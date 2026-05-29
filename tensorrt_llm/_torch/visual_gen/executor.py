import os
import queue
import secrets
import threading
import time
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
import torch.distributed as dist
import zmq

from tensorrt_llm._torch.visual_gen.output import PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm.executor.ipc import ZeroMqQueue
from tensorrt_llm.logger import logger
from tensorrt_llm.visual_gen.args import VisualGenArgs

if TYPE_CHECKING:
    from tensorrt_llm.visual_gen.params import VisualGenParams


@dataclass
class DiffusionRequest:
    """Request for diffusion inference.

    Generation parameters live in the optional ``params`` object
    (a :class:`~tensorrt_llm.visual_gen.params.VisualGenParams` instance).
    When ``params`` is ``None`` (the default), the executor creates a
    ``VisualGenParams()`` and fills it with pipeline-specific defaults
    before calling ``pipeline.infer()``.
    """

    request_id: int
    prompt: List[str]
    params: Optional["VisualGenParams"] = None


@dataclass
class DiffusionResponse:
    """Response with model-specific output.

    Attributes:
        request_id: Unique identifier for the request.
        output: Generated media as :class:`PipelineOutput` with the
            model-specific fields populated. Set to ``None`` on the error
            path; on the READY signal it carries a ``dict`` instead.
        error_msg: Error message if generation failed.
        generation: Wall-clock time the executor measured around the
            engine's inference call (host ``time.perf_counter()``), in
            seconds. Default ``0.0`` so the dataclass round-trips through
            pickling across worker/client; the error path leaves it at
            ``0.0``.
    """

    request_id: int
    output: Optional[PipelineOutput] = None
    error_msg: Optional[str] = None
    generation: float = 0.0
    # True when ``error_msg`` came from request-parameter validation
    # (i.e. the worker raised :class:`ValueError`). False on success and
    # on engine/runtime failures. The coordinator uses this flag to
    # re-raise as :class:`ValueError` (→ HTTP 400) vs
    # :class:`RuntimeError` (→ HTTP 500).
    is_validation_error: bool = False


# Python type name → accepted Python types for ExtraParamSchema validation.
_TYPE_MAP = {
    "float": (float, int),
    "int": (int,),
    "bool": (bool,),
    "str": (str,),
    "list": (list,),
}

# Generation config fields that pipelines declare defaults for.
# If a user sets one of these but the pipeline doesn't declare it in
# default_generation_params, the request is rejected so unsupported
# knobs don't get silently dropped. Conditioning inputs ``image`` and
# ``negative_prompt`` are validated at runtime by the pipeline's
# ``infer()`` and stay out of this set.
_GENERATION_CONFIG_FIELDS: tuple[str, ...] = (
    "height",
    "width",
    "num_inference_steps",
    "guidance_scale",
    "max_sequence_length",
    "num_frames",
    "frame_rate",
)


def validate_visual_gen_params(
    params: "VisualGenParams",
    *,
    pipeline_name: str,
    declared_defaults: Optional[Dict[str, Any]],
    extra_param_specs: Dict[str, Any],
) -> None:
    """Validate *params* against pipeline-declared defaults and extra specs.

    Raises :class:`ValueError` with a multi-line message listing every
    violation when one or more of:

    - Unknown ``extra_params`` keys.
    - Universal fields (e.g. ``num_frames``) set by the user but not
      declared in ``declared_defaults``. Skipped when
      ``declared_defaults`` is ``None`` — clients that don't carry
      the per-pipeline universal-field set can still validate
      ``extra_params``.
    - Type mismatches for ``extra_params`` values.
    - Out-of-range ``extra_params`` values.
    """
    messages: List[str] = []
    specs = extra_param_specs

    # --- unknown extra_params keys ---
    if params.extra_params:
        unknown = sorted(set(params.extra_params.keys()) - set(specs.keys()))
        if unknown:
            messages.append(
                f"Unknown extra_params {unknown} for {pipeline_name}. "
                f"Supported: {sorted(specs.keys())}"
            )

    # --- unsupported universal fields ---
    # Check generation config fields the user explicitly set (not None)
    # that the pipeline never declared in declared_defaults.
    # Conditioning inputs (image, negative_prompt) are excluded — they
    # are validated at runtime by the pipeline's infer().
    if declared_defaults is not None:
        for field_name in _GENERATION_CONFIG_FIELDS:
            value = getattr(params, field_name, None)
            if value is not None and field_name not in declared_defaults:
                messages.append(
                    f"Parameter '{field_name}' is set but {pipeline_name} does "
                    f"not accept it (not in default_generation_params)."
                )

    # --- extra_params type and range checks ---
    if params.extra_params:
        for key, value in params.extra_params.items():
            if key not in specs:
                continue  # already reported as unknown above
            spec = specs[key]
            # Skip None values (param left at its None default)
            if value is None:
                continue
            # Type check
            expected_types = _TYPE_MAP.get(spec.type)
            if expected_types and not isinstance(value, expected_types):
                messages.append(
                    f"extra_params['{key}'] expected type '{spec.type}', "
                    f"got {type(value).__name__}: {value!r}"
                )
                continue  # skip range check if type is wrong
            # Range check (numeric only)
            if spec.range is not None and isinstance(value, (int, float)):
                lo, hi = spec.range
                if not (lo <= value <= hi):
                    messages.append(
                        f"extra_params['{key}'] value {value} is out of range [{lo}, {hi}]"
                    )

    if not messages:
        return

    raise ValueError(
        f"Parameter validation failed for {pipeline_name}:\n"
        + "\n".join(f"  - {e}" for e in messages)
    )


class DiffusionExecutor:
    """Execution engine for diffusion models running in worker processes."""

    def __init__(
        self,
        request_queue_addr: str,
        response_queue_addr: str,
        device_id: int,
        visual_gen_args: "VisualGenArgs",
        req_hmac_key: Optional[bytes] = None,
        resp_hmac_key: Optional[bytes] = None,
    ):
        self.request_queue_addr = request_queue_addr
        self.response_queue_addr = response_queue_addr
        self.device_id = device_id
        self.visual_gen_args = visual_gen_args
        self.resp_hmac_key = resp_hmac_key

        self.pipeline = None  # initialized in _load_pipeline
        self.requests_ipc = None
        self.rank = dist.get_rank()
        self.response_queue = queue.Queue()
        self.sender_thread = None

        # Only rank 0 handles IPC
        if self.rank == 0:
            logger.info(f"Worker {device_id}: Connecting to request queue")
            self.requests_ipc = ZeroMqQueue(
                (request_queue_addr, req_hmac_key),
                is_server=False,
                socket_type=zmq.PULL,
                use_hmac_encryption=True,
            )
            self.sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
            self.sender_thread.start()

        self._load_pipeline()

    def _sender_loop(self):
        """Background thread for sending responses."""
        logger.info(f"Worker {self.device_id}: Connecting to response queue")
        responses_ipc = ZeroMqQueue(
            (self.response_queue_addr, self.resp_hmac_key),
            is_server=False,
            socket_type=zmq.PUSH,
            use_hmac_encryption=True,
        )

        while True:
            try:
                resp = self.response_queue.get()
                if resp is None:
                    break
                responses_ipc.put(resp)
            except Exception as e:
                logger.error(f"Worker {self.device_id}: Sender error: {e}")

        if responses_ipc.socket:
            responses_ipc.socket.setsockopt(zmq.LINGER, 0)
        responses_ipc.close()

    def _load_pipeline(self):
        """
        Load pipeline using proper flow:
        VisualGenArgs → PipelineLoader → DiffusionModelConfig → AutoPipeline → BasePipeline
        """
        logger.info(f"Worker {self.device_id}: Loading pipeline")

        try:
            args = self.visual_gen_args
            loader = PipelineLoader(args, device=f"cuda:{self.device_id}")
            self.pipeline = loader.load(
                skip_warmup=args.compilation_config.skip_warmup,
            )

        except Exception as e:
            logger.error(f"Worker {self.device_id}: Failed to load pipeline: {e}")
            raise

        logger.info(f"Worker {self.device_id}: Pipeline ready")

        # Sync all workers
        dist.barrier()

        # Send READY signal with pipeline metadata for the client.
        if self.rank == 0:
            logger.info(f"Worker {self.device_id}: Sending READY")
            self.response_queue.put(
                DiffusionResponse(
                    request_id=-1,
                    output={
                        "status": "READY",
                        "pipeline_name": self.pipeline.__class__.__name__,
                        "default_generation_params": self.pipeline.default_generation_params,
                        "extra_param_specs": self.pipeline.extra_param_specs,
                    },
                )
            )

    def serve_forever(self):
        """Main execution loop."""
        while True:
            req = None
            if self.rank == 0:
                req = self.requests_ipc.get()
                logger.info(f"Worker {self.device_id}: Request available")
                if req is not None:
                    # Materialize a concrete seed on the coordinator rank
                    # before broadcasting so every rank sees the same value.
                    # Drawing per-rank would diverge under multi-rank
                    # parallelism (cfg_size, ulysses_size).
                    self._resolve_seed(req)

            # Broadcast to all ranks
            obj_list = [req]
            dist.broadcast_object_list(obj_list, src=0)
            req = obj_list[0]

            if req is None:
                logger.info(f"Worker {self.device_id}: Shutdown signal received")
                if self.rank == 0 and self.sender_thread:
                    self.response_queue.put(None)
                    self.sender_thread.join()
                break

            logger.info(f"Worker {self.device_id}: Processing request {req.request_id}")
            self.process_request(req)

    def _resolve_seed(self, req: DiffusionRequest) -> None:
        """Materialize a concrete seed when the client omitted one.

        Called once per request on the coordinator rank (rank 0) before
        :func:`torch.distributed.broadcast_object_list`. After broadcast,
        all ranks see the same integer, so downstream pipelines never need
        to draw their own randomness. Direct callers (e.g. unit tests
        invoking :meth:`process_request` without going through
        :meth:`serve_forever`) can call this idempotently — when
        ``req.params`` is ``None`` a fresh :class:`VisualGenParams` is
        constructed and seeded; when ``seed`` is already an integer the
        method is a no-op.
        """
        if req.params is None:
            from tensorrt_llm.visual_gen.params import VisualGenParams

            req.params = VisualGenParams()
        if req.params.seed is None:
            # 32-bit range matches the OpenAI DALL-E seed convention that
            # vllm-omni adopts; see VisualGenParams.seed for context.
            req.params.seed = secrets.randbits(32)

    def _merge_defaults(self, req: DiffusionRequest):
        """Fill ``None`` fields in *req.params* with pipeline-specific defaults.

        Merges both universal defaults (from ``default_generation_params``)
        and extra_param defaults (from ``extra_param_specs``).
        """
        if req.params is None:
            from tensorrt_llm.visual_gen.params import VisualGenParams

            kwargs = dict(self.pipeline.default_generation_params)
            specs = self.pipeline.extra_param_specs
            if specs:
                kwargs["extra_params"] = {key: spec.default for key, spec in specs.items()}
            req.params = VisualGenParams(**kwargs)
            return

        params = req.params
        # Universal field defaults
        for field_name, default_value in self.pipeline.default_generation_params.items():
            if hasattr(params, field_name) and getattr(params, field_name) is None:
                setattr(params, field_name, default_value)

        # Extra param defaults — fill all declared keys so infer() can use direct access
        specs = self.pipeline.extra_param_specs
        if specs:
            if params.extra_params is None:
                params.extra_params = {}
            for key, spec in specs.items():
                if key not in params.extra_params:
                    params.extra_params[key] = spec.default

        self._validate_request(req)

    def _validate_request(self, req: DiffusionRequest):
        """Validate *req.params* against the loaded pipeline's declared parameters.

        Worker-side entry point — delegates to
        :func:`validate_visual_gen_params` with the pipeline's declared
        defaults and extra-param specs.
        """
        validate_visual_gen_params(
            req.params,
            pipeline_name=self.pipeline.__class__.__name__,
            declared_defaults=self.pipeline.default_generation_params,
            extra_param_specs=self.pipeline.extra_param_specs,
        )

    def process_request(self, req: DiffusionRequest):
        """Process a single request."""
        try:
            # Idempotent fallback for direct callers (unit tests). In
            # production ``serve_forever`` has already resolved the seed
            # on rank 0 before broadcast, so this is a no-op on the live
            # path.
            self._resolve_seed(req)
            self._merge_defaults(req)
            cache_key = self.pipeline.warmup_cache_key(
                req.params.height, req.params.width, num_frames=req.params.num_frames
            )
            if self.pipeline._warmed_up_shapes and cache_key not in self.pipeline._warmed_up_shapes:
                logger.warning(
                    f"Requested shape {cache_key} was not warmed up. "
                    f"First request with this shape will be slower due to "
                    f"torch.compile recompilation or CUDA graph capture. "
                    f"Warmed-up shapes: {self.pipeline._warmed_up_shapes}"
                )
            # Host wall-clock around pipeline.infer(). The pipeline already
            # syncs at the end (decode_latents path), so this captures the
            # full executor-side envelope including any pre/post-pipeline work
            # that the per-phase CUDA-event timings on PipelineOutput do not.
            generation_start = time.perf_counter()
            output = self.pipeline.infer(req)
            generation = time.perf_counter() - generation_start  # seconds
            if self.rank == 0:
                self.response_queue.put(
                    DiffusionResponse(
                        request_id=req.request_id,
                        output=output,
                        generation=generation,
                    )
                )
        except ValueError as e:
            # ``ValueError`` is the stock-Python signal that the request was
            # malformed (raised by ``validate_visual_gen_params`` and by
            # pipelines' own ``infer()`` checks). Tag the response so the
            # coordinator re-raises as ``ValueError`` and the serve layer
            # renders an HTTP 400.
            logger.error(f"Worker {self.device_id}: Validation error: {e}")
            if self.rank == 0:
                self.response_queue.put(
                    DiffusionResponse(
                        request_id=req.request_id,
                        error_msg=str(e),
                        is_validation_error=True,
                    )
                )
        except Exception as e:
            logger.error(f"Worker {self.device_id}: Error: {e}")
            logger.error(traceback.format_exc())
            if self.rank == 0:
                self.response_queue.put(
                    DiffusionResponse(request_id=req.request_id, error_msg=str(e))
                )


def run_diffusion_worker(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    request_queue_addr: Optional[str],
    response_queue_addr: Optional[str],
    visual_gen_args: "VisualGenArgs",
    log_level: str = "info",
    req_hmac_key: Optional[bytes] = None,
    resp_hmac_key: Optional[bytes] = None,
    local_rank: Optional[int] = None,
):
    """Entry point for worker process."""
    try:
        # Set log level before any other work so loading logs are visible
        logger.set_level(log_level)

        # Setup distributed env — use PyTorch distributed, not MPI
        os.environ["TLLM_DISABLE_MPI"] = "1"
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        # Determine local_rank: explicit arg > LOCAL_RANK env > global rank.
        # In multi-node runs (torchrun / srun --ntasks-per-node) SLURM/torchelastic
        # sets LOCAL_RANK; in single-node mp.Process mode it equals the global rank.
        _local_rank = (
            local_rank if local_rank is not None else int(os.environ.get("LOCAL_RANK", rank))
        )
        os.environ["LOCAL_RANK"] = str(_local_rank)

        # Use local_rank for device assignment so that each node's ranks map to
        # GPUs 0..gpus_per_node-1 rather than wrapping the global rank.
        device_id = _local_rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)

        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            device_id=torch.device(f"cuda:{device_id}") if torch.cuda.is_available() else None,
        )

        executor = DiffusionExecutor(
            request_queue_addr=request_queue_addr,
            response_queue_addr=response_queue_addr,
            device_id=device_id,
            visual_gen_args=visual_gen_args,
            req_hmac_key=req_hmac_key,
            resp_hmac_key=resp_hmac_key,
        )
        executor.serve_forever()
        if executor.pipeline is not None:
            executor.pipeline.cleanup()
        dist.destroy_process_group()

    except Exception as e:
        logger.error(f"Worker failed: {e}")
        traceback.print_exc()
