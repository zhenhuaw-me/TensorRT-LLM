# Task 4 — VisualGen Request-Level Output Wrapper, Future-like Handle, Batch Fan-Out, and Encoding Extraction

## Goal Description

Replace the raw `MediaOutput` return type with a request-aware public wrapper `VisualGenOutput` carrying `image`/`video`/`audio` tensors plus `request_id`, `error`, `frame_rate`/`audio_sample_rate`, and engine-side `VisualGenMetrics`. Promote `VisualGenResult` to a single Future-like awaitable handle that resolves to either `VisualGenOutput` (single prompt) or `List[VisualGenOutput]` (batch). Fan out batched `generate(List[str])` into per-item outputs with Option B per-item error semantics. Add `VisualGenOutput.save(path)` as the single user-facing way to write generated media to disk. Extract tensor-encoding code out of `tensorrt_llm/serve/media_storage.py` into a new `tensorrt_llm/media/encoding.py` module of free functions, while keeping `MediaStorage` in `serve/` for its file-storage role. Internally rename `MediaOutput` → `PipelineOutput` with three CUDA-event-based timing phases (`pre_denoise_ms`, `denoise_ms`, `post_denoise_ms`) plus an executor-measured `pipeline_ms` on the wire response. Update every in-tree caller (examples, bench, serve, tests) in the same change set. API-stability YAML coverage for VisualGen is explicitly deferred until the surface stabilizes post-prototype.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: New public type `VisualGenOutput` is a flat `@dataclass` at `tensorrt_llm/visual_gen/output.py` with fields `request_id: int`, `image`/`video`/`audio: Optional[torch.Tensor]`, `frame_rate: Optional[float]`, `audio_sample_rate: Optional[int]`, `error: Optional[str]`, `metrics: Optional[VisualGenMetrics]`, and is re-exported from `tensorrt_llm.visual_gen` and `tensorrt_llm`.
  - Positive Tests (expected to PASS):
    - `from tensorrt_llm import VisualGenOutput` and `from tensorrt_llm.visual_gen import VisualGenOutput` both succeed.
    - Constructing with only `request_id` and `image` leaves `video`/`audio`/`frame_rate`/`audio_sample_rate`/`error`/`metrics` at documented defaults.
    - Constructing with `video` + `frame_rate` + `audio` + `audio_sample_rate` round-trips all four values.
    - `VisualGenOutput._from_response(resp)` on a successful `DiffusionResponse` populates `request_id`, the right media tensor, rate fields, and a `VisualGenMetrics` carrying the four timing values.
    - `VisualGenOutput._from_response(resp)` on an error `DiffusionResponse` produces an output with `image is None`, `video is None`, `audio is None`, `error == resp.error_msg`, `metrics is None`.
  - Negative Tests (expected to FAIL):
    - Importing the deleted public symbol `MediaOutput` from `tensorrt_llm` or `tensorrt_llm.visual_gen` raises `ImportError`.
    - Defining `VisualGenOutput` as a `BaseModel` / Pydantic-validated class fails the dataclass-shape check.

- AC-2: `VisualGenMetrics` exists at `tensorrt_llm/visual_gen/output.py` with exactly four float fields (`pipeline_ms`, `pre_denoise_ms`, `denoise_ms`, `post_denoise_ms`), each defaulting to `0.0`. Successful single and batch outputs carry a populated metrics object; error outputs carry `metrics is None`.
  - Positive Tests (expected to PASS):
    - End-to-end Flux smoke run yields `result.metrics.pipeline_ms > 0`, `result.metrics.denoise_ms > 0`, `result.metrics.pre_denoise_ms > 0`, `result.metrics.post_denoise_ms > 0`.
    - The breakdown sums to no more than `pipeline_ms` plus a documented slack: `pre + denoise + post <= pipeline_ms + slack`.
    - Error outputs (validation failure, pipeline raise) have `metrics is None`.
  - Negative Tests (expected to FAIL):
    - A wire response with `error_msg != None` producing a `VisualGenOutput` whose `metrics is not None` fails the test.
    - Adding any field other than the four timing fields to `VisualGenMetrics` fails the field-set test.

- AC-3: `VisualGenResult` is a single Future-like awaitable handle exposing `__await__`, `done` property, sync `result(timeout=None)`, async `aresult(timeout=None)`. The same handle class resolves to `VisualGenOutput` for single-prompt input and `List[VisualGenOutput]` for batch input.
  - Positive Tests (expected to PASS):
    - `out = asyncio.run(engine.generate_async("a cat"))` returns a `VisualGenOutput`; `future.done is True` after completion.
    - `await engine.generate_async(["a","b"])` returns `List[VisualGenOutput]` of length 2; `future.done is True`.
    - Calling `future.result()` (sync) on a freshly-created handle blocks then returns the resolved value.
    - Calling `await future.aresult()` returns the same value as `await future`.
  - Negative Tests (expected to FAIL):
    - Calling `await future.result()` (treating sync `result()` as awaitable) raises `TypeError`.
    - Awaiting a handle whose underlying request errored raises `VisualGenError` from the wait call.

- AC-4: `VisualGen.generate()` returns `VisualGenOutput` for `str` input and `List[VisualGenOutput]` for `List[str]` input. Single-prompt failure raises `VisualGenError` (back-compat). Batch path never raises on per-item or whole-batch failure; failed items carry `error != None`. `params: List[VisualGenParams]` raises `NotImplementedError` mentioning §5.1.2.
  - Positive Tests (expected to PASS):
    - `engine.generate("a cat")` returns one `VisualGenOutput` with `error is None` on success.
    - `engine.generate(["a","b","c"])` returns a 3-element list; on full-batch failure all three items have the same executor-level `error` string.
    - Per-item tensors in the batch result are unbatched (sliced along dim 0); independent across list entries.
    - `engine.generate("a cat", params=[VisualGenParams()])` raises `NotImplementedError` whose message references §5.1.2.
  - Negative Tests (expected to FAIL):
    - `engine.generate(["a","b"])` raising on validation failure (must return list with errors instead).
    - Batch result tensors sharing storage with the underlying batched tensor in a way that mutating one affects another (only required when explicit independence is asserted; otherwise share-of-view is acceptable).

- AC-5: `VisualGenOutput.save(path, *, format=None, frame_rate=None, audio_sample_rate=None, quality=95) -> Path` writes the generated media to disk by routing to free functions in `tensorrt_llm.media.encoding`. Format is inferred from the path extension unless given. Rates carried on the output are used as defaults; explicit kwargs override.
  - Positive Tests (expected to PASS):
    - `out.save(tmp/"x.png")` on an image output writes a valid PNG and returns the path.
    - `out.save(tmp/"x.mp4")` on a video output with `frame_rate=16.0` writes a valid MP4.
    - Passing `frame_rate=30.0` overrides the carried value at the encoding free-function call.
    - Mocking `tensorrt_llm.media.encoding.save_image` confirms `save()` calls it (proving routing through free functions, not `MediaStorage`).
  - Negative Tests (expected to FAIL):
    - `out.save(...)` on an output with `error != None` raises `VisualGenError` mentioning the error message.
    - `out.save("x.mp4")` on a video output with `frame_rate is None` and no kwarg raises `VisualGenError`.
    - `out.save("x.png")` on an output with no media (`image`/`video`/`audio` all None) raises `VisualGenError`.

- AC-6: New `tensorrt_llm/media/` package contains `__init__.py` and `encoding.py`. `encoding.py` exposes free functions `save_image`, `save_video`, `image_to_bytes`, `video_to_bytes`, `resolve_video_format`, plus module-private helpers (`_to_pil_image`, `_save_pil_image`, `_save_encoded_video`). The free functions are not re-exported from `tensorrt_llm/__init__.py`.
  - Positive Tests (expected to PASS):
    - `from tensorrt_llm.media.encoding import save_image, save_video, image_to_bytes, video_to_bytes, resolve_video_format` succeeds.
    - `save_image(tensor, tmp/"x.png")` writes a valid PNG; `save_video(tensor, tmp/"x.mp4", frame_rate=16.0)` writes a valid MP4; `video_to_bytes(...)` returns non-empty `bytes`.
    - `resolve_video_format(Path("x.mp4"))` returns the same canonical identifier it produced before the move.
  - Negative Tests (expected to FAIL):
    - `from tensorrt_llm import save_image` raises `ImportError` (free functions are not top-level re-exports).
    - Importing `_to_pil_image` from outside `tensorrt_llm.media.encoding` works but is documented as private; nothing in-tree imports it from outside its module.

- AC-7: `tensorrt_llm/serve/media_storage.py` retains the file with `MediaStorage` for its file-storage role. The four tensor-encoding methods (`save_image`, `save_video`, `convert_image_to_bytes`, `convert_video_to_bytes`) are removed from `MediaStorage`; the free function `resolve_video_format` is moved to `media/encoding.py`. No in-tree caller references the removed methods.
  - Positive Tests (expected to PASS):
    - `import tensorrt_llm.serve.media_storage` succeeds; `MediaStorage` class exists.
    - `MediaStorage`'s remaining surface area is its file-storage role (constructor + any retained directory/path helpers).
    - `git grep "MediaStorage\.save_image\|MediaStorage\.save_video\|MediaStorage\.convert_image_to_bytes\|MediaStorage\.convert_video_to_bytes"` returns no matches under `tensorrt_llm/`, `examples/`, `tests/`, `bench/`.
  - Negative Tests (expected to FAIL):
    - `MediaStorage().save_image(...)` raises `AttributeError`.
    - Calling `from tensorrt_llm.serve.media_storage import resolve_video_format` raises `ImportError` (moved to `media/encoding.py`).

- AC-8: Internal `MediaOutput` is renamed to `PipelineOutput` at `tensorrt_llm/_torch/visual_gen/output.py`, gaining fields `frame_rate`, `audio_sample_rate`, `pre_denoise_ms`, `denoise_ms`, `post_denoise_ms`. Every reference is updated. `MediaOutput` is no longer importable from any path.
  - Positive Tests (expected to PASS):
    - `from tensorrt_llm._torch.visual_gen.output import PipelineOutput` succeeds and has all eight fields with correct types.
    - `PipelineOutput()` (no args) constructs with all-None defaults.
    - `git grep "MediaOutput"` returns no matches anywhere in tree (production code, tests, fakes, docs in repo).
  - Negative Tests (expected to FAIL):
    - `from tensorrt_llm._torch.visual_gen.output import MediaOutput` raises `ImportError`.
    - `from tensorrt_llm import MediaOutput` and `from tensorrt_llm.visual_gen import MediaOutput` raise `ImportError`.

- AC-9: Each pipeline (`pipeline_flux.py`, `pipeline_flux2.py`, `pipeline_wan.py`, `pipeline_wan_i2v.py`, `pipeline_ltx2.py`, `pipeline_ltx2_two_stages.py`) populates the metadata it owns and instruments timings via CUDA events (no host syncs added).
  - Positive Tests (expected to PASS):
    - Wan T2V/I2V successful runs produce `result.frame_rate == 16.0` and three positive timing fields.
    - LTX-2 successful runs produce `result.frame_rate == params.frame_rate`, `result.audio_sample_rate` populated from the LTX-2 audio config (looked up at runtime, not hardcoded), and three positive timing fields.
    - Flux/Flux2 successful runs produce `result.frame_rate is None`, `result.audio_sample_rate is None`, and three positive timing fields.
    - LTX-2 two-stage pipeline accounts the first stage as `denoise_ms` and rolls the second stage into `post_denoise_ms`; the docstring documents this split.
    - All six pipelines use `torch.cuda.Event(enable_timing=True)` at phase boundaries.
  - Negative Tests (expected to FAIL):
    - `git grep "torch.cuda.synchronize" -- tensorrt_llm/_torch/visual_gen/models/` returning matches inside `infer()` for timing purposes (only acceptable for non-timing reasons).
    - LTX-2 audio sample rate hardcoded as a literal in pipeline code rather than read from config.

- AC-10: `DiffusionResponse` (wire dataclass) gains `pipeline_ms: float = 0.0` measured by the executor around `pipeline.infer()`. The error path leaves `pipeline_ms` at `0.0`.
  - Positive Tests (expected to PASS):
    - Successful response carries `resp.pipeline_ms > 0` (host wall-clock).
    - `VisualGenMetrics.pipeline_ms` after `_from_response` equals `resp.pipeline_ms`.
  - Negative Tests (expected to FAIL):
    - Adding `pipeline_ms` without a default value (breaks pickling across worker/client) fails the smoke test.
    - Error-path response carries `pipeline_ms != 0.0`.

- AC-11: All in-tree callers updated in the same change set: `examples/visual_gen/quickstart_example.py`, `visual_gen_flux.py`, `visual_gen_ltx2.py`, `visual_gen_wan_i2v.py`, `visual_gen_wan_t2v.py`, `examples/visual_gen/models/wan_t2v.py`, `tensorrt_llm/bench/benchmark/visual_gen.py`, `tensorrt_llm/serve/openai_server.py`, plus test files referenced under "Files touched" in the draft.
  - Positive Tests (expected to PASS):
    - Examples import only `VisualGen`, `VisualGenParams`, `VisualGenOutput` (and friends); no `MediaStorage` import; no hardcoded `frame_rate=...` literal at the save call site.
    - `bench/benchmark/visual_gen.py` reports `result.metrics.denoise_ms`, `result.metrics.pipeline_ms`, and an externally-measured `e2e_ms` (computed by wrapping `generate()` + `save()` in `time.perf_counter()`).
    - `serve/openai_server.py` calls `tensorrt_llm.media.encoding.{save_image,save_video,image_to_bytes,video_to_bytes}` directly; logs an externally-measured `e2e_ms` in its existing per-request log/report.
    - Batch-handling sites in `bench/benchmark/visual_gen.py` use `if result.error:` rather than try/except on `VisualGenError` (warmup retains try/except).
  - Negative Tests (expected to FAIL):
    - `git grep "MediaStorage" -- examples/ bench/ tensorrt_llm/serve/openai_server.py` returning encoding-method matches.
    - `git grep "frame_rate=16.0\|frame_rate=24.0" -- examples/` returning matches at save call sites (literals are now centralized in pipelines).
    - `git grep "VisualGenError" -- tensorrt_llm/bench/benchmark/visual_gen.py` returning try/except in non-warmup batch paths.

- AC-12: API-stability YAML coverage for VisualGen surface is explicitly deferred. No new `references/visual_gen*.yaml` is added in this change set; existing `tests/unittest/api_stability/` tests pass unmodified.
  - Positive Tests (expected to PASS):
    - `pytest tests/unittest/api_stability` passes without modification.
    - The PR description records the deferral with rationale (prototype `@set_api_status` plus mid-flux surface).
  - Negative Tests (expected to FAIL):
    - A new `tests/unittest/api_stability/references/visual_gen*.yaml` introduced in this change set fails the deferral check.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

The implementation satisfies all twelve acceptance criteria with the full suite of unit tests in `tests/unittest/visual_gen/test_output.py` (covering AC-1 through AC-6 and the relevant cases of AC-9), integration assertions in `tests/integration/defs/examples/test_visual_gen.py` (covering Flux single/batch, Wan T2V, LTX-2, error-path, `NotImplementedError` guard, async path), updated test fakes and import tests, and a one-line comment in the `VisualGenMetrics` docstring documenting the GPU-event measurement methodology. `MediaOutput` is fully purged in tree. Bench and serve both report the externally-measured `e2e_ms` alongside the engine-side timings.

### Lower Bound (Minimum Acceptable Scope)

The implementation satisfies all twelve acceptance criteria. Unit tests cover AC-1, AC-2, AC-3, AC-4, AC-5, AC-6, AC-8 directly. At least one integration smoke test runs Flux end-to-end and asserts `result.image is not None`, `result.error is None`, `result.metrics.pipeline_ms > 0`, `result.metrics.denoise_ms > 0`, plus a successful `result.save(tmp/"x.png")`. The bench script reports at minimum `denoise_ms`, `pipeline_ms`, and externally-measured `e2e_ms`; `serve/openai_server.py` measures `e2e_ms` externally for its existing log line. `MediaOutput` is removed from all imports and re-exports.

### Allowed Choices

- Can use:
  - `@dataclass` for `VisualGenOutput`, `VisualGenMetrics`, `PipelineOutput`.
  - CUDA events (`torch.cuda.Event(enable_timing=True)` + `event.elapsed_time`) for the three phase timings.
  - Lazy imports inside `VisualGenOutput.save()` to avoid pulling ffmpeg/imageio at API import time.
  - `time.perf_counter()` in the executor (host wall-clock for `pipeline_ms`) and in callers (host wall-clock for external `e2e_ms`).
  - Tensor-view slicing (`t[i]`) for batch fan-out; `_split_visual_gen_output` may switch to `t[i].clone().contiguous()` if a view-aliasing issue is observed.
  - Either keeping `VisualGenError` in its current location (`tensorrt_llm/visual_gen/visual_gen.py`) or extracting to a sibling `errors.py`; consumers import from wherever it lives.
  - Either combining the encoding free-function helpers with module-level private underscores or splitting into a separate private module under `tensorrt_llm/media/`.
- Cannot use:
  - `torch.cuda.synchronize()` introduced inside `infer()` solely for timing purposes.
  - Pydantic models for `VisualGenOutput` / `VisualGenMetrics` / `PipelineOutput` (would force tensor juggling).
  - Importing from `tensorrt_llm.serve.*` in the public Python API path (`tensorrt_llm.visual_gen.*` or `tensorrt_llm.media.*`).
  - Hardcoding the LTX-2 audio sample rate as a literal in pipeline code (must be looked up from the audio config).
  - Re-exporting `MediaOutput` anywhere.
- Pinned by draft / clarifications (no implementer choice):
  - Wan T2V/I2V `frame_rate = 16.0`.
  - `VisualGenOutput` is flat (no nested wrapper).
  - `save()` routes to free functions in `tensorrt_llm.media.encoding`, not via `MediaStorage`.
  - Encoding module location is `tensorrt_llm/media/encoding.py` (top-level `media/` package).
  - `MediaStorage` stays at `tensorrt_llm/serve/media_storage.py` and the file is kept (file-storage role).
  - LTX-2 two-stage timing: first stage is `denoise_ms`, second stage is `post_denoise_ms`.
  - A single `VisualGenResult` class handles both single and batch awaits (no separate `VisualGenBatchResult`).
  - `generate_async(List[str])` returns a `VisualGenResult` whose await resolves to `List[VisualGenOutput]`.
  - `e2e_ms` is measured externally by callers; never lives on `VisualGenMetrics`.
  - No `cancel()`, `seed_used`, `prompt`, `finished`, `queue_ms`, `to_bytes`, `to_pil`, streaming, OTLP tracing, or per-modality discriminator in this change set.
  - API-stability YAML for VisualGen is deferred.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

One viable ordering minimizes blocked dependencies by building the data types from the inside out and adapting callers last:

1. Land the internal rename and timing instrumentation first. `PipelineOutput` is the sole producer of timing data; once each pipeline records the three CUDA events at phase boundaries and emits `pre_denoise_ms`/`denoise_ms`/`post_denoise_ms`, every later layer can consume it without speculation. The executor adds `pipeline_ms` to `DiffusionResponse` at the same time.
2. Extract the encoding free functions next. Move `save_image`, `save_video`, `image_to_bytes`, `video_to_bytes`, and `resolve_video_format` (plus the underscore helpers) from `serve/media_storage.py` to a new `tensorrt_llm/media/encoding.py`. Strip the moved methods from `MediaStorage` but keep the file and the class for its file-storage role.
3. Add the public types. `VisualGenMetrics`, `VisualGenOutput` (with `_from_response` and `save()`), and `_split_visual_gen_output` all live at `tensorrt_llm/visual_gen/output.py`. Re-export from `tensorrt_llm.visual_gen` and `tensorrt_llm`.
4. Promote `VisualGenResult` to the Future-like shape. The same handle resolves to `VisualGenOutput` for single input or `List[VisualGenOutput]` for batch; `_consume` chooses based on what `_split_visual_gen_output` was asked to produce.
5. Widen `generate()` to the union return type, plumb the `NotImplementedError` guard, and implement `_generate_batch` calling `_split_visual_gen_output`. Match `generate_async()` to accept the same input shapes via the same `VisualGenResult` handle.
6. Migrate every caller in the same change set: examples drop `MediaStorage` and the hardcoded `frame_rate=...` literals; bench replaces try/except in batch paths with `if result.error:` (warmup keeps try/except), uses `result.save(path)`, and adds external `e2e_ms` measurement; `serve/openai_server.py` calls the encoding free functions directly and adds external `e2e_ms` for its existing per-request log line. Update test fakes for the rename; add `tests/unittest/visual_gen/test_output.py`; tighten integration assertions for rate fields and timings.

CUDA-event methodology note (worth surfacing in the `VisualGenMetrics` docstring): events record asynchronously on the GPU stream so the pipeline doesn't stall during execution; `event.elapsed_time` performs an implicit sync at the end which is amortized into the executor-side sync that already occurs when the response is consumed. They measure GPU-stream time only, so small host-only work (e.g., tokenization) shows up as `pipeline_ms - (pre + denoise + post)`, which is itself useful information.

For LTX-2's two-stage pipeline: take the first denoising stage as `denoise_ms` and account the second stage along with VAE decode, format conversion, and audio decode as `post_denoise_ms`. Document the choice in the pipeline docstring so future readers know how to interpret the breakdown.

For batch tensor splitting, start with simple views (`t[i]`); only switch to `t[i].clone().contiguous()` if a per-pipeline assertion or a downstream encoder reports an aliasing issue. The plan's negative test does not require copies by default.

### Relevant References

- `tensorrt_llm/visual_gen/visual_gen.py` — current `VisualGen` engine; hosts `VisualGenError`, `VisualGenParamsError`, current `VisualGenResult`. The Future-like rewrite lands here.
- `tensorrt_llm/visual_gen/params.py` — current `VisualGenParams` definition (already has `frame_rate`, `extra_params`).
- `tensorrt_llm/visual_gen/__init__.py` — public re-exports; drop `MediaOutput`, add `VisualGenOutput`, `VisualGenMetrics`.
- `tensorrt_llm/__init__.py` — top-level re-exports for the same.
- `tensorrt_llm/_torch/visual_gen/output.py` — current `MediaOutput`; rename + extend here.
- `tensorrt_llm/_torch/visual_gen/__init__.py` — drop `MediaOutput`, re-export `PipelineOutput` (internal-only).
- `tensorrt_llm/_torch/visual_gen/executor.py` — `DiffusionRequest` / `DiffusionResponse` dataclasses; add `pipeline_ms` and measure around `pipeline.infer()`.
- `tensorrt_llm/_torch/visual_gen/models/{flux,wan,ltx2}/pipeline_*.py` — six pipelines that now populate metadata and three CUDA-event timings; LTX-2 audio sample rate looked up from audio config.
- `tensorrt_llm/serve/media_storage.py` — file kept; tensor-encoding methods removed; `resolve_video_format` moved out.
- `tensorrt_llm/media/{__init__.py,encoding.py}` — new package; encoding free functions live here.
- `tensorrt_llm/serve/openai_server.py` — switches to `result.image` / `result.video` / `result.error` / `result.frame_rate` / `result.audio_sample_rate`; calls `tensorrt_llm.media.encoding.*` free functions directly; adds external `e2e_ms` to its per-request log line.
- `tensorrt_llm/bench/benchmark/visual_gen.py` — switches to `result.save(path)`, `result.metrics.*`, and externally-measured `e2e_ms`; `if result.error:` for batch paths.
- `examples/visual_gen/{quickstart_example,visual_gen_flux,visual_gen_ltx2,visual_gen_wan_i2v,visual_gen_wan_t2v,models/wan_t2v}.py` — drop `MediaStorage` import, switch to `result.save(path)`, drop hardcoded `frame_rate=...`.
- `tests/unittest/_torch/visual_gen/test_trtllm_serve_endpoints.py` — `MediaOutput` → `PipelineOutput` in fakes; encoding-method references switch to `tensorrt_llm.media.encoding`.
- `tests/unittest/dynamo/test_imports.py` — `MediaOutput` → `PipelineOutput`.
- `tests/integration/defs/examples/test_visual_gen.py` — assertions on rates and timing fields.
- `tests/unittest/visual_gen/test_output.py` — new unit-test module covering `VisualGenOutput`, `VisualGenMetrics`, `_from_response`, `_split_visual_gen_output`, `save()` (via mocked encoding functions), and `VisualGenResult` awaitability.

## Dependencies and Sequence

### Milestones

1. **Milestone 1 — Internal types and timing**: `MediaOutput` → `PipelineOutput` rename, new fields, CUDA-event instrumentation across all six pipelines, executor measures `pipeline_ms` on `DiffusionResponse`.
   - Phase A: Rename and extend `PipelineOutput` (eight fields).
   - Phase B: Update every reference (~25 sites: pipelines, executor, `_torch/visual_gen/__init__.py`, test fakes).
   - Phase C: Each pipeline records three CUDA events at phase boundaries and emits `pre_denoise_ms`/`denoise_ms`/`post_denoise_ms`; LTX-2 looks up audio sample rate from audio config; LTX-2 two-stage rolls stage two into `post_denoise_ms`.
   - Phase D: `DiffusionResponse.pipeline_ms: float = 0.0`; executor wraps `pipeline.infer()` with `time.perf_counter()`.

2. **Milestone 2 — Encoding extraction**: Build the new `media/` package; pull encoding out of `MediaStorage` while preserving the file's file-storage role.
   - Phase A: Create `tensorrt_llm/media/__init__.py` and `tensorrt_llm/media/encoding.py` with the five free functions and three private helpers.
   - Phase B: Remove the four encoding methods and `resolve_video_format` from `serve/media_storage.py`. Keep the file and the class for file-storage use.

3. **Milestone 3 — Public output types**: Define `VisualGenMetrics`, `VisualGenOutput` (with `_from_response` and `save()`), and `_split_visual_gen_output`. Re-export from `tensorrt_llm.visual_gen` and `tensorrt_llm`.
   - Phase A: New file `tensorrt_llm/visual_gen/output.py` with `VisualGenMetrics` and `VisualGenOutput` (including the lazy-import `save()`).
   - Phase B: `_from_response` factory at the API boundary.
   - Phase C: `_split_visual_gen_output` helper for batch fan-out.
   - Phase D: Update `tensorrt_llm/visual_gen/__init__.py` and `tensorrt_llm/__init__.py` re-exports; drop `MediaOutput`.

4. **Milestone 4 — `VisualGenResult` Future-like handle**: Single class for single and batch.
   - Phase A: Add `done` property, sync `result()`, async `aresult()`, `__await__`.
   - Phase B: `_consume` resolves to `VisualGenOutput` or `List[VisualGenOutput]` per the request's input shape.

5. **Milestone 5 — `generate()` / `generate_async()` widening**: Union input/return types; `NotImplementedError` guard; client-side batch splitting.
   - Phase A: Widen `generate()` and `generate_async()` signatures; add the `params: List[VisualGenParams]` guard.
   - Phase B: Implement `_generate_batch` that enqueues one batched request and calls `_split_visual_gen_output` with `pipeline_ms`.

6. **Milestone 6 — Caller migration**: Examples, bench, serve, tests in the same change set.
   - Phase A: `examples/visual_gen/*.py` and `examples/visual_gen/models/wan_t2v.py` switch to `result.save(path)`; drop `MediaStorage` import and hardcoded rates.
   - Phase B: `tensorrt_llm/bench/benchmark/visual_gen.py` switches to `result.save(path)`, `result.metrics.*`, external `e2e_ms`, and `if result.error:` for batch paths.
   - Phase C: `tensorrt_llm/serve/openai_server.py` switches result-handling sites to `result.image` / `result.video` / `result.error`, replaces `MediaStorage.*` calls with `tensorrt_llm.media.encoding.*`, and adds external `e2e_ms`.
   - Phase D: Update test fakes; rename `MediaOutput` → `PipelineOutput` in `dynamo/test_imports.py` and `_torch/visual_gen/test_trtllm_serve_endpoints.py`; add `tests/unittest/visual_gen/test_output.py`; tighten integration assertions in `tests/integration/defs/examples/test_visual_gen.py`.

### Dependency Graph

- Milestone 3 depends on Milestone 1 (uses `PipelineOutput` and reads `DiffusionResponse.pipeline_ms`) and Milestone 2 (uses encoding free functions in `save()`).
- Milestone 4 depends on Milestone 3 (resolves to `VisualGenOutput`).
- Milestone 5 depends on Milestone 4 (uses the Future-like handle as the single-prompt path).
- Milestone 6 depends on Milestones 3, 4, 5 (callers consume the new public surface end-to-end).
- Within Milestone 6, the four phases are independent and can land in any order.

## Implementation Notes

### Code Style Requirements

- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead.

### Additional Guidance Captured From Clarifications

- LTX-2 two-stage timing: first stage counts as `denoise_ms`; the second stage rolls into `post_denoise_ms`. Record this choice in the LTX-2 two-stage pipeline docstring.
- `MediaStorage` retains a file-storage role; do not delete `tensorrt_llm/serve/media_storage.py` after removing the encoding methods.
- A single `VisualGenResult` class supports both single and batch; do not introduce a separate `VisualGenBatchResult`.
- API-stability YAML coverage for the VisualGen surface is deferred until the public API exits prototype status; do not add `tests/unittest/api_stability/references/visual_gen*.yaml` in this change set.

--- Original Design Draft Start ---

# Task 4 — Request-Level Output Wrapper + Batch Output Fan-Out

**Jira**: [TRTLLM-10897](https://jirasw.nvidia.com/browse/TRTLLM-10897)
**Source design doc**: [visual-gen-api-refactor-m2.md](../designs/visual-gen-api-refactor-m2.md) §5.1.1, §5.1.2, §5.2, §6.1, §6.1.1, §6.2, §6.3
**Status**: pending (last remaining task in the core VisualGen API refactor; Tasks 1, 2, 3, 5 are done)

---

## Overview

Five changes that ship together:

1. **New public output type `VisualGenOutput`** — a flat dataclass carrying `image`/`video`/`audio` tensors, their `frame_rate`/`audio_sample_rate`, plus `request_id`/`error`/`metrics`. Replaces `MediaOutput` as the user-facing return type.
2. **`VisualGenResult` becomes an awaitable Future-like handle** — `await result` resolves to a `VisualGenOutput`; sync `result()` and async `aresult()` mirror the LLM API; adds `done` property.
3. **Batch fan-out** — `generate(List[str])` returns `List[VisualGenOutput]` with per-item unbatched tensors and per-item `error`/`metrics` (Option B in §5.1.1, Option A in §6.1.1).
4. **`VisualGenOutput.save(path)`** — single user-facing way to write the generated media to disk; format inferred from path; rates carried on the output by default.
5. **Extract encoding into a dedicated module** — pull tensor-encoding code out of `tensorrt_llm/serve/media_storage.py` and expose it as **free functions** at `tensorrt_llm/media/encoding.py`. `VisualGenOutput.save()` calls these free functions directly. `MediaStorage` stays in `serve/` (its server-side role is preserved); its `save_*` and bytes-conversion methods are dropped, since those are encoding (now handled by the free functions), not serve-specific storage logic.

Internally, the per-pipeline result dataclass is renamed `MediaOutput` → `PipelineOutput` (it now carries pipeline-side timing alongside the media payload, so "Media" no longer fits). `PipelineOutput` stays internal. Four engine-side timing measurements flow through to `VisualGenMetrics`: `pre_denoise_ms`, `denoise_ms`, `post_denoise_ms` (pipeline-side breakdowns of `pipeline.infer()`), and `pipeline_ms` (full `pipeline.infer()`, executor-side). The three pipeline-side numbers sum to approximately `pipeline_ms`, letting users see where time is spent (text/latent preparation vs denoising vs VAE decode/format conversion). End-to-end latency including encoding is measured externally by callers (the bench script and `trtllm-serve` log/report) and never lives on `VisualGenMetrics`.

This is the last breaking change in the prototype→stable path for the public surface. After it lands, the end-to-end API matches the design doc.

---

## Motivation

Today `VisualGen.generate()` returns `MediaOutput` directly — a raw tensor container with no request metadata. The LLM API's equivalent (`RequestOutput`) carries `request_id`, per-item error, and metrics; users of both APIs should see the same shape of wrapper.

Concrete problems we hit without the wrapper:

1. **Batch error handling is ambiguous.** If one of 5 prompts fails validation, there is no per-item slot for `error`. The current code raises `VisualGenError` and loses the other 4 results.
2. **No place for per-request metadata.** `request_id`, timing measurements, `frame_rate`/`audio_sample_rate`, and anything we add later have nowhere to live on the return value.
3. **Encoding rates leak into every save site.** Every `examples/visual_gen/*.py` hardcodes `frame_rate=16.0` (Wan) or `24.0` (LTX-2) at the call site because the model output doesn't carry its own rate.
4. **No first-class way to write the result to disk.** Users must import `MediaStorage` from `tensorrt_llm.serve.media_storage` (an HTTP-server module) just to save a tensor — a layering inversion.
5. **Encoding is mixed into `MediaStorage`.** Tensor encoding is a general-purpose operation (used by examples, bench, serve, and now `save()`); `MediaStorage` should be a serve-side concept. Keeping encoding in a class under `serve/` forces every Python-API consumer to import from the server module.

The batch fan-out is bundled in because Option B error semantics (§5.1.1) only make sense once each prompt has its own `VisualGenOutput` slot.

---

## Design decisions (rationale captured)

| Decision | Choice | Why |
|---|---|---|
| Sync/async value-vs-handle relationship | Separate handle (`VisualGenResult`) and value (`VisualGenOutput`); handle is awaitable. | Diffusion output is atomic, not token-streamed. None of vLLM, vLLM-Omni, SGLang Diffusion unify them; only LLM does, because tokens stream and the future hosts the partial state. |
| `VisualGenOutput` field shape | **Flat** — `image`/`video`/`audio` directly on the dataclass, no nested wrapper. | Every diffusion peer (vLLM-Omni `OmniRequestOutput`, SGLang `GenerationResult`, diffusers `*PipelineOutput`) is flat. LLM nests because of `n`/`best_of` multiplicity; diffusion has no analogue. |
| Pipeline-internal type | Rename `MediaOutput` → **`PipelineOutput`**. Internal-only. | It now carries pipeline-side timing (`denoise_ms`) in addition to the media payload, so "Media" undersells it. "PipelineOutput" matches what each `pipeline.infer()` actually returns. |
| Pipeline ↔ wire ↔ user types | Three layers: `PipelineOutput` (pipeline) → `DiffusionResponse` (wire) → `VisualGenOutput` (public). | Each adds exactly one concern (pipeline contract / IPC envelope / public API). Mirrors vLLM core. Pipelines don't know `request_id` or executor-side timing. |
| Frame rate / sample rate location | Carried on the **output** (`VisualGenOutput.frame_rate`, `audio_sample_rate`); populated by pipelines on `PipelineOutput`. | A video tensor without its fps is incomplete data — torchaudio, librosa, soundfile, torchvision.io all pair samples with rate. Eliminates magic-number hardcoding in every example. |
| Naming: video rate / audio rate | `frame_rate` (not `fps`); `audio_sample_rate` (not `sr`). | Matches existing TRT-LLM spelling and scientific-Python audio convention. |
| `VisualGenResult` method names | Sync `result()` / async `aresult()` / `__await__`. | LLM-API aligned. Breaking rename from today's async-only `result()`. |
| `cancel()` | Deferred — not in M2. | Avoid shipping a `return False` stub as part of the public surface. |
| Engine-side metrics on `VisualGenMetrics` | `pre_denoise_ms`, `denoise_ms`, `post_denoise_ms`, `pipeline_ms`. | `denoise_ms` is the dominant cost; the pre/post breakdowns let users see text-encoding/latent-prep and VAE-decode/format overhead separately. `pipeline_ms` is the executor-measured envelope around `pipeline.infer()` (≥ pre + denoise + post; the gap is host-side work). |
| Sub-phase timing methodology | **CUDA events**, not `torch.cuda.synchronize()` + `time.perf_counter()`. | Events record asynchronously — zero stall during the pipeline. `event.elapsed_time()` at the end syncs once, amortized into the existing pipeline-end sync. Tradeoff: events measure GPU-stream time only, so small host-only work (e.g., tokenization) is not in `pre_denoise_ms`. The host-vs-GPU gap shows up as `pipeline_ms - (pre + denoise + post)`, which is itself useful information. |
| End-to-end latency | **Not on `VisualGenMetrics`.** Measured externally by `trtllm-serve` and the bench script (around `generate()` + `save()`); included in their logs/reports. | E2E spans caller-side encoding/save behavior the engine doesn't own. Putting it on the result would force mutable-after-construction semantics; better to let the consumer wrap with its own `time.perf_counter()`. |
| `save()` routes to | Free functions in `tensorrt_llm/media/encoding.py` (not via `MediaStorage`). | Encoding is general-purpose; `MediaStorage` is serve-side. Keeping `save()` on a clean encoding seam avoids dragging the serve module into the Python API. |
| Encoding module shape | Free functions, not class-with-static-methods. | More Pythonic; matches diffusers `export_to_video` and torchaudio `save`. The class wrapper added nothing. |
| Encoding module location | `tensorrt_llm/media/encoding.py` (new top-level `media/` package). | Per design doc §6.3 Option B: encoding is general-purpose, not server-specific and not PyTorch-backend-specific. Internal-by-convention; not re-exported from `tensorrt_llm`. |
| `MediaStorage` location | **Stays at `tensorrt_llm/serve/media_storage.py`**. | It's a server-side concept. The encoding pieces are pulled out into the new `media/encoding.py`; whatever serve-specific responsibilities `MediaStorage` accrues (now or later) live where the rest of the server code lives. |

---

## Scope

### In scope

- New public `VisualGenOutput` flat dataclass at `tensorrt_llm/visual_gen/output.py`.
- New `VisualGenMetrics` with `pipeline_ms`, `pre_denoise_ms`, `denoise_ms`, `post_denoise_ms`.
- Rename `MediaOutput` → `PipelineOutput` everywhere; add `frame_rate`, `audio_sample_rate`, `pre_denoise_ms`, `denoise_ms`, `post_denoise_ms` fields. Remove from public re-exports.
- Each pipeline populates the metadata it owns:
  - Wan T2V / I2V → `frame_rate=16.0`, all three timing fields measured.
  - LTX-2 (both pipelines) → `frame_rate=req.params.frame_rate`, `audio_sample_rate=<looked up from LTX-2 audio config>`, all three timing fields measured.
  - Flux / Flux2 → all three timing fields measured; rate fields stay `None`.
- `VisualGenOutput._from_response(resp)` factory at the API boundary.
- `VisualGenResult` becomes a Future-like awaitable handle: `__await__`, `done`, sync `result()`, async `aresult()`.
- `pipeline_ms: float = 0.0` added to `DiffusionResponse`; executor measures it around `pipeline.infer()`.
- `generate()` returns `Union[VisualGenOutput, List[VisualGenOutput]]`.
- `generate_async()` returns `VisualGenResult`. `await future` resolves to `VisualGenOutput`.
- Batch input (`List[str]`) returns `List[VisualGenOutput]` with per-item unbatched tensors — Option A in §6.1.1.
- Batch error semantics — Option B in §5.1.1.
- `NotImplementedError` guard for `params: List[VisualGenParams]` (per §5.1.2).
- `VisualGenOutput.save(path, *, format=None, frame_rate=None, audio_sample_rate=None, quality=95)` method, routed directly to encoding free functions.
- New `tensorrt_llm/media/` package:
  - `encoding.py` — free functions: `save_image`, `save_video`, `image_to_bytes`, `video_to_bytes`, `resolve_video_format`. Plus internal helpers (`_to_pil_image`, `_save_pil_image`, `_save_encoded_video`).
- `tensorrt_llm/serve/media_storage.py` stays at its current location:
  - `save_image`, `save_video`, `convert_image_to_bytes`, `convert_video_to_bytes` — **removed** from `MediaStorage` (these are encoding; they now live in `media/encoding.py` as free functions).
  - `resolve_video_format` — moved to `media/encoding.py`.
  - Any genuinely serve-side helpers stay in `MediaStorage`. If after the encoding pull-out nothing remains, `MediaStorage` ships as a near-empty placeholder class with a clear docstring on its server-side role.
- Update every caller in one PR:
  - `examples/visual_gen/*.py` and `examples/visual_gen/models/wan_t2v.py` switch to `result.save(path)`. Drop `MediaStorage` imports. Drop hardcoded `frame_rate=...` magic numbers.
  - `bench/benchmark/visual_gen.py` switches to `result.save(path)`; consumes `result.metrics.pipeline_ms` and `result.metrics.denoise_ms`; **measures `e2e_ms` externally** by wrapping `generate()` + `save()` in `time.perf_counter()` and reports all three numbers.
  - `serve/openai_server.py` switches to `tensorrt_llm.media.encoding.save_image` / `save_video` / `image_to_bytes` / `video_to_bytes` directly. Adds an external `e2e_ms` measurement around `generate()` + encoding for its existing per-request log/report.

### Out of scope — explicitly deferred

- `seed_used` field — skipped (no randomization path yet).
- `prompt` echo field — skipped.
- `finished` field — skipped (streaming is Future E).
- `queue_ms` on metrics — not yet plumbed across the IPC boundary.
- `e2e_ms` on `VisualGenMetrics` — explicitly **not** added; consumers measure externally.
- Streaming (`stream()`, `on_progress`) — Future E.
- `VisualGenOutput.to_bytes()` / `.to_pil()` — companion encoders to `save()`. Will land as a small follow-up; the encoding free functions are already in place.
- `VisualGenResult.cancel()` — Future (when executor supports cancellation).
- Per-modality discriminator (vLLM-Omni-style `final_output_type: str`) — additive; can land later.
- OTLP tracing / `trace_headers` — Future D.
- Splitting `VisualGenParams.frame_rate` out of the shared params into LTX-2-specific `extra_params` — separate concern (§4.3); not blocked by this PR.
- Cloud storage / atomic-write features inside `MediaStorage` — reserved for a future task in `serve/`.

---

## Design

### `VisualGenMetrics`

```python
# tensorrt_llm/visual_gen/output.py
from dataclasses import dataclass


@dataclass
class VisualGenMetrics:
    """Engine-side performance measurements for a single visual generation request.

    Two measurement methodologies are used:

    - ``pipeline_ms`` is host wall-clock around ``pipeline.infer()`` (executor-measured).
    - ``pre_denoise_ms`` / ``denoise_ms`` / ``post_denoise_ms`` are GPU-stream times
      measured via CUDA events at phase boundaries inside the pipeline. They cover
      GPU work; small host-only work (e.g., tokenization) is not captured by them.

    The difference ``pipeline_ms - (pre + denoise + post)`` is the host-side overhead.
    """

    pipeline_ms: float = 0.0
    """Host wall-clock time of the generation pipeline, in milliseconds."""

    pre_denoise_ms: float = 0.0
    """GPU-stream time before the denoising step (text encoding, latent
    preparation, conditioning), in milliseconds."""

    denoise_ms: float = 0.0
    """GPU-stream time inside the denoising step, in milliseconds."""

    post_denoise_ms: float = 0.0
    """GPU-stream time after the denoising step (VAE decode, format
    conversion, audio decode), in milliseconds."""
```

The breakdown sums approximately: `pre_denoise_ms + denoise_ms + post_denoise_ms ≤ pipeline_ms`. Subtraction yields useful numbers — e.g., `pipeline_ms - denoise_ms` is the non-denoising cost; `pre_denoise_ms / pipeline_ms` is the share of time spent on text encoding / latent preparation.

End-to-end latency including caller-side encoding is **measured externally** (wrap `generate()` + `save()` in `time.perf_counter()`); the bench script and `trtllm-serve` do this for their reports. Putting it on `VisualGenMetrics` would couple the engine's data type to user-controlled save behavior.

Fields are added additively as we expand engine-side timing breakdowns — no caller breaks.

### `VisualGenOutput` (flat)

```python
# tensorrt_llm/visual_gen/output.py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.executor import DiffusionResponse


@dataclass
class VisualGenOutput:
    """The result of a single visual generation request.

    Carries the generated media (one of ``image``, ``video``, or
    ``video`` + ``audio``) along with the rate metadata needed to
    interpret it, plus the request identifier, an optional error
    message, and engine-side performance metrics.

    On success, the relevant media tensor is set and ``error`` is None.
    On failure, all media tensors are None and ``error`` carries a
    human-readable message. Batch ``generate()`` never raises — callers
    must check ``error`` first. Single-prompt ``generate()`` re-raises
    ``VisualGenError`` for back-compat.
    """

    request_id: int
    image: Optional[torch.Tensor] = None
    video: Optional[torch.Tensor] = None
    audio: Optional[torch.Tensor] = None
    frame_rate: Optional[float] = None
    audio_sample_rate: Optional[int] = None
    error: Optional[str] = None
    metrics: Optional[VisualGenMetrics] = None

    def save(
        self,
        path: Union[str, Path],
        *,
        format: Optional[str] = None,
        frame_rate: Optional[float] = None,
        audio_sample_rate: Optional[int] = None,
        quality: int = 95,
    ) -> Path:
        """Save the generated media to ``path``.

        The encoding format is inferred from the file extension unless
        ``format`` is given explicitly. For video outputs, audio is muxed
        in automatically when present.

        Rates carried on this output are used as defaults; pass
        ``frame_rate=`` or ``audio_sample_rate=`` to override (e.g., for
        slow-motion or pitch-shifted encoding). Returns the path that
        was written.

        Raises:
            VisualGenError: If this output carries no media (e.g., the
                request failed) or if a video output has no frame rate.
        """
        ...

    @classmethod
    def _from_response(cls, resp: "DiffusionResponse") -> "VisualGenOutput":
        """Construct a VisualGenOutput from the executor's response."""
        if resp.error_msg or resp.output is None:
            return cls(request_id=resp.request_id, error=resp.error_msg)
        media = resp.output
        metrics = VisualGenMetrics(
            pipeline_ms=resp.pipeline_ms,
            pre_denoise_ms=media.pre_denoise_ms or 0.0,
            denoise_ms=media.denoise_ms or 0.0,
            post_denoise_ms=media.post_denoise_ms or 0.0,
        )
        return cls(
            request_id=resp.request_id,
            image=media.image,
            video=media.video,
            audio=media.audio,
            frame_rate=media.frame_rate,
            audio_sample_rate=media.audio_sample_rate,
            metrics=metrics,
        )
```

`@dataclass` (not `StrictBaseModel`) — keeps `torch.Tensor` flowing through without Pydantic juggling.

### `VisualGenOutput.save` — dispatch rules

The implementation routes directly to free functions in `tensorrt_llm.media.encoding`:

```python
from pathlib import Path

from tensorrt_llm.visual_gen.errors import VisualGenError


def save(self, path, *, format=None, frame_rate=None, audio_sample_rate=None, quality=95) -> Path:
    # Lazy import to avoid pulling in ffmpeg/imageio at API import time.
    from tensorrt_llm.media.encoding import save_image, save_video

    if self.error is not None:
        raise VisualGenError(f"Cannot save: {self.error}")

    path = Path(path)
    has_image = self.image is not None
    has_video = self.video is not None
    if not (has_image or has_video):
        raise VisualGenError("Cannot save: this output carries no media.")

    if has_image:
        return save_image(self.image, path, format=format, quality=quality)

    rate = frame_rate or self.frame_rate
    if rate is None:
        raise VisualGenError(
            "Cannot save video: no frame_rate available. Pass frame_rate= explicitly."
        )
    audio_sr = audio_sample_rate or self.audio_sample_rate
    return save_video(
        self.video, path,
        audio=self.audio,
        frame_rate=rate,
        audio_sample_rate=audio_sr,
        format=format,
    )
```

Notes:

- Image-only outputs ignore `frame_rate` / `audio_sample_rate` kwargs.
- Saving a video output to an image extension (or vice versa) lets the encoder raise a clear format error — no separate validation needed.
- The encoding module is imported lazily to avoid a top-of-module dependency on ffmpeg/imageio at API import time.

### `PipelineOutput` (renamed from `MediaOutput`, internal)

```python
# tensorrt_llm/_torch/visual_gen/output.py
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class PipelineOutput:
    """Pipeline-internal result type. Not part of the public API.

    Different pipelines populate different fields:
    - image-only models (FLUX, FLUX2): ``image`` only.
    - video-only models (Wan T2V / I2V): ``video`` and ``frame_rate``.
    - video-plus-audio models (LTX-2): ``video``, ``audio``, ``frame_rate``,
      ``audio_sample_rate``.

    All pipelines populate the three timing fields (wall-clock around
    each phase): ``pre_denoise_ms`` (text encoding, latent preparation,
    conditioning), ``denoise_ms`` (the denoising loop), and
    ``post_denoise_ms`` (VAE decode, format conversion, audio decode).
    """

    image: Optional[torch.Tensor] = None
    video: Optional[torch.Tensor] = None
    audio: Optional[torch.Tensor] = None
    frame_rate: Optional[float] = None
    audio_sample_rate: Optional[int] = None
    pre_denoise_ms: Optional[float] = None
    denoise_ms: Optional[float] = None
    post_denoise_ms: Optional[float] = None
```

Every reference to `MediaOutput` in the codebase is renamed in this PR (~25 sites: 6 pipelines, the executor, the wire dataclass field type, tests, and `_torch/visual_gen/__init__.py`).

### `VisualGenResult` (Future-like awaitable handle)

```python
# tensorrt_llm/visual_gen/visual_gen.py
from typing import Optional


class VisualGenResult:
    """An awaitable handle to one in-flight visual generation request.

    Awaiting the handle (or calling ``aresult()`` / ``result()``) returns
    the corresponding :class:`VisualGenOutput`. Failures surface as
    :class:`VisualGenError` raised from the wait call.
    """

    def __init__(self, request_id: int, executor: "DiffusionRemoteClient"):
        self.request_id = request_id
        self._executor = executor
        self._result: Optional[VisualGenOutput] = None
        self._error: Optional[str] = None
        self._finished = False

    @property
    def done(self) -> bool:
        """Whether generation has completed."""
        return self._finished

    def result(self, timeout: Optional[float] = None) -> "VisualGenOutput":
        """Block until generation completes. Raises on failure."""
        ...

    async def aresult(self, timeout: Optional[float] = None) -> "VisualGenOutput":
        """Asynchronously wait for generation to complete. Raises on failure."""
        ...

    def __await__(self):
        return self.aresult().__await__()
```

This is a **breaking rename** from today's async-only `DiffusionGenerationResult.result()`. Migration is mechanical: `await future.result()` → `await future` (preferred) or `await future.aresult()`.

### Batch output shape — §6.1.1 reminder

After this task:

- `str` input → one `VisualGenOutput`.
- `List[str]` input → `List[VisualGenOutput]`, length N, each with unbatched flat tensors (sliced along dim 0 from the underlying batched `PipelineOutput`).

Internal batching is preserved — the pipeline still runs one batched forward; we split client-side.

### Batch error semantics — §5.1.1 reminder

**Option B**: on partial failure, `generate()` returns a full-length list with `error` set on failed items, never raises. Common failure modes today:

- `VisualGenParamsError` from `DiffusionExecutor._validate_request` — applies to the entire batch.
- Pipeline `infer()` raises mid-execution — same: all N items fail.

Per-item failures aren't reachable today (every item shares the same `params`). Adopting Option B now keeps the contract stable for the future.

### Single vs batch raise behavior

- `generate("a cat")` → on error, raise `VisualGenError(error_msg)`. On success, return one `VisualGenOutput` with `error=None`.
- `generate(["a cat", "a dog"])` → never raises; returns `[VisualGenOutput(error=...), ...]`.

Documented in the docstring and codified in tests.

### Validation at `generate()` boundary (§5.1.2)

Widen the `params` type to accept (but reject with `NotImplementedError`) a list:

```python
def generate(
    self,
    inputs: Union[str, List[str]],
    params: Optional[Union[VisualGenParams, List[VisualGenParams]]] = None,
) -> Union[VisualGenOutput, List[VisualGenOutput]]:
```

If `isinstance(params, list)`, raise `NotImplementedError` with a §5.1.2 pointer.

### Encoding module — §6.3 Option B (free functions)

`tensorrt_llm/media/encoding.py` (new) holds the encoder logic as **free functions**:

```python
# tensorrt_llm/media/encoding.py
from pathlib import Path
from typing import Optional, Union

import torch


def save_image(
    image: torch.Tensor,
    path: Union[str, Path],
    *,
    format: Optional[str] = None,
    quality: int = 95,
) -> Path:
    """Encode an image tensor to ``path``. Format is inferred from
    the extension unless given explicitly. Returns the actual path written."""
    ...


def save_video(
    video: torch.Tensor,
    path: Union[str, Path],
    *,
    audio: Optional[torch.Tensor] = None,
    frame_rate: float,
    audio_sample_rate: Optional[int] = None,
    format: Optional[str] = None,
) -> Path:
    """Encode a video tensor to ``path``, muxing audio if provided.
    Format is inferred from the extension unless given explicitly.
    Returns the actual path written."""
    ...


def image_to_bytes(
    image: torch.Tensor,
    *,
    format: str = "PNG",
    quality: int = 95,
) -> bytes:
    """Encode an image tensor to bytes (e.g., for HTTP response bodies)."""
    ...


def video_to_bytes(
    video: torch.Tensor,
    *,
    audio: Optional[torch.Tensor] = None,
    frame_rate: float,
    audio_sample_rate: Optional[int] = None,
    format: str = "mp4",
) -> bytes:
    """Encode a video tensor to bytes."""
    ...


def resolve_video_format(path_or_format: Union[str, Path, None]) -> str:
    """Map a path or format string to a canonical encoder format identifier."""
    ...
```

Internal helpers (`_to_pil_image`, `_save_pil_image`, `_save_encoded_video`) move to this file as module-level private functions. The free functions are not re-exported from `tensorrt_llm/__init__.py`; they're consumed by `VisualGenOutput.save()` and by `serve/openai_server.py` directly via explicit module-path imports.

### `MediaStorage` — stays in `serve/`, encoding methods removed

`tensorrt_llm/serve/media_storage.py` is **kept at its current location**. Its tensor-encoding methods (`save_image`, `save_video`, `convert_image_to_bytes`, `convert_video_to_bytes`) are removed; that logic now lives as free functions in `tensorrt_llm/media/encoding.py`. `resolve_video_format` (a free function previously in this file) moves to `media/encoding.py` too.

`MediaStorage`'s remaining role is server-side: a placeholder for future serve-specific storage concerns (e.g., output-directory policies, atomic writes, cloud-storage integrations) that may live in this class. If after the encoding pull-out the class has no remaining methods, it ships as an empty class with a docstring describing its reserved role; cloud / atomic-write features land additively in a future task.

---

## Implementation plan

### Files touched

| File | Change |
| :--- | :--- |
| `tensorrt_llm/_torch/visual_gen/output.py` | Rename `MediaOutput` → `PipelineOutput`. Add `frame_rate`, `audio_sample_rate`, `pre_denoise_ms`, `denoise_ms`, `post_denoise_ms` fields. Update docstring. |
| `tensorrt_llm/_torch/visual_gen/__init__.py` | Re-export `PipelineOutput` (internal-only path). Drop `MediaOutput`. |
| `tensorrt_llm/_torch/visual_gen/executor.py` | `DiffusionResponse.output: Optional[PipelineOutput]`; add `pipeline_ms: float = 0.0`; measure it around `pipeline.infer()`. |
| `tensorrt_llm/_torch/visual_gen/models/flux/pipeline_flux.py`, `pipeline_flux2.py` | Return `PipelineOutput(image=..., pre_denoise_ms=..., denoise_ms=..., post_denoise_ms=...)`; instrument all three phases. |
| `tensorrt_llm/_torch/visual_gen/models/wan/pipeline_wan.py`, `pipeline_wan_i2v.py` | Return `PipelineOutput(video=..., frame_rate=16.0, pre_denoise_ms=..., denoise_ms=..., post_denoise_ms=...)`. |
| `tensorrt_llm/_torch/visual_gen/models/ltx2/pipeline_ltx2.py`, `pipeline_ltx2_two_stages.py` | Return `PipelineOutput(video=..., audio=..., frame_rate=req.params.frame_rate, audio_sample_rate=<lookup>, pre_denoise_ms=..., denoise_ms=..., post_denoise_ms=...)`. |
| `tensorrt_llm/visual_gen/output.py` (new) | Define `VisualGenOutput` (with `save()` and `_from_response`), `VisualGenMetrics`, `_split_visual_gen_output`. |
| `tensorrt_llm/visual_gen/__init__.py` | Export `VisualGenOutput`, `VisualGenMetrics`. Drop `MediaOutput` from `__all__`. |
| `tensorrt_llm/__init__.py` | Re-export `VisualGenOutput`, `VisualGenMetrics`. Drop `MediaOutput` re-export. |
| `tensorrt_llm/visual_gen/visual_gen.py` | Promote `VisualGenResult` to Future-like. `generate()` batch fan-out + return-type change; `NotImplementedError` guard on `List[VisualGenParams]`. |
| `tensorrt_llm/media/__init__.py` (new) | Empty / minimal. |
| `tensorrt_llm/media/encoding.py` (new) | `save_image`, `save_video`, `image_to_bytes`, `video_to_bytes`, `resolve_video_format` as free functions. Internal helpers (`_to_pil_image`, `_save_pil_image`, `_save_encoded_video`) live here. |
| `tensorrt_llm/serve/media_storage.py` | **Stays at this location.** Remove `save_image`, `save_video`, `convert_image_to_bytes`, `convert_video_to_bytes`, `resolve_video_format`. Anything genuinely serve-side stays. If empty after the pull-out, ship a placeholder class with a docstring on its reserved server-side role. |
| `tensorrt_llm/serve/openai_server.py` | At 9 result-handling sites: `output.image` → `result.image`, `output.video` → `result.video`. Check `result.error` before access. Replace hardcoded fps with `result.frame_rate`; pass `audio_sample_rate=result.audio_sample_rate`. Replace `MediaStorage.save_image` / `MediaStorage.save_video` / `MediaStorage.convert_image_to_bytes` / `MediaStorage.convert_video_to_bytes` calls with `tensorrt_llm.media.encoding.save_image` / `save_video` / `image_to_bytes` / `video_to_bytes`. Add an external `time.perf_counter()` measurement around `generate()` + encoding for the per-request log/report (e2e). |
| `tensorrt_llm/bench/benchmark/visual_gen.py` | At 3 sites: replace try/except on `VisualGenError` with `if result.error:` for batch paths; keep try/except for warmup. Use `result.save(path)` instead of `MediaStorage`. Wrap each request in `time.perf_counter()` to compute `e2e_ms` externally. Report includes `result.metrics.denoise_ms`, `result.metrics.pipeline_ms`, and the externally-measured `e2e_ms`. |
| `examples/visual_gen/quickstart_example.py`, `visual_gen_flux.py`, `visual_gen_ltx2.py`, `visual_gen_wan_i2v.py`, `visual_gen_wan_t2v.py`, `examples/visual_gen/models/wan_t2v.py` | Replace `MediaStorage.save_*(...)` with `result.save(path)`. Drop `MediaStorage` import. Drop hardcoded `frame_rate=...` magic numbers. |
| `tests/integration/defs/examples/test_visual_gen.py` | Update expected shapes; assert rate fields where applicable. |
| `tests/unittest/api_stability/references/*.yaml` | Accept return-type diff for `VisualGen.generate`, `generate_async`, `VisualGenResult.result`/`aresult`/`done`/`__await__`, `VisualGenOutput.save`. |
| `tests/unittest/_torch/visual_gen/test_trtllm_serve_endpoints.py` | Rename `MediaOutput` → `PipelineOutput` in fakes. Update `MediaStorage` usage if any encoding methods are referenced (use `tensorrt_llm.media.encoding` free functions instead). |
| `tests/unittest/dynamo/test_imports.py` | Rename `MediaOutput` → `PipelineOutput`. |
| `tests/unittest/visual_gen/test_output.py` (new) | Unit tests for the new types + factory + batch split + `save()`. |

### Step-by-step

**1. Rename `MediaOutput` → `PipelineOutput` and add metadata fields.**

`tensorrt_llm/_torch/visual_gen/output.py`:

```python
@dataclass
class PipelineOutput:
    image: Optional[torch.Tensor] = None
    video: Optional[torch.Tensor] = None
    audio: Optional[torch.Tensor] = None
    frame_rate: Optional[float] = None
    audio_sample_rate: Optional[int] = None
    pre_denoise_ms: Optional[float] = None
    denoise_ms: Optional[float] = None
    post_denoise_ms: Optional[float] = None
```

Update docstring per the conceptual style.

`git grep -l MediaOutput` and replace globally. Sites: 6 pipeline files, executor.py (`DiffusionResponse.output: Optional[PipelineOutput]`), `_torch/visual_gen/__init__.py`, two test files.

**2. Pipelines populate metadata fields and instrument the three timing phases.**

Pattern (each pipeline file): use **CUDA events**, not host syncs. Events record asynchronously on the GPU stream — no stalls during the pipeline; the only sync happens implicitly when `elapsed_time()` is called at the end (and is amortized into the executor-side sync that already exists when the response is consumed).

```python
import torch

def infer(self, req):
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e2 = torch.cuda.Event(enable_timing=True)
    e3 = torch.cuda.Event(enable_timing=True)

    e0.record()

    # --- pre-denoise: text encoding, latent prep, conditioning ---
    text_embeddings = self.text_encoder(...)
    latents = self._prepare_latents(...)

    e1.record()

    # --- denoise loop ---
    for step in denoising_loop:
        ...

    e2.record()

    # --- post-denoise: VAE decode, format conversion, audio decode ---
    video = self.vae.decode(latents)
    video = postprocess_video_tensor(video)

    e3.record()

    # elapsed_time() syncs on first call; the rest are essentially free
    # because the GPU is already at e3 by then.
    pre_denoise_ms = e0.elapsed_time(e1)
    denoise_ms = e1.elapsed_time(e2)
    post_denoise_ms = e2.elapsed_time(e3)

    return PipelineOutput(
        video=video,
        frame_rate=16.0,             # Wan-specific constant
        pre_denoise_ms=pre_denoise_ms,
        denoise_ms=denoise_ms,
        post_denoise_ms=post_denoise_ms,
    )
```

CUDA events measure **GPU-stream time** between record points. For diffusion this closely tracks wall-clock because the GPU is the dominant cost; small host-side work in pre-denoise (e.g., tokenization) is not captured by the events. Users who want the host+GPU envelope read `pipeline_ms` (executor-measured); the difference `pipeline_ms - (pre + denoise + post)` is the host-side overhead. Document the methodology in the `VisualGenMetrics` docstring.

Per-pipeline values:

- `pipeline_flux.py:362`, `pipeline_flux2.py:453` — image only; populate all three timing fields.
- `pipeline_wan.py:489`, `pipeline_wan_i2v.py:637` — `frame_rate=16.0`, all three timing fields.
- `pipeline_ltx2.py:1848`, `pipeline_ltx2_two_stages.py:756` — `frame_rate=req.params.frame_rate`, `audio_sample_rate=<lookup from LTX-2 audio config>`, all three timing fields. For `_two_stages`, the two stages either count as one combined `denoise_ms` (recommended for users) or split — pick during implementation; document the choice in the pipeline docstring.
- `pipeline_ltx2_two_stages.py:674` (early-out failure) — leave rate and timing fields `None` or 0.

The exact LTX-2 audio sample rate is read from the model's audio config during implementation; do not hardcode without verifying the decoder's actual rate.

**3. Add the public types.**

Create `tensorrt_llm/visual_gen/output.py` with `VisualGenMetrics` (4 fields), `VisualGenOutput` (with `save()` and `_from_response`), and `_split_visual_gen_output`.

Update `tensorrt_llm/visual_gen/__init__.py` to export `VisualGenOutput`, `VisualGenMetrics`. Re-export from `tensorrt_llm/__init__.py`.

**4. Build the new `media/` package; pull encoding out of `serve/media_storage.py`.**

- Create `tensorrt_llm/media/__init__.py` (minimal).
- Create `tensorrt_llm/media/encoding.py` with the four encoding free functions plus `resolve_video_format`. Internal helpers (`_to_pil_image`, `_save_pil_image`, `_save_encoded_video`) move here as module-level private functions.
- Edit `tensorrt_llm/serve/media_storage.py`: remove the moved methods (`save_image`, `save_video`, `convert_image_to_bytes`, `convert_video_to_bytes`) and the `resolve_video_format` free function. If `MediaStorage` is empty after this, leave it as a placeholder class with a docstring on its server-side role. Do not delete the file.

**5. Measure `pipeline_ms` and add to wire type.**

`tensorrt_llm/_torch/visual_gen/executor.py`:

```python
@dataclass
class DiffusionResponse:
    request_id: int
    output: Optional[PipelineOutput] = None
    error_msg: Optional[str] = None
    pipeline_ms: float = 0.0   # NEW

def process_request(self, req):
    try:
        ...
        t0 = time.perf_counter()
        output = self.pipeline.infer(req)
        pipeline_ms = (time.perf_counter() - t0) * 1000.0
        if self.rank == 0:
            self.response_queue.put(DiffusionResponse(
                request_id=req.request_id, output=output, pipeline_ms=pipeline_ms,
            ))
    except Exception as e:
        ...  # error path leaves pipeline_ms at 0.0
```

**6. Promote `VisualGenResult` to Future-like.**

```python
class VisualGenResult:
    def __init__(self, request_id, executor):
        self.request_id = request_id
        self._executor = executor
        self._result = None
        self._error = None
        self._finished = False

    @property
    def done(self) -> bool:
        return self._finished

    def result(self, timeout=None) -> VisualGenOutput:
        if self._finished:
            if self._error:
                raise VisualGenError(self._error)
            return self._result
        return self._consume(self._executor.await_responses_sync(self.request_id, timeout=timeout))

    async def aresult(self, timeout=None) -> VisualGenOutput:
        if self._finished:
            if self._error:
                raise VisualGenError(self._error)
            return self._result
        future = asyncio.run_coroutine_threadsafe(
            self._executor.await_responses(self.request_id, timeout=timeout),
            self._executor._event_loop,
        )
        return self._consume(await asyncio.wrap_future(future))

    def __await__(self):
        return self.aresult().__await__()

    def _consume(self, response) -> VisualGenOutput:
        if response is None:
            raise VisualGenError("Generation timed out")
        self._result = VisualGenOutput._from_response(response)
        self._finished = True
        if response.error_msg:
            self._error = response.error_msg
            raise VisualGenError(f"Generation failed: {response.error_msg}")
        return self._result
```

Audit `git grep "\.result(" -- tensorrt_llm/ examples/ tests/` and update every call site. Recommended migration: `await future` (cleanest); fallback `await future.aresult()`.

**7. Change `generate()` signature; implement single via the future.**

```python
@set_api_status("prototype")
def generate(
    self,
    inputs: Union[str, List[str]],
    params: Optional[Union[VisualGenParams, List[VisualGenParams]]] = None,
) -> Union[VisualGenOutput, List[VisualGenOutput]]:
    if isinstance(params, list):
        raise NotImplementedError(
            "Per-request params (List[VisualGenParams]) is not supported in the "
            "current milestone; see design doc §5.1.2 'Batching Support Scope'. "
            "Pass a single VisualGenParams shared across the batch."
        )

    if isinstance(inputs, str):
        return self.generate_async(inputs=inputs, params=params).result()

    if isinstance(inputs, (list, tuple)):
        return self._generate_batch(list(inputs), params)

    raise ValueError(f"Invalid inputs type: {type(inputs)}")
```

**8. Implement the batch path with client-side splitting.**

```python
def _generate_batch(self, prompts, params) -> List[VisualGenOutput]:
    if not prompts:
        raise ValueError("Batch inputs must contain at least one item")
    if not all(isinstance(p, str) for p in prompts):
        raise ValueError("Batch inputs must contain only strings")

    req_id = next(self._req_counter)
    request = DiffusionRequest(request_id=req_id, prompt=prompts, params=params)
    self.executor.enqueue_requests([request])

    resp = self.executor.await_responses_sync(req_id, timeout=None)

    if resp.error_msg or resp.output is None:
        metrics = VisualGenMetrics(pipeline_ms=resp.pipeline_ms)
        return [
            VisualGenOutput(request_id=req_id, error=resp.error_msg, metrics=metrics)
            for _ in prompts
        ]

    return _split_visual_gen_output(
        resp.output, n=len(prompts), request_id=req_id,
        pipeline_ms=resp.pipeline_ms,
    )
```

Helper at `tensorrt_llm/visual_gen/output.py`:

```python
def _split_visual_gen_output(media, n, request_id, pipeline_ms):
    """Slice a batched PipelineOutput along the leading dim into N flat
    VisualGenOutputs. Frame rate, audio sample rate, and the three timing
    fields are pipeline-level constants and repeat unchanged on every
    per-item output.
    """
    def _slice(t):
        if t is None:
            return [None] * n
        if t.shape[0] != n:
            raise VisualGenError(
                f"Pipeline returned tensor with leading dim {t.shape[0]}, "
                f"expected {n} (batch size). This is a pipeline bug."
            )
        return [t[i] for i in range(n)]

    images = _slice(media.image)
    videos = _slice(media.video)
    audios = _slice(media.audio)
    metrics = VisualGenMetrics(
        pipeline_ms=pipeline_ms,
        pre_denoise_ms=media.pre_denoise_ms or 0.0,
        denoise_ms=media.denoise_ms or 0.0,
        post_denoise_ms=media.post_denoise_ms or 0.0,
    )
    return [
        VisualGenOutput(
            request_id=request_id,
            image=images[i],
            video=videos[i],
            audio=audios[i],
            frame_rate=media.frame_rate,
            audio_sample_rate=media.audio_sample_rate,
            metrics=metrics,
        )
        for i in range(n)
    ]
```

**9. Implement `VisualGenOutput.save`.**

Per the dispatch rules above. Lazy-import the encoding free functions inside the method body. The method does not touch `metrics`; e2e timing is the consumer's job.

**10. Update every caller.**

- **`serve/openai_server.py`**: switch result-handling sites to `result.image` / `result.video` / `result.error`. Replace hardcoded fps with `result.frame_rate`; pass `audio_sample_rate=result.audio_sample_rate`. Replace every `MediaStorage.save_*` / `MediaStorage.convert_*_to_bytes` call with the corresponding free function from `tensorrt_llm.media.encoding`. Wrap each request in `time.perf_counter()` to compute `e2e_ms` for the existing per-request log line.
- **`bench/benchmark/visual_gen.py`** (3 sites): `if result.error:` for batch paths; keep try/except for warmup. Replace `MediaStorage` with `result.save(path)`. Wrap each iteration in `time.perf_counter()` to compute `e2e_ms` externally. Extend the report rows with `result.metrics.pre_denoise_ms`, `result.metrics.denoise_ms`, `result.metrics.post_denoise_ms`, `result.metrics.pipeline_ms`, and the externally-measured `e2e_ms`.
- **`examples/visual_gen/*.py`** (5 files + `models/wan_t2v.py`): mechanical rewrite to `result.save(path)`. Drop `MediaStorage` import. Drop hardcoded `frame_rate=...`.
- **Tests**: integration-test assertions on return shapes and rate fields. API-stability YAML. Test fakes get `MediaOutput` → `PipelineOutput` rename and any encoding-method references updated.

---

## Tests

### Unit (`tests/unittest/visual_gen/test_output.py`)

```python
def test_visual_gen_output_image_shape():
    out = VisualGenOutput(
        request_id=42,
        image=torch.zeros(3, 64, 64, dtype=torch.uint8),
        metrics=VisualGenMetrics(
            pipeline_ms=12.3, pre_denoise_ms=1.5, denoise_ms=10.0, post_denoise_ms=0.6
        ),
    )
    assert out.error is None
    assert out.image.shape == (3, 64, 64)
    assert out.video is None and out.audio is None
    assert out.frame_rate is None and out.audio_sample_rate is None
    assert out.metrics.pipeline_ms == 12.3
    assert out.metrics.pre_denoise_ms == 1.5
    assert out.metrics.denoise_ms == 10.0
    assert out.metrics.post_denoise_ms == 0.6


def test_visual_gen_output_video_with_rate():
    out = VisualGenOutput(
        request_id=1,
        video=torch.zeros(8, 32, 32, 3, dtype=torch.uint8),
        frame_rate=16.0,
    )
    assert out.frame_rate == 16.0
    assert out.audio is None


def test_visual_gen_output_video_audio_with_rates():
    out = VisualGenOutput(
        request_id=2,
        video=torch.zeros(8, 32, 32, 3, dtype=torch.uint8),
        audio=torch.zeros(2, 1024, dtype=torch.float32),
        frame_rate=24.0,
        audio_sample_rate=44100,
    )
    assert out.frame_rate == 24.0
    assert out.audio_sample_rate == 44100


def test_visual_gen_output_error_shape():
    out = VisualGenOutput(request_id=42, error="boom")
    assert out.image is None and out.video is None and out.audio is None
    assert out.error == "boom"
    assert out.metrics is None


def test_from_response_success_carries_all_metadata():
    media = PipelineOutput(
        video=torch.zeros(8, 16, 16, 3, dtype=torch.uint8),
        frame_rate=16.0,
        pre_denoise_ms=1.2,
        denoise_ms=8.5,
        post_denoise_ms=2.1,
    )
    resp = DiffusionResponse(request_id=7, output=media, pipeline_ms=12.0)
    out = VisualGenOutput._from_response(resp)
    assert out.request_id == 7
    assert out.frame_rate == 16.0
    assert out.metrics.pipeline_ms == 12.0
    assert out.metrics.pre_denoise_ms == 1.2
    assert out.metrics.denoise_ms == 8.5
    assert out.metrics.post_denoise_ms == 2.1


def test_from_response_error():
    resp = DiffusionResponse(request_id=7, output=None, error_msg="boom", pipeline_ms=0.0)
    out = VisualGenOutput._from_response(resp)
    assert out.image is None and out.video is None and out.audio is None
    assert out.error == "boom"
    assert out.metrics is None


def test_split_visual_gen_output_image():
    batched = PipelineOutput(image=torch.arange(2 * 4).reshape(2, 2, 2).to(torch.uint8))
    items = _split_visual_gen_output(batched, n=2, request_id=99, pipeline_ms=1.0)
    assert len(items) == 2
    assert all(item.request_id == 99 for item in items)
    assert items[0].image.shape == (2, 2)
    assert items[1].video is None


def test_split_visual_gen_output_video_propagates_rate_and_timings():
    batched = PipelineOutput(
        video=torch.zeros(2, 4, 8, 8, 3, dtype=torch.uint8),
        frame_rate=16.0,
        pre_denoise_ms=0.8,
        denoise_ms=4.2,
        post_denoise_ms=1.1,
    )
    items = _split_visual_gen_output(batched, n=2, request_id=1, pipeline_ms=5.0)
    assert items[0].frame_rate == 16.0
    assert items[0].metrics.pre_denoise_ms == 0.8
    assert items[0].metrics.denoise_ms == 4.2
    assert items[0].metrics.post_denoise_ms == 1.1
    assert items[1].metrics.pre_denoise_ms == 0.8


def test_save_image_calls_encoding_save_image(tmp_path, monkeypatch):
    captured = {}
    def fake_save_image(image, path, *, format=None, quality=95):
        captured.update(image=image, path=path, format=format, quality=quality)
        return path
    monkeypatch.setattr("tensorrt_llm.media.encoding.save_image", fake_save_image)

    out = VisualGenOutput(request_id=1, image=torch.zeros(3, 8, 8, dtype=torch.uint8))
    out.save(tmp_path / "x.png")
    assert captured["path"].name == "x.png"


def test_save_video_uses_carried_frame_rate(tmp_path, monkeypatch):
    captured = {}
    def fake_save_video(video, path, *, audio=None, frame_rate, audio_sample_rate=None, format=None):
        captured.update(frame_rate=frame_rate, audio_sample_rate=audio_sample_rate)
        return path
    monkeypatch.setattr("tensorrt_llm.media.encoding.save_video", fake_save_video)

    out = VisualGenOutput(
        request_id=1,
        video=torch.zeros(4, 8, 8, 3, dtype=torch.uint8),
        frame_rate=16.0,
    )
    out.save(tmp_path / "x.mp4")
    assert captured["frame_rate"] == 16.0


def test_save_kwarg_overrides_frame_rate(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "tensorrt_llm.media.encoding.save_video",
        lambda *a, frame_rate, **kw: captured.setdefault("rate", frame_rate) or a[1],
    )
    out = VisualGenOutput(
        request_id=1,
        video=torch.zeros(4, 8, 8, 3, dtype=torch.uint8),
        frame_rate=16.0,
    )
    out.save(tmp_path / "x.mp4", frame_rate=30.0)
    assert captured["rate"] == 30.0


def test_save_raises_on_error_output(tmp_path):
    out = VisualGenOutput(request_id=1, error="boom")
    with pytest.raises(VisualGenError):
        out.save(tmp_path / "x.png")


def test_save_raises_on_video_without_rate(tmp_path):
    out = VisualGenOutput(request_id=1, video=torch.zeros(4, 8, 8, 3, dtype=torch.uint8))
    with pytest.raises(VisualGenError):
        out.save(tmp_path / "x.mp4")


def test_visual_gen_result_is_awaitable(stub_executor):
    fut = VisualGenResult(request_id=5, executor=stub_executor)
    out = asyncio.run(fut)             # __await__ → aresult → VisualGenOutput
    assert isinstance(out, VisualGenOutput)
    assert fut.done is True
```

### Integration (small-model end-to-end, Flux is cheapest)

- Flux: `result = engine.generate("a cat")` — assert `result.image is not None`, `result.error is None`, `result.metrics.pipeline_ms > 0`, `result.metrics.denoise_ms > 0`, `result.metrics.pre_denoise_ms > 0`, `result.metrics.post_denoise_ms > 0`, and the breakdown sums close to `pipeline_ms` (allow some slack). `result.frame_rate is None`. Then `result.save(tmp / "x.png")` writes a valid file.
- Flux batch: `results = engine.generate(["a cat", "a dog"])` — `len == 2`, both `error is None`, independent `image` tensors.
- Wan T2V: `result.video is not None`, `result.frame_rate == 16.0`, `result.audio is None`. `result.save(tmp / "x.mp4")` writes a valid file.
- LTX-2: `result.video is not None`, `result.audio is not None`, rate fields populated, `result.metrics.denoise_ms > 0`.
- Error path: `VisualGenParams(extra_params={"nonsense_key": 1})`, `generate(["a cat", "a dog"])` — both items have `error` set with the same executor-level message.
- `NotImplementedError` guard: `engine.generate("a cat", params=[VisualGenParams()])` raises mentioning §5.1.2.
- Async path: `future = engine.generate_async("a cat"); out = asyncio.run(future)` returns a `VisualGenOutput`. `future.done is True`.

### API stability

`pytest tests/unittest/api_stability` — reference YAML for `VisualGen.generate`, `VisualGen.generate_async`, `VisualGenResult.result`/`aresult`/`done`/`__await__`, `VisualGenOutput.save`, `VisualGenMetrics` fields, all updated in the same PR.

---

## Risks and mitigations

| Risk | Mitigation |
| :--- | :--- |
| Pipeline returns a tensor whose leading dim isn't the batch (e.g., LTX-2 audio is shared) | `_split_visual_gen_output` asserts `t.shape[0] == n` and raises `VisualGenError`. Catch during the per-model smoke test. If audio is intentionally shared, branch in `_slice` to share the reference. |
| Tensor-sliced views keep the full batched tensor alive | Negligible at expected batch sizes. If it ever matters, replace `t[i]` with `t[i].clone().contiguous()`. |
| Single vs batch raise asymmetry confuses callers | Documented in `generate()` docstring; codified in unit tests. |
| Per-phase instrumentation perturbs perf | Use **CUDA events** (`torch.cuda.Event(enable_timing=True)`), not host-side `torch.cuda.synchronize()` + `time.perf_counter()`. Events record asynchronously on the GPU stream — the pipeline doesn't stall during execution. The only sync happens implicitly when `event.elapsed_time()` is called at the end, and that sync is amortized into the executor-side sync that already runs when the response is consumed. Verify zero-overhead with a benchmark before/after on Flux. |
| Pipeline phase boundaries differ across models (e.g., LTX-2 two-stage has two denoise loops) | Pipeline author picks the natural split and documents it in the pipeline docstring. Tests assert only that the three numbers are non-zero on success and roughly sum to `pipeline_ms`; they do not assert specific values for each phase. |
| `MediaOutput → PipelineOutput` rename is a wide diff | Mechanical; `git grep` covers all sites. PR title and description call out the rename. |
| `VisualGenResult.result()` rename (was async, now sync) breaks downstream | Audited and updated in-tree. External callers using `await future.result()` migrate to `await future` or `await future.aresult()`. PR description flags the migration. |
| `MediaStorage.save_*` / `convert_*_to_bytes` removal breaks any out-of-tree importer | Acceptable per the decision to not ship a deprecation shim. `MediaStorage` itself stays at `tensorrt_llm.serve.media_storage`; only the encoding methods move. Release notes call out that encoding callers should switch to `tensorrt_llm.media.encoding`. |
| LTX-2 audio sample rate not constant across configs | Look it up from the LTX-2 audio config in the pipeline. Do not hardcode. |
| `DiffusionResponse.pipeline_ms` field addition breaks pickling across worker/client version skew | Workers and client are spawned from the same process — no version skew. Default `0.0` preserves forward compatibility. |
| `PipelineOutput.denoise_ms` mixes timing into the pipeline-output type | The rename to `PipelineOutput` is precisely to make this fit conceptually — the dataclass now represents "what the pipeline produced", which includes both the media payload and pipeline-side timing. If more timings land later, we can refactor to a sub-struct then. |
| `MediaStorage` ends up empty after pulling encoding out | Stays at `tensorrt_llm/serve/media_storage.py` as a placeholder class with a clear docstring on its server-side role. Cloud / atomic-write features land additively in a future task. |

---

## Design-doc references

- §5.1.1 Batch Error Semantics — `visual-gen-api-refactor-m2.md:570`
- §5.1.2 Batching Support Scope (Current Milestone) — `visual-gen-api-refactor-m2.md:582`
- §5.2 Async Handle (Future-like contract) — `visual-gen-api-refactor-m2.md:604`
- §6.1 Output Object Design — `visual-gen-api-refactor-m2.md:686`
- §6.1.1 Batch Output Shape — `visual-gen-api-refactor-m2.md:755`
- §6.2 `MediaOutput` Convenience Methods & Encoding Format — `visual-gen-api-refactor-m2.md:828`
- §6.3 Where to Put the Encoding Implementation — `visual-gen-api-refactor-m2.md:921`

--- Original Design Draft End ---
