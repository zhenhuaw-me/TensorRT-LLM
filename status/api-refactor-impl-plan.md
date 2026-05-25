# VisualGen API Refactor — Implementation Plan

Source design doc: [visual-gen-api-refactor-m2.md](../designs/visual-gen-api-refactor-m2.md) | Jira: [TRTLLM-10897](https://jirasw.nvidia.com/browse/TRTLLM-10897)

> **Status as of 2026-04-23**: Phases 1, 2, 3, 5 are **done**. Only **Task 4** (the `VisualGenOutput` request-level wrapper) remains from the core refactor. Future items are tracked at the bottom. Progress callback (previously planned as Task 6) is deferred to Future E (streaming).

---

## Current State (verified in CWD code, 2026-04-23)

| Area | Current state |
| :--- | :--- |
| `tensorrt_llm/visual_gen/` | Exists; exports `VisualGen`, `VisualGenArgs`, `VisualGenParams`, `VisualGenResult`, `VisualGenError`, `VisualGenParamsError`, `ExtraParamSchema`, `MediaOutput` |
| `VisualGenArgs` | Public re-export in `visual_gen/args.py`; canonical definition still in `_torch/visual_gen/config.py` via shim |
| `VisualGenParams` | Pydantic `StrictBaseModel` with `None`-default universal fields, `image`, `mask`, `image_cond_strength`, `negative_prompt`, `extra_params: Optional[dict]` |
| `ExtraParamSchema` | Defined in `_torch/visual_gen/pipeline.py`; re-exported from `visual_gen` |
| `BasePipeline.DEFAULT_GENERATION_PARAMS` / `EXTRA_PARAM_SPECS` | Declared; populated for Wan, Wan I2V, Flux, Flux2, LTX-2, LTX-2 two-stage |
| `VisualGen.extra_param_specs` / `default_params` | Exposed as properties; READY payload carries both |
| Default merging + validation | Done in `DiffusionExecutor._merge_defaults` / `_validate_request`; raises `VisualGenParamsError` on unknown / out-of-range / type-mismatched params |
| `VisualGen(model=..., args=...)` | Renamed from `model_path` / `diffusion_args` |
| `VisualGenError(RuntimeError)` / `VisualGenParamsError(ValueError)` | Added; bare `RuntimeError` paths replaced |
| `VisualGenResult` | Renamed from `DiffusionGenerationResult`; has `result()`, `result_sync()`, `done` property, `cancel()` (`NotImplementedError`) |
| `req_counter` | Uses `itertools.count()` |
| `DiffusionRequest` | Slim — `request_id`, `prompt: List[str]`, `params: Optional[VisualGenParams]`. Field-by-field copy removed. |
| `generate()` / `generate_async()` | Signature: `(inputs: Union[str, List[str]], params: Optional[VisualGenParams] = None)`. Old `VisualGenInputs` surface dropped. |
| `VisualGenInputs` / `VisualGenTextPrompt` / `VisualGenTokensPrompt` / `VisualGenPromptInputs` / `visual_gen_inputs()` | **Removed** from `tensorrt_llm/inputs/data.py` and from all call sites |
| `openai_server.py` | All 4 sites pass `inputs=request.prompt` (plain string) directly; `negative_prompt` flows through `parse_visual_gen_params` onto `VisualGenParams` |
| Return type of `generate()` | **Still returns `MediaOutput` directly**, no request-level wrapper. Task 4 target. |
| `MediaOutput` convenience methods | None; callers still use `MediaStorage` from `serve/`. Future B. |
| OTLP tracing, streaming, `warmup(shapes)`, `on_progress` | Not implemented. Future items C / D / E. |

---

## Remaining Task

Only one remains: **Task 4** — introduce the request-level output wrapper.

---

## Task 4 — `VisualGenOutput` + `VisualGenMetrics`; batch returns list

Wrap `MediaOutput` in a minimal request-level object (`request_id`, `output`, `error`, `metrics`) and change batch `generate()` to return `List[VisualGenOutput]` with per-item unbatched tensors. Raise-on-error semantics kept for single-prompt calls; Option B (per-item `error`, never raise) for batch.

**Full spec**: [task4.md](./task4.md) — standalone implementation doc covering the scope trim (no `seed_used`/`prompt`/`finished`/`queue_ms`/preprocess-inference-postprocess breakdown — just `pipeline_ms`), the client-side tensor-splitting strategy, every file touched, and test plan.

---

## Future Tasks (design doc sections deferred)

### Future A — Sub-config exposure and `VisualGenArgs` full move (§3.1, §9)

Decide which sub-configs (`ParallelConfig`, `CompilationConfig`, `AttentionConfig`, `TeaCacheConfig`, `PipelineConfig`, `PipelineComponent`, `CudaGraphConfig`, `TorchCompileConfig`) belong on the public surface, then move the chosen ones to `tensorrt_llm/visual_gen/args.py` (or a new `visual_gen/config.py`), leaving only `DiffusionModelConfig` (internal) in `_torch/visual_gen/config.py`.

Design-doc recommendation (§9) is **Option C** — `tensorrt_llm/visual_gen/` as the public re-export layer. Today only `VisualGenArgs` lives there; sub-configs are still a shim import.

**Blocked on**: Resolving which configs are genuinely user-facing vs advanced/internal. Users already need `ParallelConfig` for `dit_cfg_size` — that one is clearly public.

### Future B — `tensorrt_llm/media/` + `MediaOutput` convenience methods (§6.2, §6.3)

Create `tensorrt_llm/media/` containing `encoding.py` (moved from `serve/media_storage.py`) and host `MediaOutput` there. Add `save()`, `to_pil()`, `to_bytes()` methods with format / codec / quality args. Deprecate `MediaStorage` (keep as thin wrapper for one release cycle).

Open sub-decisions:

- Audio muxing for LTX-2 outputs (MP4 + AAC via ffmpeg) — §13 Q5.
- `metadata: dict` on `MediaOutput` to carry `frame_rate`, `height`, `width`, `seed_used` from generation so `save("out.mp4")` works without re-specifying frame_rate.

### Future C — `VisualGen.warmup(shapes)` (§7.3)

Expose public `warmup(shapes: List[Tuple[int, int, int]])` for post-init re-warming. Discards all previously compiled CUDA graphs. Low priority, non-breaking.

### Future D — OpenTelemetry tracing (§7.4)

Reuse `llmapi/tracing.py` infrastructure. Add `VisualGenArgs.otlp_traces_endpoint`, `trace_headers` on `generate()`, and `do_tracing()` on `VisualGenResult` emitting a `"visual_gen_request"` span with visual-gen-specific attributes (`visual_gen.resolution`, `visual_gen.num_frames`, `visual_gen.num_inference_steps`, `visual_gen.latency.*`). Additive.

### Future E — Streaming / progress (§5.3, §8)

The API shapes already accommodate streaming (`VisualGenOutput.finished`). The `stream()` method yields intermediate `VisualGenOutput` objects, and `on_progress` falls out as a special case of streaming. Deferred.

### Future F — Per-request `List[VisualGenParams]` batching (§5.1.2)

Lift the `NotImplementedError` (added in Task 4 step 5) once the executor supports splitting a heterogeneous batch into homogeneous sub-batches (by resolution / step count / guidance) and the pipeline merges per-request outputs back into order-preserving results. Requires executor + scheduler + per-pipeline `infer()` changes.

### Future G — Internal `Diffusion*` → `VisualGen*` rename (§4.4, §10.3)

`DiffusionRequest`, `DiffusionResponse`, `DiffusionExecutor`, `DiffusionRemoteClient`, `DiffusionModelConfig`, `DiffusionStepProtocol`, `run_diffusion_worker`: rename in a follow-up pass once Task 4 settles. Purely internal; no user-facing impact.

---

## Open Questions (from [§13](../designs/visual-gen-api-refactor-m2.md#13-open-questions))

| Question | Relevant Task | Status |
| :--- | :--- | :--- |
| **Q1** (`VisualGenParams` Pydantic) | Task 2 | ✅ Resolved — `StrictBaseModel` |
| **Q2** (`params=None` for model defaults) | Task 2 | ✅ Resolved — yes |
| **Q3** (`extra_params` ↔ OpenAI serving mapping) | Task 4 / Future | Open — how do unknown serving fields map to `extra_params`? |
| **Q7** (`num_images_per_prompt > 1`) | Task 4 | Open — one `MediaOutput` with batch dim vs list; tentative single-with-batch-dim |
| **Q8** (batch error semantics) | Task 4 | ✅ Resolved — Option B (return all with per-item errors) |
| **Q9** (`req_counter` thread safety) | Task 1 | ✅ Resolved — `itertools.count()` |
| **Q10** (deprecation strategy) | All | API is `prototype` → breaking is acceptable; no aliases |
| **Q11** (`model_path` → `model` rename) | Task 1 | ✅ Resolved — renamed |
| **Q12** (sub-config re-export) | Future A | Open |
| **Q5** (audio muxing) | Future B | Open — confirm MP4+AAC via ffmpeg |
| **Q6** (LoRA support) | Future | Open |
