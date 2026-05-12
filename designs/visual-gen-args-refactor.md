# `VisualGenArgs` Refactor — Engine-Config Stability & Extensibility

> **Status**: Draft — converged after 9 Codex adversarial-review iterations + 1 owner review pass (2026-05-11)
> **Date**: 2026-05-07 (initial draft); converged 2026-05-11
> **Related**: [`visual-gen-api-refactor-m2.md`](./visual-gen-api-refactor-m2.md) (broader M2 API refactor — covers `VisualGen`, `VisualGenParams`, output types). This doc is a deep-dive on the **engine-config** half of M2 §3 and §13 Q12.

---

## Scope, Target & Non-Goals

This section records the requirement, scope, target, and non-goals as
clarified with the design owner before drafting. A separate review agent
reads this section first; if its content drifts from what's actually
covered in the rest of the doc, that's a flag to either re-confirm scope
with the owner or rewrite the affected sections.

### Requirement (one paragraph)

`VisualGenArgs` is the engine-level config class users instantiate when
constructing `VisualGen(...)`. It currently mixes general loading fields,
model-specific paths (LTX-2-only), cross-cutting optimization sub-configs,
testing/debug knobs, and internal parser state on a single Pydantic
class. Adding a new model or optimization keeps growing the surface, and
once we mark the API "stable" we'd be on the hook for compatibility on
fields that were never intended to be public. Users can't tell which
fields apply to their model on their GPU. The design must propose a
stable, extensible shape for `VisualGenArgs` and its sub-configs that
holds up as new models, new optimizations, and new platforms land —
without leaning on any single style choice prematurely.

### In scope

- **Top-level shape and module location** of `VisualGenArgs`, including
  whether to compose, flatten, or discriminate.
- **Cross-cutting sub-configs** (`ParallelConfig`, `CompilationConfig`,
  `AttentionConfig`, `CacheConfig`, `QuantConfig`) — what stays, what
  splits, what merges, and how deep to nest.
- **Per-architecture model-specific config** — where it lives and how
  it dispatches per model. The owner's preference is a **simple dict
  pass-through**, not a Pydantic typed schema.
- **Field-by-field disposition** of every existing field in today's
  `VisualGenArgs` and its sub-configs (keep / move-to-model-registry /
  move-to-env-debug-knob / make-internal). The owner explicitly flagged
  the debug-knob escape via `TLLM_VG_*`-style env vars as part of the
  migration to consider.
- **Stability marker convention** (`Field(status="prototype")`) — all new
  fields ship as `prototype`. Full API-stability harness is a separate
  task and deliberately out of scope here.
- **Migration plan** — direct edits without alias shims or deprecation
  cycles (VisualGen has no GA users yet).
- **Lightweight discovery affordance** — `VisualGen.supported_models()` +
  `VisualGen.pipeline_config(model)` over a registry keyed by Diffusers `_class_name` (mirrors LLM's `MODEL_MAP` keyed by HF `architectures[0]`)
  model ID. No parallel schema-metadata system.

### Out of scope (non-goals)

- **Per-request params (`VisualGenParams`)** — settled in M2.
- **Output types (`MediaOutput`, `VisualGenOutput`, `VisualGenMetrics`)** —
  settled in M2.
- **Serving-layer (HTTP / OpenAI) encoding** — settled in
  [`visual-gen-output-encoding.md`](./visual-gen-output-encoding.md)
  and [`video-encoding-options.md`](./video-encoding-options.md).
- **Internal `Diffusion*` → `VisualGen*` rename** — deferred per M2
  §10.3.
- **The internal `DiffusionModelConfig`** (merged config built by
  `PipelineLoader`) — stays internal; not migrated by this design.
- **API-stability test harness** (snapshot YAMLs, five-file split, etc.) —
  defer to a separate task. We mark fields `status="prototype"` here so
  the bookkeeping is in place; the test machinery follows.
- **Backwards-compatibility shims** — no `validation_alias`, no
  deprecation cycle. VisualGen is pre-GA; rename / move / remove is
  immediate.
- **Offload config** — declared but unwired in today's `PipelineConfig`.
  Drop entirely; reintroduce when block-wise offload actually ships.
- **`dtype` field** — currently dead code (the pipeline forces bf16
  regardless). Drop; reintroduce when fp16/fp8 inference is supported
  across models.
- **Implementation PRs / detailed impl plan** beyond the migration
  outline.
- **Public registry contract for out-of-tree plugins** — left as an
  Open Question; not in this milestone.

### Target / Audience

- **TRT-LLM VisualGen engineers** (primary) — own the refactor execution.
- **TRT-LLM API / LLM API team** (secondary) — consumers of the
  `LlmArgs` patterns we plan to reuse (`Field(status=...)`,
  `_config`-suffix naming, sub-config composition).
- **Users of `VisualGen(model=..., args=VisualGenArgs(...))`** from
  Python or YAML (tertiary) — affected by the new shape; no migration
  aliases.

### Related docs

- [`visual-gen-api-refactor-m2.md`](./visual-gen-api-refactor-m2.md) —
  broader M2 API refactor (engine class, generate, params, output).
  This doc is the engine-config half of M2 §3 / §13 Q12.
- [`visual-gen-output-encoding.md`](./visual-gen-output-encoding.md) —
  output encoding, in-process vs serving-layer.
- [`video-encoding-options.md`](./video-encoding-options.md) —
  ffmpeg / NVENC investigation notes.
- [`examples-refactor.md`](./examples-refactor.md) — examples / docs
  refactor; will reference any user-facing rename here.
- [`visual_gen_perf_sanity_design.md`](./visual_gen_perf_sanity_design.md) —
  perf-testing infra; touches `VisualGenArgs` only via YAML configs.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Today's `VisualGenArgs` and Why It Hurts](#2-todays-visualgenargs-and-why-it-hurts)
3. [Landscape Survey](#3-landscape-survey)
4. [Design Principles](#4-design-principles)
5. [Independent Design Axes](#5-independent-design-axes)
6. [Top-Level Shape — Options](#6-top-level-shape--options)
7. [Field-by-Field Review of Today's `VisualGenArgs`](#7-field-by-field-review-of-todays-visualgenargs)
8. [Cross-Cutting: Sub-Config Composition Style](#8-cross-cutting-sub-config-composition-style)
9. [Cross-Cutting: Discovery API](#9-cross-cutting-discovery-api)
10. [Cross-Cutting: Stability Marker](#10-cross-cutting-stability-marker)
11. [Cross-Cutting: Debug Knobs vs. Public Args](#11-cross-cutting-debug-knobs-vs-public-args)
12. [Cross-Cutting: YAML, CLI, dict Ingestion](#12-cross-cutting-yaml-cli-dict-ingestion)
13. [Final Design — Public API](#13-final-design--public-api)
14. [Migration Plan](#14-migration-plan)
15. [Open Questions](#15-open-questions)
16. [Appendix: Source Links](#16-appendix-source-links)

---

## 1. Executive Summary

### The problem

`VisualGenArgs` is a single Pydantic class that today carries:

- **Cross-cutting fields** (`checkpoint_path`, `dtype`, `device`, `quant_config`, `compilation`, `parallel`, `cache`, `attention`, `cuda_graph`, `torch_compile`),
- **Model-specific paths** (`text_encoder_path`, `spatial_upsampler_path`, `distilled_lora_path` — all LTX-2 only),
- **Optimization toggles coupled to model architecture** (e.g. `pipeline.fuse_qkv`, `parallel.refiner_*`, the entire `cache` discriminated union),
- **Test/debug knobs** (`skip_warmup`, `skip_components`, `pipeline.enable_layerwise_nvtx_marker`),
- **Internal/derived state leaking out** (`dynamic_weight_quant`, `force_dynamic_quantization`),
- **Dead code** (`pipeline.fuse_qkv` is declared but never read at `e527a9f785`; `dtype` is overridden to bf16 by a hardcoded `torch_dtype` property).

Every new model adds more fields. Every new optimization adds more fields. Every "stable" promise we make for one becomes an obligation we have to keep forever. Users can't tell which fields apply to *their* model on *their* GPU.

### The recommendation (one paragraph)

**Composed cross-cutting Pydantic sub-configs (typed, `_config`-suffix naming to mirror `LlmArgs`) plus a strict, `_class_name`-keyed dict `pipeline_config` for VisualGen pipeline runtime knobs (resolved via a registry mirroring LLM's `MODEL_MAP`). Sub-configs live in `tensorrt_llm.visual_gen.*`, entry-point classes (`VisualGen`, `VisualGenArgs`, `VisualGenParams`) at top-level `tensorrt_llm.*` — namespace mirrors LLM exactly.**

The complete public API surface — every class, every field, every classmethod, every import path, namespace-collision analysis with `LlmArgs` — is laid out in **[§13 Final Design — Public API](#14-final-design--public-api)**. The high-level shape:

- **Cross-cutting sub-configs stay typed and orthogonal**, with `_config`-suffix attribute names matching `LlmArgs`'s convention (`parallel_config`, `compilation_config`, `attention_config`, `cache_config`, `quant_config`). New optimization knobs go into the right sub-config.
- **`CompilationConfig` is a flat umbrella** holding `resolutions`, `num_frames`, `skip_warmup`, plus folded-in `cuda_graph_*` and `torch_compile_*` flat fields. No nested `TorchCompileConfig` / `CudaGraphConfig` siblings — they disappear, their fields are absorbed as prefixed flat fields.
- **Per-architecture config is a flat dict `pipeline_config`**, resolved at load time via a registry keyed by Diffusers `_class_name` (matches today's `pipeline_registry.py`; mirrors LLM's `MODEL_MAP` keyed by `architectures[0]`). No `WanModelConfig` / `FluxModelConfig` Pydantic classes in the public namespace. Strict validation — unknown keys raise. Discovery via `VisualGen.supported_models()` (returns canonical HF ids) + `VisualGen.pipeline_config(model)`; the latter accepts HF id, local path, or `_class_name`.
- **Debug-only knobs split by actual usage**: `skip_warmup` → `compilation_config.skip_warmup` (compile-adjacent); `skip_components` → env var `TLLM_VG_SKIP_COMP` (test-only, no production user); `enable_layerwise_nvtx_marker` → top-level flat field (mirrors `TorchLlmArgs`).
- **Dead-code removal**: `pipeline.fuse_qkv`, `dtype`, `device`, and the three offload fields go away — verified unwired or hardcoded at `e527a9f785`. `PipelineConfig` class disappears entirely.
- **Stability marker only**: every new field carries `Field(status="prototype")`. No `validation_alias`, no soft/hard deprecation cycle, no snapshot harness. VisualGen is pre-GA; the snapshot-test harness is a separate task.
- **Env-var prefix**: `TLLM_VG_*` (not `TLLM_VISUALGEN_*`).
- **CLI**: add `--visual_gen_args` as primary; keep `--extra_visual_gen_options` as alias. `--config` stays on LLM side.

The biggest single payoff: when someone proposes a fourth diffusion model, `VisualGenArgs` doesn't change. They add one registry entry per *pipeline family* (`_class_name` → pipeline class + defaults dict + docstring) — fine-tunes auto-dispatch via the inherited `_class_name`, no per-checkpoint registration. The API surface for users of *other* models is unaffected.

### What this doc does *not* commit to

- **API-stability test harness** (snapshot YAMLs, five-file split). Deferred to a separate task.
- **CLI/UX polish** on top of the discovery API (pretty-printers, `--describe` subcommands).
- **Out-of-tree model registration** — `register_model(class_name, entry)` exists internally; promotion to public API is a separate decision.
- **Backwards-compatibility with today's YAML/dict surface** — direct migration, users edit their YAMLs.

---

## 2. Today's `VisualGenArgs` and Why It Hurts

### 2.1 The current shape

From [`tensorrt_llm/_torch/visual_gen/config.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/e527a9f785/tensorrt_llm/_torch/visual_gen/config.py):

```python
class VisualGenArgs(StrictBaseModel):
    # General loading
    checkpoint_path:        str = ""           # HF id or local path
    revision:               str | None = None
    device:                 str = "cuda"
    dtype:                  str = "bfloat16"

    # LTX-2 specific (no other model uses these)
    text_encoder_path:      str = ""           # only used by LTX-2 pipelines
    spatial_upsampler_path: str = ""           # only used by LTX-2 two-stage
    distilled_lora_path:    str = ""           # only used by LTX-2 stage-2 refinement

    # Component skip + warmup skip (advanced/test)
    skip_components:        List[PipelineComponent] = []
    skip_warmup:            bool = False

    # Quant — typed sub-config + two derived flags
    quant_config:           QuantConfig = QuantConfig()
    dynamic_weight_quant:   bool = False        # populated by validator from quant_config dict
    force_dynamic_quantization: bool = False    # populated by validator from quant_config dict

    # Cross-cutting optimization sub-configs (good)
    compilation:    CompilationConfig    = CompilationConfig()
    torch_compile:  TorchCompileConfig   = TorchCompileConfig()
    cuda_graph:     CudaGraphConfig      = CudaGraphConfig()
    pipeline:       PipelineConfig       = PipelineConfig()    # mixes 4 unrelated knobs
    attention:      AttentionConfig      = AttentionConfig()
    parallel:       ParallelConfig       = ParallelConfig()    # has 7 refiner_* fields for LTX-2
    cache:          CacheConfig | None   = None                # already a discriminated union
```

### 2.2 The five categories of pain

#### 2.2.1 Model-specific fields creep onto a "shared" class

`text_encoder_path`, `spatial_upsampler_path`, `distilled_lora_path` are LTX-2-only. They sit on `VisualGenArgs` because there was nowhere else to put them. A Wan user who sees them in the API reference has to read docs to know they're inert. A new model (say a hypothetical Sora-2) will add three or four more, none of which apply to the existing models. Five models in, the API is confused. Ten models in, it is unusable.

The same problem leaks into sub-configs. `ParallelConfig.refiner_dit_*` (7 fields) and `ParallelConfig.t5_fsdp_size` are LTX-2 two-stage and Wan-T5 specific — they were tacked onto a "shared" config because the alternative (per-model parallel config) didn't exist.

This is **exactly** the SGLang #20078 anti-pattern that PR #20080 fixed — generic defaults silently doing the wrong thing for specific models because the model layer wasn't separated.

#### 2.2.2 Optimization knobs are coupled to model architecture

`PipelineConfig.fuse_qkv` was intended as a transformer-block fusion that's meaningful for some models and impossible for others. In practice it is dead code: no consumer reads it at `e527a9f785`, and `qkv_mode` is hard-coded per attention site in each model's transformer (`models/flux/attention.py:56`, `models/wan/transformer_wan.py:281`, `models/ltx2/transformer_ltx2.py:104-110`). Other knobs like `cache_dit` settings depend on the DiT block structure; `attention.backend = "TRTLLM"` requires kernels for the architecture. All exposed as if they were universal.

The user can set them to nonsensical combinations and only learn at warmup time. Worse, the validator in `VisualGenArgs` can't catch the cross-product because *it doesn't know what model is being loaded* — that resolution happens later in `PipelineLoader`.

#### 2.2.3 Internal state leaks to the public surface

`dynamic_weight_quant` and `force_dynamic_quantization` exist on `VisualGenArgs` only because the `_parse_quant_config_dict` validator splits a single user-facing `quant_config` dict into three things. They are **not** user-set fields — passing them yourself does nothing useful. But they appear in `VisualGenArgs.model_dump()`, in YAML round-trips, and in any auto-generated schema. We are committing to maintain shapes that aren't supposed to exist.

This is the inverse of vLLM's `additional_config: dict` escape hatch — there, the user has too much surface; here, the user has *fake* surface.

#### 2.2.4 Test/debug knobs masquerade as features

`skip_warmup`, `skip_components`, and `pipeline.enable_layerwise_nvtx_marker` are all "advanced" — they exist for debugging or fast iteration. None are part of the production contract. But because they live on `VisualGenArgs`, users will find them, depend on them, and ask us to keep them when we want to remove them.

Two of the three have natural homes other than the flat top-level surface:
- `skip_warmup` is *adjacent to compilation* (the warmup phase runs in the load pipeline alongside torch.compile and cuda-graph capture). Promote to `CompilationConfig.skip_warmup`.
- `skip_components` is *test-only* — no production user, no example, no CLI flag; used in ~20 unit tests to skip text encoders/VAEs so the transformer can be tested in isolation. Demote to env var `TLLM_VG_SKIP_COMP="text_encoder,vae,..."`.

The third (`enable_layerwise_nvtx_marker`) is a live debug knob with an identical analogue on the LLM side (`TorchLlmArgs.enable_layerwise_nvtx_marker` at `llm_args.py:3765`). Mirror LLM exactly: keep it as a top-level Pydantic field, marked `status="prototype"`.

#### 2.2.5 Dead fields advertise functionality that doesn't exist

Three fields are declared but unwired at `e527a9f785`:

- **`pipeline.fuse_qkv`** — declared on `PipelineConfig`, stored in `DiffusionModelConfig`, never read by production code (only test code references the name to round-trip Pydantic).
- **`dtype`** — declared at the top level with default `"bfloat16"`. But `DiffusionModelConfig.torch_dtype` at `config.py:625-627` is hard-coded to return `torch.bfloat16` regardless. The user-facing field is a lie.
- **The three `OffloadConfig`-shaped fields** (`pipeline.enable_offloading`, `pipeline.offload_device`, `pipeline.offload_param_pin_memory`) — declared but zero consumers anywhere in the VisualGen tree. Setting `enable_offloading=True` is a silent no-op. Only an aspirational comment at `pipeline.py:506`.

Fields that lie are worse than no fields. Users wire them up, hit bugs, and we get the support ticket.

#### 2.2.6 No first-class discoverability

A user of `from tensorrt_llm import VisualGen, VisualGenArgs` gets no signal about which fields apply to their model. Reading the API ref tells them everything that *exists*; nothing tells them what's *relevant*. There is no `engine.supported_models()` method, no per-model docstring, no `--help` filtered by model. This is the SGLang `_get_diffusers_model_info` problem — generic defaults dominate.

### 2.3 Why "just keep adding fields" doesn't work

vLLM ran this experiment for us. Their `EngineArgs` has 285 fields ([source](https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py#L402-L688)) and is the subject of [vllm#18707 (docs unusable)](https://github.com/vllm-project/vllm/issues/18707), [vllm#24384 (decouple from HF)](https://github.com/vllm-project/vllm/issues/24384), and a planned `ModelArchitectureConfig` rewrite. SGLang's diffusion `ServerArgs` already has 93 fields plus 2 nested sub-configs ([source](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/server_args.py#L121)) and is bracing for the same refactor. We have a window before VisualGen graduates to `stable` to avoid landing in either of those positions.

---

## 3. Landscape Survey

This is the compact version; full citations and code excerpts are in [§16](#16-appendix-source-links).

### 3.1 Compact comparison

| Aspect | TRT-LLM `LlmArgs` | vLLM `EngineArgs` / `VllmConfig` | vLLM-Omni `OmniEngineArgs` / `OmniDiffusionConfig` | SGLang `ServerArgs` (LLM + diffusion) | Diffusers (per-pipeline) | TRT-LLM `VisualGenArgs` (today) |
| --- | --- | --- | --- | --- | --- | --- |
| **Top-level shape** | Flat Pydantic + sub-configs | Flat dataclass (~285 fields) → `VllmConfig` (composed) | Subclass `EngineArgs` + parallel `OmniDiffusionConfig` | Flat dataclass (~377 + ~93 fields) | Per-pipeline class with `__init__` taking modules | Flat Pydantic with sub-configs |
| **Validation** | Pydantic `extra="forbid"` | Pydantic `extra="forbid"` (sub-configs) | Pydantic + `__post_init__` | argparse + manual `_validate_parameters()` | Lenient on `from_pretrained` (warn unknowns); strict on `__call__` | Pydantic `extra="forbid"` |
| **Model-specific fields** | HF config inheritance | Mix: flat (MoE, Mamba), nested (MultiModal lifted to flat), `additional_config: dict` | `OmniDiffusionConfig` has a few flat (`boundary_ratio`, `flow_shift`); rest via per-request `extra_args` and engine-level `custom_pipeline_args: dict` | Per-model `PipelineConfig` subclass (`@dataclass`, not Pydantic) + per-model `SamplingParams` subclass | Per-pipeline class owns everything model-specific via `__init__` signature | Flat fields on `VisualGenArgs` (LTX-2 paths) + `parallel.refiner_*` |
| **Variant config dispatch** | Manual `from_dict` discriminator (`BaseSparseAttentionConfig`, `DecodingBaseConfig`) | `quantization: str` + dict; `Annotated[Union, Field(discriminator)]` for some | Two registries: `_OMNI_PIPELINES` + `_DIFFUSION_MODELS` | Three registries: `_PIPELINE_REGISTRY`, `_PIPELINE_CONFIG_REGISTRY`, `_CONFIG_REGISTRY` (HF path + lambda detector) | `AutoPipelineForX` static OrderedDicts (closed) | None — no model dispatch on the config side |
| **Sub-config naming** | `_config` suffix on every nested attribute (`kv_cache_config`, `cuda_graph_config`, `torch_compile_config`, `lora_config`, ...) | Mix (`compilation_config`, but also `model`, `cache`) | Inherits vLLM | Flat fields (no nesting) | n/a | Mix: `compilation`, `parallel`, `attention` (no suffix); `quant_config` (with suffix) |
| **Stability marker** | `Field(status="prototype"/"beta"/"deprecated")`; YAML API-stability tests | Policy doc + `@deprecated` decorator; **no** API tests | Inherits vLLM's; nothing additional | None | `deprecate(name, target_version, message)` per call site | None on fields; `set_api_status("prototype")` on a few methods |
| **Escape hatch** | `cp_config: dict[str,Any]` (status=prototype) | `additional_config: dict` (top level) | `extra_args: dict` (per request) + `custom_pipeline_args: dict` (engine level) | `diffusers_kwargs: dict[str,Any]` on the diffusers-fallback subclass only | Lenient `__init__`/`from_pretrained` kwargs (warn unknowns) | None (`extra="forbid"` everywhere) |
| **Discoverability** | `LlmArgs.model_fields`, `model_json_schema()` | `EngineArgs` argparse `--help` (huge) | Same as vLLM | argparse `--help` | `inspect.signature` per pipeline class | `VisualGenArgs.model_fields` (no per-model filter) |
| **Famous lesson** | n/a | #18707 docs / #24384 HF coupling RFC | #2887 / #3366 / #3313 — multi-source precedence chain unsolved | #20078 / #20080 — generic defaults overrode model-specific | None equivalent (per-pipeline-class isolates it) | n/a (still pre-stable) |

### 3.2 Five takeaways from the survey

**Takeaway 1 — Composition is universal; flatness is a CLI artefact.**
Every framework that ships a Python API with sub-configs (TRT-LLM `LlmArgs`, vLLM `VllmConfig`, vLLM-Omni `OmniDiffusionConfig`, SGLang `PipelineConfig`) eventually composes them. vLLM's `EngineArgs` looks flat at first, but `EngineArgs.create_engine_config()` ([vllm/engine/arg_utils.py:1624](https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py#L1624)) explicitly assembles `VllmConfig` from sub-configs. The flatness is a CLI ergonomics layer, not a structural choice.

**Takeaway 2 — Model-specific configuration splits along ecosystem lines.**
LLM-serving frameworks (vLLM, SGLang LLM, TRT-LLM `LlmArgs`) all use **dict pass-through** for model-specific knobs: `hf_overrides`, `json_model_override_args`, `extra_llm_api_options`. Diffusion frameworks (SGLang Diffusion, HuggingFace Diffusers) use **typed per-pipeline classes** — but crucially, both use `@dataclass`, **not** Pydantic, for those classes. The cost of a typed-per-pipeline Pydantic schema (10-20+ classes exported, IDE completion benefit, but big API surface area) is asymmetric for a project whose primary identity is closer to LLM serving than to Diffusers.

**Takeaway 3 — Registries beat central enums for variant dispatch.**
TRT-LLM's `BaseSparseAttentionConfig.from_dict` and `DecodingBaseConfig.from_dict` already use the manual-dispatch idiom; vLLM-Omni and SGLang use lazy-import registries (`_LazyRegisteredModel`, `_LazyPipelineRegistry`) that allow out-of-tree registration. A simple dict keyed by an architecture-class-name (LLM uses `MODEL_MAP[architectures[0]]`; today's VisualGen uses `pipeline_registry.py` keyed by Diffusers `_class_name`) is the minimal version of this pattern.

**Takeaway 4 — Stability requires enforcement, not just intent.**
TRT-LLM's `LlmArgs` already has `Field(status="prototype"|"beta"|"deprecated")` and `tests/unittest/api_stability/llm.yaml` snapshots. vLLM has a 3-release deprecation policy but no API tests, and famously breaks fields anyway. SGLang has no markers and renames freely. We already have the right marker plumbing for `LlmArgs`; reuse the `Field(status=...)` marker on VisualGen now, and defer the full snapshot harness to a separate task.

**Takeaway 5 — `_config`-suffix is the LLM-side house style.**
Every nested sub-config attribute on `BaseLlmArgs` / `TorchLlmArgs` uses `_config` suffix: `kv_cache_config`, `speculative_config`, `scheduler_config`, `cuda_graph_config`, `torch_compile_config`, `quant_config`, `lora_config`, `peft_cache_config`, `cache_transceiver_config`, `sparse_attention_config`, `attention_dp_config`, `moe_config`. Today's VisualGen mixes `quant_config` (suffix) with `compilation` / `parallel` / `attention` (no suffix). Mirror LLM exactly: rename to `_config` suffix throughout.

---

## 4. Design Principles

The principles below extend the M2 doc's principles to the args-specific concerns. They are *normative* statements of the post-refactor design.

1. **Cross-cutting concerns get orthogonal sub-configs.** If a knob applies to every model and every backend (compilation, parallelism, attention, KV-cache-style memory, quant), it lives in its own typed Pydantic sub-config with the `_config` suffix. Sub-configs do not know about each other.
2. **Architecture-specific config is a dict.** Fields that are meaningful for one model are *not* top-level fields and *not* Pydantic classes in the public namespace. `args.pipeline_config: dict[str, Any]` carries VisualGen pipeline runtime knobs (strict — unknown keys raise) and is backed by a registry keyed by Diffusers `_class_name`, mirroring LLM's `MODEL_MAP` keyed by HF `architectures[0]`.
3. **The args class is closed for modification, open for extension.** Adding a new model must not require editing `VisualGenArgs` or any cross-cutting sub-config. Every new *pipeline family* ships as a registry entry: `(class_name, pipeline_cls, defaults_dict, doc_string)`. Fine-tunes of an existing family need zero new code.
4. **Internal state stays internal.** If a field is computed from another, it doesn't appear in the public schema. Use Pydantic's `PrivateAttr` (already used by `LlmArgs`) or move to `DiffusionModelConfig` (the internal merged config).
5. **Flat sub-configs unless multi-axis nesting is genuinely needed.** No `args.compilation_config.torch_compile_config.enable_inductor` — three-hop access hurts discovery and ergonomics. `CompilationConfig` is a flat umbrella with `cuda_graph_*` and `torch_compile_*` prefixed fields. Nested sub-configs are reserved for discriminated unions (`CacheConfig`) and cases where the sub-component has a clear, separable lifecycle.
6. **No backwards-compat shims at this stage.** VisualGen is pre-GA. Renames, moves, and removals are direct edits. No `validation_alias`, no soft/hard deprecation cycle, no compatibility code paths. Users update their YAMLs.
7. **All new fields ship as `status="prototype"`.** Promotion to `"beta"` happens later when usage stabilizes. No field starts at `"beta"` or `"stable"` in this refactor.
8. **Test/debug knobs go where they belong, not where they're easy.** `skip_warmup` → `CompilationConfig` (semantically adjacent to compile/cuda-graph). `skip_components` → env var `TLLM_VG_SKIP_COMP` (test-only, no production user). `enable_layerwise_nvtx_marker` → top-level Pydantic field (mirrors LLM exactly).
9. **Dead-code fields get deleted, not renamed.** Verified unwired fields (`fuse_qkv`, `dtype`-pretending-to-be-bf16, the three offload fields) are deleted. They reappear when the corresponding feature actually ships.

---

## 5. Independent Design Axes

The four design choices below are **independent**. We can pick the answer to each axis without being forced into a particular answer on the others. §6 then enumerates concrete combinations.

### Axis A — Where do model-specific fields live?

| Option | Pattern | Examples |
| --- | --- | --- |
| **A1** | Flat fields on the parent (status quo) | Today's `text_encoder_path` etc. |
| **A2** | Per-model typed Pydantic submodel via discriminated union | TRT-LLM `BaseSparseAttentionConfig.from_dict`; vLLM `SpeculativeConfig`; SGLang Diffusion `PipelineConfig` subclasses (but as `@dataclass`, not Pydantic) |
| **A3** | Generic dict pass-through resolved by registry | vLLM `hf_overrides`, SGLang LLM `json_model_override_args`, TRT-LLM `extra_llm_api_options` |
| **A4** | Subclass `VisualGenArgs` per model | (No major framework does this) |

A1 is the source of pain (§2.2.1). A4 fragments the import path and breaks the "one engine class, many models" property. A2 vs A3 is the substantive choice — see §6.

**Decision**: **A3** with a `_class_name`-keyed registry (mirrors LLM's `MODEL_MAP`). The argument is in §6.3.

### Axis B — Where do optimization configs live?

| Option | Pattern | Examples |
| --- | --- | --- |
| **B1** | Flat on the parent | Today's `attention.backend`, `cuda_graph.enable_cuda_graph` |
| **B2** | Orthogonal cross-cutting sub-configs | vLLM `CompilationConfig`, `CacheConfig`, `ParallelConfig`; TRT-LLM `KvCacheConfig`, `MoeConfig` |
| **B3** | Coupled to model config (per-architecture optimization classes) | None mainstream |

B2 is the universal answer. **Decision: B2.**

Note: `pipeline.fuse_qkv` is *not* an optimization knob to relocate — it's dead code. `qkv_mode` is hard-coded per attention site in the per-model transformer constructors; the config field has no consumer. Delete it (§7.6).

### Axis C — How does a user discover what applies?

| Option | Pattern | Examples |
| --- | --- | --- |
| **C1** | Static docs + `--help` | vLLM `EngineArgs` CLI help, SGLang argparse |
| **C2** | Schema introspection (`Model.model_json_schema()`) | Pydantic-native; LangChain tool schemas |
| **C3** | Per-model docstrings + class-level `inspect.signature` | Diffusers per-pipeline classes |
| **C4** | A purpose-built `engine.list_supported_args(model)` API | The user's "discovery API" idea |

For the cross-cutting parts, **C2** is free (Pydantic native). For the model-specific dicts, **C4** is the right answer — `VisualGen.supported_models()` lists registered Diffusers `_class_name`s (e.g., `"WanPipeline"`); `VisualGen.pipeline_config(model)` accepts an HF id, local path, or `_class_name` and returns the defaults dict. Source of truth is the registry.

### Axis D — How is stability enforced?

| Option | Pattern | Examples |
| --- | --- | --- |
| **D1** | Convention only | SGLang |
| **D2** | Decorator + policy doc | vLLM (`@typing_extensions.deprecated` + 3-release policy) |
| **D3** | Field-level status + API-stability snapshot tests | TRT-LLM `LlmArgs` (`status=...`, `tests/unittest/api_stability/`) |
| **D4** | Versioned schema (proto-style) | None mainstream for Python config |

**Decision**: **partial D3**. Adopt the `Field(status=...)` marker now (mark all new fields `prototype`); defer the snapshot-test harness to a separate task. No deprecation cycle for this refactor — VisualGen is pre-GA, removals are immediate.

---

## 6. Top-Level Shape — Options

Three concrete combinations. Each presented with sketch, pros, cons.

### 6.1 Option A — Status Quo + Organic Growth (Rejected)

(Axes: A1, B1, C1, D1.)

Keep `VisualGenArgs` flat-with-sub-configs as today. Add fields as needed. Per-architecture pain managed by docs.

**Sketch**: today's `VisualGenArgs` plus whatever the next model needs (e.g. `text_encoder_2_path`, `wan_special_flag`, `flux_inter_block_quant_mode`).

**Pros**:
- Zero refactor. Lowest short-term cost.
- Familiar to users who already wrote LTX-2 code.

**Cons**:
- All five categories of pain (§2.2) compound.
- Lands us in vllm#18707 / sglang#20078 territory.

**Verdict**: Rejected. The whole point of this doc is that this option doesn't scale.

### 6.2 Option B — Typed Discriminated Union Per Architecture (Rejected)

(Axes: A2, B2, C2, D3.)

Composed cross-cutting sub-configs plus a Pydantic discriminated `arch_config: ArchConfig` union with one typed Pydantic class per architecture (`WanModelConfig`, `FluxModelConfig`, `LTX2ModelConfig`, ...).

**Sketch**:
```python
class WanModelConfig(BaseModelConfig):
    arch: Literal["wan"] = "wan"
    text_encoder_fsdp_size: int = 1

class LTX2ModelConfig(BaseModelConfig):
    arch: Literal["ltx2"] = "ltx2"
    text_encoder_path: str = ""
    spatial_upsampler_path: str = ""
    distilled_lora_path: str = ""
    refiner_parallel_dit_dp_size: int = 1
    # ...

ArchConfig = Annotated[
    Union[WanModelConfig, FluxModelConfig, LTX2ModelConfig, ...],
    Field(discriminator="arch"),
]

class VisualGenArgs(StrictBaseModel):
    # ...
    arch_config: ArchConfig | None = None
```

**Pros**:
- IDE completion + JSON-schema introspection per arch.
- Pydantic discriminator handles `from YAML/dict` deserialization for free.
- Matches SGLang Diffusion and HuggingFace Diffusers (typed per-pipeline classes).

**Cons**:
- **API surface explosion**: every supported model exports one or more `XyzModelConfig` Pydantic classes. SGLang Diffusion ships ~16-38 such classes (per their `__init__.py`). At our projected 10-20 models, that's 10-20 Pydantic classes to maintain, version, and document.
- **Pydantic-specific tax**: discriminated unions require `Annotated[Union[...], Field(discriminator=...)]`, base-class forward-declaration ordering, model_validator dispatch for legacy YAML, and explicit `from_dict` plumbing. Each per-arch class needs its own model_json_schema validation. The Pydantic complexity is real — Codex iterations 1-8 of this design surfaced edge cases in the discriminator-resolution code path that don't exist with a dict.
- **Asymmetric with LLM-serving precedent**: vLLM, SGLang LLM, and TRT-LLM `LlmArgs` all use dict pass-through (`hf_overrides`, `json_model_override_args`, `extra_llm_api_options`). VisualGen sits between LLM serving and image/video generation; choosing the diffusion-side pattern doubles down on the API surface area for the model-specific knobs that the design owner explicitly does **not** want to maintain stability on.
- **Per-checkpoint vs per-family ambiguity**: typed classes naturally encode "one config per pipeline family". But real validation needs the *exact checkpoint* + the *exact pipeline variant* (i.e., HF model_index.json). The user constructing `VisualGenArgs(arch_config=WanConfig(...))` is only validating that `WanConfig` is internally consistent, not that the chosen checkpoint accepts that config. The full validation is pipeline-related anyway, not Pydantic-checkable, so much of the Pydantic value is illusory.

**Verdict**: Rejected. Strong precedent in the diffusion ecosystem, but the API-surface cost and the asymmetry with LLM-serving precedent outweigh the typing benefits for this project. The "fallback to a single flat `ModelPipelineConfig` Pydantic class" variant of this option avoids the explosion but jams unrelated model knobs into one schema — also rejected.

### 6.3 Option C — Composed Sub-Configs + Dict Model-Specific (Recommended)

(Axes: A3, B2, C2+C4, D3.)

Cross-cutting concerns get typed Pydantic sub-configs with the `_config` suffix matching `LlmArgs`. Per-architecture concerns are a flat `pipeline_config: dict[str, Any]`, validated at load time against defaults from a registry keyed by Diffusers `_class_name`. Discovery via two classmethods on `VisualGen`.

**Sketch**:

```python
# tensorrt_llm/visual_gen/args.py (new public location, M2 §3.1 / §9)

class CompilationConfig(StrictBaseModel):
    """Flat compilation umbrella — covers torch.compile, CUDA graphs, and
    warmup. No nested TorchCompileConfig / CudaGraphConfig; their fields
    are flattened in here with prefix names."""
    resolutions:                       list[tuple[int, int]] | None = Field(default=None, status="prototype")
    num_frames:                        list[int] | None             = Field(default=None, status="prototype")
    skip_warmup:                       bool                         = Field(default=False, status="prototype")
    # CUDA graphs
    cuda_graph_batch_sizes:            list[int] | None             = Field(default=None, status="prototype")
    cuda_graph_max_batch_size:         int                          = Field(default=1,    status="prototype")
    cuda_graph_enable_padding:         bool                         = Field(default=False, status="prototype")
    # torch.compile
    torch_compile_enable_fullgraph:    bool                         = Field(default=True, status="prototype")
    torch_compile_enable_inductor:     bool                         = Field(default=True, status="prototype")
    torch_compile_capture_num_tokens:  list[int] | None             = Field(default=None, status="prototype")
    torch_compile_max_num_streams:     int                          = Field(default=1,    status="prototype")

class VisualGenArgs(StrictBaseModel):
    # ── Loading ──────────────────────────────────────────────────
    model:    str                                                       = Field(description="HF id or local path.")
    revision: str | None                                                = Field(default=None, status="prototype")

    # ── Cross-cutting sub-configs (typed; _config-suffix per LlmArgs) ─
    parallel_config:    ParallelConfig    = Field(default_factory=ParallelConfig)
    compilation_config: CompilationConfig = Field(default_factory=CompilationConfig)
    attention_config:   AttentionConfig   = Field(default_factory=AttentionConfig)
    cache_config:       CacheConfig | None = None
    quant_config:       QuantConfig       = Field(default_factory=QuantConfig)

    # ── Observability (flat at top-level, mirrors TorchLlmArgs) ───
    enable_layerwise_nvtx_marker: bool = Field(default=False, status="prototype")

    # ── VisualGen pipeline runtime knobs (resolved via _class_name registry) ─
    pipeline_config: dict[str, Any] = Field(
        default_factory=dict,
        status="prototype",
        description=(
            "VisualGen-specific runtime knobs that aren't in any HF / "
            "Diffusers config file (e.g., text_encoder_path for LTX-2, "
            "refiner_dp_size for LTX-2 two-stage). Keys must match the "
            "registry entry for the resolved pipeline class; unknown keys "
            "raise at load time. Discover via VisualGen.pipeline_config(model)."
        ),
    )
```

`pipeline_config` carries pipeline-specific knobs that aren't in any HF/Diffusers config file (e.g., `text_encoder_path` for LTX-2, `refiner_dp_size` for LTX-2 two-stage, `text_encoder_fsdp_size` for Wan). Strict validation — unknown keys raise so typos surface at load time; the registry's `defaults` is the schema-by-example. (If a future need for HF/Diffusers config overrides emerges, the right shape is a separate dict mirroring `LlmArgs.model_kwargs` at `llm_args.py:2915` — deliberately deferred until a real use case appears.)

And the registry, internal:

```python
# tensorrt_llm/_torch/visual_gen/registry.py

@dataclass
class ModelEntry:
    pipeline_cls: type             # the pipeline class to construct
    hf_ids:       list[str]        # canonical HF ids for this family (for VisualGen.supported_models())
    defaults:     dict[str, Any]   # default pipeline_config knobs (schema-by-example)
    doc:          str              # short docstring for the discovery API

# Keyed by Diffusers `_class_name` (from model_index.json), mirroring
# LLM's MODEL_MAP keyed by HF `architectures[0]`. ~3-5 entries — one
# per pipeline family, not one per checkpoint. Fine-tunes auto-dispatch
# via their inherited `_class_name`.
_MODEL_REGISTRY: dict[str, ModelEntry] = {}

def register_model(class_name: str, entry: ModelEntry) -> None:
    if class_name in _MODEL_REGISTRY:
        raise ValueError(f"Model already registered: {class_name}")
    _MODEL_REGISTRY[class_name] = entry
```

And discovery API on `VisualGen`:

```python
# tensorrt_llm/visual_gen/visual_gen.py

class VisualGen:
    @classmethod
    def supported_models(cls) -> list[str]:
        """Return the canonical HF model IDs across all registered
        pipeline families (e.g., 'Wan-AI/Wan2.1-T2V-14B',
        'black-forest-labs/FLUX.1-dev', 'Lightricks/LTX-Video', ...).

        Fine-tunes of these families auto-dispatch via the inherited
        `_class_name` and need not appear in this list.
        """
        return [hf_id for e in _MODEL_REGISTRY.values() for hf_id in e.hf_ids]

    @classmethod
    def pipeline_config(cls, model: str | Path) -> dict[str, Any]:
        """Return the default pipeline_config knobs for the given model.

        `model` may be:
        - An HF model id (e.g., 'Wan-AI/Wan2.1-T2V-14B') — looked up in
          the registry's `hf_ids` list.
        - A local checkpoint path — resolved by reading
          `model_index.json::_class_name` (with a safetensors-metadata
          fallback for LTX-2's native single-file format), the same way
          `PipelineLoader` does.
        - A registered Diffusers `_class_name` directly (e.g.,
          'WanPipeline') — returned without resolution.
        """
        # 1. HF id match (most common user-facing path).
        for entry in _MODEL_REGISTRY.values():
            if model in entry.hf_ids:
                return dict(entry.defaults)
        # 2. Direct _class_name match.
        if model in _MODEL_REGISTRY:
            return dict(_MODEL_REGISTRY[model].defaults)
        # 3. Local path → resolve _class_name via the same logic
        #    PipelineLoader uses.
        class_name = AutoPipeline._detect_from_checkpoint(model)
        return dict(_MODEL_REGISTRY[class_name].defaults)
```

Load-time merge in `PipelineLoader`:

```python
class_name = AutoPipeline._detect_from_checkpoint(args.model)
entry = _MODEL_REGISTRY[class_name]
unknown = set(args.pipeline_config) - set(entry.defaults)
if unknown:
    raise ValueError(
        f"Unknown pipeline_config keys for {class_name} ({args.model}): "
        f"{sorted(unknown)}. Valid keys: {sorted(entry.defaults)}"
    )
resolved = {**entry.defaults, **args.pipeline_config}
pipeline = entry.pipeline_cls(**resolved, ...)
```

**Pros**:
- **Solves all five categories of pain** in §2.2 with the minimum API surface.
- **No per-model Pydantic classes exported** — registry entries are internal data, the public surface is one dict field + two classmethods.
- **Matches LLM-serving precedent** (vLLM `hf_overrides`, SGLang `json_model_override_args`, TRT-LLM `extra_llm_api_options`).
- **Cross-cutting sub-configs stay typed, orthogonal, discoverable** with the same `_config`-suffix naming as `LlmArgs`.
- Strict unknown-key handling catches typos at load time even without IDE completion.
- New model = one registry entry + one pipeline class. No edits to `VisualGenArgs`.

**Cons**:
- **No IDE completion for model-specific keys** — typos fail at load time, not at edit time. Mitigated by strict-on-unknown-keys + the discovery API. This is the explicit trade-off the design owner chose.
- **No JSON-schema for `pipeline_config` keys** — `VisualGenArgs.model_json_schema()` shows `pipeline_config: dict`, not the per-pipeline contents. OpenAPI generation would need to walk the registry separately.
- Validation moves from Pydantic to the pipeline class. The strict unknown-key check is the safety net; pipeline classes own deeper validation (type coercion, range checks).

**Verdict**: **Recommended.** Matches the design owner's explicit preference for dict pass-through, minimizes public API surface, mirrors LLM-serving precedent, keeps the cross-cutting typed surface that already works.

### 6.4 Variants we considered and rejected

Two flavours of typed approach, both rejected:

- **One typed `ModelPipelineConfig` Pydantic class with all model knobs flattened together** — solves the export-count problem of Option B (one class instead of 10-20), but jams unrelated fields together. An LTX2 user sees Wan-specific keys; a Wan user sees LTX2-specific keys; new models force edits to a shared schema. All the cons of A1, dressed up.
- **Typed per-arch `@dataclass` (not Pydantic), à la SGLang Diffusion** — closer to the dict approach in spirit (no Pydantic discriminator complexity), but still exports per-arch classes. We'd need a custom `from_dict` dispatch since `@dataclass` doesn't ship one. Net: more code than dict pass-through, less ergonomic than typed Pydantic.

The dict + registry path is simpler than both and matches the LLM side.

---

## 7. Field-by-Field Review of Today's `VisualGenArgs`

This section walks every field that exists today. The dispositions are:

- **Keep** on `VisualGenArgs` (cross-cutting, stable).
- **Move to `pipeline_config` registry defaults** (per-pipeline-family, in the registry's `defaults` dict for the relevant `_class_name` entry).
- **Move to env var** (`TLLM_VG_*`) — testing/debug, not part of the production contract.
- **Make internal** — derived from another field, surfaced via `PrivateAttr` or moved to `DiffusionModelConfig`.
- **Delete** — dead code, verified unwired at `e527a9f785`.

### 7.1 General loading fields

| Field | Today | Disposition | Notes |
| --- | --- | --- | --- |
| `checkpoint_path` | flat `str` | **Keep, renamed `model`** (M2 §3.4) | Matches `LlmArgs.model` |
| `revision` | flat `str \| None` | **Keep** | Live and wired — `pipeline_loader.py:91` reads `self.args.revision` and forwards to `download_hf_model(revision=...)`. Mismatch raises HF Hub `RepositoryNotFoundError`. |
| `device` | flat `str` | **Delete** | `LlmArgs` does not have a `device` field; device is implicit CUDA. We are not supporting non-CUDA inference. Reintroduce when CPU inference matters. |
| `dtype` | flat `str` | **Delete** | Dead code at `e527a9f785`: `DiffusionModelConfig.torch_dtype` (`config.py:625-627`) is hard-coded to `torch.bfloat16` regardless of the field. Reintroduce when fp16/fp8 inference is actually wired through the loader (with default `"auto"` like `LlmArgs`). |

### 7.2 LTX-2-specific paths

| Field | Today | Disposition | Notes |
| --- | --- | --- | --- |
| `text_encoder_path` | flat `str` | **Move to `LTX2Pipeline` registry entry's defaults** | E.g. `_MODEL_REGISTRY["LTX2Pipeline"].defaults["text_encoder_path"] = ""` |
| `spatial_upsampler_path` | flat `str` | **Move to `LTX2Pipeline` registry entry's defaults** | (Variant resolution into the two-stage pipeline happens in `LTX2Pipeline.resolve_variant()`.) |
| `distilled_lora_path` | flat `str` | **Move to `LTX2Pipeline` registry entry's defaults** | The LTX-2 stage-2 LoRA merge is implementation-internal to `pipeline_ltx2_two_stages.py:45-172, 617-664` — not a general LoRA story. See Open Question on a Diffusers-style runtime `load_lora_weights()` API. |

These are the textbook example of architecture-specific creep (§2.2.1). They become part of the `LTX2Pipeline` registry entry's `defaults` dict. Wan and Flux users no longer see them anywhere. LTX-2 users discover them via `VisualGen.pipeline_config("Lightricks/LTX-Video")` (HF id resolved to `_class_name` internally) or `VisualGen.pipeline_config("LTX2Pipeline")`.

### 7.3 Component skip + warmup skip

| Field | Today | Disposition | Notes |
| --- | --- | --- | --- |
| `skip_warmup` | flat `bool` | **Move to `compilation_config.skip_warmup`**, `status="prototype"` | Adjacent to torch.compile and cuda-graph capture in the load pipeline. Read by `PipelineLoader.load(...)` directly. |
| `skip_components` | flat `List[PipelineComponent]` | **Move to env var `TLLM_VG_SKIP_COMP`** | Test-only knob — no production user, no example, no CLI flag. Used in ~20 unit tests (`tests/unittest/_torch/visual_gen/test_flux_pipeline.py:60`, `test_cache_dit.py:263-266`) to skip text encoders/VAEs while testing the transformer. Env-var form: `TLLM_VG_SKIP_COMP="text_encoder,vae,tokenizer"`. Tests use `monkeypatch.setenv(...)` or a tiny pytest fixture. |

`skip_warmup` is per-instance (two `VisualGen` instances in the same process can disagree) and lives on `CompilationConfig`. `skip_components` is process-global as an env var, but since the only consumers are pytest fixtures, that's fine.

### 7.4 Quantization

| Field | Today | Disposition | Notes |
| --- | --- | --- | --- |
| `quant_config` | `QuantConfig` (from LLM side) | **Keep** | Already matches LLM `_config`-suffix convention. |
| `dynamic_weight_quant` | `bool` | **Make internal** | `PrivateAttr`; populated by `quant_config`'s parser. |
| `force_dynamic_quantization` | `bool` | **Make internal** | `PrivateAttr`; populated by `quant_config`'s parser. |

`quant_config` is the same `QuantConfig` from `tensorrt_llm.models.modeling_utils`, shared with the LLM API. The two derived booleans are implementation artefacts (§2.2.3) and should not appear on the public surface — move to `PrivateAttr` (matching `LlmArgs._parallel_config` and `_quant_config`) and populate them in the existing `_parse_quant_config_dict` validator.

### 7.5 Cross-cutting optimization sub-configs

| Field | Today | Disposition | Notes |
| --- | --- | --- | --- |
| `compilation: CompilationConfig` | typed sub-config | **Keep, renamed `compilation_config`** | Plus folds in `torch_compile` and `cuda_graph` fields (below). |
| `torch_compile: TorchCompileConfig` | typed sub-config | **Fold into `compilation_config`** | Promote fields to `torch_compile_*` prefixed flat fields on `CompilationConfig`. |
| `cuda_graph: CudaGraphConfig` | typed sub-config | **Fold into `compilation_config`** | Promote fields to `cuda_graph_*` prefixed flat fields on `CompilationConfig`. |
| `attention: AttentionConfig` | typed sub-config | **Keep, renamed `attention_config`** | |
| `parallel: ParallelConfig` | typed sub-config (with model-specific creep, see 7.7) | **Keep, renamed `parallel_config`**, with carve-outs | |
| `cache: CacheConfig \| None` | discriminated union | **Keep, renamed `cache_config`** | Already correctly designed as a `TeaCacheConfig \| CacheDiTConfig` discriminated union. |

**The `_config`-suffix rename matches `LlmArgs` exactly**: `kv_cache_config`, `cuda_graph_config`, `torch_compile_config`, `quant_config`, etc. are all `_config`-suffixed in `BaseLlmArgs` / `TorchLlmArgs`.

**Folding `torch_compile` / `cuda_graph` into `compilation_config`** is the deliberate divergence from LLM (where they're flat peers on `TorchLlmArgs`). For VisualGen, the umbrella has standalone fields (`resolutions`, `num_frames`, `skip_warmup`) that have no LLM analogue, and a single discoverable surface for everything compilation-related reads better than three peer fields. The price is field-name prefixes (`cuda_graph_batch_sizes` not `cuda_graph.batch_sizes`) — acceptable, no nesting confusion.

### 7.6 The `pipeline: PipelineConfig` mixed bag — deleted entirely

`PipelineConfig` today has four unrelated fields. All four disappear:

| Field | Today | Disposition | Notes |
| --- | --- | --- | --- |
| `pipeline.fuse_qkv` | `bool` | **Delete** | Dead code at `e527a9f785`. `qkv_mode` is hard-coded per attention site: `models/flux/attention.py:56` (FUSE_QKV), `models/wan/transformer_wan.py:281` (FUSE_QKV self) / `:294` (SEPARATE_QKV cross), `models/ltx2/transformer_ltx2.py:104-110` (driven by `_is_cross_attn`). Only `tests/unittest/_torch/visual_gen/test_fused_qkv.py:4` and `test_visual_gen_args.py:66` reference the *name* (Pydantic round-trip). No production reader. **Not moved to `attention_config`** — that would re-introduce a knob with no consumer. |
| `pipeline.enable_layerwise_nvtx_marker` | `bool` | **Move to top-level flat field** `VisualGenArgs.enable_layerwise_nvtx_marker` | Live and wired (`pipeline_loader.py:253-260` registers per-layer hooks via `LayerwiseNvtxMarker`). LLM has the same field at `TorchLlmArgs.enable_layerwise_nvtx_marker` (`llm_args.py:3765`, `status="beta"`); mirror exactly. `status="prototype"` for VisualGen. |
| `pipeline.enable_offloading` | `bool` | **Delete** | Declared but zero consumers anywhere in the VisualGen tree. Flipping it is a silent no-op. Reintroduce as a proper `OffloadConfig` when block-wise offload actually ships. |
| `pipeline.offload_device` | `Literal["cpu", "cuda"]` | **Delete** | Same — unwired today. |
| `pipeline.offload_param_pin_memory` | `bool` | **Delete** | Same — unwired today. |

After these splits, `PipelineConfig` is empty and the class disappears entirely. The name `PipelineConfig` is freed up for future reuse if needed.

### 7.7 `ParallelConfig` carve-outs

`ParallelConfig` today is mostly cross-cutting but contains seven LTX-2-specific `refiner_*` fields and one Wan-specific `t5_fsdp_size`:

| Field | Today | Disposition | Notes |
| --- | --- | --- | --- |
| `parallel.dit_*` (cfg_size, ulysses_size, attn2d_*, tp_size, dp_size, fsdp_size, ring_size, dim_order) | flat (cross-cutting) | **Keep on `parallel_config`** | |
| `parallel.enable_parallel_vae` | flat | **Keep** | |
| `parallel.parallel_vae_split_dim` | flat | **Keep** | |
| `parallel.refiner_dit_*` (7 fields) | flat (LTX-2 two-stage only) | **Move to `LTX2Pipeline` registry entry's defaults** | E.g. `_MODEL_REGISTRY["LTX2Pipeline"].defaults["refiner_dp_size"] = 1` (dit_ prefix dropped to match the new `ParallelConfig` naming). No runtime read at `e527a9f785` — intended-but-unused; reserve the registry slot now. |
| `parallel.t5_fsdp_size` | flat (only Wan T5 path) | **Move to `WanPipeline` registry entry's defaults** | E.g. `_MODEL_REGISTRY["WanPipeline"].defaults["text_encoder_fsdp_size"] = 1`. |

**Key insight from this carve-out**: when a model has *multiple* parallel-able passes (LTX-2 stage-1 / stage-2; Wan T5 / DiT), the "main DiT" parallel knobs are cross-cutting (every DiT-shaped model has them) but the auxiliary parallel knobs (refiner DiT, T5 encoder) are model-specific. Today's `ParallelConfig` mixes both. Splitting them: the DiT-shaped knobs stay on `parallel_config`; everything else moves to the relevant model's registry defaults.

### 7.8 Summary of dispositions

| Bucket | Count | Examples |
| --- | --- | --- |
| **Keep on `VisualGenArgs`** | 2 flat + 5 sub-configs + 1 flat observability | `model`, `revision`; `parallel_config`, `compilation_config`, `attention_config`, `cache_config`, `quant_config`; `enable_layerwise_nvtx_marker` |
| **Move to registry defaults** | 11 fields | LTX-2 paths (3); `parallel.refiner_*` (7); `parallel.t5_fsdp_size` (1) |
| **Delete** (verified zero runtime reads at `e527a9f785`) | 5 fields | `pipeline.fuse_qkv`; `dtype` (`torch_dtype` hardcoded bf16); `device` (LLM has none); `pipeline.enable_offloading`, `pipeline.offload_device`, `pipeline.offload_param_pin_memory` (three offload fields, all unwired) |
| **Move to env var** | 1 field | `skip_components` → `TLLM_VG_SKIP_COMP` |
| **Move to `compilation_config`** | 1 field | `skip_warmup` |
| **Move to top-level flat** | 1 field | `enable_layerwise_nvtx_marker` (mirrors LLM) |
| **Make internal (PrivateAttr)** | 2 fields | `dynamic_weight_quant`, `force_dynamic_quantization` |
| **Folded into umbrella** | 2 sub-configs | `torch_compile`, `cuda_graph` → flat prefixed fields on `compilation_config` |
| **Eliminated** | 1 sub-config | `PipelineConfig` (all four fields rehomed or deleted) |

The net effect:

- `VisualGenArgs` shrinks from "4 flat + 4 LTX-2 + 2 test + 2 internal + 7 sub-configs" (19 surface concepts) to **"2 flat + 5 cross-cutting sub-configs + 1 flat observability + 1 `pipeline_config` dict"** (9 surface concepts).
- The set of fields a Wan user sees in their schema goes from ~50 to ~25, and every field they see is meaningful for Wan.
- Pipeline-runtime knobs live in the registry (keyed by Diffusers `_class_name`); users discover them via `VisualGen.pipeline_config(model)` which accepts HF id, local path, or `_class_name`.

---

## 8. Cross-Cutting: Sub-Config Composition Style

The user explicitly flagged: *"for the args, they could be nested, for example we may extend the certain args to support new modes, so we need to be careful on such."* Plus: *"Over-nested is not friendly for config discovery and makes the coding hard."*

This section documents the three composition styles we use and when each fires.

### 8.1 Flat umbrella sub-config (the default)

When a concern has multiple related knobs but no clear sub-component lifecycle, the sub-config is a flat Pydantic class. **`CompilationConfig` is the canonical example** — `resolutions`, `num_frames`, `skip_warmup`, `cuda_graph_*`, `torch_compile_*` all live as flat fields with prefix names. No `args.compilation_config.torch_compile_config.enable_inductor` — that's three-hop access for no benefit.

This applies to:
- `CompilationConfig` (flat, with `cuda_graph_*` and `torch_compile_*` prefixed fields).
- `AttentionConfig` (flat: `backend`, plus per-backend kwargs as flat fields).
- `ParallelConfig` (flat: all `dit_*` fields).

Adding a new knob: add a flat field. Promotion / removal is by direct edit (no aliases per §10).

### 8.2 Discriminated unions for "one of N modes"

When a config has multiple mutually-exclusive backends, use a Pydantic discriminated union. **`CacheConfig` is the canonical example today** (`TeaCacheConfig | CacheDiTConfig`, discriminated by `cache_backend`).

Future candidates (not in this refactor):
- **Attention backend**: `attention_config` has `backend: Literal["VANILLA", "TRTLLM", "FA4"]` today. As backends accumulate and need per-backend kwargs, promote to `attention_config: VanillaAttentionConfig | TrtllmAttentionConfig | FA4AttentionConfig` discriminated by `backend`.
- **Quant backend**: TRT-LLM `QuantConfig` is currently enum-based. If quant backends start needing radically different config (NVFP4 group_size, FP8 dynamic flags, W4A8 exclude lists, etc.), promote.

### 8.3 Nested Pydantic — only when the sub-component has a separable lifecycle

The two-level nesting in `args.X_config.Y` is justified only when `Y` is genuinely its own concern. Currently we have **none** — `CompilationConfig` deliberately flattens, `CacheConfig` is a discriminated union, the other sub-configs are flat. If a future sub-component needs its own lifecycle (e.g. a `LoraConfig` with multiple LoRAs each carrying their own state), revisit.

The rule: prefer flat. Reach for nesting only when the sub-component reads better as its own typed object than as prefixed fields.

---

## 9. Cross-Cutting: Discovery API

The discovery API has two surfaces, matching the split between typed and dict configuration.

### 9.1 Cross-cutting sub-configs — free from Pydantic

Every Pydantic `BaseModel` already exposes:

```python
CompilationConfig.model_json_schema()       # full JSON Schema
CompilationConfig.model_fields              # dict of field info, including descriptions
VisualGenArgs.model_json_schema()           # nested schema for the whole class
```

This covers cross-cutting fields and sub-configs without writing any new API. Tools that need OpenAPI generation, IDE-completion fallbacks, or YAML schema validation get them for free.

### 9.2 Model-specific knobs — registry-backed classmethods on `VisualGen`

```python
class VisualGen:
    @classmethod
    def supported_models(cls) -> list[str]:
        """Return the canonical HF model IDs across all registered
        pipeline families. Fine-tunes auto-dispatch via inherited
        `_class_name` and need not appear in this list."""
        return [hf_id for e in _MODEL_REGISTRY.values() for hf_id in e.hf_ids]

    @classmethod
    def pipeline_config(cls, model: str | Path) -> dict[str, Any]:
        """Return the defaults dict for the given model.

        `model` may be an HF id (looked up in each entry's `hf_ids`
        list), a local checkpoint path (resolved to `_class_name` via
        `_detect_from_checkpoint`), or a registered `_class_name`
        directly. Raises KeyError if no entry matches. The returned
        dict is a copy — mutating it does not affect the registry."""
        for entry in _MODEL_REGISTRY.values():
            if model in entry.hf_ids:
                return dict(entry.defaults)
        if model in _MODEL_REGISTRY:
            return dict(_MODEL_REGISTRY[model].defaults)
        class_name = AutoPipeline._detect_from_checkpoint(model)
        return dict(_MODEL_REGISTRY[class_name].defaults)
```

**Usage**:

```python
>>> VisualGen.supported_models()
['Wan-AI/Wan2.1-T2V-14B', 'Wan-AI/Wan2.1-T2V-1.3B', 'black-forest-labs/FLUX.1-dev',
 'Lightricks/LTX-Video', ...]

>>> VisualGen.pipeline_config("Wan-AI/Wan2.1-T2V-14B")
{'text_encoder_fsdp_size': 1, ...}

>>> args = VisualGenArgs(
...     model="Wan-AI/Wan2.1-T2V-14B",
...     pipeline_config={"text_encoder_fsdp_size": 4},
... )
```

The `defaults` dict on each registry entry is **the schema-by-example**: it lists every knob the pipeline accepts, with its default value. There is no separate JSON-schema or typed class. The registry is the source of truth.

### 9.3 Strict unknown-key handling

At load time, `PipelineLoader` merges `args.pipeline_config` over `entry.defaults` and rejects unknown keys:

```python
entry = _MODEL_REGISTRY[args.model]
unknown = set(args.pipeline_config) - set(entry.defaults)
if unknown:
    raise ValueError(
        f"Unknown pipeline_config keys for {args.model}: {sorted(unknown)}. "
        f"Valid keys: {sorted(entry.defaults)}"
    )
resolved = {**entry.defaults, **args.pipeline_config}
```

This is the safety net that replaces IDE completion. Typos surface at load time with the valid key set in the error message.

### 9.4 What we deliberately do *not* ship in this milestone

- **Per-model JSON schema** — derivable from `entry.defaults` by walking the dict, but no canonical method on `VisualGen`. Add later if OpenAPI generation needs it.
- **Capability schema** — "is FA4 + NVFP4 + cache_dit supported on Wan 1.3B on H100?" — out of scope. Resolvable today only by attempting load + warmup; add a capability table later if/when needed.
- **Pretty-printer / `--describe` CLI** — `print(VisualGen.pipeline_config(model))` is sufficient for now. Polish layer; deferred.

---

## 10. Cross-Cutting: Stability Marker

### 10.1 Reuse the LLM-side `Field(status=...)` machinery

`tensorrt_llm/llmapi/llm_args.py` already defines a `Field(default, *, status="prototype"|"beta"|"deprecated", **kwargs)` wrapper that adds `status` to `json_schema_extra`. Use it on `VisualGenArgs` and every sub-config:

```python
from tensorrt_llm.llmapi.llm_args import Field, StrictBaseModel

class VisualGenArgs(StrictBaseModel):
    model:    str                  = Field(description="HF id or local path.")
    revision: str | None           = Field(default=None, status="prototype")
    # ...
    pipeline_config: dict[str, Any] = Field(
        default_factory=dict, status="prototype",
        description="Model-specific overrides; see VisualGen.pipeline_config(model).",
    )
```

**All new fields ship as `status="prototype"`.** Nothing starts at `"beta"` or `"stable"` in this refactor. Promotion to `"beta"` happens later when usage stabilizes.

### 10.2 What's deliberately out of scope here

- **API-stability test harness** (snapshot YAMLs, five-file split, capability rows, alias-cases). The five-file harness proposed in earlier iterations of this design is deferred to a separate task. We mark fields `status="prototype"` here so the marker bookkeeping is in place; the test machinery follows in its own design.
- **`validation_alias` / `AliasChoices` migration shims.** Fields rename / move / disappear directly. Users update their YAMLs.
- **Soft → hard deprecation cycle on removed flat fields.** Removed fields (`dtype`, `device`, `fuse_qkv`, `enable_offloading`, etc.) are deleted in one step. No `DeprecationWarning` → no-op → delete progression. VisualGen has no GA users; the marker discipline starts with this refactor's landing.

### 10.3 What we do for stability

- Every field on `VisualGenArgs` and every sub-config carries `Field(status="prototype")` (or higher, in future).
- The `Field(status=...)` marker surfaces in `model_json_schema()` via `json_schema_extra`, so tooling that introspects the schema sees the stability tag.
- The full enforcement story (snapshot YAMLs + per-field promotion process) is the subject of a separate design.

---

## 11. Cross-Cutting: Debug Knobs vs. Public Args

The existing TRT-LLM convention is `TLLM_*` env vars (`TLLM_LOG_LEVEL`, `TLLM_LOG_LEVEL_BY_MODULE`, `TLLM_NVTX_DEBUG`, etc.). For VisualGen-scoped knobs, the namespace is **`TLLM_VG_*`** (not `TLLM_VISUALGEN_*` — shorter, matches typical convention).

### 11.1 The migration list

| Today's field | New home | Why |
| --- | --- | --- |
| `skip_warmup` | `compilation_config.skip_warmup`, `status="prototype"` | Adjacent to torch.compile / cuda-graph capture; per-instance control (two `VisualGen` instances in the same process can disagree). |
| `skip_components` | env var `TLLM_VG_SKIP_COMP` (comma-separated) | Test-only knob — no production user, no example, no CLI flag; used in ~20 unit tests. Tests use `monkeypatch.setenv(...)` or a tiny pytest fixture. |
| `pipeline.enable_layerwise_nvtx_marker` | top-level flat field `VisualGenArgs.enable_layerwise_nvtx_marker`, `status="prototype"` | Live and wired; LLM has the identical field at `TorchLlmArgs.enable_layerwise_nvtx_marker`. Mirror LLM exactly. |

**Why `skip_components` is the only env-var migration**: it has no production caller. Every test that uses it constructs `VisualGenArgs(skip_components=[...])` so it can run with a mocked-out text encoder. Process-wide-via-env-var is the right shape for a test-only knob — it sheds a Pydantic field from the API surface, the test fixtures get a one-line setup, and no production user is affected (because there are none).

Implementation:

```python
# In PipelineLoader, reading the env var once at load time
_SKIP_COMP = [s.strip() for s in os.environ.get("TLLM_VG_SKIP_COMP", "").split(",") if s.strip()]
```

Pytest fixture:

```python
# tests/unittest/_torch/visual_gen/conftest.py
@pytest.fixture
def skip_components_env(monkeypatch):
    def _set(*components):
        monkeypatch.setenv("TLLM_VG_SKIP_COMP", ",".join(components))
    return _set
```

### 11.2 What stays as args (and why)

- **`compilation_config.resolutions` / `compilation_config.num_frames`** — production tuning knob; users explicitly want these in their YAML.
- **`attention_config.backend`** — production tuning.
- **`parallel_config.*`** — production deployment.
- **`enable_layerwise_nvtx_marker`** — live debug knob; mirrors LLM. Keeping it as a Pydantic field gives per-instance control.

### 11.3 What about `status="prototype"` fields?

`prototype` fields *can* live on the args class — that's what the status marker is for. The env-var carve-out is reserved for fields that are *test-only*. The two are orthogonal:

- `prototype` field = "this exists publicly but might break."
- env var = "this is a test/debug toggle, no public surface."

---

## 12. Cross-Cutting: YAML, CLI, dict Ingestion

The recommended shape (§6.3) preserves all current ingestion paths.

### 12.1 Direct Pydantic construction

```python
args = VisualGenArgs(
    model="Wan-AI/Wan2.1-T2V-14B",
    parallel_config=ParallelConfig(cfg_size=2),
    pipeline_config={"text_encoder_fsdp_size": 4},
)
```

### 12.2 dict ingestion

```python
args = VisualGenArgs(**config_dict)
```

`pipeline_config` is just a `dict[str, Any]` field; Pydantic deserializes it without any custom logic. Validation against the registry happens at `PipelineLoader.load(args)` (after `args.model` is known).

### 12.3 YAML

```yaml
model: "Wan-AI/Wan2.1-T2V-14B"
parallel_config:
  cfg_size: 2
compilation_config:
  resolutions:
    - [480, 832]
  num_frames: [33, 81]
  cuda_graph_max_batch_size: 4
pipeline_config:
  text_encoder_fsdp_size: 4
```

`yaml.safe_load` + `VisualGenArgs(**dict)` works directly.

#### Load / dump API

`VisualGenArgs` ships two convenience classmethods that wrap the
standard Pydantic v2 + PyYAML round-trip:

```python
from tensorrt_llm import VisualGenArgs

# Load (preferred)
args = VisualGenArgs.from_yaml("vg.yaml")

# Dump (preferred)
args.to_yaml("vg.out.yaml")
```

Implementation:

```python
@classmethod
def from_yaml(cls, path: str | Path) -> "VisualGenArgs":
    import yaml
    with open(path) as f:
        return cls(**(yaml.safe_load(f) or {}))

def to_yaml(self, path: str | Path) -> None:
    import yaml
    with open(path, "w") as f:
        yaml.safe_dump(
            self.model_dump(exclude_none=True),
            f, sort_keys=False,
        )
```

Both helpers are thin — they centralize `exclude_none=True` and the
preserved key order so the CLI side (and CI YAML comparisons) hit the
same shape every time. PyYAML is already a TRT-LLM dependency
(transitively via `pydantic-settings[yaml]` in `requirements.txt`), so
no new package surface.

For users who want the raw Pydantic path, the underlying idiom is
unchanged:

```python
import yaml
with open("vg.yaml") as f:
    args = VisualGenArgs(**yaml.safe_load(f))

with open("vg.out.yaml", "w") as f:
    yaml.safe_dump(args.model_dump(exclude_none=True), f, sort_keys=False)
```

For schemas, use `VisualGenArgs.model_json_schema()` (Pydantic v2 native).

### 12.4 CLI

VisualGen today exposes a single CLI flag for YAML injection in both `trtllm-serve` and `trtllm-bench`:

- `tensorrt_llm/commands/serve.py:824-829` declares `@click.option("--extra_visual_gen_options", ...)`.
- `tensorrt_llm/bench/benchmark/visual_gen.py:57-64` declares the same flag for the bench command.

**Add `--visual_gen_args` as the primary name**, with `--extra_visual_gen_options` retained as a backward-compat alias bound to the same destination. The new name matches the `VisualGenArgs` class name and aligns with the broader rename effort:

```python
@click.option(
    "--visual_gen_args",
    "--extra_visual_gen_options",
    "visual_gen_args",  # destination
    type=str,
    default=None,
    help="Path to a YAML file with VisualGen engine args.",
)
```

**`--config` is left exclusively on the LLM side.** At `commands/serve.py:701-712`, `--config | --extra_llm_api_options` bind to `extra_llm_api_options` (LLM args). Repurposing `--config` for VisualGen would require runtime dispatch based on `--backend` / `--server_role`, which click does not easily support. Cleaner to keep the two namespaces distinct.

For future `trtllm-serve` work, the vLLM `FlexibleArgumentParser` pattern (`--visual_gen_args config.yaml` + dotted CLI overrides like `--compilation_config.cuda_graph_max_batch_size=4`) is the most ergonomic. Not required for this milestone — the YAML route is sufficient.

---

## 13. Final Design — Public API

This section is the complete API contract this refactor lands. Every public class, every field, every classmethod, every import path, plus the namespace-collision analysis against `LlmArgs`. Source of truth for downstream readers, tooling, and docs.

### 13.1 API namespace & exports

The public namespace mirrors the LLM-side split: entry-point classes at top-level, sub-configs in the dedicated sub-package.

**Top-level `tensorrt_llm.*` — entry-point classes only**:

```python
from tensorrt_llm import (
    VisualGen,              # the engine class
    VisualGenArgs,          # engine-level config (this doc's main subject)
    VisualGenParams,        # per-request params (M2)
    VisualGenResult,        # output type (M2)
    VisualGenOutput,        # output type (M2)
    VisualGenMetrics,       # metrics type (M2)
    ExtraParamSchema,       # per-model extra-params schema (M2)
)
```

(Matches the seven names re-exported by `tensorrt_llm/__init__.py` on
`origin/main`. The earlier `VisualGen*Error` classes were dropped from
the public API in `tensorrt_llm.llmapi`; construction / load failures
surface as standard `ValueError` / `pydantic.ValidationError`.)

**Sub-package `tensorrt_llm.visual_gen.*` — sub-configs + registry helpers**:

```python
from tensorrt_llm.visual_gen import (
    CompilationConfig,      # §13.4 — flat compilation umbrella
    ParallelConfig,         # §13.5 — DiT-shaped parallelism
    AttentionConfig,        # §13.6
    CacheConfig,            # §13.7 — discriminated union TeaCacheConfig | CacheDiTConfig
    TeaCacheConfig,
    CacheDiTConfig,
    ModelEntry,             # registry entry shape (§13.8)
    QuantConfig,            # re-exported from tensorrt_llm.llmapi for convenience
)
```

`PipelineComponent` is deliberately *not* exposed publicly — the
`TLLM_VG_SKIP_COMP` env var takes comma-separated strings, and the
enum parsing is internal to `PipelineLoader`.

`QuantConfig` is **re-exported** from `tensorrt_llm.visual_gen` (same
class object as `tensorrt_llm.llmapi.QuantConfig`); users can import
it from either namespace. The `from tensorrt_llm.llmapi import QuantConfig`
path remains canonical for LLM users; VisualGen users get a single
`from tensorrt_llm.visual_gen import ...` line.

### 13.2 Naming collision vs `LlmArgs`

LLM today exports its sub-configs at `tensorrt_llm.llmapi.*` and its entry-point classes (`LLM`, `LlmArgs`, `TorchLlmArgs`, `TrtLlmArgs`) at top-level `tensorrt_llm.*`. VisualGen mirrors this split: entry-points at top-level, sub-configs in the sub-package.

After this refactor, the relevant class names — and whether they exist in both subpackages:

| Class name | `tensorrt_llm.llmapi.*` | `tensorrt_llm.visual_gen.*` | Collision risk |
| --- | --- | --- | --- |
| `CompilationConfig` | none (no umbrella; `CudaGraphConfig` + `TorchCompileConfig` are flat peers on `TorchLlmArgs`) | yes — flat umbrella | **None** — only VisualGen exports this name |
| `ParallelConfig` | none (LLM uses internal `_ParallelConfig` derived from flat `tensor_parallel_size` / `pipeline_parallel_size` / `context_parallel_size`) | yes | **None** — only VisualGen exports this name |
| `AttentionConfig` | none (LLM uses flat `attn_backend: str`) | yes | **None** |
| `CacheConfig` | none (LLM has `KvCacheConfig`, not `CacheConfig`) | yes | **None** |
| `QuantConfig` | yes | (reused from llmapi — same class object) | **None** — VisualGen imports from llmapi |
| `CudaGraphConfig` | yes (3 fields: `batch_sizes`, `max_batch_size`, `enable_padding`) | **none after this refactor** (folded into `CompilationConfig`) | Was a same-name collision before; **resolved by the fold** |
| `TorchCompileConfig` | yes (6 fields: `enable_fullgraph`, `enable_inductor`, `enable_piecewise_cuda_graph`, `capture_num_tokens`, `enable_userbuffers`, `max_num_streams`) | **none after this refactor** (folded into `CompilationConfig`) | Same — **resolved by the fold** |

**Verdict: no public name collision after the refactor.** Before, VisualGen and LLM both had `CudaGraphConfig` and `TorchCompileConfig` classes with different field sets — a latent foot-gun if either side ever exported them at top-level. Folding both into VisualGen's `CompilationConfig` (§13.4) eliminates the only same-name collision.

A user wanting both engines in the same script writes:

```python
from tensorrt_llm import LLM, LlmArgs, VisualGen, VisualGenArgs
from tensorrt_llm.llmapi   import KvCacheConfig, CudaGraphConfig
from tensorrt_llm.visual_gen import CompilationConfig, ParallelConfig as VgParallelConfig
from tensorrt_llm.llmapi   import QuantConfig    # shared
```

No name shadowing — the imports come from different namespaces. `VgParallelConfig` aliasing is optional (LLM doesn't export a `ParallelConfig` at all today), shown only as a pattern for any same-name additions in the future.

### 13.3 `VisualGenArgs` — top-level engine config

```python
# tensorrt_llm/visual_gen/args.py
# (new public location per M2 §3.1 / §9 Option C; today lives under _torch/visual_gen/)

from typing import Any
from pydantic import PrivateAttr
from tensorrt_llm.llmapi.llm_args import Field, StrictBaseModel
from tensorrt_llm.llmapi import QuantConfig

class VisualGenArgs(StrictBaseModel):
    """Engine-level configuration for VisualGen.

    Cross-cutting concerns are typed sub-configs with `_config` suffix
    (mirrors `LlmArgs`). Per-architecture knobs live in
    `pipeline_config: dict[str, Any]`, validated against the
    registry's defaults at load time. Discover via
    `VisualGen.supported_models()` / `VisualGen.pipeline_config(model)`.
    """

    # ── Loading ──────────────────────────────────────────────────
    model:    str        = Field(description="HF model id or local path.")
    revision: str | None = Field(default=None, status="prototype",
        description="HF Hub snapshot ID (branch / tag / commit SHA).")

    # ── Cross-cutting sub-configs (typed; _config suffix matches LlmArgs) ─
    parallel_config:    ParallelConfig    = Field(default_factory=ParallelConfig)
    compilation_config: CompilationConfig = Field(default_factory=CompilationConfig)
    attention_config:   AttentionConfig   = Field(default_factory=AttentionConfig)
    cache_config:       CacheConfig | None = None
    quant_config:       QuantConfig       = Field(default_factory=QuantConfig)

    # ── Observability (flat top-level; mirrors TorchLlmArgs.enable_layerwise_nvtx_marker) ─
    enable_layerwise_nvtx_marker: bool = Field(default=False, status="prototype",
        description="Emit per-layer NVTX ranges for nsys profiling.")

    # ── Per-architecture knobs (dict; resolved via HF-id registry at load) ─
    pipeline_config: dict[str, Any] = Field(
        default_factory=dict, status="prototype",
        description=(
            "Model-specific overrides. Keys must match the registry entry "
            "for `model`; unknown keys raise at load time. "
            "Discover via VisualGen.pipeline_config(model)."
        ),
    )
```

The two derived flags `dynamic_weight_quant` and `force_dynamic_quantization`
(populated by today's `_parse_quant_config_dict` validator) **do not
live on `VisualGenArgs`** — they move to the internal
`DiffusionModelConfig` (the merged config that `PipelineLoader` builds
from `VisualGenArgs` + HF metadata, at `config.py:570` on
`origin/main`). `DiffusionModelConfig` is already the home for
derived/merged state in today's code; the two flags are a clean
addition there. `VisualGenArgs` carries no `PrivateAttr` for them,
since they are not even instance-private to the user-facing class.

### 13.4 `CompilationConfig` — flat umbrella

Absorbs today's `CudaGraphConfig` (1 field) and `TorchCompileConfig` (3 fields) as prefixed flat fields, plus the existing `resolutions` / `num_frames` warmup-shape config and the new `skip_warmup`.

```python
class CompilationConfig(StrictBaseModel):
    """Flat compilation umbrella — covers torch.compile, CUDA graphs, and
    warmup. No nested TorchCompileConfig / CudaGraphConfig classes; their
    fields are flattened here with prefix names.
    """

    # ── Warmup shapes (existing; Cartesian product of resolutions x num_frames) ─
    resolutions: list[tuple[int, int]] | None = Field(default=None, status="prototype",
        description="(H, W) shapes to warmup at startup; None → per-model defaults.")
    num_frames:  list[int] | None             = Field(default=None, status="prototype",
        description="Frame counts to warmup at startup; None → per-model defaults. "
                    "For image models, use [1].")

    # ── Warmup control (moved from flat VisualGenArgs.skip_warmup) ───
    skip_warmup: bool = Field(default=False, status="prototype",
        description="Skip the warmup pass after weight load + compile.")

    # ── CUDA graphs (folded from CudaGraphConfig) ───────────────
    cuda_graph_enable: bool = Field(default=False, status="prototype",
        description="Capture and replay CUDA graphs after warmup.")

    # ── torch.compile (folded from TorchCompileConfig) ──────────
    torch_compile_enable:           bool = Field(default=True,  status="prototype")
    torch_compile_enable_fullgraph: bool = Field(default=False, status="prototype")
    torch_compile_enable_autotune:  bool = Field(default=True,  status="prototype")
```

Field renaming convention: today's `CudaGraphConfig.enable_cuda_graph` → `cuda_graph_enable`; today's `TorchCompileConfig.enable_torch_compile` → `torch_compile_enable`; the rest just gain the `torch_compile_` prefix. Discovery becomes one config to read; ergonomics are `args.compilation_config.cuda_graph_enable` instead of `args.cuda_graph.enable_cuda_graph`.

### 13.5 `ParallelConfig` — DiT-shaped parallelism

Cross-cutting DiT parallelism for every diffusion model. Model-specific parallel knobs (LTX-2 stage-2 refiner, Wan T5 encoder) move to the registry's `defaults` dict — they are *not* on this class anymore.

```python
class ParallelConfig(StrictBaseModel):
    """Parallelism axes shared across DiT-shaped diffusion models.

    Field names drop today's `dit_` prefix — the class name already
    says DiT-shaped, so the prefix is redundant on every field. Users
    write `parallel_config.dp_size`, not `parallel_config.dit_dp_size`.

    Per-architecture parallel knobs (LTX-2 stage-2 refiner, Wan T5
    encoder) live in the registry's defaults dict (§13.8), not here.
    """

    # ── VAE parallelism ─────────────────────────────────────────
    enable_parallel_vae:    bool                       = Field(default=True,    status="prototype")
    parallel_vae_split_dim: Literal["width", "height"] = Field(default="width", status="prototype")

    # ── DiT parallelism axes (prefix dropped per the rename above) ─
    dp_size:         int = Field(default=1, ge=1, status="prototype",
        description="Data parallelism degree.")
    tp_size:         int = Field(default=1, ge=1, status="prototype",
        description="Tensor parallelism degree. Not yet implemented.")
    ulysses_size:    int = Field(default=1, ge=1, status="prototype",
        description="Ulysses head-sharding degree.")
    ring_size:       int = Field(default=1, ge=1, status="prototype",
        description="Ring attention context parallelism. Not yet implemented.")
    attn2d_row_size: int = Field(default=1, ge=1, status="prototype",
        description="Attention2D row parallelism degree.")
    attn2d_col_size: int = Field(default=1, ge=1, status="prototype",
        description="Attention2D col parallelism degree.")
    cfg_size:        int = Field(default=1, ge=1, status="prototype",
        description="CFG (classifier-free guidance) batch parallelism.")
    fsdp_size:       int = Field(default=1, ge=1, status="prototype",
        description="FSDP sharding degree for DiT weights.")
    dim_order:       str = Field(default=DEFAULT_DIM_ORDER, status="prototype",
        description="Outermost-to-innermost ordering of parallel axes for "
                    "the device mesh; innermost = most contiguous ranks.")

    # ── Derived properties (unchanged from today's ParallelConfig) ─
    @property
    def seq_parallel_size(self) -> int: ...
    @property
    def n_workers(self) -> int: ...
    @property
    def total_parallel_size(self) -> int: ...

    def validate_world_size(self, world_size: int) -> None: ...
```

**Removed from today's `ParallelConfig`**:

- `refiner_dit_dp_size`, `refiner_dit_tp_size`, `refiner_dit_ulysses_size`, `refiner_dit_ring_size`, `refiner_dit_cp_size`, `refiner_dit_cfg_size`, `refiner_dit_fsdp_size` — all move to the `LTX2Pipeline` registry entry's defaults (with the same `dit_` prefix drop: `refiner_dp_size`, `refiner_tp_size`, etc.).
- `t5_fsdp_size` — moves to the `WanPipeline` registry entry's defaults as `text_encoder_fsdp_size`.

### 13.6 `AttentionConfig`

```python
class AttentionConfig(StrictBaseModel):
    """Attention backend selection. Flat for now; promote to discriminated
    union (§8.2) when backends need backend-specific kwargs.
    """
    backend: Literal["VANILLA", "TRTLLM", "FA4"] = Field(
        default="VANILLA", status="prototype",
        description="VANILLA = PyTorch SDPA; TRTLLM = TRT-LLM kernels; FA4 = FlashAttention 4.",
    )
```

### 13.7 `CacheConfig` — discriminated union (unchanged from today)

```python
class BaseCacheConfig(StrictBaseModel):
    cache_backend: str

class TeaCacheConfig(BaseCacheConfig):
    cache_backend: Literal["teacache"] = "teacache"
    teacache_thresh: float       = Field(default=0.2, gt=0.0, status="prototype")
    use_ret_steps:   bool        = Field(default=False, status="prototype")
    coefficients:    list[float] = Field(default_factory=lambda: [1.0, 0.0], status="prototype")
    # ... (full definition matches today's TeaCacheConfig; preserved verbatim)

class CacheDiTConfig(BaseCacheConfig):
    cache_backend: Literal["cache_dit"] = "cache_dit"
    Fn_compute_blocks: int    = Field(default=1, ge=0, status="prototype")
    Bn_compute_blocks: int    = Field(default=0, ge=0, status="prototype")
    max_warmup_steps:  int    = Field(default=4, ge=0, status="prototype")
    # ... (full definition matches today's CacheDiTConfig; preserved verbatim)

CacheConfig = Annotated[
    Union[TeaCacheConfig, CacheDiTConfig],
    Field(discriminator="cache_backend"),
]
```

This is the only sub-config that is *already* a discriminated union in today's code. Preserved as-is; the only change is adding `status="prototype"` markers on the individual fields.

### 13.8 Model registry & `ModelEntry`

Internal data structure; not part of the public top-level API surface. Exported from `tensorrt_llm.visual_gen.*` for inspection / testing.

**Keyed by Diffusers `_class_name`** (from `model_index.json`), mirroring the LLM side which keys `MODEL_MAP` by HF `architectures[0]` (`tensorrt_llm/models/__init__.py:139`, `tensorrt_llm/models/automodel.py:53-90`). One entry per *pipeline family*, not per checkpoint. Fine-tunes inherit the parent's `_class_name` and dispatch to the same entry automatically — no per-checkpoint registration.

```python
# tensorrt_llm/_torch/visual_gen/registry.py

from dataclasses import dataclass, field
from typing import Any

@dataclass
class ModelEntry:
    """One entry in the VisualGen model registry."""
    pipeline_cls: type                            # the pipeline class to construct at load
    hf_ids:       list[str] = field(default_factory=list)   # canonical HF ids for VisualGen.supported_models()
    defaults:     dict[str, Any] = field(default_factory=dict)  # default pipeline_config knobs (schema-by-example)
    doc:          str = ""                        # short docstring shown by discovery tooling

# Keyed by Diffusers `_class_name`, NOT by HF model id. ~3-5 entries
# total — one per pipeline family (matches today's `pipeline_registry.py`
# pattern on `origin/main`).
_MODEL_REGISTRY: dict[str, ModelEntry] = {}

def register_model(class_name: str, entry: ModelEntry) -> None:
    """Register a model in the VisualGen registry. Internal for v1."""
    if class_name in _MODEL_REGISTRY:
        raise ValueError(f"Model already registered: {class_name}")
    _MODEL_REGISTRY[class_name] = entry
```

Example entries populated in-tree by Phase 9:

```python
register_model(
    "WanPipeline",
    ModelEntry(
        pipeline_cls=WanPipeline,
        hf_ids=[
            "Wan-AI/Wan2.1-T2V-14B",
            "Wan-AI/Wan2.1-T2V-1.3B",
            "Wan-AI/Wan2.2-T2V-14B",
            # ... canonical T2V checkpoints. Fine-tunes auto-dispatch via
            # inherited `_class_name='WanPipeline'` and need not be listed.
        ],
        defaults={
            "text_encoder_fsdp_size": 1,
        },
        doc="Wan text-to-video family (T2V variants). DiT + T5 encoder.",
    ),
)

register_model(
    "LTX2Pipeline",
    ModelEntry(
        pipeline_cls=LTX2Pipeline,
        hf_ids=[
            "Lightricks/LTX-Video",
            "Lightricks/LTX-Video-13B-Distilled",
            # ... canonical LTX-Video checkpoints (single-stage and two-stage
            # share this entry; LTX2Pipeline.resolve_variant() picks the
            # right subclass at load time).
        ],
        defaults={
            # LTX-2 pipeline runtime knobs
            "text_encoder_path":      "",
            "spatial_upsampler_path": "",
            "distilled_lora_path":    "",
            # LTX-2 two-stage refiner parallelism (dit_ prefix dropped to match
            # the new ParallelConfig naming; see §13.5)
            "refiner_dp_size":      1,
            "refiner_tp_size":      1,
            "refiner_ulysses_size": 1,
            "refiner_ring_size":    1,
            "refiner_cp_size":      1,
            "refiner_cfg_size":     1,
            "refiner_fsdp_size":    1,
        },
        doc="Lightricks LTX-Video family. Single-stage and two-stage "
            "refinement variants share this entry — variant is decided at "
            "load time by `LTX2Pipeline.resolve_variant()`.",
    ),
)
```

The `defaults` dict is the schema-by-example for `pipeline_config`: it lists every pipeline-runtime knob the family accepts, with its default. There is no separate JSON-schema. The registry is the source of truth.

`hf_ids` is the canonical-checkpoints list — what `VisualGen.supported_models()` returns to the user. It is *not* the dispatch key (that's `_class_name`); it is purely a discovery hint. Fine-tunes inherit the parent's `_class_name` and dispatch automatically without needing to appear in `hf_ids`.

**Validation**: keys in `pipeline_config` (e.g., `text_encoder_path`, `refiner_dp_size`) must appear in the entry's `defaults` dict — strict validation at load time.

### 13.9 Discovery API on `VisualGen`

Two classmethods. Both read the registry directly.

```python
class VisualGen:
    @classmethod
    def supported_models(cls) -> list[str]:
        """Return the canonical HF model IDs across all registered
        pipeline families. Fine-tunes auto-dispatch and need not appear
        in this list (dispatch is keyed by Diffusers `_class_name`,
        which fine-tunes inherit from their parent)."""
        return [hf_id for e in _MODEL_REGISTRY.values() for hf_id in e.hf_ids]

    @classmethod
    def pipeline_config(cls, model: str | Path) -> dict[str, Any]:
        """Return the default pipeline_config knobs for the given model.

        `model` may be:
        - An HF model id (e.g., 'Wan-AI/Wan2.1-T2V-14B') — looked up in
          each entry's `hf_ids` list. The common user path.
        - A local checkpoint path — resolved by reading
          `model_index.json::_class_name` (with a safetensors-metadata
          fallback for LTX-2's native single-file format), the same way
          `PipelineLoader` does.
        - A registered Diffusers `_class_name` (e.g., 'WanPipeline') —
          returned directly without resolution.

        Raises KeyError if no entry matches. The returned dict is a
        copy — mutating it does not affect the registry.
        """
        # 1. HF id match (most common user-facing path).
        for entry in _MODEL_REGISTRY.values():
            if model in entry.hf_ids:
                return dict(entry.defaults)
        # 2. Direct _class_name match.
        if model in _MODEL_REGISTRY:
            return dict(_MODEL_REGISTRY[model].defaults)
        # 3. Local path — resolve _class_name via the same logic
        #    PipelineLoader uses.
        class_name = AutoPipeline._detect_from_checkpoint(model)
        return dict(_MODEL_REGISTRY[class_name].defaults)
```

Usage:

```python
>>> VisualGen.supported_models()
['Wan-AI/Wan2.1-T2V-14B', 'Wan-AI/Wan2.1-T2V-1.3B', 'Wan-AI/Wan2.2-T2V-14B',
 'black-forest-labs/FLUX.1-dev', 'black-forest-labs/FLUX.2-dev',
 'Lightricks/LTX-Video', 'Lightricks/LTX-Video-13B-Distilled', ...]

>>> VisualGen.pipeline_config("Wan-AI/Wan2.1-T2V-14B")   # by HF id (common path)
{'text_encoder_fsdp_size': 1}

>>> VisualGen.pipeline_config("/path/to/my-finetuned-wan-checkpoint")  # by local path
{'text_encoder_fsdp_size': 1}   # resolved to WanPipeline via model_index.json

>>> args = VisualGenArgs(
...     model="Wan-AI/Wan2.1-T2V-14B",
...     pipeline_config={"text_encoder_fsdp_size": 4},
... )
```

Load-time validation in `PipelineLoader` (not public API, specified here for reference):

```python
class_name = AutoPipeline._detect_from_checkpoint(args.model)
entry = _MODEL_REGISTRY[class_name]
unknown = set(args.pipeline_config) - set(entry.defaults)
if unknown:
    raise ValueError(
        f"Unknown pipeline_config keys for {class_name} ({args.model}): "
        f"{sorted(unknown)}. Valid keys: {sorted(entry.defaults)}"
    )
resolved = {**entry.defaults, **args.pipeline_config}
pipeline = entry.pipeline_cls(**resolved, ...)
```

### 13.10 End-to-end import & construction example

```python
from tensorrt_llm import VisualGen, VisualGenArgs
from tensorrt_llm.visual_gen import (
    ParallelConfig, CompilationConfig, AttentionConfig, CacheConfig, TeaCacheConfig,
)
from tensorrt_llm.llmapi import QuantConfig

args = VisualGenArgs(
    model="Wan-AI/Wan2.1-T2V-14B",
    revision=None,
    parallel_config=ParallelConfig(cfg_size=2, ulysses_size=2),
    compilation_config=CompilationConfig(
        resolutions=[(480, 832), (720, 1280)],
        num_frames=[33, 81],
        skip_warmup=False,
        torch_compile_enable=True,
        cuda_graph_enable=True,
    ),
    attention_config=AttentionConfig(backend="TRTLLM"),
    cache_config=TeaCacheConfig(teacache_thresh=0.2),
    quant_config=QuantConfig(),
    enable_layerwise_nvtx_marker=False,
    # VisualGen pipeline runtime knobs (strict; from the registry's defaults)
    pipeline_config={"text_encoder_fsdp_size": 4},
)

engine = VisualGen(args=args)
```

YAML equivalent:

```yaml
model: "Wan-AI/Wan2.1-T2V-14B"
parallel_config:
  cfg_size: 2
  ulysses_size: 2
compilation_config:
  resolutions: [[480, 832], [720, 1280]]
  num_frames: [33, 81]
  torch_compile_enable: true
  cuda_graph_enable: true
attention_config:
  backend: TRTLLM
cache_config:
  cache_backend: teacache
  teacache_thresh: 0.2
pipeline_config:
  text_encoder_fsdp_size: 4
```

### 13.11 Decision rationale (one-line per choice; cross-ref to design sections)

| Choice | Why | See |
| --- | --- | --- |
| Dispatch registry keyed by Diffusers `_class_name` | Mirrors LLM's `MODEL_MAP` keyed by HF `architectures[0]`; ~3-5 entries; fine-tunes auto-dispatch via inherited `_class_name` | §5 Axis A → A3; §13.8 |
| `pipeline_config(model)` accepts HF id / path / `_class_name` | User ergonomics keep working (query by HF id); internal `_detect_from_checkpoint` does the id → class-name resolution | §13.9 |
| Cross-cutting concerns stay typed sub-configs | Universal precedent; Pydantic gives validation + IDE completion for free | §5 Axis B → B2 |
| `_config`-suffix naming | Matches `LlmArgs` (`kv_cache_config`, `cuda_graph_config`, `quant_config`, ...) | §3.2 Takeaway 5; §7.5 |
| `CompilationConfig` as flat umbrella | Single discoverable surface; avoids three-hop `args.compilation_config.torch_compile_config.X` | §8.1 |
| Sub-configs under `tensorrt_llm.visual_gen.*`, not top-level | Mirrors LLM's `tensorrt_llm.llmapi.*` pattern; eliminates same-name collisions | §13.1, §13.2 |
| Strict unknown-key handling on `pipeline_config` | Safety net for typos in lieu of IDE completion. (LLM's analogous `model_kwargs` at `llm_args.py:2915` is loose because its keys are HF-defined and forward-compat matters; VisualGen's keys are pipeline-defined, so strictness wins.) | §9.3 |
| `Field(status="prototype")` on every new field | Reuses LLM-side marker; full snapshot harness deferred | §10 |
| No alias/deprecation shims | Pre-GA refactor; direct edits | §10.2, §13.2 |
| `TLLM_VG_*` env-var prefix | Shorter than `TLLM_VISUALGEN_*`; matches `TLLM_*` brevity convention | §11 |
| `skip_warmup` → `compilation_config`; `skip_components` → env var; `enable_layerwise_nvtx_marker` → top-level flat | Placement matches actual usage (compile-adjacent / test-only / LLM-mirror) | §11.1 |
| `--visual_gen_args` CLI primary name | Matches `VisualGenArgs` class name | §12.4 |
| `QuantConfig` reused from `tensorrt_llm.llmapi` | Same Pydantic class object on both sides; one source of truth for quant defaults | §13.1 |
| `CacheConfig` stays a discriminated union | Already correctly designed today; preserved verbatim | §13.7 |
| `dtype`, `device`, `pipeline.fuse_qkv`, three offload fields deleted | Verified unwired or hardcoded at `e527a9f785` | §2.2.5, §7.6 |

---

## 14. Migration Plan

### 14.1 Sequencing

The refactor breaks into independent phases. Each is mergeable on its own and testable in isolation. **No backwards-compatibility shims** — direct edits, users update their YAMLs.

**Phase 1 — Move `VisualGenArgs` and sub-configs to `tensorrt_llm/visual_gen/args.py`.** No structural change yet. Just move out of `_torch/visual_gen/config.py` per M2 §3.1 / §9 Option C. Keep re-exports from the old location for one release (only to avoid breaking internal imports). Adopt `Field(status="prototype")` on all fields.

**Phase 2 — Rename to `_config` suffix.** Rename `compilation → compilation_config`, `parallel → parallel_config`, `attention → attention_config`, `cache → cache_config`, `torch_compile → (folded into compilation_config)`, `cuda_graph → (folded into compilation_config)`. `quant_config` already has the suffix. Direct rename; users update YAMLs.

**Phase 3 — Fold `torch_compile` and `cuda_graph` into `compilation_config`.** Both become flat prefixed fields on `CompilationConfig` (`cuda_graph_batch_sizes`, `torch_compile_enable_inductor`, etc.). Update consumers in `PipelineLoader`.

**Phase 4 — Delete dead-code fields.**

- `dtype` (overridden to bf16 by `DiffusionModelConfig.torch_dtype`).
- `device` (LLM has no analogue; we're CUDA-only).
- `pipeline.fuse_qkv` (no production reader).
- `pipeline.enable_offloading`, `pipeline.offload_device`, `pipeline.offload_param_pin_memory` (zero consumers).

These four field removals are independent; can be done together or split. `PipelineConfig` is empty after the offload / fuse_qkv removals and disappears.

**Phase 5 — Internal-state cleanup.** Move `dynamic_weight_quant`, `force_dynamic_quantization` to `PrivateAttr`.

**Phase 6 — Move test/debug knobs.**

- `skip_warmup` → `compilation_config.skip_warmup`.
- `skip_components` → env var `TLLM_VG_SKIP_COMP`; update the test fixtures.
- `pipeline.enable_layerwise_nvtx_marker` → top-level flat `VisualGenArgs.enable_layerwise_nvtx_marker`.

**Phase 7 — Introduce the model registry.** Add `tensorrt_llm/_torch/visual_gen/registry.py` with `register_model(class_name, ModelEntry)`, keyed by Diffusers `_class_name` (mirrors today's `pipeline_registry.py` and LLM's `MODEL_MAP[architectures[0]]`). Populate one entry per currently-supported pipeline family (`WanPipeline`, `WanImageToVideoPipeline`, `FluxPipeline`, `Flux2Pipeline`, `LTX2Pipeline`), with their pipeline classes and empty-or-populated `defaults` dicts.

**Phase 8 — Add `pipeline_config: dict[str, Any]` to `VisualGenArgs`.** Strict validation. Update `PipelineLoader` to detect `_class_name` from the checkpoint (reuse `AutoPipeline._detect_from_checkpoint`), look up the registry entry, validate `pipeline_config` unknown keys against `entry.defaults`, merge, and pass the merged dict to `entry.pipeline_cls(...)`.

**Phase 9 — Move per-arch fields into registry defaults.**

- LTX-2 paths (`text_encoder_path`, `spatial_upsampler_path`, `distilled_lora_path`) → `_MODEL_REGISTRY["LTX2Pipeline"].defaults`.
- `parallel.refiner_dit_*` (7 fields) → `_MODEL_REGISTRY["LTX2Pipeline"].defaults` (with the same `dit_` prefix drop as the `ParallelConfig` rename: `refiner_dp_size`, `refiner_tp_size`, etc.).
- `parallel.t5_fsdp_size` → `_MODEL_REGISTRY["WanPipeline"].defaults["text_encoder_fsdp_size"]`.

Delete the corresponding fields from `VisualGenArgs` / `ParallelConfig` in the same PR.

**Phase 10 — Add the discovery API.** `VisualGen.supported_models()` returns the registered `_class_name`s; `VisualGen.pipeline_config(model)` accepts an HF id, local path, or `_class_name` and returns the entry's `defaults` dict (HF id / path resolved via `AutoPipeline._detect_from_checkpoint`).

**Phase 11 — Update CLI.** Add `--visual_gen_args` as primary in `commands/serve.py` and `bench/benchmark/visual_gen.py`; keep `--extra_visual_gen_options` as alias.

### 14.2 What we *don't* do for compat

- No `validation_alias` / `AliasChoices` for renames.
- No `Field(deprecated=...)` markers for removed fields.
- No soft-removal warning window.
- No backwards-compat YAML parsing for old field names.

VisualGen is pre-GA. The migration is a clean break.

---

## 15. Open Questions

1. ~~**Per-model typed Pydantic submodels.**~~ **Resolved** — rejected in favor of dict pass-through (§6.3). Trade-off: lose IDE completion for per-model knobs; gain minimal public API surface and symmetry with LLM-serving precedent.
2. **Runtime LoRA API for VisualGen.** Today's `distilled_lora_path` is LTX-2-internal (pipeline_ltx2_two_stages.py handles it at construction time). A general LoRA story for VisualGen should likely follow HF Diffusers (`pipeline.load_lora_weights(path)`, adapter naming, fusion / unfusion), not LLM's `LoraConfig`. **Tentative**: out of scope here; revisit when Wan or Flux needs LoRA. The construction-time `distilled_lora_path` registry default is acceptable for LTX-2 because it's part of the LTX-2 stage-2 algorithm, not a general LoRA loading affordance.
3. ~~**`fuse_qkv` defaults per arch.**~~ **Resolved** — moot. `fuse_qkv` is dead code; `qkv_mode` is hard-coded per attention site in the per-model transformer constructors. If/when QKV-fusion-as-config returns, it would be a knob on `attention_config`, not per-arch.
4. **When to bring back `dtype`.** Today the field is dead (hardcoded bf16). Adding it back requires the loader to actually resolve `"auto"` against per-model defaults. **Tentative**: add when a second model needs a non-bf16 default (e.g., fp16 for some legacy variant) or when fp8/fp4 inference is wired through.
5. **`AttentionConfig` evolution to a discriminated union.** §8.2 — when does this trigger? **Tentative**: as soon as a backend needs a backend-specific kwarg; not before.
6. **Promoting `pipeline_config` keys to typed fields.** When a model-specific key becomes used by 3+ commits and stable for 1+ release, should it get promoted to a typed sub-config field somewhere? **Tentative**: judgment-call; promote to a model-specific *typed* class only if multiple models share the key. Otherwise keep in the registry defaults.
7. **CLI dotted overrides.** vLLM-style `--visual_gen_args config.yaml` + `--compilation_config.cuda_graph_max_batch_size=4` overrides would be valuable but is not blocking. **Tentative**: defer to a follow-up PR.
8. **Out-of-tree model registration.** Should `register_model(class_name, entry)` be a public function? This enables third-party plugins but commits us to the registry contract (entry shape, `hf_ids` curation, namespace collisions). **Tentative**: not in this milestone; revisit if a plugin ecosystem emerges.
9. **Relationship with `DiffusionModelConfig` (the internal merged config).** `DiffusionModelConfig.from_pretrained` reads HF + args and produces the merged config. With dict-based model-specific knobs, the merging logic becomes a flat dict update — cleaner. **Tentative**: keep `DiffusionModelConfig` internal as M2 §10.3 proposes; the merge logic reads from `args.pipeline_config` directly.
10. **HF model ID matching strategy.** Exact-only today (§9.2). When checkpoint variants explode (each fine-tune as a separate HF id with the same config schema), pattern matching may be worth adding. **Tentative**: exact-only for v1; add prefix / wildcard support later if registration boilerplate becomes a pain point.

---

## 16. Appendix: Source Links

### TRT-LLM (current-state)

- [`tensorrt_llm/_torch/visual_gen/config.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/e527a9f785/tensorrt_llm/_torch/visual_gen/config.py) — `VisualGenArgs`, sub-configs, `DiffusionModelConfig`.
- [`tensorrt_llm/_torch/visual_gen/pipeline_loader.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/e527a9f785/tensorrt_llm/_torch/visual_gen/pipeline_loader.py) — `revision` consumer, `skip_warmup` / `skip_components` consumers, layer-wise NVTX hook registration.
- [`tensorrt_llm/llmapi/llm_args.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/llmapi/llm_args.py) — `LlmArgs`, `_config`-suffix convention (`kv_cache_config:1974`, `speculative_config:2023`, `cuda_graph_config:2763`, `torch_compile_config:2850`, `lora_config:1970`), `revision:1903`, `dtype:1900`, `enable_layerwise_nvtx_marker:3765`.
- [`tensorrt_llm/commands/serve.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/commands/serve.py) — `--config | --extra_llm_api_options` for LLM (`:701-712`), `--extra_visual_gen_options` for VisualGen (`:824-829`).
- [`tests/unittest/api_stability/`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/tests/unittest/api_stability) — API stability YAML reference test pattern (the LLM-side analogue; VisualGen harness deferred to a separate design).
- [`tensorrt_llm/llmapi/utils.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/llmapi/utils.py) — `set_api_status` decorator.

### vLLM

- `EngineArgs` flat → `VllmConfig` composed: [`vllm/engine/arg_utils.py:402-2175`](https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py#L402-L2175).
- `VllmConfig` composition: [`vllm/config/vllm.py:268-360`](https://github.com/vllm-project/vllm/blob/main/vllm/config/vllm.py#L268-L360).
- `hf_overrides` (dict pass-through for model-specific): [`vllm/config/model.py:85,254`](https://github.com/vllm-project/vllm/blob/main/vllm/config/model.py).
- `additional_config: dict | SupportsHash`: [`vllm/config/vllm.py:233`](https://github.com/vllm-project/vllm/blob/main/vllm/config/vllm.py#L233).
- Compilation config: [`vllm/config/compilation.py`](https://github.com/vllm-project/vllm/blob/main/vllm/config/compilation.py).
- Deprecation policy: [docs](https://docs.vllm.ai/en/v0.13.0/contributing/deprecation_policy/).
- RFC #24384 (decouple from HF): [issue](https://github.com/vllm-project/vllm/issues/24384).
- Issue #18707 (docs unusable): [issue](https://github.com/vllm-project/vllm/issues/18707).

### vLLM-Omni

- `OmniEngineArgs` (subclass of `EngineArgs`): [`vllm_omni/engine/arg_utils.py:46-75`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/engine/arg_utils.py#L46-L75) — `stage_connector_spec: dict`, `omni_kv_config: dict`.
- `OmniModelConfig(ModelConfig)`: [`vllm_omni/config/model.py:35`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/config/model.py#L35).
- Diffusion model registry: [`vllm_omni/diffusion/registry.py`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/registry.py).
- RFC #3366 (precedence chain): [issue](https://github.com/vllm-project/vllm-omni/issues/3366).

### SGLang

- `ServerArgs` (LLM) flat: [`python/sglang/srt/server_args.py:270`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py#L270) — `json_model_override_args: str` at `:422`; imperative model-arch branches at `:1168`.
- `ServerArgs` (Diffusion): [`python/sglang/multimodal_gen/runtime/server_args.py:268`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/server_args.py#L268) — `pipeline_config: PipelineConfig` with polymorphic resolution at `:826`.
- `PipelineConfig` per-arch base (`@dataclass`, not Pydantic): [`configs/pipeline_configs/base.py:149`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/configs/pipeline_configs/base.py#L149).
- Per-arch `*PipelineConfig` exports (16 in `__all__`, ~38 total): [`configs/pipeline_configs/__init__.py:33`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/configs/pipeline_configs/__init__.py#L33).
- Three-registry: [`registry.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/registry.py).
- Bug #20078 / Fix PR #20080.

### Diffusers

- `DiffusionPipeline.from_pretrained`: [`pipeline_utils.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/pipeline_utils.py).
- `ConfigMixin` + `@register_to_config`: [`configuration_utils.py:88`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/configuration_utils.py#L88) — stores kwargs in `self._internal_dict`; no separate `*PipelineConfig` classes.
- Per-pipeline typed `__init__` as the contract: [`stable_diffusion/pipeline_stable_diffusion.py:154,200`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L154).
- Runtime LoRA: `pipeline.load_lora_weights(path)`, `pipeline.set_adapters([...])`, `pipeline.fuse_lora()` / `unfuse_lora()` — on `LoraLoaderMixin` / `StableDiffusionLoraLoaderMixin`.
- Deprecation: [`utils/deprecation_utils.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/utils/deprecation_utils.py).

### Python / DL ecosystem

- Pydantic discriminated unions: [docs](https://pydantic.dev/docs/validation/latest/concepts/unions/).
- Pydantic reserved `model_*` namespace: [docs](https://docs.pydantic.dev/latest/concepts/models/#changing-the-model-namespace).
- HF Transformers `_LazyAutoMapping`: [`auto_factory.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/auto/auto_factory.py).
- PEFT registry: [`config.py`](https://github.com/huggingface/peft/blob/main/src/peft/config.py), [`mapping.py`](https://github.com/huggingface/peft/blob/main/src/peft/mapping.py).
- Hydra structured configs: [docs](https://hydra.cc/docs/tutorials/structured_config/intro/).
- PEP 702 `@warnings.deprecated`: [PEP](https://peps.python.org/pep-0702/).
- ExecuTorch deprecation policy: [docs](https://docs.pytorch.org/executorch/stable/api-life-cycle.html).

---

## Iteration Tracker

| #  | Date       | Reviewer / focus                                                            | Threads | Resolved | Open | Deferred |
|----|------------|-----------------------------------------------------------------------------|---------|----------|------|----------|
| 1  | 2026-05-08 | Codex — recommendation, field dispositions, migration ordering, stability, discovery, env-var/per-instance | 6       | 6        | 0    | 0        |
| 2  | 2026-05-08 | Codex — iter-1 normative-section drift, phase ordering, resolver/migration ownership, pseudocode bugs, capability-table rot | 5       | 5        | 0    | 0        |
| 3  | 2026-05-08 | Codex — `fuse_qkv` straggler in §6.4 sketch + §12 examples, unimplementable PipelineConfig shell, t5_fsdp_size target field bug, stale §4 principles | 4       | 4        | 0    | 0        |
| 4  | 2026-05-08 | Codex — `_advanced` is private in Pydantic v2, §10.2 stale opening + wrong layer-4 contract, residual `t5_parallel` references, Pydantic `deprecated=` doesn't fire at validation, scope-vs-Tier-2 contradiction | 5       | 5        | 0    | 0        |
| 5  | 2026-05-08 | Codex — offload migration rejects existing `pipeline:` YAML, §1 Exec Summary still defers Tier-2, discriminator both undecided and hardcoded, capability rows frozen at wrong granularity | 4       | 4        | 0    | 0        |
| 6  | 2026-05-08 | Codex — §6.4 sketch omits `advanced` namespace, four-vs-five-file contract leftover in normative sections, T5Config NameError ordering | 3       | 3        | 0    | 0        |
| 7  | 2026-05-08 | Codex — `PipelineConfig` referenced before declaration in §6.4 sketch | 1       | 1        | 0    | 0        |
| 8  | 2026-05-08 | Codex — §6.4 sketch still references `OffloadConfig` / `ObservabilityConfig` without declaring them; Scope still says discriminator is undecided | 2       | 2        | 0    | 0        |
| 9  | 2026-05-08 | Codex — strict convergence final-check pass | 0       | 0        | 0    | 0        |
| 10 | 2026-05-11 | Owner (Zhenhua) — reversed direction on §6 (dict pass-through over typed discriminated union); restored `_config` suffix to match `LlmArgs`; folded `torch_compile` / `cuda_graph` flat into `compilation_config`; dropped `dtype`/`device`/offload/`fuse_qkv` as dead code; `TLLM_VG_*` env-var prefix; `skip_warmup` → `compilation_config`, `skip_components` → env var; dropped alias / deprecation shims; `--visual_gen_args` CLI primary | 23      | 23       | 0    | 0        |
| 11 | 2026-05-11 | Owner (Zhenhua) — API-clarity pass: merged "recommendation" duplication (§1 short version + §14 list) into one consolidated **§14 Final Design — Public API** with complete class listings, namespace map, end-to-end example, decision-rationale table; added §14.2 collision analysis vs `LlmArgs` (resolved by folding `CudaGraphConfig`/`TorchCompileConfig` into `CompilationConfig` and namespacing sub-configs under `tensorrt_llm.visual_gen.*`) | 2       | 2        | 0    | 0        |
| 12 | 2026-05-12 | PR #9 review round 1 — Owner (Zhenhua) feedback on the open PR: swap §13 ↔ §14 so Final Design (API) reads before Migration; correct top-level export list (verified against `origin/main`: drop `VisualGen*Error`, add `VisualGenMetrics`/`VisualGenOutput`/`ExtraParamSchema`); drop public `PipelineComponent` export (env-var-only); re-export `QuantConfig` from `tensorrt_llm.visual_gen`; move two derived quant flags off `VisualGenArgs` into the internal `DiffusionModelConfig`; drop redundant `dit_` prefix on `ParallelConfig` fields; document YAML load/dump idiom (`yaml.safe_load` + `VisualGenArgs(**dict)` + `args.model_dump(exclude_none=True)`). | 9       | 6        | 3    | 0        |
| 13 | 2026-05-12 | PR #9 review round 2 — Owner follow-up on two open threads: (a) HF-id-vs-`_class_name` dispatch — clarified that LLM-side `MODEL_MAP` keys on HF `architectures[0]` (`automodel.py`, `models/__init__.py:139`), fine-tunes auto-dispatch via inherited architecture; proposed aligning VisualGen to the same shape (registry keyed by Diffusers `_class_name`, with `pipeline_config()` accepting HF id or path via internal resolution) — awaiting owner OK before editing §13.8. (b) `from_yaml`/`to_yaml` classmethods — verified PyYAML is already a TRT-LLM dep via `pydantic-settings[yaml]`; added the two classmethods to §12.3 per the reviewer's rule. | 2       | 1        | 1    | 0        |
| 14 | 2026-05-12 | PR #9 review round 3 — Owner clarified that LLM does have per-model user-side config (`LlmArgs.model_kwargs` at `llm_args.py:2915`, loose HF-config overrides) and asked for examples; OK'd two-dict design + strict `pipeline_config`. Doc updated: registry rekeyed by Diffusers `_class_name` (mirrors LLM's `MODEL_MAP[architectures[0]]`); ~3-5 entries instead of one-per-checkpoint; fine-tunes auto-dispatch. Added `model_kwargs: Optional[Dict[str, Any]]` to `VisualGenArgs` (loose, mirrors `LlmArgs.model_kwargs`) for HF/Diffusers config overrides; kept `pipeline_config` strict for VisualGen pipeline runtime knobs (registry-validated). `VisualGen.pipeline_config(model)` now accepts HF id / local path / `_class_name`. §1, §4, §5, §6, §7, §13.3, §13.8, §13.9, §13.10, §13.11, §14 Phases 7-10 all updated to reflect the two-dict + `_class_name`-keyed shape. | 1       | 1        | 0    | 0        |

| 15 | 2026-05-12 | PR #9 chat round 4 — Owner reversed the two-dict design: drop `model_kwargs` entirely (revisit only when a real use case appears); rename `extra_model_config` field + `VisualGen.extra_model_config()` method to `pipeline_config` (Pydantic-safe; `model_config` collides with `ConfigDict` class attribute). `supported_models()` reworked to return canonical HF ids (not `_class_name`s) via a new `ModelEntry.hf_ids: list[str]` field; dispatch still keys on `_class_name` so fine-tunes auto-resolve. `VisualGen.pipeline_config(model)` now resolves HF id → `_class_name` via `entry.hf_ids` lookup first, then `_detect_from_checkpoint` for local paths, then direct `_class_name`. | 4       | 4        | 0    | 0        |

**Converged on 2026-05-11 after iteration 11; rounds 1-3 of PR review feedback applied 2026-05-12 (8 resolved cumulatively on the PR), round 4 was an owner directive in chat that simplified the design back to one dict + HF-id-returning discovery. 1 thread still open on the PR — the acknowledgment-only thread the owner may still want to push back on.** Codex iterations 1-9 surfaced 30 thread-substantive issues all of which were resolved; owner-review iterations 10-11 raised 25 issues (all resolved) as a directional pivot + API-clarity follow-up; PR rounds 1-3 + chat round 4 raised 13 issues, of which 12 are resolved and 1 awaits a reviewer push-back decision. Severity decreased monotonically across the Codex iterations (high → medium → none). The doc's inline Codex/Claude review threads were stripped on finalize; this Iteration Tracker is the audit trail. Per-iteration commit history: `102e3df08a` (initial draft + Codex iter 1) → `09116d6604` (Codex iter 2) → `02e847c5a7` (Codex iter 3) → `d074c003bd` (Codex iter 4) → `a2b76d26cc` (Codex iter 5) → `b581b2ad0d` (Codex iter 6) → `6f152f7d3c` (Codex iter 7) → `a59ce5b2d9` (Codex iter 8) → `f469b20436` (Codex iter 9 strip + finalize) → owner-review iterations 10-11 → PR round 1 → PR round 2 → PR round 3 → chat round 4.
