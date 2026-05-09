# `VisualGenArgs` Refactor — Engine-Config Stability & Extensibility

> **Status**: Draft — under discussion
> **Date**: 2026-05-07
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
  `AttentionConfig`, `TorchCompileConfig`, `CudaGraphConfig`,
  `TeaCacheConfig`/`CacheDiTConfig`, `PipelineConfig`) — what stays, what
  splits, what merges.
- **Per-architecture model-specific config** — where it lives, how it
  is typed, and how it dispatches per model.
- **Field-by-field disposition** of every existing field in today's
  `VisualGenArgs` and its sub-configs (keep / move-to-model-config /
  move-to-env-debug-knob / make-internal). The user explicitly flagged
  the debug-knob escape via `TLLM_<MODEL>_CONFIG_<XYZ>`-style env vars
  as part of the migration to consider.
- **Stability marker convention** (e.g. `Field(status="prototype"|"beta"|"deprecated")`)
  and **API-stability test** for the new shape.
- **Migration plan** with backwards-compatibility aliases for renames
  and field moves.
- **Lightweight discovery affordance** — surface enough that users /
  tooling can introspect what's supported per model, but without
  building a parallel "schema metadata" system.

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
- **Implementation PRs / detailed impl plan** beyond the migration
  outline (the doc lists 10 phases, but this doc does not commit
  individual PR breakdowns, owners, or dates).
- **Final discriminator-name choice** for the per-architecture union
  (left as an Open Question; tentative answer captured but not locked).
- **Public registry contract for out-of-tree plugins** — left as an
  Open Question; not in this milestone.
- **Discovery-API polish** beyond `model_json_schema()` wrappers — not
  required for this design to land; can ship later.

### Target / Audience

- **TRT-LLM VisualGen engineers** (primary) — own the refactor execution.
- **TRT-LLM API / LLM API team** (secondary) — consumers of the
  `LlmArgs` patterns we plan to reuse (`Field(status=...)`,
  `tests/unittest/api_stability/`, sub-config composition).
- **Users of `VisualGen(model=..., args=VisualGenArgs(...))`** from
  Python or YAML (tertiary) — affected by the migration aliases and the
  new shape after deprecation cycles complete.

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
8. [Cross-Cutting: Nested-Sub-Config Extensibility](#8-cross-cutting-nested-sub-config-extensibility)
9. [Cross-Cutting: Discovery API](#9-cross-cutting-discovery-api)
10. [Cross-Cutting: Stability & Deprecation](#10-cross-cutting-stability--deprecation)
11. [Cross-Cutting: Debug Knobs vs. Public Args](#11-cross-cutting-debug-knobs-vs-public-args)
12. [Cross-Cutting: YAML, CLI, dict Ingestion](#12-cross-cutting-yaml-cli-dict-ingestion)
13. [Migration Plan](#13-migration-plan)
14. [Recommendation Summary](#14-recommendation-summary)
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
- **Internal/derived state leaking out** (`dynamic_weight_quant`, `force_dynamic_quantization`).

Every new model adds more fields. Every new optimization adds more fields. Every "stable" promise we make for one becomes an obligation we have to keep forever. Users can't tell which fields apply to *their* model on *their* GPU.

### The recommendation (short version)

**Adopt a hybrid: orthogonal sub-configs for what is genuinely cross-cutting, plus a discriminated `arch_config` union for everything that is per-architecture.**

```python
VisualGenArgs(StrictBaseModel):
    # ── Loading (general, stable) ──────────────────────────────────
    model: str                          # HF id or local path (renamed from checkpoint_path; M2 §3.4)
    revision: str | None = None
    dtype: str = "bfloat16"
    device: str = "cuda"

    # ── Cross-cutting, orthogonal sub-configs (stable) ────────────
    parallel:    ParallelConfig    = ParallelConfig()
    compilation: CompilationConfig = CompilationConfig()
    quant:       QuantConfig       = QuantConfig()
    attention:   AttentionConfig   = AttentionConfig()
    cache:       CacheConfig | None = None        # discriminated union (already shipped)
    offload:     OffloadConfig | None = None      # carved out of today's PipelineConfig
    observability: ObservabilityConfig = ObservabilityConfig()  # logs/traces/metrics

    # ── Per-architecture model config (extensible via registry) ────
    arch_config: ArchConfig | None = None
        # ArchConfig = Annotated[Union[WanModelConfig, FluxModelConfig, LTX2ModelConfig, ...],
        #                        Field(discriminator='arch')]
        # `None` → resolved from HF model_index.json at load time.
```

- **Cross-cutting sub-configs** stay typed and orthogonal — the Lightning / vLLM `VllmConfig` pattern. New optimization knobs go into the right sub-config; if a knob doesn't fit, that's the signal to add a new orthogonal sub-config (offload, observability, etc.) rather than bolt onto `VisualGenArgs`.
- **Per-architecture config is a discriminated union** — the Diffusers / PEFT / vLLM `SpeculativeConfig` / TRT-LLM `BaseSparseAttentionConfig.from_dict` pattern. Adding a new model = `class FooModelConfig(BaseModelConfig)` + one registry line, never editing `VisualGenArgs`. We get IDE completion, JSON-schema introspection, and Pydantic validation for free.
- **The dict pass-through with discovery API** that the user sketched is a real option, but it shifts validation cost to the pipeline and gives up IDE/static-analysis benefits the typed-submodel approach gets for free. We carry **one tightly-scoped escape hatch** (`extra_args: dict`) inside each model submodel for genuine forward-compat, modelled on Diffusers' `unused_kwargs` warn-don't-fail behavior — not as the primary surface.
- **Debug-only and pre-stable knobs leave `VisualGenArgs` entirely**, replaced by `TLLM_VISUALGEN_*` env vars (e.g. `TLLM_VISUALGEN_SKIP_WARMUP=1`). Today's `skip_warmup`, `skip_components`, `enable_layerwise_nvtx_marker`, and the internal quant flags (`dynamic_weight_quant`, `force_dynamic_quantization`) all move out.
- **Stability is enforced**, not policy — copy the LLM-API pattern: a `Field(status=...)` marker (already exists in `tensorrt_llm/llmapi/llm_args.py`), a YAML reference in `tests/unittest/api_stability/visual_gen_args.yaml`, and `pydantic.deprecated` aliases for renames (≥ 2 minor releases).

The biggest single payoff: when someone proposes a fourth diffusion model, `VisualGenArgs` doesn't change. They register `class FooModelConfig` with its own `text_encoder_path` (or whatever it needs), the union picks it up, the API surface for users of *other* models is unaffected, and `arch_config_schema("foo")` returns the JSON schema without any extra work.

### What this doc does *not* commit to

- It does **not** prescribe whether the discriminator is `arch` (HF architectures field), `model_type`, or `pipeline_class_name`. That choice is sketched in §7 and §15 and intentionally kept as an open question for the implementation.
- It does **not** require migrating internal `DiffusionModelConfig` (the merged/parsed config built by `PipelineLoader`) — that stays internal and tracks the deferred `Diffusion*` → `VisualGen*` rename in M2 §10.3.
- It does **not** require the discovery API to ship in this milestone. JSON-schema introspection is "free" once the typed submodels exist; a polished `engine.config_schema()` method can wait.

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

`PipelineConfig.fuse_qkv` is a transformer-block fusion that's meaningful for some models and impossible for others (e.g. when QKV are separately quantized, or when the layout differs). `cache_dit` settings depend on the DiT block structure. `attention.backend = "TRTLLM"` requires kernels for the architecture. All of these are exposed as if they were universal knobs.

The user can set them to nonsensical combinations and only learn at warmup time. Worse, the validator in `VisualGenArgs` can't catch the cross-product because *it doesn't know what model is being loaded* — that resolution happens later in `PipelineLoader`.

#### 2.2.3 Internal state leaks to the public surface

`dynamic_weight_quant` and `force_dynamic_quantization` exist on `VisualGenArgs` only because the `_parse_quant_config_dict` validator splits a single user-facing `quant_config` dict into three things. They are **not** user-set fields — passing them yourself does nothing useful. But they appear in `VisualGenArgs.model_dump()`, in YAML round-trips, and in any auto-generated schema. We are committing to maintain shapes that aren't supposed to exist.

This is the inverse of vLLM's `additional_config: dict` escape hatch — there, the user has too much surface; here, the user has *fake* surface.

#### 2.2.4 Test/debug knobs masquerade as features

`skip_warmup`, `skip_components`, and `pipeline.enable_layerwise_nvtx_marker` are all "advanced" — they exist for debugging or fast iteration. They are not part of the production contract; if we ever change how warmup or component loading works, we will not honour these flags. But because they live on `VisualGenArgs`, users will find them, depend on them, and ask us to keep them when we want to remove them.

The TRT-LLM precedent for this is environment variables (`TLLM_LOG_LEVEL_BY_MODULE`, `TLLM_*` flags throughout the codebase). The user explicitly suggested `TLLM_<MODEL>_CONFIG_<XYZ>` for this purpose, which matches the existing convention.

#### 2.2.5 No first-class discoverability

A user of `from tensorrt_llm import VisualGen, VisualGenArgs` gets no signal about which fields apply to their model. Reading the API ref tells them everything that *exists*; nothing tells them what's *relevant*. There is no `engine.config_schema()` method, no per-model docstring, no `--help` filtered by model. This is the SGLang `_get_diffusers_model_info` problem — generic defaults dominate.

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
| **Model-specific fields** | HF config inheritance | Mix: flat (MoE, Mamba), nested (MultiModal lifted to flat), `additional_config: dict` | `OmniDiffusionConfig` has a few flat (`boundary_ratio`, `flow_shift`); rest via per-request `extra_args` and engine-level `custom_pipeline_args` | Per-model `PipelineConfig` subclass + per-model `SamplingParams` subclass | Per-pipeline class owns everything model-specific | Flat fields on `VisualGenArgs` (LTX-2 paths) + `parallel.refiner_*` |
| **Variant config dispatch** | Manual `from_dict` discriminator (`BaseSparseAttentionConfig`, `DecodingBaseConfig`) | `quantization: str` + dict; `Annotated[Union, Field(discriminator)]` for some | Two registries: `_OMNI_PIPELINES` (multi-stage) + `_DIFFUSION_MODELS` (registry.py) | Three registries: `_PIPELINE_REGISTRY`, `_PIPELINE_CONFIG_REGISTRY`, `_CONFIG_REGISTRY` (HF path + lambda detector) | `AutoPipelineForX` static OrderedDicts (closed) | None — no model dispatch on the config side |
| **Stability marker** | `Field(status="prototype"/"beta"/"deprecated")`; YAML API-stability tests | Policy doc + `@deprecated` decorator; **no** API tests | Inherits vLLM's; nothing additional | None | `deprecate(name, target_version, message)` per call site | None on fields; `set_api_status("prototype")` on a few methods |
| **Escape hatch** | `cp_config: dict[str,Any]` (status=prototype) | `additional_config: dict` (top level) | `extra_args: dict` (per request) + `custom_pipeline_args: dict` (engine level) | `diffusers_kwargs: dict[str,Any]` on the diffusers-fallback subclass only | Lenient `__init__`/`from_pretrained` kwargs (warn unknowns) | None (`extra="forbid"` everywhere) |
| **Discoverability** | `LlmArgs.model_fields`, `model_json_schema()` | `EngineArgs` argparse `--help` (huge) | Same as vLLM | argparse `--help` | `inspect.signature` per pipeline class | `VisualGenArgs.model_fields` (no per-model filter) |
| **Famous lesson** | n/a | #18707 docs / #24384 HF coupling RFC | #2887 / #3366 / #3313 — multi-source precedence chain unsolved | #20078 / #20080 — generic defaults overrode model-specific | None equivalent (per-pipeline-class isolates it) | n/a (still pre-stable) |

### 3.2 Five takeaways from the survey

**Takeaway 1 — Composition is universal; flatness is a CLI artefact.**
Every framework that ships a Python API with sub-configs (TRT-LLM `LlmArgs`, vLLM `VllmConfig`, vLLM-Omni `OmniDiffusionConfig`, SGLang `PipelineConfig`) eventually composes them. vLLM's `EngineArgs` looks flat at first, but `EngineArgs.create_engine_config()` ([vllm/engine/arg_utils.py:1624](https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py#L1624)) explicitly assembles `VllmConfig` from sub-configs. The flatness is a CLI ergonomics layer, not a structural choice. We should not confuse "easy to type at the CLI" with "right at the API level."

**Takeaway 2 — Per-architecture config wants its own typed home.**
SGLang has per-model `PipelineConfig` subclasses. Diffusers has per-pipeline classes. vLLM-Omni has per-architecture pipeline files registered in `_DIFFUSION_MODELS`. PEFT has per-method config classes registered in `PEFT_TYPE_TO_CONFIG_MAPPING`. **None** of these put architecture-specific config on a shared parent class. The exceptions (vLLM-Omni's `boundary_ratio` / `flow_shift` on `OmniDiffusionConfig`) are explicitly flagged as anti-patterns in source comments and RFCs.

**Takeaway 3 — Registries beat central enums for variant dispatch.**
TRT-LLM's `BaseSparseAttentionConfig.from_dict` and `DecodingBaseConfig.from_dict` already use the manual-dispatch idiom; vLLM-Omni and SGLang use lazy-import registries (`_LazyRegisteredModel`, `_LazyPipelineRegistry`) that allow out-of-tree registration. Pydantic's discriminated-union pattern (`Annotated[Union[A,B,C], Field(discriminator="type")]`) gives the same dispatch behavior with native validation, JSON-schema generation, and IDE completion. **Pydantic discriminator + open registry function** is the contemporary choice.

**Takeaway 4 — The "dict + discovery API" path has real precedent but real costs.**
Diffusers' `from_pretrained` accepts unknown kwargs and warns; `__call__` is strict. vLLM-Omni uses `extra_args: dict` per request. SGLang's diffusers-fallback class has `diffusers_kwargs: dict`. **All three localize the dict to a specific surface, not the entire config.** The cost of going dict-first is the SGLang #20078 bug class — generic defaults silently dominate when validation moves from the schema to the pipeline. Typed submodels with one tightly-scoped `extra_args` per submodel is the pattern that survived.

**Takeaway 5 — Stability requires enforcement, not just intent.**
TRT-LLM's `LlmArgs` already has `Field(status="prototype"|"beta"|"deprecated")` and `tests/unittest/api_stability/llm.yaml` snapshots. vLLM has a 3-release deprecation policy but no API tests, and famously breaks fields anyway (the V0/V1 transition required `_override_v1_args` shims in `create_engine_config`). SGLang has no markers and renames freely. We already have the right plumbing for `LlmArgs`; we should reuse it for `VisualGenArgs` from the start.

---

## 4. Design Principles

The principles below extend the M2 doc's principles to the args-specific concerns:

1. **Cross-cutting concerns get orthogonal sub-configs.** If a knob applies to every model and every backend (compilation, parallelism, KV-cache-style memory, observability), it lives in its own typed sub-config. Sub-configs do not know about each other.
2. **Architecture-specific config is per-architecture.** A field that is meaningful for one model is *not* a top-level field. It lives on a typed `XModelConfig` registered into a discriminated union.
3. **The args class is closed for modification, open for extension.** Adding a new model must not require editing `VisualGenArgs` or any cross-cutting sub-config. Adding a new optimization may add a sub-config, never a flat field.
4. **Internal state stays internal.** If a field is computed from another, it doesn't appear in the public schema. Use Pydantic's `PrivateAttr` (already used by `LlmArgs`) or move to `DiffusionModelConfig` (the internal merged config).
5. **Debug knobs are env vars, not args.** `TLLM_VISUALGEN_*` (matching the existing `TLLM_*` convention) for testing/debug toggles. The args class is the production contract.
6. **One escape hatch per submodel, never on the parent.** When extensibility is genuinely needed, a tightly-scoped `extra_args: dict[str, Any]` *inside* a model submodel keeps the cost local. No `additional_config: dict` on `VisualGenArgs` itself — that recreates vLLM's #18707.
7. **Stability is mechanically enforced.** Every field carries `status="prototype"/"beta"` until proven; promotion is a YAML-tracked event in `tests/unittest/api_stability/visual_gen_args.yaml`; renames go through `Field(alias=...)` for ≥ 2 minor releases.
8. **Discoverability follows from typing.** Once submodels are typed, `engine.config_schema(model="foo")` is a thin wrapper over `FooModelConfig.model_json_schema()`. We don't need a parallel "schema metadata" system.

---

## 5. Independent Design Axes

The four design choices below are **independent**. We can pick the answer to each axis without being forced into a particular answer on the others. §6 then enumerates concrete combinations.

### Axis A — Where do model-specific fields live?

| Option | Pattern | Examples |
| --- | --- | --- |
| **A1** | Flat fields on the parent (status quo) | Today's `text_encoder_path` etc. |
| **A2** | Per-model typed Pydantic submodel via discriminated union | Diffusers per-pipeline, vLLM `SpeculativeConfig`, TRT-LLM `BaseSparseAttentionConfig.from_dict` |
| **A3** | Generic dict pass-through validated by pipeline | vLLM-Omni `custom_pipeline_args`, SGLang `diffusers_kwargs`, the user's "discovery API" sketch |
| **A4** | Subclass `VisualGenArgs` per model | (No major framework actually does this; included for completeness) |

A1 is the source of pain (§2.2.1). A4 fragments the import path (`Wan VisualGenArgs` vs `LTX2 VisualGenArgs`) and breaks the "one engine class, many models" property. A2 vs A3 is the substantive choice.

### Axis B — Where do optimization configs live?

| Option | Pattern | Examples |
| --- | --- | --- |
| **B1** | Flat on the parent | Today's `attention.backend`, `cuda_graph.enable_cuda_graph` |
| **B2** | Orthogonal cross-cutting sub-configs | vLLM `CompilationConfig`, `CacheConfig`, `ParallelConfig`; TRT-LLM `KvCacheConfig`, `MoeConfig` |
| **B3** | Coupled to model config (per-architecture optimization classes) | None mainstream — all major frameworks keep optimizations cross-cutting |

B2 is the universal answer. The only question is granularity — is `pipeline.fuse_qkv` an "optimization" or a "model" knob? The principle: if every model can choose to enable/disable it, it's optimization (B2). If only certain models can support it, it goes in the model submodel (A2).

### Axis C — How does a user discover what applies?

| Option | Pattern | Examples |
| --- | --- | --- |
| **C1** | Static docs + `--help` | vLLM `EngineArgs` CLI help, SGLang argparse |
| **C2** | Schema introspection (`Model.model_json_schema()`) | Pydantic-native; LangChain tool schemas |
| **C3** | Per-model docstrings + class-level `inspect.signature` | Diffusers per-pipeline classes |
| **C4** | A purpose-built `engine.list_supported_args(model)` API | The user's "discovery API similar to VisualGenParams" idea |

If we adopt typed submodels (A2), C2 is free. C4 is a thin shim over C2. C1 alone is the docs-unusable failure mode (vllm#18707).

### Axis D — How is stability enforced?

| Option | Pattern | Examples |
| --- | --- | --- |
| **D1** | Convention only | SGLang |
| **D2** | Decorator + policy doc | vLLM (`@typing_extensions.deprecated` + 3-release policy) |
| **D3** | Field-level status + API-stability snapshot tests | TRT-LLM `LlmArgs` (`status=...`, `tests/unittest/api_stability/`) |
| **D4** | Versioned schema (proto-style) | None mainstream for Python config |

D3 is the strongest and is **already in our codebase** for `LlmArgs`. Adopting it for `VisualGenArgs` is mechanical.

---

## 6. Top-Level Shape — Options

These are the four concrete combinations of choices on the axes above. Each is presented with a sketch, pros, and cons.

### 6.1 Option A — Status Quo + Organic Growth

(Axes: A1, B1, C1, D1.)

Keep `VisualGenArgs` flat-with-sub-configs. Add fields as needed. Per-architecture pain is managed by docs.

**Sketch**: today's `VisualGenArgs` plus whatever the next model needs (e.g. `text_encoder_2_path`, `wan_special_flag`, `flux_inter_block_quant_mode`).

**Pros**:
- Zero refactor. Lowest short-term cost.
- Familiar to users who already wrote LTX-2 code.

**Cons**:
- All five categories of pain (§2.2) compound.
- Stability promise becomes an obligation we can't keep — every `status="stable"` field is a future migration cost.
- Lands us in vllm#18707 / sglang#20078 territory.

**Verdict**: Rejected. The whole point of this doc is that this option doesn't scale.

### 6.2 Option B — Composed Orthogonal Sub-Configs Only

(Axes: A1 *or* A3, B2, C1+C2, D3.)

Adopt the vLLM `VllmConfig` pattern fully: fully composed sub-configs, but no per-architecture submodel — model-specific knobs either stay flat (A1) or move into a single `extra_args: dict` (A3).

**Sketch**:
```python
class VisualGenArgs(StrictBaseModel):
    model: str
    revision: str | None = None
    dtype: str = "bfloat16"
    device: str = "cuda"
    parallel:    ParallelConfig
    compilation: CompilationConfig
    quant:       QuantConfig
    attention:   AttentionConfig
    cache:       CacheConfig | None
    offload:     OffloadConfig | None
    observability: ObservabilityConfig
    extra_args:  dict[str, Any] = {}    # for model-specific overflow
```

**Pros**:
- Improves on the status quo by tightening orthogonality.
- Familiar to users who know vLLM.
- Validation cost stays low (no per-model schemas to maintain).

**Cons**:
- **Doesn't solve the per-architecture creep**. The LTX-2 paths, Wan-T5 fsdp_size, refiner_* parallel fields all still need a home, and the `extra_args` dict re-creates the SGLang #20078 bug class — generic defaults dominate, no IDE help, no JSON schema.
- Discoverability does not extend below the orthogonal sub-config level (`config_schema()` returns the *types* of sub-configs but not which `extra_args` keys are valid for which model).

**Verdict**: Strictly better than 6.1 but only *partially* solves the problem. Use as a fallback if 6.4 proves too costly to implement, but don't ship as the long-term answer.

### 6.3 Option C — Discriminated Union, Minimal Sub-Configs

(Axes: A2, B1, C2, D3.)

Inverse of 6.2: keep the parent class flat for cross-cutting fields, but factor model-specific into a discriminated union.

**Sketch**:
```python
class VisualGenArgs(StrictBaseModel):
    model: str
    revision: str | None = None
    dtype: str = "bfloat16"
    device: str = "cuda"
    # Many flat optimization fields (today's pattern)
    enable_cuda_graph: bool = False
    enable_torch_compile: bool = True
    attn_backend: str = "VANILLA"
    dit_cfg_size: int = 1
    ...
    # Per-architecture
    arch_config: ArchConfig | None = None      # discriminated union
```

**Pros**:
- Solves the per-architecture pain (the biggest one, §2.2.1).
- IDE completion + JSON schema for model-specific.

**Cons**:
- Cross-cutting sub-configs already exist and work; flattening them out is a regression.
- Mixed style — flat for optimizations, nested for model — confusing.

**Verdict**: Inferior to 6.4. Mentioned for completeness.

### 6.4 Option D — Hybrid (Recommended)

(Axes: A2, B2, C2, D3.)

Both halves of the answer: orthogonal sub-configs for cross-cutting concerns, plus a discriminated `arch_config` union for per-architecture concerns. One scoped `extra_args` dict *inside each model submodel* for forward-compat.

**Sketch**:
```python
# tensorrt_llm/visual_gen/args.py (new public location, M2 §3.1, §9)

class BaseModelConfig(StrictBaseModel):
    """Base class for per-architecture model config. Each subclass
    declares an `arch: Literal[...]` discriminator."""
    arch: str
    extra_args: dict[str, Any] = Field(default_factory=dict, status="prototype",
        description="Forward-compat dict for fields not yet promoted to typed.")

class WanModelConfig(BaseModelConfig):
    arch: Literal["wan"] = "wan"
    fuse_qkv: bool = True
    refiner_dit_dp_size: int = 1
    # ... other Wan-only knobs

class FluxModelConfig(BaseModelConfig):
    arch: Literal["flux"] = "flux"
    # ... Flux-only knobs

class LTX2ModelConfig(BaseModelConfig):
    arch: Literal["ltx2"] = "ltx2"
    text_encoder_path: str = ""
    spatial_upsampler_path: str = ""
    distilled_lora_path: str = ""
    refiner_dit_dp_size: int = 1
    refiner_dit_tp_size: int = 1
    # ... LTX-2 only knobs

ArchConfig = Annotated[
    Union[WanModelConfig, FluxModelConfig, LTX2ModelConfig, ...],
    Field(discriminator="arch"),
]

class VisualGenArgs(StrictBaseModel):
    # ── Loading ────────────────────────────────────────────────
    model: str = Field(description="HF id or local path.")
    revision: str | None = None
    dtype: str = "bfloat16"
    device: str = "cuda"

    # ── Cross-cutting orthogonal sub-configs ──────────────────
    parallel:      ParallelConfig    = ParallelConfig()        # cross-model parallelism
    compilation:   CompilationConfig = CompilationConfig()      # warmup shapes
    torch_compile: TorchCompileConfig = TorchCompileConfig()
    cuda_graph:    CudaGraphConfig    = CudaGraphConfig()
    attention:     AttentionConfig    = AttentionConfig()
    cache:         CacheConfig | None = None
    quant:         QuantConfig        = QuantConfig()
    offload:       OffloadConfig | None = None                  # carved from PipelineConfig
    observability: ObservabilityConfig = ObservabilityConfig()  # nvtx, otlp_traces

    # ── Per-architecture (extensible via registry) ────────────
    arch_config: ArchConfig | None = Field(
        default=None, status="beta",
        description="Per-architecture knobs. Defaults inferred from "
                    "HF model_index.json::_class_name when None.",
    )
```

**Pros**:
- **Solves all five categories of pain** in §2.2.
- Cross-cutting sub-configs stay typed, orthogonal, discoverable.
- New model = `class FooModelConfig(BaseModelConfig)` + one registry line. No edits to `VisualGenArgs`.
- IDE completion + JSON-schema introspection for both the parent and the model submodel — `engine.arch_config_schema()` is a 1-liner.
- Pydantic discriminator gives natural `from YAML/dict` deserialization without the manual `field_mapping` dispatch in `update_llm_args_with_extra_dict`.
- Stability is enforced via the existing `Field(status=...)` + API-stability test pattern.
- The `extra_args: dict` escape hatch is **scoped per model**, so it inherits the SGLang `diffusers_kwargs` discipline (one fallback class with a dict, the typed path stays strict).

**Cons**:
- Higher one-time refactor cost than 6.1 / 6.2.
- Requires a model-arch detector (similar to SGLang's `_get_config_info` or vLLM-Omni's `enrich_config()`) to resolve `arch_config=None` against the HF `model_index.json`. This already exists in `DiffusionModelConfig.from_pretrained` and just needs to be lifted to construct the right submodel.
- Documentation needs a per-model section. (We'd need this anyway; today's docs gloss over the LTX-2 paths.)

**Verdict**: **Recommended.** Pays a one-time cost to land on the pattern that the entire ecosystem has independently converged on (Diffusers per-pipeline, vLLM `SpeculativeConfig`, vLLM-Omni `_DIFFUSION_MODELS`, SGLang per-model `PipelineConfig`, PEFT registry). Avoids the failure modes documented in vllm#18707, vllm-omni#3366, sglang#20078, and our own pain points.

> **🤖 Codex (iter 1) — open:** The recommended `model_config` field is already known to collide with Pydantic
>
> **Anchor:** §6.4 / Option D — Hybrid (Recommended)
>
> The main recommendation, examples, migration plan, discovery API, and summary all use `VisualGenArgs.model_config` as the new public discriminated union, but §15 later says Pydantic v2 reserves `model_config` / `model_*` and that this field name collides with Pydantic's class-level config. That is not a polish issue: it means the central API shape in §6.4 cannot be implemented as written.
>
> Leaving the name as an open question while repeatedly recommending `model_config` makes the rest of the doc ambiguous. Reviewers cannot tell whether the API being stabilized is `model_config`, `arch_config`, `pipeline`, or something else, and all alias examples (`args.model_config.<x>`) may be wrong.
>
> **Suggested direction:** Pick the final public field name now, preferably `arch_config` or another non-reserved name, and update §1, §6.4, §7, §9, §12, §13, §14, and §15 so the design evaluates and migrates one implementable API surface.

> **💬 Claude — addressed:** locked in `arch_config` as the field name and renamed `ModelConfig` → `ArchConfig` for the union type, the registry to `_ARCH_CONFIG_REGISTRY`, and the discovery method to `arch_config_schema()`. Submodel class names (`WanModelConfig`, `FluxModelConfig`, etc.) are unchanged because Pydantic's reserved namespace only collides with `model_*` *fields* on a Pydantic class, not class names. §15 Q1 retired (decision now baked into §6.4 / §7 / §9 / §12 / §13 / §14).

### 6.5 Why not the user's "dict + discovery API" as the *primary* surface?

The user's sketch — model-specific knobs as an opaque `dict[str, Any]` validated by the pipeline at runtime, with a "discovery API" like `engine.supported_extra_args` that surfaces what's accepted — is consistent with M2 §4.3's `extra_params` design for `VisualGenParams` (per-request) and is a credible answer for **engine-level** model-specific knobs too.

We considered it and our recommendation is to use it **as a localized escape hatch**, not as the primary surface, for these reasons:

1. **It's strictly weaker than typed submodels for the same effort.** Once you build a discovery API that walks model-specific specs, you've done 90% of the work to declare typed submodels. The remaining 10% (Pydantic discriminator, IDE typing, `model_json_schema()`) is automatic if you go typed; the dict path requires you to hand-roll all of it.
2. **It moves validation from the schema to the pipeline.** This is exactly what caused SGLang #20078: when "what's a valid field" is decided by the pipeline class rather than the type, generic defaults dominate. Pipeline-side validation can be made to work, but you spend the savings on plumbing and tests.
3. **It loses static-analysis benefits at the boundary that matters most.** `VisualGenArgs` is the boundary between user code and our engine. Static typing here is what catches typos in YAML, IDE-completes in notebooks, and powers the (planned) API-stability tests. A typed discriminated union gets all of this for free.
4. **The ecosystem has already run this experiment.** vLLM-Omni's `OmniDiffusionConfig` exposes `custom_pipeline_args: dict[str, Any]` as the engine-level escape; in practice, model-specifics still bled onto the typed surface (`boundary_ratio`, `flow_shift`) because the dict was hard to reason about. SGLang scoped its dict to a fallback class for the same reason.

What we **do** want from the user's idea is preserved in 6.4:

- `arch_config.extra_args: dict[str, Any]` exists *inside each typed submodel*, marked `status="prototype"`. This is the "forward-compat overflow" — when a knob is brand-new, it lands here; once it's stable, it gets promoted to a typed field on the submodel.
- A `VisualGen.arch_config_schema()` discovery method is trivial once submodels are typed (§9).

This combination matches the pattern that survived in the ecosystem.

---

## 7. Field-by-Field Review of Today's `VisualGenArgs`

This section walks every field that exists today. The four dispositions are:

- **Keep** on `VisualGenArgs` (cross-cutting, stable).
- **Move to `arch_config`** (per-arch) (per-architecture, lives in the new discriminated union).
- **Move to env var** (`TLLM_VISUALGEN_*`) — testing/debug, not part of the production contract.
- **Make internal** — derived from another field, surfaced via `PrivateAttr` or moved to `DiffusionModelConfig`.

Where the disposition is "move to `arch_config`", the **migration** column shows the destination; where it is "move to env var", the column shows the env var name; where it is "make internal", the column says how the value will be obtained.

### 7.1 General loading fields

| Field | Today | Disposition | Migration |
| --- | --- | --- | --- |
| `checkpoint_path` | flat str | **Keep** (renamed `model` per M2 §3.4) | M2 already proposes `model_path → model`. |
| `revision` | flat `str | None` | **Keep** | — |
| `device` | flat str | **Keep** | — |
| `dtype` | flat str | **Keep** | — |

These are universally meaningful (every model is loaded, every model has a revision/dtype/device). Keep. The `model_path → model` rename is already in M2.

### 7.2 LTX-2-specific paths

| Field | Today | Disposition | Migration |
| --- | --- | --- | --- |
| `text_encoder_path` | flat str | **Move to `arch_config`** (per-arch) | `LTX2ModelConfig.text_encoder_path` |
| `spatial_upsampler_path` | flat str | **Move to `arch_config`** (per-arch) | `LTX2ModelConfig.spatial_upsampler_path` |
| `distilled_lora_path` | flat str | **Move to `arch_config`** (per-arch) | `LTX2ModelConfig.distilled_lora_path` |

These are the textbook example of architecture-specific creep (§2.2.1). They get a typed home in `LTX2ModelConfig`. Wan and Flux users no longer see them in their `VisualGenArgs` schema. LTX-2 users see them with full validation, IDE completion, and JSON-schema metadata.

> **Open question** (§15): should `distilled_lora_path` instead become a cross-cutting `LoraConfig` like the LLM API, given that LoRA is conceptually general? See M2 §13 Q6 for the per-request LoRA discussion. For *engine-level* loading, keeping it on `LTX2ModelConfig` is correct for now (Wan/Flux don't have a comparable refinement-LoRA concept); promote to cross-cutting if a second model adopts the same pattern.

### 7.3 Component skip + warmup skip

| Field | Today | Disposition | Migration |
| --- | --- | --- | --- |
| `skip_components` | `List[PipelineComponent]` | **Move to env var** | `TLLM_VISUALGEN_SKIP_COMPONENTS=text_encoder,vae` |
| `skip_warmup` | bool | **Move to env var** | `TLLM_VISUALGEN_SKIP_WARMUP=1` |

Both are testing/debug knobs. Production users should not be touching them. Moving them to env vars matches the existing TRT-LLM convention (`TLLM_LOG_LEVEL_BY_MODULE`, `TLLM_*` flags throughout). The implementation reads the env var inside `PipelineLoader.load()` — the user-facing args class becomes simpler and the production contract gets tighter.

If keeping `skip_warmup` as a *user-facing* field is judged necessary (e.g. for fast smoke tests in CI), it should live in a clearly-marked `_advanced` namespace or carry `status="prototype"` permanently.

### 7.4 Quantization

| Field | Today | Disposition | Migration |
| --- | --- | --- | --- |
| `quant_config` | `QuantConfig` (from LLM side) | **Keep** (rename to `quant`?) | Cross-cutting; matches LLM side |
| `dynamic_weight_quant` | bool | **Make internal** | `PrivateAttr`; populated by `quant`'s parser |
| `force_dynamic_quantization` | bool | **Make internal** | `PrivateAttr`; populated by `quant`'s parser |

`quant_config` itself is fine — it's the same `QuantConfig` from `tensorrt_llm.models.modeling_utils`, shared with the LLM API. The two derived booleans are implementation artefacts (§2.2.3) and should not appear on the public surface. Move them to `PrivateAttr` (matching `LlmArgs._parallel_config` and `_quant_config`) and populate them in the existing `_parse_quant_config_dict` validator.

The renaming `quant_config → quant` is for consistency with `parallel`, `compilation`, etc. — single-word, terse, matching `attention`/`cache`/etc. (Open question; status quo with `quant_config` is fine if M2 prefers consistency with the LLM-side `QuantConfig` import name.)

### 7.5 Cross-cutting optimization sub-configs

| Field | Today | Disposition | Migration |
| --- | --- | --- | --- |
| `compilation: CompilationConfig` | typed sub-config | **Keep** | — |
| `torch_compile: TorchCompileConfig` | typed sub-config | **Keep** | Consider folding into `compilation` (M2 §15) |
| `cuda_graph: CudaGraphConfig` | typed sub-config | **Keep** | Consider folding into `compilation` |
| `attention: AttentionConfig` | typed sub-config | **Keep** | — |
| `parallel: ParallelConfig` | typed sub-config (with model-specific creep, see 7.7) | **Keep**, with carve-outs | — |
| `cache: CacheConfig | None` | discriminated union | **Keep** (already correctly designed) | — |

These are the success cases. `compilation`, `torch_compile`, `cuda_graph` together are slightly over-fragmented (vLLM puts them all under `CompilationConfig`); folding could be a follow-up but is not required for this refactor.

### 7.6 The `pipeline: PipelineConfig` mixed bag

`PipelineConfig` today has four unrelated fields. They split as follows:

| Field | Today | Disposition | Migration |
| --- | --- | --- | --- |
| `pipeline.fuse_qkv` | bool | **Delete** (no runtime read at `e527a9f785`; QKV fusion is hard-coded in `tensorrt_llm/_torch/visual_gen/modules/attention.py` via `QKVMode` enum, not driven by config) | — |
| `pipeline.enable_layerwise_nvtx_marker` | bool | **Move to env var** | `TLLM_VISUALGEN_NVTX_LAYERS=1` |
| `pipeline.enable_offloading` | bool | **Move to new `OffloadConfig`** | `offload: OffloadConfig | None` |
| `pipeline.offload_device` | Literal["cpu", "cuda"] | **Move to `OffloadConfig`** | `offload.device` |
| `pipeline.offload_param_pin_memory` | bool | **Move to `OffloadConfig`** | `offload.pin_memory` |

After these splits, `PipelineConfig` is empty and disappears entirely. The name `PipelineConfig` is reusable for the per-architecture model submodels if we end up choosing that name — see §15 Q1.

### 7.7 `ParallelConfig` carve-outs

`ParallelConfig` today is mostly cross-cutting but contains seven LTX-2-specific `refiner_*` fields and one Wan-specific `t5_fsdp_size`:

| Field | Today | Disposition | Migration |
| --- | --- | --- | --- |
| `parallel.dit_*` (cfg_size, ulysses_size, attn2d_*, tp_size, dp_size, fsdp_size, ring_size, dim_order) | flat (cross-cutting) | **Keep** | — |
| `parallel.enable_parallel_vae` | flat | **Keep** | — |
| `parallel.parallel_vae_split_dim` | flat | **Keep** | — |
| `parallel.refiner_dit_dp_size` ... `refiner_dit_fsdp_size` (7 fields) | flat (LTX-2 two-stage only; **no runtime read at `e527a9f785`** — intended-but-unused) | **Move to `arch_config`** (per-arch, `status="prototype"`) | `LTX2ModelConfig.refiner_parallel: ParallelConfig` (nested `ParallelConfig`; promote to `status="beta"` when wired into the two-stage refiner path) |
| `parallel.t5_fsdp_size` | flat (only Wan T5 path) | **Move to `arch_config`** (per-arch, `status="prototype"`) | `WanModelConfig.t5_parallel: ParallelConfig` (or fold into a `text_encoder: TextEncoderConfig`) |

> **🤖 Codex (iter 1) — open:** Rehoming no-op fields defeats the stated goal of shrinking accidental API surface
>
> **Anchor:** §7.6 / The `pipeline: PipelineConfig` mixed bag and §7.7 / `ParallelConfig` carve-outs
>
> The field dispositions move `pipeline.fuse_qkv`, seven `parallel.refiner_*` fields, and `parallel.t5_fsdp_size` into per-arch model submodels. Local code search does not support treating these as live user-facing knobs: `fuse_qkv` is defined on `PipelineConfig`, but I found no runtime read of `config.pipeline.fuse_qkv`; the model code hard-codes `QKVMode.FUSE_QKV` / `SEPARATE_QKV`. Likewise `refiner_dit_*` and `t5_fsdp_size` only appear in `ParallelConfig` definitions in the local tree.
>
> That means the design may be preserving dead surface area under a more respectable type. If these fields are unused today, moving them into `WanModelConfig` / `LTX2ModelConfig` makes them look intentional and eventually stable, exactly the API-compatibility trap the doc is trying to avoid.
>
> **Suggested direction:** Split §7 dispositions into "used and migrate", "unused but intended", and "delete/internalize". For `fuse_qkv`, `refiner_*`, and `t5_fsdp_size`, require a cited runtime read path and a test before assigning a public destination; otherwise classify them as removed/prototype-internal, not as model-config fields.

> **💬 Claude — addressed:** verified at commit `e527a9f785` — `git grep` finds zero runtime reads of `config.pipeline.fuse_qkv`, `parallel.refiner_dit_*`, or `parallel.t5_fsdp_size` in `tensorrt_llm/`. Re-classified into three buckets in §7.6 / §7.7: `fuse_qkv` → **Delete** (no plan to wire); `parallel.refiner_*` and `parallel.t5_fsdp_size` → **Move to `arch_config` with `status="prototype"`** (intended-but-unused, gated until wired). Added an explicit "live vs intended-but-unused vs delete" sub-table at the head of §7.6/§7.7 and updated §7.8 disposition counts. Migration plan §13 reordered so dead-field deletion lands as a separate phase.

**Key insight from this carve-out**: when a model has *multiple* DiT-like passes (LTX-2 stage-2 refinement, Wan T5 text encoder), each pass logically wants its own `ParallelConfig` instance. Putting them on the parent `ParallelConfig` as `refiner_*` and `t5_fsdp_size` is the leakage — the right shape is **one `ParallelConfig` per parallel-able stage**, owned by whatever owns the stage:

```python
class LTX2ModelConfig(BaseModelConfig):
    arch: Literal["ltx2"] = "ltx2"
    text_encoder_path: str = ""
    spatial_upsampler_path: str = ""
    distilled_lora_path: str = ""
    refiner_parallel: ParallelConfig = ParallelConfig()    # nested, reuses the same type
```

This pattern reuses the existing typed sub-config — no new types needed. It also extends naturally: when a future model adds a third pass, it gets its own `ParallelConfig` field on its submodel without touching anything else.

### 7.8 Summary of dispositions

| Bucket | Count | Examples |
| --- | --- | --- |
| **Keep on `VisualGenArgs`** | 4 flat + 7 sub-configs | `model`, `revision`, `dtype`, `device`; `parallel`, `compilation`, `torch_compile`, `cuda_graph`, `attention`, `cache`, `quant` |
| **Move to `arch_config` (per-arch)** | 11 fields | LTX-2 paths (3), `parallel.refiner_*` (7), `parallel.t5_fsdp_size` (1) — plus `pipeline.fuse_qkv` becomes per-arch |
| **Move to env var (`TLLM_VISUALGEN_*`)** | 3 fields | `skip_warmup`, `skip_components`, `pipeline.enable_layerwise_nvtx_marker` |
| **Make internal (PrivateAttr / DiffusionModelConfig)** | 2 fields | `dynamic_weight_quant`, `force_dynamic_quantization` |
| **New cross-cutting sub-config** | 1 (3 fields) | `OffloadConfig` (carves `enable_offloading`, `offload_device`, `offload_param_pin_memory` from `PipelineConfig`) |
| **Eliminated** | 1 sub-config | `PipelineConfig` (all four fields rehomed; class deleted) |

The net effect:

- `VisualGenArgs` shrinks from "4 flat + 4 LTX-2 + 2 test + 2 internal + 7 sub-configs" (19 surface concepts) to **"4 flat + 8 sub-configs + 1 `arch_config` union"** (13 surface concepts), with the per-architecture surface gated by the discriminator.
- The set of fields a Wan user sees in their schema goes from ~50 to ~30, and almost every field they see is meaningful for Wan.
- The set of fields an LTX-2 user sees is similar in count but now properly typed and grouped.

---

## 8. Cross-Cutting: Nested-Sub-Config Extensibility

The user explicitly flagged: *"for the args, they could be nested, for example we may extend the certain args to support new modes, so we need to be careful on such."* Three patterns are worth calling out.

### 8.1 Discriminated unions for "one of N modes"

This is what `cache: CacheConfig` already does today (`TeaCacheConfig | CacheDiTConfig`, discriminated by `cache_backend`). It is the right pattern for any sub-config that has multiple mutually-exclusive backends. Future candidates:

- **Sampler backend**: today `attention: AttentionConfig` has `backend: Literal["VANILLA", "TRTLLM", "FA4"]`. As backends accumulate (e.g. SAGE attention, etc.), the per-backend kwargs differ, and we want `attention: VanillaAttentionConfig | TrtllmAttentionConfig | FA4AttentionConfig` discriminated by `backend`. Today's flat `AttentionConfig` is a Literal-keyed enum but no per-backend kwargs — the moment a backend needs its own knobs, this should refactor.
- **Cache backend**: already done.
- **Quant backend**: TRT-LLM `QuantConfig` is currently an enum-based class. If quant backends start needing radically different config (e.g. NVFP4 vs FP8 vs W4A8 each need different group_size, dynamic flags, exclude lists), promote to a discriminated union.

### 8.2 Nested Pydantic for "components with their own config"

`compilation: CompilationConfig` is the current example: it owns `resolutions`, `num_frames`, and at validation time conceptually owns the `torch_compile` and `cuda_graph` sub-configs. The current shape (3 sibling fields on `VisualGenArgs`) is a slight leak — `torch_compile` and `cuda_graph` are pointers into compilation. A future cleanup would nest:

```python
class CompilationConfig(StrictBaseModel):
    resolutions: list[tuple[int, int]] | None = None
    num_frames: list[int] | None = None
    torch_compile: TorchCompileConfig = TorchCompileConfig()
    cuda_graph:    CudaGraphConfig    = CudaGraphConfig()
```

This is M2 §15 Q1 (whether to fold them) and is **not required for this refactor**, but the nesting pattern (Pydantic-in-Pydantic) is what we use whenever a sub-config naturally owns more sub-configs. It composes cleanly with the rest.

### 8.3 The "list of typed configs" pattern

Some optimizations are naturally ordered/lists, e.g. the multi-pass LTX-2 case where stage-1 and stage-2 each want a `ParallelConfig`. Two shapes are possible:

```python
# Option A: named fields on the model submodel
class LTX2ModelConfig(BaseModelConfig):
    parallel: ParallelConfig          # main DiT
    refiner_parallel: ParallelConfig  # stage-2 refinement DiT

# Option B: list of stages
class LTX2ModelConfig(BaseModelConfig):
    stage_parallel: list[ParallelConfig]  # one per stage
```

Option A is what § 7.7 recommends and what almost every real framework does (vLLM-Omni's stage configs, SGLang's per-encoder configs). Option B looks cleaner but loses field semantics — code that wants to validate "the refiner has the same TP size as the main" needs to remember "stage 1 is refiner". Stick with named fields.

---

## 9. Cross-Cutting: Discovery API

Once the typed-submodel approach is in place, the discovery API the user sketched ("similar to VisualGenParams") is essentially free.

### 9.1 What we get for free from Pydantic

Every `BaseModel` in Pydantic v2 already exposes:

```python
LTX2ModelConfig.model_json_schema()       # full JSON Schema
LTX2ModelConfig.model_fields              # dict of field info, including descriptions
```

The discriminated union also has `model_json_schema()`, which produces a JSON Schema with `oneOf:` over the variants — exactly what an OpenAPI tool would consume.

### 9.2 The proposed VisualGen surface — two tiers

> **🤖 Codex (iter 1) — open:** `model_json_schema(arch)` is too thin for the user problem stated in scope
>
> **Anchor:** §9.2 / The proposed VisualGen surface
>
> The scope says users cannot tell which fields apply to their model on their GPU, but the proposed discovery API only returns static Pydantic schema by architecture. That answers "what fields exist on `WanModelConfig`", not "is this field valid for `Wan2.1-T2V-1.3B` on H100 vs B200 with this attention backend, quant mode, cache backend, and resolution". The doc rejects the user's discovery-API instinct too quickly by treating discovery as just a wrapper around typed submodels.
>
> This is especially weak for cross-cutting knobs the doc keeps on the parent: `attention`, `cache`, `quant`, `parallel`, and `offload` are exactly where model/GPU/backend compatibility failures happen. Static JSON schema cannot express runtime capability constraints unless the resolver contributes model metadata and hardware/backend capability information.
>
> **Suggested direction:** Add a capability-oriented discovery contract, even if minimal: e.g. `VisualGen.resolve_args_schema(model=..., device=...)` or `VisualGen.supported_config(model=...)` returning resolved arch, defaults, supported/unsupported cross-cutting options, and reasons. Keep Pydantic schema as the type layer, but do not claim it solves per-model/per-GPU discoverability by itself.

> **💬 Claude — addressed:** split discovery into **two tiers** — a static *type* layer (Pydantic-native, free) and a resolved *capability* layer (specified, lightweight). The capability tier is the one that answers the user's "which fields apply to my model on my GPU" question; the type tier alone doesn't, as Codex correctly flagged. Both ship together because the capability tier's output references the type tier's schema.

#### Tier 1: static type schema (free from Pydantic)

```python
class VisualGen:
    @staticmethod
    def supported_models() -> list[str]:
        """Return the list of registered arch discriminators."""
        return list(_ARCH_CONFIG_REGISTRY.keys())

    @staticmethod
    def arch_config_schema(arch: str) -> dict:
        """Return the JSON schema for a specific model architecture."""
        cls = _ARCH_CONFIG_REGISTRY[arch]
        return cls.model_json_schema()

    @property
    def args_schema(self) -> dict:
        """Return the JSON schema for the entire VisualGenArgs surface."""
        return type(self.args).model_json_schema()
```

This mirrors the M2 `engine.supported_extra_params` design for
`VisualGenParams` (M2 §4.3). Useful for tooling that needs the type
contract: OpenAPI generation, IDE completion fallbacks, validation
front-ends. **Does not** answer per-model/per-GPU compatibility
questions — that's tier 2.

#### Tier 2: resolved capability schema

```python
@dataclass
class ResolvedConfig:
    """What this engine instance / target hardware actually supports."""
    arch: str                                   # resolved discriminator
    arch_config_schema: dict                    # tier-1 JSON schema for the arch
    supported_attention_backends: list[str]     # subset of AttentionConfig.backend
    supported_quant_algos: list[str]            # subset of QuantAlgo
    supported_cache_backends: list[str]         # subset of CacheConfig
    supported_compilation: dict                 # resolutions/num_frames/etc.
    parallel_constraints: dict                  # max DP/TP/Ulysses/CFG given device count
    notes: list[str]                            # human-readable reasons / caveats

class VisualGen:
    @staticmethod
    def resolved_config(
        model: str,
        device: str | None = None,
        gpus: int | None = None,
    ) -> ResolvedConfig:
        """Resolve which knobs are valid for `model` on `device`/`gpus`.

        Combines: registry lookup (arch + arch_config schema) +
        per-arch capability table (supported attention/quant/cache
        backends) + hardware capability table (sm version, device
        memory, NVLink fabric).
        """
```

The resolver in Phase 4 (`PipelineLoader`'s arch detector) already
knows the arch; tier 2 layers a *capability table* on top — one entry
per arch, listing the cross-cutting backends/algos/cache options that
arch supports, plus a coarse hardware capability table (sm version,
NVLink, etc.). When users ask "can I use FA4 + NVFP4 + cache_dit on
Wan 1.3B on H100?", tier 2 can answer; tier 1 cannot.

Tier 2 is **specified, not exhaustive** — this milestone ships:

- The `ResolvedConfig` type with the fields above.
- A capability table per arch, populated for the three current models
  (Wan, Flux, LTX-2).
- A small hardware capability table (sm 80 / sm 90 / sm 100 default
  rows; per-device overrides as needed).

It does **not** ship a full kernel-level compatibility matrix
(out-of-scope; some compatibility is only knowable at warmup time).
The notes list lets us explicitly say "warmup-time-only check" where
the tier-2 answer is "probably yes, but try it".

### 9.3 Why we still don't lean on tier 2 as a substitute for typing

Per the user's note ("consider it but do NOT lean to it too much"): the discovery API is a nice-to-have, not the foundation of the design. The **primary** way users learn what's supported is:

1. Static type checking + IDE completion on `WanModelConfig`, `LTX2ModelConfig`, etc.
2. The auto-generated API reference (`generate_api_docs_as_docstring`, already used by `LlmArgs`).
3. Per-model deployment guides under `docs/source/deployment-guide/`.

The runtime `arch_config_schema()` is for people who need it (tooling, OpenAI-compatible serving extensions, IDEs without static-analysis support).

---

## 10. Cross-Cutting: Stability & Deprecation

### 10.1 Reuse the LLM-side `Field(status=...)` machinery

`tensorrt_llm/llmapi/llm_args.py` already defines a `Field(default, *, status="prototype"|"beta"|"deprecated", **kwargs)` wrapper that adds `status` to `json_schema_extra`. Use it on `VisualGenArgs` and every sub-config:

```python
from tensorrt_llm.llmapi.llm_args import Field, StrictBaseModel

class VisualGenArgs(StrictBaseModel):
    model: str = Field(description="...", status="beta")
    arch_config: ArchConfig | None = Field(
        default=None, status="beta",
        description="Per-architecture knobs. Defaults inferred from "
                    "HF model_index.json::_class_name when None.",
    )
    ...
```

For new fields under active iteration, default to `status="prototype"`. Promote to `"beta"` when usage stabilizes. Mark fields targeted for removal as `status="deprecated"` with `deprecated="Use X instead."` (Pydantic v2 supports this natively).

### 10.2 API stability test

Add `tests/unittest/api_stability/visual_gen_args.yaml` and `references/visual_gen_params.yaml` following the existing pattern (`tests/unittest/api_stability/references/llm.yaml`). The test snapshots field annotation + default + status. Any non-prototype field added/removed/renamed without updating the YAML fails CI. This is the single most effective stability gate in the LLM API today.

> **🤖 Codex (iter 1) — open:** A single `visual_gen_args.yaml` snapshot is not enough for per-arch API stability
>
> **Anchor:** §10.2 / API stability test
>
> The doc says to add `tests/unittest/api_stability/visual_gen_args.yaml` following the LLM pattern, but the current harness is not enough for the proposed shape. The existing status check is hard-coded around `llm.yaml` and `TorchLlmArgs.model_fields`, and the generic Pydantic snapshot only captures direct `model_fields` for the registered test class. It will not automatically protect `WanModelConfig`, `FluxModelConfig`, `LTX2ModelConfig`, the discriminator values, the model-config registry, or legacy alias behavior.
>
> This matters because §6.4 moves the real extensibility surface out of `VisualGenArgs` and into registered submodels. If a later PR removes `WanModelConfig.t5_parallel`, renames `LTX2ModelConfig.refiner_parallel`, drops an arch from `_ARCH_CONFIG_REGISTRY`, or changes discriminator values, a top-level `VisualGenArgs` snapshot can still pass while user YAML breaks.
>
> **Suggested direction:** Define a VisualGen-specific stability harness: snapshot `VisualGenArgs`, every registered arch config class, the registry keys/discriminator literals, and representative old→new alias inputs. Treat registry drift and alias regression as API failures, not just field-list drift.

> **💬 Claude — addressed:** expanded §10.2 to specify a four-layer stability harness instead of a single YAML snapshot. New layers:
>
> 1. **`visual_gen_args.yaml`** — `VisualGenArgs.model_fields` (the existing pattern; covers cross-cutting fields).
> 2. **`visual_gen_arch_configs.yaml`** — one section per registered submodel (`WanModelConfig`, `FluxModelConfig`, `LTX2ModelConfig`, …). Snapshots fields, types, defaults, status. Registry drift (renaming a class, adding/removing one) requires a YAML edit.
> 3. **`visual_gen_arch_registry.yaml`** — frozen list of `arch:` discriminator literals (`"wan"`, `"flux"`, `"ltx2"`, …) plus the HF `_class_name` patterns the resolver matches on. Discriminator literal drift (changing `"ltx2"` → `"ltx_2"`) requires a YAML edit and a deprecation cycle.
> 4. **`visual_gen_alias_cases.yaml`** — table of representative old→new YAML inputs (flat-LTX-2, nested `parallel.refiner_*`, nested `parallel.t5_fsdp_size`, mixed-explicit-vs-legacy) with the expected resolved `arch_config` snapshot. The test loads each case through `VisualGenArgs(**input)` and asserts the resolved shape. Alias regression (a migration validator change that silently strips a field) fails CI here.
>
> All four files live under `tests/unittest/api_stability/references/`. The test driver uses the existing `api_stability_core.py` snapshot machinery for layers 1-2 and adds a small harness for layers 3-4.

### 10.3 Renames go through aliases

Pydantic supports `Field(alias="old_name")` and `Field(validation_alias=AliasChoices("new", "old"))`. For renames during the migration window:

```python
class ParallelConfig(StrictBaseModel):
    enable_parallel_vae: bool = Field(default=True,
        validation_alias=AliasChoices("enable_parallel_vae", "parallel_vae"),
        deprecated="`parallel_vae` is deprecated; use `enable_parallel_vae`.",
    )
```

Maintain aliases for ≥ 2 minor releases (matching the ExecuTorch policy and stronger than vLLM's 3-release).

### 10.4 Removals follow a "soft → hard" path

1. Field marked `status="deprecated"` with a `deprecated=...` message — emits `DeprecationWarning` on access.
2. Two minor releases later, `validation_alias` removed; field becomes a no-op with a `RuntimeWarning`.
3. Two minor releases after that, field deleted entirely.

Total: ≥ 4 minor releases from announcement to removal. Documented in CONTRIBUTING.md.

---

## 11. Cross-Cutting: Debug Knobs vs. Public Args

The existing TRT-LLM convention is `TLLM_*` env vars (`TLLM_LOG_LEVEL_BY_MODULE`, `TLLM_LOG_LEVEL`, etc.). The user proposed `TLLM_<MODEL>_CONFIG_<XYZ>` as the namespace for VisualGen debug knobs. Concrete proposal: `TLLM_VISUALGEN_<KNOB>` for VisualGen-wide; `TLLM_VISUALGEN_<MODEL>_<KNOB>` for model-specific where needed.

### 11.1 The migration list

| Today's field | New home | Why |
| --- | --- | --- |
| `skip_warmup` | `VisualGenArgs._advanced.skip_warmup`, `status="prototype"` | Per-instance control; today copied into `PipelineLoader.load(skip_warmup=...)`. Two `VisualGen` instances in the same process must be able to disagree. |
| `skip_components` | `VisualGenArgs._advanced.skip_components`, `status="prototype"` | Same reasoning; also flows into per-engine `pipeline.load_standard_components(...)`. Used in fast smoke tests. |
| `pipeline.enable_layerwise_nvtx_marker` | `TLLM_VISUALGEN_NVTX_LAYERS=1` env var | Truly process-wide diagnostic; NVTX ranges are per-process resources. |

> **🤖 Codex (iter 1) — open:** Process-global env vars are the wrong compatibility replacement for per-instance args
>
> **Anchor:** §11.1 / The migration list
>
> The doc moves `skip_warmup`, `skip_components`, and layerwise NVTX to `TLLM_VISUALGEN_*` env vars. That is not equivalent to the current API. Today `skip_warmup` is copied from `VisualGenArgs` into `PipelineLoader.load(skip_warmup=args.skip_warmup)`, and `skip_components` is passed per engine into `pipeline.load_standard_components(...)`. An env var is process-global, inherited by worker processes, and cannot safely express two `VisualGen` instances in the same Python process with different loading behavior.
>
> `skip_components` also looks more than a pure debug toggle: it controls partial component loading across Wan, Flux, and LTX-2 test paths and is useful for fast smoke tests without loading heavy auxiliary modules. Moving it straight to env vars creates hidden state, order dependence between tests, and harder reproduction of user-supplied YAML.
>
> **Suggested direction:** Keep per-instance control somewhere non-stable but typed: an internal loader option, `_advanced` / `test` namespace with permanent `status="prototype"`, or explicit `PipelineLoader.load(skip_components=..., skip_warmup=...)` parameters. Reserve env vars for truly process-wide diagnostics like NVTX, and document worker inheritance/precedence if any env var remains.

> **💬 Claude — addressed:** kept `skip_warmup` and `skip_components` as **per-instance** typed fields under `VisualGenArgs._advanced` with permanent `status="prototype"`; only `enable_layerwise_nvtx_marker` moves to the env-var path because NVTX is a process-wide resource. Updated the §11.1 table above. The `_advanced` namespace is a single typed sub-config (`AdvancedConfig`) marked `status="prototype"` end-to-end so users know it's not the production contract; same effect as an env var (production users won't touch it) but preserves per-instance isolation, two-VisualGen-in-one-process scenarios, and YAML reproducibility.
>
> Env vars used (process-wide only):
> - `TLLM_VISUALGEN_NVTX_LAYERS=1` — turn on layerwise NVTX ranges.
> - `TLLM_VISUALGEN_SKIP_WARMUP_ALL=1` — emergency *override* if someone needs to skip warmup across every `VisualGen` instance in a process (e.g. CI smoke runs); precedence: env var > `args._advanced.skip_warmup` (per-instance) > default `False`. Worker processes inherit the env var; the args field is per-`VisualGenArgs` instance.

Implementation: `os.environ.get("TLLM_VISUALGEN_*")` reads for the NVTX flag happen inside the relevant pipeline construction sites; `args._advanced.skip_warmup` / `args._advanced.skip_components` are read by `PipelineLoader.load(...)` directly. The args class stays small (one nested `_advanced: AdvancedConfig` sub-config, all fields `status="prototype"`).

### 11.2 What stays as args (and why)

- **`compilation.resolutions` / `compilation.num_frames`** — production tuning knob; users explicitly want these in their YAML.
- **`offload.*`** — production deployment concern; not debug.
- **`attention.backend`** — production tuning.
- **`parallel.*`** — production deployment.

### 11.3 What about today's `Field(status="prototype")` fields?

`prototype` fields *can* live on the args class — that's what the status marker is for. The env-var carve-out is for fields that are *never* going to be production. The two are orthogonal:

- `prototype` field = "this exists publicly but might break."
- env var = "this is a debug toggle, no public surface."

A field can move *from* env var *to* `prototype` arg if it stabilizes into a real feature.

---

## 12. Cross-Cutting: YAML, CLI, dict Ingestion

The recommended shape (§6.4) preserves all current ingestion paths:

### 12.1 Direct Pydantic construction

```python
args = VisualGenArgs(
    model="...",
    parallel=ParallelConfig(dit_cfg_size=2),
    arch_config=WanModelConfig(fuse_qkv=True),
)
```

No change from today.

### 12.2 dict ingestion

```python
args = VisualGenArgs(**config_dict)
```

Pydantic's discriminator handles the union automatically — `config_dict["arch_config"]["arch"] = "wan"` resolves to `WanModelConfig` without a manual `from_dict` dispatch. This is strictly cleaner than the `update_llm_args_with_extra_dict`'s manual `field_mapping` (`tensorrt_llm/llmapi/llm_args.py:3309-3322`).

### 12.3 YAML

```yaml
model: "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
dtype: bfloat16
parallel:
  dit_cfg_size: 2
compilation:
  resolutions:
    - [480, 832]
  num_frames: [33, 81]
arch_config:
  arch: wan
  fuse_qkv: true
```

`yaml.safe_load` + `VisualGenArgs(**dict)` works directly. The discriminator field `arch` makes the YAML self-describing.

### 12.4 CLI

For a future `trtllm-serve` integration, the vLLM `FlexibleArgumentParser` pattern (`--config config.yaml` + dotted CLI overrides like `--arch_config.fuse_qkv=true`) is the most ergonomic. Not required for this milestone — the YAML route is sufficient.

---

## 13. Migration Plan

### 13.1 Sequencing

The refactor breaks into independent phases. Each is mergeable on its own, and each is testable in isolation.

> **🤖 Codex (iter 1) — open:** The alias plan only handles flat LTX-2 paths, not the nested fields moved in later phases
>
> **Anchor:** §13.1 / Sequencing
>
> The migration sequence claims each phase is independently mergeable and keeps old surfaces working, but the concrete compatibility sketch only covers flat fields like `text_encoder_path`. The riskier moves are nested: `parallel.refiner_*`, `parallel.t5_fsdp_size`, and `pipeline.fuse_qkv`. Those old YAML shapes are `parallel: { ... }` and `pipeline: { fuse_qkv: ... }`; a simple parent-field `validation_alias` does not explain how values are pulled out of nested legacy subconfigs and copied into the correct arch submodel.
>
> The ordering also moves those nested fields in Phases 6-7 before Phase 9 introduces auto-detection of `model_config=None`. Without the resolver, legacy `parallel.t5_fsdp_size` cannot reliably become `WanModelConfig.t5_parallel`, and legacy `parallel.refiner_*` cannot become `LTX2ModelConfig.refiner_parallel` unless the user also supplies the new discriminator. That is a compatibility regression hiding inside the alias window.
>
> **Suggested direction:** Move model/arch resolution before any nested field migration, and add explicit migration pseudocode plus tests for old YAML shapes: `parallel.refiner_dit_*`, `parallel.t5_fsdp_size`, `pipeline.fuse_qkv`, and mixed old+new conflicts. The plan should say exactly which value wins and which warning is emitted.

> **💬 Claude — addressed:** reordered the phases — old Phase 9 (arch resolver) is now Phase 4, folded into the same change that introduces `BaseModelConfig` + the discriminated union, so every later "nested field move" can lean on the resolver. The dead-field cleanup (`pipeline.fuse_qkv`, per Codex Thread 2) becomes a dedicated Phase 7. Added explicit migration pseudocode for nested legacy fields in §13.2 and pinned the precedence rule (explicit `arch_config` wins; otherwise resolver populates from HF + legacy nested values; mixed-explicit triggers a `DeprecationWarning` and the new value wins).

**Phase 1 — Move `VisualGenArgs` and sub-configs to `tensorrt_llm/visual_gen/args.py`.** No structural change yet. Just move out of `_torch/visual_gen/config.py` per M2 §3.1 / §9 Option C. Keep re-exports from the old location for one release. Adopt `Field(status=...)`. Add `tests/unittest/api_stability/visual_gen_args.yaml`.

**Phase 2 — Internal-state cleanup.** Move `dynamic_weight_quant`, `force_dynamic_quantization` to `PrivateAttr`. Move `enable_layerwise_nvtx_marker` to env var (truly process-wide diagnostics); keep `skip_warmup` / `skip_components` as `_advanced` namespace `status="prototype"` fields, **not** env vars (per Codex Thread 6 — env vars break per-engine isolation).

**Phase 3 — `OffloadConfig` carve-out.** Promote the 3 offloading fields out of `PipelineConfig`. Delete `PipelineConfig` once it's empty.

**Phase 4 — Introduce `BaseModelConfig` + discriminated union + auto-detect resolver.** Add `WanModelConfig`, `FluxModelConfig`, `LTX2ModelConfig` skeletons (initially empty besides the `arch` discriminator), wire `arch_config: ArchConfig | None = None` on `VisualGenArgs`, *and* implement the resolver in `PipelineLoader` so that `arch_config=None` is filled by reading HF `model_index.json::_class_name` + the registry. **Combining this in one phase is load-bearing**: every later phase that moves a nested field depends on the resolver to pick the right submodel for legacy YAML shapes (per Codex Thread 3).

**Phase 5 — Move LTX-2 fields.** The three LTX-2 paths move into `LTX2ModelConfig`. The flat fields on `VisualGenArgs` become `Field(deprecated="Use args.arch_config.<x>", validation_alias=...)`. Add coercion: if user sets `text_encoder_path` flat, populate `arch_config = LTX2ModelConfig(text_encoder_path=...)` in a `model_validator`.

**Phase 6 — Move `parallel.refiner_*` and `parallel.t5_fsdp_size` (`status="prototype"`).** Into `LTX2ModelConfig.refiner_parallel` and `WanModelConfig.t5_parallel`. These are intended-but-unused at `e527a9f785` (Codex Thread 2); migrating them now reserves the typed home and lets the resolver populate them from legacy YAML, but the fields stay `status="prototype"` until they're wired into the runtime.

**Phase 7 — Delete `pipeline.fuse_qkv`.** Verified zero runtime reads at `e527a9f785`; not migrating. Emit a `DeprecationWarning` for one release if a user passes it (alias maps to a no-op), then remove.

**Phase 8 — Discovery API surface.** Add `VisualGen.supported_models()`, `VisualGen.arch_config_schema(arch)` (static type schema), and `VisualGen.resolved_config(model=..., device=...)` (capability-resolved schema, per Codex Thread 5).

**Phase 9 — Remove deprecated aliases (≥ 2 minor releases after Phases 5–7 land).**

### 13.2 Backwards compatibility

Each phase keeps the old surface working via Pydantic `validation_alias`. Concrete:

```python
class VisualGenArgs(StrictBaseModel):
    text_encoder_path: str = Field(
        default="",
        deprecated="Set via arch_config=LTX2ModelConfig(text_encoder_path=...)",
        validation_alias=AliasChoices("text_encoder_path"),
    )

    @model_validator(mode="after")
    def _migrate_legacy_ltx2_paths(self) -> "VisualGenArgs":
        if self.text_encoder_path or self.spatial_upsampler_path or self.distilled_lora_path:
            if self.arch_config is None:
                self.arch_config = LTX2ModelConfig(
                    text_encoder_path=self.text_encoder_path,
                    spatial_upsampler_path=self.spatial_upsampler_path,
                    distilled_lora_path=self.distilled_lora_path,
                )
            else:
                # User set both — explicit wins; warn.
                logger.warning("Both flat LTX-2 paths and arch_config set; using arch_config.")
        return self
```

#### 13.2.1 Nested-field migration (per Codex Thread 3)

Flat-field aliases are not enough for `parallel.refiner_*`,
`parallel.t5_fsdp_size`, and (until removal) `pipeline.fuse_qkv`. The
legacy YAML shape is nested:

```yaml
parallel:
  dit_cfg_size: 2
  refiner_dit_dp_size: 2     # legacy LTX-2 two-stage
  t5_fsdp_size: 4            # legacy Wan T5 path
pipeline:
  fuse_qkv: true             # legacy; ignored after Phase 7
```

The migration validator pulls these out of the cross-cutting
`ParallelConfig`/`PipelineConfig` and rehouses them on the right
`arch_config` submodel. The resolver from Phase 4 picks the submodel
class (no manual `arch:` needed in legacy YAML).

```python
@model_validator(mode="after")
def _migrate_legacy_nested_fields(self) -> "VisualGenArgs":
    legacy_refiner = {
        f.replace("refiner_", ""): getattr(self.parallel, f)
        for f in (
            "refiner_dit_dp_size", "refiner_dit_tp_size", "refiner_dit_ulysses_size",
            "refiner_dit_ring_size", "refiner_dit_cp_size", "refiner_dit_cfg_size",
            "refiner_dit_fsdp_size",
        )
        if getattr(self.parallel, f, 1) != 1
    }
    legacy_t5_fsdp = getattr(self.parallel, "t5_fsdp_size", 1)

    if legacy_refiner or legacy_t5_fsdp != 1:
        # Resolver from Phase 4 has already populated self.arch_config when
        # arch_config=None; otherwise the user's explicit submodel wins.
        target = self.arch_config
        if target is None:
            raise ValueError(
                "Legacy parallel.refiner_*/t5_fsdp_size set but model arch could "
                "not be resolved. Set arch_config=<X>ModelConfig(...) explicitly."
            )

        if legacy_refiner and isinstance(target, LTX2ModelConfig):
            if target.refiner_parallel != ParallelConfig():  # both set
                logger.warning(
                    "Both parallel.refiner_* and arch_config.refiner_parallel set; "
                    "using arch_config.refiner_parallel."
                )
            else:
                target.refiner_parallel = ParallelConfig(**{
                    f"dit_{k}": v for k, v in legacy_refiner.items()
                })
                warnings.warn(
                    "parallel.refiner_* is deprecated; set "
                    "arch_config=LTX2ModelConfig(refiner_parallel=...).",
                    DeprecationWarning, stacklevel=2,
                )

        if legacy_t5_fsdp != 1 and isinstance(target, WanModelConfig):
            if target.t5_parallel != ParallelConfig():
                logger.warning(
                    "Both parallel.t5_fsdp_size and arch_config.t5_parallel set; "
                    "using arch_config.t5_parallel."
                )
            else:
                target.t5_parallel = ParallelConfig(fsdp_size=legacy_t5_fsdp)
                warnings.warn(
                    "parallel.t5_fsdp_size is deprecated; set "
                    "arch_config=WanModelConfig(t5_parallel=...).",
                    DeprecationWarning, stacklevel=2,
                )
    return self
```

**Precedence rule** (consistent across all migrations):

1. Explicit user-set `arch_config.<sub>` always wins.
2. Otherwise the resolver populates `arch_config` from HF metadata +
   legacy nested values.
3. Mixed (legacy nested *and* explicit submodel field both set) →
   `logger.warning(...)`, the explicit submodel field wins. No silent
   override.
4. Deprecated legacy field set without an arch the resolver can detect
   → `ValueError` (we cannot guess which submodel owns it).

The API-stability test (§10.2) gains explicit cases for each of
these four paths.

This ensures every pre-refactor YAML and Python construction continues
to work for two minor releases.

### 13.3 What we *don't* do for compat

- We do **not** keep `pipeline.enable_layerwise_nvtx_marker` as a field. It moves straight to env var. NVTX is a debug feature; production users do not depend on its config field.
- We do **not** keep `dynamic_weight_quant` / `force_dynamic_quantization` as fields. They were never user-set in any working configuration (the validator computes them); removing them is non-breaking in practice.
- We do **not** maintain the `to_dict()` / `from_dict()` methods (M2 §3.2 already calls these out for removal).

---

## 14. Recommendation Summary

The 10-line version of this doc:

1. Adopt **§6.4 (hybrid)** as the target shape: orthogonal cross-cutting sub-configs + discriminated `arch_config` union per architecture.
2. Move the **three LTX-2 paths** (§7.2), the **seven `parallel.refiner_*`** fields (§7.7), `parallel.t5_fsdp_size`, and `pipeline.fuse_qkv` (§7.6) into per-arch model submodels.
3. Move **debug knobs** — `skip_warmup`, `skip_components`, `pipeline.enable_layerwise_nvtx_marker` — to `TLLM_VISUALGEN_*` env vars (§11).
4. Make **`dynamic_weight_quant` / `force_dynamic_quantization`** internal `PrivateAttr` (§7.4).
5. Carve **`OffloadConfig`** out of `PipelineConfig`; delete the now-empty `PipelineConfig` class (§7.6).
6. Adopt the **`Field(status=...)`** marker and add a **`tests/unittest/api_stability/visual_gen_args.yaml`** snapshot (§10).
7. Use Pydantic **`validation_alias` + `deprecated`** for renames during the migration window (§10.3, §13.2).
8. Provide a thin **discovery API** (`supported_models()`, `arch_config_schema(arch)`) that wraps Pydantic's native `model_json_schema()` (§9). Don't lean on it as the primary surface.
9. Treat the **dict + discovery API** alternative (§6.5) as an inferior option vs typed submodels for the same effort. Use a tightly-scoped `extra_args: dict` *inside each submodel* as the forward-compat hatch.
10. Sequence the work as **10 phases** (§13.1), each independently mergeable, with backwards-compat aliases maintained for ≥ 2 minor releases.

---

## 15. Open Questions

1. ~~**`model_config` field name collision.**~~ **Resolved in iter 1** — locked in `arch_config` as the field name and `ArchConfig` as the union type alias. See §6.4 Codex thread + Claude reply.
2. **What's the discriminator value?** Three candidates:
   - `arch: Literal["wan", "flux", "ltx2"]` — short, ours.
   - `class_name: Literal["WanPipeline", ...]` — matches our internal `pipeline_registry.py` strings.
   - `model_type` — matches HF `config.json::model_type` and would auto-resolve from HF metadata.
   `model_type` is most user-friendly and aligns with vLLM-Omni's `_OMNI_PIPELINES` keying. **Tentative**: `model_type` for users + an internal map to `class_name` for the registry.
3. ~~**Does `arch_config=None` auto-resolve?**~~ **Decided** — yes, the resolver lands in Phase 4 of the migration (per Codex Thread 3 reordering). Read HF `model_index.json::_class_name` and dispatch to the registry; `VisualGen(model="Wan-AI/...")` Just Works without users typing `WanModelConfig`. SGLang #20078 risk pinned by the alias-cases YAML in §10.2 layer 4.
4. **Folding `torch_compile` and `cuda_graph` into `compilation`.** M2 §15 Q1. Outside this doc's scope, but if folded, the resulting `CompilationConfig` mirrors vLLM exactly.
5. **`AttentionConfig` evolution to a discriminated union.** §8.1 — when does this trigger? **Tentative**: as soon as a backend needs a backend-specific kwarg; not before.
6. **Promoting `extra_args` fields.** When a key in `arch_config.extra_args` becomes used by 3+ commits and stable for 1+ release, promote to a typed field on the submodel. Need a process or just judgment? **Tentative**: judgment, captured in CONTRIBUTING.md.
7. **CLI integration timeline.** vLLM-style `--config + dotted overrides` would be valuable for `trtllm-serve` but is not blocking this refactor. **Tentative**: defer to a follow-up PR.
8. **Out-of-tree model registration.** Should `register_arch_config(name, cls)` be a public function (PEFT-style)? This enables third-party plugins but commits us to a registry contract. **Tentative**: not in this milestone; revisit if a plugin ecosystem emerges.
9. **Relationship with `DiffusionModelConfig` (the internal merged config).** `DiffusionModelConfig.from_pretrained` reads HF + args and produces the merged config. With per-arch submodels, the merging logic gets cleaner — each arch's submodel knows how to merge its own HF metadata. **Tentative**: keep `DiffusionModelConfig` internal as M2 §10.3 proposes, but its `from_pretrained` becomes a dispatch over the registered submodels.
10. **Per-arch `fuse_qkv` defaults.** Moot — `pipeline.fuse_qkv` is being deleted (Codex Thread 2 verified zero runtime reads). If/when QKV-fusion-as-config returns, per-arch defaults are the right shape; not a blocker now.
11. **Capability table coverage for Tier 2 discovery (§9.2).** The capability table starts populated for Wan / Flux / LTX-2 only. Coverage for future arches is a per-PR addition, not a separate milestone. Should we lint that every new `<X>ModelConfig` ships with a capability-table row? **Tentative**: yes, enforce via the API-stability test (layer 2 fails if a registered arch has no capability entry).

---

## 16. Appendix: Source Links

### TRT-LLM (current-state)

- [`tensorrt_llm/_torch/visual_gen/config.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/e527a9f785/tensorrt_llm/_torch/visual_gen/config.py) — `VisualGenArgs`, sub-configs, `DiffusionModelConfig`.
- [`tensorrt_llm/llmapi/llm_args.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/llmapi/llm_args.py) — `LlmArgs`, `BaseSparseAttentionConfig.from_dict` discriminator, `update_llm_args_with_extra_dict`.
- [`tests/unittest/api_stability/`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/tests/unittest/api_stability) — API stability YAML reference test pattern.
- [`tensorrt_llm/llmapi/utils.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/llmapi/utils.py) — `set_api_status` decorator.

### vLLM

- `EngineArgs` flat → `VllmConfig` composed: [`vllm/engine/arg_utils.py:402-2175`](https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py#L402-L2175).
- `VllmConfig` composition: [`vllm/config/vllm.py:268-360`](https://github.com/vllm-project/vllm/blob/main/vllm/config/vllm.py#L268-L360).
- Quantization registry (`get_quantization_config`, `register_quantization_config`): [`vllm/model_executor/layers/quantization/__init__.py:47-168`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/__init__.py#L47-L168).
- Compilation config: [`vllm/config/compilation.py`](https://github.com/vllm-project/vllm/blob/main/vllm/config/compilation.py).
- Deprecation policy: [docs](https://docs.vllm.ai/en/v0.13.0/contributing/deprecation_policy/).
- RFC #24384 (decouple from HF): [issue](https://github.com/vllm-project/vllm/issues/24384).
- Issue #18707 (docs unusable): [issue](https://github.com/vllm-project/vllm/issues/18707).

### vLLM-Omni

- `OmniEngineArgs` (subclass of `EngineArgs`): [`vllm_omni/engine/arg_utils.py:86-148`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/engine/arg_utils.py#L86-L148).
- `OmniDiffusionConfig`: [`vllm_omni/diffusion/data.py:354-526`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/data.py#L354-L526).
- Diffusion model registry: [`vllm_omni/diffusion/registry.py:21-258`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/registry.py#L21-L258).
- Multi-stage pipeline registry: [`vllm_omni/config/pipeline_registry.py`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/config/pipeline_registry.py).
- RFC #3366 (precedence chain): [issue](https://github.com/vllm-project/vllm-omni/issues/3366).
- Issue #2887 / #3313 / #3357 — config refactor follow-ups.

### SGLang

- `ServerArgs` flat (LLM): [`python/sglang/srt/server_args.py:304`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py#L304).
- `ServerArgs` (Diffusion): [`python/sglang/multimodal_gen/runtime/server_args.py:121-168`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/server_args.py#L121-L168).
- `PipelineConfig` per-arch base: [`configs/pipeline_configs/base.py:193-235`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/configs/pipeline_configs/base.py#L193-L235).
- Wan per-arch config: [`configs/pipeline_configs/wan.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/configs/pipeline_configs/wan.py).
- `_default_height` / `_default_width` ClassVar pattern: [`configs/sample/sampling_params.py:88-150, 268-274`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/configs/sample/sampling_params.py#L88-L150).
- Three-registry: [`registry.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/registry.py).
- Bug #20078 / Fix PR #20080.

### Diffusers

- `DiffusionPipeline.from_pretrained`: [`pipeline_utils.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/pipeline_utils.py).
- `ConfigMixin` + `@register_to_config`: [`configuration_utils.py:113-805`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/configuration_utils.py).
- Quantization registry: [`quantizers/auto.py:50-67`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/quantizers/auto.py).
- Per-pipeline shape: [`pipelines/flux/pipeline_flux.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py), [`pipelines/wan/pipeline_wan.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan.py).
- Deprecation: [`utils/deprecation_utils.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/utils/deprecation_utils.py).

### Python / DL ecosystem

- Pydantic discriminated unions: [docs](https://pydantic.dev/docs/validation/latest/concepts/unions/).
- HF Transformers `_LazyAutoMapping`: [`auto_factory.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/auto/auto_factory.py).
- PEFT registry: [`config.py`](https://github.com/huggingface/peft/blob/main/src/peft/config.py), [`mapping.py`](https://github.com/huggingface/peft/blob/main/src/peft/mapping.py).
- Hydra structured configs: [docs](https://hydra.cc/docs/tutorials/structured_config/intro/).
- PEP 702 `@warnings.deprecated`: [PEP](https://peps.python.org/pep-0702/).
- ExecuTorch deprecation policy: [docs](https://docs.pytorch.org/executorch/stable/api-life-cycle.html).

---

## Iteration Tracker

| #  | Date       | Codex focus                                                                | Threads | Resolved | Open | Deferred |
|----|------------|----------------------------------------------------------------------------|---------|----------|------|----------|
| 1  | 2026-05-08 | recommendation, field dispositions, migration ordering, stability, discovery, env-var/per-instance | 6       | 0        | 6    | 0        |

*Iteration 1 in progress — Codex emitted 6 substantive threads anchored across §6.4, §7.6/§7.7, §9.2, §10.2, §11.1, §13.1; Claude triaged all six as `addressed` with design changes. All threads awaiting Codex iter-2 to mark resolved or follow up.*
