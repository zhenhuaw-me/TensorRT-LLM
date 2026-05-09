# `VisualGenArgs` Refactor — Engine-Config Stability & Extensibility

> **Status**: Draft — converged after 9 Codex adversarial-review iterations (2026-05-08)
> **Date**: 2026-05-07 (initial draft); converged 2026-05-08
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
- **Public registry contract for out-of-tree plugins** — left as an
  Open Question; not in this milestone. (The internal arch
  discriminator is **locked**: `arch: Literal["wan", "flux", "ltx2", ...]`
  — see §1 below. Only the public `register_arch_config(...)` plugin
  surface is deferred.)
- **Discovery-API CLI/UX polish** (e.g. pretty-printers, IDE
  integrations) beyond the typed schema and the `resolved_config()`
  return value — not required for this design to land. The minimal
  Tier-1 + Tier-2 API surface (§9.2) and the registry-mandatory
  capability table (§10.2 layer 3) **are** in scope; only the
  surface polish is deferred (per Codex iter-4 Thread 5).

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
- **Debug-only knobs**: `skip_warmup` and `skip_components` stay as **per-instance fields under an `advanced` namespace** with permanent `status="prototype"` (production users won't touch them; per-engine isolation is preserved — env vars cannot express two `VisualGen` instances in the same process). Only `enable_layerwise_nvtx_marker` moves to a `TLLM_VISUALGEN_NVTX_LAYERS=1` env var because NVTX is genuinely process-wide. The internal quant flags (`dynamic_weight_quant`, `force_dynamic_quantization`) become `PrivateAttr`.
- **Stability is enforced**, not policy — copy the LLM-API pattern: a `Field(status=...)` marker (already exists in `tensorrt_llm/llmapi/llm_args.py`), and a **five-file API stability harness** with two policies: `visual_gen_args.yaml` (parent fields), `visual_gen_arch_configs.yaml` (every registered submodel), `visual_gen_arch_registry.yaml` (discriminator literals + resolver patterns; stable-API policy), `visual_gen_capabilities.yaml` (per-arch capability rows; operational policy with PR-review + mandatory-row + named-test enforcement), and `visual_gen_alias_cases.yaml` (old→new YAML migration cases). `pydantic.deprecated` aliases for renames (≥ 2 minor releases).

The biggest single payoff: when someone proposes a fourth diffusion model, `VisualGenArgs` doesn't change. They register `class FooModelConfig` with its own `text_encoder_path` (or whatever it needs), the union picks it up, the API surface for users of *other* models is unaffected, and `arch_config_schema("foo")` returns the JSON schema without any extra work.

### What this doc does *not* commit to

- It does **not** require migrating internal `DiffusionModelConfig` (the merged/parsed config built by `PipelineLoader`) — that stays internal and tracks the deferred `Diffusion*` → `VisualGen*` rename in M2 §10.3.
- It does **not** require **CLI/UX polish** on top of the discovery API (pretty-printers, IDE integrations, `--describe` subcommands). The minimal Tier-1 typed schema (`arch_config_schema(arch)`) and Tier-2 capability schema (`VisualGen.resolved_config(model, device, gpus)`) **are** in scope, with the registry-mandatory capability snapshot in §10.2 layer 3 (per Codex iter-1 Thread 5 + iter-4 Thread 5 + iter-5 Thread 2). Only the surface polish on top is deferred.

**Discriminator chosen**: `arch: Literal["wan", "flux", "ltx2", ...]`. Picked over `model_type` and `class_name` for brevity and decoupling from HF metadata field names. Locked in iter 5 (Codex Thread 3); §15 Q2 retired. All sketches, examples, alias cases, and the layer-3 stability snapshot use `arch`.

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

The principles below extend the M2 doc's principles to the args-specific concerns. Each is a *normative* statement of the post-refactor design — Codex iter-3 Thread 4 caught earlier drafts that contradicted the iter-1+iter-2 decisions; this section is the single source of truth.

1. **Cross-cutting concerns get orthogonal sub-configs.** If a knob applies to every model and every backend (compilation, parallelism, KV-cache-style memory, observability), it lives in its own typed sub-config. Sub-configs do not know about each other.
2. **Architecture-specific config is per-architecture.** A field that is meaningful for one model is *not* a top-level field. It lives on a typed `XModelConfig` registered into a discriminated `arch_config: ArchConfig` union (Pydantic-reserved-namespace-safe; field name locked in iter 1).
3. **The args class is closed for modification, open for extension.** Adding a new model must not require editing `VisualGenArgs` or any cross-cutting sub-config. Every new arch ships with its own `XModelConfig` *and* a capability row in the registry (mandatory, enforced by §10.2 layer 3).
4. **Internal state stays internal.** If a field is computed from another, it doesn't appear in the public schema. Use Pydantic's `PrivateAttr` (already used by `LlmArgs`) or move to `DiffusionModelConfig` (the internal merged config).
5. **Per-instance debug knobs stay typed and per-instance.** `skip_warmup` and `skip_components` live in a `advanced: AdvancedConfig` sub-config with permanent `status="prototype"`. Env vars are reserved for *truly* process-wide diagnostics (NVTX). Per-engine isolation is non-negotiable — two `VisualGen` instances in the same process must be able to disagree (per Codex iter-1 Thread 6).
6. **One escape hatch per submodel, never on the parent.** When extensibility is genuinely needed, a tightly-scoped `extra_args: dict[str, Any]` *inside* a model submodel keeps the cost local. No `additional_config: dict` on `VisualGenArgs` itself — that recreates vLLM's #18707.
7. **Stability is mechanically enforced via a five-file harness with split policy.** Every field carries `status="prototype"/"beta"`; promotion is a YAML-tracked event across five snapshots — `visual_gen_args.yaml` (parent fields), `visual_gen_arch_configs.yaml` (per-submodel fields), `visual_gen_arch_registry.yaml` (discriminator literals + resolver patterns; stable-API policy), `visual_gen_capabilities.yaml` (per-arch capability rows; operational policy with mandatory-row + named-test enforcement), `visual_gen_alias_cases.yaml` (old→new migrations). Renames go through `Field(alias=...)` for ≥ 2 minor releases.
8. **Two-tier discovery.** Tier 1 — Pydantic-native type schema (`arch_config_schema(arch)`) — is free and answers "what fields exist on `WanModelConfig`". Tier 2 — `VisualGen.resolved_config(model, device, gpus)` — uses the registry-mandatory capability table to answer "what's actually supported on this model + this hardware". Tier 1 alone does not solve the per-model/per-GPU discoverability problem in the scope (per Codex iter-1 Thread 5); Tier 2 is part of this milestone, snapshotted by §10.2 layer 3 so it cannot rot.
9. **Migration runs after arch resolution.** Flat fields whose arch is implicit in their name (`text_encoder_path` → LTX-2) migrate in `VisualGenArgs.model_validator`. Nested fields whose arch comes from HF metadata (`parallel.refiner_*`, `parallel.t5_fsdp_size`) migrate in `PipelineLoader.load(args)` *after* the resolver populates `arch_config`. No silent drops — every legacy field has an explicit target or an explicit `ValueError` with a pointer to the live replacement (per Codex iter-2 Thread 4 + iter-3 Thread 3).

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

# Forward declarations: T5Config must be declared BEFORE WanModelConfig
# references it (Python evaluates default expressions at class-body time;
# Codex iter-6 Thread 3). Use Field(default_factory=...) where the default
# is a mutable Pydantic model so each VisualGenArgs gets its own instance.

class T5Config(StrictBaseModel):
    """Per-instance config for the T5 text encoder used by Wan."""
    fsdp_size: int = Field(default=1, ge=1, status="prototype")

class AdvancedConfig(StrictBaseModel):
    """Per-instance debug knobs. Permanent `status="prototype"` — production
    users do not reach into this namespace, but tests and CI smoke runs do
    and need per-engine isolation (env vars cannot express it; iter-1
    Thread 6, iter-4 Thread 1)."""
    skip_warmup: bool = Field(default=False, status="prototype")
    skip_components: list["PipelineComponent"] = Field(default_factory=list, status="prototype")

class PipelineConfig(StrictBaseModel):
    """Deprecated. Removed in Phase 7. Only retained to absorb legacy
    `pipeline.*` keys from existing YAML configs (`fuse_qkv` + the three
    offload fields) without breaking the soft-removal window. Full
    declaration with deprecated= fields and the `model_validator(mode="after")`
    that surfaces the warning lives in §13.1 Phase 3; declared here only
    so `VisualGenArgs` below can reference the type at class-body time
    (Codex iter-7 follow-up — Python `NameError` otherwise)."""
    fuse_qkv: bool = Field(default=True, status="deprecated")
    enable_offloading: bool = Field(default=False, status="deprecated")
    offload_device: Literal["cpu", "cuda"] = Field(default="cpu", status="deprecated")
    offload_param_pin_memory: bool = Field(default=True, status="deprecated")

class OffloadConfig(StrictBaseModel):
    """New cross-cutting offloading sub-config (§7.6 / §13.1 Phase 3).
    Field names are the migration targets for `pipeline.enable_offloading`,
    `pipeline.offload_device`, and `pipeline.offload_param_pin_memory`."""
    enabled:    bool                  = Field(default=False)
    device:     Literal["cpu", "cuda"] = Field(default="cpu")
    pin_memory: bool                  = Field(default=True)

class ObservabilityConfig(StrictBaseModel):
    """Cross-cutting observability sub-config — logs, traces, metrics.
    Minimal placeholder for this milestone; expands in a follow-up."""
    otlp_traces_endpoint: str | None = Field(default=None, status="prototype")

class WanModelConfig(BaseModelConfig):
    arch: Literal["wan"] = "wan"
    text_encoder: T5Config = Field(default_factory=T5Config, status="prototype")
    # ... other Wan-only knobs (none yet — `fuse_qkv` is being deleted, not migrated)

class FluxModelConfig(BaseModelConfig):
    arch: Literal["flux"] = "flux"
    # ... Flux-only knobs

class LTX2ModelConfig(BaseModelConfig):
    arch: Literal["ltx2"] = "ltx2"
    text_encoder_path: str = ""
    spatial_upsampler_path: str = ""
    distilled_lora_path: str = ""
    refiner_parallel: ParallelConfig = Field(default_factory=ParallelConfig, status="prototype")
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
    parallel:      ParallelConfig    = Field(default_factory=ParallelConfig)
    compilation:   CompilationConfig = Field(default_factory=CompilationConfig)
    torch_compile: TorchCompileConfig = Field(default_factory=TorchCompileConfig)
    cuda_graph:    CudaGraphConfig    = Field(default_factory=CudaGraphConfig)
    attention:     AttentionConfig    = Field(default_factory=AttentionConfig)
    cache:         CacheConfig | None = None
    quant:         QuantConfig        = Field(default_factory=QuantConfig)
    offload:       OffloadConfig | None = None                  # carved from PipelineConfig
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    # ── Per-instance debug knobs (status="prototype" forever) ─
    advanced:      AdvancedConfig     = Field(default_factory=AdvancedConfig, status="prototype")

    # ── Per-architecture (extensible via registry) ────────────
    arch_config: ArchConfig | None = Field(
        default=None, status="beta",
        description="Per-architecture knobs. Defaults inferred from "
                    "HF model_index.json::_class_name when None.",
    )

    # ── Legacy compatibility shell (deleted in Phase 7) ───────
    pipeline:      PipelineConfig | None = Field(
        default=None, status="deprecated",
        description="Deprecated. Absorbs legacy `pipeline.*` keys "
                    "(fuse_qkv + 3 offload fields) during the soft-removal "
                    "window. Removed in Phase 7.",
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
| `skip_components` | `List[PipelineComponent]` | **Move to `advanced` namespace** (`status="prototype"` forever) | `VisualGenArgs.advanced.skip_components`; read by `PipelineLoader.load(...)` per-instance |
| `skip_warmup` | bool | **Move to `advanced` namespace** (`status="prototype"` forever) | `VisualGenArgs.advanced.skip_warmup`; read by `PipelineLoader.load(...)` per-instance |

Both are testing/debug knobs. **Per-instance control is required** — today they're copied from `VisualGenArgs` into `PipelineLoader.load(skip_warmup=...)` and `pipeline.load_standard_components(skip_components=...)` per engine; two `VisualGen` instances in the same process must be able to disagree. An env var is process-global and inherited by worker processes, so it cannot express that disagreement (per Codex iter-1 Thread 6 in §11.1).

The `advanced: AdvancedConfig` sub-config is permanently `status="prototype"`. Production users won't reach for it (because it's marked unstable in the API reference and CI), so the production contract stays tight; CI test paths that need fast smoke runs keep working with full per-engine isolation and YAML reproducibility.

A single emergency override env var (`TLLM_VISUALGEN_SKIP_WARMUP_ALL=1`) exists for cases that genuinely need to skip warmup process-wide (e.g. CI infrastructure smoke runs); precedence is env-var > per-instance `advanced.skip_warmup` > default `False`. See §11.1.

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
| `parallel.t5_fsdp_size` | flat (only Wan T5 path) | **Move to `arch_config`** (per-arch, `status="prototype"`) | `WanModelConfig.text_encoder: T5Config` with `T5Config.fsdp_size: int = Field(default=1, status="prototype")`. **Not** a generic `ParallelConfig` — `ParallelConfig.dit_fsdp_size` is DiT-shaped and the T5 isn't a DiT (per Codex iter-3 Thread 3). |

**Key insight from this carve-out**: when a model has *multiple* parallel-able passes, each one wants its own typed sub-config — but **not necessarily a `ParallelConfig`**. The LTX-2 stage-2 refiner is DiT-shaped, so `LTX2ModelConfig.refiner_parallel: ParallelConfig` is a fit. The Wan T5 text encoder is *not* DiT-shaped, so reusing `ParallelConfig.dit_fsdp_size` would mis-name the field; we declare `T5Config(StrictBaseModel)` with `fsdp_size: int` and use `WanModelConfig.text_encoder: T5Config` (per Codex iter-3 Thread 3). Putting them all on the parent `ParallelConfig` as `refiner_*` and `t5_fsdp_size` is the leakage; the right shape is **one typed sub-config per parallel-able stage**, owned by whatever owns the stage:

```python
class LTX2ModelConfig(BaseModelConfig):
    arch: Literal["ltx2"] = "ltx2"
    text_encoder_path: str = ""
    spatial_upsampler_path: str = ""
    distilled_lora_path: str = ""
    refiner_parallel: ParallelConfig = ParallelConfig()  # DiT-shaped → ParallelConfig fits

class T5Config(StrictBaseModel):
    """Per-instance config for the T5 text encoder. Not DiT-shaped, so
    not a ParallelConfig."""
    fsdp_size: int = Field(default=1, ge=1, status="prototype")

class WanModelConfig(BaseModelConfig):
    arch: Literal["wan"] = "wan"
    text_encoder: T5Config = T5Config()                  # T5 → its own typed config
```

This pattern reuses an existing typed sub-config when the stage's
shape matches (e.g. DiT-shaped passes get `ParallelConfig`) and
introduces a small new typed sub-config when the stage's shape
diverges (e.g. T5 isn't a DiT, so it gets `T5Config`). It extends
naturally: when a future model adds another parallel-able pass, the
right call is "reuse if the shape fits, declare a new sub-config if
it doesn't" — never reach for `ParallelConfig.<some_random_field>` to
absorb a foreign-shape knob.

### 7.8 Summary of dispositions

| Bucket | Count | Examples |
| --- | --- | --- |
| **Keep on `VisualGenArgs`** | 4 flat + 7 sub-configs | `model`, `revision`, `dtype`, `device`; `parallel`, `compilation`, `torch_compile`, `cuda_graph`, `attention`, `cache`, `quant` |
| **Move to `arch_config` (per-arch, live)** | 3 fields | LTX-2 paths (3) — `text_encoder_path`, `spatial_upsampler_path`, `distilled_lora_path` |
| **Move to `arch_config` (per-arch, `status="prototype"` until wired)** | 8 fields | `parallel.refiner_*` (7), `parallel.t5_fsdp_size` (1) — verified zero runtime reads at `e527a9f785` (intended-but-unused; promote to `"beta"` when wired) |
| **Delete** (verified zero runtime reads) | 1 field | `pipeline.fuse_qkv` — QKV fusion is hard-coded in `tensorrt_llm/_torch/visual_gen/modules/attention.py` via `QKVMode` enum, not driven by config |
| **Move to `advanced` namespace** (`status="prototype"`, per-instance) | 2 fields | `skip_warmup`, `skip_components` |
| **Move to env var** (truly process-wide) | 1 field | `pipeline.enable_layerwise_nvtx_marker` → `TLLM_VISUALGEN_NVTX_LAYERS=1` |
| **Make internal (PrivateAttr / DiffusionModelConfig)** | 2 fields | `dynamic_weight_quant`, `force_dynamic_quantization` |
| **New cross-cutting sub-config** | 1 (3 fields) | `OffloadConfig` (carves `enable_offloading`, `offload_device`, `offload_param_pin_memory` from `PipelineConfig`) |
| **New `advanced: AdvancedConfig` sub-config** | 1 (2 fields) | Holds `skip_warmup`, `skip_components`. Permanent `status="prototype"`. |
| **Eliminated** | 1 sub-config | `PipelineConfig` deleted **only after Phase 7 finishes the `fuse_qkv` soft-removal window** (per Codex iter-2 Thread 2); the empty shell stays around until then so legacy `pipeline:` YAML still loads with a `DeprecationWarning`. |

The net effect:

- `VisualGenArgs` shrinks from "4 flat + 4 LTX-2 + 2 test + 2 internal + 7 sub-configs" (19 surface concepts) to **"4 flat + 8 cross-cutting sub-configs + 1 `advanced` sub-config + 1 `arch_config` union"** (14 surface concepts), with the per-architecture surface gated by the discriminator.
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

### 10.2 API stability test — five-file harness with split policy

The single-YAML snapshot pattern that protects `LlmArgs` is **not enough** for the proposed VisualGen shape: §6.4 moves the real extensibility surface into per-arch submodels, the registry, and the legacy-alias migration. A top-level `VisualGenArgs.model_fields` snapshot can pass while a user's YAML breaks (per Codex iter-1 Thread 4 + iter-4 Thread 2).

The harness is therefore **five files** (split from four after Codex iter-5 Thread 4 — capability cells need a different change-policy than discriminator literals), all under `tests/unittest/api_stability/references/`, all driven by one runner entrypoint (an extension of the existing `api_stability_core.py`):

1. **`visual_gen_args.yaml`** — snapshot of `VisualGenArgs.model_fields` (cross-cutting fields, plus the `arch_config` field, plus the `advanced` namespace). Same pattern as `llm.yaml`.
2. **`visual_gen_arch_configs.yaml`** — one section per registered arch submodel (`WanModelConfig`, `FluxModelConfig`, `LTX2ModelConfig`, …). Snapshots fields, types, defaults, and `status`. **Layer-2 regression assertion**: no submodel may expose `fuse_qkv` (per Codex iter-3 Thread 1). Adding/removing/renaming a submodel field, or adding/removing a registered arch, forces a YAML edit.
3a. **`visual_gen_arch_registry.yaml`** — frozen list of `arch:` discriminator literals (`"wan"`, `"flux"`, `"ltx2"`, …) and the HF `_class_name` patterns the resolver matches on. **Stable-API policy**: any drift in discriminator literals or resolver patterns is a public-API break; needs a deprecation cycle and reviewer sign-off.
3b. **`visual_gen_capabilities.yaml`** — per-arch capability rows (supported attention backends, supported quant algos, supported cache backends, parallel constraints) — split from layer 3a per Codex iter-5 Thread 4. **Operational policy**: capability cells are *expected* to drift as kernels, hardware, and backend integrations mature; this YAML tracks the live state but the change policy is "PR-reviewed, no deprecation cycle". The mandatory-row check from §9.2 still lives here (every registered arch must have a row), and named compatibility test cases (e.g. `wan_h100_fa4_nvfp4_supported = True`) anchor known-good combinations so silent regressions get caught.
4. **`visual_gen_alias_cases.yaml`** — table of representative old→new YAML inputs paired with their expected resolved `arch_config` shape. Each case has a `mode`:
   - `direct` — load via `VisualGenArgs(**input)` (covers flat aliases that migrate in `model_validator`, e.g. `text_encoder_path`).
   - `loader` — load via `PipelineLoader.load(args)` with a fixture `model_index.json` (covers nested aliases that migrate post-arch-resolution, e.g. `parallel.refiner_*`, `parallel.t5_fsdp_size`, `pipeline.fuse_qkv`).

   The case set must include — beyond the per-field happy path — `both_legacy_families`, `wrong_arch_legacy`, `explicit_new_default`, `unresolved_arch`, `legacy_fuse_qkv_warns`, and `legacy_t5_fsdp_size`.

**Test runner** (one entrypoint): walks all five YAMLs in order. Layers 1, 2, 3a are snapshotted under the stable-API policy (any drift fails CI; deprecation cycle required to resolve). Layer 3b is snapshotted under the operational policy (drift fails CI but is resolvable via a same-PR YAML edit + reviewer sign-off; mandatory-row + named compatibility test cases must always pass). Layer 4 cases are replayed through the right load path per their `mode`. Any drift in any layer fails CI with a pointer at the offending YAML.

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
| `skip_warmup` | `VisualGenArgs.advanced.skip_warmup`, `status="prototype"` | Per-instance control; today copied into `PipelineLoader.load(skip_warmup=...)`. Two `VisualGen` instances in the same process must be able to disagree. |
| `skip_components` | `VisualGenArgs.advanced.skip_components`, `status="prototype"` | Same reasoning; also flows into per-engine `pipeline.load_standard_components(...)`. Used in fast smoke tests. |
| `pipeline.enable_layerwise_nvtx_marker` | `TLLM_VISUALGEN_NVTX_LAYERS=1` env var | Truly process-wide diagnostic; NVTX ranges are per-process resources. |

Implementation: `os.environ.get("TLLM_VISUALGEN_*")` reads for the NVTX flag happen inside the relevant pipeline construction sites; `args.advanced.skip_warmup` / `args.advanced.skip_components` are read by `PipelineLoader.load(...)` directly. The args class stays small (one nested `advanced: AdvancedConfig` sub-config, all fields `status="prototype"`).

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
    arch_config=WanModelConfig(text_encoder=T5Config(fsdp_size=2)),
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
  text_encoder:
    fsdp_size: 2
```

`yaml.safe_load` + `VisualGenArgs(**dict)` works directly. The discriminator field `arch` makes the YAML self-describing.

### 12.4 CLI

For a future `trtllm-serve` integration, the vLLM `FlexibleArgumentParser` pattern (`--config config.yaml` + dotted CLI overrides like `--arch_config.text_encoder.fsdp_size=2`) is the most ergonomic. Not required for this milestone — the YAML route is sufficient.

---

## 13. Migration Plan

### 13.1 Sequencing

The refactor breaks into independent phases. Each is mergeable on its own, and each is testable in isolation.

**Phase 1 — Move `VisualGenArgs` and sub-configs to `tensorrt_llm/visual_gen/args.py`.** No structural change yet. Just move out of `_torch/visual_gen/config.py` per M2 §3.1 / §9 Option C. Keep re-exports from the old location for one release. Adopt `Field(status=...)`. Add `tests/unittest/api_stability/visual_gen_args.yaml`.

**Phase 2 — Internal-state cleanup.** Move `dynamic_weight_quant`, `force_dynamic_quantization` to `PrivateAttr`. Move `enable_layerwise_nvtx_marker` to env var (truly process-wide diagnostics); keep `skip_warmup` / `skip_components` as `advanced` namespace `status="prototype"` fields, **not** env vars (per Codex Thread 6 — env vars break per-engine isolation).

**Phase 3 — `OffloadConfig` carve-out.** Promote the 3 offloading fields out of `PipelineConfig` into a new `OffloadConfig`. **Keep `PipelineConfig` as a deprecated four-field shell** until Phase 7 closes the `fuse_qkv` soft-removal window — the shell must absorb every legacy `pipeline.*` key (not just `fuse_qkv`), or `extra="forbid"` would reject existing YAML carrying `pipeline: { enable_offloading: true }` before migration can run (Codex iter-5 Thread 1):

```python
class PipelineConfig(StrictBaseModel):
    """Deprecated. Removed in Phase 7. Only retained to absorb legacy
    `pipeline.*` keys (fuse_qkv + the three offload fields) from existing
    YAML configs without breaking the soft-removal window."""

    fuse_qkv: bool = Field(
        default=True,
        status="deprecated",
        deprecated="`pipeline.fuse_qkv` has no runtime effect since "
                   "e527a9f785. Field removed in Phase 7.",
    )
    enable_offloading: bool = Field(
        default=False,
        status="deprecated",
        deprecated="Use `offload.enabled` (offload: { enabled: true }).",
    )
    offload_device: Literal["cpu", "cuda"] = Field(
        default="cpu",
        status="deprecated",
        deprecated="Use `offload.device`.",
    )
    offload_param_pin_memory: bool = Field(
        default=True,
        status="deprecated",
        deprecated="Use `offload.pin_memory`.",
    )

    @model_validator(mode="after")
    def _warn_if_set(self) -> "PipelineConfig":
        # Pydantic's Field(deprecated=...) only sets schema metadata and
        # warns on attribute *access*, not at validation time (Codex
        # iter-4 Thread 4). The validator surfaces the deprecation
        # explicitly so YAML payloads that set the field but never read
        # it actually trigger the warning.
        for legacy in self.model_fields_set:
            warnings.warn(
                f"`pipeline.{legacy}` is deprecated. "
                f"{type(self).model_fields[legacy].deprecated}",
                DeprecationWarning, stacklevel=2,
            )
        return self
```

`VisualGenArgs.model_validator(mode="after")` then migrates the three
offload fields into `args.offload`, with the same precedence rules as
the nested-field migration (§13.2.1):

```python
@model_validator(mode="after")
def _migrate_legacy_pipeline_offload(self) -> "VisualGenArgs":
    if self.pipeline is None:
        return self
    legacy_offload = {
        "enabled":    "enable_offloading",
        "device":     "offload_device",
        "pin_memory": "offload_param_pin_memory",
    }
    for new_field, legacy_field in legacy_offload.items():
        if legacy_field not in self.pipeline.model_fields_set:
            continue
        if self.offload is None:
            self.offload = OffloadConfig()
        if new_field in self.offload.model_fields_set:
            raise ValueError(
                f"Both legacy `pipeline.{legacy_field}` and explicit "
                f"`offload.{new_field}` are set. Pick one."
            )
        setattr(self.offload, new_field, getattr(self.pipeline, legacy_field))
    return self
```

This makes the offload soft-removal window actually work: legacy YAML parses, the validator fires deprecation warnings + migrates values, explicit-vs-legacy conflicts raise `ValueError`. Phase 7 deletes the four-field shell, both validators, and the class. Layer-4 alias cases gain `legacy_pipeline_offload`, `mixed_pipeline_offload_and_offload_explicit`, and `legacy_fuse_qkv_with_offload`.

**Phase 4 — Introduce `BaseModelConfig` + discriminated union + auto-detect resolver.** Add `WanModelConfig`, `FluxModelConfig`, `LTX2ModelConfig` skeletons (initially empty besides the `arch` discriminator), wire `arch_config: ArchConfig | None = None` on `VisualGenArgs`, *and* implement the resolver in `PipelineLoader` so that `arch_config=None` is filled by reading HF `model_index.json::_class_name` + the registry. **Combining this in one phase is load-bearing**: every later phase that moves a nested field depends on the resolver to pick the right submodel for legacy YAML shapes (per Codex Thread 3).

**Phase 5 — Move LTX-2 fields.** The three LTX-2 paths move into `LTX2ModelConfig`. The flat fields on `VisualGenArgs` become `Field(deprecated="Use args.arch_config.<x>", validation_alias=...)`. Add coercion: if user sets `text_encoder_path` flat, populate `arch_config = LTX2ModelConfig(text_encoder_path=...)` in a `model_validator`.

**Phase 6 — Move `parallel.refiner_*` and `parallel.t5_fsdp_size` (`status="prototype"`).** Into `LTX2ModelConfig.refiner_parallel: ParallelConfig` and `WanModelConfig.text_encoder: T5Config` (T5 isn't DiT-shaped, so it gets its own typed sub-config; iter-3 Thread 3). These are intended-but-unused at `e527a9f785` (Codex Thread 2); migrating them now reserves the typed home and lets the resolver populate them from legacy YAML, but the fields stay `status="prototype"` until they're wired into the runtime.

**Phase 7 — Delete `pipeline.fuse_qkv` (and the `PipelineConfig` shell).** Verified zero runtime reads at `e527a9f785`; not migrating. Emit a `DeprecationWarning` for one release if a user passes it (alias maps to a no-op), then remove the field — and at the same time delete the now-stale `PipelineConfig` shell that Phase 3 kept around for the soft-removal window.

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

#### 13.2.1 Nested-field migration (per Codex iter-1 Thread 3 + iter-2 Threads 3 & 4)

Flat-field aliases are not enough for `parallel.refiner_*`,
`parallel.t5_fsdp_size`, and (until removal) `pipeline.fuse_qkv`.
**The migration must run after arch resolution** — and arch
resolution requires HF metadata that's only available after model
download. Therefore migration cannot live in
`VisualGenArgs.model_validator` (which runs at construction time, before
`PipelineLoader` reads `model_index.json`). Splitting ownership:

- **`VisualGenArgs.model_validator`** — handles flat fields whose
  arch is implicit in the field name (`text_encoder_path`,
  `spatial_upsampler_path`, `distilled_lora_path` are all
  LTX-2-specific; the validator can construct
  `arch_config = LTX2ModelConfig(...)` directly).
- **`PipelineLoader.load(args)`** — handles nested fields where the
  arch comes from the resolver (`parallel.refiner_*` →
  `LTX2ModelConfig.refiner_parallel`, `parallel.t5_fsdp_size` →
  `WanModelConfig.text_encoder.fsdp_size`). The resolver runs first; migration
  uses its output to pick the right submodel.

#### Migration pseudocode (in `PipelineLoader`)

The legacy YAML shape is nested:

```yaml
parallel:
  dit_cfg_size: 2
  refiner_dit_dp_size: 2     # legacy LTX-2 two-stage
  t5_fsdp_size: 4            # legacy Wan T5 path
pipeline:
  fuse_qkv: true             # legacy; warned and ignored until Phase 7 deletes
```

```python
# In tensorrt_llm/visual_gen/pipeline_loader.py — runs AFTER arch resolution.

# Explicit source-to-target field map. One entry per legacy field; each names
# the (cross-cutting attribute, target arch class, target submodel attribute,
# target submodel field).
_LEGACY_FIELD_MAP = {
    # parallel.refiner_dit_*  ->  LTX2ModelConfig.refiner_parallel.dit_*
    "refiner_dit_dp_size":      ("parallel", LTX2ModelConfig, "refiner_parallel", "dit_dp_size"),
    "refiner_dit_tp_size":      ("parallel", LTX2ModelConfig, "refiner_parallel", "dit_tp_size"),
    "refiner_dit_ulysses_size": ("parallel", LTX2ModelConfig, "refiner_parallel", "dit_ulysses_size"),
    "refiner_dit_ring_size":    ("parallel", LTX2ModelConfig, "refiner_parallel", "dit_ring_size"),
    "refiner_dit_cfg_size":     ("parallel", LTX2ModelConfig, "refiner_parallel", "dit_cfg_size"),
    "refiner_dit_fsdp_size":    ("parallel", LTX2ModelConfig, "refiner_parallel", "dit_fsdp_size"),
    # NOTE: refiner_dit_cp_size has NO direct ParallelConfig.dit_cp_size target
    # (today's ParallelConfig uses dit_attn2d_row_size/dit_attn2d_col_size for
    # context parallelism, not a single dit_cp_size field). This entry is
    # intentionally absent; setting parallel.refiner_dit_cp_size in legacy YAML
    # raises a clear error in _migrate_legacy_nested with a pointer to the
    # right replacement.
    # parallel.t5_fsdp_size  ->  WanModelConfig.text_encoder.fsdp_size
    # (text_encoder is a T5Config — not a ParallelConfig — because the T5
    # encoder is not DiT-shaped and ParallelConfig.dit_fsdp_size would be
    # the wrong semantic; per Codex iter-3 Thread 3.)
    "t5_fsdp_size":             ("parallel", WanModelConfig, "text_encoder",     "fsdp_size"),
}

def _migrate_legacy_nested(args: VisualGenArgs, resolved_arch_config: ArchConfig) -> ArchConfig:
    """Move legacy nested fields into the resolved arch_config, with explicit
    precedence and no silent drops."""
    parallel_set = args.parallel.model_fields_set  # which fields user actually set

    # Detect orphan legacy values: any legacy source field set under
    # an arch that doesn't own that target.
    for legacy_field, (source_attr, target_cls, _, _) in _LEGACY_FIELD_MAP.items():
        if legacy_field in parallel_set and not isinstance(resolved_arch_config, target_cls):
            warnings.warn(
                f"`parallel.{legacy_field}` was set but the resolved arch is "
                f"{type(resolved_arch_config).__name__}, not {target_cls.__name__}. "
                f"Field will be ignored. Move it to "
                f"`arch_config={target_cls.__name__}(...)` if intended.",
                DeprecationWarning, stacklevel=2,
            )

    # Migrate each owned legacy value, using model_fields_set for explicit
    # detection (no default-equality bug).
    for legacy_field, (source_attr, target_cls, target_subattr, target_field) in _LEGACY_FIELD_MAP.items():
        if legacy_field not in parallel_set:
            continue
        if not isinstance(resolved_arch_config, target_cls):
            continue                                # already warned above
        legacy_value = getattr(getattr(args, source_attr), legacy_field)
        target_sub = getattr(resolved_arch_config, target_subattr)
        if target_field in target_sub.model_fields_set:
            # Both legacy AND explicit new field set → error, not silent override.
            raise ValueError(
                f"Both legacy `parallel.{legacy_field}` and explicit "
                f"`arch_config.{target_subattr}.{target_field}` are set. "
                f"Pick one — the legacy form is deprecated."
            )
        # Migrate.
        setattr(target_sub, target_field, legacy_value)
        warnings.warn(
            f"`parallel.{legacy_field}` is deprecated; set "
            f"`arch_config={target_cls.__name__}({target_subattr}=...)`.",
            DeprecationWarning, stacklevel=2,
        )

    # The cp_size case is special: surface as ValueError, not a silent drop.
    if "refiner_dit_cp_size" in parallel_set:
        raise ValueError(
            "`parallel.refiner_dit_cp_size` has no direct target on the new "
            "shape. Today's ParallelConfig expresses CP via "
            "`dit_attn2d_row_size`/`dit_attn2d_col_size`; pick those instead."
        )

    return resolved_arch_config
```

**Precedence rule** (consistent across all nested migrations):

1. Explicit user-set `arch_config.<sub>.<field>` always wins.
2. Legacy nested field set + arch can be resolved → migrate with
   `DeprecationWarning`.
3. Legacy nested field set + the field doesn't belong to the
   resolved arch → `DeprecationWarning` and the field is dropped
   (with a pointer to the right submodel). No silent drop.
4. Both legacy and explicit submodel field set → `ValueError`.
5. Legacy nested field set + arch cannot be resolved → `ValueError`
   (we cannot guess which submodel owns it).
6. Legacy field with no migration target (e.g. `refiner_dit_cp_size`) →
   `ValueError` with a pointer to the live replacement.

The API-stability test (§10.2 layer 4) gains explicit cases for each
of these six paths plus `both_legacy_families`, `wrong_arch_legacy`,
`explicit_new_default`, `unresolved_arch`.

This ensures every pre-refactor YAML and Python construction continues
to work for two minor releases — and that migration regressions are
loud, not silent.

### 13.3 What we *don't* do for compat

- We do **not** keep `pipeline.enable_layerwise_nvtx_marker` as a field. It moves straight to env var. NVTX is a debug feature; production users do not depend on its config field.
- We do **not** keep `dynamic_weight_quant` / `force_dynamic_quantization` as fields. They were never user-set in any working configuration (the validator computes them); removing them is non-breaking in practice.
- We do **not** maintain the `to_dict()` / `from_dict()` methods (M2 §3.2 already calls these out for removal).

---

## 14. Recommendation Summary

The 10-line version of this doc:

1. Adopt **§6.4 (hybrid)** as the target shape: orthogonal cross-cutting sub-configs + discriminated `arch_config` union per architecture (Pydantic-reserved-namespace-safe; field name locked in iter 1).
2. Move the **three LTX-2 paths** (§7.2) into `LTX2ModelConfig`. **Move (with `status="prototype"` until wired)** the **seven `parallel.refiner_*`** fields (§7.7) into `LTX2ModelConfig.refiner_parallel: ParallelConfig`, and `parallel.t5_fsdp_size` into `WanModelConfig.text_encoder.fsdp_size` (a new `T5Config(StrictBaseModel)`; iter-3 Thread 3). **Delete** `pipeline.fuse_qkv` (§7.6) — verified zero runtime reads at `e527a9f785`.
3. Keep **`skip_warmup` / `skip_components`** as **per-instance `advanced` namespace fields** with permanent `status="prototype"` (§11.1) — env vars break two-VisualGen-in-one-process. Only **`enable_layerwise_nvtx_marker`** moves to env var (`TLLM_VISUALGEN_NVTX_LAYERS=1`), which is genuinely process-wide.
4. Make **`dynamic_weight_quant` / `force_dynamic_quantization`** internal `PrivateAttr` (§7.4).
5. Carve **`OffloadConfig`** out of `PipelineConfig`; keep a deprecated `PipelineConfig` shell until Phase 7 finishes the `fuse_qkv` soft-removal window, *then* delete it (§13.1 Phase 3 + Phase 7).
6. Adopt the **`Field(status=...)`** marker and add a **five-file API stability harness with split policy** (§10.2): (1) `visual_gen_args.yaml` (parent fields), (2) `visual_gen_arch_configs.yaml` (registered submodels), (3a) `visual_gen_arch_registry.yaml` (discriminator literals + resolver patterns; stable-API policy), (3b) `visual_gen_capabilities.yaml` (per-arch capability rows; operational policy + mandatory-row + named-test enforcement), (4) `visual_gen_alias_cases.yaml` (old→new YAML migrations with per-case `direct | loader` `mode`).
7. Use Pydantic **`validation_alias` + `deprecated`** for flat-field renames; run **nested-field migration in `PipelineLoader`** *after* arch resolution (not in `VisualGenArgs.model_validator`, per Codex iter-2 Thread 3) — §10.3, §13.2.
8. Provide a **two-tier discovery API** (§9.2): Tier 1 static type schema (Pydantic-native, free); Tier 2 `VisualGen.resolved_config(model, device, gpus)` returning supported attention/quant/cache/parallel options per arch+hardware. Tier 2's capability table is mandatory + harness-snapshotted (§10.2 layer 3) so it doesn't rot.
9. Treat the **dict + discovery API** alternative (§6.5) as an inferior option vs typed submodels for the same effort. Use a tightly-scoped `extra_args: dict` *inside each submodel* as the forward-compat hatch.
10. Sequence the work as **9 phases** (§13.1), with arch resolution merged into Phase 4 so nested-field migration phases (5–7) can lean on it. Backwards-compat aliases maintained for ≥ 2 minor releases.

---

## 15. Open Questions

1. ~~**`model_config` field name collision.**~~ **Resolved in iter 1** — locked in `arch_config` as the field name and `ArchConfig` as the union type alias. See §6.4 Codex thread + Claude reply.
2. ~~**What's the discriminator value?**~~ **Resolved in iter 5** — locked in `arch: Literal["wan", "flux", "ltx2", ...]`. `model_type` was tentative but tied us to HF metadata field names, and `class_name` leaked the internal registry. `arch` is short, decoupled, and consistent across sketches/YAML/CLI/stability snapshots. See §1 "What this doc does *not* commit to" + Codex iter-5 Thread 3 reply.
3. ~~**Does `arch_config=None` auto-resolve?**~~ **Decided** — yes, the resolver lands in Phase 4 of the migration (per Codex Thread 3 reordering). Read HF `model_index.json::_class_name` and dispatch to the registry; `VisualGen(model="Wan-AI/...")` Just Works without users typing `WanModelConfig`. SGLang #20078 risk pinned by the alias-cases YAML in §10.2 layer 4.
4. **Folding `torch_compile` and `cuda_graph` into `compilation`.** M2 §15 Q1. Outside this doc's scope, but if folded, the resulting `CompilationConfig` mirrors vLLM exactly.
5. **`AttentionConfig` evolution to a discriminated union.** §8.1 — when does this trigger? **Tentative**: as soon as a backend needs a backend-specific kwarg; not before.
6. **Promoting `extra_args` fields.** When a key in `arch_config.extra_args` becomes used by 3+ commits and stable for 1+ release, promote to a typed field on the submodel. Need a process or just judgment? **Tentative**: judgment, captured in CONTRIBUTING.md.
7. **CLI integration timeline.** vLLM-style `--config + dotted overrides` would be valuable for `trtllm-serve` but is not blocking this refactor. **Tentative**: defer to a follow-up PR.
8. **Out-of-tree model registration.** Should `register_arch_config(name, cls)` be a public function (PEFT-style)? This enables third-party plugins but commits us to a registry contract. **Tentative**: not in this milestone; revisit if a plugin ecosystem emerges.
9. **Relationship with `DiffusionModelConfig` (the internal merged config).** `DiffusionModelConfig.from_pretrained` reads HF + args and produces the merged config. With per-arch submodels, the merging logic gets cleaner — each arch's submodel knows how to merge its own HF metadata. **Tentative**: keep `DiffusionModelConfig` internal as M2 §10.3 proposes, but its `from_pretrained` becomes a dispatch over the registered submodels.
10. **Per-arch `fuse_qkv` defaults.** Moot — `pipeline.fuse_qkv` is being deleted (Codex Thread 2 verified zero runtime reads). If/when QKV-fusion-as-config returns, per-arch defaults are the right shape; not a blocker now.
11. ~~**Capability table coverage for Tier 2 discovery.**~~ **Resolved in iter 2** — capability rows are part of the registry entry; layer 3 of the API-stability harness fails CI if a registered arch lacks one. See §9.2 iter-2 Codex thread + Claude reply.

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
| 1  | 2026-05-08 | recommendation, field dispositions, migration ordering, stability, discovery, env-var/per-instance | 6       | 6        | 0    | 0        |
| 2  | 2026-05-08 | iter-1 normative-section drift, phase ordering, resolver/migration ownership, pseudocode bugs, capability-table rot | 5       | 5        | 0    | 0        |
| 3  | 2026-05-08 | `fuse_qkv` straggler in §6.4 sketch + §12 examples, unimplementable PipelineConfig shell, t5_fsdp_size target field bug, stale §4 principles | 4       | 4        | 0    | 0        |
| 4  | 2026-05-08 | `_advanced` is private in Pydantic v2, §10.2 stale opening + wrong layer-4 contract, residual `t5_parallel` references, Pydantic `deprecated=` doesn't fire at validation, scope-vs-Tier-2 contradiction | 5       | 5        | 0    | 0        |
| 5  | 2026-05-08 | offload migration rejects existing `pipeline:` YAML, §1 Exec Summary still defers Tier-2, discriminator both undecided and hardcoded, capability rows frozen at wrong granularity | 4       | 4        | 0    | 0        |
| 6  | 2026-05-08 | §6.4 sketch omits `advanced` namespace, four-vs-five-file contract leftover in normative sections, T5Config NameError ordering | 3       | 3        | 0    | 0        |
| 7  | 2026-05-08 | `PipelineConfig` referenced before declaration in §6.4 sketch | 1       | 1        | 0    | 0        |
| 8  | 2026-05-08 | §6.4 sketch still references `OffloadConfig` / `ObservabilityConfig` without declaring them; Scope still says discriminator is undecided | 2       | 2        | 0    | 0        |
| 9  | 2026-05-08 | strict convergence final-check pass | 0       | 0        | 0    | 0        |

**Converged on 2026-05-08 after iteration 9** — Codex's iter-9 verdict was `approve` with no material findings; the strict convergence filter found no concrete blocker the prior iterations missed. 30 substantive threads were raised across 8 review rounds (6 + 5 + 4 + 5 + 4 + 3 + 1 + 2) and all were resolved. Severity decreased monotonically from [high] (iter 1–4) to [medium] (iter 5–8) to none (iter 9). The doc's inline Codex/Claude review threads were stripped on finalize; this Iteration Tracker is the audit trail. Per-iteration commit history: `102e3df08a` (initial draft + iter 1) → `09116d6604` (iter 2) → `02e847c5a7` (iter 3) → `d074c003bd` (iter 4) → `a2b76d26cc` (iter 5) → `b581b2ad0d` (iter 6) → `6f152f7d3c` (iter 7) → `a59ce5b2d9` (iter 8) → finalize.
