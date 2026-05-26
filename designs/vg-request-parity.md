# VisualGen Serve↔Python Request-Param Parity

> **Status**: Draft — under discussion
> **Date**: 2026-05-25
> **Related**: code under `tensorrt_llm/visual_gen/`, `tensorrt_llm/_torch/visual_gen/`, and `tensorrt_llm/serve/`. Sibling design docs are referenced only where load-bearing; this doc grounds in source, not prior docs.

---

## Scope, Target & Non-Goals

This section records the requirement, scope, target, and non-goals as
clarified with the design owner before drafting. A separate review agent
reads this section first; if its content drifts from what's actually
covered in the rest of the doc, that's a flag to either re-confirm
scope with the owner or rewrite the affected sections.

### Requirement (one paragraph)

After M2 landed the Python `VisualGen` / `VisualGenParams` API, the
HTTP request schemas used by `trtllm-serve` (`ImageGenerationRequest`,
`VideoGenerationRequest`) have drifted from the Python `VisualGenParams`
they convert into. Some Python-side fields have no HTTP analog
(`max_sequence_length`, `image_cond_strength`, `extra_params` overflow).
One HTTP field (`guidance_rescale`) only works for certain models and
the executor would reject it for others. Defaults, types, and
constraints diverge in places that aren't documented and weren't
deliberately chosen. The design must propose a target shape for the
two HTTP request schemas — what to add, drop, rename, validate, and
how to transport model-specific overflow params — that lets the Python
API and the trtllm-serve API evolve together as new models land,
without silent typo bugs or per-model wire surprises.

### In scope

- **Field-by-field gap inventory** between `VisualGenParams` and the
  two HTTP request schemas (`ImageGenerationRequest`,
  `VideoGenerationRequest`).
- **Gap classification**: missing on HTTP; intentional asymmetry;
  accidental drift to fix; routed-via-`extra_params` overflow.
- **Bugs surfaced by the gap analysis**: items where the current code
  is observably wrong (silent field drops, per-model invalidity,
  silent default coercion).
- **Target shape for the two HTTP request schemas** — what to add,
  rename, drop, validate, and how to keep the OpenAI image/video
  request surface stable.
- **`extra_params` HTTP transport** — the central design choice. The
  doc presents the reference-framework survey (vLLM, vLLM-Omni,
  SGLang, SGLang-Diffusion, OpenAI, fal.ai/Replicate) and a
  recommendation tailored to the no-silent-typo constraint.
- **Conversion layer** (`tensorrt_llm/serve/visual_gen_utils.py::
  parse_visual_gen_params`) — what to keep, what to rewrite, how
  defaults flow from `generator.default_params`.
- **Validation strategy** — boundary validation at the HTTP layer
  vs. strict-key validation in the executor (`DiffusionExecutor.
  _validate_request`); error response format.
- **Migration plan** — direct edits, no compat shims. The Python API
  and the HTTP layer are both pre-GA.

### Out of scope (non-goals)

- **`--visual_gen_args` CLI flag and `VisualGenArgs` engine config** —
  this is engine-level config (YAML → `VisualGenArgs`), not per-request
  schema. Already settled.
- **`ImageEditRequest` and `/v1/images/edits`** — the route returns
  501 Not Implemented today; the request type is acknowledged but
  not redesigned here.
- **Output encoding** — base64 vs URL, MP4/AVI selection, FileResponse
  vs JSON. Already settled.
- **Async video job lifecycle** — `/v1/videos` POST (async) +
  `GET/DELETE /v1/videos/{id}` and related endpoints stay as-is.
- **Streaming intermediate frames** — Python side does not expose
  this; HTTP cannot either. Out.
- **Discovery endpoint** — no `GET /v1/models/{model}/params` or
  similar. Users discover model-specific keys via the Python API
  (`VisualGen.extra_param_specs`). Commercial deployments may want
  this later; not this design.
- **`status="prototype"` / API-stability markers on HTTP fields** —
  not a trtllm-serve convention (none in `openai_protocol.py` today,
  confirmed by `grep -nE 'status=' tensorrt_llm/serve/openai_protocol.py`
  → no matches). Stays Python-API-only.
- **Per-sample heterogeneous batch params for `n > 1`** — Python
  `generate_async` explicitly raises `NotImplementedError` for
  `List[VisualGenParams]` (`tensorrt_llm/visual_gen/visual_gen.py:852-855`).
  Out of this design; called out as an Open Question.
- **Backwards-compat shims** — no `validation_alias`, no deprecation
  cycle.
- **API-stability test harness** — separate task.
- **Implementation PRs / detailed impl plan beyond the migration
  outline.**

### Target / Audience

- **TRT-LLM VisualGen engineers** (primary) — own the refactor
  execution across `tensorrt_llm/visual_gen/`,
  `tensorrt_llm/_torch/visual_gen/`, and `tensorrt_llm/serve/`.
- **trtllm-serve maintainers** (secondary) — own the OpenAI-compatible
  HTTP surface and the conversion layer.
- **HTTP clients of `trtllm-serve` for image/video generation**
  (tertiary) — affected by the new schema; no migration aliases since
  the API is pre-GA.

### Investigation directive

Per the design owner: ground claims in actual source on this branch
(`tensorrt_llm/visual_gen/` and `tensorrt_llm/_torch/visual_gen/` for
the Python API; `tensorrt_llm/serve/` for the HTTP layer). Sibling
design docs are advisory only.

---

## Table of Contents

1. [Background: the two APIs in one page](#1-background-the-two-apis-in-one-page)
2. [Python side — what `VisualGenParams` actually is](#2-python-side--what-visualgenparams-actually-is)
3. [HTTP side — what `trtllm-serve` accepts today](#3-http-side--what-trtllm-serve-accepts-today)
4. [Gap inventory — field-by-field](#4-gap-inventory--field-by-field)
5. [Bugs surfaced by the gap analysis](#5-bugs-surfaced-by-the-gap-analysis)
6. [The central question — `extra_params` over HTTP](#6-the-central-question--extra_params-over-http)
7. [Target HTTP request schemas](#7-target-http-request-schemas)
8. [Conversion layer](#8-conversion-layer)
9. [Validation strategy](#9-validation-strategy)
10. [Migration plan](#10-migration-plan)
11. [Open Questions](#11-open-questions)
12. [Iteration Tracker](#12-iteration-tracker)

---

## 1. Background: the two APIs in one page

The VisualGen request path has three layers:

```
HTTP client                                Python client
   │                                          │
   ▼                                          │
ImageGenerationRequest  │                     │
VideoGenerationRequest  ├─ Pydantic body      │
   │                    │   in serve/         │
   ▼                                          │
parse_visual_gen_params ─ overlays request  ──┤
   │                      onto pipeline       │
   │                      defaults            │
   ▼                                          ▼
VisualGenParams  ◄── shared engine-side per-request schema ──► VisualGenParams
   │
   ▼
VisualGen.generate(inputs=..., params=...)
   │
   ▼
DiffusionExecutor._validate_request   ── strict-key validation
   │
   ▼
DiffusionExecutor._merge_defaults     ── fills None fields from pipeline
   │
   ▼
Pipeline.infer(req)                   ── reads params.field / params.extra_params[key]
```

The Python client calls `VisualGen.generate(...)` with a
`VisualGenParams` it built. The HTTP client sends an OpenAI-shaped
request body; `parse_visual_gen_params` translates it into the same
`VisualGenParams` and hands the result to the same engine.

**The gap is at the seam between the HTTP body and `VisualGenParams`.**
The translator is selective: some fields are mapped, some are
hardcoded, some are dropped, and one (`guidance_rescale`) is routed
into a Python overflow dict that only some models accept.

The endpoints in scope (`tensorrt_llm/serve/openai_server.py:738-767`):

| Endpoint | Method | Request model | Notes |
| --- | --- | --- | --- |
| `/v1/images/generations` | POST | `ImageGenerationRequest` | Sync |
| `/v1/videos/generations` | POST | `VideoGenerationRequest` | Sync, returns FileResponse |
| `/v1/videos` | POST | `VideoGenerationRequest` | Async, returns `VideoJob` (202) |

Out of scope: `/v1/images/edits` (501 today); `GET`/`DELETE` video-job
management endpoints (no request body).

---

## 2. Python side — what `VisualGenParams` actually is

`VisualGenParams` is a Pydantic class with 14 fields plus an `extra_params`
dict for model-specific overflow (`tensorrt_llm/visual_gen/params.py:1-78`).
The class is decorated `@set_api_status("prototype")` (line 22).

### 2.1 Field table

| Field | Type | Default | Note |
| --- | --- | --- | --- |
| `height` | `Optional[int]` | `None` | `None` → pipeline default |
| `width` | `Optional[int]` | `None` | `None` → pipeline default |
| `num_inference_steps` | `Optional[int]` | `None` | |
| `guidance_scale` | `Optional[float]` | `None` | |
| `max_sequence_length` | `Optional[int]` | `None` | Max tokens for text encoder |
| `seed` | `int` | `42` | Non-optional, explicit default |
| `num_frames` | `Optional[int]` | `None` | Video only |
| `frame_rate` | `Optional[float]` | `None` | Video only |
| `negative_prompt` | `Optional[str]` | `None` | |
| `image` | `Optional[Union[str, bytes, List[Union[str, bytes]]]]` | `None` | Reference image(s) — path or raw bytes |
| `mask` | `Optional[Union[str, bytes, List[bytes]]]` | `None` | Inpainting mask |
| `image_cond_strength` | `Optional[float]` | `None` | |
| `num_images_per_prompt` | `int` | `1` | Non-optional |
| `extra_params` | `Optional[Dict[str, Any]]` | `None` | Model-specific overflow; keys discovered via `VisualGen.extra_param_specs` |

**Two semantics encoded by `None` vs explicit default:** `None` means
"use pipeline default"; an explicit value overrides. `seed` is the
exception — it carries `42` so reproducibility is the documented
default behavior of the Python API. This is load-bearing in §5.

### 2.2 The `extra_params` schema — not a dict-of-anything

Each pipeline declares its accepted `extra_params` keys via an
`extra_param_specs` property returning `Dict[str, ExtraParamSchema]`
(`tensorrt_llm/_torch/visual_gen/pipeline.py:23-36`):

```python
class ExtraParamSchema(StrictBaseModel):
    type: str                                  # "float", "int", "bool", "str", "list"
    default: Any = None
    description: str = ""
    range: Optional[tuple] = None              # (min, max) for numeric
```

Each model's `extra_param_specs` lives next to its pipeline. Concrete
examples on this branch:

- **LTX2** (`tensorrt_llm/_torch/visual_gen/models/ltx2/pipeline_ltx2.py:1260+`)
  declares 7 keys: `output_type`, `guidance_rescale`, `stg_scale`,
  `stg_blocks`, `modality_scale`, `rescale_scale`, `guidance_skip_step`,
  `enhance_prompt`.
- **Wan 2.2 A14B** (`tensorrt_llm/_torch/visual_gen/models/wan/defaults.py:69-81`)
  declares 2 keys: `guidance_scale_2`, `boundary_ratio` (with
  `range=(0.0, 1.0)`).
- **Wan 2.1** and **Wan 2.2 TI2V-5B** declare zero keys
  (`tensorrt_llm/_torch/visual_gen/models/wan/defaults.py:139-147`:
  `if is_wan22_14b: return dict(_WAN22_EXTRA_SPECS); return {}`).

This is the central point: **`extra_params` is a typed, per-pipeline
dictionary, not a free-form `dict[str, Any]`.** The executor enforces
the schema at validation time (§9).

### 2.3 Engine consumes `VisualGenParams`

`VisualGen.generate(inputs, params=None)` and `generate_async(...)`
(`tensorrt_llm/visual_gen/visual_gen.py:805-829`, `832-884`) accept a
single optional `VisualGenParams` shared across all prompts in a batch.
`generate_async` deep-copies the params before enqueuing
(`visual_gen.py:880`), so callers cannot mutate a live request. Per-item
params are explicitly not yet supported:

```python
# visual_gen.py:852-855
if isinstance(params, list):
    raise NotImplementedError(
        "Per-item params (List[VisualGenParams]) are not yet supported."
    )
```

`VisualGen.default_params` (`visual_gen.py:778-802`) is the Python-side
discovery surface: it returns a fully-populated `VisualGenParams` with
universal fields filled from `executor.default_generation_params` and
`extra_params` filled from `executor.extra_param_specs[*].default`.
Python clients call this to learn what the engine accepts.

---

## 3. HTTP side — what `trtllm-serve` accepts today

### 3.1 `ImageGenerationRequest`

`tensorrt_llm/serve/openai_protocol.py:1306-1360`. Inherits
`OpenAIBaseModel`, which sets `model_config = ConfigDict(extra="forbid",
populate_by_name=True)` at line 105 — unknown top-level fields are
rejected with 400.

| Field | Type | HTTP default | Constraints |
| --- | --- | --- | --- |
| `prompt` | `str` | required | — |
| `model` | `Optional[str]` | `None` | — |
| `n` | `int` | `1` | `ge=1, le=10` |
| `output_format` | `Literal["png", "webp", "jpeg"]` | `"png"` | — |
| `size` | `Optional[str]` | `"auto"` | regex `^\d+x\d+$` or `"auto"` |
| `quality` | `Literal["standard", "hd"]` | `"standard"` | side-effect: see §5.4 |
| `response_format` | `Literal["url", "b64_json"]` | `"url"` | — |
| `style` | `Optional[Literal["vivid", "natural"]]` | `"vivid"` | — |
| `user` | `Optional[str]` | `None` | — |
| `num_inference_steps` | `Optional[int]` | `None` | — |
| `guidance_scale` | `Optional[float]` | `None` | — |
| `guidance_rescale` | `Optional[float]` | `None` | per-model validity, see §5.3 |
| `negative_prompt` | `Optional[str]` | `None` | — |
| `seed` | `Optional[int]` | `None` | not currently mapped, see §5.1 |

### 3.2 `VideoGenerationRequest`

`tensorrt_llm/serve/openai_protocol.py:1425-1483`. Same base, same
`extra="forbid"`.

| Field | Type | HTTP default | Constraints |
| --- | --- | --- | --- |
| `prompt` | `str` | required | — |
| `input_reference` | `Optional[Union[str, UploadFile]]` | `None` | Base64 string or multipart upload |
| `model` | `Optional[str]` | `None` | — |
| `size` | `Optional[str]` | `"auto"` | regex `^\d+x\d+$` or `"auto"` |
| `seconds` | `float` | `2.0` | `ge=1.0, le=16.0` |
| `n` | `int` | `1` | `ge=1, le=4` |
| `fps` | `int` | `24` | `ge=8, le=60` |
| `num_inference_steps` | `Optional[int]` | `None` | — |
| `guidance_scale` | `Optional[float]` | `None` | — |
| `guidance_rescale` | `Optional[float]` | `None` | per-model validity, see §5.3 |
| `negative_prompt` | `Optional[str]` | `None` | — |
| `seed` | `Optional[int]` | `None` | mapped, see §5.2 |
| `output_format` | `Literal["mp4", "avi", "auto"]` | `"auto"` | — |

### 3.3 Conversion — `parse_visual_gen_params`

`tensorrt_llm/serve/visual_gen_utils.py:15-83`. Starts from
`generator.default_params` (line 23), overlays request fields:

```python
# visual_gen_utils.py:15-34 (header + common section)
def parse_visual_gen_params(
    request: ImageGenerationRequest | VideoGenerationRequest | ImageEditRequest,
    id: str,
    generator: VisualGen,
    media_storage_path: Optional[str] = None,
) -> VisualGenParams:
    params = generator.default_params
    if params.extra_params is None:
        params.extra_params = {}

    if request.negative_prompt is not None:
        params.negative_prompt = request.negative_prompt
    if request.size is not None and request.size != "auto":
        params.width, params.height = map(int, request.size.split("x"))
    if request.guidance_scale is not None:
        params.guidance_scale = request.guidance_scale
    if request.guidance_rescale is not None:
        params.extra_params["guidance_rescale"] = request.guidance_rescale  # line 34
```

The image branch (lines 36-53) maps `num_inference_steps`, `n`, and
the `quality="hd"` side-effect. The video branch (lines 55-76) maps
`num_inference_steps`, `n`, `input_reference`, `fps`/`seconds`, and
`seed`. Notably, the **image branch never reads `request.seed`**; that
is the §5.1 bug.

---

## 4. Gap inventory — field-by-field

Symbols: ✅ present, ❌ absent, ⚠️ present-with-divergence, ⤵️ promoted to `extra_params` overflow.

| `VisualGenParams` field | Image HTTP | Video HTTP | Notes |
| --- | --- | --- | --- |
| `height` | ⚠️ via `size: str` | ⚠️ via `size: str` | Type/shape divergence; HTTP keeps OpenAI `"WxH"` string |
| `width` | ⚠️ via `size: str` | ⚠️ via `size: str` | Same |
| `num_inference_steps` | ✅ | ✅ | Image: also forced to `30` when `quality="hd"` (§5.4) |
| `guidance_scale` | ✅ | ✅ | — |
| `max_sequence_length` | ❌ | ❌ | Absent on both; users cannot pass text-encoder max tokens via HTTP |
| `seed` | ⚠️ present but silently dropped | ✅ mapped | **§5.1** (image bug), **§5.2** (default-divergence) |
| `num_frames` | n/a | ⚠️ derived from `seconds × fps` | Cannot send `num_frames` directly |
| `frame_rate` | n/a | ⚠️ via `fps: int` | Type divergence (HTTP `int`; Python `float`) |
| `negative_prompt` | ✅ | ✅ | — |
| `image` | n/a (image edits OOS) | ⚠️ via `input_reference` (base64 / UploadFile) | Transport difference is intentional |
| `mask` | n/a | ❌ | Video inpainting not exposed via HTTP |
| `image_cond_strength` | n/a | ❌ | Absent; Wan I2V declares it in default params |
| `num_images_per_prompt` | ⚠️ via `n: int (1..10)` | ⚠️ via `n: int (1..4)` | Constraint divergence (Python uncapped) |
| `extra_params` | ❌ | ❌ | No HTTP path to send model-specific keys other than the hardcoded `guidance_rescale` (§5.3, §6) |

| HTTP field without a `VisualGenParams` counterpart | Why it exists |
| --- | --- |
| `model` | OpenAI-compatibility (routes to the correct engine; ignored downstream when only one is loaded) |
| `output_format` (image: `png`/`webp`/`jpeg`; video: `mp4`/`avi`/`auto`) | Encoding-side; out of scope |
| `quality` (image) | OpenAI compatibility; today mapped to `num_inference_steps=30` for `"hd"` (§5.4) |
| `response_format` (image: `url`/`b64_json`) | Encoding-side; out of scope |
| `style` (image: `vivid`/`natural`) | Accepted but unused (no pipeline reads it today) |
| `user` (image) | Accepted but unused; OpenAI-shaped trace field |
| `seconds`/`fps` (video) | OpenAI video-shape (used to compute `num_frames`) |
| `output_format = "auto"` (video) | Server-side ffmpeg/encoder fallback |
| `guidance_rescale` | Hardcoded → `extra_params["guidance_rescale"]`; per-model invalid (§5.3) |

### 4.1 Gap classification

- **Genuine missing on HTTP** (could be added, with care): `max_sequence_length`, `image_cond_strength`. Long tail: model-specific keys from `extra_param_specs`.
- **Intentional asymmetry** (keep): `size` as OpenAI `"WxH"`; `n` constraint range; `input_reference` as base64/upload; `num_frames = seconds × fps`; `quality` as compat label.
- **Accidental drift to fix**: image `seed` silently dropped (§5.1); `seed` default semantics mismatch (§5.2); `quality="hd"` overriding model default (§5.4); HTTP-only fields with no pipeline consumer (`style`, `user`).
- **Routed via `extra_params` overflow**: `guidance_rescale`. Brittle today because the routing is per-field hardcoded and per-model invalid (§5.3, §6).
- **Out of scope but flagged**: video inpainting (`mask`, `image_cond_strength` for video) — not designed-in here; left as an Open Question.

---

## 5. Bugs surfaced by the gap analysis

These are concrete defects in the current `tensorrt_llm/serve/` code. They
are not the central design question but the design must decide whether to
fix them here or in a follow-up PR (see Open Question §11.1).

### 5.1 Image `seed` is silently dropped

`ImageGenerationRequest` declares `seed: Optional[int] = None`
(`openai_protocol.py:1356` area), but `parse_visual_gen_params` has no
line that maps it into `params.seed`. The conversion's image branch
(`visual_gen_utils.py:36-53`) reads `num_inference_steps`, `quality`,
`n`, and (in the `ImageEditRequest` sub-branch) `image`/`mask`. There
is no `if request.seed is not None: params.seed = int(request.seed)`
clause for `ImageGenerationRequest` — confirmed by
`grep -n 'request.seed' tensorrt_llm/serve/visual_gen_utils.py`
returning only lines 75-76 inside the video branch.

**Observable effect:** every HTTP image request runs with the Python
default `seed=42` regardless of what the client sent. Reproducibility
illusion: two requests with different `seed` values produce identical
output.

### 5.2 `seed` default semantics mismatch

Python `VisualGenParams.seed: int = 42` (explicit non-None default).
HTTP `seed: Optional[int] = None` on both image and video. The video
conversion (`visual_gen_utils.py:75-76`) only overrides when
`request.seed is not None`. When the HTTP client omits `seed`, the
engine runs with `42`, not the model's seed-defaulting convention or
a truly random seed.

**Observable effect:** HTTP `seed=None` yields reproducibility, not
randomness. Clients expecting OpenAI-style "no seed = random" get
deterministic output without knowing why.

Fix is forced by §5.1 — once image `seed` is wired up, both endpoints
should agree on what "no seed" means. Three options, picked in §7:
(a) HTTP default of `42` matching Python; (b) Python default of
`None` matching HTTP; (c) HTTP keeps `None`, conversion explicitly
maps `None` to a documented engine convention.

### 5.3 `guidance_rescale` is per-model invalid

The HTTP schema declares `guidance_rescale: Optional[float] = None`
as a top-level field on both image and video requests. The conversion
unconditionally routes it to `params.extra_params["guidance_rescale"]`
(`visual_gen_utils.py:34`), regardless of which model is loaded.

On the executor side (`_torch/visual_gen/executor.py:243-311`):

```python
# _validate_request, lines 260-266
if params.extra_params:
    unknown = set(params.extra_params.keys()) - set(specs.keys())
    if unknown:
        errors.append(
            f"Unknown extra_params {sorted(unknown)} for {pipeline_name}. "
            f"Supported: {sorted(specs.keys())}"
        )
# ...
if errors:
    raise ValueError(msg)
```

`guidance_rescale` is declared in LTX2's `extra_param_specs` but **not**
in Wan 2.1's or Wan 2.2 TI2V-5B's (which return `{}` from
`get_wan_extra_param_specs`, `tensorrt_llm/_torch/visual_gen/models/wan/defaults.py:139-147`).
A client that sends `guidance_rescale=0.7` to a Wan 2.1 model gets a
`ValueError` from the executor, surfaced as a 500 by the endpoint
handler.

**Observable effect:** the HTTP schema advertises a field that fails
at request time for 2/3 of the Wan models in tree. The same field
quietly works for LTX2.

### 5.4 `quality="hd"` side-effect overrides model default

```python
# visual_gen_utils.py:36-42
if isinstance(request, (ImageGenerationRequest, ImageEditRequest)):
    if request.num_inference_steps is not None:
        params.num_inference_steps = request.num_inference_steps
    elif isinstance(request, ImageGenerationRequest) and request.quality == "hd":
        params.num_inference_steps = 30
    if request.n is not None:
        params.num_images_per_prompt = request.n
```

When `quality="hd"` and the client did not pass `num_inference_steps`,
the conversion forces `num_inference_steps=30`. This is hardcoded
regardless of the pipeline's actual default (which may be very
different — some pipelines use 20, 28, or 50; tied to a model's
trained schedule).

**Observable effect:** `"hd"` is a no-op for some models, a regression
for others, and an unrelated semantic for the rest. The mapping
shouldn't be `quality → steps`; it should be `quality → "the model's
high-quality preset"`, which is per-model and not modelable as a
single integer.

### 5.5 Docstring/code inconsistency in executor

`tensorrt_llm/_torch/visual_gen/executor.py:71-86` comments above
`_GENERATION_CONFIG_FIELDS` say "the value will be silently ignored",
but `_validate_request` actually **raises** when one of those fields is
set but not declared by the pipeline (lines 273-280, 307-311). Minor
nit; flagging for the impl PR.

---

## 6. The central question — `extra_params` over HTTP

The Python API has `VisualGenParams.extra_params: dict[str, Any]` as the
overflow for model-specific knobs, with a typed schema
(`ExtraParamSchema`) per pipeline that the executor enforces. The
HTTP side has nothing: today only one extension key (`guidance_rescale`)
flows through, hardcoded in the conversion, and §5.3 shows that
approach already broke.

How do mature OpenAI-compatible servers handle this? The survey
shows convergence on a single pattern with a known footgun. This
section runs the survey, names the patterns, and recommends the
shape that fits the no-silent-typo constraint.

### 6.1 Patterns observed

- **Pattern A — Open `dict[str, Any]` at top level.** SGLang's native
  `/generate` endpoint takes `sampling_params: Dict | List[Dict]`
  (sglang `python/sglang/srt/managers/io_struct.py`). Zero schema
  friction, zero validation, no IDE help.
- **Pattern B — Open dict in a typed envelope.** vLLM's `vllm_xargs`
  (`vllm/entrypoints/openai/chat_completion/protocol.py`); vLLM-Omni's
  `lora` (`vllm_omni/entrypoints/openai/protocol/images.py`); SGLang
  Diffusion's `diffusers_kwargs`; SGLang chat's `custom_params`. The
  field is a documented dict; the contents are validated downstream.
- **Pattern C — Top-level promotion of well-known keys.** vLLM
  promotes ~50 keys to top-level `Optional[T] = None` fields
  (`top_k`, `min_p`, `repetition_penalty`, etc.). vLLM-Omni promotes
  the diffusion extension set (`num_inference_steps`,
  `guidance_scale`, …) with `Field(ge=..., le=...)` validation.
  This is what trtllm-serve does today for `guidance_scale`,
  `num_inference_steps`, etc.
- **Pattern D — Per-model schema dispatch.** Replicate/Cog and fal.ai
  generate a per-model OpenAPI from the Python `Predictor` class.
  Out of scope here.
- **Pattern E — `extra="allow"` Pydantic config.** vLLM uses this with
  a DEBUG-level log of unknown keys
  (`vllm/entrypoints/openai/engine/protocol.py:27-56`). The well-known
  silent-typo trap: `extra_body={"top-k": 5}` (hyphen typo) passes
  through, logged at DEBUG, production never sees it. Reference: vLLM
  Issue #7337, #11153.
- **Pattern F — Discovery endpoint** (`GET /v1/models/{model}/params`).
  fal.ai/Replicate publish per-model OpenAPI. **Explicitly out of
  scope** per design owner; users discover via the Python API.

### 6.2 Pattern × framework matrix (compressed)

| Pattern | vLLM | vLLM-Omni | SGLang | SGLang-Diff | OpenAI |
| --- | --- | --- | --- | --- | --- |
| C (promote well-known) | ✓ many | ✓ primary | ✓ many | ✓ many | ✓ only |
| B (namespaced overflow) | `vllm_xargs`, `kv_transfer_params`, `mm_processor_kwargs` | `lora` | `custom_params`, `chat_template_kwargs` | `diffusers_kwargs` | — |
| E (`extra="allow"`) | ✓ + DEBUG log | — | partial | image: ✓, video: ✗ (inconsistent) | ✗ closed |
| Discovery (F) | — | — | — | — | — |

**Footgun consensus:** Pattern E is universal among LLM/diffusion
servers because OpenAI SDK clients may add new fields the server
doesn't know yet. Every server doing E reports silent-typo bug
reports. OpenAI itself uses closed-schema (no E) because they own
both client SDK and server.

### 6.3 The recommendation: C + B with `extra="forbid"`

1. **Keep Pattern C** for every parameter trtllm-serve officially
   exposes today — OpenAI-standard fields (`prompt`, `n`, `size`,
   `seed`, `model`, `response_format`, …) and TRT-LLM extensions
   that already have top-level treatment (`negative_prompt`,
   `num_inference_steps`, `guidance_scale`). Each gets a
   `Field(ge=..., le=..., description=...)` so the OpenAPI surface
   documents bounds and the boundary catches typos and bad values.
2. **Add one Pattern-B namespaced overflow field**, named identically
   to the Python field for symmetry:
   `extra_params: Optional[Dict[str, Any]] = Field(default=None,
   description="Model-specific parameters forwarded to the underlying
   pipeline. See per-model docs for accepted keys.")`. HTTP `extra_params`
   maps 1:1 to Python `VisualGenParams.extra_params`. No rename.
3. **Keep `extra="forbid"`** on `OpenAIBaseModel`. This is the
   anti-silent-typo measure. A typo like `negative_promot` produces
   a 422 with the field name; not silent default-substitution. The
   cost — losing forward-compat when the official OpenAI SDK ships a
   new field — is small: such additions are infrequent, the failure
   mode is loud, easy to fix in a one-line schema PR, and easy to
   test for.
4. **Validate `extra_params` contents in the executor**, not in the
   HTTP layer. The HTTP layer accepts a `Dict[str, Any]`; the executor
   already enforces unknown-key and type/range checks
   (`_torch/visual_gen/executor.py:259-305`). Surface those validation
   errors as 400 at the endpoint (not 500). The executor's existing
   error message is descriptive enough to use verbatim.
5. **Do not promote `guidance_rescale` to top-level on HTTP.** Today's
   "top-level on HTTP, dict-key on Python" is exactly the per-model
   invalidity bug in §5.3. Move it into HTTP `extra_params` so it
   shares the same per-model validation as every other model-specific
   key. Clients of LTX2 send `{"extra_params": {"guidance_rescale": 0.7}}`;
   clients of Wan 2.1 get a clear 400 if they try.
6. **Do not split the schema per model** (Pattern D). One
   `ImageGenerationRequest`, one `VideoGenerationRequest`. The
   per-model variation lives inside `extra_params` and is documented
   in narrative docs.

The result is a schema shaped like vLLM-Omni's `images.py` but with
`extra="forbid"` instead of an `extra="allow"` ancestor, and with
`lora` replaced by a generic `extra_params: Dict[str, Any]` whose
contents are validated by the existing executor.

### 6.4 Trade-offs

| Concern | This design's answer |
| --- | --- |
| Silent typos at the HTTP boundary | Closed: `extra="forbid"` rejects unknown top-level keys |
| Silent typos inside `extra_params` | Closed: executor `_validate_request` rejects unknown keys with the spec list |
| Adding a new model-specific knob | No HTTP schema change; declare in pipeline's `extra_param_specs`, document |
| Per-model invalidity (the §5.3 bug) | Moved into `extra_params`; same validation as every other key |
| OpenAPI surface bloat | Bounded: only officially-supported knobs are top-level |
| OpenAI-SDK forward compat | Lost for unknown fields the SDK introduces. Mitigated by: (a) SDK additions are infrequent and announced; (b) loud 422 is easier to triage than silent drop |
| Cross-language clients | Generic `extra_params` cleanly maps to `Map<String, Object>` / `Dict[str, Any]` in any language; per-model docs cover keys |

---

## 7. Target HTTP request schemas

### 7.1 `ImageGenerationRequest` — target shape

```python
class ImageGenerationRequest(OpenAIBaseModel):
    # OpenAI-standard
    prompt: str
    model: Optional[str] = None
    n: int = Field(default=1, ge=1, le=10)
    size: Optional[str] = Field(default="auto", pattern=r"^(\d+x\d+|auto)$")
    response_format: Literal["url", "b64_json"] = "url"
    output_format: Literal["png", "webp", "jpeg"] = "png"
    user: Optional[str] = None
    seed: Optional[int] = None              # §7.4 — semantics fixed

    # TRT-LLM extensions (top-level, "well-known")
    num_inference_steps: Optional[int] = Field(default=None, ge=1, le=200)
    guidance_scale: Optional[float] = Field(default=None, ge=0.0, le=20.0)
    negative_prompt: Optional[str] = None

    # Model-specific overflow
    extra_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Model-specific parameters forwarded to the underlying pipeline. "
            "See per-model docs for accepted keys. Unknown keys are rejected."
        ),
    )

    # Removed: `quality`, `style`. Rationale below.
    # Removed: `guidance_rescale`. Moves into extra_params.
```

**Drops:**

- `quality`: today only triggers the `num_inference_steps=30` hardcoded
  override (§5.4). Either the user passes `num_inference_steps`
  explicitly, or the model default is used. `"hd"` was OpenAI-shape
  ceremony with broken semantics.
- `style`: accepted but unused (no pipeline consumer). Drop now
  rather than later when it acquires accidental semantics.
- `guidance_rescale`: moves to `extra_params`. See §6.3.

### 7.2 `VideoGenerationRequest` — target shape

```python
class VideoGenerationRequest(OpenAIBaseModel):
    # OpenAI-standard
    prompt: str
    model: Optional[str] = None
    input_reference: Optional[Union[str, UploadFile]] = None
    n: int = Field(default=1, ge=1, le=4)
    size: Optional[str] = Field(default="auto", pattern=r"^(\d+x\d+|auto)$")
    seconds: float = Field(default=2.0, ge=1.0, le=16.0)
    fps: int = Field(default=24, ge=8, le=60)
    output_format: Literal["mp4", "avi", "auto"] = "auto"
    seed: Optional[int] = None

    # TRT-LLM extensions (top-level, "well-known")
    num_inference_steps: Optional[int] = Field(default=None, ge=1, le=200)
    guidance_scale: Optional[float] = Field(default=None, ge=0.0, le=20.0)
    negative_prompt: Optional[str] = None

    # Model-specific overflow
    extra_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Model-specific parameters forwarded to the underlying pipeline. "
            "See per-model docs for accepted keys. Unknown keys are rejected."
        ),
    )

    # Removed: `guidance_rescale`. Moves into extra_params.
```

### 7.3 New `extra_params` semantics on HTTP

- Maps 1:1 to Python `VisualGenParams.extra_params`. No translation.
- Keys must match the loaded pipeline's `extra_param_specs.keys()`.
  Unknown keys → 400 from the executor's existing validation message.
- Values are validated by the executor (`type` and `range` checks
  from `ExtraParamSchema`). Type mismatch / out-of-range → 400.
- HTTP layer does not validate `extra_params` keys against any
  registry. The keys are model-specific; the executor knows.
- For LTX2 clients, this is where `guidance_rescale`, `stg_scale`,
  `modality_scale`, etc. live. For Wan 2.2 A14B clients, this is
  where `guidance_scale_2`, `boundary_ratio` live. For Wan 2.1
  clients, `extra_params` must be omitted or empty.

### 7.4 Settling the `seed` semantics

This design picks **HTTP `seed: Optional[int] = None`, conversion
maps `None` to Python `seed=42`** — the current behavior of video,
extended to image (fixes §5.1). Rationale:

- The Python default of `42` is documented and intentional —
  reproducibility is opt-out, not opt-in. Changing it to `None`
  would break callers.
- OpenAI's image API documents `seed` as optional integer; clients
  expect "no seed" to be valid input.
- A loud "did you mean to send `seed=None`?" warning isn't useful —
  most clients legitimately omit it.

The fix is **documentation + tests**: state in the schema that
`None` is normalized to `42`, and write a regression test that
omitting `seed` produces deterministic output. Open Question
§11.2 asks whether to flip Python to `None` instead.

### 7.5 What stays the same

- `size: "WxH"` string, validated by regex (already in place).
- `n` ranges (image: 1..10; video: 1..4).
- `input_reference` as base64 / `UploadFile` (no server-local paths).
- `seconds` × `fps` derivation for `num_frames`.
- `fps: int` (HTTP) → `frame_rate: float` (Python), implicit promotion.
- All "kept" fields are unchanged in shape and constraint.

---

## 8. Conversion layer

`tensorrt_llm/serve/visual_gen_utils.py::parse_visual_gen_params`
shrinks because most fields are now 1:1 instead of overlay-with-special-
cases. The function still starts from `generator.default_params` to
preserve the "None means model default" semantics, then overlays
explicit fields, including `extra_params` as a `dict.update` not a
field-by-field copy.

### 8.1 Target shape

```python
def parse_visual_gen_params(
    request: ImageGenerationRequest | VideoGenerationRequest,
    id: str,
    generator: VisualGen,
    media_storage_path: Optional[str] = None,
) -> VisualGenParams:
    params = generator.default_params

    # Universal overlays (both image and video)
    if request.size is not None and request.size != "auto":
        params.width, params.height = map(int, request.size.split("x"))
    if request.negative_prompt is not None:
        params.negative_prompt = request.negative_prompt
    if request.num_inference_steps is not None:
        params.num_inference_steps = request.num_inference_steps
    if request.guidance_scale is not None:
        params.guidance_scale = request.guidance_scale
    if request.n is not None:
        params.num_images_per_prompt = request.n
    if request.seed is not None:
        params.seed = int(request.seed)

    # Video-only overlays
    if isinstance(request, VideoGenerationRequest):
        params.frame_rate = request.fps
        params.num_frames = int(request.seconds * request.fps)
        if request.input_reference is not None:
            params.image = _materialize_reference(
                request.input_reference, id, media_storage_path,
            )

    # Model-specific overflow — dict merge, executor validates contents
    if request.extra_params:
        if params.extra_params is None:
            params.extra_params = {}
        params.extra_params.update(request.extra_params)

    # Normalize empty dict → None (matches VisualGenParams convention)
    if not params.extra_params:
        params.extra_params = None

    return params
```

(`_materialize_reference` is a small helper holding today's
base64-decode-and-write-to-disk logic from `visual_gen_utils.py:60-70`.)

### 8.2 What changes vs today

- Image branch now reads `request.seed` (fixes §5.1).
- `quality` branch removed (drops §5.4).
- `guidance_rescale` special-case removed (drops §5.3); arrives via
  `extra_params` instead.
- `extra_params` is merged from the request body (new).
- Image and video branches collapse into a single common path plus
  a small video-only block — fewer `isinstance(...)` checks, fewer
  surfaces for drift.

---

## 9. Validation strategy

Three layers, in order:

1. **HTTP boundary** — Pydantic on the request model.
   - `extra="forbid"` rejects unknown top-level keys → 422.
   - `Field(ge=..., le=...)` rejects out-of-range numeric scalars → 422.
   - `pattern=...` on `size` rejects malformed strings → 422.
   - Type errors (`seed="42"` instead of `seed=42`) → 422.
2. **Conversion** — `parse_visual_gen_params`.
   - Limited to translation, not validation. Only domain-specific
     conversion errors are raised here (e.g.
     `input_reference` without `media_storage_path` →
     `ValueError` → 400 via the endpoint's existing
     `create_error_response`).
3. **Executor** — `DiffusionExecutor._validate_request`
   (`tensorrt_llm/_torch/visual_gen/executor.py:243-311`).
   - Unknown `extra_params` keys → `ValueError`.
   - `extra_params` type mismatches → `ValueError`.
   - `extra_params` out-of-range → `ValueError`.
   - Universal fields set but not declared by pipeline → `ValueError`.

The endpoint handlers (`openai_video_routes.py:101-103, 229-231`;
parallel pattern in `openai_server.py` image endpoint) already catch
`ValueError` and surface it as 400 via `create_error_response`. The
design reuses this — no new error infra required.

**Status codes:**

| Failure mode | Layer | Status |
| --- | --- | --- |
| Unknown top-level field | Pydantic | 422 |
| Out-of-range scalar / bad `size` string | Pydantic | 422 |
| Unknown `extra_params` key for this model | Executor | 400 |
| `extra_params` type mismatch | Executor | 400 |
| `extra_params` out-of-range | Executor | 400 |
| Conversion failure (e.g. missing `media_storage_path`) | Conversion | 400 |

The 422 / 400 split is intentional and matches FastAPI's defaults:
422 is "request body could not be parsed", 400 is "request was
parsed but is semantically invalid for the loaded model".

---

## 10. Migration plan

Pre-GA, direct edits. No `validation_alias`, no deprecation cycle.
Coordinated PR (or short series) touching:

| File | Change |
| --- | --- |
| `tensorrt_llm/serve/openai_protocol.py` | Edit `ImageGenerationRequest` and `VideoGenerationRequest` per §7. Drop `quality`, `style`, `guidance_rescale`. Add `extra_params`. |
| `tensorrt_llm/serve/visual_gen_utils.py` | Rewrite `parse_visual_gen_params` per §8. |
| `tensorrt_llm/serve/openai_video_routes.py` | Verify `ValueError → 400` path covers `_validate_request` errors. Likely no code change. |
| `tensorrt_llm/serve/openai_server.py` | Verify same for the image endpoint. |
| `tests/integration/defs/serve/` | Add HTTP-level tests for image + video generation: `extra_params` valid keys, unknown keys, out-of-range, missing `seed` → reproducible, `extra="forbid"` rejection. This is new territory — there are no HTTP-level tests today (`grep -r 'ImageGenerationRequest' tests/` returns nothing under the integration test tree). |
| `docs/source/features/visual_gen.md` (or equivalent) | Document the per-model `extra_params` accepted keys. Defer to a follow-up doc PR if the doc structure isn't settled. |

### 10.1 Ordering

1. Land schema + conversion + executor pass-through together in one
   PR. The HTTP schema and the conversion are coupled; splitting them
   leaves the tree in a state where `extra="forbid"` rejects the
   field the next PR adds.
2. Add HTTP integration tests in the same PR or immediately after.
   Without tests, regressions across the three layers (HTTP →
   conversion → executor) won't surface.
3. Update docs in a follow-up; not blocking.

### 10.2 What this design intentionally does not specify

- The exact text of error messages (executor messages are reused
  verbatim).
- Whether the impl PR should be one PR or split (engineer's call).
- Whether the `seed` Python default flips to `None` (Open Question
  §11.2).
- Whether `quality` removal counts as a breaking change for any
  out-of-tree client (none known; flag in the PR description).

---

## 11. Open Questions

### 11.1 Fix the four §5 bugs in this design's PR, or separately?

The image `seed` drop (§5.1), the per-model `guidance_rescale`
invalidity (§5.3), and the `quality="hd"` semantic (§5.4) are all
resolved as side-effects of the §7 redesign. The `seed` default
mismatch (§5.2) requires a deliberate choice (kept current behavior
in §7.4). The §5.5 docstring nit is a one-line edit.

**Tentative answer:** all five resolve in the redesign PR. If the
PR gets too big, peel §5.5 off (zero-risk one-line nit).

### 11.2 Should Python `VisualGenParams.seed` flip to `Optional[int] = None`?

Current: `seed: int = 42` (Python), `seed: Optional[int] = None`
(HTTP). The mismatch is small but persistent.

**Option A** (this design's pick): keep Python at `42`, document HTTP
`None → 42` in the schema.

**Option B**: flip Python to `None`, route `None` to a per-pipeline
default. Aligns the two sides but breaks every Python caller that
relies on the documented `42` default.

Option A wins on stability; Option B is the cleaner long-term shape.
Flagging for the design owner.

### 11.3 Heterogeneous batch params (`n > 1` with per-sample overrides)

Python `generate_async` explicitly does not accept
`List[VisualGenParams]` (`visual_gen.py:852-855`). HTTP `n > 1`
multiplexes by repeating the same `params` N times. There's no
shape for "give me 4 samples with 4 different seeds".

Out of scope here. When/if Python gains per-item support, the HTTP
side likely grows a `seeds: Optional[List[int]] = None` or similar
sibling. Not designing that today.

### 11.4 Video inpainting

`VisualGenParams.mask` and `image_cond_strength` exist Python-side
and are wired through I2V pipelines. HTTP has no path. Either
extend `VideoGenerationRequest` with `mask` + `image_cond_strength`,
or leave video inpainting Python-only.

Out of scope here; flagged for a follow-up design when the I2V HTTP
surface is decided.

### 11.5 `max_sequence_length` exposure

Genuinely absent on HTTP today, present on Python. It controls how
many tokens the text encoder accepts before truncation. Whether to
expose:

- **Yes (top-level):** safe; bounded; documented as text-encoder
  truncation; small additional schema surface.
- **No (extra_params per model):** keeps it Python-only for now.

**Tentative:** add as a top-level optional integer in both image and
video schemas, with `Field(ge=1, le=4096)` or similar. Cheap to add,
cheap to remove if it never gets used. Open for input.

### 11.6 Validation message format for `extra_params` errors

The executor's current message
(`"Unknown extra_params [...] for {pipeline_name}. Supported: [...]"`)
is descriptive but server-flavor. Should the endpoint reshape this
into a JSON error response with structured fields (`{"code":
"unknown_extra_param", "key": "...", "supported": [...]}`)? Pure
API ergonomics question; flag for the impl PR.

---

## 12. Iteration Tracker

| # | Date | Codex focus | Threads | Resolved | Open | Deferred |
| --- | --- | --- | --- | --- | --- | --- |
| _placeholder_ | | | | | | |

(Iteration rows appended as Codex adversarial-review passes complete.)

---
