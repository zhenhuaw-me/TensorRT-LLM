# VisualGen Serve↔Python Request-Param Parity

> **Status**: Draft — converged after 5 Codex adversarial-review iterations (2026-05-26)
> **Date**: 2026-05-25 (initial draft); converged 2026-05-26
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
should agree on what "no seed" means. §7.4 settles this as **option
(b): Python flips to `Optional[int] = None`**, HTTP semantics become
"no seed = random," and the conversion only overrides when the
request explicitly sends an integer.

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

### 6.5 Why `forbid` over `ignore+warn` or `allow`

Three options side-by-side:

| Option | Unknown top-level key behavior | Caller experience | Production failure mode |
| --- | --- | --- | --- |
| **`extra="forbid"`** *(this design)* | 422 with structured Pydantic error; hint generator may rewrite the message for known-relocated keys (§9.2) | "field rejected" surfaces immediately in any environment; one-line fix to send through `extra_params` | If upstream OpenAI ships a new field the server hasn't mirrored, well-shaped requests start failing at validation until trtllm-serve catches up. Loud, traceable, easy to triage. |
| **`extra="ignore"` + WARNING log** | Server silently drops unknown fields; logs them at WARNING | Looks like success; the dropped knob has no observable effect on output until the user notices wrong colors / wrong step count / etc. | Warning logs are filtered by most production aggregators by default; "field doesn't work" tickets accumulate at the support tier, not the schema tier. This is vLLM's documented failure mode (Issues #7337, #11153). |
| **`extra="allow"`** | Unknown keys are accepted and stored on the model; conversion ignores them | Same caller experience as `ignore`; even less server-side visibility | Same as `ignore` plus: any downstream code that loops over `request.model_dump()` may forward typo'd keys deeper into the stack. |

Two observations narrow the trade-off:

- **OpenAI image-API schema additions appear infrequent in the
  surface this design mirrors.** This is not a quantified longitudinal
  survey of OpenAI's release history — only an assumption based on
  the current `tensorrt_llm/serve/openai_protocol.py` mirroring what
  the team has chosen to track. If upstream OpenAI velocity is
  higher than assumed, the failure-mode argument for `forbid` weakens
  proportionally. The implementation PR should explicitly note this
  as a watched risk in its description.
- **The dynamic hint generator (§9.2) recovers most of the ignore-
  but-warn DX without the silent-drop downside.** A client that sends
  `guidance_rescale` top-level — or any other key that the loaded
  model accepts under `extra_params` — gets a 422 body that names
  `extra_params` as the right destination, derived from the loaded
  pipeline's `extra_param_specs` at request time, not a bare "extra
  inputs not permitted."

Net: `forbid` is the right default. **It is not literally a one-
character flip to `"ignore"`** — switching the `model_config` value
also requires re-establishing the unknown-field code path that today
runs only because Pydantic raises. A workable compatibility-mode
plan, named here so future maintainers don't underestimate the cost:

1. **Capture raw unknown top-level keys before model parse.** A
   FastAPI dependency or middleware on the visual_gen routes reads
   `await request.json()` and computes `unknown = set(payload.keys())
   - set(KNOWN_TOP_LEVEL_KEYS)` (where `KNOWN_TOP_LEVEL_KEYS` is
   derived from `Request.model_fields`). The dependency stores the
   set on `request.state` and lets the model parse with
   `extra="ignore"`.
2. **Re-emit the §9.1 envelope or a deliberate warning** based on
   `request.state.unknown_top_level_keys` and the §9.2 cache. The
   choice — 422 (parity with `forbid`) vs. WARNING log + 200 (true
   ignore) vs. metric counter — is a policy lever the escape-hatch
   user picks based on the upstream-velocity scenario they're
   handling.
3. **Update the schema-rejection tests** to assert the new shape:
   `unknown_top_level_field` no longer comes from
   `RequestValidationError`; tests instead inspect the warning log
   or the JSON envelope produced by the middleware.
4. **Restore observability** via the chosen policy: structured
   WARNING logs at minimum, ideally a Prometheus counter
   (`visual_gen_unknown_top_level_field_total`) labelled by field
   name. Without this, `ignore` slides back into the silent-drop
   failure mode `forbid` was meant to avoid.

Realistic scope: roughly half a day of work, mostly tests and the
middleware shim. Far cheaper than the silent-drop debugging cost,
but not a config-line change. The doc records this as the explicit
escape hatch — the impl PR can leave it as a comment in
`openai_protocol.py` referencing this section.

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

    # OpenAI-shaped, no-op pass-through (compat only)
    quality: Literal["standard", "hd"] = Field(
        default="standard",
        description=(
            "Accepted for OpenAI-SDK compatibility; ignored by TRT-LLM. "
            "Pass `num_inference_steps` for explicit step control."
        ),
    )

    # TRT-LLM extensions (top-level, "well-known")
    num_inference_steps: Optional[int] = Field(default=None, ge=1, le=200)
    guidance_scale: Optional[float] = Field(default=None, ge=0.0, le=20.0)
    max_sequence_length: Optional[int] = Field(default=None, ge=1, le=4096)
    negative_prompt: Optional[str] = None

    # Model-specific overflow
    extra_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Model-specific parameters forwarded to the underlying pipeline. "
            "See per-model docs for accepted keys. Unknown keys are rejected "
            "by the executor with a hint naming the supported keys for the "
            "loaded model."
        ),
    )

    # Removed: `style` (no callers in tree).
    # Removed: top-level `guidance_rescale` (moves into `extra_params`).
```

**Schema disposition for fields with broken or absent semantics:**

| Field | Adoption evidence | Disposition |
| --- | --- | --- |
| `quality` | `examples/visual_gen/serve/sync_image_gen.py:31,57`; `examples/visual_gen/serve/README.md:274`; `test_image_generation_hd_quality` test | **Keep schema field, drop conversion behavior.** Accepted as a no-op for OpenAI-SDK compatibility. Schema docstring says: "Accepted for OpenAI compatibility; ignored by TRT-LLM. Pass `num_inference_steps` for explicit step control." |
| `style` | No reference in `tests/`, `examples/`, or `docs/` (grep confirmed) | **Drop.** Zero callers; remove cleanly. |
| `guidance_rescale` (top-level) | No top-level HTTP usage in `tests/`, `examples/`, or `docs/`; only Python-side `extra_params` dict access in `examples/visual_gen/visual_gen_ltx2.py:418` | **Drop top-level; available via `extra_params` for LTX2 callers.** Hint-error from §6.3 catches the legacy top-level spelling. |

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
    max_sequence_length: Optional[int] = Field(default=None, ge=1, le=4096)
    negative_prompt: Optional[str] = None

    # Model-specific overflow
    extra_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Model-specific parameters forwarded to the underlying pipeline. "
            "See per-model docs for accepted keys."
        ),
    )

    # Removed: top-level `guidance_rescale` (moves into `extra_params`).
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

This design picks **HTTP `seed: Optional[int] = None` means "no seed =
random."** OpenAI's image-API documents `seed` as optional, and clients
have strong intuition that omitted seed implies non-deterministic
generation; mapping `None` to a documented `42` default would be a UX
trap that documentation alone cannot fix. The implementation has two
parts:

- **HTTP:** `seed: Optional[int] = None`. When the client omits it,
  the conversion does NOT set `params.seed`; the engine generates a
  fresh seed per request.
- **Python:** `VisualGenParams.seed: Optional[int] = None` (flipped
  from `int = 42`). The engine treats `None` as "fresh random seed
  per request"; explicit integers reproduce.

This **closes Open Question §11.2**. It is a deliberate Python-side
break: every Python caller that today relies on the documented `42`
default for reproducibility must now pass `seed=42` explicitly. The
change is small and discoverable (the default flip is in
`VisualGenParams.seed`'s `Field(default=None, description="...")`),
and the affected test files (`test_visual_gen_params.py`,
`test_visual_gen.py`) are the natural place to add the regression.

The §5.1 bug (image `seed` silently dropped at the HTTP layer) is
fixed by the universal seed overlay in §8.

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
    if request.max_sequence_length is not None:
        params.max_sequence_length = request.max_sequence_length
    if request.n is not None:
        params.num_images_per_prompt = request.n
    if request.seed is not None:
        params.seed = int(request.seed)
    # `quality` is intentionally not consumed (no-op pass-through per §7.1).

    # Video-only overlays
    if isinstance(request, VideoGenerationRequest):
        params.frame_rate = request.fps
        params.num_frames = int(request.seconds * request.fps)
        if request.input_reference is not None:
            params.image = _materialize_reference(
                request.input_reference, id, media_storage_path,
            )

    # Model-specific overflow — see §8.3 for normative merge semantics.
    _merge_extra_params(
        params, request.extra_params, generator.extra_param_specs,
    )

    return params
```

(`_materialize_reference` and `_merge_extra_params` are small helpers
defined below. `_materialize_reference` holds today's base64-decode-
and-write-to-disk logic from `visual_gen_utils.py:60-70`.)

### 8.2 What changes vs today

- Image branch now reads `request.seed` (fixes §5.1).
- `quality` branch removed: the field stays on the schema (per §7.1
  adoption check), but the conversion no longer maps it to anything.
  Closes §5.4.
- `guidance_rescale` special-case removed (drops §5.3); arrives via
  `extra_params` instead, routed through `_merge_extra_params`.
- `extra_params` is merged from the request body via the normative
  rule in §8.3 (new).
- `max_sequence_length` is now mapped (was previously absent — see
  the schema addition in §7.1).
- Image and video branches collapse into a single common path plus
  a small video-only block — fewer `isinstance(...)` checks, fewer
  surfaces for drift.

### 8.3 Normative merge semantics for `extra_params`

The conversion starts from `generator.default_params`, which already
includes a populated `extra_params` dict seeded from the pipeline's
`extra_param_specs[*].default` values. The request body may contain
its own `extra_params` dict. Merge must specify three cases
unambiguously and **must not silently swallow typos** even when the
client sends `null`:

| Client behavior | Effect on `params.extra_params[key]` | Notes |
| --- | --- | --- |
| Omits `key` | Retains the pipeline default from `generator.default_params` | — |
| Sends `key: <non-null value>` where `key` is in `extra_param_specs` | Overrides the default with the client value | Executor's existing type/range checks apply (`executor.py:282-305`) |
| Sends `key: null` where `key` is in `extra_param_specs` | Use pipeline default (key stripped before executor) | Lets clients explicitly opt back into the default |
| Sends `key: <any value>` where `key` is NOT in `extra_param_specs` | **Pass through unchanged** (including `null`) | Executor's strict-key validation (`executor.py:259-266`) rejects with `unknown_extra_param` |

The "null on a known key = use default" rule is **schema-aware**:
the helper consults the loaded pipeline's `extra_param_specs` before
deciding whether `null` is a sentinel. Unknown keys with `null`
values are *not* dropped; they reach the executor and trigger the
same `unknown_extra_param` error as any other typo. A schema-blind
"strip every null" rule would create a silent-typo loophole — a
client sending `{"extra_params": {"stg_sclae": null}}` (typo)
would otherwise see a 200 and silently retained defaults.

For known keys whose spec genuinely permits `None` as a value (e.g.
list-valued knobs like `stg_blocks` where "disabled" and "use
default" are both meaningful), the spec author currently must rely
on the `default=None` convention — there is no way through this HTTP
shape to express "explicit None override" distinct from "use
default." This is an intentional limitation; an explicit
disable-vs-default flag would expand the HTTP surface beyond what
diffusion clients ask for today. Flagged as Open Question §11.7.

The helper:

```python
def _merge_extra_params(
    params: VisualGenParams,
    request_extras: Optional[Dict[str, Any]],
    extra_param_specs: Dict[str, "ExtraParamSchema"],
) -> None:
    """Shallow-merge request extras into the params dict.

    Model defaults are already populated in params.extra_params (from
    generator.default_params). Per-key behavior:

    - Known key + non-null value  -> override.
    - Known key + null value      -> strip override; default remains.
    - Unknown key (any value)     -> pass through; executor will reject.
    """
    if not request_extras:
        return

    if params.extra_params is None:
        params.extra_params = {}

    for key, value in request_extras.items():
        if key in extra_param_specs and value is None:
            # Known key + null sentinel: "use pipeline default."
            # The default was already populated into params.extra_params
            # by generator.default_params; leave it untouched, even when
            # the default itself is None. Distinguishing "key absent" vs
            # "key present with value None" matters to some pipelines
            # (e.g. stg_blocks), and the pre-seeded default already
            # encodes the right state. Do not pop.
            continue
        # All other cases (known + non-null, unknown + anything):
        # let it through. Executor validates.
        params.extra_params[key] = value

    if not params.extra_params:
        params.extra_params = None
```

The executor's existing strict validation
(`_torch/visual_gen/executor.py:243-311`) catches unknown keys *and*
type/range violations against the per-pipeline schema. The merge
helper does not validate; it only normalizes the `null` sentinel for
known keys.

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
design extends this with a structured JSON envelope and a hint
catalogue (§9.1, §9.2) so the HTTP response body is a stable contract
rather than the executor's raw Python string.

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

### 9.1 JSON error envelope

All HTTP error bodies from the visual_gen endpoints share one shape,
independent of which layer raised the error. This decouples the HTTP
API contract from the executor's internal Python `ValueError` text.

```json
{
  "error": {
    "code": "<stable_error_code>",
    "message": "<human-readable summary>",
    "param": "<json-path to the offending field, when applicable>",
    "hint": "<optional remediation; see §9.2>",
    "details": { /* layer-specific structured fields */ }
  }
}
```

Stable error codes (extensible):

| `code` | Layer | When | `param` |
| --- | --- | --- | --- |
| `unknown_top_level_field` | Pydantic | Unknown key at the request root | `<field name>` |
| `field_validation_error` | Pydantic | Range/type/regex failure on a known field | `<field name>` |
| `unknown_extra_param` | Executor | Key not in `extra_param_specs` for the loaded model | `extra_params.<key>` |
| `extra_param_type_mismatch` | Executor | Value type doesn't match `ExtraParamSchema.type` | `extra_params.<key>` |
| `extra_param_out_of_range` | Executor | Value outside `ExtraParamSchema.range` | `extra_params.<key>` |
| `unsupported_universal_field` | Executor | Universal field (e.g. `num_frames`) set but pipeline doesn't declare it | `<field name>` |
| `conversion_error` | Conversion | e.g. missing `media_storage_path` | varies |

**Ownership boundary (HTTP codes live in `serve/`, not in the
engine).** Iter 2 review correctly flagged that pushing HTTP code
strings like `"unknown_extra_param"` into the executor blurs layering
and risks breaking Python callers that catch `ValueError`. The
revised plan:

- **Engine:** a transport-neutral validation error in
  `tensorrt_llm/_torch/visual_gen/executor.py`:

  ```python
  class VisualGenValidationError(ValueError):
      """Raised by DiffusionExecutor._validate_request for parameter
      violations. Subclasses ValueError so existing Python call
      sites that ``except ValueError`` continue to work.
      """
      def __init__(
          self,
          reason: Literal[
              "unknown_extra_param",
              "extra_param_type_mismatch",
              "extra_param_out_of_range",
              "unsupported_universal_field",
          ],
          param: str,           # field name or extra_params.<key>
          message: str,         # human-readable; today's text
          details: Optional[Dict[str, Any]] = None,
      ):
          super().__init__(message)
          self.reason = reason
          self.param = param
          self.details = details or {}
  ```

- **Serve layer:** the endpoint handlers catch
  `VisualGenValidationError` and translate `reason` → HTTP `code`,
  populate `param` and `details`, build the §9.1 envelope. The
  engine never sees the HTTP `code` string. The
  `reason → code` mapping is a small dict in
  `tensorrt_llm/serve/`.

- **Backwards compat:** because `VisualGenValidationError` subclasses
  `ValueError`, every existing `except ValueError` (Python callers,
  tests, endpoint exception handlers) keeps working unchanged. The
  serve layer adds the more specific catch as an inner `except`
  clause inside the existing `except ValueError` block.

The previous `ValueError`-as-API-contract pattern is preserved as the
fallback (for any executor code path that hasn't been migrated to
raise `VisualGenValidationError`), but new validation cases go
through the typed exception.

### 9.2 Dynamic hint generation for relocated top-level keys

The hint is **derived dynamically** from the loaded pipeline's
`extra_param_specs`, not from a hand-maintained static catalogue.
This keeps the discoverability story honest as new models add new
`extra_param_specs` keys — no second source of truth to drift.

```python
# Cached on the server at generator load / model swap time.
# Set to None when no generator is loaded (startup, mid-swap).
_extra_param_keys_cache: Optional[frozenset[str]] = None

def _refresh_extra_param_keys_cache(generator: Optional[VisualGen]) -> None:
    """Called by the server on startup, on model swap, and after any
    generator reload. Idempotent.
    """
    global _extra_param_keys_cache
    if generator is None:
        _extra_param_keys_cache = None
    else:
        _extra_param_keys_cache = frozenset(generator.extra_param_specs.keys())

def _hint_for_unknown_top_level_key(key: str) -> Optional[str]:
    """Look up the offending unknown top-level key against:
    (a) the loaded pipeline's extra_param_specs (cached at load time —
        not recomputed per request, cannot raise on attacker-driven
        bad input), and
    (b) a small frozen alias map for keys removed from the schema
        (historical typo defenses; today: empty).
    """
    if _extra_param_keys_cache is not None and key in _extra_param_keys_cache:
        return (
            f"'{key}' is not a top-level field for this model. Pass it "
            f"via extra_params: {{'extra_params': {{'{key}': ...}}}}."
        )
    # Reserved for historical aliases (currently empty). Add entries
    # only when a top-level field is removed from the schema and we
    # want a corrective message for clients still sending it.
    historical_aliases: dict[str, str] = {}
    return historical_aliases.get(key)
```

When the Pydantic `RequestValidationError` indicates an
`unknown_top_level_field`, the FastAPI exception handler calls
`_hint_for_unknown_top_level_key(field_name)` and sets `error.hint`
to the returned string when non-`None`. Two robustness properties:

- **Cached**, not recomputed per request — the validation-error
  path runs on attacker-controllable input, so a per-error lazy
  `generator.extra_param_specs` call would be a tiny DoS surface.
  The cache is a `frozenset` populated once at load.
- **No-generator-loaded fallback** — during server startup or a
  model swap, the cache is `None` and the helper returns `None`.
  The error path falls back to a plain "field not permitted" 422
  rather than crashing with `AttributeError`.

If the key is unknown to both the cache *and* the historical-alias
map, the error stays a plain 422 — the typo case continues to fail
loud.

The single key that has Python usage in tests/examples
(`guidance_rescale`, used in `examples/visual_gen/visual_gen_ltx2.py`
through Python's `extra_params` dict) is automatically covered for
any model that declares it in `extra_param_specs` (today: LTX2). No
manual entry required. Test coverage in §10.

---

## 10. Migration plan

Pre-GA, direct edits. No `validation_alias`, no deprecation cycle.
Coordinated PR (or short series) touching:

| File | Change |
| --- | --- |
| `tensorrt_llm/serve/openai_protocol.py` | Edit `ImageGenerationRequest` and `VideoGenerationRequest` per §7. Drop `style` and top-level `guidance_rescale`. Add `extra_params` and `max_sequence_length`. Keep `quality` as no-op. |
| `tensorrt_llm/serve/visual_gen_utils.py` | Rewrite `parse_visual_gen_params` per §8 (including `_merge_extra_params` per §8.3). |
| `tensorrt_llm/serve/openai_video_routes.py` | Wire the JSON error envelope (§9.1) into `create_error_response`. |
| `tensorrt_llm/serve/openai_server.py` | Same for the image endpoint. Add the hint-catalogue exception handler for unknown-top-level Pydantic errors (§9.2). |
| `tensorrt_llm/visual_gen/params.py` | Flip `seed: int = 42` → `seed: Optional[int] = None` (§7.4). |
| `tensorrt_llm/_torch/visual_gen/executor.py` | Introduce `VisualGenValidationError(ValueError)` carrying transport-neutral `reason`/`param`/`details` (§9.1); switch `_validate_request` to raise it instead of plain `ValueError` for the four validation cases. Resolve `params.seed is None` at this layer (§10.4) by drawing once and broadcasting to ranks. Fix the docstring/code inconsistency from §5.5. |
| `tensorrt_llm/serve/` (new small file or co-located with `openai_server.py`) | The `reason → HTTP code` mapping (`unknown_extra_param`, `extra_param_type_mismatch`, etc.) lives here, not in the engine. |
| `tests/unittest/_torch/visual_gen/test_trtllm_serve_endpoints.py` | Extend (§10.1). The file already mocks `VisualGen` via `MockVisualGen` and uses FastAPI's `TestClient` — no GPU required. |
| `tests/unittest/_torch/visual_gen/test_visual_gen_params.py` | Add a regression for `seed=None` random generation; update any existing case that relied on `42`. |
| `examples/visual_gen/serve/README.md` | Update parameter list: remove `style`; note `quality` is a no-op pass-through; document `extra_params` with per-model accepted keys. |
| `docs/source/models/visual-generation.md` (and related) | Document the per-model `extra_params` accepted keys. Defer to a follow-up doc PR if the doc structure isn't settled. |

### 10.1 Test plan (HTTP unit + integration split)

HTTP-level tests for the visual_gen endpoints already exist
(`tests/unittest/_torch/visual_gen/test_trtllm_serve_endpoints.py`,
~1134 lines) and run as **unit tests** thanks to a `MockVisualGen`
class plus FastAPI's `TestClient`. No GPU, no real generation, no
model weights required. This is exactly the schema/adapter test
surface the redesign needs.

The redesign extends the existing file rather than creating new
infrastructure. Two test layers:

**Unit tests** — split across three files so the assertion target is
obvious and capture-point issues are avoided:

- **Schema-only tests** (`test_trtllm_serve_endpoints.py`,
  `TestClient`-driven, no `params` capture needed): unknown top-level
  fields → 422 with `code=unknown_top_level_field`; mismatched
  types/ranges → 422 with `code=field_validation_error`; hint
  generator (§9.2) fires for keys in the loaded mock's
  `extra_param_specs`. Schema acceptance for every preserved field
  on both image and video request models.
- **Conversion tests** (a new `test_visual_gen_utils.py` or extension
  of an existing file): call `parse_visual_gen_params(...)` and
  `_merge_extra_params(...)` directly with constructed Pydantic
  request objects and a stub `VisualGen` (just the
  `default_params` and `extra_param_specs` properties). Assert the
  produced `VisualGenParams` field-by-field. Covers: the three rows
  of §8.3 (omit / override / null on known key); null on unknown key
  passes through; `quality="hd"` is a no-op; HTTP `seed=None` →
  `params.seed=None`; HTTP `seed=123` → `params.seed=123`. This
  avoids the race risk of `MockVisualGen.last_params` because we
  never enqueue an async request.
- **End-to-end mocked tests** (`test_trtllm_serve_endpoints.py`): use
  only the **sync** endpoints (`/v1/images/generations` and
  `/v1/videos/generations`) for `last_params` assertions, and reset
  `MockVisualGen.last_params = None` in a per-test fixture (the
  existing fixtures already do this; document explicitly). Do *not*
  use the async `/v1/videos` job route for merge-semantics tests —
  the deep-copy on enqueue plus the background task makes the
  capture point ambiguous. Async-route tests assert only that the
  job was accepted (202) and that the response shape is valid.

This split makes the conversion tests assertion-precise (direct
function call, no transport machinery), while the schema and e2e
tests cover the FastAPI integration boundary. The plan does not
introduce new test infrastructure beyond a small `test_visual_gen_utils.py`
that imports the conversion helpers directly.

**Integration tests** (GPU + real model weights, opt-in CI lane):

- Existing real-engine tests (`test_wan21_*`, `test_ltx2_*`) continue
  to assert the engine path. Add at most one HTTP-driven smoke test
  per supported model that uses `test_trtllm_serve_endpoints.py`'s
  infrastructure but with the real `VisualGen`. These are slow and
  should not gate the unit-test lane.

The unit layer is the verification surface for everything in §7-§9.
The integration layer guards the executor's per-pipeline `extra_param_specs`
contract.

### 10.2 Ordering

1. Land schema + conversion + executor pass-through together with
   unit-test extensions in one PR. The HTTP schema, conversion, and
   executor exception type are coupled; splitting them leaves the
   tree in a state where unit tests would fail mid-stack.
2. Update `params.py` (`seed` flip) in the same PR. The actual
   migration impact, measured by running:

   ```
   grep -rnE 'seed=42|seed: int = 42|default=42|VisualGenParams\(' \
     tensorrt_llm/ tests/ examples/ docs/
   ```

   produces a bounded set of touch points on this branch:
   - `tensorrt_llm/visual_gen/params.py:49` — the canonical default (the flip itself).
   - `tests/unittest/_torch/visual_gen/test_visual_gen_params.py:45` — `VisualGenParams()` with no kwargs (verifies the default); needs the assertion updated from `seed == 42` to `seed is None`.
   - `tests/unittest/_torch/visual_gen/test_trtllm_serve_endpoints.py:150` — `MockVisualGen.default_params` returns `VisualGenParams()`; the mock should be updated to explicitly populate `seed` to whatever the test wants observed.
   - `tests/integration/defs/examples/test_visual_gen.py:810, 899` — construct `VisualGenParams(seed=...)` with explicit values; unaffected by the flip.
   - `examples/visual_gen/visual_gen_{wan_t2v,wan_i2v,flux,ltx2}.py` — define argparse `--seed default=42` and pass the explicit value into `VisualGenParams(...)`; unaffected by the flip.
   - Per-model pipeline internal `seed: int = 42` (e.g.
     `pipeline_flux.py:268`, `pipeline_ltx2.py:1381`, `pipeline_wan.py:392`)
     are independent internal defaults on the pipeline's own `infer()`
     kwargs, not the `VisualGenParams.seed` default. **The executor
     resolves `None` to a concrete integer before pipeline dispatch
     (§10.4),** so each pipeline's `infer()` becomes `seed: int` with
     no default — pipelines do not draw their own randomness. Tracked
     in §10.4 below.

3. Update README/docs in a follow-up PR; not blocking.

### 10.3 Backwards-compat audit before introducing `VisualGenValidationError`

Because `VisualGenValidationError` subclasses `ValueError`, *normal*
`except ValueError` blocks continue to work. Two patterns can still
break and must be checked before merge:

- **Strict identity checks** — `type(e) is ValueError` (rare but
  possible). A repo-wide grep before the PR lands:
  `grep -rnE 'type\([^)]*\) is ValueError|type\([^)]*\) == ValueError' tensorrt_llm/ tests/`.
  Expected result: zero hits in the visual_gen + serve paths. If a
  hit exists, switch to `isinstance(...)`.
- **`pytest.raises(ValueError, match="exact substring of old message")`** —
  if existing tests assert message text, the new structured exception's
  `__str__` may not match. Grep:
  `grep -rnE 'pytest.raises\(ValueError.*match=' tests/unittest/_torch/visual_gen/`.
  Affected assertions update to either drop the `match=` or anchor on
  the new structured fields.

Both audits are mechanical, expected to be empty or near-empty, and
land in the same PR as the exception introduction.

### 10.4 Seed resolution at the executor, not the pipeline

The `seed` flip introduces an engine-side requirement: when
`params.seed is None`, the engine must draw a fresh random seed
*once per request*. Iter 3 review correctly flagged that drawing the
seed inside each pipeline's `infer()` is wrong for distributed
execution — `VisualGen.args.parallel_config` can split work across
ranks (`cfg_size`, `ulysses_size`), and a per-rank `torch.seed()` call
would give the same logical request *different* random seeds on
different ranks, producing non-deterministic and rank-divergent
output for what the user perceives as one request.

The correct seam is the **request coordinator** (the executor or its
caller), which sees the request exactly once and runs on rank 0 in
multi-rank topologies. Concretely, in
`tensorrt_llm/_torch/visual_gen/executor.py`, immediately after
`_merge_defaults` and `_validate_request` (lines ~313-316), add:

```python
def _resolve_seed(self, req: DiffusionRequest) -> None:
    """Materialize a concrete seed for the request when the client
    omitted one. Drawn once on the coordinator rank and stored back
    on the params so downstream broadcasts/pipeline stages see the
    same value.
    """
    if req.params.seed is None:
        # Use a single per-process generator to avoid coupling to
        # PyTorch's global state, which a model load may have re-seeded.
        req.params.seed = secrets.randbits(63)
```

Pipelines do **not** add their own `seed is None` handling. The
existing per-pipeline `seed: int = 42` signatures (`pipeline_flux.py`,
`pipeline_flux2.py`, `pipeline_ltx2.py`, `pipeline_ltx2_two_stages.py`,
`pipeline_wan.py`, `pipeline_wan_i2v.py`) become `seed: int` (no
default; the executor always provides one). This:

- guarantees the same seed reaches every rank for a given request,
  because broadcast/scatter happens *after* the resolution step;
- lets each pipeline keep its existing per-device generator and
  seed-broadcast convention unchanged;
- removes the temptation to draw the seed at multiple layers (one
  source of randomness per request, surface fully owned by the
  executor).

If a particular pipeline currently has bespoke seed handling that
diverges from "use this integer," the impl PR is the place to
document why. The audit list (six pipeline files above) is bounded
and unambiguous: read each `infer()` signature; remove the `= 42`
default; verify no `if seed == 42` / `if not seed` branches remain.

### 10.3 What this design intentionally does not specify

- Whether the impl PR should be one PR or split (engineer's call;
  default is one).
- Whether `quality` no-op should grow a deprecation log line. Not
  required by this design; the schema docstring is enough.

---

## 11. Open Questions

### 11.1 Fix the five §5 bugs in this design's PR, or separately?

All five bugs resolve as side-effects of the redesign:

- §5.1 (image `seed` silently dropped) — fixed by the universal seed
  overlay in §8.1.
- §5.2 (`seed` default semantics mismatch) — fixed by the Python flip
  to `Optional[int] = None` in §7.4, which makes "no seed = random"
  the consistent semantics on both sides.
- §5.3 (`guidance_rescale` per-model invalid as top-level) — fixed
  by moving it into `extra_params` (§7.1, §7.2) plus the hint catalogue
  for the legacy spelling (§9.2).
- §5.4 (`quality="hd"` overrides model default) — fixed by keeping
  the field as no-op pass-through and removing the conversion mapping
  (§7.1, §8.2).
- §5.5 (executor docstring/code inconsistency) — one-line comment
  edit listed in the §10 migration plan.

**Answer:** all five resolve in the redesign PR. If the PR gets too
big, peel §5.5 off (zero-risk one-line nit).

### 11.2 ~~Should Python `VisualGenParams.seed` flip to `Optional[int] = None`?~~ *Resolved in §7.4.*

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

### 11.5 ~~`max_sequence_length` exposure~~ *Resolved: added top-level (§7.1, §7.2, §8.1).*

### 11.6 ~~Validation message format for `extra_params` errors~~ *Resolved: §9.1 defines the JSON error envelope.*

### 11.7 Explicit `None` override for `extra_params` keys whose "disabled" state is distinct from "use default"

The §8.3 merge rule treats `null` on a known `extra_params` key as
"use pipeline default." For most keys this is correct, but a small
category — list-valued knobs like `stg_blocks` whose `None` is a
*meaningful disable* rather than just absence — can't currently
distinguish "client wants the disabled state explicitly" from
"client doesn't care, use default." Both routes produce
`params.extra_params[key] = <default>` and the spec's default may
itself be `None`.

This is an intentional limitation; an explicit disable-vs-default
distinction would expand the HTTP shape beyond what diffusion
clients ask for today. If a real use case appears, two options:
(a) add a sentinel string like `"__disabled__"` for that key; or
(b) document per-key semantics in the model's docs. **Open for
later input.**

---

## 12. Iteration Tracker

| #  | Date       | Codex focus                                                                                       | Threads | Resolved | Open | Deferred |
|----|------------|---------------------------------------------------------------------------------------------------|---------|----------|------|----------|
| 1  | 2026-05-25 | extra="forbid" vs OpenAI-SDK drift; schema-drop adoption check; seed semantics; error envelope; test plan realism; open-question scoping | 10      | 10       | 0    | 0        |
| 2  | 2026-05-26 | forbid escape-hatch realism; hint catalogue drift vs `extra_param_specs`; null sentinel + silent-typo loophole; seed migration inventory breadth; executor error ownership / layering; mock-test race risk; §11.1 stale phrasing | 7       | 7        | 0    | 0        |
| 3  | 2026-05-26 | FastAPI ordering for compat-mode escape hatch; null+default=None equivalence; §10 stale exception name; hint-lookup cache + no-generator fallback; distributed-seed correctness | 5       | 5        | 0    | 0        |
| 4  | 2026-05-26 | Convergence pass: verified iter-3 resolutions; caught one stale §10.2 reference to pre-§10.4 pipeline-local seed logic | 5       | 5        | 0    | 0        |
| 5  | 2026-05-26 | Final convergence check: verified the §10.2 fix landed and no third-place stale references remain. Codex returned "Convergence reached." | 0       | —        | 0    | 0        |

**Converged on 2026-05-26 after 5 iterations.** Total: 27 substantive
threads raised across the loop, 27 resolved (including the iter-2
re-opens of partially-addressed iter-1 critiques and the iter-3
re-opens of iter-2 critiques). The four Open Questions in §11
(§11.3 heterogeneous batch params, §11.4 video inpainting, §11.7
disable-vs-default sentinel) are intentional carry-outs flagged for
follow-up designs, not unresolved iteration threads.

---
