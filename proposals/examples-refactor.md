# VisualGen Examples, Docs & Testing Refactor

> **Status**: Draft — under discussion
> **Date**: 2026-03-24
> **Related**: [API Refactor](visual-gen-api-refactor-m2.md) ·
> [Perf Sanity Design](visual_gen_perf_sanity_design.md)

---

## Table of Contents

1. [User Requirement Analysis](#1-user-requirement-analysis)
2. [Current State Assessment](#2-current-state-assessment)
3. [Design Principles](#3-design-principles)
4. [Proposed Structure](#4-proposed-structure)
5. [Example Specifications](#5-example-specifications)
6. [Documentation Refactor](#6-documentation-refactor)
7. [CI / Testing Integration](#7-ci--testing-integration)
8. [Migration Plan](#8-migration-plan)
9. [Open Questions](#9-open-questions)

---

## 1. User Requirement Analysis

### User Personas

We identify two primary personas, informed by industry patterns from diffusers,
SGLang, vLLM, and ComfyUI communities.

| Dimension | Persona A — "Getting Started" | Persona B — "Performance-Eager" |
|-----------|-------------------------------|----------------------------------|
| **Goal** | Generate a first image or video on a single GPU | Serve a model in production or benchmark throughput/latency |
| **Skill** | Familiar with Python and HuggingFace; new to TRT-LLM | Familiar with TRT-LLM or inference servers; may operate at scale |
| **Setup** | 1 GPU (A100/H100/B200), Docker container or pip install | 1–8 GPUs, likely inside NGC container, potentially multi-node |
| **Expectations** | A self-contained script that works end-to-end in < 5 min. No multi-node, no tuning. Clear output (saved file). | `trtllm-serve` one-liner with tuned YAML configs. Benchmarking across batch sizes and concurrency levels. Quantization and parallelism knobs. |
| **Success metric** | "I ran the script, got a video, and I understand how to change the prompt / model / resolution." | "I can deploy and benchmark this model with a copy-paste command and a known-good config." |
| **Comparable experience** | `diffusers` README (5-line pipeline example), SGLang Diffusion quickstart | TRT-LLM LLM deployment guides + recipe database |

### Industry Best Practices

Surveying how successful ML inference projects organize examples:

| Practice | Source | Implication |
|----------|--------|-------------|
| **Runnable examples are the documentation** | diffusers, PyTorch tutorials, LangChain | Docs should embed or `literalinclude` from real scripts, not maintain parallel code snippets that drift. |
| **Progressive disclosure** | Gradio, FastAPI, Keras | Quickstart → Advanced → Feature deep-dives. Don't overwhelm beginners with parallelism or quantization. |
| **Examples double as regression tests** | HuggingFace transformers, JAX, Lightning | CI runs the same scripts users run. If the example breaks, CI catches it before users hit it. |
| **Config-driven, not script-per-variant** | TRT-LLM LLM recipes, MLflow | One script + many YAML configs > many scripts with hardcoded knobs. Scales to 170+ configs without code duplication. |
| **Per-model README with shared scripts** | TRT-LLM `examples/models/core/` pattern | Model READMEs own hardware requirements and feature notes; execution defers to shared scripts. |
| **Docs as index, not encyclopedia** | Stripe, Vercel, Next.js docs | Feature docs link to examples and API reference; they don't duplicate inline code. |

### User Journey Map

```
Persona A                              Persona B
   │                                      │
   ▼                                      ▼
Quick Start Guide (docs)             Deployment Guide (docs)
   │                                      │
   ▼                                      ▼
quickstart_example.py               trtllm-serve + YAML config
   │                                      │
   ▼                                      ▼
models/<model>.py                   benchmark_visual_gen.sh
(per-model API example script)     (perf benchmark)
   │                                      │
   ▼                                      ▼
"I want to tune perf" ──────────►  configs/curated/<model>.yaml
                                   configs/database/<model>/...
```

### Key Requirements (Derived)

1. **Minimal time-to-first-output** for Persona A — a single script, a single
   command, a visible output file.
2. **Copy-paste production deployment** for Persona B — `trtllm-serve` +
   battle-tested YAML, not raw Python.
3. **Examples are the source of truth** — docs reference them, tests run them.
   No code duplication across these three surfaces.
4. **Per-model coverage** — each supported model family (Wan, FLUX, LTX-2) has
   at least one example that doubles as a CI sanity test.
5. **Config reuse** — offline examples and serve examples share the same YAML
   config files for parallelism, quantization, and TeaCache settings.
6. **Progressive complexity** — quickstart → per-model example scripts →
   serve/benchmark. Feature-specific examples come later as the feature set
   matures.

---

## 2. Current State Assessment

### What Exists Today

```
examples/visual_gen/
├── README.md                      # 260-line guide covering all models
├── quickstart_example.py          # Wan T2V, minimal (28 lines)
├── visual_gen_flux.py             # FLUX T2I, ~348 lines, full CLI
├── visual_gen_wan_t2v.py          # Wan T2V, ~260 lines, full CLI
├── visual_gen_wan_i2v.py          # Wan I2V, ~256 lines, full CLI
├── visual_gen_ltx2.py             # LTX-2, ~298 lines, full CLI
└── serve/
    ├── README.md                  # Operational guide for trtllm-serve
    ├── sync_image_gen.py          # OpenAI SDK client for images
    ├── sync_video_gen.py          # HTTP client for sync video
    ├── async_video_gen.py         # OpenAI SDK client for async video
    ├── delete_video.py            # Video lifecycle demo
    ├── benchmark_visual_gen.sh    # Benchmark wrapper
    └── configs/
        ├── flux1.yml              # FLUX.1 serve config
        ├── flux2.yml              # FLUX.2 serve config
        ├── wan.yml                # Wan serve config
        └── ltx2.yml               # LTX-2 serve config

docs/source/models/visual-generation.md    # 176 lines, feature overview + dev guide
```

### Problems

| Problem | Impact |
|---------|--------|
| **Monolithic README** | The 260-line `examples/visual_gen/README.md` mixes all models, all modes (offline/serve), all features (quant, TeaCache, parallelism). Hard to scan. |
| **Per-model scripts are heavyweight CLIs, not examples** | Each `visual_gen_*.py` is 250–350 lines with extensive `argparse`, building a CLI tool on top of the API. Users must learn the CLI abstraction instead of the API itself. |
| **Serve clients are protocol demos, not model examples** | `sync_video_gen.py` etc. demonstrate HTTP/SDK usage, not model-specific workflows. They're useful but orthogonal to model coverage. |
| **Configs are serve-only** | YAML configs under `serve/configs/` are only used with `trtllm-serve`. Offline examples don't consume them (they use `argparse` flags instead). |
| **Docs duplicate code** | `visual-generation.md` inlines CLI commands and Python snippets that aren't tied to real scripts. These will drift. |
| **No curated configs** | Unlike LLM models, there are no VisualGen entries in `examples/configs/curated/` or `examples/configs/database/`. |
| **Tests partially cover examples** | `test_visual_gen_quickstart` runs the quickstart script. Per-model CLI scripts are NOT directly tested by CI — they're wrapped by integration fixtures that import fragments. |
| **No example index** | No root-level README or index page listing all VisualGen examples. |

---

## 3. Design Principles

Aligned with the user's stated priorities:

| # | Principle | Rationale |
|---|-----------|-----------|
| 1 | **Examples over docs** | Docs route to examples. Examples are readable Python scripts with inline comments. Docs should not carry standalone code snippets. |
| 2 | **Examples are CI-tested** | Every example script is run by at least one CI test. If the example breaks, CI catches it. |
| 3 | **Per-model sanity coverage** | Each supported model family has one offline example that serves as a functional sanity test. |
| 4 | **Two tiers: functional & perf** | Tier 1 (functional) = per-model API example scripts for Persona A, run as sanity tests. Tier 2 (perf) = `trtllm-serve` + benchmark for Persona B, run as perf tests. |
| 5 | **Shared configs** | Both tiers consume the same YAML config files. `VisualGenArgs.from_yaml()` in example scripts; `--extra_visual_gen_options` in serve. |
| 6 | **Docs as index** | `visual-generation.md` becomes a routing page: supported models table, feature matrix, links to examples. No inline code beyond `literalinclude` of quickstart. |
| 7 | **Don't over-example** | No per-feature example pages yet. Features (quant, TeaCache, parallelism) are config YAML fields, shown inline in per-model example scripts. |
| 8 | **Examples use the API, not CLI** | Per-model examples focus on model-specific request construction and output processing. The only CLI flag is `--extra_visual_gen_args` (optional, shared with `trtllm-serve`). No per-model CLI abstraction. |

---

## 4. Proposed Structure

### 4.1 Examples Layout

```
examples/visual_gen/
├── README.md                          # Overview + index of subdirectories
│
├── quickstart_example.py             # [KEEP] Persona A entry point (Wan T2V, 1 GPU)
│
├── models/                            # Tier 1: Per-model offline examples
│   ├── wan_t2v.py                     # Wan text-to-video (replaces visual_gen_wan_t2v.py)
│   ├── wan_i2v.py                     # Wan image-to-video (replaces visual_gen_wan_i2v.py)
│   ├── flux.py                        # FLUX text-to-image (replaces visual_gen_flux.py)
│   └── ltx2.py                        # LTX-2 text-to-video (replaces visual_gen_ltx2.py)
│
├── serve/                             # Tier 2: Serve + benchmark examples
│   ├── README.md                      # How to use trtllm-serve for visual gen
│   ├── serve_and_benchmark.sh         # Unified: launch server + run benchmark
│   └── clients/                       # Client examples (protocol demos)
│       ├── sync_image_gen.py
│       ├── sync_video_gen.py
│       ├── async_video_gen.py
│       └── delete_video.py
│
└── configs/                           # Shared YAML configs (offline + serve)
    ├── wan_t2v_1gpu.yaml              # 1-GPU baseline (sanity / getting started)
    ├── wan_t2v_8gpu.yaml              # 8-GPU performance config
    ├── wan_i2v_1gpu.yaml
    ├── wan_i2v_8gpu.yaml
    ├── flux1_1gpu.yaml
    ├── flux1_8gpu.yaml
    ├── flux2_1gpu.yaml
    ├── flux2_8gpu.yaml
    ├── ltx2_1gpu.yaml
    └── ltx2_8gpu.yaml
```

### 4.2 What Changes

| Current | Proposed | Why |
|---------|----------|-----|
| `visual_gen_wan_t2v.py` (260-line CLI tool) | `models/wan_t2v.py` (~35-line API script) | Readable example, not a CLI abstraction |
| `argparse` flags for parallelism, quant, TeaCache | Single `--extra_visual_gen_args` flag pointing to shared YAML config (optional) | Same config for offline examples and `trtllm-serve`, no per-model CLI layer |
| Serve configs in `serve/configs/` only | Single `configs/` directory shared by offline examples and `trtllm-serve` | One set of configs, no duplication |
| 4 client scripts at `serve/` root | Moved to `serve/clients/` | Separate client demos from server launch scripts |
| `benchmark_visual_gen.sh` | `serve_and_benchmark.sh` | Combined launch + benchmark for Persona B |
| No root README | `README.md` as index | Navigability |

### 4.3 Config Structure

A single `configs/` directory serves both offline examples
(`--extra_visual_gen_args`) and `trtllm-serve`
(`--extra_visual_gen_options`) — same YAML format, same files:

```yaml
# examples/visual_gen/configs/wan_t2v_1gpu.yaml
# 1-GPU Wan T2V — functional sanity / getting started
teacache:
  enable_teacache: false
attention:
  backend: VANILLA
parallel:
  dit_cfg_size: 1
  dit_ulysses_size: 1
```

```yaml
# examples/visual_gen/configs/wan_t2v_8gpu.yaml
# 8-GPU Wan T2V — performance config
teacache:
  enable_teacache: true
  teacache_thresh: 0.2
attention:
  backend: TRTLLM
parallel:
  dit_cfg_size: 2
  dit_ulysses_size: 4
```

Model-specific fields (e.g., `text_encoder_path` for LTX-2) go in that
model's config file — no separate serve-only config directory needed.

### 4.4 Curated Configs (Stretch Goal)

Follow the LLM pattern and add VisualGen entries to
`examples/configs/curated/`:

```
examples/configs/curated/
├── wan-t2v-latency.yaml
├── wan-t2v-throughput.yaml
├── flux-latency.yaml
└── ltx2-latency.yaml
```

This enables the deployment guide to reference them with the standard
`trtllm-serve <model> --extra_visual_gen_options $TRTLLM_DIR/examples/configs/curated/<config>.yaml`
pattern. **Deferred** until configs are validated by perf sanity CI.

---

## 5. Example Specifications

### 5.1 Tier 1 — Per-Model Example Scripts (Functional Sanity)

**Audience**: Persona A (getting started)
**Tested by**: CI sanity tests (1-GPU, correctness-focused)

#### Why per-model?

The reason examples are per-model is that **modality differs per model**.
What varies across models is:

- **Request construction**: T2V needs `num_frames`/`frame_rate`, I2V needs
  an input image, T2I needs `num_images_per_prompt`, LTX-2 adds audio params.
- **Output processing**: video models produce `output.video` (save as MP4),
  image models produce `output.images` (save as PNG), LTX-2 may produce audio.

What does NOT vary per model is the engine config (parallelism, quantization,
TeaCache, attention backend) — those are YAML config fields, shared across
all models.

#### Script pattern

Each per-model script is a small, focused API example (~30–40 lines) that
demonstrates the model-specific request and output handling. The only CLI
flag is `--extra_visual_gen_args`, which optionally points to a YAML config
for performance tuning — the same YAML that `trtllm-serve
--extra_visual_gen_options` consumes. By default it is not used (1-GPU,
default settings).

```python
# examples/visual_gen/models/wan_t2v.py
"""Wan 2.1 / 2.2 Text-to-Video generation.

Usage:
    python wan_t2v.py
    python wan_t2v.py --extra_visual_gen_args ../configs/wan_t2v_8gpu.yaml
"""
import argparse

from tensorrt_llm import VisualGen, VisualGenArgs, VisualGenParams
from tensorrt_llm.serve.media_storage import MediaStorage

parser = argparse.ArgumentParser()
parser.add_argument("--extra_visual_gen_args", type=str, default=None,
                    help="Path to YAML config (same as trtllm-serve --extra_visual_gen_options)")
args = parser.parse_args()

diffusion_args = VisualGenArgs.from_yaml(args.extra_visual_gen_args) if args.extra_visual_gen_args else None
visual_gen = VisualGen(model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", diffusion_args=diffusion_args)

# --- Model-specific: request construction (T2V modality) ---
params = VisualGenParams(
    height=480, width=832,
    num_frames=81,
    guidance_scale=5.0,
    num_inference_steps=50,
    seed=42,
)
output = visual_gen.generate(inputs="A cat playing piano in a sunny room", params=params)

# --- Model-specific: output processing (video) ---
MediaStorage.save_video(output.video, "wan_t2v_output.mp4", frame_rate=params.frame_rate)
print("Saved: wan_t2v_output.mp4")
```

The user can run as-is on 1 GPU for a quick functional test, or point
`--extra_visual_gen_args` to a performance config to see tuned behavior:

```bash
# Default: 1 GPU, no tuning
python wan_t2v.py

# With perf config: multi-GPU, TeaCache, quantization
python wan_t2v.py --extra_visual_gen_args ../configs/wan_t2v_8gpu.yaml
```

#### Design constraints

| Constraint | Rationale |
|------------|-----------|
| ~30–40 lines | Readable at a glance — it's an example, not a tool |
| Only CLI flag is `--extra_visual_gen_args` (optional) | Shared with `trtllm-serve --extra_visual_gen_options`; no per-model CLI abstraction |
| Script body focuses on request + output (the per-model parts) | Engine config is in YAML, not in the script |
| Model-specific defaults come from the model, not the script | Avoid SGLang's default-value bug |
| Prints the output file path on success | Verifiable by CI (check file exists) |

#### Comparison with current scripts

The current `visual_gen_wan_t2v.py` (260 lines) rebuilds the entire
`VisualGenArgs` via `argparse` — parallelism, quantization, TeaCache,
attention backend, etc. are all CLI flags. This is a CLI tool on top of the
API. The new approach eliminates this layer: the script shows only the
model-specific parts (request + output), and engine config is an optional
YAML file that is shared with the serve path.

### 5.2 Tier 2 — Serve + Benchmark Examples (Performance)

**Audience**: Persona B (performance-eager)
**Tested by**: Perf sanity CI (multi-GPU, latency/throughput-focused)

The serve tier is simpler — it's `trtllm-serve` + a benchmark script:

```bash
# examples/visual_gen/serve/serve_and_benchmark.sh
# Usage: ./serve_and_benchmark.sh <model_path> <config> [benchmark_args...]
#
# Example:
#   ./serve_and_benchmark.sh Wan-AI/Wan2.2-T2V-A14B-Diffusers \
#       configs/wan_t2v.yml \
#       --num-prompts 10 --max-concurrency 4

trtllm-serve "$MODEL" --extra_visual_gen_options "$CONFIG" &
wait_for_health
python -m tensorrt_llm.serve.scripts.benchmark_visual_gen "$@"
```

**Client examples** (`serve/clients/`) remain as protocol demos for users
who want to integrate with the API. They are not per-model — they're
per-protocol (sync image, sync video, async video).

### 5.3 What We Intentionally Omit

| Omission | Reason |
|----------|--------|
| Per-feature example pages (quantization, TeaCache, parallelism) | Not enough features yet to justify separate pages. These are YAML config fields passed via `--extra_visual_gen_args`. |
| Notebook examples | Maintenance cost is high; scripts are more CI-friendly. |
| Multi-node examples | VisualGen is single-node only today. |
| Model-specific deployment guides (à la LLM) | Premature — 5 model families is manageable with one doc page. Add when we reach 10+. |

---

## 6. Documentation Refactor

### 6.1 `docs/source/models/visual-generation.md` — Proposed Outline

The doc becomes an **index + feature overview**, not a tutorial:

```
# Visual Generation (Prototype)

## Background                           [KEEP - 1 paragraph]
## Supported Models                     [KEEP - table]
## Feature Matrix                       [KEEP - table]

## Quick Start
  - literalinclude quickstart_example.py  [KEEP - single embed]
  - Link to examples/visual_gen/ for more

## Serving
  - 1-liner trtllm-serve command
  - Endpoint table                      [KEEP]
  - Link to examples/visual_gen/serve/  for configs, clients, benchmark

## Optimizations                        [SIMPLIFY]
  - Quantization: 1-sentence + link to config YAML
  - TeaCache: 1-sentence + link to config YAML
  - Multi-GPU: 1-sentence + link to config YAML
  (Remove inline CLI examples — settings are in YAML configs and shown
   as commented-out lines in per-model example scripts)

## Developer Guide                      [KEEP as-is]
  - Architecture overview
  - Adding a new model
```

**What gets removed from docs**:
- Inline `python visual_gen_wan_t2v.py --linear_type trtllm-fp8-per-tensor ...`
  CLI commands (these are replaced by YAML configs and readable example scripts)
- Inline `VisualGenArgs` Python snippet (this lives in example scripts)
- Duplicated serve launch commands (these live in `serve/README.md`)

**What gets added**:
- Links to specific example scripts for each model
- Links to YAML configs
- A "Next Steps" section mirroring the LLM quick-start-guide pattern

### 6.2 `examples/visual_gen/README.md` — Proposed Outline

Becomes a concise overview of direct children only (subdirs have their own
READMEs):

```markdown
# VisualGen Examples

## Quick Start
- quickstart_example.py — generate a video in ~30 lines

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| models/   | Per-model example scripts (Wan T2V/I2V, FLUX, LTX-2) |
| configs/  | YAML configs shared by offline examples and `trtllm-serve` |
| serve/    | trtllm-serve usage, benchmarking, and client examples |
```

---

## 7. CI / Testing Integration

### 7.1 Test Tiers

| Tier | What It Tests | Scripts | CI Cadence | GPU |
|------|---------------|---------|------------|-----|
| **Sanity (Tier 1)** | Per-model API example scripts produce valid output | `models/wan_t2v.py`, `models/flux.py`, etc. | Pre-merge (L0) | 1 GPU |
| **Quickstart** | The quickstart script runs end-to-end | `quickstart_example.py` | Pre-merge (L0) | 1 GPU |
| **Serve E2E (Tier 2)** | `trtllm-serve` launches, serves requests, responds correctly | `serve_and_benchmark.sh` | Post-merge | 1–8 GPU |
| **Perf Sanity** | Latency/throughput regression detection | Perf sanity YAML configs (see [perf sanity design](visual_gen_perf_sanity_design.md)) | Post-merge | 8 GPU |
| **vbench** | Quality metrics on generated outputs | Existing `test_visual_gen.py` vbench fixtures | Post-merge | 1+ GPU |

### 7.2 How Tests Invoke Examples

Per-model examples have only one optional CLI flag (`--extra_visual_gen_args`),
so CI tests simply run the script as-is for sanity, or with a config for
perf testing:

```python
# tests/integration/defs/examples/test_visual_gen.py

def test_wan_t2v_sanity(llm_venv):
    """Runs the per-model Wan T2V example — default settings, 1 GPU."""
    script = EXAMPLE_ROOT / "models" / "wan_t2v.py"
    venv_check_call(llm_venv, ["python", str(script)])
    assert Path("wan_t2v_output.mp4").exists()

def test_flux_sanity(llm_venv):
    """Runs the per-model FLUX example — default settings, 1 GPU."""
    script = EXAMPLE_ROOT / "models" / "flux.py"
    venv_check_call(llm_venv, ["python", str(script)])
    assert Path("flux_output.png").exists()

def test_wan_t2v_perf_config(llm_venv):
    """Runs Wan T2V with a perf config to validate config loading."""
    script = EXAMPLE_ROOT / "models" / "wan_t2v.py"
    config = EXAMPLE_ROOT / "configs" / "wan_t2v_8gpu.yaml"
    venv_check_call(llm_venv, ["python", str(script),
                               "--extra_visual_gen_args", str(config)])
    assert Path("wan_t2v_output.mp4").exists()
```

The scripts run identically whether invoked by a user or by CI — no
special env vars or patching needed.

### 7.3 CI List Entries

```yaml
# tests/integration/test_lists/test-db/l0_b200.yml (additions)
- examples/test_visual_gen.py::test_wan_t2v_sanity
- examples/test_visual_gen.py::test_flux_sanity
- examples/test_visual_gen.py::test_ltx2_sanity
- examples/test_visual_gen.py::test_visual_gen_quickstart
```

### 7.4 Relationship to Perf Sanity

The perf sanity system (designed in
[visual_gen_perf_sanity_design.md](visual_gen_perf_sanity_design.md)) is the
Tier 2 CI layer. It uses `trtllm-serve` + `benchmark_visual_gen.py` with
dedicated YAML configs under `tests/scripts/perf-sanity/visual_gen/`.
This is orthogonal to the example refactor but benefits from it:

- Perf sanity serve configs can reference the same shared configs from
  `examples/visual_gen/configs/` as a baseline.
- The benchmark script (`serve_and_benchmark.sh`) can be reused by perf
  sanity tests with different concurrency/iteration parameters.

---

## 8. Migration Plan

### Phase 1 — Restructure (Non-Breaking)

1. Create `examples/visual_gen/models/` directory
2. Create `examples/visual_gen/configs/` with 1-GPU baseline configs
3. Write `models/wan_t2v.py` as a simple API script (~35 lines, no argparse)
4. Write remaining per-model example scripts (same pattern)
5. Move serve client scripts to `serve/clients/`
6. Create new `serve_and_benchmark.sh`
7. Update `examples/visual_gen/README.md` to new index format
8. Keep old scripts as deprecated aliases (import + deprecation warning)

### Phase 2 — Tests

1. Update `test_visual_gen.py` fixtures to point at new script paths
2. Add per-model sanity tests for FLUX and LTX-2 (Wan already exists)
3. Verify CI list entries pass
4. Remove deprecated alias scripts once CI is green

### Phase 3 — Docs

1. Slim down `visual-generation.md` to index format
2. Replace inline CLI examples with `literalinclude` from real scripts
3. Add links to per-model examples and configs
4. Consider adding VisualGen entry to deployment guide index (stretch)

### Phase 4 — Curated Configs (Stretch)

1. Validate performance configs through perf sanity CI
2. Promote validated configs to `examples/configs/curated/`
3. Add VisualGen entries to deployment guide recipe table

---

## 9. Open Questions

| # | Question | Options | Leaning |
|---|----------|---------|---------|
| 1 | **Should the model path in per-model scripts be editable inline or also a CLI flag?** | (a) inline only (user edits the script), (b) `--model_path` flag alongside `--extra_visual_gen_args` | (a) — keeps scripts minimal; CI can use the default HF ID or symlink. If CI needs a local path, (b) adds one more flag without bloating the script. |
| 2 | **Where should shared configs live?** | (a) `examples/visual_gen/configs/`, (b) `examples/configs/curated/`, (c) both | (a) for now, (b) once validated by perf sanity |
| 3 | **Should we keep separate per-model scripts or consolidate into 1 script with auto-detection?** | (a) per-model, (b) unified | (a) — per-model scripts are self-contained and readable. Auto-detection exists at the API level already. |
| 4 | **Config naming convention?** | (a) `<model>_<gpus>.yaml` (e.g., `wan_t2v_8gpu.yaml`), (b) `<model>_<scenario>.yaml` (e.g., `wan_t2v_latency.yaml`) | (a) for now — GPU count is the primary variable. Scenario-based naming when we have enough data points. |
| 5 | **How thin should the quickstart be?** | (a) current 28 lines, (b) even thinner (remove MediaStorage, just print tensor shape) | (a) — saving a file is the satisfying "it worked" moment |
| 6 | **When to add per-feature examples?** | (a) now, (b) when we have 3+ features that need dedicated pages | (b) — premature now. Quant, TeaCache, parallelism are YAML config fields passed via `--extra_visual_gen_args`. |
| 7 | **Should the I2V example (Wan I2V) include a sample input image in the repo?** | (a) yes (small JPEG), (b) no (download from URL) | (a) — avoids network dependency in CI, keeps example self-contained |
| 8 | **Should `visual-generation.md` link to per-model example scripts, or to per-model sections in README?** | (a) direct links to scripts, (b) link to README sections | (a) — fewer indirections. README is for browsing on GitHub. |
