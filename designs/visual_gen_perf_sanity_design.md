# VisualGen Perf Sanity Test Design

## Background

The existing LLM perf sanity system runs benchmarks against `trtllm-serve` using
`benchmark_serving.py`, parses throughput/latency metrics from stdout, uploads to
OpenSearch, and detects regressions via baseline comparison. VisualGen
(image/video generation) requires a parallel system because the server command,
benchmark script, metrics, and regression logic are fundamentally different.

This document summarizes the investigation of the existing LLM perf sanity
infrastructure and proposes a design for adding VisualGen perf sanity tests.

## Existing LLM Perf Sanity Architecture

### Key Files

| File | Role |
|------|------|
| `tests/integration/defs/perf/test_perf_sanity.py` | Main pytest entry point; `ServerConfig`, `ClientConfig`, `PerfSanityTestConfig` classes |
| `tests/integration/defs/perf/open_search_db_utils.py` | OpenSearch upload, baseline computation, regression detection |
| `jenkins/scripts/perf/perf_utils.py` | Shared constants, regression classification, SVG chart generation, HTML dashboard |
| `jenkins/scripts/perf/get_pre_merge_html.py` | Pre-merge HTML report with history charts |
| `jenkins/scripts/perf/perf_sanity_triage.py` | Post-merge triage dashboard + Slack notifications |
| `tests/scripts/perf-sanity/aggregated/*.yaml` | Aggregated test config files |
| `tests/scripts/perf-sanity/disaggregated/*.yaml` | Disaggregated test config files |
| `tensorrt_llm/serve/scripts/benchmark_serving.py` | LLM benchmark client script |
| `tensorrt_llm/serve/scripts/benchmark_visual_gen.py` | VisualGen benchmark client script |
| `tensorrt_llm/bench/benchmark/visual_gen_utils.py` | VisualGen metrics calculation and output formatting |

### Flow

1. pytest parametrizes test cases from config YAML filenames + server config names
2. `PerfSanityTestConfig` parses the YAML, creates `ServerConfig` and `ClientConfig` objects
3. `ServerConfig.to_cmd()` generates `trtllm-serve <model> --backend pytorch --config <extra-llm-api-config.yml>`
4. `ClientConfig.to_cmd()` generates `python -m tensorrt_llm.serve.scripts.benchmark_serving --dataset random ...`
5. Server is launched, health endpoint polled, benchmark client run, stdout captured
6. Metrics parsed via regex from stdout (16 patterns for throughput + latency)
7. Results uploaded to OpenSearch with server/client config fields
8. Baseline computed from history (rolling smooth + P95), regression detected via threshold comparison

## Key Differences: LLM vs VisualGen

### Server Command

| LLM | VisualGen |
|-----|-----------|
| `trtllm-serve <model> --backend pytorch --config <extra-llm-api-config.yml>` | `trtllm-serve <model> --extra_visual_gen_options <config.yml>` |

LLM uses `--config` with a YAML containing `tensor_parallel_size`, `moe_config`,
`kv_cache_config`, `speculative_config`, `cuda_graph_config`, etc. VisualGen uses
`--extra_visual_gen_options` with a YAML containing `parallel.dit_cfg_size`,
`parallel.dit_ulysses_size`, and pipeline-specific options. The health endpoint
(`/health`) is the same for both.

### Benchmark Client Command

| LLM | VisualGen |
|-----|-----------|
| `python -m tensorrt_llm.serve.scripts.benchmark_serving` | `python -m tensorrt_llm.serve.scripts.benchmark_visual_gen` |
| `--dataset random --random-input-len 1024 --random-output-len 1024 --ignore-eos --percentile-metrics ttft,tpot,itl,e2el` | `--backend openai-videos --prompt "..." --size 480x832 --num-frames 81 --fps 16 --num-inference-steps 50` |

LLM client is token-oriented (ISL/OSL, datasets, tokenizer). VisualGen client is
media-oriented (prompt, resolution, frame count, diffusion steps).

### Performance Metrics

**LLM metrics** (parsed from `benchmark_serving.py` stdout):
- `d_seq_throughput` — Request throughput (req/s)
- `d_token_throughput` — Output token throughput (tok/s)
- `d_total_token_throughput` — Total token throughput (tok/s)
- `d_user_throughput` — User throughput (tok/s)
- `d_mean_ttft`, `d_median_ttft`, `d_p99_ttft` — Time to first token
- `d_mean_tpot`, `d_median_tpot`, `d_p99_tpot` — Time per output token
- `d_mean_itl`, `d_median_itl`, `d_p99_itl` — Inter-token latency
- `d_mean_e2el`, `d_median_e2el`, `d_p99_e2el` — End-to-end latency

**VisualGen metrics** (parsed from `benchmark_visual_gen.py` stdout via
`print_visual_gen_results()` in `visual_gen_utils.py`):
- `d_mean_e2e_latency_ms` — Mean E2E latency (ms)
- `d_median_e2e_latency_ms` — Median E2E latency (ms)
- `d_std_e2e_latency_ms` — Std dev E2E latency (ms)
- `d_min_e2e_latency_ms` — Min E2E latency (ms)
- `d_max_e2e_latency_ms` — Max E2E latency (ms)
- `d_p50_e2e_latency_ms`, `d_p90_e2e_latency_ms`, `d_p99_e2e_latency_ms` — Percentile latencies
- `d_request_throughput` — Request throughput (req/s)
- `d_per_gpu_throughput` — Per-GPU throughput (req/s/GPU)

### Regression Detection Direction

LLM regression is based on throughput metrics (higher is better):
regression if `new_value < baseline * (1 - threshold)`.

VisualGen regression is based on latency (lower is better):
regression if `new_value > baseline * (1 + threshold)`.

The existing code in `open_search_db_utils.py` already supports both directions
via `MAXIMIZE_METRICS` and `MINIMIZE_METRICS` lists and handles baseline
computation differently (P95 for maximize, P5 for minimize).

### Server Config Fields

LLM `ServerConfig` has 40+ fields: `tensor_parallel_size`, `moe_expert_parallel_size`,
`pipeline_parallel_size`, `max_batch_size`, `max_num_tokens`, `attn_backend`,
`enable_attention_dp`, `moe_config`, `cuda_graph_config`, `kv_cache_config`,
`speculative_config`, `cache_transceiver_config`, etc.

VisualGen server config is much simpler: `parallel.dit_cfg_size`,
`parallel.dit_ulysses_size`, and pipeline-specific options. These are completely
disjoint from LLM config fields.

### Error Detection in Benchmark Output

LLM output format:
```
Failed requests:                         3
=======================!FAILED REQUESTS!=======================
```

VisualGen output format:
```
  !!! 3 FAILED REQUESTS - CHECK LOG FOR ERRORS !!!
```

## Design Decisions

### Separate Test File (not extending existing)

Create `test_visual_gen_perf_sanity.py` alongside the existing
`test_perf_sanity.py`. Rationale:

- `ServerConfig` and `ClientConfig` are deeply LLM-specific with 40+ fields.
  Adding VisualGen logic to them would be awkward and fragile.
- VisualGen is single-node only (no disaggregated mode), so the disagg
  orchestration code is not needed.
- Separate files reduce the risk of breaking existing LLM tests.
- Shared utilities (OpenSearch upload, baseline, regression) live in
  `open_search_db_utils.py` and are reused by both test files.

### Same OpenSearch Index

Use the existing `perf_sanity_info` index with a new discriminator field
`s_test_type: "visual_gen"`. OpenSearch uses dynamic mapping, so new fields
are auto-detected without schema migration. This keeps the infrastructure
simple and allows unified querying.

### Primary Regression Metric

`d_median_e2e_latency_ms` is the primary regression indicator. It is the most
stable central-tendency measure (robust to outliers unlike mean, captures
typical-case latency). This is a MINIMIZE metric: regression means latency
increased beyond the threshold.

### Initial Model

Wan T2V (text-to-video) only, to validate the framework before expanding to
other models.

### Dashboards / Reports

Reuse existing reporting infrastructure as-is. VisualGen data flows into the
same OpenSearch index and will be picked up by existing queries. Follow-up work
can parameterize `CHART_METRICS` and classification direction by `s_test_type`
for proper VisualGen chart rendering.

## Proposed Config File Format

**Location**: `tests/scripts/perf-sanity/visual_gen/`

**Naming**: `<model_short_name>_<gpu_suffix>.yaml`
(e.g., `wan_t2v_a14b_blackwell.yaml`)

```yaml
metadata:
  model_name: "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
  model_type: "video"
  supported_gpus:
    - B200

hardware:
  gpus_per_node: 8

environment:
  server_env_var: ""
  client_env_var: ""

server_config:
  extra_visual_gen_options:
    parallel:
      dit_cfg_size: 1
      dit_ulysses_size: 4

benchmark_configs:
  - name: "wan_t2v_a14b_480x832_81f_50steps_con1"
    backend: "openai-videos"
    prompt: "A cat playing in the park"
    num_prompts: 5
    size: "480x832"
    num_frames: 81
    fps: 16
    num_inference_steps: 50
    max_concurrency: 1
    iterations: 3
```

## Proposed Test File Structure

**File**: `tests/integration/defs/perf/test_visual_gen_perf_sanity.py`

### VisualGenServerConfig

Stores VisualGen server configuration and generates the server launch command.

- `to_cmd()` returns `["trtllm-serve", model_path, "--extra_visual_gen_options", config_path]`
- `to_db_data()` returns server config fields for OpenSearch:
  `s_model_name`, `s_model_type`, `l_dit_cfg_size`, `l_dit_ulysses_size`, `l_num_gpus`

### VisualGenClientConfig

Stores VisualGen benchmark parameters and generates the benchmark command.

- `to_cmd()` returns `["python", "-m", "tensorrt_llm.serve.scripts.benchmark_visual_gen",
  "--backend", backend, "--model", model, "--prompt", prompt, "--size", size, ...]`
- `to_db_data()` returns benchmark config fields for OpenSearch:
  `s_visual_gen_backend`, `l_num_prompts`, `s_size`, `l_num_frames`, `l_fps`,
  `l_num_inference_steps`, `l_max_concurrency`

### VisualGenPerfSanityTestConfig

Orchestrates the test: config parsing, command generation, server launch,
benchmark execution, metric parsing, and DB upload. Reuses the same
health-check pattern (`wait_for_endpoint_ready` on `/health`).

### Metric Regex Patterns

```python
VISUAL_GEN_METRIC_LOG_QUERIES = {
    "mean_e2e_latency_ms":   re.compile(r"Mean E2E Latency \(ms\):\s+(-?[\d\.]+)"),
    "median_e2e_latency_ms": re.compile(r"Median E2E Latency \(ms\):\s+(-?[\d\.]+)"),
    "std_e2e_latency_ms":    re.compile(r"Std Dev E2E Latency \(ms\):\s+(-?[\d\.]+)"),
    "min_e2e_latency_ms":    re.compile(r"Min E2E Latency \(ms\):\s+(-?[\d\.]+)"),
    "max_e2e_latency_ms":    re.compile(r"Max E2E Latency \(ms\):\s+(-?[\d\.]+)"),
    "p50_e2e_latency_ms":    re.compile(r"P50 E2E Latency \(ms\):\s+(-?[\d\.]+)"),
    "p90_e2e_latency_ms":    re.compile(r"P90 E2E Latency \(ms\):\s+(-?[\d\.]+)"),
    "p99_e2e_latency_ms":    re.compile(r"P99 E2E Latency \(ms\):\s+(-?[\d\.]+)"),
    "request_throughput":    re.compile(r"Request throughput \(req\/s\):\s+(-?[\d\.]+)"),
    "per_gpu_throughput":    re.compile(r"Per-GPU throughput \(req\/s\/GPU\):\s+(-?[\d\.]+)"),
}
```

### Test Entry Point

```python
VG_CONFIG_FOLDER = os.environ.get(
    "VG_CONFIG_FOLDER", "tests/scripts/perf-sanity/visual_gen"
)

@pytest.mark.parametrize("perf_sanity_test_case", VISUAL_GEN_TEST_CASES)
def test_e2e(output_dir, perf_sanity_test_case):
    config = VisualGenPerfSanityTestConfig(perf_sanity_test_case, output_dir)
    config.parse_config_file()
    commands = config.get_commands()
    outputs = config.run_ex(commands)
    config.get_perf_result(outputs)
    config.check_test_failure()
    config.upload_test_results_to_database()
```

### Test Case Naming

Format: `vg_upload-<config_base_name>-<benchmark_name>`

The `vg_upload` prefix indicates upload to OpenSearch (analogous to `aggr_upload`
in LLM tests). Without `_upload`, the test runs locally without DB upload.

## OpenSearch Schema

### New Fields

Each VisualGen document includes all standard job/CI fields (same as LLM) plus:

**Discriminator**:
- `s_test_type: "visual_gen"`

**Server config**:
- `s_model_name`, `s_model_type`
- `l_dit_cfg_size`, `l_dit_ulysses_size`, `l_num_gpus`

**Benchmark config**:
- `s_visual_gen_backend` (openai-images / openai-videos)
- `l_num_prompts`, `s_size`, `l_num_frames`, `l_fps`
- `l_num_inference_steps`, `l_max_concurrency`

**Metrics**:
- `d_mean_e2e_latency_ms`, `d_median_e2e_latency_ms`, `d_std_e2e_latency_ms`
- `d_min_e2e_latency_ms`, `d_max_e2e_latency_ms`
- `d_p50_e2e_latency_ms`, `d_p90_e2e_latency_ms`, `d_p99_e2e_latency_ms`
- `d_request_throughput`, `d_per_gpu_throughput`

### Changes to open_search_db_utils.py

Add VisualGen metrics to existing classification lists:

```python
VISUAL_GEN_MINIMIZE_METRICS = [
    "d_mean_e2e_latency_ms",
    "d_median_e2e_latency_ms",
    "d_p99_e2e_latency_ms",
]

VISUAL_GEN_MAXIMIZE_METRICS = [
    "d_request_throughput",
    "d_per_gpu_throughput",
]

VISUAL_GEN_REGRESSION_METRICS = [
    "d_median_e2e_latency_ms",
]
```

The `prepare_regressive_test_cases()` and `calculate_baseline_metrics()` functions
need to select the appropriate metric lists based on `s_test_type`. The existing
direction logic (MAXIMIZE vs MINIMIZE) already handles both correctly:
- MAXIMIZE: baseline = P95, regression if `new < baseline * (1 - threshold)`
- MINIMIZE: baseline = P5, regression if `new > baseline * (1 + threshold)`

Default threshold: 5% post-merge / 10% pre-merge (same as LLM, tunable per-metric).

## CI Integration

### Test-db YAML

**File**: `tests/integration/test_lists/test-db/l0_b200_multi_gpus_visual_gen_perf_sanity.yml`

```yaml
conditions:
  gpu_type: B200
  system_gpu_count: 8
tests:
  - perf/test_visual_gen_perf_sanity.py::test_e2e[vg_upload-wan_t2v_a14b_blackwell-wan_t2v_a14b_480x832_81f_50steps_con1] TIMEOUT (120)
```

### Jenkins Pipeline

Add to `jenkins/L0_Test.groovy` under `launchTestJobs()`:

```groovy
visualGenSlurmTestConfigs += buildStageConfigs(
    "B200-8_GPUs-VisualGen-PerfSanity-Post-Merge",
    "auto:b200-x8",
    "l0_b200_multi_gpus_visual_gen_perf_sanity",
    1,   // testCount
    8,   // gpuCount
    1,   // nodeCount
)
```

## Follow-Up Work

- **Dashboard charts**: Parameterize `CHART_METRICS`, `METRIC_LABELS`, and
  `CLASSIFICATION_METRICS` in `perf_utils.py` by `s_test_type` so VisualGen
  tests render with latency charts instead of throughput charts
- **Additional models**: Add Flux (text-to-image), Wan I2V (image-to-video)
  configs once the framework is validated
- **Disaggregated VisualGen**: If VisualGen supports multi-node in the future,
  extend the test with disaggregated mode support
