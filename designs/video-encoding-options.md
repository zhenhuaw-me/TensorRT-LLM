# Video Encoding Options for TRT-LLM on NVIDIA GPUs — Investigation Notes

**Status:** investigation in progress. Captures findings on license surface,
hardware availability, library landscape, and order-of-magnitude compute cost
for video encoding alongside LLM workloads on NVIDIA data-center GPUs.
No commitments here; design follow-ups are listed at the end.

**Scope question:** TRT-LLM workloads that produce image tensors on GPU (e.g.,
diffusion / video generation) may need to emit encoded video (H.264 today,
potentially HEVC / AV1 later). What are the viable paths on B200-class
hardware, and what are the trade-offs?

---

## 1. License Surface of FFmpeg and Adjacent Libraries

### 1.1 What triggers GPL in the FFmpeg ecosystem

FFmpeg is LGPL 2.1+ by default. Building with `--enable-gpl` pulls in
GPL-only components and flips the resulting FFmpeg binary to GPL. Anything
statically / dynamically linked into a single derivative work can then be
transitively affected.

| Component | Purpose | License |
|---|---|---|
| `libx264` | H.264 encode (CPU) | GPL |
| `libx265` | HEVC encode (CPU) | GPL |
| `libxvid` | MPEG-4 ASP encode (CPU) | GPL |
| `libpostproc` | Post-processing filter | GPL |

### 1.2 Non-GPL CPU encoder alternatives

| Component | Purpose | License |
|---|---|---|
| `openh264` | H.264 encode/decode (Cisco) | BSD-2 (+ Cisco-hosted binaries cover AVC patent royalties) |
| `libvpx` | VP8 / VP9 | BSD |
| `libaom` | AV1 (reference) | BSD-2 + AOM patent grant |
| `SVT-AV1` | AV1 (Intel, fast) | BSD-3 + AOM patent grant |
| `SVT-HEVC` | HEVC | BSD-2 + patent grant |
| `SVT-VP9` | VP9 | BSD |

### 1.3 NVENC path and the GPL question

`NVENC` ships under NVIDIA's SDK EULA, royalty-free. Headers for binding it
into FFmpeg (`nv-codec-headers`) are MIT / LGPL-compatible. An FFmpeg built
with `--disable-gpl --enable-nvenc --enable-nvdec` is LGPL, GPL-free, and
invokes hardware H.264 / HEVC / AV1. `--enable-nonfree` is **not** required
for NVENC (that flag is for cases like fdk-aac or OpenSSL-linked-with-GPL
combinations).

### 1.4 What opencv-python-headless actually ships

TRT-LLM depends on `opencv-python-headless` (see `requirements.txt`,
`docker/common/install.sh`). The bundled FFmpeg in that wheel is built
**LGPL-only**, without `libx264` / `libx265`. Consequences:

- `cv2.VideoCapture` for **decoding** works and is covered by LGPL §6.
- `cv2.VideoWriter` for **H.264 / HEVC encoding** does not work in that wheel.
- No transitive GPL contamination for TRT-LLM.

### 1.5 Open license question

- If NVIDIA redistributes a container that bundles a custom FFmpeg with
  `--enable-gpl` (to get `libx264`), that redistribution would need full GPL
  compliance. Need to confirm whether any shipped TRT-LLM artifact requires
  this path.

---

## 2. NVENC Hardware Availability on Data-Center GPUs

NVENC is a fixed-function ASIC separate from SMs and tensor cores. NVIDIA
selectively removes NVENC from compute-focused data-center SKUs. This is
the single most important finding of the investigation so far.

| GPU | NVENC engines | NVDEC engines | Notes |
|---|---|---|---|
| A100 | **0** | 5 | No encode |
| H100 / H200 | **0** | 7 | No encode |
| **B100 / B200** | **0** | 7 | **No encode** — primary TRT-LLM target today |
| GB200 (2× B200) | **0** | 14 (sum) | Grace CPU (72× Neoverse V2) available for CPU encode |
| L40S (Ada) | 3 | 3 | Candidate sidecar encoder for DGX B200 |
| RTX PRO 6000 Blackwell | yes (9th-gen) | yes | Pro workstation Blackwell has NVENC |
| RTX 50-series (consumer Blackwell) | yes (9th-gen) | yes | Consumer NVENC |

**Source:**
[NVIDIA Video Encode and Decode GPU Support Matrix](https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new)
and
[Video Codec SDK 13.0 Blackwell blog](https://developer.nvidia.com/blog/nvidia-video-codec-sdk-13-0-powered-by-nvidia-blackwell/).

**Implication:** any path that relies on `PyNvVideoCodec` / `h264_nvenc` /
`hevc_nvenc` assumes NVENC hardware. On B200, that path is unavailable unless
a sidecar NVENC-equipped GPU is introduced.

---

## 3. GPU Encoding Library Landscape

Survey of libraries that could sit between "CUDA tensor of frames" and
"encoded bitstream". Ordered by how closely they fit a PyTorch/CUDA producer.

| Library | Strength | License | Notes |
|---|---|---|---|
| **PyNvVideoCodec** | Thin Python binding over Video Codec SDK; zero-copy from CUDA tensors via DLPack / `__cuda_array_interface__` | Permissive (MIT-style) | Successor to VPF. Needs NVENC hardware. |
| **FFmpeg (LGPL) + `h264_nvenc`** | Full muxer/demuxer/filter graph, container formats, HLS/DASH | LGPL (build-dependent) | Call via PyAV or subprocess. `hwupload_cuda` avoids host round-trip. |
| **NVIDIA DALI** | `nvidia.dali.fn.experimental.video` encode/decode | Apache 2.0 | Heavy dependency if used only for encoding. |
| **torchcodec** | PyTorch-native, CUDA-accelerated via FFmpeg under the hood | BSD-3 | Decoder mature; encoder support younger than PyNvVideoCodec. |
| **VPF** | Predecessor to PyNvVideoCodec | — | Archived; do not add. |
| **NVIDIA DeepStream / Video Codec SDK (C++)** | Lowest-level, absolute SOL | Proprietary / SDK EULA | Overkill unless PyNvVideoCodec is a measured bottleneck. |
| **OpenCV `VideoWriter`** via `opencv-python-headless` | — | — | Does not actually encode H.264/HEVC in the shipped wheel (LGPL FFmpeg, no x264/x265). |

All of the above except DeepStream/SDK fall back to invoking NVENC under the
hood, so they share the B200 limitation in Section 2.

---

## 4. Compute Complexity: H.264 Encode vs LLM GEMM

### 4.1 Rough arithmetic budget for H.264 encoding

Per second of output at 1080p60, "medium"-class preset, integer ALU ops:

| Stage | Share of total | Character |
|---|---|---|
| Motion estimation | 60–80% | SAD on 16×16 blocks, hierarchical search, branch-heavy |
| Intra prediction | 5–10% | mode search |
| Integer transform + quantization | 2–5% | trivial, regular |
| Deblocking filter | 3–5% | memory-bound |
| Rate-distortion optimization | multiplies 2–10× on top | tries multiple encoding choices |
| CABAC entropy coding | ~5% compute, serial | cannot parallelize inside a slice |

### 4.2 Order-of-magnitude comparison

| Workload | Sustained ops/sec (rough) |
|---|---|
| H.264 1080p60 "medium" encode | ~5–30 Gops/sec (integer, 16-bit) |
| H.264 4K60 "medium" encode | ~30–150 Gops/sec |
| x264 "veryslow" 1080p60 (max quality) | ~50–200 Gops/sec |
| **LLM inference on B200 (FP8, sustained)** | **~1–3 × 10⁶ Gops/sec** |
| LLM training on B200 (FP8, sustained) | ~2–4 × 10⁶ Gops/sec |

**Ratio: H.264 encode is 4–6 orders of magnitude smaller than LLM GEMM.**
Video encoding is effectively noise next to the model forward pass in pure
compute terms. The cost is architectural, not FLOPS-bound.

### 4.3 Open questions for quantitative follow-up

- Tighter FPS targets for the actual TRT-LLM video-gen workloads (is it ever
  real-time, or always offline post-generation?). This changes whether a CPU
  encoder is adequate even at `slow` / `veryslow`.
- Bit-rate / quality targets (and therefore acceptable BD-rate loss of the
  non-GPL encoders like openh264 / SVT-HEVC vs. x264).
- Stream fan-out: are we encoding one video at a time, or dozens concurrently
  (batched diffusion)?

---

## 5. Feasibility of H.264 Encoding on Tensor Cores / CUDA Cores

### 5.1 Tensor cores — structurally wrong fit

Tensor cores are dense low-precision matmul accumulators (FP8 / FP16 / BF16
/ INT8 → FP32 / INT32). H.264 encoding is dominated by:

- **SAD** (`Σ |a_i − b_i|`) — not a matmul.
- Branch-heavy, data-dependent control flow in ME and mode decision.
- Irregular memory access patterns (search window hops, reference frames).
- Inherently serial stages: CABAC inside a slice.

A rewrite to SSE (`Σ (a_i − b_i)²`) can express the distance metric as an
inner product — matmul-friendly — but changes the rate-distortion metric and
degrades quality-at-bitrate noticeably against tuned reference encoders.

### 5.2 CUDA cores — historically tried, abandoned

- NVIDIA shipped a CUDA-based H.264 encoder pre-NVENC (Fermi era), deprecated.
- MainConcept shipped a commercial CUDA H.264 encoder, deprecated.
- Academic pieces exist (parallel ME, parallel CABAC variants), but none are
  competitive end-to-end.

Consistent pattern:
1. ME suffers from warp divergence (data-dependent search paths across blocks).
2. CABAC cannot meaningfully use thousands of SMs.
3. RDO's inter-choice dependencies throttle parallelism.

End result: typically **slower than x264 on a handful of CPU cores** at
matched quality, and **100× slower than NVENC** on the same GPU's video ASIC.

### 5.3 Why the ASIC wins

A fixed-function encoder is roughly **50–200× more energy-efficient** per
frame than general-purpose compute running the same algorithm. This is why
every major vendor (NVIDIA NVENC, Intel QSV, AMD VCN, Apple VideoToolbox)
ships a dedicated ASIC rather than using shaders.

### 5.4 Rough cost estimate to build a CUDA-based H.264 encoder

- Scope: ME, intra prediction, transform+quant, deblocking, CABAC, RDO,
  reference-frame management, bitstream writer, rate control.
- Estimate: **10–50 person-years** for a specialist codec team.
- Expected outcome: still slower than x264 on modern CPUs at matched quality,
  still far slower than NVENC on an NVENC-equipped GPU.
- No compelling reason to build this unless strategic (e.g., guaranteed
  software-only path on data-center SKUs with no NVENC, matching a specific
  compliance bar). This is the question that most justifies a follow-up
  design discussion.

### 5.5 Open design question

- Would a tensor-core-friendly **lossy-but-acceptable** encoder (SSE-based
  metric, simplified CABAC, no B-frames) be worth prototyping for a
  B200-only future where an encoder must live on the same GPU as the LLM?
  What is the target BD-rate relative to x264 at which this becomes
  attractive vs. simply pairing B200 with an NVENC sidecar?

---

## 6. CPU Encoding Realistic Performance

Per-core throughput on a modern CPU at 1080p60 (approximate):

| Encoder / preset | Throughput | Quality character |
|---|---|---|
| x264 `ultrafast` | 400–800 fps | much worse than default |
| x264 `fast` | 200–400 fps | below default |
| x264 `medium` | 80–150 fps | default reference |
| x264 `slow` | 30–60 fps | better |
| x264 `veryslow` | 8–15 fps | best, still well below HEVC |
| openh264 | comparable to x264 `fast` at Baseline/Main | BSD, license-clean |

Context:
- Compute budget for H.264 encode is 4–6 orders of magnitude smaller than
  the LLM GEMM it accompanies (Section 4). The question is core availability,
  not per-frame compute.
- **GB200** has a Grace CPU with 72× Neoverse V2 cores. Real-time 4K60 x264
  `medium` uses a small fraction of that.
- Typical **DGX B200** host has 2× 112-core Intel/AMD CPU. Dozens of cores
  are idle during inference.
- Video generation from LLMs typically produces short clips (seconds) per
  request; real-time encoding is usually not even required.

**Finding:** "CPU encoding is pretty slow" is a 2010-era intuition against
NVENC of that era. For modern many-core CPUs at LLM-generated video rates,
CPU encoding is not the bottleneck and typically uses cores that would
otherwise sit idle.

### 6.1 Open performance question

- Measure actual concurrent CPU encode throughput on a GB200 or DGX B200
  with production LLM inference running, to confirm that CPU contention
  doesn't interact badly with the LLM host-side path (dataloader, scheduler,
  multimodal preprocessing, etc.).

### 6.2 Measured: TRT-LLM serve E2E vs bench on Wan 2.2 T2V, 4×GB200

First data point answering "what fraction of `trtllm-serve` end-to-end time
is the encode step" on a B200-class (NVENC-less) host.

**Setup**
- Host: 1× GB200 node, 4× GB200 GPUs (aarch64, sm100), Docker dev container.
- Model: `Wan-AI/Wan2.2-T2V-A14B-Diffusers`, NVFP4 dynamic quantization.
- Config: `examples/visual_gen/configs/wan2.2-t2v-fp4-4gpu.yaml`
  (`dit_cfg_size=2`, `dit_ulysses_size=2`, `enable_parallel_vae=true`,
  CUDA graphs off, no step-caching backend).
- Request: 1280×720, 81 frames, 40 inference steps, CFG 4.0, seed 42.
- Encoder: **pure-Python MJPEG/AVI** (shipped with TRT-LLM; used because
  `ffmpeg` was not installed in the container). Output 12.9 MB / 5 s clip.
- Instrumentation: `tensorrt_llm/serve/openai_server.py` logs two `[timing]`
  lines per request, one around `self.generator.generate*(...)` (the
  "VisualGen Python API pipeline") and one around `MediaStorage.save_video`
  (the "media storage and encode part"). NVTX markers surface the same
  regions in the nsys timeline.
- Scripts: `~/projects/aigv.m1/encoding/run_serve_nsys.sh` (serve + nsys),
  `~/projects/aigv.m1/encoding/run_bench.sh` (`trtllm-bench visual-gen`).

**Numbers (per request, seconds)**

|                         | pipeline | encode | total |
|-------------------------|---------:|-------:|------:|
| `trtllm-bench`          |        — |    n/a | 130.22 (mean of 2) |
| `trtllm-serve` req 1    |   131.71 |   1.03 | 132.74 |
| `trtllm-serve` req 2    |   131.91 |   0.98 | 132.88 |
| `trtllm-serve` mean     |   131.81 |   1.00 | 132.81 |

**Observations**
- **Encode is ~0.75% of serve E2E** at this config (1.0 s out of 132.8 s).
  Confirms Section 4.2's order-of-magnitude argument for the `trtllm-serve`
  video path on Wan 2.2 T2V: the encoder is a rounding error next to the
  DiT forward + VAE decode.
- **Serve pipeline is ~1.6 s (~1.2%) above bench.** Attributable to nsys
  profiling overhead, no serve-side warmup (bench does 1 warmup request),
  FastAPI / asyncio dispatch, and request parsing. Directionally, serve
  and bench measure the same pipeline cost.
- **Encoder choice was MJPEG/AVI, not H.264/MP4.** The pure-Python encoder
  is JPEG-per-frame; it is CPU-bound but trivially parallel across frames
  and has no motion search. Switching to `h264_nvenc` via FFmpeg would
  require NVENC, which is not present on B200 (Section 2) — so on this
  host, the remaining options are CPU `libx264` (GPL) or `openh264` (BSD),
  both via FFmpeg.
- **AVI output size.** 12.9 MB for a 5 s 1280×720 clip is ~20 Mbps.
  MJPEG compresses each frame independently with no inter-frame prediction,
  so bitrate is much higher than H.264 at matched visual quality. The
  bitrate gap, not the encode-CPU gap, is the reason to move to H.264.

**Caveats**
- Single-concurrency, 2 measured requests only. Noise envelope at this
  scale is on the order of a few hundred ms (bench std 77 ms).
- No concurrent inference traffic during encode; Section 6.1's question
  about CPU contention under real traffic is still open.
- nsys profiling is active during the serve run; some of the 1.6 s
  pipeline delta vs bench is nsys overhead, not a property of serve.
- With `ffmpeg` now installed on this host, the immediate follow-up is to
  rerun serve and measure H.264/MP4 encode time + file size, which closes
  the MJPEG-vs-H.264 comparison on the same 4×GB200 box.

---

## 7. Candidate Architectures (for follow-up design)

Not recommendations — these are the shapes we could pick between.

**A. NVENC sidecar GPU.**
- Producer: B200 / GB200. Encoder: L40S or RTX PRO Blackwell, on the same
  node (PCIe / NVLink) or on a separate node.
- Pros: hardware encode at ~0 GPU time and ~0 CPU time on the main node.
- Cons: extra GPU SKU, routing overhead, more topology variations to test.

**B. Host CPU encoder.**
- Producer: B200 / GB200. Encoder: host CPU via `openh264` (BSD) or
  `libx264` (GPL).
- Pros: no extra hardware; openh264 is license-clean; Grace / Xeon / EPYC
  have spare cores during inference.
- Cons: touches host memory (round-trip out of CUDA). Need a CUDA → pinned
  host memory path that does not serialize the LLM forward.

**C. Offline / batch encoding.**
- Producer: B200. Output: raw frames or losslessly compressed (FFV1 /
  lossless HEVC) dumped to storage. Separate node encodes later.
- Pros: decouples inference latency from encode SLA.
- Cons: storage intermediate; not real-time.

**D. Future: on-GPU compute-shader encoder.**
- Producer and encoder both on B200; encoder implemented on CUDA / tensor
  cores.
- Pros: single-GPU story, no sidecar, no host round-trip.
- Cons: Section 5 — historically uncompetitive. Needs fresh investigation
  before committing engineering.

---

## 8. Open Follow-Ups

- [ ] Confirm which TRT-LLM deployment targets actually need to emit H.264
      today (and at what FPS / resolution / concurrency).
- [ ] Decide whether HEVC or AV1 are acceptable substitutes for H.264 in
      any deployment target (opens up BSD-licensed SVT-HEVC / SVT-AV1 on CPU
      and wider GPU options).
- [ ] Benchmark `openh264` and `x264 medium` on GB200 Grace CPU and on a
      DGX B200 host at target resolutions, concurrent with representative
      LLM inference.  *(Partially measured — Section 6.2. Single-request,
      pure-Python MJPEG only so far. H.264/MP4 via newly-installed FFmpeg
      and concurrent-traffic numbers still to do.)*
- [ ] Benchmark PyNvVideoCodec on L40S as a sidecar to a DGX B200 to
      quantify frame-routing overhead.
- [ ] Confirm license posture for any NVIDIA-redistributed artifact that
      bundles FFmpeg with `--enable-gpl`. If nothing we ship requires it,
      the whole GPL question collapses to a non-issue.
- [ ] Decide whether investigation into a compute-shader / tensor-core
      H.264 encoder is worth funding, given the historical evidence in
      Section 5.

---

## 9. References

- [FFmpeg license and legal considerations](https://ffmpeg.org/legal.html)
- [NVIDIA Video Codec SDK](https://developer.nvidia.com/video-codec-sdk)
- [NVIDIA Video Codec SDK 13.0 Powered by Blackwell](https://developer.nvidia.com/blog/nvidia-video-codec-sdk-13-0-powered-by-nvidia-blackwell/)
- [NVIDIA Video Encode and Decode GPU Support Matrix](https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new)
- [NVENC Application Note (13.0)](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvenc-application-note/index.html)
- [PyNvVideoCodec](https://docs.nvidia.com/video-technologies/pynvvideocodec/)
- [Cisco openh264](https://www.openh264.org/)
