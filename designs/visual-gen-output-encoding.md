# VisualGen Python API — Should `generate()` Return Raw Tensors or Encoded Bytes?

> **Status**: Investigation / design discussion — no commitments
> **Scope**: The Python-level `VisualGen.generate()` return type. Specifically whether the "image/video" carried by `VisualGenOutput.output` (today `MediaOutput`) is a raw PyTorch tensor, a decoded PIL/numpy container, or a codec-encoded byte blob (PNG/AVI/MP4).
> **Not in scope**: the HTTP / OpenAI-compatible serving layer (that layer *has* to emit encoded bytes — it is a transport constraint, not a design question). Also not in scope: which H.264 codec path we pick on which hardware — see [`video-encoding-options.md`](./video-encoding-options.md) for that.
> **Related**: [`visual-gen-api-refactor-m2.md §6`](./visual-gen-api-refactor-m2.md), [`status/api-refactor-impl-plan.md` Task 4](../status/api-refactor-impl-plan.md), [`video-encoding-options.md`](./video-encoding-options.md).

---

## 1. The Question

`trtllm-serve` already returns encoded bytes today (MJPEG-in-AVI by default, MP4/H.264 if ffmpeg is available on the host). The Python `VisualGen.generate()` API, by contrast, returns a raw `MediaOutput` whose `image`/`video`/`audio` are `torch.Tensor` objects. That's a gap:

- A user who calls `trtllm-serve` gets an `.avi`/`.mp4` byte blob back over HTTP.
- A user who calls `VisualGen` in-process gets a float tensor and has to know about `MediaStorage.save_video(...)` (which lives under `serve/`) or write their own ffmpeg/PIL code.

The refactor already addresses the *usability* side of that gap by adding `MediaOutput.save(path)` / `.to_bytes(format)` / `.to_pil()` convenience methods (§6.2 of the refactor doc). The remaining design question is:

> **Should `VisualGen.generate()` itself return the encoded byte blob (mirroring `trtllm-serve`), or keep returning the raw tensor and leave encoding to an explicit post-processing step?**

This doc argues the question by surveying what the diffusion/video-gen ecosystem does, then laying out pros/cons, and closes with a recommendation.

---

## 2. What the Ecosystem Does

Across every major open-source diffusion/video-generation Python API we've looked at, the **engine-level Python API returns decoded-but-not-codec-encoded data**. Codec encoding (PNG/JPEG/MP4/H.264) is always a separate, explicit step — a utility function, a method on the output object, a CLI flag, or a serving-layer responsibility.

| Framework | Python API return type | Encoding path | Serving layer encoding |
|---|---|---|---|
| **diffusers** (HF) | `output_type` kwarg: `pil` (default), `pt`, `np`, `latent`. Returns `PipelineOutput.images` / `.frames`. | Separate utility: [`diffusers.utils.export_utils.export_to_video(frames, path)`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/utils/export_utils.py) (imageio-ffmpeg backend). Never returns encoded bytes from the pipeline itself. | N/A (library, not a server) |
| **vLLM-Omni** | `omni.generate(...)` → `OmniRequestOutput` with `images: list[PIL.Image.Image]`. Decoded to PIL, **not** base64/PNG. | Post-processing converts tensors → PIL inside the worker; further encoding is the caller's job. | OpenAI-compatible HTTP server encodes to base64 PNG in `data[i].b64_json`. |
| **SGLang Diffusion** | `Engine.generate(SamplingParams)` returns frames / tensors. | CLI offers `sglang generate --save-output` for end-to-end "generate + write file" convenience. Engine-level callers still handle encoding themselves. | Separate OpenAI-compatible server layer. |
| **ComfyUI** | Workflow nodes pass `IMAGE` / video tensors between nodes. | `SaveVideo` / `SaveImage` nodes call ffmpeg/PIL at the *end* of a graph. No "encode inside generate" option. | N/A (node-graph UI, not a Python lib API) |
| **TRT-LLM VisualGen (today)** | `generate()` → `MediaOutput` with raw `torch.Tensor` fields. | User imports `MediaStorage` from `serve/` — awkward cross-module dependency (the refactor fixes this with `MediaOutput.save()` convenience methods). | `trtllm-serve` calls `MediaStorage` in the response path. |

### Cross-cutting pattern

1. **Engine returns "ready-to-render" pixels, not bytes.** Every engine-level API above hands back tensors, numpy, or PIL — all "the pixels, uncompressed in some object". None hand back an `.mp4`/`.png` byte blob.
2. **Codec encoding lives in one of three places, never the engine's hot path:**
   - a utility function (diffusers `export_to_video`),
   - a method on the returned output object (our proposed `MediaOutput.save/to_bytes`),
   - a convenience CLI or HTTP serving layer (SGLang `--save-output`, vLLM-Omni OpenAI server, `trtllm-serve`).
3. **HTTP/serving layers encode because the transport demands it** — JSON can only carry base64; an HTTP download has to be a self-contained container file. This is a transport constraint, not an API design preference.

So "the Python API returns raw pixels, the serving layer returns encoded bytes" is not a TRT-LLM quirk — it is the dominant ecosystem pattern.

---

## 3. Why the Ecosystem Converged on "Raw in Python, Encoded at the Edge"

Five structural reasons explain why every project ended up here independently:

1. **Lossy encoding is a one-way door.** Once `generate()` hands back an H.264 MP4, the frames have been through motion estimation, quantization, CABAC, and chroma subsampling. A researcher who wanted to compute SSIM, feed frames into another model, do inpainting, or extract a keyframe has to *decode* the video you just encoded. That's strictly worse than having kept the tensor.
2. **Codec choice is policy, not generation.** "AVI vs MP4", "libx264 vs openh264 vs NVENC", "CRF 18 vs CRF 23", "yuv420p vs yuv444p", "strip audio vs mux LTX-2 audio" — these are deployment-time decisions that depend on the consumer (browser player, video editor, dataset storage, QoE dashboard). Baking them into `generate()` means either a bloated API surface with a dozen encode knobs on `VisualGenParams`, or a one-size-fits-nobody default.
3. **Encoding dependencies are heavyweight and optional.** FFmpeg is a 100 MB native dependency with a GPL vs LGPL license surface that TRT-LLM has already had to think about carefully (see [`video-encoding-options.md §1`](./video-encoding-options.md)). Users doing pure research on a headless cluster do not want `generate()` to fail or silently degrade because the image doesn't have ffmpeg. Opt-in encoding keeps the core hot path dependency-light.
4. **The latency argument is real but small.** `video-encoding-options.md §4.2` and §6.2 measured encode at **~1.0 s out of ~132.8 s** of serve E2E for Wan 2.2 T2V on 4×GB200 — ~0.75%. So the pro-encoding argument "it's free, just bake it in" is compute-true. But the cost isn't CPU — the cost is **optionality**, see (1)–(3). An always-on encode forces the cost on users who don't want it, and it's still 1 s of wall-clock for a batch job that wants raw frames.
5. **Streaming-readiness only works on tensors.** The refactor design explicitly keeps `VisualGenOutput.finished: bool` to leave room for a future streaming API that yields partial denoising states. You cannot stream partial H.264 slices from a diffusion step — but you *can* stream partial tensors and let the consumer encode at the end. Returning encoded bytes from `generate()` pre-commits against streaming.

---

## 4. Pros/Cons of Each Option

### Option A — Keep raw tensors. Encoding is a method on `MediaOutput` (current refactor plan)

`generate()` → `VisualGenOutput` → `.output: MediaOutput` with `image/video/audio` as `torch.Tensor`.  Convenience: `output.save("x.mp4")`, `output.to_bytes(format="mp4")`, `output.to_pil()`.

**Pros**
- Matches the entire ecosystem (diffusers, vLLM-Omni, SGLang, ComfyUI).
- No wasted work: users who want tensors pay nothing for encoding; users who want files call `.save()` once.
- Codec/container/quality are decoupled from generation — deployment can pick AVI for license-clean fallback, MP4 for polished output, PNG for evaluation, without touching `VisualGenParams`.
- FFmpeg stays **optional** — a user without ffmpeg installed can still call `generate()` and get tensors. `.save("x.avi")` works via the pure-Python MJPEG path (same one `MediaStorage` uses today, see `video-encoding-options.md §6.2`); `.save("x.mp4")` raises a clear error pointing at how to install ffmpeg.
- Preserves streaming-readiness — a future `stream()` yields `VisualGenOutput` objects whose `MediaOutput` carries a partial tensor, no codec involvement.
- Lossless: downstream ML work (metrics, chained models, frame extraction) sees the actual model output, not an H.264 re-quantized version.

**Cons**
- Users who *only* want a file-on-disk write one extra line (`out.save(path)`). Tiny ergonomic cost.
- Divergence from `trtllm-serve`'s return shape: serve gives bytes, Python gives tensors+method. The refactor design already accepts this divergence and documents it — serving is a separate transport.
- Users must be told where encoding happens (we have the `MediaOutput` methods to point at). Mitigated by documentation and a working example.

### Option B — Always return encoded bytes from `generate()`

`generate()` → `VisualGenOutput` with `output: bytes` (or `output.encoded: bytes`) — e.g., an AVI blob by default, MP4 if ffmpeg is present, similar to what `trtllm-serve` produces today.

**Pros**
- Single call, ready-to-write-to-disk output. Shortest path for "give me a file" use case.
- Consistent with `trtllm-serve`'s return shape — Python users get "the same thing" they'd get over HTTP.
- Smaller output object in memory for long videos (compressed bytes are ≪ raw FP16 tensor).

**Cons**
- **Fights the entire ecosystem pattern** from §2 — no peer framework does this.
- Forces encode cost and FFmpeg/encoder dependency on users who don't want them. In-container without ffmpeg, `generate()` would have to fall back to MJPEG/AVI (which the refactor notes is ~20 Mbps vs H.264 at matched quality) or refuse to run. Both are worse than just returning the tensor.
- Lossy one-way door (§3.1). Research users who want SSIM/PSNR or chained models have to decode what we just encoded.
- Codec parameters either bloat `VisualGenParams` (adding `codec`, `crf`, `container`, `pixel_format`, `audio_codec`, …) or are hidden from the user entirely — both are worse than the method-on-output design.
- Blocks streaming — can't meaningfully yield H.264 mid-generation.
- Breaks the "typed, introspectable" property of `MediaOutput` — a `bytes` blob is opaque; a `torch.Tensor` has `shape`, `dtype`, `device` for debugging/logging/metrics.

### Option C — Per-request switch: `VisualGenParams.output_format: "tensor" | "mp4" | "avi" | "png"`

`generate()` inspects the format and returns either a tensor `MediaOutput` or an encoded `bytes`. Union return type.

**Pros**
- Single entry point for both use cases.
- Matches `trtllm-serve`'s request-level `response_format` knob closely.

**Cons**
- Union return types are a usability trap — callers must type-check or conditionally branch. IDE/type-checker integration suffers.
- Resurrects the `output_type` problem the refactor just removed from `VisualGenParams` — the refactor doc §4.2.2 explicitly deleted `output_type` because [`MediaOutput`'s methods subsume it](./visual-gen-api-refactor-m2.md#422-remove-output_type-from-visualgenparams). Reintroducing it for encoded formats retraces that history.
- Codec knobs (codec, quality, frame_rate) still need somewhere to live — either on `VisualGenParams` (bloat) or as additional kwargs on `generate()` (also bloat), or ignored (surprise).
- The Python user who wants both "the tensor for metrics" and "the mp4 for archival" now has to call `generate()` twice, or encode themselves after getting the tensor — which is exactly Option A with extra steps.

### Option D — Return both: tensor on `.output`, encoded bytes on `.encoded` (eagerly)

`generate()` → `VisualGenOutput` with `output: MediaOutput` (tensor) *and* `encoded: bytes` pre-computed.

**Pros**
- Callers get both without branching.

**Cons**
- Pays the encode cost every time, even when the caller only wants the tensor. For the measured ~1 s on 4×GB200 that's 0.75% of E2E — not catastrophic, but strictly wasted for research/eval workflows.
- Doubles memory footprint until GC.
- Still has to pick a codec/container/quality on the user's behalf — same policy-in-the-engine problem as Option B.

---

## 5. Recommendation

**Keep Option A** — the current refactor plan already gets this right. `VisualGen.generate()` returns `VisualGenOutput` with `output: MediaOutput` holding raw tensors; encoding is on the output via `save() / to_bytes() / to_pil()`, with the same "AVI by default, MP4 when ffmpeg is present" policy `trtllm-serve` already uses. This matches every peer framework, preserves streaming readiness, keeps ffmpeg optional, and leaves codec policy where it belongs (deployment time, not generation time).

Concretely this means no API change from what the refactor plan already proposes — this doc is validation that that plan is on the well-trodden path, not a proposal for a new design.

### What we should do on top of the current plan

1. **Document the symmetry explicitly.** A short "Python API vs `trtllm-serve` — same pipeline, different transport" paragraph in the user-facing docs so readers understand why the Python API hands back tensors but HTTP hands back bytes. The `trtllm-serve` response uses `MediaOutput.to_bytes()` internally — make that link visible so users know they are the same code path.
2. **Make `MediaOutput.save()` work out-of-the-box without ffmpeg.** For the common "quickstart" path (a user running locally, maybe no ffmpeg installed), `save("out.avi")` should succeed via the existing pure-Python MJPEG path (same encoder `trtllm-serve` uses today when ffmpeg is absent; see `video-encoding-options.md §6.2`). `save("out.mp4")` without ffmpeg should raise a clear error naming the install fix. The goal: the three-line example in docs (`generate → save → play`) never fails because of a missing dep.
3. **Expose a `to_bytes(format=...)` shortcut.** For users who *do* want `trtllm-serve`-style bytes from the Python API (e.g., writing their own HTTP server on top of VisualGen, or shipping blobs over a queue), one call `output.to_bytes("mp4")` gives them exactly what `trtllm-serve` returns. That covers Option B's "I want bytes" use case without making it the default.
4. **Keep `VisualGenOutput.output` typed as `MediaOutput`, not a union.** No per-request `output_format` knob. If a serving-layer caller wants bytes, they call `.to_bytes()` themselves — `trtllm-serve` already does exactly that.

---

## 6. Open Questions

- **Q-E1 — Where does the pure-Python MJPEG/AVI encoder live after the refactor?** `video-encoding-options.md §6.2` references it as "shipped with TRT-LLM" and used by `MediaStorage`. The refactor's §6.3 moves encoding to `tensorrt_llm/media/encoding.py`. Confirm the MJPEG path moves with it.
- **Q-E2 — FFmpeg detection policy.** Does `MediaOutput.save("x.mp4")` probe for ffmpeg once and cache, every call, or at import time? Probing at import time blocks cold start; probing every call adds overhead. One-time lazy probe (first `.mp4` save triggers detection) is probably right. Who owns this cache — `media/encoding.py` or `MediaOutput`?
- **Q-E3 — What does `trtllm-serve` default to after the refactor?** Serve currently picks AVI if ffmpeg is absent, MP4 otherwise. Refactor-wise, serve should call `MediaOutput.to_bytes(format=...)` — so the *policy* (pick best available container) probably belongs on `to_bytes()` via a `format="auto"` value, not re-implemented in serve. Needs a concrete decision before Task 4 of the impl plan ships.
- **Q-E4 — Batch output**: for `num_images_per_prompt > 1`, does `.save(path)` write `path_0.png, path_1.png, …` or require the caller to iterate? (This is the same question as Q7 in [`status/api-refactor-impl-plan.md`](../status/api-refactor-impl-plan.md) — noting it here because the encoded-output story has to agree with the tensor-shape story.)
- **Q-E5 — Audio muxing policy on `.save("x.mp4")`.** For LTX-2 the `MediaOutput` has both `video` and `audio`. Muxing is ffmpeg-only. If ffmpeg is absent, do we silently drop audio with a warning, write a separate `.wav`, or refuse? Confirm the user expectation before Task 4.

---

## 7. References

- [Hugging Face Diffusers — `export_to_video`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/utils/export_utils.py)
- [Hugging Face Diffusers — Pipelines overview (`output_type`)](https://huggingface.co/docs/diffusers/api/pipelines/overview)
- [vLLM-Omni — Python offline API (DeepWiki)](https://deepwiki.com/vllm-project/vllm-omni/8.4-python-api-(omni-and-omnillm)) — `OmniRequestOutput.images: list[PIL.Image.Image]`
- [vLLM-Omni — Image Generation API (OpenAI-compatible, base64)](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/image_generation_api/)
- [SGLang Diffusion — blog](https://www.lmsys.org/blog/2025-11-07-sglang-diffusion/)
- [SGLang Diffusion — supported models / CLI `--save-output`](https://github.com/sgl-project/sglang/blob/main/docs/supported_models/diffusion_models.md)
- [ComfyUI-VideoHelperSuite — Save Video nodes](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
- [`video-encoding-options.md`](./video-encoding-options.md) — license/hardware/complexity of video encoding on NVIDIA GPUs
- [`visual-gen-api-refactor-m2.md §6`](./visual-gen-api-refactor-m2.md) — output/post-processing design
- [`status/api-refactor-impl-plan.md` Task 4](../status/api-refactor-impl-plan.md) — VisualGenOutput + MediaOutput implementation
