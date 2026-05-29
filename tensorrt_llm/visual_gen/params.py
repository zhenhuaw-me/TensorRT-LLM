# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from tensorrt_llm.llmapi.utils import StrictBaseModel, set_api_status

# Match the OpenAI DALL-E API range that vllm-omni adopts for seed
# (see https://github.com/vllm-project/vllm-omni — api_server.py clamps
# user-supplied seeds to MAX_UINT32_SEED before dispatch). Engine-side
# random seed generation also stays in this range so reproducibility
# transports cleanly between client and server.
MAX_UINT32_SEED = 2**32 - 1


@set_api_status("prototype")
class VisualGenParams(StrictBaseModel):
    """Parameters for visual generation.

    Fields default to ``None``, meaning "use the model's default".
    Per-model defaults are declared by each pipeline via
    ``DEFAULT_GENERATION_PARAMS`` and merged automatically before
    inference.

    Model-specific parameters (e.g. LTX-2's ``stg_scale``, Wan's
    ``guidance_scale_2``) should be passed via ``extra_params``.
    Use ``VisualGen.extra_param_specs`` to discover valid keys
    for the loaded pipeline.
    """

    # Core — None means "use model default"
    height: Optional[int] = Field(default=None, description="Output height in pixels.")
    width: Optional[int] = Field(default=None, description="Output width in pixels.")
    num_inference_steps: Optional[int] = Field(
        default=None, description="Number of denoising steps."
    )
    guidance_scale: Optional[float] = Field(
        default=None, description="Classifier-free guidance scale."
    )
    max_sequence_length: Optional[int] = Field(
        default=None, description="Max tokens for text encoding."
    )
    # When ``num_images_per_prompt > 1`` is honored end-to-end (future),
    # the implementation follows the diffusers/vllm-omni convention:
    # one ``torch.Generator(seed=s)`` drives ``N`` latents from a single
    # RNG stream (batched ``randn``), not SGLang's per-image
    # ``[s, s+1, …]`` expansion. Adding ``seed: int | list[int]`` is
    # left as an additive extension if explicit per-image seeds become
    # a requirement.
    seed: Optional[int] = Field(
        default=None,
        ge=0,
        le=MAX_UINT32_SEED,
        description=(
            "Random seed for reproducibility. ``None`` means the engine draws "
            "a fresh seed on the coordinator rank before pipeline dispatch. "
            f"Must be in ``[0, {MAX_UINT32_SEED}]`` for compatibility with the "
            "OpenAI DALL-E API seed range."
        ),
    )

    # Video
    num_frames: Optional[int] = Field(
        default=None, description="Number of frames. None = model default."
    )
    frame_rate: Optional[float] = Field(default=None, description="Video frame rate in fps.")

    # Conditioning inputs
    negative_prompt: Optional[str] = Field(default=None, description="Negative prompt for CFG.")
    image: Optional[Union[str, bytes, List[Union[str, bytes]]]] = Field(
        default=None, description="Reference image(s) for I2V/I2I."
    )

    # Per-prompt multiplier
    num_images_per_prompt: int = Field(default=1, description="Number of images per prompt.")

    # Model-specific overflow
    extra_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Model-specific parameters. Use VisualGen.extra_param_specs "
        "to discover valid keys for the loaded pipeline.",
    )
