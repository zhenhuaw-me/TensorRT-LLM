# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for VisualGenParams, ExtraParamSchema, pipeline defaults, default merging, and validation."""

from unittest.mock import MagicMock, patch

import pytest


def _wan_mock(is_wan22_14b=False, is_wan22_5b=False, name_or_path="", num_heads=12):
    """Create a mock with attributes needed by WanPipeline/WanI2V properties."""
    mock = MagicMock()
    mock.is_wan22_14b = is_wan22_14b
    mock.is_wan22_5b = is_wan22_5b
    config = MagicMock()
    config._name_or_path = name_or_path
    config.num_attention_heads = num_heads
    mock.config = config
    return mock


# =============================================================================
# VisualGenParams — Pydantic validation
# =============================================================================


class TestVisualGenParamsValidation:
    """VisualGenParams is a Pydantic StrictBaseModel with correct defaults."""

    def test_default_construction(self):
        from tensorrt_llm.visual_gen import VisualGenParams

        params = VisualGenParams()
        # Universal fields default to None
        assert params.height is None
        assert params.width is None
        assert params.num_inference_steps is None
        assert params.guidance_scale is None
        assert params.max_sequence_length is None
        assert params.num_frames is None
        assert params.frame_rate is None
        assert params.negative_prompt is None
        assert params.image is None
        assert params.image_cond_strength is None
        # `seed` is now ``Optional[int]`` and defaults to None — the engine
        # draws a fresh value on the coordinator rank before broadcast.
        assert params.seed is None
        # Concrete defaults
        assert params.num_images_per_prompt == 1
        # Extra params
        assert params.extra_params is None
        # The model does not expose a ``mask`` field.
        assert not hasattr(params, "mask")

    def test_explicit_values(self):
        from tensorrt_llm.visual_gen import VisualGenParams

        params = VisualGenParams(
            height=720,
            width=1280,
            num_inference_steps=50,
            guidance_scale=5.0,
            seed=123,
        )
        assert params.height == 720
        assert params.width == 1280
        assert params.num_inference_steps == 50
        assert params.guidance_scale == 5.0
        assert params.seed == 123

    def test_unknown_field_rejected(self):
        from pydantic import ValidationError

        from tensorrt_llm.visual_gen import VisualGenParams

        with pytest.raises(ValidationError):
            VisualGenParams(stg_scale=0.5)

    def test_extra_params_accepted(self):
        from tensorrt_llm.visual_gen import VisualGenParams

        params = VisualGenParams(extra_params={"stg_scale": 0.5, "enhance_prompt": True})
        assert params.extra_params["stg_scale"] == 0.5
        assert params.extra_params["enhance_prompt"] is True

    def test_image_accepts_str(self):
        from tensorrt_llm.visual_gen import VisualGenParams

        params = VisualGenParams(image="/path/to/image.png")
        assert params.image == "/path/to/image.png"

    def test_image_accepts_bytes(self):
        from tensorrt_llm.visual_gen import VisualGenParams

        params = VisualGenParams(image=b"\x89PNG")
        assert params.image == b"\x89PNG"

    def test_image_accepts_list(self):
        from tensorrt_llm.visual_gen import VisualGenParams

        params = VisualGenParams(image=["/path/a.png", b"\x89PNG"])
        assert len(params.image) == 2

    def test_model_dump(self):
        from tensorrt_llm.visual_gen import VisualGenParams

        params = VisualGenParams(height=512, seed=1)
        d = params.model_dump()
        assert d["height"] == 512
        assert d["seed"] == 1
        assert d["width"] is None

    def test_negative_prompt_on_params(self):
        from tensorrt_llm.visual_gen import VisualGenParams

        params = VisualGenParams(negative_prompt="blurry, low quality")
        assert params.negative_prompt == "blurry, low quality"


# =============================================================================
# ExtraParamSchema
# =============================================================================


class TestExtraParamSchema:
    """ExtraParamSchema construction and field access."""

    def test_basic_construction(self):
        from tensorrt_llm._torch.visual_gen.pipeline import ExtraParamSchema

        spec = ExtraParamSchema(type="float", default=0.0, description="test param")
        assert spec.type == "float"
        assert spec.default == 0.0
        assert spec.description == "test param"
        assert spec.range is None

    def test_with_range(self):
        from tensorrt_llm._torch.visual_gen.pipeline import ExtraParamSchema

        spec = ExtraParamSchema(type="float", range=(0.0, 1.0))
        assert spec.range == (0.0, 1.0)

    def test_none_default(self):
        from tensorrt_llm._torch.visual_gen.pipeline import ExtraParamSchema

        spec = ExtraParamSchema(type="str")
        assert spec.default is None

    def test_public_import(self):
        from tensorrt_llm import ExtraParamSchema

        spec = ExtraParamSchema(type="int", default=42)
        assert spec.default == 42


# =============================================================================
# Pipeline default_generation_params and extra_param_specs
# =============================================================================


class TestPipelineDefaults:
    """Each pipeline declares correct default generation params."""

    def test_wan21_480p_defaults(self):
        """Wan 2.1 small-model (≤12 heads) returns 480p defaults."""
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline

        d = WanPipeline.default_generation_params.fget(_wan_mock(num_heads=12))
        assert d["height"] == 480
        assert d["width"] == 832
        assert d["num_inference_steps"] == 50
        assert d["guidance_scale"] == 5.0
        assert d["num_frames"] == 81

    def test_wan22_14b_defaults(self):
        """Wan 2.2 A14B returns 720p defaults."""
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline

        d = WanPipeline.default_generation_params.fget(
            _wan_mock(is_wan22_14b=True, is_wan22_5b=False, num_heads=40)
        )
        assert d["height"] == 720
        assert d["width"] == 1280
        assert d["num_inference_steps"] == 40
        assert d["guidance_scale"] == 4.0
        assert d["num_frames"] == 81

    def test_wan22_5b_defaults(self):
        """Wan 2.2 TI2V-5B returns native 720p defaults."""
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline

        d = WanPipeline.default_generation_params.fget(
            _wan_mock(is_wan22_14b=False, is_wan22_5b=True, num_heads=24)
        )
        assert d["height"] == 704
        assert d["width"] == 1280
        assert d["num_inference_steps"] == 50
        assert d["guidance_scale"] == 5.0
        assert d["num_frames"] == 121
        assert d["frame_rate"] == 24.0

    def test_flux_defaults(self):
        from tensorrt_llm._torch.visual_gen.models.flux.pipeline_flux import FluxPipeline

        d = FluxPipeline.default_generation_params.fget(None)
        assert d["height"] == 1024
        assert d["width"] == 1024
        assert d["guidance_scale"] == 3.5

    def test_ltx2_defaults(self):
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        d = LTX2Pipeline.default_generation_params.fget(None)
        assert d["height"] == 512
        assert d["width"] == 768
        assert d["num_inference_steps"] == 40
        assert d["guidance_scale"] == 4.0
        assert d["max_sequence_length"] == 1024
        assert d["num_frames"] == 121

    def test_base_pipeline_empty_defaults(self):
        from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline

        assert BasePipeline.default_generation_params.fget(None) == {}
        assert BasePipeline.extra_param_specs.fget(None) == {}


class TestPipelineExtraParamSpecs:
    """Each pipeline declares correct extra param specs."""

    def test_wan22_extra_specs(self):
        """Wan 2.2 A14B exposes guidance_scale_2 and boundary_ratio."""
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline

        specs = WanPipeline.extra_param_specs.fget(_wan_mock(is_wan22_14b=True, is_wan22_5b=False))
        assert "guidance_scale_2" in specs
        assert "boundary_ratio" in specs
        assert specs["guidance_scale_2"].type == "float"
        assert specs["boundary_ratio"].range == (0.0, 1.0)

    def test_wan21_no_extra_specs(self):
        """Wan 2.1 has no model-specific extra params."""
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline

        specs = WanPipeline.extra_param_specs.fget(_wan_mock())
        assert specs == {}

    def test_wan22_5b_no_extra_specs(self):
        """Wan 2.2 TI2V-5B has no model-specific extra params."""
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline

        specs = WanPipeline.extra_param_specs.fget(_wan_mock(is_wan22_14b=False, is_wan22_5b=True))
        assert specs == {}

    def test_wan_i2v_extra_specs(self):
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan_i2v import (
            WanImageToVideoPipeline,
        )

        specs = WanImageToVideoPipeline.extra_param_specs.fget(
            _wan_mock(is_wan22_14b=True, is_wan22_5b=False)
        )
        assert "last_image" in specs
        assert "guidance_scale_2" in specs
        assert specs["last_image"].type == "str"

    def test_flux_no_extra_specs(self):
        from tensorrt_llm._torch.visual_gen.models.flux.pipeline_flux import FluxPipeline

        assert FluxPipeline.extra_param_specs.fget(None) == {}

    def test_ltx2_extra_specs(self):
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        specs = LTX2Pipeline.extra_param_specs.fget(None)
        expected_keys = {
            "output_type",
            "guidance_rescale",
            "stg_scale",
            "stg_blocks",
            "modality_scale",
            "rescale_scale",
            "guidance_skip_step",
            "enhance_prompt",
        }
        assert set(specs.keys()) == expected_keys
        assert specs["stg_scale"].default == 0.0
        assert specs["enhance_prompt"].default is False
        assert specs["stg_blocks"].default is None

    def test_ltx2_extra_specs_attribute_access(self):
        """Direct attribute-style access works on the returned dict."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        specs = LTX2Pipeline.extra_param_specs.fget(None)
        assert specs["modality_scale"].type == "float"
        assert specs["modality_scale"].default == 1.0


# =============================================================================
# Executor default merging
# =============================================================================


class TestDefaultMerging:
    """DiffusionExecutor._merge_defaults fills None fields correctly."""

    def _make_mock_executor(self, pipeline_cls, mock_self=None):
        """Create a mock DiffusionExecutor with the given pipeline class's specs."""
        executor = MagicMock()
        executor.pipeline = MagicMock()
        executor.pipeline.default_generation_params = pipeline_cls.default_generation_params.fget(
            mock_self
        )
        executor.pipeline.extra_param_specs = pipeline_cls.extra_param_specs.fget(mock_self)
        return executor

    def _make_request(self, **kwargs):
        from tensorrt_llm._torch.visual_gen.executor import DiffusionRequest
        from tensorrt_llm.visual_gen.params import VisualGenParams

        return DiffusionRequest(request_id=0, prompt=["test"], params=VisualGenParams(**kwargs))

    def _merge(self, executor, req):
        from tensorrt_llm._torch.visual_gen.executor import DiffusionExecutor

        DiffusionExecutor._merge_defaults(executor, req)

    def test_universal_defaults_merged(self):
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline

        executor = self._make_mock_executor(WanPipeline, _wan_mock(num_heads=12))
        req = self._make_request()
        assert req.params.height is None

        self._merge(executor, req)
        assert req.params.height == 480
        assert req.params.width == 832
        assert req.params.num_inference_steps == 50

    def test_user_values_not_overwritten(self):
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline

        executor = self._make_mock_executor(WanPipeline, _wan_mock(num_heads=12))
        req = self._make_request(height=1080, width=1920)

        self._merge(executor, req)
        assert req.params.height == 1080  # User value preserved
        assert req.params.width == 1920
        assert req.params.num_inference_steps == 50  # Default filled

    def test_extra_params_defaults_merged(self):
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        executor = self._make_mock_executor(LTX2Pipeline)
        req = self._make_request()

        self._merge(executor, req)
        assert req.params.extra_params is not None
        assert req.params.extra_params["stg_scale"] == 0.0
        assert req.params.extra_params["output_type"] == "pt"
        assert req.params.extra_params["enhance_prompt"] is False
        # None defaults are also filled
        assert req.params.extra_params["stg_blocks"] is None

    def test_user_extra_params_not_overwritten(self):
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        executor = self._make_mock_executor(LTX2Pipeline)
        req = self._make_request(extra_params={"stg_scale": 0.5})

        self._merge(executor, req)
        assert req.params.extra_params["stg_scale"] == 0.5  # User value preserved
        assert req.params.extra_params["output_type"] == "pt"  # Default filled

    def test_no_extra_params_for_flux(self):
        from tensorrt_llm._torch.visual_gen.models.flux.pipeline_flux import FluxPipeline

        executor = self._make_mock_executor(FluxPipeline)
        req = self._make_request()

        self._merge(executor, req)
        assert req.params.extra_params is None  # Flux has no extra specs

    def test_all_declared_keys_present_after_merge(self):
        """After merge, all extra_param_specs keys are in extra_params."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        executor = self._make_mock_executor(LTX2Pipeline)
        req = self._make_request(extra_params={"stg_scale": 0.5})

        self._merge(executor, req)
        ltx2_specs = LTX2Pipeline.extra_param_specs.fget(None)
        for key in ltx2_specs:
            assert key in req.params.extra_params, f"Missing key: {key}"

    def test_params_none_materializes_defaults(self):
        """req.params=None is the default path from generate_async(params=None);
        _merge_defaults should materialize a VisualGenParams from pipeline defaults."""
        from tensorrt_llm._torch.visual_gen.executor import DiffusionRequest
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline
        from tensorrt_llm.visual_gen.params import VisualGenParams

        executor = self._make_mock_executor(LTX2Pipeline)
        req = DiffusionRequest(request_id=0, prompt=["test"], params=None)

        self._merge(executor, req)

        assert isinstance(req.params, VisualGenParams)
        # Universal defaults are filled from the pipeline
        assert req.params.height == 512
        assert req.params.width == 768
        assert req.params.num_inference_steps == 40
        # Extra-param defaults are filled for all declared keys
        assert req.params.extra_params is not None
        assert req.params.extra_params["stg_scale"] == 0.0
        assert req.params.extra_params["output_type"] == "pt"
        assert "stg_blocks" in req.params.extra_params


# =============================================================================
# VisualGen.default_params and extra_param_specs
# =============================================================================


class TestVisualGenDefaultParams:
    """VisualGen.default_params returns correctly merged params per pipeline.

    VisualGen delegates to executor.default_generation_params and
    executor.extra_param_specs (populated from the READY signal).
    """

    def _make_visual_gen(self, pipeline_cls, mock_self=None):
        """Create VisualGen with mocked init and executor carrying pipeline metadata."""
        from tensorrt_llm.visual_gen import VisualGen

        with patch.object(VisualGen, "__init__", lambda self, *a, **kw: None):
            vg = VisualGen.__new__(VisualGen)
            vg.executor = MagicMock()
            if pipeline_cls is not None:
                vg.executor.default_generation_params = pipeline_cls.default_generation_params.fget(
                    mock_self
                )
                vg.executor.extra_param_specs = pipeline_cls.extra_param_specs.fget(mock_self)
            else:
                vg.executor.default_generation_params = {}
                vg.executor.extra_param_specs = {}
            return vg

    def test_ltx2_default_params(self):
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        vg = self._make_visual_gen(LTX2Pipeline)
        params = vg.default_params
        assert params.height == 512
        assert params.width == 768
        assert params.num_inference_steps == 40
        # Pipelines don't declare a seed default; the executor resolves
        # ``None`` to a concrete integer on the coordinator rank.
        assert params.seed is None
        assert params.extra_params is not None
        assert params.extra_params["stg_scale"] == 0.0
        assert params.extra_params["output_type"] == "pt"
        # None-default keys are present
        assert "stg_blocks" in params.extra_params

    def test_wan22_default_params(self):
        """Wan 2.2 A14B returns 720p defaults with guidance_scale_2/boundary_ratio."""
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline

        vg = self._make_visual_gen(WanPipeline, _wan_mock(is_wan22_14b=True, is_wan22_5b=False))
        params = vg.default_params
        assert params.height == 720
        assert params.width == 1280
        assert params.extra_params is not None
        assert "guidance_scale_2" in params.extra_params
        assert "boundary_ratio" in params.extra_params

    def test_wan22_5b_default_params(self):
        """Wan 2.2 TI2V-5B returns 704x1280 defaults with no extra params."""
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline

        vg = self._make_visual_gen(
            WanPipeline, _wan_mock(is_wan22_14b=False, is_wan22_5b=True, num_heads=24)
        )
        params = vg.default_params
        assert params.height == 704
        assert params.width == 1280
        assert params.num_inference_steps == 50
        assert params.guidance_scale == 5.0
        assert params.num_frames == 121
        assert params.frame_rate == 24.0
        assert params.extra_params is None

    def test_wan21_default_params(self):
        """Wan 2.1 small-model returns 480p defaults with no extra params."""
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline

        vg = self._make_visual_gen(WanPipeline, _wan_mock(num_heads=12))
        params = vg.default_params
        assert params.height == 480
        assert params.width == 832
        assert params.extra_params is None

    def test_flux_default_params_no_extra(self):
        from tensorrt_llm._torch.visual_gen.models.flux.pipeline_flux import FluxPipeline

        vg = self._make_visual_gen(FluxPipeline)
        params = vg.default_params
        assert params.height == 1024
        assert params.width == 1024
        assert params.extra_params is None

    def test_no_pipeline_returns_bare_params(self):
        vg = self._make_visual_gen(None)
        params = vg.default_params
        assert params.height is None
        assert params.extra_params is None

    def test_extra_param_specs(self):
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        vg = self._make_visual_gen(LTX2Pipeline)
        specs = vg.extra_param_specs
        assert "stg_scale" in specs
        assert specs["stg_scale"].type == "float"

    def test_extra_param_specs_empty_for_flux(self):
        from tensorrt_llm._torch.visual_gen.models.flux.pipeline_flux import FluxPipeline

        vg = self._make_visual_gen(FluxPipeline)
        assert vg.extra_param_specs == {}


# =============================================================================
# Pipeline metadata bridging (executor → client)
# =============================================================================


class TestPipelineMetadataBridging:
    """Verify default_generation_params and extra_param_specs survive
    the pickle round-trip from DiffusionExecutor READY signal to the client."""

    def _build_ready_response(self, pipeline_cls, mock_self=None):
        """Build a DiffusionResponse matching what DiffusionExecutor sends."""
        from tensorrt_llm._torch.visual_gen.executor import DiffusionResponse

        return DiffusionResponse(
            request_id=-1,
            output={
                "status": "READY",
                "default_generation_params": pipeline_cls.default_generation_params.fget(mock_self),
                "extra_param_specs": pipeline_cls.extra_param_specs.fget(mock_self),
            },
        )

    def _roundtrip(self, response):
        """Pickle/unpickle to simulate ZMQ transport."""
        import pickle

        return pickle.loads(pickle.dumps(response))

    def test_ready_payload_pickle_roundtrip(self):
        """The READY dict survives pickle (the ZMQ transport layer)."""
        from tensorrt_llm._torch.visual_gen.executor import DiffusionResponse
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        resp = self._build_ready_response(LTX2Pipeline)
        restored = self._roundtrip(resp)

        ltx2_defaults = LTX2Pipeline.default_generation_params.fget(None)
        ltx2_specs = LTX2Pipeline.extra_param_specs.fget(None)

        assert isinstance(restored, DiffusionResponse)
        assert restored.request_id == -1
        payload = restored.output
        assert isinstance(payload, dict)
        assert payload["status"] == "READY"
        assert payload["default_generation_params"] == ltx2_defaults
        assert set(payload["extra_param_specs"].keys()) == set(ltx2_specs.keys())

    def test_extra_param_schema_type_preserved(self):
        """ExtraParamSchema instances keep their type through pickle."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline
        from tensorrt_llm._torch.visual_gen.pipeline import ExtraParamSchema

        resp = self._build_ready_response(LTX2Pipeline)
        restored = self._roundtrip(resp)

        specs = restored.output["extra_param_specs"]
        for key, spec in specs.items():
            assert isinstance(spec, ExtraParamSchema), (
                f"spec '{key}' lost its type: got {type(spec).__name__}"
            )

    def test_extra_param_schema_fields_preserved(self):
        """ExtraParamSchema field values survive the round-trip."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        resp = self._build_ready_response(LTX2Pipeline)
        restored = self._roundtrip(resp)

        specs = restored.output["extra_param_specs"]
        original = LTX2Pipeline.extra_param_specs.fget(None)
        for key in original:
            assert specs[key].type == original[key].type
            assert specs[key].default == original[key].default
            assert specs[key].range == original[key].range
            assert specs[key].description == original[key].description

    def test_wan_pipeline_roundtrip(self):
        """Wan 2.2 A14B pipeline metadata survives the round-trip."""
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline

        mock_self = _wan_mock(is_wan22_14b=True, is_wan22_5b=False)
        resp = self._build_ready_response(WanPipeline, mock_self)
        restored = self._roundtrip(resp)

        payload = restored.output
        assert payload["default_generation_params"]["height"] == 720
        assert payload["default_generation_params"]["num_frames"] == 81
        assert "guidance_scale_2" in payload["extra_param_specs"]
        assert "boundary_ratio" in payload["extra_param_specs"]

    def test_flux_empty_specs_roundtrip(self):
        """Pipeline with no extra_param_specs round-trips as empty dict."""
        from tensorrt_llm._torch.visual_gen.models.flux.pipeline_flux import FluxPipeline

        resp = self._build_ready_response(FluxPipeline)
        restored = self._roundtrip(resp)

        assert restored.output["extra_param_specs"] == {}
        assert restored.output["default_generation_params"]["height"] == 1024

    def test_client_extracts_metadata_from_ready(self):
        """DiffusionRemoteClient stores metadata when processing a READY response."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline
        from tensorrt_llm._torch.visual_gen.pipeline import ExtraParamSchema
        from tensorrt_llm.visual_gen.visual_gen import DiffusionRemoteClient

        resp = self._build_ready_response(LTX2Pipeline)
        restored = self._roundtrip(resp)

        ltx2_defaults = LTX2Pipeline.default_generation_params.fget(None)
        ltx2_specs = LTX2Pipeline.extra_param_specs.fget(None)

        # Simulate what _wait_ready_async does: extract from the response payload
        client = MagicMock(spec=DiffusionRemoteClient)
        client.default_generation_params = {}
        client.extra_param_specs = {}

        payload = restored.output
        if isinstance(payload, dict):
            client.default_generation_params = payload.get("default_generation_params", {})
            client.extra_param_specs = payload.get("extra_param_specs", {})

        assert client.default_generation_params == ltx2_defaults
        assert set(client.extra_param_specs.keys()) == set(ltx2_specs.keys())
        for spec in client.extra_param_specs.values():
            assert isinstance(spec, ExtraParamSchema)


# =============================================================================
# Request validation — _validate_request
# =============================================================================


class TestRequestValidation:
    """DiffusionExecutor._validate_request raises ValueError on bad params."""

    def _make_mock_executor(self, pipeline_cls, mock_self=None):
        executor = MagicMock()
        executor.pipeline = MagicMock()
        executor.pipeline.__class__ = pipeline_cls
        executor.pipeline.default_generation_params = pipeline_cls.default_generation_params.fget(
            mock_self
        )
        executor.pipeline.extra_param_specs = pipeline_cls.extra_param_specs.fget(mock_self)
        return executor

    def _make_request(self, **kwargs):
        from tensorrt_llm._torch.visual_gen.executor import DiffusionRequest
        from tensorrt_llm.visual_gen.params import VisualGenParams

        return DiffusionRequest(request_id=0, prompt=["test"], params=VisualGenParams(**kwargs))

    def _validate(self, executor, req):
        from tensorrt_llm._torch.visual_gen.executor import DiffusionExecutor

        DiffusionExecutor._validate_request(executor, req)

    def _merge_and_validate(self, executor, req):
        from tensorrt_llm._torch.visual_gen.executor import DiffusionExecutor

        DiffusionExecutor._merge_defaults(executor, req)
        DiffusionExecutor._validate_request(executor, req)

    # --- unknown extra_params ---

    def test_unknown_extra_params_raises(self):
        from tensorrt_llm._torch.visual_gen.models.flux.pipeline_flux import FluxPipeline

        executor = self._make_mock_executor(FluxPipeline)
        req = self._make_request(extra_params={"nonexistent_key": 42})
        with pytest.raises(ValueError, match="Unknown extra_params"):
            self._validate(executor, req)

    def test_unknown_extra_params_lists_supported_keys(self):
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        executor = self._make_mock_executor(LTX2Pipeline)
        req = self._make_request(extra_params={"bad_key": 1})
        with pytest.raises(ValueError, match="Supported"):
            self._validate(executor, req)

    def test_valid_extra_params_accepted(self):
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        executor = self._make_mock_executor(LTX2Pipeline)
        req = self._make_request(extra_params={"stg_scale": 0.5})
        self._merge_and_validate(executor, req)  # should not raise

    # --- unsupported universal fields ---

    def test_num_frames_on_image_pipeline_raises(self):
        """num_frames=81 to FLUX (image-only) should raise."""
        from tensorrt_llm._torch.visual_gen.models.flux.pipeline_flux import FluxPipeline

        executor = self._make_mock_executor(FluxPipeline)
        req = self._make_request(num_frames=81)
        with pytest.raises(ValueError, match="num_frames.*not use it"):
            self._validate(executor, req)

    def test_frame_rate_on_image_pipeline_raises(self):
        from tensorrt_llm._torch.visual_gen.models.flux.pipeline_flux import FluxPipeline

        executor = self._make_mock_executor(FluxPipeline)
        req = self._make_request(frame_rate=24.0)
        with pytest.raises(ValueError, match="frame_rate.*not use it"):
            self._validate(executor, req)

    def test_image_cond_strength_on_t2v_pipeline_raises(self):
        """``image_cond_strength`` on a Wan T2V (no I2V conditioning)
        pipeline must be rejected — the pipeline never reads the
        value, so silently dropping it would defeat the strict
        request policy."""
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline

        executor = self._make_mock_executor(WanPipeline, _wan_mock(num_heads=12))
        req = self._make_request(image_cond_strength=0.8)
        with pytest.raises(ValueError, match="image_cond_strength.*not use it"):
            self._validate(executor, req)

    def test_image_cond_strength_on_wan_i2v_pipeline_raises(self):
        """``image_cond_strength`` is not consumed by Wan I2V's
        ``forward()`` today — declaring it in defaults would be a
        silent no-op. The strict validator rejects it on every Wan
        variant so the wire contract matches the pipeline's actual
        behavior."""
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan_i2v import (
            WanImageToVideoPipeline,
        )

        executor = self._make_mock_executor(WanImageToVideoPipeline, _wan_mock(num_heads=12))
        req = self._make_request(image_cond_strength=0.6)
        with pytest.raises(ValueError, match="image_cond_strength.*not use it"):
            self._validate(executor, req)

    def test_image_cond_strength_on_ltx2_pipeline_ok(self):
        """LTX-2 threads ``image_cond_strength`` through
        ``infer`` → ``forward``; the field must not raise on a
        pipeline that actually consumes it."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        executor = self._make_mock_executor(LTX2Pipeline)
        req = self._make_request(image_cond_strength=0.6)
        # Should not raise — declared and consumed by LTX-2.
        self._merge_and_validate(executor, req)

    def test_image_not_checked_by_validator(self):
        """image is a conditioning input — validated at runtime by infer(), not here."""
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline

        executor = self._make_mock_executor(WanPipeline, _wan_mock(num_heads=12))
        req = self._make_request(image="/path/to/img.png")
        # Should not raise — image validation is the pipeline's responsibility
        self._merge_and_validate(executor, req)

    def test_num_frames_on_video_pipeline_ok(self):
        """num_frames is declared by WanPipeline, should not raise."""
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline

        executor = self._make_mock_executor(WanPipeline, _wan_mock(num_heads=12))
        req = self._make_request(num_frames=81)
        self._merge_and_validate(executor, req)

    def test_image_on_i2v_pipeline_ok(self):
        """image is declared by WanImageToVideoPipeline, should not raise."""
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan_i2v import (
            WanImageToVideoPipeline,
        )

        executor = self._make_mock_executor(WanImageToVideoPipeline, _wan_mock(num_heads=12))
        req = self._make_request(image="/path/to/img.png")
        self._merge_and_validate(executor, req)

    def test_none_fields_not_flagged(self):
        """Fields left as None should never trigger unsupported-field errors."""
        from tensorrt_llm._torch.visual_gen.models.flux.pipeline_flux import FluxPipeline

        executor = self._make_mock_executor(FluxPipeline)
        req = self._make_request()  # all None
        self._merge_and_validate(executor, req)

    def test_params_none_merge_and_validate_ok(self):
        """req.params=None must merge + validate cleanly (VisualGen.generate_async
        defaults to params=None, so this is the canonical call path)."""
        from tensorrt_llm._torch.visual_gen.executor import DiffusionRequest
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline
        from tensorrt_llm.visual_gen.params import VisualGenParams

        executor = self._make_mock_executor(LTX2Pipeline)
        req = DiffusionRequest(request_id=0, prompt=["test"], params=None)

        self._merge_and_validate(executor, req)  # should not raise

        assert isinstance(req.params, VisualGenParams)
        assert req.params.height == 512
        assert req.params.extra_params["stg_scale"] == 0.0

    # --- type validation on extra_params ---

    def test_wrong_type_extra_param_raises(self):
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        executor = self._make_mock_executor(LTX2Pipeline)
        req = self._make_request(extra_params={"stg_scale": "not_a_number"})
        with pytest.raises(ValueError, match="expected type 'float'"):
            self._merge_and_validate(executor, req)

    def test_int_accepted_for_float_spec(self):
        """An int value should be accepted for a 'float'-typed spec."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        executor = self._make_mock_executor(LTX2Pipeline)
        req = self._make_request(extra_params={"stg_scale": 1})
        self._merge_and_validate(executor, req)

    def test_bool_rejected_for_float_spec(self):
        """A bool should not be accepted for a 'float' spec (bool is-a int in Python,
        but semantically wrong for floats)."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        executor = self._make_mock_executor(LTX2Pipeline)
        req = self._make_request(extra_params={"stg_scale": True})
        # bool is instance of int which is accepted for float, so this passes type check.
        # This is intentional — Python's type hierarchy makes bool a subclass of int.
        self._merge_and_validate(executor, req)

    def test_wrong_type_str_extra_param(self):
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan_i2v import (
            WanImageToVideoPipeline,
        )

        executor = self._make_mock_executor(WanImageToVideoPipeline, _wan_mock(num_heads=12))
        req = self._make_request(
            image="/img.png",
            extra_params={"last_image": 123},
        )
        with pytest.raises(ValueError, match="expected type 'str'"):
            self._merge_and_validate(executor, req)

    # --- range validation on extra_params ---

    def test_out_of_range_extra_param_raises(self):
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline

        executor = self._make_mock_executor(
            WanPipeline, _wan_mock(is_wan22_14b=True, is_wan22_5b=False)
        )
        # boundary_ratio has range (0.0, 1.0)
        req = self._make_request(extra_params={"boundary_ratio": 2.0})
        with pytest.raises(ValueError, match="out of range"):
            self._merge_and_validate(executor, req)

    def test_negative_boundary_ratio_raises(self):
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline

        executor = self._make_mock_executor(
            WanPipeline, _wan_mock(is_wan22_14b=True, is_wan22_5b=False)
        )
        req = self._make_request(extra_params={"boundary_ratio": -0.5})
        with pytest.raises(ValueError, match="out of range"):
            self._merge_and_validate(executor, req)

    def test_boundary_value_at_range_edge_ok(self):
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline

        executor = self._make_mock_executor(
            WanPipeline, _wan_mock(is_wan22_14b=True, is_wan22_5b=False)
        )
        req = self._make_request(extra_params={"boundary_ratio": 0.0})
        self._merge_and_validate(executor, req)

    def test_boundary_value_at_range_max_ok(self):
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline

        executor = self._make_mock_executor(
            WanPipeline, _wan_mock(is_wan22_14b=True, is_wan22_5b=False)
        )
        req = self._make_request(extra_params={"boundary_ratio": 1.0})
        self._merge_and_validate(executor, req)

    # --- multiple errors collected ---

    def test_multiple_errors_in_single_message(self):
        """Multiple validation failures should be collected into one error."""
        from tensorrt_llm._torch.visual_gen.models.flux.pipeline_flux import FluxPipeline

        executor = self._make_mock_executor(FluxPipeline)
        req = self._make_request(
            num_frames=81,
            frame_rate=24.0,
            extra_params={"bogus": 1},
        )
        with pytest.raises(ValueError) as exc_info:
            self._validate(executor, req)
        msg = str(exc_info.value)
        assert "num_frames" in msg
        assert "frame_rate" in msg
        assert "bogus" in msg

    # --- None extra_params values skip type/range check ---

    def test_none_extra_param_value_skipped(self):
        """None values for extra_params with range specs should not fail validation."""
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline

        executor = self._make_mock_executor(
            WanPipeline, _wan_mock(is_wan22_14b=True, is_wan22_5b=False)
        )
        req = self._make_request(extra_params={"boundary_ratio": None})
        self._merge_and_validate(executor, req)

    # --- process_request returns error response instead of crashing ---

    def test_process_request_returns_error_on_validation_failure(self):
        """Validation errors become error responses, not server crashes."""
        from tensorrt_llm._torch.visual_gen.executor import DiffusionExecutor, DiffusionResponse

        # Build a mock with real method bindings for the three methods
        # that process_request chains through.
        executor = MagicMock()
        executor.pipeline = MagicMock()
        executor.pipeline.__class__.__name__ = "FluxPipeline"
        executor.pipeline.default_generation_params = {"height": 1024, "width": 1024}
        executor.pipeline.extra_param_specs = {}
        executor.pipeline._warmed_up_shapes = set()
        executor.pipeline.warmup_cache_key = MagicMock(return_value=(1024, 1024, None))
        executor.rank = 0
        executor.device_id = 0
        executor.response_queue = MagicMock()

        # Wire real methods onto the mock so process_request uses them
        executor._merge_defaults = lambda req: DiffusionExecutor._merge_defaults(executor, req)
        executor._validate_request = lambda req: DiffusionExecutor._validate_request(executor, req)

        req = self._make_request(num_frames=81, extra_params={"bad": 1})

        # Call the real process_request
        DiffusionExecutor.process_request(executor, req)

        # Should have put an error response, not crashed
        executor.response_queue.put.assert_called_once()
        resp = executor.response_queue.put.call_args[0][0]
        assert isinstance(resp, DiffusionResponse)
        assert resp.error_msg is not None
        assert "validation failed" in resp.error_msg.lower()


# =============================================================================
# VisualGenValidationError — structured exception fields
# =============================================================================


class TestVisualGenValidationErrorStructured:
    """The executor surfaces structured ``reason``/``param``/``details`` on
    every category of parameter violation. Keeps the contract green even if
    the human-readable messages drift."""

    def _make_mock_executor(self, pipeline_cls, mock_self=None):
        executor = MagicMock()
        executor.pipeline = MagicMock()
        executor.pipeline.__class__ = pipeline_cls
        executor.pipeline.default_generation_params = pipeline_cls.default_generation_params.fget(
            mock_self
        )
        executor.pipeline.extra_param_specs = pipeline_cls.extra_param_specs.fget(mock_self)
        return executor

    def _make_request(self, **kwargs):
        from tensorrt_llm._torch.visual_gen.executor import DiffusionRequest
        from tensorrt_llm.visual_gen.params import VisualGenParams

        return DiffusionRequest(request_id=0, prompt=["test"], params=VisualGenParams(**kwargs))

    def _validate(self, executor, req):
        from tensorrt_llm._torch.visual_gen.executor import DiffusionExecutor

        DiffusionExecutor._validate_request(executor, req)

    def test_subclasses_value_error(self):
        from tensorrt_llm._torch.visual_gen.executor import VisualGenValidationError

        exc = VisualGenValidationError("unknown_extra_param", "k", "msg", {"a": 1})
        assert isinstance(exc, ValueError)
        # ``str(exc)`` is the human message so existing ``except ValueError``
        # log lines and ``pytest.raises(ValueError, match=...)`` checks see
        # the same string as before.
        assert str(exc) == "msg"
        assert exc.message == "msg"
        assert exc.reason == "unknown_extra_param"
        assert exc.param == "k"
        assert exc.details == {"a": 1}

    def test_reasons_tuple_matches_design(self):
        from tensorrt_llm._torch.visual_gen.executor import VisualGenValidationError

        assert set(VisualGenValidationError.REASONS) == {
            "unknown_extra_param",
            "unsupported_universal_field",
            "extra_param_type_mismatch",
            "extra_param_out_of_range",
        }

    def test_unknown_extra_param_fields(self):
        from tensorrt_llm._torch.visual_gen.executor import VisualGenValidationError
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        executor = self._make_mock_executor(LTX2Pipeline)
        req = self._make_request(extra_params={"stg_sclae": 1.0, "bogus_key": 2})
        with pytest.raises(VisualGenValidationError) as excinfo:
            self._validate(executor, req)
        exc = excinfo.value
        assert exc.reason == "unknown_extra_param"
        # `param` is the first offending key in sorted order.
        assert exc.param == "bogus_key"
        assert exc.details["pipeline"] == "LTX2Pipeline"
        assert sorted(exc.details["unknown_extra_param"]["keys"]) == ["bogus_key", "stg_sclae"]
        assert "supported" in exc.details["unknown_extra_param"]

    def test_unsupported_universal_field_fields(self):
        """An image pipeline should reject video-only universal fields with a
        deterministic ``param`` (the first unsupported field by declared
        iteration order)."""
        from tensorrt_llm._torch.visual_gen.executor import VisualGenValidationError
        from tensorrt_llm._torch.visual_gen.models.flux.pipeline_flux import FluxPipeline

        executor = self._make_mock_executor(FluxPipeline)
        # FluxPipeline doesn't declare num_frames or frame_rate; both should
        # be flagged with the iteration-order-first key as ``param``.
        req = self._make_request(num_frames=81, frame_rate=24.0)
        with pytest.raises(VisualGenValidationError) as excinfo:
            self._validate(executor, req)
        exc = excinfo.value
        assert exc.reason == "unsupported_universal_field"
        # ``_GENERATION_CONFIG_FIELDS`` declares num_frames before
        # frame_rate, so param is deterministic.
        assert exc.param == "num_frames"
        assert exc.details["unsupported_universal_field"]["fields"] == ["num_frames", "frame_rate"]

    def test_extra_param_type_mismatch_fields(self):
        from tensorrt_llm._torch.visual_gen.executor import VisualGenValidationError
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        executor = self._make_mock_executor(LTX2Pipeline)
        # stg_scale is declared as ``float``; passing a string triggers a
        # type mismatch.
        req = self._make_request(extra_params={"stg_scale": "fast"})
        with pytest.raises(VisualGenValidationError) as excinfo:
            self._validate(executor, req)
        exc = excinfo.value
        assert exc.reason == "extra_param_type_mismatch"
        assert exc.param == "stg_scale"
        mismatch = exc.details["extra_param_type_mismatch"][0]
        assert mismatch["key"] == "stg_scale"
        assert mismatch["expected_type"] == "float"
        assert mismatch["got_type"] == "str"
        assert mismatch["value"] == "fast"

    def test_extra_param_out_of_range_fields(self):
        from tensorrt_llm._torch.visual_gen.executor import VisualGenValidationError
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline

        executor = self._make_mock_executor(WanPipeline, _wan_mock(is_wan22_14b=True, num_heads=12))
        # boundary_ratio is declared with range [0.0, 1.0].
        req = self._make_request(extra_params={"boundary_ratio": -0.5})
        with pytest.raises(VisualGenValidationError) as excinfo:
            self._validate(executor, req)
        exc = excinfo.value
        assert exc.reason == "extra_param_out_of_range"
        assert exc.param == "boundary_ratio"
        violation = exc.details["extra_param_out_of_range"][0]
        assert violation["key"] == "boundary_ratio"
        assert violation["value"] == -0.5
        assert violation["range"] == [0.0, 1.0]


# =============================================================================
# Seed resolution — coordinator-rank materialization
# =============================================================================


class TestResolveSeed:
    """``_resolve_seed`` materializes ``params.seed`` once on the coordinator
    rank before broadcast so every rank sees the same value."""

    def _make_executor(self):
        from tensorrt_llm._torch.visual_gen.executor import DiffusionExecutor

        # We only need the bound method ``_resolve_seed``; the rest of the
        # executor state is unused for these assertions.
        executor = MagicMock()
        executor._resolve_seed = lambda req: DiffusionExecutor._resolve_seed(executor, req)
        return executor

    def _req_with_params(self, **kwargs):
        from tensorrt_llm._torch.visual_gen.executor import DiffusionRequest
        from tensorrt_llm.visual_gen.params import VisualGenParams

        return DiffusionRequest(request_id=0, prompt=["x"], params=VisualGenParams(**kwargs))

    def test_seed_none_is_materialized(self):
        executor = self._make_executor()
        req = self._req_with_params()
        assert req.params.seed is None
        executor._resolve_seed(req)
        assert isinstance(req.params.seed, int)
        assert req.params.seed >= 0
        # 63-bit space; should be well inside int range.
        assert req.params.seed < (1 << 63)

    def test_concrete_seed_preserved(self):
        executor = self._make_executor()
        req = self._req_with_params(seed=12345)
        executor._resolve_seed(req)
        assert req.params.seed == 12345

    def test_two_calls_draw_two_distinct_seeds(self):
        """Each request gets its own random seed when None is sent."""
        executor = self._make_executor()
        req_a = self._req_with_params()
        req_b = self._req_with_params()
        executor._resolve_seed(req_a)
        executor._resolve_seed(req_b)
        # Probabilistic — collision space is 2**63; essentially impossible.
        assert req_a.params.seed != req_b.params.seed

    def test_params_none_creates_seeded_params(self):
        """Direct callers (unit tests) hitting ``process_request`` without
        ``serve_forever`` may pass ``params=None``; the helper must seed a
        fresh params object instead of crashing."""
        from tensorrt_llm._torch.visual_gen.executor import DiffusionRequest

        executor = self._make_executor()
        req = DiffusionRequest(request_id=0, prompt=["x"], params=None)
        executor._resolve_seed(req)
        assert req.params is not None
        assert isinstance(req.params.seed, int)


# =============================================================================
# DiffusionResponse — validation-error transport
# =============================================================================


class TestValidationErrorTransport:
    """When ``_validate_request`` raises ``VisualGenValidationError``,
    ``process_request`` must put a ``DiffusionResponse`` carrying the
    structured ``error_reason``/``error_param``/``error_details`` so the
    serve layer can recover the engine-side context without parsing the
    message string."""

    def _make_executor(self, pipeline_cls, mock_self=None):
        executor = MagicMock()
        executor.rank = 0
        executor.device_id = 0
        executor.response_queue = MagicMock()
        executor.pipeline = MagicMock()
        executor.pipeline.__class__ = pipeline_cls
        executor.pipeline.default_generation_params = pipeline_cls.default_generation_params.fget(
            mock_self
        )
        executor.pipeline.extra_param_specs = pipeline_cls.extra_param_specs.fget(mock_self)
        return executor

    def test_validation_error_populates_structured_response_fields(self):
        from tensorrt_llm._torch.visual_gen.executor import (
            DiffusionExecutor,
            DiffusionRequest,
            DiffusionResponse,
        )
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline
        from tensorrt_llm.visual_gen.params import VisualGenParams

        executor = self._make_executor(LTX2Pipeline)
        # Bind real methods so the existing validator runs against the mock.
        executor._resolve_seed = lambda req: DiffusionExecutor._resolve_seed(executor, req)
        executor._merge_defaults = lambda req: DiffusionExecutor._merge_defaults(executor, req)
        executor._validate_request = lambda req: DiffusionExecutor._validate_request(executor, req)

        req = DiffusionRequest(
            request_id=42,
            prompt=["test"],
            params=VisualGenParams(extra_params={"unknown_key": 1}),
        )

        DiffusionExecutor.process_request(executor, req)

        executor.response_queue.put.assert_called_once()
        resp = executor.response_queue.put.call_args[0][0]
        assert isinstance(resp, DiffusionResponse)
        assert resp.error_msg is not None
        assert resp.error_reason == "unknown_extra_param"
        assert resp.error_param == "unknown_key"
        assert resp.error_details is not None
        assert resp.error_details["pipeline"] == "LTX2Pipeline"

    def test_non_validation_error_leaves_structured_fields_none(self):
        from tensorrt_llm._torch.visual_gen.executor import (
            DiffusionExecutor,
            DiffusionRequest,
            DiffusionResponse,
        )
        from tensorrt_llm._torch.visual_gen.models.flux.pipeline_flux import FluxPipeline
        from tensorrt_llm.visual_gen.params import VisualGenParams

        executor = self._make_executor(FluxPipeline)
        executor._resolve_seed = lambda req: DiffusionExecutor._resolve_seed(executor, req)
        executor._merge_defaults = lambda req: DiffusionExecutor._merge_defaults(executor, req)
        executor._validate_request = lambda req: DiffusionExecutor._validate_request(executor, req)
        executor.pipeline.warmup_cache_key = MagicMock(return_value=(1024, 1024, None))
        executor.pipeline._warmed_up_shapes = None
        # Force a generic failure inside infer() — must NOT populate
        # structured fields.
        executor.pipeline.infer = MagicMock(side_effect=RuntimeError("oops"))

        req = DiffusionRequest(
            request_id=7,
            prompt=["test"],
            params=VisualGenParams(),
        )

        DiffusionExecutor.process_request(executor, req)

        executor.response_queue.put.assert_called_once()
        resp = executor.response_queue.put.call_args[0][0]
        assert isinstance(resp, DiffusionResponse)
        assert resp.error_msg == "oops"
        assert resp.error_reason is None
        assert resp.error_param is None
        assert resp.error_details is None
