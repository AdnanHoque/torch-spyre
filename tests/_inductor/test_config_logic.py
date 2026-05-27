# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import subprocess
import sys
import textwrap


_FLASH_CONFIG_KEYS = [
    "flash_attention_prefill",
    "flash_attention_prefill_block_size",
    "flash_attention_onchip_sdpa",
    "flash_attention_onchip_sdpa_layout_xform",
    "flash_attention_mixed_pipeline",
    "flash_attention_mixed_pipeline_overlap",
    "flash_attention_mixed_pipeline_artifact",
    "flash_attention_mixed_pipeline_execute_tile",
    "flash_attention_mixed_pipeline_value_flow_tile",
    "flash_attention_mixed_pipeline_ifn_pair_tile",
    "flash_attention_mixed_pipeline_ifn_prefix_force",
    "flash_attention_mixed_pipeline_layout_xform_pair_tile",
    "flash_attention_mixed_pipeline_layout_xform_pair_overlap",
    "flash_attention_mixed_pipeline_layout_xform_lookahead_tile",
    "flash_attention_mixed_pipeline_layout_xform_hoist_tile",
    "flash_attention_pointwise_handoff",
    "flash_attention_score_scale_handoff",
    "causal_idx_to_mask_plan_artifact",
]
_HERE = os.path.dirname(os.path.abspath(__file__))
_CONFIG = os.path.normpath(
    os.path.join(_HERE, "..", "..", "torch_spyre", "_inductor", "config.py")
)


def _read_flash_config(extra_env=None):
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("SPYRE_FLASH_ATTENTION_") or key.startswith(
            "SPYRE_CAUSAL_"
        ):
            env.pop(key)
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    if extra_env:
        env.update(extra_env)

    script = textwrap.dedent(
        f"""
        import importlib.util
        import json
        import sys
        import types

        torch = types.ModuleType("torch")
        torch_utils = types.ModuleType("torch.utils")
        torch_config_module = types.ModuleType("torch.utils._config_module")
        torch_config_module.install_config_module = lambda module: None
        torch.utils = torch_utils
        torch_utils._config_module = torch_config_module
        sys.modules["torch"] = torch
        sys.modules["torch.utils"] = torch_utils
        sys.modules["torch.utils._config_module"] = torch_config_module

        spec = importlib.util.spec_from_file_location(
            "torch_spyre._inductor.config",
            {json.dumps(_CONFIG)},
        )
        config = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = config
        spec.loader.exec_module(config)

        keys = {json.dumps(_FLASH_CONFIG_KEYS)}
        print(json.dumps({{key: getattr(config, key) for key in keys}}, sort_keys=True))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(result.stdout)


def test_flash_attention_onchip_sdpa_master_gate_defaults_off():
    cfg = _read_flash_config()

    assert cfg["flash_attention_prefill"] is False
    assert cfg["flash_attention_prefill_block_size"] == 128
    assert cfg["flash_attention_onchip_sdpa"] is False
    assert cfg["flash_attention_onchip_sdpa_layout_xform"] is False
    assert cfg["flash_attention_mixed_pipeline"] is False
    assert cfg["flash_attention_pointwise_handoff"] is False
    assert cfg["flash_attention_score_scale_handoff"] is False
    assert cfg["flash_attention_mixed_pipeline_overlap"] is False
    assert cfg["flash_attention_mixed_pipeline_artifact"] is False
    assert cfg["flash_attention_mixed_pipeline_execute_tile"] == -1
    assert cfg["flash_attention_mixed_pipeline_value_flow_tile"] == -1
    assert cfg["flash_attention_mixed_pipeline_ifn_pair_tile"] == -1
    assert cfg["flash_attention_mixed_pipeline_ifn_prefix_force"] is False
    assert cfg["flash_attention_mixed_pipeline_layout_xform_pair_tile"] == -1
    assert cfg["flash_attention_mixed_pipeline_layout_xform_pair_overlap"] is False
    assert cfg["flash_attention_mixed_pipeline_layout_xform_lookahead_tile"] == -1
    assert cfg["flash_attention_mixed_pipeline_layout_xform_hoist_tile"] == -1
    assert cfg["causal_idx_to_mask_plan_artifact"] is False


def test_flash_attention_onchip_sdpa_master_gate_enables_certified_path_only():
    cfg = _read_flash_config({"SPYRE_FLASH_ATTENTION_ONCHIP_SDPA": "1"})

    assert cfg["flash_attention_onchip_sdpa"] is True
    assert cfg["flash_attention_onchip_sdpa_layout_xform"] is False
    assert cfg["flash_attention_prefill_block_size"] == 512
    assert cfg["flash_attention_mixed_pipeline"] is True
    assert cfg["flash_attention_pointwise_handoff"] is True
    assert cfg["flash_attention_score_scale_handoff"] is True

    assert cfg["flash_attention_prefill"] is False
    assert cfg["flash_attention_mixed_pipeline_overlap"] is False
    assert cfg["flash_attention_mixed_pipeline_artifact"] is False
    assert cfg["flash_attention_mixed_pipeline_execute_tile"] == -1
    assert cfg["flash_attention_mixed_pipeline_value_flow_tile"] == -1
    assert cfg["flash_attention_mixed_pipeline_ifn_pair_tile"] == -1
    assert cfg["flash_attention_mixed_pipeline_ifn_prefix_force"] is False
    assert cfg["flash_attention_mixed_pipeline_layout_xform_pair_tile"] == -1
    assert cfg["flash_attention_mixed_pipeline_layout_xform_pair_overlap"] is False
    assert cfg["flash_attention_mixed_pipeline_layout_xform_lookahead_tile"] == -1
    assert cfg["flash_attention_mixed_pipeline_layout_xform_hoist_tile"] == -1
    assert cfg["causal_idx_to_mask_plan_artifact"] is False


def test_causal_idx_to_mask_plan_artifact_is_independently_gated():
    cfg = _read_flash_config({"SPYRE_CAUSAL_IDX_TO_MASK_PLAN_ARTIFACT": "1"})

    assert cfg["causal_idx_to_mask_plan_artifact"] is True
    assert cfg["flash_attention_onchip_sdpa"] is False
    assert cfg["flash_attention_mixed_pipeline"] is False


def test_flash_attention_ifn_prefix_force_is_independently_gated():
    cfg = _read_flash_config(
        {"SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_IFN_PREFIX_FORCE": "1"}
    )

    assert cfg["flash_attention_mixed_pipeline_ifn_prefix_force"] is True
    assert cfg["flash_attention_onchip_sdpa"] is False
    assert cfg["flash_attention_mixed_pipeline"] is False


def test_flash_attention_layout_xform_pair_overlap_is_independently_gated():
    cfg = _read_flash_config(
        {"SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_PAIR_OVERLAP": "1"}
    )

    assert cfg["flash_attention_mixed_pipeline_layout_xform_pair_overlap"] is True
    assert cfg["flash_attention_onchip_sdpa"] is False
    assert cfg["flash_attention_mixed_pipeline"] is False


def test_flash_attention_layout_xform_lookahead_accepts_concrete_tile():
    cfg = _read_flash_config(
        {"SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_LOOKAHEAD_TILE": "3"}
    )

    assert cfg["flash_attention_mixed_pipeline_layout_xform_lookahead_tile"] == 3
    assert cfg["flash_attention_onchip_sdpa"] is False
    assert cfg["flash_attention_mixed_pipeline"] is False


def test_flash_attention_layout_xform_hoist_accepts_concrete_tile():
    cfg = _read_flash_config(
        {"SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_HOIST_TILE": "2"}
    )

    assert cfg["flash_attention_mixed_pipeline_layout_xform_hoist_tile"] == 2
    assert cfg["flash_attention_onchip_sdpa"] is False
    assert cfg["flash_attention_mixed_pipeline"] is False


def test_flash_attention_onchip_sdpa_layout_xform_adjunct_enables_auto_pair():
    cfg = _read_flash_config(
        {
            "SPYRE_FLASH_ATTENTION_ONCHIP_SDPA": "1",
            "SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM": "1",
        }
    )

    assert cfg["flash_attention_onchip_sdpa"] is True
    assert cfg["flash_attention_onchip_sdpa_layout_xform"] is True
    assert cfg["flash_attention_mixed_pipeline"] is True
    assert cfg["flash_attention_pointwise_handoff"] is True
    assert cfg["flash_attention_score_scale_handoff"] is True
    assert cfg["flash_attention_mixed_pipeline_layout_xform_pair_tile"] == -2


def test_flash_attention_onchip_sdpa_layout_xform_adjunct_requires_master_gate():
    cfg = _read_flash_config(
        {"SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM": "1"}
    )

    assert cfg["flash_attention_onchip_sdpa"] is False
    assert cfg["flash_attention_onchip_sdpa_layout_xform"] is False
    assert cfg["flash_attention_mixed_pipeline_layout_xform_pair_tile"] == -1


def test_flash_attention_onchip_sdpa_master_gate_respects_block_size_override():
    cfg = _read_flash_config(
        {
            "SPYRE_FLASH_ATTENTION_ONCHIP_SDPA": "1",
            "SPYRE_FLASH_ATTENTION_PREFILL_BLOCK_SIZE": "128",
        }
    )

    assert cfg["flash_attention_onchip_sdpa"] is True
    assert cfg["flash_attention_prefill_block_size"] == 128


def test_flash_attention_onchip_sdpa_master_gate_preserves_individual_flags():
    cfg = _read_flash_config(
        {
            "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE": "1",
            "SPYRE_FLASH_ATTENTION_POINTWISE_HANDOFF": "1",
            "SPYRE_FLASH_ATTENTION_SCORE_SCALE_HANDOFF": "0",
            "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_PAIR_TILE": "-2",
        }
    )

    assert cfg["flash_attention_onchip_sdpa"] is False
    assert cfg["flash_attention_onchip_sdpa_layout_xform"] is False
    assert cfg["flash_attention_mixed_pipeline"] is True
    assert cfg["flash_attention_pointwise_handoff"] is True
    assert cfg["flash_attention_score_scale_handoff"] is False
    assert cfg["flash_attention_mixed_pipeline_layout_xform_pair_tile"] == -2


def test_flash_attention_layout_xform_pair_accepts_concrete_tile():
    cfg = _read_flash_config(
        {
            "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_PAIR_TILE": "2",
        }
    )

    assert cfg["flash_attention_mixed_pipeline_layout_xform_pair_tile"] == 2


def _run_all():
    tests = sorted(
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
    fails = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            fails.append(name)
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {name}")
    print(f"\n{len(tests) - len(fails)}/{len(tests)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
