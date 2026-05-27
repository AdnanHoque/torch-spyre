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
    "flash_attention_mixed_pipeline",
    "flash_attention_mixed_pipeline_overlap",
    "flash_attention_mixed_pipeline_artifact",
    "flash_attention_mixed_pipeline_execute_tile",
    "flash_attention_mixed_pipeline_value_flow_tile",
    "flash_attention_mixed_pipeline_ifn_pair_tile",
    "flash_attention_mixed_pipeline_layout_xform_pair_tile",
    "flash_attention_pointwise_handoff",
    "flash_attention_score_scale_handoff",
]


def _read_flash_config(extra_env=None):
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("SPYRE_FLASH_ATTENTION_"):
            env.pop(key)
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    if extra_env:
        env.update(extra_env)

    script = textwrap.dedent(
        f"""
        import json
        from torch_spyre._inductor import config

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
    assert cfg["flash_attention_mixed_pipeline"] is False
    assert cfg["flash_attention_pointwise_handoff"] is False
    assert cfg["flash_attention_score_scale_handoff"] is False
    assert cfg["flash_attention_mixed_pipeline_overlap"] is False
    assert cfg["flash_attention_mixed_pipeline_artifact"] is False
    assert cfg["flash_attention_mixed_pipeline_execute_tile"] == -1
    assert cfg["flash_attention_mixed_pipeline_value_flow_tile"] == -1
    assert cfg["flash_attention_mixed_pipeline_ifn_pair_tile"] == -1
    assert cfg["flash_attention_mixed_pipeline_layout_xform_pair_tile"] == -1


def test_flash_attention_onchip_sdpa_master_gate_enables_certified_path_only():
    cfg = _read_flash_config({"SPYRE_FLASH_ATTENTION_ONCHIP_SDPA": "1"})

    assert cfg["flash_attention_onchip_sdpa"] is True
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
            "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_PAIR_TILE": "2",
        }
    )

    assert cfg["flash_attention_onchip_sdpa"] is False
    assert cfg["flash_attention_mixed_pipeline"] is True
    assert cfg["flash_attention_pointwise_handoff"] is True
    assert cfg["flash_attention_score_scale_handoff"] is False
    assert cfg["flash_attention_mixed_pipeline_layout_xform_pair_tile"] == 2
