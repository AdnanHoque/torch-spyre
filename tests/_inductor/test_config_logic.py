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
    "flash_attention_kv_repack_broadcast_plan_artifact",
    "flash_attention_kv_repack_broadcast_pair_tile",
    "flash_attention_kv_repack_broadcast_pair_ifn_transfer",
    "flash_attention_kv_repack_broadcast_pair_subpiece_reuse",
    "flash_attention_kv_repack_broadcast_pair_group_size",
    "flash_attention_kv_repack_broadcast_pair_self_resident_source",
    "flash_attention_kv_repack_broadcast_pair_hbm_source",
    "flash_attention_kv_repack_broadcast_pair_hbm_direct_load",
    "flash_attention_kv_repack_broadcast_pair_hbm_staged",
    "flash_attention_kv_repack_broadcast_pair_consumer_core_state_init",
    "flash_attention_kv_repack_broadcast_pair_consumer_ds_type",
    "flash_attention_kv_repack_broadcast_pair_consumer_lx_alloc_style",
    "flash_attention_kv_repack_broadcast_pair_use_unicast",
    "flash_attention_kv_repack_broadcast_pair_force_mc_mode",
    "flash_attention_kv_repack_hbm_staged_hoist_tile",
    "flash_attention_kv_repack_hbm_prefetch_hoist_tile",
    "flash_attention_kv_repack_hbm_prefetch_lx_base",
    "flash_attention_kv_repack_hbm_prefetch_serial",
    "flash_attention_kv_repack_hbm_prefetch_prefill_current",
    "flash_attention_kv_repack_hbm_prefetch_redundant_future",
    "flash_attention_kv_repack_hbm_prefetch_serialize_current",
    "flash_attention_kv_repack_hbm_prefetch_external_future",
    "flash_attention_kv_repack_hbm_prefetch_overlap_after_sync",
    "flash_attention_kv_repack_hbm_prefetch_tail_current",
    "flash_attention_kv_repack_hbm_prefetch_source_fanout",
    "flash_attention_kv_repack_hbm_prefetch_loader_fanout",
    "flash_attention_kv_repack_hbm_prefetch_loader_core",
    "flash_attention_kv_repack_hbm_prefetch_loader_lx_base",
    "flash_attention_kv_repack_hbm_prefetch_fanout_use_unicast",
    "flash_attention_kv_repack_hbm_prefetch_fanout_use_lxsfp_lx_transfers",
    "flash_attention_kv_repack_hbm_prefetch_fanout_copyback_core",
    "flash_attention_kv_repack_hbm_prefetch_fanout_restrict_to_copyback_core",
    "flash_attention_kv_repack_hbm_prefetch_loader_copyback_without_fanout",
    "flash_attention_kv_repack_hbm_prefetch_loader_fanout_full_tile_pieces",
    "flash_attention_kv_repack_hbm_prefetch_serialize_loader_core",
    "flash_attention_kv_repack_hbm_prefetch_lx_roundtrip",
    "flash_attention_kv_repack_hbm_prefetch_corelet1",
    "flash_attention_kv_repack_broadcast_copyback_tile",
    "flash_attention_kv_repack_broadcast_copyback_core",
    "flash_attention_kv_repack_broadcast_copyback_direct_source",
    "flash_attention_kv_repack_broadcast_copyback_hbm_roundtrip",
    "flash_attention_kv_repack_broadcast_copyback_hbm_source_fanout",
    "flash_attention_kv_repack_broadcast_copyback_hbm_direct_load",
    "flash_attention_kv_repack_broadcast_copyback_hbm_roundtrip_load_only",
    "flash_attention_kv_repack_broadcast_copyback_hbm_roundtrip_barrier_only",
    "flash_attention_kv_repack_broadcast_copyback_data_only",
    "flash_attention_kv_repack_broadcast_copyback_replace_consumer",
    "flash_attention_kv_repack_broadcast_copyback_compute_only",
    "flash_attention_kv_repack_broadcast_copyback_exact_clone",
    "flash_attention_kv_repack_broadcast_copyback_preserve_consumer_name",
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
    assert cfg["flash_attention_kv_repack_broadcast_plan_artifact"] is False
    assert cfg["flash_attention_kv_repack_broadcast_pair_tile"] == -1
    assert cfg["flash_attention_kv_repack_broadcast_pair_ifn_transfer"] is True
    assert cfg["flash_attention_kv_repack_broadcast_pair_subpiece_reuse"] is True
    assert cfg["flash_attention_kv_repack_broadcast_pair_group_size"] == 0
    assert cfg["flash_attention_kv_repack_broadcast_pair_self_resident_source"] is False
    assert cfg["flash_attention_kv_repack_broadcast_pair_hbm_source"] is False
    assert cfg["flash_attention_kv_repack_broadcast_pair_hbm_direct_load"] is False
    assert cfg["flash_attention_kv_repack_broadcast_pair_hbm_staged"] is False
    assert (
        cfg["flash_attention_kv_repack_broadcast_pair_consumer_core_state_init"]
        is True
    )
    assert cfg["flash_attention_kv_repack_broadcast_pair_consumer_ds_type"] == ""
    assert (
        cfg["flash_attention_kv_repack_broadcast_pair_consumer_lx_alloc_style"]
        == ""
    )
    assert cfg["flash_attention_kv_repack_broadcast_pair_use_unicast"] == -1
    assert cfg["flash_attention_kv_repack_broadcast_pair_force_mc_mode"] == -1
    assert cfg["flash_attention_kv_repack_hbm_staged_hoist_tile"] == -1
    assert cfg["flash_attention_kv_repack_hbm_prefetch_hoist_tile"] == -1
    assert cfg["flash_attention_kv_repack_hbm_prefetch_lx_base"] == -1
    assert cfg["flash_attention_kv_repack_hbm_prefetch_serial"] is False
    assert (
        cfg["flash_attention_kv_repack_hbm_prefetch_prefill_current"] is False
    )
    assert (
        cfg["flash_attention_kv_repack_hbm_prefetch_redundant_future"] is False
    )
    assert (
        cfg["flash_attention_kv_repack_hbm_prefetch_serialize_current"] is False
    )
    assert cfg["flash_attention_kv_repack_hbm_prefetch_external_future"] is False
    assert cfg["flash_attention_kv_repack_hbm_prefetch_overlap_after_sync"] is True
    assert cfg["flash_attention_kv_repack_hbm_prefetch_tail_current"] is False
    assert cfg["flash_attention_kv_repack_hbm_prefetch_source_fanout"] is False
    assert cfg["flash_attention_kv_repack_hbm_prefetch_loader_fanout"] is False
    assert cfg["flash_attention_kv_repack_hbm_prefetch_loader_core"] == 0
    assert cfg["flash_attention_kv_repack_hbm_prefetch_loader_lx_base"] == -1
    assert cfg["flash_attention_kv_repack_hbm_prefetch_fanout_use_unicast"] == -1
    assert (
        cfg[
            "flash_attention_kv_repack_hbm_prefetch_fanout_use_lxsfp_lx_transfers"
        ]
        == -1
    )
    assert cfg["flash_attention_kv_repack_hbm_prefetch_fanout_copyback_core"] == -2
    assert (
        cfg[
            "flash_attention_kv_repack_hbm_prefetch_fanout_restrict_to_copyback_core"
        ]
        is False
    )
    assert (
        cfg[
            "flash_attention_kv_repack_hbm_prefetch_loader_copyback_without_fanout"
        ]
        is False
    )
    assert (
        cfg[
            "flash_attention_kv_repack_hbm_prefetch_loader_fanout_full_tile_pieces"
        ]
        is False
    )
    assert (
        cfg["flash_attention_kv_repack_hbm_prefetch_serialize_loader_core"]
        is False
    )
    assert cfg["flash_attention_kv_repack_hbm_prefetch_lx_roundtrip"] is False
    assert cfg["flash_attention_kv_repack_hbm_prefetch_corelet1"] is False
    assert cfg["flash_attention_kv_repack_broadcast_copyback_tile"] == -1
    assert cfg["flash_attention_kv_repack_broadcast_copyback_core"] == -1
    assert cfg["flash_attention_kv_repack_broadcast_copyback_direct_source"] is False
    assert cfg["flash_attention_kv_repack_broadcast_copyback_hbm_roundtrip"] is False
    assert (
        cfg["flash_attention_kv_repack_broadcast_copyback_hbm_source_fanout"]
        is False
    )
    assert (
        cfg["flash_attention_kv_repack_broadcast_copyback_hbm_direct_load"]
        is False
    )
    assert (
        cfg["flash_attention_kv_repack_broadcast_copyback_hbm_roundtrip_load_only"]
        is False
    )
    assert (
        cfg["flash_attention_kv_repack_broadcast_copyback_hbm_roundtrip_barrier_only"]
        is False
    )
    assert cfg["flash_attention_kv_repack_broadcast_copyback_data_only"] is False
    assert (
        cfg["flash_attention_kv_repack_broadcast_copyback_replace_consumer"] is False
    )
    assert cfg["flash_attention_kv_repack_broadcast_copyback_compute_only"] is False
    assert cfg["flash_attention_kv_repack_broadcast_copyback_exact_clone"] is False
    assert (
        cfg["flash_attention_kv_repack_broadcast_copyback_preserve_consumer_name"]
        is False
    )
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
    assert cfg["flash_attention_kv_repack_broadcast_plan_artifact"] is False
    assert cfg["flash_attention_kv_repack_broadcast_pair_tile"] == -1
    assert cfg["flash_attention_kv_repack_broadcast_pair_ifn_transfer"] is True
    assert cfg["flash_attention_kv_repack_broadcast_pair_subpiece_reuse"] is True
    assert cfg["flash_attention_kv_repack_broadcast_pair_group_size"] == 0
    assert cfg["flash_attention_kv_repack_broadcast_pair_hbm_source"] is False
    assert cfg["flash_attention_kv_repack_broadcast_pair_hbm_direct_load"] is False
    assert cfg["flash_attention_kv_repack_broadcast_pair_hbm_staged"] is False
    assert (
        cfg["flash_attention_kv_repack_broadcast_pair_consumer_core_state_init"]
        is True
    )
    assert cfg["flash_attention_kv_repack_broadcast_pair_consumer_ds_type"] == ""
    assert (
        cfg["flash_attention_kv_repack_broadcast_pair_consumer_lx_alloc_style"]
        == ""
    )
    assert cfg["flash_attention_kv_repack_hbm_staged_hoist_tile"] == -1
    assert cfg["flash_attention_kv_repack_hbm_prefetch_hoist_tile"] == -1
    assert cfg["flash_attention_kv_repack_hbm_prefetch_lx_base"] == -1
    assert cfg["flash_attention_kv_repack_hbm_prefetch_serial"] is False
    assert cfg["flash_attention_kv_repack_hbm_prefetch_prefill_current"] is False
    assert cfg["flash_attention_kv_repack_hbm_prefetch_redundant_future"] is False
    assert cfg["flash_attention_kv_repack_hbm_prefetch_serialize_current"] is False
    assert cfg["flash_attention_kv_repack_hbm_prefetch_external_future"] is False
    assert cfg["flash_attention_kv_repack_hbm_prefetch_overlap_after_sync"] is True
    assert cfg["flash_attention_kv_repack_hbm_prefetch_tail_current"] is False
    assert cfg["flash_attention_kv_repack_hbm_prefetch_source_fanout"] is False
    assert cfg["flash_attention_kv_repack_hbm_prefetch_loader_fanout"] is False
    assert cfg["flash_attention_kv_repack_hbm_prefetch_loader_core"] == 0
    assert cfg["flash_attention_kv_repack_hbm_prefetch_loader_lx_base"] == -1
    assert cfg["flash_attention_kv_repack_hbm_prefetch_fanout_use_unicast"] == -1
    assert (
        cfg[
            "flash_attention_kv_repack_hbm_prefetch_fanout_use_lxsfp_lx_transfers"
        ]
        == -1
    )
    assert cfg["flash_attention_kv_repack_hbm_prefetch_fanout_copyback_core"] == -2
    assert (
        cfg[
            "flash_attention_kv_repack_hbm_prefetch_fanout_restrict_to_copyback_core"
        ]
        is False
    )
    assert (
        cfg[
            "flash_attention_kv_repack_hbm_prefetch_loader_copyback_without_fanout"
        ]
        is False
    )
    assert (
        cfg[
            "flash_attention_kv_repack_hbm_prefetch_loader_fanout_full_tile_pieces"
        ]
        is False
    )
    assert (
        cfg["flash_attention_kv_repack_hbm_prefetch_serialize_loader_core"]
        is False
    )
    assert cfg["flash_attention_kv_repack_hbm_prefetch_lx_roundtrip"] is False
    assert cfg["flash_attention_kv_repack_hbm_prefetch_corelet1"] is False
    assert cfg["flash_attention_kv_repack_broadcast_copyback_tile"] == -1
    assert cfg["flash_attention_kv_repack_broadcast_copyback_core"] == -1
    assert cfg["flash_attention_kv_repack_broadcast_copyback_direct_source"] is False
    assert cfg["flash_attention_kv_repack_broadcast_copyback_hbm_roundtrip"] is False
    assert (
        cfg["flash_attention_kv_repack_broadcast_copyback_hbm_source_fanout"]
        is False
    )
    assert (
        cfg["flash_attention_kv_repack_broadcast_copyback_hbm_direct_load"]
        is False
    )
    assert (
        cfg["flash_attention_kv_repack_broadcast_copyback_hbm_roundtrip_load_only"]
        is False
    )
    assert (
        cfg["flash_attention_kv_repack_broadcast_copyback_hbm_roundtrip_barrier_only"]
        is False
    )
    assert cfg["flash_attention_kv_repack_broadcast_copyback_data_only"] is False
    assert (
        cfg["flash_attention_kv_repack_broadcast_copyback_replace_consumer"] is False
    )
    assert cfg["flash_attention_kv_repack_broadcast_copyback_compute_only"] is False
    assert cfg["flash_attention_kv_repack_broadcast_copyback_exact_clone"] is False
    assert (
        cfg["flash_attention_kv_repack_broadcast_copyback_preserve_consumer_name"]
        is False
    )
    assert cfg["causal_idx_to_mask_plan_artifact"] is False


def test_flash_attention_kv_repack_pair_ifn_transfer_can_be_disabled():
    cfg = _read_flash_config(
        {"SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_IFN_TRANSFER": "0"}
    )

    assert cfg["flash_attention_kv_repack_broadcast_pair_ifn_transfer"] is False


def test_flash_attention_kv_repack_pair_subpiece_reuse_can_be_disabled():
    cfg = _read_flash_config(
        {"SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_SUBPIECE_REUSE": "0"}
    )

    assert cfg["flash_attention_kv_repack_broadcast_pair_subpiece_reuse"] is False


def test_flash_attention_kv_repack_pair_group_size_accepts_concrete_value():
    cfg = _read_flash_config(
        {"SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_GROUP_SIZE": "16"}
    )

    assert cfg["flash_attention_kv_repack_broadcast_pair_group_size"] == 16


def test_flash_attention_kv_repack_pair_self_resident_source_can_be_enabled():
    cfg = _read_flash_config(
        {"SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_SELF_RESIDENT_SOURCE": "1"}
    )

    assert cfg["flash_attention_kv_repack_broadcast_pair_self_resident_source"] is True


def test_flash_attention_kv_repack_pair_hbm_source_can_be_enabled():
    cfg = _read_flash_config(
        {"SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_HBM_SOURCE": "1"}
    )

    assert cfg["flash_attention_kv_repack_broadcast_pair_hbm_source"] is True


def test_flash_attention_kv_repack_pair_hbm_direct_load_can_be_enabled():
    cfg = _read_flash_config(
        {"SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_HBM_DIRECT_LOAD": "1"}
    )

    assert cfg["flash_attention_kv_repack_broadcast_pair_hbm_direct_load"] is True


def test_flash_attention_kv_repack_pair_hbm_staged_can_be_enabled():
    cfg = _read_flash_config(
        {"SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_HBM_STAGED": "1"}
    )

    assert cfg["flash_attention_kv_repack_broadcast_pair_hbm_staged"] is True


def test_flash_attention_kv_repack_pair_consumer_core_state_init_can_be_disabled():
    cfg = _read_flash_config(
        {
            (
                "SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_"
                "CONSUMER_CORE_STATE_INIT"
            ): "0"
        }
    )

    assert (
        cfg["flash_attention_kv_repack_broadcast_pair_consumer_core_state_init"]
        is False
    )


def test_flash_attention_kv_repack_pair_consumer_ds_type_can_be_overridden():
    cfg = _read_flash_config(
        {
            (
                "SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_"
                "CONSUMER_DS_TYPE"
            ): "INPUT"
        }
    )

    assert cfg["flash_attention_kv_repack_broadcast_pair_consumer_ds_type"] == "INPUT"


def test_flash_attention_kv_repack_pair_consumer_lx_alloc_style_can_be_overridden():
    cfg = _read_flash_config(
        {
            (
                "SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_"
                "CONSUMER_LX_ALLOC_STYLE"
            ): "canonical_loop"
        }
    )

    assert (
        cfg["flash_attention_kv_repack_broadcast_pair_consumer_lx_alloc_style"]
        == "canonical_loop"
    )


def test_flash_attention_kv_repack_pair_use_unicast_accepts_concrete_value():
    cfg = _read_flash_config(
        {"SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_USE_UNICAST": "1"}
    )

    assert cfg["flash_attention_kv_repack_broadcast_pair_use_unicast"] == 1


def test_flash_attention_kv_repack_hbm_staged_hoist_tile_can_be_enabled():
    cfg = _read_flash_config(
        {"SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_STAGED_HOIST_TILE": "-2"}
    )

    assert cfg["flash_attention_kv_repack_hbm_staged_hoist_tile"] == -2


def test_flash_attention_kv_repack_hbm_prefetch_hoist_tile_can_be_enabled():
    cfg = _read_flash_config(
        {"SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_HOIST_TILE": "-2"}
    )

    assert cfg["flash_attention_kv_repack_hbm_prefetch_hoist_tile"] == -2


def test_flash_attention_kv_repack_hbm_prefetch_lx_base_can_be_overridden():
    cfg = _read_flash_config(
        {"SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_LX_BASE": "1625344"}
    )

    assert cfg["flash_attention_kv_repack_hbm_prefetch_lx_base"] == 1625344


def test_flash_attention_kv_repack_hbm_prefetch_serial_can_be_enabled():
    cfg = _read_flash_config(
        {"SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_SERIAL": "1"}
    )

    assert cfg["flash_attention_kv_repack_hbm_prefetch_serial"] is True


def test_flash_attention_kv_repack_hbm_prefetch_prefill_current_can_be_enabled():
    cfg = _read_flash_config(
        {"SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_PREFILL_CURRENT": "1"}
    )

    assert cfg["flash_attention_kv_repack_hbm_prefetch_prefill_current"] is True


def test_flash_attention_kv_repack_hbm_prefetch_redundant_future_can_be_enabled():
    cfg = _read_flash_config(
        {
            "SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_REDUNDANT_FUTURE": "1"
        }
    )

    assert cfg["flash_attention_kv_repack_hbm_prefetch_redundant_future"] is True


def test_flash_attention_kv_repack_hbm_prefetch_serialize_current_can_be_enabled():
    cfg = _read_flash_config(
        {
            "SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_SERIALIZE_CURRENT": "1"
        }
    )

    assert cfg["flash_attention_kv_repack_hbm_prefetch_serialize_current"] is True


def test_flash_attention_kv_repack_hbm_prefetch_external_future_can_be_enabled():
    cfg = _read_flash_config(
        {
            "SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_EXTERNAL_FUTURE": "1"
        }
    )

    assert cfg["flash_attention_kv_repack_hbm_prefetch_external_future"] is True


def test_flash_attention_kv_repack_hbm_prefetch_overlap_after_sync_can_be_disabled():
    cfg = _read_flash_config(
        {
            "SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_OVERLAP_AFTER_SYNC": "0"
        }
    )

    assert cfg["flash_attention_kv_repack_hbm_prefetch_overlap_after_sync"] is False


def test_flash_attention_kv_repack_hbm_prefetch_tail_current_can_be_enabled():
    cfg = _read_flash_config(
        {"SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_TAIL_CURRENT": "1"}
    )

    assert cfg["flash_attention_kv_repack_hbm_prefetch_tail_current"] is True


def test_flash_attention_kv_repack_hbm_prefetch_source_fanout_can_be_enabled():
    cfg = _read_flash_config(
        {"SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_SOURCE_FANOUT": "1"}
    )

    assert cfg["flash_attention_kv_repack_hbm_prefetch_source_fanout"] is True


def test_flash_attention_kv_repack_hbm_prefetch_loader_fanout_can_be_enabled():
    cfg = _read_flash_config(
        {"SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT": "1"}
    )

    assert cfg["flash_attention_kv_repack_hbm_prefetch_loader_fanout"] is True


def test_flash_attention_kv_repack_hbm_prefetch_loader_core_can_be_forced():
    cfg = _read_flash_config(
        {"SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_LOADER_CORE": "31"}
    )

    assert cfg["flash_attention_kv_repack_hbm_prefetch_loader_core"] == 31


def test_flash_attention_kv_repack_hbm_prefetch_loader_lx_base_can_be_forced():
    cfg = _read_flash_config(
        {"SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_LOADER_LX_BASE": "-2"}
    )

    assert cfg["flash_attention_kv_repack_hbm_prefetch_loader_lx_base"] == -2


def test_flash_attention_kv_repack_hbm_prefetch_fanout_use_unicast_can_be_forced():
    cfg = _read_flash_config(
        {"SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_FANOUT_USE_UNICAST": "1"}
    )

    assert cfg["flash_attention_kv_repack_hbm_prefetch_fanout_use_unicast"] == 1


def test_flash_attention_kv_repack_hbm_prefetch_fanout_use_lxsfp_lx_transfers_can_be_forced():
    cfg = _read_flash_config(
        {
            "SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_FANOUT_USE_LXSFP_LX_TRANSFERS": "0"
        }
    )

    assert (
        cfg[
            "flash_attention_kv_repack_hbm_prefetch_fanout_use_lxsfp_lx_transfers"
        ]
        == 0
    )


def test_flash_attention_kv_repack_hbm_prefetch_fanout_copyback_core_can_be_forced():
    cfg = _read_flash_config(
        {"SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_FANOUT_COPYBACK_CORE": "0"}
    )

    assert cfg["flash_attention_kv_repack_hbm_prefetch_fanout_copyback_core"] == 0


def test_flash_attention_kv_repack_hbm_prefetch_fanout_restrict_to_copyback_core_can_be_enabled():
    cfg = _read_flash_config(
        {
            "SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_FANOUT_RESTRICT_TO_COPYBACK_CORE": "1"
        }
    )

    assert (
        cfg[
            "flash_attention_kv_repack_hbm_prefetch_fanout_restrict_to_copyback_core"
        ]
        is True
    )


def test_flash_attention_kv_repack_hbm_prefetch_loader_copyback_without_fanout_can_be_enabled():
    cfg = _read_flash_config(
        {
            "SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_LOADER_COPYBACK_WITHOUT_FANOUT": "1"
        }
    )

    assert (
        cfg[
            "flash_attention_kv_repack_hbm_prefetch_loader_copyback_without_fanout"
        ]
        is True
    )


def test_flash_attention_kv_repack_hbm_prefetch_loader_fanout_full_tile_pieces_can_be_enabled():
    cfg = _read_flash_config(
        {
            "SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT_FULL_TILE_PIECES": "1"
        }
    )

    assert (
        cfg[
            "flash_attention_kv_repack_hbm_prefetch_loader_fanout_full_tile_pieces"
        ]
        is True
    )


def test_flash_attention_kv_repack_hbm_prefetch_serialize_loader_core_can_be_enabled():
    cfg = _read_flash_config(
        {
            "SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_SERIALIZE_LOADER_CORE": "1"
        }
    )

    assert (
        cfg["flash_attention_kv_repack_hbm_prefetch_serialize_loader_core"]
        is True
    )


def test_flash_attention_kv_repack_hbm_prefetch_corelet1_can_be_enabled():
    cfg = _read_flash_config(
        {"SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_CORELET1": "1"}
    )

    assert cfg["flash_attention_kv_repack_hbm_prefetch_corelet1"] is True


def test_flash_attention_kv_repack_hbm_prefetch_lx_roundtrip_can_be_enabled():
    cfg = _read_flash_config(
        {"SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_LX_ROUNDTRIP": "1"}
    )

    assert cfg["flash_attention_kv_repack_hbm_prefetch_lx_roundtrip"] is True


def test_flash_attention_kv_repack_pair_force_mc_accepts_concrete_value():
    cfg = _read_flash_config(
        {"SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_FORCE_MC_MODE": "3"}
    )

    assert cfg["flash_attention_kv_repack_broadcast_pair_force_mc_mode"] == 3


def test_flash_attention_kv_repack_pair_accepts_concrete_tile():
    cfg = _read_flash_config(
        {"SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_TILE": "2"}
    )

    assert cfg["flash_attention_kv_repack_broadcast_pair_tile"] == 2
    assert cfg["flash_attention_onchip_sdpa"] is False
    assert cfg["flash_attention_mixed_pipeline"] is False


def test_flash_attention_kv_repack_copyback_accepts_concrete_tile_and_core():
    cfg = _read_flash_config(
        {
            "SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_TILE": "2",
            "SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_CORE": "31",
        }
    )

    assert cfg["flash_attention_kv_repack_broadcast_copyback_tile"] == 2
    assert cfg["flash_attention_kv_repack_broadcast_copyback_core"] == 31


def test_flash_attention_kv_repack_copyback_direct_source_is_gated():
    cfg = _read_flash_config(
        {"SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_DIRECT_SOURCE": "1"}
    )

    assert cfg["flash_attention_kv_repack_broadcast_copyback_direct_source"] is True


def test_flash_attention_kv_repack_copyback_hbm_roundtrip_is_gated():
    cfg = _read_flash_config(
        {"SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_HBM_ROUNDTRIP": "1"}
    )

    assert cfg["flash_attention_kv_repack_broadcast_copyback_hbm_roundtrip"] is True


def test_flash_attention_kv_repack_copyback_hbm_source_fanout_is_gated():
    cfg = _read_flash_config(
        {
            (
                "SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_"
                "HBM_SOURCE_FANOUT"
            ): "1"
        }
    )

    assert (
        cfg["flash_attention_kv_repack_broadcast_copyback_hbm_source_fanout"]
        is True
    )


def test_flash_attention_kv_repack_copyback_hbm_direct_load_is_gated():
    cfg = _read_flash_config(
        {
            (
                "SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_"
                "HBM_DIRECT_LOAD"
            ): "1"
        }
    )

    assert (
        cfg["flash_attention_kv_repack_broadcast_copyback_hbm_direct_load"]
        is True
    )


def test_flash_attention_kv_repack_copyback_hbm_roundtrip_load_only_is_gated():
    cfg = _read_flash_config(
        {
            (
                "SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_"
                "HBM_ROUNDTRIP_LOAD_ONLY"
            ): "1"
        }
    )

    assert (
        cfg["flash_attention_kv_repack_broadcast_copyback_hbm_roundtrip_load_only"]
        is True
    )


def test_flash_attention_kv_repack_copyback_hbm_roundtrip_barrier_only_is_gated():
    cfg = _read_flash_config(
        {
            (
                "SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_"
                "HBM_ROUNDTRIP_BARRIER_ONLY"
            ): "1"
        }
    )

    assert (
        cfg["flash_attention_kv_repack_broadcast_copyback_hbm_roundtrip_barrier_only"]
        is True
    )


def test_flash_attention_kv_repack_copyback_data_only_is_gated():
    cfg = _read_flash_config(
        {"SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_DATA_ONLY": "1"}
    )

    assert cfg["flash_attention_kv_repack_broadcast_copyback_data_only"] is True


def test_flash_attention_kv_repack_copyback_replace_consumer_is_gated():
    cfg = _read_flash_config(
        {"SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_REPLACE_CONSUMER": "1"}
    )

    assert (
        cfg["flash_attention_kv_repack_broadcast_copyback_replace_consumer"] is True
    )


def test_flash_attention_kv_repack_copyback_compute_only_is_gated():
    cfg = _read_flash_config(
        {"SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_COMPUTE_ONLY": "1"}
    )

    assert cfg["flash_attention_kv_repack_broadcast_copyback_compute_only"] is True


def test_flash_attention_kv_repack_copyback_exact_clone_is_gated():
    cfg = _read_flash_config(
        {"SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_EXACT_CLONE": "1"}
    )

    assert cfg["flash_attention_kv_repack_broadcast_copyback_exact_clone"] is True


def test_flash_attention_kv_repack_copyback_preserve_consumer_name_is_gated():
    cfg = _read_flash_config(
        {
            (
                "SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_"
                "PRESERVE_CONSUMER_NAME"
            ): "1"
        }
    )

    assert (
        cfg["flash_attention_kv_repack_broadcast_copyback_preserve_consumer_name"]
        is True
    )


def test_flash_attention_kv_repack_plan_artifact_is_independently_gated():
    cfg = _read_flash_config(
        {"SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PLAN_ARTIFACT": "1"}
    )

    assert cfg["flash_attention_kv_repack_broadcast_plan_artifact"] is True
    assert cfg["flash_attention_onchip_sdpa"] is False
    assert cfg["flash_attention_mixed_pipeline"] is False


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
