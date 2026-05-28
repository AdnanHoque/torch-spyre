# Copyright 2026 The Torch-Spyre Authors.
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

"""Standalone tests for the on-chip realization first cut (same-core same-shard).

onchip_realize.py and codegen/onchip_bridge.py are torch-free, so both are
loaded by file path (no torch_spyre import). Asserts: LX bases non-overlapping
and in-capacity, datadscs_ structure (sharding match, memId per core), and that
over-capacity / mismatched-shard edges fail closed.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODEGEN = os.path.normpath(
    os.path.join(_HERE, "..", "..", "torch_spyre", "_inductor", "codegen")
)
_REAL = os.path.normpath(
    os.path.join(_HERE, "..", "..", "torch_spyre", "_inductor", "onchip_realize.py")
)
_BUNDLE = os.path.normpath(
    os.path.join(_HERE, "..", "..", "torch_spyre", "_inductor", "codegen", "bundle.py")
)
_MISSING = object()


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Build a minimal package shim so onchip_realize's relative import resolves.
for pkg in ("torch_spyre", "torch_spyre._inductor", "torch_spyre._inductor.codegen"):
    sys.modules.setdefault(pkg, types.ModuleType(pkg))
_load("torch_spyre._inductor.codegen.onchip_bridge", os.path.join(_CODEGEN, "onchip_bridge.py"))
rz = _load("torch_spyre._inductor.onchip_realize", _REAL)


def _hbm_dataop_addr(byte_addr):
    assert byte_addr % rz.STICK_BYTES == 0
    return byte_addr // rz.STICK_BYTES


class _Logger:
    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass


def _install_bundle_stubs(
    *,
    pointwise_handoff=False,
    layout_xform_pair_tile=None,
    layout_xform_pair_overlap=False,
    layout_xform_pair_result=True,
    layout_xform_lookahead_tile=-1,
    layout_xform_lookahead_result=True,
    layout_xform_hoist_tile=-1,
    layout_xform_hoist_result=True,
    layout_xform_pointwise_region0=None,
    causal_plan_artifact=False,
    kv_repack_plan_artifact=False,
    kv_repack_pair_tile=-1,
    kv_repack_pair_ifn_transfer=True,
    kv_repack_pair_subpiece_reuse=True,
    kv_repack_pair_group_size=0,
    kv_repack_pair_self_resident_source=False,
    kv_repack_pair_hbm_source=False,
    kv_repack_pair_hbm_direct_load=False,
    kv_repack_pair_hbm_staged=False,
    kv_repack_pair_consumer_core_state_init=True,
    kv_repack_pair_consumer_ds_type="",
    kv_repack_pair_consumer_lx_alloc_style="",
    kv_repack_pair_use_unicast=-1,
    kv_repack_pair_force_mc_mode=-1,
    kv_repack_hbm_staged_hoist_tile=-1,
    kv_repack_hbm_staged_hoist_result=True,
    kv_repack_hbm_prefetch_hoist_tile=-1,
    kv_repack_hbm_prefetch_lx_base=-1,
    kv_repack_hbm_prefetch_serial=False,
    kv_repack_hbm_prefetch_prefill_current=False,
    kv_repack_hbm_prefetch_redundant_future=False,
    kv_repack_hbm_prefetch_serialize_current=False,
    kv_repack_hbm_prefetch_external_future=False,
    kv_repack_hbm_prefetch_overlap_after_sync=True,
    kv_repack_hbm_prefetch_tail_current=False,
    kv_repack_hbm_prefetch_source_fanout=False,
    kv_repack_hbm_prefetch_loader_fanout=False,
    kv_repack_hbm_prefetch_loader_core=0,
    kv_repack_hbm_prefetch_loader_lx_base=-1,
    kv_repack_hbm_prefetch_fanout_use_unicast=-1,
    kv_repack_hbm_prefetch_fanout_use_lxsfp_lx_transfers=-1,
    kv_repack_hbm_prefetch_fanout_copyback_core=-2,
    kv_repack_hbm_prefetch_fanout_restrict_to_copyback_core=False,
    kv_repack_hbm_prefetch_loader_copyback_without_fanout=False,
    kv_repack_hbm_prefetch_loader_fanout_full_tile_pieces=False,
    kv_repack_hbm_prefetch_serialize_loader_core=False,
    kv_repack_hbm_prefetch_lx_roundtrip=False,
    kv_repack_hbm_prefetch_corelet1=False,
    kv_repack_hbm_prefetch_hoist_result=True,
    kv_repack_pair_result=True,
    kv_repack_copyback_tile=-1,
    kv_repack_copyback_core=-1,
    kv_repack_copyback_hbm_roundtrip=False,
    kv_repack_copyback_hbm_source_fanout=False,
    kv_repack_copyback_hbm_direct_load=False,
    kv_repack_copyback_hbm_roundtrip_load_only=False,
    kv_repack_copyback_hbm_roundtrip_barrier_only=False,
    kv_repack_copyback_data_only=False,
    kv_repack_copyback_replace_consumer=False,
    kv_repack_copyback_compute_only=False,
    kv_repack_copyback_exact_clone=False,
    kv_repack_copyback_preserve_consumer_name=False,
    kv_repack_copyback_result=True,
    ifn_prefix_force=False,
    execute_tile=-1,
    tile_artifacts=None,
):
    calls = {
        "layout_xform": [],
        "layout_xform_overlap": [],
        "layout_xform_lookahead": [],
        "layout_xform_hoist": [],
        "kv_repack_plan": [],
        "kv_repack_pair": [],
        "kv_repack_pair_ifn_transfer": [],
        "kv_repack_pair_subpiece_reuse": [],
        "kv_repack_pair_group_size": [],
        "kv_repack_pair_self_resident_source": [],
        "kv_repack_pair_hbm_source": [],
        "kv_repack_pair_hbm_direct_load": [],
        "kv_repack_pair_hbm_staged": [],
        "kv_repack_pair_consumer_core_state_init": [],
        "kv_repack_pair_consumer_ds_type": [],
        "kv_repack_pair_consumer_lx_alloc_style": [],
        "kv_repack_pair_use_unicast": [],
        "kv_repack_pair_force_mc_mode": [],
        "kv_repack_hbm_staged_hoist": [],
        "kv_repack_hbm_prefetch_hoist": [],
        "kv_repack_hbm_prefetch_lx_base": [],
        "kv_repack_hbm_prefetch_serial": [],
        "kv_repack_hbm_prefetch_prefill_current": [],
        "kv_repack_hbm_prefetch_redundant_future": [],
        "kv_repack_hbm_prefetch_serialize_current": [],
        "kv_repack_hbm_prefetch_external_future": [],
        "kv_repack_hbm_prefetch_overlap_after_sync": [],
        "kv_repack_hbm_prefetch_tail_current": [],
        "kv_repack_hbm_prefetch_source_fanout": [],
        "kv_repack_hbm_prefetch_loader_fanout": [],
        "kv_repack_hbm_prefetch_loader_core": [],
        "kv_repack_hbm_prefetch_loader_lx_base": [],
        "kv_repack_hbm_prefetch_fanout_use_unicast": [],
        "kv_repack_hbm_prefetch_fanout_use_lxsfp_lx_transfers": [],
        "kv_repack_hbm_prefetch_fanout_copyback_core": [],
        "kv_repack_hbm_prefetch_fanout_restrict_to_copyback_core": [],
        "kv_repack_hbm_prefetch_loader_copyback_without_fanout": [],
        "kv_repack_hbm_prefetch_loader_fanout_full_tile_pieces": [],
        "kv_repack_hbm_prefetch_serialize_loader_core": [],
        "kv_repack_hbm_prefetch_lx_roundtrip": [],
        "kv_repack_hbm_prefetch_corelet_id": [],
        "kv_repack_copyback": [],
        "kv_repack_copyback_core": [],
        "kv_repack_copyback_hbm_roundtrip": [],
        "kv_repack_copyback_hbm_source_fanout": [],
        "kv_repack_copyback_hbm_direct_load": [],
        "kv_repack_copyback_hbm_roundtrip_load_only": [],
        "kv_repack_copyback_hbm_roundtrip_barrier_only": [],
        "kv_repack_copyback_data_only": [],
        "kv_repack_copyback_replace_consumer": [],
        "kv_repack_copyback_compute_only": [],
        "kv_repack_copyback_exact_clone": [],
        "kv_repack_copyback_preserve_consumer_name": [],
        "pointwise": [],
    }

    config = types.ModuleType("torch_spyre._inductor.config")
    config.onchip_handoff_realize = False
    config.onchip_attention_score_handoff = False
    config.onchip_static_matmul_handoff = False
    config.onchip_handoff_min_bytes = 1
    config.flash_attention_mixed_pipeline = True
    config.flash_attention_pointwise_handoff = pointwise_handoff
    config.flash_attention_score_scale_handoff = False
    config.flash_attention_mixed_pipeline_artifact = False
    config.flash_attention_mixed_pipeline_execute_tile = execute_tile
    config.flash_attention_mixed_pipeline_value_flow_tile = -1
    config.flash_attention_mixed_pipeline_ifn_pair_tile = -1
    config.flash_attention_mixed_pipeline_ifn_prefix_force = ifn_prefix_force
    if layout_xform_pair_tile is None:
        layout_xform_pair_tile = rz.LAYOUT_XFORM_PAIR_AUTO_TILE
    if layout_xform_pointwise_region0 is None:
        layout_xform_pointwise_region0 = rz.LAYOUT_XFORM_COMPOSE_POINTWISE_LX_BASE
    config.flash_attention_mixed_pipeline_layout_xform_pair_tile = (
        layout_xform_pair_tile
    )
    config.flash_attention_mixed_pipeline_layout_xform_pair_overlap = (
        layout_xform_pair_overlap
    )
    config.flash_attention_mixed_pipeline_layout_xform_lookahead_tile = (
        layout_xform_lookahead_tile
    )
    config.flash_attention_mixed_pipeline_layout_xform_hoist_tile = (
        layout_xform_hoist_tile
    )
    config.flash_attention_mixed_pipeline_overlap = False
    config.causal_idx_to_mask_plan_artifact = causal_plan_artifact
    config.flash_attention_kv_repack_broadcast_plan_artifact = (
        kv_repack_plan_artifact
    )
    config.flash_attention_kv_repack_broadcast_pair_tile = kv_repack_pair_tile
    config.flash_attention_kv_repack_broadcast_pair_ifn_transfer = (
        kv_repack_pair_ifn_transfer
    )
    config.flash_attention_kv_repack_broadcast_pair_subpiece_reuse = (
        kv_repack_pair_subpiece_reuse
    )
    config.flash_attention_kv_repack_broadcast_pair_group_size = (
        kv_repack_pair_group_size
    )
    config.flash_attention_kv_repack_broadcast_pair_self_resident_source = (
        kv_repack_pair_self_resident_source
    )
    config.flash_attention_kv_repack_broadcast_pair_hbm_source = (
        kv_repack_pair_hbm_source
    )
    config.flash_attention_kv_repack_broadcast_pair_hbm_direct_load = (
        kv_repack_pair_hbm_direct_load
    )
    config.flash_attention_kv_repack_broadcast_pair_hbm_staged = (
        kv_repack_pair_hbm_staged
    )
    config.flash_attention_kv_repack_broadcast_pair_consumer_core_state_init = (
        kv_repack_pair_consumer_core_state_init
    )
    config.flash_attention_kv_repack_broadcast_pair_consumer_ds_type = (
        kv_repack_pair_consumer_ds_type
    )
    config.flash_attention_kv_repack_broadcast_pair_consumer_lx_alloc_style = (
        kv_repack_pair_consumer_lx_alloc_style
    )
    config.flash_attention_kv_repack_broadcast_pair_use_unicast = (
        kv_repack_pair_use_unicast
    )
    config.flash_attention_kv_repack_broadcast_pair_force_mc_mode = (
        kv_repack_pair_force_mc_mode
    )
    config.flash_attention_kv_repack_hbm_staged_hoist_tile = (
        kv_repack_hbm_staged_hoist_tile
    )
    config.flash_attention_kv_repack_hbm_prefetch_hoist_tile = (
        kv_repack_hbm_prefetch_hoist_tile
    )
    config.flash_attention_kv_repack_hbm_prefetch_lx_base = (
        kv_repack_hbm_prefetch_lx_base
    )
    config.flash_attention_kv_repack_hbm_prefetch_serial = (
        kv_repack_hbm_prefetch_serial
    )
    config.flash_attention_kv_repack_hbm_prefetch_prefill_current = (
        kv_repack_hbm_prefetch_prefill_current
    )
    config.flash_attention_kv_repack_hbm_prefetch_redundant_future = (
        kv_repack_hbm_prefetch_redundant_future
    )
    config.flash_attention_kv_repack_hbm_prefetch_serialize_current = (
        kv_repack_hbm_prefetch_serialize_current
    )
    config.flash_attention_kv_repack_hbm_prefetch_external_future = (
        kv_repack_hbm_prefetch_external_future
    )
    config.flash_attention_kv_repack_hbm_prefetch_overlap_after_sync = (
        kv_repack_hbm_prefetch_overlap_after_sync
    )
    config.flash_attention_kv_repack_hbm_prefetch_tail_current = (
        kv_repack_hbm_prefetch_tail_current
    )
    config.flash_attention_kv_repack_hbm_prefetch_source_fanout = (
        kv_repack_hbm_prefetch_source_fanout
    )
    config.flash_attention_kv_repack_hbm_prefetch_loader_fanout = (
        kv_repack_hbm_prefetch_loader_fanout
    )
    config.flash_attention_kv_repack_hbm_prefetch_loader_core = (
        kv_repack_hbm_prefetch_loader_core
    )
    config.flash_attention_kv_repack_hbm_prefetch_loader_lx_base = (
        kv_repack_hbm_prefetch_loader_lx_base
    )
    config.flash_attention_kv_repack_hbm_prefetch_fanout_use_unicast = (
        kv_repack_hbm_prefetch_fanout_use_unicast
    )
    config.flash_attention_kv_repack_hbm_prefetch_fanout_use_lxsfp_lx_transfers = (
        kv_repack_hbm_prefetch_fanout_use_lxsfp_lx_transfers
    )
    config.flash_attention_kv_repack_hbm_prefetch_fanout_copyback_core = (
        kv_repack_hbm_prefetch_fanout_copyback_core
    )
    config.flash_attention_kv_repack_hbm_prefetch_fanout_restrict_to_copyback_core = (
        kv_repack_hbm_prefetch_fanout_restrict_to_copyback_core
    )
    config.flash_attention_kv_repack_hbm_prefetch_loader_copyback_without_fanout = (
        kv_repack_hbm_prefetch_loader_copyback_without_fanout
    )
    config.flash_attention_kv_repack_hbm_prefetch_loader_fanout_full_tile_pieces = (
        kv_repack_hbm_prefetch_loader_fanout_full_tile_pieces
    )
    config.flash_attention_kv_repack_hbm_prefetch_serialize_loader_core = (
        kv_repack_hbm_prefetch_serialize_loader_core
    )
    config.flash_attention_kv_repack_hbm_prefetch_lx_roundtrip = (
        kv_repack_hbm_prefetch_lx_roundtrip
    )
    config.flash_attention_kv_repack_hbm_prefetch_corelet1 = (
        kv_repack_hbm_prefetch_corelet1
    )
    config.flash_attention_kv_repack_broadcast_copyback_tile = (
        kv_repack_copyback_tile
    )
    config.flash_attention_kv_repack_broadcast_copyback_core = (
        kv_repack_copyback_core
    )
    config.flash_attention_kv_repack_broadcast_copyback_hbm_roundtrip = (
        kv_repack_copyback_hbm_roundtrip
    )
    config.flash_attention_kv_repack_broadcast_copyback_hbm_source_fanout = (
        kv_repack_copyback_hbm_source_fanout
    )
    config.flash_attention_kv_repack_broadcast_copyback_hbm_direct_load = (
        kv_repack_copyback_hbm_direct_load
    )
    config.flash_attention_kv_repack_broadcast_copyback_hbm_roundtrip_load_only = (
        kv_repack_copyback_hbm_roundtrip_load_only
    )
    config.flash_attention_kv_repack_broadcast_copyback_hbm_roundtrip_barrier_only = (
        kv_repack_copyback_hbm_roundtrip_barrier_only
    )
    config.flash_attention_kv_repack_broadcast_copyback_data_only = (
        kv_repack_copyback_data_only
    )
    config.flash_attention_kv_repack_broadcast_copyback_replace_consumer = (
        kv_repack_copyback_replace_consumer
    )
    config.flash_attention_kv_repack_broadcast_copyback_compute_only = (
        kv_repack_copyback_compute_only
    )
    config.flash_attention_kv_repack_broadcast_copyback_exact_clone = (
        kv_repack_copyback_exact_clone
    )
    config.flash_attention_kv_repack_broadcast_copyback_preserve_consumer_name = (
        kv_repack_copyback_preserve_consumer_name
    )

    superdsc = types.ModuleType("torch_spyre._inductor.codegen.superdsc")
    superdsc.compile_op_spec = lambda _idx, spec: getattr(spec, "sdsc_json", spec)

    op_spec = types.ModuleType("torch_spyre._inductor.op_spec")
    op_spec.OpSpec = object

    logging_utils = types.ModuleType("torch_spyre._inductor.logging_utils")
    logging_utils.get_inductor_logger = lambda _name: _Logger()

    onchip_realize = types.ModuleType("torch_spyre._inductor.onchip_realize")
    onchip_realize.LAYOUT_XFORM_COMPOSE_POINTWISE_LX_BASE = (
        rz.LAYOUT_XFORM_COMPOSE_POINTWISE_LX_BASE
    )

    def build_flash_attention_layout_xform_pair_tile_artifacts(
        _sdscs,
        tile_index,
        *,
        name_prefix="mixed_flash_layout_xform_pair_tile",
        overlap_consumer=False,
    ):
        calls["layout_xform"].append(tile_index)
        calls["layout_xform_overlap"].append(overlap_consumer)
        if not layout_xform_pair_result:
            return None
        pred_name = f"{name_prefix}_2_predecessor"
        cons_name = f"{name_prefix}_2_consumer"
        return {
            "artifacts": [
                {
                    pred_name: {
                        "flashAttentionPipeline_": {
                            "tile_index": 2,
                            "requested_tile_index": tile_index,
                            "layout_xform_overlap_consumer": overlap_consumer,
                        }
                    }
                },
                {
                    cons_name: {
                        "flashAttentionPipeline_": {
                            "tile_index": 2,
                            "requested_tile_index": tile_index,
                            "layout_xform_overlap_consumer": overlap_consumer,
                        }
                    }
                },
            ],
            "replacements": {
                "0_batchmatmul": pred_name,
                "1_batchmatmul": cons_name,
            },
            "bundle_attrs": {},
            "pointwise_lx_region0": layout_xform_pointwise_region0,
        }

    onchip_realize.build_flash_attention_layout_xform_pair_tile_artifacts = (
        build_flash_attention_layout_xform_pair_tile_artifacts
    )

    def build_flash_attention_layout_xform_lookahead_tile_artifacts(
        _sdscs,
        tile_index,
        *,
        name_prefix="mixed_flash_pipeline_tile_layout_xform_lookahead",
    ):
        calls["layout_xform_lookahead"].append(tile_index)
        if not layout_xform_lookahead_result:
            return None
        current_pred = f"{name_prefix}_0_current_predecessor"
        future_pred = f"{name_prefix}_0_future_predecessor"
        current_cons = f"{name_prefix}_0_current_consumer"
        future_cons = f"{name_prefix}_0_future_consumer"
        return {
            "artifacts": [
                {current_pred: {"flashAttentionPipeline_": {}}},
                {future_pred: {"flashAttentionPipeline_": {}}},
                {current_cons: {"flashAttentionPipeline_": {}}},
                {future_cons: {"flashAttentionPipeline_": {}}},
            ],
            "replacements": {
                "0_batchmatmul": current_pred,
                "1_batchmatmul": future_pred,
                "2_batchmatmul": current_cons,
                "3_batchmatmul": future_cons,
            },
            "bundle_attrs": {},
            "pointwise_lx_region0": layout_xform_pointwise_region0,
        }

    onchip_realize.build_flash_attention_layout_xform_lookahead_tile_artifacts = (
        build_flash_attention_layout_xform_lookahead_tile_artifacts
    )

    def build_flash_attention_layout_xform_hoist_tile_artifacts(
        _sdscs,
        tile_index,
        *,
        name_prefix="mixed_flash_pipeline_tile_layout_xform_hoist",
    ):
        calls["layout_xform_hoist"].append(tile_index)
        if not layout_xform_hoist_result:
            return None
        current_cons = f"{name_prefix}_0_current_consumer"
        future_cons = f"{name_prefix}_0_future_consumer"
        return {
            "artifacts": [
                {current_cons: {"flashAttentionPipeline_": {}}},
                {future_cons: {"flashAttentionPipeline_": {}}},
            ],
            "replacements": {
                "0_batchmatmul": current_cons,
                "2_batchmatmul": future_cons,
            },
            "omissions": {"1_ReStickifyOpHBM"},
            "bundle_attrs": {},
            "pointwise_lx_region0": layout_xform_pointwise_region0,
        }

    onchip_realize.build_flash_attention_layout_xform_hoist_tile_artifacts = (
        build_flash_attention_layout_xform_hoist_tile_artifacts
    )

    def build_flash_attention_kv_repack_broadcast_plan_artifact(
        _sdscs,
        tile_index,
        *,
        input_idx,
        name_prefix="flash_kv_repack_broadcast_plan",
    ):
        calls["kv_repack_plan"].append((tile_index, input_idx))
        if not kv_repack_plan_artifact or tile_index != 1 or input_idx != 1:
            return None
        name = f"{name_prefix}_{tile_index}_input{input_idx}"
        return {
            name: {
                "numCoresUsed_": 32,
                "coreIdToDscSchedule": {"0": [[0, -1, 0, 0]]},
                "datadscs_": [],
                "dscs_": [],
                "opFuncsUsed_": ["STCDPOpLx"],
                "flashAttentionPipeline_": {
                    "kv_repack_broadcast_plan": True,
                    "kv_repack_broadcast_executable": False,
                    "tile_index": tile_index,
                },
            }
        }

    onchip_realize.build_flash_attention_kv_repack_broadcast_plan_artifact = (
        build_flash_attention_kv_repack_broadcast_plan_artifact
    )

    def build_flash_attention_kv_repack_broadcast_pair_artifacts(
        _sdscs,
        tile_index,
        *,
        name_prefix="mixed_flash_kv_repack_broadcast_pair",
        include_input_fetch_transfer=True,
        stcdp_subpiece_reuse=True,
        broadcast_group_size=0,
        self_resident_source=False,
        hbm_source=False,
        hbm_direct_load=False,
        hbm_staged=False,
        consumer_core_state_init=True,
        consumer_ds_type="",
        consumer_lx_alloc_style="",
        stcdp_use_unicast=-1,
        stcdp_force_mc_mode=-1,
    ):
        calls["kv_repack_pair"].append(tile_index)
        calls["kv_repack_pair_ifn_transfer"].append(include_input_fetch_transfer)
        calls["kv_repack_pair_subpiece_reuse"].append(stcdp_subpiece_reuse)
        calls["kv_repack_pair_group_size"].append(broadcast_group_size)
        calls["kv_repack_pair_self_resident_source"].append(self_resident_source)
        calls["kv_repack_pair_hbm_source"].append(hbm_source)
        calls["kv_repack_pair_hbm_direct_load"].append(hbm_direct_load)
        calls["kv_repack_pair_hbm_staged"].append(hbm_staged)
        calls["kv_repack_pair_consumer_core_state_init"].append(
            consumer_core_state_init
        )
        calls["kv_repack_pair_consumer_ds_type"].append(consumer_ds_type)
        calls["kv_repack_pair_consumer_lx_alloc_style"].append(
            consumer_lx_alloc_style
        )
        calls["kv_repack_pair_use_unicast"].append(stcdp_use_unicast)
        calls["kv_repack_pair_force_mc_mode"].append(stcdp_force_mc_mode)
        if not kv_repack_pair_result:
            return None
        pred_name = f"{name_prefix}_1_input1_producer"
        cons_name = f"{name_prefix}_1_input1_consumer"
        artifacts = [{cons_name: {"flashAttentionPipeline_": {}}}]
        replacements = {"2_batchmatmul": cons_name}
        if not (hbm_source or hbm_direct_load or hbm_staged):
            artifacts.insert(0, {pred_name: {"flashAttentionPipeline_": {}}})
            replacements = {
                "1_ReStickifyOpHBM": pred_name,
                **replacements,
            }
        return {
            "artifacts": artifacts,
            "replacements": replacements,
            "bundle_attrs": {},
            "pointwise_lx_region0": layout_xform_pointwise_region0,
        }

    onchip_realize.build_flash_attention_kv_repack_broadcast_pair_artifacts = (
        build_flash_attention_kv_repack_broadcast_pair_artifacts
    )

    def build_flash_attention_kv_repack_hbm_staged_hoist_tile_artifacts(
        _sdscs,
        tile_index,
        *,
        name_prefix="mixed_flash_kv_repack_hbm_staged_hoist",
    ):
        calls["kv_repack_hbm_staged_hoist"].append(tile_index)
        if not kv_repack_hbm_staged_hoist_result:
            return None
        producer_name = f"{name_prefix}_0_future_producer"
        future_name = f"{name_prefix}_0_future_kv_1_input1_consumer"
        return {
            "artifacts": [
                {producer_name: {"flashAttentionPipeline_": {}}},
                {future_name: {"flashAttentionPipeline_": {}}},
            ],
            "replacements": {
                "2_batchmatmul": future_name,
            },
            "insertions_before": {"0_batchmatmul": [producer_name]},
            "omissions": {"1_ReStickifyOpHBM"},
            "bundle_attrs": {},
            "pointwise_lx_region0": layout_xform_pointwise_region0,
        }

    onchip_realize.build_flash_attention_kv_repack_hbm_staged_hoist_tile_artifacts = (
        build_flash_attention_kv_repack_hbm_staged_hoist_tile_artifacts
    )

    def build_flash_attention_kv_repack_hbm_prefetch_hoist_tile_artifacts(
        _sdscs,
        tile_index,
        *,
        name_prefix="mixed_flash_kv_repack_hbm_prefetch_hoist",
        prefetch_lx_base=None,
        serial_prefetch=False,
        prefill_current_input=False,
        redundant_future_prefetch=False,
        serialize_current_prefetch=False,
        external_future_prefetch=False,
        overlap_after_sync=True,
        tail_current_prefetch=False,
        prefetch_source_fanout=False,
        prefetch_loader_fanout=False,
        prefetch_loader_core_id=0,
        prefetch_loader_lx_base=-1,
        prefetch_fanout_use_unicast=-1,
        prefetch_fanout_use_lxsfp_lx_transfers=-1,
        prefetch_fanout_copyback_core=-2,
        prefetch_fanout_restrict_to_copyback_core=False,
        prefetch_loader_copyback_without_fanout=False,
        prefetch_loader_fanout_full_tile_pieces=False,
        serialize_loader_core_prefetch=False,
        prefetch_lx_roundtrip=False,
        prefetch_corelet_id=None,
    ):
        calls["kv_repack_hbm_prefetch_hoist"].append(tile_index)
        calls["kv_repack_hbm_prefetch_lx_base"].append(prefetch_lx_base)
        calls["kv_repack_hbm_prefetch_serial"].append(serial_prefetch)
        calls["kv_repack_hbm_prefetch_prefill_current"].append(
            prefill_current_input
        )
        calls["kv_repack_hbm_prefetch_redundant_future"].append(
            redundant_future_prefetch
        )
        calls["kv_repack_hbm_prefetch_serialize_current"].append(
            serialize_current_prefetch
        )
        calls["kv_repack_hbm_prefetch_external_future"].append(
            external_future_prefetch
        )
        calls["kv_repack_hbm_prefetch_overlap_after_sync"].append(
            overlap_after_sync
        )
        calls["kv_repack_hbm_prefetch_tail_current"].append(
            tail_current_prefetch
        )
        calls["kv_repack_hbm_prefetch_source_fanout"].append(
            prefetch_source_fanout
        )
        calls["kv_repack_hbm_prefetch_loader_fanout"].append(
            prefetch_loader_fanout
        )
        calls["kv_repack_hbm_prefetch_loader_core"].append(
            prefetch_loader_core_id
        )
        calls["kv_repack_hbm_prefetch_loader_lx_base"].append(
            prefetch_loader_lx_base
        )
        calls["kv_repack_hbm_prefetch_fanout_use_unicast"].append(
            prefetch_fanout_use_unicast
        )
        calls["kv_repack_hbm_prefetch_fanout_use_lxsfp_lx_transfers"].append(
            prefetch_fanout_use_lxsfp_lx_transfers
        )
        calls["kv_repack_hbm_prefetch_fanout_copyback_core"].append(
            prefetch_fanout_copyback_core
        )
        calls["kv_repack_hbm_prefetch_fanout_restrict_to_copyback_core"].append(
            prefetch_fanout_restrict_to_copyback_core
        )
        calls["kv_repack_hbm_prefetch_loader_copyback_without_fanout"].append(
            prefetch_loader_copyback_without_fanout
        )
        calls["kv_repack_hbm_prefetch_loader_fanout_full_tile_pieces"].append(
            prefetch_loader_fanout_full_tile_pieces
        )
        calls["kv_repack_hbm_prefetch_serialize_loader_core"].append(
            serialize_loader_core_prefetch
        )
        calls["kv_repack_hbm_prefetch_lx_roundtrip"].append(
            prefetch_lx_roundtrip
        )
        calls["kv_repack_hbm_prefetch_corelet_id"].append(prefetch_corelet_id)
        if not kv_repack_hbm_prefetch_hoist_result:
            return None
        producer_name = f"{name_prefix}_0_future_producer"
        current_name = f"{name_prefix}_0_current_prefetch"
        future_name = f"{name_prefix}_0_future_consumer"
        return {
            "artifacts": [
                {producer_name: {"flashAttentionPipeline_": {}}},
                {current_name: {"flashAttentionPipeline_": {}}},
                {future_name: {"flashAttentionPipeline_": {}}},
            ],
            "replacements": {
                "0_batchmatmul": current_name,
                "2_batchmatmul": future_name,
            },
            "insertions_before": {"0_batchmatmul": [producer_name]},
            "omissions": {"1_ReStickifyOpHBM"},
            "bundle_attrs": {},
            "pointwise_lx_region0": layout_xform_pointwise_region0,
        }

    onchip_realize.build_flash_attention_kv_repack_hbm_prefetch_hoist_tile_artifacts = (
        build_flash_attention_kv_repack_hbm_prefetch_hoist_tile_artifacts
    )

    def build_flash_attention_kv_repack_broadcast_copyback_artifacts(
        _sdscs,
        tile_index,
        *,
        name_prefix="mixed_flash_kv_repack_broadcast_copyback",
        stcdp_subpiece_reuse=True,
        broadcast_group_size=0,
        self_resident_source=False,
        stcdp_use_unicast=-1,
        stcdp_force_mc_mode=-1,
        readback_core=-1,
        direct_source=False,
        hbm_roundtrip=False,
        hbm_source_fanout=False,
        hbm_direct_load=False,
        hbm_roundtrip_load_only=False,
        hbm_roundtrip_barrier_only=False,
        data_only=False,
        replace_consumer=False,
        compute_only=False,
        exact_clone=False,
        preserve_consumer_name=False,
    ):
        calls["kv_repack_copyback"].append(tile_index)
        calls["kv_repack_pair_subpiece_reuse"].append(stcdp_subpiece_reuse)
        calls["kv_repack_pair_group_size"].append(broadcast_group_size)
        calls["kv_repack_pair_self_resident_source"].append(self_resident_source)
        calls["kv_repack_pair_use_unicast"].append(stcdp_use_unicast)
        calls["kv_repack_pair_force_mc_mode"].append(stcdp_force_mc_mode)
        calls["kv_repack_copyback_core"].append(readback_core)
        calls.setdefault("kv_repack_copyback_direct_source", []).append(
            direct_source
        )
        calls["kv_repack_copyback_hbm_roundtrip"].append(hbm_roundtrip)
        calls["kv_repack_copyback_hbm_source_fanout"].append(hbm_source_fanout)
        calls["kv_repack_copyback_hbm_direct_load"].append(hbm_direct_load)
        calls["kv_repack_copyback_hbm_roundtrip_load_only"].append(
            hbm_roundtrip_load_only
        )
        calls["kv_repack_copyback_hbm_roundtrip_barrier_only"].append(
            hbm_roundtrip_barrier_only
        )
        calls["kv_repack_copyback_data_only"].append(data_only)
        calls["kv_repack_copyback_replace_consumer"].append(replace_consumer)
        calls["kv_repack_copyback_compute_only"].append(compute_only)
        calls["kv_repack_copyback_exact_clone"].append(exact_clone)
        calls["kv_repack_copyback_preserve_consumer_name"].append(
            preserve_consumer_name
        )
        if not kv_repack_copyback_result:
            return None
        pred_name = f"{name_prefix}_1_input1_producer"
        copyback_name = f"{name_prefix}_1_input1_copyback"
        return {
            "artifacts": [
                {pred_name: {"flashAttentionPipeline_": {}}},
                {copyback_name: {"flashAttentionPipeline_": {}}},
            ],
            "replacements": {
                "1_ReStickifyOpHBM": pred_name,
            },
            "insertions_before": {"2_batchmatmul": [copyback_name]},
            "bundle_attrs": {},
            "pointwise_lx_region0": layout_xform_pointwise_region0,
        }

    onchip_realize.build_flash_attention_kv_repack_broadcast_copyback_artifacts = (
        build_flash_attention_kv_repack_broadcast_copyback_artifacts
    )
    onchip_realize.build_flash_attention_ifn_pair_tile_artifacts = (
        lambda *_args, **_kwargs: None
    )
    onchip_realize.build_flash_attention_pipeline_artifact = (
        lambda *_args, **_kwargs: None
    )
    onchip_realize.build_flash_attention_pipeline_tile_artifacts = (
        lambda *_args, **_kwargs: list(tile_artifacts or [])
    )
    onchip_realize.build_flash_attention_value_flow_tile_artifact = (
        lambda *_args, **_kwargs: None
    )
    onchip_realize.flash_attention_ifn_pair_tile_rejection_reasons = (
        lambda *_args, **_kwargs: []
    )
    onchip_realize.flash_attention_layout_xform_pair_rejection_reasons = (
        lambda *_args, **_kwargs: []
    )
    onchip_realize.flash_attention_layout_xform_lookahead_rejection_reasons = (
        lambda *_args, **_kwargs: []
    )
    onchip_realize.flash_attention_layout_xform_hoist_rejection_reasons = (
        lambda *_args, **_kwargs: []
    )
    onchip_realize.flash_attention_kv_repack_broadcast_pair_rejection_reasons = (
        lambda *_args, **_kwargs: []
    )
    onchip_realize.flash_attention_kv_repack_hbm_staged_hoist_rejection_reasons = (
        lambda *_args, **_kwargs: []
    )
    onchip_realize.flash_attention_kv_repack_hbm_prefetch_hoist_rejection_reasons = (
        lambda *_args, **_kwargs: []
    )
    onchip_realize.flash_attention_kv_repack_broadcast_copyback_rejection_reasons = (
        lambda *_args, **_kwargs: []
    )
    onchip_realize.flash_attention_value_flow_tile_rejection_reasons = (
        lambda *_args, **_kwargs: []
    )

    def realize_flash_attention_pointwise_handoffs(*_args, **_kwargs):
        calls["pointwise"].append(_kwargs)
        return 1

    onchip_realize.realize_flash_attention_pointwise_handoffs = (
        realize_flash_attention_pointwise_handoffs
    )
    onchip_realize.realize_onchip_handoff = lambda *_args, **_kwargs: False

    packages = {
        "torch_spyre": types.ModuleType("torch_spyre"),
        "torch_spyre._inductor": types.ModuleType("torch_spyre._inductor"),
        "torch_spyre._inductor.codegen": types.ModuleType(
            "torch_spyre._inductor.codegen"
        ),
        "torch_spyre._inductor.config": config,
        "torch_spyre._inductor.codegen.superdsc": superdsc,
        "torch_spyre._inductor.op_spec": op_spec,
        "torch_spyre._inductor.logging_utils": logging_utils,
        "torch_spyre._inductor.onchip_realize": onchip_realize,
    }
    for name, module in packages.items():
        sys.modules[name] = module
    _load(
        "torch_spyre._inductor.codegen.causal_mask_dataop",
        os.path.join(_CODEGEN, "causal_mask_dataop.py"),
    )

    return calls


def _load_bundle_with_stubs(
    *,
    pointwise_handoff=False,
    layout_xform_pair_tile=None,
    layout_xform_pair_overlap=False,
    layout_xform_pair_result=True,
    layout_xform_lookahead_tile=-1,
    layout_xform_lookahead_result=True,
    layout_xform_hoist_tile=-1,
    layout_xform_hoist_result=True,
    layout_xform_pointwise_region0=None,
    causal_plan_artifact=False,
    kv_repack_plan_artifact=False,
    kv_repack_pair_tile=-1,
    kv_repack_pair_ifn_transfer=True,
    kv_repack_pair_subpiece_reuse=True,
    kv_repack_pair_group_size=0,
    kv_repack_pair_self_resident_source=False,
    kv_repack_pair_hbm_source=False,
    kv_repack_pair_hbm_direct_load=False,
    kv_repack_pair_hbm_staged=False,
    kv_repack_pair_consumer_core_state_init=True,
    kv_repack_pair_consumer_ds_type="",
    kv_repack_pair_consumer_lx_alloc_style="",
    kv_repack_pair_use_unicast=-1,
    kv_repack_pair_force_mc_mode=-1,
    kv_repack_hbm_staged_hoist_tile=-1,
    kv_repack_hbm_staged_hoist_result=True,
    kv_repack_hbm_prefetch_hoist_tile=-1,
    kv_repack_hbm_prefetch_lx_base=-1,
    kv_repack_hbm_prefetch_serial=False,
    kv_repack_hbm_prefetch_prefill_current=False,
    kv_repack_hbm_prefetch_redundant_future=False,
    kv_repack_hbm_prefetch_serialize_current=False,
    kv_repack_hbm_prefetch_external_future=False,
    kv_repack_hbm_prefetch_overlap_after_sync=True,
    kv_repack_hbm_prefetch_tail_current=False,
    kv_repack_hbm_prefetch_source_fanout=False,
    kv_repack_hbm_prefetch_loader_fanout=False,
    kv_repack_hbm_prefetch_loader_core=0,
    kv_repack_hbm_prefetch_loader_lx_base=-1,
    kv_repack_hbm_prefetch_fanout_use_unicast=-1,
    kv_repack_hbm_prefetch_fanout_use_lxsfp_lx_transfers=-1,
    kv_repack_hbm_prefetch_fanout_copyback_core=-2,
    kv_repack_hbm_prefetch_fanout_restrict_to_copyback_core=False,
    kv_repack_hbm_prefetch_loader_copyback_without_fanout=False,
    kv_repack_hbm_prefetch_loader_fanout_full_tile_pieces=False,
    kv_repack_hbm_prefetch_serialize_loader_core=False,
    kv_repack_hbm_prefetch_lx_roundtrip=False,
    kv_repack_hbm_prefetch_corelet1=False,
    kv_repack_hbm_prefetch_hoist_result=True,
    kv_repack_pair_result=True,
    kv_repack_copyback_tile=-1,
    kv_repack_copyback_core=-1,
    kv_repack_copyback_hbm_roundtrip=False,
    kv_repack_copyback_hbm_source_fanout=False,
    kv_repack_copyback_hbm_direct_load=False,
    kv_repack_copyback_hbm_roundtrip_load_only=False,
    kv_repack_copyback_hbm_roundtrip_barrier_only=False,
    kv_repack_copyback_data_only=False,
    kv_repack_copyback_replace_consumer=False,
    kv_repack_copyback_compute_only=False,
    kv_repack_copyback_exact_clone=False,
    kv_repack_copyback_preserve_consumer_name=False,
    kv_repack_copyback_result=True,
    ifn_prefix_force=False,
    execute_tile=-1,
    tile_artifacts=None,
):
    names = [
        "torch_spyre",
        "torch_spyre._inductor",
        "torch_spyre._inductor.codegen",
        "torch_spyre._inductor.config",
        "torch_spyre._inductor.codegen.superdsc",
        "torch_spyre._inductor.codegen.causal_mask_dataop",
        "torch_spyre._inductor.op_spec",
        "torch_spyre._inductor.logging_utils",
        "torch_spyre._inductor.onchip_realize",
        "_test_bundle_under_test",
    ]
    saved = {name: sys.modules.get(name, _MISSING) for name in names}
    calls = _install_bundle_stubs(
        pointwise_handoff=pointwise_handoff,
        layout_xform_pair_tile=layout_xform_pair_tile,
        layout_xform_pair_overlap=layout_xform_pair_overlap,
        layout_xform_pair_result=layout_xform_pair_result,
        layout_xform_lookahead_tile=layout_xform_lookahead_tile,
        layout_xform_lookahead_result=layout_xform_lookahead_result,
        layout_xform_hoist_tile=layout_xform_hoist_tile,
        layout_xform_hoist_result=layout_xform_hoist_result,
        layout_xform_pointwise_region0=layout_xform_pointwise_region0,
        causal_plan_artifact=causal_plan_artifact,
        kv_repack_plan_artifact=kv_repack_plan_artifact,
        kv_repack_pair_tile=kv_repack_pair_tile,
        kv_repack_pair_ifn_transfer=kv_repack_pair_ifn_transfer,
        kv_repack_pair_subpiece_reuse=kv_repack_pair_subpiece_reuse,
        kv_repack_pair_group_size=kv_repack_pair_group_size,
        kv_repack_pair_self_resident_source=kv_repack_pair_self_resident_source,
        kv_repack_pair_hbm_source=kv_repack_pair_hbm_source,
        kv_repack_pair_hbm_direct_load=kv_repack_pair_hbm_direct_load,
        kv_repack_pair_hbm_staged=kv_repack_pair_hbm_staged,
        kv_repack_pair_consumer_core_state_init=(
            kv_repack_pair_consumer_core_state_init
        ),
        kv_repack_pair_consumer_ds_type=kv_repack_pair_consumer_ds_type,
        kv_repack_pair_consumer_lx_alloc_style=(
            kv_repack_pair_consumer_lx_alloc_style
        ),
        kv_repack_pair_use_unicast=kv_repack_pair_use_unicast,
        kv_repack_pair_force_mc_mode=kv_repack_pair_force_mc_mode,
        kv_repack_hbm_staged_hoist_tile=kv_repack_hbm_staged_hoist_tile,
        kv_repack_hbm_staged_hoist_result=kv_repack_hbm_staged_hoist_result,
        kv_repack_hbm_prefetch_hoist_tile=kv_repack_hbm_prefetch_hoist_tile,
        kv_repack_hbm_prefetch_lx_base=kv_repack_hbm_prefetch_lx_base,
        kv_repack_hbm_prefetch_serial=kv_repack_hbm_prefetch_serial,
        kv_repack_hbm_prefetch_prefill_current=(
            kv_repack_hbm_prefetch_prefill_current
        ),
        kv_repack_hbm_prefetch_redundant_future=(
            kv_repack_hbm_prefetch_redundant_future
        ),
        kv_repack_hbm_prefetch_serialize_current=(
            kv_repack_hbm_prefetch_serialize_current
        ),
        kv_repack_hbm_prefetch_external_future=(
            kv_repack_hbm_prefetch_external_future
        ),
        kv_repack_hbm_prefetch_overlap_after_sync=(
            kv_repack_hbm_prefetch_overlap_after_sync
        ),
        kv_repack_hbm_prefetch_tail_current=(
            kv_repack_hbm_prefetch_tail_current
        ),
        kv_repack_hbm_prefetch_source_fanout=(
            kv_repack_hbm_prefetch_source_fanout
        ),
        kv_repack_hbm_prefetch_loader_fanout=(
            kv_repack_hbm_prefetch_loader_fanout
        ),
        kv_repack_hbm_prefetch_loader_core=kv_repack_hbm_prefetch_loader_core,
        kv_repack_hbm_prefetch_loader_lx_base=(
            kv_repack_hbm_prefetch_loader_lx_base
        ),
        kv_repack_hbm_prefetch_fanout_use_unicast=(
            kv_repack_hbm_prefetch_fanout_use_unicast
        ),
        kv_repack_hbm_prefetch_fanout_use_lxsfp_lx_transfers=(
            kv_repack_hbm_prefetch_fanout_use_lxsfp_lx_transfers
        ),
        kv_repack_hbm_prefetch_fanout_copyback_core=(
            kv_repack_hbm_prefetch_fanout_copyback_core
        ),
        kv_repack_hbm_prefetch_fanout_restrict_to_copyback_core=(
            kv_repack_hbm_prefetch_fanout_restrict_to_copyback_core
        ),
        kv_repack_hbm_prefetch_loader_copyback_without_fanout=(
            kv_repack_hbm_prefetch_loader_copyback_without_fanout
        ),
        kv_repack_hbm_prefetch_loader_fanout_full_tile_pieces=(
            kv_repack_hbm_prefetch_loader_fanout_full_tile_pieces
        ),
        kv_repack_hbm_prefetch_serialize_loader_core=(
            kv_repack_hbm_prefetch_serialize_loader_core
        ),
        kv_repack_hbm_prefetch_lx_roundtrip=(
            kv_repack_hbm_prefetch_lx_roundtrip
        ),
        kv_repack_hbm_prefetch_corelet1=kv_repack_hbm_prefetch_corelet1,
        kv_repack_hbm_prefetch_hoist_result=kv_repack_hbm_prefetch_hoist_result,
        kv_repack_pair_result=kv_repack_pair_result,
        kv_repack_copyback_tile=kv_repack_copyback_tile,
        kv_repack_copyback_core=kv_repack_copyback_core,
        kv_repack_copyback_hbm_roundtrip=kv_repack_copyback_hbm_roundtrip,
        kv_repack_copyback_hbm_source_fanout=(
            kv_repack_copyback_hbm_source_fanout
        ),
        kv_repack_copyback_hbm_direct_load=kv_repack_copyback_hbm_direct_load,
        kv_repack_copyback_hbm_roundtrip_load_only=(
            kv_repack_copyback_hbm_roundtrip_load_only
        ),
        kv_repack_copyback_hbm_roundtrip_barrier_only=(
            kv_repack_copyback_hbm_roundtrip_barrier_only
        ),
        kv_repack_copyback_data_only=kv_repack_copyback_data_only,
        kv_repack_copyback_replace_consumer=kv_repack_copyback_replace_consumer,
        kv_repack_copyback_compute_only=kv_repack_copyback_compute_only,
        kv_repack_copyback_exact_clone=kv_repack_copyback_exact_clone,
        kv_repack_copyback_preserve_consumer_name=(
            kv_repack_copyback_preserve_consumer_name
        ),
        kv_repack_copyback_result=kv_repack_copyback_result,
        ifn_prefix_force=ifn_prefix_force,
        execute_tile=execute_tile,
        tile_artifacts=tile_artifacts,
    )
    spec = importlib.util.spec_from_file_location("_test_bundle_under_test", _BUNDLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_test_bundle_under_test"] = module
    spec.loader.exec_module(module)
    return module, calls, saved


def _restore_modules(saved):
    for name, module in saved.items():
        if module is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def test_same_core_same_shard_realizes_two_regions():
    r = rz.realize_same_core_handoff(
        iter_sizes={"mb_": 2048, "out_": 2048}, layout=["mb_", "out_"],
        stick_dim="out_", split_dim="out_", stick_size=64, num_cores=32,
        producer_ldsidx=2, consumer_ldsidx=0,
    )
    assert r is not None and r.realizable
    assert r.producer_base != r.consumer_base
    assert r.producer_base == 0 and r.consumer_base == r.slice_bytes
    assert r.consumer_base + r.slice_bytes <= rz.LX_CAPACITY_BYTES
    assert r.opfuncs == ["STCDPOpLx"]
    assert r.producer_flip.ldsidx == 2 and r.consumer_flip.ldsidx == 0


def test_datadsc_sharding_and_memid_match_consumer():
    r = rz.realize_same_core_handoff(
        iter_sizes={"mb_": 2048, "out_": 2048}, layout=["mb_", "out_"],
        stick_dim="out_", split_dim="out_", stick_size=64, num_cores=32,
        producer_ldsidx=0, consumer_ldsidx=0,
    )
    dataop = r.datadscs[0]["0_STCDPOpLx_dataop"]
    in_pieces = dataop["labeledDs_"][0]["PieceInfo"]
    out_pieces = dataop["labeledDs_"][1]["PieceInfo"]
    assert len(in_pieces) == 32 and len(out_pieces) == 32
    # same-shard => piece i on core i both sides (no ring); chunk = 2048/32.
    for i in range(32):
        assert in_pieces[i]["PlacementInfo"][0]["memId"] == [i]
        assert out_pieces[i]["PlacementInfo"][0]["memId"] == [i]
        assert in_pieces[i]["dimToSize_"]["out_"] == 64


def test_substick_split_on_stick_dim_pads_dataop_frame():
    r = rz.realize_same_core_handoff(
        iter_sizes={"mb_": 512, "out_": 512}, layout=["mb_", "out_"],
        stick_dim="out_", split_dim="out_", stick_size=64, num_cores=32,
        producer_ldsidx=0, consumer_ldsidx=0,
        region0=rz.PRODUCER_LX_BASE,
    )
    dataop = r.datadscs[0]["0_STCDPOpLx_dataop"]
    in_ld = dataop["labeledDs_"][0]
    out_ld = dataop["labeledDs_"][1]
    assert r.slice_bytes == rz.MIN_BRIDGE_REGION_BYTES
    assert r.consumer_base == rz.PRODUCER_LX_BASE + rz.MIN_BRIDGE_REGION_BYTES
    assert in_ld["dimToLayoutSize_"]["mb_"] == 2048
    assert in_ld["dimToLayoutSize_"]["out_"] == 2048
    assert out_ld["dimToLayoutSize_"]["mb_"] == 2048
    assert out_ld["dimToLayoutSize_"]["out_"] == 2048
    assert in_ld["PieceInfo"][0]["dimToSize_"]["mb_"] == 2048
    assert in_ld["PieceInfo"][0]["dimToSize_"]["out_"] == 64
    assert out_ld["PieceInfo"][0]["dimToSize_"]["mb_"] == 2048
    assert out_ld["PieceInfo"][0]["dimToSize_"]["out_"] == 64


def test_over_capacity_fails_closed():
    # 2048x(2048/2) cols = 1 MB/region; 2 regions = 2 MB == capacity, but the
    # slice doubles to >1MB at 1-core split -> 2 regions exceed 2 MB.
    r = rz.realize_same_core_handoff(
        iter_sizes={"mb_": 4096, "out_": 4096}, layout=["mb_", "out_"],
        stick_dim="out_", split_dim="out_", stick_size=64, num_cores=2,
        producer_ldsidx=0, consumer_ldsidx=0,
    )
    assert r is None


def test_indivisible_split_fails_closed():
    r = rz.realize_same_core_handoff(
        iter_sizes={"mb_": 100, "out_": 100}, layout=["mb_", "out_"],
        stick_dim="out_", split_dim="out_", stick_size=64, num_cores=32,
        producer_ldsidx=0, consumer_ldsidx=0,
    )
    assert r is None


def test_is_same_shard_diff_shard_false():
    assert rz.is_same_shard({"out": 32}, {"out": 32}, {"out": "out"})
    # producer splits mb, consumer splits out, identity map -> divergent shard.
    assert not rz.is_same_shard({"mb": 32}, {"out": 32}, {"out": "out", "mb": "mb"})


def _fake_sdsc(
    idx,
    op,
    shard,
    n_sizes,
    inputs,
    outputs,
    pdi,
    *,
    lx_pinned=False,
    input_neighbor_transfer=False,
    num_cores=32,
    core_slices=None,
):
    def lds(label, role):
        i = int(label.rsplit("-idx", 1)[1])
        mem_org = {"lx": {"isPresent": 1}}
        if not lx_pinned:
            mem_org = {"hbm": {"isPresent": 1}, "lx": {"isPresent": 1}}
        return {
            "ldsIdx_": i,
            "dsName_": f"Tensor{i}",
            "dsType_": role,
            "wordLength": 2,
            "dataFormat_": "SEN169_FP16",
            "memOrg_": mem_org,
        }

    def alloc(label, addr):
        i = int(label.rsplit("-idx", 1)[1])
        component = "lx" if lx_pinned else "hbm"
        return {
            "nodeType_": "allocate",
            "name_": f"allocate-Tensor{i}_{component}",
            "ldsIdx_": i,
            "component_": component,
            "startAddressCoreCorelet_": {
                "data_": {f"[{c}, 0, 0]": str(addr) for c in range(num_cores)}
            },
        }

    labels = {}
    for label, role, addr in inputs + outputs:
        labels[label] = (role, addr)
    dl = {
        "numCoresUsed_": 32,
        "N_": {"name_": "n", **n_sizes},
        "primaryDsInfo_": pdi,
        "labeledDs_": [lds(label, role) for label, (role, _addr) in labels.items()],
        "scheduleTree_": [alloc(label, addr) for label, (_role, addr) in labels.items()],
        "computeOp_": [
            {
                "inputLabeledDs": [label for label, _role, _addr in inputs],
                "outputLabeledDs": [label for label, _role, _addr in outputs],
            }
        ],
    }
    if input_neighbor_transfer:
        dl["scheduleTree_"].append(
            {
                "nodeType_": "transfer",
                "name_": "dummy_transfer_to_lx_neighbor_input",
                "src_": {
                    "unit_": "no_component",
                    "storage_": "no_component",
                },
                "dstVias_": [
                    {
                        "loc_": {
                            "unit_": "no_component",
                            "storage_": "lx",
                        },
                        "via_": [],
                    }
                ],
                "dstLdsAndLoopOffsets_": [{"myLdsIdx_": 0}],
            }
        )
    if core_slices is None:
        core_slices = {
            str(c): {
                dim: c if factor == num_cores else 0
                for dim, factor in shard.items()
            }
            for c in range(num_cores)
        }
    else:
        core_slices = {str(c): dict(slices) for c, slices in core_slices.items()}
    return {
        f"{idx}_{op}": {
            "sdscFoldProps_": [{"factor_": 1, "label_": "time"}],
            "sdscFolds_": {
                "dim_prop_func": [{"Affine": {"alpha_": 1, "beta_": 0}}],
                "dim_prop_attr": [{"factor_": 1, "label_": "time"}],
                "data_": {"[0]": "0"},
            },
            "coreFoldProp_": {"factor_": num_cores, "label_": "core"},
            "coreletFoldProp_": {"factor_": 1, "label_": "corelet"},
            "numCoresUsed_": num_cores,
            "coreIdToDsc_": {str(c): 0 for c in range(num_cores)},
            "numWkSlicesPerDim_": shard,
            "coreIdToWkSlice_": core_slices,
            "coreIdToDscSchedule": {},
            "dscs_": [{op: dl}],
        }
    }


def _fake_attention_sdscs(include_max=True):
    score_addr = 4096
    score_pdi = {
        "OUTPUT": {
            "layoutDimOrder_": ["out", "x", "mb"],
            "stickDimOrder_": ["out"],
            "stickSize_": [64],
        },
        "KERNEL": {
            "layoutDimOrder_": ["x", "mb", "out"],
            "stickDimOrder_": ["out"],
            "stickSize_": [64],
        },
    }
    producer_pdi = {
        "OUTPUT": {
            "layoutDimOrder_": ["out", "mb", "x"],
            "stickDimOrder_": ["out"],
            "stickSize_": [64],
        }
    }
    bmm = _fake_sdsc(
        0,
        "batchmatmul",
        {"x": 1, "mb": 32, "out": 1, "in": 1},
        {"x_": 32, "mb_": 64, "out_": 64, "in_": 128},
        [],
        [("Tensor2-idx2", "OUTPUT", score_addr)],
        producer_pdi,
    )
    sdscs = [bmm]
    if include_max:
        sdscs.append(
            _fake_sdsc(
                1,
                "max",
                {"mb": 1, "x": 32, "out": 1},
                {"x_": 64, "mb_": 32, "out_": 64},
                [("Tensor0-idx0", "OUTPUT", score_addr)],
                [("Tensor1-idx1", "KERNEL", 8192)],
                score_pdi,
            )
        )
    sdscs.append(
        _fake_sdsc(
            2,
            "sub",
            {"mb": 1, "x": 32, "out": 1},
            {"x_": 64, "mb_": 32, "out_": 64},
            [("Tensor0-idx0", "OUTPUT", score_addr), ("Tensor1-idx1", "KERNEL", 8192)],
            [("Tensor2-idx2", "OUTPUT", 12288)],
            score_pdi,
        )
    )
    return sdscs


def _fake_static_matmul_sdscs(stick_position="last", extra_consumer=False):
    shared_addr = 4096
    producer_pdi = {
        "OUTPUT": {
            "layoutDimOrder_": ["mb", "out"],
            "stickDimOrder_": ["out"],
            "stickSize_": [64],
        }
    }
    consumer_layout = ["mb", "in"] if stick_position == "last" else ["in", "mb"]
    consumer_pdi = {
        "INPUT": {
            "layoutDimOrder_": consumer_layout,
            "stickDimOrder_": ["in"],
            "stickSize_": [64],
        },
        "KERNEL": {
            "layoutDimOrder_": ["out", "in"],
            "stickDimOrder_": ["in"],
            "stickSize_": [64],
        },
        "OUTPUT": {
            "layoutDimOrder_": ["mb", "out"],
            "stickDimOrder_": ["out"],
            "stickSize_": [64],
        },
    }
    producer = _fake_sdsc(
        0,
        "batchmatmul",
        {"mb": 32, "out": 1},
        {"mb_": 512, "out_": 1024, "in_": 512},
        [],
        [("Tensor2-idx2", "OUTPUT", shared_addr)],
        producer_pdi,
    )
    consumer = _fake_sdsc(
        1,
        "batchmatmul",
        {"mb": 32, "out": 1, "in": 1},
        {"mb_": 512, "in_": 1024, "out_": 256},
        [("Tensor0-idx0", "INPUT", shared_addr), ("Tensor1-idx1", "KERNEL", 8192)],
        [("Tensor2-idx2", "OUTPUT", 12288)],
        consumer_pdi,
    )
    sdscs = [producer, consumer]
    if extra_consumer:
        sdscs.append(
            _fake_sdsc(
                2,
                "identity",
                {"mb": 32, "in": 1},
                {"mb_": 512, "in_": 1024},
                [("Tensor0-idx0", "INPUT", shared_addr)],
                [("Tensor1-idx1", "OUTPUT", 16384)],
                consumer_pdi,
            )
        )
    return sdscs


def _fake_flash_pipeline_sdscs(
    num_tiles=3,
    *,
    lx_pinned=False,
    input_neighbor_transfer=False,
    ij_input_layout=False,
    sdpa_layout_transform=False,
    size_overrides=None,
):
    if ij_input_layout:
        input_layout = ["i", "j", "in"]
        output_layout = ["i", "j", "out"]
    elif sdpa_layout_transform:
        input_layout = ["x", "mb", "in"]
        output_layout = ["mb", "x", "out"]
    else:
        input_layout = ["mb", "x", "in"]
        output_layout = ["mb", "x", "out"]
    if ij_input_layout:
        n_sizes = {"i_": 64, "j_": 2, "x_": 2, "out_": 192, "in_": 64}
    elif sdpa_layout_transform:
        n_sizes = {"x_": 2, "mb_": 96, "out_": 64, "in_": 64}
    else:
        n_sizes = {"x_": 2, "mb_": 96, "out_": 192, "in_": 64}
    if size_overrides:
        n_sizes.update(size_overrides)
    shard = (
        {"i": 32, "j": 1, "out": 1, "in": 1}
        if ij_input_layout
        else {"x": 1, "mb": 32, "out": 1, "in": 1}
    )
    pdi = {
        "INPUT": {
            "layoutDimOrder_": input_layout,
            "stickDimOrder_": ["in"],
            "stickSize_": [64],
        },
        "KERNEL": {
            "layoutDimOrder_": ["in", "x", "out"],
            "stickDimOrder_": ["out"],
            "stickSize_": [64],
        },
        "OUTPUT": {
            "layoutDimOrder_": output_layout,
            "stickDimOrder_": ["out"],
            "stickSize_": [64],
        },
    }
    sdscs = []
    for idx in range(num_tiles):
        sdscs.append(
            _fake_sdsc(
                idx,
                "batchmatmul",
                shard,
                n_sizes,
                [("Tensor0-idx0", "INPUT", 4096 + idx * 4096)],
                [("Tensor2-idx2", "OUTPUT", 8192 + idx * 4096)],
                pdi,
                lx_pinned=lx_pinned,
                input_neighbor_transfer=input_neighbor_transfer,
            )
        )
    return sdscs


def _fake_flash_layout_xform_relation_sdscs():
    shared_addr = 4096
    producer_pdi = {
        "OUTPUT": {
            "layoutDimOrder_": ["mb", "x", "out"],
            "stickDimOrder_": ["out"],
            "stickSize_": [64],
        }
    }
    consumer_pdi = {
        "INPUT": {
            "layoutDimOrder_": ["x", "mb", "in"],
            "stickDimOrder_": ["in"],
            "stickSize_": [64],
        },
        "KERNEL": {
            "layoutDimOrder_": ["in", "x", "out"],
            "stickDimOrder_": ["out"],
            "stickSize_": [64],
        },
        "OUTPUT": {
            "layoutDimOrder_": ["mb", "x", "out"],
            "stickDimOrder_": ["out"],
            "stickSize_": [64],
        },
    }
    producer = _fake_sdsc(
        0,
        "ReStickifyOpHBM",
        {"mb": 2, "x": 2, "out": 1},
        {"mb_": 2, "x_": 128, "out_": 64},
        [],
        [("Tensor1-idx1", "OUTPUT", shared_addr)],
        producer_pdi,
        num_cores=4,
        core_slices={
            0: {"mb": 0, "x": 0, "out": 0},
            1: {"mb": 1, "x": 0, "out": 0},
            2: {"mb": 0, "x": 1, "out": 0},
            3: {"mb": 1, "x": 1, "out": 0},
        },
    )
    consumer = _fake_sdsc(
        1,
        "batchmatmul",
        {"x": 1, "mb": 32, "out": 1, "in": 1},
        {"x_": 2, "mb_": 128, "out_": 64, "in_": 64},
        [("Tensor0-idx0", "INPUT", shared_addr)],
        [("Tensor2-idx2", "OUTPUT", 8192)],
        consumer_pdi,
    )
    return [producer, consumer]


def _fake_flash_layout_xform_lookahead_sdscs():
    current_addr = 4096
    future_addr = 12288
    producer_pdi = {
        "OUTPUT": {
            "layoutDimOrder_": ["mb", "x", "out"],
            "stickDimOrder_": ["out"],
            "stickSize_": [64],
        }
    }
    consumer_pdi = {
        "INPUT": {
            "layoutDimOrder_": ["x", "mb", "in"],
            "stickDimOrder_": ["in"],
            "stickSize_": [64],
        },
        "KERNEL": {
            "layoutDimOrder_": ["in", "x", "out"],
            "stickDimOrder_": ["out"],
            "stickSize_": [64],
        },
        "OUTPUT": {
            "layoutDimOrder_": ["mb", "x", "out"],
            "stickDimOrder_": ["out"],
            "stickSize_": [64],
        },
    }
    shard = {"x": 1, "mb": 32, "out": 1, "in": 1}
    sizes = {"x_": 2, "mb_": 96, "out_": 64, "in_": 64}
    return [
        _fake_sdsc(
            0,
            "ReStickifyOpHBM",
            shard,
            sizes,
            [],
            [("Tensor1-idx1", "OUTPUT", current_addr)],
            producer_pdi,
        ),
        _fake_sdsc(
            1,
            "ReStickifyOpHBM",
            shard,
            sizes,
            [],
            [("Tensor1-idx1", "OUTPUT", future_addr)],
            producer_pdi,
        ),
        _fake_sdsc(
            2,
            "batchmatmul",
            shard,
            sizes,
            [("Tensor0-idx0", "INPUT", current_addr)],
            [("Tensor2-idx2", "OUTPUT", 8192)],
            consumer_pdi,
        ),
        _fake_sdsc(
            3,
            "batchmatmul",
            shard,
            sizes,
            [("Tensor0-idx0", "INPUT", future_addr)],
            [("Tensor2-idx2", "OUTPUT", 16384)],
            consumer_pdi,
        ),
    ]


def _fake_flash_layout_xform_hoist_sdscs(
    dependent_future=False,
    future_producer_op="ReStickifyOpHBM",
):
    future_addr = 12288
    future_input_addr = 24576
    if dependent_future:
        future_input_addr = 8192
    producer_pdi = {
        "INPUT": {
            "layoutDimOrder_": ["mb", "x", "in"],
            "stickDimOrder_": ["in"],
            "stickSize_": [64],
        },
        "KERNEL": {
            "layoutDimOrder_": ["mb", "x", "out"],
            "stickDimOrder_": ["out"],
            "stickSize_": [64],
        },
    }
    consumer_pdi = {
        "INPUT": {
            "layoutDimOrder_": ["x", "mb", "in"],
            "stickDimOrder_": ["in"],
            "stickSize_": [64],
        },
        "KERNEL": {
            "layoutDimOrder_": ["in", "x", "out"],
            "stickDimOrder_": ["out"],
            "stickSize_": [64],
        },
        "OUTPUT": {
            "layoutDimOrder_": ["mb", "x", "out"],
            "stickDimOrder_": ["out"],
            "stickSize_": [64],
        },
    }
    current_pdi = {
        "INPUT": {
            "layoutDimOrder_": ["mb", "x", "in"],
            "stickDimOrder_": ["in"],
            "stickSize_": [64],
        },
        "KERNEL": {
            "layoutDimOrder_": ["in", "x", "out"],
            "stickDimOrder_": ["out"],
            "stickSize_": [64],
        },
        "OUTPUT": {
            "layoutDimOrder_": ["mb", "x", "out"],
            "stickDimOrder_": ["out"],
            "stickSize_": [64],
        },
    }
    shard = {"x": 1, "mb": 32, "out": 1, "in": 1}
    sizes = {"x_": 2, "mb_": 64, "out_": 64, "in_": 64}
    return [
        _fake_sdsc(
            0,
            "batchmatmul",
            shard,
            sizes,
            [("Tensor0-idx0", "INPUT", 4096), ("Tensor1-idx1", "KERNEL", 5120)],
            [("Tensor2-idx2", "OUTPUT", 8192)],
            current_pdi,
        ),
        _fake_sdsc(
            1,
            future_producer_op,
            shard,
            sizes,
            [("Tensor0-idx0", "INPUT", future_input_addr)],
            [("Tensor1-idx1", "KERNEL", future_addr)],
            producer_pdi,
        ),
        _fake_sdsc(
            2,
            "batchmatmul",
            shard,
            sizes,
            [
                ("Tensor0-idx0", "INPUT", 32768),
                ("Tensor1-idx1", "KERNEL", future_addr),
            ],
            [("Tensor2-idx2", "OUTPUT", 16384)],
            consumer_pdi,
        ),
    ]


def _fake_flash_layout_xform_kv_repack_sdscs():
    future_addr = 12288
    producer_pdi = {
        "INPUT": {
            "layoutDimOrder_": ["out", "mb", "x"],
            "stickDimOrder_": ["x"],
            "stickSize_": [64],
        },
        "KERNEL": {
            "layoutDimOrder_": ["x", "mb", "out"],
            "stickDimOrder_": ["out"],
            "stickSize_": [64],
        },
    }
    current_pdi = {
        "INPUT": {
            "layoutDimOrder_": ["x", "mb", "in"],
            "stickDimOrder_": ["in"],
            "stickSize_": [64],
        },
        "KERNEL": {
            "layoutDimOrder_": ["in", "x", "out"],
            "stickDimOrder_": ["out"],
            "stickSize_": [64],
        },
        "OUTPUT": {
            "layoutDimOrder_": ["mb", "x", "out"],
            "stickDimOrder_": ["out"],
            "stickSize_": [64],
        },
    }
    future_pdi = {
        "INPUT": {
            "layoutDimOrder_": ["mb", "x", "in"],
            "stickDimOrder_": ["in"],
            "stickSize_": [64],
        },
        "KERNEL": {
            "layoutDimOrder_": ["in", "x", "out"],
            "stickDimOrder_": ["out"],
            "stickSize_": [64],
        },
        "OUTPUT": {
            "layoutDimOrder_": ["mb", "x", "out"],
            "stickDimOrder_": ["out"],
            "stickSize_": [64],
        },
    }
    return [
        _fake_sdsc(
            0,
            "batchmatmul",
            {"x": 1, "mb": 32, "out": 1, "in": 1},
            {"x_": 2, "mb_": 128, "out_": 64, "in_": 64},
            [("Tensor0-idx0", "INPUT", 4096), ("Tensor1-idx1", "KERNEL", 5120)],
            [("Tensor2-idx2", "OUTPUT", 8192)],
            current_pdi,
        ),
        _fake_sdsc(
            1,
            "ReStickifyOpHBM",
            {"mb": 2, "x": 1, "out": 1},
            {"x_": 64, "mb_": 2, "out_": 64},
            [("Tensor0-idx0", "INPUT", 24576)],
            [("Tensor1-idx1", "KERNEL", future_addr)],
            producer_pdi,
            num_cores=2,
        ),
        _fake_sdsc(
            2,
            "batchmatmul",
            {"x": 1, "mb": 32, "out": 1, "in": 1},
            {"x_": 2, "mb_": 128, "out_": 64, "in_": 64},
            [
                ("Tensor0-idx0", "INPUT", 32768),
                ("Tensor1-idx1", "KERNEL", future_addr),
            ],
            [("Tensor2-idx2", "OUTPUT", 16384)],
            future_pdi,
        ),
    ]


def _fake_flash_layout_xform_kv_repack_multisplit_sdscs():
    sdscs = _fake_flash_layout_xform_kv_repack_sdscs()
    producer = rz._body(sdscs[1])
    producer["numCoresUsed_"] = 4
    producer["coreFoldProp_"] = {"factor_": 4, "label_": "core"}
    producer["coreIdToDsc_"] = {str(c): 0 for c in range(4)}
    producer["numWkSlicesPerDim_"] = {"mb": 2, "x": 2, "out": 1}
    producer["coreIdToWkSlice_"] = {
        "0": {"mb": 0, "x": 0, "out": 0},
        "1": {"mb": 1, "x": 0, "out": 0},
        "2": {"mb": 0, "x": 1, "out": 0},
        "3": {"mb": 1, "x": 1, "out": 0},
    }
    for dsc in producer["dscs_"]:
        dl = next(iter(dsc.values()))
        for node in dl.get("scheduleTree_", []):
            data = node.get("startAddressCoreCorelet_", {}).get("data_")
            if isinstance(data, dict):
                first = next(iter(data.values()))
                node["startAddressCoreCorelet_"]["data_"] = {
                    f"[{c}, 0, 0]": first for c in range(4)
                }
    return sdscs


def _fake_flash_layout_xform_kv_repack_extra_consumer_sdscs():
    sdscs = _fake_flash_layout_xform_kv_repack_sdscs()
    future_addr = 12288
    extra_pdi = {
        "INPUT": {
            "layoutDimOrder_": ["mb", "x", "out"],
            "stickDimOrder_": ["out"],
            "stickSize_": [64],
        },
        "KERNEL": {
            "layoutDimOrder_": ["x", "mb", "out"],
            "stickDimOrder_": ["out"],
            "stickSize_": [64],
        },
        "OUTPUT": {
            "layoutDimOrder_": ["mb", "x", "out"],
            "stickDimOrder_": ["out"],
            "stickSize_": [64],
        },
    }
    sdscs.append(
        _fake_sdsc(
            3,
            "maxnonstick",
            {"mb": 2, "x": 1, "out": 1},
            {"mb_": 2, "x_": 128, "out_": 64},
            [
                ("Tensor0-idx0", "INPUT", 49152),
                ("Tensor1-idx1", "KERNEL", future_addr),
            ],
            [("Tensor2-idx2", "OUTPUT", 57344)],
            extra_pdi,
            num_cores=2,
        )
    )
    return sdscs


def _fake_flash_pointwise_sdscs(multisplit=False, chain=False):
    shared_addr = 4096
    second_addr = 12288
    shard = {"mb": 1, "x": 1, "out": 32}
    if multisplit:
        shard = {"mb": 2, "x": 1, "out": 16}
    pdi = {
        "INPUT": {
            "layoutDimOrder_": ["mb", "x", "out"],
            "stickDimOrder_": ["x"],
            "stickSize_": [64],
        },
        "KERNEL": {
            "layoutDimOrder_": ["mb", "x", "out"],
            "stickDimOrder_": ["x"],
            "stickSize_": [64],
        },
        "OUTPUT": {
            "layoutDimOrder_": ["mb", "x", "out"],
            "stickDimOrder_": ["x"],
            "stickSize_": [64],
        },
    }
    producer = _fake_sdsc(
        0,
        "add",
        shard,
        {"mb_": 2, "x_": 128, "out_": 64},
        [("Tensor0-idx0", "INPUT", 1024)],
        [("Tensor2-idx2", "OUTPUT", shared_addr)],
        pdi,
    )
    consumer = _fake_sdsc(
        1,
        "mul",
        shard,
        {"mb_": 2, "x_": 128, "out_": 64},
        [
            ("Tensor0-idx0", "INPUT", shared_addr),
            ("Tensor1-idx1", "KERNEL", 8192),
        ],
        [("Tensor2-idx2", "OUTPUT", second_addr)],
        pdi,
    )
    if chain:
        downstream = _fake_sdsc(
            2,
            "add",
            shard,
            {"mb_": 2, "x_": 128, "out_": 64},
            [
                ("Tensor0-idx0", "INPUT", second_addr),
                ("Tensor1-idx1", "KERNEL", 16384),
            ],
            [("Tensor2-idx2", "OUTPUT", 20480)],
            pdi,
        )
        return [producer, consumer, downstream]
    return [producer, consumer]


def _fake_flash_score_scale_sdscs(score_block=64):
    shared_addr = 4096
    producer_pdi = {
        "OUTPUT": {
            "layoutDimOrder_": ["mb", "x", "out"],
            "stickDimOrder_": ["out"],
            "stickSize_": [64],
        }
    }
    consumer_pdi = {
        "OUTPUT": {
            "layoutDimOrder_": ["mb", "x", "out"],
            "stickDimOrder_": ["out"],
            "stickSize_": [64],
        }
    }
    producer = _fake_sdsc(
        0,
        "batchmatmul",
        {"x": 1, "mb": 32, "out": 1, "in": 1},
        {"x_": 2, "mb_": 128, "out_": score_block, "in_": 64},
        [],
        [("Tensor2-idx2", "OUTPUT", shared_addr)],
        producer_pdi,
    )
    consumer = _fake_sdsc(
        1,
        "mul",
        {"x": 1, "out": 1, "mb": 32},
        {"x_": 2, "mb_": 128, "out_": score_block},
        [("Tensor0-idx0", "OUTPUT", shared_addr), ("Tensor1-idx1", "OUTPUT", 8192)],
        [("Tensor2-idx2", "OUTPUT", 12288)],
        consumer_pdi,
    )
    return [producer, consumer]


def test_attention_score_handoff_bridges_full_score_fanout():
    sdscs = _fake_attention_sdscs()
    assert rz.realize_onchip_handoff(
        sdscs, attention_score_handoff=True, min_handoff_bytes=0
    )
    bmm, max_sdsc, sub_sdsc = sdscs
    bmm_out = rz._dl_op(bmm)["labeledDs_"][0]
    assert bmm_out["hbmSize_"] == 0
    for sdsc in (max_sdsc, sub_sdsc):
        body = sdsc[next(iter(sdsc))]
        assert body["opFuncsUsed_"] == ["STCDPOpLx", "STCDPOpLx"]
        assert len(body["datadscs_"]) == 2
        for dataop in body["datadscs_"]:
            op_body = dataop[next(iter(dataop))]
            assert op_body["labeledDs_"][0]["hbmSize_"] == 0
            assert op_body["labeledDs_"][1]["hbmSize_"] == 0
        assert rz._dl_op(sdsc)["labeledDs_"][0]["hbmSize_"] == 0


def test_attention_score_handoff_respects_min_size_gate():
    sdscs = _fake_attention_sdscs()
    assert not rz.realize_onchip_handoff(
        sdscs, attention_score_handoff=True, min_handoff_bytes=1 << 40
    )
    assert "datadscs_" not in sdscs[1][next(iter(sdscs[1]))]


def test_attention_score_handoff_requires_full_score_fanout():
    sdscs = _fake_attention_sdscs(include_max=False)
    assert not rz.realize_onchip_handoff(
        sdscs, attention_score_handoff=True, min_handoff_bytes=0
    )


def test_static_matmul_handoff_detects_same_stick_layout():
    sdscs = _fake_static_matmul_sdscs()
    edge = rz.detect_static_matmul_handoff(sdscs, min_handoff_bytes=0)
    assert edge is not None
    assert edge["layout"] == ["mb_", "in_"]
    assert edge["stick_dim"] == "in_"
    assert edge["split_dim"] == "mb_"
    assert edge["slice_bytes"] == 512 // 32 * 1024 * 2


def test_static_matmul_handoff_realizes_roundtrip_consumer():
    sdscs = _fake_static_matmul_sdscs()
    assert rz.realize_onchip_handoff(
        sdscs, static_matmul_handoff=True, min_handoff_bytes=0
    )
    prod, cons = sdscs[:2]
    assert rz._lds_by_idx(rz._dl_op(prod), 2)["hbmSize_"] == 0
    assert rz._lds_by_idx(rz._dl_op(cons), 0)["hbmSize_"] == 0
    body = cons[next(iter(cons))]
    assert body["opFuncsUsed_"] == ["STCDPOpLx", "STCDPOpLx"]
    assert len(body["datadscs_"]) == 2
    assert rz._dl_op(cons)["numCoreletsUsed_DSC2_"] == 1
    for dataop in body["datadscs_"]:
        op_body = dataop[next(iter(dataop))]
        assert op_body["labeledDs_"][0]["hbmSize_"] == 0
        assert op_body["labeledDs_"][1]["hbmSize_"] == 0


def test_static_matmul_handoff_respects_min_size_gate():
    sdscs = _fake_static_matmul_sdscs()
    assert not rz.realize_onchip_handoff(
        sdscs, static_matmul_handoff=True, min_handoff_bytes=1 << 40
    )
    assert "datadscs_" not in sdscs[1][next(iter(sdscs[1]))]


def test_static_matmul_handoff_rejects_layout_change_and_fanout():
    assert (
        rz.detect_static_matmul_handoff(
            _fake_static_matmul_sdscs(stick_position="first"), min_handoff_bytes=0
        )
        is None
    )
    assert (
        rz.detect_static_matmul_handoff(
            _fake_static_matmul_sdscs(extra_consumer=True), min_handoff_bytes=0
        )
        is None
    )


def test_pointwise_handoff_uses_actual_stick_when_split_differs():
    sdscs = _fake_flash_pointwise_sdscs()
    assert rz.realize_onchip_handoff(sdscs, min_handoff_bytes=0)
    root = sdscs[1]["1_mul"]
    dataop = root["datadscs_"][0]["0_STCDPOpLx_dataop"]
    in_ld = dataop["labeledDs_"][0]
    assert root["opFuncsUsed_"] == ["STCDPOpLx"]
    assert in_ld["layoutDimOrder_"] == ["mb_", "x_", "out_"]
    assert in_ld["stickDimOrder_"] == ["x_"]
    assert in_ld["dimToLayoutSize_"] == {"mb_": 2, "x_": 128, "out_": 64}


def test_pointwise_handoff_rejects_multisplit_flash_edge():
    sdscs = _fake_flash_pointwise_sdscs(multisplit=True)
    assert not rz.realize_onchip_handoff(sdscs, min_handoff_bytes=0)


def test_flash_pointwise_handoffs_realize_eligible_chain():
    sdscs = _fake_flash_pointwise_sdscs(chain=True)
    assert rz.realize_flash_attention_pointwise_handoffs(sdscs) == 2
    assert rz._lds_by_idx(rz._dl_op(sdscs[0]), 2)["hbmSize_"] == 0
    assert rz._lds_by_idx(rz._dl_op(sdscs[1]), 0)["hbmSize_"] == 0
    assert rz._lds_by_idx(rz._dl_op(sdscs[1]), 2)["hbmSize_"] == 0
    assert rz._lds_by_idx(rz._dl_op(sdscs[2]), 0)["hbmSize_"] == 0
    assert sdscs[1]["1_mul"]["opFuncsUsed_"] == ["STCDPOpLx"]
    assert sdscs[2]["2_add"]["opFuncsUsed_"] == ["STCDPOpLx"]


def test_flash_pointwise_handoffs_accept_disjoint_region():
    def alloc_base(sdsc, lds_idx):
        for node in rz._dl_op(sdsc).get("scheduleTree_", []):
            if node.get("nodeType_") == "allocate" and node.get("ldsIdx_") == lds_idx:
                data = node["startAddressCoreCorelet_"]["data_"]
                return int(next(iter(data.values())))
        raise AssertionError(f"missing allocate node for lds{lds_idx}")

    sdscs = _fake_flash_pointwise_sdscs(chain=True)
    region0 = rz.LAYOUT_XFORM_COMPOSE_POINTWISE_LX_BASE

    assert (
        rz.realize_flash_attention_pointwise_handoffs(
            sdscs,
            pointwise_region0=region0,
        )
        == 2
    )
    assert alloc_base(sdscs[0], 2) == region0
    assert alloc_base(sdscs[1], 0) == region0 + rz.MIN_BRIDGE_REGION_BYTES
    assert alloc_base(sdscs[1], 2) == region0
    assert alloc_base(sdscs[2], 0) == region0 + rz.MIN_BRIDGE_REGION_BYTES


def test_layout_xform_compose_pointwise_lx_base_tracks_layout_footprint():
    assert (
        rz.layout_xform_compose_pointwise_lx_base(rz.MIN_BRIDGE_REGION_BYTES)
        == rz.LAYOUT_XFORM_COMPOSE_POINTWISE_LX_BASE
    )
    larger_slice = rz.MIN_BRIDGE_REGION_BYTES * 3
    assert rz.layout_xform_compose_pointwise_lx_base(larger_slice) == (
        rz.PRODUCER_LX_BASE + 2 * larger_slice
    )


def test_flash_score_scale_handoff_realizes_batchmatmul_to_mul():
    sdscs = _fake_flash_score_scale_sdscs()
    edge = rz.detect_flash_score_scale_handoff(sdscs)
    assert edge is not None
    assert edge["layout"] == ["mb_", "x_", "out_"]
    assert edge["stick_dim"] == "out_"
    assert edge["split_dim"] == "mb_"
    assert (
        rz.realize_flash_attention_pointwise_handoffs(
            sdscs,
            score_scale_handoff=True,
        )
        == 1
    )
    assert rz._lds_by_idx(rz._dl_op(sdscs[0]), 2)["hbmSize_"] == 0
    assert rz._lds_by_idx(rz._dl_op(sdscs[1]), 0)["hbmSize_"] == 0
    assert "coreStateInit_" not in rz._lds_by_idx(rz._dl_op(sdscs[0]), 2)
    assert rz._dl_op(sdscs[0])["numCoreletsUsed_DSC2_"] == 1
    body = sdscs[1]["1_mul"]
    assert body["opFuncsUsed_"] == ["STCDPOpLx"]
    dataop = body["datadscs_"][0]["0_STCDPOpLx_dataop"]
    assert dataop["labeledDs_"][0]["PieceInfo"][0]["PlacementInfo"][0][
        "startAddr"
    ] == [0]
    assert dataop["labeledDs_"][1]["PieceInfo"][0]["PlacementInfo"][0][
        "startAddr"
    ] == [rz.MIN_BRIDGE_REGION_BYTES]
    assert dataop["labeledDs_"][0]["layoutDimOrder_"] == ["mb_", "x_", "out_"]
    assert dataop["labeledDs_"][0]["stickDimOrder_"] == ["out_"]


def test_flash_score_scale_handoff_is_default_disabled():
    sdscs = _fake_flash_score_scale_sdscs()
    assert rz.realize_flash_attention_pointwise_handoffs(sdscs) == 0
    assert rz._hbm_base(rz._dl_op(sdscs[0]), 2) == "4096"
    assert rz._hbm_base(rz._dl_op(sdscs[1]), 0) == "4096"
    assert "coreStateInit_" not in rz._lds_by_idx(rz._dl_op(sdscs[0]), 2)
    assert "coreStateInit_" not in rz._lds_by_idx(rz._dl_op(sdscs[1]), 0)
    assert "datadscs_" not in sdscs[1]["1_mul"]


def test_flash_score_scale_handoff_rejects_wide_score_block():
    sdscs = _fake_flash_score_scale_sdscs(score_block=256)
    assert rz.detect_flash_score_scale_handoff(sdscs) is None
    assert (
        rz.realize_flash_attention_pointwise_handoffs(
            sdscs,
            score_scale_handoff=True,
        )
        == 0
    )
    assert rz._hbm_base(rz._dl_op(sdscs[0]), 2) == "4096"
    assert rz._hbm_base(rz._dl_op(sdscs[1]), 0) == "4096"


def test_flash_value_flow_tile_flips_real_single_consumer_edge():
    sdscs = _fake_static_matmul_sdscs()
    artifact, replaced = rz.build_flash_attention_value_flow_tile_artifact(
        sdscs,
        tile_index=1,
    )

    assert replaced == "1_batchmatmul"
    assert rz._lds_by_idx(rz._dl_op(sdscs[0]), 2)["hbmSize_"] == 0
    assert rz._lds_by_idx(rz._dl_op(sdscs[1]), 0)["hbmSize_"] == 0

    root = artifact["mixed_flash_value_flow_tile_1"]
    assert len(root["dscs_"]) == 1
    assert len(root["datadscs_"]) == 2
    assert root["opFuncsUsed_"] == ["STCDPOpLx", "STCDPOpLx"]
    assert root["coreIdToDscSchedule"]["0"] == [
        [0, -1, 0, 1],
        [1, -1, 1, 1],
        [-1, 0, 1, 0],
    ]
    meta = root["flashAttentionPipeline_"]
    assert meta["source"] == "generated-flash-prefill-real-value-flow"
    assert meta["tile_index"] == 1
    assert meta["replaces_sdsc"] == "1_batchmatmul"
    assert len(meta["edges"]) == 1


def test_flash_value_flow_tile_requires_latest_single_consumer_producer():
    assert rz.build_flash_attention_value_flow_tile_artifact(
        _fake_flash_pipeline_sdscs(num_tiles=1),
        tile_index=0,
    ) is None
    assert rz.build_flash_attention_value_flow_tile_artifact(
        _fake_static_matmul_sdscs(extra_consumer=True),
        tile_index=1,
    ) is None


def test_flash_value_flow_tile_reports_rejection_reasons():
    assert rz.flash_attention_value_flow_tile_rejection_reasons(
        _fake_static_matmul_sdscs(),
        tile_index=1,
    ) == []
    assert rz.flash_attention_value_flow_tile_rejection_reasons(
        _fake_static_matmul_sdscs(extra_consumer=True),
        tile_index=1,
    ) == [
        "input0:not_single_consumer:1_batchmatmul:input0,2_identity:input0",
        "input1:no_latest_producer",
    ]
    assert rz.flash_attention_value_flow_tile_rejection_reasons(
        _fake_flash_pipeline_sdscs(num_tiles=1),
        tile_index=0,
    ) == ["input0:no_latest_producer"]
    assert rz.flash_attention_value_flow_tile_rejection_reasons(
        _fake_flash_pipeline_sdscs(num_tiles=1),
        tile_index=3,
    ) == ["tile_not_found"]


def test_flash_ifn_pair_tile_builds_predecessor_backed_sidecars():
    result = rz.build_flash_attention_ifn_pair_tile_artifacts(
        _fake_static_matmul_sdscs(),
        tile_index=1,
    )

    assert result is not None
    pred = result["artifacts"][0]["mixed_flash_ifn_pair_tile_1_predecessor"]
    cons = result["artifacts"][1]["mixed_flash_ifn_pair_tile_1_consumer"]
    assert result["replacements"] == {
        "0_batchmatmul": "mixed_flash_ifn_pair_tile_1_predecessor",
        "1_batchmatmul": "mixed_flash_ifn_pair_tile_1_consumer",
    }
    assert result["bundle_attrs"] == {}

    pred_dl = rz._dl_op({"p": pred})
    cons_dl = rz._dl_op({"c": cons})
    assert rz._lds_by_idx(pred_dl, 2)["hbmSize_"] == 0
    assert rz._lds_by_idx(cons_dl, 0)["hbmSize_"] == 0
    assert rz._has_input_fetch_neighbor_transfer(cons_dl, 0)
    assert cons["coreIdToDscSchedule"]["0"] == [[0, -1, 0, 1], [-1, 0, 1, 0]]
    dataop_name = next(iter(cons["datadscs_"][0]))
    assert dataop_name == "0_STCDPOpLx_predecessor_fetch_Tensor0_idx0_tile1"
    assert "STCDPOpLx_ifn_Tensor" not in dataop_name
    dataop = next(iter(cons["datadscs_"][0].values()))
    src_piece = dataop["labeledDs_"][0]["PieceInfo"][0]
    dst_piece = dataop["labeledDs_"][1]["PieceInfo"][0]
    assert src_piece["PlacementInfo"][0]["startAddr"] == [rz.PRODUCER_LX_BASE]
    assert dst_piece["PlacementInfo"][0]["startAddr"] == [rz.CONSUMER_LX_BASE]

    pred_meta = pred["flashAttentionPipeline_"]
    assert pred_meta["ifn_pair_role"] == "predecessor"
    assert pred_meta["ifn_runtime_safe"] is True
    cons_meta = cons["flashAttentionPipeline_"]
    assert cons_meta["source"] == (
        "generated-flash-prefill-predecessor-ifn-pair-consumer"
    )
    assert cons_meta["ifn_mode"] == "predecessor_backed_lx_copy_pair"
    assert cons_meta["ifn_runtime_safe"] is True
    assert cons_meta["ifn_predecessor_sdsc"] == "0_batchmatmul"
    assert cons_meta["ifn_consumer_sdsc"] == "1_batchmatmul"
    assert cons_meta["ifn_predecessor_output_idx"] == 2
    assert cons_meta["ifn_attached_input_idx"] == 0
    assert cons_meta["ifn_shared_hbm_addr"] == "4096"
    assert cons_meta["ifn_predecessor_lx_base"] == rz.PRODUCER_LX_BASE
    assert cons_meta["ifn_input_lx_base"] == rz.CONSUMER_LX_BASE


def test_flash_ifn_pair_tile_rejects_not_physically_equivalent_edge():
    sdscs = _fake_flash_pipeline_sdscs(num_tiles=3)

    assert rz.build_flash_attention_ifn_pair_tile_artifacts(
        sdscs,
        tile_index=1,
    ) is None
    assert rz.flash_attention_ifn_pair_tile_rejection_reasons(
        sdscs,
        tile_index=1,
    ) == [
        "input0:physical_layout_mismatch:"
        "producer=['mb_', 'x_', 'out_']/out_:"
        "consumer=['mb_', 'x_', 'in_']/in_"
    ]


def test_flash_ifn_pair_tile_reports_layout_transform_required_edge():
    sdscs = _fake_flash_pipeline_sdscs(
        num_tiles=3,
        sdpa_layout_transform=True,
    )

    assert rz.build_flash_attention_ifn_pair_tile_artifacts(
        sdscs,
        tile_index=1,
    ) is None
    assert rz.flash_attention_ifn_pair_tile_rejection_reasons(
        sdscs,
        tile_index=1,
    ) == [
        "input0:layout_transform_required:"
        "producer=['mb_', 'x_', 'out_']/out_:"
        "consumer=['x_', 'mb_', 'in_']/in_"
    ]


def test_flash_layout_xform_pair_tile_builds_experimental_sidecars():
    result = rz.build_flash_attention_layout_xform_pair_tile_artifacts(
        _fake_flash_pipeline_sdscs(num_tiles=3, sdpa_layout_transform=True),
        tile_index=1,
    )

    assert result is not None
    pred = result["artifacts"][0]["mixed_flash_layout_xform_pair_tile_1_predecessor"]
    cons = result["artifacts"][1]["mixed_flash_layout_xform_pair_tile_1_consumer"]
    assert result["replacements"] == {
        "0_batchmatmul": "mixed_flash_layout_xform_pair_tile_1_predecessor",
        "1_batchmatmul": "mixed_flash_layout_xform_pair_tile_1_consumer",
    }
    assert result["bundle_attrs"] == {}
    assert (
        rz.flash_attention_layout_xform_pair_tile_rejection_reasons(
            _fake_flash_pipeline_sdscs(num_tiles=3, sdpa_layout_transform=True),
            tile_index=1,
        )
        == []
    )

    pred_dl = rz._dl_op({"p": pred})
    cons_dl = rz._dl_op({"c": cons})
    assert rz._lds_by_idx(pred_dl, 2)["hbmSize_"] == 0
    assert rz._lds_by_idx(cons_dl, 0)["hbmSize_"] == 0
    assert rz._has_input_fetch_neighbor_transfer(cons_dl, 0)
    assert cons["coreIdToDscSchedule"]["0"] == [[0, -1, 0, 1], [-1, 0, 1, 0]]
    dataop_name = next(iter(cons["datadscs_"][0]))
    assert dataop_name == "0_STCDPOpLx_layout_xform_Tensor0_idx0_tile1"
    dataop = next(iter(cons["datadscs_"][0].values()))
    src_ld = dataop["labeledDs_"][0]
    dst_ld = dataop["labeledDs_"][1]
    assert dataop["dimPool_"] == ["mb_", "x_", "in_"]
    assert src_ld["layoutDimOrder_"] == ["mb_", "x_", "in_"]
    assert src_ld["stickDimOrder_"] == ["in_"]
    assert dst_ld["layoutDimOrder_"] == ["x_", "mb_", "in_"]
    assert dst_ld["stickDimOrder_"] == ["in_"]
    assert src_ld["PieceInfo"][0]["PlacementInfo"][0]["startAddr"] == [
        rz.PRODUCER_LX_BASE
    ]
    assert dst_ld["PieceInfo"][0]["PlacementInfo"][0]["startAddr"] == [
        rz.CONSUMER_LX_BASE
    ]

    pred_meta = pred["flashAttentionPipeline_"]
    assert pred_meta["layout_xform_pair_role"] == "predecessor"
    assert pred_meta["layout_xform_experimental"] is True
    cons_meta = cons["flashAttentionPipeline_"]
    assert cons_meta["layout_xform_mode"] == "same_dim_lx_copy_pair"
    assert cons_meta["layout_xform_pair_role"] == "consumer"
    assert cons_meta["layout_xform_source_layout"] == ["mb_", "x_", "in_"]
    assert cons_meta["layout_xform_consumer_layout"] == ["x_", "mb_", "in_"]
    assert cons_meta["layout_xform_original_predecessor_layout"] == [
        "mb_",
        "x_",
        "out_",
    ]


def test_flash_layout_xform_pair_overlap_schedules_copy_with_compute():
    result = rz.build_flash_attention_layout_xform_pair_tile_artifacts(
        _fake_flash_pipeline_sdscs(num_tiles=3, sdpa_layout_transform=True),
        tile_index=1,
        name_prefix="mixed_flash_pipeline_tile_layout_xform_pair",
        overlap_consumer=True,
    )

    assert result is not None
    pred = result["artifacts"][0][
        "mixed_flash_pipeline_tile_layout_xform_pair_1_predecessor"
    ]
    cons = result["artifacts"][1][
        "mixed_flash_pipeline_tile_layout_xform_pair_1_consumer"
    ]
    assert result["replacements"] == {
        "0_batchmatmul": "mixed_flash_pipeline_tile_layout_xform_pair_1_predecessor",
        "1_batchmatmul": "mixed_flash_pipeline_tile_layout_xform_pair_1_consumer",
    }
    assert cons["coreIdToDscSchedule"]["0"] == [[0, 0, 0, 0]]
    dataop_name = next(iter(cons["datadscs_"][0]))
    assert dataop_name == "0_STCDPOpLx_prefetch_layout_xform_Tensor0_idx0_tile1"

    pred_meta = pred["flashAttentionPipeline_"]
    assert pred_meta["source"] == (
        "generated-flash-prefill-layout-xform-overlap-pair-producer"
    )
    assert pred_meta["layout_xform_overlap_consumer"] is True
    cons_meta = cons["flashAttentionPipeline_"]
    assert cons_meta["source"] == (
        "generated-flash-prefill-layout-xform-overlap-pair-consumer"
    )
    assert cons_meta["layout_xform_overlap_consumer"] is True
    assert cons_meta["layout_xform_runtime_safe"] is False
    assert cons_meta["layout_xform_runtime_forced"] is True
    assert cons_meta["layout_xform_attached_input_idx"] == 0


def test_flash_layout_xform_lookahead_copies_current_then_prefetches_future():
    result = rz.build_flash_attention_layout_xform_lookahead_tile_artifacts(
        _fake_flash_layout_xform_lookahead_sdscs(),
        tile_index=0,
    )

    assert result is not None
    prefix = "mixed_flash_pipeline_tile_layout_xform_lookahead_0"
    current_cons = result["artifacts"][2][f"{prefix}_current_consumer"]
    future_cons = result["artifacts"][3][f"{prefix}_future_consumer"]
    assert result["replacements"] == {
        "0_ReStickifyOpHBM": f"{prefix}_current_predecessor",
        "1_ReStickifyOpHBM": f"{prefix}_future_predecessor",
        "2_batchmatmul": f"{prefix}_current_consumer",
        "3_batchmatmul": f"{prefix}_future_consumer",
    }
    assert current_cons["coreIdToDscSchedule"]["0"] == [
        [0, -1, 0, 1],
        [1, 0, 1, 0],
    ]
    assert [next(iter(item)) for item in current_cons["datadscs_"]] == [
        "0_STCDPOpLx_layout_xform_current_Tensor0_idx0_tile0",
        "1_STCDPOpLx_prefetch_layout_xform_future_Tensor0_idx0_tile1",
    ]
    future_cons_dl = rz._dl_op({"future": future_cons})
    assert rz._hbm_base(future_cons_dl, 0) is None
    assert "datadscs_" not in future_cons

    meta = current_cons["flashAttentionPipeline_"]
    assert meta["layout_xform_lookahead_role"] == "current_consumer"
    assert meta["layout_xform_current_tile"] == 0
    assert meta["layout_xform_future_tile"] == 1
    assert meta["layout_xform_runtime_safe"] is False
    assert meta["layout_xform_runtime_forced"] is True
    assert meta["layout_xform_attached_input_idx"] == 0
    assert meta["layout_xform_prefetch_input_idx"] == 0
    assert meta["layout_xform_future_predecessor_sdsc"] == "1_ReStickifyOpHBM"
    assert future_cons["flashAttentionPipeline_"][
        "layout_xform_future_input_lx_base"
    ] == meta["layout_xform_future_input_lx_base"]


def test_flash_layout_xform_lookahead_rejects_future_producer_not_ready():
    assert rz.build_flash_attention_layout_xform_lookahead_tile_artifacts(
        _fake_flash_pipeline_sdscs(num_tiles=3, sdpa_layout_transform=True),
        tile_index=1,
    ) is None
    reasons = rz.flash_attention_layout_xform_lookahead_rejection_reasons(
        _fake_flash_pipeline_sdscs(num_tiles=3, sdpa_layout_transform=True),
        tile_index=1,
    )
    assert reasons == [
        "future_tile2:producer_not_ready:producer=1:current=1",
    ]


def test_flash_layout_xform_hoist_runs_future_producer_then_prefetches_input1():
    result = rz.build_flash_attention_layout_xform_hoist_tile_artifacts(
        _fake_flash_layout_xform_hoist_sdscs(),
        tile_index=0,
    )

    assert result is not None
    prefix = "mixed_flash_pipeline_tile_layout_xform_hoist_0"
    current = result["artifacts"][0][f"{prefix}_current_consumer"]
    future = result["artifacts"][1][f"{prefix}_future_consumer"]
    assert result["replacements"] == {
        "0_batchmatmul": f"{prefix}_current_consumer",
        "2_batchmatmul": f"{prefix}_future_consumer",
    }
    assert result["omissions"] == {"1_ReStickifyOpHBM"}
    assert len(current["dscs_"]) == 2
    assert current["coreIdToDscSchedule"]["0"] == [
        [-1, 0, 0, 1],
        [0, 1, 1, 0],
    ]
    dataop_name = next(iter(current["datadscs_"][0]))
    assert dataop_name == (
        "0_STCDPOpLx_prefetch_layout_xform_hoisted_future_"
        "Tensor0_idx1_tile1"
    )
    future_dl = rz._dl_op({"future": future})
    assert rz._hbm_base(future_dl, 1) is None
    assert rz._lds_by_idx(future_dl, 1)["hbmSize_"] == 0

    meta = current["flashAttentionPipeline_"]
    assert meta["layout_xform_hoist_role"] == "current_consumer"
    assert meta["layout_xform_future_predecessor_sdsc"] == "1_ReStickifyOpHBM"
    assert (
        meta["layout_xform_omitted_future_predecessor_sdsc"]
        == "1_ReStickifyOpHBM"
    )
    assert meta["layout_xform_future_consumer_sdsc"] == "2_batchmatmul"
    assert meta["layout_xform_prefetch_input_idx"] == 1
    assert meta["layout_xform_runtime_safe"] is False
    assert meta["layout_xform_runtime_forced"] is True
    assert future["flashAttentionPipeline_"][
        "layout_xform_future_input_lx_base"
    ] == meta["layout_xform_future_input_lx_base"]


def test_flash_layout_xform_hoist_rejects_future_dependency_not_ready():
    assert rz.build_flash_attention_layout_xform_hoist_tile_artifacts(
        _fake_flash_layout_xform_hoist_sdscs(dependent_future=True),
        tile_index=0,
    ) is None
    reasons = rz.flash_attention_layout_xform_hoist_rejection_reasons(
        _fake_flash_layout_xform_hoist_sdscs(dependent_future=True),
        tile_index=0,
    )
    assert reasons == [
        "future_tile1:input0:no_latest_producer",
        (
            "future_tile1:input1:producer_input0:"
            "dependency_not_ready:producer=0:before=0"
        ),
    ]


def test_flash_layout_xform_hoist_rejects_non_restickify_producer():
    assert rz.build_flash_attention_layout_xform_hoist_tile_artifacts(
        _fake_flash_layout_xform_hoist_sdscs(future_producer_op="add"),
        tile_index=0,
    ) is None
    reasons = rz.flash_attention_layout_xform_hoist_rejection_reasons(
        _fake_flash_layout_xform_hoist_sdscs(future_producer_op="add"),
        tile_index=0,
    )
    assert reasons == [
        "future_tile1:input0:no_latest_producer",
        "future_tile1:input1:producer_not_restickify_hbm:add",
    ]


def test_flash_layout_xform_hoist_reports_kv_repack_boundary():
    sdscs = _fake_flash_layout_xform_kv_repack_sdscs()

    edge, reasons = rz._flash_attention_layout_xform_pair_edge(
        sdscs,
        tile_index=1,
        input_idx=1,
        allow_nonzero_input=True,
    )

    assert edge is None
    assert reasons == [
        (
            "input1:requires_kv_repack_broadcast:"
            "producer_split=mb_:mapped_split=x_:consumer_split=mb_:"
            "producer_cores=2:consumer_cores=32"
        ),
    ]
    assert rz.build_flash_attention_layout_xform_hoist_tile_artifacts(
        sdscs,
        tile_index=0,
    ) is None
    assert rz.flash_attention_layout_xform_hoist_rejection_reasons(
        sdscs,
        tile_index=0,
    ) == [
        "future_tile1:input0:no_latest_producer",
        (
            "future_tile1:input1:requires_kv_repack_broadcast:"
            "producer_split=mb_:mapped_split=x_:consumer_split=mb_:"
            "producer_cores=2:consumer_cores=32"
        ),
    ]


def test_flash_kv_repack_hbm_staged_hoist_runs_future_producer_before_current():
    result = rz.build_flash_attention_kv_repack_hbm_staged_hoist_tile_artifacts(
        _fake_flash_layout_xform_kv_repack_sdscs(),
        tile_index=0,
    )

    assert result is not None
    prefix = "mixed_flash_kv_repack_hbm_staged_hoist_0"
    producer_name = f"{prefix}_future_producer"
    future_name = f"{prefix}_future_kv_1_input1_consumer"
    producer = result["artifacts"][0][producer_name]
    future = result["artifacts"][1][future_name]
    assert result["replacements"] == {
        "2_batchmatmul": future_name,
    }
    assert result["insertions_before"] == {"0_batchmatmul": [producer_name]}
    assert result["omissions"] == {"1_ReStickifyOpHBM"}
    assert len(producer["dscs_"]) == 1
    assert producer["coreIdToDscSchedule"] == rz._body(
        _fake_flash_layout_xform_kv_repack_sdscs()[1]
    )["coreIdToDscSchedule"]
    meta = producer["flashAttentionPipeline_"]
    assert meta["kv_repack_hbm_staged_hoist_role"] == "future_producer"
    assert meta["kv_repack_hbm_staged_hoist_future_tile"] == 1
    assert meta["kv_repack_hbm_staged_hoist_future_input_idx"] == 1
    assert meta["kv_repack_hbm_staged_hoist_source_core_ids"] == [0, 1]

    future_meta = future["flashAttentionPipeline_"]
    assert future_meta["kv_repack_hbm_staged"] is True
    assert future_meta["kv_repack_hbm_source"] is True
    future_dl = rz._dl_op({"future": future})
    assert "hbm" in rz._lds_by_idx(future_dl, 1)["memOrg_"]


def test_flash_kv_repack_hbm_prefetch_hoist_prefetches_during_current_compute():
    sdscs = _fake_flash_layout_xform_kv_repack_sdscs()
    current_dl = rz._dl_op(sdscs[0])
    current_dl["dataStageParam_"] = {
        "0": {
            "ss_": {
                "name_": "core",
                "x_": 2,
                "mb_": 4,
                "in_": 64,
                "out_": 64,
            },
            "el_": {
                "name_": "core",
                "x_": 2,
                "mb_": 4,
                "in_": 64,
                "out_": 64,
            },
        }
    }
    current_k_alloc = next(
        node
        for node in current_dl["scheduleTree_"]
        if node.get("nodeType_") == "allocate" and node.get("ldsIdx_") == 1
    )
    current_k_alloc["backGapCore_"] = {"in": {"-1": "192"}}
    current_k_alloc["gapStickSpread_"] = {"in": 4}

    result = rz.build_flash_attention_kv_repack_hbm_prefetch_hoist_tile_artifacts(
        sdscs,
        tile_index=0,
        prefill_current_input=True,
    )

    assert result is not None
    prefix = "mixed_flash_kv_repack_hbm_prefetch_hoist_0"
    producer_name = f"{prefix}_future_producer"
    current_name = f"{prefix}_current_prefetch"
    future_name = f"{prefix}_future_consumer"
    producer = result["artifacts"][0][producer_name]
    current = result["artifacts"][1][current_name]
    future = result["artifacts"][2][future_name]
    assert result["replacements"] == {
        "0_batchmatmul": current_name,
        "2_batchmatmul": future_name,
    }
    assert result["insertions_before"] == {"0_batchmatmul": [producer_name]}
    assert result["omissions"] == {"1_ReStickifyOpHBM"}
    assert producer["flashAttentionPipeline_"][
        "kv_repack_hbm_prefetch_hoist_role"
    ] == "future_producer"

    current_meta = current["flashAttentionPipeline_"]
    assert current_meta["kv_repack_hbm_prefetch_hoist_role"] == "current_prefetch"
    assert current_meta["kv_repack_hbm_prefetch_hoist_future_tile"] == 1
    assert current_meta["kv_repack_hbm_prefetch_hoist_dataop_count"] == 1
    assert current_meta["kv_repack_hbm_prefetch_hoist_prefetch_corelet_id"] is None
    assert (
        current_meta["kv_repack_hbm_prefetch_hoist_prefetch_lx_roundtrip"]
        is False
    )
    assert (
        current_meta["kv_repack_hbm_prefetch_hoist_tail_current_prefetch"]
        is False
    )
    assert (
        current_meta["kv_repack_hbm_prefetch_hoist_prefetch_source_fanout"]
        is False
    )
    assert current_meta["kv_repack_hbm_prefetch_hoist_prefilled_current_inputs"] == [
        {"lds_idx": 0, "lx_base": 540672},
        {"lds_idx": 1, "lx_base": 802816},
    ]
    assert current["opFuncsUsed_"] == ["STCDPOpHBM", "STCDPOpHBM", "STCDPOpHBM"]
    assert current["coreIdToDscSchedule"]["0"] == [
        [0, -1, 0, 1],
        [1, -1, 1, 1],
        [2, 0, 1, 1],
    ]
    current_dl = rz._dl_op({"current": current})
    current_k_alloc = next(
        node
        for node in current_dl["scheduleTree_"]
        if node.get("nodeType_") == "allocate" and node.get("ldsIdx_") == 1
    )
    assert current_k_alloc["component_"] == "lx"
    assert "backGapCore_" not in current_k_alloc
    assert "gapStickSpread_" not in current_k_alloc
    current_q_prefill = next(iter(current["datadscs_"][0].values()))
    assert current_q_prefill["labeledDs_"][0]["PieceInfo"][0]["dimToSize_"] == {
        "x_": 2,
        "mb_": 4,
        "in_": 64,
    }
    first_load_name, first_load = next(iter(current["datadscs_"][-1].items()))
    assert first_load_name == (
        "0_STCDPOpHBM_kv_repack_hbm_prefetch_Tensor0_idx1_tile1_piece0_load"
    )
    first_hbm = next(
        placement
        for placement in first_load["labeledDs_"][0]["PieceInfo"][0][
            "PlacementInfo"
        ]
        if placement["type"] == "hbm"
    )
    assert first_hbm["startAddr"] == [
        _hbm_dataop_addr(
            int(current_meta["kv_repack_hbm_prefetch_hoist_source_hbm_addr"])
        )
    ]
    assert first_load["primaryDs_"] == [
        {"name_": "dataIN", "dimNames": ["in_", "x_", "out_"]}
    ]
    assert first_load["labeledDs_"][1]["pdsName_"] == "dataIN"
    assert first_load["labeledDs_"][1]["PieceInfo"] == []
    assert first_load["coreIdsUsed_"] == list(range(32))
    assert first_load["labeledDs_"][0]["PieceInfo"][0]["PlacementInfo"][0] == {
        "type": "lx",
        "memId": [0],
        "startAddr": [current_meta["kv_repack_hbm_prefetch_hoist_consumer_lx_base"]],
    }
    assert first_load["labeledDs_"][0]["PieceInfo"][0]["PlacementInfo"] == [
        {
            "type": "lx",
            "memId": [0],
            "startAddr": [current_meta["kv_repack_hbm_prefetch_hoist_consumer_lx_base"]],
        },
        first_hbm,
    ]

    future_meta = future["flashAttentionPipeline_"]
    assert future_meta["kv_repack_hbm_prefetch_hoist_role"] == "future_consumer"
    assert future_meta["kv_repack_hbm_prefetch_hoist_prefetch_corelet_id"] is None
    assert (
        future_meta["kv_repack_hbm_prefetch_hoist_prefetch_lx_roundtrip"]
        is False
    )
    assert (
        future_meta["kv_repack_hbm_prefetch_hoist_tail_current_prefetch"]
        is False
    )
    assert (
        future_meta["kv_repack_hbm_prefetch_hoist_prefetch_source_fanout"]
        is False
    )
    assert future["opFuncsUsed_"] == []
    assert future["datadscs_"] == []
    assert future["coreIdToDscSchedule"]["0"] == [[-1, 0, 0, 0]]
    future_dl = rz._dl_op({"future": future})
    future_lds = rz._lds_by_idx(future_dl, 1)
    assert "lx" in future_lds["memOrg_"]
    assert "hbm" not in future_lds["memOrg_"]
    assert "isExternal_" not in future_lds
    assert "isFirstUse_" not in future_lds
    assert not any(
        node.get("name_") == "input_fetch_neighbor_transfer_lds1"
        for node in future_dl["scheduleTree_"]
    )


def test_flash_kv_repack_hbm_prefetch_hoist_allows_preserved_extra_hbm_consumer():
    sdscs = _fake_flash_layout_xform_kv_repack_extra_consumer_sdscs()

    assert (
        rz.build_flash_attention_kv_repack_broadcast_plan_artifact(
            sdscs,
            tile_index=1,
            input_idx=1,
        )
        is None
    )
    assert rz.flash_attention_kv_repack_broadcast_rejection_reasons(
        sdscs,
        1,
        input_idx=1,
    ) == ["input1:not_single_consumer:2_batchmatmul:input1,3_maxnonstick:input1"]

    result = rz.build_flash_attention_kv_repack_hbm_prefetch_hoist_tile_artifacts(
        sdscs,
        tile_index=0,
        prefetch_loader_fanout=True,
        prefetch_loader_core_id=31,
        prefetch_loader_fanout_full_tile_pieces=True,
        serialize_loader_core_prefetch=True,
    )

    assert result is not None
    prefix = "mixed_flash_kv_repack_hbm_prefetch_hoist_0"
    assert result["insertions_before"] == {"0_batchmatmul": [f"{prefix}_future_producer"]}
    assert result["omissions"] == {"1_ReStickifyOpHBM"}
    current = result["artifacts"][1][f"{prefix}_current_prefetch"]
    future = result["artifacts"][2][f"{prefix}_future_consumer"]
    current_meta = current["flashAttentionPipeline_"]
    future_meta = future["flashAttentionPipeline_"]
    assert current_meta["kv_repack_hbm_prefetch_hoist_future_tile"] == 1
    assert current_meta["kv_repack_hbm_prefetch_hoist_future_consumer_sdsc"] == (
        "2_batchmatmul"
    )
    assert current_meta["kv_repack_additional_consumers"] == [
        "3_maxnonstick:input1"
    ]
    assert future_meta["kv_repack_additional_consumers"] == [
        "3_maxnonstick:input1"
    ]


def test_flash_kv_repack_hbm_prefetch_hoist_can_force_lx_base_probe():
    forced_lx_base = 1625344
    result = rz.build_flash_attention_kv_repack_hbm_prefetch_hoist_tile_artifacts(
        _fake_flash_layout_xform_kv_repack_sdscs(),
        tile_index=0,
        prefetch_lx_base=forced_lx_base,
    )

    assert result is not None
    prefix = "mixed_flash_kv_repack_hbm_prefetch_hoist_0"
    current = result["artifacts"][1][f"{prefix}_current_prefetch"]
    future = result["artifacts"][2][f"{prefix}_future_consumer"]
    current_meta = current["flashAttentionPipeline_"]
    future_meta = future["flashAttentionPipeline_"]
    assert current_meta["kv_repack_hbm_prefetch_hoist_native_load_prologue"] is True
    assert current["opFuncsUsed_"] == ["nop", "STCDPOpHBM"]
    assert current["coreIdToDscSchedule"]["0"] == [
        [0, -1, 0, 1],
        [1, 0, 1, 1],
    ]
    prologue = next(iter(current["datadscs_"][0].values()))
    assert prologue["op"]["name"] == "nop"
    assert "dldsc_native_load_prologue" in next(iter(current["datadscs_"][0]))
    assert (
        current_meta["kv_repack_hbm_prefetch_hoist_consumer_lx_base"]
        == forced_lx_base
    )
    assert (
        future_meta["kv_repack_hbm_prefetch_hoist_input_lx_base"]
        == forced_lx_base
    )
    assert current_meta[
        "kv_repack_hbm_prefetch_hoist_original_consumer_lx_base"
    ] != forced_lx_base
    first_load = next(iter(current["datadscs_"][-1].values()))
    assert first_load["labeledDs_"][1]["PieceInfo"] == []
    assert first_load["labeledDs_"][0]["PieceInfo"][0]["PlacementInfo"][0][
        "startAddr"
    ] == [forced_lx_base]
    future_dl = rz._dl_op({"future": future})
    future_lds = rz._lds_by_idx(future_dl, 1)
    assert future_lds["coreStateInit_"][0]["lbrInit_"] == [
        forced_lx_base
    ]
    assert "isExternal_" not in future_lds
    assert "isFirstUse_" not in future_lds


def test_flash_kv_repack_hbm_prefetch_hoist_can_disable_overlap_after_sync():
    result = rz.build_flash_attention_kv_repack_hbm_prefetch_hoist_tile_artifacts(
        _fake_flash_layout_xform_kv_repack_sdscs(),
        tile_index=0,
        overlap_after_sync=False,
    )

    assert result is not None
    prefix = "mixed_flash_kv_repack_hbm_prefetch_hoist_0"
    current = result["artifacts"][1][f"{prefix}_current_prefetch"]
    assert current["coreIdToDscSchedule"]["0"] == [
        [0, -1, 0, 1],
        [1, 0, 1, 0],
    ]
    current_meta = current["flashAttentionPipeline_"]
    assert (
        current_meta["kv_repack_hbm_prefetch_hoist_overlap_after_sync"]
        is False
    )


def test_flash_kv_repack_hbm_prefetch_hoist_can_tail_current_prefetch():
    result = rz.build_flash_attention_kv_repack_hbm_prefetch_hoist_tile_artifacts(
        _fake_flash_layout_xform_kv_repack_sdscs(),
        tile_index=0,
        tail_current_prefetch=True,
    )

    assert result is not None
    prefix = "mixed_flash_kv_repack_hbm_prefetch_hoist_0"
    current = result["artifacts"][1][f"{prefix}_current_prefetch"]
    future = result["artifacts"][2][f"{prefix}_future_consumer"]
    assert current["coreIdToDscSchedule"]["0"] == [
        [0, -1, 0, 1],
        [-1, 0, 1, 1],
        [1, -1, 1, 0],
    ]
    assert (
        current["flashAttentionPipeline_"][
            "kv_repack_hbm_prefetch_hoist_tail_current_prefetch"
        ]
        is True
    )
    assert (
        future["flashAttentionPipeline_"][
            "kv_repack_hbm_prefetch_hoist_tail_current_prefetch"
        ]
        is True
    )


def test_flash_kv_repack_hbm_prefetch_hoist_can_overlap_source_fanout_loads():
    result = rz.build_flash_attention_kv_repack_hbm_prefetch_hoist_tile_artifacts(
        _fake_flash_layout_xform_kv_repack_sdscs(),
        tile_index=0,
        prefetch_source_fanout=True,
    )

    assert result is not None
    prefix = "mixed_flash_kv_repack_hbm_prefetch_hoist_0"
    current = result["artifacts"][1][f"{prefix}_current_prefetch"]
    future = result["artifacts"][2][f"{prefix}_future_consumer"]
    assert current["opFuncsUsed_"] == [
        "nop",
        "STCDPOpHBM",
        "STCDPOpHBM",
        "nop",
        "STCDPOpLx",
    ]
    assert current["coreIdToDscSchedule"]["0"] == [
        [0, -1, 0, 1],
        [1, 0, 1, 1],
        [3, -1, 1, 1],
        [4, -1, 1, 0],
    ]
    assert current["coreIdToDscSchedule"]["2"] == [
        [0, -1, 0, 1],
        [-1, 0, 1, 1],
        [3, -1, 1, 1],
        [4, -1, 1, 0],
    ]
    first_load = next(iter(current["datadscs_"][1].values()))
    fanout = next(iter(current["datadscs_"][-1].values()))
    assert first_load["op"]["name"] == "STCDPOpHBM"
    assert len(first_load["labeledDs_"][1]["PieceInfo"]) == 0
    assert fanout["op"]["name"] == "STCDPOpLx"
    assert len(fanout["labeledDs_"][1]["PieceInfo"]) == 64
    assert (
        current["flashAttentionPipeline_"][
            "kv_repack_hbm_prefetch_hoist_prefetch_source_fanout"
        ]
        is True
    )
    assert (
        future["flashAttentionPipeline_"][
            "kv_repack_hbm_prefetch_hoist_prefetch_source_fanout"
        ]
        is True
    )


def test_flash_kv_repack_hbm_prefetch_hoist_can_tail_source_fanout_loads():
    result = rz.build_flash_attention_kv_repack_hbm_prefetch_hoist_tile_artifacts(
        _fake_flash_layout_xform_kv_repack_sdscs(),
        tile_index=0,
        prefetch_source_fanout=True,
        tail_current_prefetch=True,
    )

    assert result is not None
    prefix = "mixed_flash_kv_repack_hbm_prefetch_hoist_0"
    current = result["artifacts"][1][f"{prefix}_current_prefetch"]
    future = result["artifacts"][2][f"{prefix}_future_consumer"]
    assert current["opFuncsUsed_"] == [
        "nop",
        "STCDPOpHBM",
        "STCDPOpHBM",
        "nop",
        "STCDPOpLx",
    ]
    assert current["coreIdToDscSchedule"]["0"] == [
        [0, -1, 0, 1],
        [-1, 0, 1, 1],
        [1, -1, 1, 1],
        [3, -1, 1, 1],
        [4, -1, 1, 0],
    ]
    assert current["coreIdToDscSchedule"]["2"] == [
        [0, -1, 0, 1],
        [-1, 0, 1, 1],
        [3, -1, 1, 1],
        [4, -1, 1, 0],
    ]
    assert (
        current["flashAttentionPipeline_"][
            "kv_repack_hbm_prefetch_hoist_prefetch_source_fanout"
        ]
        is True
    )
    assert (
        current["flashAttentionPipeline_"][
            "kv_repack_hbm_prefetch_hoist_tail_current_prefetch"
        ]
        is True
    )
    assert (
        future["flashAttentionPipeline_"][
            "kv_repack_hbm_prefetch_hoist_prefetch_source_fanout"
        ]
        is True
    )
    assert (
        future["flashAttentionPipeline_"][
            "kv_repack_hbm_prefetch_hoist_tail_current_prefetch"
        ]
        is True
    )


def test_flash_kv_repack_hbm_prefetch_hoist_can_tail_loader_fanout_load():
    result = rz.build_flash_attention_kv_repack_hbm_prefetch_hoist_tile_artifacts(
        _fake_flash_layout_xform_kv_repack_sdscs(),
        tile_index=0,
        prefetch_loader_fanout=True,
        tail_current_prefetch=True,
    )

    assert result is not None
    prefix = "mixed_flash_kv_repack_hbm_prefetch_hoist_0"
    current = result["artifacts"][1][f"{prefix}_current_prefetch"]
    future = result["artifacts"][2][f"{prefix}_future_consumer"]
    assert current["opFuncsUsed_"] == [
        "nop",
        "STCDPOpHBM",
        "nop",
        "STCDPOpLx",
    ]
    assert current["coreIdToDscSchedule"]["0"] == [
        [0, -1, 0, 1],
        [-1, 0, 1, 1],
        [1, -1, 1, 1],
        [2, -1, 1, 1],
        [3, -1, 1, 0],
    ]
    assert current["coreIdToDscSchedule"]["2"] == [
        [0, -1, 0, 1],
        [-1, 0, 1, 1],
        [2, -1, 1, 1],
        [3, -1, 1, 0],
    ]
    load = next(iter(current["datadscs_"][1].values()))
    fanout = next(iter(current["datadscs_"][-1].values()))
    current_meta = current["flashAttentionPipeline_"]
    assert load["coreIdsUsed_"] == [0]
    assert len(load["labeledDs_"][1]["PieceInfo"]) == 0
    assert fanout["labeledDs_"][0]["PieceInfo"][0]["PlacementInfo"] == [
        {"type": "lx", "memId": [0], "startAddr": [rz.PRODUCER_LX_BASE]}
    ]
    assert fanout["labeledDs_"][0]["PieceInfo"][1]["PlacementInfo"] == [
        {"type": "lx", "memId": [0], "startAddr": [rz.PRODUCER_LX_BASE + 128]}
    ]
    assert fanout["labeledDs_"][0]["PieceInfo"][1]["validGap_"]["x_"] == [[2, 0]]
    assert fanout["labeledDs_"][1]["PieceInfo"][1]["PlacementInfo"] == [
        {
            "type": "lx",
            "memId": [0],
            "startAddr": [
                current_meta["kv_repack_hbm_prefetch_hoist_consumer_lx_base"] + 128
            ],
        }
    ]
    assert fanout["labeledDs_"][1]["PieceInfo"][1]["validGap_"]["x_"] == [[2, 0]]
    assert (
        current_meta["kv_repack_hbm_prefetch_hoist_prefetch_loader_fanout"]
        is True
    )
    assert (
        future["flashAttentionPipeline_"][
            "kv_repack_hbm_prefetch_hoist_prefetch_loader_fanout"
        ]
        is True
    )


def test_flash_kv_repack_hbm_prefetch_hoist_can_force_fanout_unicast():
    result = rz.build_flash_attention_kv_repack_hbm_prefetch_hoist_tile_artifacts(
        _fake_flash_layout_xform_kv_repack_sdscs(),
        tile_index=0,
        prefetch_loader_fanout=True,
        prefetch_fanout_use_unicast=1,
        tail_current_prefetch=True,
    )

    assert result is not None
    prefix = "mixed_flash_kv_repack_hbm_prefetch_hoist_0"
    current = result["artifacts"][1][f"{prefix}_current_prefetch"]
    future = result["artifacts"][2][f"{prefix}_future_consumer"]
    fanout = next(iter(current["datadscs_"][-1].values()))
    assert fanout["op"]["useUnicast"] == 1
    assert (
        current["flashAttentionPipeline_"][
            "kv_repack_hbm_prefetch_hoist_prefetch_fanout_use_unicast"
        ]
        == 1
    )
    assert (
        future["flashAttentionPipeline_"][
            "kv_repack_hbm_prefetch_hoist_prefetch_fanout_use_unicast"
        ]
        == 1
    )


def test_flash_kv_repack_hbm_prefetch_hoist_can_force_fanout_lxsfp_lx_transfer_mode():
    result = rz.build_flash_attention_kv_repack_hbm_prefetch_hoist_tile_artifacts(
        _fake_flash_layout_xform_kv_repack_sdscs(),
        tile_index=0,
        prefetch_loader_fanout=True,
        prefetch_fanout_use_lxsfp_lx_transfers=0,
        tail_current_prefetch=True,
    )

    assert result is not None
    prefix = "mixed_flash_kv_repack_hbm_prefetch_hoist_0"
    current = result["artifacts"][1][f"{prefix}_current_prefetch"]
    future = result["artifacts"][2][f"{prefix}_future_consumer"]
    fanout = next(iter(current["datadscs_"][-1].values()))
    assert fanout["op"]["useLXSFPLXTransfers"] == 0
    assert (
        current["flashAttentionPipeline_"][
            "kv_repack_hbm_prefetch_hoist_prefetch_fanout_use_lxsfp_lx_transfers"
        ]
        == 0
    )
    assert (
        future["flashAttentionPipeline_"][
            "kv_repack_hbm_prefetch_hoist_prefetch_fanout_use_lxsfp_lx_transfers"
        ]
        == 0
    )


def test_flash_kv_repack_hbm_prefetch_hoist_can_copyback_fanout_to_hbm():
    result = rz.build_flash_attention_kv_repack_hbm_prefetch_hoist_tile_artifacts(
        _fake_flash_layout_xform_kv_repack_sdscs(),
        tile_index=0,
        prefetch_loader_fanout=True,
        prefetch_fanout_copyback_core=0,
        tail_current_prefetch=True,
    )

    assert result is not None
    prefix = "mixed_flash_kv_repack_hbm_prefetch_hoist_0"
    current = result["artifacts"][1][f"{prefix}_current_prefetch"]
    future = result["artifacts"][2][f"{prefix}_future_consumer"]
    assert current["opFuncsUsed_"] == [
        "nop",
        "STCDPOpHBM",
        "nop",
        "STCDPOpLx",
        "STCDPOpHBM",
    ]
    assert current["coreIdToDscSchedule"]["0"] == [
        [0, -1, 0, 1],
        [-1, 0, 1, 1],
        [1, -1, 1, 1],
        [2, -1, 1, 1],
        [3, -1, 1, 1],
        [4, -1, 1, 0],
    ]
    assert current["coreIdToDscSchedule"]["2"] == [
        [0, -1, 0, 1],
        [-1, 0, 1, 1],
        [2, -1, 1, 1],
        [3, -1, 1, 0],
    ]
    copyback = next(iter(current["datadscs_"][-1].values()))
    assert copyback["op"]["name"] == "STCDPOpHBM"
    assert copyback["coreIdsUsed_"] == [0]
    assert copyback["labeledDs_"][0]["PieceInfo"][0]["PlacementInfo"] == [
        {
            "type": "lx",
            "memId": [0],
            "startAddr": [
                current["flashAttentionPipeline_"][
                    "kv_repack_hbm_prefetch_hoist_consumer_lx_base"
                ]
            ],
        }
    ]
    assert copyback["labeledDs_"][1]["PieceInfo"][0]["PlacementInfo"][-1] == {
        "type": "hbm",
        "memId": [-1],
        "startAddr": [_hbm_dataop_addr(12288)],
    }
    assert future["opFuncsUsed_"] == []
    assert future["datadscs_"] == []
    assert (
        future["flashAttentionPipeline_"][
            "kv_repack_hbm_prefetch_hoist_prefetch_fanout_copyback_core"
        ]
        == 0
    )


def test_flash_kv_repack_hbm_prefetch_hoist_can_restrict_fanout_to_copyback_core():
    result = rz.build_flash_attention_kv_repack_hbm_prefetch_hoist_tile_artifacts(
        _fake_flash_layout_xform_kv_repack_sdscs(),
        tile_index=0,
        prefetch_loader_fanout=True,
        prefetch_fanout_copyback_core=0,
        prefetch_fanout_restrict_to_copyback_core=True,
        tail_current_prefetch=True,
    )

    assert result is not None
    prefix = "mixed_flash_kv_repack_hbm_prefetch_hoist_0"
    current = result["artifacts"][1][f"{prefix}_current_prefetch"]
    future = result["artifacts"][2][f"{prefix}_future_consumer"]
    assert current["opFuncsUsed_"] == [
        "nop",
        "STCDPOpHBM",
        "nop",
        "STCDPOpLx",
        "STCDPOpHBM",
    ]
    assert current["coreIdToDscSchedule"]["0"] == [
        [0, -1, 0, 1],
        [-1, 0, 1, 1],
        [1, -1, 1, 1],
        [2, -1, 1, 1],
        [3, -1, 1, 1],
        [4, -1, 1, 0],
    ]
    assert current["coreIdToDscSchedule"]["2"] == [
        [0, -1, 0, 1],
        [-1, 0, 1, 1],
        [2, -1, 1, 0],
    ]
    fanout = next(iter(current["datadscs_"][3].values()))
    assert fanout["coreIdsUsed_"] == [0]
    dst_pieces = fanout["labeledDs_"][1]["PieceInfo"]
    assert len(dst_pieces) == 2
    assert dst_pieces[0]["PlacementInfo"] == [
        {
            "type": "lx",
            "memId": [0],
            "startAddr": [
                current["flashAttentionPipeline_"][
                    "kv_repack_hbm_prefetch_hoist_consumer_lx_base"
                ]
            ],
        }
    ]
    assert dst_pieces[1]["PlacementInfo"] == [
        {
            "type": "lx",
            "memId": [0],
            "startAddr": [
                current["flashAttentionPipeline_"][
                    "kv_repack_hbm_prefetch_hoist_consumer_lx_base"
                ]
                + 128
            ],
        }
    ]
    assert (
        current["flashAttentionPipeline_"][
            "kv_repack_hbm_prefetch_hoist_prefetch_fanout_restrict_to_copyback_core"
        ]
        is True
    )
    assert (
        future["flashAttentionPipeline_"][
            "kv_repack_hbm_prefetch_hoist_prefetch_fanout_restrict_to_copyback_core"
        ]
        is True
    )


def test_flash_kv_repack_hbm_prefetch_hoist_can_copyback_loader_without_fanout():
    result = rz.build_flash_attention_kv_repack_hbm_prefetch_hoist_tile_artifacts(
        _fake_flash_layout_xform_kv_repack_sdscs(),
        tile_index=0,
        prefetch_loader_fanout=True,
        prefetch_fanout_copyback_core=0,
        prefetch_loader_copyback_without_fanout=True,
        tail_current_prefetch=True,
    )

    assert result is not None
    prefix = "mixed_flash_kv_repack_hbm_prefetch_hoist_0"
    current = result["artifacts"][1][f"{prefix}_current_prefetch"]
    future = result["artifacts"][2][f"{prefix}_future_consumer"]
    assert current["opFuncsUsed_"] == [
        "nop",
        "STCDPOpHBM",
        "nop",
        "STCDPOpHBM",
    ]
    assert current["coreIdToDscSchedule"]["0"] == [
        [0, -1, 0, 1],
        [-1, 0, 1, 1],
        [1, -1, 1, 1],
        [2, -1, 1, 1],
        [3, -1, 1, 0],
    ]
    assert current["coreIdToDscSchedule"]["2"] == [
        [0, -1, 0, 1],
        [-1, 0, 1, 1],
        [2, -1, 1, 0],
    ]
    copyback = next(iter(current["datadscs_"][-1].values()))
    assert copyback["op"]["name"] == "STCDPOpHBM"
    assert copyback["coreIdsUsed_"] == [0]
    assert copyback["labeledDs_"][0]["PieceInfo"][0]["PlacementInfo"] == [
        {
            "type": "lx",
            "memId": [0],
            "startAddr": [rz.PRODUCER_LX_BASE],
        }
    ]
    assert copyback["labeledDs_"][1]["PieceInfo"][0]["PlacementInfo"][-1] == {
        "type": "hbm",
        "memId": [-1],
        "startAddr": [_hbm_dataop_addr(12288)],
    }
    assert (
        current["flashAttentionPipeline_"][
            "kv_repack_hbm_prefetch_hoist_prefetch_loader_copyback_without_fanout"
        ]
        is True
    )
    assert (
        future["flashAttentionPipeline_"][
            "kv_repack_hbm_prefetch_hoist_prefetch_loader_copyback_without_fanout"
        ]
        is True
    )


def test_flash_kv_repack_hbm_prefetch_hoist_direct_copyback_uses_loader_lx_base():
    result = rz.build_flash_attention_kv_repack_hbm_prefetch_hoist_tile_artifacts(
        _fake_flash_layout_xform_kv_repack_sdscs(),
        tile_index=0,
        prefetch_loader_fanout=True,
        prefetch_loader_lx_base=-2,
        prefetch_fanout_copyback_core=0,
        prefetch_loader_copyback_without_fanout=True,
        tail_current_prefetch=True,
    )

    assert result is not None
    prefix = "mixed_flash_kv_repack_hbm_prefetch_hoist_0"
    current = result["artifacts"][1][f"{prefix}_current_prefetch"]
    loader_lx_base = (
        current["flashAttentionPipeline_"][
            "kv_repack_hbm_prefetch_hoist_consumer_lx_base"
        ]
        + (256 << 10)
    )
    copyback = next(iter(current["datadscs_"][-1].values()))
    assert copyback["labeledDs_"][0]["PieceInfo"][0]["PlacementInfo"] == [
        {"type": "lx", "memId": [0], "startAddr": [loader_lx_base]}
    ]
    assert (
        current["flashAttentionPipeline_"][
            "kv_repack_hbm_prefetch_hoist_prefetch_loader_lx_base"
        ]
        == loader_lx_base
    )


def test_flash_kv_repack_hbm_prefetch_hoist_loader_copyback_can_disable_overlap_after_sync():
    result = rz.build_flash_attention_kv_repack_hbm_prefetch_hoist_tile_artifacts(
        _fake_flash_layout_xform_kv_repack_sdscs(),
        tile_index=0,
        prefetch_loader_fanout=True,
        prefetch_fanout_copyback_core=0,
        prefetch_loader_copyback_without_fanout=True,
        overlap_after_sync=False,
    )

    assert result is not None
    prefix = "mixed_flash_kv_repack_hbm_prefetch_hoist_0"
    current = result["artifacts"][1][f"{prefix}_current_prefetch"]
    assert current["coreIdToDscSchedule"]["0"] == [
        [0, -1, 0, 1],
        [1, 0, 1, 0],
        [2, -1, 1, 1],
        [3, -1, 1, 0],
    ]
    assert (
        current["flashAttentionPipeline_"][
            "kv_repack_hbm_prefetch_hoist_overlap_after_sync"
        ]
        is False
    )


def test_flash_kv_repack_hbm_prefetch_hoist_can_fanout_full_tile_piece():
    result = rz.build_flash_attention_kv_repack_hbm_prefetch_hoist_tile_artifacts(
        _fake_flash_layout_xform_kv_repack_sdscs(),
        tile_index=0,
        prefetch_loader_fanout=True,
        prefetch_fanout_copyback_core=0,
        prefetch_fanout_restrict_to_copyback_core=True,
        prefetch_loader_fanout_full_tile_pieces=True,
        tail_current_prefetch=True,
    )

    assert result is not None
    prefix = "mixed_flash_kv_repack_hbm_prefetch_hoist_0"
    current = result["artifacts"][1][f"{prefix}_current_prefetch"]
    future = result["artifacts"][2][f"{prefix}_future_consumer"]
    assert current["opFuncsUsed_"] == [
        "nop",
        "STCDPOpHBM",
        "nop",
        "STCDPOpLx",
        "STCDPOpHBM",
    ]
    fanout = next(iter(current["datadscs_"][3].values()))
    assert fanout["coreIdsUsed_"] == [0]
    src_pieces = fanout["labeledDs_"][0]["PieceInfo"]
    dst_pieces = fanout["labeledDs_"][1]["PieceInfo"]
    assert len(src_pieces) == 1
    assert len(dst_pieces) == 1
    assert src_pieces[0]["dimToSize_"]["x_"] == 2
    assert dst_pieces[0]["dimToSize_"]["x_"] == 2
    assert src_pieces[0]["PlacementInfo"] == [
        {"type": "lx", "memId": [0], "startAddr": [rz.PRODUCER_LX_BASE]}
    ]
    assert dst_pieces[0]["PlacementInfo"] == [
        {
            "type": "lx",
            "memId": [0],
            "startAddr": [
                current["flashAttentionPipeline_"][
                    "kv_repack_hbm_prefetch_hoist_consumer_lx_base"
                ]
            ],
        }
    ]
    assert (
        current["flashAttentionPipeline_"][
            "kv_repack_hbm_prefetch_hoist_prefetch_loader_fanout_full_tile_pieces"
        ]
        is True
    )
    assert (
        future["flashAttentionPipeline_"][
            "kv_repack_hbm_prefetch_hoist_prefetch_loader_fanout_full_tile_pieces"
        ]
        is True
    )


def test_flash_kv_repack_hbm_prefetch_hoist_can_select_loader_core():
    result = rz.build_flash_attention_kv_repack_hbm_prefetch_hoist_tile_artifacts(
        _fake_flash_layout_xform_kv_repack_sdscs(),
        tile_index=0,
        prefetch_loader_fanout=True,
        prefetch_loader_core_id=31,
        prefetch_fanout_copyback_core=31,
        prefetch_fanout_restrict_to_copyback_core=True,
        prefetch_loader_fanout_full_tile_pieces=True,
        tail_current_prefetch=True,
    )

    assert result is not None
    prefix = "mixed_flash_kv_repack_hbm_prefetch_hoist_0"
    current = result["artifacts"][1][f"{prefix}_current_prefetch"]
    future = result["artifacts"][2][f"{prefix}_future_consumer"]
    assert current["coreIdToDscSchedule"]["0"] == [
        [0, -1, 0, 1],
        [-1, 0, 1, 1],
        [2, -1, 1, 0],
    ]
    assert current["coreIdToDscSchedule"]["31"] == [
        [0, -1, 0, 1],
        [-1, 0, 1, 1],
        [1, -1, 1, 1],
        [2, -1, 1, 1],
        [3, -1, 1, 1],
        [4, -1, 1, 0],
    ]
    load = next(iter(current["datadscs_"][1].values()))
    fanout = next(iter(current["datadscs_"][3].values()))
    assert load["coreIdsUsed_"] == [31]
    assert fanout["coreIdsUsed_"] == [31]
    assert fanout["labeledDs_"][0]["PieceInfo"][0]["PlacementInfo"] == [
        {"type": "lx", "memId": [31], "startAddr": [rz.PRODUCER_LX_BASE]}
    ]
    assert (
        current["flashAttentionPipeline_"][
            "kv_repack_hbm_prefetch_hoist_prefetch_loader_core_id"
        ]
        == 31
    )
    assert (
        future["flashAttentionPipeline_"][
            "kv_repack_hbm_prefetch_hoist_prefetch_loader_core_id"
        ]
        == 31
    )


def test_flash_kv_repack_hbm_prefetch_hoist_can_serialize_loader_core_compute():
    result = rz.build_flash_attention_kv_repack_hbm_prefetch_hoist_tile_artifacts(
        _fake_flash_layout_xform_kv_repack_sdscs(),
        tile_index=0,
        prefetch_loader_fanout=True,
        prefetch_loader_core_id=31,
        prefetch_loader_fanout_full_tile_pieces=True,
        serialize_loader_core_prefetch=True,
    )

    assert result is not None
    prefix = "mixed_flash_kv_repack_hbm_prefetch_hoist_0"
    current = result["artifacts"][1][f"{prefix}_current_prefetch"]
    future = result["artifacts"][2][f"{prefix}_future_consumer"]
    assert current["coreIdToDscSchedule"]["0"] == [
        [0, -1, 0, 1],
        [-1, 0, 1, 1],
        [2, -1, 1, 1],
        [3, -1, 1, 0],
    ]
    assert current["coreIdToDscSchedule"]["31"] == [
        [0, -1, 0, 1],
        [1, -1, 1, 1],
        [-1, 0, 1, 1],
        [2, -1, 1, 1],
        [3, -1, 1, 0],
    ]
    assert (
        current["flashAttentionPipeline_"][
            "kv_repack_hbm_prefetch_hoist_serialize_loader_core_prefetch"
        ]
        is True
    )
    assert (
        future["flashAttentionPipeline_"][
            "kv_repack_hbm_prefetch_hoist_serialize_loader_core_prefetch"
        ]
        is True
    )


def test_flash_kv_repack_hbm_prefetch_hoist_can_move_loader_source_lx_base():
    result = rz.build_flash_attention_kv_repack_hbm_prefetch_hoist_tile_artifacts(
        _fake_flash_layout_xform_kv_repack_sdscs(),
        tile_index=0,
        prefetch_loader_fanout=True,
        prefetch_loader_lx_base=-2,
        prefetch_fanout_copyback_core=0,
        prefetch_fanout_restrict_to_copyback_core=True,
        prefetch_loader_fanout_full_tile_pieces=True,
        tail_current_prefetch=True,
    )

    assert result is not None
    prefix = "mixed_flash_kv_repack_hbm_prefetch_hoist_0"
    current = result["artifacts"][1][f"{prefix}_current_prefetch"]
    future = result["artifacts"][2][f"{prefix}_future_consumer"]
    loader_lx_base = (
        current["flashAttentionPipeline_"][
            "kv_repack_hbm_prefetch_hoist_consumer_lx_base"
        ]
        + (256 << 10)
    )
    load = next(iter(current["datadscs_"][1].values()))
    fanout = next(iter(current["datadscs_"][3].values()))
    copyback = next(iter(current["datadscs_"][4].values()))
    assert load["labeledDs_"][0]["PieceInfo"][0]["PlacementInfo"][0] == {
        "type": "lx",
        "memId": [0],
        "startAddr": [loader_lx_base],
    }
    assert fanout["labeledDs_"][0]["PieceInfo"][0]["PlacementInfo"] == [
        {"type": "lx", "memId": [0], "startAddr": [loader_lx_base]}
    ]
    assert copyback["labeledDs_"][0]["PieceInfo"][0]["PlacementInfo"] == [
        {
            "type": "lx",
            "memId": [0],
            "startAddr": [
                current["flashAttentionPipeline_"][
                    "kv_repack_hbm_prefetch_hoist_consumer_lx_base"
                ]
            ],
        }
    ]
    assert (
        current["flashAttentionPipeline_"][
            "kv_repack_hbm_prefetch_hoist_prefetch_loader_lx_base"
        ]
        == loader_lx_base
    )
    assert (
        current["flashAttentionPipeline_"][
            "kv_repack_hbm_prefetch_hoist_prefetch_loader_lx_base_request"
        ]
        == -2
    )
    assert (
        future["flashAttentionPipeline_"][
            "kv_repack_hbm_prefetch_hoist_prefetch_loader_lx_base"
        ]
        == loader_lx_base
    )


def test_flash_kv_repack_hbm_prefetch_hoist_can_route_prefetch_corelet1():
    result = rz.build_flash_attention_kv_repack_hbm_prefetch_hoist_tile_artifacts(
        _fake_flash_layout_xform_kv_repack_sdscs(),
        tile_index=0,
        prefetch_corelet_id=1,
    )

    assert result is not None
    prefix = "mixed_flash_kv_repack_hbm_prefetch_hoist_0"
    current = result["artifacts"][1][f"{prefix}_current_prefetch"]
    future = result["artifacts"][2][f"{prefix}_future_consumer"]
    first_load = next(iter(current["datadscs_"][-1].values()))

    assert first_load["op"]["coreletId"] == 1
    assert (
        current["flashAttentionPipeline_"][
            "kv_repack_hbm_prefetch_hoist_prefetch_corelet_id"
        ]
        == 1
    )
    assert (
        future["flashAttentionPipeline_"][
            "kv_repack_hbm_prefetch_hoist_prefetch_corelet_id"
        ]
        == 1
    )


def test_flash_kv_repack_hbm_prefetch_hoist_can_emit_lx_roundtrip_prefetch():
    result = rz.build_flash_attention_kv_repack_hbm_prefetch_hoist_tile_artifacts(
        _fake_flash_layout_xform_kv_repack_sdscs(),
        tile_index=0,
        prefetch_lx_roundtrip=True,
        prefetch_corelet_id=1,
    )

    assert result is not None
    prefix = "mixed_flash_kv_repack_hbm_prefetch_hoist_0"
    current = result["artifacts"][1][f"{prefix}_current_prefetch"]
    future = result["artifacts"][2][f"{prefix}_future_consumer"]
    first_load = next(iter(current["datadscs_"][-1].values()))

    assert first_load["op"]["coreletId"] == 1
    assert (
        current["flashAttentionPipeline_"][
            "kv_repack_hbm_prefetch_hoist_prefetch_lx_roundtrip"
        ]
        is True
    )
    assert (
        future["flashAttentionPipeline_"][
            "kv_repack_hbm_prefetch_hoist_prefetch_lx_roundtrip"
        ]
        is True
    )
    assert len(first_load["labeledDs_"][1]["PieceInfo"]) == 32
    assert first_load["labeledDs_"][1]["PieceInfo"][0]["PlacementInfo"] == [
        first_load["labeledDs_"][0]["PieceInfo"][0]["PlacementInfo"][0]
    ]


def test_flash_kv_repack_hbm_prefetch_hoist_can_serialize_prefetch_probe():
    result = rz.build_flash_attention_kv_repack_hbm_prefetch_hoist_tile_artifacts(
        _fake_flash_layout_xform_kv_repack_sdscs(),
        tile_index=0,
        serial_prefetch=True,
    )

    assert result is not None
    prefix = "mixed_flash_kv_repack_hbm_prefetch_hoist_0"
    future_name = f"{prefix}_future_consumer"
    future = result["artifacts"][1][future_name]
    assert result["replacements"] == {
        "2_batchmatmul": future_name,
    }
    assert result["insertions_before"] == {
        "0_batchmatmul": [f"{prefix}_future_producer"],
    }
    assert len(result["artifacts"]) == 2
    assert len(future["dscs_"]) == 1
    assert future["opFuncsUsed_"] == ["STCDPOpHBM"]
    assert future["coreIdToDscSchedule"]["0"] == [
        [0, -1, 0, 1],
        [-1, 0, 1, 0],
    ]
    future_meta = future["flashAttentionPipeline_"]
    assert future_meta["kv_repack_hbm_prefetch_hoist_serial_prefetch"] is True
    assert future_meta["kv_repack_hbm_prefetch_hoist_dataop_count"] == 1
    assert future_meta["compute_tile_count"] == 1


def test_flash_kv_repack_hbm_prefetch_hoist_can_keep_redundant_future_prefetch():
    result = rz.build_flash_attention_kv_repack_hbm_prefetch_hoist_tile_artifacts(
        _fake_flash_layout_xform_kv_repack_sdscs(),
        tile_index=0,
        redundant_future_prefetch=True,
    )

    assert result is not None
    prefix = "mixed_flash_kv_repack_hbm_prefetch_hoist_0"
    current_name = f"{prefix}_current_prefetch"
    future_name = f"{prefix}_future_consumer"
    current = result["artifacts"][1][current_name]
    future = result["artifacts"][2][future_name]
    assert result["replacements"] == {
        "0_batchmatmul": current_name,
        "2_batchmatmul": future_name,
    }
    assert current["opFuncsUsed_"] == ["nop", "STCDPOpHBM"]
    assert future["opFuncsUsed_"] == ["STCDPOpHBM"]
    assert future["coreIdToDscSchedule"]["0"] == [
        [0, -1, 0, 1],
        [-1, 0, 1, 0],
    ]
    future_meta = future["flashAttentionPipeline_"]
    assert future_meta["kv_repack_hbm_prefetch_hoist_serial_prefetch"] is False
    assert (
        future_meta["kv_repack_hbm_prefetch_hoist_redundant_future_prefetch"]
        is True
    )
    assert future_meta["kv_repack_hbm_prefetch_hoist_dataop_count"] == 1
    future_load_name = next(iter(future["datadscs_"][0]))
    assert future_load_name.startswith("future_redundant_")
    future_dl = rz._dl_op({"future": future})
    future_lds = rz._lds_by_idx(future_dl, 1)
    assert "isExternal_" not in future_lds
    assert "isFirstUse_" not in future_lds


def test_flash_kv_repack_hbm_prefetch_hoist_can_serialize_current_prefetch():
    result = rz.build_flash_attention_kv_repack_hbm_prefetch_hoist_tile_artifacts(
        _fake_flash_layout_xform_kv_repack_sdscs(),
        tile_index=0,
        redundant_future_prefetch=True,
        serialize_current_prefetch=True,
    )

    assert result is not None
    prefix = "mixed_flash_kv_repack_hbm_prefetch_hoist_0"
    current = result["artifacts"][1][f"{prefix}_current_prefetch"]
    future = result["artifacts"][2][f"{prefix}_future_consumer"]
    assert current["opFuncsUsed_"] == ["nop", "STCDPOpHBM"]
    assert current["coreIdToDscSchedule"]["0"] == [
        [0, -1, 0, 1],
        [1, -1, 1, 1],
        [-1, 0, 1, 0],
    ]
    current_meta = current["flashAttentionPipeline_"]
    assert (
        current_meta["kv_repack_hbm_prefetch_hoist_serialize_current_prefetch"]
        is True
    )
    future_meta = future["flashAttentionPipeline_"]
    assert (
        future_meta["kv_repack_hbm_prefetch_hoist_serialize_current_prefetch"]
        is True
    )


def test_flash_kv_repack_hbm_prefetch_hoist_uses_lx_local_future_marker_by_default():
    result = rz.build_flash_attention_kv_repack_hbm_prefetch_hoist_tile_artifacts(
        _fake_flash_layout_xform_kv_repack_sdscs(),
        tile_index=0,
    )

    assert result is not None
    prefix = "mixed_flash_kv_repack_hbm_prefetch_hoist_0"
    future = result["artifacts"][2][f"{prefix}_future_consumer"]
    future_meta = future["flashAttentionPipeline_"]
    assert (
        future_meta["kv_repack_hbm_prefetch_hoist_external_future_prefetch"]
        is False
    )
    future_dl = rz._dl_op({"future": future})
    future_lds = rz._lds_by_idx(future_dl, 1)
    assert "isExternal_" not in future_lds
    assert "isFirstUse_" not in future_lds


def test_flash_kv_repack_hbm_prefetch_hoist_can_force_external_future_marker():
    result = rz.build_flash_attention_kv_repack_hbm_prefetch_hoist_tile_artifacts(
        _fake_flash_layout_xform_kv_repack_sdscs(),
        tile_index=0,
        external_future_prefetch=True,
    )

    assert result is not None
    prefix = "mixed_flash_kv_repack_hbm_prefetch_hoist_0"
    future = result["artifacts"][2][f"{prefix}_future_consumer"]
    future_meta = future["flashAttentionPipeline_"]
    assert (
        future_meta["kv_repack_hbm_prefetch_hoist_external_future_prefetch"]
        is True
    )
    future_dl = rz._dl_op({"future": future})
    future_lds = rz._lds_by_idx(future_dl, 1)
    assert future_lds["isExternal_"] == 1
    assert future_lds["isFirstUse_"] == 0


def test_flash_kv_repack_broadcast_plan_fans_out_low_core_input1():
    sdscs = _fake_flash_layout_xform_kv_repack_sdscs()

    plan = rz.build_flash_attention_kv_repack_broadcast_plan_artifact(
        sdscs,
        tile_index=1,
        input_idx=1,
    )

    assert plan is not None
    root = plan["flash_kv_repack_broadcast_plan_1_input1"]
    meta = root["flashAttentionPipeline_"]
    assert meta["kv_repack_broadcast_executable"] is False
    assert meta["kv_repack_source_sdsc"] == "1_ReStickifyOpHBM"
    assert meta["kv_repack_consumer_sdsc"] == "2_batchmatmul"
    assert meta["kv_repack_input_idx"] == 1
    assert meta["kv_repack_producer_cores"] == 2
    assert meta["kv_repack_consumer_cores"] == 32
    assert meta["kv_repack_source_layout"] == ["in_", "x_", "out_"]
    assert meta["kv_repack_consumer_layout"] == ["in_", "x_", "out_"]
    assert meta["kv_repack_iter_sizes"] == {"in_": 64, "x_": 2, "out_": 64}
    assert meta["kv_repack_stick_dim"] == "out_"
    assert meta["kv_repack_producer_split"] == "mb_"
    assert meta["kv_repack_mapped_split"] == "x_"
    assert meta["kv_repack_consumer_split"] == "mb_"
    assert meta["kv_repack_source_lx_base"] == rz.PRODUCER_LX_BASE
    assert meta["kv_repack_consumer_lx_base"] == (
        rz.PRODUCER_LX_BASE + meta["slice_bytes"]
    )
    assert meta["kv_repack_source_piece_count"] == 2
    assert meta["kv_repack_destination_piece_count"] == 64
    assert root["kvRepackBroadcastPlan_"]["runtime_status"] == "not_executed"

    assert root["numCoresUsed_"] == 32
    assert root["opFuncsUsed_"] == ["STCDPOpLx"]
    assert root["dscs_"] == []
    assert root["coreIdToDscSchedule"]["0"] == [[0, -1, 0, 0]]
    assert root["coreIdToDscSchedule"]["31"] == [[0, -1, 0, 0]]
    dataop = root["datadscs_"][0][
        "0_STCDPOpLx_kv_repack_broadcast_tile1_input1"
    ]
    assert dataop["coreIdsUsed_"] == list(range(32))
    assert dataop["primaryDs_"] == [
        {"name_": "dataIN", "dimNames": ["in_", "x_", "out_"]},
        {"name_": "dataOUT", "dimNames": ["in_", "x_", "out_"]},
    ]
    src_ld, dst_ld = dataop["labeledDs_"]
    assert src_ld["layoutDimOrder_"] == ["in_", "x_", "out_"]
    assert dst_ld["layoutDimOrder_"] == ["in_", "x_", "out_"]
    src_pieces = src_ld["PieceInfo"]
    dst_pieces = dst_ld["PieceInfo"]
    assert len(src_pieces) == 2
    assert len(dst_pieces) == 64
    assert src_pieces[0]["dimToStartCordinate"] == {
        "in_": 0,
        "x_": 0,
        "out_": 0,
    }
    assert src_pieces[1]["dimToStartCordinate"] == {
        "in_": 0,
        "x_": 1,
        "out_": 0,
    }
    assert src_pieces[0]["PlacementInfo"][0]["memId"] == [0]
    assert src_pieces[1]["PlacementInfo"][0]["memId"] == [1]
    assert src_pieces[0]["PlacementInfo"][0]["startAddr"] == [
        meta["kv_repack_source_lx_base"]
    ]
    assert src_pieces[1]["PlacementInfo"][0]["startAddr"] == [
        meta["kv_repack_source_lx_base"]
    ]
    assert dst_pieces[0]["PlacementInfo"][0]["memId"] == [0]
    assert dst_pieces[1]["PlacementInfo"][0]["memId"] == [0]
    assert dst_pieces[2]["PlacementInfo"][0]["memId"] == [1]
    assert dst_pieces[-1]["PlacementInfo"][0]["memId"] == [31]
    assert dst_pieces[0]["PlacementInfo"][0]["startAddr"] == [
        meta["kv_repack_consumer_lx_base"]
    ]
    assert dst_pieces[1]["PlacementInfo"][0]["startAddr"] == [
        meta["kv_repack_consumer_lx_base"] + 64 * rz.WORD_LENGTH
    ]
    assert src_pieces[1]["validGap_"]["x_"] == [[1, 0]]
    assert dst_pieces[1]["validGap_"]["x_"] == [[2, 0]]
    assert dst_pieces[0]["broadcastSourcePieceKey_"] == src_pieces[0]["key_"]
    assert dst_pieces[1]["broadcastSourcePieceKey_"] == src_pieces[1]["key_"]
    assert dst_pieces[-1]["broadcastConsumerCore_"] == 31
    assert dst_pieces[0]["dimToStartCordinate"] == src_pieces[0][
        "dimToStartCordinate"
    ]
    assert dst_pieces[1]["dimToStartCordinate"] == src_pieces[1][
        "dimToStartCordinate"
    ]


def test_flash_kv_repack_broadcast_plan_accepts_multisplit_low_core_input1():
    sdscs = _fake_flash_layout_xform_kv_repack_multisplit_sdscs()

    plan = rz.build_flash_attention_kv_repack_broadcast_plan_artifact(
        sdscs,
        tile_index=1,
        input_idx=1,
    )

    assert plan is not None
    root = plan["flash_kv_repack_broadcast_plan_1_input1"]
    meta = root["flashAttentionPipeline_"]
    assert meta["kv_repack_producer_cores"] == 4
    assert meta["kv_repack_consumer_cores"] == 32
    assert meta["kv_repack_producer_split"] == ["mb_", "x_"]
    assert meta["kv_repack_mapped_split"] == ["x_", "in_"]
    assert meta["kv_repack_source_piece_count"] == 4
    dataop = root["datadscs_"][0][
        "0_STCDPOpLx_kv_repack_broadcast_tile1_input1"
    ]
    src_pieces = dataop["labeledDs_"][0]["PieceInfo"]
    assert [piece["dimToStartCordinate"] for piece in src_pieces] == [
        {"in_": 0, "x_": 0, "out_": 0},
        {"in_": 0, "x_": 1, "out_": 0},
        {"in_": 32, "x_": 0, "out_": 0},
        {"in_": 32, "x_": 1, "out_": 0},
    ]


def test_flash_kv_repack_broadcast_pair_wraps_producer_and_consumer():
    sdscs = _fake_flash_layout_xform_kv_repack_sdscs()

    result = rz.build_flash_attention_kv_repack_broadcast_pair_artifacts(
        sdscs,
        tile_index=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
    )

    assert result is not None
    assert result["replacements"] == {
        "1_ReStickifyOpHBM": "mixed_flash_kv_repack_broadcast_pair_1_input1_producer",
        "2_batchmatmul": "mixed_flash_kv_repack_broadcast_pair_1_input1_consumer",
    }
    pred_name = "mixed_flash_kv_repack_broadcast_pair_1_input1_producer"
    cons_name = "mixed_flash_kv_repack_broadcast_pair_1_input1_consumer"
    producer = result["artifacts"][0][pred_name]
    consumer = result["artifacts"][1][cons_name]
    prod_meta = producer["flashAttentionPipeline_"]
    cons_meta = consumer["flashAttentionPipeline_"]
    assert prod_meta["kv_repack_broadcast_role"] == "producer"
    assert cons_meta["kv_repack_broadcast_role"] == "consumer"
    assert cons_meta["kv_repack_broadcast_executable"] is True
    assert cons_meta["kv_repack_runtime_forced"] is True
    assert cons_meta["kv_repack_source_sdsc"] == "1_ReStickifyOpHBM"
    assert cons_meta["kv_repack_consumer_sdsc"] == "2_batchmatmul"
    assert cons_meta["kv_repack_source_piece_count"] == 2
    assert cons_meta["kv_repack_destination_piece_count"] == 64
    assert cons_meta["kv_repack_input_fetch_transfer"] is True
    assert cons_meta["kv_repack_stcdp_subpiece_reuse"] is True
    assert cons_meta["kv_repack_broadcast_group_count"] == 1
    assert cons_meta["kv_repack_broadcast_group_size"] == 0
    assert cons_meta["kv_repack_source_lx_base"] == rz.PRODUCER_LX_BASE
    assert cons_meta["kv_repack_consumer_lx_base"] == (
        rz.PRODUCER_LX_BASE + cons_meta["slice_bytes"]
    )

    assert rz._lds_by_idx(rz._dl_op({pred_name: producer}), 1)["hbmSize_"] == 0
    assert len(consumer["dscs_"]) == 1
    assert len(consumer["datadscs_"]) == 1
    assert consumer["opFuncsUsed_"] == ["STCDPOpLx"]
    compute_dl = next(iter(consumer["dscs_"][0].values()))
    assert rz._has_input_fetch_neighbor_transfer(compute_dl, 1)
    assert consumer["coreIdToDscSchedule"]["0"] == [
        [0, -1, 0, 1],
        [-1, 0, 1, 0],
    ]
    assert consumer["coreIdToDscSchedule"]["31"] == [
        [0, -1, 0, 1],
        [-1, 0, 1, 0],
    ]
    dataop_name, dataop = next(iter(consumer["datadscs_"][0].items()))
    assert dataop_name == "0_STCDPOpLx_kv_repack_broadcast_Tensor0_idx1_tile1"
    src_ld, dst_ld = dataop["labeledDs_"]
    assert len(src_ld["PieceInfo"]) == 2
    assert len(dst_ld["PieceInfo"]) == 64
    assert "broadcastSourcePieceKey_" not in dst_ld["PieceInfo"][0]
    assert "broadcastConsumerCore_" not in dst_ld["PieceInfo"][0]
    assert dst_ld["PieceInfo"][-1]["PlacementInfo"][0]["memId"] == [31]
    assert dataop["op"] == {"name": "STCDPOpLx"}


def test_flash_kv_repack_broadcast_pair_hbm_source_keeps_original_producer():
    sdscs = _fake_flash_layout_xform_kv_repack_sdscs()

    result = rz.build_flash_attention_kv_repack_broadcast_pair_artifacts(
        sdscs,
        tile_index=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        hbm_source=True,
    )

    assert result is not None
    assert result["replacements"] == {
        "2_batchmatmul": "mixed_flash_kv_repack_broadcast_pair_1_input1_consumer",
    }
    assert len(result["artifacts"]) == 1
    cons_name = "mixed_flash_kv_repack_broadcast_pair_1_input1_consumer"
    consumer = result["artifacts"][0][cons_name]
    cons_meta = consumer["flashAttentionPipeline_"]
    assert cons_meta["kv_repack_broadcast_role"] == "consumer"
    assert cons_meta["kv_repack_producer_sidecar"] is None
    assert cons_meta["kv_repack_hbm_source"] is True
    assert cons_meta["kv_repack_broadcast_group_count"] == 1
    assert cons_meta["kv_repack_source_piece_count"] == 2
    assert cons_meta["kv_repack_destination_piece_count"] == 64
    assert len(consumer["dscs_"]) == 1
    assert len(consumer["datadscs_"]) == 4
    assert consumer["opFuncsUsed_"] == [
        "STCDPOpHBM",
        "STCDPOpHBM",
        "STCDPOpLx",
        "nop",
    ]
    assert consumer["coreIdToDscSchedule"]["0"] == [
        [0, -1, 0, 1],
        [2, -1, 1, 1],
        [3, -1, 1, 1],
        [-1, 0, 1, 0],
    ]
    assert consumer["coreIdToDscSchedule"]["1"] == [
        [1, -1, 0, 1],
        [2, -1, 1, 1],
        [3, -1, 1, 1],
        [-1, 0, 1, 0],
    ]
    assert consumer["coreIdToDscSchedule"]["31"] == [
        [2, -1, 0, 1],
        [3, -1, 1, 1],
        [-1, 0, 1, 0],
    ]

    compute_dl = next(iter(consumer["dscs_"][0].values()))
    assert rz._has_input_fetch_neighbor_transfer(compute_dl, 1)

    load0_name, load0 = next(iter(consumer["datadscs_"][0].items()))
    load1_name, load1 = next(iter(consumer["datadscs_"][1].items()))
    fanout_name, fanout = next(iter(consumer["datadscs_"][2].items()))
    nop_name, nop = next(iter(consumer["datadscs_"][3].items()))
    assert load0_name == (
        "0_STCDPOpHBM_kv_repack_broadcast_"
        "Tensor0_idx1_tile1_hbm_source_piece0_load"
    )
    assert load1_name == (
        "1_STCDPOpHBM_kv_repack_broadcast_"
        "Tensor0_idx1_tile1_hbm_source_piece1_load"
    )
    assert fanout_name == "2_STCDPOpLx_kv_repack_broadcast_Tensor0_idx1_tile1"
    assert load0["coreIdsUsed_"] == [0]
    assert load1["coreIdsUsed_"] == [1]
    assert load0["labeledDs_"][0]["PieceInfo"][0]["PlacementInfo"] == [
        {
            "type": "lx",
            "memId": [0],
            "startAddr": [cons_meta["kv_repack_source_lx_base"]],
        },
        {"type": "hbm", "memId": [-1], "startAddr": [_hbm_dataop_addr(12288)]},
    ]
    assert load0["labeledDs_"][1]["PieceInfo"][0]["PlacementInfo"] == [
        {
            "type": "lx",
            "memId": [0],
            "startAddr": [cons_meta["kv_repack_source_lx_base"]],
        },
    ]
    assert load1["labeledDs_"][0]["PieceInfo"][0]["PlacementInfo"][1] == {
        "type": "hbm",
        "memId": [-1],
        "startAddr": [_hbm_dataop_addr(12416)],
    }
    assert fanout["op"] == {"name": "STCDPOpLx"}
    assert nop_name == "3_nop_kv_repack_broadcast_barrier_Tensor0_idx1_tile1"
    assert nop["coreIdsUsed_"] == list(range(32))
    assert nop["op"] == {"name": "nop"}


def test_flash_kv_repack_broadcast_pair_hbm_direct_load_skips_fanout():
    sdscs = _fake_flash_layout_xform_kv_repack_sdscs()

    result = rz.build_flash_attention_kv_repack_broadcast_pair_artifacts(
        sdscs,
        tile_index=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        hbm_direct_load=True,
    )

    assert result is not None
    assert result["replacements"] == {
        "2_batchmatmul": "mixed_flash_kv_repack_broadcast_pair_1_input1_consumer",
    }
    assert len(result["artifacts"]) == 1
    cons_name = "mixed_flash_kv_repack_broadcast_pair_1_input1_consumer"
    consumer = result["artifacts"][0][cons_name]
    cons_meta = consumer["flashAttentionPipeline_"]
    assert cons_meta["kv_repack_broadcast_role"] == "consumer"
    assert cons_meta["kv_repack_producer_sidecar"] is None
    assert cons_meta["kv_repack_hbm_source"] is True
    assert cons_meta["kv_repack_hbm_direct_load"] is True
    assert cons_meta["kv_repack_broadcast_group_count"] == 0
    assert cons_meta["kv_repack_source_piece_count"] == 2
    assert cons_meta["kv_repack_destination_piece_count"] == 64
    assert len(consumer["dscs_"]) == 1
    assert len(consumer["datadscs_"]) == 3
    assert consumer["opFuncsUsed_"] == ["STCDPOpHBM", "STCDPOpHBM", "nop"]
    assert consumer["coreIdToDscSchedule"]["0"] == [
        [0, -1, 0, 1],
        [1, -1, 1, 1],
        [2, -1, 1, 1],
        [-1, 0, 1, 0],
    ]
    assert consumer["coreIdToDscSchedule"]["31"] == [
        [0, -1, 0, 1],
        [1, -1, 1, 1],
        [2, -1, 1, 1],
        [-1, 0, 1, 0],
    ]

    compute_dl = next(iter(consumer["dscs_"][0].values()))
    assert rz._has_input_fetch_neighbor_transfer(compute_dl, 1)

    load0_name, load0 = next(iter(consumer["datadscs_"][0].items()))
    load1_name, load1 = next(iter(consumer["datadscs_"][1].items()))
    nop_name, nop = next(iter(consumer["datadscs_"][2].items()))
    assert load0_name == (
        "0_STCDPOpHBM_kv_repack_broadcast_"
        "Tensor0_idx1_tile1_hbm_direct_piece0_load"
    )
    assert load1_name == (
        "1_STCDPOpHBM_kv_repack_broadcast_"
        "Tensor0_idx1_tile1_hbm_direct_piece1_load"
    )
    assert load0["coreIdsUsed_"] == list(range(32))
    assert load1["coreIdsUsed_"] == list(range(32))
    assert load0["labeledDs_"][0]["PieceInfo"][0]["PlacementInfo"] == [
        {
            "type": "lx",
            "memId": [0],
            "startAddr": [cons_meta["kv_repack_consumer_lx_base"]],
        },
        {"type": "hbm", "memId": [-1], "startAddr": [_hbm_dataop_addr(12288)]},
    ]
    assert load0["labeledDs_"][0]["PieceInfo"][-1]["PlacementInfo"] == [
        {
            "type": "lx",
            "memId": [31],
            "startAddr": [cons_meta["kv_repack_consumer_lx_base"]],
        },
        {"type": "hbm", "memId": [-1], "startAddr": [_hbm_dataop_addr(12288)]},
    ]
    assert load0["labeledDs_"][1]["PieceInfo"][-1]["PlacementInfo"] == [
        {
            "type": "lx",
            "memId": [31],
            "startAddr": [cons_meta["kv_repack_consumer_lx_base"]],
        },
    ]
    assert load1["labeledDs_"][0]["PieceInfo"][0]["PlacementInfo"][1] == {
        "type": "hbm",
        "memId": [-1],
        "startAddr": [_hbm_dataop_addr(12416)],
    }
    assert nop_name == "2_nop_kv_repack_broadcast_barrier_Tensor0_idx1_tile1"
    assert nop["coreIdsUsed_"] == list(range(32))
    assert nop["op"] == {"name": "nop"}


def test_flash_kv_repack_broadcast_pair_hbm_direct_load_can_omit_ifn_transfer():
    sdscs = _fake_flash_layout_xform_kv_repack_sdscs()

    result = rz.build_flash_attention_kv_repack_broadcast_pair_artifacts(
        sdscs,
        tile_index=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        hbm_direct_load=True,
        include_input_fetch_transfer=False,
    )

    assert result is not None
    cons_name = "mixed_flash_kv_repack_broadcast_pair_1_input1_consumer"
    consumer = result["artifacts"][0][cons_name]
    cons_meta = consumer["flashAttentionPipeline_"]
    assert cons_meta["kv_repack_hbm_direct_load"] is True
    assert cons_meta["kv_repack_input_fetch_transfer"] is False
    assert consumer["opFuncsUsed_"] == ["STCDPOpHBM", "STCDPOpHBM", "nop"]
    compute_dl = next(iter(consumer["dscs_"][0].values()))
    assert not rz._has_input_fetch_neighbor_transfer(compute_dl, 1)


def test_flash_kv_repack_broadcast_pair_hbm_staged_keeps_hbm_pinned_consumer():
    sdscs = _fake_flash_layout_xform_kv_repack_sdscs()

    result = rz.build_flash_attention_kv_repack_broadcast_pair_artifacts(
        sdscs,
        tile_index=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        hbm_staged=True,
        consumer_core_state_init=False,
        consumer_ds_type="INPUT",
        consumer_lx_alloc_style="canonical_loop",
    )

    assert result is not None
    assert result["replacements"] == {
        "2_batchmatmul": "mixed_flash_kv_repack_broadcast_pair_1_input1_consumer",
    }
    assert len(result["artifacts"]) == 1
    cons_name = "mixed_flash_kv_repack_broadcast_pair_1_input1_consumer"
    consumer = result["artifacts"][0][cons_name]
    cons_meta = consumer["flashAttentionPipeline_"]
    assert cons_meta["kv_repack_producer_sidecar"] is None
    assert cons_meta["kv_repack_hbm_source"] is True
    assert cons_meta["kv_repack_hbm_direct_load"] is False
    assert cons_meta["kv_repack_hbm_staged"] is True
    assert cons_meta["kv_repack_broadcast_group_count"] == 0
    assert consumer["opFuncsUsed_"] == ["nop"]
    assert len(consumer["datadscs_"]) == 1
    assert consumer["coreIdToDscSchedule"]["0"] == [
        [0, -1, 0, 1],
        [-1, 0, 1, 0],
    ]
    assert consumer["coreIdToDscSchedule"]["31"] == [
        [0, -1, 0, 1],
        [-1, 0, 1, 0],
    ]

    compute_dl = next(iter(consumer["dscs_"][0].values()))
    consumer_lds = rz._lds_by_idx(compute_dl, 1)
    assert consumer_lds["dsType_"] == "KERNEL"
    assert "hbm" in consumer_lds["memOrg_"]
    assert consumer_lds["memOrg_"]["hbm"]["isPresent"] == 1
    assert "coreStateInit_" not in consumer_lds
    assert not rz._has_input_fetch_neighbor_transfer(compute_dl, 1)
    alloc_node = next(
        node
        for node in compute_dl["scheduleTree_"]
        if node.get("nodeType_") == "allocate" and node.get("ldsIdx_") == 1
    )
    assert alloc_node["component_"] == "hbm"
    assert alloc_node["name_"] == "allocate-Tensor1_hbm"


def test_flash_kv_repack_broadcast_pair_can_omit_consumer_core_state_init():
    sdscs = _fake_flash_layout_xform_kv_repack_sdscs()

    result = rz.build_flash_attention_kv_repack_broadcast_pair_artifacts(
        sdscs,
        tile_index=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        hbm_direct_load=True,
        consumer_core_state_init=False,
    )

    assert result is not None
    cons_name = "mixed_flash_kv_repack_broadcast_pair_1_input1_consumer"
    consumer = result["artifacts"][0][cons_name]
    cons_meta = consumer["flashAttentionPipeline_"]
    assert cons_meta["kv_repack_hbm_direct_load"] is True
    assert cons_meta["kv_repack_consumer_core_state_init"] is False
    compute_dl = next(iter(consumer["dscs_"][0].values()))
    consumer_lds = rz._lds_by_idx(compute_dl, 1)
    assert "coreStateInit_" not in consumer_lds
    assert consumer_lds["hbmSize_"] == 0
    assert consumer_lds["memOrg_"]["lx"]["allocateNode_"] == "allocate-Tensor1_lx"


def test_flash_kv_repack_broadcast_pair_can_override_consumer_ds_type():
    sdscs = _fake_flash_layout_xform_kv_repack_sdscs()

    result = rz.build_flash_attention_kv_repack_broadcast_pair_artifacts(
        sdscs,
        tile_index=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        hbm_direct_load=True,
        consumer_ds_type="INPUT",
    )

    assert result is not None
    cons_name = "mixed_flash_kv_repack_broadcast_pair_1_input1_consumer"
    consumer = result["artifacts"][0][cons_name]
    cons_meta = consumer["flashAttentionPipeline_"]
    assert cons_meta["kv_repack_hbm_direct_load"] is True
    assert cons_meta["kv_repack_consumer_ds_type"] == "INPUT"
    compute_dl = next(iter(consumer["dscs_"][0].values()))
    consumer_lds = rz._lds_by_idx(compute_dl, 1)
    assert consumer_lds["dsType_"] == "INPUT"


def test_flash_kv_repack_broadcast_pair_can_retarget_consumer_lx_alloc_style():
    sdscs = _fake_flash_layout_xform_kv_repack_sdscs()
    consumer_dl = rz._dl_op(sdscs[2])
    consumer_dl["scheduleTree_"].append(
        {
            "nodeType_": "loop",
            "name_": "loop_ds0_ds1_in",
            "next_": ["lx_below_schedule"],
        }
    )

    result = rz.build_flash_attention_kv_repack_broadcast_pair_artifacts(
        sdscs,
        tile_index=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        hbm_direct_load=True,
        consumer_lx_alloc_style="canonical_loop",
    )

    assert result is not None
    cons_name = "mixed_flash_kv_repack_broadcast_pair_1_input1_consumer"
    consumer = result["artifacts"][0][cons_name]
    cons_meta = consumer["flashAttentionPipeline_"]
    assert cons_meta["kv_repack_consumer_lx_alloc_style"] == "canonical_loop"
    compute_dl = next(iter(consumer["dscs_"][0].values()))
    consumer_lds = rz._lds_by_idx(compute_dl, 1)
    assert consumer_lds["memOrg_"]["lx"]["allocateNode_"] == "allocate_lds1_lx"
    alloc_node = next(
        node
        for node in compute_dl["scheduleTree_"]
        if node.get("nodeType_") == "allocate" and node.get("ldsIdx_") == 1
    )
    loop_node = next(
        node
        for node in compute_dl["scheduleTree_"]
        if node.get("name_") == "loop_ds0_ds1_in"
    )
    assert alloc_node["name_"] == "allocate_lds1_lx"
    assert alloc_node["prev_"] == "loop_ds0_ds1_in"
    assert alloc_node["numBuffers_"] == 2
    assert loop_node["next_"] == ["allocate_lds1_lx", "lx_below_schedule"]


def test_flash_kv_repack_broadcast_copyback_inserts_before_original_consumer():
    sdscs = _fake_flash_layout_xform_kv_repack_sdscs()

    result = rz.build_flash_attention_kv_repack_broadcast_copyback_artifacts(
        sdscs,
        tile_index=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
    )

    assert result is not None
    assert result["replacements"] == {
        "1_ReStickifyOpHBM": "mixed_flash_kv_repack_broadcast_copyback_1_input1_producer",
    }
    assert result["insertions_before"] == {
        "2_batchmatmul": [
            "mixed_flash_kv_repack_broadcast_copyback_1_input1_copyback"
        ],
    }
    copyback = result["artifacts"][1][
        "mixed_flash_kv_repack_broadcast_copyback_1_input1_copyback"
    ]
    meta = copyback["flashAttentionPipeline_"]
    assert meta["kv_repack_broadcast_role"] == "copyback"
    assert meta["kv_repack_copyback_readback_core"] == 31
    assert meta["kv_repack_copyback_original_consumer"] == "2_batchmatmul"
    assert meta["kv_repack_copyback_replaces_consumer"] is False
    assert meta["kv_repack_copyback_inserts_before_consumer"] is True
    assert len(copyback["dscs_"]) == 1
    assert copyback["opFuncsUsed_"] == [
        "STCDPOpLx",
        "STCDPOpHBM",
        "STCDPOpHBM",
        "nop",
    ]
    assert copyback["coreIdToDscSchedule"]["31"] == [
        [0, -1, 0, 1],
        [1, -1, 1, 1],
        [2, -1, 1, 1],
        [3, -1, 1, 1],
        [-1, 0, 1, 0],
    ]
    assert copyback["coreIdToDscSchedule"]["0"] == [
        [0, -1, 0, 1],
        [3, -1, 1, 1],
        [-1, 0, 1, 0],
    ]

    stcdp_name, stcdp = next(iter(copyback["datadscs_"][0].items()))
    assert stcdp_name == "0_STCDPOpLx_kv_repack_copyback_Tensor0_idx1_tile1"
    assert stcdp["op"] == {"name": "STCDPOpLx"}

    hbm_name, hbm = next(iter(copyback["datadscs_"][1].items()))
    assert (
        hbm_name
        == "1_STCDPOpHBM_kv_repack_copyback_Tensor0_idx1_tile1_piece0_core31"
    )
    assert hbm["coreIdsUsed_"] == [31]
    assert hbm["op"]["name"] == "STCDPOpHBM"
    assert hbm["op"]["coreIDtoANInfo"]["31"]["isAnalyticalMode"] == 0
    assert sorted(hbm["op"]["coreIDtoANInfo"].keys(), key=int) == ["31"]
    in_pieces = hbm["labeledDs_"][0]["PieceInfo"]
    out_pieces = hbm["labeledDs_"][1]["PieceInfo"]
    assert len(in_pieces) == 1
    assert len(out_pieces) == 1
    assert in_pieces[0]["PlacementInfo"] == [
        {
            "type": "lx",
            "memId": [31],
            "startAddr": [meta["kv_repack_consumer_lx_base"]],
        },
    ]
    assert out_pieces[0]["dimToSize_"]["x_"] == 1
    assert out_pieces[0]["dimToStartCordinate"]["x_"] == 0
    assert out_pieces[0]["PlacementInfo"] == [
        {
            "type": "lx",
            "memId": [31],
            "startAddr": [meta["kv_repack_consumer_lx_base"]],
        },
        {"type": "hbm", "memId": [-1], "startAddr": [_hbm_dataop_addr(12288)]},
    ]
    hbm1_name, hbm1 = next(iter(copyback["datadscs_"][2].items()))
    assert (
        hbm1_name
        == "2_STCDPOpHBM_kv_repack_copyback_Tensor0_idx1_tile1_piece1_core31"
    )
    out1_pieces = hbm1["labeledDs_"][1]["PieceInfo"]
    assert len(out1_pieces) == 1
    assert out1_pieces[0]["dimToStartCordinate"]["x_"] == 1
    assert out1_pieces[0]["PlacementInfo"][1] == {
        "type": "hbm",
        "memId": [-1],
        "startAddr": [_hbm_dataop_addr(12416)],
    }
    nop_name, nop = next(iter(copyback["datadscs_"][3].items()))
    assert nop_name == "3_nop_kv_repack_copyback_barrier_Tensor0_idx1_tile1"
    assert nop["coreIdsUsed_"] == list(range(32))
    assert nop["op"] == {"name": "nop"}


def test_flash_kv_repack_copyback_hbm_roundtrip_keeps_original_producer():
    sdscs = _fake_flash_layout_xform_kv_repack_sdscs()

    result = rz.build_flash_attention_kv_repack_broadcast_copyback_artifacts(
        sdscs,
        tile_index=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        hbm_roundtrip=True,
    )

    assert result is not None
    assert result["replacements"] == {}
    assert len(result["artifacts"]) == 1
    assert result["insertions_before"] == {
        "2_batchmatmul": [
            "mixed_flash_kv_repack_broadcast_copyback_1_input1_copyback"
        ],
    }
    copyback = result["artifacts"][0][
        "mixed_flash_kv_repack_broadcast_copyback_1_input1_copyback"
    ]
    meta = copyback["flashAttentionPipeline_"]
    assert meta["kv_repack_broadcast_role"] == "copyback"
    assert meta["kv_repack_broadcast_group_count"] == 0
    assert meta["kv_repack_copyback_direct_source"] is False
    assert meta["kv_repack_copyback_hbm_roundtrip"] is True
    assert copyback["opFuncsUsed_"] == [
        "STCDPOpHBM",
        "STCDPOpHBM",
        "STCDPOpHBM",
        "STCDPOpHBM",
        "nop",
    ]
    assert copyback["coreIdToDscSchedule"]["0"] == [
        [0, -1, 0, 1],
        [1, -1, 1, 1],
        [4, -1, 1, 1],
        [-1, 0, 1, 0],
    ]
    assert copyback["coreIdToDscSchedule"]["1"] == [
        [2, -1, 0, 1],
        [3, -1, 1, 1],
        [4, -1, 1, 1],
        [-1, 0, 1, 0],
    ]
    assert copyback["coreIdToDscSchedule"]["31"] == [
        [4, -1, 0, 1],
        [-1, 0, 1, 0],
    ]

    load0_name, load0 = next(iter(copyback["datadscs_"][0].items()))
    assert (
        load0_name
        == "0_STCDPOpHBM_kv_repack_copyback_"
        "Tensor0_idx1_tile1_roundtrip_source_piece0_load"
    )
    assert load0["coreIdsUsed_"] == [0]
    assert load0["op"]["coreIDtoANInfo"] == {
        "0": {
            "isAnalyticalMode": 0,
            "inpPieceOrder": ["p1"],
            "outPieceOrder": ["p1"],
        }
    }
    piece0 = load0["labeledDs_"][0]["PieceInfo"][0]
    assert piece0["dimToStartCordinate"]["x_"] == 0
    assert piece0["PlacementInfo"] == [
        {
            "type": "lx",
            "memId": [0],
            "startAddr": [meta["kv_repack_source_lx_base"]],
        },
        {"type": "hbm", "memId": [-1], "startAddr": [_hbm_dataop_addr(12288)]},
    ]
    assert load0["labeledDs_"][1]["PieceInfo"][0]["PlacementInfo"] == [
        {
            "type": "lx",
            "memId": [0],
            "startAddr": [meta["kv_repack_source_lx_base"]],
        },
    ]

    store0_name, store0 = next(iter(copyback["datadscs_"][1].items()))
    assert (
        store0_name
        == "1_STCDPOpHBM_kv_repack_copyback_"
        "Tensor0_idx1_tile1_roundtrip_source_piece0_store"
    )
    assert store0["labeledDs_"][0]["PieceInfo"][0]["PlacementInfo"] == [
        {
            "type": "lx",
            "memId": [0],
            "startAddr": [meta["kv_repack_source_lx_base"]],
        },
    ]
    assert store0["labeledDs_"][1]["PieceInfo"][0]["PlacementInfo"] == piece0[
        "PlacementInfo"
    ]

    load1_name, load1 = next(iter(copyback["datadscs_"][2].items()))
    assert (
        load1_name
        == "2_STCDPOpHBM_kv_repack_copyback_"
        "Tensor0_idx1_tile1_roundtrip_source_piece1_load"
    )
    piece1 = load1["labeledDs_"][0]["PieceInfo"][0]
    assert load1["coreIdsUsed_"] == [1]
    assert piece1["dimToStartCordinate"]["x_"] == 1
    assert piece1["PlacementInfo"] == [
        {
            "type": "lx",
            "memId": [1],
            "startAddr": [meta["kv_repack_source_lx_base"]],
        },
        {"type": "hbm", "memId": [-1], "startAddr": [_hbm_dataop_addr(12416)]},
    ]
    store1_name, store1 = next(iter(copyback["datadscs_"][3].items()))
    assert (
        store1_name
        == "3_STCDPOpHBM_kv_repack_copyback_"
        "Tensor0_idx1_tile1_roundtrip_source_piece1_store"
    )
    assert store1["labeledDs_"][1]["PieceInfo"][0]["PlacementInfo"] == piece1[
        "PlacementInfo"
    ]


def test_flash_kv_repack_copyback_hbm_source_fanout_loads_before_fanout():
    sdscs = _fake_flash_layout_xform_kv_repack_sdscs()

    result = rz.build_flash_attention_kv_repack_broadcast_copyback_artifacts(
        sdscs,
        tile_index=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        hbm_source_fanout=True,
    )

    assert result is not None
    assert result["replacements"] == {}
    assert len(result["artifacts"]) == 1
    assert result["insertions_before"] == {
        "2_batchmatmul": [
            "mixed_flash_kv_repack_broadcast_copyback_1_input1_copyback"
        ],
    }
    copyback = result["artifacts"][0][
        "mixed_flash_kv_repack_broadcast_copyback_1_input1_copyback"
    ]
    meta = copyback["flashAttentionPipeline_"]
    assert meta["kv_repack_broadcast_role"] == "copyback"
    assert meta["kv_repack_broadcast_group_count"] == 1
    assert meta["kv_repack_copyback_direct_source"] is False
    assert meta["kv_repack_copyback_hbm_roundtrip"] is False
    assert meta["kv_repack_copyback_hbm_source_fanout"] is True
    assert copyback["opFuncsUsed_"] == [
        "STCDPOpHBM",
        "STCDPOpHBM",
        "STCDPOpLx",
        "STCDPOpHBM",
        "STCDPOpHBM",
        "nop",
    ]
    assert copyback["coreIdToDscSchedule"]["0"] == [
        [0, -1, 0, 1],
        [2, -1, 1, 1],
        [5, -1, 1, 1],
        [-1, 0, 1, 0],
    ]
    assert copyback["coreIdToDscSchedule"]["1"] == [
        [1, -1, 0, 1],
        [2, -1, 1, 1],
        [5, -1, 1, 1],
        [-1, 0, 1, 0],
    ]
    assert copyback["coreIdToDscSchedule"]["31"] == [
        [2, -1, 0, 1],
        [3, -1, 1, 1],
        [4, -1, 1, 1],
        [5, -1, 1, 1],
        [-1, 0, 1, 0],
    ]

    load0_name, load0 = next(iter(copyback["datadscs_"][0].items()))
    load1_name, load1 = next(iter(copyback["datadscs_"][1].items()))
    fanout_name, fanout = next(iter(copyback["datadscs_"][2].items()))
    store0_name, store0 = next(iter(copyback["datadscs_"][3].items()))
    store1_name, store1 = next(iter(copyback["datadscs_"][4].items()))
    nop_name, nop = next(iter(copyback["datadscs_"][5].items()))

    assert load0_name == (
        "0_STCDPOpHBM_kv_repack_copyback_"
        "Tensor0_idx1_tile1_hbm_source_piece0_load"
    )
    assert load1_name == (
        "1_STCDPOpHBM_kv_repack_copyback_"
        "Tensor0_idx1_tile1_hbm_source_piece1_load"
    )
    assert fanout_name == "2_STCDPOpLx_kv_repack_copyback_Tensor0_idx1_tile1"
    assert store0_name == (
        "3_STCDPOpHBM_kv_repack_copyback_Tensor0_idx1_tile1_piece0_core31"
    )
    assert store1_name == (
        "4_STCDPOpHBM_kv_repack_copyback_Tensor0_idx1_tile1_piece1_core31"
    )
    assert nop_name == "5_nop_kv_repack_copyback_barrier_Tensor0_idx1_tile1"
    assert load0["coreIdsUsed_"] == [0]
    assert load1["coreIdsUsed_"] == [1]
    assert fanout["op"] == {"name": "STCDPOpLx"}
    assert store0["coreIdsUsed_"] == [31]
    assert store1["coreIdsUsed_"] == [31]
    assert nop["coreIdsUsed_"] == list(range(32))
    assert load0["labeledDs_"][0]["PieceInfo"][0]["PlacementInfo"] == [
        {
            "type": "lx",
            "memId": [0],
            "startAddr": [meta["kv_repack_source_lx_base"]],
        },
        {"type": "hbm", "memId": [-1], "startAddr": [_hbm_dataop_addr(12288)]},
    ]
    assert load0["labeledDs_"][1]["PieceInfo"][0]["PlacementInfo"] == [
        {
            "type": "lx",
            "memId": [0],
            "startAddr": [meta["kv_repack_source_lx_base"]],
        },
    ]
    assert store0["labeledDs_"][0]["PieceInfo"][0]["PlacementInfo"] == [
        {
            "type": "lx",
            "memId": [31],
            "startAddr": [meta["kv_repack_consumer_lx_base"]],
        },
    ]
    assert store0["labeledDs_"][1]["PieceInfo"][0]["PlacementInfo"][1] == {
        "type": "hbm",
        "memId": [-1],
        "startAddr": [_hbm_dataop_addr(12288)],
    }


def test_flash_kv_repack_copyback_hbm_direct_load_reads_consumer_lx():
    sdscs = _fake_flash_layout_xform_kv_repack_sdscs()

    result = rz.build_flash_attention_kv_repack_broadcast_copyback_artifacts(
        sdscs,
        tile_index=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        hbm_direct_load=True,
    )

    assert result is not None
    assert result["replacements"] == {}
    assert len(result["artifacts"]) == 1
    assert result["insertions_before"] == {
        "2_batchmatmul": [
            "mixed_flash_kv_repack_broadcast_copyback_1_input1_copyback"
        ],
    }
    copyback = result["artifacts"][0][
        "mixed_flash_kv_repack_broadcast_copyback_1_input1_copyback"
    ]
    meta = copyback["flashAttentionPipeline_"]
    assert meta["kv_repack_broadcast_role"] == "copyback"
    assert meta["kv_repack_broadcast_group_count"] == 0
    assert meta["kv_repack_copyback_direct_source"] is False
    assert meta["kv_repack_copyback_hbm_roundtrip"] is False
    assert meta["kv_repack_copyback_hbm_source_fanout"] is False
    assert meta["kv_repack_copyback_hbm_direct_load"] is True
    assert copyback["opFuncsUsed_"] == [
        "STCDPOpHBM",
        "STCDPOpHBM",
        "STCDPOpHBM",
        "STCDPOpHBM",
        "nop",
    ]
    assert copyback["coreIdToDscSchedule"]["0"] == [
        [0, -1, 0, 1],
        [1, -1, 1, 1],
        [4, -1, 1, 1],
        [-1, 0, 1, 0],
    ]
    assert copyback["coreIdToDscSchedule"]["31"] == [
        [0, -1, 0, 1],
        [1, -1, 1, 1],
        [2, -1, 1, 1],
        [3, -1, 1, 1],
        [4, -1, 1, 1],
        [-1, 0, 1, 0],
    ]

    load0_name, load0 = next(iter(copyback["datadscs_"][0].items()))
    load1_name, load1 = next(iter(copyback["datadscs_"][1].items()))
    store0_name, store0 = next(iter(copyback["datadscs_"][2].items()))
    store1_name, store1 = next(iter(copyback["datadscs_"][3].items()))
    nop_name, nop = next(iter(copyback["datadscs_"][4].items()))
    assert load0_name == (
        "0_STCDPOpHBM_kv_repack_copyback_"
        "Tensor0_idx1_tile1_hbm_direct_piece0_load"
    )
    assert load1_name == (
        "1_STCDPOpHBM_kv_repack_copyback_"
        "Tensor0_idx1_tile1_hbm_direct_piece1_load"
    )
    assert store0_name == (
        "2_STCDPOpHBM_kv_repack_copyback_Tensor0_idx1_tile1_piece0_core31"
    )
    assert store1_name == (
        "3_STCDPOpHBM_kv_repack_copyback_Tensor0_idx1_tile1_piece1_core31"
    )
    assert nop_name == "4_nop_kv_repack_copyback_barrier_Tensor0_idx1_tile1"
    assert load0["coreIdsUsed_"] == list(range(32))
    assert load1["coreIdsUsed_"] == list(range(32))
    assert store0["coreIdsUsed_"] == [31]
    assert store1["coreIdsUsed_"] == [31]
    assert nop["coreIdsUsed_"] == list(range(32))
    assert load0["labeledDs_"][0]["PieceInfo"][0]["PlacementInfo"] == [
        {
            "type": "lx",
            "memId": [0],
            "startAddr": [meta["kv_repack_consumer_lx_base"]],
        },
        {"type": "hbm", "memId": [-1], "startAddr": [_hbm_dataop_addr(12288)]},
    ]
    assert load0["labeledDs_"][0]["PieceInfo"][-1]["PlacementInfo"] == [
        {
            "type": "lx",
            "memId": [31],
            "startAddr": [meta["kv_repack_consumer_lx_base"]],
        },
        {"type": "hbm", "memId": [-1], "startAddr": [_hbm_dataop_addr(12288)]},
    ]
    assert store0["labeledDs_"][0]["PieceInfo"][0]["PlacementInfo"] == [
        {
            "type": "lx",
            "memId": [31],
            "startAddr": [meta["kv_repack_consumer_lx_base"]],
        },
    ]
    assert store0["labeledDs_"][1]["PieceInfo"][0]["PlacementInfo"][1] == {
        "type": "hbm",
        "memId": [-1],
        "startAddr": [_hbm_dataop_addr(12288)],
    }


def test_flash_kv_repack_copyback_hbm_direct_load_can_readback_core0():
    sdscs = _fake_flash_layout_xform_kv_repack_sdscs()

    result = rz.build_flash_attention_kv_repack_broadcast_copyback_artifacts(
        sdscs,
        tile_index=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        hbm_direct_load=True,
        readback_core=0,
    )

    assert result is not None
    copyback = result["artifacts"][0][
        "mixed_flash_kv_repack_broadcast_copyback_1_input1_copyback"
    ]
    meta = copyback["flashAttentionPipeline_"]
    assert meta["kv_repack_copyback_hbm_direct_load"] is True
    assert meta["kv_repack_copyback_readback_core"] == 0
    assert copyback["coreIdToDscSchedule"]["0"] == [
        [0, -1, 0, 1],
        [1, -1, 1, 1],
        [2, -1, 1, 1],
        [3, -1, 1, 1],
        [4, -1, 1, 1],
        [-1, 0, 1, 0],
    ]
    assert copyback["coreIdToDscSchedule"]["31"] == [
        [0, -1, 0, 1],
        [1, -1, 1, 1],
        [4, -1, 1, 1],
        [-1, 0, 1, 0],
    ]
    store0_name, store0 = next(iter(copyback["datadscs_"][2].items()))
    assert store0_name == (
        "2_STCDPOpHBM_kv_repack_copyback_Tensor0_idx1_tile1_piece0_core0"
    )
    assert store0["coreIdsUsed_"] == [0]
    assert store0["labeledDs_"][0]["PieceInfo"][0]["PlacementInfo"] == [
        {
            "type": "lx",
            "memId": [0],
            "startAddr": [meta["kv_repack_consumer_lx_base"]],
        },
    ]


def test_flash_kv_repack_copyback_hbm_load_only_skips_hbm_store():
    sdscs = _fake_flash_layout_xform_kv_repack_sdscs()

    result = rz.build_flash_attention_kv_repack_broadcast_copyback_artifacts(
        sdscs,
        tile_index=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        hbm_roundtrip=True,
        hbm_roundtrip_load_only=True,
    )

    assert result is not None
    assert result["replacements"] == {}
    assert len(result["artifacts"]) == 1
    copyback = result["artifacts"][0][
        "mixed_flash_kv_repack_broadcast_copyback_1_input1_copyback"
    ]
    meta = copyback["flashAttentionPipeline_"]
    assert meta["kv_repack_copyback_hbm_roundtrip"] is True
    assert meta["kv_repack_copyback_hbm_roundtrip_load_only"] is True
    assert copyback["opFuncsUsed_"] == ["STCDPOpHBM", "STCDPOpHBM", "nop"]
    assert copyback["coreIdToDscSchedule"]["0"] == [
        [0, -1, 0, 1],
        [2, -1, 1, 1],
        [-1, 0, 1, 0],
    ]
    assert copyback["coreIdToDscSchedule"]["1"] == [
        [1, -1, 0, 1],
        [2, -1, 1, 1],
        [-1, 0, 1, 0],
    ]
    assert copyback["coreIdToDscSchedule"]["31"] == [
        [2, -1, 0, 1],
        [-1, 0, 1, 0],
    ]

    assert len(copyback["datadscs_"]) == 3
    load0_name, load0 = next(iter(copyback["datadscs_"][0].items()))
    load1_name, load1 = next(iter(copyback["datadscs_"][1].items()))
    assert load0_name.endswith("_roundtrip_source_piece0_load")
    assert load1_name.endswith("_roundtrip_source_piece1_load")
    assert "store" not in "".join(
        next(iter(dataop)) for dataop in copyback["datadscs_"]
    )
    assert load0["labeledDs_"][0]["PieceInfo"][0]["PlacementInfo"] == [
        {
            "type": "lx",
            "memId": [0],
            "startAddr": [meta["kv_repack_source_lx_base"]],
        },
        {"type": "hbm", "memId": [-1], "startAddr": [_hbm_dataop_addr(12288)]},
    ]
    assert load0["labeledDs_"][1]["PieceInfo"][0]["PlacementInfo"] == [
        {
            "type": "lx",
            "memId": [0],
            "startAddr": [meta["kv_repack_source_lx_base"]],
        },
    ]
    assert load1["labeledDs_"][1]["PieceInfo"][0]["PlacementInfo"] == [
        {
            "type": "lx",
            "memId": [1],
            "startAddr": [meta["kv_repack_source_lx_base"]],
        },
    ]


def test_flash_kv_repack_copyback_hbm_barrier_only_skips_hbm_dataops():
    sdscs = _fake_flash_layout_xform_kv_repack_sdscs()

    result = rz.build_flash_attention_kv_repack_broadcast_copyback_artifacts(
        sdscs,
        tile_index=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        hbm_roundtrip=True,
        hbm_roundtrip_barrier_only=True,
    )

    assert result is not None
    assert result["replacements"] == {}
    assert len(result["artifacts"]) == 1
    copyback = result["artifacts"][0][
        "mixed_flash_kv_repack_broadcast_copyback_1_input1_copyback"
    ]
    meta = copyback["flashAttentionPipeline_"]
    assert meta["kv_repack_copyback_hbm_roundtrip"] is True
    assert meta["kv_repack_copyback_hbm_roundtrip_load_only"] is False
    assert meta["kv_repack_copyback_hbm_roundtrip_barrier_only"] is True
    assert copyback["opFuncsUsed_"] == ["nop"]
    assert len(copyback["datadscs_"]) == 1
    nop_name, nop = next(iter(copyback["datadscs_"][0].items()))
    assert nop_name == "0_nop_kv_repack_copyback_barrier_Tensor0_idx1_tile1"
    assert nop["op"] == {"name": "nop"}
    assert copyback["coreIdToDscSchedule"]["0"] == [
        [0, -1, 0, 1],
        [-1, 0, 1, 0],
    ]
    assert copyback["coreIdToDscSchedule"]["1"] == [
        [0, -1, 0, 1],
        [-1, 0, 1, 0],
    ]
    assert copyback["coreIdToDscSchedule"]["31"] == [
        [0, -1, 0, 1],
        [-1, 0, 1, 0],
    ]


def test_flash_kv_repack_copyback_can_replace_original_consumer():
    sdscs = _fake_flash_layout_xform_kv_repack_sdscs()

    result = rz.build_flash_attention_kv_repack_broadcast_copyback_artifacts(
        sdscs,
        tile_index=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        hbm_roundtrip=True,
        hbm_roundtrip_barrier_only=True,
        replace_consumer=True,
    )

    assert result is not None
    copyback_name = "mixed_flash_kv_repack_broadcast_copyback_1_input1_copyback"
    assert result["replacements"] == {"2_batchmatmul": copyback_name}
    assert result["insertions_before"] == {}
    copyback = result["artifacts"][0][copyback_name]
    meta = copyback["flashAttentionPipeline_"]
    assert meta["kv_repack_copyback_replaces_consumer"] is True
    assert meta["kv_repack_copyback_inserts_before_consumer"] is False
    assert copyback["opFuncsUsed_"] == ["nop"]
    assert copyback["coreIdToDscSchedule"]["0"] == [
        [0, -1, 0, 1],
        [-1, 0, 1, 0],
    ]


def test_flash_kv_repack_copyback_compute_only_wraps_original_consumer():
    sdscs = _fake_flash_layout_xform_kv_repack_sdscs()

    result = rz.build_flash_attention_kv_repack_broadcast_copyback_artifacts(
        sdscs,
        tile_index=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        hbm_roundtrip=True,
        replace_consumer=True,
        compute_only=True,
    )

    assert result is not None
    copyback_name = "mixed_flash_kv_repack_broadcast_copyback_1_input1_copyback"
    assert result["replacements"] == {"2_batchmatmul": copyback_name}
    copyback = result["artifacts"][0][copyback_name]
    meta = copyback["flashAttentionPipeline_"]
    assert meta["kv_repack_copyback_compute_only"] is True
    assert meta["compute_tile_count"] == 1
    assert copyback["datadscs_"] == []
    assert copyback["opFuncsUsed_"] == []
    assert len(copyback["dscs_"]) == 1
    assert copyback["coreIdToDscSchedule"]["0"] == [[-1, 0, 0, 0]]
    assert copyback["coreIdToDscSchedule"]["31"] == [[-1, 0, 0, 0]]


def test_flash_kv_repack_copyback_compute_only_can_preserve_consumer_name():
    sdscs = _fake_flash_layout_xform_kv_repack_sdscs()

    result = rz.build_flash_attention_kv_repack_broadcast_copyback_artifacts(
        sdscs,
        tile_index=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        hbm_roundtrip=True,
        compute_only=True,
        preserve_consumer_name=True,
    )

    assert result is not None
    assert result["replacements"] == {}
    assert result["insertions_before"] == {}
    copyback = result["artifacts"][0]["2_batchmatmul"]
    meta = copyback["flashAttentionPipeline_"]
    assert meta["kv_repack_copyback_compute_only"] is True
    assert meta["kv_repack_copyback_preserve_consumer_name"] is True
    assert copyback["datadscs_"] == []
    assert copyback["opFuncsUsed_"] == []


def test_flash_kv_repack_copyback_exact_clone_replaces_original_consumer():
    sdscs = _fake_flash_layout_xform_kv_repack_sdscs()

    result = rz.build_flash_attention_kv_repack_broadcast_copyback_artifacts(
        sdscs,
        tile_index=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        hbm_roundtrip=True,
        exact_clone=True,
    )

    assert result is not None
    clone_name = "mixed_flash_kv_repack_broadcast_copyback_1_input1_exact_clone"
    assert result["replacements"] == {"2_batchmatmul": clone_name}
    assert result["insertions_before"] == {}
    assert result["artifacts"] == [{clone_name: rz._body(sdscs[2])}]
    clone = result["artifacts"][0][clone_name]
    assert "flashAttentionPipeline_" not in clone
    assert "datadscs_" not in clone
    assert "opFuncsUsed_" not in clone
    assert clone["coreIdToDscSchedule"] == rz._body(sdscs[2])["coreIdToDscSchedule"]


def test_flash_kv_repack_copyback_exact_clone_can_preserve_consumer_name():
    sdscs = _fake_flash_layout_xform_kv_repack_sdscs()

    result = rz.build_flash_attention_kv_repack_broadcast_copyback_artifacts(
        sdscs,
        tile_index=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        hbm_roundtrip=True,
        exact_clone=True,
        preserve_consumer_name=True,
    )

    assert result is not None
    assert result["replacements"] == {}
    assert result["insertions_before"] == {}
    assert result["artifacts"] == [{"2_batchmatmul": rz._body(sdscs[2])}]


def test_flash_kv_repack_copyback_data_only_omits_copied_compute():
    sdscs = _fake_flash_layout_xform_kv_repack_sdscs()

    result = rz.build_flash_attention_kv_repack_broadcast_copyback_artifacts(
        sdscs,
        tile_index=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        hbm_roundtrip=True,
        hbm_roundtrip_barrier_only=True,
        data_only=True,
    )

    assert result is not None
    copyback = result["artifacts"][0][
        "mixed_flash_kv_repack_broadcast_copyback_1_input1_copyback"
    ]
    meta = copyback["flashAttentionPipeline_"]
    assert meta["kv_repack_copyback_data_only"] is True
    assert meta["compute_tile_count"] == 0
    assert copyback["dscs_"] == []
    assert copyback["opFuncsUsed_"] == ["nop"]
    assert "coreIdToDsc_" not in copyback
    assert copyback["coreIdToDscSchedule"]["0"] == [[0, -1, 0, 0]]
    assert copyback["coreIdToDscSchedule"]["31"] == [[0, -1, 0, 0]]


def test_flash_kv_repack_copyback_hbm_load_data_only_keeps_hbm_loads():
    sdscs = _fake_flash_layout_xform_kv_repack_sdscs()

    result = rz.build_flash_attention_kv_repack_broadcast_copyback_artifacts(
        sdscs,
        tile_index=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        hbm_roundtrip=True,
        hbm_roundtrip_load_only=True,
        data_only=True,
    )

    assert result is not None
    copyback = result["artifacts"][0][
        "mixed_flash_kv_repack_broadcast_copyback_1_input1_copyback"
    ]
    meta = copyback["flashAttentionPipeline_"]
    assert meta["kv_repack_copyback_data_only"] is True
    assert copyback["dscs_"] == []
    assert copyback["opFuncsUsed_"] == ["STCDPOpHBM", "STCDPOpHBM", "nop"]
    assert copyback["coreIdToDscSchedule"]["0"] == [
        [0, -1, 0, 1],
        [2, -1, 1, 0],
    ]
    assert copyback["coreIdToDscSchedule"]["1"] == [
        [1, -1, 0, 1],
        [2, -1, 1, 0],
    ]
    assert copyback["coreIdToDscSchedule"]["31"] == [[2, -1, 0, 0]]


def test_flash_kv_repack_broadcast_pair_can_omit_ifn_transfer_marker():
    result = rz.build_flash_attention_kv_repack_broadcast_pair_artifacts(
        _fake_flash_layout_xform_kv_repack_sdscs(),
        tile_index=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        include_input_fetch_transfer=False,
    )

    assert result is not None
    consumer = result["artifacts"][1][
        "mixed_flash_kv_repack_broadcast_pair_1_input1_consumer"
    ]
    cons_meta = consumer["flashAttentionPipeline_"]
    compute_dl = next(iter(consumer["dscs_"][0].values()))
    assert cons_meta["kv_repack_input_fetch_transfer"] is False
    assert not rz._has_input_fetch_neighbor_transfer(compute_dl, 1)


def test_flash_kv_repack_broadcast_pair_can_disable_subpiece_reuse():
    result = rz.build_flash_attention_kv_repack_broadcast_pair_artifacts(
        _fake_flash_layout_xform_kv_repack_sdscs(),
        tile_index=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        stcdp_subpiece_reuse=False,
    )

    assert result is not None
    consumer = result["artifacts"][1][
        "mixed_flash_kv_repack_broadcast_pair_1_input1_consumer"
    ]
    cons_meta = consumer["flashAttentionPipeline_"]
    dataop = next(iter(consumer["datadscs_"][0].values()))
    assert cons_meta["kv_repack_stcdp_subpiece_reuse"] is False
    assert dataop["op"] == {"name": "STCDPOpLx", "enSubPieceReuse": 0}


def test_flash_kv_repack_broadcast_pair_can_skip_producer_self_destinations():
    result = rz.build_flash_attention_kv_repack_broadcast_pair_artifacts(
        _fake_flash_layout_xform_kv_repack_sdscs(),
        tile_index=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        self_resident_source=True,
    )

    assert result is not None
    producer = result["artifacts"][0][
        "mixed_flash_kv_repack_broadcast_pair_1_input1_producer"
    ]
    consumer = result["artifacts"][1][
        "mixed_flash_kv_repack_broadcast_pair_1_input1_consumer"
    ]
    cons_meta = consumer["flashAttentionPipeline_"]
    assert cons_meta["kv_repack_self_resident_source"] is True
    assert cons_meta["kv_repack_source_lx_base"] == cons_meta["kv_repack_consumer_lx_base"]
    assert cons_meta["kv_repack_destination_piece_count"] == 62
    assert rz._lds_by_idx(rz._dl_op({"producer": producer}), 1)["hbmSize_"] == 0

    dataop = next(iter(consumer["datadscs_"][0].values()))
    src_ld, dst_ld = dataop["labeledDs_"]
    assert src_ld["PieceInfo"][0]["PlacementInfo"][0]["startAddr"] == [
        cons_meta["kv_repack_consumer_lx_base"]
    ]
    assert len(dst_ld["PieceInfo"]) == 62
    assert dst_ld["PieceInfo"][0]["PlacementInfo"][0]["memId"] == [0]
    assert dst_ld["PieceInfo"][0]["dimToStartCordinate"]["x_"] == 1
    assert dst_ld["PieceInfo"][1]["PlacementInfo"][0]["memId"] == [1]
    assert dst_ld["PieceInfo"][1]["dimToStartCordinate"]["x_"] == 0


def test_flash_kv_repack_broadcast_pair_can_force_multicast_mode():
    result = rz.build_flash_attention_kv_repack_broadcast_pair_artifacts(
        _fake_flash_layout_xform_kv_repack_sdscs(),
        tile_index=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        stcdp_force_mc_mode=3,
    )

    assert result is not None
    consumer = result["artifacts"][1][
        "mixed_flash_kv_repack_broadcast_pair_1_input1_consumer"
    ]
    cons_meta = consumer["flashAttentionPipeline_"]
    dataop = next(iter(consumer["datadscs_"][0].values()))
    assert cons_meta["kv_repack_stcdp_force_mc_mode"] == 3
    assert dataop["op"] == {
        "name": "STCDPOpLx",
        "forceModeMC": {"force": 1, "val": 3},
    }


def test_flash_kv_repack_broadcast_pair_can_split_broadcast_groups():
    result = rz.build_flash_attention_kv_repack_broadcast_pair_artifacts(
        _fake_flash_layout_xform_kv_repack_sdscs(),
        tile_index=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        broadcast_group_size=16,
    )

    assert result is not None
    consumer = result["artifacts"][1][
        "mixed_flash_kv_repack_broadcast_pair_1_input1_consumer"
    ]
    cons_meta = consumer["flashAttentionPipeline_"]
    assert cons_meta["kv_repack_broadcast_group_size"] == 16
    assert cons_meta["kv_repack_broadcast_group_count"] == 2
    assert len(consumer["datadscs_"]) == 2
    assert consumer["opFuncsUsed_"] == ["STCDPOpLx", "STCDPOpLx"]
    assert consumer["coreIdToDscSchedule"]["0"] == [
        [0, -1, 0, 1],
        [1, -1, 1, 1],
        [-1, 0, 1, 0],
    ]
    assert consumer["coreIdToDscSchedule"]["2"] == [
        [0, -1, 0, 1],
        [-1, 0, 1, 0],
    ]
    assert consumer["coreIdToDscSchedule"]["16"] == [
        [1, -1, 0, 1],
        [-1, 0, 1, 0],
    ]
    first_name, first = next(iter(consumer["datadscs_"][0].items()))
    second_name, second = next(iter(consumer["datadscs_"][1].items()))
    assert first_name.endswith("_group0")
    assert second_name.endswith("_group1")
    assert first["coreIdsUsed_"] == list(range(16))
    assert second["coreIdsUsed_"] == [0, 1, *range(16, 32)]
    assert len(first["labeledDs_"][1]["PieceInfo"]) == 32
    assert len(second["labeledDs_"][1]["PieceInfo"]) == 32
    assert first["labeledDs_"][1]["PieceInfo"][-1]["PlacementInfo"][0]["memId"] == [
        15
    ]
    assert second["labeledDs_"][1]["PieceInfo"][0]["PlacementInfo"][0]["memId"] == [
        16
    ]


def test_flash_kv_repack_broadcast_pair_rejections_are_not_double_prefixed():
    reasons = rz.flash_attention_kv_repack_broadcast_pair_rejection_reasons(
        _fake_flash_layout_xform_kv_repack_sdscs(),
        tile_index=0,
    )

    assert reasons == ["input1:no_latest_producer", "input2:not_consumer_input"]


def test_flash_layout_xform_pair_reports_dynamic_pointwise_region():
    result = rz.build_flash_attention_layout_xform_pair_tile_artifacts(
        _fake_flash_pipeline_sdscs(
            num_tiles=3,
            sdpa_layout_transform=True,
            size_overrides={"mb_": 65536},
        ),
        tile_index=1,
    )

    assert result is not None
    cons = result["artifacts"][1]["mixed_flash_layout_xform_pair_tile_1_consumer"]
    cons_meta = cons["flashAttentionPipeline_"]
    assert cons_meta["slice_bytes"] == 512 << 10
    assert result["pointwise_lx_region0"] == (
        rz.layout_xform_compose_pointwise_lx_base(cons_meta["slice_bytes"])
    )
    assert result["pointwise_lx_region0"] > (
        rz.LAYOUT_XFORM_COMPOSE_POINTWISE_LX_BASE
    )


def test_flash_layout_xform_pair_auto_selects_first_eligible_tile():
    result = rz.build_flash_attention_layout_xform_pair_tile_artifacts(
        _fake_flash_pipeline_sdscs(num_tiles=3, sdpa_layout_transform=True),
        tile_index=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
    )

    assert result is not None
    pred = result["artifacts"][0]["mixed_flash_layout_xform_pair_tile_1_predecessor"]
    cons = result["artifacts"][1]["mixed_flash_layout_xform_pair_tile_1_consumer"]
    assert result["replacements"] == {
        "0_batchmatmul": "mixed_flash_layout_xform_pair_tile_1_predecessor",
        "1_batchmatmul": "mixed_flash_layout_xform_pair_tile_1_consumer",
    }
    assert (
        rz.flash_attention_layout_xform_pair_rejection_reasons(
            _fake_flash_pipeline_sdscs(num_tiles=3, sdpa_layout_transform=True),
            tile_index=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        )
        == []
    )
    assert pred["flashAttentionPipeline_"]["tile_index"] == 1
    assert pred["flashAttentionPipeline_"]["requested_tile_index"] == (
        rz.LAYOUT_XFORM_PAIR_AUTO_TILE
    )
    assert cons["flashAttentionPipeline_"]["tile_index"] == 1
    assert cons["flashAttentionPipeline_"]["requested_tile_index"] == (
        rz.LAYOUT_XFORM_PAIR_AUTO_TILE
    )


def test_bundle_executes_layout_xform_pair_auto_gate():
    bundle, calls, saved = _load_bundle_with_stubs()
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["layout_xform"] == [rz.LAYOUT_XFORM_PAIR_AUTO_TILE]
            assert calls["pointwise"] == []
            with open(os.path.join(output_dir, "bundle.mlir")) as file:
                bundle_mlir = file.read()
            assert (
                "sdsc_mixed_flash_layout_xform_pair_tile_2_predecessor.json"
                in bundle_mlir
            )
            assert (
                "sdsc_mixed_flash_layout_xform_pair_tile_2_consumer.json"
                in bundle_mlir
            )

            consumer_path = os.path.join(
                output_dir,
                "sdsc_mixed_flash_layout_xform_pair_tile_2_consumer.json",
            )
            with open(consumer_path) as file:
                consumer = json.load(file)
            meta = consumer["mixed_flash_layout_xform_pair_tile_2_consumer"][
                "flashAttentionPipeline_"
            ]
            assert meta["tile_index"] == 2
            assert meta["requested_tile_index"] == rz.LAYOUT_XFORM_PAIR_AUTO_TILE
    finally:
        _restore_modules(saved)


def test_bundle_executes_layout_xform_pair_overlap_gate():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_overlap=True,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["layout_xform"] == [rz.LAYOUT_XFORM_PAIR_AUTO_TILE]
            assert calls["layout_xform_overlap"] == [True]
            with open(os.path.join(output_dir, "bundle.mlir")) as file:
                bundle_mlir = file.read()
            assert (
                "sdsc_mixed_flash_pipeline_tile_layout_xform_pair_2_predecessor.json"
                in bundle_mlir
            )
            assert (
                "sdsc_mixed_flash_pipeline_tile_layout_xform_pair_2_consumer.json"
                in bundle_mlir
            )

            consumer_path = os.path.join(
                output_dir,
                "sdsc_mixed_flash_pipeline_tile_layout_xform_pair_2_consumer.json",
            )
            with open(consumer_path) as file:
                consumer = json.load(file)
            meta = consumer[
                "mixed_flash_pipeline_tile_layout_xform_pair_2_consumer"
            ]["flashAttentionPipeline_"]
            assert meta["layout_xform_overlap_consumer"] is True
    finally:
        _restore_modules(saved)


def test_bundle_executes_layout_xform_lookahead_gate():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        layout_xform_lookahead_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"3_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["layout_xform_lookahead"] == [
                rz.LAYOUT_XFORM_PAIR_AUTO_TILE
            ]
            with open(os.path.join(output_dir, "bundle.mlir")) as file:
                bundle_mlir = file.read()
            assert (
                "sdsc_mixed_flash_pipeline_tile_layout_xform_lookahead_0_"
                "current_consumer.json"
            ) in bundle_mlir
            assert (
                "sdsc_mixed_flash_pipeline_tile_layout_xform_lookahead_0_"
                "future_consumer.json"
            ) in bundle_mlir
    finally:
        _restore_modules(saved)


def test_bundle_executes_layout_xform_hoist_gate_and_omits_future_producer():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        layout_xform_hoist_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["layout_xform_hoist"] == [
                rz.LAYOUT_XFORM_PAIR_AUTO_TILE
            ]
            with open(os.path.join(output_dir, "bundle.mlir")) as file:
                bundle_mlir = file.read()
            assert (
                "sdsc_mixed_flash_pipeline_tile_layout_xform_hoist_0_"
                "current_consumer.json"
            ) in bundle_mlir
            assert (
                "sdsc_mixed_flash_pipeline_tile_layout_xform_hoist_0_"
                "future_consumer.json"
            ) in bundle_mlir
            assert "sdsc_1_ReStickifyOpHBM.json" not in bundle_mlir
            assert os.path.exists(
                os.path.join(output_dir, "sdsc_1_ReStickifyOpHBM.json")
            )
    finally:
        _restore_modules(saved)


def test_bundle_executes_kv_repack_pair_gate():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_pair_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_pair"] == [rz.LAYOUT_XFORM_PAIR_AUTO_TILE]
            assert calls["kv_repack_pair_ifn_transfer"] == [True]
            assert calls["kv_repack_pair_subpiece_reuse"] == [True]
            assert calls["kv_repack_pair_group_size"] == [0]
            assert calls["kv_repack_pair_self_resident_source"] == [False]
            assert calls["kv_repack_pair_hbm_source"] == [False]
            assert calls["kv_repack_pair_hbm_direct_load"] == [False]
            assert calls["kv_repack_pair_hbm_staged"] == [False]
            assert calls["kv_repack_pair_consumer_core_state_init"] == [True]
            assert calls["kv_repack_pair_consumer_ds_type"] == [""]
            assert calls["kv_repack_pair_consumer_lx_alloc_style"] == [""]
            assert calls["kv_repack_pair_use_unicast"] == [-1]
            assert calls["kv_repack_pair_force_mc_mode"] == [-1]
            with open(os.path.join(output_dir, "bundle.mlir")) as file:
                bundle_mlir = file.read()
            assert (
                "sdsc_mixed_flash_kv_repack_broadcast_pair_1_input1_producer.json"
                in bundle_mlir
            )
            assert (
                "sdsc_mixed_flash_kv_repack_broadcast_pair_1_input1_consumer.json"
                in bundle_mlir
            )
    finally:
        _restore_modules(saved)


def test_bundle_can_combine_non_conflicting_kv_repack_and_layout_xform_pairs():
    bundle, calls, saved = _load_bundle_with_stubs(
        kv_repack_pair_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_pair_hbm_staged=True,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"1_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_pair"] == [rz.LAYOUT_XFORM_PAIR_AUTO_TILE]
            assert calls["layout_xform"] == [rz.LAYOUT_XFORM_PAIR_AUTO_TILE]
            with open(os.path.join(output_dir, "bundle.mlir")) as file:
                bundle_mlir = file.read()
            assert (
                "sdsc_mixed_flash_kv_repack_broadcast_pair_1_input1_consumer.json"
                in bundle_mlir
            )
            assert (
                "sdsc_mixed_flash_layout_xform_pair_tile_2_predecessor.json"
                in bundle_mlir
            )
            assert (
                "sdsc_mixed_flash_layout_xform_pair_tile_2_consumer.json"
                in bundle_mlir
            )
    finally:
        _restore_modules(saved)


def test_bundle_rebuilds_kv_repack_pair_after_pointwise_handoff():
    bundle, calls, saved = _load_bundle_with_stubs(
        pointwise_handoff=True,
        layout_xform_pair_tile=-1,
        kv_repack_pair_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_pair_hbm_staged=True,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["pointwise"] == [
                {
                    "score_scale_handoff": False,
                    "pointwise_region0": rz.LAYOUT_XFORM_COMPOSE_POINTWISE_LX_BASE,
                }
            ]
            assert calls["kv_repack_pair"] == [
                rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
                rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
            ]
            with open(os.path.join(output_dir, "bundle.mlir")) as file:
                bundle_mlir = file.read()
            assert (
                "sdsc_mixed_flash_kv_repack_broadcast_pair_1_input1_consumer.json"
                in bundle_mlir
            )
    finally:
        _restore_modules(saved)


def test_bundle_executes_kv_hbm_staged_hoist_gate_and_omits_future_producer():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_hbm_staged_hoist_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_hbm_staged_hoist"] == [
                rz.LAYOUT_XFORM_PAIR_AUTO_TILE
            ]
            with open(os.path.join(output_dir, "bundle.mlir")) as file:
                bundle_mlir = file.read()
            assert (
                "sdsc_mixed_flash_kv_repack_hbm_staged_hoist_0_"
                "future_producer.json"
            ) in bundle_mlir
            assert (
                "sdsc_mixed_flash_kv_repack_hbm_staged_hoist_0_"
                "future_kv_1_input1_consumer.json"
            ) in bundle_mlir
            assert "sdsc_1_ReStickifyOpHBM.json" not in bundle_mlir
            assert bundle_mlir.index(
                "sdsc_mixed_flash_kv_repack_hbm_staged_hoist_0_"
                "future_producer.json"
            ) < bundle_mlir.index("sdsc_0_batchmatmul.json")
            assert os.path.exists(
                os.path.join(output_dir, "sdsc_1_ReStickifyOpHBM.json")
            )
    finally:
        _restore_modules(saved)


def test_bundle_executes_kv_hbm_prefetch_hoist_gate_and_omits_future_producer():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_hbm_prefetch_hoist_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_hbm_prefetch_hoist"] == [
                rz.LAYOUT_XFORM_PAIR_AUTO_TILE
            ]
            assert calls["kv_repack_hbm_prefetch_lx_base"] == [None]
            assert calls["kv_repack_hbm_prefetch_serial"] == [False]
            assert calls["kv_repack_hbm_prefetch_prefill_current"] == [False]
            assert calls["kv_repack_hbm_prefetch_redundant_future"] == [False]
            assert calls["kv_repack_hbm_prefetch_serialize_current"] == [False]
            assert calls["kv_repack_hbm_prefetch_external_future"] == [False]
            assert calls["kv_repack_hbm_prefetch_lx_roundtrip"] == [False]
            assert calls["kv_repack_hbm_prefetch_tail_current"] == [False]
            assert calls["kv_repack_hbm_prefetch_source_fanout"] == [False]
            assert calls["kv_repack_hbm_prefetch_loader_fanout"] == [False]
            assert calls["kv_repack_hbm_prefetch_loader_core"] == [0]
            assert calls["kv_repack_hbm_prefetch_loader_lx_base"] == [-1]
            assert calls["kv_repack_hbm_prefetch_fanout_use_unicast"] == [-1]
            assert (
                calls[
                    "kv_repack_hbm_prefetch_fanout_use_lxsfp_lx_transfers"
                ]
                == [-1]
            )
            assert calls["kv_repack_hbm_prefetch_fanout_copyback_core"] == [-2]
            assert (
                calls[
                    "kv_repack_hbm_prefetch_fanout_restrict_to_copyback_core"
                ]
                == [False]
            )
            assert (
                calls[
                    "kv_repack_hbm_prefetch_loader_copyback_without_fanout"
                ]
                == [False]
            )
            assert (
                calls[
                    "kv_repack_hbm_prefetch_loader_fanout_full_tile_pieces"
                ]
                == [False]
            )
            assert (
                calls["kv_repack_hbm_prefetch_serialize_loader_core"]
                == [False]
            )
            assert calls["kv_repack_hbm_prefetch_corelet_id"] == [None]
            with open(os.path.join(output_dir, "bundle.mlir")) as file:
                bundle_mlir = file.read()
            assert (
                "sdsc_mixed_flash_kv_repack_hbm_prefetch_hoist_0_"
                "future_producer.json"
            ) in bundle_mlir
            assert (
                "sdsc_mixed_flash_kv_repack_hbm_prefetch_hoist_0_"
                "current_prefetch.json"
            ) in bundle_mlir
            assert (
                "sdsc_mixed_flash_kv_repack_hbm_prefetch_hoist_0_"
                "future_consumer.json"
            ) in bundle_mlir
            assert "sdsc_1_ReStickifyOpHBM.json" not in bundle_mlir
            assert bundle_mlir.index(
                "sdsc_mixed_flash_kv_repack_hbm_prefetch_hoist_0_"
                "future_producer.json"
            ) < bundle_mlir.index(
                "sdsc_mixed_flash_kv_repack_hbm_prefetch_hoist_0_"
                "current_prefetch.json"
            )
    finally:
        _restore_modules(saved)


def test_bundle_forwards_kv_hbm_prefetch_lx_base_probe():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_hbm_prefetch_hoist_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_hbm_prefetch_lx_base=1625344,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_hbm_prefetch_hoist"] == [
                rz.LAYOUT_XFORM_PAIR_AUTO_TILE
            ]
            assert calls["kv_repack_hbm_prefetch_lx_base"] == [1625344]
            assert calls["kv_repack_hbm_prefetch_serial"] == [False]
            assert calls["kv_repack_hbm_prefetch_prefill_current"] == [False]
            assert calls["kv_repack_hbm_prefetch_redundant_future"] == [False]
            assert calls["kv_repack_hbm_prefetch_serialize_current"] == [False]
            assert calls["kv_repack_hbm_prefetch_external_future"] == [False]
    finally:
        _restore_modules(saved)


def test_bundle_forwards_kv_hbm_prefetch_serial_probe():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_hbm_prefetch_hoist_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_hbm_prefetch_serial=True,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_hbm_prefetch_hoist"] == [
                rz.LAYOUT_XFORM_PAIR_AUTO_TILE
            ]
            assert calls["kv_repack_hbm_prefetch_lx_base"] == [None]
            assert calls["kv_repack_hbm_prefetch_serial"] == [True]
            assert calls["kv_repack_hbm_prefetch_prefill_current"] == [False]
            assert calls["kv_repack_hbm_prefetch_redundant_future"] == [False]
            assert calls["kv_repack_hbm_prefetch_serialize_current"] == [False]
            assert calls["kv_repack_hbm_prefetch_external_future"] == [False]
    finally:
        _restore_modules(saved)


def test_bundle_forwards_kv_hbm_prefetch_prefill_current_probe():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_hbm_prefetch_hoist_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_hbm_prefetch_prefill_current=True,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_hbm_prefetch_hoist"] == [
                rz.LAYOUT_XFORM_PAIR_AUTO_TILE
            ]
            assert calls["kv_repack_hbm_prefetch_prefill_current"] == [True]
            assert calls["kv_repack_hbm_prefetch_redundant_future"] == [False]
            assert calls["kv_repack_hbm_prefetch_serialize_current"] == [False]
            assert calls["kv_repack_hbm_prefetch_external_future"] == [False]
    finally:
        _restore_modules(saved)


def test_bundle_forwards_kv_hbm_prefetch_redundant_future_probe():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_hbm_prefetch_hoist_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_hbm_prefetch_redundant_future=True,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_hbm_prefetch_hoist"] == [
                rz.LAYOUT_XFORM_PAIR_AUTO_TILE
            ]
            assert calls["kv_repack_hbm_prefetch_serial"] == [False]
            assert calls["kv_repack_hbm_prefetch_prefill_current"] == [False]
            assert calls["kv_repack_hbm_prefetch_redundant_future"] == [True]
            assert calls["kv_repack_hbm_prefetch_serialize_current"] == [False]
            assert calls["kv_repack_hbm_prefetch_external_future"] == [False]
    finally:
        _restore_modules(saved)


def test_bundle_forwards_kv_hbm_prefetch_serialize_current_probe():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_hbm_prefetch_hoist_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_hbm_prefetch_redundant_future=True,
        kv_repack_hbm_prefetch_serialize_current=True,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_hbm_prefetch_hoist"] == [
                rz.LAYOUT_XFORM_PAIR_AUTO_TILE
            ]
            assert calls["kv_repack_hbm_prefetch_redundant_future"] == [True]
            assert calls["kv_repack_hbm_prefetch_serialize_current"] == [True]
            assert calls["kv_repack_hbm_prefetch_external_future"] == [False]
    finally:
        _restore_modules(saved)


def test_bundle_forwards_kv_hbm_prefetch_external_future_probe():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_hbm_prefetch_hoist_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_hbm_prefetch_external_future=False,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_hbm_prefetch_hoist"] == [
                rz.LAYOUT_XFORM_PAIR_AUTO_TILE
            ]
            assert calls["kv_repack_hbm_prefetch_external_future"] == [False]
    finally:
        _restore_modules(saved)


def test_bundle_forwards_kv_hbm_prefetch_tail_current_probe():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_hbm_prefetch_hoist_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_hbm_prefetch_tail_current=True,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_hbm_prefetch_hoist"] == [
                rz.LAYOUT_XFORM_PAIR_AUTO_TILE
            ]
            assert calls["kv_repack_hbm_prefetch_tail_current"] == [True]
    finally:
        _restore_modules(saved)


def test_bundle_forwards_kv_hbm_prefetch_source_fanout_probe():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_hbm_prefetch_hoist_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_hbm_prefetch_source_fanout=True,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_hbm_prefetch_hoist"] == [
                rz.LAYOUT_XFORM_PAIR_AUTO_TILE
            ]
            assert calls["kv_repack_hbm_prefetch_source_fanout"] == [True]
    finally:
        _restore_modules(saved)


def test_bundle_forwards_kv_hbm_prefetch_loader_fanout_probe():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_hbm_prefetch_hoist_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_hbm_prefetch_loader_fanout=True,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_hbm_prefetch_hoist"] == [
                rz.LAYOUT_XFORM_PAIR_AUTO_TILE
            ]
            assert calls["kv_repack_hbm_prefetch_loader_fanout"] == [True]
            assert calls["kv_repack_hbm_prefetch_loader_core"] == [0]
            assert calls["kv_repack_hbm_prefetch_loader_lx_base"] == [-1]
    finally:
        _restore_modules(saved)


def test_bundle_forwards_kv_hbm_prefetch_loader_core_probe():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_hbm_prefetch_hoist_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_hbm_prefetch_loader_fanout=True,
        kv_repack_hbm_prefetch_loader_core=31,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_hbm_prefetch_hoist"] == [
                rz.LAYOUT_XFORM_PAIR_AUTO_TILE
            ]
            assert calls["kv_repack_hbm_prefetch_loader_fanout"] == [True]
            assert calls["kv_repack_hbm_prefetch_loader_core"] == [31]
    finally:
        _restore_modules(saved)


def test_bundle_forwards_kv_hbm_prefetch_loader_lx_base_probe():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_hbm_prefetch_hoist_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_hbm_prefetch_loader_fanout=True,
        kv_repack_hbm_prefetch_loader_lx_base=-2,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_hbm_prefetch_hoist"] == [
                rz.LAYOUT_XFORM_PAIR_AUTO_TILE
            ]
            assert calls["kv_repack_hbm_prefetch_loader_fanout"] == [True]
            assert calls["kv_repack_hbm_prefetch_loader_lx_base"] == [-2]
    finally:
        _restore_modules(saved)


def test_bundle_forwards_kv_hbm_prefetch_serialize_loader_core_probe():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_hbm_prefetch_hoist_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_hbm_prefetch_loader_fanout=True,
        kv_repack_hbm_prefetch_serialize_loader_core=True,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_hbm_prefetch_hoist"] == [
                rz.LAYOUT_XFORM_PAIR_AUTO_TILE
            ]
            assert calls["kv_repack_hbm_prefetch_loader_fanout"] == [True]
            assert calls["kv_repack_hbm_prefetch_serialize_loader_core"] == [
                True
            ]
    finally:
        _restore_modules(saved)


def test_bundle_forwards_kv_hbm_prefetch_fanout_unicast_probe():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_hbm_prefetch_hoist_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_hbm_prefetch_fanout_use_unicast=1,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_hbm_prefetch_hoist"] == [
                rz.LAYOUT_XFORM_PAIR_AUTO_TILE
            ]
            assert calls["kv_repack_hbm_prefetch_fanout_use_unicast"] == [1]
    finally:
        _restore_modules(saved)


def test_bundle_forwards_kv_hbm_prefetch_fanout_lxsfp_lx_transfer_probe():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_hbm_prefetch_hoist_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_hbm_prefetch_fanout_use_lxsfp_lx_transfers=0,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_hbm_prefetch_hoist"] == [
                rz.LAYOUT_XFORM_PAIR_AUTO_TILE
            ]
            assert (
                calls[
                    "kv_repack_hbm_prefetch_fanout_use_lxsfp_lx_transfers"
                ]
                == [0]
            )
    finally:
        _restore_modules(saved)


def test_bundle_forwards_kv_hbm_prefetch_fanout_copyback_probe():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_hbm_prefetch_hoist_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_hbm_prefetch_fanout_copyback_core=0,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_hbm_prefetch_hoist"] == [
                rz.LAYOUT_XFORM_PAIR_AUTO_TILE
            ]
            assert calls["kv_repack_hbm_prefetch_fanout_copyback_core"] == [0]
    finally:
        _restore_modules(saved)


def test_bundle_forwards_kv_hbm_prefetch_fanout_restrict_to_copyback_probe():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_hbm_prefetch_hoist_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_hbm_prefetch_fanout_copyback_core=0,
        kv_repack_hbm_prefetch_fanout_restrict_to_copyback_core=True,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_hbm_prefetch_hoist"] == [
                rz.LAYOUT_XFORM_PAIR_AUTO_TILE
            ]
            assert (
                calls[
                    "kv_repack_hbm_prefetch_fanout_restrict_to_copyback_core"
                ]
                == [True]
            )
    finally:
        _restore_modules(saved)


def test_bundle_forwards_kv_hbm_prefetch_loader_copyback_without_fanout_probe():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_hbm_prefetch_hoist_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_hbm_prefetch_fanout_copyback_core=0,
        kv_repack_hbm_prefetch_loader_copyback_without_fanout=True,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_hbm_prefetch_hoist"] == [
                rz.LAYOUT_XFORM_PAIR_AUTO_TILE
            ]
            assert (
                calls[
                    "kv_repack_hbm_prefetch_loader_copyback_without_fanout"
                ]
                == [True]
            )
    finally:
        _restore_modules(saved)


def test_bundle_forwards_kv_hbm_prefetch_loader_fanout_full_tile_probe():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_hbm_prefetch_hoist_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_hbm_prefetch_loader_fanout_full_tile_pieces=True,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_hbm_prefetch_hoist"] == [
                rz.LAYOUT_XFORM_PAIR_AUTO_TILE
            ]
            assert (
                calls[
                    "kv_repack_hbm_prefetch_loader_fanout_full_tile_pieces"
                ]
                == [True]
            )
    finally:
        _restore_modules(saved)


def test_bundle_forwards_kv_hbm_prefetch_roundtrip_corelet1_probe():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_hbm_prefetch_hoist_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_hbm_prefetch_lx_roundtrip=True,
        kv_repack_hbm_prefetch_corelet1=True,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_hbm_prefetch_hoist"] == [
                rz.LAYOUT_XFORM_PAIR_AUTO_TILE
            ]
            assert calls["kv_repack_hbm_prefetch_lx_roundtrip"] == [True]
            assert calls["kv_repack_hbm_prefetch_corelet_id"] == [1]
    finally:
        _restore_modules(saved)


def test_bundle_can_enable_kv_repack_pair_hbm_source():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_pair_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_pair_hbm_source=True,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_pair"] == [rz.LAYOUT_XFORM_PAIR_AUTO_TILE]
            assert calls["kv_repack_pair_hbm_source"] == [True]
            with open(os.path.join(output_dir, "bundle.mlir")) as file:
                bundle_mlir = file.read()
            assert (
                "sdsc_mixed_flash_kv_repack_broadcast_pair_1_input1_producer.json"
                not in bundle_mlir
            )
            assert (
                "sdsc_mixed_flash_kv_repack_broadcast_pair_1_input1_consumer.json"
                in bundle_mlir
            )
            assert "sdsc_1_ReStickifyOpHBM.json" in bundle_mlir
    finally:
        _restore_modules(saved)


def test_bundle_can_enable_kv_repack_pair_hbm_direct_load():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_pair_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_pair_hbm_direct_load=True,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_pair"] == [rz.LAYOUT_XFORM_PAIR_AUTO_TILE]
            assert calls["kv_repack_pair_hbm_direct_load"] == [True]
            with open(os.path.join(output_dir, "bundle.mlir")) as file:
                bundle_mlir = file.read()
            assert (
                "sdsc_mixed_flash_kv_repack_broadcast_pair_1_input1_producer.json"
                not in bundle_mlir
            )
            assert (
                "sdsc_mixed_flash_kv_repack_broadcast_pair_1_input1_consumer.json"
                in bundle_mlir
            )
            assert "sdsc_1_ReStickifyOpHBM.json" in bundle_mlir
    finally:
        _restore_modules(saved)


def test_bundle_can_enable_kv_repack_pair_hbm_staged():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_pair_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_pair_hbm_staged=True,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_pair"] == [rz.LAYOUT_XFORM_PAIR_AUTO_TILE]
            assert calls["kv_repack_pair_hbm_staged"] == [True]
            with open(os.path.join(output_dir, "bundle.mlir")) as file:
                bundle_mlir = file.read()
            assert (
                "sdsc_mixed_flash_kv_repack_broadcast_pair_1_input1_producer.json"
                not in bundle_mlir
            )
            assert (
                "sdsc_mixed_flash_kv_repack_broadcast_pair_1_input1_consumer.json"
                in bundle_mlir
            )
            assert "sdsc_1_ReStickifyOpHBM.json" in bundle_mlir
    finally:
        _restore_modules(saved)


def test_bundle_can_disable_kv_repack_pair_consumer_core_state_init():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_pair_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_pair_consumer_core_state_init=False,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_pair"] == [rz.LAYOUT_XFORM_PAIR_AUTO_TILE]
            assert calls["kv_repack_pair_consumer_core_state_init"] == [False]
    finally:
        _restore_modules(saved)


def test_bundle_can_override_kv_repack_pair_consumer_ds_type():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_pair_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_pair_consumer_ds_type="INPUT",
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_pair"] == [rz.LAYOUT_XFORM_PAIR_AUTO_TILE]
            assert calls["kv_repack_pair_consumer_ds_type"] == ["INPUT"]
    finally:
        _restore_modules(saved)


def test_bundle_can_override_kv_repack_pair_consumer_lx_alloc_style():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_pair_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_pair_consumer_lx_alloc_style="canonical_loop",
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_pair"] == [rz.LAYOUT_XFORM_PAIR_AUTO_TILE]
            assert calls["kv_repack_pair_consumer_lx_alloc_style"] == [
                "canonical_loop"
            ]
    finally:
        _restore_modules(saved)


def test_bundle_executes_kv_repack_copyback_before_original_consumer():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_copyback_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_copyback"] == [rz.LAYOUT_XFORM_PAIR_AUTO_TILE]
            assert calls["kv_repack_copyback_core"] == [-1]
            assert calls["kv_repack_copyback_direct_source"] == [False]
            assert calls["kv_repack_copyback_hbm_roundtrip"] == [False]
            assert calls["kv_repack_copyback_hbm_source_fanout"] == [False]
            with open(os.path.join(output_dir, "bundle.mlir")) as file:
                bundle_mlir = file.read()
            producer = (
                "sdsc_mixed_flash_kv_repack_broadcast_copyback_"
                "1_input1_producer.json"
            )
            copyback = (
                "sdsc_mixed_flash_kv_repack_broadcast_copyback_"
                "1_input1_copyback.json"
            )
            consumer = "sdsc_2_batchmatmul.json"
            assert producer in bundle_mlir
            assert copyback in bundle_mlir
            assert consumer in bundle_mlir
            assert bundle_mlir.index(producer) < bundle_mlir.index(copyback)
            assert bundle_mlir.index(copyback) < bundle_mlir.index(consumer)
    finally:
        _restore_modules(saved)


def test_bundle_can_enable_kv_repack_copyback_hbm_roundtrip():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_copyback_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_copyback_hbm_roundtrip=True,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_copyback"] == [rz.LAYOUT_XFORM_PAIR_AUTO_TILE]
            assert calls["kv_repack_copyback_hbm_roundtrip"] == [True]
            with open(os.path.join(output_dir, "bundle.mlir")) as file:
                bundle_mlir = file.read()
            assert (
                "sdsc_mixed_flash_kv_repack_broadcast_copyback_"
                "1_input1_copyback.json"
            ) in bundle_mlir
    finally:
        _restore_modules(saved)


def test_bundle_can_enable_kv_repack_copyback_hbm_source_fanout():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_copyback_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_copyback_hbm_source_fanout=True,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_copyback"] == [rz.LAYOUT_XFORM_PAIR_AUTO_TILE]
            assert calls["kv_repack_copyback_hbm_roundtrip"] == [False]
            assert calls["kv_repack_copyback_hbm_source_fanout"] == [True]
    finally:
        _restore_modules(saved)


def test_bundle_can_enable_kv_repack_copyback_hbm_direct_load():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_copyback_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_copyback_hbm_direct_load=True,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_copyback"] == [rz.LAYOUT_XFORM_PAIR_AUTO_TILE]
            assert calls["kv_repack_copyback_hbm_roundtrip"] == [False]
            assert calls["kv_repack_copyback_hbm_source_fanout"] == [False]
            assert calls["kv_repack_copyback_hbm_direct_load"] == [True]
    finally:
        _restore_modules(saved)


def test_bundle_can_enable_kv_repack_copyback_hbm_load_only():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_copyback_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_copyback_hbm_roundtrip=True,
        kv_repack_copyback_hbm_roundtrip_load_only=True,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_copyback"] == [rz.LAYOUT_XFORM_PAIR_AUTO_TILE]
            assert calls["kv_repack_copyback_hbm_roundtrip"] == [True]
            assert calls["kv_repack_copyback_hbm_roundtrip_load_only"] == [True]
    finally:
        _restore_modules(saved)


def test_bundle_can_enable_kv_repack_copyback_hbm_barrier_only():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_copyback_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_copyback_hbm_roundtrip=True,
        kv_repack_copyback_hbm_roundtrip_barrier_only=True,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_copyback"] == [rz.LAYOUT_XFORM_PAIR_AUTO_TILE]
            assert calls["kv_repack_copyback_hbm_roundtrip"] == [True]
            assert calls["kv_repack_copyback_hbm_roundtrip_barrier_only"] == [True]
    finally:
        _restore_modules(saved)


def test_bundle_can_enable_kv_repack_copyback_data_only():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_copyback_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_copyback_hbm_roundtrip=True,
        kv_repack_copyback_data_only=True,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_copyback"] == [rz.LAYOUT_XFORM_PAIR_AUTO_TILE]
            assert calls["kv_repack_copyback_hbm_roundtrip"] == [True]
            assert calls["kv_repack_copyback_data_only"] == [True]
    finally:
        _restore_modules(saved)


def test_bundle_can_enable_kv_repack_copyback_replace_consumer():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_copyback_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_copyback_hbm_roundtrip=True,
        kv_repack_copyback_replace_consumer=True,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_copyback"] == [rz.LAYOUT_XFORM_PAIR_AUTO_TILE]
            assert calls["kv_repack_copyback_hbm_roundtrip"] == [True]
            assert calls["kv_repack_copyback_replace_consumer"] == [True]
    finally:
        _restore_modules(saved)


def test_bundle_can_enable_kv_repack_copyback_compute_only():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_copyback_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_copyback_hbm_roundtrip=True,
        kv_repack_copyback_replace_consumer=True,
        kv_repack_copyback_compute_only=True,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_copyback"] == [rz.LAYOUT_XFORM_PAIR_AUTO_TILE]
            assert calls["kv_repack_copyback_replace_consumer"] == [True]
            assert calls["kv_repack_copyback_compute_only"] == [True]
    finally:
        _restore_modules(saved)


def test_bundle_can_enable_kv_repack_copyback_exact_clone():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_copyback_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_copyback_hbm_roundtrip=True,
        kv_repack_copyback_exact_clone=True,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_copyback"] == [rz.LAYOUT_XFORM_PAIR_AUTO_TILE]
            assert calls["kv_repack_copyback_hbm_roundtrip"] == [True]
            assert calls["kv_repack_copyback_exact_clone"] == [True]
    finally:
        _restore_modules(saved)


def test_bundle_can_enable_kv_repack_copyback_preserve_consumer_name():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_copyback_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_copyback_hbm_roundtrip=True,
        kv_repack_copyback_exact_clone=True,
        kv_repack_copyback_preserve_consumer_name=True,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_copyback"] == [rz.LAYOUT_XFORM_PAIR_AUTO_TILE]
            assert calls["kv_repack_copyback_exact_clone"] == [True]
            assert calls["kv_repack_copyback_preserve_consumer_name"] == [True]
    finally:
        _restore_modules(saved)


def test_bundle_can_disable_kv_repack_pair_ifn_transfer_marker():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_pair_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_pair_ifn_transfer=False,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_pair"] == [rz.LAYOUT_XFORM_PAIR_AUTO_TILE]
            assert calls["kv_repack_pair_ifn_transfer"] == [False]
            assert calls["kv_repack_pair_subpiece_reuse"] == [True]
            assert calls["kv_repack_pair_group_size"] == [0]
    finally:
        _restore_modules(saved)


def test_bundle_can_disable_kv_repack_pair_subpiece_reuse():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_pair_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_pair_subpiece_reuse=False,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_pair"] == [rz.LAYOUT_XFORM_PAIR_AUTO_TILE]
            assert calls["kv_repack_pair_ifn_transfer"] == [True]
            assert calls["kv_repack_pair_subpiece_reuse"] == [False]
            assert calls["kv_repack_pair_group_size"] == [0]
    finally:
        _restore_modules(saved)


def test_bundle_can_split_kv_repack_pair_broadcast_groups():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_pair_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_pair_group_size=16,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_pair"] == [rz.LAYOUT_XFORM_PAIR_AUTO_TILE]
            assert calls["kv_repack_pair_group_size"] == [16]
    finally:
        _restore_modules(saved)


def test_bundle_can_enable_kv_repack_pair_self_resident_source():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_pair_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_pair_self_resident_source=True,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_pair"] == [rz.LAYOUT_XFORM_PAIR_AUTO_TILE]
            assert calls["kv_repack_pair_self_resident_source"] == [True]
            assert calls["kv_repack_pair_force_mc_mode"] == [-1]
    finally:
        _restore_modules(saved)


def test_bundle_can_force_kv_repack_pair_multicast_mode():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_pair_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        kv_repack_pair_force_mc_mode=3,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["kv_repack_pair"] == [rz.LAYOUT_XFORM_PAIR_AUTO_TILE]
            assert calls["kv_repack_pair_self_resident_source"] == [False]
            assert calls["kv_repack_pair_force_mc_mode"] == [3]
    finally:
        _restore_modules(saved)


def test_bundle_falls_back_to_layout_pair_when_lookahead_fails_closed():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        layout_xform_lookahead_tile=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
        layout_xform_lookahead_result=False,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert calls["layout_xform_lookahead"] == [
                rz.LAYOUT_XFORM_PAIR_AUTO_TILE
            ]
            assert calls["layout_xform"] == [rz.LAYOUT_XFORM_PAIR_AUTO_TILE]
            with open(os.path.join(output_dir, "bundle.mlir")) as file:
                bundle_mlir = file.read()
            assert "sdsc_mixed_flash_layout_xform_pair_tile_2_consumer.json" in (
                bundle_mlir
            )
    finally:
        _restore_modules(saved)


def _ifn_prefix_tile_artifact():
    return {
        "mixed_flash_pipeline_tile_0": {
            "flashAttentionPipeline_": {
                "source": "generated-flash-prefill-overlap-prefix-ifn-tile",
                "tile_index": 0,
                "replaces_sdsc": "0_batchmatmul",
                "overlap_prefix": True,
                "overlap_candidate": True,
                "ifn_attached_input_idx": 0,
                "ifn_runtime_safe": False,
            },
            "dscs_": [{"batchmatmul": {"computeOp_": []}}],
            "datadscs_": [
                {"0_STCDPOpLx_prefetch_ifn_Tensor0_idx0_tile0": {}}
            ],
            "opFuncsUsed_": ["STCDPOpLx"],
        }
    }


def test_bundle_keeps_ifn_prefix_probe_non_executed_without_force():
    bundle, _calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        execute_tile=0,
        tile_artifacts=[_ifn_prefix_tile_artifact()],
    )
    try:
        specs = [{"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}}]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            with open(os.path.join(output_dir, "bundle.mlir")) as file:
                bundle_mlir = file.read()
            assert "sdsc_0_batchmatmul.json" in bundle_mlir
            assert "sdsc_mixed_flash_pipeline_tile_0.json" not in bundle_mlir
    finally:
        _restore_modules(saved)


def test_bundle_force_executes_ifn_prefix_probe():
    bundle, _calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        ifn_prefix_force=True,
        execute_tile=0,
        tile_artifacts=[_ifn_prefix_tile_artifact()],
    )
    try:
        specs = [{"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}}]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            with open(os.path.join(output_dir, "bundle.mlir")) as file:
                bundle_mlir = file.read()
            assert "sdsc_mixed_flash_pipeline_tile_0.json" in bundle_mlir

            sidecar_path = os.path.join(
                output_dir,
                "sdsc_mixed_flash_pipeline_tile_0.json",
            )
            with open(sidecar_path) as file:
                sidecar = json.load(file)
            meta = sidecar["mixed_flash_pipeline_tile_0"][
                "flashAttentionPipeline_"
            ]
            assert meta["ifn_runtime_forced"] is True
    finally:
        _restore_modules(saved)


def test_bundle_shifts_pointwise_handoffs_when_layout_xform_pair_is_active():
    region0 = rz.LAYOUT_XFORM_COMPOSE_POINTWISE_LX_BASE + rz.MIN_BRIDGE_REGION_BYTES
    bundle, calls, saved = _load_bundle_with_stubs(
        pointwise_handoff=True,
        layout_xform_pointwise_region0=region0,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

        assert calls["layout_xform"] == [rz.LAYOUT_XFORM_PAIR_AUTO_TILE]
        assert calls["pointwise"] == [
            {
                "score_scale_handoff": False,
                "pointwise_region0": region0,
            }
        ]
    finally:
        _restore_modules(saved)


def test_bundle_keeps_pointwise_handoffs_when_layout_xform_pair_fails_closed():
    bundle, calls, saved = _load_bundle_with_stubs(
        pointwise_handoff=True,
        layout_xform_pair_result=False,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

        assert calls["layout_xform"] == [rz.LAYOUT_XFORM_PAIR_AUTO_TILE]
        assert calls["pointwise"] == [{"score_scale_handoff": False}]
    finally:
        _restore_modules(saved)


def test_bundle_keeps_pointwise_handoffs_when_layout_xform_pair_is_disabled():
    bundle, calls, saved = _load_bundle_with_stubs(
        pointwise_handoff=True,
        layout_xform_pair_tile=-1,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

        assert calls["layout_xform"] == []
        assert calls["pointwise"] == [{"score_scale_handoff": False}]
    finally:
        _restore_modules(saved)


def _fake_causal_score_bias_sdsc():
    return {
        "0_causal_score_bias_like": {
            "numWkSlicesPerDim_": {"mb": 2, "x": 4, "out": 1},
            "coreIdToWkSlice_": {
                "0": {"mb": 0, "x": 0, "out": 0},
                "1": {"mb": 1, "x": 0, "out": 0},
                "2": {"mb": 0, "x": 1, "out": 0},
                "3": {"mb": 1, "x": 1, "out": 0},
                "4": {"mb": 0, "x": 2, "out": 0},
                "5": {"mb": 1, "x": 2, "out": 0},
                "6": {"mb": 0, "x": 3, "out": 0},
                "7": {"mb": 1, "x": 3, "out": 0},
            },
            "dscs_": [
                {
                    "causal_score_bias_like": {
                        "numCoresUsed_": 8,
                        "N_": {"name_": "n", "mb_": 2, "x_": 4, "out_": 64},
                        "primaryDsInfo_": {
                            "OUTPUT": {
                                "layoutDimOrder_": ["x", "mb", "out"],
                                "stickDimOrder_": ["out"],
                                "stickSize_": [64],
                            }
                        },
                        "labeledDs_": [
                            {
                                "ldsIdx_": 0,
                                "dsName_": "Tensor0",
                                "dsType_": "OUTPUT",
                            },
                            {
                                "ldsIdx_": 1,
                                "dsName_": "Tensor1",
                                "dsType_": "OUTPUT",
                            },
                        ],
                        "constantInfo_": {
                            "0": {
                                "dataFormat_": "SEN169_FP16",
                                "name_": "keyStart",
                            }
                        },
                        "computeOp_": [
                            {
                                "opFuncName": "causal_score_bias_like",
                                "inputLabeledDs": ["Tensor0-idx0"],
                                "outputLabeledDs": ["Tensor1-idx1"],
                            }
                        ],
                    }
                }
            ],
        }
    }


def test_bundle_emits_non_executed_causal_idx_to_mask_plan_artifact():
    class FakeCausalSpec:
        op = "causal_score_bias_like"
        op_info = {"constants": {"keyStart": 2}}
        sdsc_json = _fake_causal_score_bias_sdsc()

    bundle, _calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        causal_plan_artifact=True,
    )
    try:
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, [FakeCausalSpec()])

            plan_path = os.path.join(output_dir, "causal_idx_to_mask_plan_0.json")
            assert os.path.exists(plan_path)
            with open(plan_path) as file:
                plan = json.load(file)
            body = plan["causal_idx_to_mask_plan_0"]
            dataop = body["datadscs_"][0]["0_IdxToMask_dataop"]
            assert dataop["op"] == {
                "name": "IdxToMask",
                "idxToMaskDimIdx": 2,
                "idxToMaskValidElementOffset": -2,
                "invertedMask": 0,
                "reversedMask": 0,
                "causalMask": 1,
            }
            assert body["coreIdToDscSchedule"]["0"] == [
                [0, -1, 0, 1],
                [-1, 0, 1, 0],
            ]
            assert body["where3_compute_fragment"]["computeOp_"][0][
                "opFuncName"
            ] == "where3"

            with open(os.path.join(output_dir, "bundle.mlir")) as file:
                bundle_mlir = file.read()
            assert "causal_idx_to_mask_plan_0.json" not in bundle_mlir
    finally:
        _restore_modules(saved)


def test_bundle_emits_non_executed_kv_repack_plan_artifact():
    bundle, calls, saved = _load_bundle_with_stubs(
        layout_xform_pair_tile=-1,
        kv_repack_plan_artifact=True,
    )
    try:
        specs = [
            {"0_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
            {"1_ReStickifyOpHBM": {"dscs_": [{"ReStickifyOpHBM": {}}]}},
            {"2_batchmatmul": {"dscs_": [{"batchmatmul": {}}]}},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            bundle.generate_bundle("kernel", output_dir, specs)

            assert (1, 1) in calls["kv_repack_plan"]
            plan_path = os.path.join(
                output_dir,
                "sdsc_flash_kv_repack_broadcast_plan_1_input1.json",
            )
            assert os.path.exists(plan_path)
            with open(plan_path) as file:
                plan = json.load(file)
            body = plan["flash_kv_repack_broadcast_plan_1_input1"]
            meta = body["flashAttentionPipeline_"]
            assert meta["kv_repack_broadcast_plan"] is True
            assert meta["kv_repack_broadcast_executable"] is False
            assert body["opFuncsUsed_"] == ["STCDPOpLx"]

            with open(os.path.join(output_dir, "bundle.mlir")) as file:
                bundle_mlir = file.read()
            assert "sdsc_flash_kv_repack_broadcast_plan_1_input1.json" not in (
                bundle_mlir
            )
    finally:
        _restore_modules(saved)


def test_flash_layout_xform_pair_auto_reports_rejections():
    assert rz.flash_attention_layout_xform_pair_rejection_reasons(
        _fake_static_matmul_sdscs(),
        tile_index=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
    ) == [
        "tile0:input0:not_consumer_input",
        "tile1:input0:same_physical_layout_use_ifn_pair",
    ]
    assert rz.flash_attention_layout_xform_pair_rejection_reasons(
        [],
        tile_index=rz.LAYOUT_XFORM_PAIR_AUTO_TILE,
    ) == ["auto:no_candidate_tiles"]


def test_flash_layout_xform_pair_tile_rejects_same_physical_edge():
    assert rz.build_flash_attention_layout_xform_pair_tile_artifacts(
        _fake_static_matmul_sdscs(),
        tile_index=1,
    ) is None
    assert rz.flash_attention_layout_xform_pair_tile_rejection_reasons(
        _fake_static_matmul_sdscs(),
        tile_index=1,
    ) == ["input0:same_physical_layout_use_ifn_pair"]


def test_flash_layout_xform_pair_tile_maps_producer_work_slices():
    result = rz.build_flash_attention_layout_xform_pair_tile_artifacts(
        _fake_flash_layout_xform_relation_sdscs(),
        tile_index=0,
    )

    assert result is not None
    cons = result["artifacts"][1]["mixed_flash_layout_xform_pair_tile_0_consumer"]
    dataop = next(iter(cons["datadscs_"][0].values()))
    src_ld = dataop["labeledDs_"][0]
    dst_ld = dataop["labeledDs_"][1]
    assert src_ld["layoutDimOrder_"] == ["x_", "mb_", "in_"]
    assert dst_ld["layoutDimOrder_"] == ["x_", "mb_", "in_"]
    src_pieces = src_ld["PieceInfo"]
    assert len(src_pieces) == 4
    assert len(dst_ld["PieceInfo"]) == 32
    assert src_pieces[0]["dimToStartCordinate"] == {
        "x_": 0,
        "mb_": 0,
        "in_": 0,
    }
    assert src_pieces[1]["dimToStartCordinate"] == {
        "x_": 1,
        "mb_": 0,
        "in_": 0,
    }
    assert src_pieces[2]["dimToStartCordinate"] == {
        "x_": 0,
        "mb_": 64,
        "in_": 0,
    }
    assert src_pieces[0]["dimToSize_"] == {"x_": 1, "mb_": 64, "in_": 64}
    assert src_pieces[0]["PlacementInfo"][0]["memId"] == [0]
    assert src_pieces[3]["PlacementInfo"][0]["memId"] == [3]


def test_flash_pipeline_artifact_wraps_generated_batchmatmul_tiles():
    artifact = rz.build_flash_attention_pipeline_artifact(
        _fake_flash_pipeline_sdscs(num_tiles=3),
        overlap=False,
    )
    assert artifact is not None
    root = artifact["mixed_flash_pipeline_artifact"]
    assert len(root["dscs_"]) == 3
    assert len(root["datadscs_"]) == 6
    assert root["opFuncsUsed_"] == ["STCDPOpLx"] * 6
    assert root["numWkSlicesPerDim_"] == {"x": 1, "mb": 32, "out": 1, "in": 1}
    assert root["coreIdToDsc_"] == _fake_flash_pipeline_sdscs()[0][
        "0_batchmatmul"
    ]["coreIdToDsc_"]
    meta = root["flashAttentionPipeline_"]
    assert meta["tile_count"] == 3
    assert meta["dataop_count"] == 6
    assert meta["overlap_candidate"] is False
    assert meta["source"] == "generated-flash-prefill-batchmatmul-tiles"
    assert meta["layout"] == ["mb_", "x_", "out_"]
    assert meta["split_dim"] == "mb_"
    assert meta["stick_dim"] == "out_"
    assert meta["row_dim"] == "out_"
    assert root["coreIdToDscSchedule"]["0"] == [
        [0, -1, 0, 1],
        [1, -1, 1, 1],
        [-1, 0, 1, 1],
        [2, -1, 1, 1],
        [3, -1, 1, 1],
        [-1, 1, 1, 1],
        [4, -1, 1, 1],
        [5, -1, 1, 1],
        [-1, 2, 1, 0],
    ]


def test_flash_pipeline_artifact_overlap_marks_candidate_rows():
    artifact = rz.build_flash_attention_pipeline_artifact(
        _fake_flash_pipeline_sdscs(num_tiles=3),
        overlap=True,
    )
    root = artifact["mixed_flash_pipeline_artifact"]
    assert root["flashAttentionPipeline_"]["overlap_candidate"] is True
    assert [2, 0, 1, 1] in root["coreIdToDscSchedule"]["0"]


def test_flash_pipeline_artifact_returns_none_without_batchmatmul_tiles():
    assert rz.build_flash_attention_pipeline_artifact([]) is None


def test_flash_pipeline_tile_artifacts_are_one_compute_each():
    artifacts = rz.build_flash_attention_pipeline_tile_artifacts(
        _fake_flash_pipeline_sdscs(num_tiles=3)
    )
    assert len(artifacts) == 3
    for idx, artifact in enumerate(artifacts):
        name = f"mixed_flash_pipeline_tile_{idx}"
        root = artifact[name]
        assert len(root["dscs_"]) == 1
        assert len(root["datadscs_"]) == 2
        assert root["flashAttentionPipeline_"]["tile_count"] == 1
        assert root["flashAttentionPipeline_"]["tile_index"] == idx
        assert root["flashAttentionPipeline_"]["replaces_sdsc"] == (
            f"{idx}_batchmatmul"
        )
        assert root["coreIdToDscSchedule"]["0"] == [
            [0, -1, 0, 1],
            [1, -1, 1, 1],
            [-1, 0, 1, 0],
        ]


def test_flash_pipeline_overlap_prefix_tile_artifacts_overlap_one_compute():
    artifacts = rz.build_flash_attention_pipeline_tile_artifacts(
        _fake_flash_pipeline_sdscs(
            num_tiles=3,
            lx_pinned=True,
            input_neighbor_transfer=True,
            ij_input_layout=True,
        ),
        overlap_prefix=True,
    )
    assert len(artifacts) == 3

    root0 = artifacts[0]["mixed_flash_pipeline_tile_0"]
    assert len(root0["dscs_"]) == 1
    assert len(root0["datadscs_"]) == 1
    assert root0["opFuncsUsed_"] == ["STCDPOpLx"]
    dataop_name, dataop = next(iter(root0["datadscs_"][0].items()))
    assert dataop_name == "0_STCDPOpLx_prefetch_ifn_Tensor0_idx0_tile0"
    assert dataop["op"] == {"name": "STCDPOpLx"}
    meta0 = root0["flashAttentionPipeline_"]
    assert meta0["source"] == "generated-flash-prefill-overlap-prefix-ifn-tile"
    assert meta0["tile_count"] == 1
    assert meta0["dataop_count"] == 1
    assert meta0["ifn_attached_input_idx"] == 0
    assert meta0["ifn_input_lx_base"] == rz.CONSUMER_LX_BASE
    assert meta0["ifn_runtime_safe"] is False
    assert meta0["ifn_runtime_rejection_reason"] == (
        "single_sdsc_ifn_no_real_predecessor"
    )
    assert meta0["compute_tile_count"] == 1
    assert meta0["overlap_prefix"] is True
    assert meta0["overlap_candidate"] is True
    assert meta0["tile_index"] == 0
    assert meta0["replaces_sdsc"] == "0_batchmatmul"
    assert root0["coreIdToDscSchedule"]["0"] == [
        [0, 0, 0, 0],
    ]
    compute = next(iter(root0["dscs_"][0].values()))
    input_lds = next(lds for lds in compute["labeledDs_"] if lds["ldsIdx_"] == 0)
    assert input_lds["memOrg_"] == {"lx": {"isPresent": 1, "allocateNode_": "allocate-Tensor0_lx"}}
    assert rz._has_input_fetch_neighbor_transfer(compute, 0)
    transfer = next(
        node
        for node in compute["scheduleTree_"]
        if rz._is_input_fetch_neighbor_transfer_node(node, 0)
    )
    assert transfer["prev_"] == ""
    assert transfer["src_"] == {
        "unit_": "no_component",
        "storage_": "no_component",
    }
    assert transfer["srcLdsAndLoopOffsets_"]["myLdsIdx_"] == -1
    assert transfer["dstLdsAndLoopOffsets_"][0]["myLdsIdx_"] == 0
    assert transfer["dstLdsAndLoopOffsets_"][0]["startAddr_"] == "0"
    alloc = next(
        node
        for node in compute["scheduleTree_"]
        if node.get("nodeType_") == "allocate"
        and node.get("ldsIdx_") == 0
        and node.get("component_") == "lx"
    )
    assert alloc["allocUsers_"][transfer["name_"]] == 1
    assert compute["CoreD_"]["i_"] == 2
    assert compute["CoreD_"]["j_"] == 2
    assert compute["CoreD_"]["in_"] == 64
    assert compute["CoreletD_"]["i_"] == 2
    assert compute["B_"]["i_"] == 2

    root2 = artifacts[2]["mixed_flash_pipeline_tile_2"]
    assert len(root2["dscs_"]) == 1
    assert len(root2["datadscs_"]) == 2
    assert root2["flashAttentionPipeline_"]["overlap_prefix"] is False


def test_flash_pipeline_overlap_prefix_allows_hbm_backed_compute():
    artifacts = rz.build_flash_attention_pipeline_tile_artifacts(
        _fake_flash_pipeline_sdscs(num_tiles=3),
        overlap_prefix=True,
    )
    root0 = artifacts[0]["mixed_flash_pipeline_tile_0"]
    assert len(root0["dscs_"]) == 1
    assert len(root0["datadscs_"]) == 1
    assert root0["flashAttentionPipeline_"]["overlap_prefix"] is True
    assert root0["flashAttentionPipeline_"]["overlap_candidate"] is True


def test_flash_pipeline_overlap_prefix_allows_lx_compute_without_transfer():
    artifacts = rz.build_flash_attention_pipeline_tile_artifacts(
        _fake_flash_pipeline_sdscs(num_tiles=3, lx_pinned=True),
        overlap_prefix=True,
    )
    root0 = artifacts[0]["mixed_flash_pipeline_tile_0"]
    assert len(root0["dscs_"]) == 1
    assert len(root0["datadscs_"]) == 1
    assert root0["flashAttentionPipeline_"]["overlap_prefix"] is True
    assert root0["flashAttentionPipeline_"]["overlap_candidate"] is True


def test_flash_pipeline_overlap_prefix_allows_non_ij_shape():
    artifacts = rz.build_flash_attention_pipeline_tile_artifacts(
        _fake_flash_pipeline_sdscs(
            num_tiles=3,
            lx_pinned=True,
            input_neighbor_transfer=True,
        ),
        overlap_prefix=True,
    )
    root0 = artifacts[0]["mixed_flash_pipeline_tile_0"]
    assert len(root0["dscs_"]) == 1
    assert len(root0["datadscs_"]) == 1
    assert root0["flashAttentionPipeline_"]["overlap_prefix"] is True
    assert root0["flashAttentionPipeline_"]["overlap_candidate"] is True


def test_flash_pipeline_overlap_prefix_rejects_mismatched_next_tile():
    sdscs = _fake_flash_pipeline_sdscs(
        num_tiles=3,
        lx_pinned=True,
        input_neighbor_transfer=True,
        ij_input_layout=True,
    )
    sdscs[1]["1_batchmatmul"]["dscs_"][0]["batchmatmul"]["N_"]["out_"] = 128
    artifacts = rz.build_flash_attention_pipeline_tile_artifacts(
        sdscs,
        overlap_prefix=True,
    )
    root0 = artifacts[0]["mixed_flash_pipeline_tile_0"]
    assert len(root0["dscs_"]) == 1
    assert len(root0["datadscs_"]) == 2
    assert root0["flashAttentionPipeline_"]["overlap_prefix"] is False
    assert root0["flashAttentionPipeline_"]["overlap_candidate"] is False
    assert root0["flashAttentionPipeline_"][
        "overlap_prefix_rejection_reasons"
    ] == ["next_tile_iter_sizes_mismatch"]


def _run_all():
    tests = sorted(
        (n, o) for n, o in globals().items()
        if n.startswith("test_") and callable(o)
    )
    fails = []
    for n, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            fails.append(n)
            print(f"FAIL {n}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {n}")
    print(f"\n{len(tests) - len(fails)}/{len(tests)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
