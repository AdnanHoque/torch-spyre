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
    layout_xform_pointwise_region0=None,
    causal_plan_artifact=False,
    ifn_prefix_force=False,
    execute_tile=-1,
    tile_artifacts=None,
):
    calls = {
        "layout_xform": [],
        "layout_xform_overlap": [],
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
    config.flash_attention_mixed_pipeline_overlap = False
    config.causal_idx_to_mask_plan_artifact = causal_plan_artifact

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
    layout_xform_pointwise_region0=None,
    causal_plan_artifact=False,
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
        layout_xform_pointwise_region0=layout_xform_pointwise_region0,
        causal_plan_artifact=causal_plan_artifact,
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
