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
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODEGEN = os.path.normpath(
    os.path.join(_HERE, "..", "..", "torch_spyre", "_inductor", "codegen")
)
_REAL = os.path.normpath(
    os.path.join(_HERE, "..", "..", "torch_spyre", "_inductor", "onchip_realize.py")
)


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
                "data_": {f"[{c}, 0, 0]": str(addr) for c in range(32)}
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
    return {
        f"{idx}_{op}": {
            "sdscFoldProps_": [{"factor_": 1, "label_": "time"}],
            "sdscFolds_": {
                "dim_prop_func": [{"Affine": {"alpha_": 1, "beta_": 0}}],
                "dim_prop_attr": [{"factor_": 1, "label_": "time"}],
                "data_": {"[0]": "0"},
            },
            "coreFoldProp_": {"factor_": 32, "label_": "core"},
            "coreletFoldProp_": {"factor_": 1, "label_": "corelet"},
            "numCoresUsed_": 32,
            "coreIdToDsc_": {str(c): 0 for c in range(32)},
            "numWkSlicesPerDim_": shard,
            "coreIdToWkSlice_": {
                str(c): {
                    dim: c if factor == 32 else 0
                    for dim, factor in shard.items()
                }
                for c in range(32)
            },
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
):
    input_layout = ["i", "j", "in"] if ij_input_layout else ["mb", "x", "in"]
    output_layout = ["i", "j", "out"] if ij_input_layout else ["mb", "x", "out"]
    n_sizes = (
        {"i_": 64, "j_": 2, "x_": 2, "out_": 192, "in_": 64}
        if ij_input_layout
        else {"x_": 2, "mb_": 96, "out_": 192, "in_": 64}
    )
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
    assert len(root0["datadscs_"]) == 4
    assert root0["opFuncsUsed_"] == ["STCDPOpLx"] * 4
    meta0 = root0["flashAttentionPipeline_"]
    assert meta0["source"] == "generated-flash-prefill-overlap-prefix-tile"
    assert meta0["tile_count"] == 1
    assert meta0["dataop_count"] == 4
    assert meta0["prefetch_tile_count"] == 2
    assert meta0["compute_tile_count"] == 1
    assert meta0["overlap_prefix"] is True
    assert meta0["overlap_candidate"] is True
    assert meta0["tile_index"] == 0
    assert meta0["replaces_sdsc"] == "0_batchmatmul"
    assert root0["coreIdToDscSchedule"]["0"] == [
        [0, -1, 0, 1],
        [1, -1, 1, 1],
        [2, 0, 1, 1],
        [3, -1, 1, 0],
    ]

    root2 = artifacts[2]["mixed_flash_pipeline_tile_2"]
    assert len(root2["dscs_"]) == 1
    assert len(root2["datadscs_"]) == 2
    assert root2["flashAttentionPipeline_"]["overlap_prefix"] is False


def test_flash_pipeline_overlap_prefix_rejects_hbm_backed_compute():
    artifacts = rz.build_flash_attention_pipeline_tile_artifacts(
        _fake_flash_pipeline_sdscs(num_tiles=3),
        overlap_prefix=True,
    )
    root0 = artifacts[0]["mixed_flash_pipeline_tile_0"]
    assert len(root0["dscs_"]) == 1
    assert len(root0["datadscs_"]) == 2
    assert root0["flashAttentionPipeline_"]["overlap_prefix"] is False
    assert root0["flashAttentionPipeline_"]["overlap_candidate"] is False
    assert root0["flashAttentionPipeline_"]["overlap_prefix_requested"] is True
    reasons = root0["flashAttentionPipeline_"][
        "overlap_prefix_rejection_reasons"
    ]
    assert set(reasons) >= {
        "compute_dsc:lds0_pinned_hbm",
        "compute_dsc:input_lds0_pinned_hbm",
        "compute_dsc:input_layout_missing_i_j",
        "compute_dsc:missing_no_component_to_lx_transfer_lds0",
    }


def test_flash_pipeline_overlap_prefix_rejects_lx_compute_without_transfer():
    artifacts = rz.build_flash_attention_pipeline_tile_artifacts(
        _fake_flash_pipeline_sdscs(num_tiles=3, lx_pinned=True),
        overlap_prefix=True,
    )
    root0 = artifacts[0]["mixed_flash_pipeline_tile_0"]
    assert len(root0["dscs_"]) == 1
    assert len(root0["datadscs_"]) == 2
    assert root0["flashAttentionPipeline_"]["overlap_prefix"] is False
    assert root0["flashAttentionPipeline_"]["overlap_candidate"] is False
    assert root0["flashAttentionPipeline_"][
        "overlap_prefix_rejection_reasons"
    ] == [
        "compute_dsc:input_layout_missing_i_j",
        "compute_dsc:missing_no_component_to_lx_transfer_lds0",
    ]


def test_flash_pipeline_overlap_prefix_rejects_non_ij_input_neighbor_shape():
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
    assert len(root0["datadscs_"]) == 2
    assert root0["flashAttentionPipeline_"]["overlap_prefix"] is False
    assert root0["flashAttentionPipeline_"]["overlap_candidate"] is False
    assert root0["flashAttentionPipeline_"][
        "overlap_prefix_rejection_reasons"
    ] == ["compute_dsc:input_layout_missing_i_j"]


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
