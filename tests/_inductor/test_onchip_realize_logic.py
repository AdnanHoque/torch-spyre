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
):
    def lds(label, role):
        i = int(label.rsplit("-idx", 1)[1])
        return {
            "ldsIdx_": i,
            "dsName_": f"Tensor{i}",
            "dsType_": role,
            "wordLength": 2,
            "dataFormat_": "SEN169_FP16",
            "memOrg_": {"hbm": {"isPresent": 1}, "lx": {"isPresent": 1}},
        }

    def alloc(label, addr):
        i = int(label.rsplit("-idx", 1)[1])
        return {
            "nodeType_": "allocate",
            "name_": f"allocate-Tensor{i}_hbm",
            "ldsIdx_": i,
            "component_": "hbm",
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
    return {
        f"{idx}_{op}": {
            "numCoresUsed_": 32,
            "numWkSlicesPerDim_": shard,
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
