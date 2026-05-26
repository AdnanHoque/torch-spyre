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

"""Standalone tests for the mixed-SDSC flash-attention pipeline proof.

The proof is descriptor-only: it validates double-buffer allocation, tiled
STCDPOpLx prefetch data-ops, and the serial/overlap schedule shapes without
requiring torch or a device runtime.
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


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


for pkg in ("torch_spyre", "torch_spyre._inductor", "torch_spyre._inductor.codegen"):
    sys.modules.setdefault(pkg, types.ModuleType(pkg))
ob = _load(
    "torch_spyre._inductor.codegen.onchip_bridge",
    os.path.join(_CODEGEN, "onchip_bridge.py"),
)

_ITER = {"q_": 1024, "kv_": 512}
_LAYOUT = ["q_", "kv_"]
_KW = dict(stick_size=64, num_cores=32, lx_size=ob.LX_CAPACITY_BYTES)
_TILE_BYTES = 8 << 10


def _alloc():
    return ob.allocate_flash_attention_pipeline_bases(
        num_lanes=2,
        tile_bytes=_TILE_BYTES,
        scratch_regions=2,
        region0=16 << 10,
    )


def _bridge(overlap=False):
    bases = _alloc()
    return ob.build_flash_attention_pipeline_bridge(
        dim_pool=_LAYOUT,
        iter_sizes=_ITER,
        src_bases=bases["source_bases"],
        dst_lane_bases=bases["lane_bases"],
        layout=_LAYOUT,
        stick_dim="kv_",
        split_dim="q_",
        row_dim="kv_",
        lane_names=["k", "v"],
        tile_bytes=_TILE_BYTES,
        overlap=overlap,
        **_KW,
    )


def _bridge_on_corelet(corelet_id):
    bases = _alloc()
    return ob.build_flash_attention_pipeline_bridge(
        dim_pool=_LAYOUT,
        iter_sizes=_ITER,
        src_bases=bases["source_bases"],
        dst_lane_bases=bases["lane_bases"],
        layout=_LAYOUT,
        stick_dim="kv_",
        split_dim="q_",
        row_dim="kv_",
        lane_names=["k", "v"],
        tile_bytes=_TILE_BYTES,
        overlap=True,
        stcdp_corelet_id=corelet_id,
        **_KW,
    )


def _dataop_body(dataop):
    return dataop[next(iter(dataop))]


def _compute_dscs(count=4):
    return [
        {
            f"flash_tile_{tile}": {
                "computeOp_": [{"opFuncName": f"flash_tile_{tile}"}],
            }
        }
        for tile in range(count)
    ]


def test_allocate_flash_attention_pipeline_bases_fits_double_buffered_kv():
    bases = _alloc()
    assert len(bases["source_bases"]) == 2
    assert len(bases["lane_bases"]) == 2
    assert all(len(lane) == 2 for lane in bases["lane_bases"])
    assert len(bases["scratch_bases"]) == 2
    assert bases["tile_bytes"] == _TILE_BYTES
    assert bases["footprint"] <= ob.LX_CAPACITY_BYTES


def test_allocate_flash_attention_pipeline_bases_rejects_over_capacity():
    try:
        ob.allocate_flash_attention_pipeline_bases(
            num_lanes=2,
            tile_bytes=1 << 20,
            scratch_regions=1,
        )
    except ValueError as exc:
        assert "exceeds per-core LX capacity" in str(exc)
    else:
        raise AssertionError("expected over-capacity allocation to fail")


def test_build_flash_attention_pipeline_bridge_emits_kv_prefetch_dataops():
    datadscs, opfuncs, _ = _bridge()
    # q chunk = 1024/32, kv stick = 512 => 32 * 512 * 2 = 32 KiB/core.
    # With 8 KiB staging tiles this becomes 4 tiles, each with K and V lanes.
    assert len(datadscs) == 8
    assert opfuncs == ["STCDPOpLx"] * 8
    assert "0_STCDPOpLx_prefetch_k_tile0" in datadscs[0]
    assert "1_STCDPOpLx_prefetch_v_tile0" in datadscs[1]
    assert "2_STCDPOpLx_prefetch_k_tile1" in datadscs[2]
    assert "coreletId" not in _dataop_body(datadscs[0])["op"]


def test_build_flash_attention_pipeline_bridge_can_target_prefetch_corelet():
    datadscs, _, _ = _bridge_on_corelet(1)
    assert _dataop_body(datadscs[0])["op"] == {
        "name": "STCDPOpLx",
        "coreletId": 1,
    }
    assert _dataop_body(datadscs[-1])["op"]["coreletId"] == 1


def test_flash_pipeline_tiles_partition_row_dim_and_alternate_buffers():
    bases = _alloc()
    datadscs, _, _ = _bridge()

    first_k = _dataop_body(datadscs[0])["labeledDs_"][1]["PieceInfo"][0]
    second_k = _dataop_body(datadscs[2])["labeledDs_"][1]["PieceInfo"][0]
    third_k = _dataop_body(datadscs[4])["labeledDs_"][1]["PieceInfo"][0]

    assert first_k["dimToStartCordinate"]["kv_"] == 0
    assert second_k["dimToStartCordinate"]["kv_"] == 128
    assert third_k["dimToStartCordinate"]["kv_"] == 256
    assert first_k["dimToSize_"]["kv_"] == 128
    assert second_k["dimToSize_"]["kv_"] == 128

    k_buf0, k_buf1 = bases["lane_bases"][0]
    assert first_k["PlacementInfo"][0]["startAddr"] == [k_buf0]
    assert second_k["PlacementInfo"][0]["startAddr"] == [k_buf1]
    assert third_k["PlacementInfo"][0]["startAddr"] == [k_buf0]


def test_flash_pipeline_serial_schedule_runs_prefetch_then_compute_per_tile():
    _, _, sched = _bridge(overlap=False)
    rows = sched["0"]
    assert rows == [
        [0, -1, 0, 1],
        [1, -1, 1, 1],
        [-1, 0, 1, 1],
        [2, -1, 1, 1],
        [3, -1, 1, 1],
        [-1, 1, 1, 1],
        [4, -1, 1, 1],
        [5, -1, 1, 1],
        [-1, 2, 1, 1],
        [6, -1, 1, 1],
        [7, -1, 1, 1],
        [-1, 3, 1, 0],
    ]


def test_flash_pipeline_overlap_schedule_pairs_prefetch_with_compute():
    _, _, sched = _bridge(overlap=True)
    rows = sched["0"]
    assert rows == [
        [0, -1, 0, 1],
        [1, -1, 1, 1],
        [2, 0, 1, 1],
        [3, -1, 1, 1],
        [4, 1, 1, 1],
        [5, -1, 1, 1],
        [6, 2, 1, 1],
        [7, -1, 1, 1],
        [-1, 3, 1, 0],
    ]


def test_flash_pipeline_overlap_prefix_schedule_stays_one_compute():
    sched = ob.flash_pipeline_overlap_prefix_schedule(num_lanes=2, num_cores=32)
    assert sched["0"] == [
        [0, -1, 0, 1],
        [1, -1, 1, 1],
        [2, 0, 1, 1],
        [3, -1, 1, 0],
    ]


def test_flash_pipeline_mixed_sdsc_wraps_compute_tiles():
    datadscs, opfuncs, sched = _bridge(overlap=True)
    mixed = ob.build_flash_attention_pipeline_mixed_sdsc(
        "flash_pipeline",
        datadscs,
        opfuncs,
        sched,
        _compute_dscs(),
        num_cores=32,
    )
    root = mixed["flash_pipeline"]
    assert root["numCoresUsed_"] == 32
    assert len(root["datadscs_"]) == 8
    assert len(root["dscs_"]) == 4
    assert root["opFuncsUsed_"] == ["STCDPOpLx"] * 8
    assert root["flashAttentionPipeline_"] == {
        "tile_count": 4,
        "dataop_count": 8,
        "overlap_candidate": True,
    }
    assert root["coreIdToDscSchedule"]["0"][2] == [2, 0, 1, 1]


def test_flash_pipeline_mixed_sdsc_rejects_bad_schedule_refs():
    datadscs, opfuncs, sched = _bridge(overlap=False)
    bad = {core: [list(row) for row in rows] for core, rows in sched.items()}
    bad["0"][-1] = [-1, 99, 1, 0]
    try:
        ob.build_flash_attention_pipeline_mixed_sdsc(
            "flash_pipeline",
            datadscs,
            opfuncs,
            bad,
            _compute_dscs(),
            num_cores=32,
        )
    except ValueError as exc:
        assert "compute DSC index out of range" in str(exc)
    else:
        raise AssertionError("expected invalid compute DSC reference to fail")


def test_flash_pipeline_rejects_row_dim_equal_to_split_dim():
    bases = _alloc()
    try:
        ob.build_flash_attention_pipeline_bridge(
            dim_pool=_LAYOUT,
            iter_sizes=_ITER,
            src_bases=bases["source_bases"],
            dst_lane_bases=bases["lane_bases"],
            layout=_LAYOUT,
            stick_dim="kv_",
            split_dim="q_",
            row_dim="q_",
            lane_names=["k", "v"],
            tile_bytes=_TILE_BYTES,
            **_KW,
        )
    except ValueError as exc:
        assert "row_dim must differ" in str(exc)
    else:
        raise AssertionError("expected invalid row_dim to fail")


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
