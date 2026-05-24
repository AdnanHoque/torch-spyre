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

"""Standalone tests for the streamed cross-core bridge (slices > LX buffers).

onchip_realize.py and codegen/onchip_bridge.py are torch-free, so both are
loaded by file path (no torch_spyre import). Asserts: K = ceil(slice/tile) STCDP
data-ops; tiles partition rows with no gaps/overlap and reconstruct the slice;
the 2 fixed buffers fit LX; the schedule has K+1 rows with correct sync flags;
realize streams only when the single move won't fit and fail-closes otherwise.
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


for pkg in ("torch_spyre", "torch_spyre._inductor", "torch_spyre._inductor.codegen"):
    sys.modules.setdefault(pkg, types.ModuleType(pkg))
ob = _load(
    "torch_spyre._inductor.codegen.onchip_bridge",
    os.path.join(_CODEGEN, "onchip_bridge.py"),
)
rz = _load("torch_spyre._inductor.onchip_realize", _REAL)

# A 16384x2048 fp16 slab, out_ split 32 ways -> 16384 rows x 64 cols x 2B = 2 MB
# per core: bigger than the single 2-region move can fit -> stream in tiles.
_ITER = {"mb_": 16384, "out_": 2048}
_LAYOUT = ["mb_", "out_"]
_KW = dict(stick_size=64, num_cores=32, lx_size=ob.LX_CAPACITY_BYTES)


def _slice_bytes():
    return ob.per_core_slice_bytes(_ITER, "out_", 64, 32)


def test_num_tiles_is_ceil_slice_over_buffer():
    sb = _slice_bytes()
    assert sb == 16384 * 64 * 2  # 2 MB
    k = ob.num_stream_tiles(sb)
    assert k == -(-sb // ob.STREAM_TILE_BYTES) == 16


def test_build_streamed_bridge_emits_k_stcdp_dataops():
    sb = _slice_bytes()
    datadscs, opfuncs, _ = ob.build_streamed_bridge(
        dim_pool=_LAYOUT, iter_sizes=_ITER, src_base=0, dst_base=ob.STREAM_TILE_BYTES,
        layout=_LAYOUT, stick_dim="out_", src_split_dim="out_", dst_split_dim="out_",
        row_dim="mb_", slice_bytes=sb, **_KW,
    )
    k = ob.num_stream_tiles(sb)
    assert len(datadscs) == k and opfuncs == ["STCDPOpLx"] * k
    for j, dd in enumerate(datadscs):
        assert f"{j}_STCDPOpLx_dataop" in dd


def test_tiles_partition_rows_no_gaps_overlap():
    sb = _slice_bytes()
    datadscs, _, _ = ob.build_streamed_bridge(
        dim_pool=_LAYOUT, iter_sizes=_ITER, src_base=0, dst_base=ob.STREAM_TILE_BYTES,
        layout=_LAYOUT, stick_dim="out_", src_split_dim="out_", dst_split_dim="out_",
        row_dim="mb_", slice_bytes=sb, **_KW,
    )
    covered, expect = [], 0
    for dd in datadscs:
        in_p = dd[next(iter(dd))]["labeledDs_"][0]["PieceInfo"][0]
        r0 = in_p["dimToStartCordinate"]["mb_"]
        nr = in_p["dimToSize_"]["mb_"]
        assert r0 == expect  # contiguous, no gap
        covered.append((r0, r0 + nr))
        expect = r0 + nr
        # split dim full chunk every tile (sticks intact)
        assert in_p["dimToSize_"]["out_"] == 64
    assert covered[0][0] == 0 and covered[-1][1] == _ITER["mb_"]  # full slice


def test_buffers_fit_capacity():
    bases = ob.allocate_stream_bases()
    assert bases[0] == 0 and bases[1] == ob.STREAM_TILE_BYTES
    assert bases[1] + ob.STREAM_TILE_BYTES <= ob.LX_CAPACITY_BYTES


def test_schedule_has_k_plus_one_rows_with_sync_flags():
    sb = _slice_bytes()
    _, _, sched = ob.build_streamed_bridge(
        dim_pool=_LAYOUT, iter_sizes=_ITER, src_base=0, dst_base=ob.STREAM_TILE_BYTES,
        layout=_LAYOUT, stick_dim="out_", src_split_dim="out_", dst_split_dim="out_",
        row_dim="mb_", slice_bytes=sb, **_KW,
    )
    k = ob.num_stream_tiles(sb)
    rows = sched["0"]
    assert len(rows) == k + 1
    assert rows[0] == [0, -1, 0, 1]  # first tile: no after-sync, before-sync
    for j in range(1, k):
        assert rows[j] == [j, -1, 1, 1]  # reuse buffer only after prior drains
    assert rows[-1] == [-1, 0, 1, 0]  # DL op last


def test_memid_mirrors_single_move_per_tile():
    sb = _slice_bytes()
    datadscs, _, _ = ob.build_streamed_bridge(
        dim_pool=_LAYOUT, iter_sizes=_ITER, src_base=0, dst_base=ob.STREAM_TILE_BYTES,
        layout=_LAYOUT, stick_dim="out_", src_split_dim="out_", dst_split_dim="out_",
        row_dim="mb_", slice_bytes=sb, **_KW,
    )
    dd = datadscs[0][next(iter(datadscs[0]))]
    in_p, out_p = dd["labeledDs_"][0]["PieceInfo"], dd["labeledDs_"][1]["PieceInfo"]
    assert len(in_p) == 32 and len(out_p) == 32
    for i in range(32):  # same-shard: piece i on core i both sides, no ring
        assert in_p[i]["PlacementInfo"][0]["memId"] == [i]
        assert out_p[i]["PlacementInfo"][0]["memId"] == [i]


def test_realize_streams_when_single_move_wont_fit():
    r = rz.realize_streamed_handoff(
        iter_sizes=_ITER, layout=_LAYOUT, stick_dim="out_", split_dim="out_",
        stick_size=64, num_cores=32, producer_ldsidx=2, consumer_ldsidx=0,
    )
    assert r is not None and r.realizable
    assert r.num_tiles == 16
    assert len(r.datadscs) == 16 and r.opfuncs == ["STCDPOpLx"] * 16
    assert r.producer_base + r.tile_bytes <= rz.LX_CAPACITY_BYTES
    assert r.consumer_base + r.tile_bytes <= rz.LX_CAPACITY_BYTES


def test_realize_fails_closed_when_tiles_too_big():
    # tile_bytes set so 2 fixed tiles exceed the 2 MB LX -> None.
    r = rz.realize_streamed_handoff(
        iter_sizes=_ITER, layout=_LAYOUT, stick_dim="out_", split_dim="out_",
        stick_size=64, num_cores=32, producer_ldsidx=0, consumer_ldsidx=0,
        tile_bytes=2 << 20,
    )
    assert r is None


def test_single_move_fits_below_threshold():
    # 2048x2048/32 = 256 KB/region < half cap: single move regime, not stream.
    sb = ob.per_core_slice_bytes({"mb_": 2048, "out_": 2048}, "out_", 64, 32)
    assert sb <= rz.STREAM_THRESHOLD
    sb_big = ob.per_core_slice_bytes(_ITER, "out_", 64, 32)
    assert sb_big > rz.STREAM_THRESHOLD  # streams


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
