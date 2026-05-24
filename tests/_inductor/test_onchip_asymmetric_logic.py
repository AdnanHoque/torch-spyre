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

"""Standalone tests for the asymmetric N->M same-stick reshard bridge.

onchip_realize.py + codegen/onchip_bridge.py are torch-free, loaded by file path
(no torch_spyre import). Asserts: producer and consumer partitions each tile the
stick dim with no gaps/overlap; the single STCDPOpLx datadsc carries N producer
pieces in dataIN and M consumer pieces in dataOUT; bands cross cores (DCG cells
ride the ring); sub-stick partitions fail-closed. The 8->25 granite bmm->mul edge
is the motivating case; the equal-cell builder is the special case.
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

_ITER = {"out_": 1600, "mb_": 64}
_LAYOUT = ["out_", "mb_"]


def _tiles_exactly(owners, starts, lens, length):
    assert starts[0] == 0
    assert starts[-1] + lens[-1] == length
    for i in range(1, len(starts)):
        assert starts[i] == starts[i - 1] + lens[i - 1]  # no gap, no overlap
    assert sum(lens) == length
    assert len(set(owners)) == len(owners)  # each piece a distinct core


def test_uniform_partition_tiles_no_gaps_overlap():
    for n in (8, 25):
        part = rz.uniform_partition(1600, n, 64)
        assert part is not None
        _tiles_exactly(*part, 1600)
        assert all(x % 64 == 0 for x in part[2])  # whole-stick


def test_partition_fails_closed_when_not_whole_stick():
    assert rz.uniform_partition(100, 8, 64) is None  # 100 not stick-aligned


def test_asymmetric_emits_single_stcdp_with_n_and_m_pieces():
    prod = rz.uniform_partition(1600, 8, 64)
    cons = rz.uniform_partition(1600, 25, 64)
    datadscs, opfuncs, sched = ob.build_asymmetric_reshard_bridge(
        dim_pool=_LAYOUT, iter_sizes=_ITER, stick_size=64, num_cores=32,
        lx_size=ob.LX_CAPACITY_BYTES, src_base=0, dst_base=1 << 20,
        layout=_LAYOUT, stick_dim="out_",
        prod_owners=prod[0], prod_starts=prod[1], prod_lens=prod[2],
        cons_owners=cons[0], cons_starts=cons[1], cons_lens=cons[2],
    )
    assert opfuncs == ["STCDPOpLx"] and len(datadscs) == 1
    dd = datadscs[0]["0_STCDPOpLx_dataop"]
    in_p, out_p = dd["labeledDs_"][0]["PieceInfo"], dd["labeledDs_"][1]["PieceInfo"]
    assert len(in_p) == len(prod[0]) and len(out_p) == 25  # N != M
    assert len(sched["0"]) == 2  # one data-op + DL op


def test_emitted_pieces_tile_tensor_and_cross_cores():
    prod = rz.uniform_partition(1600, 8, 64)
    cons = rz.uniform_partition(1600, 25, 64)
    datadscs, _, _ = ob.build_asymmetric_reshard_bridge(
        dim_pool=_LAYOUT, iter_sizes=_ITER, stick_size=64, num_cores=32,
        lx_size=ob.LX_CAPACITY_BYTES, src_base=0, dst_base=1 << 20,
        layout=_LAYOUT, stick_dim="out_",
        prod_owners=prod[0], prod_starts=prod[1], prod_lens=prod[2],
        cons_owners=cons[0], cons_starts=cons[1], cons_lens=cons[2],
    )
    dd = datadscs[0]["0_STCDPOpLx_dataop"]
    for side in dd["labeledDs_"]:
        starts = [p["dimToStartCordinate"]["out_"] for p in side["PieceInfo"]]
        lens = [p["dimToSize_"]["out_"] for p in side["PieceInfo"]]
        cov = sorted(zip(starts, lens))
        exp = 0
        for s, ln in cov:
            assert s == exp  # no gap/overlap
            exp = s + ln
        assert exp == 1600  # full tensor
    cons_owner = {p["PlacementInfo"][0]["memId"][0]
                  for p in dd["labeledDs_"][1]["PieceInfo"]}
    prod_owner = {p["PlacementInfo"][0]["memId"][0]
                  for p in dd["labeledDs_"][0]["PieceInfo"]}
    assert cons_owner != prod_owner  # genuine cross-core gather


def test_realize_asymmetric_two_regions_fit():
    r = rz.realize_asymmetric_handoff(
        iter_sizes=_ITER, layout=_LAYOUT, stick_dim="out_", prod_n=8, cons_n=25,
        stick_size=64, num_cores=32, producer_ldsidx=2, consumer_ldsidx=1,
    )
    assert r is not None and r.realizable
    assert r.opfuncs == ["STCDPOpLx"]
    assert r.producer_base != r.consumer_base
    assert r.consumer_base + r.slice_bytes <= rz.LX_CAPACITY_BYTES


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
