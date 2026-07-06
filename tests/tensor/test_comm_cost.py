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

"""Unit tests for the G3 communication cost model (comm_cost.py).

These tests are PURE PYTHON: they import comm_cost.py directly by file path so
they never touch the torch runtime (the package ``__init__`` imports torch).
They pin the MEASURED anchors and structural invariants from this session's
findings (validated against the seed model g1_ring_model.py):

  * F / band / LX-cap / same-core-free invariants.
  * C1: naive_all2all beats ring at 1 MiB/head (22.3 vs 57.7 us).
  * C2: the 5.2 MiB/head ring-vs-naive crossover.
  * C3: LX-cap interception (ring never wins for a <=2 MiB resident tile).
  * C6: reduce F-domination (plain (P-1)-hop ring vs log2 tree fold).
"""

import importlib.util
import math
import os
import sys

import pytest

# Load comm_cost.py directly, bypassing the torch-importing package __init__.
_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULE_PATH = os.path.normpath(
    os.path.join(_HERE, "..", "..", "torch_spyre", "_inductor", "comm_cost.py")
)
_spec = importlib.util.spec_from_file_location("comm_cost_under_test", _MODULE_PATH)
cc = importlib.util.module_from_spec(_spec)
sys.modules["comm_cost_under_test"] = cc
_spec.loader.exec_module(cc)

MiB = 1024 ** 2
KiB = 1024


# --------------------------------------------------------------------
# Calibration constants (MEASURED — pin them so a drift is caught).
# --------------------------------------------------------------------
def test_calibration_constants():
    assert cc.F == pytest.approx(7.3e-6)
    assert cc.RAW_RATE == pytest.approx(140e9)
    assert cc.LX_CAP == 2 * MiB
    assert cc.HBM_BW == pytest.approx(204.8e9)


# --------------------------------------------------------------------
# band(): keyed on per-link TRANSFER COUNT (occ), not burst size.
# --------------------------------------------------------------------
def test_band_table():
    # occ<=1: uniform shift, ~140 GB/s.
    assert cc.band(0) == pytest.approx(140e9)
    assert cc.band(1) == pytest.approx(140e9)
    # occ 2..9: scatter/all-to-all plateau, ~36 GB/s.
    assert cc.band(2) == pytest.approx(36e9)
    assert cc.band(9) == pytest.approx(36e9)
    # occ>9: strictly worse than 36 GB/s ("16/link is worse than 36").
    assert cc.band(16) < 36e9
    assert cc.band(16) == pytest.approx(36e9 * 9 / 16)
    # monotonically worse past the plateau.
    assert cc.band(32) < cc.band(16)


# --------------------------------------------------------------------
# link_occupancy: shortest-arc routing, same-core FREE, the anchor 16/link.
# --------------------------------------------------------------------
def test_same_core_is_free():
    # A pure self-map (every core -> itself) contributes NO links, NO cross.
    cmap = {i: {i} for i in range(8)}
    per, max_occ, n_cross, n_same = cc.link_occupancy(cmap, 8, group_size=8)
    assert max_occ == 0
    assert n_cross == 0
    assert n_same == 8
    assert per == {}


def test_anchor_link_occupancy_is_16():
    # The realized attention broadcast: 4 contiguous groups of 8, within-group
    # all-to-all. MEASURED: 32 same-core (free) + 224 cross-core, MAX occ 16.
    cmap = cc._synthesize_coord_map(4, 8)
    per, max_occ, n_cross, n_same = cc.link_occupancy(cmap, 32, group_size=8)
    assert max_occ == 16
    assert n_cross == 224
    assert n_same == 32


def test_all2all_max_occ_g8():
    assert cc._all2all_max_occ(8) == 16


# --------------------------------------------------------------------
# C1 — the anchor case: naive_all2all beats ring at 1 MiB/head.
# --------------------------------------------------------------------
def test_c1_attention_anchor_schedule_ranking():
    chunk = (1 * MiB) // 8
    scheds = cc._broadcast_schedules(chunk, 8)
    t = {k: v[0] * 1e6 for k, v in scheds.items()}
    assert t["naive_all2all_1exec"] == pytest.approx(22.3, abs=0.2)
    assert t["recursive_doubling"] == pytest.approx(28.5, abs=0.2)
    assert t["bidir_ring"] == pytest.approx(32.9, abs=0.2)
    assert t["ring_carousel"] == pytest.approx(57.7, abs=0.2)
    # naive is the fastest; ring is 2.6x slower.
    assert min(t, key=t.get) == "naive_all2all_1exec"
    assert t["ring_carousel"] / t["naive_all2all_1exec"] == pytest.approx(2.6, abs=0.1)


def test_c1_cost_picks_naive():
    edge = cc.CommEdge(
        comm_class=cc.COMM_CLASS_ALL_GATHER,
        operand_bytes=1 * MiB,
        group_count=4,
        group_size=8,
    )
    out = cc.cost(edge)
    assert out.schedule == "naive_all2all_1exec"
    assert out.time_us == pytest.approx(22.3, abs=0.2)
    assert out.executes == 1
    assert out.max_link_occupancy == 16
    assert out.n_tiles == 1
    # 1 MiB fits in LX, so highwater is the whole operand.
    assert out.lx_highwater == 1 * MiB
    # HBM round-trip elimination = write + read of the operand.
    assert out.dram_bytes_saved_vs_spill == 2 * MiB


def test_c1_execute_count_dominates():
    # F=7.3us is ~89% of a 128 KiB single-transfer time (execute count wins).
    chunk = 128 * KiB
    single_transfer = cc.F + chunk / cc.RAW_RATE
    assert cc.F / single_transfer == pytest.approx(0.89, abs=0.02)


# --------------------------------------------------------------------
# C2 — crossover: ring beats naive-1-exec only above 5.2 MiB/head.
# --------------------------------------------------------------------
def test_c2_crossover_5_2_mib():
    xover = cc.crossover_mib_per_head(8)
    assert xover == pytest.approx(5.20, abs=0.05)


def test_c2_naive_wins_below_crossover():
    # At 1 MiB and 2 MiB, naive still wins/ties vs ring_carousel.
    for operand in (1 * MiB, 2 * MiB):
        chunk = operand // 8
        scheds = cc._broadcast_schedules(chunk, 8)
        assert (
            scheds["naive_all2all_1exec"][0] < scheds["ring_carousel"][0]
        ), f"naive should beat ring at {operand} bytes"


# --------------------------------------------------------------------
# C3 — LX-capacity interception (the load-bearing structural result).
# --------------------------------------------------------------------
def test_c3_crossover_exceeds_lx_cap():
    # 5.2 MiB crossover > 2 MiB LX cap: the interception premise.
    assert cc.crossover_mib_per_head(8) * MiB > cc.LX_CAP


def test_c3_16mib_operand_is_tiled_to_lx_cap():
    edge = cc.CommEdge(
        comm_class=cc.COMM_CLASS_ALL_GATHER,
        operand_bytes=16 * MiB,
        group_count=4,
        group_size=8,
    )
    out = cc.cost(edge)
    # 16 MiB / 2 MiB = 8 tiles; highwater never exceeds the LX cap.
    assert out.n_tiles == 8
    assert out.lx_highwater == cc.LX_CAP


def test_c3_ring_never_wins_for_resident_tile():
    # Every LX-resident tile is <= 2 MiB < 5.2 MiB (the ring_carousel-vs-naive
    # crossover), so the bandwidth-optimal ring carousel is NEVER the cheapest
    # schedule for a resident tile. NB the per-tile winner is a low-execute
    # schedule -- naive below ~1.73 MiB, recursive_doubling in [1.73, 2.0] MiB --
    # not necessarily naive; assert the real theorem (carousel is never argmin).
    tile = 2 * MiB
    chunk = tile // 8
    scheds = cc._broadcast_schedules(chunk, 8)
    assert scheds["naive_all2all_1exec"][0] * 1e6 == pytest.approx(37.3, abs=0.3)
    assert scheds["ring_carousel"][0] * 1e6 == pytest.approx(64.2, abs=0.3)
    # The interception theorem: ring_carousel is not the argmin at any resident
    # tile size up to the LX cap.
    for mib in (0.25, 0.5, 1.0, 1.73, 2.0):
        sc = cc._broadcast_schedules(int(mib * MiB) // 8, 8)
        best = min(sc, key=lambda name: sc[name][0])
        assert best != "ring_carousel", f"carousel won at {mib} MiB tile"


# --------------------------------------------------------------------
# C6 — LSE reduce over P=32: F-domination; tree beats plain ring.
# --------------------------------------------------------------------
def test_c6_reduce_f_domination():
    payload = 16500  # ~16.5 KiB LSE merge, tiny vs F.
    P = 32
    scheds = cc._reduce_schedules(payload, P)
    plain = scheds["reduce_plain_ring"][0] * 1e6
    tree = scheds["reduce_tree_fold"][0] * 1e6
    # (P-1)=31 sequential hops, F-dominated (~226 us; payload adds a few us).
    assert scheds["reduce_plain_ring"][1] == 31
    assert plain == pytest.approx(226.3, abs=5.0)
    # log2(32)=5 rounds tree fold; ~6x faster.
    assert scheds["reduce_tree_fold"][1] == 5
    assert tree == pytest.approx(36.5, abs=2.0)
    assert plain / tree == pytest.approx(6.2, abs=0.5)
    # >= 98% of the plain-ring cost is fixed (F), not payload.
    fixed_fraction = (31 * cc.F) / scheds["reduce_plain_ring"][0]
    assert fixed_fraction >= 0.98


def test_c6_cost_picks_tree_fold():
    edge = cc.CommEdge(
        comm_class=cc.COMM_CLASS_REDUCE,
        operand_bytes=16500,
        group_count=1,
        group_size=32,
        payload_per_hop=16500,
        is_reduction=True,
    )
    out = cc.cost(edge)
    # The model's own gradient selects the fewer-hop tree fold.
    assert out.schedule == "reduce_tree_fold"
    assert out.executes == 5


# --------------------------------------------------------------------
# C4 / C5 — regime splits: no ring edge for streamed / seam-transparent.
# --------------------------------------------------------------------
def test_c4_tiny_q_broadcast_is_l_independent():
    # Q broadcast (~8 KiB) as a single-execute broadcast is F-dominated and
    # L-independent (~7.4 us).
    edge = cc.CommEdge(
        comm_class=cc.COMM_CLASS_BROADCAST,
        operand_bytes=8 * KiB,
        group_count=1,
        group_size=8,
    )
    out = cc.cost(edge)
    assert out.time_us == pytest.approx(7.4, abs=0.3)


def test_c5_no_communication_edge_is_zero():
    # An M-split activation handoff is seam-transparent: no cross-core movement.
    edge = cc.CommEdge(
        comm_class=cc.COMM_CLASS_UNSUPPORTED,
        operand_bytes=4 * MiB,
        group_count=1,
        group_size=1,
    )
    out = cc.cost(edge)
    assert out.time_us == 0.0
    assert out.executes == 0


# --------------------------------------------------------------------
# C7 — realized 16/link broadcast lands in the SLOW effective band.
# --------------------------------------------------------------------
def test_c7_occ16_is_slow_band():
    edge = cc.CommEdge(
        comm_class=cc.COMM_CLASS_ALL_GATHER,
        operand_bytes=1 * MiB,
        group_count=4,
        group_size=8,
    )
    out = cc.cost(edge)
    # occ=16 -> effective bandwidth strictly below the 36 GB/s plateau, far
    # below the 140 GB/s uniform-shift band. NOT ring-efficient.
    assert out.max_link_occupancy == 16
    assert out.effective_bw_bps < 36e9
    assert out.effective_bw_bps < cc.band(1)
    # The HBM-round-trip elimination (what the 1.053x actually banks) is a
    # SEPARATE resource, priced against the operand, not the ring links.
    assert out.dram_bytes_saved_vs_spill == 2 * MiB


# --------------------------------------------------------------------
# Seam invariant: comm_edge_cost_us is a plain additive scalar (us).
# --------------------------------------------------------------------
def test_comm_edge_cost_us_scalar():
    edge = cc.CommEdge(
        comm_class=cc.COMM_CLASS_ALL_GATHER,
        operand_bytes=1 * MiB,
        group_count=4,
        group_size=8,
    )
    val = cc.comm_edge_cost_us(edge)
    assert isinstance(val, float)
    assert val == pytest.approx(cc.cost(edge).time_us)


def test_t_step_matches_seed_model():
    # t_step = F + max_link_bytes / RAW_RATE (the g1 seed time model).
    chunk = 128 * KiB
    assert cc._t_step(16 * chunk) == pytest.approx(cc.F + 16 * chunk / cc.RAW_RATE)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
