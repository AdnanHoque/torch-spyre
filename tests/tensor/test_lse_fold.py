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

"""Value-oracle + cost-wiring tests for the LSE ring-fold REDUCE lane (G2).

PURE PYTHON: the modules under test are loaded by file path so they never touch
the torch runtime (the package ``__init__`` imports torch).  Covers:

  * the LSE-combine fold operator value oracle (lse_fold_ref):
    associativity, commutativity, run-to-run bit determinism (static ring
    order), fp32-carry vs fp16 drift, equivalence to single-pass softmax over
    the union of key sets, tree-fold == linear-fold.
  * the frontend contract builder (layout_allgather_restickify):
    make_lse_ring_fold_contract carries communication_class=reduce + the
    LSE-combine semantics + the L-independent payload triple.
  * the G3 cost wiring (comm_cost): a reduce edge prices reduce_tree_fold (not
    plain_ring) and reproduces the MEASURED ~37 us at P=32 anchor.
"""

from __future__ import annotations

import importlib.util
import itertools
import math
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_INDUCTOR = os.path.normpath(
    os.path.join(_HERE, "..", "..", "torch_spyre", "_inductor")
)


def _load(mod_name: str, filename: str):
    path = os.path.join(_INDUCTOR, filename)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


lse = _load("lse_fold_ref_under_test", "lse_fold_ref.py")
contract = _load(
    "layout_allgather_restickify_under_test", "layout_allgather_restickify.py"
)
cc = _load("comm_cost_under_test_lse", "comm_cost.py")


# ====================================================================
# helpers
# ====================================================================
def _rng(seed):
    return np.random.default_rng(seed)


def _make_shards(rng, n_shards, n_k, d_v, scale=8.0):
    shards = []
    for _ in range(n_shards):
        scores = rng.standard_normal(n_k).astype(np.float32) * scale
        values = rng.standard_normal((n_k, d_v)).astype(np.float32)
        shards.append((scores, values))
    return shards


def _partials(shards):
    return [lse.make_partial(s, v) for (s, v) in shards]


def _union(shards):
    scores = np.concatenate([s for (s, _) in shards])
    values = np.concatenate([v for (_, v) in shards], axis=0)
    return scores, values


# ====================================================================
# value oracle
# ====================================================================
def test_equivalence_to_single_pass():
    rng = _rng(0)
    for n_shards in (2, 3, 7, 32):
        shards = _make_shards(rng, n_shards, n_k=5, d_v=16)
        folded = lse.normalize(lse.fold(_partials(shards)))
        ref = lse.single_pass_softmax_attention(*_union(shards))
        assert np.allclose(folded, ref, rtol=1e-5, atol=1e-6), (
            f"n_shards={n_shards} max|d|={np.abs(folded - ref).max()}"
        )


def test_associativity():
    rng = _rng(1)
    shards = _make_shards(rng, 4, n_k=6, d_v=8)
    p = _partials(shards)
    left = lse.lse_combine(lse.lse_combine(lse.lse_combine(p[0], p[1]), p[2]), p[3])
    right = lse.lse_combine(p[0], lse.lse_combine(p[1], lse.lse_combine(p[2], p[3])))
    bal = lse.lse_combine(lse.lse_combine(p[0], p[1]), lse.lse_combine(p[2], p[3]))
    for other in (right, bal):
        assert np.allclose(
            lse.normalize(left), lse.normalize(other), rtol=1e-5, atol=1e-6
        )
        assert np.allclose(left.m, other.m)
        assert np.allclose(left.l, other.l, rtol=1e-5, atol=1e-6)


def test_commutativity():
    rng = _rng(2)
    shards = _make_shards(rng, 5, n_k=4, d_v=12)
    p = _partials(shards)
    base = lse.normalize(lse.fold(p, order=range(5)))
    perms = list(itertools.islice(itertools.permutations(range(5)), 0, 40, 3))
    for perm in perms:
        got = lse.normalize(lse.fold(p, order=perm))
        assert np.allclose(base, got, rtol=1e-5, atol=1e-6), f"perm={perm}"
    ab = lse.normalize(lse.lse_combine(p[0], p[1]))
    ba = lse.normalize(lse.lse_combine(p[1], p[0]))
    assert np.allclose(ab, ba, rtol=1e-6, atol=1e-7)


def test_run_to_run_bit_determinism():
    rng = _rng(3)
    shards = _make_shards(rng, 8, n_k=5, d_v=16)
    p = _partials(shards)
    static_order = (0, 1, 2, 3, 4, 5, 6, 7)
    r1 = lse.fold(p, order=static_order)
    r2 = lse.fold(p, order=static_order)
    assert r1.A.tobytes() == r2.A.tobytes()
    assert r1.m.tobytes() == r2.m.tobytes()
    assert r1.l.tobytes() == r2.l.tobytes()
    assert lse.normalize(r1).tobytes() == lse.normalize(r2).tobytes()


def test_fp32_carry_vs_fp16_drift():
    rng = _rng(4)
    d_v = 32
    shards = []
    for i in range(6):
        base = 30.0 * i
        scores = rng.standard_normal(4).astype(np.float32) + base
        values = rng.standard_normal((4, d_v)).astype(np.float32)
        shards.append((scores, values))
    ref = lse.single_pass_softmax_attention(*_union(shards))
    fp32_out = lse.normalize(lse.fold(_partials(shards)))
    err_fp32 = np.abs(fp32_out - ref).max()

    def combine_fp16(a, b):
        m = np.float16(max(a["m"], b["m"]))
        fa = np.float16(np.exp(np.float16(a["m"]) - m))
        fb = np.float16(np.exp(np.float16(b["m"]) - m))
        A = (fa * a["A"].astype(np.float16) + fb * b["A"].astype(np.float16)).astype(
            np.float16
        )
        l = np.float16(fa * np.float16(a["l"]) + fb * np.float16(b["l"]))
        return {"m": m, "l": l, "A": A}

    acc = {"m": np.float16(-np.inf), "l": np.float16(0.0),
           "A": np.zeros(d_v, np.float16)}
    for (s, v) in shards:
        pp = lse.make_partial(s, v)
        acc = combine_fp16(
            acc,
            {"m": np.float16(pp.m), "l": np.float16(pp.l),
             "A": pp.A.astype(np.float16)},
        )
    fp16_out = acc["A"].astype(np.float32) / np.float32(acc["l"])
    err_fp16 = np.abs(fp16_out - ref).max()
    assert err_fp32 < 1e-3, f"fp32 carry should be tight, got {err_fp32}"
    assert err_fp16 > err_fp32 * 5, (
        f"fp16 should drift more than fp32: fp16={err_fp16} fp32={err_fp32}"
    )


def test_identity_and_empty_shard():
    rng = _rng(5)
    shards = _make_shards(rng, 3, n_k=5, d_v=8)
    p = _partials(shards)
    ident = lse.Partial.identity(8)
    got = lse.lse_combine(p[0], ident)
    assert np.allclose(lse.normalize(got), lse.normalize(p[0]), rtol=0, atol=0)
    assert got.A.tobytes() == p[0].A.tobytes()
    empty = lse.make_partial(np.zeros(0, np.float32), np.zeros((0, 8), np.float32))
    assert not np.isfinite(empty.m) and empty.l == 0.0


def test_tree_fold_equals_linear_fold():
    rng = _rng(6)
    P = 32
    shards = _make_shards(rng, P, n_k=4, d_v=16)
    p = _partials(shards)
    lin = lse.fold(p)
    trf = lse.tree_fold(p)
    assert np.allclose(lse.normalize(lin), lse.normalize(trf), rtol=1e-5, atol=1e-6)
    ref = lse.single_pass_softmax_attention(*_union(shards))
    assert np.allclose(lse.normalize(trf), ref, rtol=1e-5, atol=1e-6)


# ====================================================================
# payload sizing helpers
# ====================================================================
def test_lse_partial_nbytes_is_L_independent():
    # d_v=64 fp32 + m,l scalars, one head-row -> (2+64)*4 = 264 B.
    assert lse.lse_partial_nbytes(64) == 264
    # L-independent: bytes do not depend on key-tile count, only d_v/heads.
    assert lse.lse_partial_nbytes(64, n_head_rows=4) == 264 * 4


def test_reduce_tree_hops_matches_comm_cost():
    for p in (1, 2, 3, 4, 8, 16, 32):
        assert lse.reduce_tree_hops(p) == (
            0 if p < 2 else math.ceil(math.log2(p))
        )


# ====================================================================
# contract builder
# ====================================================================
def test_contract_reduce_carries_lse_semantics():
    c = contract.make_lse_ring_fold_contract(
        producer_name="buf_flash_partial",
        consumer_name="buf_out_proj_in",
        reduce_axis="Lk",
        scatter_axis="head",
        participant_count=8,
        d_v=64,
        n_head_rows=1,
        all_reduce=False,
    )
    assert c["communication_class"] == contract.COMM_CLASS_REDUCE
    assert c["communication_pattern"] == "lse_ring_fold"
    assert c["reduce_op"] == "lse_combine"
    assert c["is_reduction"] is True
    assert c["fp32_carry"] is True
    assert c["static_ring_order"] is True
    # payload is the L-independent (m, l, A) triple.
    assert c["payload"]["estimated_tensor_bytes"] == 264
    assert c["payload"]["A_shape"] == [1, 64]
    assert c["payload"]["carry_dtype"] == "float32"
    # reduce (not all_reduce) => result scattered over heads (layout dividend).
    assert c["reduce_axis"] == "Lk"
    assert c["scatter_axis"] == "head"
    # target schedule is the F-dominated tree fold.
    assert c["target_schedule"] == "reduce_tree_fold"
    assert c["tree_fold_hops"] == math.ceil(math.log2(8))  # 3


def test_contract_all_reduce_class():
    c = contract.make_lse_ring_fold_contract(
        producer_name="p",
        consumer_name="c",
        reduce_axis="Lk",
        scatter_axis=None,
        participant_count=32,
        d_v=64,
        all_reduce=True,
    )
    assert c["communication_class"] == contract.COMM_CLASS_ALL_REDUCE
    assert c["tree_fold_hops"] == 5  # ceil(log2 32)


def test_contract_rejects_bad_participant_count():
    with pytest.raises(ValueError):
        contract.make_lse_ring_fold_contract(
            producer_name="p", consumer_name="c", reduce_axis="Lk",
            scatter_axis="head", participant_count=0, d_v=64,
        )


# ====================================================================
# G3 cost wiring — the reduce edge prices reduce_tree_fold (F-dominated).
# ====================================================================
def _reduce_edge(payload_bytes, P):
    return cc.CommEdge(
        comm_class=cc.COMM_CLASS_REDUCE,
        operand_bytes=payload_bytes,
        group_count=1,
        group_size=P,
        payload_per_hop=payload_bytes,
        is_reduction=True,
    )


def test_reduce_prices_tree_fold_at_P32_brief_anchor():
    # Brief MEASURED anchor: P=32, ~16.5 KiB payload -> tree_fold ~37 us,
    # plain_ring ~230 us, and cost() selects tree_fold.
    payload = 16896  # ~16.5 KiB
    edge = _reduce_edge(payload, P=32)
    out = cc.cost(edge)
    assert out.schedule == "reduce_tree_fold"
    assert out.executes == 5  # ceil(log2 32)
    assert out.time_us == pytest.approx(37.10, abs=0.5)

    scheds = cc._reduce_schedules(payload, 32)
    plain = scheds["reduce_plain_ring"][0] * 1e6
    tree = scheds["reduce_tree_fold"][0] * 1e6
    assert plain == pytest.approx(230.04, abs=1.0)
    assert tree == pytest.approx(37.10, abs=0.5)
    # tree fold strictly wins (6.2x).
    assert plain / tree > 6.0


def test_flash_lk_reduce_real_payload_is_F_dominated():
    # Real LSE payload: d_v=64 fp32 + m,l = 264 B, reduce over the Lk split
    # factor T (NOT a 32-core ring).  Tree fold, F-dominated.
    payload = lse.lse_partial_nbytes(64)  # 264 B
    assert payload == 264
    edge = _reduce_edge(payload, P=8)  # T=8-way Lk split
    out = cc.cost(edge)
    assert out.schedule == "reduce_tree_fold"
    assert out.executes == 3  # ceil(log2 8)
    # per hop = F + payload/RAW_RATE; the transfer term is negligible (F-dom).
    per_hop = cc.F + payload / cc.RAW_RATE
    assert out.time_us == pytest.approx(3 * per_hop * 1e6, rel=1e-6)
    # transfer term is <1% of a hop -> F-DOMINATED.
    assert (payload / cc.RAW_RATE) / per_hop < 0.01


def test_participant_count_discipline_split_dim_only():
    # Reducing over ONLY the split factor T is far cheaper than a 32-core ring.
    payload = 264
    t_edge = cc.cost(_reduce_edge(payload, P=4))       # T=4-way split
    ring32 = cc._reduce_schedules(payload, 32)["reduce_plain_ring"][0] * 1e6
    assert t_edge.schedule == "reduce_tree_fold"
    assert t_edge.executes == 2  # ceil(log2 4)
    # a full-32-core plain ring would be >10x the T=4 tree fold.
    assert ring32 / t_edge.time_us > 10.0


def test_decode_merge_prices_via_tree_fold():
    # The sequence-sharded DECODE merge (KV-carousel) is the same reduce shape
    # with P = seq-shard count; it also prices via reduce_tree_fold.
    payload = lse.lse_partial_nbytes(64)
    edge = _reduce_edge(payload, P=16)  # 16 sequence shards
    out = cc.cost(edge)
    assert out.schedule == "reduce_tree_fold"
    assert out.executes == 4  # ceil(log2 16)


if __name__ == "__main__":
    import sys as _sys

    _sys.exit(pytest.main([__file__, "-v"]))
