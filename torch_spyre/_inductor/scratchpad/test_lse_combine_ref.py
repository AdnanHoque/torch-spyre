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

"""Value-oracle tests for the LSE-combine fold operator (A2, G2 reduce lane).

Pure numpy; no torch. Run directly:  python3 test_lse_combine_ref.py
Asserts: associativity, commutativity-up-to-rounding, run-to-run bit
determinism (static ring order), fp32 carry vs fp16 drift, equivalence to
single-pass softmax over the union of key sets.
"""

from __future__ import annotations

import importlib.util
import itertools
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "lse_combine_ref", os.path.join(_HERE, "lse_combine_ref.py")
)
lse = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lse)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _rng(seed):
    return np.random.default_rng(seed)


def _make_shards(rng, n_shards, n_k, d_v, scale=8.0):
    """A list of (scores, values) shards over disjoint key subsets, one head-row.
    scale is deliberately large so the running-max rescale actually bites."""
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


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
def test_equivalence_to_single_pass():
    """Fold over disjoint shards == single-pass softmax over the union."""
    rng = _rng(0)
    for n_shards in (2, 3, 7, 32):
        shards = _make_shards(rng, n_shards, n_k=5, d_v=16)
        folded = lse.normalize(lse.fold(_partials(shards)))
        ref = lse.single_pass_softmax_attention(*_union(shards))
        # fp32 fold vs fp32 single-pass: agree to fp32 rounding.
        assert np.allclose(folded, ref, rtol=1e-5, atol=1e-6), (
            f"n_shards={n_shards} max|d|={np.abs(folded - ref).max()}"
        )
    print("PASS test_equivalence_to_single_pass")


def test_associativity():
    """(p1 . p2) . p3 == p1 . (p2 . p3) up to fp32 rounding, and the fully
    left-folded reduction matches an arbitrary parenthesization."""
    rng = _rng(1)
    shards = _make_shards(rng, 4, n_k=6, d_v=8)
    p = _partials(shards)

    left = lse.lse_combine(lse.lse_combine(lse.lse_combine(p[0], p[1]), p[2]), p[3])
    right = lse.lse_combine(p[0], lse.lse_combine(p[1], lse.lse_combine(p[2], p[3])))
    bal = lse.lse_combine(lse.lse_combine(p[0], p[1]), lse.lse_combine(p[2], p[3]))

    for other in (right, bal):
        assert np.allclose(lse.normalize(left), lse.normalize(other),
                           rtol=1e-5, atol=1e-6)
        assert np.allclose(left.m, other.m)
        assert np.allclose(left.l, other.l, rtol=1e-5, atol=1e-6)
    print("PASS test_associativity")


def test_commutativity():
    """lse_combine is commutative up to rounding; every fold order over the
    shards yields the same normalized output."""
    rng = _rng(2)
    shards = _make_shards(rng, 5, n_k=4, d_v=12)
    p = _partials(shards)
    base = lse.normalize(lse.fold(p, order=range(5)))
    # sample several permutations
    perms = list(itertools.islice(itertools.permutations(range(5)), 0, 40, 3))
    for perm in perms:
        got = lse.normalize(lse.fold(p, order=perm))
        assert np.allclose(base, got, rtol=1e-5, atol=1e-6), f"perm={perm}"
    # pairwise commute
    ab = lse.normalize(lse.lse_combine(p[0], p[1]))
    ba = lse.normalize(lse.lse_combine(p[1], p[0]))
    assert np.allclose(ab, ba, rtol=1e-6, atol=1e-7)
    print("PASS test_commutativity")


def test_run_to_run_bit_determinism():
    """A STATIC ring order gives bit-identical results run to run (this is the
    on-device determinism guarantee: same order => same rounding => same bits)."""
    rng = _rng(3)
    shards = _make_shards(rng, 8, n_k=5, d_v=16)
    p = _partials(shards)
    static_order = (0, 1, 2, 3, 4, 5, 6, 7)  # the fixed ring order
    r1 = lse.fold(p, order=static_order)
    r2 = lse.fold(p, order=static_order)
    # BITWISE identical (not just allclose): same order, same rounding.
    assert r1.A.tobytes() == r2.A.tobytes()
    assert r1.m.tobytes() == r2.m.tobytes()
    assert r1.l.tobytes() == r2.l.tobytes()
    o1, o2 = lse.normalize(r1), lse.normalize(r2)
    assert o1.tobytes() == o2.tobytes()
    print("PASS test_run_to_run_bit_determinism")


def test_fp32_carry_vs_fp16_drift():
    """The carry dtype is load-bearing. Fold in fp32 (the oracle) vs a fp16
    carry: fp16 drifts materially from the single-pass reference; fp32 does not.
    We construct shards with a wide max spread so the rescale factors span the
    fp16 subnormal/precision cliff."""
    rng = _rng(4)
    # Wide spread of per-shard maxima -> small exp(m_i - m) rescale factors.
    d_v = 32
    shards = []
    for i in range(6):
        base = 30.0 * i  # each shard's scores live at a very different level
        scores = (rng.standard_normal(4).astype(np.float32) + base)
        values = rng.standard_normal((4, d_v)).astype(np.float32)
        shards.append((scores, values))
    ref = lse.single_pass_softmax_attention(*_union(shards))

    # fp32 carry (the oracle): tight.
    fp32_out = lse.normalize(lse.fold(_partials(shards)))
    err_fp32 = np.abs(fp32_out - ref).max()

    # fp16 carry: rebuild the fold with everything squeezed through fp16.
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
        acc = combine_fp16(acc, {"m": np.float16(pp.m), "l": np.float16(pp.l),
                                 "A": pp.A.astype(np.float16)})
    fp16_out = (acc["A"].astype(np.float32) / np.float32(acc["l"]))
    err_fp16 = np.abs(fp16_out - ref).max()

    assert err_fp32 < 1e-3, f"fp32 carry should be tight, got {err_fp32}"
    assert err_fp16 > err_fp32 * 5, (
        f"fp16 carry should drift materially more than fp32: "
        f"fp16={err_fp16} fp32={err_fp32}"
    )
    print(f"PASS test_fp32_carry_vs_fp16_drift (fp32 err={err_fp32:.2e}, "
          f"fp16 err={err_fp16:.2e})")


def test_identity_and_empty_shard():
    """Folding the identity (empty key set) is a no-op; empty shards are skipped."""
    rng = _rng(5)
    shards = _make_shards(rng, 3, n_k=5, d_v=8)
    p = _partials(shards)
    ident = lse.Partial.identity(8)
    got = lse.lse_combine(p[0], ident)
    assert np.allclose(lse.normalize(got), lse.normalize(p[0]), rtol=0, atol=0)
    assert got.A.tobytes() == p[0].A.tobytes()  # exact no-op
    # empty shard produces identity
    empty = lse.make_partial(np.zeros(0, np.float32), np.zeros((0, 8), np.float32))
    assert not np.isfinite(empty.m) and empty.l == 0.0
    print("PASS test_identity_and_empty_shard")


def test_tree_fold_equals_linear_fold():
    """The COST winner is a log2-depth tree fold; assert it gives the SAME value
    as the linear chain (associativity => tree order is value-equivalent). This
    is the value guarantee the G3 reduce_tree_fold schedule relies on."""
    rng = _rng(6)
    P = 32
    shards = _make_shards(rng, P, n_k=4, d_v=16)
    p = _partials(shards)

    def tree(parts):
        cur = list(parts)
        while len(cur) > 1:
            nxt = []
            for i in range(0, len(cur) - 1, 2):
                nxt.append(lse.lse_combine(cur[i], cur[i + 1]))
            if len(cur) % 2 == 1:
                nxt.append(cur[-1])
            cur = nxt
        return cur[0]

    lin = lse.fold(p)
    trf = tree(p)
    assert np.allclose(lse.normalize(lin), lse.normalize(trf),
                       rtol=1e-5, atol=1e-6)
    ref = lse.single_pass_softmax_attention(*_union(shards))
    assert np.allclose(lse.normalize(trf), ref, rtol=1e-5, atol=1e-6)
    print("PASS test_tree_fold_equals_linear_fold")


if __name__ == "__main__":
    test_equivalence_to_single_pass()
    test_associativity()
    test_commutativity()
    test_run_to_run_bit_determinism()
    test_fp32_carry_vs_fp16_drift()
    test_identity_and_empty_shard()
    test_tree_fold_equals_linear_fold()
    print("\nALL 7 LSE-COMBINE ORACLE TESTS PASSED")
