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

"""NumPy validators for the weight- and KV-carousel work-division plans.

No device, no torch. Pure NumPy. These prove the two carousel schedules are
data-movement rewrites, not numeric changes: each computes the same numbers as
an ordinary local matmul / single-pass softmax attention, and the answer does
not depend on the plan knobs (replication radius r, K-slab K_t, KV block B_kv,
core placement, or ring fold order).

Run directly to execute every check and print PASS/FAIL:

    python3 reference.py

Two references live here:

  weight carousel  Y[S,N] = X[S,K] @ W[K,N], output M-split across P cores.
                   Naive M-split replicates W to every core through DRAM. The
                   carousel loads each W N-tile once and rotates the tiles
                   around the ring, so every core still computes its full-K
                   local matmul but sees each N-tile exactly once over P steps.

  KV carousel      decode attention with the KV cache sharded along the
                   sequence across all P cores. Each core runs a flash pass
                   over its owned key blocks and emits an unnormalized partial
                   (A, l, m); a ring fold with the LSE-combine operator merges
                   the partials; normalize gives the same output as a single
                   softmax over all keys.
"""

from __future__ import annotations

import numpy as np

try:  # package import when loaded as torch_spyre..._inductor.carousel.reference
    from . import fold
except ImportError:  # script import: `python3 reference.py` (same-dir fold.py)
    import fold

# AIU defaults (constants for this hardware; overridable per check).
P_CORES = 32  # cores per accelerator
H_KV = 8  # KV heads (GQA); query heads are a multiple of this


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _shard_ranges(size: int, parts: int):
    """Split [0, size) into `parts` contiguous shards (front shards absorb the
    remainder). Returns (lo, hi) pairs -- the head-axis shards the ring
    reduce-scatter owns one-per-core."""
    base, rem = divmod(size, parts)
    ranges, start = [], 0
    for s in range(parts):
        stop = start + base + (1 if s < rem else 0)
        ranges.append((start, stop))
        start = stop
    return ranges


def _pad_to(x: np.ndarray, axis: int, mult: int) -> np.ndarray:
    """Zero-pad `axis` up to a multiple of `mult`. Zeros are seam-transparent:
    padded rows/cols produce padded (zero) output that is sliced back off."""
    n = x.shape[axis]
    pad = _ceil_div(n, mult) * mult - n
    if pad == 0:
        return x
    width = [(0, 0)] * x.ndim
    width[axis] = (0, pad)
    return np.pad(x, width)


# ======================================================================
# weight carousel
# ======================================================================


def rotation_schedule(p_prime: int) -> np.ndarray:
    """schedule[step, core] = index of the N-tile that core holds at `step`.

    A ring rotation: the tiles shift one core position per step, so over
    p_prime steps every core sees every tile exactly once."""
    core = np.arange(p_prime)
    return np.stack([(core + step) % p_prime for step in range(p_prime)])


def tiled_matmul(
    x: np.ndarray,
    w: np.ndarray,
    p_cores: int = P_CORES,
    r: int = 1,
    k_t: int | None = None,
) -> np.ndarray:
    """Ordinary local matmul under the carousel's M/N/K tiling, NO rotation.

    Same slices and same K_t-slab accumulation order as the carousel, so it is
    the bit-exact ground truth that isolates the rotation schedule. Matching the
    N-tiling matters: a narrow BLAS tile rounds a K-dot differently from one wide
    matmul, so a full-N reference is only close, not bit-exact."""
    s, k = x.shape
    n = w.shape[1]
    assert p_cores % r == 0, "r must divide the core count"
    p_prime = p_cores // r
    if k_t is None:
        k_t = k
    xp = _pad_to(x, 0, p_cores)
    wp = _pad_to(w, 1, p_prime)
    rows = xp.shape[0] // p_cores
    tile = wp.shape[1] // p_prime
    y = np.zeros((xp.shape[0], wp.shape[1]), dtype=x.dtype)
    for gp in range(p_cores):
        for t in range(p_prime):
            xb = xp[gp * rows : (gp + 1) * rows, :]
            wt = wp[:, t * tile : (t + 1) * tile]
            acc = np.zeros((rows, tile), dtype=x.dtype)
            for k0 in range(0, k, k_t):
                acc += xb[:, k0 : k0 + k_t] @ wt[k0 : k0 + k_t, :]
            y[gp * rows : (gp + 1) * rows, t * tile : (t + 1) * tile] = acc
    return y[:s, :n]


def carousel_matmul(
    x: np.ndarray,
    w: np.ndarray,
    p_cores: int = P_CORES,
    r: int = 1,
    k_t: int | None = None,
) -> np.ndarray:
    """Simulate the N-tile rotation and return Y[S, N].

    M-split: S rows split into `p_cores` blocks, one per core.
    Group carousel: the ring is cut into `r` arcs of P' = p_cores // r cores;
    W is replicated once per arc (DRAM = r*|W|), and inside each arc the full N
    is cut into P' tiles that rotate among the arc's P' cores. r == p_cores is
    full replication (today's plan); r == 1 is the full carousel.

    Each core accumulates its (M-block, N-tile) cell over K_t slabs in the same
    order as `tiled_matmul`, so the rotation is bit-for-bit inert."""
    s, k = x.shape
    n = w.shape[1]
    assert p_cores % r == 0, "r must divide the core count"
    p_prime = p_cores // r
    if k_t is None:
        k_t = k

    xp = _pad_to(x, 0, p_cores)  # M-split needs S divisible by p_cores
    wp = _pad_to(w, 1, p_prime)  # N-tiling needs N divisible by p_prime
    rows = xp.shape[0] // p_cores
    tile = wp.shape[1] // p_prime
    y = np.zeros((xp.shape[0], wp.shape[1]), dtype=x.dtype)

    sched = rotation_schedule(p_prime)
    for arc in range(r):
        for step in range(p_prime):
            for c in range(p_prime):
                held = sched[step, c]  # which N-tile this core holds now
                gp = arc * p_prime + c  # global core -> its M-block
                xb = xp[gp * rows : (gp + 1) * rows, :]
                wt = wp[:, held * tile : (held + 1) * tile]
                acc = np.zeros((rows, tile), dtype=x.dtype)
                for k0 in range(0, k, k_t):  # local K-loop, core-resident
                    acc += xb[:, k0 : k0 + k_t] @ wt[k0 : k0 + k_t, :]
                y[gp * rows : (gp + 1) * rows, held * tile : (held + 1) * tile] = acc
    return y[:s, :n]


# ======================================================================
# flash partial + LSE ring fold  (KV carousel)
# ======================================================================


def flash_partial(q: np.ndarray, k_blk: np.ndarray, v_blk: np.ndarray, scale: float):
    """One core's unnormalized partial over its key block: (A, l, m).

    m running max, l = sum exp(score - m) mass, A = (exp(score - m)) @ V.
    An empty block returns the identity partial (A=0, l=0, m=-inf)."""
    nq = q.shape[0]
    d = v_blk.shape[1] if v_blk.ndim == 2 else q.shape[1]
    if k_blk.shape[0] == 0:
        return (
            np.zeros((nq, d), q.dtype),
            np.zeros(nq, q.dtype),
            np.full(nq, -np.inf, q.dtype),
        )
    sc = (q @ k_blk.T) * scale  # [nq, Lb]
    m = sc.max(axis=1)
    p = np.exp(sc - m[:, None])
    return p @ v_blk, p.sum(axis=1), m


def lse_combine(pa, pb):
    """Associative LSE merge of two partials. fp32 carry, static order in the
    caller -> deterministic. Leading batch dims (e.g. GQA G,nq) broadcast."""
    a1, l1, m1 = pa
    a2, l2, m2 = pb
    m = np.maximum(m1, m2)
    with np.errstate(invalid="ignore"):  # -inf minus -inf on empty partials
        c1 = np.where(np.isneginf(m1), 0.0, np.exp(m1 - m))
        c2 = np.where(np.isneginf(m2), 0.0, np.exp(m2 - m))
    a = c1[..., None] * a1 + c2[..., None] * a2
    return a, c1 * l1 + c2 * l2, m


def fold_linear(partials, order):
    """Sequential ring fold in the given order (topology A: P-1 hops)."""
    acc = partials[order[0]]
    for i in order[1:]:
        acc = lse_combine(acc, partials[i])
    return acc


def fold_tree(partials):
    """Recursive-halving fold (topology C: log2 rounds)."""
    cur = list(partials)
    while len(cur) > 1:
        cur = [
            lse_combine(cur[i], cur[i + 1]) if i + 1 < len(cur) else cur[i]
            for i in range(0, len(cur), 2)
        ]
    return cur[0]


def fold_reduce_scatter(partials, p_cores: int):
    """Ring reduce-scatter + all-gather fold (topology B), simulating the actual
    cross-core movement rather than a shortcut linear fold.

    P cores sit on a ring; core c sends only to its neighbor (c+1) mod P. Each
    core starts holding its own full partial, split into P head-shards. The fold
    runs in two phases, every hop distance-1:

      reduce-scatter  P-1 combining rounds. Round t: every core forwards one
                      head-shard to its right neighbor, which LSE-combines it.
                      After P-1 rounds each core owns exactly one FULLY reduced
                      head-shard -- the head-split layout the O-projection wants.
      all-gather      P-1 move-only rounds that circulate the reduced shards so
                      the head-gathered output can be assembled.

    Returns ((a, l, m), shard_owner, hops):
      (a, l, m)    the head-gathered merged partial (equals a dense fold),
      shard_owner  shard_owner[s] = core that owns fully-reduced shard s,
      hops         [(kind, distance)] one entry per ring round (all distance 1).
    """
    p = p_cores
    assert len(partials) == p, "one partial per core"
    heads = partials[0][0].shape[0]
    ranges = _shard_ranges(heads, p)

    # state[c][s] = core c's running (a, l, m) for head-shard s.
    # prov[c][s]  = set of source cores folded into it (self-describing owner).
    state = [
        [(a[lo:hi].copy(), mass[lo:hi].copy(), m[lo:hi].copy()) for (lo, hi) in ranges]
        for (a, mass, m) in partials
    ]
    prov = [[{c} for _ in range(p)] for c in range(p)]

    hops = []
    # --- reduce-scatter: P-1 neighbor combining rounds ---
    for t in range(p - 1):
        # All P links fire in one round; collect then apply (reads see the
        # pre-round state, and each (dst, shard) is written exactly once).
        writes = {}
        for c in range(p):
            dst = (c + 1) % p
            s = (c - t) % p
            writes[(dst, s)] = (
                lse_combine(state[dst][s], state[c][s]),
                prov[dst][s] | prov[c][s],
            )
        for (dst, s), (val, pv) in writes.items():
            state[dst][s], prov[dst][s] = val, pv
        hops.append(("reduce_scatter", 1))

    # Each shard is fully reduced on exactly one core (provenance == all cores):
    # that permutation IS the head-split endpoint.
    shard_owner = [None] * p
    for s in range(p):
        owners = [c for c in range(p) if len(prov[c][s]) == p]
        assert len(owners) == 1, f"shard {s} not owned by exactly one core"
        shard_owner[s] = owners[0]

    # --- all-gather: P-1 neighbor move-only rounds (no combine) ---
    for _ in range(p - 1):
        hops.append(("all_gather", 1))

    a_full = np.empty_like(partials[0][0])
    l_full = np.empty_like(partials[0][1])
    m_full = np.empty_like(partials[0][2])
    for s, (lo, hi) in enumerate(ranges):
        a_s, l_s, m_s = state[shard_owner[s]][s]
        a_full[lo:hi], l_full[lo:hi], m_full[lo:hi] = a_s, l_s, m_s
    return (a_full, l_full, m_full), shard_owner, hops


def normalize(a: np.ndarray, mass: np.ndarray) -> np.ndarray:
    return a / mass[..., None]


def softmax_attention(q, k, v, scale):
    """Single-pass reference: O = softmax(scale * Q K^T) V (max-shifted)."""
    sc = (q @ k.T) * scale
    sc = sc - sc.max(axis=1, keepdims=True)
    e = np.exp(sc)
    return (e / e.sum(axis=1, keepdims=True)) @ v


def blocks_of(seq_len: int, b_kv: int):
    """Contiguous key blocks of length B_kv (last one may be short)."""
    return [(i, min(i + b_kv, seq_len)) for i in range(0, seq_len, b_kv)]


def static_placement(seq_len: int, b_kv: int, p_cores: int):
    """core(pos) = (pos // B_kv) mod P: which core owns each sequence block.

    Returns (blocks, order) where `order` walks the blocks core-by-core along
    the ring -- the fold order the sequence-sharded plan actually produces."""
    blks = blocks_of(seq_len, b_kv)
    owner = [(j % p_cores) for j in range(len(blks))]
    order = sorted(range(len(blks)), key=lambda j: (owner[j], j))
    return blks, order


def sharded_attention(q, k, v, scale, blocks, order=None, tree=False):
    """Flash partial per block + LSE fold + normalize."""
    partials = [flash_partial(q, k[s:e], v[s:e], scale) for (s, e) in blocks]
    if tree:
        a, mass, _ = fold_tree(partials)
    else:
        if order is None:
            order = list(range(len(partials)))
        a, mass, _ = fold_linear(partials, list(order))
    return normalize(a, mass)


# ======================================================================
# GQA: each stored KV head streamed once, applied to its G query heads
# ======================================================================


def gqa_attention(q, k, v, scale, blocks):
    """q:[Hq,nq,d]  k,v:[Hkv,Lk,d].  Returns (O, reads) where reads[h] counts
    how many times KV head h's blocks were streamed. A KV block is read ONCE
    per query group of G=Hq/Hkv heads, not once per query head."""
    hq, hkv = q.shape[0], k.shape[0]
    g = hq // hkv
    reads = np.zeros(hkv, dtype=int)
    out = np.zeros_like(q)
    for h in range(hkv):
        qg = q[h * g : (h + 1) * g]  # [G, nq, d]
        partials = []
        for (s, e) in blocks:
            reads[h] += 1  # one stream feeds all G heads
            kb, vb = k[h, s:e], v[h, s:e]
            sc = np.einsum("gqd,ld->gql", qg, kb) * scale  # [G, nq, Lb]
            m = sc.max(axis=2)
            p = np.exp(sc - m[..., None])
            partials.append((np.einsum("gql,ld->gqd", p, vb), p.sum(axis=2), m))
        a, mass, _ = fold_linear(partials, list(range(len(partials))))
        out[h * g : (h + 1) * g] = normalize(a, mass)
    return out, reads


def gqa_reference(q, k, v, scale):
    hq, hkv = q.shape[0], k.shape[0]
    g = hq // hkv
    out = np.zeros_like(q)
    for h in range(hq):
        out[h] = softmax_attention(q[h], k[h // g], v[h // g], scale)
    return out


# ======================================================================
# checks
# ======================================================================


def check_carousel_tile_exact():
    """Rotation is numerically inert: carousel == same-tiling local matmul
    bit-for-bit, and both equal the plain matmul up to fp rounding."""
    rng = np.random.default_rng(1)
    p = 8
    s, k, n = 4 * p, 128, 6 * p  # divisible: isolates the schedule
    x = rng.standard_normal((s, k)).astype(np.float32)
    w = rng.standard_normal((k, n)).astype(np.float32)
    for r in (1, 2, 4, 8):
        for k_t in (k, k // 2, k // 4):
            car = carousel_matmul(x, w, p_cores=p, r=r, k_t=k_t)
            ref = tiled_matmul(x, w, p_cores=p, r=r, k_t=k_t)
            assert np.array_equal(car, ref), f"not bit-exact r={r} k_t={k_t}"
            assert np.allclose(car, x @ w, rtol=1e-3, atol=1e-4)


def check_carousel_schedule_cover():
    """The rotation schedule is a permutation cover: every core sees every
    N-tile exactly once. Combinatorial, BLAS-independent."""
    for p_prime in (1, 2, 3, 5, 8):
        sched = rotation_schedule(p_prime)
        assert sched.shape == (p_prime, p_prime)
        for c in range(p_prime):  # each core, over all steps, sees all tiles
            assert sorted(sched[:, c]) == list(range(p_prime))
        for step in range(p_prime):  # each step is a permutation (no collision)
            assert sorted(sched[step, :]) == list(range(p_prime))


def check_carousel_r_invariance():
    """Result is invariant to replication radius r: the r=P full-replication
    plan and the r=1 full carousel are endpoints of one plan family. Different
    r means different N-tile widths, so this is close (BLAS rounds narrow tiles
    differently), not bit-exact."""
    rng = np.random.default_rng(2)
    p = 12
    s, k, n = 3 * p, 96, 12  # n divisible by every p_prime = p//r
    x = rng.standard_normal((s, k)).astype(np.float32)
    w = rng.standard_normal((k, n)).astype(np.float32)
    base = carousel_matmul(x, w, p_cores=p, r=1, k_t=k)
    for r in (1, 2, 3, 4, 6, 12):
        got = carousel_matmul(x, w, p_cores=p, r=r, k_t=k)
        assert np.allclose(got, base, rtol=1e-3, atol=1e-4), f"r={r}"


def check_carousel_kt_invariance():
    """K-slab size is a transport knob: it regroups the K sum, so the result
    matches only up to fp rounding (not bit-exact), which is expected."""
    rng = np.random.default_rng(3)
    p = 8
    s, k, n = 2 * p, 256, 4 * p
    x = rng.standard_normal((s, k)).astype(np.float32)
    w = rng.standard_normal((k, n)).astype(np.float32)
    gold = x @ w
    for k_t in (k, k // 2, k // 4, k // 8):
        got = carousel_matmul(x, w, p_cores=p, r=2, k_t=k_t)
        assert np.allclose(got, gold, rtol=1e-3, atol=1e-4), f"k_t={k_t}"


def check_carousel_padding():
    """Non-divisible S and N handled by a zero-padded last tile."""
    rng = np.random.default_rng(4)
    p = 8
    s, k, n = 37, 100, 45  # neither S nor N divides the core / tile count
    x = rng.standard_normal((s, k)).astype(np.float32)
    w = rng.standard_normal((k, n)).astype(np.float32)
    got = carousel_matmul(x, w, p_cores=p, r=2, k_t=k // 2)
    assert got.shape == (s, n)
    assert np.allclose(got, x @ w, rtol=1e-3, atol=1e-4)


def check_flash_matches_softmax():
    """Flash-partial + LSE fold == single-pass softmax attention."""
    rng = np.random.default_rng(5)
    nq, lk, d = 5, 200, 32
    scale = 1.0 / np.sqrt(d)
    q = rng.standard_normal((nq, d)).astype(np.float32)
    k = rng.standard_normal((lk, d)).astype(np.float32)
    v = rng.standard_normal((lk, d)).astype(np.float32)
    ref = softmax_attention(q, k, v, scale)
    got = sharded_attention(q, k, v, scale, blocks_of(lk, 64))
    assert np.allclose(got, ref, rtol=1e-4, atol=1e-5)


def check_fold_bit_determinism():
    """A fixed static ring order folds bit-identically every run; a different
    order agrees only up to fp rounding (associativity up to rounding)."""
    rng = np.random.default_rng(6)
    nq, lk, d = 4, 130, 16
    scale = 1.0 / np.sqrt(d)
    q = rng.standard_normal((nq, d)).astype(np.float32)
    k = rng.standard_normal((lk, d)).astype(np.float32)
    v = rng.standard_normal((lk, d)).astype(np.float32)
    blks = blocks_of(lk, 16)
    order = list(range(len(blks)))
    a = sharded_attention(q, k, v, scale, blks, order=order)
    b = sharded_attention(q, k, v, scale, blks, order=order)
    assert np.array_equal(a, b), "fixed order not bit-deterministic"
    rev = sharded_attention(q, k, v, scale, blks, order=order[::-1])
    assert np.allclose(a, rev, rtol=1e-4, atol=1e-5)  # close, not bit-exact


def check_fold_order_invariance():
    """Linear, reversed, random, and recursive-halving folds all agree with
    the single-pass reference -> merge is order-invariant up to rounding."""
    rng = np.random.default_rng(7)
    nq, lk, d = 6, 300, 24
    scale = 1.0 / np.sqrt(d)
    q = rng.standard_normal((nq, d)).astype(np.float32)
    k = rng.standard_normal((lk, d)).astype(np.float32)
    v = rng.standard_normal((lk, d)).astype(np.float32)
    ref = softmax_attention(q, k, v, scale)
    blks = blocks_of(lk, 32)
    idx = list(range(len(blks)))
    orders = [idx, idx[::-1], list(rng.permutation(len(blks)))]
    for o in orders:
        assert np.allclose(
            sharded_attention(q, k, v, scale, blks, order=o), ref, rtol=1e-4, atol=1e-5
        )
    assert np.allclose(
        sharded_attention(q, k, v, scale, blks, tree=True), ref, rtol=1e-4, atol=1e-5
    )


def check_block_placement_invariance():
    """The static placement function core(pos)=(pos//B)%P only reorders the
    fold; the merged output is invariant to it."""
    rng = np.random.default_rng(8)
    nq, lk, d = 4, 260, 16
    scale = 1.0 / np.sqrt(d)
    q = rng.standard_normal((nq, d)).astype(np.float32)
    k = rng.standard_normal((lk, d)).astype(np.float32)
    v = rng.standard_normal((lk, d)).astype(np.float32)
    ref = softmax_attention(q, k, v, scale)
    for p_cores in (4, 8, 32):
        blks, order = static_placement(lk, 32, p_cores)
        got = sharded_attention(q, k, v, scale, blks, order=order)
        assert np.allclose(got, ref, rtol=1e-4, atol=1e-5), f"P={p_cores}"


def check_bkv_invariance():
    """Result is invariant to the KV block size B_kv (only the merge grouping
    changes; the softmax over all keys does not)."""
    rng = np.random.default_rng(9)
    nq, lk, d = 5, 384, 32
    scale = 1.0 / np.sqrt(d)
    q = rng.standard_normal((nq, d)).astype(np.float32)
    k = rng.standard_normal((lk, d)).astype(np.float32)
    v = rng.standard_normal((lk, d)).astype(np.float32)
    ref = softmax_attention(q, k, v, scale)
    for b_kv in (1, 16, 64, 128, 384):
        got = sharded_attention(q, k, v, scale, blocks_of(lk, b_kv))
        assert np.allclose(got, ref, rtol=1e-4, atol=1e-5), f"B_kv={b_kv}"


def check_adversarial_spread():
    """Max-shift stress: logits spanning ~+/-80 across blocks. The online LSE
    merge stays finite and matches an fp64 single-pass reference, while a naive
    non-shifted fp32 sum-of-exp overflows -- so the stress is real."""
    rng = np.random.default_rng(10)
    nq, lk, d = 3, 128, 8
    scale = 1.0
    q = (rng.standard_normal((nq, d)) * 6.0).astype(np.float32)
    k = (rng.standard_normal((lk, d)) * 6.0).astype(np.float32)
    v = rng.standard_normal((lk, d)).astype(np.float32)
    logits = (q @ k.T) * scale
    assert np.abs(logits).max() > 60.0, "stress not extreme enough"

    ref = softmax_attention(
        q.astype(np.float64), k.astype(np.float64), v.astype(np.float64), scale
    )
    got = sharded_attention(q, k, v, scale, blocks_of(lk, 16))
    assert np.isfinite(got).all(), "online merge not finite under stress"
    assert np.allclose(got, ref, rtol=1e-2, atol=1e-3)

    with np.errstate(over="ignore"):  # naive (no max shift) blows up
        naive = np.exp(logits.astype(np.float32))
    assert not np.isfinite(naive).all(), "expected naive path to overflow"


def check_gqa_each_head_once():
    """GQA: each KV head's blocks are streamed exactly once (shared across its
    G query heads), and the fused pass matches per-head reference attention."""
    rng = np.random.default_rng(11)
    hkv, g, nq, lk, d = H_KV, 4, 3, 192, 16
    hq = hkv * g
    scale = 1.0 / np.sqrt(d)
    q = rng.standard_normal((hq, nq, d)).astype(np.float32)
    k = rng.standard_normal((hkv, lk, d)).astype(np.float32)
    v = rng.standard_normal((hkv, lk, d)).astype(np.float32)
    blks = blocks_of(lk, 48)
    out, reads = gqa_attention(q, k, v, scale, blks)
    nblk = len(blks)
    assert (reads == nblk).all(), "each KV head must stream each block once"
    assert reads.sum() == hkv * nblk  # not hq * nblk
    assert reads.sum() < hq * nblk  # strictly cheaper than per-query-head
    assert np.allclose(out, gqa_reference(q, k, v, scale), rtol=1e-4, atol=1e-5)


def check_fold_reduce_scatter_matches_softmax():
    """Cross-core NEIGHBOR reduce-scatter fold (topology B): P cores each own a
    contiguous KV shard and emit a full-head partial; a ring reduce-scatter folds
    them and a move-only all-gather assembles the output. Asserts:
      (a) the merged output equals a single-pass softmax over all keys,
      (b) EVERY ring round is a distance-1 neighbor move -- the ~130 GB/s uniform
          shift band, never a distance>1 fold-to-one that drops to the floor,
      (c) the pre-gather endpoint is head-split: each core owns exactly one
          fully-reduced head-shard (shard_owner is a permutation), the layout the
          O-projection consumes relayout-free.
    The frontend hop schedule (fold._hops_for) must agree on both counts."""
    rng = np.random.default_rng(12)
    nq, lk, d, p = 16, 208, 32, 8
    scale = 1.0 / np.sqrt(d)
    q = rng.standard_normal((nq, d)).astype(np.float32)
    k = rng.standard_normal((lk, d)).astype(np.float32)
    v = rng.standard_normal((lk, d)).astype(np.float32)
    ref = softmax_attention(q, k, v, scale)

    # Shard the KV sequence contiguously across the p cores; each core's partial
    # spans all nq query rows (the head axis the fold reduce-scatters).
    shards = np.array_split(np.arange(lk), p)
    partials = [flash_partial(q, k[idx], v[idx], scale) for idx in shards]
    (a, mass, _), shard_owner, hops = fold_reduce_scatter(partials, p)

    # (a) numeric parity with a dense softmax over all keys.
    assert np.allclose(normalize(a, mass), ref, rtol=1e-4, atol=1e-5), "!= softmax"

    # (b) all rounds distance-1, and the two phases have the P-1 + P-1 shape.
    assert all(dist == 1 for (_, dist) in hops), "fold not all distance-1"
    rs = [h for h in hops if h[0] == "reduce_scatter"]
    ag = [h for h in hops if h[0] == "all_gather"]
    assert len(rs) == p - 1 and len(ag) == p - 1, "wrong hop counts"

    # (c) head-split endpoint: owners are a permutation of the shards.
    assert sorted(shard_owner) == list(range(p)), "endpoint not head-split"

    # The frontend schedule/cost port (fold.py) must describe the same topology.
    payload = int(sum(x.nbytes for x in partials[0]))
    sched, endpoint = fold._hops_for(fold.B_REDUCE_SCATTER, p, payload)
    assert endpoint == "head_split", endpoint
    assert all(h.distance == 1 for h in sched), "fold.py schedule not distance-1"
    assert len(sched) == 2 * (p - 1), "fold.py schedule wrong length"


CHECKS = [
    ("carousel_tile_exact", check_carousel_tile_exact),
    ("carousel_schedule_cover", check_carousel_schedule_cover),
    ("carousel_r_invariance", check_carousel_r_invariance),
    ("carousel_kt_invariance", check_carousel_kt_invariance),
    ("carousel_padding", check_carousel_padding),
    ("flash_matches_softmax", check_flash_matches_softmax),
    ("fold_bit_determinism", check_fold_bit_determinism),
    ("fold_order_invariance", check_fold_order_invariance),
    ("block_placement_invariance", check_block_placement_invariance),
    ("bkv_invariance", check_bkv_invariance),
    ("adversarial_spread", check_adversarial_spread),
    ("gqa_each_head_once", check_gqa_each_head_once),
    ("fold_reduce_scatter", check_fold_reduce_scatter_matches_softmax),
]


def main() -> int:
    failures = 0
    for name, fn in CHECKS:
        try:
            fn()
        except Exception as exc:  # report every check, do not abort the suite
            failures += 1
            print(f"FAIL  {name}: {exc!r}")
        else:
            print(f"PASS  {name}")
    total = len(CHECKS)
    print(f"\n{total - failures}/{total} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
