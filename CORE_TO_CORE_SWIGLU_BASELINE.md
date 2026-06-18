# Core-to-core: Granite SwiGLU MLP baseline

Device-measured baseline of the Granite SwiGLU MLP (fused + unfused) to scope the
on-chip core-to-core reshard. Companion to the applicability analysis in the
`spyre-onchip-core-to-core` repo (`docs/07-swiglu-mlp-applicability.md`).

## Environment (what we actually ran on)

- **Code: `cf67411` (the `latest-main` branch) — the device-latest build.** It
  carries the always-on matmul cost model (`cost_model_matmul_division`,
  unconditional in `passes.py`) AND the flex-API adaptation needed to run on the
  harvest +292 SDK. A `_C.so` built with the profiler (`libaiupti`).
- **True upstream `3a1d9d9` does NOT build on this device:** flex-API skew — the
  harvest +292 flex ships the unified `launchOperation(...)`, but `3a1d9d9`'s csrc
  still calls the split `launchOperationH2D/D2H/Compute/HostCallback` (+
  `flex::ComputeParams`, non-const `getAllocator`). Building true upstream needs a
  flex split→unified port (a code change). The 62-commit `cf67411→3a1d9d9` gap is
  almost all **work-division *hints*** (#2416/#2613) + symbolic-batch (#2499) —
  off by default, so they do **not** change the concrete-shape no-hints SwiGLU
  worksplits below.
- **Run env:** `.venv` (torch 2.11) + harvest LD (`/home/adnan/opt-newer/{runtime,
  spyre-comms,deeptools,senlib}/lib`), `unset PYTHONPATH`, perf-suite
  `jamie/dev`, `python benchmark.py --op fms_granite_micro.swiglu[_unfused]
  --stack torch-spyre --shape 1 512 4096`.
- **Device caveat:** the profiler build triggers a flex *profiling-in-streams*
  thread-lock — `RuntimeStream::synchronize()` stalls ~60s ("lost completion").
  Runs with `--with-profiling` time out / crash; `--no-with-profiling` sidesteps
  it. There is also an intermittent start-up stall. So **wall-clock here is noisy
  and only meaningful as an A/B delta** (host stalls cancel); kernel-time needs
  the flex thread-lock fix (`tmhoangt/fix-profiling-in-streams`).

## Worksplits (device-confirmed, cost-model code)

Both fused and unfused: **matmuls split `(m4, n8)`** (`numWkSlicesPerDim_ =
{mb:4, out:8, in:1}`), **pointwise chain pure-M** (`{mb:32}`). So every
matmul→pointwise hand-off is a **cross-division, same-stick (`out`) edge**.

| | SDSCs | weight ReStickify (Class A) | matmuls | cross-div edge (Class C) |
|---|---|---|---|---|
| **fused** (`linear_mul_silu_split_with_sizes`) | 9 | **2** (gate+up combined `[4096,25600]`, down) | 2 (gate+up, down), `(m4,n8)` | `sdsc_1`→`sdsc_2` @ `0xc800000` |
| **unfused** (`linear_mul_silu` + separate down) | 11 (2 bundles) | **3** (gate, up, down separately) | 3 (gate, up, down), `(m4,n8)` | `sdsc_1`→`sdsc_2` @ `0x6400000` |

Fusion's win here is **one fewer weight relayout** (combined gate+up vs separate);
the pointwise LX-residency (`exp/add/realdiv` resident) is identical and is
main's LX_PLANNING, not ours.

## The edge classes for this MLP (see doc 07 for the 40-edge taxonomy)

- **(A) Weight relayout — `ReStickifyOpHBM`, 25 cores, HBM→HBM, per-forward.**
  Transposes the matmul weights to stationary layout. The byte-dominant cost
  (~52% bucket). Stick-orientation change → **cannot** be an on-chip move
  (`STCDPOpLx` is same-stick only); lever is **weight prelayout / freezing**
  (orthogonal to core-to-core).
- **(B) Same-division pointwise (silu chain).** `exp/add/realdiv` already
  LX-resident — **main's LX_PLANNING does this**; our symmetric pass is subsumed.
- **(C) Cross-division matmul→pointwise edge.** The on-chip reshard target — see
  below.

## The reshard target (Class C) — Phase-0 owner pin

The matmul→neg edge, same HBM tensor, same stick (`out`), different split:

- **Producer (matmul):** `{mb:4, out:8, in:1}`, owner(core) = **mb + 4·out**;
  each core owns a `128 rows × 3200 cols` tile of `[512, 25600]`.
- **Consumer (neg):** `{mb:32, out:1}`, owner(core) = c; each owns `16 rows ×
  full-12800-gate`.
- **Reshard map:** consumer core `c` ← producer cores `{c//8, c//8+4, c//8+8,
  c//8+12}` (mb-band `c//8`; the 4 `out`-bands spanning the gate half `[0,12800)`
  of the combined `25600`).
- **`in:1` ⇒ no K-reduction**, so producer owners are direct — **no rep-core
  ambiguity**, simpler than the documented granite `bmm{out:8,in:4}→mul{out:25}`
  edge that the `frontiers/asymmetric-reshard.md` plan worked on. This is exactly
  the `~250 LOC pure-Inductor` asymmetric same-stick reshard (feed
  `createSubPieces` the native unequal pieces); the broken `0b994bb` (max_err
  0.669) failed by *guessing* these owners — they are pinned above.

## Perf (cf67411; wall-clock is A/B-only per the device caveat)

| shape | op | metric | value |
|---|---|---|---|
| 1×512×4096 | fused | kernel_ms (profiler), pt_util | 16.3 ms, 17.1% |
| 1×512×4096 | fused | wall_clock (clean) | min 23.8 / median 31.0 ms (high variance — start-up stalls) |
| 1×512×4096 | unfused | wall_clock (clean) | min 23.2 / median 24.5 ms |

**Wall-clock fused ≈ unfused (≈23–24 ms at the stall-free min)** — the ~5%
structural difference (unfused's extra weight restickify) is **below this
device's noise floor**; resolving it needs kernel-time, which needs the flex
thread-lock fix. Kernel-time was only capturable for fused (16.3 ms) before the
profiler stall. (Decode `4×1×4096` shapes: pending.)

## Next

1. Clean fused wall-clock + the `4×1×4096` decode worksplits/perf.
2. Reshard A/B: keep `(m4,n8)` + on-chip reshard (the Phase-0 map above) vs.
   **steer the matmul to pure-M** (→ Class B, main's LX_PLANNING persists it
   free). The A/B decides whether the matmul-speed of `(m4,n8)` outweighs the
   cross-division hand-off it creates. Wall-clock deltas cancel the device noise.
3. Weight prelayout/freezing for the Class-A restickifies (the byte-dominant lever).
