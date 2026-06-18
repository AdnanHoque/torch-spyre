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

## Perf — device kernel time (the only metric we trust)

Method: `torch.profiler` with `ProfilerActivity.PrivateUse1`, reading
`self_device_time_total` per kernel (`<1%` noise). Requires the
`USE_SPYRE_PROFILER=1` build (verified `nm -D _C.so | grep SpyreActivityProfiler`
present on the latest-main tree) + harvest libs + `.venv` torch 2.11. Wall-clock
is discarded — host-side, polluted by the flex profiling-in-streams 60 s sync
stall. `kernel_ms` = Σ device kernel time; `PT-util` = PT-array active fraction.

| shape | op | kernel_ms | PT-util |
|---|---|---|---|
| prefill 1×512×4096 | **fused** | 19.8 | 16.9% |
| prefill 1×512×4096 | **unfused** | **13.9** | 20.1% |
| decode 4×1×4096 | **fused** | 13.2 | 0.20% |
| decode 4×1×4096 | **unfused** | **8.07** | 0.22% |

### Findings

1. **Unfused beats fused in BOTH regimes — consistently** (prefill 19.8→13.9 =
   1.42×; decode 13.2→8.07 = 1.63×; higher PT-util unfused, 20.1 vs 16.9). The
   `linear_mul_silu_split_with_sizes` fusion is **counterproductive on Spyre**:
   the combined gate+up matmul `[4096,25600]` + `split_with_sizes` is
   scheduled/utilised worse than two separate `[4096,12800]` matmuls, and that
   outweighs fusion's one-fewer-weight-restickify saving. *Actionable
   independent of core-to-core: prefer the unfused SwiGLU lowering.*
2. **Decode runs at ~0.2% PT-util** — the array is essentially **idle** (decode
   M=4 → tiny matmul). So decode kernel time is **almost entirely data movement**
   (restickify + cross-division round-trips), making decode the **prime target**
   for the reshard/prelayout levers.
3. **Both prefill and decode are Class C** (cross-division matmul→pointwise). In
   decode the matmul is still `(m4,n8)` and feeds a pointwise split `out/25`
   (25 cores) — same-stick, different shard → the reshard applies to decode too.

## Next

1. **Reshard A/B (kernel-time):** keep `(m4,n8)` + on-chip reshard (the Phase-0
   map above) vs. **steer the matmul to pure-M** (→ Class B, main's LX_PLANNING
   persists it free). Decides whether `(m4,n8)`'s matmul-speed outweighs the
   cross-division hand-off it creates. Measure with the profiler kernel-time
   recipe above.
2. **Weight prelayout/freezing** for the Class-A restickifies (byte-dominant
   lever; orthogonal to core-to-core).
3. **Unfused vs fused** is a free, independent win (1.4–1.6× kernel) — worth a
   separate look at why the fused lowering schedules worse.
4. The **decode 0.2%-util / movement-bound** regime is where data-movement
   elimination should pay off most.
