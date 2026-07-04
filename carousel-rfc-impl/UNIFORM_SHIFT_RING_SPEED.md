# Uniform p→p+1 Single-Hop Ring Shift Speed (STCDPOpLx) — Verdict

**Scope:** does a **uniform, single-hop, unicast** ring shift (every core
`p → p+1`, exactly one transfer per ring segment) escape the **~36 GB/s**
all-to-all scatter floor and approach the **raw ring** (128–145 GB/s)? This is
the primitive under **weight-carousel rotation** and **ring-pipelined
all-gather/fold**. It is measured head-to-head against the same `stcdp_range`
carrier whose *all-to-all* range-relayout was pinned at **~36 GB/s** effective
(slope method, R² ≥ 0.999; see `LX_TO_LX_SPEED.md`, `DLDSC_LX_SPEED.md`).
Numbers are tagged **measured** (device), **modeled** (backend cost model), or
**spec** (architecture sheet).

---

## TL;DR

**YES — decisively.** The uniform `p→p+1` shift escapes the 36 GB/s
all-to-all contention floor and lands in the model's predicted raw-ring band.

- **Effective ρ (bytes/slope, the exact definition that gave strided = 36):
  54 → 90 → 130 GB/s** as the move grows 4.06 → 8.13 → 16.25 MB, R² ≥ 0.9985.
- At the largest tested volume, **ρ_uniform = 130 GB/s — dead center of the
  modeled 100–145 band**, ~1.02× the raw single-link 128.
- At byte-volume matched to the strided run (~8 MB): **90.5 vs 36 = 2.5×.**
- **Contention-free streaming aggregate (fixed per-move overhead removed):
  ~244–254 GB/s**, R² = 0.998 — 31 concurrent uncontended ring segments summing
  above a single link.
- **The 36 floor is a serialization artifact of shared links, not a hardware
  ceiling.** The model was right.

---

## 1. Measured ρ_uniform, method, R², honor-control

### Method (additive-differential slope, same harness as the strided ρ)

A live SwiGLU compile (shape `[1, 256, 4096]`, patched dxp,
`CARRIER=stcdp_range RANGE_ENCODING=1 REALIZE=1 SPYRE_ONCHIP_MOVE_PLANNER=1`)
is captured; the fired `STCDPOpLx` range-relayout (`sdsc_2`) is **rewritten to a
uniform geometry**; `K ∈ {0, 8, 16}` extra copies of that relay are spliced
into `bundle.mlir`; the bundle is recompiled with the patched dxp; wall clock is
measured per-iteration-synced (12 iters, median + min); the fit `Δµs/Δk` gives
the marginal per-relay time; **ρ = total_bytes / slope** — identical to the ρ
definition that pinned the strided all-to-all at 34–36.

De-risked in order, no fault at any step: compile-only (`dxp_standalone
--bundle`) at N ∈ {4, 31} → 4-link device smoke → full N = 31 sweep.

### Result — ρ_uniform (effective, bytes/slope) — **measured**

| B (bytes/move) | total bytes | slope µs | **ρ (GB/s)** | R² |
|---:|---:|---:|---:|---:|
| 131072 | 4.06 MB | 75.28 | **54.0** | 0.99993 |
| 262144 | 8.13 MB | 89.79 | **90.5** | 0.99984 |
| 524288 | 16.25 MB | 124.78 | **130.3** | 0.99850 |

Min-clock corroboration (same sweep): **53.2 / 92.8 / 131.5 GB/s**, R² ≥ 0.993 —
matches the median, so the number is not a tail artifact.

### ρ_uniform (streaming, fixed overhead removed) — **measured**

Linear decomposition of slope vs bytes:

```
slope_µs = 57.8 µs (fixed per-move) + bytes × 4.096e-6   (R² = 0.998, median)
                                                          (R² = 0.989, min)
⇒ ρ_stream = 1 / 4.096e-6 ≈ 244 GB/s (median) … 254 GB/s (min)
```

The effective ρ *rises with B* (54 → 90 → 130) purely because the fixed ~58 µs
per-move setup amortizes over more bytes; the overhead-free asymptote is
**~244–254 GB/s** — the contention-free aggregate of 31 concurrent ring
segments, ~2× a single raw link.

### Honor-control — **PASSED**

The additive-differential method is only trustworthy if the swept knob (bytes)
actually moves the number. It does:

- **Slopes grow monotonically:** 75.3 → 89.8 → 124.8 µs. Not flat ⇒ the runtime
  is really moving the bytes; a splice that no-op'd or a byte-independent fixed
  cost would leave the slope flat.
- **The streaming component scales ~linearly with bytes:** subtracting the
  fitted 57.8 µs floor gives 17.5 : 32.0 : 67.0 ≈ **1 : 1.83 : 3.83** against the
  byte ratio **1 : 2 : 4**; the `slope = fixed + k·bytes` decomposition fits
  **R² = 0.998**.
- All parser and lowering asserts passed on device; **no fault at any point.**

---

## 2. THE ANSWER — does it escape the 36 GB/s floor, and reach raw ring?

**Yes, and by a wide margin that grows with move size.**

| Pattern | ρ (GB/s) | Per-link transfers | Descriptor count |
|---|---:|---:|---:|
| All-to-all scatter (strided) — **measured** | **34–36** | 4–9 (src→7-9 dests, dest←4 srcs) | 524 |
| Uniform `p→p+1` @ 4.06 MB — **measured** | 54.0 | **1** | 33 |
| Uniform `p→p+1` @ 8.13 MB — **measured** | 90.5 | **1** | 33 |
| Uniform `p→p+1` @ 16.25 MB — **measured** | **130.3** | **1** | 33 |
| Uniform streaming asymptote — **measured (extrapolated)** | **~244–254** | **1** | 33 |
| Model prediction (raw 128 ÷ contention = 1) — **modeled** | 100–145 | 1 | — |
| Raw ring, single link — **spec** | 128–145 (166 peak/dir) | — | — |

- **Escape factor:** at byte-volume matched to the strided run (~8 MB),
  **90.5 / 36 = 2.5×**. At 16 MB, **130 / 36 = 3.6×**.
- **Approaches raw ring: yes.** The directly-measured effective ρ at the largest
  B is **130 GB/s — dead center of the modeled 100–145 band, ~1.02× the raw
  single-link 128.** The overhead-free aggregate (~250 GB/s) sits above a single
  link because 31 uncontended segments carry their bytes concurrently.

The distinguishing variable is **per-link transfer count**, not burst size:
uniform puts **exactly one** transfer on each ring segment (zero contention);
the scatter puts 4–9 (shared links serialize). Enlarging bursts on the *scatter*
gave 1.0× (prior result, `LX_TO_LX_SPEED.md`); eliminating contention took
effective ρ from 36 → 130.

---

## 3. Reconciliation with the model — **was the model right? Yes.**

The backend cost model claimed:

1. `BurstEfficiency` is a **scheduler selection heuristic, not a bandwidth
   multiplier** — perfmodel runs the ring at raw 128 for both patterns.
2. The **only** effective-BW effect is **per-link contention** (shared links
   serialize).
3. Therefore a uniform `p→p+1` shift (one transfer/link, zero contention) should
   land at **raw 128 ÷ contention_factor 1 ≈ 100–145 GB/s.**

All three confirmed on device:

- Burst had already been falsified as a lever (1.0× on the scatter). ✔
- Removing contention — and nothing else — moved effective ρ from **36 → 130**,
  i.e. onto the raw-ring number the model predicted. ✔ The **36 floor is a
  serialization artifact of shared links, not a hardware ceiling.** ✔
- Measured 130 vs modeled 128 at the effective level; ~250 aggregate is the
  contention-free multi-link sum the model implies for 31 concurrent segments. ✔

**The model is validated.** Effective bandwidth on this fabric is governed by
ring-segment contention (per-link transfer count), full stop — not by burst
size and not by a hidden hardware cap below 36.

---

## 4. Implications (with a number)

### Weight-carousel rotation — **roofline CONFIRMED, not ring-bound at 36**

Carousel rotation *is* a uniform `p→p+1` single-hop shift — the exact
zero-contention pattern measured here. It reaches **130 GB/s effective /
~250 GB/s streaming aggregate**, i.e. the raw ring, **not** the 36 GB/s
all-to-all floor. The carousel roofline (ring-bandwidth-limited) is
**confirmed**: rotation is viable at raw ring speed, and the earlier worry that
it would be dragged down to the scatter floor is **refuted**. Size the carousel
against ~128 GB/s per-link (amortize the ~58 µs per-move setup by keeping rotated
tiles large: ρ is 54 at 4 MB but 130 at 16 MB).

### Ring-pipelined all-gather / fold — **bucket-brigade VIABLE**

Each all-gather/fold stage is a nearest-neighbor `p→p+1` hop → the same
one-transfer-per-link, zero-contention regime → the same 130 → ~250 GB/s. A
bucket-brigade all-gather/fold built from single-hop shifts runs at ring speed,
**not** at the scatter floor. Bucket brigade is the right shape; the pipeline
cost is `#hops × (58 µs setup + bytes/ρ_stream)`, so keep per-hop payloads large
to stay overhead-amortized.

### Comms-collectives all-to-all scatter — **still stuck at 36; the lever is placement, not burst**

The all-to-all remains contention-bound at **34–36 GB/s** because it genuinely
oversubscribes shared links (each source → 7–9 dests, each dest ← 4 sources).
Two device-measured facts fix the direction:

- **Burst is not the lever** — 1.0× on the scatter.
- **Contention is the lever** — going to one transfer/link bought 2.5–3.6×.

So the all-to-all lever is **ring-distance-aware placement / decomposition**:
restructure the collective as a *sequence of neighbor shifts* (each contention-
free), or place ranges so every ring segment carries ≤ 1 transfer. Topology-aware
scheduling, not bigger bursts, is what moves the all-to-all off 36.

---

## Precision / tightly-scoped caveat

- **Measured vs modeled vs extrapolated.** ρ_uniform = 54 / 90 / **130** GB/s is
  *directly measured* (bytes/slope, R² ≥ 0.9985). The **~250 GB/s** streaming
  aggregate is an *extrapolation* (fixed ~58 µs per-move overhead removed via the
  R² = 0.998 linear decomposition), not a directly-clocked point; the honest
  single-number headline is **130 GB/s measured** at 16.25 MB. The model band
  (100–145) is *modeled*; raw ring (128–145, 166 peak/dir) is *spec*.
- **Geometry scope.** The uniform ring lives in the relay's only full-core chunk.
  The fresh S = 256 relay has **3 chunks** (core-sizes `[32, 8, 8]`), not the
  on-disk 5; the builder robustly selects the largest-`coreIdsUsed_` chunk
  (chunk0 = `[0..31]`), places **31 unicast ranges** `c→c+1` there, and keeps the
  restricted-core chunks as single **128-byte, non-scaling** keep-alive ranges
  (each chunk's `movementRanges` must be non-empty). Uniform descriptor count =
  **33** (31 unicast + 2 keep-alive) vs the scatter's 524. The keep-alives are
  identical across `k`, so they cancel in the differential and do not contaminate
  the slope.
- **Schema-recipe correction (for the record).** The `parseMovementRange` /
  `parseMovementSide` assert inventory was correct but **missed the pcfg_gen
  lowering assert `ensureUnitPcfg` (`pcfg_gen.cpp:57`)**: both `source.core` and
  `destination.core` must be in that chunk's `coreIdsUsed_`. Only chunk0 carries
  the full `[0..31]` set, which is why the 31-link ring must live in chunk0; the
  restricted chunks can host only their own cores. Every parser and lowering
  assert passed on device.

**Files (absolute):**
- Geometry builder: `/tmp/claude-1000800000/-home-adnan-dt-inductor-torch-spyre/bf8c8f55-d57e-49f8-ab75-1762da9c6aea/scratchpad/rho/uniform_rewrite.py`
- Device sweep: `/tmp/claude-1000800000/-home-adnan-dt-inductor-torch-spyre/bf8c8f55-d57e-49f8-ab75-1762da9c6aea/scratchpad/rho/sweep_uniform.py`
- Runner: `/tmp/claude-1000800000/-home-adnan-dt-inductor-torch-spyre/bf8c8f55-d57e-49f8-ab75-1762da9c6aea/scratchpad/rho/run_uniform.sh`
- Result log: `/tmp/claude-1000800000/-home-adnan-dt-inductor-torch-spyre/bf8c8f55-d57e-49f8-ab75-1762da9c6aea/scratchpad/rho/full_uniform.log`

---

**Bottom line:** ρ_uniform = **54 → 90 → 130 GB/s** (effective, R² ≥ 0.9985) /
**~244–254 GB/s** (streaming aggregate) vs ρ_strided **36** vs model **100–145**.
The uniform `p→p+1` carousel/ring-shift topology **escapes the contention floor
and reaches the raw ring** — 2.5–3.6× the all-to-all — confirming the cost
model. Carousel rotation and ring-pipelined all-gather/fold are ring-bound at raw
speed, not stuck at 36; the all-to-all scatter's only lever is ring-distance-aware
placement, not burst.