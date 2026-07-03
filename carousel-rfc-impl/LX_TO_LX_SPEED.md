Written to `/home/adnan/dt-inductor/torch-spyre/carousel-rfc-impl/LX_TO_LX_SPEED.md`. Markdown text below.

---

# LX-to-LX Ring Transfer Speed (STCDPOpLx) — Verdict

**Scope:** effective bandwidth ρ and per-hop latency λ for an on-chip LX→LX relayout carried by `OnChipMoveSTCDPOpLx` (`stcdp_range` carrier, 32 cores). This is the primitive the carousels' on-chip shuffle is built on. Numbers are tagged **measured** (device), **modeled** (backend cost model, from source), or **spec** (architecture sheet).

---

## 1. Measured ρ vs. modeled/assumed ρ + λ

### Measured (device) — ρ ≈ 36 GB/s effective, HIGH confidence

| shape | bytes / move | slope µs/move (median) | R² | ρ_eff (GB/s) |
|---|---|---|---|---|
| S=512 | 13,107,200 | 362.40 | 0.99990 | 36.17 |
| S=256 | 6,553,600  | 184.86 | 0.99997 | 35.45 |

- **Method:** additive K-replication differential. The isolated relayout SDSC faults on device (needs its live LX context), so I kept the *full valid* SwiGLU bundle and inserted K extra copies of the relayout's own `sdsc_execute` into `bundle.mlir`, recompiled with the patched `dxp_standalone`, and replayed with the **exact captured device-tensor args**. `slope = Δdevice_time / Δmove_count` cancels all fixed program/host overhead. Per-iteration-synced wall clock (profiler and the batched loop both wedged the device — the known flex profiling-in-streams thread-lock). R² ≥ 0.9999 on every fit.
- **Byte-sweep decomposition** (two sizes → separate the fixed per-execute cost F from the pure ring rate; `slope = F + bytes/ρ_ring`): **ρ_ring = 36.9 GB/s** (F removed), **F = 7.3 µs** per execute (2–5 % of the S512 move, so effective ≈ asymptotic). slope512/slope256 = 1.96 vs bytes-ratio 2.00 — physically consistent.
- **Cross-check:** an early device-*profiler* 2-point S512 run gave device-slope 370.2 µs == wall-slope 368.7 µs (0.4 %) — proves the slope is device work, not host dispatch.
- **What ρ_eff means:** it is defined on the **logical tensor bytes** (`bytes_moved` read live from the SDSC). A range-relayout permutes / multicasts across 32 cores, so physical link traffic ≥ logical traffic; 36 GB/s is therefore a **floor on the raw per-direction link BW**, and the **exact effective rate** for this relayout pattern.

### Modeled (backend cost model, from source) — ρ = 128–166 GB/s/dir, λ = 0

Two perf models, both `time = bytes / bandwidth`, **no per-hop latency term** anywhere (λ = 0; multi-hop cost emerges only from serialized per-link occupancy + sync nodes).

- **Path A — perfmodel (primary, hardcoded):** `ringBw = 128 GB/s/dir` (`dsc/sysdef.cpp:216`); `trCycles = ceil(trSize/bandwidth * coreFreq)` (`sharedtools/perfmodel.cpp:648-651`), coreFreq 1.5 GHz → 85.3 B/cycle. **Multicast modeled** (`trSize = trSize/2` split across cw+ccw arcs, `perfmodel.cpp:1888,1986`); **duplex modeled** (separate `cwlink-*`/`ccwlink-*` link resources). STCDP lowering asserts `coreFreq == lxCoreletBw/ringBw` (1.5 == 192/128, `stcdpOp.cpp:3307,4121`).
- **Path B — Sentient JSON estimator:** `RIU-To-RIU-Link` = 128 B/cyc × freq → 153.6 (1.5), **166.4 (dd2, = the BiRing spec)**, 179.2 (multi-dd2); BiRing `mul_factor = 2` = duplex; `cycles = totVolume/bandwidth`, no latency (`RCUIntraEntityScheduler.cpp:329-330`).
- **Only ring-BW modifier:** the **heuristic** `BurstEfficiency.def` table (`L3DlOpsScheduler.cpp:275,1431-1447`) — a multiplier on effective ring BW keyed on burst size × multicast degree ("approximations based on heuristics"). Larger multicast degree → lower efficiency.
- **Spec:** ~166 GB/s/dir BiRing (matches Path B dd2).

### Side-by-side

| quantity | value | kind |
|---|---|---|
| ρ effective, this relayout | **36 GB/s** (35.4–37 GB/s) | measured, HIGH |
| ρ ring, perfmodel constant | 128 GB/s/dir | modeled |
| ρ ring, Sentient dd2 / spec | 166 GB/s/dir | modeled / spec |
| λ per-hop | 0 | modeled (both paths) |
| λ per-hop | no direct signal | measured — needs instrumentation |
| F per-execute (fixed) | **7.3 µs** | measured |

---

## 2. Reconciliation — do they agree?

**Not at face value.** Measured effective 36 GB/s is **21.7 %** of the 166 GB/s/dir spec and **~28 %** of the 128 GB/s perfmodel constant. A naive model of this move (13.1 MB / 128 GB/s) predicts **~102 µs**; with the multicast halving it predicts **~51 µs**. Device is **362 µs** — the model is **3.5–7× optimistic** for this specific move.

**The gap is not a contradiction; the two numbers measure different things.**

- Device ρ_eff is the **effective end-to-end throughput of a strided, multicast range-relayout** (`dataop_chunks = 5`, permutation across 32 cores) on **logical** bytes.
- Model ρ is the **ideal per-direction link BW** for a contiguous stream, which the model then *is supposed to* de-rate via the **`BurstEfficiency` heuristic**. The 3.5–7× is exactly the regime that de-rate exists to cover — a strided/multicast pattern moving more physical bytes per logical byte and paying permutation overhead the flat `bytes/128` term ignores.

**Is the device number a bound?** Two different roles:

- As the **effective rate for costing this pattern** (logical bytes ÷ ρ → time): it is **not a bound, it is the measured value** — use 36 GB/s directly. Tightest defensible range **35–37 GB/s** (median vs. min fits, two sizes, R² ≥ 0.9999).
- As an inference about the **raw physical link BW**: 36 GB/s is a **lower bound only** (logical ≤ physical traffic). The raw link rate sits in **[36, 166] GB/s/dir** and cannot be pinned without counting physical hops / multicast degree.

**Verdict:** they **do not agree as stated**, and they **should not** — the honest effective rate for a carousel-style range-relayout is **36 GB/s**, not the 128–166 ideal-link ceiling. The ceiling is only reachable by a contiguous single-arc stream, which a permuting relayout is not.

---

## 3. Per-hop latency λ

- **Measured:** **no direct signal.** Wall clock cannot resolve individual ring hops, and the profiler's finest granularity is the **fused bundle** (one event per inductor bundle — it cannot expose a single STCDPOpLx event). Only the aggregate is observable: 362 µs to relayout 13.1 MB across 32 cores. **A per-hop λ needs dedicated per-hop instrumentation** (cycle-accurate senulator ring timer `senulator/timerRing.cpp` + `senulator/progs/pcfg_ringdttest/`, or hardware hop counters).
- **Modeled:** **λ = 0** in both cost paths — there is no fixed ns/hop constant; multi-hop cost is pure serialized link occupancy plus sync nodes.
- **Do not confuse with F.** The measured **F ≈ 7.3 µs** is a fixed **per-execute (per-SDSC)** cost, **not** per-hop latency. It is real and matters for small moves, but it is one constant per STCDP execute, not a ×hop-count term.

---

## 4. Implications

### For the carousels' cost model (`comm_cost.py`, ring-occupancy term)

- **Use the measured effective rate, not the ideal link BW.** The ring-occupancy term should divide logical move bytes by **ρ_eff ≈ 36 GB/s**, and add a fixed **F ≈ 7 µs per STCDP execute**:

  ```
  t_ring = F + bytes_moved / rho_eff      # rho_eff = 36e9 B/s, F = 7e-6 s
  lambda_per_hop = 0                        # consistent with backend; no per-hop term
  ```

  Using the 128–166 GB/s spec/model constant here would **under-cost on-chip ring moves by 3.5–7×**, biasing the carousel planner toward on-chip relayout in cases where it is actually far slower — the exact failure mode this measurement exists to prevent.
- **Equivalent framing:** if `comm_cost.py` mirrors the perfmodel path (128 GB/s raw), it must apply a **BurstEfficiency-style de-rate ≈ 0.22–0.28** for the range/multicast relayout to land at 36 GB/s. Hardcoding the measured 36 GB/s is simpler and better anchored for the carousel pattern.
- **Scope caveat:** 36 GB/s is measured at 32 cores, `dataop_chunks = 5`, two sizes (6.55 / 13.1 MB). A different multicast degree or stride pattern will shift ρ_eff; if carousel moves vary that, parametrize by a BurstEfficiency-like factor rather than trusting one flat constant across regimes. λ = 0 is safe to keep until per-hop instrumentation says otherwise.

### For M0 (go / no-go)

**No change — M0 stays GO.** M0's premise is that the on-chip relayout removes an HBM round-trip. At the honest 36 GB/s, the two SwiGLU relayout moves (2 × 13.1 MB = 26.2 MB) cost **~0.72 ms** of ring time — larger than the ~0.16 ms the loose M0 A/B had attributed to the ring (the earlier number under-counted the ring by ~4.5×), but still the **minority** of the 2.36 ms A/B win: the ring move is ~23 % of the ~3.0 ms HBM round-trip it displaces, which still **~77 %** dominates the delta. The go/no-go decision is unaffected; the relayout remains net-positive.

What *does* update: the previous loose bound **ρ ≥ ~11 GB/s** is **superseded** by **ρ ≈ 36 GB/s** (effective, HIGH confidence). Any M0 follow-on that budgets ring time should use 36 GB/s + 7 µs/execute, and should note the ring slice is a bigger fraction of the win than the loose bound implied — a tighter margin, not a flipped decision.

---

## Provenance

- **Measured:** device, harvest env, solo; harness `…/scratchpad/rho/sweep.py`, logs `wall.log` / `wall256.log`; relayout confirmed firing (`grep OnChipMoveSTCDPOpLx` + live `bytes_moved` capture), carrier `stcdp_range`, 32 cores. Two sizes, R² ≥ 0.9999.
- **Modeled:** source read-only — `dsc/sysdef.cpp:216` (ringBw 128), `sharedtools/perfmodel.cpp:648-651,1888,1986` (cost + multicast), `dsc/HardwareArchMapping/baseAccelSystem.cpp:1390,1286` + `sysConfigs2.0/*.json` (Sentient 153.6/166.4/179.2), `RCUIntraEntityScheduler.cpp:329-330` (no latency), `dcg/dcg_fe/scheduler/BurstEfficiency.def` (heuristic de-rate).
- **Spec:** ~166 GB/s/dir BiRing.
- **Not conflated:** `dsm/coll/cost_estimator.cpp:472-482` (`BANDWIDTH_EST_DMA`, `LATENCY_EST_DMA 0.162us`) are cross-device HDMA/spyreccl links, **not** the on-chip STCDP ring.