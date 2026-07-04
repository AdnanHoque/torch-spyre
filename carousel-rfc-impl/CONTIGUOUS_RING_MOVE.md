# Contiguous Ring Move — Verdict

**Question:** Does the AIU ring ever deliver its raw ~128–166 GB/s/dir, or is the ~36 GB/s measured on the STCDP range-relayout a floor for any pattern? Specifically, does a *contiguous* (large-burst, unicast) ring move escape the BurstEfficiency derate?

**Answer up front:** For the range-relayout scatter that was actually built and run, **~36 GB/s is a floor.** Enlarging the per-descriptor burst ~100× produced **zero** speedup on device. Burst size is not the device-visible bottleneck. The model's *strided* level checks out; the model's *contiguous* prediction does not.

---

## 1. Modeled contiguous rho — and does the strided prediction reproduce 36?

**Raw ring bandwidth.** `sysDef.ringBw = 128` GB/s/dir hardcoded default (`dsc/sysdef.cpp:216`); the dd2 JSON RIU-To-RIU-Link raises this to **166**. Both are per-direction, duplex, multicast-modeled, λ=0.

**BurstEfficiency table** (`dcg/dcg_fe/scheduler/BurstEfficiency.def`, 32×32, closed form):

```
eff(burst b, degree m) = 0.1000 + (b-1)*0.025 - (m-1)*0.0005
```

Burst-size axis dominates (0.025/step, span 0.775); multicast axis is nearly inert (−0.0005/step, full span only −0.0155). Multicast alone can never produce a large derate.

**Modeled rho (raw × efficiency):**

| pattern | burst | eff | ×128 | ×166 |
|---|---|---|---|---|
| **contiguous, unicast** | 32 (clamped to `l3BurstSize`) | 0.8750 | **112 GB/s** | **145 GB/s** |
| strided as-emitted | 8 sticks | 0.275 | 35.2 | 45.6 |
| single-stick worst | 1 | 0.100 | 12.8 | 16.6 |

**Does the strided prediction reproduce the measured 36?** *Yes, tightly.* Fed the relayout's actual 8-stick burst, the table gives eff(8,1)=0.275 → **35.2 GB/s** off the 128 baseline — a near-exact match to the measured ~36. (From 166, matching 36 needs burst 5–6; eff(5,1)=0.20→33.2, eff(6,1)=0.225→37.4, which brackets 36.) So the measured relayout behaves like a **mid-burst ~5–8-stick** transfer, *not* the single-stick floor (12.8–16.6 is too low). **The table's absolute strided level is consistent with the measurement to within its own granularity.**

The table therefore predicts contiguous should clear the strided regime by **~2.9×** (0.875/0.275 at burst 32), or ~1.8× at a 16-stick burst. That is the model's falsifiable claim.

**Critical modeling caveat — two different "models."** BurstEfficiency **never multiplies ring bandwidth in the latency model.** In `sharedtools/perfmodel.cpp` (lines 1889, 1897, 1987, 1997) the ring links are assigned raw `ringBw` (128) with **no** efficiency factor; the only ring adjustment is `trSize/2` for multicast — a duplex *speed-up*, not a derate. The table is consumed **only** by the scheduler (`L3DlOpsScheduler` `findBestParamsForMemoryBandwidth`, :1560/:1820/:1871/:2035) as a **chunk-selection objective** — it steers which chunk parameters get picked; it does not emit a bandwidth or a time. Consequence:

- The deeptools **latency model** already runs the ring at full **128 for both** patterns → it predicts neither the 36 nor a strided/contiguous gap (over-optimistic for both).
- The BurstEfficiency table used as a **manual roofline** reproduces the strided 36 but predicts contiguous 112–145.
- The only other effective-BW effect in perfmodel is **per-link contention** (transfers sharing a link serialize). A uniform p→p+1 shift puts exactly one transfer on each link → no contention → the model *doubly* favors contiguous.

So the model — read either way — predicts contiguous escapes the derate *before the device runs*. Device confirmation was required.

---

## 2. Measured contiguous rho — method and confidence

**Device, solo, honor-controlled. High confidence.**

Additive-differential slope method (same harness as the rho baseline): monkeypatch `launch_kernel` to capture the real device-tensor args, insert *K* copies of the relayout `sdsc_execute` into `bundle.mlir`, recompile with the patched `stcdp_range` dxp (`RANGE_ENCODING=1`), replay per-iter-synced (wall + device), fit Δtime/ΔK. R²>0.998. Every variant preserves the identical **6.758 MB** move (524 movement-ranges, 32-core all-to-all scatter; each dest ← 4 sources, each source → 7–9 dests). S=256 (S=512 wedged — see obstacles).

| variant | per-move slope | effective rho | note |
|---|---|---|---|
| **strided** (as-emitted, burst = 8 sticks / 1024 B) | 192.4 µs (min 185.6) | **34.1 / 35.3 GB/s** | reproduces established ~36 |
| **contig** (same bytes + topology, gather → count=1 burst up to ~50× larger) | 197.7 / 196.4 µs | **33.1 / 33.4 GB/s** | — |
| **sparse** (honor control: count=1, ~12.6× fewer bytes) | 80.3 / 77.9 µs | — | slope collapsed |

**rho_contig / rho_strided = 0.958× (median), 1.004× (min) → zero improvement.**

**The honor control is decisive.** Cutting the moved bytes 12.6× via `movementRanges` dropped the slope 192→80 µs, proving the runtime *actually consumes* the range-encoded movement — so the contig null is **real device physics, not a cosmetic-edit artifact.** (Verified off-device that dxp preserves the enlarged `bytesPerMove=102400` through recompile, and that the scheduler's own burst accounting reads the geometry.)

**Decomposition** (two-point fit, strided 6.758 MB vs sparse 0.537 MB): **fixed per-move overhead ≈ 69–71 µs; streaming (marginal) BW ≈ 55–58 GB/s; effective (full move) 34–35 GB/s.** Even subtracting *all* fixed overhead, the scatter streams at only ~55 GB/s = **33–45% of raw**.

**Measured vs modeled:** the model predicts contiguous is 1.8–2.9× faster; the device shows **1.0×.** Enlarging the per-descriptor burst from 8 sticks to ~800 sticks changed nothing. The ~34–36 GB/s is set by a **~70 µs fixed ring/descriptor overhead + a ~55 GB/s streaming rate**, not by burst efficiency. The BurstEfficiency heuristic ("approximations based on heuristics") does not govern the device rate for this move.

**Scope caveat (the one untested lever).** All variants retained the 524-range all-to-all **multi-hop** scatter. I did **not** device-test the pure uniform **p→p+1** neighbor shift (32 contiguous blocks, unicast, single-hop) — building a valid from-scratch movement geometry (logicalSlice/coverage) risked faulting the device. That geometry attacks the *other two* costs — the 70 µs fixed overhead (≈16× fewer descriptors) and the multi-hop contention behind the 55 GB/s ceiling — not burst. It remains unmeasured; I do not claim it reaches raw.

**Obstacles (for the record):** a parallel codex `dldsc_ah_comms_relayout` job SIGTERM'd the first run and held `/dev/vfio`; and the flex "lost-completion" wedge reproducibly hung the 2nd program reload at S=512 (both attempts, 300 s+). Switched to known-good S=256; rho is shape-independent, so the comparison holds.

---

## 3. The answer

**~36 GB/s is a floor for the STCDP range-relayout, and a contiguous (large-burst) move does NOT escape the ~4× derate — it does not approach raw 128–166.**

- Burst size is **not** the device-visible bottleneck. The BurstEfficiency table's contiguous prediction (~112–145 GB/s, ~2.9× faster) is **falsified on device**: contiguity bought 1.0×.
- The real limiters are a **~70 µs fixed per-move overhead** and a **~55 GB/s streaming ceiling** (marginal), yielding ~34–36 GB/s effective. Even the pure streaming rate is only ~1/3 of raw.
- **One lever remains genuinely untested:** the pure uniform p→p+1 single-hop unicast shift. It targets the fixed overhead and multi-hop contention (which the tested scatter could not isolate), so it *could* beat 36 — but it is unmeasured and there is no evidence it reaches raw. Treat "contiguous single-hop escapes the floor" as an open hypothesis, not a result.

---

## 4. Implications

### Weight carousel (uniform-shift rotation)
The carousel *is* the one untested geometry — uniform p→p+1, contiguous blocks, unicast, single-hop. The measured null does **not** directly condemn it, because the tested scatter was multi-hop all-to-all. **But the roofline claim ("if contiguous is fast, roofline holds") is now conditional on the untested lever, not on burst size** — we proved burst-enlargement alone gives nothing. The carousel's only remaining hope is that single-hop unicast slashes the 70 µs fixed cost (fewer descriptors) and dodges multi-hop contention. **Until that probe runs, plan conservatively:** prefill weight movement is **ring-bound near ~36 GB/s effective / ~55 GB/s marginal, not raw.** Roofline holds only if the uniform-shift probe clears *both* the fixed-overhead floor and the streaming ceiling — currently unproven on device.

### Ring-pipelined all-gather / fold (bucket-brigade contiguous hops)
Same favorable-but-unproven regime as the carousel (each hop is a single-hop p→p+1 unicast pass). **Two device-measured cautions apply directly:** (a) the **~70 µs fixed per-move overhead** is paid *per hop* — a bucket-brigade of N−1 hops is overhead-dominated for small buckets, so buckets must be large enough to amortize; (b) the best case even fully amortized is the **~55 GB/s marginal streaming rate**, still ~1/3 of raw. So a well-sized ring pipeline could plausibly reach ~55 GB/s at the limit, but not 128–166. Size buckets to amortize the fixed cost; do not budget for raw.

### Comms-collectives scatter (inherently strided)
This **is** the measured case — multi-hop all-to-all, 524 movement-ranges. It is stuck at **~36 GB/s** with no available lever: it inherently eats both the streaming ceiling and multi-hop link contention, and burst-enlargement was shown not to help. **Budget collective relayouts at 36 GB/s.** This is the firm, device-confirmed floor.

---

**Files.** Table `dcg/dcg_fe/scheduler/BurstEfficiency.def`; lookup `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1431-1447`; objective-only use `:1560,:1820,:1871,:2035`; raw ringBw `dsc/sysdef.cpp:216`; raw-BW latency (no efficiency factor) `sharedtools/perfmodel.cpp:1889,1897,1987,1997`. Harness/artifacts `scratchpad/rho/sweep_contig.py`, `contig_rewrite.py`, `contig_run.log` (+ `_s256`, `_wedge*`, `_busy`, `_killed`).