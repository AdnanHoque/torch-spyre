# On-Chip Handoff: Theoretical Model vs Observed (worked calculations)

This doc answers three questions for the on-chip core-to-core (LX↔LX over the RIU
ring) data-movement optimization:

1. **How do we know the data actually traverses the RIU ring** (and not HBM)?
2. **What does a first-principles bandwidth/roofline model predict** for the gain?
3. **Does the prediction match the measured MoE and attention A/B**, and where it
   does not, **what explains the delta**?

All hardware constants are the **deeptools compiler cost model** (`dsc/sysdef.cpp`,
coreFreq = 1.5 GHz), cross-checked against the IBM Spyre KB primary papers. Every
number is cited. Inferences are flagged `[INFER]`.

> Note on a prior figure: earlier notes used "166 GB/s" for HBM and the ring.
> That is a stale 1.3 GHz derivation; `166` does not appear as a bandwidth in
> deeptools. The compiler model uses **HBM 170 GB/s**, **ring 128 GB/s/direction
> per link** (= 128 B/cyc, 192 GB/s at 1.5 GHz), **LX 192 GB/s/corelet**.

---

## 1. How we know we are on the RIU ring

The cross-core move lowers to the **L3 ring DMA units**, and the device trace
shows them firing on every core with the mirror permutation. This is structural,
not inferred:

- **L3LU / L3SU are the ring load/store DMA units**, not HBM units. L3LU (inbound)
  is wired to `core2RingLoadReqFifo` and L3SU (outbound) to
  `core2RingStoreReqFifo`; "L3 only connects to LX"
  (`wiki/concepts/core-functional-units.md:34-35,136,220`;
  `sources/schedule-ir-spec.md:291-293`). The RING itself is a transport, not a
  programmable unit (`schedule-ir-spec.md:203-205`).
- **`STCDPOpLx` lowers to L3SU + L3LU.** `DcgFE::createPcfgsSTCDPOp` emits a
  per-core PCFG entry tagged `SenComponents::L3SU` and one tagged
  `SenComponents::L3LU` (`deeptools/dcg/dcg_fe/pcfg_gen/stcdpOp.cpp:459-518`), and
  prints `: L3SU` / `: L3LU` (`:476,500,522,540`). The HBM variant is a different
  op, `STCDPOpHBM` (`dsm/dsm.cpp:1771`).
- **The cost model attaches `ringBw` to these moves** (every ring-link entry uses
  `ringBw`, `perfmodel.cpp:1885,1983,1993`); HBM moves use a single `hbm-dt`
  component at `hbmBw` (`perfmodel.cpp:1880`). So the IR-level test is exact:
  **ring iff there are L3SU/L3LU PCFG entries fed by `ringBw`; HBM iff a single
  `hbm-dt` component.**

Device evidence (verbose dxp traces):

| bundle | `DataDsc` PCFGs | `L3SU:L3LU` (per core) | mirror map `i --> [31-i]` |
|---|---|---|---|
| MoE spliced (on-chip) | 2 (`dxp_moe_verbose.log:3658`) | all 32 cores (`:3659-3690`) | `:3692-3723` (`0-->[31] … 31-->[0]`) |
| Attention (on-chip) | 2 (`dxp_attn512_verbose.log:8678`) | 64 occurrences | `:8645-8676` |
| MoE baseline (HBM) | **0** (`dxp_baseline_verbose.log`) | **0** | **0** |

The baseline bundle has **zero** DataDsc / L3SU / L3LU / permutation lines — it is
pure HBM. The on-chip bundle moves data core→core over the ring with the explicit
mirror permutation `i --> [31-i]` on all 32 cores. That is the proof.

---

## 2. The bandwidth model

The canonical deeptools cost model is **`t = bytes / BW` per fabric component, with
parallel components overlapped (max), no additive DMA-setup latency**
(`sharedtools/perfmodel.cpp`; roofline knee `peak/hbmBw` at `sysdef.cpp:262-273`).
The three fabrics that matter:

| fabric | bandwidth | shared? | source |
|---|---|---|---|
| HBM | **170 GB/s** | one HMI, shared by all 32 cores | `sysdef.cpp:209-211` |
| RIU ring | 128 GB/s/dir per link; **aggregate ≈ `128×2×numSeg×0.8`** ≈ 6.5–8.2 TB/s | 32 parallel links | `sysdef.cpp:212`; `dsm/spadprefetch.cpp:89-91` |
| LX | 192 GB/s/corelet; aggregate ≈ 6.1–12.3 TB/s | per corelet | `sysdef.cpp:213` |

The single load-bearing fact: **HBM is one shared 170 GB/s pipe, while the ring
and LX aggregate to ~6–8 TB/s — roughly 40–60× faster.** The ring is pipelined
(per-link model, no per-hop bandwidth penalty), so the mirror permutation costs
≈ a 1-hop move for an isolated transfer (`perfmodel.cpp:1860-1993`). Sub-4 KB
transfers miss peak BW (L3 burst = 32 sticks = 4 KB, `schedule-ir-spec.md:295`).

### Expected gain

A producer→consumer handoff of `S` bytes:

- **HBM baseline:** producer stores `S` to HBM, consumer loads `S` from HBM →
  `2S` through the shared pipe → `T_hbm = 2S / 170 GB/s`.
- **On-chip:** producer's output already sits in LX; the ring moves it core→core;
  the consumer reads it from LX → `T_onchip = S/ring_agg + 2·S/lx_agg`.

Because ring/LX are ~40–60× HBM, `T_onchip` is negligible. Concretely, for the
**largest (8 MB) handoff, using the measured 2× round-trip construct** (two ring
moves, `i→31−i→i`):

```
ring  : 2·8 MB / 6.55 TB/s  ≈ 0.0026 ms
LX r/w: 2·8 MB / 6.1  TB/s  ≈ 0.0028 ms
T_onchip ≈ 0.005 ms   vs   T_hbm = 2·8 MB / 170 GB/s ≈ 0.099 ms   (~5%)
```

So even with the round-trip construct's doubled ring traffic, the on-chip side is
~5% of the eliminated HBM time. The **expected saving** is therefore, to within a
few percent:

> **ΔT_expected ≈ 2S / 170 GB/s = 0.01234 ms per MB of handoff** (peak HBM BW).

This is the prediction. Note it is a prediction of the **saving (ΔT)**, not of the
full op time or the speedup ratio — the model does not include the matmul compute
or the other operands, only the eliminated handoff round-trip. It is also a
**lower bound**, because it assumes HBM runs at its 170 GB/s peak.

---

## 3. Predicted vs observed

Handoff size `S` per workload: MoE dispatch/combine `S = EC·H·2` bytes; attention
`S = bh·seq_q·seq_k·2` (the score matrix). `ΔT_observed = HBM_ms − onchip_ms`
(device, N=50, from [PerformanceResults.md](PerformanceResults.md)). `B_eff` is the
**effective HBM bandwidth implied by the observation**, `B_eff = 2S / ΔT_observed`.

| workload | S | HBM ms | on-chip ms | ΔT obs | ΔT exp (peak) | **obs/exp** | implied B_eff |
|---|---|---|---|---|---|---|---|
| MoE dispatch E8 T512 H2048 | 2 MB | 0.2746 | 0.2094 | 0.0652 | 0.0247 | **2.6×** | 64 GB/s |
| MoE dispatch E8 T512 H4096 | 4 MB | 0.9425 | 0.6725 | 0.2700 | 0.0493 | **5.5×** | 31 GB/s |
| MoE dispatch E8 T1024 H2048 | 4 MB | 0.4465 | 0.3225 | 0.1240 | 0.0493 | **2.5×** | 68 GB/s |
| MoE dispatch E8 T2048 H2048 | 8 MB | 1.3323 | 0.6865 | 0.6458 | 0.0987 | **6.5×** | 26 GB/s |
| MoE combine E8 T512 H2048 | 2 MB | 0.2760 | 0.2093 | 0.0667 | 0.0247 | **2.7×** | 63 GB/s |
| Attention seq=512 bh=32 | 16 MB | 2.5595 | 1.9778 | 0.5817 | 0.1974 | **2.9×** | 58 GB/s |
| Attention Q512 KV4096 bh=1 | 4 MB | 0.5198 | 0.4523 | 0.0675 | 0.0493 | **1.4×** | 124 GB/s |
| Attention seq=64 bh=32 | 256 KB | 0.1811 | 0.1832 | −0.002 | 0.0031 | n/a | below floor |

**The model has the right sign and order of magnitude, but it under-predicts the
saving by 1.4×–6.5×.** Observed savings are always ≥ the peak-BW floor, and the
implied effective HBM bandwidth is **26–124 GB/s — i.e. 15–73% of the 170 GB/s
peak.**

---

## 4. The delta: why observed beats the peak model

The expected gain assumed HBM runs at 170 GB/s. The observations say it does not —
the baseline HBM path runs **well below peak**, and that is precisely the traffic
the on-chip path removes. Four compounding causes, in order of evidence:

1. **Scattered / sub-burst HBM access (the dominant cause for MoE).** Routing
   dispatch/combine is a gather/scatter — a permutation. HBM is bursty (L3 burst =
   4 KB, `schedule-ir-spec.md:295`); scattered per-token accesses below the burst
   quantum fetch whole bursts to use a fraction, so realized bytes ≫ logical
   bytes and effective BW collapses. The implied `B_eff` for MoE (26–68 GB/s)
   sits **below even the contiguous `ReStickifyOpHBM` anchor (~107 GB/s)**,
   consistent with scatter amplification. The same-stick on-chip path moves whole
   sticks contiguously over the ring and pays none of this.

2. **Restickify staging the same-stick path skips.** The attention bundle contains
   a `2_ReStickifyOpHBM` (`dxp_attn512_verbose.log:4302`) — a layout staging that
   routes through HBM (`…OpHBM` uses `hbmBw`; `…OpLx` stays on-ring,
   `restickifyOp.cpp:18-20`). The baseline pays this HBM traffic on top of the
   logical handoff; the on-chip path keeps it on-chip. This inflates the eliminated
   bytes beyond the nominal `S`.

3. **Super-linear HMI contention.** A single shared pipe degrades as it saturates.
   The implied `B_eff` **falls as the handoff grows**: 64 GB/s @ 2 MB → 26 GB/s
   @ 8 MB. So the per-MB saving *grows* with size (0.033 → 0.081 ms/MB) — the
   linear `bytes/BW` model cannot capture this; the on-chip win is largest exactly
   where HBM is most congested.

4. **Layout / shape dependence.** Two 4 MB handoffs, same logical bytes, differ
   2.2× in saving: `T512 H4096` (B_eff 31) vs `T1024 H2048` (B_eff 68). The wider
   hidden axis (the stick dimension) yields a worse HBM access footprint. A single
   scalar bandwidth cannot predict this — it is an access-pattern effect.

**The control that proves the theory:** the **clean, contiguous** attention
handoff (Q512 KV4096, bh=1) lands at **1.4× of the peak model (B_eff ≈ 124 GB/s ≈
73% of peak)** — the closest to prediction of any row. Where the handoff is
contiguous and large-per-core, HBM runs near peak and the model is accurate; where
it is scattered (MoE) or restickified (attention seq=512), HBM collapses and the
on-chip advantage balloons. **The model is right; the delta is the gap between HBM's
peak spec and its realized bandwidth, and the on-chip handoff wins biggest in
exactly the regime where that gap is widest.**

---

## 5. What the model gets right (the matches)

- **Sign and scaling.** Every above-floor handoff wins, and the saving scales with
  handoff bytes (the bandwidth-bound signature) — predicted and observed.
- **Ring/LX cost is negligible → the round-trip construct still wins.** The model
  says `T_onchip` is ~5% of the eliminated HBM even with the doubled ring traffic;
  observed: the round-trip construct (which a production single-move would beat)
  wins by 1.3–1.9× anyway. Confirmed.
- **Sub-MB neutrality.** seq=64 (256 KB) has expected gain 0.003 ms — below the
  STCDP setup + the 3-region construct overhead + measurement noise — so the model
  predicts no usable win, and the measurement is neutral/slight regression. Match.
- **E-invariance** (in [PerformanceResults.md](PerformanceResults.md)): the model's
  `S = EC·H·2` has no expert-count term, and E=8 vs E=64 at matched EC·H measure
  identically. Match.

## 6. Caveats and how to tighten this

- The model uses **peak** bandwidths; the delta analysis backs out **effective**
  HBM BW from op-level ΔT, which conflates the handoff with restickify and any
  compute non-overlap. The numbers are internally consistent but the per-cause
  attribution in §4 is reasoned from the access pattern, not isolated.
- **The decisive next measurement** (device, ~15 min, currently un-run) is two
  isolated microbenchmarks: (a) a pure HBM store→load of `S` (no compute) to
  measure realized HBM round-trip BW directly, and (b) a pure `STCDPOpLx` move of
  `S` (no compute) to measure realized ring BW. With those two measured constants
  the model becomes predictive rather than bounding, and §4's "effective ≪ peak"
  claim is validated head-on instead of inferred.
- All on-chip numbers include the **2× round-trip construct**; a production
  single-move handoff would save marginally more (ring cost is ~5%, so the
  difference is within noise).

## Source index

- Bandwidths: `deeptools/dsc/sysdef.cpp:179,191,209-213` (stick, freq, hbmBw,
  ringBw, lxCoreletBw); `dsm/spadprefetch.cpp:89-91` (ring aggregate + 0.8 derate).
- Cost model: `sharedtools/perfmodel.cpp:1880,1885,1983,1993` (hbm/ring BW
  assignment), `:1860-1993` (per-link pipelined ring); `sysdef.cpp:262-273`
  (roofline knee); `schedule-ir-spec.md:295` (4 KB L3 burst).
- Ring DMA units: `wiki/concepts/core-functional-units.md:34-35,136,220`;
  `schedule-ir-spec.md:203-205,291-293`; `stcdpOp.cpp:459-518` (STCDPOpLx →
  L3SU/L3LU); `restickifyOp.cpp:18-20` (…OpHBM vs …OpLx).
- Device traces: `/tmp/ab_moe_routing/dxp_moe_verbose.log`,
  `dxp_baseline_verbose.log`; `/tmp/ab_attention_512/dxp_attn512_verbose.log`.
- Primary papers: `sources/DNNDaSher.pdf` p.1,3 (ring, stick); `sources/rapid.pdf`
  p.7 (LX 128 B/cyc).
- Observed numbers: [PerformanceResults.md](PerformanceResults.md) (MoE A/B,
  attention A/B).
