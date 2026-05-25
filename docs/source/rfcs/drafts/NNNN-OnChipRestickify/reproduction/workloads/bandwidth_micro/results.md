# Spyre realized-bandwidth microbenchmarks (roofline parameterization)

All numbers device-measured on the shared AIU (SOLO, sequential). torch 2.11
`.venv` + `uv`; torch.profiler PrivateUse1 device events (`device_time_total`).
Microbench A used stock dxp; Microbench B used the patched on-chip dxp
(`/home/adnan/dt-inductor/build/deeptools-onchip/dxp/dxp_standalone`).

Peak HBM reference = 170 GB/s (doc). Stick = 128 B = 64 fp16 elements.

## Microbench A — effective HBM round-trip bandwidth vs size

Op: `y = x * 2.0` (fp16), one fused pointwise SFP kernel (`sdsc_fused_mul_0`).
Reads S bytes, writes S bytes -> HBM traffic = 2S. `B_hbm_eff = 2S / kernel_ms`.
Shapes: rows x 2048 (2048 stick-aligned), N=50 timed iters after 12 warmup.
Each size ran in its own fresh process (a shared-state run wedged the runtime
DMA scheduler at >2 MB; per-process isolation fixed it). max_err = 4.9e-4 (fp16
rounding of *2.0), value-correct.

| S (MB) | kernel_ms | spyre_ms | B_hbm_eff (kernel) GB/s | % of 170 peak |
|-------:|----------:|---------:|------------------------:|--------------:|
|  1 | 0.0318 | 0.0341 |  66.0 | 38.8% |
|  2 | 0.0387 | 0.0409 | 108.3 | 63.7% |
|  4 | 0.0801 | 0.0834 | 104.8 | 61.6% |
|  8 | 0.1649 | 0.1671 | 101.8 | 59.9% |
| 16 | 0.3383 | 0.3406 |  99.2 | 58.3% |
| 32 | 0.6626 | 0.6655 | 101.3 | 59.6% |

(spyre_ms is total device time incl. tiny memcpy/setup events; kernel_ms is the
pointwise kernel only. They differ by <0.003 ms here — the op is pure-kernel.)

### Trend: flat, NOT degrading

- B_hbm_eff does **not** degrade super-linearly with S. It **rises** 1->2 MB
  (small-S fixed-cost / setup floor) then **plateaus flat at ~100 GB/s
  (~59% of 170 peak)** for all S >= 2 MB (range 99.2-108.3 GB/s, no downward
  trend through 32 MB).
- The **1 MB point is fixed-cost-dominated**: 66 GB/s (38.8%). Linear-fit of
  kernel_ms vs S over {2,4,8,16,32} MB: slope ~= 0.0198 ms/MB (=> asymptotic
  ~106 GB/s round-trip) + intercept ~= 0.0 ms; the 1 MB point sits above the
  fit line (extra fixed cost), consistent with a setup floor at small S, not
  HMI contention at large S.
- A pure scale is firmly HBM-bound (far left of the roofline knee): zero
  reuse, 2 bytes moved per fp16 element for ~1 flop.

**Verdict A:** realized HBM round-trip bandwidth is **~100 GB/s (~59% of the
170 GB/s peak), flat across 2-32 MB**. The doc's "effective BW degrades
super-linearly with size (HMI contention)" is **not supported** — effective BW
is a constant ~0.59x derate, with an *additional* small-S setup penalty below
~2 MB.

## Microbench B — effective ring move bandwidth

**Path chosen:** the preferred elementwise producer->consumer splice was NOT
viable — Inductor fuses `m = x*2; y = m*3` into a *single* pointwise kernel
(`sdsc_fused_mul_0`), so there is no HBM handoff buffer to keep on-chip. Used
the proven 2-SDSC matmul handoff instead: `(perm @ x) @ wexp` -> producer
dispatch matmul writes [EC,H] to HBM base 0, consumer linear reads it back.
BASELINE keeps the handoff in HBM; ON-CHIP keeps it in LX and moves it
cross-core via an STCDP round-trip i->31-i->i (patched dxp). Compute is
identical in both, so `dT = baseline - onchip = HBM_handoff - ring_move`.
Handoff buffer swept 2/4/8 MB (E8, H=2048 constant stick, EC=512/1024/2048).
N=50 timed iters after 12 warmup; device-time via torch.profiler PrivateUse1.

### A/B device times (clean re-measurement)

| handoff S (MB) | baseline_ms | onchip_ms | dT (saving) ms | speedup |
|---------------:|------------:|----------:|---------------:|--------:|
| 2 | 0.2737 | 0.2086 | 0.0651 | 1.31x |
| 4 | 0.4497 | 0.3224 | 0.1273 | 1.39x |
| 8 | 1.3217 | 0.6860 | 0.6357 | 1.93x |

(kernel_ms == spyre_ms for every point: the matmul bundle has no separate
memcpy/memset device events.) Value-correct: max_err ~0.0017 vs CPU for all
spliced runs (vs ~0.0016 baseline) — see checks below.

dT-vs-S linear fit over the two well-behaved points {2,4 MB}:
**slope = 0.0311 ms/MB, intercept = +0.0029 ms** (matches the prior MoE anchor
of ~0.029 ms/MB + ~0.005 ms setup). The 8 MB baseline shows a knee (1.32 ms,
disproportionately slow), so the all-3 fit is distorted; the 2-4 MB rate is the
reliable per-MB number.

### Ring move cost: below measurement resolution (cannot invert to B_ring_eff)

Using `B_hbm_eff = 103 GB/s` (the A plateau) the predicted eliminated HBM
handoff is `2S/103GB/s` = 0.041 / 0.081 / 0.163 ms for S = 2/4/8 MB. Then
`ring_move = HBM_handoff_actual - dT`, and the *lower-bound* estimate
`2S/B_hbm_eff - dT` is **negative at every size** (-0.024 / -0.046 / -0.47 ms).

This is not a bug — it is the result: the **ring move is so much cheaper than
the HBM handoff that it is swamped**. Two independent facts confirm it:

1. The theoretical ring time is tiny: a round-trip moving 2S of ring traffic at
   the ~6.5 TB/s aggregate is **0.0006 / 0.0013 / 0.0026 ms**; at 128 GB/s/dir
   per-link on the per-core slice (S/32 each way, 2 hops) it is
   **0.0010 / 0.0021 / 0.0041 ms**. Either way the ring move is **2-3 orders of
   magnitude below dT** — far below the per-iter device-time resolution here.
2. dT therefore = (eliminated HBM handoff) almost entirely. dT EXCEEDS the
   contiguous-pointwise HBM estimate `2S/103GB/s`, which means the matmul's
   *strided/tiled* HBM handoff is even more expensive than A's contiguous
   round-trip (the 8 MB baseline knee is the extreme case). So the subtraction
   `2S/B_hbm_eff - dT` cannot recover the ring number; it only proves
   ring_move << HBM_handoff.

**B_ring_eff** (INFERRED, not directly measured): bounded below by the
multi-TB/s aggregate (consistent with ~6.5 TB/s / ~128 GB/s-per-link); the A/B
cannot put an upper bound tighter than "negligible vs HBM." No fixed-vs-per-byte
ring split is recoverable from these data — the ring term is dominated out; the
~0.0031 ms small-S setup we see is the STCDP/DL launch overhead, not ring bytes.

### Required checks (all PASS)

1. **Value correctness:** all spliced runs validate vs CPU ref, max_err
   ~0.0017 (close_rtol/atol 5e-2). Distinct from baseline max_err -> the
   on-chip senprog really ran.
2. **Negative control:** removing the spliced senprog
   (`loadprogram_to_device/<bundle>-SenProgSend/init.txt`) makes the device
   load HARD-FAIL — `RuntimeError: Failed to open file: .../init.txt`,
   exit code 1, no DIRECT_VALIDATE_OK. No silent fallback.
3. **Ring signature:** the spliced *consumer* SDSC carries
   `opFuncsUsed_ = ['STCDPOpLx','STCDPOpLx']` (the 2-move cross-core round trip)
   with 2 datadscs; the baseline consumer SDSC has `opFuncsUsed_ = None`,
   0 datadscs. The matching dxp L3-scheduler verbose signature (2x
   `Creating PCFG for DataDsc`, 64 = 2 ops x 32 cores `: L3SU : L3LU` lines,
   `i --> [31-i]` mirror) is in the reference verbose log for the identical
   splice (`/tmp/ab_moe_routing/dxp_moe_verbose.log`); the baseline verbose log
   has 0. (Note: the deeptools-onchip `dxp_standalone --bundle -d` used here did
   not re-emit the PCFG lines to the captured stream even with
   `DT_DEEPRT_VERBOSE=1`; the SDSC-level signature above is the per-run proof.)

## Verdict

- **Realized HBM bandwidth ~100 GB/s (~59% of the 170 GB/s peak), FLAT across
  2-32 MB** (not degrading; 1 MB is fixed-cost-dominated at 66 GB/s). The doc's
  super-linear-degradation-with-size claim is not supported; instead there is a
  constant ~0.59x derate plus a small-S setup floor below ~2 MB.
- **Realized ring bandwidth is effectively unbounded relative to HBM at these
  sizes** — the cross-core STCDP round trip costs 0.0006-0.004 ms (theory),
  2-3 orders of magnitude below the eliminated HBM handoff, so it cannot be
  isolated by the A/B subtraction (residual goes negative). The ring is
  consistent with the multi-TB/s aggregate; treat it as "free vs HBM."
- **This explains the 1.4-6.5x op-level under-prediction:** a roofline using the
  170 GB/s peak over-predicts achievable HBM throughput by ~1.7x at the plateau
  (and up to ~2.6x at 1 MB / small ops). Layering on the matmul-handoff
  inefficiency (strided HBM access, the 8 MB knee where baseline jumps to
  1.32 ms => effective handoff BW well under 100 GB/s) compounds the gap. Use
  **B_hbm_eff ~= 100 GB/s (flat)** in the roofline, not 170; and treat ring
  moves as ~zero-cost so on-chip handoffs are modeled as pure HBM-elimination.

### Inference flags
- Ring B_ring_eff is INFERRED (theory floor + aggregate), not directly measured
  — the A/B resolution cannot isolate a sub-0.005 ms ring term.
- The matmul A/B confounds HBM-handoff *access-pattern* inefficiency with the
  handoff bytes; the clean per-byte HBM number is from Microbench A, not B.
- All HBM/ring derate numbers are device-measured (A) or device-bounded (B);
  no projected throughput is presented as measured.
