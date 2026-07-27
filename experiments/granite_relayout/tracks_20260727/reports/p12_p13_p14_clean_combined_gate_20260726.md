# P12 + P13 + P14 clean combined gate

Date: 2026-07-26  
Status: **PASS for combined correctness, realized transport, and positive timing; not SenDNN parity**

## Frozen environment

- Pod: `a6-quantization/adnan-spyre-dev-pf`
- Clean candidate root:
  `/home/adnan/codex-isolated/device_parity_tracks_20260726/p14/p14_promotion_clean_20260726`
- Torch-Spyre: `59545440f0e7091ff1b2f90df63580da1842f3fe`
- FMS: `61bc991b175103e80cb8202b24a66ba7dbe79d1b`
- DeepTools: `406142afb9f080b9271e7c565a757ab8d8b5ed8f`
- `DXP_LX_FRAC_AVAIL=0.2`
- No attention oracle, phase-program change, commit, push, merge, or PR.

The bracket enabled exactly the accepted P12 residual handoff and the fused
P14/P13 final-stage sequence. Rejected P03/P05 normalization/MLP work and all
attention relayouts were disabled.

## Full-model T-C-T-C

Granite 3.3 8B, B1/S512 prefill, FP16, unfused weights, SDPA, five measured
requests per run:

| Run | Planner | 40-layer block mean (us) | Final stage mean (us) | Kernel-sum mean (us) | First-to-last span (us) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `combined_clean_t1_5x_20260726_a` | 1 | 385,200.901 | 3,529.141 | 388,815.284 | 506,267.800 |
| `combined_clean_c1_5x_20260726_b` | 0 | 388,135.724 | 4,172.215 | 392,393.885 | 512,736.099 |
| `combined_clean_t2_5x_20260726_c` | 1 | 382,632.224 | 3,704.486 | 386,421.871 | 505,137.061 |
| `combined_clean_c2_5x_20260726_d` | 0 | 385,889.551 | 4,372.903 | 390,347.678 | 509,158.872 |

Midpoint treatment versus control:

- 40-layer block: **-3.096075 ms (-0.800%)**.
- Fused final stage: **-655.745 us (-15.348%)**.
- Whole-request device-kernel sum: **-3.752204 ms (-0.959%)**.
- First-to-last device span: **-5.245055 ms (-1.027%)**.
- Every trace contained 210 kernel events and zero zero-duration events.

This is a clean composition result. It does not pass the SenDNN prefill device
gate: the candidate midpoint kernel sum is 387.619 ms versus the 192.310 ms
pass condition. These measurements are for the current accepted-edge stack,
not a claim that every previously measured Torch optimization is represented.

## Correctness

Each run saved six full-model FP16 logit tensors. Using T1 as the aligned
reference for C1, T2, and C2 compared 887,040 values:

- zero mismatches;
- maximum absolute difference zero;
- no shape or metadata mismatch;
- token 203 in every dump.

## Realized on-chip transport

Each treatment emitted exactly three plan/allocation records and the three
intended post-insertion payloads:

- P12: `relayout_debug_45_shuffle_input0.json`;
- P14: `relayout_debug_7_shuffle_input0.json`;
- P13: `relayout_debug_9_shuffle_input0.json`.

All three payloads contain `STCDPOpLx`. Every labeled source and destination
memory organization is `type: lx`, and both LDS entries on each edge have
`hbmSize_ = -1` and `hbmStartAddress_ = -1`. Thus none of the three claimed
transports has an HBM endpoint.

The two treatment compilations produced byte-identical payloads:

| Edge | Relayout payload SHA-256 | Original shuffle SHA-256 |
| --- | --- | --- |
| P12 | `24380524b8e5efe05ca6fb7516fc42874ce7db673f0d8292b1fb25ad77f3c05e` | `e685684a27662c4cc8ecea554138b6eabb3461b8bc9638cd5a932eed97fad3fb` |
| P14 | `1d879de6d302e5c93338d2ddf291cfcd98391808c6db75b13a31755a2a25e0d2` | `69cb1b6b0c20008d6c30431f3b408c0ce0fa58581f25784d18f24924c706205f` |
| P13 | `1f2173ab0be5ddcb31c537f288eae6a9266b1b952cba813c74418fa5c1b02760` | `c63363f020e80bc731b4412811668c2274563c4f19433fabd255c51ce39e0e25` |

## Decision

P12, P13, and P14 compose correctly and retain a positive measured effect in
one clean full-model candidate. Keep the stack. The source patch still needs
hunk minimization because the validated P14-selected Torch patch includes
disabled experiment infrastructure. Attention and decode lanes should remain
independent until they pass the same full-model correctness and emitted-
transport gate.
