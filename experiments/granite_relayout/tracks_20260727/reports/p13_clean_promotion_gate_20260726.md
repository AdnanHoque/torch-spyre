# P13 clean promotion gate — Granite 3.3 8B final LM-head all-gather

Date: 2026-07-26  
Status: **PASS for isolated promotion; targeted timing claim only**  
Attention scope: frozen; P13 is a non-attention final-hidden-state → LM-head edge.

## Frozen environment

- Pod: `a6-quantization/adnan-clc-spyre-dev-pf`
- Clean root: `/home/adnan/codex-isolated/device_parity_tracks_20260726/p13_clean_20260726`
- Torch-Spyre: `59545440f0e7091ff1b2f90df63580da1842f3fe`
- FMS: `61bc991b175103e80cb8202b24a66ba7dbe79d1b`
- DeepTools: `406142afb9f080b9271e7c565a757ab8d8b5ed8f`
- `DXP_LX_FRAC_AVAIL=0.2`
- No phase-program changes, commits, pushes, merges, or PRs.

## Clean patch surface

P13 contains only the following behavior:

- 28-owner LM-head output split with unsplit K reduction.
- Exact 32-source → 28-destination subset all-gather.
- Dense participant-union materialization and hidden-axis SDSC bridge.
- Fused last-token slice/head stage, opt-in `last_n_tokens=1`, and 50,176 padded-vocabulary support.
- Sparse destination-owner handling in DeepTools.

Patch checksums:

- Torch: `db651d8715dbac2eb508b8be077cd81ca41a62635b2e3aabb1aa4ab394df22b5`
- FMS: `1f91e61b1df4de1e7418d40d42ae291f2d22781ee60966e1ed3aaf25267f9b9a`
- DeepTools: `55500b8c52430f5c34e3c7c87b28ce86f66adeb9ba11d12115450474559b062f`

Validation before the device bracket:

- `git diff --check`: pass in all three trees.
- Three focused Torch P13 tests: pass.
- Three FMS last-token-policy tests: pass.
- Python syntax compilation: pass.
- Clean DeepTools 402-target build: pass.

## Full-model T–C–T–C gate

Granite 3.3 8B, B1/S512 prefill, one generated token, five measured requests per run:

| Run | Planner | Final-head mean (µs) | Whole-request kernel sum (µs) | First-to-last kernel span (µs) |
|---|---:|---:|---:|---:|
| T1 | 1 | 3,435.449 | 371,291.687 | 474,449.113 |
| C1 | 0 | 4,081.899 | 365,486.666 | 473,210.530 |
| T2 | 1 | 3,411.616 | 371,441.040 | 474,160.520 |
| C2 | 0 | 3,913.621 | 370,419.301 | 475,375.851 |

Pooled targeted result:

- Control final head: **3,997.760 µs**
- Treatment final head: **3,423.532 µs**
- Saving: **574.228 µs (14.364%)**
- Bracket deltas: **646.450 µs** and **502.006 µs**, both favorable.

The whole-request first-to-last span was effectively neutral: treatment was 11.626 µs slower on a 474 ms span (0.0025%). The summed-kernel metric moved against treatment by 3.413 ms because the forty-layer block varied between runs. Therefore P13 earns a targeted final-head promotion claim, not an independent full-model latency claim.

## Correctness gate

Each run saved six aligned logit dumps. Comparing all 24 dumps against C1 produced:

- 0 element mismatches.
- `max_abs = 0.0`.
- No shape or metadata mismatches.
- Generated token 203 in every dump.

This is bit-exact full-model output correctness, not a reduced subgraph check.

## Emitted on-chip transport proof

The two treatment compilations emitted byte-identical final-head artifacts:

- `bundle.mlir`: `18b6d52d7dab95a23ccfef829f3bf8b4a65a14eaa10974cec58177ab0c946224`
- Relayout `sdsc_1.json`: `c5a8e8ecb54f661e76883f0c6f43eca6c0bc14b4bf673e0e332e014f99b69e41`
- `init_binary.bin`: `2a13f1cb8e81fab2a50258f7486d3e968e03fdb6ae363f8594abef6d87e02e96`
- `spyrecode.json`: `760ddecf93e1e5c84e3787d618c36fe23b5bb4ef2cf13d8700025c23f444d79a`

The exact emitted relayout SDSC was then compiled alone with the clean DeepTools build and decoded through DIP:

- Standalone init binary: `414d159ab6b17278060ac4ba05370b4429392562d3598d139a8d378f3910a724`
- Standalone Spyre code: `508d0f2cc1a08e0f9b864563f31e14d53317891f305c9ed2f89fcffd0ef6d0a8`
- Both standalone hashes reproduce the independent earlier isolated decode.
- SDSC allocation and labeled memory organization place both `Tensor0` and `Tensor1` in LX.
- Decoded base programs contain 31 `LDGU`, 2 `STGU`, and LX `LDSTU` transfer instructions. The isolated relayout contains no ordinary HBM `LD`/`ST` program.

Topology derived from the emitted plan and the byte-identical emitted SDSC:

- Source: 32 owner shards × 128 FP16 elements = **32 × 256 B**.
- Destination: **28 owner cores**, each materializing the full 4,096-element hidden vector.
- Deliveries: **32 × 28 = 896**.
- Local: **28 deliveries / 7,168 B**.
- Remote: **868 deliveries / 222,208 B**.
- Endpoints: LX-only source and destination allocations.

This proof combines realized bundle identity, emitted SDSC placement, and decoded init-packet/ISA transport. Planner telemetry alone was not used as realization proof.

## Decision

P13 is independently promotable into the non-attention integration stack. Its isolated effect is a repeatable 574.228 µs reduction in the final LM-head kernel with bit-exact full-model outputs and realized on-chip transport. The next gate is a combined P12+P13+P14 clean bracket to determine how much of the targeted savings survives at the full-model SenDNN device gate.
