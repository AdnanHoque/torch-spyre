# Granite 3.3 8B B1/S512 prefill relayout snapshot

Date: 2026-07-26

Scope: device kernel/program time only, `DXP_LX_FRAC_AVAIL=0.2`, five measured requests. The accepted arm contains P12/P13/P14. P06 is excluded because the combined stack failed during compilation before device execution.

| Arm | Median device time | Mean | Min–max |
|---|---:|---:|---:|
| Relayout-disabled control | 394.507 ms | 394.556 ms | 394.221–394.951 ms |
| P12/P13/P14 | 386.286 ms | 386.243 ms | 386.029–386.502 ms |

The accepted stack saves 8.222 ms, a 2.084% device-time reduction (1.0213x speedup).

The measured SenDNN gate is 192.310 ms. The accepted stack remains 193.976 ms above it and is 2.0087x the gate time.

Correctness: all five measured prefill logits, plus the captured warmup prefill logits, are bit-exact against control (`max_abs=0`).

Emitted transport: prefill shuffles 45, 7, and 9 each have a post-PCFG payload containing exactly one `STCDPOpLx`; none of those three payloads contains an HBM transport marker.

Decode is intentionally outside this timing snapshot.
