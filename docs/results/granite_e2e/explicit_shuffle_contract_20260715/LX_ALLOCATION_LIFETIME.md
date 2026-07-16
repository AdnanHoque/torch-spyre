# LX Allocation And Lifetime Evidence

## Required Live Regions

The K-side all-gather expands each core's compact K shard from 128 KiB to a
complete 1 MiB score-BMM operand. S1 and S2 overlap in time during SHUFFLE.

The real workload also has a 512 KiB attention buffer live at the same boundary.
The observed allocation is:

| Region | Address range | Size/core | Boundary lifetime |
|---|---|---:|---|
| `buf5` | `0x0` onward | 16 KiB | live |
| S1 / `buf26` | `0x24000..0x44000` | 128 KiB | live through relayout |
| `buf7` | `0x44000..0xc4000` | 512 KiB | live |
| S2 | `0xc4000..0x1c4000` | 1 MiB | relayout through score BMM |

Consequently, the isolated fixture's `0x44000` destination is invalid for the
full workload even though it does not overlap S1. It aliases live `buf7`.

## Capacity Evidence

With the default `DXP_LX_FRAC_AVAIL=0.2`, Torch reports an approximately
1638 KiB scratchpad limit. S2 ends around 1808 KiB (`0x1c4000`), so the explicit
full operand does not fit and the candidate falls back.

The value-correct custom-materializer control used one consistent partition:

```text
DXP_LX_FRAC_AVAIL=0.07
```

Both Torch and Deeptools saw that same value. Torch then had enough frontend LX
for the explicit S2 placement while leaving approximately seven percent for
backend workspace. This is distinct from the unsafe historical split-environment
experiment where Torch and Deeptools each behaved as if they owned the same LX.

## Production Requirement

The production contract must not depend on a hard-coded global fraction. The
frontend allocator must model S2 as a real 1 MiB/core lifetime-bound region and
emit its address and extent. Deeptools must consume that exact allocation and
must not allocate another full-size destination. If the combined live set does
not fit, the compiler must fail closed to the existing HBM path.

Evidence archived in this directory:

```text
allocation_evidence/default_0p2_scratchpad_excerpt.txt
allocation_evidence/frac0p07_value_correct/env.txt
allocation_evidence/frac0p07_value_correct/structural_summary.json
allocation_evidence/frac0p07_value_correct/post_relayout_sdsc_6.json
allocation_evidence/frac0p07_value_correct/report.txt
```

The latter records destination address `802816` (`0xc4000`) and a value-correct
Lq=512 integration control.
