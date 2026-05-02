# Cross-kernel weight residency probe

PyTorch:        2.10.0+cpu
torch_spyre:    (editable)
warmup iters:   5
measure iters:  30
N back-to-back: 8

**Hypothesis under test**: does Spyre's device-side runtime / scratchpad reuse weight data across consecutive kernel calls when the same W tensor is referenced? If yes, MoE-style expert weight residency is a real planner-level lever.

**Method**: bench N back-to-back `mm(a, W)` calls with the SAME W vs. with N DIFFERENT Ws. Per-call ratio < 1 indicates same-W is faster (caching evidence); ratio = 1 indicates per-launch overhead and DDR streaming reset between every kernel.

| shape | config | total ms | per-call ms |
|---|---|---:|---:|
| small (per-launch dominated) | single (baseline) | 2.93 | 2.934 |
| small (per-launch dominated) | same-W ×8 | 23.52 | 2.940 |
| small (per-launch dominated) | different-W ×8 | 23.34 | 2.918 |
| medium (compute visible) | single (baseline) | 3.92 | 3.925 |
| medium (compute visible) | same-W ×8 | 31.32 | 3.915 |
| medium (compute visible) | different-W ×8 | 31.17 | 3.897 |
| large weight (BW potentially visible) | single (baseline) | 9.24 | 9.236 |
| large weight (BW potentially visible) | same-W ×8 | 73.22 | 9.153 |
| large weight (BW potentially visible) | different-W ×8 | 74.21 | 9.276 |

### Per-call ratio (same-W vs different-W)

| shape | same-W per call | different-W per call | ratio | verdict |
|---|---:|---:|---:|---|
| small (per-launch dominated) | 2.940 | 2.918 | 1.008× | tied (no reuse at kernel-boundary granularity) |
| medium (compute visible) | 3.915 | 3.897 | 1.005× | tied (no reuse at kernel-boundary granularity) |
| large weight (BW potentially visible) | 9.153 | 9.276 | 0.987× | tied (no reuse at kernel-boundary granularity) |
