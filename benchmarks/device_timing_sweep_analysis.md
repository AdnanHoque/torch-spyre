# Granite matmul work-division: profiled device-timing sweep

Definitive measurement of whether the work-division cost model picks the
**device-optimal** core split for each of the 12 golden Granite-8B matmul
shapes. Supersedes the earlier wall-clock `stable_resweep_verdict.md` (wall-clock
buried real per-kernel gaps under ~5.7 ms fixed host overhead).

## Method

- **Metric:** `self_device_time_total` from `torch.profiler` (CPU +
  PrivateUse1 activities), summed over the matmul kernel, per rep (20 reps).
  On-device, <1% run-to-run noise.
- **Device:** harvested +148 SDK on NFS, clc card `ba:00.0` (the stable
  firmware). `USE_SPYRE_PROFILER=1` `_C.so`.
- **Sweep:** for each shape, every 32-core-feasible `(batch, M, N, K)` split
  is *forced* (cost-function patch) and timed — 240 splits total, 236 valid.
  Device-best = min over a shape's splits.
- **Cost-model picks:** ground-truth, read from the emitted SDSC
  `numWkSlicesPerDim_` of a real compile (not a standalone cost estimate).
  - **MAIN** = upstream `main` cost model.
  - **decode-fix** = `cost-model-min-cores-fix` branch (argmin restricted to
    cores ≥ default + K-tax + scoped batch penalty), run on-device via a
    pure-Python overlay of its `work_division.py`.

Splits are core-counts per axis: `batch_M_N_K`. N and K are in stick groups
(64 fp16 elems/stick). `1_4_8_1` = M split 4, N split 8, 32 cores.

## device-best vs cost-model pick

| op | phase | shape B·M·N·K | device-best | µs | MAIN pick | µs | gap | decode-fix pick | µs | gap |
|---|---|---|---|--:|---|--:|--:|---|--:|--:|
| QK^T | prefill | 512x32x512x128 | `1_2_8_2` | 1009 | `32_1_1_1` | 1827 | +81% | `1_4_8_1` | 1630 | +62% |
| attn@V | prefill | 32x512x128x512 | `4_4_2_1` | 395 | `1_32_1_1` | 796 | +101% | `2_8_2_1` | 406 | +3% |
| Q/O proj | prefill | 1x512x4096x4096 | `1_4_8_1` | 317 | `1_4_8_1` | 317 | +0% | `1_4_8_1` | 317 | +0% |
| K/V proj | prefill | 1x512x1024x4096 | `1_4_8_1` | 89 | `1_4_8_1` | 89 | +0% | `1_4_8_1` | 89 | +0% |
| MLP up | prefill | 1x512x12800x4096 | `1_4_8_1` | 1017 | `1_4_8_1` | 1017 | +0% | `1_4_8_1` | 1017 | +0% |
| MLP down | prefill | 1x512x4096x12800 | `1_4_8_1` | 899 | `1_4_8_1` | 899 | +0% | `1_4_8_1` | 899 | +0% |
| QK^T | decode | 64x32x576x128 | `4_4_1_2` | 158 | `32_1_1_1` | 259 | +64% | `1_32_1_1` | 793 | **+401%** |
| attn@V | decode | 32x64x128x576 | `2_8_2_1` | 60 | `1_32_1_1` | 343 | +474% | `1_16_2_1` | 105 | +75% |
| Q/O proj | decode | 1x64x4096x4096 | `1_4_8_1` | 221 | `1_32_1_1` | 843 | +282% | `1_4_8_1` | 221 | +0% |
| K/V proj | decode | 1x64x1024x4096 | `1_8_4_1` | 60 | `1_32_1_1` | 211 | +249% | `1_4_8_1` | 62 | +2% |
| MLP up | decode | 1x64x12800x4096 | `1_4_8_1` | 705 | `1_4_8_1` | 705 | +0% | `1_4_8_1` | 705 | +0% |
| MLP down | decode | 1x64x4096x12800 | `1_4_8_1` | 685 | `1_32_1_1` | 2561 | +274% | `1_4_8_1` | 685 | +0% |

**Totals:** device-best **5617 µs**, MAIN **9868 µs (+76%)**, decode-fix
**6930 µs (+23%)**.

## Findings

1. **MAIN is device-optimal on only 5 of 12 shapes** — the four prefill
   projection/MLP shapes plus decode MLP-up. It misses all four attention
   bmms and three of four decode projections, +76% device time aggregate.

2. **Every miss collapses to a default-distributor split** — pure-M
   `1_32_1_1` or pure-batch `32_1_1_1`. These are not cost argmins; the cost
   *function* correctly ranks them as the **worst** options (e.g. decode Q/O
   `1_32_1_1` = 888 cost-units, the highest of all its splits; QK^T prefill
   `32_1_1_1` = 65 405).

3. **Root cause: the trade-down guard** (`work_division.py:815`, *"never
   trade down to fewer cores than the default distributor"*). For these
   memory-bound decode/attention shapes the genuinely-cheapest split uses
   **fewer than 32 cores** (decode Q/O wants `1_1_8_2` = 16 cores; attn@V
   wants `1_8_2_1` = 16 cores). The guard rejects the good <32-core argmin and
   falls back to the default — which is the single worst split. The planner
   *engages* (cost called 55–86×) and computes the right answer, then discards
   it.

4. **decode-fix (restrict argmin to ≥ default cores) cuts the gap +76% → +23%.**
   Clean wins to ~device-best on attn@V prefill (+101%→+3%), Q/O decode
   (+282%→+0%), K/V decode (+249%→+2%), MLP-down decode (+274%→+0%); the five
   already-optimal shapes are unchanged.

5. **decode-fix does not solve QK^T, and regresses QK^T decode** (+64% →
   **+401%**, device-confirmed). With M=32 the only 32-core splits are heavy
   M-splits (`1_32_1_1` → 1 row/core) or batch-splits; the cost model ranks the
   M-split cheapest, but device wants the batch+K split `4_4_1_2`. attn@V
   decode also stays off-best (+75%, the cost model picks an M-split over the
   device-best batch-split). **QK^T (both phases) and decode attn@V need
   explicit batch/K-split modeling for tiny-M bmms — the core-count restriction
   alone is not enough, and for QK^T decode it is a net regression.**

## Raw data

- `device_timing_sweep_raw.txt` — all 236 valid `(shape, split) → µs` rows.
- `device_best_vs_picks.csv` — the table above, machine-readable.
