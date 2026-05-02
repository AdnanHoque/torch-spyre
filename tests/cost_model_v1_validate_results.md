Loaded 208 (shape, split) samples from diag_split_gap_results.md
Across 13 unique shapes

# Cost-model v1 validation — Phase 1.2

## Default constants (initial guesses)

  PER_CORE_TFLOPS    = 0.5
  EFFECTIVE_DDR_BW   = 200.0 GB/s
  LAUNCH_FLOOR_MS    = 3.0

  Wall-time MAPE   : 32.1%
  Top-1 best-split : 15.4% (13 shapes)
  Top-3 best-split : 15.4%
  Top-5 best-split : 23.1%

## Calibrated constants (grid search)

  Grid:  TFLOPS ∈ [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
         BW     ∈ [50.0, 100.0, 150.0, 200.0, 300.0, 500.0, 800.0, 1200.0, 2000.0] GB/s

  Best PER_CORE_TFLOPS = 0.1
  Best EFFECTIVE_DDR_BW = 800.0 GB/s

  Wall-time MAPE   : 15.4%  (default 32.1%)
  Top-1 best-split : 15.4%  (default 15.4%)
  Top-3 best-split : 15.4%  (default 15.4%)
  Top-5 best-split : 23.1%  (default 23.1%)

## Per-shape breakdown

| shape | best measured | best predicted | rank of best meas | shape MAPE |
|---|---|---|---:|---:|
| L3-8B q_proj prefill | (1,32,1) @ 3.24ms | (1,1,32) | 6/21 | 16.7% |
| L3-8B GQA kv_proj prefill | (2,16,1) @ 3.07ms | (1,1,32) | 10/20 | 5.6% |
| L3-8B MLP gate/up prefill | (1,32,1) @ 3.77ms | (1,1,32) | 6/21 | 16.6% |
| L3-8B MLP down prefill | (2,1,16) @ 4.20ms | (1,1,32) | 7/21 | 14.5% |
| L3-70B q_proj prefill | (2,16,1) @ 4.02ms | (1,1,32) | 11/21 | 20.7% |
| L3-70B GQA kv_proj prefill | (2,16,1) @ 3.16ms | (1,1,32) | 10/20 | 8.6% |
| L3-70B GQA TP=8 kv prefill | (32,1,1) @ 3.00ms | (1,1,32) | 11/11 | 3.0% |
| L3-70B MLP down prefill | (16,2,1) @ 8.03ms | (1,2,16) | 15/15 | 61.0% |
| Mixtral down per-expert | (2,1,16) @ 4.23ms | (1,1,32) | 7/21 | 14.3% |
| Qwen3-MoE gate per-expert | (1,1,32) @ 3.10ms | (1,1,32) | 1/18 | 5.9% |
| DeepSeek-MoE gate (M=192) | (8,1,4) @ 3.15ms | (1,1,32) | 7/11 | 6.7% |
| L3-8B q_proj decode | (1,16,2) @ 3.21ms | (1,1,32) | 5/6 | 6.8% |
| L3-70B GQA TP=8 kv decode | (1,1,32) @ 3.06ms | (1,1,32) | 1/2 | 2.1% |

## Verdict

  v1 top-1 = 15%, top-3 = 15%. Model is too coarse — Phase 1.3 should add missing terms (e.g. split-dependent BW, output-reduction cost for k>1).
