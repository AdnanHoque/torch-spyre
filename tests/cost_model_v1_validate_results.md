Loaded 208 (shape, split) samples from diag_split_gap_results.md
Across 13 unique shapes

# Cost-model v1 validation — Phase 1.2

## Default constants (initial guesses)

  PER_CORE_TFLOPS    = 0.5
  EFFECTIVE_DDR_BW   = 200.0 GB/s
  SHARING_FACTOR     = 0.0
  LAUNCH_FLOOR_MS    = 3.0

  Wall-time MAPE   : 32.1%
  Top-1 best-split : 15.4% (13 shapes)
  Top-3 best-split : 15.4%
  Top-5 best-split : 23.1%
  Mean regret      : 1.098×  (1.0 = optimal pick)
  Max regret       : 1.356×

## Calibrated for wall-time MAPE

  PER_CORE_TFLOPS  = 0.1
  EFFECTIVE_DDR_BW = 500.0 GB/s
  SHARING_FACTOR   = 0.25

  Wall-time MAPE   : 15.1%  (default 32.1%)
  Top-1 best-split : 23.1%  (default 15.4%)
  Top-3 best-split : 30.8%  (default 15.4%)
  Top-5 best-split : 46.2%  (default 23.1%)
  Mean regret      : 1.059×  (default 1.098×)
  Max regret       : 1.356×  (default 1.356×)

## Calibrated for mean regret (planner-relevant)

  PER_CORE_TFLOPS  = 0.05
  EFFECTIVE_DDR_BW = 150.0 GB/s
  SHARING_FACTOR   = 0.0

  Wall-time MAPE   : 63.5%  (default 32.1%)
  Top-1 best-split : 23.1%  (default 15.4%)
  Top-3 best-split : 30.8%  (default 15.4%)
  Top-5 best-split : 38.5%  (default 23.1%)
  Mean regret      : 1.059×  (default 1.098×)
  Max regret       : 1.356×  (default 1.356×)

## Per-shape breakdown

| shape | best measured | model picks | regret | rank | shape MAPE |
|---|---|---|---:|---:|---:|
| L3-8B q_proj prefill | (1,32,1) @ 3.24ms | (1,1,32) @ 3.53ms | 1.09× | 6/21 | 18.7% |
| L3-8B GQA kv_proj prefill | (2,16,1) @ 3.07ms | (1,1,32) @ 3.15ms | 1.03× | 10/20 | 5.6% |
| L3-8B MLP gate/up prefill | (1,32,1) @ 3.77ms | (1,32,1) @ 3.77ms | 1.00× | 1/21 | 113.7% |
| L3-8B MLP down prefill | (2,1,16) @ 4.20ms | (1,32,1) @ 4.64ms | 1.10× | 17/21 | 101.3% |
| L3-70B q_proj prefill | (2,16,1) @ 4.02ms | (1,32,1) @ 4.06ms | 1.01× | 2/21 | 128.6% |
| L3-70B GQA kv_proj prefill | (2,16,1) @ 3.16ms | (1,1,32) @ 3.27ms | 1.03× | 10/20 | 8.1% |
| L3-70B GQA TP=8 kv prefill | (32,1,1) @ 3.00ms | (1,1,32) @ 3.06ms | 1.02× | 11/11 | 3.0% |
| L3-70B MLP down prefill | (16,2,1) @ 8.03ms | (1,32,1) @ 10.89ms | 1.36× | 15/15 | 197.8% |
| Mixtral down per-expert | (2,1,16) @ 4.23ms | (1,32,1) @ 4.72ms | 1.12× | 17/21 | 99.9% |
| Qwen3-MoE gate per-expert | (1,1,32) @ 3.10ms | (1,1,32) @ 3.10ms | 1.00× | 1/18 | 5.9% |
| DeepSeek-MoE gate (M=192) | (8,1,4) @ 3.15ms | (1,1,32) @ 3.17ms | 1.01× | 7/11 | 6.7% |
| L3-8B q_proj decode | (1,16,2) @ 3.21ms | (1,1,32) @ 3.22ms | 1.00× | 5/6 | 6.8% |
| L3-70B GQA TP=8 kv decode | (1,1,32) @ 3.06ms | (1,1,32) @ 3.06ms | 1.00× | 1/2 | 2.1% |

## Verdict

  Top-1: 23%   Top-3: 31%   Mean regret: 1.06×   Max regret: 1.36×

  Mean regret ≤ 10%. Useful as a tiebreaker on top of the current planner, but not yet good enough to replace it. Consider Phase 1.3 (refine missing terms).
