# Granite Fused-Block Effective Peak Table

This table answers two separate questions from the current fused GraniteBlock oracle:

1. Did the cost model pick the best measured work split?
2. Given the measured fused-window time, what is the target matmul effective throughput relative to the nominal DL16 peak?

Important caveat: the profiler window is the selected fused block part, not an isolated matmul kernel. For `mlp` it includes the MLP fused region; for `attn` it includes attention fused work; for `block` it can include the whole block path. Therefore target-matmul TFLOP/s is a conservative lower bound and should not be read as isolated kernel PT-util.

Nominal DL16 peak used here: `98.304 TFLOP/s` (`49.152 TMAC/s`).

| phase | role | shape | best split | best fused-window us | cost-model split | split gap | target TFLOP/s lower bound | target peak % lower bound | notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| prefill | QK^T | `B=32, M=512, N=512, K=128` | `1_4_4_2` | 12452.7 | `1_16_2_1` | 2.40% | 0.17 | 0.2% | attention-window; target-only lower bound |
| prefill | attn@V | `B=32, M=512, N=128, K=512` | `1_32_1_1` | 12573.7 | `1_32_1_1` | 0.00% | 0.17 | 0.2% | attention-window; target-only lower bound |
| prefill | K/V proj | `[[1, 512, 4096], [4096, 1024]]` | `1_8_4_1` | 12561.7 | `1_8_4_1` | 0.00% | 0.34 | 0.3% | attention-window; target-only lower bound |
| prefill | Q/O proj | `[[1, 512, 4096], [4096, 4096]]` | `1_8_4_1` | 23190.1 | `1_4_8_1` | 0.38% | 0.74 | 0.8% | block-window; target-only lower bound is especially conservative |
| prefill | MLP up/gate | `[[1, 512, 4096], [4096, 12800]]` | `1_4_8_1` | 10901.9 | `1_4_8_1` | 0.00% | 4.92 | 5.0% | MLP-window approx full-MLP 14.8 TFLOP/s / 15.0% peak |
| prefill | MLP down | `[[1, 512, 12800], [12800, 4096]]` | `1_8_4_1` | 10366.4 | `1_8_4_1` | 0.00% | 5.18 | 5.3% | MLP-window approx full-MLP 15.5 TFLOP/s / 15.8% peak |
| decode | QK^T | `B=32, M=64, N=576, K=128` | `1_4_3_2` | 5018.2 | `4_4_1_2` | 0.30% | 0.06 | 0.1% | attention-window; target-only lower bound |
| decode | attn@V | `B=32, M=64, N=128, K=576` | `16_2_1_1` | 7058.9 | `8_4_1_1` | 0.43% | 0.04 | 0.0% | attention-window; target-only lower bound |
| decode | K/V proj | `[[1, 64, 4096], [4096, 1024]]` | `1_4_8_1` | 5414.8 | `1_4_4_2` | 0.84% | 0.10 | 0.1% | attention-window; target-only lower bound |
| decode | Q/O proj | `[[1, 64, 4096], [4096, 4096]]` | `1_4_8_1` | 15533.7 | `1_4_8_1` | 0.00% | 0.14 | 0.1% | block-window; target-only lower bound is especially conservative |
| decode | MLP up/gate | `[[1, 64, 4096], [4096, 12800]]` | `1_4_8_1` | 9266.3 | `1_4_8_1` | 0.00% | 0.72 | 0.7% | MLP-window approx full-MLP 2.2 TFLOP/s / 2.2% peak |
| decode | MLP down | `[[1, 64, 12800], [12800, 4096]]` | `1_4_8_1` | 7119.8 | `1_4_8_1` | 0.00% | 0.94 | 1.0% | MLP-window approx full-MLP 2.8 TFLOP/s / 2.9% peak |

## Readout

- Work-split capture is strong: over the rows in this table, the selected measured time is `0.36%` over the best measured forced split.
- The low target-only peak percentages should not be interpreted as isolated matmul inefficiency. They mostly reflect that the measurement window includes surrounding fused work, and for `block` roles can include much more than the target projection.
- The clean isolated-kernel peak question still requires a matmul-only profiler table using the same selected splits. This fused-block table is suitable for split-selection validation and conservative lower-bound throughput, not final PT-util claims.

## Source Artifacts

- first pass: `fused_split_oracle_trace_firstpass_20260618_220301`
- fill holes: `fused_split_oracle_trace_fillholes_20260618_222115`
- full-core ambiguous roles: `fused_split_oracle_fullcore_20260618_224201`
- candidate decode K/V check: `fused_split_oracle_trace_candidate_kv_20260619_022103`
