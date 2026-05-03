# LX scratchpad catalog sweep — Phase 1

# Shapes: 13, Configs: 7, Total runs: 91
# warmup=3 iters=20


## L3-8B q_proj prefill (128, 4096, 4096)
  [1/91 t=0s] control ...  3.772 ms
  [2/91 t=19s] frac=0.2 (default) ...  3.868 ms
  [3/91 t=30s] frac=0.4 ...  3.659 ms
  [4/91 t=40s] frac=0.6 ...  3.566 ms
  [5/91 t=51s] frac=0.8 ...  3.484 ms
  [6/91 t=61s] frac=0.95 ...  3.502 ms
  [7/91 t=71s] compound: ep + frac=0.8 ...  3.162 ms

## L3-8B GQA kv_proj prefill (128, 1024, 4096)
  [8/91 t=82s] control ...  3.125 ms
  [9/91 t=92s] frac=0.2 (default) ...  3.186 ms
  [10/91 t=102s] frac=0.4 ...  3.077 ms
  [11/91 t=112s] frac=0.6 ...  3.128 ms
  [12/91 t=122s] frac=0.8 ...  3.126 ms
  [13/91 t=133s] frac=0.95 ...  3.052 ms
  [14/91 t=143s] compound: ep + frac=0.8 ...  2.962 ms

## L3-8B MLP gate/up prefill (128, 14336, 4096)
  [15/91 t=153s] control ...  3.780 ms
  [16/91 t=163s] frac=0.2 (default) ...  3.692 ms
  [17/91 t=174s] frac=0.4 ...  4.408 ms
  [18/91 t=185s] frac=0.6 ...  4.420 ms
  [19/91 t=196s] frac=0.8 ...  4.505 ms
  [20/91 t=206s] frac=0.95 ...  4.546 ms
  [21/91 t=217s] compound: ep + frac=0.8 ...  4.501 ms

## L3-8B MLP down prefill (128, 4096, 14336)
  [22/91 t=228s] control ...  5.984 ms
  [23/91 t=239s] frac=0.2 (default) ...  5.908 ms
  [24/91 t=249s] frac=0.4 ...  5.292 ms
  [25/91 t=260s] frac=0.6 ...  5.129 ms
  [26/91 t=271s] frac=0.8 ...  5.167 ms
  [27/91 t=282s] frac=0.95 ...  5.080 ms
  [28/91 t=293s] compound: ep + frac=0.8 ...  4.573 ms

## L3-70B q_proj prefill (128, 8192, 8192)
  [29/91 t=304s] control ...  6.478 ms
  [30/91 t=315s] frac=0.2 (default) ...  6.434 ms
  [31/91 t=326s] frac=0.4 ...  5.774 ms
  [32/91 t=337s] frac=0.6 ...  5.390 ms
  [33/91 t=348s] frac=0.8 ...  5.417 ms
  [34/91 t=359s] frac=0.95 ...  5.350 ms
  [35/91 t=370s] compound: ep + frac=0.8 ...  3.965 ms

## L3-70B GQA kv_proj prefill (128, 1024, 8192)
  [36/91 t=381s] control ...  3.324 ms
  [37/91 t=391s] frac=0.2 (default) ...  3.351 ms
  [38/91 t=401s] frac=0.4 ...  3.337 ms
  [39/91 t=412s] frac=0.6 ...  3.303 ms
  [40/91 t=422s] frac=0.8 ...  3.280 ms
  [41/91 t=432s] frac=0.95 ...  3.275 ms
  [42/91 t=442s] compound: ep + frac=0.8 ...  3.044 ms

## L3-70B GQA TP=8 kv prefill (128, 128, 8192)
  [43/91 t=452s] control ...  2.881 ms
  [44/91 t=462s] frac=0.2 (default) ...  2.881 ms
  [45/91 t=472s] frac=0.4 ...  2.950 ms
  [46/91 t=482s] frac=0.6 ...  2.964 ms
  [47/91 t=492s] frac=0.8 ...  2.911 ms
  [48/91 t=501s] frac=0.95 ...  2.897 ms
  [49/91 t=511s] compound: ep + frac=0.8 ...  2.904 ms

## L3-70B MLP down prefill (128, 8192, 28672)
  [50/91 t=521s] control ...  7.881 ms
  [51/91 t=535s] frac=0.2 (default) ...  7.884 ms
  [52/91 t=548s] frac=0.4 ...  7.598 ms
  [53/91 t=562s] frac=0.6 ...  7.413 ms
  [54/91 t=575s] frac=0.8 ...  7.575 ms
  [55/91 t=589s] frac=0.95 ...  7.552 ms
  [56/91 t=602s] compound: ep + frac=0.8 ...  7.591 ms

## Mixtral down per-expert (128, 4096, 14336)
  [57/91 t=616s] control ...  5.921 ms
  [58/91 t=627s] frac=0.2 (default) ...  5.962 ms
  [59/91 t=637s] frac=0.4 ...  5.301 ms
  [60/91 t=648s] frac=0.6 ...  5.107 ms
  [61/91 t=659s] frac=0.8 ...  5.085 ms
  [62/91 t=670s] frac=0.95 ...  5.011 ms
  [63/91 t=681s] compound: ep + frac=0.8 ...  4.574 ms

## Qwen3-MoE gate per-expert (128, 1536, 2048)
  [64/91 t=692s] control ...  3.099 ms
  [65/91 t=702s] frac=0.2 (default) ...  3.090 ms
  [66/91 t=713s] frac=0.4 ...  3.014 ms
  [67/91 t=723s] frac=0.6 ...  3.054 ms
  [68/91 t=733s] frac=0.8 ...  2.991 ms
  [69/91 t=743s] frac=0.95 ...  2.996 ms
  [70/91 t=753s] compound: ep + frac=0.8 ...  3.006 ms

## DeepSeek-MoE gate (M=192) (192, 1408, 2048)
  [71/91 t=763s] control ...  3.085 ms
  [72/91 t=773s] frac=0.2 (default) ...  3.066 ms
  [73/91 t=783s] frac=0.4 ...  3.138 ms
  [74/91 t=793s] frac=0.6 ...  3.081 ms
  [75/91 t=803s] frac=0.8 ...  3.153 ms
  [76/91 t=813s] frac=0.95 ...  3.177 ms
  [77/91 t=823s] compound: ep + frac=0.8 ...  2.891 ms

## L3-8B q_proj decode (1, 4096, 4096)
  [78/91 t=833s] control ...  3.111 ms
  [79/91 t=843s] frac=0.2 (default) ...  3.081 ms
  [80/91 t=853s] frac=0.4 ...  3.092 ms
  [81/91 t=863s] frac=0.6 ...  3.134 ms
  [82/91 t=873s] frac=0.8 ...  3.093 ms
  [83/91 t=883s] frac=0.95 ...  3.094 ms
  [84/91 t=893s] compound: ep + frac=0.8 ...  3.106 ms

## L3-70B GQA TP=8 kv decode (1, 128, 8192)
  [85/91 t=903s] control ...  2.911 ms
  [86/91 t=913s] frac=0.2 (default) ...  2.995 ms
  [87/91 t=923s] frac=0.4 ...  2.932 ms
  [88/91 t=933s] frac=0.6 ...  2.900 ms
  [89/91 t=942s] frac=0.8 ...  2.967 ms
  [90/91 t=952s] frac=0.95 ...  2.941 ms
  [91/91 t=962s] compound: ep + frac=0.8 ...  2.974 ms


## Median wall time per (shape, config) — ms

| shape | control | frac=0.2 (default) | frac=0.4 | frac=0.6 | frac=0.8 | frac=0.95 | compound: ep + frac=0.8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| L3-8B q_proj prefill | 3.772 | 3.868 | 3.659 | 3.566 | 3.484 | 3.502 | 3.162 |
| L3-8B GQA kv_proj prefill | 3.125 | 3.186 | 3.077 | 3.128 | 3.126 | 3.052 | 2.962 |
| L3-8B MLP gate/up prefill | 3.780 | 3.692 | 4.408 | 4.420 | 4.505 | 4.546 | 4.501 |
| L3-8B MLP down prefill | 5.984 | 5.908 | 5.292 | 5.129 | 5.167 | 5.080 | 4.573 |
| L3-70B q_proj prefill | 6.478 | 6.434 | 5.774 | 5.390 | 5.417 | 5.350 | 3.965 |
| L3-70B GQA kv_proj prefill | 3.324 | 3.351 | 3.337 | 3.303 | 3.280 | 3.275 | 3.044 |
| L3-70B GQA TP=8 kv prefill | 2.881 | 2.881 | 2.950 | 2.964 | 2.911 | 2.897 | 2.904 |
| L3-70B MLP down prefill | 7.881 | 7.884 | 7.598 | 7.413 | 7.575 | 7.552 | 7.591 |
| Mixtral down per-expert | 5.921 | 5.962 | 5.301 | 5.107 | 5.085 | 5.011 | 4.574 |
| Qwen3-MoE gate per-expert | 3.099 | 3.090 | 3.014 | 3.054 | 2.991 | 2.996 | 3.006 |
| DeepSeek-MoE gate (M=192) | 3.085 | 3.066 | 3.138 | 3.081 | 3.153 | 3.177 | 2.891 |
| L3-8B q_proj decode | 3.111 | 3.081 | 3.092 | 3.134 | 3.093 | 3.094 | 3.106 |
| L3-70B GQA TP=8 kv decode | 2.911 | 2.995 | 2.932 | 2.900 | 2.967 | 2.941 | 2.974 |

## Speedup vs control (LX_PLANNING=0)

| shape | frac=0.2 (default) | frac=0.4 | frac=0.6 | frac=0.8 | frac=0.95 | compound: ep + frac=0.8 |
|---|---:|---:|---:|---:|---:|---:|
| L3-8B q_proj prefill | 0.975x | 1.031x | 1.058x ✓ | 1.083x ✓ | 1.077x ✓ | 1.193x ✓✓ |
| L3-8B GQA kv_proj prefill | 0.981x | 1.016x | 0.999x | 1.000x | 1.024x | 1.055x ✓ |
| L3-8B MLP gate/up prefill | 1.024x | 0.857x ✗ | 0.855x ✗ | 0.839x ✗ | 0.831x ✗ | 0.840x ✗ |
| L3-8B MLP down prefill | 1.013x | 1.131x ✓✓ | 1.167x ✓✓ | 1.158x ✓✓ | 1.178x ✓✓ | 1.309x ✓✓ |
| L3-70B q_proj prefill | 1.007x | 1.122x ✓✓ | 1.202x ✓✓ | 1.196x ✓✓ | 1.211x ✓✓ | 1.634x ✓✓ |
| L3-70B GQA kv_proj prefill | 0.992x | 0.996x | 1.006x | 1.013x | 1.015x | 1.092x ✓ |
| L3-70B GQA TP=8 kv prefill | 1.000x | 0.977x | 0.972x | 0.990x | 0.995x | 0.992x |
| L3-70B MLP down prefill | 1.000x | 1.037x | 1.063x ✓ | 1.040x | 1.044x | 1.038x |
| Mixtral down per-expert | 0.993x | 1.117x ✓✓ | 1.159x ✓✓ | 1.164x ✓✓ | 1.182x ✓✓ | 1.294x ✓✓ |
| Qwen3-MoE gate per-expert | 1.003x | 1.028x | 1.015x | 1.036x | 1.034x | 1.031x |
| DeepSeek-MoE gate (M=192) | 1.006x | 0.983x | 1.001x | 0.978x | 0.971x | 1.067x ✓ |
| L3-8B q_proj decode | 1.010x | 1.006x | 0.993x | 1.006x | 1.005x | 1.002x |
| L3-70B GQA TP=8 kv decode | 0.972x | 0.993x | 1.004x | 0.981x | 0.990x | 0.979x |

## Best LX frac per shape (no element_priority)

| shape | best frac | speedup vs control |
|---|---|---:|
| L3-8B q_proj prefill | frac=0.8 | 1.083x |
| L3-8B GQA kv_proj prefill | frac=0.95 | 1.024x |
| L3-8B MLP gate/up prefill | frac=0.2 (default) | 1.024x |
| L3-8B MLP down prefill | frac=0.95 | 1.178x |
| L3-70B q_proj prefill | frac=0.95 | 1.211x |
| L3-70B GQA kv_proj prefill | frac=0.95 | 1.015x |
| L3-70B GQA TP=8 kv prefill | frac=0.2 (default) | 1.000x |
| L3-70B MLP down prefill | frac=0.6 | 1.063x |
| Mixtral down per-expert | frac=0.95 | 1.182x |
| Qwen3-MoE gate per-expert | frac=0.8 | 1.036x |
| DeepSeek-MoE gate (M=192) | frac=0.2 (default) | 1.006x |
| L3-8B q_proj decode | frac=0.2 (default) | 1.010x |
| L3-70B GQA TP=8 kv decode | frac=0.6 | 1.004x |

## Compound: element_priority + frac=0.8 vs each lever alone

| shape | control | LX frac=0.8 alone | element_priority alone* | compound (both) | compound speedup |
|---|---:|---:|---:|---:|---:|

*element_priority alone numbers from previous compare results (committed `0ff598a`).

| L3-8B q_proj prefill | 3.77 | 3.48 | 3.24 | 3.16 | 1.193x |
| L3-8B GQA kv_proj prefill | 3.13 | 3.13 | 3.04 | 2.96 | 1.055x |
| L3-8B MLP gate/up prefill | 3.78 | 4.50 | 3.78 | 4.50 | 0.840x |
| L3-8B MLP down prefill | 5.98 | 5.17 | 4.64 | 4.57 | 1.309x |
| L3-70B q_proj prefill | 6.48 | 5.42 | 4.05 | 3.97 | 1.634x |
| L3-70B GQA kv_proj prefill | 3.32 | 3.28 | 3.13 | 3.04 | 1.092x |
| L3-70B GQA TP=8 kv prefill | 2.88 | 2.91 | 3.00 | 2.90 | 0.992x |
| L3-70B MLP down prefill | 7.88 | 7.57 | 8.03 | 7.59 | 1.038x |
| Mixtral down per-expert | 5.92 | 5.08 | 4.65 | 4.57 | 1.294x |
| Qwen3-MoE gate per-expert | 3.10 | 2.99 | 3.05 | 3.01 | 1.031x |
| DeepSeek-MoE gate (M=192) | 3.08 | 3.15 | 3.00 | 2.89 | 1.067x |
| L3-8B q_proj decode | 3.11 | 3.09 | 3.15 | 3.11 | 1.002x |
| L3-70B GQA TP=8 kv decode | 2.91 | 2.97 | 3.00 | 2.97 | 0.979x |


# Total wall time: 972s
