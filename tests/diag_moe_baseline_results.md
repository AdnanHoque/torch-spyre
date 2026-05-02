# MoE naive-baseline diagnostic — Phase 0a

PyTorch:        2.10.0+cpu
torch_spyre:    (editable)
warmup iters:   5
measure iters:  30
decode M:       1
per-iter sync:  torch_spyre.streams.synchronize() inside the timed loop

**Naive MoE step**: K active experts run as separate (gate, up, down) matmul calls in a Python loop, output is the weighted sum. Each expert's call is `silu(x @ W_gate) * (x @ W_up) @ W_down` — three matmul launches per active expert.

**Dense fallback**: column-stacked weights `(hidden -> E*intermediate)` for gate/up and `(E*intermediate -> hidden)` for down. ONE matmul per stage regardless of E, but computes all E experts' outputs (so compute is `E×` more than what an oracle MoE would need).

## H=1024, I=2048, E=8

| variant | median ms | per-active-expert ms | vs single-expert |
|---|---:|---:|---:|
| naive K=1 | 15.45 | 15.45 | 1.00× |
| naive K=2 | 30.72 | 15.36 | 1.99× |
| naive K=4 | 61.45 | 15.36 | 3.98× |
| naive K=8 | 123.10 | 15.39 | 7.97× |
| dense fallback (E experts always run) | 10.12 | — | 0.65× |

## H=1024, I=4096, E=8

| variant | median ms | per-active-expert ms | vs single-expert |
|---|---:|---:|---:|
| naive K=1 | 15.77 | 15.77 | 1.00× |
| naive K=2 | 31.24 | 15.62 | 1.98× |
| naive K=4 | 62.30 | 15.57 | 3.95× |
| naive K=8 | 123.51 | 15.44 | 7.83× |
| dense fallback (E experts always run) | 10.91 | — | 0.69× |

