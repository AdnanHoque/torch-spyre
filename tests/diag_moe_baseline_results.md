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
| empty step (zeros_like + sync only) | 0.29 | — | 0.02× |
| single mm (no SwiGLU pointwise) | 2.89 | — | 0.19× |
| naive K=1 | 15.44 | 15.44 | 1.00× |
| naive K=2 | 30.74 | 15.37 | 1.99× |
| naive K=4 | 60.97 | 15.24 | 3.95× |
| naive K=8 | 122.01 | 15.25 | 7.90× |
| dense fallback (E experts always run) | 10.02 | — | 0.65× |

## H=1024, I=4096, E=8

| variant | median ms | per-active-expert ms | vs single-expert |
|---|---:|---:|---:|
| empty step (zeros_like + sync only) | 0.29 | — | 0.02× |
| single mm (no SwiGLU pointwise) | 2.94 | — | 0.19× |
| naive K=1 | 15.74 | 15.74 | 1.00× |
| naive K=2 | 30.85 | 15.42 | 1.96× |
| naive K=4 | 61.82 | 15.46 | 3.93× |
| naive K=8 | 123.31 | 15.41 | 7.84× |
| dense fallback (E experts always run) | 10.89 | — | 0.69× |

## H=4096, I=14336, E=8

| variant | median ms | per-active-expert ms | vs single-expert |
|---|---:|---:|---:|
| empty step (zeros_like + sync only) | 0.27 | — | 0.01× |
| single mm (no SwiGLU pointwise) | 3.87 | — | 0.21× |
| naive K=1 | 18.53 | 18.53 | 1.00× |
| naive K=2 | 36.94 | 18.47 | 1.99× |
| naive K=4 | 73.76 | 18.44 | 3.98× |
| naive K=8 | 147.61 | 18.45 | 7.97× |
| dense fallback (E experts always run) | 40.23 | — | 2.17× |

## Token permute cost (H=4096)

Decode-relevant batch sizes. Permuted-token grouped-GEMM needs a gather pre-pass + a scatter post-pass; if either is >5ms it eats into the win, and if either is unsupported on the Spyre op set (`n/a` below) the format is fully blocked until the op is registered.

| op | median ms | note |
|---|---:|---|
| gather M=1, H=4096 | n/a | NotImplementedError: Could not run 'aten::index.Tensor_out' with arguments from the 'spyre' backend.  |
| scatter M=1, H=4096 | n/a | NotImplementedError: Could not run 'aten::_index_put_impl_' with arguments from the 'spyre' backend.  |
| gather M=4, H=4096 | n/a | NotImplementedError: Could not run 'aten::index.Tensor_out' with arguments from the 'spyre' backend.  |
| scatter M=4, H=4096 | n/a | NotImplementedError: Could not run 'aten::_index_put_impl_' with arguments from the 'spyre' backend.  |
| gather M=8, H=4096 | n/a | NotImplementedError: Could not run 'aten::index.Tensor_out' with arguments from the 'spyre' backend.  |
| scatter M=8, H=4096 | n/a | NotImplementedError: Could not run 'aten::_index_put_impl_' with arguments from the 'spyre' backend.  |
| gather M=16, H=4096 | n/a | NotImplementedError: Could not run 'aten::index.Tensor_out' with arguments from the 'spyre' backend.  |
| scatter M=16, H=4096 | n/a | NotImplementedError: Could not run 'aten::_index_put_impl_' with arguments from the 'spyre' backend.  |
| gather M=64, H=4096 | n/a | NotImplementedError: Could not run 'aten::index.Tensor_out' with arguments from the 'spyre' backend.  |
| scatter M=64, H=4096 | n/a | NotImplementedError: Could not run 'aten::_index_put_impl_' with arguments from the 'spyre' backend.  |

