# Static Unit-Batch BMM Canonicalization Probe

## Scope

This is the follow-up prefill-only probe for the remaining full MLP gap after
the shared-weight unit-BMM layout fix.

Target shape:

```text
mlp [[1, 512, 4096]]
```

The hypothesis was that the full perf-suite MLP still sent gate/up/down through
an awkward static unit-batch BMM layout:

```text
[1, M, K] @ [1, K, N] -> [1, M, N]
```

For this exact static batch-1 case, the batch axis is semantically real for the
user-visible output shape but should not force the poor batched-weight schedule.
The pass now marks it as shared-weight unit-BMM metadata, and codegen recovers
the sendnn-like unit layout even when the unit axis was squeezed out before
layout emission.

This does not apply to true batched/MoE BMM where `B > 1`.

## Implementation

Files changed:

```text
torch_spyre/_inductor/temp_passes.py
torch_spyre/_inductor/spyre_kernel.py
tests/inductor/test_temp_passes.py
tests/inductor/test_coarse_tiling.py
```

The implementation is deliberately narrow:

- `temp_passes.py` marks plain `aten.bmm` nodes when lhs, rhs, and output are
  all rank-3 with static leading batch `1`.
- `spyre_kernel.py` accepts either `[1,M,K] @ [K,N]` or
  `[1,M,K] @ [1,K,N]` as shared-weight unit-BMM candidates.
- If input/output TensorArgs arrive with the unit axis already squeezed away,
  codegen inserts a synthetic `_spyre_bmm_unit` axis before the stick dim, then
  applies the same shared-weight unit-BMM layout preservation path.

Generated code evidence from the benchmark cache:

```text
op_info={'shared_weight_unit_bmm': {'batch_dim': 0}}

input device_size=[1, 64, 512, 64]
input device_coordinates=[_spyre_bmm_unit, floor(c2/64), c0, Mod(c2, 64)]

output device_size=[1, 200, 512, 64]
output device_coordinates=[_spyre_bmm_unit, floor(c1/64), c0, Mod(c1, 64)]
```

## Validation

Pure unit tests:

```text
python -m pytest tests/inductor/test_temp_passes.py \
  tests/inductor/test_coarse_tiling.py \
  tests/inductor/test_work_division.py -q

126 passed
```

Benchmark environment:

```text
LX_PLANNING=0
torch-spyre:      fb65d27 (branch: pr-mlp-prefill-explained)
flex:             2457d3fc (branch: main)
deeptools:        60b12999e4 (branch: master)
spyre-perf-suite: 7450624 (branch: HEAD)
runroot:          /tmp/mlp-prefill-unitcanon-tsp-lx0-20260608_054919
```

The raw report version block shows `6942fd9` because the benchmark was run just
before this commit, while the static-unit-BMM patch was still uncommitted. The
committed code in `fb65d27` is the same patch plus this archived report.

Primary perf-suite command:

```text
python run_benchmark.py --op mlp --shape 1 512 4096 \
  --stacks torch-spyre sendnn --runs 3 \
  --perf-dir perf \
  --report report_tsp_sendnn.txt \
  --kernel_report kernel_report_tsp_sendnn.txt
```

## Results

Full MLP prefill:

```text
Op: mlp  Shape: [[1, 512, 4096]]
Metric                             torch-spyre          sendnn tsp/sendnn
----------------------------------------------------------------------
kernel_ms.mean_ms                      6.169           5.788     1.066
spyre_ms.mean_ms                      11.596          25.870     0.448
memory_transfer_ms.mean_ms             5.427          20.082     0.270
pt_util%                              36.217           0.000        N/A
```

Kernel breakdown:

```text
torch-spyre:
  sdsc_fused_bmm_mul_silu_0_sy9tete0   4.588 ms   32.464% PT
  sdsc_fused_bmm_1_nzjfmlwd            1.581 ms   47.106% PT

sendnn:
  bmm                                  5.788 ms
```

Compared to the prior current-PR full MLP number, this moves torch-spyre from
about `11.702 ms` to `6.169 ms`, a `1.90x` torch-spyre-side speedup. Relative
to sendnn in the paired report, the remaining kernel gap is only `1.066x`.

Standalone wide prefill projection remained near parity:

```text
Op: matmul  Shape: [[1, 512, 4096], [4096, 12800]]
Metric                             torch-spyre          sendnn tsp/sendnn
----------------------------------------------------------------------
kernel_ms.mean_ms                      1.018           0.960     1.060
pt_util%                              73.165           0.000        N/A
```

Decode MLP was checked as a non-regression only:

```text
Op: mlp  Shape: [[4, 1, 4096]]
torch-spyre kernel_ms.mean_ms: 24.765
```

That remains the known true batched/MoE-style BMM issue and is not changed by
this prefill-only canonicalization.

## Conclusion

The probe succeeded. The remaining full prefill MLP gap was not fundamentally a
DeepTools-only kernel issue in this static batch-1 case. Torch-spyre was still
presenting the MLP BMM as an awkward unit-batch BMM layout; preserving and
recovering the shared-weight unit-BMM schedule brings the full prefill MLP to
`1.066x` of sendnn kernel time with `LX_PLANNING=0`.

The follow-up decode/MoE work should stay separate: real `B > 1` batched BMM
still needs a different fix.
