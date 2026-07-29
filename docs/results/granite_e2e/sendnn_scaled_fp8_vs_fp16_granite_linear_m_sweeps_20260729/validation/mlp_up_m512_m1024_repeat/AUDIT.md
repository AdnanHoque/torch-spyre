# Granite MLP up FP8 M=512 to M=1024 anomaly audit

## Scope

Shape: `[M,4096] @ [4096,12800]`, with fixed per-row activation scales
and per-output-channel weight scales. The timed FP8 kernel includes Qfp8,
FP8 BatchMatMul, and both scale-recovery stages.

This audit used the current-chip SenDNN production stack on
`adnan-clc-spyre-dev-pf`. Architecture override variables, including
`SENARCH`, `SENTARGET`, and `DATA_PREC`, were unset. Nothing labeled
1p5/SEN1P5 was used.

Baseline run:

`/home/adnan/codex-isolated/fp8_sendnn_linear_sweeps_20260729/runs/granite_mlp_up_m_sweep_20260729_040527`

Successful independent repeat:

`/home/adnan/codex-isolated/fp8_sendnn_linear_sweeps_20260729/runs/granite_mlp_up_m512_m1024_repeat2_20260729_041803`

The repeat used benchmark SHA256
`3536cfcb912779e2f04013df04d534e0c11b2d38a43152ca235e21b713bbd046`,
five warmups, 20 Kineto repetitions, fresh compile/export directories, and
one isolated process per case.

## Timing reproduction

Effective TFLOP/s is `2*M*K*N / kernel_time`.

| M | Mode | Baseline mean us | Repeat mean us | Repeat delta | Repeat TFLOP/s | Repeat event stdev us |
|---:|---|---:|---:|---:|---:|---:|
| 512 | FP16 | 1091.6944 | 1091.6414 | -0.005% | 49.180 | 2.249 |
| 512 | scaled FP8 | 1821.6872 | 1823.0628 | +0.076% | 29.449 | 5.202 |
| 1024 | FP16 | 1849.6730 | 1849.0149 | -0.036% | 58.071 | 3.181 |
| 1024 | scaled FP8 | 1189.2080 | 1188.0514 | -0.097% | 90.378 | 1.222 |

The FP8/FP16 speed ratio reproduced from `0.5993x` to `0.5988x` at M=512
and from `1.5554x` to `1.5563x` at M=1024. All four repeat cases passed
CPU-reference correctness and compile/load/prepare/execute checks, with
exactly 20 positive Kineto kernel events.

## Emitted-plan comparison

The FP8 BatchMatMul grid does not change:

| Stage | M=512 | M=1024 |
|---|---|---|
| Qfp8 | 32 cores, 2 corelets/core, `OUT:1, MB:32` | same grid |
| Qfp8 primary corelet M split | `8+8` | `16+16` |
| FP8 BatchMatMul | 32 cores, 2 corelets/core, `IN:1, OUT:4, MB:8` | same grid |
| FP8 BatchMatMul primary corelet M split | `32+32` | `64+64` |
| Activation-scale recovery | 1 core, 2 corelets, `OUT:1, MB:1` | 32 cores, 2 corelets/core, `OUT:4, MB:8` |
| Weight-scale recovery | 1 core, 2 corelets, `OUT:1, MB:1` | 32 cores, 2 corelets/core, `OUT:4, MB:8` |
| LX optimization | one inserted relayout; not in-place | no relayout; in-place |

The final M=512 chain contains:

1. two folded scale preloads
2. Qfp8
3. `BatchMatMulV2...-LxRelayout`
4. FP8 BatchMatMul
5. activation-scale recovery
6. weight-scale recovery

At M=1024 the chain is identical except that the LX relayout is absent.

Additional deterministic artifact changes:

| Artifact property | M=512 | M=1024 |
|---|---:|---:|
| `RelayoutIns` | 1 | 0 |
| `LxOptInPlace` | 0 | 1 |
| Qfp8 prepared-program bytes | 38,016 | 27,392 |
| runtime stack segment bytes | 204,800 | 32,768 |
| modeled FP8 BMM ideal cycles | 409,600 | 819,200 |

The compiler model assigns zero ideal cycles to Qfp8, relayout, and both
scale-recovery stages. It therefore does not represent the overhead that
dominates this threshold behavior.

The baseline and repeat emitted Qfp8 `init.txt`, `prog_size.json`,
`segment_size.json`, and scale-preload program files are byte-identical for
each M. The work-division facts were captured at the DeepTools DCG boundary
with a read-only preload audit; they are structural compiler evidence, not
hardware counters.

Audit logs:

- `.../granite_mlp_up_m512_m1024_repeat2_20260729_041803/artifact_audit/live_m512/run.log`
- `.../granite_mlp_up_m512_m1024_repeat2_20260729_041803/artifact_audit/live_m1024/run.log`

## Conclusion

The discontinuity is real and deterministic. It is not a change in FP8
BatchMatMul core count, corelet count, or core work grid. At M=1024,
DeepTools changes the surrounding fused schedule: both scale-recovery stages
expand from one core to 32 cores, and the Qfp8-to-matmul handoff becomes
in-place so an LX relayout disappears. The fused Kineto event cannot assign
an exact time contribution to either change, so this audit establishes the
joint structural cause rather than a percentage attribution.

## Operational note

An earlier focused attempt at
`granite_mlp_up_m512_m1024_repeat_20260729_041430` completed M=512 FP8, then
stalled during M=512 FP16 profiler teardown. That exact process was
terminated after the log stopped advancing; its data is excluded. The
successful repeat above used a clean process per case and reproduced all
baseline means within 0.1%.
