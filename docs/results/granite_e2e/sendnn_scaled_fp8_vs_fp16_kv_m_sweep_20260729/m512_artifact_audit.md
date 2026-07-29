# DD2 M512 K4096 N1024 FP16/FP8 Artifact Audit

## Scope and evidence boundary

This is a structural compiler-artifact audit of the already correctness-checked
direct-SenDNN pair:

```text
M=512, K=4096, N=1024
activation [1,1,512,4096]
weight     [1,1,4096,1024]
output     [1,1,512,1024]
```

The installed current-chip stack is:

```text
DeepTools +1401 (ee2f97a)
Flex      +388  (81385a4)
senlib DD2 +194 (951e4c4)
```

Layout, work-division, and corelet facts below come from the exact PBDIs
recompiled for the `sentient` backend with a read-only audit at the DCG
boundary. Instruction mnemonics come from the same exact PBDIs recompiled by
the same installed stack for the `senulator` backend so that textual programs
are exported. Compiler ideal cycles and static instruction-token counts are
not device timings.

## Exact lowering

FP16 has one executable compute node:

```text
batchmatmul
```

Scaled FP8 has this final execution order:

```text
activation-scale preload
weight-scale preload
qfp8mb
LX relayout
batchmatmulfp8mb
activation-scale transfer and reshape
LX relayout
BnPrecZeroShft1
weight-scale transfer and reshape
BnPrecZeroShft2
```

Both recovery nodes use the emitted `batchnormfwd` compute operation.

## Work division and corelets

| Stage | Cores | Corelets/core | Core work division | Corelet split |
|---|---:|---:|---|---|
| FP16 batchmatmul | 32 | 2 | `IN:1, OUT:4, MB:8` | `MB:32+32` |
| FP8 qfp8mb | 32 | 2 | `OUT:1, MB:32` | `MB:8+8` at the primary stage |
| FP8 batchmatmulfp8mb | 32 | 2 | `IN:1, OUT:4, MB:8` | `MB:32+32` |
| FP8 BnPrecZeroShft1 | 1 | 2 | `OUT:1, MB:1` | output dimension split |
| FP8 BnPrecZeroShft2 | 1 | 2 | `OUT:1, MB:1` | output dimension split |

For both matmuls, the 32-core grid is therefore:

```text
M:8, N:4, K:1
per-core logical tile: M=64, N=256, K=4096
per-corelet logical tile: M=32, N=256, K=4096
```

FP16 and FP8 do not differ in matmul core count, work division, or corelet
partition for this shape. FP8 quantization uses a different ownership grid
(`MB:32`) from the FP8 matmul (`MB:8, OUT:4`), and the compiler inserts an LX
relayout between them.

## Layout and stick contracts

| Tensor | FP16 contract | FP8 scaled-pipeline contract |
|---|---|---|
| matmul activation | layout `MB,IN`; word 2 bytes; stick `IN:64` | layout `MB,IN`; word 1 byte; compound stick `IN:8, MB:2, IN:8` |
| weight | layout `IN,OUT`; word 2 bytes; stick `OUT:64` | layout `IN,OUT`; word 1 byte; stick `IN:2, OUT:64` |
| output | layout `MB,OUT`; word 2 bytes; stick `OUT:64` | layout `MB,OUT`; word 2 bytes; stick `OUT:64` |

The FP8 activation contract is the minibatch-packed variant selected by
`qfp8mb`/`batchmatmulfp8mb`: one compound stick contains two M rows and 64 K
values per row, or 128 FP8 values total. Its outer physical coordinates are
`M/2=256` by `K/64=64`.

The exported host/device physical shapes agree:

```text
FP16 activation: [1,1,64,512,64]
FP16 weight:     [1,1,16,4096,64]
FP16 output:     [1,1,16,512,64]

FP8 static weight: [1,1,16,2048,64,2]
FP8 output:        [1,1,16,512,64]
```

The FP8 pipeline receives the primary activation as FP16 and converts it in
`qfp8mb`; it is not a host-prequantized activation path.

## PT instruction evidence

Both matmuls export 512 PT-row programs:

```text
32 cores * 2 corelets * 8 PT rows = 512
```

Every FP16 PT-row program contains `PTOP_FMA` and no `PTOP_FMA8`. Every FP8
matmul PT-row program contains `PTOP_FMA8` and no `PTOP_FMA`.

Static emitted-token counts are:

| Path | `PTOP_FMA` | `PTOP_FMA8` |
|---|---:|---:|
| FP16 batchmatmul | 18,624 | 0 |
| FP8 batchmatmulfp8mb | 0 | 14,208 |

These are static program-text token counts, not dynamic issued-instruction
counts. The compiler perf model reports 65,536 ideal cycles for FP16 and
32,768 for the FP8 matmul node; that is an arithmetic proxy, not a measured
latency.

## Preserved evidence

Run root:

```text
/home/adnan-cdx/codex-isolated/fp8_sendnn_study_20260728_210752/benchmarks/direct_pair_m512_k4096_n1024_20260729_012100
```

Key files:

```text
artifact_recompile/fp16_layout_audit_op/run.log
artifact_recompile/fp8_layout_audit_op/run.log
artifact_recompile/fp16_senulator/execute/fp16_bmm/senprog.json
artifact_recompile/fp8_senulator/execute/fp8_scaled_bmm-Qfp8/senprog.json
artifact_recompile/fp16_senulator/run.log
artifact_recompile/fp8_senulator/run.log
```

Audit source and library:

```text
/home/adnan-cdx/codex-isolated/fp8_sendnn_study_20260728_210752/deeptools_corelet_audit_preload.cpp
/home/adnan-cdx/codex-isolated/fp8_sendnn_study_20260728_210752/libdeeptools_corelet_audit.so
```
