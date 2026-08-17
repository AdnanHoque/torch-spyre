# Clean-source reproduction

## Purpose

This gate removes the staged-file provenance boundary from the original
activation-stationary evidence. The exact `moe-asgemm` branch was checked out
cleanly on the device host and exercised without copying modified Python source
files over another checkout.

The native extension was not rebuilt because this branch changes only Python
compiler code. The clean checkout used the exact compatible base native
extension and records its hash below.

## Source and compiler identity

```text
branch head                 6dd7132b3468ba9c01e2b5992fa5a4fe1d267a13
git status                  clean
git bundle                  6a9342d7ce57b8c73693d24f961275ef08e8cbd5b99c663b2aead13845c0ba86
torch_spyre/_C.so           9db452635696b9090da944a57a0657f9ae12f092023be886f4355edd07726325
L3 scheduler               67f49291120d3dcd4a6796d174125cb045685f874019a4027fe9285dce396f2b
dxp_standalone              2b1814572c3fc402db094030d8dcb82327c7a959fbe7372c98c57537b840ff6b
private bmm.ddl             e453b889440cf8f9aaf779f87ea99e98b9891a4fb7eadba4c8ea0ac74afcb4ea
```

The private BMM template removes the INPUT mapping requirement that is invalid
for a shared-LHS tensor with no expert dimension. It does not change the
matmul arithmetic.

## Compiler tests

The exact compiler environment ran the four affected suites:

```text
530 passed
90 skipped
2 xfailed
6 subtests passed
```

The suites cover coarse tiling, core mapping, scratchpad allocation, and work
division hints.

## Reduced C1 gate

The reduced `E=2,T=64,H=64,F=64,C=1` program passed:

- real DeepTools compilation;
- exact four loop-dependent 128-byte expert address advances;
- one HBM-to-LX activation preheader;
- one fixed LX accumulator and one final HBM drain;
- zero HBM-pool intermediates and zero HBM restickify operations; and
- two distinct runtime alpha payloads through the same callable and bundle.

The emitted bundle hash is:

```text
977d62e74ca5f4a3ff28a89be32e4531ecb858828b77cedbf21e007c05c0c720
```

## Full representative correctness gate

The clean checkout passed `E=128,T=512,H=2816,F=704,C=32` with two runtime
payloads. It retained the accepted structure:

- one flat expert loop;
- one activation HBM-to-LX load;
- all internal activation and accumulator storage in LX;
- direct, affine-advanced expert weight and alpha operands from HBM;
- zero HBM-pool intermediates;
- zero HBM restickify operations; and
- one final HBM output.

The emitted bundle hash is exactly the prior accepted hash:

```text
976e5c8101370a6f482247652b31ec81c5be55c2419011b06746000693fd1727
```

## Two-AIU timing confirmation

The user requested that the clean-source timing stop after two completed AIUs.
Each completed AIU produced 540 records and 900 measured calls under the fixed
protocol. Compilation, copies, reference computation, and artifact validation
were outside the samples.

| AIU | Identity single | Identity block | Permutation block | Hot-eight block |
|---|---:|---:|---:|---:|
| cdx | 46.318 ms | 42.408 ms | 42.453 ms | 42.402 ms |
| clc | 46.611 ms | 42.592 ms | 42.620 ms | 42.623 ms |

Both AIUs emitted the same bundle and passed all structural and correctness
gates. Relative to the retained same-device grouped measurements, the identity
block ratios are `4.033x` on cdx and `4.017x` on clc.

A third run was explicitly stopped during compilation when the measurement
scope was reduced. It produced no accepted result and is excluded. A fourth
run was not launched.

## Artifact pointers

Compact text evidence is retained under:

```text
moe_asgemm/artifacts/clean_reproduction/cdx
moe_asgemm/artifacts/clean_reproduction/clc
```

Each directory contains the result, generated wrapper, bundle, and all twelve
SDSCs. Multi-gigabyte tensor payloads and backend binaries remain outside Git.

## Claim boundary

This establishes clean-source reproducibility of the compiler-generated
activation-stationary dense kernel and confirms its latency on two AIUs. It is
not full-model timing, energy evidence, a native-DDL comparison, or a universal
claim about every grouped schedule.
