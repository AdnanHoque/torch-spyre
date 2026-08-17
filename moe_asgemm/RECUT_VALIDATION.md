# Recut validation on current main

## Scope

The six production commits were rebased onto Torch-Spyre
`3fd7a0f954a84817a417b6b45639b0d5f3499575`, tested independently, and then
validated on one AIU. No device timing was collected.

The tested production head is:

```text
dffd639f Emit expert operand bindings and physical core maps
```

The device probe source was validation snapshot
`dcac0292e19468092f408ef74b7905a39e2e6c6a`. The final validation commit adds
this note, compact copies of the already-generated evidence, and removes one
unused analyzer-local variable. It also applies formatting-only cleanup to the
validation utilities. The compiler and graph-producing probe code used by the
device runs are unchanged.

## Commit-by-commit compiler gates

```text
contract commit       import passed; 4 focused tests passed
hoist commit          import passed; 6 focused tests passed
accumulator commit    import passed; 13 focused tests passed
lifetime commit       import passed; 5 focused tests passed, 2 skipped
ownership commit      import passed; 14 focused tests passed
binding/codegen head  import passed; 537 passed, 90 skipped,
                      2 xfailed, 6 subtests passed
```

The ownership gate found one current-main interaction during the recut: the
marked accumulator's own synthetic allocation-only fallback was being treated
as an unrelated external kernel. The final rule ignores only that synthetic
self-use. Any other external kernel in the live interval still rejects LX
placement.

## Reduced C1 correctness

Shape:

```text
E=2, T=64, H=64, F=64, C=1, FP16
```

The independent acceptance checker passed:

- one wrapper call, one bundle, and one static expert loop;
- one X HBM-to-LX copy before the loop;
- one fixed LX accumulator and one final HBM drain;
- zero HBM-pool intermediates and zero restickify operations;
- exactly four loop-dependent expert operands;
- one affine map with a 128-byte step for Wg, Wu, Wd, and alpha;
- X, accumulator initializer, and final output addresses fixed;
- two distinct nonbinary `[E,T,1]` alpha payloads through the same callable;
  and
- no second-call bundle generation.

Measured correctness against fresh FP32 references:

| Payload | Max abs | Relative L2 | Cosine |
|---|---:|---:|---:|
| alpha A | 0.0004457533 | 0.0041910365 | 0.9999964833 |
| alpha B | 0.0003192276 | 0.0041069494 | 0.9999965429 |
| B minus A | 0.0002084821 | 0.0044958913 | 0.9999948144 |

Key hashes:

```text
generated_module.py  3fed6b2ef32ef6ce18664935255c0b70c49f3948ca276848c5e8748ffd31c483
bundle.mlir          977d62e74ca5f4a3ff28a89be32e4531ecb858828b77cedbf21e007c05c0c720
strict_accept.json   0807dd437e3a678cfb3350440d26bdac56324c9cb00b3058f0ea0b0a6db2b55b
```

The first correctness harness used `0.01` scaling for all three matmul weight
stages, which underflowed the reduced FP16 output to zero while the FP32 output
norm was about `1.3e-6`. That run is excluded as a validation-range error. The
accepted run uses the previously proven C1 range: X scaled by `0.25` and each
weight bank by `0.1`. Two later setup attempts failed before backend
compilation because the artifact-capture shim used the wrong module import and
then an obsolete keyword. They are also excluded and consumed no kernel run.

## Representative C32 correctness

Shape:

```text
E=128, T=512, H=2816, F=704, C=32, FP16
```

The representative gate passed:

- one source bundle, one wrapper call, and one flat expert loop;
- one X HBM-to-LX preheader shared by gate and up on all 32 cores;
- direct affine-advanced Wg, Wu, Wd, and alpha operands from HBM;
- all internal compute and the fixed accumulator in LX;
- zero HBM-pool intermediates and zero restickify operations;
- runtime top-8 weights applied after down;
- one final HBM output; and
- two distinct balanced identity/permutation payloads through the same
  callable.

Measured correctness against fresh FP32 references:

| Payload | Max abs | Relative L2 | Cosine |
|---|---:|---:|---:|
| identity | 0.0085942149 | 0.0088168671 | 1.0000096560 |
| permutation | 0.0077630281 | 0.0089123510 | 1.0000089407 |
| permutation minus identity | 0.0105085373 | 0.0088609057 | 1.0000239611 |

Key hashes:

```text
generated_module.py  b724049b58731d4c49de316c9d12415ef16f77bafb3fd3e28fcfce37b0c7e247
bundle.mlir          976e5c8101370a6f482247652b31ec81c5be55c2419011b06746000693fd1727
compile_result.json  4c8ea0e82690b159bfefc360938685ce1a27cb0e3ae157969c36282cd514a57f
```

## Runtime provenance

```text
pod                     adnan-cdx-spyre-dev-pf
PCI                     0000:ac:00.0
torch_spyre/_C.so       9db452635696b9090da944a57a0657f9ae12f092023be886f4355edd07726325
dxp_standalone          2b1814572c3fc402db094030d8dcb82327c7a959fbe7372c98c57537b840ff6b
L3 scheduler            67f49291120d3dcd4a6796d174125cb045685f874019a4027fe9285dce396f2b
private bmm.ddl         e453b889440cf8f9aaf779f87ea99e98b9891a4fb7eadba4c8ea0ac74afcb4ea
```

Compact generated sources, bundles, SDSCs, and JSON reports are under:

```text
moe_asgemm/artifacts/recut_validation/c1
moe_asgemm/artifacts/recut_validation/c32
```

The multi-gigabyte full-shape tensor payload and backend binaries remain on
the device host. They are not Git artifacts.

## Claim boundary

This proves the recut compiler series still generates and executes the dense
activation-stationary all-expert kernel correctly on current main. It does not
measure latency, energy, model integration, native custom-DDL performance,
indexed expert binding, grouped execution, or superiority over grouped GEMM.
