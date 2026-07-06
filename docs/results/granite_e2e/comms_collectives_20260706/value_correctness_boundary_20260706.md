# Value Correctness Boundary - 2026-07-06

This note separates two correctness questions that were starting to get mixed together:

1. Is the current flash attention baseline value-correct?
2. Does the LX communication work introduce an additional value-correctness problem?

The answer today is that flash attention value correctness is not a clean signal for this lane because there is an independent baseline lowering issue around broadcast / zero-stride views. For the communication work, the useful signal is synthetic, non-broadcast row-pattern coverage plus structural SDSC evidence from Granite and flash.

## Jamie's Flash Baseline Read

Jamie's current diagnosis is that the flash baseline can be value-wrong before our relayout work is involved.

The problematic pattern is:

```text
running_max:               [B, H, Lq]
running_max.unsqueeze(-1): [B, H, Lq, 1]
broadcast consumer view:   [B, H, Lq, D or Lk]
```

The final dimension should be a broadcast dimension. It should not change the source address. Every value along that broadcast dimension should read the same original `running_max[b,h,lq]`.

The suspected lowering gap is:

- `SpyreTensorLayout` can represent a broadcast dimension with a zero stride.
- `TensorArg` only carries `device_size` and `device_coordinates`.
- `create_tensor_arg()` drops the layout `stride_map`.
- SuperDSC generation later recomputes dense strides from `device_size`.

So a logical broadcast view with a device shape like:

```text
[1, 512, 4, 2, 64]
```

can be linearized as dense. The `2` in the broadcasted outer-stick dimension incorrectly participates in address stride calculation.

Example:

```text
Producer writes [1, 512, 4, 64]
offset = c1 * 4 * 64 + c0 * 64
       = c1 * 256 + c0 * 64

Consumer reads [1, 512, 4, 2, 64]
offset = c1 * 4 * 2 * 64 + c0 * 2 * 64
       = c1 * 512 + c0 * 128
```

Even with `device_coordinates=[0, c1, c0, 0, 0]`, the dense stride calculation reads the wrong address.

Likely fixes belong in the separate flash/value-correctness lane:

- carry `stride_map` / zero-stride metadata through `TensorArg` into SuperDSC generation, or
- canonicalize broadcast device dimensions to size `1` before creating `TensorArg`s.

## Boundary For This Lane

For LX communication, do not use flash attention end-to-end value mismatch as the blocker until the zero-stride issue is fixed.

The acceptance question for this lane is narrower:

```text
Given a value-correct producer/consumer contract without zero-stride ambiguity,
does the on-chip communication path move the right bytes to the right cores and
layouts?
```

That means the primary value-correctness tests should be:

- synthetic row-pattern tests with no broadcast/zero-stride ambiguity;
- small non-broadcast matmul operand gather/restickify tests;
- DXP/unit tests that fail closed when a layout-converting path would otherwise take the unsafe direct carrier.

Flash and Granite are still useful, but mainly for:

- confirming the compiler classifies the right edge;
- confirming HBM `ReStickifyOpHBM` rows are converted to on-chip `ReStickifyOpLx` or relayout metadata;
- confirming backend plans are emitted;
- measuring performance after the communication path is value-correct on clean synthetic cases.

## Current Communication Correctness Evidence

The useful artifact root is:

```text
docs/results/granite_e2e/dldsc_collectives_artifacts_20260704/matmul_operand_staged_gather_20260704
```

The direct KERNEL-neighbor path is not value-correct:

| Variant | Shape | Result |
| --- | --- | --- |
| Direct KERNEL-neighbor, max run 8 | M=16,K=64,N=256 | FAIL, 2048/4096 mismatches |
| Direct KERNEL-neighbor, out-major | M=32,K=64,N=256 | FAIL, 6144/8192 mismatches |

The staged gather plus local restickify path is value-correct:

| Variant | Shape | Result |
| --- | --- | --- |
| Staged gather + `ReStickifyOpLx` | M=16,K=64,N=256 | PASS, `ALLCLOSE True`, 0/4096 mismatches |
| Staged gather + `ReStickifyOpLx` | M=32,K=64,N=256 | PASS, `ALLCLOSE True`, 0/8192 mismatches |
| Staged gather + `ReStickifyOpLx` | M=64,K=64,N=256 | PASS, `ALLCLOSE True`, 0/16384 mismatches |

Interpretation:

- The communication class is not a plain scatter.
- It is an all-gather/replicate into a matmul operand plus a layout conversion.
- Correct lowering needs two conceptual steps:
  1. gather producer LX shards into a source-layout LX staging view;
  2. run local `ReStickifyOpLx` or equivalent conversion into the consumer operand layout before PT consumes it.
- The direct KERNEL-neighbor path can be useful as a diagnostic for schedule/ring placement, but it should not be treated as production-correct.

## Current Branch State

Torch artifact branch:

```text
AdnanHoque/torch-spyre ah/comms-collectives
latest checkpoint before this note: 891a7ef
```

Deeptools artifact branch:

```text
Adnan-Hoque1/deeptools ah/comms-collectives
latest checkpoint before this note: f23ab8b85
```

The Deeptools checkpoint intentionally fails closed for direct layout-converting KERNEL-neighbor movement unless a diagnostic bypass is set:

```text
DEEPTOOLS_ALLOW_DIRECT_KERNEL_NEIGHBOR_LAYOUT_BYPASS
```

That guard is correct: it prevents the known value-unsafe path from silently becoming the production lowering.

## Validated Gates At This Checkpoint

On pod `adnan-spyre-dev-pf`, the current Deeptools branch passes:

```text
./build-deeptools/util/util_unit_test --gtest_filter="LayoutAllgatherRestickify.*"
25 tests passed

./build-deeptools/dxp/dxp_unit_test --gtest_filter="DxpTestFixture.CoreWorkDivIncomptLxRelayout*"
2 tests passed
```

These gates validate the metadata contract and existing scatter/cardinality relayout path. They do not yet prove production staged matmul-operand lowering.

## Next Implementation Step

Do not chase flash value correctness here.

The next backend task is to turn the already-proven staged shape into production Deeptools lowering:

```text
producer LX/source-layout shards
  -> ring/local gather into source-layout LX staging
  -> local ReStickifyOpLx/layout conversion
  -> consumer matmul operand layout
```

Recommended order:

1. Add a DXP/DCG unit fixture for the staged matmul operand contract.
2. Allocate source-layout LX staging explicitly.
3. Route ring transfers into that staging allocation.
4. Run local `ReStickifyOpLx` or equivalent conversion before the consumer matmul.
5. Validate on synthetic row-pattern cases first.
6. Re-run flash/Granite only as structural and performance checks until the broadcast zero-stride lowering bug is fixed.
