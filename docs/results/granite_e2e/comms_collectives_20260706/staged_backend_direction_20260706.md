# DLDSC Collectives Staged Backend Direction - 2026-07-06

This note records the current source-of-truth after Jamie identified a separate flash value-correctness issue in zero-stride/broadcast view lowering.

## Correctness Boundary

Flash attention is currently useful as a structural/backend stress test, not as the numeric oracle for DLDSC relayout. The flash baseline can already be wrong before relayout because `unsqueeze`/broadcast zero-stride information is represented in `SpyreTensorLayout.stride_map` but is not carried through `TensorArg` into SDSC emission. SDSC then reconstructs dense strides from `device_size`, so a broadcast dimension can incorrectly change linear offsets.

For the communication-collectives work, value correctness should be judged on cases that do not depend on that zero-stride view lowering. The current clean evidence is the synthetic matmul RHS operand path:

| path | result |
|---|---|
| staged gather/all-gather into LX plus local `ReStickifyOpLx` | value-correct within dtype tolerance for M16/M32/M64 |
| direct KERNEL-neighbor write | value-wrong, row-map corruption |

The archived evidence is under:

```text
docs/results/granite_e2e/dldsc_collectives_artifacts_20260704/matmul_operand_staged_gather_20260704/
```

## Source Of Truth

Use the pod branches as source of truth:

```text
Torch:      AdnanHoque/torch-spyre ah/comms-collectives
Deeptools:  Adnan-Hoque1/deeptools ah/comms-collectives
```

As of this note, the active DEV pod Deeptools branch is:

```text
16e9c4f4e ddc: preserve relayout core maps during fold propagation
```

There are older local clones with additional exploratory staged-scheduler names such as `StagedLayoutConversionInfo`; do not treat those as source-of-truth unless they are explicitly ported onto the pod branch.

## Current Backend Gap

The frontend/DLDSC contract already identifies the useful Granite/attention class as:

```text
kind = matmul_operand_broadcast
communication_pattern = all_gather_replicate
materialization_pattern = all_gather_replicate_with_layout_conversion
requires_layout_conversion = true
staged_destination.scope = matmul_transfer_loop
```

Deeptools also already expands the logical transfers and can emit ring send/recv nodes from `TransferNode::lxNeighborRingTransfers_`.

The missing production lowering is the two-stage backend realization:

1. Ring/LX-neighbor gather or all-gather producer shards into a source-layout LX staging buffer scoped to the matmul transfer loop.
2. Perform local layout conversion from that staging buffer into the matmul KERNEL RHS operand view, using `ReStickifyOpLx` or equivalent existing restickify lowering.
3. Bind the matmul to the converted KERNEL operand tile.

The current direct KERNEL-neighbor shortcut skips step 2. It moves bytes into the final KERNEL operand address space as if producer layout and KERNEL layout were identical. The synthetic row-pattern tests show this is value-wrong.

## Next Patch Shape

The next Deeptools patch should not broaden the direct shortcut. It should add explicit staged lowering state and fail closed until both stages are implemented.

Minimal first implementation slice:

1. Keep `matmul_operand_broadcast` recognition in `dxp/SdscRelayoutInsertion.cpp`.
2. Add an env-gated staged path such as `DEEPTOOLS_MATMUL_OPERAND_BROADCAST_STAGED_RESTICKIFY=1`.
3. In the scheduler, separate the source-layout staging destination from the final KERNEL operand destination. Today `_lx_neighbor` has only one destination labeled DS.
4. Populate ring transfers to the staging allocation, not the KERNEL allocation.
5. Add a local `ReStickifyOpLx`/layout-conversion step from staging LX to KERNEL RHS tile before matmul compute.
6. Add a unit/fixture assertion that direct KERNEL-neighbor is not considered the production lowering when `requires_layout_conversion=true`.

Unsupported cases should fail with a clear diagnostic rather than silently falling back to HBM or the direct KERNEL-neighbor path.

## Why This Scope

Scatter/permutation-style DLDSC relayout is already the PR1 class: ownership moves without duplication or arithmetic. The Granite/attention RHS case is different: it is `all_gather_replicate` plus a local format/layout conversion. It is still copy-style communication, not reduce/all-reduce, but it needs two backend stages to be value-correct.

Reduce and all-reduce remain separate arithmetic collectives and should not be conflated with this staged copy-plus-layout-conversion work.

## Focused Gate Status

Ran on `adnan-spyre-dev-pf` against the pod source-of-truth branches:

```text
Deeptools util:
  ./build-deeptools/util/util_unit_test --gtest_filter="LayoutAllgatherRestickify.*"
  PASS: 25 tests

Deeptools DXP relayout:
  ./build-deeptools/dxp/dxp_unit_test --gtest_filter="DxpTestFixture.CoreWorkDivIncomptLxRelayout*"
  PASS: 2 tests

Torch pure-Python metadata:
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q tests/inductor/test_layout_allgather_restickify_import_light.py
  PASS: 2 tests
```

The extension-backed Torch test `tests/inductor/test_lx_relayout_dldsc.py` could not be collected in this pod because the local runtime library stack failed to load `torch_spyre._C`:

```text
ImportError: /opt/ibm/spyre/spyre-comms/lib/libspyre_comms.so.1: undefined symbol: ...
```

This is an environment/runtime mismatch for the local test pod, not evidence of a DLDSC relayout code failure.
