# Gather + ReStickify Branch Split - 2026-07-06

This note records the clean branch split for the staged gather plus local
`ReStickifyOpLx` path. It is meant as a handoff point for the next agent or
human picking up the Granite/flash communication-collectives work.

## Pod Ownership

Current pod allocation:

| Pod | Owner for this slice | Notes |
|---|---|---|
| `adnan-clc-spyre-dev-pf` | Claude | Do not use while Claude is running there. |
| `adnan-spyre-dev-pf` | Codex | DEV lane for Deeptools build/tests and AIU probes. |
| `adnan-cdx-spyre-dev-pf` | Codex | CDX lane for source inspection, replay, and artifact checks. |

## Clean Branches

Two clean implementation branches were prepared from the larger
`ah/comms-collectives` artifact branch:

| Repository | Branch | Head | Base | Contents |
|---|---|---|---|---|
| `AdnanHoque/torch-spyre` | `gather-restickify` | `b84528d7e32ad0aea5f31d7de107344b35617695` | `7e45168f1d56ca1cec4889a3e19b14719dcdd23f` (`origin/main`) | Torch DLDSC relayout metadata, topology classification, matmul-operand/layout-allgather contracts, and focused tests. |
| `Adnan-Hoque1/deeptools` | `gather-restickify` | `393403f8205a089045e364a4e98ab7291e584618` | newer Deeptools fork base stack ending at parent `949cfeea885e05cb12dd37ea07d480d82f1ee27c` | Deeptools support for `ReStickifyOpLx`, layout-allgather/gather-restickify metadata handling, DXP insertion, DCG/DCC plumbing, and focused tests. |

Both feature commits have DCO signoff with `adnan.hoque1@ibm.com`. DEV did
not have a usable GPG secret key, so the branch-packaging commits are not
PGP-signed.

## What This Branch Split Is For

The target path is the attention/granite class where a matmul operand needs
pieces from other cores and also needs a local layout conversion before the
consumer matmul can use it:

```text
producer LX shards
  -> STCDPOpLx gather/all-gather into temporary LX
  -> local ReStickifyOpLx into the consumer matmul operand layout
  -> consumer batchmatmul
```

This is the path associated with the useful validated signals:

- DXP replay succeeded with backend plans.
- `LayoutAllgatherRestickify.*`: 27/27 passing.
- `CoreWorkDivIncomptLxRelayout*`: 2/2 passing.
- DEV flash compile probe reached:
  - `ReStickifyOpHBM: 0`
  - `ReStickifyOpLx: 64`
  - `matmul_operand_broadcast: 32`
  - backend plans all classified as `gather_then_restickify`.

## What Was Deliberately Excluded

The later standalone data-op carrier experiments are not included in the clean
branches.

Those experiments taught us something important:

1. Standalone data-op-only `STCDPOpLx` SDSCs initially failed DXP replay
   because inserted relayout rows did not extend the memory-tracker timeline.
2. Adding `memTrackers->insertPsBefore(ps)` fixed DXP replay for standalone
   gather-only and gather-plus-restickify cases.
3. AIU runtime still hung at final synchronize, even for gather-only. This
   showed the runtime issue was not caused by the local `ReStickifyOpLx` row.
4. Therefore standalone data-op-only rows are useful diagnostics, but not a
   production carrier for the attention path without deeper runtime/backend
   support for generic data-op program boundaries and completion.

The clean branch direction stays with the DLDSC contract and the staged
gather-restickify implementation, not the standalone data-op runtime probe.

## Current Communication-Class Read

| Class | Current status | Why it matters |
|---|---|---|
| Scatter/permutation | Covered by PR1-style DLDSC coordinate mismatch when the destination view can be resident. | This removes simple one-to-one HBM relayouts. |
| Broadcast/multicast | Partly supported in backend movement substrate, but not generally dispatched from compact DLDSC classification metadata. | Needed for one source shard feeding several consumers. |
| Gather | Needs grouped LX-to-LX fan-in and local assembly/layout conversion. | Needed when one consumer operand is assembled from several producer cores. |
| All-gather/replicate | The attention RHS case is the key example. Dense full-resident materialization is too large; staged movement is required. | Needed for flash/granite attention operand spills. |
| Reduce/all-reduce | Not a pure copy problem. Requires arithmetic accumulation semantics, dtype/precision choices, ordering, and synchronization. | Future work for split-K and reduction-shaped relayouts. |

## Validation Already Rechecked In This Slice

On DEV, using the existing Deeptools build:

```text
./util/util_unit_test --gtest_filter="LayoutAllgatherRestickify.*"
27/27 passed

./dxp/dxp_unit_test --gtest_filter="DxpTestFixture.CoreWorkDivIncomptLxRelayout*"
2/2 passed
```

Torch-side focused tests were not fully rerun in this slice because the
available DEV/CDX Python environments did not load `torch_spyre._C` cleanly:

- CDX: `ModuleNotFoundError: No module named 'torch_spyre._C'`
- DEV: undefined symbol from `/opt/ibm/spyre/spyre-comms/lib/libspyre_comms.so.1`

The Torch branch worker did run:

```text
git diff --check origin/main..origin/gather-restickify
tests/inductor/test_layout_allgather_restickify_import_light.py: 2 passed
python compile check over touched inductor modules: passed
```

## Next Step

Use the two clean `gather-restickify` branches for the next production-shaped
prototype:

1. Rebuild Deeptools from `Adnan-Hoque1/deeptools:gather-restickify`.
2. Use Torch from `AdnanHoque/torch-spyre:gather-restickify`.
3. Reproduce the DXP replay and flash compile probe.
