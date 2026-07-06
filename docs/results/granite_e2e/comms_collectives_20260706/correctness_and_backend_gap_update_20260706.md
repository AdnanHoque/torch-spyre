# DLDSC Collectives Correctness and Backend Gap Update - 2026-07-06

This note updates the communication-collectives status after separating flash attention baseline correctness from DLDSC relayout correctness.

## Correctness Scope

Jamie's current read is that the standalone flash attention baseline still has an unrelated value-correctness issue: an `unsqueeze`/broadcast view can lose zero-stride layout information before SDSC generation. `TensorArg` carries `device_size` and `device_coordinates`, but not the original `stride_map`, so SDSC generation can recompute dense strides that make a broadcast dimension change the linear offset.

That is not the communication-collectives bug we are chasing here.

For this track, the clean question is:

1. Does on-chip DLDSC relayout preserve values on cases that do not rely on the known zero-stride/broadcast-view lowering?
2. Does enabling relayout introduce a new mismatch relative to the non-relayout baseline on those cases?
3. Does flash structurally expose the right communication classes and backend lowering gaps?

Until the zero-stride issue is fixed separately, flash value comparison is not a clean oracle for relayout correctness.

## Zero-Stride-Independent Value Evidence

The strongest current value-correct evidence is the staged matmul-operand path:

```text
communication: matmul_operand_broadcast / all_gather_replicate
implementation: gather/all-gather source-layout chunks into LX, then local ReStickifyOpLx into the matmul KERNEL operand view
```

Existing contiguous row-pattern synthetic runs avoid `unsqueeze`, `expand`, and zero-stride input views:

| case | result |
|---|---|
| M16 | `ALLCLOSE True`, `MISMATCH 0 / 4096` |
| M32 | `ALLCLOSE True`, `MISMATCH 0 / 8192` |
| M64 | `ALLCLOSE True`, `MISMATCH 0 / 16384` |

That means the staged gather/all-gather plus local restickify decomposition is not showing an independent communication-value bug in the evidence we have.

Fresh CDX reruns were attempted under:

```text
/home/adnan-cdx/codex-isolated/comms_collectives_gather_reduce_probe_20260706_005444/runs/zero_stride_independent_correctness_20260706_014405
```

Those fresh runs failed due runtime/device state (`ComputeHardwareError` followed by `StreamInErrorState`), not a numerical mismatch. The relayout-on run still emitted one plan:

```text
kind=matmul_operand_broadcast
communication_pattern=all_gather_replicate
physical_lowering_status=lowered_gather_then_restickify
```

## Unsafe Shortcut Evidence

The direct loop-scoped KERNEL-neighbor shortcut is not the value-correct path:

| case | result |
|---|---|
| direct KERNEL-neighbor M16 | `ALLCLOSE False`, `MISMATCH 8181 / 8192` |

The conclusion is precise: the staged design is the value-correct decomposition; direct ring writes into the PT KERNEL view are not currently safe.

## Flash Structural Backend Gap

The flash relayout-on bundle structurally does the right high-level thing:

- `ReStickifyOpHBM` rows disappear.
- `ReStickifyOpLx` rows appear.
- Deeptools emits `matmul_operand_broadcast` / `all_gather_replicate` backend plans.

The current DDC blocker is after insertion, during coordinate propagation for the generated transfer. A minimized failing edge has:

```text
producer: sdsc_2 ReStickifyOpLx
consumer: sdsc_3 batchmatmul Tensor1
failure: l3_lx_kernel transfer cannot solve coordinates
```

The important observation from the backend investigation:

- The referenced LX allocation carries a custom `coreIdToWkSlice_`.
- The generated transfer coordinate does not carry that map.
- DDC falls back to the SDSC/global consumer map.
- For replicated producer chunks, that fallback can ask the solver for a coordinate/core pairing that cannot exist.

The smallest principled backend fix direction is to propagate an explicit `coreIdToWkSlice_` from the effective reference coordinate onto the transfer coordinate when the transfer is derived from an explicitly mapped LX allocation. This should be implemented carefully in Deeptools, not as the current local diagnostic bypass.

### Map Propagation Replay

A local Deeptools experiment added that map propagation rule in `ddc/ddc_fold.cpp` and rebuilt `dxp_standalone` successfully:

```text
cmake --build build-deeptools --target dxp_standalone -j8
```

Replaying the minimized flash bundle moved the failure:

```text
before: coordCoreMap=0, allocation=allocate_lds1_lx
after:  coordCoreMap=32, allocation=allocate_lds1_ptxrf
```

The replay log after the map propagation experiment:

```text
/home/adnan/codex-isolated/flash_attention_comms_backend2162_20260706_005751/ddc_replay_after_map_propagation_20260706_015429.log
```

The new failure still occurs in `lexiAffineSolve`, but now the transfer coordinate carries the explicit core map. That means the first bug was real and fixed by the experiment, and the remaining blocker is the deeper KERNEL operand layout conversion: direct LX/LXLU-to-PTXRF movement is still not enough to materialize the PT KERNEL view.

This strengthens the design conclusion:

- Preserve explicit `coreIdToWkSlice_` through DDC propagation; this is a legitimate backend fix.
- Do not rely on direct ring writes into the KERNEL operand as the production path.
- Implement a loop/tile-scoped staged path: source-layout gather/all-gather into LX, then local `ReStickifyOpLx` or equivalent KERNEL layout conversion, then matmul consumption.

## Current Architecture Takeaway

The useful distinction is:

1. **Pure coordinate movement**: scatter, broadcast, multicast, gather, all-gather are expressible in DLDSC metadata.
2. **Matmul operand broadcast/all-gather**: movement is only half the story; the gathered source-layout bytes must become the PT KERNEL operand layout.
3. **Reduce/all-reduce**: these are arithmetic collectives, not copy-only relayouts.

For the current flash path, the safe target remains:

```text
loop/tile-scoped source-layout gather/all-gather
then local KERNEL layout conversion
then matmul reads the converted tile
```

That matches the value-correct staged synthetic evidence while avoiding full-resident materialization.

## Artifacts

Synthetic correctness and fresh rerun logs:

```text
/home/adnan-cdx/codex-isolated/comms_collectives_gather_reduce_probe_20260706_005444/runs/zero_stride_independent_correctness_20260706_014405/probe_summary.md
/home/adnan-cdx/codex-isolated/comms_collectives_gather_reduce_probe_20260706_005444/runs/zero_stride_independent_correctness_20260706_014405/commands.md
```

Flash structural run:

```text
/home/adnan/codex-isolated/flash_attention_comms_backend2162_20260706_005751/runs/relayout_on_20260706_010236
```

Minimized flash DDC reproducer:

```text
/home/adnan/codex-isolated/flash_attention_comms_backend2162_20260706_005751/ddc_abort_minimize_20260706_012508/subset_2_3_trimmed
```
