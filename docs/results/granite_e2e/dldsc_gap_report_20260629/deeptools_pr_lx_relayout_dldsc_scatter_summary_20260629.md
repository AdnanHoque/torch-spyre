# Deeptools PR1 Scatter Patch Summary - 2026-06-29

Patch files in this artifact directory:

- `deeptools_pr_lx_relayout_dldsc_scatter.patch`: plain diff against `ai-chip-toolchain/deeptools` master.
- `deeptools_pr_lx_relayout_dldsc_scatter.format-patch`: signed-off commit patch preserving author/message metadata.

## Branch

- Fork branch: `https://github.ibm.com/Adnan-Hoque1/deeptools/tree/pr-lx-relayout-dldsc-scatter`
- Commit: `b8c09743c46505b4cac46b434b9eb3243ae0b685`
- Base observed in review clone: `ai-chip-toolchain/deeptools` master `0a9da5eb19`

## What Changed

The patch touches two files:

- `dxp/SdscRelayoutInsertion.cpp`
- `ddc/ddc_fold.cpp`

Functional changes:

1. DXP resident relayout now sizes the post-relayout LX piece from the tensor dimensions that actually participate in the tensor layout, instead of treating every consumer core as if it needs the full tensor form.
2. DXP probes and allocates the post-relayout LX piece at the inserted relayout program step, so the capacity check and the final allocation use the same memory-tracker state.
3. DXP builds relayout LDS metadata from the relayout operand's own `primaryDsInfo_`, rather than assuming `OUTPUT` metadata.
4. DXP ignores compute-only/non-tensor split dimensions while creating relayout pieces and HBM fallback offsets.
5. DDC coordinate propagation permits custom `coreIdToWkSlice_` maps when they are constant along the propagated corelet split dimension, rejecting only maps that actually vary on that dimension.

## Why These Changes Were Needed

Torch PR1 emits the logical handoff through dl-dsc allocation coordinates: an LX input can carry producer-residency `coreIdToWkSlice_` while the consumer SDSC carries its own compute `coreIdToWkSlice_`. That is enough information for Deeptools to infer a resident scatter relayout.

Current Deeptools master has the relayout insertion mechanism, but the useful Torch-generated Granite/SwiGLU-style scatter cases exposed correctness and capacity gaps:

- The backend was overestimating resident materialization size by using full form size per consumer core. For PR1 scatter, the backend should allocate the consumer resident piece, not the entire tensor on every core.
- The backend needed to ignore dimensions that exist in compute splitting but not in the operand tensor layout. Otherwise real matmul operands with extra compute dimensions become inconsistent or oversized.
- Some Torch-emitted coordinate maps include full dim entries with constants. DDC should reject only the dimension that truly varies against a corelet split, not every custom coordinate map.

## Why Not Other Changes

We did not add a new public data movement op. We used the backend's existing dl-dsc relayout insertion path and internal `STCDPOpLx` realization.

We did not move frontend policy into Deeptools. Torch still selects the work division and records the logical producer residency. Deeptools only synthesizes physical movement for the mismatch already described in dl-dsc coordinates.

We did not add debug prints or diagnostic-only instrumentation to the patch. Earlier local diagnostics were intentionally left out of this clean branch.

## Interaction With Torch PR1

Torch PR1:

1. Extends LX planning to keep resident producer tensors when producer and consumer `PerCoreView`s differ.
2. Emits producer-residency coordinates on the consumer LX input allocation.
3. Reserves conservative LX space so backend-inserted resident relayout has somewhere to materialize.

Deeptools PR1:

1. Sees the allocation-vs-compute coordinate mismatch.
2. Inserts an internal `LxRelayout` SuperDSC.
3. Realizes the scatter movement through `STCDPOpLx` when the consumer piece fits in LX.

Together, they replace HBM round trips on resident scatter edges with on-chip LX relayout.

Torch interaction note: during review we fixed Torch's conservative reservation formula to reserve the backend resident piece (`source_bytes / consumer_slices`) rather than `source_bytes * consumer_core_count`. Without that Torch-side fix the planner could clear relayout metadata before codegen, leaving no coordinate-bearing SDSC rows for Deeptools to realize.

## Evidence And Speedup

The corrected dev-pf Granite causal prefill run (`B=1,S=512,E=4096`) produced:

| Variant | Kernel ms/iter | Median wall ms | Kernel speedup | Wall speedup |
|---|---:|---:|---:|---:|
| baseline, relayout off | 14.697693 | 34.857512 | 1.000x | 1.000x |
| PR1 scatter, full frontend LX + backend split wrapper + local GraphEditor fix + corrected Torch reservation | 12.014579 | 31.895638 | 1.223x | 1.093x |

The before/after SDSC evidence is in:

- `devpf_isolated_20260629/jamie_style_baseline_off.md`
- `devpf_isolated_20260629/jamie_style_boundary_full_torch_lx_backend1_graphfix.md`
- `devpf_isolated_20260629/sdsc_comm_classes_devpf.md`
- `pr1_scatter_review_and_artifacts_20260629.md`

## Remaining Communication Classes

PR1 should stay scoped to resident scatter. The next named communication classes are:

- `matmul_operand_broadcast` / `all_gather_replicate`: attention value/PV-style operands where consumer cores need a replicated/gathered operand view, not a one-to-one resident remap.
- Reduction-aware movement: producer output is partial, so movement placement must occur after reduction or include reduction semantics.
- Layout-changing movement: LX movement plus restickify/reformat, not just same-stick resident scatter.
- Scheduling/overlap: pipeline movement with compute once the communication classes are present.

## Reproduction Notes

See `pr1_scatter_review_and_artifacts_20260629.md` and `devpf_isolated_20260629/isolated_run_summary.md` for the exact pod runbook. The most important local caveat is the split frontend/backend LX fraction wrapper: full frontend LX means Torch sees `DXP_LX_FRAC_AVAIL=0`, but DXP must see `DXP_BACKEND_LX_FRAC_AVAIL=1` remapped to `DXP_LX_FRAC_AVAIL=1` to avoid backend chunk-capacity failure.
