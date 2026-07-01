# Flash layout/restickify gap - 2026-07-01

## Scope

Analysis-only lane on pod `adnan-spyre-dev-pf`, namespace `a6-quantization`. I used the existing SDSCs under `/home/adnan/codex-isolated/flash-sdsc-20260701-033044` and did not rerun the flash test.

Source runs:

- Baseline: `/home/adnan/codex-isolated/flash-sdsc-20260701-033044/runs/baseline_noh2d_20260701_040758/cache/inductor-spyre/sdsc_fused_add_amax_exp_maximum_mul_sub_sum_transpose_unsqueeze_1_41uw4ts1`
- Optimized PR2939/scatter run: `/home/adnan/codex-isolated/flash-sdsc-20260701-033044/runs/optimized_noh2d_20260701_041326/cache/inductor-spyre/sdsc_fused_add_amax_exp_maximum_mul_sub_sum_transpose_unsqueeze_1_eg9zimnr`
- Prior classification: `/home/adnan/codex-isolated/flash-sdsc-20260701-033044/artifacts/flash_classification_20260701/`

## Representative edge

The representative edge is `sdsc_1.json -> sdsc_2.json -> sdsc_3.json`:

- `sdsc_1.json`: `mul`, 32 cores, `numWkSlicesPerDim_={mb:4, x:8, out:1}`, `coreIdToWkSlice_ = mb=core_id%4, x=floor(core_id/4)%8, out=0`.
- `mul` output is `Tensor2_lx`, DS type/layout `OUTPUT`, `layoutDimOrder_=[out,x,mb]`, `stickDimOrder_=[out]`, shape dims `mb=4, x=4096, out=128`, per-core tile `mb=1, x=512, out=128`.
- `sdsc_2.json`: `ReStickifyOpHBM`, same work division as `mul`. It reads `Tensor0_lx` in `OUTPUT [out,x,mb]`/stick `out`, and writes `Tensor1_hbm`/pool in `KERNEL [x,out,mb]`/stick `x`.
- `sdsc_3.json`: `batchmatmul`, 32 cores, `numWkSlicesPerDim_={x:4, mb:8, out:1, in:1}`, `coreIdToWkSlice_ = x=core_id%4, mb=floor(core_id/4)%8, out=0, in=0`.
- The restickified tensor is the `batchmatmul` `KERNEL` operand after dimension rename: `restickify.x -> bmm.out`, `restickify.out -> bmm.in`, `restickify.mb -> bmm.x`. The BMM KERNEL expects `layoutDimOrder_=[out,in,x]`, stick `out`, shape dims `x=4, mb=1024, out=4096, in=128`, per-core compute tile `x=1, mb=128, out=4096, in=128`.

## Structural identity

All 32 repeated rows are structurally identical in the fields relevant to this edge:

- Baseline: 32 / 32 identical, `diff_ids=[]`.
- Optimized: 32 / 32 identical, `diff_ids=[]`.
- Baseline representative and optimized representative are identical after ignoring file paths/top-level numeric op names.

The repeated IDs are `[2, 19, 36, 53, 70, 87, 104, 121, 138, 155, 172, 189, 206, 223, 240, 257, 274, 291, 308, 325, 342, 359, 376, 393, 410, 427, 444, 461, 478, 495, 512, 529]`.

## Classification

This is best modeled as **layout-aware on-chip restickify plus grouped all-gather/broadcast into `batchmatmul` KERNEL**, not as scatter-only, not as a plain gather from HBM, and not as a capacity/streaming issue.

Reason: for each batch (`restickify.mb` / `bmm.x`), the producer and consumer use the same group of eight cores, but the role of the second split changes. The producer split is along `restickify.x` / `bmm.out`, so each core owns one 512-wide chunk of the BMM KERNEL out dimension. The consumer split is along BMM `mb`; every consumer core in that batch group computes a different 128-row `mb` slice but needs the full `out=4096, in=128` KERNEL for the batch. Removing the HBM materialization therefore requires each of the eight consumer cores to see all eight producer chunks while keeping the KERNEL stick/layout.

The optimized log confirms why current LX pinning does not apply: `stderr.log:11776` reports `lx_pinning: buf27 (restickify) -> core div mismatch`, and generated `SDSCArgs` in this region have `lx_residency_core_id_to_wk_slice=None`.

## Required frontend support

The frontend needs metadata and a pass decision that can represent a producer/consumer core-division remap with layout transform, not only same-core LX residency or scatter-capable relayout:

- Preserve the producer LX PerCoreView for the `mul` output through the inserted restickify edge: `OUTPUT [out,x,mb]`, stick `out`, `mb=4,x=8,out=1` core division.
- Attach to the `batchmatmul` KERNEL operand a consumer view: `KERNEL [out,in,x]`, stick `out`, `x=4,mb=8,out=1,in=1` core division, with the dimension rename above.
- Classify this edge as a grouped replication/all-gather remap: within each batch group, producer chunks split on BMM `out` must be replicated to all consumer cores split on BMM `mb`.
- Extend the existing LX residency/remap metadata beyond `lx_residency_core_id_to_wk_slice=<same owner map>` so it can encode source owner view, destination consumer view, layout transform, and replication group.

Likely frontend files for a focused prototype:

- `/home/adnan/codex-isolated/flash-sdsc-20260701-033044/repos/torch-spyre-optimized/torch_spyre/_inductor/lx_relayout.py`
- `/home/adnan/codex-isolated/flash-sdsc-20260701-033044/repos/torch-spyre-optimized/torch_spyre/_inductor/scratchpad/allocator.py`
- `/home/adnan/codex-isolated/flash-sdsc-20260701-033044/repos/torch-spyre-optimized/torch_spyre/_inductor/op_spec.py`
- `/home/adnan/codex-isolated/flash-sdsc-20260701-033044/repos/torch-spyre-optimized/torch_spyre/_inductor/spyre_kernel.py`
- `/home/adnan/codex-isolated/flash-sdsc-20260701-033044/repos/torch-spyre-optimized/torch_spyre/_inductor/codegen/superdsc.py`
- Restickify/layout planning touchpoints: `propagate_layouts.py`, `insert_restickify.py`, `optimize_restickify.py`, `pass_utils.py`.

## Required backend support

For the DLDSC path, the backend needs an LX-resident restickify/remap primitive that can change stick/layout and change core distribution in one producer-to-consumer handoff. In this case it must gather/broadcast eight `out` chunks within each batch-local core group into the BMM KERNEL operand, without materializing the restickified KERNEL in HBM/pool.

For the explicit-remap path, the backend lowering needs an explicit core-to-core remap primitive that is not just scatter of a primary output. It must support many-source-to-many-destination replication/all-gather plus layout/restickify semantics, then feed the PT `batchmatmul` KERNEL operand from LX-resident data. A minimal explicit lowering could split this into `restickify-to-LX chunks` plus `group_allgather/remap_to_bmm_kernel_view`, but the scheduler/codegen still needs to understand the consumer view and lifetime.

Likely backend files for investigation/prototype:

- `/home/adnan/codex-isolated/flash-sdsc-20260701-033044/repos/deeptools-optimized/dsm/dsm.cpp` (`ReStickifyOpHBM`, `ReStickifyOpWithPTLx`, `ReStickifyOpWithPTHBM` lowering areas)
- `/home/adnan/codex-isolated/flash-sdsc-20260701-033044/repos/deeptools-optimized/dsm/graphOptimizer.cpp` (DLDSC/restickify graph transforms)
- `/home/adnan/codex-isolated/flash-sdsc-20260701-033044/repos/deeptools-optimized/dsc/` and `/home/adnan/codex-isolated/flash-sdsc-20260701-033044/repos/deeptools-optimized/ddc/` if the SDSC/DLDSC arg schema must grow.

## Tiny prototype plan, not implemented

1. Add a tiny detector for the exact edge pattern: producer `mul`/pointwise LX `OUTPUT [out,x,mb]`, inserted `ReStickifyOpHBM` to KERNEL `[x,out,mb]`, consumer `batchmatmul` KERNEL with renamed dims and `producer work_slices=(mb=4,x=8)` vs `consumer work_slices=(x=4,mb=8)`.
2. Teach `lx_relayout.py`/scratchpad allocation to classify it as `layout_allgather_restickify` instead of rejecting only as `core div mismatch`.
3. Extend `TensorArg`/SuperDSC metadata with source and destination core maps plus replication axis, leaving lowering gated behind a feature flag.
4. Add a metadata-only unit test first; only then wire a backend primitive once DLDSC/explicit-remap semantics are agreed.

## Granite overlap

This does not directly overlap the Granite block remaining spills if those are PR2939 scatter/remap cases. The flash gap is an activation `mul -> restickify -> batchmatmul KERNEL` handoff with layout transform plus grouped all-gather/replication across a core-division transpose. Scatter-only support can remove HBM stores where each value has a destination owner; it does not remove this flash class unless extended to layout-aware all-gather/broadcast.

## Artifacts written

- `/home/adnan/codex-isolated/flash-sdsc-20260701-033044/artifacts/flash_layout_restickify_gap_20260701/flash_layout_restickify_gap.md`
- `/home/adnan/codex-isolated/flash-sdsc-20260701-033044/artifacts/flash_layout_restickify_gap_20260701/representative_edge.json`
- `/home/adnan/codex-isolated/flash-sdsc-20260701-033044/artifacts/flash_layout_restickify_gap_20260701/sdsc_triplet_snippets.json`
