# DLDSC Backend Next Collectives Scope - 2026-07-02

Scope inspected on pod `adnan-cdx-spyre-dev-pf`.

- Deeptools: `/home/adnan-cdx/codex-isolated/dldsc_backend_path_20260702_074814/deeptools`, branch `ah/comms-collectives`, SHA `00a37826a8c8e1b32f97c7d6edbc2527f1359076`.
- Torch artifact cross-check: `/home/adnan-cdx/codex-isolated/flash_contract_validate_20260701_082005/torch-spyre`, branch `ah/comms-collectives`.
- Verification run: `./util/util_unit_test --gtest_filter="LayoutAllgatherRestickify.*"` passed 13/13.

## Current Backend Support

The compact DLDSC coordinate path is real, but narrow. Torch can serialize producer residency as `scheduleTree_[].coordinates_.coreIdToWkSlice_` on an LX input allocation and can serialize top-level `lxRelayoutClassifications_`. Deeptools now imports/exports that classification map in `SuperDsc`.

DXP's executable path is still driven by coordinate incompatibility, not by the classification taxonomy. `Dxp::insertRelayoutSdsc` looks for an LX-pinned input whose allocation `coreIdToWkSlice_` is non-empty and differs from the consumer SDSC `coreIdToWkSlice_`. It then synthesizes a relayout SDSC, usually `STCDPOpLx` if LX space is available, otherwise an HBM identity relayout. This works as a backend-derived resident relayout for direct scatter/permutation-style pointwise edges.

`layout_allgather_restickify` has a new backend helper and DXP hook, but it is not a general all-gather engine. The helper accepts only the flash-shaped `mul -> ReStickifyOpHBM/ReStickifyOpLx -> batchmatmul` contract with `communication_class == all_gather`, complete layout/stick metadata, matching producer/restickify core counts, group/chunk dimensions, unsplit `out/in`, and complete `dimension_rename`. It can emit a deterministic logical movement-plan artifact and logical source/destination core pairs. DXP recognizes that classification, fail-closes invalid metadata, emits the artifact, then proceeds through the existing generic LX relayout insertion path.

## Remaining Backend Gaps

| Class if Torch emits DLDSC metadata | Current state | Concrete gap |
|---|---|---|
| pointwise scatter / permutation | Supported by coordinate mismatch when a full resident destination view is legal. | Needs more negative tests, but no major architectural gap for PR1-style resident relayout. |
| pointwise broadcast / multicast | `STCDPOpLx` has multicast metadata/GTR paths, and generic resident relayout may express some one-to-many piece movement. | No classification-aware DLDSC dispatcher for `broadcast`/`multicast`; no loop-scoped fan-out lowering from compact metadata; DXP ignores `communication_class` except the special layout-allgather case. |
| pointwise gather | `GatherOpHBM` exists as an HBM-oriented DataOp; generic `STCDPOpLx` can reason about overlapping pieces. | No LX-to-LX DLDSC-coordinate gather synthesizer; no production grouped fan-in descriptor; no local partial-stick assemble/extract primitive to merge byte ranges from multiple sticks into one consumer stick. |
| all-gather | Special flash layout-restickify metadata is validated and planned logically. | No general grouped all-gather lowering from metadata. The current path still relies on resident STCDP-style relayout, which is not safe for attention-sized full replication; it needs loop-scoped IFN/STCDP plus whole-stick remote staging and local assemble/extract. |
| reduce | DL compute reductions exist (`SUM`, `MAX`, nonstick variants, `REDUCE` lowering), but not as a relayout primitive. | Compact coordinate metadata needs an arithmetic fan-in primitive, not just byte movement: reduce op kind, dtype/accumulation precision, destination ownership, local partial-stick reduce/accumulate, and synchronization. |
| all-reduce | DSM/multi-AIU collective code has `AllReduce`/`AllGather` algorithms, separate from this single-SDSC LX relayout path. | Need reduce plus redistribution wired to DLDSC coordinate metadata; no current bridge from `lxRelayoutClassifications_.communication_class == all_reduce` to generated single-AIU LX movement/reduction. |

## Cross-Cutting Backend Work

1. Add a DLDSC classification dispatcher in DXP/DCG after SuperDSC import. It should consume `lxRelayoutClassifications_`, validate the producer/consumer coordinate contract, and choose a lowering by class instead of treating every mismatch as resident STCDP.
2. Synthesize executable movement from metadata. Either create DataOpDSCs/IFN modules internally or generate equivalent PCFG/dataflow modules, then schedule them with the consumer DLDSC. Metadata alone currently does not create executable movement.
3. Add a grouped, loop-scoped LX movement primitive for `broadcast`, `multicast`, `gather`, and `all_gather`: compact producer/consumer groups, whole-stick ring transfers, and local LE128 assemble/extract with independent `srcByteOffset`, `dstByteOffset`, and `numBytes`.
4. Extend DCC stitching/scheduling for generated collective modules. The current DataOp+DLDSC path has a special input-neighbor-fetch case and module ordering based on `coreIdToDscSchedule`; generalized metadata-generated collectives need explicit schedule/module ownership.
5. Add reduction-specific lowering: local and cross-core accumulation, op semantics, accumulation format, identity values, ordering/sync, then optional redistribution for all-reduce.
6. Add focused tests per class: tiny SuperDSC import tests for metadata preservation, DXP tests that prove the correct lowering is selected, and one DCC/PCFG smoke case for each of broadcast/multicast, gather, all-gather, reduce, and all-reduce.

## Bottom Line

The backend is ready for compact DLDSC metadata for direct resident scatter/permutation and for validating the one known flash `layout_allgather_restickify` shape. It is not yet ready for Torch to replace explicit DataOpDSCs with metadata for general broadcast/multicast, gather/all-gather, or reduce/all-reduce. Those classes still need backend-owned collective synthesis, grouped scheduling, local sub-stick assembly, and reduction arithmetic support.
