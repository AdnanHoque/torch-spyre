# Flash DLDSC layout/restickify checkpoint

Date: 2026-07-01
Pod/namespace: adnan-cdx-spyre-dev-pf / a6-quantization
DLDSC workspace inspected: /home/adnan-cdx/codex-isolated/dldsc_collectives_stage_local_20260701_015300
Flash Torch workspace inspected: /home/adnan-cdx/dt-inductor-mixed/torch-spyre-flash-ws-stage105
Master-ish Deeptools inspected: /home/adnan-cdx/dt-inductor-mixed/deeptools

## Scope change

Stopped the generic DLDSC CFG replay. A gated CFG diagnostic patch had already been added and dxp_standalone rebuilt in /home/adnan-cdx/codex-isolated/dldsc_collectives_stage_local_20260701_015300/deeptools, but no CFG replay was run after the priority change. This checkpoint focuses on flash attention activation-spill layout/restickify.

## Question

Can the flash mul -> ReStickifyOpHBM -> batchmatmul activation spill be represented as DLDSC tensor-vs-compute distribution metadata plus backend-generated relayout, or does it need explicit layout/restickify support beyond scatter?

## Short answer

Not on Deeptools master/PR4408 as-is. The shape of the solution should be metadata-driven, but current master/PR4408 does not have a serialized Torch-to-Deeptools edge contract that says: this consumer input is LX-resident with producer distribution P, consumer distribution C, source layout Lp, target layout Lc, and layout transform/restickify semantics. Existing master has explicit ReStickifyOpHBM parser/DDL/PCFG support and scatter support; it does not infer the flash activation-spill bridge from ordinary DLDSC layout fields alone.

The staged DLDSC branch is closer: dxp/SdscRelayoutInsertion.cpp can detect an LX input allocation whose allocateCoordinates_.coreIdToWkSlice_ differs from the consumer SuperDsc::coreIdToWkSlice_ and inject STCDPOpLx movement. It also has a loop-scoped operand movement prototype gated by lxRelayoutClassifications_ fields, including kind=layout_restickify_activation and communication_pattern=layout_transform_then_operand_broadcast. That is the right backend direction, but it is not yet a stable public metadata contract from Torch.

## Evidence from Torch flash code

Relevant source paths inspected:
- /home/adnan-cdx/dt-inductor-mixed/torch-spyre-flash-ws-stage105/torch_spyre/_inductor/onchip_realize.py
- /home/adnan-cdx/dt-inductor-mixed/torch-spyre-flash-ws-stage105/torch_spyre/_inductor/insert_restickify.py
- /home/adnan-cdx/dt-inductor-mixed/torch-spyre-flash-ws-stage105/torch_spyre/_inductor/optimize_restickify.py
- /home/adnan-cdx/dt-inductor-mixed/torch-spyre-flash-ws-stage105/torch_spyre/_inductor/restickify_ring.py
- /home/adnan-cdx/dt-inductor-mixed/torch-spyre-flash-ws-stage105/tools/onchip_flash_artifact_inspect.py

Key observations:
1. insert_restickify.py records restickify_plan entries from incompatible input edges and inserts spyre.restickify before consumers. It commits FixedTiledLayout/SpyreTensorLayout, but that alone is compiler IR metadata; backend-visible SDSC metadata must be emitted later.
2. optimize_restickify.py classifies impossible cases as gather/scatter stick incompatibility: input stick zero -> output nonzero means gather, output zero -> input nonzero means scatter. This is a Torch planner limitation, not enough to describe full producer-vs-consumer distribution for backend movement.
3. restickify_ring.py already computes producer/restickify splits, symbol maps, core mapping overrides, and locality certificates. These are exactly the minimal Torch-side facts Deeptools needs, but today they are telemetry/optimizer metadata, not a durable SDSC edge contract.
4. onchip_realize.py has explicit flash layout-xform pair artifact builders. build_flash_attention_layout_xform_pair_tile_artifacts emits predecessor and consumer sidecars and a concrete STCDPOpLx dataop with source_layout, consumer_layout, split_dim, stick_dim, iter_sizes, source_pieces, and LX bases. This is more than plain DLDSC metadata decoration; Torch is currently materializing the bridge.
5. _flash_attention_kv_repack_broadcast_edge requires producer op ReStickifyOpHBM for K/V fanout style probes and records blockers for non-executed plan artifacts: executable mixed SDSC ownership and one-to-many PieceInfo broadcast support are not fully proven.

## Evidence from Deeptools

Relevant source paths inspected:
- /home/adnan-cdx/dt-inductor-mixed/deeptools/dcg/dcg_fe/pcfg_gen/restickifyOp.cpp
- /home/adnan-cdx/dt-inductor-mixed/deeptools/ddc/ddl_templates/restickify.ddl
- /home/adnan-cdx/dt-inductor-mixed/deeptools/ddc/ddl_templates/test/sdsc_restickify.json
- /home/adnan-cdx/codex-isolated/dldsc_collectives_stage_local_20260701_015300/deeptools/dxp/SdscRelayoutInsertion.cpp

Master-side Deeptools has explicit ReStickifyOpHBM/ReStickifyOpLx/ReStickifyOpWithPT* structs, JSON parsing, DDL template, and DCG PCFG generation. That supports an explicit restickify SDSC/op path. It does not by itself let an ordinary batchmatmul input allocation express the flash bridge without either an explicit ReStickifyOpHBM SDSC or extra relayout metadata.

The staged branch adds backend relayout insertion. The key trigger is an LX input with non-empty allocateCoordinates_.coreIdToWkSlice_ that differs from the consumer SuperDsc::coreIdToWkSlice_. The prototype then creates STCDPOpLx dataops or HBM relayout SDSCs. For classified loop-scoped movement it recognizes kind=layout_restickify_activation and communication_pattern=layout_transform_then_operand_broadcast and injects staged STCDPOpLx movement inside the consumer schedule.

## Minimal Torch metadata changes

Add a backend-visible per-edge layout/restickify contract for compiler-inserted restickify edges, serialized into the consumer SDSC input allocation or a sidecar classification block. Minimal fields:
- edge kind: layout_restickify_activation
- communication_pattern: layout_transform_then_operand_broadcast, or layout_transform_same_core when no fanout
- producer op/name and output lds index; consumer op/name and input lds index
- source_hbm_addr/shared_hbm_addr when HBM spill is the rendezvous point
- source_layout and consumer_layout in Deeptools dimension names
- source_stick_dim and consumer_stick_dim; for this path reject differing stick dims initially
- iter_sizes, split_dim/consumer_split, producer_split/mapped_split when producer distribution differs
- producer coreIdToWkSlice and consumer coreIdToWkSlice, or compact equivalent core mapping override
- source PieceInfo/LX starts when the producer is already LX-resident; otherwise HBM load start addresses
- runtime policy bits: executable=true, requires_broadcast=true/false, one_to_many_pieceinfo_expected=true/false

Torch files/functions to extend:
- torch_spyre/_inductor/restickify_ring.py: promote locality certificate/core mapping override/source metadata into a serializable backend contract.
- torch_spyre/_inductor/onchip_realize.py: instead of only materializing flash layout_xform_pair STCDPOpLx artifacts, attach the contract to the original consumer SDSC input when the edge is mul/ReStickifyOpHBM -> batchmatmul and the current checks pass.
- tools/onchip_flash_artifact_inspect.py and tests/_inductor/test_onchip_flash_artifact_inspect.py: validate the new contract fields and DXP-generated dataops/debug components.

## Minimal Deeptools lowering changes

Implement backend generation in the staged relayout insertion path, not in scatter:
1. Parse/preserve the Torch edge contract on SuperDsc, likely adjacent to the current lxRelayoutClassifications_ mechanism used by SdscRelayoutInsertion.cpp.
2. In dxp/SdscRelayoutInsertion.cpp, generalize isLoopScopedLxOperandMovement() to select this contract for the consumer lds index, not only ad hoc classification strings.
3. Populate the consumer input allocation allocateCoordinates_.coreIdToWkSlice_ from producer distribution and SuperDsc::coreIdToWkSlice_ from consumer distribution before relayout insertion, so existing mismatch-trigger logic can create STCDPOpLx movement.
4. For same stick dim and same element set with layout-order transform, use STCDPOpLx movement with source_layout -> consumer_layout and existing PieceInfo.
5. For producer low-core to consumer 32-core fanout, add proven one-to-many PieceInfo/STCDPOpLx broadcast semantics or fail closed; do not map this to ScatterOpHBM. Scatter only covers HBM scatter correction style and lacks the layout/restickify activation contract.
6. Keep ReStickifyOpHBM explicit lowering as fallback for standalone restickify SDSCs and for cases that need PT/SFP semantics beyond simple movement.

## Decision for flash mul -> ReStickifyOpHBM -> batchmatmul

Use DLDSC tensor-vs-compute distribution metadata plus backend-generated relayout as the intended final design, but add explicit layout/restickify edge metadata beyond scatter. Plain master/PR4408 fields are insufficient; explicit ReStickifyOpHBM support exists but would keep Torch emitting an explicit op/SDSC rather than letting backend synthesize the transfer from the consumer edge.

For the first low-risk implementation, support only: same element set, same stick dim, one producer, one consumer, HBM rendezvous or producer LX pieces known, no additional future consumers, and no stick transform. Fail closed to existing explicit sidecar/restickify artifacts otherwise.

## Replay status

No focused SDSC replay was run. I did not find an existing flash cache bundle quickly under /home/adnan-cdx/dt-inductor-mixed or /home/adnan-cdx at bounded search depth, and generating one would require a broader Torch compile/perf run. The checkpoint is therefore an implementation plan with source evidence, not a replay result.

## Artifacts

- analysis: /home/adnan-cdx/codex-isolated/dldsc_collectives_stage_local_20260701_015300/runs/flash_dldsc_layout_restickify_20260701_20260701_062102/analysis.md
- Deeptools diff: /home/adnan-cdx/codex-isolated/dldsc_collectives_stage_local_20260701_015300/runs/flash_dldsc_layout_restickify_20260701_20260701_062102/deeptools.diff
- Deeptools status: /home/adnan-cdx/codex-isolated/dldsc_collectives_stage_local_20260701_015300/runs/flash_dldsc_layout_restickify_20260701_20260701_062102/deeptools_status_short.txt
