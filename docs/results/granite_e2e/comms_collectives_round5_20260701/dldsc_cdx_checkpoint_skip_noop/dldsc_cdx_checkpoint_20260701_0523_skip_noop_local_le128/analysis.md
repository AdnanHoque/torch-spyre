# CDX-pf DLDSC/backend collectives checkpoint 2026-07-01

Workspace: /home/adnan-cdx/codex-isolated/dldsc_collectives_stage_local_20260701_015300
Pod/namespace: adnan-cdx-spyre-dev-pf / a6-quantization

## Prior blocker

Prior run: /home/adnan-cdx/codex-isolated/dldsc_collectives_stage_local_20260701_015300/runs/local_le128_ring_sequence_multiaddr_burst_20260701_043805_dcc_l3le128
Result: exit_code=143
Captured stack: AgenToSentientLoweringPass::fuseLoadOrStoreChainOps(...) while walking/lowering agen vector memory ops.

## Narrow cause identified

The staged local LE128 ring helper generator produced LX-to-LX helper nodes whose source and destination were identical. In generated dataops, every localAssembleBeforeRing/localExtractAfterRing helper byte-range had srcStartAddr == destStartAddr and srcByteOffset == dstByteOffset. That injected pure no-op L3 load/store chains before/after the ring.

Count from prior dataops: 7,168 local helper nodes, 48,128 byte-range entries, 48,128 no-op entries.

The code path is:
- dcg/dcg_fe/pcfg_gen/stcdpOp.cpp:createLocalSubStickLe128NodeForRingDt(...)
- cloneAnchorAddrToLocalLe128Node(...) clones the same ring anchor into both src and dest addresses.
- The local helper then materialized ranges with identical src/dst intra-stick offsets.
- dcc/src/Conversion/PCFGToDataflowIR/PCFGToDataflowIR.cpp:createLE128DataTransferOp(...) lowered those no-op helper entries to L3 agen::VectorLoadOp -> agen::VectorStoreOp chains.
- dcc/src/Conversion/AgenToSentient/AgenToSentient.cpp:fuseLoadOrStoreChainOps(...) repeatedly walked/lowered the inflated unit.

## Patch prototyped

Patched dcg/dcg_fe/pcfg_gen/stcdpOp.cpp only: filter out local helper ranges where srcIntraStickOffset == dstIntraStickOffset, and return nullptr if no materialized ranges remain. This preserves current semantics for the generated no-op helpers while avoiding the L3 Agen lowering blow-up.

Build command:
cmake --build /home/adnan-cdx/codex-isolated/dldsc_collectives_stage_local_20260701_015300/deeptools/build-stage-local-safe --target dxp_standalone -j 8

Build completed successfully.

## Rerun result

New run: /home/adnan-cdx/codex-isolated/dldsc_collectives_stage_local_20260701_015300/runs/local_le128_ring_sequence_multiaddr_burst_20260701_051839_skip_noop_local_le128
Result: exit_code=143; terminated manually after stack sampling.

Generated dataops after patch: 0 local helper nodes, 0 local helper entries.

The replay advanced past the prior AgenToSentientLoweringPass::fuseLoadOrStoreChainOps blocker. Stack samples showed DCC later CPU-bound in sentient CFG conditional-tree passes:
- First sample: CFGDeepMergingPass -> dcc::CFGDeepMergingConditionalTree::mergeConditionals(...) -> dcc::ConditionalTree::compute()
- Final sample before termination: CFGSimplificationSentientLevelPass -> dcc::CFGSSentientLevelConditionalTree::compute()

## Recommended next patch

The next blocker is not the local LE128 helper fuser spin. Inspect CFG conditional tree construction and repeated walks for the generated sub-stick ring PCFG without local helpers:
- dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:CFGSimplificationSentientLevelPass::runOnOperation(), especially construction of dcc::CFGSSentientLevelConditionalTree at lines around 703 and 887.
- dcc/src/Transform/Sentient/Analyses/CFGSSentientLevelConditionalTree.cpp:CFGSSentientLevelConditionalTree::compute().
- dcc/src/Analysis/ConditionalTree.cpp:ConditionalTree::compute().

A narrow next investigation is to count sentient if/conditional ops per ProgramUnit after Agen lowering and before CFG simplification, then gate or batch CFGSSentientLevelConditionalTree recomputation for units with the repeated sub-stick ring branch pattern.
