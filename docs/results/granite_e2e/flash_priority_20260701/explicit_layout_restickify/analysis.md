# Flash explicit layout-restickify checkpoint

## Scope

CLC-pf pod only: `adnan-clc-spyre-dev-pf` in namespace `a6-quantization`.

Priority was changed to the latest flash edge. The literal `test_flash.py` source used by the newer flash lane is:

`/home/adnan/codex-isolated/flash_main_probe_20260701_015234/test-spyre-scripts/test_flash.py`

Identity captured from the existing flash workspace:

`/home/adnan/codex-isolated/flash-sdsc-20260701-033044/logs/test_flash_identity.log`

No local Mac files were modified and no push was performed.

## Latest Flash Edge

Existing latest flash artifacts live under:

`/home/adnan/codex-isolated/flash-sdsc-20260701-033044`

The representative edge is:

`sdsc_1.json:mul -> sdsc_2.json:ReStickifyOpHBM -> sdsc_3.json:batchmatmul KERNEL`

Source SDSC directories:

- baseline: `/home/adnan/codex-isolated/flash-sdsc-20260701-033044/runs/baseline_noh2d_20260701_040758/cache/inductor-spyre/sdsc_fused_add_amax_exp_maximum_mul_sub_sum_transpose_unsqueeze_1_41uw4ts1`
- optimized: `/home/adnan/codex-isolated/flash-sdsc-20260701-033044/runs/optimized_noh2d_20260701_041326/cache/inductor-spyre/sdsc_fused_add_amax_exp_maximum_mul_sub_sum_transpose_unsqueeze_1_eg9zimnr`

Summary from the existing artifact:

- producer `mul` output: LX `OUTPUT`, layout `[out,x,mb]`, stick `out`, shape `mb=4,x=4096,out=128`, work slices `mb=4,x=8,out=1`
- restickify: `ReStickifyOpHBM`, input LX `OUTPUT [out,x,mb]` stick `out`, output HBM/pool `KERNEL [x,out,mb]` stick `x`
- consumer `batchmatmul`: consumes the restickified tensor as KERNEL with rename `restickify.x -> bmm.out`, `restickify.out -> bmm.in`, `restickify.mb -> bmm.x`
- consumer work slices: `x=4,mb=8,out=1,in=1`; every consumer in a batch-local group needs the full `out=4096,in=128` KERNEL

Classification: **layout-aware on-chip restickify plus grouped all-gather/broadcast into `batchmatmul` KERNEL**. It is not scatter-only and not just capacity/streaming.

## Can Grouped Explicit Movement Express It?

At the physical communication level, yes: the needed movement is grouped many-source-to-many-destination replication. For each of 4 `bmm.x` / `restickify.mb` groups, eight producer cores own 512-wide chunks of the BMM `out` dimension, and all eight consumer `mb`-slice cores in that group need all eight chunks. A logical carrier needs about `4 * 8 * 8 = 256` grouped transfers before lower-level stick/substick splitting.

The current grouped explicit carrier is not sufficient as-is. It can express grouped strided byte copies, but the latest flash edge also needs:

- source and destination layout views, not only byte ranges
- dimension rename across restickify and BMM KERNEL
- grouped all-gather/replication semantics
- a consumer-operand lifetime contract so PT `batchmatmul` reads the KERNEL from LX

I wrote a proposed logical schema here:

`/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/metadata/latest_flash_proposed_explicit_schema.json`

## Replay Decision

I did not mutate the latest flash SDSCs into an explicit replay bundle. That would be unsafe with the current carrier because it would require inventing backend semantics for layout-aware all-gather and consumer KERNEL residency.

To keep a tiny executable result, I preserved and reran the nearest existing grouped explicit carrier probe under this CLC checkpoint:

- source bundle: `/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/explicit_range_grouped_sdsc10_20260701_024349/bundle_input`
- focused copy: `/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/focused_bundle_input`
- carrier edge: `sdsc_8.json:8_mul -> sdsc_9.json:ReStickifyOpHBM -> sdsc_10.json:batchmatmul`

Normal DXP / sentient-target replay:

`DEEPTOOLS_PATH=/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/deeptools DEEPTOOLS_EXPLICIT_LX_RANGE_PROTO=1 /home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/deeptools/build-explicit-range-agent-nollvm/dxp/dxp_standalone --bundle -d /home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/focused_bundle_input`

Result: `rc=0`, stderr empty.

Senulator backend replay:

`DEEPTOOLS_PATH=/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/deeptools DEEPTOOLS_EXPLICIT_LX_RANGE_PROTO=1 DEEPTOOLS_EXPLICIT_LX_RANGE_PROTO_DUMP=/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/replay_senulator/explicit_range_failure_dump.txt /home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/deeptools/build-explicit-range-agent-nollvm/dxp/dxp_standalone -b senulator --bundle -d /home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/focused_bundle_input`

Result: `rc=134`.

Failure:

`DtException: skv.second <= layoutSize, file /home/adnan/codex-isolated/explicit_range_agent_20260630/deeptools/dcg/dcg_fe/transfer_compute/transfer_compute.cpp line 639`

The diagnostic dump shows explicit grouped import and routing happened first: it parsed 128 ranges and produced L3SU/L3LU/LX route maps. The first new blocker is later:

`pieceVerificationFailure dataOpDsc=ProgCorrectionScatter0 op=ScatterOpHBM lds=ProgCorrectionFlit piece=p0 dim=d1 pieceSize=1 layoutSize=0`

Classification for that executable probe: senulator/backend compile-time-correction blocker, not Torch emitter/schema import and not the previous DCC stitcher collision.

## Required Changes

Frontend/schema changes:

- classify this exact flash edge as `layout_allgather_restickify` or equivalent, not as scatter and not as same-owner LX residency
- preserve producer LX PerCoreView for `mul` output and attach consumer KERNEL view for `batchmatmul`
- encode source owner map, destination consumer map, dimension rename, replication group, and layout/stick transform
- keep the feature gated until backend lowering is real

Backend changes:

- add an explicit core-to-core remap primitive for layout-aware many-source-to-many-destination replication/all-gather
- support restickify-to-LX chunks plus grouped remap into BMM KERNEL view, or a single combined primitive with equivalent semantics
- make DCG/DCC route grouped transfers without expanding them into invalid legacy LDS pieces
- teach senulator compile-time correction / ProgramCorrection scatter generation to avoid zero-layout synthetic `ProgCorrectionFlit` LDS dimensions for grouped explicit transfers

## Artifacts

- analysis: `/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/analysis.md`
- results: `/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/results.txt`
- latest `test_flash.py` copy: `/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/metadata/latest_test_flash.py`
- latest flash edge artifact: `/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/metadata/latest_flash_representative_edge.json`
- latest flash analysis copy: `/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/metadata/latest_flash_layout_restickify_gap.md`
- proposed explicit schema: `/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/metadata/latest_flash_proposed_explicit_schema.json`
- carrier focused bundle: `/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/focused_bundle_input`
- carrier sentient replay: `/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/replay_sentient`
- carrier senulator replay: `/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/replay_senulator`
- carrier senulator failure dump: `/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/replay_senulator/explicit_range_failure_dump.txt`
- workspace diffs/status: `/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/diffs`

## Notes

A tiny result from the interrupted generic lane was preserved at:

`/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/explicit_grouped_checkpoint_20260701_next_20260701_061242`

That lane was stopped after the priority change.
