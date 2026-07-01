# Layout All-Gather Explicit-Remap Schema Extension

## Implemented

Added a semantic checker lane for the flash edge contract, without attempting unsafe backend lowering.

Changed checker file:

`/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/deeptools/scripts/explicit_lx_range_semantic_check.py`

New/changed functions:

- `_validate_layout_view`: validates source/destination layout views, stick dims, shape/work-slice/per-core-tile consistency.
- `_validate_dimension_rename`: validates `restickify.* -> batchmatmul.*` rename and extent preservation.
- `_logical_allgather_plan`: derives the grouped all-gather transfer cardinality and a logical transfer excerpt.
- `check_layout_allgather_schema`: validates schema v2, communication class, replication kind, and `batchmatmul.KERNEL` LX lifetime.
- `_render_layout_check_text` and `main`: add `--layout-schema` while preserving the existing `--sdsc` byte-range checker path.

Updated schema artifact:

`/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/metadata/latest_flash_proposed_explicit_schema.json`

The schema now explicitly carries:

- `communicationClass: layout_allgather_restickify`
- source `mul.OUTPUT` and destination `batchmatmul.KERNEL` layout views
- dimension rename: `restickify.x -> batchmatmul.out`, `restickify.out -> batchmatmul.in`, `restickify.mb -> batchmatmul.x`
- grouped all-gather replication metadata: 4 groups, 8 producers/group, 8 consumers/group, 256 logical transfers
- consumer operand lifetime contract: `batchmatmul.KERNEL` resides in LX, has no HBM materialization, and is valid only from all-gather completion through the KERNEL operand read

## Validation

Commands run from Deeptools workspace:

```bash
python3 -m py_compile scripts/explicit_lx_range_semantic_check.py
python3 scripts/explicit_lx_range_semantic_check.py   --layout-schema ../runs/flash_explicit_layout_restickify_20260701_062020/metadata/latest_flash_proposed_explicit_schema.json   --json-out ../runs/flash_explicit_layout_restickify_20260701_062020/metadata/layout_allgather_semantic_check.json   --text-out ../runs/flash_explicit_layout_restickify_20260701_062020/metadata/layout_allgather_semantic_check.txt
python3 scripts/explicit_lx_range_semantic_check.py   --sdsc ../runs/flash_explicit_layout_restickify_20260701_062020/focused_bundle_input/sdsc_10.json   --diag ../runs/flash_explicit_layout_restickify_20260701_062020/replay_senulator/explicit_range_failure_dump.txt   --dataop sdsc_10_Tensor1_explicit_range   --json-out ../runs/flash_explicit_layout_restickify_20260701_062020/metadata/grouped_semantic_check_after_layout_extension.json   --text-out ../runs/flash_explicit_layout_restickify_20260701_062020/metadata/grouped_semantic_check_after_layout_extension.txt
```

Results:

- layout schema check: `pass`, `logicalTransferCount=256`, `modeledByteCount=33554432`
- existing grouped byte-range check: `pass`, `destinationSha256=5fe9ce7f75a692627342aa2a9e2740c6304ccea2a54e19cac994fc5b17dbc5d8`

## Remaining Backend Blockers

Full lowering is still too large for a safe minimal patch. Exact backend areas:

- `deeptools/dsc/dataOpDsc.cpp`: current `rangedLxRemap` import parses v1 byte movements only. It needs a gated v2 import for layout views, dimension rename, replication groups, and consumer operand lifetime.
- `deeptools/dsc/dataOpDsc.h`: `STCDPOpLx` prototype state only stores byte-range movement records. It needs a separate layout-all-gather/remap contract type, not more v1 fields.
- `deeptools/dcg/dcg_fe/transfer_compute/transfer_compute.cpp`: `materializeExplicitLxRangeProto` materializes one producer and one consumer per byte-range row. The flash edge needs grouped many-source-to-many-destination all-gather while preserving layout/stick semantics and KERNEL operand residency.
- `deeptools/dcg/dcg_fe/pcfg_gen/stcdpOp.cpp`: current prototype relaxes small piece/stick checks only for v1 explicit range import. A v2 lowering needs native piece construction for grouped layout remap instead of relying on those relaxations.
- Program correction / senulator path still hits the prior zero-layout synthetic piece issue for grouped explicit transfers: `ProgCorrectionScatter0`, `ProgCorrectionFlit`, `pieceSize=1`, `layoutSize=0`.

## Decision

Do not mutate the flash SDSCs into a replay bundle yet. The safe next backend step is a gated v2 import and a real layout-aware grouped all-gather primitive, then a tiny synthetic SDSC that consumes `batchmatmul.KERNEL` from LX under the lifetime contract.
