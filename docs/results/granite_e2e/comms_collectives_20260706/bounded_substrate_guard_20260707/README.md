# Bounded DLDSC Communication Substrate Guard - 2026-07-07

This snapshot records the current state of the DLDSC LX communication substrate work after adding the explicit boundary between communication collectives and WSR/tile-scoping.

## Current Position

The communication branch should classify and express on-chip communication edges, then prove Deeptools can realize a bounded LX-resident tile correctly. It should not invent a full-tensor streaming relayout system. When an activation is too large to materialize safely in LX, the compiler should preserve HBM fallback or fail closed with a clear reason until WSR can tile the region.

The key new guard is for matmul-operand collectives. With `SPYRE_LX_PLANNER_RELAYOUT=1`, Torch now defaults the bounded matmul-operand relayout budget to 4 MiB:

```bash
SPYRE_LX_PLANNER_RELAYOUT_MAX_MATMUL_OPERAND_BYTES=4194304
```

That keeps the bounded attention-side collectives active, while avoiding the 13.1 MiB MLP down-projection full-activation path that explodes into thousands of chunks. That MLP case is WSR-shaped.

## Code Heads

Torch branch:

```text
AdnanHoque/torch-spyre:gather-restickify
commit bced14b4 inductor: enrich partial-view gather contracts
previous partial-view metadata checkpoint 6bc8b00d inductor: emit partial-view gather relayout metadata
previous bounded-budget checkpoint 99f95650 test: cover bounded matmul operand relayout budget
base behavior commit de76531b inductor: bound matmul operand relayout contracts
```

Deeptools branch:

```text
Adnan-Hoque1/deeptools:ah/comms-collectives
commit 2fa9220a6 [DXP] realize bounded partial-view gather relayout
previous partial-view guard checkpoint faa78233e [DXP] fail closed for partial-view gather relayout
previous bounded-broadcast checkpoint 9e9b20b42 [DXP] test bounded matmul operand broadcast patterns
base behavior commit c8c259061 [DXP] fail closed for oversized matmul operand relayout
```

Patch files archived here:

- `patches/torch_gather_restickify_bounded_operand_guard_de76531b.patch`
- `patches/torch_bounded_operand_budget_tests_99f95650.patch`
- `patches/deeptools_fail_closed_oversized_matmul_relayout_c8c259061.patch`
- `patches/deeptools_dxp_chunk_cap_fixture_61ffb6b3a.patch`
- `patches/deeptools_bounded_broadcast_multicast_dxp_tests_9e9b20b42.patch`

The follow-on partial-view gather guard is archived in:

- `../partial_view_gather_guard_20260707/patches/torch_partial_view_gather_metadata_6bc8b00d.patch`
- `../partial_view_gather_guard_20260707/patches/deeptools_partial_view_gather_fail_closed_faa78233e.patch`

The bounded offset-aware realization is archived in:

- `../partial_view_gather_bounded_relayout_20260707/patches/torch_partial_view_gather_contract_enrichment_bced14b4.patch`
- `../partial_view_gather_bounded_relayout_20260707/patches/deeptools_bounded_partial_view_gather_2fa9220a6.patch`

## Validation

Focused checks run on `adnan-cdx-spyre-dev-pf`:

```text
Torch py_compile: passed
Torch tests/inductor/test_lx_relayout_dldsc.py at 99f95650: 30/30 passed
Torch partial-view update at 6bc8b00d: py_compile passed
Torch partial_view_gather helper smoke at 6bc8b00d: passed with _C stub
Deeptools LayoutAllgatherRestickify.*: 32/32 passed
Deeptools CoreWorkDivIncomptLxRelayout*: 2/2 passed
Deeptools MatmulOperandBroadcastChunkCapFailsClosed: passed
Deeptools MatmulOperandBroadcastPattern*: 2/2 passed
Deeptools PartialViewGatherFailsClosedBeforeGenericLxRelayout: passed
Deeptools PartialViewGatherBoundedOffsetRelayoutCompiles at 2fa9220a6: passed
Combined DXP focused regression at 2fa9220a6: 6/6 passed
LayoutAllgatherRestickify.* at 2fa9220a6: 32/32 passed
```

Flash compile probe:

```text
run: /home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/test_flash_one_flag_gather_restickify_20260706_175419
returncode: 0
ReStickifyOpHBM: 0 in the structural probe summary
ReStickifyOpLx: 64 files mentioning ReStickifyOpLx
matmul_operand_broadcast plans: 32
backend plans realized: 32
realization strategy: gather_then_restickify
logical transfers: 8192
```

Granite prefill structural run:

```text
run: /home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/granite_s512_bounded_substrate_default4m_20260707_050254
case: prefill, seq_len=512, sdpa_causal
status: generated SDSC/plans, then hit an unrelated runtime flex convert_address path
```

The Granite run is still useful for classification and lowering evidence. It generated the bounded collectives and then failed later during runtime launch of an RMSNorm/linear bundle with:

```text
RuntimeError: convert_address not yet implemented - waiting for flex support
```

That is not the matmul-operand relayout descriptor failure.

## Granite Evidence

Bounded attention-side plans emitted:

```text
16_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json: 1024 logical transfers
7_batchmatmul_Tensor0_0_matmul_operand_broadcast_plan.json: 256 logical transfers
8_batchmatmul_Tensor0_0_matmul_operand_broadcast_plan.json: 64 logical transfers
8_batchmatmul_Tensor1_1_matmul_operand_broadcast_plan.json: 1024 logical transfers
```

All four are classified as:

```text
kind: matmul_operand_broadcast
pattern: all_gather_replicate
physical lowering: lowered_loop_scoped_kernel_neighbor
strategy: loop_scoped_input_fetch
```

See `summaries/granite_bounded_plan_table.md` for the table.

The MLP down-projection full activation is now explicitly skipped:

```text
lx relayout skip: buf35 -> buf36 matmul operand is 13107200 bytes, exceeding bounded relayout budget 4194304 bytes; preserving HBM fallback until WSR/tile-scoping handles this edge
```

This is intentional. The full tensor path had previously expanded into approximately:

```text
25,600 gathered pieces
6,400 chunks
12,800 data ops
```

That is a descriptor explosion, not a healthy bounded-tile communication test. Truncating it is value-wrong. The right handoff is: communication substrate proves the class and bounded realization; WSR makes the large activation tile-scoped.

## SDSC Classification Summary

See `summaries/granite_sdsc_classification_table.md`.

High-level readout:

- First attention bundle: contains bounded `matmul_operand_broadcast`, `partial_view_gather`, and `ReStickifyOpLx` evidence.
- Later attention/MLP bundles: remaining `ReStickifyOpHBM` rows are either outside this bounded communication class, weight/restickify related, or WSR/tile-scoping candidates.
- MLP down-projection activation remains HBM-backed in this branch by design, because the full tensor is too large for this substrate-only scope.

## Why The Deeptools Guard Matters

`DEEPTOOLS_MATMUL_OPERAND_BROADCAST_MAX_CHUNKS` previously truncated the movement list when the cap was exceeded. That can make a compile appear to pass while silently dropping required transfers. The new behavior fails closed with an explicit reason instead of producing value-wrong movement.

That matters for the MLP down-projection case: it should not be made to pass by keeping only the first few chunks. It should fall back or wait for WSR tile-scoping.

Backend regression coverage now includes:

- `DxpTestFixture.MatmulOperandBroadcastChunkCapFailsClosed`, which replays a compact generated `matmul_operand_broadcast` SDSC fixture with `DEEPTOOLS_MATMUL_OPERAND_BROADCAST_MAX_CHUNKS=1` and verifies that DXP rejects the case with the explicit fail-closed message instead of truncating movement.
- `DxpTestFixture.MatmulOperandBroadcastPatternBroadcastCompiles`, which rewrites the same compact SDSC fixture into a bounded one-source / 32-consumer `communication_pattern=broadcast` case and verifies DXP accepts it.
- `DxpTestFixture.MatmulOperandBroadcastPatternMulticastCompiles`, which rewrites the fixture into a bounded four-source / 32-consumer grouped `communication_pattern=multicast` case and verifies DXP accepts it.

## Reproduction Notes

Use the one-feature gate:

```bash
export SPYRE_LX_PLANNER_RELAYOUT=1
export DEEPTOOLS_ENABLE_MATMUL_OPERAND_GATHER_RESTICKIFY=1
```

For Granite full-LX local reproduction, use the split LX environment that has been necessary on the pods:

```bash
export DXP_LX_FRAC_AVAIL=0
export DXP_BACKEND_LX_FRAC_AVAIL=1
```

In this setup, Torch sees full frontend LX planning, while the DXP subprocess receives backend relayout workspace. Without this split, DXP can fail because it interprets the same knob as reserving too much LX for frontend tensors.

Typical pinned environment on CDX:

```bash
ROOT=/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236
TORCH_ROOT=$ROOT/torch-spyre
DEE=$ROOT/deeptools
BENCH=/home/adnan-cdx/spyre-granite-e2e-bench
FMS=/home/adnan-cdx/dt-inductor-codex-clean/profiler_runs/decode_regression_rev_ab_20260610_163300/foundation-model-stack-eager_spyre
PY212=/home/adnan-cdx/dt-inductor-codex-clean/.venv-py212/bin/python

source /home/adnan-cdx/dt-inductor-codex-clean/env.sh
source /home/adnan-cdx/dt-inductor-codex-clean/matmul_gap_env.sh
use_py212_localflex_optdeeptools_spyre_runtime

export PYTHONPATH=$TORCH_ROOT:$TORCH_ROOT/tests/inductor:$FMS:$BENCH:${PYTHONPATH:-}
export PATH=$DEE/install-deeptools/bin:$DEE/build-deeptools/dxp:$PATH
export LD_LIBRARY_PATH=$DEE/install-deeptools/lib64:$DEE/install-deeptools/lib:$DEE/build-deeptools/dxp:$DEE/build-deeptools/dcc/lib:${LD_LIBRARY_PATH:-}
export DEEPTOOLS_PATH=$DEE/install-deeptools/share
export DXP_STANDALONE=$DEE/install-deeptools/bin/dxp_standalone
export SPYRE_LX_PLANNER_RELAYOUT=1
export DEEPTOOLS_ENABLE_MATMUL_OPERAND_GATHER_RESTICKIFY=1
export DXP_LX_FRAC_AVAIL=0
export DXP_BACKEND_LX_FRAC_AVAIL=1
export SPYRE_INDUCTOR_LOG=1
export SPYRE_INDUCTOR_LOG_LEVEL=DEBUG
```

Granite command used for this snapshot:

```bash
cd $BENCH
$PY212 benchmarks/granite_block_layer_probe.py \
  --fms-root $FMS \
  --run-root $RUN \
  --case prefill \
  --seq-len 512 \
  --compile-block \
  --attn-name sdpa_causal \
  --iters 1 \
  --warmups 0
```

## Next Work

The next substrate work is not full-tensor streaming. The next useful steps are:

1. Keep tightening bounded collectives: scatter/permutation, broadcast/multicast, gather/all-gather.
2. Add value-oriented bounded synthetic tests for each class where possible.
3. For any full Granite edge that still spills because the live tensor is too large, label it as WSR/tile-scoping unless the bounded-tile version itself fails.
4. After WSR provides smaller live tiles, rerun the same communication substrate against the tile-scoped Granite graph.
