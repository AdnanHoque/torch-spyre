# Explicit Remap Senulator Failure Investigation - Flash LayoutAllGather

## Scope

Pod-local CLC investigation only.
Workspace: `/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212`.

I did not touch Torch or the DLDSC backend. The only code artifact in this explicit lane remains the prototype checker/emitter script:

- `/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/deeptools/scripts/explicit_lx_range_semantic_check.py`

## Concrete Probe

Latest probe pointer:

- `/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/metadata/latest_concrete_layout_allgather_range_probe.txt`

Latest concrete probe:

- `/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/concrete_layout_allgather_range_probe_20260701_083445`

The probe represents the latest flash edge:

- `sdsc_1.json:mul -> sdsc_2.json:ReStickifyOpHBM -> sdsc_3.json:batchmatmul KERNEL`
- communication class: `layout_allgather_restickify`
- concrete carrier: `STCDPOpLx.rangedLxRemap.movementRanges`
- range count: 256
- bytes per range: 131072
- modeled movement: 33554432 bytes
- destination coverage: each core gets contiguous `[0x100000, 0x200000)` bytes

## Generated Explicit Range SDSC Fields Checked

In replay bundle:

- `/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/concrete_layout_allgather_range_probe_20260701_083445/dxp_bundle_replay_senulator/bundle_input/sdsc_10.json`

Data-op:

- `datadscs_[...]/sdsc_10_Tensor1_explicit_range/op/name = STCDPOpLx`
- `op/rangedLxRemap/schema = layout-aware-grouped-lx-remap-v2-concrete-ranges`
- `op/rangedLxRemap/groupedRangeSchema/schema = layout-aware-grouped-byte-range-v1`
- `op/rangedLxRemap/movementRanges` length = 256
- `coreIdsUsed_` length = 32
- `dimPool_ = [mb, out, in]`
- `labeledDs_[0].dimToLayoutSize_ = {in:128, mb:32, out:256}`
- `labeledDs_[1].dimToLayoutSize_ = {in:128, mb:32, out:256}`
- range `logicalSlice` dims use only existing LDS dims: `mb/out/in`
- true flash layout semantics are carried separately in `source.layoutSlice`, `destination.layoutSlice`, `sourceView`, `destinationView`, and `dimensionRename`

This means the generated explicit range SDSC is not malformed for the current importer. Normal DXP imports and routes it.

## Commands Run

### Static/probe checks

```bash
cd /home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/deeptools
python3 -m py_compile scripts/explicit_lx_range_semantic_check.py

python3 scripts/explicit_lx_range_semantic_check.py \
  --layout-schema /home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/metadata/latest_flash_proposed_explicit_schema.json \
  --emit-ranged-sdsc /home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/concrete_layout_allgather_range_probe_20260701_083445/layout_allgather_concrete_range_sdsc.json \
  --emit-ranged-remap-json /home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/concrete_layout_allgather_range_probe_20260701_083445/layout_allgather_concrete_ranged_remap.json \
  --text-out /home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/concrete_layout_allgather_range_probe_20260701_083445/layout_schema_to_ranges_check.txt \
  --json-out /home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/concrete_layout_allgather_range_probe_20260701_083445/layout_schema_to_ranges_check.json

python3 scripts/explicit_lx_range_semantic_check.py \
  --sdsc /home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/concrete_layout_allgather_range_probe_20260701_083445/layout_allgather_concrete_range_sdsc.json \
  --text-out /home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/concrete_layout_allgather_range_probe_20260701_083445/concrete_range_semantic_check.txt \
  --json-out /home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/concrete_layout_allgather_range_probe_20260701_083445/concrete_range_semantic_check.json
```

Results:

- Python compile: PASS
- layout schema check: PASS
- concrete byte-range semantic check: PASS

### Normal DXP import/routing

```bash
DEEPTOOLS_PATH=/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/deeptools \
DEEPTOOLS_EXPLICIT_LX_RANGE_PROTO=1 \
DEEPTOOLS_EXPLICIT_LX_RANGE_PROTO_DUMP=/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/concrete_layout_allgather_range_probe_20260701_083445/dxp_bundle_replay/explicit_range_diag.txt \
/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/deeptools/build-explicit-range-agent-nollvm/dxp/dxp_standalone \
  --bundle -d /home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/concrete_layout_allgather_range_probe_20260701_083445/dxp_bundle_replay/bundle_input
```

Result: PASS, `rc=0`.

Diagnostic evidence:

- `/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/concrete_layout_allgather_range_probe_20260701_083445/dxp_bundle_replay/explicit_range_diag.txt`
- `parsedRangeCount: 256`
- `dtTableCount: 256`

### Senulator with default runtime correction path

```bash
DEEPTOOLS_PATH=/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/deeptools \
DEEPTOOLS_EXPLICIT_LX_RANGE_PROTO=1 \
DEEPTOOLS_EXPLICIT_LX_RANGE_PROTO_DUMP=/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/concrete_layout_allgather_range_probe_20260701_083445/dxp_bundle_replay_senulator/explicit_range_diag.txt \
/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/deeptools/build-explicit-range-agent-nollvm/dxp/dxp_standalone \
  -b senulator --bundle -d /home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/concrete_layout_allgather_range_probe_20260701_083445/dxp_bundle_replay_senulator/bundle_input
```

Result: FAIL, `rc=134`.

Failure:

```text
DtException: skv.second <= layoutSize, file .../dcg/dcg_fe/transfer_compute/transfer_compute.cpp line 639
pieceVerificationFailure dataOpDsc=ProgCorrectionScatter0 op=ScatterOpHBM lds=ProgCorrectionFlit piece=p0 dim=d1 pieceSize=1 layoutSize=0
```

### Senulator with compile-time correction enabled

```bash
DEEPTOOLS_PATH=/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/deeptools \
DEEPTOOLS_EXPLICIT_LX_RANGE_PROTO=1 \
DXP_ENABLE_COMPILE_TIME_CORRECTION=1 \
DEEPTOOLS_EXPLICIT_LX_RANGE_PROTO_DUMP=/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/concrete_layout_allgather_range_probe_20260701_083445/dxp_bundle_replay_senulator_compiletime_correction/explicit_range_diag.txt \
/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/deeptools/build-explicit-range-agent-nollvm/dxp/dxp_standalone \
  -b senulator --bundle -d /home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/concrete_layout_allgather_range_probe_20260701_083445/dxp_bundle_replay_senulator_compiletime_correction/bundle_input
```

Result: PASS, `rc=0`.

## Root Cause

This is not caused by:

- malformed generated explicit range SDSC
- missing explicit range LDS/layout fields needed by normal DXP import
- unsupported 256-transfer all-gather shape
- legacy HBM scatter assumptions in the explicit `STCDPOpLx` transfer itself

It is caused by the runtime program-correction path creating a zero-sized correction tensor and then forcing a nonzero scatter piece.

Backend path:

1. `dxp/dxp.cpp::prepareRuntimeCorrection()` creates runtime correction info whenever a correction-program node exists.
2. `dxp/dxp.cpp::fillCorrectionProperties(...)` computes correction statistics from the associated SDSCs.
3. In this replay, all relevant symbols are compile-time constants, so the correction-flit count can be zero.
4. `dxp/util.cpp::createProgCorrectNodeInfo(...)` creates `ProgCorrectionFlit` / `ProgramBinary` shapes with `d1 = totalProgFlits` or `numProgCorrectFlits`; in the failure case this is zero.
5. `dxp/util.cpp::createDataDscProgCorrection(...)` still creates `ProgCorrectionScatter0` and, for array C, forcibly sets `piece.dimToSize_[gatherScatterDim] = 1`.
6. `dcg/dcg_fe/transfer_compute/transfer_compute.cpp::verificationCheckDataOpDSC(...)` checks every piece against its LDS layout and fails because `pieceSize=1 > layoutSize=0` for `ProgCorrectionFlit` dim `d1`.

## Why Compile-Time Correction Works

With `DXP_ENABLE_COMPILE_TIME_CORRECTION=1`, DXP resolves constant symbols before runtime correction. The zero-correction runtime scatter is not materialized, so senulator completes successfully. This proves the explicit remap carrier is acceptable to senulator once the unrelated zero-flit correction path is avoided.

## Next Concrete Fix

Backend fix, not probe-generation fix:

- In `dxp/dxp.cpp::fillCorrectionProperties(...)` or immediately before `createDataDscProgCorrection(...)`, skip runtime correction SDSC generation when both:
  - `stats.numProgCorrectFlits == 0`
  - no unresolved correction symbols remain for the associated SDSCs

or equivalently:

- In `dxp/util.cpp::createDataDscProgCorrection(...)`, handle zero correction flits by emitting a no-op correction SDSC instead of a `ScatterOpHBM` with a forced one-element piece.

Short-term runbook workaround for this explicit prototype:

- run senulator validation with `DXP_ENABLE_COMPILE_TIME_CORRECTION=1`.

Remaining e2e lowering gap after this fix:

- The copied-bundle replay proves explicit carrier import/routing and senulator acceptance with compile-time correction. It still does not remove the real flash HBM row. The production lowering must bind the generated LX-resident KERNEL view to the downstream PT `batchmatmul` operand and remove/replace the original `ReStickifyOpHBM` materialization.
