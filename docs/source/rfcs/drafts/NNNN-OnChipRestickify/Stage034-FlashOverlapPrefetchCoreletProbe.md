# Stage 034: Flash Overlap Prefetch Corelet Probe

Date: 2026-05-26

## Purpose

Stage 033 proved that the independent-overlap row now reaches DCC stitching,
but fails because both sides of the row claim the same physical unit:

```text
component=lxlu
corelet=0
schedule_idx=2
```

This stage tightens the next hypothesis: the current generated batchmatmul
compute stays on corelet 0, while the independent prefetch data-op should use
the other corelet.  That preserves the intended high-level row:

```text
corelet 0: current tile compute
corelet 1: next tile prefetch
```

instead of serializing both programs through the same `lxlu` slot.

## Local Deeptools Source Audit

The pod remained unreachable through the cluster API:

```text
Unable to connect to the server:
context deadline exceeded
```

I inspected local Deeptools source mirrors instead.  They are advisory until
the pod state can be rechecked, but they explain the Stage 033 failure:

```text
dcc/src/Conversion/PCFGToDataflowIR/PCFGToDFManager.cpp
dcc/src/Stitcher/ModuleStitcher.cpp
```

For a paired `[dataop, dldsc]` schedule row, `PCFGToDFManager` associates the
schedule slot with the data module while `ModuleStitcher` also places DLDSC
program units into the same slot because the row has a DL index.  If both
modules emit a `GetUnitOp` for the same core/corelet/component, the stitch map
has only one slot and fails with:

```text
unit already set for associated schedule step
```

The local Deeptools sources also show:

```text
dsc/dataOpDsc.cpp
  STCDPOpLx currently parses fields such as enOptMC, gateLXCodeGen,
  enSubPieceReuse, and forceModeMC, but not coreletId.

dcg/dcg_fe/pcfg_gen/stcdpOp.cpp
  ordinary STCDPOpLx PCFG creation hardcodes LXLU0/LXSU0/PE0 paths.
```

So a Torch-only descriptor change will not be sufficient until Deeptools is
patched to honor `STCDPOpLx.coreletId`.

## Torch-Spyre Change

Changed:

```text
torch_spyre/_inductor/codegen/onchip_bridge.py
torch_spyre/_inductor/onchip_realize.py
tests/_inductor/test_onchip_flash_pipeline_logic.py
tests/_inductor/test_onchip_realize_logic.py
```

`build_flash_attention_pipeline_bridge(...)` now accepts:

```text
stcdp_corelet_id
```

When set, generated `STCDPOpLx` data-op descriptors include:

```json
{"name": "STCDPOpLx", "coreletId": 1}
```

The overlap-prefix tile builder passes `stcdp_corelet_id=1` for its prefetch
data-ops and records:

```text
flashAttentionPipeline_.prefetch_corelet_id = 1
```

Serial flash sidecars still omit `coreletId`, preserving their existing
descriptor shape.

## Validation

Local:

```text
python3 tests/_inductor/test_onchip_flash_pipeline_logic.py
  11/11 passed

python3 tests/_inductor/test_onchip_realize_logic.py
  31/31 passed

python3 -m py_compile \
  torch_spyre/_inductor/codegen/onchip_bridge.py \
  torch_spyre/_inductor/onchip_realize.py \
  tests/_inductor/test_onchip_flash_pipeline_logic.py \
  tests/_inductor/test_onchip_realize_logic.py

git diff --check
  passed
```

Pod/device validation did not run because `oc exec` still timed out.

## Interpretation

This does not yet prove the attention variant executes with overlap.  It moves
the emitted descriptor toward the resource split implied by the Stage 033
diagnostic:

```text
compute:  corelet 0
prefetch: corelet 1
```

The next Deeptools patch should make STCDPOpLx honor that field:

1. add a `coreletId` field to the STCDPOpLx data-op model, or to the common
   STCDP base if that is the cleaner ownership boundary;
2. parse and print `coreletId` for STCDPOpLx in `dsc/dataOpDsc.cpp`;
3. route STCDPOpLx PCFG generation through `LXLU1/LXSU1/PE1` when
   `coreletId == 1`, at least for the ordinary LX-to-LX path used by the flash
   prefetch sidecar;
4. rerun `warp_overlap_probe` against the patched DXP and check whether the
   first duplicate moves from `lxlu/corelet=0` to a different real resource
   conflict or the bundle compiles.

If that exposes a new duplicate on `PE0`/`SFP0` or another compute unit, the
same corelet split must be extended to the STCDP helper path that currently
uses `PE0` for LX-SFP-LX transfers.
