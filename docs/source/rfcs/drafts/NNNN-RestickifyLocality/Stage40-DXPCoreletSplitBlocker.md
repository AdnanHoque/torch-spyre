# Stage 40: DXP Corelet-Split Blocker

## Summary

Stage 39 found the right high-level contract: Deeptools DDC can expand the
`sdsc_restickify.json` fixture through `restickify_sen1p5.ddl`, and the DDC
output lowers through DCC to real `lxlu/lxsu/sfp/pt` work.

Stage 40 looked at why that still does not pass through DXP.

The blocker is earlier than codegen. DXP runs a generic corelet-split pass
before it runs DDC:

```text
Dxp::runDxpOnSdsc
  Dxp::runDsmClSplit(sdsc)
  Dxp::runDdc(sdsc)
  Dxp::runCodegen(sdsc)
```

That pre-DDC splitter currently assumes each DSC has exactly one data-stage
parameter:

```text
dsm/SdscCoreletSplit.cpp
auto& data_stage_params = dsc.dataStageParam_;
DT_CHECK(data_stage_params.size() == 1);
```

The restickify DDL fixture has two data-stage params before DDC. DDC itself can
handle it, but DXP never reaches DDC because the pre-DDC splitter aborts.

## Code Change

Added a diagnostic tool:

```text
tools/restickify_dxp_corelet_split_probe.py
```

The tool generates a small variant matrix from the Deeptools restickify fixture
and runs both `ddc_standalone` and `dxp_standalone`:

- `raw`
- `one_ds_repoint_loops`
- `output_scale_out_zero`
- `fake_restickify_reduction_after`

These are not candidate lowerings. They are boundary probes to distinguish
"can DDC understand this?" from "can DXP get past its pre-DDC pass?"

## Probe Command

On the pod:

```sh
python3 /tmp/restickify_dxp_corelet_split_probe.py \
  --sdsc /tmp/stage39-ddc-fixture/sdsc_restickify.json \
  --output-dir /tmp/stage40-dxp-corelet-split-probe \
  --deeptools-bin /opt/ibm/spyre/deeptools/bin \
  --senarch rcudd1a \
  --run-deeptools
```

The summary artifact was copied locally to:

```text
artifacts/stage40_dxp_corelet_split_probe/summary.json
```

## Results

| Variant | DDC | DXP | Interpretation |
|---|---:|---:|---|
| `raw` | 0 | -6 | DDC succeeds, but DXP aborts before DDC at `data_stage_params.size() == 1`. |
| `one_ds_repoint_loops` | -6 | -6 | Collapsing to one data stage gets past the first assertion, but breaks DDC and DCG schedule validity. |
| `output_scale_out_zero` | 1 | -6 | A psum-style skip hack gets past corelet split, but breaks DDL matching. |
| `fake_restickify_reduction_after` | -6 | -6 | A fake reduction op trips DDC's compute-op invariants. |

The important asymmetry is:

```text
raw fixture:
  ddc_standalone  -> succeeds
  dxp_standalone  -> fails before DDC
```

So the raw fixture is not inherently malformed for DDC. It is incompatible with
DXP's current pre-DDC corelet-split assumption.

## Why JSON Hacks Are Not Enough

The tempting hack is to force DXP to skip corelet splitting by making the op
look like it needs psum reduction, because `doCoreletSplitSdsc()` returns early
for psum cases. That does get past the exact assertion, but every version we
tried breaks a later invariant:

- changing output scale makes DDL matching fail
- appending a fake compute op makes DDC reject the op
- collapsing data-stage params loses schedule-tree consistency

This is good news in the boring way: there is probably not a hidden one-line
Torch-Spyre JSON tweak that makes this work. The fix belongs at the Deeptools
pipeline boundary.

## Proposed Deeptools Fix Direction

The smallest plausible Deeptools patch is to make DXP's pre-DDC corelet splitter
skip restickify DDL-template inputs that already use the multi-data-stage
contract, and leave restickify corelet handling to DDC's restickify-specific
path.

Conceptually:

```cpp
for (auto& dsc : sdsc->dscs_) {
  bool is_restickify =
      dsc.computeOp_.size() == 1 &&
      (dsc.computeOp_.front().opFuncName == OpFuncs::ReStickifyOpLx ||
       dsc.computeOp_.front().opFuncName == OpFuncs::ReStickifyOpHBM);

  if (is_restickify && dsc.dataStageParam_.size() != 1) {
    continue;
  }

  DT_CHECK(dsc.dataStageParam_.size() == 1);
  ...
}
```

This is only a proposed direction. It still needs a Deeptools build validation,
because skipping the splitter may expose another DXP/DIP issue later in the
pipeline. But the Stage 39 and Stage 40 evidence says this is the next
high-signal thing to test.

## What This Means For The Restickify Project

For the Torch-Spyre side, the conclusion is unchanged:

1. Stage 3B mapping alignment is still a narrow, default-off locality prototype.
2. Real LX-local restickify needs a first-class lowerer, not `ReStickifyOpHBM`
   renaming.
3. Deeptools already contains most of the schedule knowledge in
   `restickify_sen1p5.ddl`.
4. The immediate blocker is that DXP's pre-DDC corelet splitter rejects the
   fixture shape DDC expects.

If we want to continue this path, the next experiment should be a small
Deeptools branch that skips or relaxes `doCoreletSplitSdsc()` for restickify
multi-data-stage inputs, then reruns the same Stage 40 fixture through DXP.

## Validation

Local:

```text
python3 -m py_compile tools/restickify_dxp_corelet_split_probe.py
```

Pod:

```text
python3 /tmp/restickify_dxp_corelet_split_probe.py ... --run-deeptools
```

