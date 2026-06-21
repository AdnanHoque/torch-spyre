# LX Relayout Lean PR Scope

Date: 2026-06-21

## Question

What is the smallest production PR that keeps the same working functionality as
`pr-lx-relayout`, and can it be done with zero Deeptools changes?

## Current Branch Baseline

`origin/pr-lx-relayout` is one Torch commit on top of current upstream
`torch-spyre/main`:

```text
upstream/main:        f63dc3c4
origin/pr-lx-relayout: 69b3f29a
merge-base:           f63dc3c4
```

Current Torch diff:

```text
7 files changed, 3536 insertions(+), 12 deletions(-)
```

File breakdown:

```text
279    tests/inductor/test_onchip_move.py
23/7   torch_spyre/_inductor/codegen/bundle.py
1461   torch_spyre/_inductor/codegen/onchip_move.py
34     torch_spyre/_inductor/config.py
1706   torch_spyre/_inductor/onchip_move.py
11     torch_spyre/_inductor/passes.py
22/5   torch_spyre/_inductor/spyre_kernel.py
```

Most of the blast radius is isolated in two new files:

- `torch_spyre/_inductor/onchip_move.py`
- `torch_spyre/_inductor/codegen/onchip_move.py`

The existing-file hooks are small and opt-in.

## Essential Torch Surface

The smallest honest Torch PR still needs these pieces:

1. Config flags, default off:
   - `SPYRE_ONCHIP_MOVE_PLANNER`
   - `SPYRE_ONCHIP_MOVE_REALIZE`
   - `SPYRE_ONCHIP_MOVE_CARRIER=coordinate_remap`
   - LX base and chunk-size controls

2. A pass hook after work distribution and before LX planning:
   - work distribution gives producer/consumer `PerCoreView`s;
   - existing `LX_PLANNER` still owns same-view same-core persistence;
   - this pass only handles mismatched same-stick LX producer/consumer views.

3. Planner:
   - classify producer-to-consumer edges;
   - skip same-view edges;
   - reject partial K-split/reduce-like cases;
   - compute common-refinement movement cells;
   - support subviews needed by fused SwiGLU gate/up halves;
   - attach movement metadata to the consumer op.

4. Codegen realization:
   - delay SDSC writeout until bundle-level patching;
   - patch producer output and consumer input to LX;
   - emit mixed SDSC rows with `LXCoordinateRemapOp`;
   - schedule movement before consumer compute;
   - support range encoding and chunking so artifacts stay tractable;
   - support non-adjacent/reuse and local relay cases already needed by the
     correctness-proven path.

5. Focused unit coverage:
   - common-refinement cell coverage;
   - coordinate-remap support/rejection checks;
   - emitted mixed SDSC contains `LXCoordinateRemapOp` and schedule order.

The existing `tests/inductor/test_onchip_move.py` is already small: three tests,
279 lines. It is worth keeping in the PR, though one test could be adjusted to
cover fused-SwiGLU subview/reuse explicitly.

## Lean Trims

Recommended `pr-lx-relayout-lean` code trim:

1. Remove dormant STCDP carrier code from
   `torch_spyre/_inductor/codegen/onchip_move.py`.

   The branch only permits `coordinate_remap`, and the test asserts
   `STCDPOpLx` is not emitted. AST inspection shows the removable dead/legacy
   functions are about 535 lines:

   ```text
   build_mixed_onchip_move_sdsc              87
   build_stcdp_datadsc                      117
   _logical_dataop_layout                    36
   _dsc_logical_layout                       42
   _logical_host_strides                     10
   _device_to_logical_mapping                62
   _logical_cells                            54
   _project_logical_cells                    23
   _span_partial_stick_dim_for_output        41
   _coalesce_piece_cells                     49
   _word_length                               2
   _consumer_mixed_schedule                  11
   ```

2. Remove `onchip_move_output_piece_mode`.

   This knob exists for the old STCDP/output-piece path and should not be in
   the coordinate-remap PR.

3. Consider keeping JSONL observability but dropping bulk debug-directory
   output from the production PR.

   This saves little LOC, so reviewer value matters more than raw size. JSONL
   is useful for proving whether the opt-in pass fired or skipped.

4. Keep range encoding and chunking.

   These are production hygiene, not benchmark scaffolding. Without them the
   generated SDSC can become too large for realistic remaps.

Expected lean Torch size after the obvious trims:

```text
approximately 3000 total inserted lines
approximately 2700 production inserted lines
```

Further reductions are possible only by narrowing the feature below the
correctness-proven behavior, for example dropping fused-SwiGLU subviews,
non-adjacent/reuse, or local relay. Those would make the PR smaller but would
not preserve the same functionality.

## Can This Be Zero Deeptools?

For the same functionality: no.

The working feature requires the backend to understand an explicit movement
primitive:

```text
LXCoordinateRemapOp:
  source core
  source LX byte address
  destination core
  destination LX byte address
  byte count / range
  schedule before consumer compute
```

Torch can plan and emit this metadata, but stock Deeptools cannot infer or
lower arbitrary remote-LX movement from Torch metadata alone.

Zero-Deeptools alternatives do not preserve the feature:

| Alternative | Zero Deeptools? | Same functionality? | Issue |
| --- | --- | --- | --- |
| HBM/ReStickify fallback | Yes | No | Keeps the round trip we are removing. |
| Co-assignment | Yes | No | Avoids movement by harming matmul layout. |
| Existing DL ops | Maybe | No proven path | Local compute semantics, not arbitrary source-core to destination-core copy. |
| `STCDPOpLx` | Maybe for narrow cases | No | Not a general coordinate remap; earlier probes were value-wrong/fault-prone. |
| `InputFetchNeighbor` | No/unclear | No | Backend assumptions and shape limits do not cover general remap. |
| Torch-only metadata | Yes | No | Useful for diagnostics only; no realized on-chip movement. |

The lowest-friction backend request is still a small Deeptools PR, not zero:

1. accept scheduled mixed SDSCs containing both `dscs_` and `dataOpdscs_`;
2. route scheduled mixed SDSCs through the existing data-op + DL-op path;
3. import `LXCoordinateRemapOp`;
4. lower range-encoded whole-stick LX copies to existing ring transfer nodes.

The lean Deeptools prototype branch previously measured as:

```text
8 files changed, 468 insertions(+), 34 deletions(-)
```

with these conceptual commits:

```text
Allow scheduled mixed data-op SDSCs
Add LX coordinate remap ring lowering prototype
Lower ranged LX coordinate remaps
```

## Recommendation

Do not contort PR 1 into a zero-Deeptools workaround. The clean primitive is
more reviewable than a Torch-only escape hatch that silently gives up either
matmul layout quality or the HBM-round-trip removal.

For `pr-lx-relayout-lean`:

1. start from `origin/pr-lx-relayout`;
2. remove dormant STCDP carrier code and `onchip_move_output_piece_mode`;
3. keep coordinate-remap planner/codegen, range encoding, chunking, subviews,
   reuse, and local relay;
4. keep the three focused unit tests;
5. keep docs, benchmark harnesses, and large artifact records off the final PR.

This gives the lowest-friction honest PR: small hooks in existing Torch files,
two isolated implementation files, a default-off experimental flag, focused
unit tests, and a narrow Deeptools companion change that implements exactly the
primitive Torch emits.
