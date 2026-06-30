# dl-dsc communication agent notes - 2026-06-30

## Targeted Granite gaps

The guarded inventory scope is limited to the two remaining non-weight runtime
classes:

- `layout_restickify_activation`: attention computed activation
  `ReStickifyOpHBM` feeding the downstream batchmatmul.
- `matmul_operand_broadcast`: attention value-side batchmatmul operand
  all-gather/replicate.

Scatter is already handled for the guarded Granite edges. Weight restickifies
remain out of scope and should stay on the offline/preload weight-layout path.

## dl-dsc contract assessment

`allocateCoordinates_.coreIdToWkSlice_` can express producer tensor ownership
and consumer ownership/compute mismatch. That is sufficient for resident
scatter and for naming the producer-vs-consumer distribution mismatch behind a
matmul operand broadcast.

It cannot, by itself, express a pre/post stick-layout transform. A
`layout_restickify_activation` edge changes the physical stick/layout form of a
computed tensor, not only which core owns each logical slice. The production
contract therefore needs an explicit layout-restickify activation field/class
that carries at least the source layout, destination layout, operand identity,
and computed-vs-weight scope. Deeptools then needs to lower that contract to an
LX layout transform such as `ReStickifyOpLx` or an equivalent internal data-op.

## Narrow prototype

The experimental Torch patch maps a restickify SDSC to `ReStickifyOpLx` only
when all of the following are true:

- `SPYRE_LX_PLANNER_RELAYOUT_RESTICKIFY_OUTPUTS=1` is enabled.
- The restickify source is a `ComputedBuffer`.
- Every tensor argument in the restickify SDSC is already allocated in LX.

This keeps graph-input/weight restickifies as `ReStickifyOpHBM`.
