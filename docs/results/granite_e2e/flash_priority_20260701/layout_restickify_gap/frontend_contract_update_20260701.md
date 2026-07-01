# Flash Layout-All-Gather Frontend Contract Update

## Summary

Torch branch `ah/comms-collectives` now emits the same logical contract shape that the Deeptools checker validates for the latest flash edge:

`mul -> ReStickifyOpHBM -> batchmatmul KERNEL`

This edge is classified as:

`layout_allgather_restickify` / `all_gather`

It is not scatter. The metadata now carries the layout/stick views and dimension rename needed by the backend to avoid interpreting the edge as a direct owner-to-owner copy.

## Code Delta

Commit: `ea98e174` (`inductor: emit flash layout allgather contract`)

Touched files:

- `torch_spyre/_inductor/layout_allgather_restickify.py`
- `torch_spyre/_inductor/lx_relayout.py`
- `tests/inductor/test_lx_relayout_dldsc.py`

The shared helper is:

`make_layout_allgather_restickify_contract(...)`

The planner-side payload now includes:

- `kind = layout_allgather_restickify`
- `communication_class = all_gather`
- `producer_layout = [out, x, mb]`, stick `out`
- `restickify_kernel_layout = [x, out, mb]`, stick `x`
- `consumer_kernel_layout = [out, in, x]`, stick `out`
- `dimension_rename = {restickify.x -> batchmatmul.out, restickify.out -> batchmatmul.in, restickify.mb -> batchmatmul.x}`
- `requires_staged_realization = true`

Unrelated relayout classifications now strip `None` fields before serialization, so scatter metadata does not grow layout-specific null fields.

## Validation Run On DEV

Pod: `adnan-spyre-dev-pf`

Passing commands:

```bash
cd /home/adnan/codex-isolated/artifact_push/torch-spyre-ah-comms-collectives
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=. python3 -m pytest -q tests/inductor/test_layout_allgather_restickify_import_light.py
python3 -m py_compile torch_spyre/_inductor/layout_allgather_restickify.py torch_spyre/_inductor/lx_relayout.py
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=. python3 tools/classify_layout_allgather_restickify.py /home/adnan/codex-isolated/flash-sdsc-20260701-033044/artifacts/flash_layout_restickify_gap_20260701/sdsc_triplet_snippets.json
```

Results:

```text
tests/inductor/test_layout_allgather_restickify_import_light.py: 2 passed
py_compile: passed
real flash snippet classifier: emitted layout_allgather_restickify/all_gather contract with layout views and dimension rename
```

Known DEV limitation:

`tests/inductor/test_lx_relayout_dldsc.py` could not collect on this pod because importing `torch_spyre._C` currently fails with a `libspyre_comms` undefined symbol. That is an environment/runtime-library mismatch in this artifact worktree, not a test assertion failure. The new test was added there for a properly pinned Torch runtime environment.

## Next Backend Step

The backend still needs to turn this validated contract into movement:

1. produce an on-chip restickified KERNEL view from the producer LX tensor;
2. group by batch-local consumer group;
3. all-gather/replicate producer chunks into each consumer core that executes the BMM KERNEL operand;
4. preserve the KERNEL operand lifetime until `batchmatmul` consumes it.
