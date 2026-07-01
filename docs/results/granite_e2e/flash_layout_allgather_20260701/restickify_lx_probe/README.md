# ReStickifyOpLx Probe

This probe reran the H=4 flash staged all-gather contract with Torch emitting `ReStickifyOpLx` when the restickify input and output are both LX-resident.

Result: Torch emitted `ReStickifyOpLx` rows and preserved `layout_allgather_restickify` metadata on four consumer `batchmatmul` rows, but stock DXP still aborts with `std::out_of_range: map::at`.

Interpretation: the op-name mismatch is not the only blocker. The useful remaining backend work is still grouped all-gather realization plus rebinding the post-restickify LX KERNEL view into the consumer BMM.

Key rows copied here:

- `sdsc_2.json`, `sdsc_19.json`, `sdsc_36.json`, `sdsc_53.json`: `ReStickifyOpLx` rows.
- `sdsc_3.json`, `sdsc_20.json`, `sdsc_37.json`, `sdsc_54.json`: consumer `batchmatmul` rows carrying `layout_allgather_restickify` metadata.

Artifacts:

- `run_summary.json`: exit status and SDSC count.
- `run.log`: raw compile log with DXP abort.
- `sdsc_summary.csv`: compact row summary.
- `classifications.json`: extracted all-gather contracts.
