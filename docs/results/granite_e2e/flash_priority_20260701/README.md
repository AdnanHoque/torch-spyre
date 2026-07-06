# Flash Priority Checkpoint - 2026-07-01

Priority shifted to the latest `test_flash.py` attention spill.

Key finding: PR1 scatter removes 0 flash HBM spills because the remaining repeated edge is not scatter. It is an activation layout/restickify plus grouped all-gather/broadcast into the following `batchmatmul` KERNEL operand.

Representative edge:

```text
mul -> ReStickifyOpHBM -> batchmatmul
sdsc_1.json -> sdsc_2.json -> sdsc_3.json
```

Producer split:

```text
mul: {mb:4, x:8, out:1}
output: LX OUTPUT [out,x,mb], stick out
```

Consumer split:

```text
batchmatmul: {x:4, mb:8, out:1, in:1}
KERNEL operand expects renamed/layout-transformed view
```

Why scatter is insufficient:

- each value does not have one destination owner;
- each batch-local group needs producer chunks split over BMM `out` to be visible to consumer cores split over BMM `mb`;
- the handoff also changes stick/layout from pointwise output form into BMM KERNEL form.

Artifacts:

- `layout_restickify_gap/flash_layout_restickify_gap.md`
- `layout_restickify_gap/representative_edge.json`
- `layout_restickify_gap/sdsc_triplet_snippets.json`
