# Flash Layout-AllGather Restickify Checkpoint - 2026-07-01

This directory records the latest flash-attention communication-class checkpoint.
The workload is the `test_flash.py` attention kernel reduced to the same four-head
shape used in the earlier SDSC discussions:

- `B=1`, `H=4`, `Lq=4096`, `Lk=4096`, `D=128`
- Torch source: pod-local `ah/comms-collectives-current`
- Primary edge: computed activation `mul` -> `ReStickifyOpHBM` -> `batchmatmul`

## What Case This Is

This is not the PR1 scatter case. The selected edge is a layout-aware grouped
all-gather/restickify:

1. The producer activation is already in LX.
2. The inserted restickify changes the stick/layout interpretation needed by the
   downstream `batchmatmul` KERNEL operand.
3. The consumer wants each group of KERNEL cores to see replicated chunks from
   producer cores, so the movement is grouped all-gather, not 1:1 scatter.

In baseline/current behavior, this shows up as an HBM handoff:

```text
mul OUTPUT in LX -> ReStickifyOpHBM -> batchmatmul KERNEL
```

With the staged contract enabled, Torch can express the intended on-chip handoff:

```text
mul OUTPUT in LX -> ReStickifyOpLx-style LX view -> grouped all-gather -> batchmatmul KERNEL in LX
```

The backend lowering is not complete yet, so the staged run intentionally reaches
a DXP/runtime gap instead of a value-correct optimized execution.

## Artifacts

`baseline_no_staged_contract/` is the successful run before preserving staged
collective metadata after scatter-reservation failure. It compiles and runs, but
the interesting restickify-to-batchmatmul handoff is hidden behind HBM.

`staged_allgather_contract/` is the run after preserving metadata-only staged
collective classifications. It emits four `layout_allgather_restickify`
classifications on the relevant `batchmatmul` rows, and the paired restickify rows
show LX input and LX output allocations. Stock DXP then aborts with:

```text
std::out_of_range: map::at
```

`explicit_remap_probe/` contains the parallel explicit-remap prototype. It lowers
the same logical edge into concrete ranged movement:

- 4 groups
- 8 producer chunks per group
- 8 consumer replicas per group
- 256 logical transfers
- 131072 bytes per transfer
- 33554432 modeled movement bytes

Normal DXP import/routing accepts that synthetic explicit carrier, but senulator
realization still fails in the legacy program-correction path.

## Selected Files

- `flash_layout_allgather_summary.json`: compact machine-readable run summary.
- `baseline_no_staged_contract/sdsc_summary.csv`: selected baseline SDSC rows.
- `staged_allgather_contract/sdsc_summary.csv`: selected staged SDSC rows.
- `staged_allgather_contract/classifications.json`: extracted DLDSC contract.
- `explicit_remap_probe/layout_allgather_concrete_ranged_remap.json`: explicit
  ranged transfer list for the same communication class.
- `explicit_remap_probe/explicit_remap_next_step_20260701.md`: explicit-remap
  lane report.

## Current Read

PR1 scatter handles 1:1 producer-to-consumer LX relayout when the source tensor
and consumer operand have compatible stick/layout semantics. This flash edge is
broader: it combines layout restickification, dimension rename, and grouped
replication into a KERNEL operand.

The next backend work is therefore not another scatter tweak. It is a real
collective class:

1. Carry the DLDSC contract into the backend relayout/PerfDSC mutation point.
2. Convert the restickify to an LX-side layout transform.
3. Generate grouped `STCDPOpLx` transfers for the all-gather.
4. Bind the resulting LX-resident KERNEL view to the consumer `batchmatmul`.
5. Schedule movement before the consumer compute.

Weight restickifies are intentionally out of scope for this directory; those should be handled by offline weight prelayout/preload work.

## 2026-07-01 Follow-Up Findings

`restickify_lx_probe/` reruns the staged DLDSC contract after changing Torch to emit `ReStickifyOpLx` whenever the restickify input and output are both LX-resident. The generated SDSC rows do switch from `ReStickifyOpHBM` to `ReStickifyOpLx`, and the four consumer `batchmatmul` rows still carry `layout_allgather_restickify` metadata. Stock DXP still aborts in `Dxp::insertRelayoutSdsc(...)`, so the op-name mismatch is not the only blocker.

`restickify_lx_probe/dxp_gdb_bt.txt` captures the stock DXP stack trace. The abort is inside backend relayout insertion, not Python import or frontend metadata propagation.

`explicit_remap_probe/explicit_remap_senulator_failure_report_20260701.md` records the explicit-remap lane finding: the concrete ranged transfer SDSC imports/routes successfully, and the senulator failure is avoided with `DXP_ENABLE_COMPILE_TIME_CORRECTION=1`. That points to a runtime program-correction zero-flit/zero-layout issue rather than malformed explicit movement metadata.

