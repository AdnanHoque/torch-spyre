# Comms Collectives Round 4 - 2026-07-01

This checkpoint records two pieces of progress:

- latest `test_flash.py` SDSC before/after with torch-spyre main versus
  PR2939 + Deeptools PR4408;
- compact DLDSC/STCDP progress on local LE128 assemble/extract around whole
  stick ring movement.

## Flash Attention SDSC Comparison

Script:

```text
repo: github.ibm.com/aviros/test-spyre-scripts
commit: 04f813edb0e3902d248600cede951a6b210fc71d
test_flash.py blob: 970123d788bd112b0f713103d0a5298122fff2f9
test_flash.py sha256: f8074b5b32910b3419a1cdd57f68fd65f761cdecf53f704d045c28c7423c385f
```

Runs:

| variant | torch | deeptools | run | result |
|---|---|---|---|---|
| baseline | `7cef216af0f53998741a6c27164e25798a5839d8` | `/opt/ibm/spyre/deeptools/bin/dxp_standalone` | `baseline_noh2d_20260701_040758` | compile/SDSC probe passed |
| optimized | local merge `99ac46a9fa1eb1b168534a44e1aee15b546d6dc3` = main `7cef216` + PR2939 `d56dc699035c317aa7c2db3f0eda944ce6ebd1bd` | PR4408 DXP at `/home/adnan/codex-isolated/pr4408_perf_check_20260630_231905/deeptools/build-dxp-pr4408/dxp/dxp_standalone` | `optimized_noh2d_20260701_041326` | SDSCs generated; probe then failed final correctness assert because this was no-H2D |

Important caveat:

The actual unpatched script still timed out before SDSC generation during
runtime/H2D setup in both lanes.  The comparison here uses the established
no-H2D compile probe to answer the SDSC question, not runtime correctness or
performance.

### SDSC Counts

| metric | baseline | optimized |
|---|---:|---:|
| SDSC JSON files | 550 | 550 |
| SuperDSC bundles | 3 | 3 |
| `ReStickifyOpHBM` rows | 32 | 32 |
| tensor components in HBM | 781 | 781 |
| tensor components in LX | 704 | 704 |
| relayout metadata keys | 0 | 0 |
| files with `lx_residency_core_id_to_wk_slice` | 0 | 0 |

Op-class counts are also identical:

| op class | count |
|---|---:|
| `mul` | 128 |
| `add` | 96 |
| `batchmatmul` | 64 |
| `exp` | 64 |
| `sub` | 64 |
| `max` | 34 |
| `maximum` | 32 |
| `sum` | 32 |
| `ReStickifyOpHBM` | 32 |
| `identity` | 3 |
| `realdiv` | 1 |

Interpretation:

PR2939 scatter relayout does not fire for this latest `test_flash.py` lowering.
The optimized run does not remove any of the 32 `ReStickifyOpHBM` rows and does
not add LX relayout metadata.  This is different from the Granite-block scatter
cases where PR2939 did classify resident scatter edges.  For this flash script,
the remaining HBM traffic appears to be layout/restickify or operand-movement
work outside PR1 scatter's class.

Artifacts:

```text
flash_attention/flash_baseline_summary.json
flash_attention/flash_optimized_summary.json
flash_attention/flash_opclass_summary.json
flash_attention/flash_baseline_sdsc_table.csv
flash_attention/flash_baseline_sdsc_table.md
flash_attention/flash_optimized_sdsc_table.csv
flash_attention/flash_optimized_sdsc_table.md
flash_attention/flash_attention_latest_noh2d_sdsc_20260701.tgz
```

The tarball contains compact run logs and manifests, not the full 99 MB / 87 MB
SDSC caches.

## DLDSC Compact Path

Two artifacts in this directory capture the latest compact-path progression:

```text
dldsc_local_ring/local_le128_ring_sequence_20260701_032217.tgz
dldsc_local_ring/local_le128_ring_sequence_multiaddr_burst_20260701_041424.tgz
```

Progression:

1. `local_le128_ring_sequence_20260701_032217`
   - emitted local `LE128BTRANSFER` assemble/extract nodes around whole-stick
     `ringdatatransfer`;
   - focused unit tests passed;
   - replay advanced to `addr_vec.size() == 1` in `stcdpOp.cpp`.

2. `local_le128_ring_sequence_multiaddr_burst_20260701_041424`
   - allowed folded/multi-address polarity collection when all addresses
     collapse to one polarity;
   - disabled pure-ring burst metadata for the local-staged ring loop;
   - replay advanced past both prior blockers.

Current first DLDSC blocker:

```text
DtException: is_lxunit || is_l0unit
dcc/src/Conversion/PCFGToDataflowIR/PCFGToDataflowIR.cpp:2409
```

Reason:

The generated dataop now contains `le128btransfer` nodes such as
`c0-l3lu-localExtractAfterRing-0-7` under an L3LU schedule.  Deeptools'
`createLE128DataTransferOp` currently accepts LE128 transfer nodes only under
LX/L0 units.  The next backend step is to model this local extract/assemble as
an LX/L0 unit adjacent to the ring load/store, or extend the conversion rule to
legally bridge the local LE128 node attached to the L3 ring unit.

## Current Read

The latest evidence strengthens the communication taxonomy:

- PR1 scatter remains a narrow resident ownership-permutation feature.
- Latest flash attention does not expose that class to PR2939.
- Attention/Granite remaining spills need at least:
  - layout/restickify activation movement;
  - operand gather/all-gather/broadcast;
  - local partial-stick assemble/extract around whole-stick ring movement;
  - later, reduction/all-reduction semantics for true accumulating edges.

The compact DLDSC path is now the more promising production direction for
these non-scatter classes because it avoids expanded frontend movement lists
and lets backend synthesize legal ring plus local movement.
