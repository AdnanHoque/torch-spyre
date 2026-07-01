# Comms Collectives Round 5 - 2026-07-01

This checkpoint starts from the round4 finding:

- latest `test_flash.py` does not expose PR1 resident scatter;
- compact DLDSC/STCDP local-staged movement now reaches DCC lowering and
  currently spins in `AgenToSentientLoweringPass::fuseLoadOrStoreChainOps`.

## DLDSC Compact Path: DCC L3 LE128 Prototype

Artifact:

```text
dldsc_dcc_spin/local_le128_ring_sequence_multiaddr_burst_20260701_043805_dcc_l3le128.tgz
```

What advanced:

- The guarded prototype in `PCFGToDataflowIR.cpp` allowed local-staged L3
  `le128btransfer` helpers named `localAssembleBeforeRing` /
  `localExtractAfterRing` when:
  - the transfer is LX-to-LX;
  - `isPartStick` is set;
  - it belongs to the local-staged ring path.
- LE128 conversion now uses side-aware `srcByteOffset` / `dstByteOffset`
  metadata rather than the legacy shared `Offset`.
- Safe rebuild passed.
- `dcg_unit_test` passed: 11/11.
- DXP replay advanced past the prior blocker:

```text
DtException: is_lxunit || is_l0unit
dcc/src/Conversion/PCFGToDataflowIR/PCFGToDataflowIR.cpp:2409
```

Current first blocker:

```text
AgenToSentientLoweringPass::fuseLoadOrStoreChainOps(...)
inside DCC::DFtoProgIR()
```

The replay dumped all 8 dataops and then became CPU-bound in DCC lowering. The
artifact includes `gdb_bt.txt`; the run was terminated with SIGTERM, so
`result.txt` records `exit_code=143`.

Interpretation:

The compact path has now reached a lower-level DCC optimization/lowering issue.
The frontend communication contract and DCG dataop construction are no longer
the immediate blockers for this focused repro.  The next patch should identify
the load/store chain shape created by local-staged LE128 helpers and either:

- lower the helper in a form that the existing chain-fusion pass accepts; or
- guard the chain-fusion pass away from this local-staged helper pattern.

## Flash Attention Follow-Up

Round4 showed that latest `test_flash.py` has identical SDSC counts before and
after PR2939 + Deeptools PR4408:

| metric | baseline | optimized |
|---|---:|---:|
| SDSC JSON files | 550 | 550 |
| `ReStickifyOpHBM` rows | 32 | 32 |
| HBM tensor components | 781 | 781 |
| LX tensor components | 704 | 704 |
| relayout metadata | 0 | 0 |

The current follow-up is classifying those 32 `ReStickifyOpHBM` rows into
communication classes so we can state exactly which primitive is missing for
flash attention.  The hypothesis before that classification is:

```text
latest flash is not resident scatter;
it is layout/restickify activation movement and/or operand gather/broadcast.
```

Round5 classification artifact:

```text
flash_attention_classification/flash_classification_20260701/
```

Result:

- Baseline and optimized runs still have the same 32 `ReStickifyOpHBM` rows.
- PR2939 + Deeptools PR4408 removed 0 flash HBM spills for this test.
- Every row is an activation handoff:

```text
mul -> ReStickifyOpHBM -> batchmatmul
```

- The restickify input is LX `OUTPUT` with layout `[out,x,mb]`.
- The restickify output is HBM `KERNEL` with layout `[x,out,mb]`.
- The following `batchmatmul` uses a different reduction/core division.
- This is not a plain resident scatter case. It needs an LX-resident
  layout/restickify handoff into `batchmatmul`, or consumer support for that
  LX-resident activation layout.

The generated optimized SDSCs also reported no relayout metadata and no
`lx_residency_core_id_to_wk_slice`, so the PR1 scatter contract is not firing
for this flash pattern.

## DLDSC Compact Path: No-Op LE128 Helper Skip

Artifact:

```text
dldsc_cdx_checkpoint_skip_noop/dldsc_cdx_checkpoint_20260701_0523_skip_noop_local_le128/
```

What advanced:

- The local-staged LE128 helper generator was producing LX-to-LX helper ranges
  whose source and destination were identical.
- Those no-op helpers inflated DCC lowering: the prior run had 7,168 local
  helper nodes and 48,128 no-op helper entries.
- A narrow prototype in `dcg/dcg_fe/pcfg_gen/stcdpOp.cpp` now skips local
  helper ranges where `srcIntraStickOffset == dstIntraStickOffset`.
- The focused rebuild passed.
- Replay advanced past the prior
  `AgenToSentientLoweringPass::fuseLoadOrStoreChainOps` blocker.

Current first blocker:

```text
CFGSimplificationSentientLevelPass
  -> CFGSSentientLevelConditionalTree::compute()
```

The next backend investigation should count sentient conditional ops per
program unit after Agen lowering and before CFG simplification, then determine
whether the repeated sub-stick ring branch pattern needs batching, guarding, or
a simpler lowering shape.

## Explicit Grouped-Range Path

The explicit grouped-remap lane is still pending a checkpoint.  Last known
state from round3:

- grouped schema compressed `sdsc_10 Tensor1` from 2,097,152 modeled moves to
  128 grouped rows;
- semantic checker passed;
- DXP diagnostic DT rows materialized;
- DCC stitching failed with:

```text
DtException: unit already set for associated schedule step
dcc/src/Stitcher/ModuleStitcher.cpp:279
```

That path remains useful as a diagnostic carrier, but the compact DLDSC path is
currently advancing faster toward a production-shaped backend realization.
