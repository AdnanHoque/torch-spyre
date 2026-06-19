# Experimental On-Chip Core Movement

Status: experimental on branch `swiglu-ws`.

## Goal

SwiGLU exposes a common layout conflict:

- Matmul wants a two-dimensional split such as `{mb:4, out:8}`.
- Pointwise SiLU and multiply want a pure-M split across 32 cores.
- Existing LX planning handles same-core persistence, but not general
  cross-core LX-to-LX reshards when producer and consumer `PerCoreView`s differ.

The prototype adds a planner and optional mixed-SuperDSC carrier for those
same-stick, mismatched-view edges. SwiGLU is the first benchmark, not a special
case.

## Torch-Spyre Prototype

The current branch adds:

- `SPYRE_ONCHIP_MOVE_PLANNER` to classify producer-to-consumer edges.
- `SPYRE_ONCHIP_MOVE_REALIZE` to emit an experimental mixed SuperDSC carrier.
- `SPYRE_ONCHIP_MOVE_DEBUG_DIR` and `SPYRE_ONCHIP_MOVE_JSONL` for artifact
  inspection.
- `SPYRE_SWIGLU_WARPSPEC_AUDIT` and `SPYRE_SWIGLU_WARPSPEC_AUDIT_JSONL` to
  record whether SwiGLU lowers as epilogue work or separate pointwise ops.

The planner reuses `PerCoreView` metadata after work distribution. It leaves
same-view edges to the existing LX planner and plans only mismatched same-stick
LX edges. Unsupported cases are skipped with explicit fallback reasons.

The v1 carrier is mixed SuperDSC with an `STCDPOpLx` data-op row inserted before
the consumer compute row. The producer remains a standalone SDSC whose output is
pinned to LX. This avoids DeepTools' current assumption that multiple DL rows in
one SuperDSC are compatible pieces of the same operation. The carrier is a proof
carrier, not necessarily the final interface.

The carrier now records packed per-core source and destination byte offsets for
each movement cell. Realization rejects overlapping or over-capacity LX regions
before emitting mixed SDSC JSON.

## Validation Snapshot

On the `1x512x4096` MLP/SwiGLU benchmark, fresh compilation with planner and
realization enabled produced:

- 1 realized movement edge in the emitted mixed SDSC.
- 256 common-refinement cells for the `{mb:4, out:8} -> pure-M` reshard.
- 13,107,200 bytes moved.
- Producer and consumer LX regions of 409,600 bytes each, placed at
  non-overlapping bases.
- The down-projection edge is skipped as `consumer-duplicate-owner`, which is
  intentional for v1.

The current SwiGLU lowering is not a matmul epilogue. It lowers as:

```text
batchmatmul, batchmatmul, neg, exp, add, realdiv, mul
```

That makes warp specialization a plausible follow-on after movement is
value-correct: PT-heavy matmul work and SFP-heavy pointwise work are separate
operations in the generated SuperDSC sequence.

## Backend Status

With realization enabled, Torch-Spyre now emits:

- a standalone producer SDSC with the selected output pinned to LX;
- one mixed consumer SDSC containing one `STCDPOpLx` in `datadscs_`;
- one DL consumer row in `dscs_`;
- `coreIdToDscSchedule` entries that run the data-op before the consumer row.

The clean DeepTools/DXP worktree `/tmp/deeptools-swiglu-ws` has branch
`swiglu-ws-dxp` commit `aa101d41e6`:

```text
Allow scheduled mixed data-op SDSCs
```

It builds `dxp_standalone` successfully with `DT_USE_DCC_DDC=ON`, reusing
`/home/adnan-cdx/dt-inductor-mixed/llvm-project` and
`/home/adnan-cdx/dt-inductor-mixed/build/llvm`.

That patch clears the original import rejection:

```text
DtException: Datadsc not allowed, use dldsc
```

The earlier producer-plus-dataop-plus-consumer carrier exposed an unsupported
DeepTools scheduler shape:

```text
DtException: isValidDimParam(...) && "Expect the chunk dimension has a valid parameter value."
file dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp line 1200
```

That failure happens because `L3DlOpsScheduler` builds chunk dimensions from DSC
0 and applies them to every DL row. A block-only DDC bypass is not enough:

```text
DtException: Missing below-lx schedule insert block
file ddc/ddcv1.cpp line 2241
```

DDC needs the loop, allocation, transfer, and sync metadata normally created by
the L3 scheduler.

The consumer-only carrier gets through direct DXP codegen:

```text
dxp_standalone --bundle -d .../sdsc_fused_bmm_mul_silu_0_9d148k53
```

The generated bundle contains `2_OnChipMoveMixedSTCDP` with one `neg` DL row and
one STCDP data-op. Direct DXP exits successfully.

Runtime execution is not yet value-correct. The current blocker is physical LX
layout: after DCC lowers the producer batchmatmul, the visible output is carried
through generated LX/PERF labels and corelet/chunk loop offsets. The current
planner emits STCDP cells from the logical per-core view, so it can read the
wrong physical LX subpieces even though the JSON compiles.

### Physicalizer Probe

DeepTools branch `swiglu-ws-dxp` now has an env-gated post-DDC/pre-DCG
physicalizer:

```text
DXP_ONCHIP_MOVE_PHYSICALIZE=1
DXP_ONCHIP_MOVE_PHYSICALIZE_OUT_STICK_GROUP=4
DXP_ONCHIP_MOVE_FORCE_NO_OPT_STCDP=1
```

The physicalizer rewrites the 256 logical STCDP cells into grouped physical
pieces before DCG. It uses the selected producer compute output LDS and consumer
compute input LDS to derive LX bases, then rewrites each piece as
`d0=16` by `d1={1,4,8}` out-stick groups. The important address discovery is
that DXP `startAddr` values are in 64-byte LX address units, not raw bytes.

Validation results on `mlp 1x512x4096` with identical seeded inputs:

- Baseline torch-spyre without on-chip movement is bit-stable across separate
  runs.
- The first physicalizer version used byte offsets. It compiled and ran without
  the compute-CB hardware error, but was value-wrong:

  ```text
  max_abs 0.00012183189392089844
  mean_abs 3.706010465975851e-05
  nonzero_diff 2094939 of 2097152
  allclose_atol_0.0001_rtol_0.01 False
  ```

- After fixing offsets to 64-byte address units, the generated `.phys.json`
  matches the original logical address scale, for example source `d0=16,d1=0`
  uses `startAddr=32` and destination `d1=50` uses `base+1600`.
- Correct-address variants with `d1` group sizes 4 and 8 both compile and then
  hit the runtime compute-CB hardware error.
- Routing the row through conservative STCDP PCFG generation with
  `DXP_ONCHIP_MOVE_FORCE_NO_OPT_STCDP=1` still hits the same hardware error.

Conclusion: mixed import and physical address derivation are no longer the only
blockers. The remaining blocker is in the generated STCDP LX transfer
descriptor/program for this non-IJ `{mb,out}` relayout. We do not yet have a
value-correct running prototype.

### InputFetchNeighbor Probe

The existing DeepTools `runDcgForInputFetchNeighbor(main, pre)` path was tested
as a possible backend hook because it sees both the consumer DL row and previous
producer DL row.

That path is not a drop-in carrier for the current SwiGLU reshards:

- Raw imported torch-spyre pointwise rows use `computeOp_.inputLabeledDs` to
  identify operands; the input tensor can still have `DsTypes::OUTPUT`.
  `InputFetchNeighbor` assumed a distinct `DsTypes::INPUT` primary.
- It checked all consumer tensors and all producer `OUTPUT` tensors for LX-only
  placement. Modern rows may contain HBM+LX outputs or internal PT/PE
  temporaries; the relevant tensors are the consumer compute input and producer
  compute output.
- Post-DDC artifacts no longer populate the old `coreStateInit_` LBR list for
  these tensors. The modern address source is the LX allocation node reachable
  through `DesignSpaceConfig::getAddress(...)`.
- Under the L3/LX planner, `CoreD_` and `CoreletD_` can be unset in exported
  JSON while equivalent data lives in `dataStageParam_`.
- After compatibility patches for those issues, the helper reached its real
  algorithmic limit: it still assumes IJ-style spatial coordinates and fails on
  non-IJ SwiGLU movement domains such as `{mb, out}`:

```text
DtException: op->outSP_.at(mainOutSPIdx).dimToStartCordinate.count("i")
file dcg/dcg_fe/pcfg_gen/inputNeighFetchOp.cpp line 986
```

Conclusion: `InputFetchNeighbor` is a useful reference for where a backend hook
should live and how DCG wants producer/consumer movement expressed, but it needs
a generalized non-IJ relayout path before it can serve the scalable LX-to-LX
movement goal. Continuing to patch the old helper is likely to produce a narrow
SwiGLU workaround rather than the desired general movement planner.

## Next Step

Close the logical-to-physical LX gap. The next implementation should either:

- add a backend physical-piece rewriter after DDC and before DCG, using the
  selected producer output and consumer input `LabeledDs` plus allocation
  metadata; or
- generalize the `InputFetchNeighbor` algorithm to support arbitrary layout
  dimensions, not just IJ/spatial movement, then route the mixed carrier through
  that backend hook.

After the STCDP pieces match physical LX layout, run patterned value-correctness
tests before evaluating performance or warp-specialization scheduling.
