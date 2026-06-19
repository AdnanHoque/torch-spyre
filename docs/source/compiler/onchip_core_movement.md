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
`swiglu-ws-dxp`. The original mixed-import enabling commit was `aa101d41e6`;
the latest physicalizer probe seen in the pod was `aeaaa65949`:

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
`d0=16` by `d1={1,4,8}` out-stick groups. The current open question is address
units. The corrected 64-byte-unit artifacts line up with some pre-DCG JSON
patterns, but DeepTools' STCDP LX PCFG/codegen path appears to treat LX
`startAddr` as bytes.

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

- After changing offsets to 64-byte address units, the generated `.phys.json`
  matches the original logical address scale, for example source `d0=16,d1=0`
  uses `startAddr=32` and destination `d1=50` uses `base+1600`. A later code
  audit makes this interpretation suspicious for the final STCDP LX PCFG path,
  because that path copies LX placement addresses into PCFG/LRF fields without
  dividing by 64.
- Correct-address variants with `d1` group sizes 4 and 8 both compile and then
  hit the runtime compute-CB hardware error.
- Routing the row through conservative STCDP PCFG generation with
  `DXP_ONCHIP_MOVE_FORCE_NO_OPT_STCDP=1` still hits the same hardware error.

Artifact diff from the current probe:

| Run | Physical pieces | Final `p/c/dtTable_` | `useUnicast` | `collapseFactor` | Result |
| --- | ---: | ---: | ---: | --- | --- |
| old byte-offset physicalizer | 1792 | 1792 / 1792 / 1792 | 1 | 1536 x `1`, 256 x `4` | runs, value-wrong |
| corrected 64-byte units, group 4 | 1792 | 1792 / 1792 / 1792 | 1 | all `4` | runtime compute-CB fault |
| corrected 64-byte units, group 8 | 1024 | 1024 / 1024 / 1024 | 1 | all `4` | runtime compute-CB fault |
| corrected group 4 plus no-opt PCFG | 1792 | 1792 / 1792 / 1792 | 1 | all `4` | runtime compute-CB fault |

The address examples show the ambiguity. The old byte-offset run used wrong
piece extents but likely used the address scale expected by STCDP LX PCFG. The
64-byte-unit runs fixed physical extents but may have underscaled LX addresses:

```text
source d0=16,d1=0: old startAddr=2048, corrected startAddr=32
dest   d0=0,d1=50: old startAddr=1150976, corrected startAddr=1050176
source d0=448,d1=70: old startAddr=335872, corrected startAddr=5248
dest   d0=448,d1=70: old startAddr=1191936, corrected startAddr=1050816
```

The strongest remaining signal is that corrected compact addresses push DCG into
a different STCDP program shape. The old, wrong byte-offset run spread addresses
far enough to execute but read the wrong values. The corrected group-4 run emits
much more `LX_LDSTI` traffic in `smc.txt`; the corrected group-8 run avoids that
specific explosion but still faults.

DeepTools code audit points at this experiment order:

1. Test the physicalizer with corrected `d0=16`/grouped pieces but byte
   `startAddr` offsets.
2. If byte addresses run but remain value-wrong, test the STCDP LX-local stick
   adjustment path. In `stcdpOp.cpp`, the LXLU0/LXSU0 path appears to use the
   opposite LDS for `makeStickLevelAdjustments(...)`, which is suspicious for
   non-IJ `{mb,out}` relayouts.
3. If still unstable, disable local LX burst as a separate diagnostic. The
   force-no-opt switch does not de-risk this path; it only selects between the
   optimized and non-optimized L3 STCDP paths.

### Minimal STCDP Legality Probe

`tools/deeptools_onchip_move_physicalizer_filter_probe.patch` is an incremental
DeepTools patch for `swiglu-ws-dxp`. It adds an address-unit toggle, address
dumping, and diagnostic filters to the existing physicalizer:

```text
DXP_ONCHIP_MOVE_PHYSICALIZE_LX_BYTE_ADDRS
DXP_ONCHIP_MOVE_PHYSICALIZE_DUMP_ADDRS
DXP_ONCHIP_MOVE_PHYSICALIZE_MAX_PIECES
DXP_ONCHIP_MOVE_PHYSICALIZE_SRC_CORE
DXP_ONCHIP_MOVE_PHYSICALIZE_DST_CORE
DXP_ONCHIP_MOVE_PHYSICALIZE_MB_START
DXP_ONCHIP_MOVE_PHYSICALIZE_OUT_STICK_START
```

Run the next probe in this order, keeping the seeded SwiGLU input fixed:

```bash
cd /tmp/deeptools-swiglu-ws
git apply /tmp/torch-spyre-main/tools/deeptools_onchip_move_physicalizer_filter_probe.patch
# rebuild dxp_standalone with the same build command used for swiglu-ws-dxp

export SPYRE_ONCHIP_MOVE_PLANNER=1
export SPYRE_ONCHIP_MOVE_REALIZE=1
export DXP_ONCHIP_MOVE_PHYSICALIZE=1
export DXP_ONCHIP_MOVE_PHYSICALIZE_OUT_STICK_GROUP=4
export DXP_ONCHIP_MOVE_FORCE_NO_OPT_STCDP=1
export DXP_ONCHIP_MOVE_PHYSICALIZE_LX_BYTE_ADDRS=1
export DXP_ONCHIP_MOVE_PHYSICALIZE_DUMP_ADDRS=1

# First run the full physicalized tensor with byte addresses.
# If it runs, compare against baseline correctness.

# If it faults, keep byte addresses enabled and shrink to a minimal legality
# probe.
for n in 1 2 4 8 16 32 64; do
  export DXP_ONCHIP_MOVE_PHYSICALIZE_MAX_PIECES=$n
  # run the same seeded 1x512x4096 SwiGLU correctness command
done
```

The first transfer in the existing corrected group-4 artifact is:

```text
src_core=11 -> dst_core=28
coord: d0/mb=448, d1/out_stick=70, d2=0, d3=0
size:  d0=16, d1=4, d2=1, d3=64
64B-unit addresses: src=5248, dst=1050816
```

To isolate that exact move with byte addresses:

```bash
export DXP_ONCHIP_MOVE_PHYSICALIZE_MAX_PIECES=1
export DXP_ONCHIP_MOVE_PHYSICALIZE_SRC_CORE=11
export DXP_ONCHIP_MOVE_PHYSICALIZE_DST_CORE=28
export DXP_ONCHIP_MOVE_PHYSICALIZE_MB_START=448
export DXP_ONCHIP_MOVE_PHYSICALIZE_OUT_STICK_START=70
# run the same seeded 1x512x4096 SwiGLU correctness command
```

Interpretation:

- Full byte-address run completes and improves correctness: the 64-byte-unit
  address interpretation was wrong for this backend path; continue with
  value-correctness debugging from that run.
- Full byte-address run faults but `MAX_PIECES=1` runs: binary-search piece
  count, producer core, destination core, `mb`, and `out-stick` with the new
  filters.
- `MAX_PIECES=1` faults even with byte addresses: inspect the single generated
  transfer descriptor and STCDP microprogram. The minimal physical LX transfer
  is illegal or miscompiled.
- Small `MAX_PIECES` runs but a larger one faults: binary-search piece count,
  producer core, destination core, `mb`, and `out-stick` with the new filters.
- Any filtered run that completes will be value-wrong unless it transfers the
  full tensor. Use it only as a hardware/codegen legality probe, not as a
  correctness result.

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

Run the minimal STCDP legality probe above. That is now the shortest path to a
working conclusion:

- If byte-addressed physical pieces cannot run even for one piece, fix the STCDP
  LX descriptor or microprogram generation for arbitrary non-IJ dimensions
  before touching the torch-spyre planner.
- If one piece runs, binary-search the smallest failing set and decide whether
  the issue is transaction count, GTR grouping, route pressure, sync placement,
  or a specific producer/destination core pair.

Only after byte-addressed or otherwise correctly scaled physical pieces run
without a hardware fault should we move back to full tensor value-correctness
and then performance or warp-specialized SwiGLU scheduling.
