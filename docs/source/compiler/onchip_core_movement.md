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
the consumer compute row. This is a proof carrier, not necessarily the final
interface.

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

With realization enabled, Torch-Spyre emits a mixed SuperDSC containing:

- one `STCDPOpLx` in `datadscs_`;
- two DL compute rows in `dscs_`;
- `coreIdToDscSchedule` entries for all participating cores.

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

The current backend blockers are later in DXP/DDC:

```text
DtException: isValidDimParam(...) && "Expect the chunk dimension has a valid parameter value."
file dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp line 1200
```

With `DXP_SKIP_MIXED_MULTI_DL_L3_SCHEDULER=1`, the next blocker is:

```text
DtException: Missing below-lx schedule insert block
file ddc/ddcv1.cpp line 2241
```

## Next Step

Fix or bypass the mixed multi-DL scheduler path in a principled way, then teach
DDC how to create the below-LX schedule insert block for scheduled mixed
DL/data-op SDSCs. After DXP emits runnable code for the mixed carrier, run
patterned value-correctness tests before evaluating performance or warp-
specialization scheduling.
