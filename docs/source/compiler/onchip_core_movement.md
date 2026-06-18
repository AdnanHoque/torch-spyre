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

## Validation Snapshot

On the `1x512x4096` MLP/SwiGLU benchmark, planner-only compilation produced:

- 3 planned movement edges.
- 5 skipped edges.
- Each planned `{mb:4, out:8} -> pure-M` edge has 256 common-refinement cells.
- Each planned edge moves 13,107,200 bytes.
- The down-projection edge is skipped as `consumer-duplicate-owner`, which is
  intentional for v1.

The current SwiGLU lowering is not a matmul epilogue. It lowers as:

```text
batchmatmul, batchmatmul, neg, exp, add, realdiv, mul
```

That makes warp specialization a plausible follow-on after movement is
value-correct: PT-heavy matmul work and SFP-heavy pointwise work are separate
operations in the generated SuperDSC sequence.

## Current Backend Blocker

With realization enabled, Torch-Spyre emits a mixed SuperDSC containing:

- one `STCDPOpLx` in `datadscs_`;
- two DL compute rows in `dscs_`;
- `coreIdToDscSchedule` entries for all participating cores.

The installed DeepTools/DXP in the validation pod rejects that form during
import:

```text
DtException: Datadsc not allowed, use dldsc
```

The required backend change is to allow `datadscs_` only for scheduled mixed
SDSCs that also contain DL `dscs_` and a non-empty `coreIdToDscSchedule`, then
route those SDSCs through `runDcgForDataOpsDlOps`.

## Next Step

Build and validate a DeepTools/DXP with the scheduled mixed-SDSC import path.
After DXP accepts the emitted `STCDPOpLx`, run value-correctness tests with
patterned tensors before evaluating performance or warp-specialization
scheduling.
