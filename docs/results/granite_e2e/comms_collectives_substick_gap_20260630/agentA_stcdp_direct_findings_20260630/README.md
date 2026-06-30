# Agent A: Direct STCDPOpLx Findings For Attention sdsc_10

Root tree: `/home/adnan-cdx/codex-isolated/comms_collectives_stcdp_agent_20260630_190747`

Focused bundle:
`runs/stcdp_agent_validation_20260630_191625/dxp_replay_fresh_attention_subpiece_20260630_193832/bundle_input`

Important file: `sdsc_10.json`, root op `10_batchmatmul`.

## Case Shape

`sdsc_10` consumes Tensor1 from LX. The consumer compute split is:

```json
{"x":16,"mb":1,"out":2,"in":1}
```

Tensor1's LX allocation/distribution is still producer-shaped:

```text
core 0  -> out 0
core 1  -> out 1
...
core 31 -> out 31
```

The KERNEL/Tensor1 stick dimension is `out` with stick size 64. The consumer wants two large `out` slices, while producer ownership is fragmented across many `out` slices. This is therefore a partial-stick fragment assembly problem: multiple producer fragments must be placed into different offsets inside a destination slice/stick.

## Conclusion

The current STCDPOpLx direct data-op descriptor cannot fully represent this movement without schema/codegen extension. Existing structures identify producer/consumer subpieces and core IDs, but do not carry source intra-stick offset, destination intra-stick offset, or valid element count for a fragment inside a stick.

The direct path therefore has no unambiguous way to say:

```text
copy producer core P's out-fragment [src offset/count]
into consumer core C's destination out-stick at [dst intra-stick offset]
```

Simply relaxing stick-size assertions is not correct: the next failures are coverage/addressing failures, including an invalid negative immediate.

## Code Paths That Need Extension

- `dsc/dataOpDsc.h`
  - `transferInfo` needs fragment metadata, e.g. transfer dim, source intra-stick element offset, destination intra-stick element offset, and element count.
- `dcg/dcg_fe/pcfg_gen/stcdpOp.cpp`
  - `setPlacementInfoSubPiece()` currently stick-adjusts offsets with `ceil(diff / stickDim)` and loses `diff % stickDim`.
  - `insertSubPieces()` / overlap logic needs to preserve the exact intersection range rather than only piece-level starts/sizes.
  - `computeInnerLoopCollapseFactor()` assumes relevant outer dims are at least one stick; partial-stick transfer needs either no collapse or fragment-aware collapse.
  - STCDP unroll/codegen must consume fragment offsets to generate correct LX/L3 addresses or use a staged whole-stick assembly path.

## Small Safe Patch Produced

`grouped_destination_fix.diff` changes `dcg/dcg_fe/pcfg_gen/stcdpOp.cpp` so grouped L3LU destination placements select the start-address lane matching `coreID`, rather than always lane 0.

This is an independent addressing correctness fix for grouped STCDPOpLx destinations. It does not solve the partial-stick schema gap by itself.

## Replay Results Consulted

Original compact direct replay:

```bash
/home/adnan-cdx/codex-isolated/comms_collectives_stcdp_agent_20260630_190747/tools/dxp-split-wrapper-stcdp-agent/dxp_standalone \
  --bundle -d /home/adnan-cdx/codex-isolated/comms_collectives_stcdp_agent_20260630_190747/runs/stcdp_agent_validation_20260630_191625/dxp_replay_fresh_attention_subpiece_20260630_193832/bundle_input
```

Result: `RC=134`.

Failure:

```text
DtException: op->inpSP_.at(inpSPIdx).dimToSize_.at(dimNameOuter) >= stickDim,
file .../dcg/dcg_fe/pcfg_gen/stcdpOp.cpp line 4374
```

Noncompact stage 8/16 replays:

```text
DtException: 0, file .../dcg/dcg_fe/pcfg_gen/stcdpOp.cpp line 440
```

Noncompact stage 32 replay:

```text
Program verification failed for core 0 node 10_batchmatmul
Immediate value out of boundary in instruction:
LX_MODLRFIMM :: lrfimm:-2101120 src0:0
```

Compact stage 32 replay:

```text
rc=124 timeout
```

## Build Note

A clean separate worktree was created at:

`/home/adnan-cdx/codex-isolated/comms_collectives_stcdp_agent_20260630_190747/deeptools-agentA-direct`

A separate clean build was attempted, but CMake began cloning/building a fresh LLVM/MLIR tree. That build was stopped to avoid burning time; the diagnosis above uses the archived direct replay artifacts plus static code inspection.
