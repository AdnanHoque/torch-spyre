# Stage 72: InputFetchNeighbor Adaptation Probe

## Goal

Continue the Stage 71 `dcg_inpfetch_standalone` probe now that pod access is restored. The immediate question was whether a Torch-Spyre producer/restickify/consumer bundle can be adapted into the Deeptools `InputFetchNeighbor` contract far enough to generate an LX-to-LX neighbor-fetch data op.

## What Changed

`tools/restickify_input_fetch_neighbor_probe.py` now has a probe-only mode:

```sh
--adapt-scheduled-lx-neighbor
```

That mode:

1. Runs `L3DlOpsScheduler_standalone` on the staged producer and consumer SDSCs.
2. Retags the first consumer input as `INPUT`, because `InputFetchNeighbor` requires `primaryDsInfo_[INPUT]`.
3. Marks the staged producer/consumer labeled DS as LX-pinned by setting HBM `isPresent=0` while keeping the HBM memOrg key present.
4. Populates `coreStateInit_` from the scheduler's per-core LX allocation nodes.
5. Copies DSC2 staging dimensions from `dataStageParam_` into the aggregate DSC fields that the input-fetch path reads directly.

This is not a production lowering. It is a visibility harness for testing the Deeptools contract.

## Pod Command

```sh
cd /tmp/torch-spyre-stage2
python tools/restickify_input_fetch_neighbor_probe.py \
  --code-dir /tmp/restickify-input-fetch-capture/kernel_code/computed_transpose_adds_then_matmul_2048/0001_sdsc_fused_add_t_0 \
  --output-dir /tmp/restickify-input-fetch-neighbor-adapted \
  --adapt-scheduled-lx-neighbor \
  --run \
  --senprog
```

## Result

The probe now gets past the earlier blockers:

- missing consumer `INPUT`
- non-LX-pinned labeled DS
- missing `coreStateInit_`
- missing aggregate `CoreD_`/`B_`/`CoreletD_`

The current blocker is deeper and semantic:

```text
DtException: op->outSP_.at(mainOutSPIdx).dimToStartCordinate.count("i"),
file /project_src/deeptools/dcg/dcg_fe/pcfg_gen/inputNeighFetchOp.cpp line 1644
```

This means the existing `InputFetchNeighbor` implementation expects `i`/`j` image-style subpiece coordinates when ordering neighbor-fetch pieces. The Torch-Spyre restickify case under test is a pointwise tensor with logical dimensions `mb/out`, so it reaches the coordinate-ordering path without the `i` and `j` fields that the Deeptools code asserts.

## Interpretation

This is progress, but it is not yet proof of a usable Torch-Spyre restickify LX-to-LX path.

What we have shown:

- Deeptools has a real `InputFetchNeighbor` path that is intended to generate LX/ring movement.
- Torch-Spyre SDSCs can be mechanically adapted through several of that path's contract checks.
- The next incompatibility is not a missing metadata field; it is that the current Deeptools path appears specialized around `i/j` coordinate ordering.

What remains unproven:

- That this path can directly lower the `mb/out` restickify cases emitted by Torch-Spyre.
- That the resulting generated program avoids HBM for the actual restickify replacement.

## Next Options

1. Build a tiny `i/j`-shaped fixture that satisfies the existing `InputFetchNeighbor` assumptions and verify the generated `senprog.txt` has no `HBM` plus ring-facing `L3LU/L3SU` traffic. Stage 73 corrected the expected signature: cross-core LX-to-LX RIU movement can show up as `L3LU/L3SU`, because those are the ring-facing units.
2. Patch a local Deeptools experiment to generalize `inputNeighborFetchL3LUSubpieceOrder` away from hard-coded `i/j` ordering.
3. Return to the Torch-Spyre-side mapping-only Stage 3B path, since it remains the smallest production-shaped optimization.

Option 1 is the safest next proof step because it tests whether the hardware/compiler path works at all before changing Deeptools semantics.
