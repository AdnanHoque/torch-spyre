# DLDSC Agent Branch Assessment - 2026-07-02

Source branch inspected:

- Torch: `AdnanHoque/torch-spyre:ah/comms-collectives-dldsc-agent`
- SHA: `75040ee6d9f48518d0c194b72d1075035bb37b7b`
- Compared against artifact branch: `ah/comms-collectives` at `95b818680cfffd94baeb474420f4436467474feb`

## Summary

The branch is not a competing architecture. It is a narrow prototype for the
same DLDSC direction: keep computed activation restickifies on LX and let
Deeptools synthesize movement from DLDSC tensor-vs-compute coordinates.

The useful idea is the guard shape. The prototype only emits `ReStickifyOpLx`
when all of the following are true:

- the experimental restickify-output lane is enabled;
- the restickify source is a computed activation, not a graph input or weight;
- the restickify SDSC arguments are already LX-allocated.

That matches the intended scope: non-weight activation spills are in scope for
the communication substrate; weight restickifies remain out of scope because
offline/preloaded weight layout work should handle them.

## Relationship To Current Artifact Branch

The current artifact branch already contains a broader version of the same
direction:

- `ReStickifyOpLx` is represented in Torch constants and SDSC handling.
- The scratchpad allocator has a computed-source guard for restickify outputs.
- Flash activation edges can be classified as `layout_allgather_restickify`.
- The emitted DLDSC bundle now carries explicit LX residency coordinates and a
  communication classification for the flash activation handoff.

The branch therefore reinforces the current path rather than replacing it. The
main lesson to keep is the computed-activation guard; it avoids accidentally
pulling graph-input or weight restickifies into this research lane.

## Current Evidence

The latest flash artifact proves the frontend contract can now be emitted:

- before optimization: 32 `ReStickifyOpHBM` rows, 0 layout-allgather
  classifications, 0 LX residency coordinate entries;
- after optimization: 0 `ReStickifyOpHBM` rows, 32 `ReStickifyOpLx` rows, 32
  `layout_allgather_restickify` classifications, and 32 LX residency coordinate
  entries.

The remaining failure is backend scheduling/lowering, not frontend expression:

```text
DtException: Scheduler failed to find a suitable op mapping for sdsc: 2_ReStickifyOpLx
```

## Next Work

1. Keep DLDSC as the forward path.
2. Preserve the guard: only computed activation relayout/restickify edges should
   enter this lane; graph-input and weight relayouts stay out of scope.
3. Close the Deeptools gap for `ReStickifyOpLx` / `layout_allgather_restickify`
   physical lowering and scheduling.
4. Once flash schedules, rerun correctness and profiler traces before claiming a
