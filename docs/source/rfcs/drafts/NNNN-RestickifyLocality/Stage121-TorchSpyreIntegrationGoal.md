# Stage 121: Torch-Spyre Integration Goal

## Goal

Move the LX-to-LX restickify work from a standalone Deeptools compiler proof
toward a Torch-Spyre-integrated prototype.

Stage 120 proved that Deeptools can generate verified HBM-free core-to-core
movement using `InputFetchNeighbor` / `STCDPOpLx`. Stage 121 should make
Torch-Spyre express that path as an internal edge in a real compiled graph.

## Remaining Work

The next prototype should:

1. Generate an internal-edge descriptor from Torch-Spyre when a producer output
   feeds a restickify input and the consumer can read an LX-neighbor input.
2. Avoid the `mb/out` to `ij/in` probe alias by either:
   - generalizing `InputFetchNeighbor` dimension ordering; or
   - emitting a Deeptools-native shape that satisfies the existing contract.
3. Package the generated `InputFetchNeighbor` program into the Torch-Spyre/Flex
   runtime artifact, rather than only printing `senprog.txt`.
4. Validate with a small value-correct graph:

```text
producer add -> LX-to-LX restickify/input-fetch -> consumer add
```

## Success Criteria

The next stage succeeds when the first fused bundle:

- retires on hardware;
- does not emit or launch `ReStickifyOpHBM` for the internal edge;
- uses the LX-to-LX `InputFetchNeighbor` path;
- avoids HBM traffic for the restickify movement;
- avoids the stream hardware error seen in earlier boundary-splicing attempts;
- produces the correct tensor value for the small add/restickify/add graph.

## Current Best Hypothesis

The production-shaped abstraction is not a standalone replacement of
`ReStickifyOpHBM`. It is an internal scheduled edge where producer, restickify
movement, and consumer agree on:

```text
producer LX allocation
  -> InputFetchNeighbor/STCDPOpLx movement
  -> consumer LX-neighbor input
```

Torch-Spyre must therefore carry enough metadata across codegen/runtime
packaging for Flex to treat the restickify edge as an internal LX edge rather
than a graph boundary tensor reloaded from HBM.
