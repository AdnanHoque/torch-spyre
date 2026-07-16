# Four-Pod Execution Plan

## Why this parallelization is useful

The experiment has four independent proof obligations:

1. Can redundant-coordinate `SHUFFLE` lower into real bounded transfers?
2. Is the explicit meta-dimension encoding a viable fallback?
3. Is direct `S1 -> S2` physically value-correct, or is a local layout
   conversion required?
4. Does the winning contract preserve correctness and outperform the frozen
   HBM and custom-materializer controls?

Running multiple backend implementations would create conflicting patches and
would not shorten the critical path. Instead, each pod owns one question and
produces an artifact that can be consumed by the single integration lane.

## Pod ownership

| Pod | Write scope | Required output | Merge role |
|---|---|---|---|
| `adnan-cdx-spyre-dev-pf` | Exact Deeptools Variant A worktree only | Endpoint-aware eight-row `STCDPOpLx` lowering, replay, patch, physical transfer proof | Sole production-backend candidate |
| `adnan-clc-spyre-dev-pf` | Variant B fixture/replay directory only | Smallest meta-dimension fixture and decisive viability/expansion report | Diagnostic fallback only |
| `adnan-spyre-dev-pf` | Direct-versus-staged diagnostic directory only | Post-DCG and patterned value comparison of direct and restickified paths | Selects contract shape; no direct merge |
| `adnan-spyre-current-pf` | Frozen control and benchmark run directories only | Hash-verified HBM/custom controls, patterned oracle, Kineto runs | Independent acceptance lane |

No lane may modify or force-push an active Torch or Deeptools PR branch.

## Dependency graph

```text
Variant A backend lowering ---------+
                                     +--> structural verifier
Variant B diagnostic ---------------+          |
                                                v
Direct-vs-staged value diagnostic --+--> select explicit contract
                                                |
Frozen HBM/custom controls ---------+            v
                                          patterned AIU test
                                                |
                                                v
                                      four-shape benchmark matrix
```

Variant B is not on the critical path unless Variant A proves impossible.
Control capture and harness validation run concurrently with compiler work.

## Integration gates

The integration lane accepts a candidate only in this order:

1. DXP imports the bundle.
2. Post-DCG rows contain nonempty transfer tables; a NOP is a failure.
3. Source and destination addresses match frontend S1/S2 allocations.
4. Eight bounded shard rows cover 256 placements: 224 remote and 32 local.
5. No HBM row, untracked allocation, overlap, or consumer-before-movement
   schedule appears.
6. Patterned AIU output has every shard exactly once, in order, without
   cross-head contamination or out-of-bounds writes.
7. Flash integration compiles and runs; known baseline numerical defects are
   reported separately from communication correctness.
8. Kineto kernel time is compared against both frozen controls.

No performance run is accepted before gates 1-6 pass.

## Benchmark matrix

| Lq | Mask | HBM fallback | Custom materializer | Explicit contract |
|---:|:---:|:---:|:---:|:---:|
| 512 | off | required | required | required |
| 512 | on | required | required | required |
| 1024 | off | required | required | required |
| 1024 | on | required | required | required |

Every row records Torch, Deeptools, perf-suite and runtime identities, complete
environment, LX map, transfer counts and bytes, correctness/fallback state,
Kineto kernel time, and wall time.

## Expected time savings

The exact backend build/replay, fallback encoding, layout-correctness decision,
and control preparation are independent and therefore run concurrently. The
only serialized work is final integration, patterned hardware validation, and
benchmarking. This removes two full backend build/debug cycles from the
critical path while keeping one authoritative implementation branch.
