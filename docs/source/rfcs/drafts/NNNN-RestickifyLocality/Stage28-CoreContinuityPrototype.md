# Stage 28: Core-Division Continuity Prototype

## Summary

This stage implements a default-off prototype for **ring-aware core-division
continuity**. The prototype targets exact in-graph pointwise producer-consumer
edges, not graph inputs, weights, constants, matmul reductions, or true
restickify placement.

The narrow goal is: when one in-graph producer has a single dominant split
dimension, prefer the corresponding consumer output dimension during work
distribution so the same physical cores keep ownership of the same logical
tensor regions. This can remove modeled RIU byte-hops without changing tensor
semantics, restickify count, or selected layouts.

The feature is disabled by default:

```sh
SPYRE_ALIGN_CORE_DIVISION_CONTINUITY=1
```

Telemetry remains separate:

```sh
SPYRE_CORE_CONTINUITY_TELEMETRY=1
SPYRE_CORE_CONTINUITY_TELEMETRY_JSONL=/path/to/core_continuity.jsonl
```

## Implementation

The prototype adds `torch_spyre/_inductor/core_continuity_alignment.py`.

It has two pieces:

- **Work-distribution steering**: for exact pointwise in-graph edges, use the
  same stride/symbol correspondence machinery as the core-continuity telemetry
  and prefer the consumer dim mapped to the producer's single split dim.
- **Certified mapping override path**: build a producer-aligned
  `coreIdToWkSlice_` override only when producer and consumer split factors
  match exactly and the modeled byte-hop estimator proves the override is zero
  hop.

During validation, a less conservative version allowed multi-dimensional split
steering and multi-dimensional mapping overrides. That found two hazards:

- small multi-split cases can get worse, even if large single-split cases win
- one multi-split pointwise override hit a deeptools fold-manager failure on
  `fanout_diamond` size `512`

The prototype was tightened accordingly:

- only producers with exactly one split dimension are eligible for steering
- only producer/consumer pairs with exactly one split dimension each are
  eligible for a mapping override
- restickify ops themselves remain handled by the separate Stage 3B flags

This leaves the high-signal `2048` cases intact and avoids the small-shape
regressions observed in the first attempt.

## Probe Command

Run from the pod checkout:

```sh
export HOME=/home/adnan-cdx
export DTI_PROJECT_ROOT=$HOME/dt-inductor
source $DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh
export SENTIENT_BASE_INSTALL_DIR=/opt/ibm/spyre
export DEEPTOOLS_INSTALL_DIR=/opt/ibm/spyre/deeptools
export DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share
export PATH=/opt/ibm/spyre/deeptools/bin:/opt/ibm/spyre/runtime/bin:$PATH
export LD_LIBRARY_PATH=/opt/ibm/spyre/deeptools/lib:/opt/ibm/spyre/runtime/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/tmp/torch-spyre-core-continuity-test:${PYTHONPATH:-}
export SENCORES=32
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
cd /tmp/torch-spyre-core-continuity-test

SPYRE_ALIGN_CORE_DIVISION_CONTINUITY=1 \
SPYRE_CORE_CONTINUITY_TELEMETRY=1 \
SPYRE_CORE_CONTINUITY_TELEMETRY_JSONL=/tmp/core_continuity.jsonl \
python tools/restickify_scenario_probe.py \
  --case pointwise_transpose_add \
  --case transpose_chain \
  --case fanout_diamond \
  --case adds_then_matmul \
  --size 512 \
  --size 2048 \
  --ring-telemetry \
  --skip-correctness \
  --output-dir /tmp/restickify-stage28/probe \
  --fail-on-error
```

## Results

Final artifact directory:

```text
artifacts/stage28_core_continuity_prototype/
```

Final telemetry run:

```text
/tmp/restickify-stage28-core-continuity-1779062717
```

| Case | Size | Baseline byte-hops | Prototype byte-hops | Delta |
|---|---:|---:|---:|---:|
| `adds_then_matmul` | 512 | 2,752,512 | 2,752,512 | 0 |
| `adds_then_matmul` | 2048 | 134,217,728 | 0 | 134,217,728 |
| `fanout_diamond` | 512 | 2,752,512 | 2,752,512 | 0 |
| `fanout_diamond` | 2048 | 134,217,728 | 0 | 134,217,728 |
| `pointwise_transpose_add` | 512 | 1,376,256 | 1,376,256 | 0 |
| `pointwise_transpose_add` | 2048 | 67,108,864 | 0 | 67,108,864 |
| `transpose_chain` | 512 | 1,376,256 | 1,376,256 | 0 |
| `transpose_chain` | 2048 | 67,108,864 | 0 | 67,108,864 |

The 2048 wins come from conservative work-distribution steering. The final
telemetry reports the resulting exact edges as `already-local`; the modeled
byte-hop count is zero after work distribution, so no core mapping override is
needed for these cases.

The 512 rows remain unchanged because their producers use multi-dimensional
splits such as `d0:8,d1:4`; those are intentionally skipped.

## Validation

Pod validation:

```text
python -m pytest tests/inductor/test_restickify_mapping_alignment.py -q
25 passed in 0.09s
```

Default-off regression, rerun with the required `/opt/ibm/spyre/deeptools`
environment:

```text
python -m pytest tests/inductor/test_restickify.py -q
97 passed in 70.42s
```

Probe validation:

```text
baseline: 8 rows, 0 errors
prototype: 8 rows, 0 errors
```

## Conclusion

This is a useful prototype, but still a prototype. It gives a clean evidence
point for a second ring-aware optimization family beyond Stage 3B: preserving
producer-consumer core ownership across ordinary pointwise edges.

The most important lesson is the conservative gate. Single-split continuity is
promising and non-invasive. Multi-split continuity needs a more formal codegen
and fold legality check before it should be considered.
