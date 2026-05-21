# Stage 213: Relaxed PT-LX Gate

## Summary

This stage removed the hard requirement that a PT-LX restickify bridge must
already have a Stage 3B core-mapping override. The override is now treated as a
second-order locality optimization: if a zero-hop certificate exists, telemetry
records it and the bridge uses the same proven path; if not, the bridge may
still be planned when the compiler can safely create allocator-backed LX
endpoints.

The implementation remains default-off behind:

- `LX_PLANNING=1`
- `SPYRE_RESTICKIFY_PTLX_MIXED_SCHEDULE_E2E=1`
- `SPYRE_RESTICKIFY_PTLX_VALUE_FLOW_ASSERT=1`

## What Changed

- PT-LX mixed-schedule eligibility now requires `restickify_source_kind ==
  "in_graph_computed"` but no longer requires `core_id_to_work_slice_override`
  or a locality certificate.
- Scratchpad endpoint forcing now considers uncertified in-graph restickifies,
  but only when the prototype can preserve correctness:
  - producer and restickify split core counts match
  - split pieces are at least one 64-element stick wide
  - estimated endpoint bytes fit conservatively inside the reserved LX budget
  - large uncertified shapes stay skipped until endpoint reservation is atomic
- The bridge generator can now accept explicit producer/restickify/consumer
  `numWkSlicesPerDim_` and `coreIdToWkSlice_` maps from generated SDSCs instead
  of hard-coding the intermediate split.
- Audit rows now include a `core_locality` block when patched, so transport
  success and zero-hop locality remain separable.

## Validation

Focused tests:

```sh
python -m pytest tests/inductor/test_restickify_lx_dataop.py -q
# 17 passed

python -m pytest tests/inductor/test_scratchpad_patterns.py -k certified_ptlx -q
# 1 passed, 14 deselected
```

Hardware correctness sweep:

```sh
python tools/restickify_scenario_probe.py \
  --case computed_transpose_adds_then_matmul_tuple \
  --size 512 --size 1024 --size 1536 --size 2048 --size 3072 --size 4096 \
  --ring-telemetry \
  --fail-on-error
```

All sizes passed correctness with PT-LX enabled. The only patched size in this
family remains `2048`; the other sizes skip before endpoint forcing and run the
stock HBM restickify path.

## Timing Sweep

Case: `computed_transpose_adds_then_matmul_tuple`

| Size | Stock HBM median | Stage3B HBM median | Relaxed PT-LX median | PT-LX audit |
|---:|---:|---:|---:|---|
| 512 | 0.119 ms | 0.111 ms | 0.115 ms | skipped |
| 1024 | 0.292 ms | 0.288 ms | 0.289 ms | skipped |
| 1536 | 0.588 ms | 0.594 ms | 0.586 ms | skipped |
| 2048 | 1.309 ms | 1.314 ms | 1.023 ms | patched |
| 3072 | 2.866 ms | 2.885 ms | 2.873 ms | skipped |
| 4096 | 7.012 ms | 7.020 ms | 7.027 ms | skipped |

At `2048`, relaxed PT-LX is `1.28x` faster than Stage3B HBM and `1.28x`
faster than stock HBM for this probe.

## Interpretation

Removing the override requirement is structurally correct, but it does not by
itself make every size patchable. The bridge has stricter physical contracts
than the stock HBM restickify path:

- `512`, `1024`, and `1536` split the producer too finely for the current
  single `ReStickifyOpWithPTLx` bridge; a piece in the output-stick dimension
  would be smaller than one 64-element stick.
- `3072` uses a different restickify core count than the adjacent producer in
  this graph, so a single bridge cannot safely preserve the value-flow contract.
- `4096` needs larger LX endpoint reservations than the current greedy endpoint
  forcing can safely reserve without risking partial LX allocation.

The next production-shaped step is not another locality certificate tweak. It
is atomic endpoint planning: reserve the producer endpoint, restickify bridge
workspace, and consumer endpoint as one allocation decision before SDSC
generation. That should let the compiler either patch the whole edge or leave
the stock HBM path completely untouched.
