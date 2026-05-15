# Stage 8: Locality-Certified Stage 3B Prototype

This note records the default-off prototype that turns Stage 3B restickify
alignment from best-effort into compiler-certified locality for eligible
in-graph restickifies.

## What Changed

The prototype adds `SPYRE_RESTICKIFY_LOCALITY_ASSERT=1`. This flag only matters
when Stage 3B alignment is also enabled:

- `SPYRE_ALIGN_RESTICKIFY_WORK_DISTRIBUTION=1`
- `SPYRE_ALIGN_RESTICKIFY_CORE_MAPPING=1`
- `SPYRE_RESTICKIFY_LOCALITY_ASSERT=1`

When enabled, any restickify core mapping override must certify zero modeled
RIU byte-hops before it reaches codegen. Graph-input and weight restickifies are
not failures; they are explicitly marked as skipped because the compiler has no
in-graph producer core ownership to preserve.

Telemetry JSONL now includes:

- `locality_certified`
- `locality_assertion`
- `locality_skip_reason`
- `certified_byte_hops`
- `certified_bytes_moved`
- `certified_max_hops`
- `certified_core_count`

## Validation Probe

The high-signal validation used:

```sh
python3.12 -u tools/restickify_hierarchy_sweep.py \
  --case adds_then_matmul_x \
  --size 2048 \
  --mode baseline \
  --mode stage3b \
  --locality-assert \
  --time \
  --warmup 5 \
  --iters 10 \
  --skip-correctness \
  --output-dir /tmp/restickify-locality-certified-prototype
```

Raw artifacts are saved locally under:

- `artifacts/restickify_locality_certified_prototype/probe/hierarchy_rows.csv`
- `artifacts/restickify_locality_certified_prototype/probe/hierarchy_rows.jsonl`
- `artifacts/restickify_locality_certified_prototype/probe/hierarchy_pairs.csv`

## Result

| Mode | Restickifies | Bytes moved | Byte-hops | Locality assertions | Certified rows | Median |
|---|---:|---:|---:|---|---:|---:|
| Baseline | 2 | 16,777,216 | 67,108,864 | `not-run: 2` | 0 | 1.746 ms |
| Stage 3B + assert | 2 | 16,777,216 | 0 | `passed: 1`, `skipped: 1` | 1 | 1.709 ms |

The certified row is the in-graph `buf2 -> restickify -> matmul` edge. The
skipped row is the graph-input restickify, which remains out of scope for Stage
3B. Restickify count and bytes moved stayed unchanged.

This validates the prototype claim: for the eligible in-graph restickify, the
compiler can require a zero-byte-hop certificate before emitting the
producer-aligned `coreIdToWkSlice_` override.

## Tests

Validation completed in the pod against `/tmp/torch-spyre-refresh`:

| Check | Result |
|---|---|
| `python3.12 -m py_compile` on changed files | passed |
| `python3.12 -m pytest tests/inductor/test_restickify_mapping_alignment.py -q` | 19 passed |
| selected `tests/inductor/test_restickify.py` families with flags off | 5 passed, 92 deselected |
| `adds_then_matmul_x` size `2048` with locality assert | passed, one certified row |

## Interpretation

This is still a compiler model, not physical-fabric proof. The certificate
means producer/restickify ownership geometry implies zero RIU byte-hops after
the Stage 3B override. Direct confirmation of actual RIU/HBM traffic still
requires AIUPTI/Kineto traces or hardware counters.

No PR was created and no merge was performed.
