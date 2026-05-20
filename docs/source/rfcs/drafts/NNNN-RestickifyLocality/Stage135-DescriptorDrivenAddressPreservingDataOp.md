# Stage 135: Descriptor-Driven Address-Preserving Data-Op

## Summary

This stage made the address-preserving LX data-op probe consume the schema v3
`restickify_lx_neighbor_edges.json` descriptor directly. The probe now prefers
the real LX endpoint contract when it is present, and falls back to the older
HBM-base matching heuristic only when no usable descriptor exists.

This is a small but important shift: the prototype is no longer guessing which
producer output and consumer input belong to the restickify by comparing HBM
base addresses. It is using the contract that Torch-Spyre emitted for the
producer/restickify/consumer edge.

## What Changed

`tools/restickify_address_preserving_dataop_probe.py` now:

- Accepts `--descriptor`, defaulting to `--code-dir/restickify_lx_neighbor_edges.json`.
- Selects schema v3 edges with both `lx_endpoint_contract` and `sdsc_contract`.
- Reads the producer/restickify/consumer SDSC files from the descriptor.
- Reads producer output, restickify source/destination, and consumer input LDS
  roles from the descriptor instead of inferring them through HBM alias bases.
- Records `endpoint_contract.source = schema-v3-lx-endpoint-contract` in the
  summary when the descriptor path is used.

The old heuristic remains available for older generated artifacts.

## Pod Validation

First, I generated a fresh 2048 high-signal edge with the LX endpoint descriptor
enabled:

```sh
SPYRE_RESTICKIFY_RING_TELEMETRY=1 \
SPYRE_ALIGN_RESTICKIFY_CORE_MAPPING=1 \
SPYRE_ALIGN_RESTICKIFY_WORK_DISTRIBUTION=1 \
SPYRE_RESTICKIFY_LOCALITY_ASSERT=1 \
SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR=1 \
python tools/restickify_scenario_probe.py \
  --case computed_transpose_adds_then_matmul_tuple \
  --size 2048 \
  --skip-correctness \
  --copy-kernel-code \
  --output-dir /tmp/stage135-real-lx-contract-2048 \
  --fail-on-error
```

Result:

```text
ok size=2048 case=computed_transpose_adds_then_matmul_tuple restickifies=1 bytes=8388608 byte_hops=0
schema 3 edges 1 skipped 0
edge 0:1:2 has_sdsc True endpoint lx
```

Then I ran the address-preserving data-op probe against that generated code
directory:

```sh
python tools/restickify_address_preserving_dataop_probe.py \
  --code-dir /tmp/stage135-real-lx-contract-2048/kernel_code/computed_transpose_adds_then_matmul_tuple_2048/0001_sdsc_fused_add_t_0 \
  --output-dir /tmp/stage135-address-from-descriptor-standalone \
  --mode stage3b \
  --size 2048
```

Key summary:

| Field | Value |
|---|---|
| Endpoint source | `schema-v3-lx-endpoint-contract` |
| Producer output LDS | `2` |
| Restickify input LDS | `0` |
| Restickify output LDS | `1` |
| Consumer input LDS | `1` |
| Producer pieces patched | `32` |
| Consumer pieces patched | `32` |
| `DataOpStandalone` return code | `0` |
| Lowered data-op units | `LXLU`, `LXSU` present |

Artifacts copied locally:

- `artifacts/stage135_real_lx_endpoint_contract/summary.json`
- `artifacts/stage135_real_lx_endpoint_contract/restickify_lx_neighbor_edges.json`

## Interpretation

This validates the new plumbing: the real LX endpoint contract can drive the
address-preserving `ReStickifyOpLx -> STCDPOpLx` data-op prototype. That is the
right direction for a production solution because the endpoint identities come
from Torch-Spyre's own generated SDSC metadata.

One caveat: `DataOpStandalone`'s MLIR still declares HBM/L3 units globally, so a
plain string count of `HBM` or `L3` in the file is not enough to prove active
traffic. The meaningful signal in this stage is that the descriptor-driven
patch selected LX endpoints and `DataOpStandalone` lowered successfully. The
next proof should inspect active transfer edges or runtime counters, not unit
declarations.

## Next Step

The next integration blocker is still mixed runtime packaging. We have a
descriptor-driven standalone data-op artifact; now we need Torch-Spyre/Flex to
package the producer compute, endpoint-preserving restickify data-op, and
consumer compute as one value-correct runtime unit without falling back to a
stock `ReStickifyOpHBM` boundary.
