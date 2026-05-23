# Stage 329: Normal Mixed PT-LX Success

## Summary

Stage 329 moved the value-correct Stage195 bridge contract back into the normal
Torch-Spyre mixed-schedule lowering path for the same-bundle fixture:

```python
def computed_transpose_adds_then_matmul_tuple(a, b, c, d):
    u = a + (b + c).t()
    return u, u @ d
```

At size `2048`, the normal lowering path now:

- patches the producer/restickify/consumer triple;
- removes the standalone `ReStickifyOpHBM` SDSC from the fused-add bundle;
- emits `MixedReStickifyOpWithPTLxConsumer`;
- validates the producer-to-bridge and bridge-to-consumer LX endpoints;
- passes hardware correctness.

This is not yet production-ready because it still uses forced diagnostic LX
bases, but it is materially stronger than the earlier late binary splice:
the mixed SDSC is generated during Torch-Spyre lowering rather than inserted by
post-DXP frame surgery.

## Command

```sh
SPYRE_RESTICKIFY_PTLX_MIXED_SCHEDULE_E2E=1
SPYRE_RESTICKIFY_PTLX_FORCE_ENV_ENDPOINTS=1
SPYRE_RESTICKIFY_PTLX_BRIDGE_PRODUCER_BASE=0
SPYRE_RESTICKIFY_PTLX_BRIDGE_CONSUMER_BASE=524288
SPYRE_RESTICKIFY_PTLX_BRIDGE_AUDIT_JSONL=/tmp/stage329-stage195-mixed-run-2048.jsonl
python tools/restickify_scenario_probe.py \
  --case computed_transpose_adds_then_matmul_tuple \
  --size 2048 \
  --copy-kernel-code \
  --output-dir /tmp/stage329-stage195-mixed-run-2048 \
  --fail-on-error
```

`SPYRE_RESTICKIFY_PTLX_STREAMING_E2E` was intentionally unset.  This uses the
full-tensor `ReStickifyOpWithPTLx -> STCDPOpLx` bridge contract from Stage195,
not the newer streaming/direct-tile diagnostic candidates.

## Result

Probe result:

```text
ok size=2048 case=computed_transpose_adds_then_matmul_tuple
restickifies=1 bytes=8388608 byte_hops=0
Completed 1 rows with 0 errors
```

Generated fused-add bundle:

```text
sdsc_0_add.json
sdsc_1_MixedReStickifyOpWithPTLxConsumer.json
```

String check:

```text
sdsc_0_add.json                              ReStickifyOpHBM=false
sdsc_1_MixedReStickifyOpWithPTLxConsumer.json ReStickifyOpHBM=false
sdsc_1_MixedReStickifyOpWithPTLxConsumer.json ReStickifyOpWithPTLx=true
sdsc_1_MixedReStickifyOpWithPTLxConsumer.json STCDPOpLx=true
```

Audit:

```text
status: patched
kind: ptlx-mixed-schedule
replacement_sdsc: 1_MixedReStickifyOpWithPTLxConsumer
producer_lx_unique_starts: [0]
consumer_lx_unique_starts: [524288]
```

Value-flow contract:

```text
producer_to_bridge_input_match: true
bridge_output_to_consumer_match: true
producer_core_count: 32
bridge_input_core_count: 32
bridge_output_core_count: 32
consumer_core_count: 32
valid: true
```

## Interpretation

This is the strongest current evidence for the production-shaped path:

1. The PT-aware bridge contract can be value-correct.
2. The consumer can read the bridge output through an LX endpoint.
3. The stock `ReStickifyOpHBM` SDSC can be removed from the normal generated
   fused bundle for this same-bundle case.

The remaining production blocker is allocation planning.  The successful run
uses forced bases:

```text
producer base: 0
consumer base: 524288
intermediate base: 262144
per-core span: 262144 bytes
```

An earlier attempt with the historical Stage195 bases overlapped the per-core
ranges and correctly skipped:

```text
producer base: 16384
consumer base: 8192
reason: ptlx-endpoint-ranges-overlap
```

So the next compiler step is not another data-op spelling.  It is to allocate
non-overlapping producer, intermediate, and consumer LX ranges automatically
before SDSC generation.

## Next Step

Implement allocator-backed endpoint planning for the normal mixed PT-LX path:

- reserve producer output, bridge intermediate, and consumer input LX ranges;
- attach those ranges to `PTLX_ENDPOINT_ALLOCATION_OP_INFO_KEY`;
- remove the forced-env endpoint requirement for this fixture;
- rerun the same `2048` correctness probe with no forced bases.

After that works, run the size sweep to see which shapes patch, skip, or need
the streaming/chunked path.

## Note

An optional sweep was attempted after this run, but the OpenShift session had
expired:

```text
error: You must be logged in to the server (Unauthorized)
```
