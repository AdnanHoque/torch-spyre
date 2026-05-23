# Stage 326: Runtime Allocation Audit

## Summary

Stage 326 audits the value-correctness failure from Stage 325. The question was
simple:

> Do the PT-LX sidecar's gather/scatter LX addresses match the actual runtime
> allocations around the stock `ReStickifyOpHBM` bundle?

Result: no.

The normal bundle being patched still allocates the relevant producer,
restickify, and consumer tensors in HBM. The PT-LX sidecar uses synthetic LX
addresses. That explains why the Stage 325 patched path can launch and avoid
explicit HBM tokens in the bridge frame, but still computes wrong values.

## Tool

The new audit helper is:

```sh
python3 tools/restickify_runtime_allocation_audit.py \
  --code-dir <patched-or-unpatched-code-dir> \
  --output <audit.json>
```

It reports:

- SDSC order and compute input/output LDS indices;
- allocation maps by LDS and component (`hbm`, `lx`, `sfp`, etc.);
- restickify position;
- sidecar gather/source LX starts;
- sidecar scatter/destination LX starts;
- whether producer output and consumer inputs have actual LX allocations.

## Result On Stage 325 Failure

Audited code directory:

```text
/tmp/torchinductor_1000800000/tmp4nhez89v/inductor-spyre/sdsc_fused_addmm_t_0_7ivo6q5l
```

Summary:

| Field | Value |
|---|---|
| SDSC order | `batchmatmul`, `ReStickifyOpHBM`, `add` |
| producer output components | `hbm` |
| producer output has LX | `false` |
| consumer input components | input 0: `hbm`, input 1: `hbm` |
| consumer any input has LX | `false` |
| sidecar gather LX starts | `[0]` |
| sidecar scatter LX starts | `[262144]` |

So the bridge is not reading the real producer output. It is reading from a
sidecar-modeled LX source address that the stock runtime bundle does not use
for this tensor edge.

## Interpretation

This rules out the idea that we can get a correct implementation by only
replacing the `ReStickifyOpHBM` program frame after the normal bundle has
already been planned.

At that point, the producer/restickify/consumer contract is already HBM-shaped:

```text
batchmatmul output -> HBM
ReStickifyOpHBM input -> HBM
ReStickifyOpHBM output -> HBM
add input -> HBM
```

The PT-LX sidecar is internally HBM-free, but it is not connected to live
producer/consumer LX allocations:

```text
sidecar gather source -> synthetic LX start 0
sidecar scatter dest  -> synthetic LX start 262144
```

That makes the Stage 325 correctness failure expected rather than mysterious.

## Production Implication

The production-shaped fix must happen before final SDSC/codegen for the affected
bundle. A late frame splice is useful as a launchability probe, but it cannot
create a real LX value-flow contract if the producer and consumer were already
planned through HBM.

The next implementation target is therefore:

1. during bundle/codegen planning, identify the eligible
   producer -> restickify -> consumer edge;
2. force or request an LX allocation for the producer output that survives until
   the bridge consumes it;
3. force or request an LX allocation for the consumer input that matches the
   bridge scatter/output;
4. generate the PT-LX bridge from those real allocation maps, not from synthetic
   sidecar addresses;
5. only then remove or bypass the stock `ReStickifyOpHBM` frame.

This is stricter than Stage 3B core mapping. Stage 3B optimizes locality if a
restickify exists. The PT-LX path needs an actual internal LX allocation
contract.

## Artifacts

Pod:

```text
/tmp/stage326-runtime-allocation-audit-512.json
```

Local copy:

```text
artifacts/stage326_runtime_allocation_audit/audit_512.json
```
