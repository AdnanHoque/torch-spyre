# Stage 134: Real LX Endpoint Contract For Stock Restickification

## Summary

This stage pivots away from the diagnostic "stock LX alias" path and toward a
production-shaped contract: reuse the stock restickification semantics, but make
the producer, restickify, and consumer agree on real LX endpoints before
Deeptools schedules and lowers the program.

The alias experiment was useful because it showed what not to do. Patching an
HBM restickify boundary after scheduling can compile, but it does not preserve
the source/destination LDS roles, dimension contract, or per-core ownership that
the stock op expected when it was created. The failure is not just a small
address bug; it is an endpoint contract mismatch.

## What Changed

`restickify_lx_neighbor_edges.json` is now schema version 3. For each eligible
certified internal restickify edge, it records:

- `source_view_contract`: the producer physical output view, restickify logical
  source view, restickify destination view, and consumer input view.
- `sdsc_contract`: SDSC-level role metadata when JSON payloads are available,
  including producer output LDS, restickify source/destination LDS, consumer
  input LDS, primary layout metadata, allocation summaries, and compute labels.
- `lx_endpoint_contract`: the production-shaped handoff object. It states that
  the target memory space is LX, names the endpoints that must be preserved, and
  explicitly marks post-hoc HBM aliasing as disallowed.

The endpoint contract names four endpoints:

| Endpoint | Meaning |
|---|---|
| `producer_lx_source` | The producer output already resident in LX. |
| `restickify_lx_input` | The stock restickify input endpoint that must read the producer LX source. |
| `restickify_lx_output` | The stock restickify output endpoint produced in LX. |
| `consumer_lx_sink` | The consumer input endpoint that must read the restickify LX output. |

The descriptor remains metadata-only. Normal bundle execution is unchanged.

## Why This Path Is Better

This is the closest path to a production solution because it describes the
contract before runtime packaging instead of mutating an already-lowered HBM
boundary. It also lines up with the strongest evidence from previous stages:

- Stage 70 showed that a descriptor-driven neighbor movement path can reference
  scheduled producer and consumer LX addresses.
- Stage 74 showed an address-preserving `ReStickifyOpLx -> STCDPOpLx` data-op
  artifact with no HBM or L3 traffic in the generated program summary.
- Stage 75 showed that the address-preserving data-op artifact can be exported
  and launched as a minimal Deeprt runtime artifact.

The remaining gap is not "can we spell an LX movement." It is making the
Torch-Spyre/Flex/Deeptools contract carry these endpoints through mixed compute
and data-op packaging.

## Required Contract

The real contract must preserve:

- Producer LX allocation identity.
- Consumer LX allocation identity.
- The `coreStateInit_` addresses or an equivalent explicit endpoint reference.
- Core ownership and work-slice mapping.
- Layout and stick metadata for the source and destination views.
- Runtime lifetime/synchronization across producer, restickify, and consumer.

## Non-Solutions

The descriptor intentionally records these as known non-solutions:

- Patch `ReStickifyOpHBM` HBM allocations to LX after scheduling.
- Compact every source core to local LX address zero.
- Copy `coreIdToWkSlice_` without also satisfying the split/layout endpoint
  contract.

## Next Steps

1. Make the address-preserving data-op probe consume schema v3 directly instead
   of reverse-engineering endpoints from generated SDSC JSON.
2. Recreate the Stage 74 address-preserving `ReStickifyOpLx -> STCDPOpLx`
   artifact from the descriptor and confirm the generated program remains
   HBM/L3-free.
3. Work on mixed runtime packaging: either support the data-op inside the same
   DXP/Flex bundle as compute SDSCs, or add a Deeptools-native endpoint reference
   that lets stock restickification consume producer LX and produce consumer LX.
4. Only after mixed packaging is value-correct should this move from prototype
   metadata to a compiler/runtime integration patch.

## Validation

Local syntax validation passed:

```sh
python3 -m py_compile \
  torch_spyre/_inductor/codegen/lx_neighbor_descriptor.py \
  tests/inductor/test_restickify_lx_neighbor_descriptor.py \
  torch_spyre/_inductor/codegen/bundle.py
```

Focused pod validation also passed in the Torch-Spyre development environment:

```sh
python -m pytest tests/inductor/test_restickify_lx_neighbor_descriptor.py -q
# 7 passed in 0.03s
```
