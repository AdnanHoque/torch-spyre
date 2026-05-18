# Stage 43: DDL Bridge E2E Prototype

## Summary

Stage 42 proved that a real Torch-Spyre `ReStickifyOpHBM` SDSC can be reshaped
into the compact restickify DDL contract and compiled to an LX/SFP/PT-only
senprog with no visible L3/HBM tokens.

Stage 43 turns that result into a default-off Torch-Spyre prototype. The
prototype does not touch Deeptools and does not change default behavior.

## Code Change

Added a new generator:

```text
torch_spyre/_inductor/codegen/restickify_ddl_bridge.py
```

Added default-off config:

```text
SPYRE_RESTICKIFY_DDL_BRIDGE_E2E=1
SPYRE_RESTICKIFY_DDL_BRIDGE_AUDIT_JSONL=/path/to/audit.jsonl
SPYRE_RESTICKIFY_DDL_BRIDGE_PREDDC_SHIM=1
```

When enabled, `compile_op_spec()` can replace a normal `ReStickifyOpHBM`
compute SDSC with a compact DDL-style restickify input only if a narrow contract
holds:

- exactly one input and one output,
- no constants, padding, or coordinate masking,
- exactly one split dimension,
- that split dimension covers all cores,
- the output stick dimension is the split dimension,
- the input stick dimension is not the split dimension,
- estimated per-core LX footprint is at most 512 KiB.

The generated SDSC keeps the normal root `dscs_` path, but changes the
restickify payload to the DDL input form:

- two LX-only labeled data spaces,
- `INPUT` and `OUTPUT` primary data-space roles,
- two `dataStageParam_` entries,
- an LX-local transfer/loop skeleton,
- preserved `coreIdToWkSlice_` and fold metadata.

## DXP Shim

The installed DXP still runs generic corelet splitting and L3 scheduling before
DDC. Those passes reject the compact restickify DDL input before the DDL
template can expand it.

To keep this prototype self-contained and avoid a Deeptools push, Torch-Spyre
can compile and apply the Stage 41 `LD_PRELOAD` shim, but only for bundles where
every SDSC is a DDL-bridge restickify:

```text
_bundle_contains_only_restickify_ddl_bridge(output_dir)
```

Mixed bundles are explicitly not shimmed. This avoids skipping L3 scheduling
for normal pointwise or matmul SDSCs.

## High-Signal Probe Result

Command:

```sh
SPYRE_RESTICKIFY_DDL_BRIDGE_E2E=1 \
SPYRE_RESTICKIFY_DDL_BRIDGE_AUDIT_JSONL=/tmp/stage43-ddl-bridge-e2e/audit.jsonl \
python tools/restickify_scenario_probe.py \
  --case adds_then_matmul \
  --size 2048 \
  --ring-telemetry \
  --skip-correctness \
  --output-dir /tmp/stage43-ddl-bridge-e2e \
  --fail-on-error
```

Result:

```text
ok size=2048 case=adds_then_matmul restickifies=2 bytes=16777216 byte_hops=67108864
```

The audit rows show both restickifies were skipped:

```text
status=skipped reason=mixed-kernel-bundle
```

This is expected. The real generated fused kernels are not standalone
restickify bundles:

- `sdsc_fused_add_t_0` contains the bridge-eligible restickify plus two normal
  `add` SDSCs.
- Applying the global preload shim to that mixed bundle fails later in DXP with
  `std::out_of_range: map::at`.

So Stage 43 deliberately keeps the high-signal runtime probe on the existing
HBM path until we have a targeted Deeptools bypass or a Torch-Spyre kernel
split that isolates restickify into its own bundle.

## Interpretation

The prototype is useful but intentionally not a production solution yet:

- The DDL bridge generator exists in Torch-Spyre and is unit-tested.
- The emitted DDL shape matches the successful Stage 42 standalone contract.
- Stock DXP still cannot consume that shape inside a mixed bundle.
- The guarded pre-DDC shim is safe only for all-restickify DDL bundles.
- Real high-signal cases currently place restickify inside mixed fused kernels,
  so they are skipped and remain on the HBM path.

This is the correct boundary for a no-Deeptools-push prototype.

## Next Step

There are two credible paths from here:

1. Deeptools targeted bypass: skip generic pre-DDC corelet splitting and L3
   scheduling only for DDL-bridge restickify SDSCs, not globally.
2. Torch-Spyre isolation: split eligible restickify ops into their own DDL
   bridge bundle so the Stage 41 shim can be applied without touching normal
   SDSCs.

The Deeptools targeted bypass is cleaner for production. The Torch-Spyre split
is attractive if we want to continue proving this from our side without pushing
anything to Deeptools.

## Validation

Pod:

```text
python -m py_compile \
  torch_spyre/_inductor/config.py \
  torch_spyre/execution/async_compile.py \
  torch_spyre/_inductor/codegen/bundle.py \
  torch_spyre/_inductor/codegen/superdsc.py \
  torch_spyre/_inductor/codegen/restickify_ddl_bridge.py \
  tests/inductor/test_restickify_ddl_bridge.py

python -m pytest \
  tests/inductor/test_restickify_ddl_bridge.py \
  tests/inductor/test_restickify_mapping_alignment.py \
  -q
```

Result:

```text
22 passed
```

The high-signal `adds_then_matmul` probe also completed with zero errors when
the flag was enabled, because mixed kernels were skipped back to the existing
path.
