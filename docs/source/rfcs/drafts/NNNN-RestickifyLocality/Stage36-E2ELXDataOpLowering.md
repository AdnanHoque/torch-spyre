# Stage 36: E2E LX Data-Op Lowering Probe

## Summary

Stage 35 showed that Deeptools DCG can accept an LX-resident
`ReStickifyOpLx -> STCDPOpLx` data-op sequence. Stage 36 tested the next
question: can the normal Torch-Spyre bundle path replace a compiled
`ReStickifyOpHBM` with a default-off LX data-op SDSC?

The short answer is: not yet. The guarded compiler hook works, but the current
DXP bundle path rejects `datadscs_` and asks for `dldsc`.

## Code Changes

Added default-off config flags:

- `SPYRE_RESTICKIFY_LX_DATAOP_E2E=1`
- `SPYRE_RESTICKIFY_LX_DATAOP_AUDIT_JSONL=/path/to/audit.jsonl`
- `SPYRE_ALLOW_ALL_OPS_IN_LX_PLANNING=1`

Added a guarded path in `compile_op_spec`:

- only considers `ReStickifyOpHBM`
- requires `SPYRE_RESTICKIFY_LOCALITY_ASSERT=1`
- requires a Stage 3B core mapping override
- requires both input and output SDSC args to be LX-resident
- emits `ReStickifyOpLx` data-op SDSC only if all checks pass
- otherwise preserves normal `ReStickifyOpHBM` compute SDSC and writes an audit
  row when requested

Also normalized known Deeptools data-op dimension labels so parsed OpSpecs using
`mb` / `out` can emit data-op SDSCs with `mb_` / `out_`, matching the standalone
data-op contract observed in prior stages.

## Validation

Static and focused tests:

```text
python -m py_compile torch_spyre/_inductor/codegen/superdsc.py \
  torch_spyre/_inductor/codegen/restickify_lx_dataop.py \
  torch_spyre/_inductor/config.py

python -m pytest tests/inductor/test_restickify_lx_dataop.py -q
5 passed
```

Known Stage 3B case:

```sh
LX_PLANNING=1 \
DXP_LX_FRAC_AVAIL=1 \
SPYRE_ALIGN_RESTICKIFY_CORE_MAPPING=1 \
SPYRE_ALIGN_RESTICKIFY_WORK_DISTRIBUTION=1 \
SPYRE_RESTICKIFY_LOCALITY_ASSERT=1 \
SPYRE_RESTICKIFY_LX_DATAOP_E2E=1 \
SPYRE_RESTICKIFY_LX_DATAOP_AUDIT_JSONL=/tmp/restickify-stage36-e2e-audit/frac1.jsonl \
python tools/restickify_scenario_probe.py \
  --case adds_then_matmul \
  --size 2048 \
  --ring-telemetry \
  --skip-correctness \
  --output-dir /tmp/restickify-stage36-e2e-frac1 \
  --fail-on-error
```

Result:

```text
ok size=2048 case=adds_then_matmul restickifies=2 bytes=16777216 byte_hops=0
```

Audit:

```json
{"status": "skipped", "reason": "missing-certified-core-mapping-override", "arg_allocations": [{}, {}]}
{"status": "skipped", "reason": "arg0-not-lx-resident", "arg_allocations": [{}, {}]}
```

Interpretation:

- The first restickify is not Stage 3B-eligible.
- The Stage 3B-certified restickify is still HBM-backed in the normal compiled
  graph, so replacing it with `ReStickifyOpLx` would be unsound.

## DXP_LX_FRAC_AVAIL Probe

`DXP_LX_FRAC_AVAIL=1` preserves all LX for Deeptools backend use under the
current allocator formula:

```python
frontend_lx_limit = 2MB * (1.0 - DXP_LX_FRAC_AVAIL)
```

So `DXP_LX_FRAC_AVAIL=1` gives the Torch-Spyre scratchpad planner no frontend
allocation budget and the restickifies remain HBM-backed.

`DXP_LX_FRAC_AVAIL=0` makes the planner more aggressive, but the known
`adds_then_matmul` graph then fails before an eligible replacement:

```text
Unable to map graph within architecture constraints:
The initial chunk parameters must fit in LX for SuperDSC: 0_ReStickifyOpHBM
```

The same failure appeared for smaller `128` and `512` sizes when forcing all ops
into LX, because the first restickify in the graph is not Stage 3B-certified and
remains `ReStickifyOpHBM`.

## Synthetic Bundle Probe

To isolate the bundle path from scratchpad planning, we synthesized a single
LX-resident, locality-certified restickify `OpSpec` and passed it through normal
`generate_bundle`.

The new e2e hook emitted a data-op SDSC:

```text
sdsc_0_ReStickifyOpLx_dataop.json
```

The audit showed emission:

```json
{
  "status": "emitted",
  "reason": null,
  "op": "ReStickifyOpHBM",
  "arg_allocations": [{"lx": 0}, {"lx": 65536}],
  "has_core_mapping_override": true
}
```

But DXP rejected the bundle:

```text
DtException: Datadsc not allowed, use dldsc
file /project_src/deeptools/dxp/dxp.cpp line 489
```

## Conclusion

The current production bundle path cannot consume `datadscs_` directly. The
Stage 35 standalone evidence remains valid as a Deeptools data-op contract
probe, but Stage 36 shows that normal Torch-Spyre execution needs a DLDSc-based
integration path before an e2e `ReStickifyOpLx` replacement can run.

## Next Step

The next engineering task is not more Stage 3B tuning. It is to find or build
the DLDSc equivalent of the data-op prototype:

1. Locate current DLDSc/data-op lowering entry points in Deeptools/Torch-Spyre.
2. Translate the single-dataop `ReStickifyOpLx` JSON contract into the DLDSc form
   DXP accepts.
3. Re-run the synthetic bundle probe until `dxp_standalone --bundle` accepts it.
4. Only then revisit the real `adds_then_matmul` graph, likely with a graph that
   avoids the first non-certified `ReStickifyOpHBM` or with a broader replacement
   strategy.

This keeps the claim honest: LX-resident restickify is representable in
Deeptools data-op form, but the production bundle path currently requires DLDSc.
