# Stage 50: DDL Bridge End-to-End Validation Attempt

## Summary

Stage 50 tried to move from the Stage 49 DDL-template contract into a real
Torch-Spyre end-to-end path.

The most important result is mixed:

1. A real device execution smoke did not complete.
2. The failing smoke did not actually exercise the DDL bridge.
3. Artifact-only validation on real Torch-Spyre-generated restickify SDSCs now
   passes DDC, DCC, and DXP with the Stage 49 template spelling, and the final
   DXP `senprog.txt` contains no HBM work tokens.

So we have a stronger compiler-contract result, but not a numerical correctness
result yet.

## Template Patch Under Test

The temporary Deeptools share was created by copying:

```text
/opt/ibm/spyre/deeptools/share
```

to:

```text
/tmp/stage50-template-share
```

and patching only:

```text
/tmp/stage50-template-share/ddc/ddl_templates/restickify.ddl
```

from:

```ddl
%src_inp_lxsfp =
  ddl.unit(%inptensor, %inptensor_lx_allocation)
    {unit="lxlu", data_connect="lxlu_input"}
```

to:

```ddl
%src_inp_lxsfp =
  ddl.unit(%inptensor)
    {unit="lxlu", data_connect="sfp_input"}
```

This is the Stage 49 passing variant.

## Device Smoke Attempt

We attempted a small Torch-Spyre run:

```sh
DEEPTOOLS_PATH=/tmp/stage50-template-share \
SPYRE_RESTICKIFY_DDL_BRIDGE_E2E=1 \
SPYRE_RESTICKIFY_DDL_BRIDGE_PREDDC_SHIM=1 \
SPYRE_RESTICKIFY_DDL_BRIDGE_AUDIT_JSONL=/tmp/stage50-ddl-e2e/smoke/audit.jsonl \
python3 tools/restickify_scenario_probe.py \
  --case adds_then_matmul \
  --size 512 \
  --ring-telemetry \
  --output-dir /tmp/stage50-ddl-e2e/smoke \
  --fail-on-error
```

The run timed out in hardware execution:

```text
Timed-out: PendingRequest(... PipelineId(COMPUTE) ...)
Compute CB hardware error detected
Fail on RB time-out
```

But the audit file shows the DDL bridge did not fire:

```json
{"status":"skipped","reason":"source-not-in-graph-computed","source_kind":"graph_input_or_weight","source_name":"arg1_1","work_slices":{"mb":8,"out":4}}
{"status":"skipped","reason":"expected-one-split-dim","source_kind":"in_graph_computed","source_name":"buf1","work_slices":{"mb":8,"out":4}}
```

That means the timeout is not evidence against the DDL bridge. This 512 case
stayed on the normal generated path.

## Artifact-Only Validation

To avoid more hardware timeouts, we next validated real cached 2048
Torch-Spyre-generated restickify SDSCs through the compiler toolchain only.

Inputs:

```text
/tmp/torchinductor_1000800000/tmpc91aha5k/inductor-spyre/sdsc_fused_add_t_0_vs3e120g/sdsc_0_ReStickifyOpHBM.json
/tmp/torchinductor_1000800000/tmpc91aha5k/inductor-spyre/sdsc_fused_mm_1_lkf0ek44/sdsc_0_ReStickifyOpHBM.json
```

Command shape:

```sh
DEEPTOOLS_PATH=/tmp/stage50-template-share \
python3 tools/restickify_torch_spyre_ddl_bridge_probe.py \
  --sdsc <sdsc_0_ReStickifyOpHBM.json> \
  --output-dir /tmp/stage50-ddl-e2e/<artifact> \
  --deeptools-bin /opt/ibm/spyre/deeptools/bin \
  --senarch rcudd1a \
  --run-deeptools \
  --run-dxp-preload
```

This path performs:

```text
Torch-Spyre ReStickifyOpHBM SDSC
  -> synthesized LX-local DDL bridge SDSC
  -> DDC
  -> DCC
  -> DXP with the Stage 41 pre-DDC shim
```

No device execution is involved.

## Results

| Artifact | Source allocation | Synth allocation | DDC | DCC | DXP | Final HBM tokens | Final active units |
|---|---|---|---:|---:|---:|---:|---|
| add-side restickify | `hbm:2` | `lx:2` | pass | pass | pass | `0` | `LXSU:32`, `SFP:896`, `PT:8928` |
| matmul-side restickify | `hbm:2` | `lx:2` | pass | pass | pass | `0` | `LXSU:32`, `SFP:896`, `PT:8928` |

The source SDSCs were normal generated `ReStickifyOpHBM` kernels with HBM
allocations:

```text
source allocate_components = {"hbm": 2}
source labeled_ds mem_org = [["hbm", "lx"], ["hbm", "lx"]]
```

The synthesized bridge SDSCs use LX-local allocation at the Torch-Spyre bridge
input:

```text
synthesized allocate_components = {"lx": 2}
synthesized labeled_ds mem_org = [["lx"], ["lx"]]
```

After DDC, Deeptools expands the bridge into internal LX/SFP/PT storage:

```text
ddc output allocate_components = {"l0": 1, "lx": 3, "ptxrf": 1, "sfplrf": 4}
```

The final DXP `senprog.txt` summaries were identical for both artifacts:

```json
{"HBM":0,"L3LU":0,"L3SU":0,"LXLU":0,"LXSU":32,"PT":8928,"SFP":896}
```

DXP still emits empty lowercase `l3lu`/`l3su` program stubs with immediate
returns, but there are no uppercase HBM/L3 work tokens in the summarized
program.

Local copied summaries:

```text
artifacts/stage50_ddl_bridge_e2e/artifact_add_summary.json
artifacts/stage50_ddl_bridge_e2e/artifact_mm_summary.json
artifacts/stage50_ddl_bridge_e2e/smoke_audit.jsonl
```

## Interpretation

This is the strongest result so far for the DDL bridge contract:

```text
real Torch-Spyre restickify SDSC
  -> synthesized LX-local bridge
  -> patched restickify.ddl
  -> DDC/DCC/DXP success
  -> final generated program with no HBM work tokens
```

It still does not prove correctness.

The missing proof is semantic: we must show that the generated program reads the
intended producer values from the preceding op's LX-resident output. The artifact
probe proves that Deeptools can lower the shape of program we want; it does not
yet prove the runtime binding between producer output and restickify bridge input.

## Current Blocker

The immediate e2e blocker is not DCC anymore. It is getting a Torch-Spyre graph
case that:

1. emits an eligible DDL-bridge restickify,
2. avoids unrelated graph-input/HBM restickify rows,
3. runs on hardware without the unrelated 512 timeout, and
4. validates numerical parity against CPU.

The 512 smoke is not a good candidate because the DDL bridge audit skipped both
restickifies.

## Next Step

The next useful experiment should be compile-artifact-first:

1. Generate or find a 2048 `adds_then_matmul` case whose audit marks the
   in-graph restickify as DDL-bridge eligible.
2. Confirm the emitted `_ddl_bridge` SDSC follows the same DDC/DCC/DXP behavior
   as the artifact-only probe above.
3. Only then attempt device execution.
4. If device execution fails, reduce the graph while preserving bridge
   eligibility, instead of falling back to the 512 skipped case.

If that still fails, the question for Deeptools/Flex owners becomes precise:

```text
Given an LX-local restickify bridge SDSC that DDC/DCC/DXP can lower without HBM,
what runtime binding is required so its INPUT labeled DS aliases the previous
op's LX-resident OUTPUT labeled DS?
```

