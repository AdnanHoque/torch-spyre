# Stage 125: Boundary-Unit Patch Goal

## Active Goal

Move the LX-to-LX restickify prototype from "replace the middle restickify
SDSC" to "patch the adjacent boundary as one unit":

```text
producer output LX
  -> DDL bridge input LX/LXLU
  -> DDL bridge output LX/LXSU
  -> consumer input LX/LXLU
```

The consumer must not describe the restickified logical tensor as an HBM-backed
input. If it does, DXP regenerates a normal `hbm -> lx` reload and the bridge's
LX output is not the value consumed by the following op.

## Prototype Change

I added a default-off codegen flag:

```sh
SPYRE_RESTICKIFY_DDL_BRIDGE_BOUNDARY_PATCH=1
```

When enabled together with the existing DDL bridge prototype, bundle generation
now patches eligible adjacent triples before writing SDSC files:

1. producer output LDS is marked LX-only;
2. DDL bridge input allocation aliases the producer output start map;
3. DDL bridge output remains LX-only;
4. consumer matching input LDS is marked LX-only and uses the bridge output
   start map;
5. the bridge transfer offsets are updated so the DDL bridge reads from the
   producer-side map and writes to the consumer-side map.

This is still a prototype. It does not prove the final hardware stream is
value-correct yet, but it makes the intended contract visible in generated
Torch-Spyre SDSC JSON rather than relying only on a launch-time patch script.

## Why This Is The Right Next Step

Stage 124 showed the bridge frame itself can emit and retire, but correctness
fails because the consumer still reloads the same logical tensor from HBM:

```text
consumer:
  transfer_lds1_src:hbm_dst:lx
  transfer_lds1_src:lxlu_dst:sfp
```

The next experiment must therefore inspect the generated/scheduled consumer
after this patch. Success at this stage is not yet "full model speedup"; it is:

```text
same-bundle add/restickify/add fixture
  + DDL bridge emitted
  + no consumer hbm_dst:lx reload for the restickified input
  + consumer reads that input via lxlu_dst:sfp
  + bundle retires without stream error
  + CPU parity passes
```

## Validation Commands

Static/local:

```sh
python3 -m py_compile \
  torch_spyre/_inductor/codegen/restickify_lx_boundary.py \
  torch_spyre/_inductor/codegen/bundle.py \
  tests/inductor/test_restickify_ddl_bridge.py
```

Pod unit tests:

```sh
cd $DTI_PROJECT_ROOT/torch-spyre
python3 -m pytest tests/inductor/test_restickify_ddl_bridge.py -q
```

Pod runtime probe:

```sh
SPYRE_RESTICKIFY_DDL_BRIDGE_E2E=1 \
SPYRE_RESTICKIFY_DDL_BRIDGE_PREDDC_SHIM=1 \
SPYRE_RESTICKIFY_DDL_BRIDGE_SOURCE_ADDRESS=compact-lxlu \
SPYRE_RESTICKIFY_DDL_BRIDGE_BOUNDARY_PATCH=1 \
SPYRE_RESTICKIFY_DDL_BRIDGE_AUDIT_JSONL=/tmp/stage125-boundary/audit.jsonl \
python tools/restickify_scenario_probe.py \
  --case computed_transpose_adds_then_matmul \
  --size 2048 \
  --kernel-launch-log \
  --copy-kernel-code \
  --output-dir /tmp/stage125-boundary \
  --fail-on-error
```

If correctness still fails, inspect the copied kernel code:

```sh
rg -n 'hbm_dst:lx|lxlu_dst:sfp|ReStickifyOpHBM|ddl_bridge' \
  /tmp/stage125-boundary/kernel_code
```

## Expected Outcomes

- If the consumer still has `hbm_dst:lx`, the pre-DXP JSON patch is incomplete.
- If `hbm_dst:lx` disappears but correctness fails, the next problem is address
  identity or synchronization between the bridge output and consumer input.
- If the bundle hits a stream hardware error, the next problem is likely the
  control-block/runtime packaging contract rather than the SDSC-level memory
  annotation.

## Pod Results

Unit validation in the disposable pod checkout passed:

```text
python3 -m pytest tests/inductor/test_restickify_ddl_bridge.py -q
15 passed in 0.06s
```

Running the codegen boundary patch with the real Stage125 code imported via
`PYTHONPATH` no longer leaves the consumer input HBM-backed in the raw generated
SDSC:

```text
sdsc_2_add:
  Tensor1 INPUT mem ['lx']
  allocate-Tensor1_lx component lx
```

The raw copied bundle has no `hbm_dst:lx` transfer for that consumer input.
However, the end-to-end correctness run still fails:

```text
Mismatched elements: 3637417 / 4194304 (86.7%)
```

So the Stage125 pre-DXP patch expresses the intended memory organization, but
it is not sufficient for value correctness. The likely reason is that raw SDSC
allocation addresses are not the final scheduled producer `LXSU` and consumer
`LXLU` addresses.

I also reran the older launch-time boundary stitch path, which discovers
scheduled producer/consumer LX maps before patching:

```sh
SPYRE_RESTICKIFY_LX_BOUNDARY_MATCH_CONSUMER=1
```

That failed in Deeptools on the final DXP rerun:

```text
DtException: Different cardinality between json and caller
```

The discovered scheduled maps had `corelet factor = 2` and 64 entries, while
other allocations in the same consumer SDSC still had `corelet factor = 1` and
32 entries. I added a probe-only option:

```sh
SPYRE_RESTICKIFY_LX_BOUNDARY_COLLAPSE_CORELETS=1
```

which collapses transplanted scheduled maps back to corelet 0. That moved the
failure, but did not solve it:

```text
DtException: child_ff_vec_.size() > dim_index
```

## Current Blocker

We have now separated the problem into two layers:

1. **Raw codegen boundary patch**: DXP accepts and launches the bundle, but the
   result is numerically wrong.
2. **Scheduled-address stitch**: uses more plausible LX addresses, but DXP
   rejects the patched bundle because the transplanted start-address fold maps
   are not globally consistent with the surrounding SDSC.

The next useful step is to stop transplanting final scheduled maps directly
into otherwise unscheduled SDSC JSON. Either:

- build a first-class internal-edge schedule object before DXP finalizes all
  three SDSCs; or
- patch at a lower runtime-frame level after DXP, where the generated frame
  cardinalities are already fixed.
