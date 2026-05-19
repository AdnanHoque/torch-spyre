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
