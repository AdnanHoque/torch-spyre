# Stage 041: On-Chip Layout-Transform Master Adjunct

Date: 2026-05-26

## Purpose

Stage040 made the layout-transform matrix repeatable, but it still exercised the
lower-level probe flag set directly.  This stage adds a production-shaped opt-in
adjunct to the existing on-chip SDPA master gate:

```text
SPYRE_FLASH_ATTENTION_ONCHIP_SDPA=1
SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM=1
```

The adjunct selects `layout_xform_pair_auto` for the composed path while leaving
the plain master gate unchanged.  This is not default enablement: users must still
opt into the layout-transform pair separately from
`SPYRE_FLASH_ATTENTION_ONCHIP_SDPA=1`.

## Implementation

Updated:

```text
torch_spyre/_inductor/config.py
```

The new config key is:

```text
flash_attention_onchip_sdpa_layout_xform
```

It is true only when both env vars are set.  When true, and when the lower-level
`SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_PAIR_TILE` override is absent,
the pair-tile config defaults to `-2` (auto selection).  The lower-level override
still wins for probe work.

Updated:

```text
tools/onchip_sdpa_sweep.py
```

Added:

```text
onchip_master_layout_xform:
  SPYRE_FLASH_ATTENTION_ONCHIP_SDPA=1
  SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM=1
```

The sweep harness can now clear a base env key by setting its variant value to
`None`.  The new variant uses that to remove
`SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_PAIR_TILE`, otherwise the base
`-1` would mask the config adjunct and disable auto selection.

Updated:

```text
tools/onchip_sdpa_promotion_gate.py
```

The `onchip_layout_xform` promotion gate now defaults to the production-shaped
`onchip_master_layout_xform` sweep variant.  Stage040's historical lower-level
probe validation remains reproducible by passing `--variant onchip_layout_xform`.

## Tests

Updated:

```text
tests/_inductor/test_config_logic.py
tests/_inductor/test_onchip_sdpa_sweep_logic.py
tests/_inductor/test_onchip_sdpa_promotion_gate_logic.py
```

The config tests cover:

- master gate alone keeps layout-transform auto selection disabled;
- the adjunct requires the master gate;
- master plus adjunct selects pair tile `-2`; and
- explicit lower-level pair-tile envs still work.

The sweep tests cover inherited-env protection: if the parent shell has
`SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_PAIR_TILE` set, the
`onchip_master_layout_xform` child env still removes it so config can provide the
adjunct default.

## Pod Validation

Pod:

```text
adnan-cdx-spyre-dev-pf
DTI_PROJECT_ROOT=/home/adnan-cdx/dt-inductor-mixed
worktree=/home/adnan-cdx/dt-inductor-mixed/torch-spyre-stage039-two-sdsc-ifn
PATCHED_DXP=/home/adnan-cdx/dt-inductor-mixed/build/deeptools-onchip-foundation-clean/dxp/dxp_standalone
```

Command:

```sh
"$PYTHON" tests/_inductor/test_config_logic.py

"$PYTHON" tools/onchip_sdpa_promotion_gate.py \
  --gate onchip_layout_xform \
  --case-output-dir /tmp/sdpa-stage041-master-layout-xform-gate-json2 \
  --cache-prefix /tmp/sdpa-stage041-master-layout-xform-gate2 \
  --output-json /tmp/sdpa-stage041-master-layout-xform-gate2.json
```

Result:

```text
tests/_inductor/test_config_logic.py 7/7 passed
PROMOTION_GATE_PASSED gate=onchip_layout_xform cases=5 rows=14
```

Aggregate evidence:

```text
rows=14
max_err=0.00732421875
mixed_minmax=3..39
layout-transform consumer rows=12/14
```

Representative rows:

```text
B1 H2 L64  D64  block=64  mixed=6  layout_consumer=0 median=0.223923 max_err=0.00732422
B1 H2 L512 D64  block=64  mixed=39 layout_consumer=1 median=0.698440 max_err=0.00244141
B2 H2 L256 D64  block=64  mixed=15 layout_consumer=1 median=0.573520 max_err=0.00317383
B1 H4 L256 D64  block=64  mixed=19 layout_consumer=1 median=0.482594 max_err=0.00317383
B1 H2 L256 D128 block=64  mixed=19 layout_consumer=1 median=0.476648 max_err=0.00585938
B1 H2 L512 D64  block=128 mixed=19 layout_consumer=1 median=0.591151 max_err=0.00195312
```

## Local Validation

```text
tests/_inductor/test_config_logic.py                  7/7 passed
tests/_inductor/test_onchip_sdpa_sweep_logic.py       2/2 passed
tests/_inductor/test_onchip_sdpa_promotion_gate_logic.py 5/5 passed
tests/_inductor/test_onchip_realize_logic.py         46/46 passed
py_compile(config.py, bundle.py, onchip_realize.py, tools, tests) passed
git diff --check passed
```

## Interpretation

The layout-transform path is now wired to the production-shaped master switch as
an explicit adjunct and verified through the same 14-row promotion matrix.  This
is the right step before defaulting: the code path is no longer only a low-level
probe, but the default master behavior remains conservative.

## Next

- expand the gate beyond the current matrix (causal/masking variants, longer L,
  and larger batch/head/depth stress);
- once that evidence is in, consider defaulting the adjunct under
  `SPYRE_FLASH_ATTENTION_ONCHIP_SDPA`;
- keep the DXP predecessor-generated IFN path separate from this explicit
  LX-copy layout-transform path.
