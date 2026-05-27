# Stage 040: On-Chip Layout-Transform Promotion Gate

Date: 2026-05-26

## Purpose

Stage039 made the composed `onchip_layout_xform` path value-correct, then proved
the current broad matrix manually on the pod.  This stage turns that matrix into a
repeatable promotion gate around `tools/onchip_sdpa_sweep.py`.

The gate is intentionally stricter than "all rows exited ok".  For every expected
row it checks:

- status is `ok`;
- shape and effective block size match the matrix;
- `max_abs_error <= 0.01`;
- mixed SDSC count is at least the Stage039 floor for that shape; and
- a layout-transform consumer sidecar appears for rows that have an eligible
  layout-transform edge.

Two rows are still useful coverage but do not have an eligible layout-transform
pair: `B1 H2 D64 L64 block=64` and `B1 H2 D64 L128 block=128`.  The gate keeps
them in the matrix, but only requires pointwise/mixed coverage for those rows.

## Tool

Added:

```text
tools/onchip_sdpa_promotion_gate.py
tests/_inductor/test_onchip_sdpa_promotion_gate_logic.py
```

Default command:

```sh
python3 tools/onchip_sdpa_promotion_gate.py \
  --gate onchip_layout_xform \
  --variant onchip_layout_xform \
  --case-output-dir /tmp/sdpa-stage040-layout-xform-gate-json \
  --cache-prefix /tmp/sdpa-stage040-layout-xform-gate \
  --output-json /tmp/sdpa-stage040-layout-xform-gate.json
```

The encoded matrix is:

| Shape | Block | Lengths |
| --- | --- | --- |
| B1 H2 D64 | 64 | 64, 128, 256, 384, 512 |
| B2 H2 D64 | 64 | 128, 256 |
| B1 H4 D64 | 64 | 128, 256 |
| B1 H2 D128 | 64 | 128, 256 |
| B1 H2 D64 | 128 | 128, 256, 512 |

## Pod Validation

Pod:

```text
adnan-cdx-spyre-dev-pf
DTI_PROJECT_ROOT=/home/adnan-cdx/dt-inductor-mixed
worktree=/home/adnan-cdx/dt-inductor-mixed/torch-spyre-stage039-two-sdsc-ifn
PATCHED_DXP=/home/adnan-cdx/dt-inductor-mixed/build/deeptools-onchip-foundation-clean/dxp/dxp_standalone
```

The first full run executed all five cases and all fourteen rows were
value-correct.  After tightening the validator to model the two no-layout-pair
rows explicitly, the saved per-case JSON files revalidated cleanly:

```sh
"$PYTHON" tools/onchip_sdpa_promotion_gate.py \
  --gate onchip_layout_xform \
  --variant onchip_layout_xform \
  --reuse-existing \
  --case-output-dir /tmp/sdpa-stage040-layout-xform-gate-json \
  --cache-prefix /tmp/sdpa-stage040-layout-xform-gate \
  --output-json /tmp/sdpa-stage040-layout-xform-gate-revalidated.json
```

Result:

```text
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
B1 H2 L64  D64  block=64  mixed=6  layout_consumer=0 median=0.247305 max_err=0.00732422
B1 H2 L512 D64  block=64  mixed=39 layout_consumer=1 median=0.700361 max_err=0.00244141
B2 H2 L256 D64  block=64  mixed=15 layout_consumer=1 median=0.560191 max_err=0.00317383
B1 H4 L256 D64  block=64  mixed=19 layout_consumer=1 median=0.491405 max_err=0.00317383
B1 H2 L256 D128 block=64  mixed=19 layout_consumer=1 median=0.458186 max_err=0.00585938
B1 H2 L512 D64  block=128 mixed=19 layout_consumer=1 median=0.593640 max_err=0.00195312
```

## Local Validation

```text
tests/_inductor/test_onchip_sdpa_promotion_gate_logic.py 5/5 passed
py_compile(onchip_sdpa_promotion_gate.py, test_onchip_sdpa_promotion_gate_logic.py) passed
git diff --check passed
```

## Interpretation

The composed layout-transform path now has a durable promotion gate for the
current matrix.  This is still not a default enablement proof for
`SPYRE_FLASH_ATTENTION_ONCHIP_SDPA`: causal/masking coverage, longer lengths, and
larger batch/head/depth stress are still outside the matrix.

## Next

- wire this gate into whatever device CI lane can run patched-DXP SDPA sweeps;
- keep extending the matrix before defaulting layout-transform pair selection
  under `SPYRE_FLASH_ATTENTION_ONCHIP_SDPA`;
- continue tracking the DXP predecessor-generated IFN path separately from the
  explicit LX-copy sidecar path.
