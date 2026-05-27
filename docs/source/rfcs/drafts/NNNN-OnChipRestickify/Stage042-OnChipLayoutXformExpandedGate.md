# Stage 042: On-Chip Layout-Transform Expanded Gate

Date: 2026-05-27

## Purpose

Stage041 wired the layout-transform path to a production-shaped master adjunct:

```text
SPYRE_FLASH_ATTENTION_ONCHIP_SDPA=1
SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM=1
```

This stage expands the promotion gate beyond the first matrix before considering
default enablement.  The new coverage adds:

- long sequence lengths `L=768` and `L=1024` for `B1 H2 D64 block=64`; and
- a combined stress case `B2 H4 D128 block=64` at `L=128,256`.

Causal/masked SDPA remains outside this gate.  The current flash-prefill
predicate intentionally returns false when `attn_bias is not None` or
`is_causal=True`, so those modes route through the non-flash fallback today.

## Implementation

Updated:

```text
tools/onchip_sdpa_promotion_gate.py
tests/_inductor/test_onchip_sdpa_promotion_gate_logic.py
```

The default `onchip_layout_xform` promotion gate now has seven cases and eighteen
rows:

| Shape | Block | Lengths |
| --- | --- | --- |
| B1 H2 D64 | 64 | 64, 128, 256, 384, 512 |
| B2 H2 D64 | 64 | 128, 256 |
| B2 H4 D128 | 64 | 128, 256 |
| B1 H4 D64 | 64 | 128, 256 |
| B1 H2 D128 | 64 | 128, 256 |
| B1 H2 D64 | 128 | 128, 256, 512 |
| B1 H2 D64 | 64 | 768, 1024 |

The added rows require the same invariants as the previous matrix: `status=ok`,
shape/block match, `max_abs_error <= 0.01`, a mixed-SDSC count floor, and a
layout-transform consumer sidecar for rows with an eligible layout-transform
edge.

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
"$PYTHON" tests/_inductor/test_onchip_sdpa_promotion_gate_logic.py

"$PYTHON" tools/onchip_sdpa_promotion_gate.py \
  --gate onchip_layout_xform \
  --case-output-dir /tmp/sdpa-stage042-expanded-layout-xform-gate-json \
  --cache-prefix /tmp/sdpa-stage042-expanded-layout-xform-gate \
  --timeout-s 700 \
  --output-json /tmp/sdpa-stage042-expanded-layout-xform-gate.json
```

Result:

```text
tests/_inductor/test_onchip_sdpa_promotion_gate_logic.py 5/5 passed
PROMOTION_GATE_PASSED gate=onchip_layout_xform cases=7 rows=18
```

Aggregate evidence:

```text
rows=18
max_err=0.00732421875
mixed_minmax=3..78
```

New representative rows:

```text
B2 H4 L128  D128 block=64 mixed=7  layout_consumer=1 median=0.450583 max_err=0.00634766
B2 H4 L256  D128 block=64 mixed=15 layout_consumer=1 median=0.854025 max_err=0.00323486
B1 H2 L768  D64  block=64 mixed=59 layout_consumer=1 median=1.083863 max_err=0.00183105
B1 H2 L1024 D64  block=64 mixed=78 layout_consumer=1 median=1.467725 max_err=0.00134277
```

## Local Validation

```text
tests/_inductor/test_onchip_sdpa_promotion_gate_logic.py 5/5 passed
py_compile(onchip_sdpa_promotion_gate.py, test_onchip_sdpa_promotion_gate_logic.py) passed
git diff --check passed
```

## Interpretation

The opt-in master layout-transform path now has longer-sequence evidence and a
combined batch/head/depth stress case.  The remaining default-enable blockers are
more about coverage scope than the mechanics already proven here:

- causal/masked SDPA flash-prefill support is still not implemented;
- no rows beyond `L=1024` are certified; and
- broader combined stress, for example larger `B/H/D` with longer `L`, remains
  untested.

## Next

- implement or explicitly scope causal/masked flash-prefill support;
- add a larger long-shape stress case if device time allows; and
- once coverage expectations are settled, decide whether the layout-transform
  adjunct should become part of plain `SPYRE_FLASH_ATTENTION_ONCHIP_SDPA`.
