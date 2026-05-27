# Stage 044: Full Causal Gate and Native-Mask Probes

Date: 2026-05-27

## Purpose

Stage043 added square causal flash-prefill support and a causal promotion-gate
case, but only the causal rows had been smoke-tested on the pod.  This stage
runs the full expanded gate with the causal case included, then records the
attempts to remove the remaining `aten.triu.default` CPU fallback from causal
mask construction.

## Full Gate Validation

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
  --case-output-dir /tmp/sdpa-stage044-full-layout-xform-causal-gate-json \
  --cache-prefix /tmp/sdpa-stage044-full-layout-xform-causal-gate \
  --timeout-s 700 \
  --output-json /tmp/sdpa-stage044-full-layout-xform-causal-gate.json
```

Result:

```text
tests/_inductor/test_onchip_sdpa_promotion_gate_logic.py 7/7 passed
PROMOTION_GATE_PASSED gate=onchip_layout_xform cases=8 rows=20
```

Aggregate evidence:

```text
rows=20
max_err=0.00732421875
mixed_minmax=3..78
causal_rows=2
```

Representative rows:

```text
B1 H2 L64   D64  C0 block=64  mixed=6  median=0.225689 max_err=0.0073242188
B1 H2 L512  D64  C0 block=64  mixed=39 median=0.697589 max_err=0.0024414062
B1 H2 L128  D64  C1 block=64  mixed=8  median=0.516252 max_err=0.005859375
B1 H2 L256  D64  C1 block=64  mixed=16 median=0.676267 max_err=0.00390625
B2 H4 L128  D128 C0 block=64  mixed=7  median=0.477624 max_err=0.0063476562
B2 H4 L256  D128 C0 block=64  mixed=15 median=0.867500 max_err=0.0032348633
B1 H2 L512  D64  C0 block=128 mixed=19 median=0.595878 max_err=0.001953125
B1 H2 L768  D64  C0 block=64  mixed=59 median=1.101866 max_err=0.0018310547
B1 H2 L1024 D64  C0 block=64  mixed=78 median=1.447286 max_err=0.0013427734
```

## Native-Mask Replacement Probes

The current causal implementation is still the Stage043 triangular additive bias:

```text
torch.full((L, L), -inf, device=query.device, dtype=query.dtype).triu(diagonal=1)
```

That path is value-correct and now passes the full gate, but the unit test still
reports:

```text
FallbackWarning: aten.triu.default is falling back to cpu
```

Several device-native replacements were tried on the pod and rejected by the
current compiler/layout constraints:

| Probe | Shape | Failure |
| --- | --- | --- |
| `arange` query/key comparison plus `masked_fill` | full or block-local mask | restickify could not resolve pointwise stick incompatibility |
| `zeros`/`full`/`cat`/`stack` block-local bias | `[block, block]` bias added to diagonal rows | `Unsupported: Unexpected stick expression 63` |
| Python-constructed block-local constant bias | `[L, block]` bias added to scores | constant buffer had `FixedLayout`, not `FixedTiledLayout` |
| `full_like(scores, -inf)` plus slice assignment of valid score regions | full score-shaped mask | AOT functional graph rejected generated `copy_` mutations |
| mask-free row-prefix diagonal update | row-wise valid key prefixes and `cat` recomposition | `Unsupported: Unexpected stick expression 63` |

These probes suggest that removing the causal CPU fallback is not just a local
decomposition rewrite.  The viable fix likely needs one of:

- a backend-supported triangular/causal mask primitive that produces
  `FixedTiledLayout`;
- layout propagation support for constant mask buffers in the score layout; or
- restickify support for the non-stick-aligned slices inherent in triangular
  score regions.

## Interpretation

The expanded on-chip layout-transform gate is now proven with the causal rows in
the default matrix.  Causal square prefill can be promoted as opt-in coverage,
but broad/default causal enablement should still keep the `aten.triu` fallback on
the blocker list until one of the backend-level mask paths above exists.

## Local Validation

```text
py_compile(decompositions.py) passed
git diff --check passed
```

## Next

- add a small backend/compiler issue or task for a tiled causal mask primitive;
- keep the causal rows in the promotion gate; and
- avoid replacing `triu` with decomposition-only mask rewrites until the layout
  constraints above are addressed.
