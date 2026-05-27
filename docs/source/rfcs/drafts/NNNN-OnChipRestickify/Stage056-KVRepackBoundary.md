# Stage 056: K/V Repack Boundary Diagnostic

Date: 2026-05-27

## Purpose

Stage055 showed that the real block64 graph has the independent future edge we
wanted to hoist:

```text
3_ReStickifyOpHBM -> 4_batchmatmul input1
```

but that edge is not a same-core layout-transform copy.  The producer emits a
low-core K/V operand, while the future `batchmatmul` is a 32-core consumer split
over query rows (`mb_`).  The K/V operand layout does not contain `mb_`, so the
missing primitive is a K/V repack or broadcast:

```text
2-core or 4-core ReStickifyOpHBM LX output
  -> duplicate/repack into the 32-core future batchmatmul input1 LX layout
```

Stage056 makes that boundary explicit in the fail-closed selector.  This keeps
the hoist probe from reporting a generic `invalid_split:mb_` and records the
actual next implementation contract.

## Change

When a nonzero layout-transform input fails because the consumer split dimension
is not part of the selected operand layout, `_flash_attention_layout_xform_pair_edge`
now checks whether the producer split maps cleanly into an operand dimension
that is divisible by the producer core count.  If so, it reports:

```text
input1:requires_kv_repack_broadcast:
  producer_split=mb_:mapped_split=x_:consumer_split=mb_:
  producer_cores=2:consumer_cores=32
```

The hoist scanner carries that through as:

```text
future_tile1:input1:requires_kv_repack_broadcast:...
```

No executable repack sidecar is emitted yet.  The current behavior remains
fail-closed.

## Validation

The new synthetic graph mirrors the real block64 K/V shape:

- current 32-core `batchmatmul`;
- future 2-core `ReStickifyOpHBM`;
- future 32-core `batchmatmul`;
- selected edge is future `input1`;
- producer split `mb_` maps to operand `x_`;
- consumer split `mb_` is absent from the K/V operand layout.

Local validation:

```text
python3 -m py_compile torch_spyre/_inductor/onchip_realize.py \
  tests/_inductor/test_onchip_realize_logic.py
python3 tests/_inductor/test_onchip_realize_logic.py
```

Result:

```text
test_onchip_realize_logic.py: 60/60 passed
```

Pod validation in:

```text
/home/adnan-cdx/dt-inductor-mixed/torch-spyre-stage039-two-sdsc-ifn
```

also passed:

```text
test_onchip_realize_logic.py: 60/60 passed
```

The block64 L128 device probe used:

```text
variant=layout_xform_hoist_auto
cache=/tmp/sdpa-stage056-kv-repack-boundary-layout_xform_hoist_auto-B1-H2-L128-D64-C0-639754-414580
```

The hoist warning now reports the real K/V boundary directly:

```text
future_tile1:input1:requires_kv_repack_broadcast:
  producer_split=mb_:mapped_split=x_:consumer_split=mb_:
  producer_cores=2:consumer_cores=32
```

No hoist sidecar was selected.  The run then hit the known raw block64 HBM path
value mismatch:

```text
Mismatched elements: 16286 / 16384 (99.4%)
Greatest absolute difference: nan at index (0, 1, 104, 16)
```

## Current Status

This stage does not complete the warp-specialized prefill attention path.  It
turns the next required primitive into a named, test-covered boundary.

The next executable probe should build a default-off K/V repack descriptor with:

```text
dataIN_L0:
  PieceInfo from the low-core ReStickifyOpHBM output
  PlacementInfo.memId = producer cores

dataOUT_L0:
  PieceInfo duplicated across all 32 future batchmatmul consumer cores
  PlacementInfo.memId = consumer cores

schedule:
  producer runs before the current tile, or remains an inserted predecessor
  current tile overlaps K/V repack with current batchmatmul
```

That probe must stay non-promoting until DXP accepts the descriptor and device
execution is value-correct.
