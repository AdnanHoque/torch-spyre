# Stage 067: Overlap Candidate Diagnostics

Date: 2026-05-27

## Purpose

Stage066 added HBM-KV-safe lookahead and hoist sweep rows, then showed that the
real H8/L256 graph falls back to the serial layout-transform pair.  The fallback
reason was only visible after a manual cache inspection.

Stage067 makes that information first-class in the sweep JSON.  Every successful
sweep row now includes a bounded `layout_xform_candidates` summary, so future
device runs directly report whether query-side lookahead, hoist, or the serial
pair is selectable for each generated flash bundle.

## Implementation

`tools/onchip_sdpa_sweep.py` now scans non-mixed `sdsc_*.json` files in each
generated bundle directory and calls the existing torch-free rejection helpers:

```text
flash_attention_layout_xform_lookahead_rejection_reasons(...)
flash_attention_layout_xform_hoist_rejection_reasons(...)
flash_attention_layout_xform_pair_rejection_reasons(...)
```

The emitted JSON shape is:

```text
layout_xform_candidates: [
  {
    dir: "...",
    sdscs: 62,
    lookahead_selectable: false,
    lookahead_rejections: {count: 11, first: [...], truncated: false},
    hoist_selectable: false,
    hoist_rejections: {count: 43, first: [...], truncated: true},
    pair_selectable: true,
    pair_rejections: {count: 0, first: [], truncated: false},
  }
]
```

Rejection lists are capped at twelve entries per builder.  The full count and a
`truncated` flag remain present, so the summary stays compact without hiding
whether the scan found many candidate failures.

## Device Result

Run:

```text
B=1, H=8, L=256, D=64, block=64, causal=0, seed=0
variant = onchip_hbm_kv_layout_xform_hoist
cache prefix = /tmp/sdpa-stage067-diagnostics
output = /tmp/sdpa-stage067-diagnostics.json
```

Result:

```text
status = ok
median = 0.560703 ms
max_abs_error = 0.00488281
mixed_sdscs = 19
```

The long flash bundle reports:

```text
dir = inductor-spyre/sdsc_fused__scaled_dot_product_fused_attention_overrideable_1_9rawmvpt
sdscs = 62
pair_selectable = true
lookahead_selectable = false
hoist_selectable = false
```

Representative lookahead blockers:

```text
tile2:future_tile3:input0:no_latest_producer
tile2:future_tile4:producer_not_ready:producer=29:current=15
tile2:future_tile5:input0:no_latest_producer
tile2:future_tile6:producer_not_ready:producer=44:current=15
```

Representative hoist blockers:

```text
tile0:future_tile3:input1:requires_kv_repack_broadcast:
  producer_split=mb_
  mapped_split=x_
  consumer_split=mb_
  producer_cores=8
  consumer_cores=32

tile0:future_tile5:input1:requires_kv_repack_broadcast:
  producer_split=mb_
  mapped_split=x_
  consumer_split=mb_
  producer_cores=8
  consumer_cores=32
```

The serial pair has no rejection reasons:

```text
pair_rejections.count = 0
```

## Interpretation

This is not a new overlap implementation yet, but it removes a slow feedback
loop.  We can now search across shapes and variants for a real query-side
lookahead selection or a hoistable producer without manually reconstructing
original bundle order from cache directories.

For the current H8/L256 neutral K/V shape, the JSON now directly proves the
same conclusion as Stage066:

- serial query-side layout-transform pair is legal and selected;
- query-side lookahead has no ready future input0 producer;
- hoistable future work is dominated by K/V input1 candidates that need an
  8-core to 32-core broadcast/staging contract;
- no evidence supports retrying the value-wrong raw LX-only K/V input1 path.

The next step is to run the HBM-KV lookahead/hoist rows across the promotion
shape matrix and use `layout_xform_candidates` to identify whether any real
query-side overlap candidate is selectable before designing another K/V staging
primitive.

## Verification

Local:

```text
python3 -m py_compile tools/onchip_sdpa_sweep.py tests/_inductor/test_onchip_sdpa_sweep_logic.py
python3 tests/_inductor/test_onchip_sdpa_sweep_logic.py  # 58/58 pass
```

Pod:

```text
python3 -m py_compile tools/onchip_sdpa_sweep.py tests/_inductor/test_onchip_sdpa_sweep_logic.py
python3 tests/_inductor/test_onchip_sdpa_sweep_logic.py  # 58/58 pass
```

Device:

```text
tools/onchip_sdpa_sweep.py \
  --lengths 256 \
  --variants onchip_hbm_kv_layout_xform_hoist \
  --batch 1 --heads 8 --dim 64 --block-size 64 \
  --seed -256 --warmup 1 --iters 2 --timeout-s 300 \
  --cache-prefix /tmp/sdpa-stage067-diagnostics \
  --output-json /tmp/sdpa-stage067-diagnostics.json
```
