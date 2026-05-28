# Stage 066: HBM-KV Overlap Probe Fallback

Date: 2026-05-27

## Purpose

Stage065 made the HBM-KV-safe composite path explicit in the promotion gate:
query-side layout-transform plus flash pointwise handoffs on chip, with K/V
input1 left on Foundation's HBM-backed `batchmatmul` path.

Stage066 asks the next overlap question without reopening the known-bad
same-input row-overlap probe:

```text
Can the HBM-KV-safe path request a lookahead or hoisted prefetch builder and
stay value-correct, while falling back to the serial layout-transform pair if
the real graph has no legal overlap candidate?
```

## New Sweep Rows

Two HBM-KV-safe diagnostic rows were added:

```text
onchip_hbm_kv_layout_xform_lookahead
onchip_hbm_kv_layout_xform_hoist
```

Both rows enable:

```text
SPYRE_FLASH_ATTENTION_ONCHIP_SDPA=1
SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM=1
SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_PAIR_TILE=<unset>
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PLAN_ARTIFACT=0
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_TILE=-1
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_TILE=-1
SPYRE_ONCHIP_HANDOFF_MIN_BYTES=0
```

The lookahead row additionally sets:

```text
SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_LOOKAHEAD_TILE=-2
```

The hoist row additionally sets:

```text
SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_HOIST_TILE=-2
```

The low-level pair tile env is removed so the master layout-transform adjunct
can still provide the serial pair fallback.  K/V-repack probes remain explicitly
disabled.

## Device Result

Run:

```text
B=1, H=8, L=256, D=64, block=64, causal=0, seed=0
warmup=1, iters=2
cache prefix: /tmp/sdpa-stage066-hbmkv-overlap
```

Results:

```text
variant                                 status  median_ms  max_abs_error  mixed
onchip_hbm_kv_layout_xform              pass    0.560277   0.00488281     19
onchip_hbm_kv_layout_xform_lookahead    pass    0.585820   0.00488281     19
onchip_hbm_kv_layout_xform_hoist        pass    0.589024   0.00488281     19
```

All three rows selected the same executed layout-transform sidecar:

```text
mixed_flash_layout_xform_pair_tile_2_consumer
source = generated-flash-prefill-layout-xform-pair-consumer
```

Neither diagnostic row emitted a `lookahead` or `hoist` sidecar.  Neither row
emitted any K/V-repack sidecar.

## Rejection Evidence

The long flash bundle for the lookahead row had no legal lookahead candidate.
Representative rejection reasons were:

```text
tile2:future_tile3:input0:no_latest_producer
tile2:future_tile4:producer_not_ready:producer=29:current=15
tile2:future_tile5:input0:no_latest_producer
tile2:future_tile6:producer_not_ready:producer=44:current=15
```

The hoist scan also found no legal candidate.  The independent future input0
producers are not ready before the current consumer, while the future input1
K/V candidates require a broadcast/repack shape:

```text
tile2:future_tile3:input1:requires_kv_repack_broadcast:
  producer_split=mb_
  mapped_split=x_
  consumer_split=mb_
  producer_cores=8
  consumer_cores=32

tile2:future_tile5:input1:requires_kv_repack_broadcast:
  producer_split=mb_
  mapped_split=x_
  consumer_split=mb_
  producer_cores=8
  consumer_cores=32
```

The serial pair scan still succeeds:

```text
flash_attention_layout_xform_pair_rejection_reasons(...) == []
```

## Interpretation

The HBM-KV-safe overlap probe variants are useful and safe, but they do not yet
exercise a runtime-overlapped sidecar on the real H8/L256 graph.  They pass
because the overlap-oriented builders fail closed and the serial
layout-transform pair remains available.

This result narrows the next implementation requirement:

- Same-input row overlap is still the wrong direction; Stage053 already showed
  the read-after-write hazard.
- Lookahead needs a future query-side layout-transform producer that is already
  ready before the current consumer.  The current H8 graph does not expose one.
- Hoist sees future K/V input1 candidates, but those require an AIU-supported
  K/V broadcast/staging contract from an 8-core producer to a 32-core consumer.
  The raw LX-only K/V input1 route remains disqualified by Stages061 and 062.

The next productive path is therefore either to find a real shape where the
query-side lookahead builder selects, or to design a backend-supported HBM-KV
staging/overlap contract that does not force `batchmatmul` input1 into the
value-wrong LX-only K/V path.

## Verification

Local:

```text
python3 -m py_compile tools/onchip_sdpa_sweep.py tests/_inductor/test_onchip_sdpa_sweep_logic.py
python3 tests/_inductor/test_onchip_sdpa_sweep_logic.py  # 57/57 pass
```

Pod:

```text
python3 -m py_compile tools/onchip_sdpa_sweep.py tests/_inductor/test_onchip_sdpa_sweep_logic.py
python3 tests/_inductor/test_onchip_sdpa_sweep_logic.py  # 57/57 pass
```

Device:

```text
tools/onchip_sdpa_sweep.py \
  --lengths 256 \
  --variants onchip_hbm_kv_layout_xform,onchip_hbm_kv_layout_xform_lookahead,onchip_hbm_kv_layout_xform_hoist \
  --batch 1 --heads 8 --dim 64 --block-size 64 \
  --seed -256 --warmup 1 --iters 2 --timeout-s 300 \
  --cache-prefix /tmp/sdpa-stage066-hbmkv-overlap \
  --output-json /tmp/sdpa-stage066-hbmkv-overlap.json
```
