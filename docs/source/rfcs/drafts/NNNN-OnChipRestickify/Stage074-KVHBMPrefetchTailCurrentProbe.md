# Stage 074: K/V HBM Prefetch Tail-Current Probe

Date: 2026-05-28

## Purpose

Stage073 proved the serialized HBM K/V prefetch path can be value-correct, while
the true current-compute/future-prefetch overlap row corrupts attention output.
Stage074 narrows the hazard by moving the same future `STCDPOpHBM` prefetch to
the tail of the current mixed SDSC:

```text
native current input prologue
current attention compute
future K/V HBM prefetch
future attention consumer reads prefilled LX
```

This is not the final warp-specialized form, but it identifies the earliest
safe same-SDSC placement currently found.

## Implementation

Added a default-off diagnostic gate:

```text
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_TAIL_CURRENT=1
```

and sweep variant:

```text
onchip_hbm_kv_layout_xform_kv_hbm_prefetch_tail_current_probe
```

The generated current sidecar schedule is:

```text
[[0, -1, 0, 1], [-1, 0, 1, 1], [1, -1, 1, 0]]
```

where dataop 0 is the native-load prologue `nop`, compute DSC 0 is the current
attention tile, and dataop 1 is the future K/V `STCDPOpHBM` prefetch.

The sweep base environment now also clears:

```text
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_TAIL_CURRENT=0
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_LX_ROUNDTRIP=0
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_CORELET1=0
```

so stale shell state cannot leak between prefetch probes.

## Device Result

Run:

```text
tools/onchip_sdpa_sweep.py \
  --variants onchip_hbm_kv_layout_xform_kv_hbm_prefetch_tail_current_probe \
  --batch 1 --heads 8 --lengths 256 --dim 64 --block-size 64 \
  --warmup 1 --iters 2 --seed 0 --timeout-s 900 --dxp-debug \
  --cache-prefix /tmp/sdpa-stage130-tail-current \
  --output-json /tmp/sdpa-stage130-tail-current.json
```

Result:

```text
status = ok
median = 0.589672 ms
max_abs_error = 0.00463867
mixed_sdscs = 20
cache = /tmp/sdpa-stage130-tail-current-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_tail_current_probe-B1-H8-L256-D64-C0-752031-600405
```

## Contrast With Overlap Probes

The direct current-compute/future-prefetch overlap row fails deterministically:

```text
Mismatched elements: 5205 / 131072 (4.0%)
Greatest absolute difference: 0.74658203125 at (0, 1, 242, 11)
```

The `coreletId=1` plus LX-roundtrip probe lowered to SMC with `lxlu1/lxsu1`,
but failed with the same mismatch.  That rules out a simple LX-side corelet
routing fix.

Tail-current uses the same future HBM payload and the same future prefilled-LX
consumer contract, but passes.  The remaining incorrectness is therefore tied
to placing the future `L3_LDMU` in the true current attention compute overlap
window.

## Interpretation

Current evidence says:

- HBM K/V address/layout generation is correct.
- Cross-SDSC prefilled-LX lifetime is correct for this shape.
- Direct overlap with active attention compute is unsafe.
- Moving LX-side participation to corelet 1 is not sufficient, because L3 HBM
  traffic remains the shared resource.
- A tail slot after current compute is safe, but it does not yet provide the
  intended warp-specialized speedup.

## Next Direction

The next implementation step should target the actual resource conflict rather
than more Torch-side descriptor reshaping:

1. Add a lower-stack scheduling/resource model for `L3_LDMU` overlap with active
   attention PE/LX work, so unsafe rows are either delayed to a safe bubble or
   assigned to a truly independent path.
2. Alternatively, introduce a loader-core/core-to-core plan: load future K/V
   away from the current attention cores, then fan out the prefetched LX payload
   before the future consumer.

Tail-current should remain as a correctness-preserving diagnostic fallback while
we keep pushing toward real compute/prefetch overlap.
