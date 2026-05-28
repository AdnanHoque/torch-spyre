# Stage088 - Warpspec Layout Decoupling

## Change

The new `onchip_warpspec_kv_hbm_prefetch_loader_core31_decoupled` sweep variant
disables the layout-transform adjunct and requests only the serialized
loader-core K/V HBM prefetch schedule.

```text
SPYRE_FLASH_ATTENTION_ONCHIP_SDPA=1
SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM=0
SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_PAIR_TILE=-1
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_HOIST_TILE=-2
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_LOADER_CORE=31
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT=1
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT_FULL_TILE_PIECES=1
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_SERIALIZE_LOADER_CORE=1
```

## Why

The layout-transform pair is useful for some earlier on-chip layout experiments,
but it is not required for the loader-core prefetch schedule. Coupling the two
hid valid warpspec rows behind layout-pair numerical failures.

The decoupled path recovers:

```text
B1 H4 D64 block64 L768,L1024
B2 H4 D128 block64 L384,L512,L768,L1024
```

All recovered rows keep the core warpspec invariant:

```text
current_prefetch sidecar
opFuncsUsed contains STCDPOpHBM
loader_core == 31
loader fanout == true
full tile fanout pieces == true
serialize loader-core prefetch == true
```

## Gate Implication

The `onchip_warpspec_decoupled` promotion gate checks for the serialized
loader-core K/V prefetch artifact directly and does not require a
layout-transform consumer. The existing `onchip_warpspec` gate remains on the
layout-coupled default variant for rows that still rely on that path.

## Promotion Result

The exact final variant passed the focused hardware gate:

```text
PROMOTION_GATE_PASSED gate=onchip_warpspec_decoupled cases=2 rows=6
```

Per-row medians from the gate:

```text
B1 H4 L768  D64  block64: 1.781614 ms, max abs 0.00195312, mixed 59
B1 H4 L1024 D64  block64: 2.518849 ms, max abs 0.00268555, mixed 78
B2 H4 L384  D128 block64: 1.339696 ms, max abs 0.00390625, mixed 22
B2 H4 L512  D128 block64: 1.679618 ms, max abs 0.00317383, mixed 31
B2 H4 L768  D128 block64: 3.318368 ms, max abs 0.00366211, mixed 47
B2 H4 L1024 D128 block64: 5.053475 ms, max abs 0.00219727, mixed 63
```
