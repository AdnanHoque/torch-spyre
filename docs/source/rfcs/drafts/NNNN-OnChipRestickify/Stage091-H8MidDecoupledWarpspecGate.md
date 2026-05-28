# Stage091 - H8 Mid-Length Decoupled Warpspec Gate

## Question

Can the layout-decoupled loader-specialized path safely cover the middle
`B1 H8 D64 block64` rows, while keeping the longer H8 boundary out of the
promotion gate?

The short answer is yes for L384 and L512, and no for L768/L1024.

## Evidence

Stage231 first checked the exact decoupled variant:

```text
B1 H8 D64 block64 L384:  ok, mixed=29
B1 H8 D64 block64 L512:  ok, mixed=39
B1 H8 D64 block64 L768:  failed
B1 H8 D64 block64 L1024: failed
```

Stage232 then probed the failing L768 row across the stack:

| Variant | Result | Median ms |
| --- | --- | ---: |
| `flash_hbm` | ok | 3.1902976334095 |
| `onchip_master` | failed | n/a |
| `onchip_hbm_kv_layout_xform` | failed | n/a |
| `onchip_warpspec_kv_hbm_prefetch_loader_core31` | failed | n/a |
| `onchip_warpspec_kv_hbm_prefetch_loader_core31_decoupled` | failed | n/a |
| `onchip_warpspec_kv_hbm_prefetch_loader_core31_decoupled_unicast` | failed | n/a |

That makes L768 a broader on-chip H8 boundary, not a failure unique to the
decoupled loader-prefetch sidecar.

Stage233 reran the passing mid rows with `warmup=2`, `iters=7`, and seed 42865:

| L | Variant | Median ms | Max abs error | Mixed SDSCs |
| ---: | --- | ---: | ---: | ---: |
| 384 | `flash_hbm` | 1.035977 | 0.00292969 | 0 |
| 384 | `onchip_master` | 0.955282 | 0.00292969 | 16 |
| 384 | `onchip_warpspec_kv_hbm_prefetch_loader_core31_decoupled` | 0.960641 | 0.00292969 | 29 |
| 384 | `onchip_warpspec_kv_hbm_prefetch_loader_core31_decoupled_unicast` | 0.967214 | 0.00292969 | 29 |
| 512 | `flash_hbm` | 1.458632 | 0.00305176 | 0 |
| 512 | `onchip_master` | 1.275601 | 0.00305176 | 22 |
| 512 | `onchip_warpspec_kv_hbm_prefetch_loader_core31_decoupled` | 1.273712 | 0.00305176 | 39 |
| 512 | `onchip_warpspec_kv_hbm_prefetch_loader_core31_decoupled_unicast` | 1.284823 | 0.00305176 | 39 |

The default decoupled path emitted the expected current-prefetch sidecar on
both rows:

```text
name:        mixed_flash_kv_repack_hbm_prefetch_hoist_0_current_prefetch
ops:         nop, STCDPOpHBM, nop, STCDPOpLx
loader_core: 31
fanout:      true
full_tile:   true
serialize:   true
```

## Change

Add a narrow decoupled promotion case:

```text
b1h8d64_block64_mid_decoupled_loader_core31
```

The case covers only:

```text
B1 H8 D64 block64 L384,L512
```

It requires the same loader-prefetch invariant as the existing decoupled gate:

- current-prefetch role;
- loader core 31;
- loader fanout;
- full-tile fanout pieces;
- serialized loader-core prefetch;
- `STCDPOpHBM` in the mixed sidecar.

The full decoupled promotion gate then passed with the new case included:

```text
PROMOTION_GATE_PASSED gate=onchip_warpspec_decoupled cases=3 rows=8
```

The new case's gate medians were:

| L | Median ms | Max abs error | Mixed SDSCs |
| ---: | ---: | ---: | ---: |
| 384 | 1.045306 | 0.00292969 | 29 |
| 512 | 1.268466 | 0.00305176 | 39 |

## Interpretation

This extends the certified shape island without pretending the H8 long rows are
fixed. The mid rows are faster than `flash_hbm` and effectively tied with
`onchip_master`, matching the broader decoupled-gate performance pattern.

The longer H8 rows should stay out of the gate until the broader on-chip
failure at L768/L1024 is understood.
