# Stage093 - Decoupled Warpspec Knob Screen

## Question

Can small schedule knobs make the layout-decoupled loader-specialized path beat
`onchip_master` on the rows where the default decoupled target is currently
weak?

The answer is: not yet. Safe-source is the most useful diagnostic lane, but it
does not beat the current default decoupled target across the full gate island.

## New A/B Aliases

This stage adds three named decoupled A/B aliases:

```text
onchip_warpspec_kv_hbm_prefetch_loader_core31_decoupled_safesrc
onchip_warpspec_kv_hbm_prefetch_loader_core31_decoupled_no_after_sync
onchip_warpspec_kv_hbm_prefetch_loader_core31_decoupled_corelet1
```

All three keep the layout-decoupled loader-specialized invariant:

```text
SPYRE_FLASH_ATTENTION_ONCHIP_SDPA=1
SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM=0
SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_PAIR_TILE=-1
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_HOIST_TILE=-2
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_LOADER_CORE=31
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT=1
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT_FULL_TILE_PIECES=1
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_SERIALIZE_LOADER_CORE=1
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_TAIL_CURRENT=0
```

They differ only by one knob:

| Alias suffix | Extra setting |
| --- | --- |
| `safesrc` | `SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_LOADER_LX_BASE=-2` |
| `no_after_sync` | `SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_OVERLAP_AFTER_SYNC=0` |
| `corelet1` | `SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_CORELET1=1` |

## Short Screens

Stage235 screened B1/H8/D64 L384,L512 with `warmup=1`, `iters=3`:

| L | Best row | Median ms | Default decoupled ms | `onchip_master` ms |
| ---: | --- | ---: | ---: | ---: |
| 384 | `decoupled_safesrc` | 0.960283 | 0.986068 | 0.986004 |
| 512 | `decoupled_safesrc` | 1.265379 | 1.343824 | 1.331601 |

Stage236 screened B2/H4/D128 L384,L512 with `warmup=1`, `iters=3`:

| L | Best row | Median ms | Default decoupled ms | `onchip_master` ms |
| ---: | --- | ---: | ---: | ---: |
| 384 | `decoupled_safesrc` | 1.129966 | 1.263643 | 1.158366 |
| 512 | `onchip_master` | 1.497725 | 1.511576 | 1.497725 |

The short screens made safe-source look promising enough to run across the
full gate island.

## Full Safe-Source Compare

Stage237 ran safe-source as the target across the full decoupled gate with
`warmup=2`, `iters=7`:

```text
PERF_COMPARE_PASSED gate=onchip_warpspec_decoupled cases=3 comparisons=24
PERF_SUMMARY baseline=flash_hbm ok_pairs=8/8 geomean_speedup=1.1416x
PERF_SUMMARY baseline=onchip_master ok_pairs=8/8 geomean_speedup=0.9948x
PERF_SUMMARY baseline=onchip_warpspec_kv_hbm_prefetch_loader_core31_decoupled ok_pairs=8/8 geomean_speedup=0.9904x
```

Per-row safe-source medians:

| Shape | L | Safe-source ms | Speedup vs default decoupled | Speedup vs `onchip_master` |
| --- | ---: | ---: | ---: | ---: |
| B1 H4 D64 block64 | 768 | 1.572341 | 0.9984x | 1.0041x |
| B1 H4 D64 block64 | 1024 | 2.197515 | 0.9903x | 1.0051x |
| B1 H8 D64 block64 | 384 | 1.006003 | 0.9580x | 0.9504x |
| B1 H8 D64 block64 | 512 | 1.289943 | 0.9825x | 1.0376x |
| B2 H4 D128 block64 | 384 | 1.115384 | 1.0392x | 0.9762x |
| B2 H4 D128 block64 | 512 | 1.548773 | 0.9628x | 0.9713x |
| B2 H4 D128 block64 | 768 | 3.178602 | 0.9881x | 1.0076x |
| B2 H4 D128 block64 | 1024 | 4.860772 | 1.0060x | 1.0091x |

## Interpretation

Safe-source is correct and occasionally useful, but it is not better than the
default decoupled target as a gate-wide replacement. It improves B2/H4/D128
L384 and B2/H4/D128 L1024 versus the default in this run, but loses on most
other rows.

The default decoupled target remains the right promotion-gate default. The new
aliases should stay as diagnostic A/B lanes until a shape-selective routing
policy or a more stable schedule improvement exists.

## Next Step

The next higher-leverage work is probably not another one-bit selector. The
branch needs either:

- a shape-selective performance routing policy that can use the certified
  loader-specialized path only where it beats `onchip_master`; or
- a schedule change that reduces the fixed mixed-SDSC/fanout overhead without
  giving up the serialized loader-core correctness invariant.
