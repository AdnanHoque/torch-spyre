# Stage090 - Warpspec Fanout Tuning

## Question

Can the layout-decoupled loader-specialized path beat `onchip_master` by tuning
the loader/fanout sidecar, without changing the correctness-critical invariant
from Stage078?

The invariant remains:

```text
Do not overlap the loader core's HBM prefetch data movement with that same
core's current attention compute slice.
```

## Screened Knobs

Stage227 screened two representative rows with `warmup=1` and `iters=3`:

```text
B1 H4 D64  block64 L768
B2 H4 D128 block64 L1024
```

The screened knobs were:

```text
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_LOADER_CORE=0
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_LOADER_LX_BASE=-2
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_OVERLAP_AFTER_SYNC=0
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_FANOUT_USE_UNICAST=1
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_CORELET1=1
```

All screened rows were value-correct. The useful signal was small:

| Shape | Default ms | Best screened ms | Best knob |
| --- | ---: | ---: | --- |
| B1 H4 L768 D64 | 1.586923 | 1.574749 | fanout unicast |
| B2 H4 L1024 D128 | 4.873151 | 4.841749 | corelet1, roughly tied with safe source/unicast |

Stage228 tested combined safe-source plus unicast on four representative rows.
The combination was correct but did not compose into a better schedule:

| Shape | Combined ms |
| --- | ---: |
| B1 H4 L768 D64 | 1.583857 |
| B1 H4 L1024 D64 | 2.209391 |
| B2 H4 L512 D128 | 1.510890 |
| B2 H4 L1024 D128 | 4.816279 |

## Unicast Full-Island Check

Stage229 reran the full `onchip_warpspec_decoupled` island with fanout unicast:

```text
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_FANOUT_USE_UNICAST=1
warmup=2
iters=7
```

All six rows remained value-correct:

| Shape | Unicast ms | Previous default ms |
| --- | ---: | ---: |
| B1 H4 L768 D64 | 1.565760 | 1.567068 |
| B1 H4 L1024 D64 | 2.181850 | 2.182102 |
| B2 H4 L384 D128 | 1.118608 | 1.121148 |
| B2 H4 L512 D128 | 1.493568 | 1.495212 |
| B2 H4 L768 D128 | 3.105724 | 3.116855 |
| B2 H4 L1024 D128 | 4.810600 | 4.802847 |

The result is a tiny tuning signal, not a default-change signal. The geomean
speedup over the previous decoupled default is approximately 1.002x, with one
row slightly slower.

## Change

Added a named sweep alias for repeatable future A/B testing:

```text
onchip_warpspec_kv_hbm_prefetch_loader_core31_decoupled_unicast
```

The alias is identical to
`onchip_warpspec_kv_hbm_prefetch_loader_core31_decoupled` except for:

```text
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_FANOUT_USE_UNICAST=1
```

It is not the promotion-gate default.

## Interpretation

The loader-specialized path is now better instrumented, but the immediate
performance limiter does not appear to be multicast-vs-unicast fanout mode
alone. The tuned unicast sidecar is useful as a controlled variant, while the
main default should remain the simpler decoupled loader-core path until a larger
performance delta appears.

The next higher-leverage work is probably not another one-bit fanout selector.
It is either:

- reducing the extra mixed-SDSC/fanout rows relative to `onchip_master`;
- making the loader-side prefetch hide more latency without violating the
  loader-core serialization invariant;
- redistributing or avoiding the loader core's serialized compute slice; or
- finding shapes where the future K/V HBM prefetch removes enough HBM pressure
  to clearly exceed `onchip_master`.
