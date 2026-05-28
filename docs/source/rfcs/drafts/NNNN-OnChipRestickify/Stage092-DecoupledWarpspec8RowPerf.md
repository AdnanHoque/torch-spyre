# Stage092 - Decoupled Warpspec 8-Row Performance Envelope

## Question

After adding the `B1 H8 D64 block64 L384,L512` mid-length rows to the
layout-decoupled loader-specialized gate, what is the performance envelope of
the full eight-row gate island?

## Run

Stage234 ran:

```text
tools/onchip_sdpa_perf_compare.py
  --gate onchip_warpspec_decoupled
  --cases all
  --baseline-variants flash_hbm,onchip_master
  --warmup 2
  --iters 7
  --seed 42865
```

The command validates the target rows with the same promotion-gate invariants
before reporting timings:

- value correctness within the gate max error;
- pointwise handoff;
- K/V repack allowed for this gate;
- current-prefetch sidecar;
- loader core 31;
- loader fanout with full-tile pieces;
- serialized loader-core prefetch;
- `STCDPOpHBM` in the sidecar.

## Result

```text
PERF_COMPARE_PASSED gate=onchip_warpspec_decoupled cases=3 comparisons=16
PERF_SUMMARY baseline=flash_hbm ok_pairs=8/8 geomean_speedup=1.1518x
PERF_SUMMARY baseline=onchip_master ok_pairs=8/8 geomean_speedup=0.9929x
```

Per-row medians:

| Shape | L | `flash_hbm` ms | `onchip_master` ms | decoupled ms | Speedup vs `flash_hbm` | Speedup vs `onchip_master` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 H4 D64 block64 | 768 | 1.738489 | 1.626099 | 1.571266 | 1.1064x | 1.0349x |
| B1 H4 D64 block64 | 1024 | 2.550084 | 2.194906 | 2.173174 | 1.1734x | 1.0100x |
| B1 H8 D64 block64 | 384 | 1.049221 | 0.954747 | 0.968222 | 1.0837x | 0.9861x |
| B1 H8 D64 block64 | 512 | 1.465706 | 1.267971 | 1.275051 | 1.1495x | 0.9944x |
| B2 H4 D128 block64 | 384 | 1.249995 | 1.102760 | 1.151739 | 1.0853x | 0.9575x |
| B2 H4 D128 block64 | 512 | 1.764253 | 1.486903 | 1.555549 | 1.1342x | 0.9559x |
| B2 H4 D128 block64 | 768 | 3.712762 | 3.115857 | 3.109740 | 1.1939x | 1.0020x |
| B2 H4 D128 block64 | 1024 | 6.250052 | 4.821906 | 4.796391 | 1.3031x | 1.0053x |

## Interpretation

The decoupled loader-specialized path is now consistently faster than the
FlashAttention HBM baseline on every promoted row. The current eight-row
geomean speedup is `1.1518x`, with a per-row range from `1.0837x` to `1.3031x`.

The comparison against `onchip_master` is still near break-even. The full gate
geomean is `0.9929x`, with row-level wins on:

```text
B1 H4 D64  L768,L1024
B2 H4 D128 L768,L1024
```

and row-level losses on:

```text
B1 H8 D64  L384,L512
B2 H4 D128 L384,L512
```

This suggests the loader-specialized prefetch becomes more attractive as the
shape gets long enough for HBM pressure to dominate the extra mixed-SDSC and
fanout rows. The shorter promoted rows are useful for coverage and correctness,
but they are not the performance target versus `onchip_master`.

## Next Performance Work

The next tuning pass should focus on the rows that are close to or below
`onchip_master`:

```text
B1 H8 D64  L384,L512
B2 H4 D128 L384,L512
```

The likely question is whether the decoupled path can reduce its fixed
sidecar/fanout overhead for shorter rows, or whether the promotion gate should
eventually distinguish a correctness-certified island from a
performance-preferred routing island.
