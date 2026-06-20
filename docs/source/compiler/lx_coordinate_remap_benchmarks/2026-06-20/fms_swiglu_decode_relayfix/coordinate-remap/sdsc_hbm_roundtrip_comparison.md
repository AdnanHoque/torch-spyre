# Fused SwiGLU HBM Round-Trip Comparison

This file is the direct before/after readout for an FMS fused SwiGLU run. The important signal is whether pointwise consumers read the projection halves from HBM or from LX after coordinate remap.

| edge | baseline alloc/addr | coordinate-remap alloc/addr | interpretation |
| --- | --- | --- | --- |
| Projection output | 2_hbm @ 0xc800000..0xc80c000 (25 unique) | 2_hbm @ 0xc800000..0xc80c000 (25 unique) | Still HBM-backed. |
| SiLU neg first-half input | 0_hbm @ 0xc800000..0xc806000 (25 unique) | 0_hbm @ 0xc800000..0xc806000 (25 unique) | Still HBM-backed. |
| SiLU realdiv first-half input | 0_hbm @ 0xc800000..0xc806000 (25 unique) | 0_hbm @ 0xc800000..0xc806000 (25 unique) | Still HBM-backed. |
| Gate mul second-half input | 1_hbm @ 0xc806400..0xc80c400 (25 unique) | 1_hbm @ 0xc806400..0xc80c400 (25 unique) | Still HBM-backed. |
| Gate mul output | 2_hbm @ 0x0..0x6000 (25 unique) | 2_hbm @ 0x0..0x6000 (25 unique) | Still HBM-backed. |

## Structural Counters

| metric | baseline | coordinate-remap |
| --- | ---: | ---: |
| sdsc_count | 9 | 9 |
| row_count | 23 | 23 |
| sdsc_with_dataops | 0 | 0 |
| remap_chunks | 0 | 0 |
| remap_movements | 0 | 0 |
| remap_bytes | 0 | 0 |

## Interpretation

- No `LXCoordinateRemapOp` chunks were emitted for this run, so the pointwise consumers remain HBM-backed.
- This is expected to have little or no pass-driven speedup; any timing delta is benchmark noise or secondary compiler effects.
