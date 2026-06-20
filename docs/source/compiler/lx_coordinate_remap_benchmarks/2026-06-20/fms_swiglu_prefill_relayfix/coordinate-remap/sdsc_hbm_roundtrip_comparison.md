# Fused SwiGLU HBM Round-Trip Comparison

This file is the direct before/after readout for an FMS fused SwiGLU run. The important signal is whether pointwise consumers read the projection halves from HBM or from LX after coordinate remap.

| edge | baseline alloc/addr | coordinate-remap alloc/addr | interpretation |
| --- | --- | --- | --- |
| Projection output | 2_hbm @ 0xc800000..0xdacaf00 (32 unique) | 2_lx @ 0x0 | HBM read eliminated for this input. |
| SiLU neg first-half input | 0_hbm @ 0xc800000..0xe038000 (32 unique) | 0_lx @ 0x100000 | HBM read eliminated for this input. |
| SiLU realdiv first-half input | 0_hbm @ 0xc800000..0xe038000 (32 unique) | 0_lx @ 0x100000 | HBM read eliminated for this input. |
| Gate mul second-half input | 1_hbm @ 0xc806400..0xe03e400 (32 unique) | 1_lx @ 0x100000 | HBM read eliminated for this input. |
| Gate mul output | 2_hbm @ 0x0..0xf800 (32 unique) | 2_hbm @ 0x0..0xf800 (32 unique) | Still HBM-backed. |

## Structural Counters

| metric | baseline | coordinate-remap |
| --- | ---: | ---: |
| sdsc_count | 9 | 9 |
| row_count | 23 | 33 |
| sdsc_with_dataops | 0 | 2 |
| remap_chunks | 0 | 10 |
| remap_movements | 0 | 13200 |
| remap_bytes | 0 | 27033600 |

## Interpretation

- The first projection output moves from HBM-backed SDSC rows to LX output in the coordinate-remap run.
- `neg` and `realdiv` consume the first half from LX at `0x100000` after the remap carrier runs.
- `mul` consumes the second half from LX at `0x100000` after the second remap carrier runs.
- The pointwise chain still writes its final product to HBM for the downstream matmul; the weight restickifies also remain.
