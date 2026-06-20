# SDSC Structural Diff

| metric | baseline | current |
| --- | ---: | ---: |
| sdsc_count | 8 | 8 |
| row_count | 22 | 25 |
| sdsc_with_dataops | 0 | 1 |
| remap_chunks | 0 | 3 |
| remap_movements | 0 | 6600 |
| remap_bytes | 0 | 13516800 |

## Operation Counts

| op | baseline | current |
| --- | ---: | ---: |
| LXCoordinateRemapOp | 0 | 3 |
| add | 1 | 1 |
| batchmatmul | 3 | 3 |
| exp | 1 | 1 |
| mul | 1 | 1 |
| neg | 1 | 1 |
| realdiv | 1 | 1 |

## Tensor Location Counts

| loc | baseline | current |
| --- | ---: | ---: |
| hbm+lx | 18 | 15 |
| lx | 4 | 7 |
| lx->lx | 0 | 3 |
