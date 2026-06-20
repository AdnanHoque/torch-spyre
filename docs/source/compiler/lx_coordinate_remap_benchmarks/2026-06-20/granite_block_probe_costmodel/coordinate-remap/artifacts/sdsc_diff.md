# SDSC Structural Diff

| metric | baseline | current |
| --- | ---: | ---: |
| sdsc_count | 9 | 9 |
| row_count | 23 | 33 |
| sdsc_with_dataops | 0 | 2 |
| remap_chunks | 0 | 10 |
| remap_movements | 0 | 13200 |
| remap_bytes | 0 | 27033600 |

## Operation Counts

| op | baseline | current |
| --- | ---: | ---: |
| LXCoordinateRemapOp | 0 | 10 |
| ReStickifyOpHBM | 4 | 4 |
| add | 1 | 1 |
| batchmatmul | 2 | 2 |
| exp | 1 | 1 |
| mul | 1 | 1 |
| neg | 1 | 1 |
| realdiv | 1 | 1 |

## Tensor Location Counts

| loc | baseline | current |
| --- | ---: | ---: |
| hbm+lx | 19 | 15 |
| lx | 4 | 8 |
| lx->lx | 0 | 10 |
