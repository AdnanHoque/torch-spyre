# SDSC Structural Diff

No baseline SDSC directory was provided.

| metric | baseline | current |
| --- | ---: | ---: |
| sdsc_count |  | 9 |
| row_count |  | 33 |
| sdsc_with_dataops |  | 2 |
| remap_chunks |  | 10 |
| remap_movements |  | 13200 |
| remap_bytes |  | 27033600 |

## Operation Counts

| op | baseline | current |
| --- | ---: | ---: |
| LXCoordinateRemapOp |  | 10 |
| ReStickifyOpHBM |  | 4 |
| add |  | 1 |
| batchmatmul |  | 2 |
| exp |  | 1 |
| mul |  | 1 |
| neg |  | 1 |
| realdiv |  | 1 |

## Tensor Location Counts

| loc | baseline | current |
| --- | ---: | ---: |
| hbm+lx |  | 15 |
| lx |  | 8 |
| lx->lx |  | 10 |
