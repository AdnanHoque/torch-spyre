# SDSC Structural Diff

No baseline SDSC directory was provided.

| metric | baseline | current |
| --- | ---: | ---: |
| sdsc_count |  | 44 |
| row_count |  | 114 |
| sdsc_with_dataops |  | 0 |
| remap_chunks |  | 0 |
| remap_movements |  | 0 |
| remap_bytes |  | 0 |

## Operation Counts

| op | baseline | current |
| --- | ---: | ---: |
| ReStickifyOpHBM |  | 10 |
| add |  | 5 |
| batchmatmul |  | 6 |
| exp |  | 1 |
| identity |  | 3 |
| max |  | 1 |
| mean |  | 2 |
| mul |  | 13 |
| realdiv |  | 1 |
| rsqrt |  | 2 |
| silu |  | 1 |
| sub |  | 1 |
| sum |  | 1 |
| sumnonstick |  | 2 |

## Tensor Location Counts

| loc | baseline | current |
| --- | ---: | ---: |
| hbm+lx |  | 51 |
| lx |  | 63 |
