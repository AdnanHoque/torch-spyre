# main vs pr-mlp-fix shape-aware summary

| case | main kernel | PR kernel | PR/main | speedup | main spyre | PR spyre | PR/main | speedup | main PT% | PR PT% | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| matmul_prefill_kv | 0.127 | 0.091 | 0.717x | 1.40x | 0.331 | 0.336 | 1.015x | 0.99x | 46.800 | 65.306 | ok/ok |
| matmul_prefill_qo | 0.559 | 0.323 | 0.578x | 1.73x | 1.105 | 0.916 | 0.829x | 1.21x | 42.656 | 73.833 | ok/ok |
| matmul_prefill_mlp_proj | 3.749 | 1.023 | 0.273x | 3.66x | 5.790 | 3.159 | 0.546x | 1.83x | 29.794 | 72.799 | ok/ok |
| matmul_decode_kv | 0.058 | 0.058 | 1.000x | 1.00x | 0.231 | 0.274 | 1.186x | 0.84x | 0.803 | 0.806 | ok/ok |
| matmul_decode_qo | 0.215 | 0.216 | 1.005x | 1.00x | 0.717 | 0.648 | 0.904x | 1.11x | 0.864 | 0.863 | ok/ok |
| matmul_decode_mlp_proj | 0.718 | 0.722 | 1.006x | 0.99x | 2.587 | 2.635 | 1.019x | 0.98x | 0.810 | 0.806 | ok/ok |
| mlp_prefill |  |  |  |  |  |  |  |  |  |  | failed/failed |
| mlp_decode | 24.600 | 24.807 | 1.008x | 0.99x | 46.601 | 46.883 | 1.006x | 0.99x | 0.080 | 0.079 | ok/ok |

Remote runroot: `/tmp/spyre-main-vs-pr-mlp-fix-final-20260606-133103`

Failure notes are in `summary_shape_aware.tsv`.
