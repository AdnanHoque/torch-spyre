# Flash PR81 Relayout Matrix

`split` is the benchmark-only configuration where Torch sees full LX and DXP sees 0.6 backend workspace. It is not a production-safe allocation policy.

| Operation | Lq | Mask | Variant | Kernel ms | Wall ms | ReStickify HBM | ReStickify LX | Explicit LX distributions | Fired | Kernel speedup vs 0.2 | Wall speedup vs 0.2 | Kernel speedup vs 0.6 | Wall speedup vs 0.6 |
|---|---:|---:|---|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|
| flash_attn_online_softmax | 512 | 0 | off0p2 | 0.306 | 15.760 | 1 | 0 | 0 | no | 1.000 | 1.000 | 0.935 | 1.055 |
| flash_attn_online_softmax | 512 | 0 | off0p6 | 0.286 | 16.624 | 1 | 0 | 0 | no | 1.070 | 0.948 | 1.000 | 1.000 |
| flash_attn_online_softmax | 512 | 0 | split | 0.232 | 15.669 | 0 | 1 | 1 | yes | 1.319 | 1.006 | 1.233 | 1.061 |
| flash_attn_online_softmax | 1024 | 0 | off0p2 | 0.439 | 16.840 | 1 | 0 | 0 | no | 1.000 | 1.000 | 6.383 | 1.100 |
| flash_attn_online_softmax | 1024 | 0 | off0p6 | 2.802 | 18.528 | 1 | 0 | 0 | no | 0.157 | 0.909 | 1.000 | 1.000 |
| flash_attn_online_softmax | 1024 | 0 | split | 1.296 | 17.943 | 0 | 1 | 1 | yes | 0.339 | 0.939 | 2.162 | 1.033 |
| flash_attn_softmax | 512 | 0 | off0p2 | 0.296 | 15.378 | 1 | 0 | 0 | no | 1.000 | 1.000 | 0.970 | 0.956 |
| flash_attn_softmax | 512 | 0 | off0p6 | 0.287 | 14.702 | 1 | 0 | 0 | no | 1.031 | 1.046 | 1.000 | 1.000 |
| flash_attn_softmax | 512 | 0 | split | 0.247 | 14.551 | 0 | 1 | 1 | yes | 1.198 | 1.057 | 1.162 | 1.010 |
| flash_attn_softmax | 512 | 1 | off0p2 | 0.329 | 21.484 | 1 | 0 | 0 | no | 1.000 | 1.000 | 0.982 | 1.035 |
| flash_attn_softmax | 512 | 1 | off0p6 | 0.323 | 22.226 | 1 | 0 | 0 | no | 1.019 | 0.967 | 1.000 | 1.000 |
| flash_attn_softmax | 512 | 1 | split | 0.279 | 21.159 | 0 | 1 | 1 | yes | 1.179 | 1.015 | 1.158 | 1.050 |
| flash_attn_softmax | 1024 | 0 | off0p2 | 0.430 | 16.398 | 1 | 0 | 0 | no | 1.000 | 1.000 | 7.458 | 1.186 |
| flash_attn_softmax | 1024 | 0 | off0p6 | 3.207 | 19.445 | 1 | 0 | 0 | no | 0.134 | 0.843 | 1.000 | 1.000 |
| flash_attn_softmax | 1024 | 0 | split | 1.279 | 17.134 | 0 | 1 | 1 | yes | 0.336 | 0.957 | 2.507 | 1.135 |
| flash_attn_softmax | 1024 | 1 | off0p2 | 0.491 | 29.014 | 1 | 0 | 0 | no | 1.000 | 1.000 | 8.010 | 1.099 |
| flash_attn_softmax | 1024 | 1 | off0p6 | 3.933 | 31.900 | 1 | 0 | 0 | no | 0.125 | 0.910 | 1.000 | 1.000 |
| flash_attn_softmax | 1024 | 1 | split | 1.158 | 29.948 | 0 | 1 | 1 | yes | 0.424 | 0.969 | 3.396 | 1.065 |
| flash_attn_stable_softmax | 512 | 0 | off0p2 | 0.245 | 16.570 | 1 | 0 | 0 | no | 1.000 | 1.000 | 0.963 | 0.991 |
| flash_attn_stable_softmax | 512 | 0 | off0p6 | 0.236 | 16.420 | 1 | 0 | 0 | no | 1.038 | 1.009 | 1.000 | 1.000 |
| flash_attn_stable_softmax | 512 | 0 | split | 0.191 | 17.017 | 0 | 1 | 1 | yes | 1.283 | 0.974 | 1.236 | 0.965 |
| flash_attn_stable_softmax | 512 | 1 | off0p2 | 0.280 | 23.144 | 1 | 0 | 0 | no | 1.000 | 1.000 | 0.979 | 1.025 |
| flash_attn_stable_softmax | 512 | 1 | off0p6 | 0.274 | 23.714 | 1 | 0 | 0 | no | 1.022 | 0.976 | 1.000 | 1.000 |
| flash_attn_stable_softmax | 512 | 1 | split | 0.225 | 23.663 | 0 | 1 | 1 | yes | 1.244 | 0.978 | 1.218 | 1.002 |
| flash_attn_stable_softmax | 1024 | 0 | off0p2 | 0.330 | 18.063 | 1 | 0 | 0 | no | 1.000 | 1.000 | 9.691 | 1.155 |
| flash_attn_stable_softmax | 1024 | 0 | off0p6 | 3.198 | 20.857 | 1 | 0 | 0 | no | 0.103 | 0.866 | 1.000 | 1.000 |
| flash_attn_stable_softmax | 1024 | 0 | split | 1.180 | 18.693 | 0 | 1 | 1 | yes | 0.280 | 0.966 | 2.710 | 1.116 |
| flash_attn_stable_softmax | 1024 | 1 | off0p2 | 0.383 | 31.923 | 1 | 0 | 0 | no | 1.000 | 1.000 | 10.266 | 1.053 |
| flash_attn_stable_softmax | 1024 | 1 | off0p6 | 3.932 | 33.627 | 1 | 0 | 0 | no | 0.097 | 0.949 | 1.000 | 1.000 |
| flash_attn_stable_softmax | 1024 | 1 | split | 1.051 | 32.176 | 0 | 1 | 1 | yes | 0.364 | 0.992 | 3.741 | 1.045 |
