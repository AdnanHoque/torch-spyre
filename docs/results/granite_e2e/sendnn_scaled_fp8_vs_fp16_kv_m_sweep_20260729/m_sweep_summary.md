# Granite KV standalone SenDNN M sweep

Fixed logical operation: `[M, 4096] @ [4096, 1024]`.

| M | FP16 mean kernel (us) | Scaled FP8 mean kernel (us) | FP16 effective TFLOP/s | Scaled FP8 effective TFLOP/s | FP8 / FP16 |
|---:|---:|---:|---:|---:|---:|
| 1 | 62.772 | 30.816 | 0.134 | 0.272 | 2.037x |
| 2 | 64.599 | 30.906 | 0.260 | 0.543 | 2.090x |
| 4 | 64.947 | 31.575 | 0.517 | 1.063 | 2.057x |
| 8 | 65.542 | 33.963 | 1.024 | 1.976 | 1.930x |
| 16 | 71.511 | 38.266 | 1.877 | 3.508 | 1.869x |
| 32 | 73.663 | 45.530 | 3.644 | 5.896 | 1.618x |
| 64 | 66.055 | 47.240 | 8.128 | 11.365 | 1.398x |
| 128 | 69.964 | 54.900 | 15.347 | 19.558 | 1.274x |
| 256 | 82.702 | 69.225 | 25.967 | 31.022 | 1.195x |
| 512 | 105.845 | 108.175 | 40.578 | 39.704 | 0.978x |
| 1024 | 174.177 | 240.931 | 49.317 | 35.653 | 0.723x |
| 2048 | 301.347 | 464.971 | 57.010 | 36.948 | 0.648x |

TFLOP/s counts only the logical matmul FLOPs (`2*M*K*N`) and uses the mean Kineto device-kernel duration. The scaled-FP8 kernel includes on-device FP16-to-FP8 Qfp8 conversion, FP8 matmul, and both scale-recovery stages. It uses unit-valued per-row activation scales and per-output-channel weight scales; scale derivation and activation normalization are not included.
