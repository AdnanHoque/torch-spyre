# Q/O scaled-FP8 weight-preload PoC

Logical operation: `[M,4096] @ [4096,4096]`. Each number is the mean of 20 Kineto device-kernel events after five warmups on DD2.

| M | FP16 us | Stock FP8 us | FP8 `weipreload=0` us | Stock FP8 / FP16 | PoC FP8 / FP16 | PoC / stock FP8 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 232.145 | 112.256 | 112.977 | 2.068x | 2.055x | 0.994x |
| 2 | 235.038 | 112.805 | 112.079 | 2.084x | 2.097x | 1.006x |
| 4 | 235.894 | 115.392 | 117.416 | 2.044x | 2.009x | 0.983x |
| 8 | 259.903 | 124.069 | 125.031 | 2.095x | 2.079x | 0.992x |
| 16 | 253.719 | 138.740 | 137.781 | 1.829x | 1.841x | 1.007x |
| 32 | 250.908 | 131.852 | 129.884 | 1.903x | 1.932x | 1.015x |
| 64 | 224.362 | 138.448 | 138.945 | 1.621x | 1.615x | 0.996x |
| 128 | 253.924 | 188.974 | 159.731 | 1.344x | 1.590x | 1.183x |
| 256 | 279.261 | 273.753 | 175.855 | 1.020x | 1.588x | 1.557x |
| 512 | 349.637 | 414.814 | 220.793 | 0.843x | 1.584x | 1.879x |
| 1024 | 604.878 | 782.266 | 398.995 | 0.773x | 1.516x | 1.961x |
| 2048 | 1142.804 | 1665.736 | 767.702 | 0.686x | 1.489x | 2.170x |

The FP8 kernel includes FP16-to-FP8 `Qfp8`, any inserted relayout, FP8 BatchMatMul, both scale-recovery stages, and FP16 output production. Scale derivation is outside this standalone graph; per-row and per-output-channel scale inputs are fixed at one for correctness isolation.

The PoC changes one DeepTools option only: `DT_OPT=autopilot=1,weipreload=0`. It is an experimental whole-graph switch, not a production recommendation.
