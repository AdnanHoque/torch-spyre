# Upstream Main Device Pick Measurement

- torch-spyre upstream/main: `f9519fcaeb73a62e77328ae8c430ac3de3ff7d8a`
- pristine upstream/main on this PyTorch 2.12 lane failed before timing due the fake-tensor / joint-graph attention setup issue.
- measurement workaround: `torch._inductor.config.use_joint_graph_passes = False` in the probe process only; no upstream code changes.
- profiler device times from pristine upstream were `0.0`, so total below is split-imputed by looking up emitted upstream SDSC splits in the completed forced-split timing table.

| phase | op | upstream split | imputed us | device-best split | best us |
|---|---|---:|---:|---:|---:|
| prefill | QK^T | `32_1_1_1` | 989.49 | `4_1_8_1` | 731.06 |
| prefill | attn@V | `1_32_1_1` | 327.39 | `1_16_2_1` | 197.72 |
| prefill | Q/O proj | `1_4_8_1` | 340.14 | `1_8_4_1` | 331.07 |
| prefill | K/V proj | `1_4_8_1` | 174.50 | `1_8_4_1` | 117.55 |
| prefill | MLP up | `1_4_8_1` | 1037.61 | `1_4_8_1` | 1037.61 |
| prefill | MLP down | `1_4_8_1` | 926.95 | `1_4_8_1` | 926.95 |
| decode | QK^T | `32_1_1_1` | 203.05 | `8_2_1_2` | 89.93 |
| decode | attn@V | `1_32_1_1` | 94.42 | `1_4_2_3` | 55.04 |
| decode | Q/O proj | `1_32_1_1` | 622.82 | `1_4_8_1` | 231.84 |
| decode | K/V proj | `1_32_1_1` | 142.62 | `1_8_4_1` | 66.78 |
| decode | MLP up | `1_4_8_1` | 673.44 | `1_4_8_1` | 673.44 |
| decode | MLP down | `1_32_1_1` | 2043.51 | `1_4_4_1` | 689.20 |

- upstream split-imputed total: `7575.95 us`
- device-best total: `5148.19 us`
- upstream / device-best: `1.47x`
