# Codex-pod cost-model discrepancy artifacts

This directory compares the Codex-pod forced-split oracle against Claude-pod numbers transcribed from the June 12 discussion. The purpose is to isolate whether disagreements come from cost-model picks or from different device/compiler/runtime timing oracles.

## Contents

- `four_way_cost_model_codex_pod_summary.md`: four-way table scored on the Codex-pod oracle.
- `four_way_cost_model_codex_pod.csv`: machine-readable four-way table.
- `device_best_splits_codex_vs_claude.csv`: device-best split comparison across pods.
- `codex_pod_disputed_split_metadata.json`: Codex-pod timing samples plus SDSC/MLIR/ideal-cycle hashes for disputed splits.
- `artifacts/`: copied Codex-pod SDSC JSON, bundle MLIR, and ideal-cycle files for selected disputed split cases.

## Device-best split comparison

Only 5 of 12 device-best splits match exactly across the two pod oracles. This is the main evidence that the pods are not interchangeable timing oracles.

| phase | op | shape | Codex-pod best | Claude-pod best | same? |
|---|---|---|---|---|---|
| prefill | QK^T | 512x32x512x128 | 4_1_8_1 (731 us) | 1_2_8_2 (1009 us) | False |
| prefill | attn@V | 32x512x128x512 | 1_16_2_1 (198 us) | 4_4_2_1 (395 us) | False |
| prefill | Q/O proj | 1x512x4096x4096 | 1_8_4_1 (331 us) | 1_4_8_1 (317 us) | False |
| prefill | K/V proj | 1x512x1024x4096 | 1_8_4_1 (118 us) | 1_4_8_1 (89 us) | False |
| prefill | MLP up | 1x512x12800x4096 | 1_4_8_1 (1038 us) | 1_4_8_1 (1017 us) | True |
| prefill | MLP down | 1x512x4096x12800 | 1_4_8_1 (927 us) | 1_4_8_1 (899 us) | True |
| decode | QK^T | 64x32x576x128 | 8_2_1_2 (90 us) | 4_4_1_2 (158 us) | False |
| decode | attn@V | 32x64x128x576 | 1_4_2_3 (55 us) | 2_8_2_1 (60 us) | False |
| decode | Q/O proj | 1x64x4096x4096 | 1_4_8_1 (232 us) | 1_4_8_1 (221 us) | True |
| decode | K/V proj | 1x64x1024x4096 | 1_8_4_1 (67 us) | 1_8_4_1 (60 us) | True |
| decode | MLP up | 1x64x12800x4096 | 1_4_8_1 (673 us) | 1_4_8_1 (705 us) | True |
| decode | MLP down | 1x64x4096x12800 | 1_4_4_1 (689 us) | 1_4_8_1 (685 us) | False |

## Suggested mirror experiment for Claude

Run the same extraction on Claude pod for the listed disputed splits, then compare SDSC JSON fields, bundle MLIR hashes/content, ideal-cycle JSON, and version/env blocks. If SDSCs are identical but timings differ, the discrepancy is runtime/backend/device. If SDSCs differ for the same logical split, the discrepancy is lowering/layout/environment.
