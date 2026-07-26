| ID | Expanded | Remote MiB | Route | Source -> destination | Input | Consumers |
| --- | ---: | ---: | --- | --- | ---: | --- |
| P01 | 40 | 1240.000 | grouped_all_gather_with_replication | 32x owners[1] -> 1x owners[32] | 1 | bmm-BMM_1, bmm_2-BMM_1, bmm_78-BMM_1 |
| P02 | 40 | 1240.000 | grouped_all_gather_with_replication | 32x owners[1] -> 1x owners[32] | 1 | bmm_1-BMM_1, bmm_3-BMM_1, bmm_79-BMM_1 |
| P03 | 80 | 960.000 | grouped_all_gather_with_replication | 32x owners[1] -> 8x owners[4] | 0 | mm_11-BMM_1, mm_12-BMM_1, mm_277-BMM_1, mm_278-BMM_1, mm_4-BMM_1, mm_5-BMM_1 |
| P04 | 40 | 480.000 | grouped_all_gather_with_replication | 32x owners[1] -> 8x owners[4] | 0 | mm_10-BMM_1, mm_276-BMM_1, mm_3-BMM_1 |
| P05 | 41 | 123.000 | all_gather | 32x owners[1] -> 8x owners[1] | 0 | mean_1-Exx2, mean_3-Exx2, mean_79-Exx2, mean_80-Exx2 |
| P06 | 40 | 120.000 | all_gather | 32x owners[1] -> 32x owners[1] | 0 | bmm-BMM_1, bmm_2-BMM_1, bmm_78-BMM_1 |
| P07 | 160 | 60.000 | grouped_all_gather_with_replication | 16x owners[1] -> 8x owners[4] | 1 | mul_14-mul_1, mul_14-mul_2, mul_15-mul_1, mul_15-mul_2, mul_3-mul_1, mul_3-mul_2, mul_4-mul_1, mul_4-mul_2, mul_432-mul_1, mul_432-mul_2, mul_433-mul_1, mul_433-mul_2 |
| P08 | 40 | 37.500 | permutation | 32x owners[1] -> 32x owners[1] | 0 | bmm-wtAttnHeadBreak-VirtualReshape-Output-Restickify, bmm_2-wtAttnHeadBreak-VirtualReshape-Output-Restickify, bmm_78-wtAttnHeadBreak-VirtualReshape-Output-Restickify |
| P09 | 3 | 36.000 | grouped_all_gather_with_replication | 16x owners[1] -> 8x owners[4] | 0 | mm-BMM_1, mm_1-BMM_1, mm_2-BMM_1 |
| P10 | 80 | 15.000 | replicate_or_owner_remap | 8x owners[1] -> 8x owners[4] | 1 | mean_1-LayerNormNorm, mean_2-LayerNormNorm, mean_3-LayerNormNorm, mean_4-LayerNormNorm, mean_79-LayerNormNorm, mean_80-LayerNormNorm |
| P11 | 80 | 15.000 | replicate_or_owner_remap | 8x owners[1] -> 8x owners[4] | 2 | mean_1-LayerNormNorm, mean_2-LayerNormNorm, mean_3-LayerNormNorm, mean_4-LayerNormNorm, mean_79-LayerNormNorm, mean_80-LayerNormNorm |
| P12 | 1 | 3.000 | all_gather | 16x owners[1] -> 32x owners[1] | 1 | add_3 |
| P13 | 1 | 0.212 | grouped_all_gather_with_replication | 32x owners[1] -> 1x owners[28] | 0 | mm_280-BMM_1 |
| P14 | 1 | 0.008 | permutation | 32x owners[1] -> 32x owners[1] | 0 | slice_161-Stcdp |
