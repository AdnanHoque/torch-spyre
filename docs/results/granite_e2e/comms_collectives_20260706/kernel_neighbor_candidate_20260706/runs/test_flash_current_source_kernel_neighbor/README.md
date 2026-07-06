# Flash test_flash.py Current Source Kernel-Neighbor Structural Run

Source script SHA: afda166e58b23519d0b4ca871350b011b56d91a3  
Torch SHA: c9e0e9ae  
Deeptools SHA: e3e265d22  
Return code: 0

This is a structural compile probe against the current aviros/test-spyre-scripts test_flash.py. The probe patches host-to-device movement and CPU assert_close so it can validate compiler/backend lowering without being blocked by the separate baseline value-correctness issue.

## Result

| Metric | Value |
|---|---:|
| SDSC files | 550 |
| ReStickifyOpHBM | 0 |
| ReStickifyOpLx | 32 |
| Backend plans | 32 |
| Plan kind | {'matmul_operand_broadcast': 32} |
| Communication pattern | {'all_gather_replicate': 32} |
| Realization strategy | {'loop_scoped_input_fetch': 32} |
| Physical lowering | {'lowered_loop_scoped_kernel_neighbor': 32} |

## Readout

The current DLDSC path removes HBM restickify from this flash compile shape: ReStickifyOpHBM_total=0. The backend sees 32 matmul_operand_broadcast contracts and lowers all of them through loop-scoped KERNEL-neighbor input fetch, represented as all_gather_replicate into the matmul operand transfer loop.

This is not a value-correctness claim. The current baseline issue is the zero-stride/broadcast view lowering bug; this artifact only proves our communication path is not reintroducing HBM spills in the current flash structural compile path.

Sample backend plan: matmul_plans_sample/54_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json.
