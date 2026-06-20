# FMS Fused SwiGLU Prefill Branch Baseline Jamie-Style SDSC

| Op | cores | alloc_tensor {i}_{loc} | Role | Layout* extent/wkSlices | Address | coreIdToWkSlice | schedule | json files |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ReStickifyOpHBM | 25 | 0_hbm | INPUT | layout=mb,out; stick=out; stick_size=[64] | 0x400000000..0x400300000 (25 unique) | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_0 |
| ReStickifyOpHBM | 25 | 1_hbm | OUTPUT | layout=mb,out; stick=out; stick_size=[64] | 0x0..0xc000000 (25 unique) | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_0 |
| batchmatmul | 32 | 0_hbm | INPUT | layout=in,mb; stick=in; stick_size=[64] | 0x800000000..0x800300000 (4 unique) | in=0 mb=0:3 (4 unique) out=0:7 (8 unique) | all [[-1,0,0,0]] | sdsc_1 |
| batchmatmul | 32 | 1_hbm | INPUT | layout=in,mb; stick=in; stick_size=[64] | 0x0..0xaf00000 (8 unique) | in=0 mb=0:3 (4 unique) out=0:7 (8 unique) | all [[-1,0,0,0]] | sdsc_1 |
| batchmatmul | 32 | 2_hbm | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0xc800000..0xdacaf00 (32 unique) | in=0 mb=0:3 (4 unique) out=0:7 (8 unique) | all [[-1,0,0,0]] | sdsc_1 |
| neg | 32 | 0_hbm | INPUT | layout=out,mb; stick=out; stick_size=[64] | 0xc800000..0xe038000 (32 unique) | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_2 |
| neg | 32 | 1_hbm | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0x0..0xf800 (32 unique) | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_2 |
| exp | 32 | 0_hbm | INPUT | layout=mb,out; stick=out; stick_size=[64] | 0x0..0xf800 (32 unique) | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_3 |
| exp | 32 | 1_lx | OUTPUT | layout=mb,out; stick=out; stick_size=[64] | 0x0 | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_3 |
| add | 32 | 0_lx | INPUT | layout=mb,out; stick=out; stick_size=[64] | 0x0 | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_4 |
| add | 32 | 1_hbm | INPUT | layout=mb,out; stick=out; stick_size=[64] | 0xc00000000 | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_4 |
| add | 32 | 2_lx | OUTPUT | layout=mb,out; stick=out; stick_size=[64] | 0x0 | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_4 |
| realdiv | 32 | 0_hbm | INPUT | layout=out,mb; stick=out; stick_size=[64] | 0xc800000..0xe038000 (32 unique) | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_5 |
| realdiv | 32 | 1_lx | INPUT | layout=out,mb; stick=out; stick_size=[64] | 0x0 | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_5 |
| realdiv | 32 | 2_hbm | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0xc80000..0xc8f800 (32 unique) | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_5 |
| mul | 32 | 0_hbm | INPUT | layout=mb,out; stick=out; stick_size=[64] | 0xc80000..0xc8f800 (32 unique) | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_6 |
| mul | 32 | 1_hbm | INPUT | layout=mb,out; stick=out; stick_size=[64] | 0xc806400..0xe03e400 (32 unique) | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_6 |
| mul | 32 | 2_hbm | OUTPUT | layout=mb,out; stick=out; stick_size=[64] | 0x0..0xf800 (32 unique) | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_6 |
| ReStickifyOpHBM | 25 | 0_hbm | INPUT | layout=mb,out; stick=out; stick_size=[64] | 0x1000000000..0x1006000000 (25 unique) | mb=0 out=core_id | all [[-1,0,0,0]] | sdsc_7 |
| ReStickifyOpHBM | 25 | 1_hbm | OUTPUT | layout=mb,out; stick=out; stick_size=[64] | 0x1900000..0x1a80000 (25 unique) | mb=0 out=core_id | all [[-1,0,0,0]] | sdsc_7 |
| batchmatmul | 32 | 0_hbm | INPUT | layout=mb,in; stick=in; stick_size=[64] | 0x0..0xc000 (4 unique) | in=0 mb=0:3 (4 unique) out=0:7 (8 unique) | all [[-1,0,0,0]] | sdsc_8 |
| batchmatmul | 32 | 1_hbm | INPUT | layout=mb,in; stick=in; stick_size=[64] | 0x1900000..0x7080000 (8 unique) | in=0 mb=0:3 (4 unique) out=0:7 (8 unique) | all [[-1,0,0,0]] | sdsc_8 |
| batchmatmul | 32 | 2_hbm | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0x1400000000..0x1400301c00 (32 unique) | in=0 mb=0:3 (4 unique) out=0:7 (8 unique) | all [[-1,0,0,0]] | sdsc_8 |
