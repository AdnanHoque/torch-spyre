# FMS Fused SwiGLU Decode Coordinate-Remap Jamie-Style SDSC

| Op | cores | alloc_tensor {i}_{loc} | Role | Layout* extent/wkSlices | Address | coreIdToWkSlice | schedule | json files |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ReStickifyOpHBM | 25 | 0_hbm | INPUT | layout=mb,out; stick=out; stick_size=[64] | 0x400000000..0x400300000 (25 unique) | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_0 |
| ReStickifyOpHBM | 25 | 1_hbm | OUTPUT | layout=mb,out; stick=out; stick_size=[64] | 0x0..0xc000000 (25 unique) | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_0 |
| batchmatmul | 25 | 0_hbm | INPUT | layout=in; stick=in; stick_size=[64] | 0x800000000 | in=0 out=core_id | all [[-1,0,0,0]] | sdsc_1 |
| batchmatmul | 25 | 1_hbm | INPUT | layout=in; stick=in; stick_size=[64] | 0x0..0xc000000 (25 unique) | in=0 out=core_id | all [[-1,0,0,0]] | sdsc_1 |
| batchmatmul | 25 | 2_hbm | OUTPUT | layout=out; stick=out; stick_size=[64] | 0xc800000..0xc80c000 (25 unique) | in=0 out=core_id | all [[-1,0,0,0]] | sdsc_1 |
| neg | 25 | 0_hbm | INPUT | layout=out; stick=out; stick_size=[64] | 0xc800000..0xc806000 (25 unique) | out=core_id | all [[-1,0,0,0]] | sdsc_2 |
| neg | 25 | 1_hbm | OUTPUT | layout=out; stick=out; stick_size=[64] | 0x0..0x6000 (25 unique) | out=core_id | all [[-1,0,0,0]] | sdsc_2 |
| exp | 25 | 0_hbm | INPUT | layout=out; stick=out; stick_size=[64] | 0x0..0x6000 (25 unique) | out=core_id | all [[-1,0,0,0]] | sdsc_3 |
| exp | 25 | 1_lx | OUTPUT | layout=out; stick=out; stick_size=[64] | 0x0 | out=core_id | all [[-1,0,0,0]] | sdsc_3 |
| add | 25 | 0_lx | INPUT | layout=out; stick=out; stick_size=[64] | 0x0 | out=core_id | all [[-1,0,0,0]] | sdsc_4 |
| add | 25 | 1_hbm | INPUT | layout=out; stick=out; stick_size=[64] | 0xc00000000 | out=core_id | all [[-1,0,0,0]] | sdsc_4 |
| add | 25 | 2_lx | OUTPUT | layout=out; stick=out; stick_size=[64] | 0x0 | out=core_id | all [[-1,0,0,0]] | sdsc_4 |
| realdiv | 25 | 0_hbm | INPUT | layout=out; stick=out; stick_size=[64] | 0xc800000..0xc806000 (25 unique) | out=core_id | all [[-1,0,0,0]] | sdsc_5 |
| realdiv | 25 | 1_lx | INPUT | layout=out; stick=out; stick_size=[64] | 0x0 | out=core_id | all [[-1,0,0,0]] | sdsc_5 |
| realdiv | 25 | 2_hbm | OUTPUT | layout=out; stick=out; stick_size=[64] | 0x6400..0xc400 (25 unique) | out=core_id | all [[-1,0,0,0]] | sdsc_5 |
| mul | 25 | 0_hbm | INPUT | layout=out; stick=out; stick_size=[64] | 0x6400..0xc400 (25 unique) | out=core_id | all [[-1,0,0,0]] | sdsc_6 |
| mul | 25 | 1_hbm | INPUT | layout=out; stick=out; stick_size=[64] | 0xc806400..0xc80c400 (25 unique) | out=core_id | all [[-1,0,0,0]] | sdsc_6 |
| mul | 25 | 2_hbm | OUTPUT | layout=out; stick=out; stick_size=[64] | 0x0..0x6000 (25 unique) | out=core_id | all [[-1,0,0,0]] | sdsc_6 |
| ReStickifyOpHBM | 25 | 0_hbm | INPUT | layout=mb,out; stick=out; stick_size=[64] | 0x1000000000..0x1006000000 (25 unique) | mb=0 out=core_id | all [[-1,0,0,0]] | sdsc_7 |
| ReStickifyOpHBM | 25 | 1_hbm | OUTPUT | layout=mb,out; stick=out; stick_size=[64] | 0xc800..0x18c800 (25 unique) | mb=0 out=core_id | all [[-1,0,0,0]] | sdsc_7 |
| batchmatmul | 32 | 0_hbm | INPUT | layout=in; stick=in; stick_size=[64] | 0x0 | in=0 out=core_id | all [[-1,0,0,0]] | sdsc_8 |
| batchmatmul | 32 | 1_hbm | INPUT | layout=in; stick=in; stick_size=[64] | 0xc800..0x60ec800 (32 unique) | in=0 out=core_id | all [[-1,0,0,0]] | sdsc_8 |
| batchmatmul | 32 | 2_hbm | OUTPUT | layout=out; stick=out; stick_size=[64] | 0x1400000000..0x1400001f00 (32 unique) | in=0 out=core_id | all [[-1,0,0,0]] | sdsc_8 |
