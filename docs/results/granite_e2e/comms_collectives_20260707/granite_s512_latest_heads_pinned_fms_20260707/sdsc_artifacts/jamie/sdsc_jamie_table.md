# Granite S512 Current Head Relayout Enabled Pinned FMS

| Op | cores | alloc_tensor {i}_{loc} | Role | Layout* extent/wkSlices | Address | coreIdToWkSlice | schedule | json files |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mul | 32 | 0_hbm | INPUT | layout=out,y,mb,i,x; stick=i; stick_size=[64] | -0x20..-0x1 (32 unique) | i=0 mb=core_id out=0 x=0 y=0 | all [[-1,0,0,0]] | sdsc_0 |
| mul | 32 | 1_hbm | INPUT | layout=out,y,mb,i,x; stick=i; stick_size=[64] | -0x40..-0x21 (32 unique) | i=0 mb=core_id out=0 x=0 y=0 | all [[-1,0,0,0]] | sdsc_0 |
| mul | 32 | 2_lx | OUTPUT | layout=out,y,mb,i,x; stick=i; stick_size=[64] | 0x0 | i=0 mb=core_id out=0 x=0 y=0 | all [[-1,0,0,0]] | sdsc_0 |
| ReStickifyOpHBM | 32 | 0_hbm | INPUT | layout=mb,out; stick=out; stick_size=[64] | -0x20..-0x1 (32 unique) | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_0 |
| ReStickifyOpHBM | 32 | 1_hbm | OUTPUT | layout=mb,out; stick=out; stick_size=[64] | -0x40..-0x21 (32 unique) | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_0 |
| ReStickifyOpHBM | 25 | 0_hbm | INPUT | layout=mb,out; stick=out; stick_size=[64] | -0x19..-0x1 (25 unique) | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_0 |
| ReStickifyOpHBM | 25 | 1_hbm | OUTPUT | layout=mb,out; stick=out; stick_size=[64] | -0x32..-0x1a (25 unique) | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_0 |
| mul | 32 | 0_hbm | INPUT | layout=out,mb; stick=out; stick_size=[64] | -0x20..-0x1 (32 unique) | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_0 |
| mul | 32 | 1_hbm | INPUT | layout=out,mb; stick=out; stick_size=[64] | -0x20..-0x1 (32 unique) | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_0 |
| mul | 32 | 2_lx | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0x0 | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_0 |
| sumnonstick | 32 | 0_lx | INPUT | layout=mb,out,y,x,i; stick=i; stick_size=[64] | 0x0 | i=0 mb=core_id out=0 x=0 y=0 | all [[-1,0,0,0]] | sdsc_1 |
| sumnonstick | 32 | 1_lx | INPUT/OUTPUT | layout=mb,out,y,x,i; stick=i; stick_size=[64] | 0x40000 | i=0 mb=core_id out=0 x=0 y=0 | all [[-1,0,0,0]] | sdsc_1 |
| batchmatmul | 32 | 0_hbm | INPUT | layout=in,mb; stick=in; stick_size=[64] | -0x44..-0x41 (4 unique) | in=0 mb=0:3 (4 unique) out=0:7 (8 unique) | all [[-1,0,0,0]] | sdsc_1 |
| batchmatmul | 32 | 1_hbm | INPUT | layout=in,mb; stick=in; stick_size=[64] | -0x4c..-0x45 (8 unique) | in=0 mb=0:3 (4 unique) out=0:7 (8 unique) | all [[-1,0,0,0]] | sdsc_1 |
| batchmatmul | 32 | 2_lx | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0x0 | in=0 mb=0:3 (4 unique) out=0:7 (8 unique) | all [[-1,0,0,0]] | sdsc_1 |
| batchmatmul | 32 | 0_hbm | INPUT | layout=mb,in; stick=in; stick_size=[64] | -0x36..-0x33 (4 unique) | in=0 mb=0:3 (4 unique) out=0:7 (8 unique) | all [[-1,0,0,0]] | sdsc_1 |
| batchmatmul | 32 | 1_hbm | INPUT | layout=mb,in; stick=in; stick_size=[64] | -0x3e..-0x37 (8 unique) | in=0 mb=0:3 (4 unique) out=0:7 (8 unique) | all [[-1,0,0,0]] | sdsc_1 |
| batchmatmul | 32 | 2_hbm | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | -0x5e..-0x3f (32 unique) | in=0 mb=0:3 (4 unique) out=0:7 (8 unique) | all [[-1,0,0,0]] | sdsc_1 |
| mean | 32 | 0_lx | INPUT | layout=out,mb; stick=out; stick_size=[64] | 0x0 | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_1 |
| mean | 32 | 1_lx | INPUT/OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0x20000 | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_1 |
| mul | 32 | 0_lx | INPUT | layout=mb,out,x; stick=out; stick_size=[64] | 0x40000 | mb=core_id out=0 x=0 | all [[-1,0,0,0]] | sdsc_2 |
| mul | 32 | 1_hbm | INPUT | layout=mb,out,x; stick=out; stick_size=[64] | -0x41 | mb=core_id out=0 x=0 | all [[-1,0,0,0]] | sdsc_2 |
| mul | 32 | 2_hbm | OUTPUT | layout=mb,out,x; stick=out; stick_size=[64] | -0x61..-0x42 (32 unique) | mb=core_id out=0 x=0 | all [[-1,0,0,0]] | sdsc_2 |
| mul | 32 | 0_lx | INPUT | layout=out,mb; stick=out; stick_size=[64] | 0x0 | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_2 |
| mul | 32 | 1_hbm | INPUT | layout=out,mb; stick=out; stick_size=[64] | -0x4d | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_2 |
| mul | 32 | 2_lx | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0x0 | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_2 |
| silu | 32 | 0_hbm | INPUT | layout=out,mb; stick=out; stick_size=[64] | -0x7e..-0x5f (32 unique) | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_2 |
| silu | 32 | 1_lx | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0x20000 | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_2 |
| add | 32 | 0_lx | INPUT | layout=out,x; stick=x; stick_size=[64] | 0x20000 | out=core_id x=0 | all [[-1,0,0,0]] | sdsc_2 |
| add | 32 | 1_hbm | INPUT | layout=out,x; stick=x; stick_size=[64] | -0x21 | out=core_id x=0 | all [[-1,0,0,0]] | sdsc_2 |
| add | 32 | 2_lx | OUTPUT | layout=out,x; stick=x; stick_size=[64] | 0x20000 | out=core_id x=0 | all [[-1,0,0,0]] | sdsc_2 |
| mul | 32 | 0_hbm | INPUT | layout=out,y,mb,i,x; stick=i; stick_size=[64] | -0x81..-0x62 (32 unique) | i=0 mb=core_id out=0 x=0 y=0 | all [[-1,0,0,0]] | sdsc_3 |
| mul | 32 | 1_hbm | INPUT | layout=out,y,mb,i,x; stick=i; stick_size=[64] | -0xa2..-0x83 (32 unique) | i=0 mb=core_id out=0 x=0 y=0 | all [[-1,0,0,0]] | sdsc_3 |
| mul | 32 | 2_lx | OUTPUT | layout=out,y,mb,i,x; stick=i; stick_size=[64] | 0x0 | i=0 mb=core_id out=0 x=0 y=0 | all [[-1,0,0,0]] | sdsc_3 |
| add | 32 | 0_lx | INPUT | layout=out,mb; stick=out; stick_size=[64] | 0x0 | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_3 |
| add | 32 | 1_hbm | INPUT | layout=out,mb; stick=out; stick_size=[64] | -0x6d..-0x4e (32 unique) | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_3 |
| add | 32 | 2_lx | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0x0 | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_3 |
| mul | 32 | 0_lx | INPUT | layout=mb,out; stick=out; stick_size=[64] | 0x20000 | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_3 |
| mul | 32 | 1_hbm | INPUT | layout=mb,out; stick=out; stick_size=[64] | -0x9e..-0x7f (32 unique) | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_3 |
| mul | 32 | 2_hbm | OUTPUT | layout=mb,out; stick=out; stick_size=[64] | -0xbe..-0x9f (32 unique) | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_3 |
| rsqrt | 32 | 0_lx | INPUT | layout=out,x; stick=x; stick_size=[64] | 0x20000 | out=core_id x=0 | all [[-1,0,0,0]] | sdsc_3 |
| rsqrt | 32 | 1_lx | OUTPUT | layout=out,x; stick=x; stick_size=[64] | 0x20000 | out=core_id x=0 | all [[-1,0,0,0]] | sdsc_3 |
| sumnonstick | 32 | 0_lx | INPUT | layout=mb,out,y,x,i; stick=i; stick_size=[64] | 0x0 | i=0 mb=core_id out=0 x=0 y=0 | all [[-1,0,0,0]] | sdsc_4 |
| sumnonstick | 32 | 1_hbm | INPUT/OUTPUT | layout=mb,out,y,x,i; stick=i; stick_size=[64] | -0xc2..-0xa3 (32 unique) | i=0 mb=core_id out=0 x=0 y=0 | all [[-1,0,0,0]] | sdsc_4 |
| mul | 32 | 0_lx | INPUT | layout=out,mb; stick=out; stick_size=[64] | 0x0 | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_4 |
| mul | 32 | 1_lx | INPUT | layout=out,mb; stick=out; stick_size=[64] | 0x0 | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_4 |
| mul | 32 | 2_lx | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0x20000 | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_4 |
| ReStickifyOpHBM | 25 | 0_hbm | INPUT | layout=mb,out; stick=out; stick_size=[64] | -0xd7..-0xbf (25 unique) | mb=0 out=core_id | all [[-1,0,0,0]] | sdsc_4 |
| ReStickifyOpHBM | 25 | 1_hbm | OUTPUT | layout=mb,out; stick=out; stick_size=[64] | -0xf0..-0xd8 (25 unique) | mb=0 out=core_id | all [[-1,0,0,0]] | sdsc_4 |
| mul | 32 | 0_hbm | INPUT | layout=out,mb; stick=out; stick_size=[64] | -0x41..-0x22 (32 unique) | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_4 |
| mul | 32 | 1_lx | INPUT | layout=out,mb; stick=out; stick_size=[64] | 0x20000 | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_4 |
| mul | 32 | 2_lx | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0x0 | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_4 |
| identity | 32 | 0_hbm | INPUT | layout=y,out,mb,x; stick=out; stick_size=[64] | -0xe2..-0xc3 (32 unique) | mb=0 out=0 x=0 y=core_id | all [[-1,0,0,0]] | sdsc_5 |
| identity | 32 | 1_lx | OUTPUT | layout=y,out,mb,x; stick=out; stick_size=[64] | 0x0 | mb=0 out=0 x=0 y=core_id | all [[-1,0,0,0]] | sdsc_5 |
| mean | 32 | 0_lx | INPUT | layout=out,mb; stick=out; stick_size=[64] | 0x20000 | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_5 |
| mean | 32 | 1_lx | INPUT/OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0x40000 | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_5 |
| batchmatmul | 32 | 0_hbm | INPUT | layout=mb,in; stick=in; stick_size=[64] | -0xf8..-0xf1 (8 unique) | in=0 mb=0:7 (8 unique) out=0:3 (4 unique) | all [[-1,0,0,0]] | sdsc_5 |
| batchmatmul | 32 | 1_hbm | INPUT | layout=mb,in; stick=in; stick_size=[64] | -0xfc..-0xf9 (4 unique) | in=0 mb=0:7 (8 unique) out=0:3 (4 unique) | all [[-1,0,0,0]] | sdsc_5 |
| batchmatmul | 32 | 2_lx | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0x20000 | in=0 mb=0:7 (8 unique) out=0:3 (4 unique) | all [[-1,0,0,0]] | sdsc_5 |
| mul | 32 | 0_lx | INPUT | layout=mb,out; stick=out; stick_size=[64] | 0x0 | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_5 |
| mul | 32 | 1_hbm | INPUT | layout=mb,out; stick=out; stick_size=[64] | -0x42 | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_5 |
| mul | 32 | 2_hbm | OUTPUT | layout=mb,out; stick=out; stick_size=[64] | -0x62..-0x43 (32 unique) | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_5 |
| mul | 32 | 0_lx | INPUT | layout=out,x,mb; stick=out; stick_size=[64] | 0x0 | mb=0 out=0 x=core_id | all [[-1,0,0,0]] | sdsc_6 |
| mul | 32 | 1_hbm | INPUT | layout=out,x,mb; stick=out; stick_size=[64] | -0xe3 | mb=0 out=0 x=core_id | all [[-1,0,0,0]] | sdsc_6 |
| mul | 32 | 2_lx | OUTPUT | layout=out,x,mb; stick=out; stick_size=[64] | 0x20000 | mb=0 out=0 x=core_id | all [[-1,0,0,0]] | sdsc_6 |
| add | 32 | 0_lx | INPUT | layout=out,x; stick=x; stick_size=[64] | 0x40000 | out=core_id x=0 | all [[-1,0,0,0]] | sdsc_6 |
| add | 32 | 1_hbm | INPUT | layout=out,x; stick=x; stick_size=[64] | -0x6e | out=core_id x=0 | all [[-1,0,0,0]] | sdsc_6 |
| add | 32 | 2_lx | OUTPUT | layout=out,x; stick=x; stick_size=[64] | 0x40000 | out=core_id x=0 | all [[-1,0,0,0]] | sdsc_6 |
| mul | 32 | 0_lx | INPUT | layout=out,mb; stick=out; stick_size=[64] | 0x20000 | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_6 |
| mul | 32 | 1_hbm | INPUT | layout=out,mb; stick=out; stick_size=[64] | -0xfd | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_6 |
| mul | 32 | 2_lx | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0x20000 | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_6 |
| ReStickifyOpHBM | 32 | 0_hbm | INPUT | layout=mb,out; stick=out; stick_size=[64] | -0x82..-0x63 (32 unique) | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_6 |
| ReStickifyOpHBM | 32 | 1_hbm | OUTPUT | layout=mb,out; stick=out; stick_size=[64] | -0xa2..-0x83 (32 unique) | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_6 |
| ReStickifyOpHBM | 32 | 0_lx | INPUT | layout=mb,out,x; stick=out; stick_size=[64] | 0x20000 | mb=core_id out=0 x=0 | all [[-1,0,0,0]] | sdsc_7 |
| ReStickifyOpHBM | 32 | 1_hbm | OUTPUT | layout=mb,out,x; stick=out; stick_size=[64] | -0x103..-0xe4 (32 unique) | mb=core_id out=0 x=0 | all [[-1,0,0,0]] | sdsc_7 |
| rsqrt | 32 | 0_lx | INPUT | layout=out,x; stick=x; stick_size=[64] | 0x40000 | out=core_id x=0 | all [[-1,0,0,0]] | sdsc_7 |
| rsqrt | 32 | 1_lx | OUTPUT | layout=out,x; stick=x; stick_size=[64] | 0x40000 | out=core_id x=0 | all [[-1,0,0,0]] | sdsc_7 |
| add | 32 | 0_lx | INPUT | layout=out,mb; stick=out; stick_size=[64] | 0x20000 | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_7 |
| add | 32 | 1_lx | INPUT | layout=out,mb; stick=out; stick_size=[64] | 0x0 | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_7 |
| add | 32 | 2_hbm | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | -0x11d..-0xfe (32 unique) | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_7 |
| batchmatmul | 32 | 0_hbm | INPUT | layout=mb,in; stick=in; stick_size=[64] | -0xa6..-0xa3 (4 unique) | in=0 mb=0:3 (4 unique) out=0:7 (8 unique) | all [[-1,0,0,0]] | sdsc_7 |
| batchmatmul | 32 | 1_hbm | INPUT | layout=mb,in; stick=in; stick_size=[64] | -0xae..-0xa7 (8 unique) | in=0 mb=0:3 (4 unique) out=0:7 (8 unique) | all [[-1,0,0,0]] | sdsc_7 |
| batchmatmul | 32 | 2_hbm | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | -0xce..-0xaf (32 unique) | in=0 mb=0:3 (4 unique) out=0:7 (8 unique) | all [[-1,0,0,0]] | sdsc_7 |
| batchmatmul | 32 | 0_hbm | INPUT | layout=mb,in,x; stick=in; stick_size=[64] | -0x113..-0x104 (16 unique) | in=0 mb=0 out=0:1 (2 unique) x=0:15 (16 unique) | all [[-1,0,0,0]] | sdsc_8 |
| batchmatmul | 32 | 1_hbm | INPUT | layout=mb,in,x; stick=in; stick_size=[64] | -0x115..-0x114 (2 unique) | in=0 mb=0 out=0:1 (2 unique) x=0:15 (16 unique) | all [[-1,0,0,0]] | sdsc_8 |
| batchmatmul | 32 | 2_lx | OUTPUT | layout=out,x,mb; stick=out; stick_size=[64] | 0x0 | in=0 mb=0 out=0:1 (2 unique) x=0:15 (16 unique) | all [[-1,0,0,0]] | sdsc_8 |
| mul | 32 | 0_lx | INPUT | layout=out,mb; stick=out; stick_size=[64] | 0x0 | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_8 |
| mul | 32 | 1_lx | INPUT | layout=out,mb; stick=out; stick_size=[64] | 0x40000 | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_8 |
| mul | 32 | 2_lx | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0x40800 | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_8 |
| add | 32 | 0_lx | INPUT | layout=out,x,mb; stick=out; stick_size=[64] | 0x0 | mb=0 out=0 x=core_id | all [[-1,0,0,0]] | sdsc_9 |
| add | 32 | 1_hbm | INPUT | layout=out,x,mb; stick=out; stick_size=[64] | -0x135..-0x116 (32 unique) | mb=0 out=0 x=core_id | all [[-1,0,0,0]] | sdsc_9 |
| add | 32 | 2_lx | OUTPUT | layout=out,x,mb; stick=out; stick_size=[64] | 0x80000 | mb=0 out=0 x=core_id | all [[-1,0,0,0]] | sdsc_9 |
| mul | 32 | 0_lx | INPUT | layout=mb,out; stick=out; stick_size=[64] | 0x40800 | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_9 |
| mul | 32 | 1_hbm | INPUT | layout=mb,out; stick=out; stick_size=[64] | -0x6f | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_9 |
| mul | 32 | 2_hbm | OUTPUT | layout=mb,out; stick=out; stick_size=[64] | -0x8f..-0x70 (32 unique) | mb=core_id out=0 | all [[-1,0,0,0]] | sdsc_9 |
| max | 32 | 0_lx | INPUT | layout=mb,out,x; stick=out; stick_size=[64] | 0x80000 | mb=0 out=0 x=core_id | all [[-1,0,0,0]] | sdsc_10 |
| max | 32 | 1_lx | INPUT/OUTPUT | layout=mb,out,x; stick=out; stick_size=[64] | 0x0 | mb=0 out=0 x=core_id | all [[-1,0,0,0]] | sdsc_10 |
| sub | 32 | 0_lx | INPUT | layout=mb,out,x; stick=out; stick_size=[64] | 0x80000 | mb=0 out=0 x=core_id | all [[-1,0,0,0]] | sdsc_11 |
| sub | 32 | 1_lx | INPUT | layout=mb,out,x; stick=out; stick_size=[64] | 0x0 | mb=0 out=0 x=core_id | all [[-1,0,0,0]] | sdsc_11 |
| sub | 32 | 2_lx | OUTPUT | layout=mb,out,x; stick=out; stick_size=[64] | 0x100000 | mb=0 out=0 x=core_id | all [[-1,0,0,0]] | sdsc_11 |
| exp | 32 | 0_lx | INPUT | layout=mb,out,x; stick=out; stick_size=[64] | 0x100000 | mb=0 out=0 x=core_id | all [[-1,0,0,0]] | sdsc_12 |
| exp | 32 | 1_lx | OUTPUT | layout=mb,out,x; stick=out; stick_size=[64] | 0x100000 | mb=0 out=0 x=core_id | all [[-1,0,0,0]] | sdsc_12 |
| sum | 32 | 0_lx | INPUT | layout=mb,out,x; stick=out; stick_size=[64] | 0x100000 | mb=0 out=0 x=core_id | all [[-1,0,0,0]] | sdsc_13 |
| sum | 32 | 1_lx | INPUT/OUTPUT | layout=mb,out,x; stick=out; stick_size=[64] | 0x0 | mb=0 out=0 x=core_id | all [[-1,0,0,0]] | sdsc_13 |
| realdiv | 32 | 0_lx | INPUT | layout=mb,out,x; stick=out; stick_size=[64] | 0x100000 | mb=0 out=0 x=core_id | all [[-1,0,0,0]] | sdsc_14 |
| realdiv | 32 | 1_lx | INPUT | layout=mb,out,x; stick=out; stick_size=[64] | 0x0 | mb=0 out=0 x=core_id | all [[-1,0,0,0]] | sdsc_14 |
| realdiv | 32 | 2_lx | OUTPUT | layout=mb,out,x; stick=out; stick_size=[64] | 0x100000 | mb=0 out=0 x=core_id | all [[-1,0,0,0]] | sdsc_14 |
| identity | 32 | 0_hbm | INPUT | layout=out,mb,y,x; stick=out; stick_size=[64] | -0x156..-0x137 (32 unique) | mb=0 out=0 x=0 y=core_id | all [[-1,0,0,0]] | sdsc_15 |
| identity | 32 | 1_hbm | OUTPUT | layout=out,mb,y,x; stick=out; stick_size=[64] | -0x176..-0x157 (32 unique) | mb=0 out=0 x=0 y=core_id | all [[-1,0,0,0]] | sdsc_15 |
| batchmatmul | 32 | 0_lx | INPUT | layout=x,in,mb; stick=in; stick_size=[64] | 0x100000 | in=0 mb=core_id out=0 x=0 | all [[-1,0,0,0]] | sdsc_16 |
| batchmatmul | 32 | 1_hbm | INPUT | layout=x,in,mb; stick=in; stick_size=[64] | -0x177 | in=0 mb=core_id out=0 x=0 | all [[-1,0,0,0]] | sdsc_16 |
| batchmatmul | 32 | 2_lx | OUTPUT | layout=out,mb,x; stick=out; stick_size=[64] | 0x0 | in=0 mb=core_id out=0 x=0 | all [[-1,0,0,0]] | sdsc_16 |
| identity | 32 | 0_lx | INPUT | layout=out,x,mb; stick=out; stick_size=[64] | 0x0 | mb=0 out=0 x=core_id | all [[-1,0,0,0]] | sdsc_17 |
| identity | 32 | 1_hbm | OUTPUT | layout=out,x,mb; stick=out; stick_size=[64] | -0x197..-0x178 (32 unique) | mb=0 out=0 x=core_id | all [[-1,0,0,0]] | sdsc_17 |
