# jamie_mlp coordinate-remap SDSC Breakdown

| Op | cores | alloc_tensor {i}_{loc} | Role | Layout* extent/wkSlices | Address | coreIdToWkSlice | json files |
| --- | --- | --- | --- | --- | --- | --- | --- |
| batchmatmul | 32 | 0_hbm+lx | INPUT | layout=mb,in,x; stick=in; stick_size=[64] | 0xc80000..0xc8c000 (4 unique) | in=0 mb=0:3 (4 unique) out=0:7 (8 unique) x=0 | sdsc_fused_bmm_1_1zu_u1v7/sdsc_0.json |
| batchmatmul | 32 | 1_hbm+lx | KERNEL | layout=out,in; stick=out; stick_size=[64] | 0x400000000..0x400001c00 (8 unique) | in=0 mb=0:3 (4 unique) out=0:7 (8 unique) x=0 | sdsc_fused_bmm_1_1zu_u1v7/sdsc_0.json |
| batchmatmul | 32 | 2_hbm+lx | OUTPUT | layout=mb,out,x; stick=out; stick_size=[64] | 0x800000000..0x80038c000 (32 unique) | in=0 mb=0:3 (4 unique) out=0:7 (8 unique) x=0 | sdsc_fused_bmm_1_1zu_u1v7/sdsc_0.json |
| batchmatmul | 32 | 0_hbm+lx | INPUT | layout=mb,in,x; stick=in; stick_size=[64] | 0x400000000..0x40000c000 (4 unique) | in=0 mb=0:3 (4 unique) out=0:7 (8 unique) x=0 | sdsc_fused_bmm_mul_silu_0_bqy6z255/sdsc_0.json |
| batchmatmul | 32 | 1_hbm+lx | KERNEL | layout=out,in; stick=out; stick_size=[64] | 0x800000000..0x800005780 (8 unique) | in=0 mb=0:3 (4 unique) out=0:7 (8 unique) x=0 | sdsc_fused_bmm_mul_silu_0_bqy6z255/sdsc_0.json |
| batchmatmul | 32 | 2_hbm+lx | OUTPUT | layout=mb,out,x; stick=out; stick_size=[64] | 0x0..0xafc000 (32 unique) | in=0 mb=0:3 (4 unique) out=0:7 (8 unique) x=0 | sdsc_fused_bmm_mul_silu_0_bqy6z255/sdsc_0.json |
| batchmatmul | 32 | 0_hbm+lx | INPUT | layout=mb,in,x; stick=in; stick_size=[64] | 0x400000000..0x40000c000 (4 unique) | in=0 mb=0:3 (4 unique) out=0:7 (8 unique) x=0 | sdsc_fused_bmm_mul_silu_0_bqy6z255/sdsc_1.json |
| batchmatmul | 32 | 1_hbm+lx | KERNEL | layout=out,in; stick=out; stick_size=[64] | 0xc00000000..0xc00005780 (8 unique) | in=0 mb=0:3 (4 unique) out=0:7 (8 unique) x=0 | sdsc_fused_bmm_mul_silu_0_bqy6z255/sdsc_1.json |
| batchmatmul | 32 | 2_lx | OUTPUT | layout=mb,out,x; stick=out; stick_size=[64] | 0x0 | in=0 mb=0:3 (4 unique) out=0:7 (8 unique) x=0 | sdsc_fused_bmm_mul_silu_0_bqy6z255/sdsc_1.json |
| LXCoordinateRemapOp | 32 | dataop_0 (lx->lx) | MOVE | coverage=[512,200,1,64]; ranges=248; movements=6200; bytes=12697600; coalesced=6400 | src=0x0..0x63800; dst=0x100000..0x163800 | [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31] | sdsc_fused_bmm_mul_silu_0_bqy6z255/sdsc_2.json |
| LXCoordinateRemapOp | 15 | dataop_1 (lx->lx) | MOVE | coverage=[512,200,1,64]; ranges=200; movements=200; bytes=409600; coalesced=6400 | src=0x0..0x63800; dst=0x164000..0x1c7800 | [0,1,4,5,9,10,13,14,18,19,22,23,27,28,31] | sdsc_fused_bmm_mul_silu_0_bqy6z255/sdsc_2.json |
| LXCoordinateRemapOp | 15 | dataop_2 (lx->lx) | MOVE | coverage=[512,200,1,64]; ranges=200; movements=200; bytes=409600; coalesced=6400 | src=0x164000..0x1c7800; dst=0x100000..0x163800 | [0,1,4,5,9,10,13,14,18,19,22,23,27,28,31] | sdsc_fused_bmm_mul_silu_0_bqy6z255/sdsc_2.json |
| neg | 32 | 0_lx | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0x100000 | mb=core_id out=0 | sdsc_fused_bmm_mul_silu_0_bqy6z255/sdsc_2.json |
| neg | 32 | 1_hbm+lx | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0x1900000..0x251c000 (32 unique) | mb=core_id out=0 | sdsc_fused_bmm_mul_silu_0_bqy6z255/sdsc_2.json |
| exp | 32 | 0_hbm+lx | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0x1900000..0x251c000 (32 unique) | mb=core_id out=0 | sdsc_fused_bmm_mul_silu_0_bqy6z255/sdsc_3.json |
| exp | 32 | 1_lx | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0x0 | mb=core_id out=0 | sdsc_fused_bmm_mul_silu_0_bqy6z255/sdsc_3.json |
| add | 32 | 0_lx | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0x0 | mb=core_id out=0 | sdsc_fused_bmm_mul_silu_0_bqy6z255/sdsc_4.json |
| add | 32 | 1_hbm+lx | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0x1000000000 | mb=core_id out=0 | sdsc_fused_bmm_mul_silu_0_bqy6z255/sdsc_4.json |
| add | 32 | 2_lx | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0x0 | mb=core_id out=0 | sdsc_fused_bmm_mul_silu_0_bqy6z255/sdsc_4.json |
| realdiv | 32 | 0_lx | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0x100000 | mb=core_id out=0 | sdsc_fused_bmm_mul_silu_0_bqy6z255/sdsc_5.json |
| realdiv | 32 | 1_lx | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0x0 | mb=core_id out=0 | sdsc_fused_bmm_mul_silu_0_bqy6z255/sdsc_5.json |
| realdiv | 32 | 2_hbm+lx | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0x1900000..0x251c000 (32 unique) | mb=core_id out=0 | sdsc_fused_bmm_mul_silu_0_bqy6z255/sdsc_5.json |
| mul | 32 | 0_hbm+lx | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0x0..0xc1c000 (32 unique) | mb=core_id out=0 | sdsc_fused_bmm_mul_silu_0_bqy6z255/sdsc_6.json |
| mul | 32 | 1_hbm+lx | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0x1900000..0x251c000 (32 unique) | mb=core_id out=0 | sdsc_fused_bmm_mul_silu_0_bqy6z255/sdsc_6.json |
| mul | 32 | 2_hbm+lx | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0xc80000..0x189c000 (32 unique) | mb=core_id out=0 | sdsc_fused_bmm_mul_silu_0_bqy6z255/sdsc_6.json |
