# decode_bmm branch-baseline SDSC Breakdown

| Op | cores | alloc_tensor {i}_{loc} | Role | Layout* extent/wkSlices | Address | coreIdToWkSlice | json files |
| --- | --- | --- | --- | --- | --- | --- | --- |
| batchmatmul | 32 | 0_hbm+lx | INPUT | layout=in; stick=in; stick_size=[64] | 0x6400 | in=0 out=core_id | sdsc_fused_bmm_1__ben_gcp/sdsc_0.json |
| batchmatmul | 32 | 1_hbm+lx | KERNEL | layout=out,in; stick=out; stick_size=[64] | 0x400000000..0x400001f00 (32 unique) | in=0 out=core_id | sdsc_fused_bmm_1__ben_gcp/sdsc_0.json |
| batchmatmul | 32 | 2_hbm+lx | OUTPUT | layout=out; stick=out; stick_size=[64] | 0x800000000..0x800001f00 (32 unique) | in=0 out=core_id | sdsc_fused_bmm_1__ben_gcp/sdsc_0.json |
| batchmatmul | 25 | 0_hbm+lx | INPUT | layout=in; stick=in; stick_size=[64] | 0x400000000 | in=0 out=core_id | sdsc_fused_bmm_mul_silu_0_zpitha0g/sdsc_0.json |
| batchmatmul | 25 | 1_hbm+lx | KERNEL | layout=out,in; stick=out; stick_size=[64] | 0x800000000..0x800006000 (25 unique) | in=0 out=core_id | sdsc_fused_bmm_mul_silu_0_zpitha0g/sdsc_0.json |
| batchmatmul | 25 | 2_hbm+lx | OUTPUT | layout=out; stick=out; stick_size=[64] | 0x0..0x6000 (25 unique) | in=0 out=core_id | sdsc_fused_bmm_mul_silu_0_zpitha0g/sdsc_0.json |
| batchmatmul | 25 | 0_hbm+lx | INPUT | layout=in; stick=in; stick_size=[64] | 0x400000000 | in=0 out=core_id | sdsc_fused_bmm_mul_silu_0_zpitha0g/sdsc_1.json |
| batchmatmul | 25 | 1_hbm+lx | KERNEL | layout=out,in; stick=out; stick_size=[64] | 0xc00000000..0xc00006000 (25 unique) | in=0 out=core_id | sdsc_fused_bmm_mul_silu_0_zpitha0g/sdsc_1.json |
| batchmatmul | 25 | 2_hbm+lx | OUTPUT | layout=out; stick=out; stick_size=[64] | 0x6400..0xc400 (25 unique) | in=0 out=core_id | sdsc_fused_bmm_mul_silu_0_zpitha0g/sdsc_1.json |
| neg | 25 | 0_hbm+lx | OUTPUT | layout=out; stick=out; stick_size=[64] | 0x6400..0xc400 (25 unique) | out=core_id | sdsc_fused_bmm_mul_silu_0_zpitha0g/sdsc_2.json |
| neg | 25 | 1_hbm+lx | OUTPUT | layout=out; stick=out; stick_size=[64] | 0xc800..0x12800 (25 unique) | out=core_id | sdsc_fused_bmm_mul_silu_0_zpitha0g/sdsc_2.json |
| exp | 25 | 0_hbm+lx | OUTPUT | layout=out; stick=out; stick_size=[64] | 0xc800..0x12800 (25 unique) | out=core_id | sdsc_fused_bmm_mul_silu_0_zpitha0g/sdsc_3.json |
| exp | 25 | 1_lx | OUTPUT | layout=out; stick=out; stick_size=[64] | 0x0 | out=core_id | sdsc_fused_bmm_mul_silu_0_zpitha0g/sdsc_3.json |
| add | 25 | 0_lx | OUTPUT | layout=out; stick=out; stick_size=[64] | 0x0 | out=core_id | sdsc_fused_bmm_mul_silu_0_zpitha0g/sdsc_4.json |
| add | 25 | 1_hbm+lx | OUTPUT | layout=out; stick=out; stick_size=[64] | 0x1000000000 | out=core_id | sdsc_fused_bmm_mul_silu_0_zpitha0g/sdsc_4.json |
| add | 25 | 2_lx | OUTPUT | layout=out; stick=out; stick_size=[64] | 0x0 | out=core_id | sdsc_fused_bmm_mul_silu_0_zpitha0g/sdsc_4.json |
| realdiv | 25 | 0_hbm+lx | OUTPUT | layout=out; stick=out; stick_size=[64] | 0x6400..0xc400 (25 unique) | out=core_id | sdsc_fused_bmm_mul_silu_0_zpitha0g/sdsc_5.json |
| realdiv | 25 | 1_lx | OUTPUT | layout=out; stick=out; stick_size=[64] | 0x0 | out=core_id | sdsc_fused_bmm_mul_silu_0_zpitha0g/sdsc_5.json |
| realdiv | 25 | 2_hbm+lx | OUTPUT | layout=out; stick=out; stick_size=[64] | 0xc800..0x12800 (25 unique) | out=core_id | sdsc_fused_bmm_mul_silu_0_zpitha0g/sdsc_5.json |
| mul | 25 | 0_hbm+lx | OUTPUT | layout=out; stick=out; stick_size=[64] | 0x0..0x6000 (25 unique) | out=core_id | sdsc_fused_bmm_mul_silu_0_zpitha0g/sdsc_6.json |
| mul | 25 | 1_hbm+lx | OUTPUT | layout=out; stick=out; stick_size=[64] | 0xc800..0x12800 (25 unique) | out=core_id | sdsc_fused_bmm_mul_silu_0_zpitha0g/sdsc_6.json |
| mul | 25 | 2_hbm+lx | OUTPUT | layout=out; stick=out; stick_size=[64] | 0x6400..0xc400 (25 unique) | out=core_id | sdsc_fused_bmm_mul_silu_0_zpitha0g/sdsc_6.json |
