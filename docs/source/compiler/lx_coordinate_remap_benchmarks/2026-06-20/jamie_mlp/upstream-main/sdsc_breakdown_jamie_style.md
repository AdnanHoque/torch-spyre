# jamie_mlp upstream-main SDSC Breakdown

| Op | cores | alloc_tensor {i}_{loc} | Role | Layout* extent/wkSlices | Address | coreIdToWkSlice | json files |
| --- | --- | --- | --- | --- | --- | --- | --- |
| batchmatmul | 32 | 0_hbm+lx | INPUT | layout=mb,in,x; stick=in; stick_size=[64] | 0xc80000..0xc8c000 (4 unique) | in=0 mb=0:3 (4 unique) out=0:7 (8 unique) x=0 | sdsc_fused_bmm_1_0dsrhvvu/sdsc_0.json |
| batchmatmul | 32 | 1_hbm+lx | KERNEL | layout=out,in; stick=out; stick_size=[64] | 0x400000000..0x400001c00 (8 unique) | in=0 mb=0:3 (4 unique) out=0:7 (8 unique) x=0 | sdsc_fused_bmm_1_0dsrhvvu/sdsc_0.json |
| batchmatmul | 32 | 2_hbm+lx | OUTPUT | layout=mb,out,x; stick=out; stick_size=[64] | 0x800000000..0x80038c000 (32 unique) | in=0 mb=0:3 (4 unique) out=0:7 (8 unique) x=0 | sdsc_fused_bmm_1_0dsrhvvu/sdsc_0.json |
| batchmatmul | 32 | 0_hbm+lx | INPUT | layout=mb,in,x; stick=in; stick_size=[64] | 0x400000000..0x40000c000 (4 unique) | in=0 mb=0:3 (4 unique) out=0:7 (8 unique) x=0 | sdsc_fused_bmm_mul_silu_0_nhmeo7o2/sdsc_0.json |
| batchmatmul | 32 | 1_hbm+lx | KERNEL | layout=out,in; stick=out; stick_size=[64] | 0x800000000..0x800005780 (8 unique) | in=0 mb=0:3 (4 unique) out=0:7 (8 unique) x=0 | sdsc_fused_bmm_mul_silu_0_nhmeo7o2/sdsc_0.json |
| batchmatmul | 32 | 2_hbm+lx | OUTPUT | layout=mb,out,x; stick=out; stick_size=[64] | 0x0..0xafc000 (32 unique) | in=0 mb=0:3 (4 unique) out=0:7 (8 unique) x=0 | sdsc_fused_bmm_mul_silu_0_nhmeo7o2/sdsc_0.json |
| batchmatmul | 32 | 0_hbm+lx | INPUT | layout=mb,in,x; stick=in; stick_size=[64] | 0x400000000..0x40000c000 (4 unique) | in=0 mb=0:3 (4 unique) out=0:7 (8 unique) x=0 | sdsc_fused_bmm_mul_silu_0_nhmeo7o2/sdsc_1.json |
| batchmatmul | 32 | 1_hbm+lx | KERNEL | layout=out,in; stick=out; stick_size=[64] | 0xc00000000..0xc00005780 (8 unique) | in=0 mb=0:3 (4 unique) out=0:7 (8 unique) x=0 | sdsc_fused_bmm_mul_silu_0_nhmeo7o2/sdsc_1.json |
| batchmatmul | 32 | 2_hbm+lx | OUTPUT | layout=mb,out,x; stick=out; stick_size=[64] | 0xc80000..0x177c000 (32 unique) | in=0 mb=0:3 (4 unique) out=0:7 (8 unique) x=0 | sdsc_fused_bmm_mul_silu_0_nhmeo7o2/sdsc_1.json |
| neg | 32 | 0_hbm+lx | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0xc80000..0x189c000 (32 unique) | mb=core_id out=0 | sdsc_fused_bmm_mul_silu_0_nhmeo7o2/sdsc_2.json |
| neg | 32 | 1_hbm+lx | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0x1900000..0x251c000 (32 unique) | mb=core_id out=0 | sdsc_fused_bmm_mul_silu_0_nhmeo7o2/sdsc_2.json |
| exp | 32 | 0_hbm+lx | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0x1900000..0x251c000 (32 unique) | mb=core_id out=0 | sdsc_fused_bmm_mul_silu_0_nhmeo7o2/sdsc_3.json |
| exp | 32 | 1_lx | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0x0 | mb=core_id out=0 | sdsc_fused_bmm_mul_silu_0_nhmeo7o2/sdsc_3.json |
| add | 32 | 0_lx | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0x0 | mb=core_id out=0 | sdsc_fused_bmm_mul_silu_0_nhmeo7o2/sdsc_4.json |
| add | 32 | 1_hbm+lx | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0x1000000000 | mb=core_id out=0 | sdsc_fused_bmm_mul_silu_0_nhmeo7o2/sdsc_4.json |
| add | 32 | 2_lx | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0x0 | mb=core_id out=0 | sdsc_fused_bmm_mul_silu_0_nhmeo7o2/sdsc_4.json |
| realdiv | 32 | 0_hbm+lx | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0xc80000..0x189c000 (32 unique) | mb=core_id out=0 | sdsc_fused_bmm_mul_silu_0_nhmeo7o2/sdsc_5.json |
| realdiv | 32 | 1_lx | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0x0 | mb=core_id out=0 | sdsc_fused_bmm_mul_silu_0_nhmeo7o2/sdsc_5.json |
| realdiv | 32 | 2_hbm+lx | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0x1900000..0x251c000 (32 unique) | mb=core_id out=0 | sdsc_fused_bmm_mul_silu_0_nhmeo7o2/sdsc_5.json |
| mul | 32 | 0_hbm+lx | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0x0..0xc1c000 (32 unique) | mb=core_id out=0 | sdsc_fused_bmm_mul_silu_0_nhmeo7o2/sdsc_6.json |
| mul | 32 | 1_hbm+lx | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0x1900000..0x251c000 (32 unique) | mb=core_id out=0 | sdsc_fused_bmm_mul_silu_0_nhmeo7o2/sdsc_6.json |
| mul | 32 | 2_hbm+lx | OUTPUT | layout=out,mb; stick=out; stick_size=[64] | 0xc80000..0x189c000 (32 unique) | mb=core_id out=0 | sdsc_fused_bmm_mul_silu_0_nhmeo7o2/sdsc_6.json |
