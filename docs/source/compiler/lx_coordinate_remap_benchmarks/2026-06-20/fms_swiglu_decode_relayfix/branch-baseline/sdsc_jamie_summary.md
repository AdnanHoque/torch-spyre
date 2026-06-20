SDSC Operations Summary - Batch Report
Directory: docs/source/compiler/lx_coordinate_remap_benchmarks/2026-06-20/fms_swiglu_decode_relayfix/branch-baseline/sdsc_json
Total sdsc.json files found: 9

Operations Summary:

ReStickifyOpHBM      - INPUT (hbm), OUTPUT (hbm)
batchmatmul          - INPUT (hbm), INPUT (hbm), OUTPUT (hbm)
neg                  - INPUT (hbm), OUTPUT (hbm)
exp                  - INPUT (hbm), OUTPUT (lx)
add                  - INPUT (lx), INPUT (hbm), OUTPUT (lx)
realdiv              - INPUT (hbm), INPUT (lx), OUTPUT (hbm)
mul                  - INPUT (hbm), INPUT (hbm), OUTPUT (hbm)

Tensor Details:

sdsc_0: ReStickifyOpHBM (25 cores)
  - 0_hbm: role=INPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x400000000..0x400300000 (25 unique)
  - 1_hbm: role=OUTPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x0..0xc000000 (25 unique)

sdsc_1: batchmatmul (25 cores)
  - 0_hbm: role=INPUT, layout=in; stick=in; stick_size=[64], wkSlice=in=0 out=core_id, address=0x800000000
  - 1_hbm: role=INPUT, layout=in; stick=in; stick_size=[64], wkSlice=in=0 out=core_id, address=0x0..0xc000000 (25 unique)
  - 2_hbm: role=OUTPUT, layout=out; stick=out; stick_size=[64], wkSlice=in=0 out=core_id, address=0xc800000..0xc80c000 (25 unique)

sdsc_2: neg (25 cores)
  - 0_hbm: role=INPUT, layout=out; stick=out; stick_size=[64], wkSlice=out=core_id, address=0xc800000..0xc806000 (25 unique)
  - 1_hbm: role=OUTPUT, layout=out; stick=out; stick_size=[64], wkSlice=out=core_id, address=0x0..0x6000 (25 unique)

sdsc_3: exp (25 cores)
  - 0_hbm: role=INPUT, layout=out; stick=out; stick_size=[64], wkSlice=out=core_id, address=0x0..0x6000 (25 unique)
  - 1_lx: role=OUTPUT, layout=out; stick=out; stick_size=[64], wkSlice=out=core_id, address=0x0

sdsc_4: add (25 cores)
  - 0_lx: role=INPUT, layout=out; stick=out; stick_size=[64], wkSlice=out=core_id, address=0x0
  - 1_hbm: role=INPUT, layout=out; stick=out; stick_size=[64], wkSlice=out=core_id, address=0xc00000000
  - 2_lx: role=OUTPUT, layout=out; stick=out; stick_size=[64], wkSlice=out=core_id, address=0x0

sdsc_5: realdiv (25 cores)
  - 0_hbm: role=INPUT, layout=out; stick=out; stick_size=[64], wkSlice=out=core_id, address=0xc800000..0xc806000 (25 unique)
  - 1_lx: role=INPUT, layout=out; stick=out; stick_size=[64], wkSlice=out=core_id, address=0x0
  - 2_hbm: role=OUTPUT, layout=out; stick=out; stick_size=[64], wkSlice=out=core_id, address=0x6400..0xc400 (25 unique)

sdsc_6: mul (25 cores)
  - 0_hbm: role=INPUT, layout=out; stick=out; stick_size=[64], wkSlice=out=core_id, address=0x6400..0xc400 (25 unique)
  - 1_hbm: role=INPUT, layout=out; stick=out; stick_size=[64], wkSlice=out=core_id, address=0xc806400..0xc80c400 (25 unique)
  - 2_hbm: role=OUTPUT, layout=out; stick=out; stick_size=[64], wkSlice=out=core_id, address=0x0..0x6000 (25 unique)

sdsc_7: ReStickifyOpHBM (25 cores)
  - 0_hbm: role=INPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=0 out=core_id, address=0x1000000000..0x1006000000 (25 unique)
  - 1_hbm: role=OUTPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=0 out=core_id, address=0xc800..0x18c800 (25 unique)

sdsc_8: batchmatmul (32 cores)
  - 0_hbm: role=INPUT, layout=in; stick=in; stick_size=[64], wkSlice=in=0 out=core_id, address=0x0
  - 1_hbm: role=INPUT, layout=in; stick=in; stick_size=[64], wkSlice=in=0 out=core_id, address=0xc800..0x60ec800 (32 unique)
  - 2_hbm: role=OUTPUT, layout=out; stick=out; stick_size=[64], wkSlice=in=0 out=core_id, address=0x1400000000..0x1400001f00 (32 unique)
