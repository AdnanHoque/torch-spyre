SDSC Operations Summary - Batch Report
Directory: docs/source/compiler/lx_coordinate_remap_benchmarks/2026-06-20/fms_swiglu_prefill_relayfix/branch-baseline/sdsc_json
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

sdsc_1: batchmatmul (32 cores)
  - 0_hbm: role=INPUT, layout=in,mb; stick=in; stick_size=[64], wkSlice=in=0 mb=0:3 (4 unique) out=0:7 (8 unique), address=0x800000000..0x800300000 (4 unique)
  - 1_hbm: role=INPUT, layout=in,mb; stick=in; stick_size=[64], wkSlice=in=0 mb=0:3 (4 unique) out=0:7 (8 unique), address=0x0..0xaf00000 (8 unique)
  - 2_hbm: role=OUTPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=in=0 mb=0:3 (4 unique) out=0:7 (8 unique), address=0xc800000..0xdacaf00 (32 unique)

sdsc_2: neg (32 cores)
  - 0_hbm: role=INPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0xc800000..0xe038000 (32 unique)
  - 1_hbm: role=OUTPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x0..0xf800 (32 unique)

sdsc_3: exp (32 cores)
  - 0_hbm: role=INPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x0..0xf800 (32 unique)
  - 1_lx: role=OUTPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x0

sdsc_4: add (32 cores)
  - 0_lx: role=INPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x0
  - 1_hbm: role=INPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0xc00000000
  - 2_lx: role=OUTPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x0

sdsc_5: realdiv (32 cores)
  - 0_hbm: role=INPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0xc800000..0xe038000 (32 unique)
  - 1_lx: role=INPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x0
  - 2_hbm: role=OUTPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0xc80000..0xc8f800 (32 unique)

sdsc_6: mul (32 cores)
  - 0_hbm: role=INPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0xc80000..0xc8f800 (32 unique)
  - 1_hbm: role=INPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0xc806400..0xe03e400 (32 unique)
  - 2_hbm: role=OUTPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x0..0xf800 (32 unique)

sdsc_7: ReStickifyOpHBM (25 cores)
  - 0_hbm: role=INPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=0 out=core_id, address=0x1000000000..0x1006000000 (25 unique)
  - 1_hbm: role=OUTPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=0 out=core_id, address=0x1900000..0x1a80000 (25 unique)

sdsc_8: batchmatmul (32 cores)
  - 0_hbm: role=INPUT, layout=mb,in; stick=in; stick_size=[64], wkSlice=in=0 mb=0:3 (4 unique) out=0:7 (8 unique), address=0x0..0xc000 (4 unique)
  - 1_hbm: role=INPUT, layout=mb,in; stick=in; stick_size=[64], wkSlice=in=0 mb=0:3 (4 unique) out=0:7 (8 unique), address=0x1900000..0x7080000 (8 unique)
  - 2_hbm: role=OUTPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=in=0 mb=0:3 (4 unique) out=0:7 (8 unique), address=0x1400000000..0x1400301c00 (32 unique)
