SDSC Operations Summary - Batch Report
Directory: docs/source/compiler/lx_coordinate_remap_benchmarks/2026-06-20/fms_swiglu_prefill_relayfix/coordinate-remap/sdsc_json
Total sdsc.json files found: 9

Operations Summary:

ReStickifyOpHBM      - INPUT (hbm), OUTPUT (hbm)
batchmatmul          - INPUT (hbm), INPUT (hbm), OUTPUT (lx); INPUT (hbm), INPUT (hbm), OUTPUT (hbm)
LXCoordinateRemapOp  - MOVE (lx->lx), MOVE (lx->lx), MOVE (lx->lx), MOVE (lx->lx), MOVE (lx->lx)
neg                  - INPUT (lx), OUTPUT (hbm)
exp                  - INPUT (hbm), OUTPUT (lx)
add                  - INPUT (lx), INPUT (hbm), OUTPUT (lx)
realdiv              - INPUT (lx), INPUT (lx), OUTPUT (hbm)
mul                  - INPUT (hbm), INPUT (lx), OUTPUT (hbm)

Tensor Details:

sdsc_0: ReStickifyOpHBM (25 cores)
  - 0_hbm: role=INPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x400000000..0x400300000 (25 unique)
  - 1_hbm: role=OUTPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x0..0xc000000 (25 unique)

sdsc_1: batchmatmul (32 cores)
  - 0_hbm: role=INPUT, layout=in,mb; stick=in; stick_size=[64], wkSlice=in=0 mb=0:3 (4 unique) out=0:7 (8 unique), address=0x800000000..0x800300000 (4 unique)
  - 1_hbm: role=INPUT, layout=in,mb; stick=in; stick_size=[64], wkSlice=in=0 mb=0:3 (4 unique) out=0:7 (8 unique), address=0x0..0xaf00000 (8 unique)
  - 2_lx: role=OUTPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=in=0 mb=0:3 (4 unique) out=0:7 (8 unique), address=0x0

sdsc_2: LXCoordinateRemapOp + neg (32 cores)
  - dataop_0_lx->lx: role=MOVE, coverage=[512,400,1,64]; ranges=124; movements=6200; bytes=12697600; coalesced=6400, wkSlice=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31], address=src=0x0..0xc7800; dst=0x100000..0x163800
  - dataop_1_lx->lx: role=MOVE, coverage=[512,400,1,64]; ranges=112; movements=112; bytes=229376; coalesced=6400, wkSlice=[0,1,4,5,9,10], address=src=0x0..0xc6000; dst=0xc8000..0xff800
  - dataop_2_lx->lx: role=MOVE, coverage=[512,400,1,64]; ranges=112; movements=112; bytes=229376; coalesced=6400, wkSlice=[0,1,4,5,9,10], address=src=0xc8000..0xff800; dst=0x100000..0x137800
  - dataop_3_lx->lx: role=MOVE, coverage=[512,400,1,64]; ranges=88; movements=88; bytes=180224; coalesced=6400, wkSlice=[9,10,13,14], address=src=0x2800..0xc6800; dst=0xc8000..0xf3800
  - dataop_4_lx->lx: role=MOVE, coverage=[512,400,1,64]; ranges=88; movements=88; bytes=180224; coalesced=6400, wkSlice=[9,10,13,14], address=src=0xc8000..0xf3800; dst=0x138000..0x163800
  - 0_lx: role=INPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x100000
  - 1_hbm: role=OUTPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x0..0xf800 (32 unique)

sdsc_3: exp (32 cores)
  - 0_hbm: role=INPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x0..0xf800 (32 unique)
  - 1_lx: role=OUTPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x0

sdsc_4: add (32 cores)
  - 0_lx: role=INPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x0
  - 1_hbm: role=INPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0xc00000000
  - 2_lx: role=OUTPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x0

sdsc_5: realdiv (32 cores)
  - 0_lx: role=INPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x100000
  - 1_lx: role=INPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x0
  - 2_hbm: role=OUTPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0xc80000..0xc8f800 (32 unique)

sdsc_6: LXCoordinateRemapOp + mul (32 cores)
  - dataop_0_lx->lx: role=MOVE, coverage=[512,400,1,64]; ranges=124; movements=6200; bytes=12697600; coalesced=6400, wkSlice=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31], address=src=0x0..0xc7800; dst=0x164000..0x1c7800
  - dataop_1_lx->lx: role=MOVE, coverage=[512,400,1,64]; ranges=112; movements=112; bytes=229376; coalesced=6400, wkSlice=[18,19,22,23,27,28], address=src=0x1000..0xc7000; dst=0xc8000..0xff800
  - dataop_2_lx->lx: role=MOVE, coverage=[512,400,1,64]; ranges=112; movements=112; bytes=229376; coalesced=6400, wkSlice=[18,19,22,23,27,28], address=src=0xc8000..0xff800; dst=0x164000..0x19b800
  - dataop_3_lx->lx: role=MOVE, coverage=[512,400,1,64]; ranges=88; movements=88; bytes=180224; coalesced=6400, wkSlice=[0,27,28,31], address=src=0x3800..0xc7800; dst=0xc8000..0xf3800
  - dataop_4_lx->lx: role=MOVE, coverage=[512,400,1,64]; ranges=88; movements=88; bytes=180224; coalesced=6400, wkSlice=[0,27,28,31], address=src=0xc8000..0xf3800; dst=0x19c000..0x1c7800
  - 0_hbm: role=INPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0xc80000..0xc8f800 (32 unique)
  - 1_lx: role=INPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x100000
  - 2_hbm: role=OUTPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x0..0xf800 (32 unique)

sdsc_7: ReStickifyOpHBM (25 cores)
  - 0_hbm: role=INPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=0 out=core_id, address=0x1000000000..0x1006000000 (25 unique)
  - 1_hbm: role=OUTPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=0 out=core_id, address=0x1900000..0x1a80000 (25 unique)

sdsc_8: batchmatmul (32 cores)
  - 0_hbm: role=INPUT, layout=mb,in; stick=in; stick_size=[64], wkSlice=in=0 mb=0:3 (4 unique) out=0:7 (8 unique), address=0x0..0xc000 (4 unique)
  - 1_hbm: role=INPUT, layout=mb,in; stick=in; stick_size=[64], wkSlice=in=0 mb=0:3 (4 unique) out=0:7 (8 unique), address=0x1900000..0x7080000 (8 unique)
  - 2_hbm: role=OUTPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=in=0 mb=0:3 (4 unique) out=0:7 (8 unique), address=0x1400000000..0x1400301c00 (32 unique)
