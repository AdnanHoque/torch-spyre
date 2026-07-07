SDSC Operations Summary - Batch Report
Directory: /home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/granite_s512_latest_acceptance_20260707_204801/relayout_enabled_pinned_fms/block_prefill/cache/inductor-spyre
Total sdsc.json files found: 18

Operations Summary:

mul                  - INPUT (hbm), INPUT (hbm), OUTPUT (lx), INPUT (hbm), INPUT (hbm), OUTPUT (lx); INPUT (lx), INPUT (hbm), OUTPUT (hbm), INPUT (lx), INPUT (hbm), OUTPUT (lx); INPUT (hbm), INPUT (hbm), OUTPUT (lx), INPUT (lx), INPUT (hbm), OUTPUT (hbm); INPUT (lx), INPUT (lx), OUTPUT (lx), INPUT (hbm), INPUT (lx), OUTPUT (lx); INPUT (lx), INPUT (hbm), OUTPUT (hbm); INPUT (lx), INPUT (hbm), OUTPUT (lx), INPUT (lx), INPUT (hbm), OUTPUT (lx); INPUT (lx), INPUT (lx), OUTPUT (lx)
ReStickifyOpHBM      - INPUT (hbm), OUTPUT (hbm), INPUT (hbm), OUTPUT (hbm); INPUT (hbm), OUTPUT (hbm); INPUT (lx), OUTPUT (hbm)
sumnonstick          - INPUT (lx), INPUT/OUTPUT (lx); INPUT (lx), INPUT/OUTPUT (hbm)
batchmatmul          - INPUT (hbm), INPUT (hbm), OUTPUT (lx), INPUT (hbm), INPUT (hbm), OUTPUT (hbm); INPUT (hbm), INPUT (hbm), OUTPUT (lx); INPUT (hbm), INPUT (hbm), OUTPUT (hbm); INPUT (lx), INPUT (hbm), OUTPUT (lx)
mean                 - INPUT (lx), INPUT/OUTPUT (lx)
silu                 - INPUT (hbm), OUTPUT (lx)
add                  - INPUT (lx), INPUT (hbm), OUTPUT (lx); INPUT (lx), INPUT (lx), OUTPUT (hbm)
rsqrt                - INPUT (lx), OUTPUT (lx)
identity             - INPUT (hbm), OUTPUT (lx); INPUT (hbm), OUTPUT (hbm); INPUT (lx), OUTPUT (hbm)
max                  - INPUT (lx), INPUT/OUTPUT (lx)
sub                  - INPUT (lx), INPUT (lx), OUTPUT (lx)
exp                  - INPUT (lx), OUTPUT (lx)
sum                  - INPUT (lx), INPUT/OUTPUT (lx)
realdiv              - INPUT (lx), INPUT (lx), OUTPUT (lx)

Tensor Details:

sdsc_0: mul + ReStickifyOpHBM (32 cores)
  - 0_hbm: role=INPUT, layout=out,y,mb,i,x; stick=i; stick_size=[64], wkSlice=i=0 mb=core_id out=0 x=0 y=0, address=-0x20..-0x1 (32 unique)
  - 1_hbm: role=INPUT, layout=out,y,mb,i,x; stick=i; stick_size=[64], wkSlice=i=0 mb=core_id out=0 x=0 y=0, address=-0x40..-0x21 (32 unique)
  - 2_lx: role=OUTPUT, layout=out,y,mb,i,x; stick=i; stick_size=[64], wkSlice=i=0 mb=core_id out=0 x=0 y=0, address=0x0
  - 0_hbm: role=INPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=-0x20..-0x1 (32 unique)
  - 1_hbm: role=OUTPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=-0x40..-0x21 (32 unique)
  - 0_hbm: role=INPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=-0x19..-0x1 (25 unique)
  - 1_hbm: role=OUTPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=-0x32..-0x1a (25 unique)
  - 0_hbm: role=INPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=-0x20..-0x1 (32 unique)
  - 1_hbm: role=INPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=-0x20..-0x1 (32 unique)
  - 2_lx: role=OUTPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x0

sdsc_1: sumnonstick + batchmatmul + mean (32 cores)
  - 0_lx: role=INPUT, layout=mb,out,y,x,i; stick=i; stick_size=[64], wkSlice=i=0 mb=core_id out=0 x=0 y=0, address=0x0
  - 1_lx: role=INPUT/OUTPUT, layout=mb,out,y,x,i; stick=i; stick_size=[64], wkSlice=i=0 mb=core_id out=0 x=0 y=0, address=0x40000
  - 0_hbm: role=INPUT, layout=in,mb; stick=in; stick_size=[64], wkSlice=in=0 mb=0:3 (4 unique) out=0:7 (8 unique), address=-0x44..-0x41 (4 unique)
  - 1_hbm: role=INPUT, layout=in,mb; stick=in; stick_size=[64], wkSlice=in=0 mb=0:3 (4 unique) out=0:7 (8 unique), address=-0x4c..-0x45 (8 unique)
  - 2_lx: role=OUTPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=in=0 mb=0:3 (4 unique) out=0:7 (8 unique), address=0x0
  - 0_hbm: role=INPUT, layout=mb,in; stick=in; stick_size=[64], wkSlice=in=0 mb=0:3 (4 unique) out=0:7 (8 unique), address=-0x36..-0x33 (4 unique)
  - 1_hbm: role=INPUT, layout=mb,in; stick=in; stick_size=[64], wkSlice=in=0 mb=0:3 (4 unique) out=0:7 (8 unique), address=-0x3e..-0x37 (8 unique)
  - 2_hbm: role=OUTPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=in=0 mb=0:3 (4 unique) out=0:7 (8 unique), address=-0x5e..-0x3f (32 unique)
  - 0_lx: role=INPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x0
  - 1_lx: role=INPUT/OUTPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x20000

sdsc_2: mul + silu + add (32 cores)
  - 0_lx: role=INPUT, layout=mb,out,x; stick=out; stick_size=[64], wkSlice=mb=core_id out=0 x=0, address=0x40000
  - 1_hbm: role=INPUT, layout=mb,out,x; stick=out; stick_size=[64], wkSlice=mb=core_id out=0 x=0, address=-0x41
  - 2_hbm: role=OUTPUT, layout=mb,out,x; stick=out; stick_size=[64], wkSlice=mb=core_id out=0 x=0, address=-0x61..-0x42 (32 unique)
  - 0_lx: role=INPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x0
  - 1_hbm: role=INPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=-0x4d
  - 2_lx: role=OUTPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x0
  - 0_hbm: role=INPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=-0x7e..-0x5f (32 unique)
  - 1_lx: role=OUTPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x20000
  - 0_lx: role=INPUT, layout=out,x; stick=x; stick_size=[64], wkSlice=out=core_id x=0, address=0x20000
  - 1_hbm: role=INPUT, layout=out,x; stick=x; stick_size=[64], wkSlice=out=core_id x=0, address=-0x21
  - 2_lx: role=OUTPUT, layout=out,x; stick=x; stick_size=[64], wkSlice=out=core_id x=0, address=0x20000

sdsc_3: mul + add + rsqrt (32 cores)
  - 0_hbm: role=INPUT, layout=out,y,mb,i,x; stick=i; stick_size=[64], wkSlice=i=0 mb=core_id out=0 x=0 y=0, address=-0x81..-0x62 (32 unique)
  - 1_hbm: role=INPUT, layout=out,y,mb,i,x; stick=i; stick_size=[64], wkSlice=i=0 mb=core_id out=0 x=0 y=0, address=-0xa2..-0x83 (32 unique)
  - 2_lx: role=OUTPUT, layout=out,y,mb,i,x; stick=i; stick_size=[64], wkSlice=i=0 mb=core_id out=0 x=0 y=0, address=0x0
  - 0_lx: role=INPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x0
  - 1_hbm: role=INPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=-0x6d..-0x4e (32 unique)
  - 2_lx: role=OUTPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x0
  - 0_lx: role=INPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x20000
  - 1_hbm: role=INPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=-0x9e..-0x7f (32 unique)
  - 2_hbm: role=OUTPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=-0xbe..-0x9f (32 unique)
  - 0_lx: role=INPUT, layout=out,x; stick=x; stick_size=[64], wkSlice=out=core_id x=0, address=0x20000
  - 1_lx: role=OUTPUT, layout=out,x; stick=x; stick_size=[64], wkSlice=out=core_id x=0, address=0x20000

sdsc_4: sumnonstick + mul + ReStickifyOpHBM (32 cores)
  - 0_lx: role=INPUT, layout=mb,out,y,x,i; stick=i; stick_size=[64], wkSlice=i=0 mb=core_id out=0 x=0 y=0, address=0x0
  - 1_hbm: role=INPUT/OUTPUT, layout=mb,out,y,x,i; stick=i; stick_size=[64], wkSlice=i=0 mb=core_id out=0 x=0 y=0, address=-0xc2..-0xa3 (32 unique)
  - 0_lx: role=INPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x0
  - 1_lx: role=INPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x0
  - 2_lx: role=OUTPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x20000
  - 0_hbm: role=INPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=0 out=core_id, address=-0xd7..-0xbf (25 unique)
  - 1_hbm: role=OUTPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=0 out=core_id, address=-0xf0..-0xd8 (25 unique)
  - 0_hbm: role=INPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=-0x41..-0x22 (32 unique)
  - 1_lx: role=INPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x20000
  - 2_lx: role=OUTPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x0

sdsc_5: identity + mean + batchmatmul + mul (32 cores)
  - 0_hbm: role=INPUT, layout=y,out,mb,x; stick=out; stick_size=[64], wkSlice=mb=0 out=0 x=0 y=core_id, address=-0xe2..-0xc3 (32 unique)
  - 1_lx: role=OUTPUT, layout=y,out,mb,x; stick=out; stick_size=[64], wkSlice=mb=0 out=0 x=0 y=core_id, address=0x0
  - 0_lx: role=INPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x20000
  - 1_lx: role=INPUT/OUTPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x40000
  - 0_hbm: role=INPUT, layout=mb,in; stick=in; stick_size=[64], wkSlice=in=0 mb=0:7 (8 unique) out=0:3 (4 unique), address=-0xf8..-0xf1 (8 unique)
  - 1_hbm: role=INPUT, layout=mb,in; stick=in; stick_size=[64], wkSlice=in=0 mb=0:7 (8 unique) out=0:3 (4 unique), address=-0xfc..-0xf9 (4 unique)
  - 2_lx: role=OUTPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=in=0 mb=0:7 (8 unique) out=0:3 (4 unique), address=0x20000
  - 0_lx: role=INPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x0
  - 1_hbm: role=INPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=-0x42
  - 2_hbm: role=OUTPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=-0x62..-0x43 (32 unique)

sdsc_6: mul + add + ReStickifyOpHBM (32 cores)
  - 0_lx: role=INPUT, layout=out,x,mb; stick=out; stick_size=[64], wkSlice=mb=0 out=0 x=core_id, address=0x0
  - 1_hbm: role=INPUT, layout=out,x,mb; stick=out; stick_size=[64], wkSlice=mb=0 out=0 x=core_id, address=-0xe3
  - 2_lx: role=OUTPUT, layout=out,x,mb; stick=out; stick_size=[64], wkSlice=mb=0 out=0 x=core_id, address=0x20000
  - 0_lx: role=INPUT, layout=out,x; stick=x; stick_size=[64], wkSlice=out=core_id x=0, address=0x40000
  - 1_hbm: role=INPUT, layout=out,x; stick=x; stick_size=[64], wkSlice=out=core_id x=0, address=-0x6e
  - 2_lx: role=OUTPUT, layout=out,x; stick=x; stick_size=[64], wkSlice=out=core_id x=0, address=0x40000
  - 0_lx: role=INPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x20000
  - 1_hbm: role=INPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=-0xfd
  - 2_lx: role=OUTPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x20000
  - 0_hbm: role=INPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=-0x82..-0x63 (32 unique)
  - 1_hbm: role=OUTPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=-0xa2..-0x83 (32 unique)

sdsc_7: ReStickifyOpHBM + rsqrt + add + batchmatmul (32 cores)
  - 0_lx: role=INPUT, layout=mb,out,x; stick=out; stick_size=[64], wkSlice=mb=core_id out=0 x=0, address=0x20000
  - 1_hbm: role=OUTPUT, layout=mb,out,x; stick=out; stick_size=[64], wkSlice=mb=core_id out=0 x=0, address=-0x103..-0xe4 (32 unique)
  - 0_lx: role=INPUT, layout=out,x; stick=x; stick_size=[64], wkSlice=out=core_id x=0, address=0x40000
  - 1_lx: role=OUTPUT, layout=out,x; stick=x; stick_size=[64], wkSlice=out=core_id x=0, address=0x40000
  - 0_lx: role=INPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x20000
  - 1_lx: role=INPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x0
  - 2_hbm: role=OUTPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=-0x11d..-0xfe (32 unique)
  - 0_hbm: role=INPUT, layout=mb,in; stick=in; stick_size=[64], wkSlice=in=0 mb=0:3 (4 unique) out=0:7 (8 unique), address=-0xa6..-0xa3 (4 unique)
  - 1_hbm: role=INPUT, layout=mb,in; stick=in; stick_size=[64], wkSlice=in=0 mb=0:3 (4 unique) out=0:7 (8 unique), address=-0xae..-0xa7 (8 unique)
  - 2_hbm: role=OUTPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=in=0 mb=0:3 (4 unique) out=0:7 (8 unique), address=-0xce..-0xaf (32 unique)

sdsc_8: batchmatmul + mul (32 cores)
  - 0_hbm: role=INPUT, layout=mb,in,x; stick=in; stick_size=[64], wkSlice=in=0 mb=0 out=0:1 (2 unique) x=0:15 (16 unique), address=-0x113..-0x104 (16 unique)
  - 1_hbm: role=INPUT, layout=mb,in,x; stick=in; stick_size=[64], wkSlice=in=0 mb=0 out=0:1 (2 unique) x=0:15 (16 unique), address=-0x115..-0x114 (2 unique)
  - 2_lx: role=OUTPUT, layout=out,x,mb; stick=out; stick_size=[64], wkSlice=in=0 mb=0 out=0:1 (2 unique) x=0:15 (16 unique), address=0x0
  - 0_lx: role=INPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x0
  - 1_lx: role=INPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x40000
  - 2_lx: role=OUTPUT, layout=out,mb; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x40800

sdsc_9: add + mul (32 cores)
  - 0_lx: role=INPUT, layout=out,x,mb; stick=out; stick_size=[64], wkSlice=mb=0 out=0 x=core_id, address=0x0
  - 1_hbm: role=INPUT, layout=out,x,mb; stick=out; stick_size=[64], wkSlice=mb=0 out=0 x=core_id, address=-0x135..-0x116 (32 unique)
  - 2_lx: role=OUTPUT, layout=out,x,mb; stick=out; stick_size=[64], wkSlice=mb=0 out=0 x=core_id, address=0x80000
  - 0_lx: role=INPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=0x40800
  - 1_hbm: role=INPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=-0x6f
  - 2_hbm: role=OUTPUT, layout=mb,out; stick=out; stick_size=[64], wkSlice=mb=core_id out=0, address=-0x8f..-0x70 (32 unique)

sdsc_10: max (32 cores)
  - 0_lx: role=INPUT, layout=mb,out,x; stick=out; stick_size=[64], wkSlice=mb=0 out=0 x=core_id, address=0x80000
  - 1_lx: role=INPUT/OUTPUT, layout=mb,out,x; stick=out; stick_size=[64], wkSlice=mb=0 out=0 x=core_id, address=0x0

sdsc_11: sub (32 cores)
  - 0_lx: role=INPUT, layout=mb,out,x; stick=out; stick_size=[64], wkSlice=mb=0 out=0 x=core_id, address=0x80000
  - 1_lx: role=INPUT, layout=mb,out,x; stick=out; stick_size=[64], wkSlice=mb=0 out=0 x=core_id, address=0x0
  - 2_lx: role=OUTPUT, layout=mb,out,x; stick=out; stick_size=[64], wkSlice=mb=0 out=0 x=core_id, address=0x100000

sdsc_12: exp (32 cores)
  - 0_lx: role=INPUT, layout=mb,out,x; stick=out; stick_size=[64], wkSlice=mb=0 out=0 x=core_id, address=0x100000
  - 1_lx: role=OUTPUT, layout=mb,out,x; stick=out; stick_size=[64], wkSlice=mb=0 out=0 x=core_id, address=0x100000

sdsc_13: sum (32 cores)
  - 0_lx: role=INPUT, layout=mb,out,x; stick=out; stick_size=[64], wkSlice=mb=0 out=0 x=core_id, address=0x100000
  - 1_lx: role=INPUT/OUTPUT, layout=mb,out,x; stick=out; stick_size=[64], wkSlice=mb=0 out=0 x=core_id, address=0x0

sdsc_14: realdiv (32 cores)
  - 0_lx: role=INPUT, layout=mb,out,x; stick=out; stick_size=[64], wkSlice=mb=0 out=0 x=core_id, address=0x100000
  - 1_lx: role=INPUT, layout=mb,out,x; stick=out; stick_size=[64], wkSlice=mb=0 out=0 x=core_id, address=0x0
  - 2_lx: role=OUTPUT, layout=mb,out,x; stick=out; stick_size=[64], wkSlice=mb=0 out=0 x=core_id, address=0x100000

sdsc_15: identity (32 cores)
  - 0_hbm: role=INPUT, layout=out,mb,y,x; stick=out; stick_size=[64], wkSlice=mb=0 out=0 x=0 y=core_id, address=-0x156..-0x137 (32 unique)
  - 1_hbm: role=OUTPUT, layout=out,mb,y,x; stick=out; stick_size=[64], wkSlice=mb=0 out=0 x=0 y=core_id, address=-0x176..-0x157 (32 unique)

sdsc_16: batchmatmul (32 cores)
  - 0_lx: role=INPUT, layout=x,in,mb; stick=in; stick_size=[64], wkSlice=in=0 mb=core_id out=0 x=0, address=0x100000
  - 1_hbm: role=INPUT, layout=x,in,mb; stick=in; stick_size=[64], wkSlice=in=0 mb=core_id out=0 x=0, address=-0x177
  - 2_lx: role=OUTPUT, layout=out,mb,x; stick=out; stick_size=[64], wkSlice=in=0 mb=core_id out=0 x=0, address=0x0

sdsc_17: identity (32 cores)
  - 0_lx: role=INPUT, layout=out,x,mb; stick=out; stick_size=[64], wkSlice=mb=0 out=0 x=core_id, address=0x0
  - 1_hbm: role=OUTPUT, layout=out,x,mb; stick=out; stick_size=[64], wkSlice=mb=0 out=0 x=core_id, address=-0x197..-0x178 (32 unique)
