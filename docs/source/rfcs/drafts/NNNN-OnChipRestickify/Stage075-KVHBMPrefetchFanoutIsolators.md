# Stage 075: K/V HBM Prefetch Fanout Isolators

Date: 2026-05-28

## Purpose

Stage074 showed that a direct all-core future K/V HBM prefetch is value-correct
when delayed until after current attention compute.  Stage075 probes whether a
more warp-specialized shape can replace that all-core HBM fill:

```text
current attention compute
loader/source HBM fill into LX
LX fanout to future consumer cores
future attention consumer reads prefilled LX
```

The goal is to find a correct loader/fanout primitive before trying to overlap
it with current attention compute.

## Implemented Probes

Added default-off gates:

```text
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_SOURCE_FANOUT=1
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT=1
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_LOADER_CORE={0..31}
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_LOADER_LX_BASE={-2,-1,addr}
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_FANOUT_USE_UNICAST=1
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_FANOUT_USE_LXSFP_LX_TRANSFERS={0,1}
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_FANOUT_COPYBACK_CORE={-1,0..31}
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_FANOUT_RESTRICT_TO_COPYBACK_CORE=1
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_LOADER_COPYBACK_WITHOUT_FANOUT=1
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT_FULL_TILE_PIECES=1
```

and sweep variants:

```text
onchip_hbm_kv_layout_xform_kv_hbm_prefetch_source_fanout_probe
onchip_hbm_kv_layout_xform_kv_hbm_prefetch_source_fanout_tail_probe
onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_tail_probe
onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_unicast_tail_probe
onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_copyback_tail_probe
onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_lxfifo_copyback_tail_probe
onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_local_copyback_tail_probe
onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_direct_copyback_tail_probe
onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_local_copyback_tail_probe
onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_tail_probe
onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_overlap_probe
onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_overlap_core31_probe
onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_overlap_safesrc_probe
onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_overlap_no_after_sync_probe
onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_direct_copyback_overlap_core31_probe
onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_direct_copyback_overlap_safesrc_probe
onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_direct_copyback_overlap_no_after_sync_probe
```

`source_fanout` loads one future K/V source slice on each original source core,
then fans those source LX slices out to all 32 future consumer cores.

`loader_fanout` loads the whole future K/V tile into core 0, then fans that
single loader-core LX copy out to all 32 future consumer cores.

`fanout_copyback` runs the same loader/fanout path, then stores one selected
consumer LX replica back to the original future K/V HBM address and leaves the
future attention consumer HBM-backed.  This isolates whether the fanout output
is already value-wrong before the future batchmatmul consumer reads it.

Both fanout schedules now include an explicit all-core `nop` rendezvous between
the loader HBM fill and the `STCDPOpLx` fanout, so non-loader cores cannot enter
the fanout row before loader cores have issued their HBM loads.

`local_copyback` restricts the fanout output pieces and fanout-row execution to
the selected copyback core.  This tests the same-core loader copy path without
the full all-core fanout ring.

`loader_direct_copyback` uses the same loader HBM fill and HBM store machinery
but removes the `STCDPOpLx` fanout row entirely.  It stores the loader-core LX
buffer directly back to the original future K/V HBM address.

`loader_fanout_fulltile` keeps the single-loader shape but collapses the
`STCDPOpLx` fanout descriptor from eight `x_` subpieces to one full-tile
producer piece and one full-tile destination piece per participating consumer
core.  This isolates subpiece splitting from the LX copy contract.

`loader_core` selects which core performs the single-loader HBM fill.  This
distinguishes a core-0-specific hazard from a general hazard when any current
compute core also runs the loader HBM fill in the same row.

`loader_lx_base` selects the transient loader source LX buffer.  The value `-1`
keeps the original source base, and `-2` places the loader source after the
future consumer LX region.  This tests whether the same-row failure is caused
by current-compute scratch clobbering the low loader source buffer.

The full-tile fanout and direct-copyback overlap schedules now honor
`SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_OVERLAP_AFTER_SYNC=0` for the
paired loader-HBM/current-compute row.  This probes whether the failure is an
after-sync placement issue rather than the paired row itself.

## Device Results

Overlapped source-fanout:

```text
cache = /tmp/sdpa-stage131-source-fanout-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_source_fanout_probe-B1-H8-L256-D64-C0-752566-797348
status = failed
Mismatched elements: 5658 / 131072 (4.3%)
Greatest absolute difference: 0.5986328125 at index (0, 2, 129, 63)
```

Tail source-fanout with explicit barrier:

```text
cache = /tmp/sdpa-stage133-source-fanout-tail-barrier-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_source_fanout_tail_probe-B1-H8-L256-D64-C0-753700-357405
status = failed
Mismatched elements: 3141 / 131072 (2.4%)
Greatest absolute difference: 0.5986328125 at index (0, 2, 129, 63)
```

Tail source-fanout using direct HBM-to-LX fills rather than LX-roundtrip output
pieces was effectively unchanged:

```text
cache = /tmp/sdpa-stage134-source-fanout-tail-directfill-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_source_fanout_tail_probe-B1-H8-L256-D64-C0-754062-776216
status = failed
Mismatched elements: 3139 / 131072 (2.4%)
```

Tail single-loader fanout lowered cleanly but was still value-wrong:

```text
cache = /tmp/sdpa-stage135-loader-fanout-tail-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_tail_probe-B1-H8-L256-D64-C0-754418-680995
status = failed
Mismatched elements: 2358 / 131072 (1.8%)
Greatest absolute difference: 0.5986328125 at index (0, 2, 129, 63)
```

Disabling STCDP subpiece reuse on the loader fanout did not produce a useful
value result; DXP rejected the shape:

```text
DtException: maxGrpId <= sysDef.maxGroupID
dcg/dcg_fe/pcfg_gen/stcdpOp.cpp line 2695
```

After fixing generated fanout LX piece addresses so each `x_` slice uses
`base + piece_offset`, the loader-fanout tail probe improved but still failed:

```text
cache = /tmp/sdpa-stage139-loader-fanout-tail-offsetfix-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_tail_probe-B1-H8-L256-D64-C0-757204-395379
status = failed
Mismatched elements: 2261 / 131072 (1.7%)
Greatest absolute difference: 0.349609375 at index (0, 1, 159, 48)
```

Forcing the same fanout to STCDP-LX unicast carried `useUnicast: 1` in the
generated dataop but produced the same value result:

```text
cache = /tmp/sdpa-stage140-loader-fanout-unicast-tail-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_unicast_tail_probe-B1-H8-L256-D64-C0-757596-898700
status = failed
Mismatched elements: 2261 / 131072 (1.7%)
Greatest absolute difference: 0.349609375 at index (0, 1, 159, 48)
```

Marking the future consumer K/V LX input as external worsened the result:

```text
cache = /tmp/sdpa-stage141-loader-fanout-external-tail-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_tail_probe-B1-H8-L256-D64-C0-757899-818767
status = failed
Mismatched elements: 3016 / 131072 (2.3%)
Greatest absolute difference: 0.5087890625 at index (0, 4, 181, 20)
```

Changing fanout source/destination slice `validGap_` to describe the full
contiguous K/V allocation, while keeping each slice extent at `x_=1`, also
made it into the generated JSON but did not change the value result:

```text
cache = /tmp/sdpa-stage142-loader-fanout-stridefix-tail-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_tail_probe-B1-H8-L256-D64-C0-758213-162134
status = failed
Mismatched elements: 2261 / 131072 (1.7%)
Greatest absolute difference: 0.349609375 at index (0, 1, 159, 48)
```

Copying the core-0 fanout replica back to HBM and keeping the future consumer
HBM-backed still failed:

```text
cache = /tmp/sdpa-stage143-loader-fanout-copyback-tail-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_copyback_tail_probe-B1-H8-L256-D64-C0-758936-350709
status = failed
Mismatched elements: 2288 / 131072 (1.7%)
Greatest absolute difference: 0.346923828125 at index (0, 1, 159, 48)
```

After making the lower-stack `useLXSFPLXTransfers` STCDP-LX mode configurable
instead of hard-coded true, forcing the fanout to the LX-LU/SU FIFO transport
did not reach device execution.  The generated fanout op carried
`useLXSFPLXTransfers: 0`, but DXP aborted during PCFG-to-dataflow lowering:

```text
cache = /tmp/sdpa-stage144-loader-fanout-lxfifo-copyback-tail-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_lxfifo_copyback_tail_probe-B1-H8-L256-D64-C0-763012-497301
status = failed
DtException: senpcfgs_.count(pair)
dcc/src/Conversion/PCFGToDataflowIR/PCFGToDataflowIR.cpp line 851
```

Rerunning the default SFP-mediated path after the lower-stack rebuild confirmed
the previous copyback result and exported `useLXSFPLXTransfers: 1`:

```text
cache = /tmp/sdpa-stage145-loader-fanout-copyback-tail-postlxsfp-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_copyback_tail_probe-B1-H8-L256-D64-C0-763537-958380
status = failed
Mismatched elements: 2288 / 131072 (1.7%)
Greatest absolute difference: 0.346923828125 at index (0, 1, 159, 48)
```

Copying back from core 31 rather than core 0 was also wrong, with a slightly
different mismatch count and the same hot coordinate:

```text
cache = /tmp/sdpa-stage146-loader-fanout-copyback-core31-tail-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_copyback_tail_probe-B1-H8-L256-D64-C0-763800-303438
status = failed
Mismatched elements: 2257 / 131072 (1.7%)
Greatest absolute difference: 0.349609375 at index (0, 1, 159, 48)
```

After teaching DCC to map `LXLUSUFIFO` endpoints onto the paired `LXLU`/`LXSU`
units, the forced-FIFO probe advanced past the previous
`senpcfgs_.count(pair)` abort but hit a later ProgIR guard:

```text
cache = /tmp/sdpa-stage147-loader-fanout-lxfifo-copyback-dccfix-tail-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_lxfifo_copyback_tail_probe-B1-H8-L256-D64-C0-765404-103448
status = failed
DtException: is_any_of(consumer, PE, SFP)
dcc/src/Conversion/SentientToProgIR/Utils.cpp line 93
```

After allowing `LXLU`/`LXSU` in that ProgIR helper, the FIFO transport path
reached device execution.  It produced the same value result as the default
SFP-mediated path:

```text
cache = /tmp/sdpa-stage148-loader-fanout-lxfifo-copyback-progirfix-tail-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_lxfifo_copyback_tail_probe-B1-H8-L256-D64-C0-766549-889683
status = failed
Mismatched elements: 2288 / 131072 (1.7%)
Greatest absolute difference: 0.346923828125 at index (0, 1, 159, 48)
```

Restricting loader fanout to only the copyback core also failed.  The generated
shape was accepted and the current sidecar proved the restriction was active:
`coreIdsUsed_ == [0]`, the `STCDPOpLx` fanout had 8 producer subpieces and 8
consumer subpieces, and non-copyback cores stopped after the barrier.

```text
cache = /tmp/sdpa-stage149-loader-fanout-local-copyback-tail-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_local_copyback_tail_probe-B1-H8-L256-D64-C0-766934-213206
status = failed
Mismatched elements: 2257 / 131072 (1.7%)
Greatest absolute difference: 0.349609375 at index (0, 1, 159, 48)
```

Forcing that same local-only probe to the FIFO LX transport was identical:

```text
cache = /tmp/sdpa-stage150-loader-fanout-local-lxfifo-copyback-tail-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_local_copyback_tail_probe-B1-H8-L256-D64-C0-767499-59795
status = failed
Mismatched elements: 2257 / 131072 (1.7%)
Greatest absolute difference: 0.349609375 at index (0, 1, 159, 48)
```

Removing the intervening `STCDPOpLx` row and copying the loader LX buffer
directly back to HBM passed:

```text
cache = /tmp/sdpa-stage151-loader-direct-copyback-tail-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_direct_copyback_tail_probe-B1-H8-L256-D64-C0-767898-3873
status = ok
median = 0.551034ms
max_abs_error = 0.00439453125
```

Collapsing the local fanout copyback descriptor to full-tile LX pieces passed:

```text
cache = /tmp/sdpa-stage152-loader-fanout-fulltile-local-copyback-tail-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_local_copyback_tail_probe-B1-H8-L256-D64-C0-768285-69262
status = ok
median = 0.567868ms
max_abs_error = 0.00439453125
```

Running the same full-tile fanout copyback across all consumer cores also
passed:

```text
cache = /tmp/sdpa-stage153-loader-fanout-fulltile-allcore-copyback-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_local_copyback_tail_probe-B1-H8-L256-D64-C0-768462-603290
status = ok
median = 0.559472ms
max_abs_error = 0.00439453125
```

Using the all-core full-tile fanout as the actual future K/V consumer input
passed in the tail-current schedule:

```text
cache = /tmp/sdpa-stage154-loader-fanout-fulltile-tail-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_tail_probe-B1-H8-L256-D64-C0-768639-524393
status = ok
median = 0.548592ms
max_abs_error = 0.00439453125
```

The named full-tile tail variant is the current clean passing artifact:

```text
cache = /tmp/sdpa-stage161-loader-fanout-fulltile-tail-named-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_tail_probe-B1-H8-L256-D64-C0-770198-88027
status = ok
median = 0.554030ms
mean = 0.552374ms
max_abs_error = 0.00439453125
```

Reintroducing same-row overlap of the loader HBM fill with current compute
failed, even with the full-tile fanout descriptor:

```text
cache = /tmp/sdpa-stage155-loader-fanout-fulltile-overlap-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_tail_probe-B1-H8-L256-D64-C0-768816-328498
status = failed
Mismatched elements: 152 / 131072 (0.1%)
Greatest absolute difference: 0.36181640625 at index (0, 7, 4, 3)
```

Routing that full-tile overlap probe through corelet 1 did not change the
result:

```text
cache = /tmp/sdpa-stage156-loader-fanout-fulltile-overlap-corelet1-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_tail_probe-B1-H8-L256-D64-C0-769104-982314
status = failed
Mismatched elements: 152 / 131072 (0.1%)
```

Serializing the full-tile loader fanout before current compute passed:

```text
cache = /tmp/sdpa-stage157-loader-fanout-fulltile-serial-before-current-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_tail_probe-B1-H8-L256-D64-C0-769297-216650
status = ok
median = 0.568464ms
max_abs_error = 0.00439453125
```

The same-row hazard is not caused by the future consumer or the LX fanout.
Full-tile all-core copyback with the future consumer still HBM-backed failed in
the same way:

```text
cache = /tmp/sdpa-stage158-loader-fanout-fulltile-copyback-overlap-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_local_copyback_tail_probe-B1-H8-L256-D64-C0-769624-935547
status = failed
Mismatched elements: 152 / 131072 (0.1%)
```

Direct loader copyback without any fanout row also failed when the loader HBM
fill overlapped current compute:

```text
cache = /tmp/sdpa-stage159-loader-direct-copyback-overlap-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_direct_copyback_tail_probe-B1-H8-L256-D64-C0-769847-660832
status = failed
Mismatched elements: 152 / 131072 (0.1%)
```

Prefilling the current compute inputs before the direct-copyback overlap did
not repair the issue and made the mismatch much larger:

```text
cache = /tmp/sdpa-stage160-loader-direct-copyback-overlap-prefill-current-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_direct_copyback_tail_probe-B1-H8-L256-D64-C0-770023-931694
status = failed
Mismatched elements: 13107 / 131072 (10.0%)
Greatest absolute difference: 0.8564453125
```

Moving the direct-copyback overlap loader from core 0 to core 31 also failed:

```text
cache = /tmp/sdpa-stage162-loader-direct-copyback-overlap-core31-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_direct_copyback_overlap_core31_probe-B1-H8-L256-D64-C0-772717-939632
status = failed
Mismatched elements: 151 / 131072 (0.1%)
Greatest absolute difference: 0.2213134765625 at index (0, 3, 252, 48)
```

The actual full-tile fanout overlap path with the loader on core 31 failed in
the same way:

```text
cache = /tmp/sdpa-stage163-loader-fanout-fulltile-overlap-core31-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_overlap_core31_probe-B1-H8-L256-D64-C0-772979-924444
status = failed
Mismatched elements: 151 / 131072 (0.1%)
Greatest absolute difference: 0.2213134765625 at index (0, 3, 252, 48)
```

Moving the direct-copyback overlap loader source LX buffer from the original
low source base to the auto safe source base after the consumer region did not
repair the paired-row failure:

```text
cache = /tmp/sdpa-stage164-loader-direct-copyback-overlap-safesrc-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_direct_copyback_overlap_safesrc_probe-B1-H8-L256-D64-C0-774266-747011
status = failed
Mismatched elements: 199 / 131072 (0.2%)
Greatest absolute difference: 0.429443359375 at index (0, 2, 6, 38)
loader_lx_base = 540672
```

Disabling the after-sync bit on the direct-copyback paired row also failed:

```text
cache = /tmp/sdpa-stage165-loader-direct-copyback-overlap-no-after-sync-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_direct_copyback_overlap_no_after_sync_probe-B1-H8-L256-D64-C0-774757-155139
status = failed
Mismatched elements: 199 / 131072 (0.2%)
Greatest absolute difference: 0.429443359375 at index (0, 2, 6, 38)
```

Forcing the loader HBM fill/copyback dataops into the older HBM/LX roundtrip
STCDP shape failed with the same signature:

```text
cache = /tmp/sdpa-stage166-loader-direct-copyback-overlap-roundtrip-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_direct_copyback_tail_probe-B1-H8-L256-D64-C0-774977-997287
status = failed
Mismatched elements: 199 / 131072 (0.2%)
Greatest absolute difference: 0.429443359375 at index (0, 2, 6, 38)
```

Rerunning the current default direct-copyback overlap path still failed:

```text
cache = /tmp/sdpa-stage167-loader-direct-copyback-overlap-current-default-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_direct_copyback_tail_probe-B1-H8-L256-D64-C0-775197-261754
status = failed
Mismatched elements: 199 / 131072 (0.2%)
Greatest absolute difference: 0.429443359375 at index (0, 2, 6, 38)
```

The actual full-tile fanout path with the after-sync bit disabled also failed:

```text
cache = /tmp/sdpa-stage168-loader-fanout-fulltile-overlap-no-after-sync-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_overlap_no_after_sync_probe-B1-H8-L256-D64-C0-775417-786679
status = failed
Mismatched elements: 199 / 131072 (0.2%)
Greatest absolute difference: 0.429443359375 at index (0, 2, 6, 38)
```

## Generated Shape

The source-fanout tail schedule after the barrier fix is:

```text
core 0: [[0,-1,0,1],[-1,0,1,1],[1,-1,1,1],[9,-1,1,1],[10,-1,1,0]]
core 8: [[0,-1,0,1],[-1,0,1,1],[9,-1,1,1],[10,-1,1,0]]
```

The single-loader fanout tail schedule is:

```text
core 0: [[0,-1,0,1],[-1,0,1,1],[1,-1,1,1],[2,-1,1,1],[3,-1,1,0]]
core 1: [[0,-1,0,1],[-1,0,1,1],[2,-1,1,1],[3,-1,1,0]]
```

where dataop 1 is the core-0 full-tile `STCDPOpHBM`, dataop 2 is the all-core
barrier, and dataop 3 is the all-core `STCDPOpLx` fanout.

The copyback variant appends dataop 4, a selected-core `STCDPOpHBM` store from
the chosen consumer LX replica back to the original future K/V HBM address:

```text
core 0: [[0,-1,0,1],[-1,0,1,1],[1,-1,1,1],[2,-1,1,1],[3,-1,1,1],[4,-1,1,0]]
core 2: [[0,-1,0,1],[-1,0,1,1],[2,-1,1,1],[3,-1,1,0]]
```

The local-copyback variant uses the same dataop list but restricts dataop 3 to
the copyback core:

```text
core 0: [[0,-1,0,1],[-1,0,1,1],[1,-1,1,1],[2,-1,1,1],[3,-1,1,1],[4,-1,1,0]]
core 1: [[0,-1,0,1],[-1,0,1,1],[2,-1,1,0]]
core 31: [[0,-1,0,1],[-1,0,1,1],[2,-1,1,0]]
```

The generated Stage149 fanout op lowered to one PCFG entry with `lxlu0`,
`lxsu0`, and `pe0`; `pSubPiece=8`, `cSubPiece=8`, and `dtTable_=8`.

The Stage151 direct-copyback schedule removes the fanout row:

```text
core 0: [[0,-1,0,1],[-1,0,1,1],[1,-1,1,1],[2,-1,1,1],[3,-1,1,0]]
core 1: [[0,-1,0,1],[-1,0,1,1],[2,-1,1,0]]
core 31: [[0,-1,0,1],[-1,0,1,1],[2,-1,1,0]]
```

Its current sidecar had `opFuncsUsed_ = ["nop", "STCDPOpHBM", "nop",
"STCDPOpHBM"]`.  The copyback input placement was loader LX base `16384`, and
the output placement stored to HBM start `5216`.

The Stage152 full-tile local fanout copyback keeps the same row structure as
Stage149 but lowers the fanout op with `pSubPiece=1` and `cSubPiece=1`.  The
Stage153 all-core copyback keeps `pSubPiece=1` and expands to one full-tile
consumer piece per participating core.

The Stage155 overlap schedule placed the loader HBM fill in the same row as
current compute on core 0:

```text
core 0: [[0,-1,0,1],[1,0,1,1],[2,-1,1,1],[3,-1,1,0]]
core 1: [[0,-1,0,1],[-1,0,1,1],[2,-1,1,1],[3,-1,1,0]]
core 31: [[0,-1,0,1],[-1,0,1,1],[2,-1,1,1],[3,-1,1,0]]
```

Dataop 1 is the loader-core `STCDPOpHBM` fill.  That `[1,0,1,1]` row is the
remaining unsafe shape.

The Stage162 direct-copyback core-31 schedule moved that same unsafe row from
core 0 to core 31 and removed the fanout row:

```text
core 0: [[0,-1,0,1],[-1,0,1,1],[2,-1,1,0]]
core 31: [[0,-1,0,1],[1,0,1,1],[2,-1,1,1],[3,-1,1,0]]
```

The Stage163 full-tile fanout core-31 schedule kept all-core fanout but still
placed the loader HBM fill in the current-compute row on core 31:

```text
core 0: [[0,-1,0,1],[-1,0,1,1],[2,-1,1,1],[3,-1,1,0]]
core 31: [[0,-1,0,1],[1,0,1,1],[2,-1,1,1],[3,-1,1,0]]
```

The Stage164 safe-source direct-copyback schedule kept the paired row but moved
the loader source LX address:

```text
core 0: [[0,-1,0,1],[1,0,1,1],[2,-1,1,1],[3,-1,1,0]]
loader_lx_base = 540672
```

The Stage165 and Stage168 no-after-sync schedules changed only the after-sync
bit on the paired row:

```text
direct copyback core 0:
[[0,-1,0,1],[1,0,1,0],[2,-1,1,1],[3,-1,1,0]]

full-tile fanout core 0:
[[0,-1,0,1],[1,0,1,0],[2,-1,1,1],[3,-1,1,0]]
```

## Interpretation

The latest evidence changes the target:

- The split-by-x loader fanout descriptor is value-wrong.  SFP-mediated and
  FIFO LX transports both reproduce the failure, so the local transfer backend
  is not the root cause.
- The full-tile loader fanout descriptor is value-correct for local copyback,
  all-core copyback, and the future-consumer tail path.
- The current usable warp-specialized-like K/V path is therefore
  `loader_fanout_fulltile_tail_probe`: one loader-core HBM fill, one full-tile
  LX fanout to consumer cores, then a future attention consumer reading the
  prefilled LX input.
- Same-row overlap of the loader HBM fill with current compute corrupts the
  loader-filled future data.  This holds even when the future consumer is
  HBM-backed and even when the fanout row is removed, so the overlap hazard is
  before fanout and before the future consumer.
- Corelet 1 routing did not make the same-row HBM fill safe.
- Moving the loader from core 0 to core 31 did not make the same-row HBM fill
  safe.  The hazard is not a core-0 special case.
- Moving the loader source LX buffer out of the low source region did not make
  the same-row HBM fill safe.  The hazard is not simply current-compute scratch
  clobbering the loader source address.
- Disabling the after-sync bit on the paired row did not make the same-row HBM
  fill safe.
- Forcing the older HBM/LX roundtrip STCDP shape did not make the same-row HBM
  fill safe.
- Serializing the same loader work before current compute is value-correct,
  which narrows the remaining unsafe contract to the paired loader-HBM/current-
  compute schedule row.

The next target is the scheduling/lower-stack contract for running a loader HBM
fill concurrently with the current attention compute.  A true performance
variant needs either a genuinely independent loader lane/core/corelet, a
lower-stack fix that makes the `[loader STCDPOpHBM, current DL]` row safe, or a
new primitive that models loader-to-consumer prefetch without sharing the
current compute row's unsafe resource path.
