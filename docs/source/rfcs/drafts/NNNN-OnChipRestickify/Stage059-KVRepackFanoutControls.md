# Stage 059: K/V Repack Fanout Controls

Date: 2026-05-27

## Purpose

Stage058 proved that the executable K/V repack pair can be selected and run, but
the 2-source to 64-destination `STCDPOpLx` fanout is not value-correct.  Stage059
keeps the work on the warp-specialized prefill blocker: make the future K/V
batchmatmul input available on the 32 compute cores without falling back to HBM.

This stage adds focused descriptor controls for the K/V fanout.  It does not
complete the warp-specialized attention variant; all device A/Bs still fail
value correctness.

## Change

New gates:

```text
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_SELF_RESIDENT_SOURCE=0
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_USE_UNICAST=-1
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_FORCE_MC_MODE=-1
```

`SELF_RESIDENT_SOURCE=1` makes the low-core producer write its source pieces
directly at the consumer input LX base on producer cores, then skips those
producer-owned self destinations in the fanout.

`FORCE_MC_MODE=1/2/3` emits:

```json
{"name": "STCDPOpLx", "forceModeMC": {"force": 1, "val": N}}
```

The grouped fanout builder now includes producer cores in each nonzero group
data-op's `coreIdsUsed_`.  This lets Deeptools emit producer-side L3SU PCFGs for
all groups instead of lowering a group whose sources are absent from its core
set.

New sweep variants:

```text
kv_repack_pair_self_resident_auto
kv_repack_pair_force_mc1_auto
kv_repack_pair_force_mc2_auto
kv_repack_pair_force_mc3_auto
kv_repack_pair_group8_auto
kv_repack_pair_group4_auto
```

The existing `kv_repack_pair_group16_auto` now uses the corrected grouped
descriptor shape.

## Validation

Local syntax and whitespace checks:

```text
python3 -m py_compile torch_spyre/_inductor/config.py \
  torch_spyre/_inductor/codegen/bundle.py \
  torch_spyre/_inductor/onchip_realize.py \
  tools/onchip_sdpa_sweep.py \
  tests/_inductor/test_config_logic.py \
  tests/_inductor/test_onchip_realize_logic.py \
  tests/_inductor/test_onchip_sdpa_sweep_logic.py
git diff --check
```

Pod focused gates:

```text
python3 -m py_compile ...
pytest tests/_inductor/test_config_logic.py \
  tests/_inductor/test_onchip_sdpa_sweep_logic.py \
  tests/_inductor/test_onchip_realize_logic.py -q
```

Result:

```text
122 passed in 0.85s
```

The local checkout used for this turn does not have `pytest` installed, so
pytest validation was run in the dev pod worktree.

## Device Results

All device probes used `B=1,H=2,L=128,D=64,seed=0` with DXP debug enabled under:

```text
/home/adnan-cdx/dt-inductor-mixed/torch-spyre-stage039-two-sdsc-ifn
```

### Self-resident source

Cache:

```text
/tmp/sdpa-stage059-kv-repack-self-resident-kv_repack_pair_self_resident_auto-B1-H2-L128-D64-C0-645134-875843
```

Generated descriptor:

```text
source_lx_base=278528
consumer_lx_base=278528
destination_piece_count=62
prodConsList: {0: 31 consumers, 1: 31 consumers}
```

Result:

```text
Mismatched elements: 16289 / 16384 (99.4%)
Greatest absolute difference: nan at index (0, 1, 119, 4)
```

Skipping producer-self destinations does not fix the corruption.

### Forced multicast modes

Caches:

```text
/tmp/sdpa-stage059-kv-repack-force-mc12-kv_repack_pair_force_mc1_auto-B1-H2-L128-D64-C0-645821-579015
/tmp/sdpa-stage059-kv-repack-force-mc12-kv_repack_pair_force_mc2_auto-B1-H2-L128-D64-C0-645821-165507
/tmp/sdpa-stage059-kv-repack-force-mc3-kv_repack_pair_force_mc3_auto-B1-H2-L128-D64-C0-645453-215017
```

The final debug JSON preserves `forceModeMC` for all three values:

```text
forceModeMC={"force": 1, "val": 1}
forceModeMC={"force": 1, "val": 2}
forceModeMC={"force": 1, "val": 3}
```

All three fail the same way:

```text
Mismatched elements: 16283 / 16384 (99.4%)
Greatest absolute difference: nan at index (0, 1, 119, 4)
```

Forcing the multicast instruction route is not the missing piece.

### Corrected grouped fanout

Stage058 `group16` failed in DXP lowering because group1 omitted producer cores.
Stage059 includes producer cores in every group data-op.  For group16:

```text
group0 coreIdsUsed_=0..15
group1 coreIdsUsed_=[0, 1, 16..31]
core0 schedule=[[0,-1,0,1],[1,-1,1,1],[-1,0,1,0]]
core16 schedule=[[1,-1,0,1],[-1,0,1,0]]
```

Cache:

```text
/tmp/sdpa-stage059-kv-repack-group16-fixed-kv_repack_pair_group16_auto-B1-H2-L128-D64-C0-645637-272342
```

This now lowers and runs.  Final debug shows two data-ops with 16 consumers per
producer:

```text
prodConsList group0: {0: 16, 1: 16}
prodConsList group1: {0: 16, 1: 16}
```

But values are still wrong:

```text
Mismatched elements: 16292 / 16384 (99.4%)
Greatest absolute difference: nan at index (0, 1, 119, 4)
```

Smaller groups also run and remain wrong:

```text
group8 cache=/tmp/sdpa-stage059-kv-repack-groups84-kv_repack_pair_group8_auto-B1-H2-L128-D64-C0-646252-356928
prodConsList per data-op: {0: 8, 1: 8}
Mismatched elements: 16284 / 16384 (99.4%)
Greatest absolute difference: nan at index (0, 1, 119, 4)

group4 cache=/tmp/sdpa-stage059-kv-repack-groups84-kv_repack_pair_group4_auto-B1-H2-L128-D64-C0-646252-434892
prodConsList per data-op: {0: 4, 1: 4}
Mismatched elements: 16286 / 16384 (99.4%)
Greatest absolute difference: nan at index (0, 1, 119, 4)
```

## Current Status

Stage059 rules out several simple explanations for the K/V repack corruption:

- the failure is not only caused by producer cores also being destinations;
- the failure is not solved by forcing multicast mode 1, 2, or 3;
- the Stage058 grouped-fanout DXP assert was a descriptor bug, now fixed;
- reducing fanout from 32 consumers to 16, 8, or 4 consumers per producer still
  produces wrong values.

The useful next step is a copy-validation probe at this exact boundary: verify
the generated `STCDPOpLx` output in LX before the batchmatmul consumes it.  If
the copied K/V tiles are already corrupt, the blocker is the L3/LX fanout
contract.  If the copied tiles are correct, the blocker is the mixed-SDSC
handoff into the DL batchmatmul consumer.
