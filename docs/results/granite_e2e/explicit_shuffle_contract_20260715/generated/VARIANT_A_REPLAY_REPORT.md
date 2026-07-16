# Variant A Explicit SHUFFLE Replay

## Verdict

Variant A is **not realized correctly** by the tested Deeptools candidate.

After repairing two fixture-schema omissions in the isolated pod copy, DXP
recognizes one grouped all-gather relationship and enumerates 256 logical
source-to-destination relationships. It emits an `STCDPOpLx` DataOp, but DCG
fails before producing a physical transfer program.

The corrected compact fixture also exposes the more important semantic gap:
the generated relayout DataDsc is byte-for-byte identical to the full-stride
diagnostic after names are normalized. The honest source has a 1024-byte row
stride, but the generated DataDsc describes both source and destination with
an 8192-byte row stride. The compact physical source fold is therefore lost.

This is not a successful Variant A replay. Import and mismatch recognition
work; correct physical materialization does not.

## Tested Source And Build

- Pod: `adnan-spyre-dev-pf`
- Deeptools source:
  `/home/adnan/codex-isolated/swagath_shuffle_contract_20260715/deeptools-shuffle`
- Deeptools SHA: `704c19f8fb7f0cc972f20404f9dd0010895a35e2`
- Source worktree status: clean
- DXP binary:
  `/home/adnan/codex-isolated/explicit-shuffle-contract-20260715-variant-a/deeptools-build-704c19f8/dxp/dxp_standalone`
- DXP SHA256:
  `56f50a9250cdeae5b930ee1c54aacb92c8023123586fb89b95fa1d2ba6b6f2cb`
- Build log:
  `/home/adnan/codex-isolated/explicit-shuffle-contract-20260715-variant-a/build_704c19f8.log`
- Configure log:
  `/home/adnan/codex-isolated/explicit-shuffle-contract-20260715-variant-a/configure_704c19f8.log`

The exact source directory must also be used as `DEEPTOOLS_PATH`. Without that
override, `dxp_standalone` loads older installed DDL templates and reports a
false `DDL found but not suitable` failure for `SHUFFLE`.

## Replay Command

The same command shape was used for each fixture:

```bash
env -u DXP_LX_FRAC_AVAIL \
  DEEPTOOLS_PATH=/home/adnan/codex-isolated/swagath_shuffle_contract_20260715/deeptools-shuffle \
  DXP_VERBOSE=3 \
  DXP_DEBUG=1 \
  /home/adnan/codex-isolated/explicit-shuffle-contract-20260715-variant-a/deeptools-build-704c19f8/dxp/dxp_standalone \
  -b senulator \
  --dump-bundle-module \
  -d <fixture-directory>
```

## Isolated Fixture Repairs

No generator, active branch, or remote branch was changed. Repairs exist only
under the isolated pod experiment directory.

1. The generated unary SHUFFLE descriptor declared one corelet while DXP's
   legal schedule uses two. The isolated copy changes the relevant root and
   allocation coordinate cardinalities from one to two.
2. The generated SuperDSC omitted root-level `N_` and `unpadN_`. The relayout
   inserter reads piece extents from the SuperDSC fields rather than the nested
   DLDSC fields, so the omission produced zero-sized pieces. The isolated copy
   mirrors the nested DLDSC `N_` into those root fields.

Patches:

- Corelet repair, compact fixture:
  `/home/adnan/codex-isolated/explicit-shuffle-contract-20260715-variant-a/experiments/fix6_corelet_coordinate_cardinality/compact_variant_combined.patch`
- Corelet repair, full-stride diagnostic:
  `/home/adnan/codex-isolated/explicit-shuffle-contract-20260715-variant-a/experiments/fix6_corelet_coordinate_cardinality/full_stride_diagnostic_combined.patch`
- Corelet repair, negative control:
  `/home/adnan/codex-isolated/explicit-shuffle-contract-20260715-variant-a/experiments/fix6_corelet_coordinate_cardinality/negative_control_combined.patch`
- SuperDSC shape repair:
  `/home/adnan/codex-isolated/explicit-shuffle-contract-20260715-variant-a/experiments/fix7_superdsc_N/compact_schema_fix.patch`

## Results

| Fixture | Import / DL compile | Relayout rows | Logical relationships | Physical realization |
|---|---|---:|---:|---|
| Honest compact Variant A | Pass through original SHUFFLE DCC | 1 `STCDPOpLx` | 256 | Fail in DCG coverage check |
| Full-stride diagnostic | Pass through original SHUFFLE DCC | 1 `STCDPOpLx` | 256 | Same DCG failure |
| Negative control | Pass | 0 | 0 | Pass; original SDSC only |
| Producer -> SHUFFLE -> consumer | Producer passes; SHUFFLE reached | Not completed | Not completed | Segmentation fault while compiling SHUFFLE |

### Logical relationship evidence

For both mismatch fixtures, DXP prints 32 producer-core entries. Each producer
maps to eight destination cores in its `x` group, and reports
`maxConsumers: 8`:

```text
0 --> [ 0 4 8 12 16 20 24 28 ]
1 --> [ 1 5 9 13 17 21 25 29 ]
2 --> [ 2 6 10 14 18 22 26 30 ]
3 --> [ 3 7 11 15 19 23 27 31 ]
...
maxConsumers: 8
```

This is 32 x 8 = 256 logical relationships: 32 local and 224 remote. They are
recognized but not lowered into an executable transfer table.

### DCG failure

Both mismatch fixtures terminate with:

```text
DtException: 0, file .../dcg/dcg_fe/pcfg_gen/stcdpOp.cpp line 440
```

That check verifies that generated subpieces cover every destination element.
The inserted pieces use work-slice indices as starts (`0..7`) while each shard
has extent 512. Correct starts for this dimension are `0, 512, ..., 3584`.
The resulting subpieces cannot cover the 4096-wide destination.

The tested Deeptools source constructs the start directly from the work-slice
index in `dxp/SdscRelayoutInsertion.cpp`; it does not scale by the piece extent.

## Required Stride Check

The honest input allocation encodes:

```text
source out fold alpha = 512 FP16 values
source row stride = 512 * 2 bytes = 1024 bytes
```

The destination allocation encodes:

```text
destination out fold alpha = 4096 FP16 values
destination row stride = 4096 * 2 bytes = 8192 bytes
```

The full-stride diagnostic deliberately encodes 4096 FP16 values for both.

However, DXP creates the same relayout DataDsc for the honest and diagnostic
inputs. After normalizing names, their `.out.json` files have no differences.
Both relayout inputs are described as:

```text
wordLength = 2
layoutDimOrder = [out, in, x]
dimToLayoutSize.out = 4096
```

Thus the generated input row stride is 8192 bytes in both cases. The honest
1024-byte source stride is not preserved. The tested insertion code selects
the output allocation's layout order for both relayout LDSes in the direct
output-copy path.

Normalized comparison artifacts:

- Honest:
  `/home/adnan/codex-isolated/explicit-shuffle-contract-20260715-variant-a/runs/exact704c19f8/fix7_compact_shuffle_only/normalized_relayout.out.json`
- Full-stride diagnostic:
  `/home/adnan/codex-isolated/explicit-shuffle-contract-20260715-variant-a/runs/exact704c19f8/fix7_full_stride_diagnostic/normalized_relayout.out.json`
- `diff -u` result: no differences.

## Address, Allocation, And Fallback Evidence

- Source LX base on every core: `147456` (`0x24000`)
- Destination LX base on every core: `278528` (`0x44000`)
- Honest source footprint per core: `131072` bytes
- Destination footprint per core: `1048576` bytes
- Honest source and destination do not overlap.
- The full-stride diagnostic source footprint is `1048576` bytes and therefore
  overlaps the destination; it is structural diagnostic input, not value-safe.
- Both generated relayout LDSes have `hbmStartAddress_ = -1`; no HBM fallback
  is selected before the failure.
- The destination is the existing explicit S2 LX allocation at `278528`.
  No separate dynamic relayout allocation is present.

## Schedule Evidence

The insertion path intends to place the relayout immediately before the
consumer SHUFFLE. The earlier import-only replay emitted this order:

```text
0_shuffle-Relayout0.json
sdsc_0.json
```

With valid piece extents, DCG aborts before an optimized bundle can be emitted.
The full producer-to-consumer bundle also does not complete, so end-to-end
`producer -> relayout -> SHUFFLE -> consumer` ordering is not validated.

The negative control emits only:

```text
sdsc_0.json
```

## Artifact Paths

- Honest run:
  `/home/adnan/codex-isolated/explicit-shuffle-contract-20260715-variant-a/runs/exact704c19f8/fix7_compact_shuffle_only`
- Full-stride diagnostic run:
  `/home/adnan/codex-isolated/explicit-shuffle-contract-20260715-variant-a/runs/exact704c19f8/fix7_full_stride_diagnostic`
- Negative control run:
  `/home/adnan/codex-isolated/explicit-shuffle-contract-20260715-variant-a/runs/exact704c19f8/fix7_negative_control`
- Full-bundle run:
  `/home/adnan/codex-isolated/explicit-shuffle-contract-20260715-variant-a/runs/exact704c19f8/fix7_compact_full_bundle`
- Known Deeptools control fixture replay:
  `/home/adnan/codex-isolated/explicit-shuffle-contract-20260715-variant-a/experiments/upstream_core_work_div_incompt`

The known Deeptools control passes and produces non-empty transfer tables,
showing that the empty/failing transfer behavior is specific to the wider
grouped-shard case rather than a generally broken DCG build.

## Scope Note

No endpoint-fix work from CDX was copied, modified, or duplicated. This replay
stops at the independently observed Variant A contract/materialization gaps.
