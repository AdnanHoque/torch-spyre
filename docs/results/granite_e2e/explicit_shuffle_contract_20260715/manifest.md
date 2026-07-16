# Explicit DLDSC SHUFFLE Contract Experiment

## Objective

Prove or disprove that a frontend-allocated `S1 -> SHUFFLE -> S2` DLDSC
contract is sufficient for the grouped K-side all-gather in flash attention,
without custom all-gather metadata, backend destination allocation, HBM, or an
overcommitted LX partition.

Target shapes:

```text
Q   = [1, 4,  512, 128]
K/V = [1, 4, 4096, 128]
```

## Frozen Inputs

| Component | Reference |
|---|---|
| Source K producer SDSC | `generated_v2/source/producer_sdsc.json` |
| Score BMM consumer SDSC | `generated_v2/source/consumer_sdsc.json` |
| Torch state that produced the source SDSCs | `bb8218347114d8b23ed97b2add09210675252d73` |
| Deeptools PR1 base | `704c19f8fb7f0cc972f20404f9dd0010895a35e2` |
| spyre-perf-suite PR81 state | `3d33ca0ab3ec94f6713cb05bef2deb70d18fb170` |

The source hashes are recorded in `generated/fixture_validation.json` so later
fixture regeneration cannot silently use a different input.

## Explicit Memory Contract

The first isolated fixtures deliberately used adjacent regions:

| State | Diagnostic address/core | Size/core | Lifetime |
|---|---:|---:|---|
| S1 compact source shard | `0x24000` | 128 KiB | producer through SHUFFLE |
| S2 complete K operand | `0x44000` | 1 MiB | SHUFFLE through score BMM |

Those two regions do not overlap each other, which makes `0x44000` useful for
backend-only contract diagnosis. It is **not** a valid full-workload placement:
the real attention allocation has a live 512 KiB `buf7` region beginning at
`0x44000` during the score-BMM boundary.

The value-correct custom-materializer run instead uses the following
integration placement:

| State | Integration address/core | Size/core | Lifetime |
|---|---:|---:|---|
| S1 compact source shard | `0x24000` | 128 KiB | producer through SHUFFLE |
| existing live attention state | `0x44000` | 512 KiB | overlaps SHUFFLE/BMM boundary |
| S2 complete K operand | `0xc4000` | 1 MiB | SHUFFLE through score BMM |

Both S1 and S2 must be supplied by the frontend allocation plan. Deeptools must
not allocate another destination. See `LX_ALLOCATION_LIFETIME.md` for the
capacity and partition evidence.

## Variant A

`generated_v2/variant_a_redundant_coordinates` is the authoritative preferred
contract:

- The SHUFFLE input allocation map identifies 32 distinct `(head, Lk shard)`
  owners.
- The SHUFFLE input allocation keeps the producer's compact 512-wide physical
  fold while renaming its logical dimensions into the consumer schema.
- The SHUFFLE compute map redundantly assigns the same `(head, full-K)` work
  slice to the eight cores in each head group. The consumer BMM's `mb`/Lq
  coordinate is deliberately absent from this K-only operation.
- The SHUFFLE output has an explicit S2 allocation.
- The following BMM reads S2 with an allocation map equal to its compute map,
  so it must not request a second relayout.
- `lxRelayoutClassifications_` is empty throughout.

The earlier `generated/variant_a_redundant_coordinates` directory is retained
as historical replay input. It predates removal of the irrelevant score-BMM
`mb`/Lq loop from the K-only SHUFFLE and predates the corrected two-corelet
fold geometry. Do not use it for a production conclusion. The authoritative
SHUFFLE-only fixture has SHA-256
`89fbba9f699fc2fdae6b189a04464990ce85f560fa49c8db19a97a8e7b5bec51`.

`generated/diagnostic_full_stride_input` has the same logical mismatch but
clones the consumer's 4096-wide physical allocation for S1. It is deliberately
not a valid production contract. If it lowers while Variant A fails, the result
isolates missing support for distinct input/output physical strides rather than
a failure to infer the grouped all-gather.

`generated/negative_control_no_mismatch` uses the same input and output compute
map. DXP must not insert movement for this control.

`generated/diagnostic_scaled_redundant_allgather` gathers eight 64-wide source
shards into a 512-wide destination. It preserves the redundant-coordinate
all-gather and compact-to-expanded stride relationship while avoiding the
4096-wide stick geometry. This is the clean control for whether redundant
coordinates themselves are supported by SHUFFLE.

## Layout Question Under Test

The semantic dimensions and stick dimension align positionally:

```text
producer.x   -> consumer.out
producer.out -> consumer.in
producer.mb  -> consumer.x
```

However, that does not by itself prove physical compatibility. S1 is a compact
512-wide per-core allocation with a 1024-byte row stride; S2 embeds each shard
inside a complete 4096-wide K tensor with an 8192-byte row stride. The backend
must represent both physical strides or an explicit layout step is required.

`generated/expected_transfer_summary.json` captures this invariant. A replay
that reports 256 correct core pairs but uses the wrong row strides is not a
successful result.

## Reproduction

```bash
python3 generate_explicit_shuffle_fixtures.py \
  --producer generated_v2/source/producer_sdsc.json \
  --consumer generated_v2/source/consumer_sdsc.json \
  --output generated_v2

python3 build_expected_transfer_plan.py \
  --shuffle generated_v2/variant_a_redundant_coordinates/sdsc_1.json \
  --output generated_v2
```

## Stock And Patched Backend Result

On the clean SHUFFLE-capable Deeptools base `704c19f8fb`, both the preferred
redundant-coordinate fixture and the explicit meta-dimension fixture fail with:

```text
Scheduler failed to find a suitable op mapping for sdsc: 0_shuffle
```

A bounded experimental materializer gets past scheduler selection but reaches
DCG output-coverage validation with uncovered destination coordinates. A
manually enumerated bounded `STCDPOpLx` control lowers structurally and produces
the expected 256 shard-level placements (224 remote and 32 local). Therefore,
the ring carrier can represent the traffic, while current coordinate-only
SHUFFLE materialization cannot yet realize the expanding replicated output.

An isolated 141-line patch to `dxp/SdscRelayoutInsertion.cpp` closes that
structural backend gap for Variant A. With the same redundant-coordinate
contract, the patched backend:

- lowers through DXP, DCG, and DCC;
- emits eight bounded `STCDPOpLx` rows;
- realizes all 256 placements, including 224 remote and 32 local placements;
- preserves the compact 1024-byte S1 row stride and expanded 8192-byte S2 row
  stride; and
- uses the explicit LX endpoints without HBM, dynamic LX allocation, or
  `ReStickifyOpLx`.

The proof and exact patch are under `variant_a_exact_backend/`. This establishes
that coordinates plus explicit S1/S2 storage are structurally sufficient for
this affine grouped all-gather. It does not establish that the unmodified
backend supports the contract, nor does the diagnostic `S2=0x44000` fixture
prove full-workload allocation safety. Patterned AIU correctness and performance
remain separate gates.

This is a statement about current compiler behavior, not a proof that the
coordinate representation is mathematically incapable of describing the
distribution. See `COORDINATE_ONLY_STATUS.md` for the precise distinction.

## Parallel Tracks

| Pod | Track | Status |
|---|---|---|
| `adnan-cdx-spyre-dev-pf` | Variant A exact-backend materialization | structural DXP/DCG/DCC proof complete; AIU value/performance pending |
| `adnan-clc-spyre-dev-pf` | Variant B meta-dimension encoding | coordinate-only scheduler rejection reproduced |
| `adnan-spyre-dev-pf` | direct SHUFFLE versus staged layout patterned values | in progress |
| `adnan-spyre-current-pf` | HBM/custom controls and workload allocation evidence | controls archived |

Results will be archived under this directory without modifying either active
production PR branch.
