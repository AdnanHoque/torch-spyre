# DLDSC SHUFFLE Evidence Matrix

## Decision

The tested DLDSC coordinates describe the intended grouped K distribution. The
unmodified Deeptools SHUFFLE path does not materialize it, but an isolated
141-line bounded-SHUFFLE patch derives and lowers the complete transfer plan
from the same coordinates and explicit LX endpoints.

This conclusion separates two claims:

1. **Representation:** redundant coordinates can express that each of eight
   cores owns the complete K operand after the operation.
2. **Implementation:** Deeptools must map, lower, place, and synchronize the
   corresponding one-to-many ring transfers.

The first claim is supported by the fixtures. The second requires the bounded,
replication-aware materialization logic demonstrated by the exact-backend
experiment.

## Proof Obligations

| Layer | Experiment | Observed result | What it proves | Artifact |
|---|---|---|---|---|
| Logical representation | Variant A assigns one distinct K shard per source core and redundantly assigns the full-K result to all eight destination cores in each head group | Fixture validation derives 256 required shard placements | Coordinates contain enough information to compute the desired logical source/destination relation | `generated_v2/fixture_validation.json`, `generated_v2/expected_transfer_summary.json` |
| Alternate representation | Variant B adds an explicit nonphysical replication dimension with input cardinality 1 and output cardinality 8 | Clean backend rejects the SHUFFLE mapping | Making replication explicit does not activate a supported all-gather mapping | `variant_b_meta/fixtures/variant_b_explicit_meta/sdsc_0.json` |
| Clean backend mapping | Variant A is replayed on Deeptools `704c19f8fb` | `Scheduler failed to find a suitable op mapping for sdsc: 0_shuffle` | Current DDL/scheduler mapping does not accept the expanding replicated SHUFFLE | `exact_backend/raw/v6-honest-a-shuffle-only/stderr.log`, `exact_backend/audit.json` |
| Physical materialization | A bounded backend recognizer emits transfer rows from the coordinate mismatch | DCG reaches `checkSubPieceCoverage` with uncovered output coordinates | Output-piece generation and coverage still assume unsupported nonreplicated semantics | `generated/VARIANT_A_REPLAY_REPORT.md`, `generated/variant_a_replay_summary.json` |
| Exact patched backend | Variant A is replayed after adding bounded replicated-SHUFFLE lowering with independent endpoint layouts | DXP/DCG/DCC pass; 8 rows, 256 placements, 224 remote, 32 local, no HBM/dynamic allocation/restickify | Coordinates and explicit endpoints are structurally sufficient; the gap is the stock backend materializer | `variant_a_exact_backend/README.md`, `variant_a_exact_backend/verification.md`, `variant_a_exact_backend/deeptools.patch` |
| Ring carrier | The required placements are enumerated directly as bounded `STCDPOpLx` rows | 256 placements lower structurally: 224 remote and 32 local | Existing LX ring primitives can carry the traffic | `bounded_direct_datadsc/`, `generated_v2/expected_transfer_summary.json` |
| End-to-end operation | The custom staged materializer emits the grouped gather and local layout conversion | Value-correct; K-side HBM handoff is removed | The grouped K all-gather is feasible on the device | `custom_materializer_control/CUSTOM_MATERIALIZER_PHYSICAL_LOWERING.md`, `allocation_evidence/frac0p07_value_correct/report.txt` |
| Destination capacity | Default allocation is replayed with the 1 MiB/core expanded S2 live at the score-BMM boundary | Default 1638 KiB frontend limit cannot fit the full live set | S2 must be a tracked frontend allocation with capacity-aware HBM fallback | `LX_ALLOCATION_LIFETIME.md`, `allocation_evidence/default_0p2_scratchpad_excerpt.txt` |
| Physical layout | S1 uses a compact 512-wide fold; S2 embeds shards in a complete 4096-wide K tensor | Direct source/destination row strides differ | Backend needs independent endpoint layouts or an explicit local `ReStickifyOpLx` stage | `manifest.md`, `generated_v2/expected_physical_address_samples.json` |

## Experiments Tried

### X: Redundant-coordinate SHUFFLE

```text
S1: 8 cores x one distinct K shard
SHUFFLE output coordinates: same complete-head coordinate on all 8 cores
S2: 8 cores x complete K operand
```

No custom collective classification is present. The clean backend rejects the
operation before physical transfer generation.

### Y: Explicit replication dimension

```text
input replication coordinate cardinality  = 1
output replication coordinate cardinality = 8
```

This is also rejected. The failure is therefore not specific to redundant
coordinate spelling.

### Z: Bounded materialization patch

The backend recognizes the expansion and emits bounded movement rows. This
passes the initial mapping point but fails DCG output coverage, showing that
mapping support alone is insufficient.

### Z2: Exact bounded materialization

The corrected narrow materializer uses the nested DLDSC shape, retains compact
S1 and expanded S2 as independent physical layouts, emits one bounded row per
source shard, and treats the SHUFFLE bundle slot as a data-only program. It
passes DXP/DCG/DCC and the physical verifier for all 256 placements.

### Control: Explicit bounded transfers

Manually enumerated `STCDPOpLx` rows lower the exact logical transfer set. This
isolates the limitation to automatic SHUFFLE materialization rather than ring
transport capability.

## Required Backend Behavior

To cover this use case through the preferred SHUFFLE contract, production
Deeptools needs the behavior demonstrated by the isolated patch:

1. **Replication-aware mapping** that accepts one source shard contributing to
   multiple destination residencies.
2. **Expansion-aware piece construction** that derives all 256 placements and
   treats repeated logical coordinates on distinct cores as valid residency.
3. **Explicit destination ownership** that consumes the frontend-provided S2
   address and extent rather than allocating hidden dynamic LX.
4. **Endpoint layout handling** that either supports independent S1/S2 strides
   or inserts a defined local restickify stage.
5. **Coverage and synchronization** that proves every destination shard is
   written exactly once before the score BMM begins.
6. **Capacity-safe fallback** that preserves the HBM route when the full live
   set cannot fit without overlap.

The structural experiment closes the diagnostic-fixture portions of items 1
through 5, including independent endpoint strides and bundle ordering.
Frontend-tracked S2 lifetime in the real graph, full-workload capacity safety,
patterned values, and performance remain integration gates rather than
representation questions.

## What Not To Claim

- Do not claim that coordinates are mathematically unable to express an
  all-gather.
- Do not claim that current coordinates automatically make the all-gather work.
- Do not infer success from DXP import alone; post-DCG movement rows, physical
  addresses, coverage, ordering, and patterned values are required gates.
