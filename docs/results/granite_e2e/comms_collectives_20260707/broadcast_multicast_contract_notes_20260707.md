# Broadcast/Multicast Contract Notes - 2026-07-07

## Why The Minimal Enablement Did Not Land

We tried the smallest Deeptools change:

- accept `broadcast` and `multicast` in the existing staged
  `STCDPOpLx + ReStickifyOpLx` matmul-operand carrier;
- convert the existing broadcast/multicast fail-closed unit tests into bounded
  compile tests.

That was the right experiment, but the current test fixture is not a valid
physical broadcast/multicast proof.

## First-Principles Contract

A communication label is not enough.

For a copy-only LX relayout, Deeptools needs two pieces of information:

1. Logical communication pattern:
   - scatter/permutation: each destination gets one distinct source shard;
   - broadcast: one source shard is copied to all destination cores;
   - multicast: one source shard per group is copied to that group's destination
     cores;
   - all-gather/replicate: every destination gets every source shard.
2. Physical tensor residency:
   - which source coordinates actually exist on each source core;
   - which destination tensor coordinates each consumer core expects;
   - source and destination LX addresses for those concrete cells.

The failed bounded test changed only the communication pattern. It reused an
all-gather fixture where the source tensor was physically sharded across 32
cores. After rewriting the pattern to `broadcast`, the plan asked source core 0
to feed several destination coordinates that were not resident on source core 0.

That is not a legal broadcast. It is an inconsistent contract.

## What Failed

The experimental diff is archived in:

`replay_payloads/artifact_payload_20260707_overnight/broadcast_multicast_bounded_experiment_20260707.tgz`

Observed failures:

- broadcast bounded test: initially failed because the diagnostic chunk cap was
  too small for the rewritten fixture;
- multicast bounded test: failed later with `Invalid start address or buffer
  offset`, because the rewritten fixture could imply reads beyond the actual
  source shard allocation.

This is a useful failure. It says the utility-level logical plan is not enough;
the DXP physical test must also make the source tensor genuinely resident for
the fanout coordinates being requested.

## Required Next Fixture

A valid bounded broadcast fixture should have:

- `communication_pattern = broadcast`;
- one source core/group;
- multiple destination cores in the same destination group;
- a `source_lx_tensor` whose `coreIdToWkSlice_` and `numWkSlicesPerDim_`
  describe a source piece that truly covers the destination tensor coordinates;
- a `target_kernel_tensor` whose tensor distribution is redundant across the
  destination cores while compute coordinates may still differ.

A valid bounded multicast fixture should have:

- `communication_pattern = multicast`;
- multiple source groups;
- each source group faning out to a subset of destination cores;
- each source group's source tensor cell physically covering the tensor
  coordinates requested by that destination subset.

## Backend Guard To Add

Before enabling broadcast/multicast through the staged carrier, Deeptools should
reject inconsistent contracts before DCC lowering.

The likely guard:

- when deriving a source shard cell, use the tensor contract's declared
  `numWkSlicesPerDim_` when present, not only the filtered source core map;
- verify that every required fanout destination has a concrete overlapping
  source cell;
- fail closed with an explicit message if a communication pattern asks a source
  core to provide coordinates it does not own.

That keeps the communication substrate honest and prevents a misleading
green test that only changes labels.

## Current Status

- scatter/permutation: supported by PR1 path;
- partial gather: bounded tests pass;
- all-gather/replicate: bounded tests pass and full saved flash DXP replay
  passes after the default chunk-policy fix;
- broadcast/multicast: logical classification and utility expansion exist, but
  production lowering remains fail-closed until a valid physical fixture and
  contract guard are added;
- reduce/all-reduce: future arithmetic communication primitive, not part of the
  copy-only relayout carrier.
