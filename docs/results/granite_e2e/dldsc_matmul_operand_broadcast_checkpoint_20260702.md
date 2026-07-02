# DLDSC Matmul Operand Broadcast Checkpoint - 2026-07-02

## Summary

DLDSC is still the forward path for LX relayout/collective work. The older
`ah/comms-collectives-dldsc-agent` branch was audited and should be treated as
historical: its only unique code commit was a guarded activation `ReStickifyOpLx`
prototype, and the current `ah/comms-collectives` branch has already superseded
that with the `ReStickifyOpLx` op plus the flash `layout_allgather_restickify`
path.

This checkpoint adds and validates the next Granite communication contract:
matmul RHS/value operand broadcast/all-gather replication. This is the class
behind the Granite attention value operand spill, often referred to in the
reduced artifacts as the `buf21` / Tensor1 edge.

## Branches

Torch artifact branch:

- repo: `AdnanHoque/torch-spyre`
- branch: `ah/comms-collectives`
- relevant commit: `4e9a25544c712da435f3d3f95553b8cf2100eb37`
- commit message: `Add DLDSC contract for matmul operand all-gather`

Deeptools artifact branch:

- repo: `Adnan-Hoque1/deeptools`
- branch: `ah/comms-collectives`
- relevant commit: `6ef1771a6115d6e8bd38d1a0ddb06723fefb22a5`
- commit message: `Support matmul operand broadcast planning`

## What This Communication Class Is

This is not the PR1 scatter class.

The Granite attention value operand is produced with tensor ownership sharded
across an operand dimension, then consumed by a `batchmatmul` whose compute work
is sharded across `mb`. Because `mb` is not a dimension of the RHS/value operand,
a naive resident relayout interprets the post-relayout form as a full operand per
consumer core. That is exactly the full-materialization behavior we need to
avoid.

The new contract describes this as:

- `kind = matmul_operand_broadcast`
- `communication_class = all_gather`
- `communication_pattern = all_gather_replicate`
- `staging_scope = matmul_transfer_loop`

Meaning: keep the producer-owned operand shards in LX, and let Deeptools derive a
loop-scoped movement plan so the consumer matmul sees the operand pieces it needs
without allocating one complete replicated operand per core.

## Torch-Side Change

Torch now emits a compact DLDSC contract for non-primary matmul operand
collectives when enabled. The important payload fields are:

- producer op and consumer op
- operand read index (`1` for RHS/value operand)
- producer work-slice dimensions
- consumer tensor work-slice dimensions
- consumer compute work-slice dimensions
- operand kernel layout
- communication class and pattern
- staged-realization hint

This keeps the physical transfer schedule out of Torch. Torch defines the
logical handoff and the communication class; Deeptools is still responsible for
physical movement synthesis.

Focused Torch validation was limited by a pod ABI issue importing
`torch_spyre._C`, but lightweight checks passed in the worker:

- `git diff --check`
- `python3 -m py_compile`
- `ruff check`
- direct contract smoke

## Deeptools-Side Change

Deeptools now parses and validates the `matmul_operand_broadcast` /
`all_gather_replicate` contract and synthesizes a deterministic backend movement
plan.

For the reduced `buf21`-shaped case, the backend plan is:

- producer shards: 32
- consumer replicas: 32
- logical transfers: 1024
- stages:
  - `source_operand_shards`
  - `grouped_all_gather_replicate`
  - `loop_scoped_input_fetch`
  - `bind_matmul_kernel_operand`

Verification on `adnan-cdx-spyre-dev-pf`:

```bash
cmake --build build-focused --target util_unit_test -j2
./build-focused/util/util_unit_test --gtest_filter=LayoutAllgatherRestickify.*
```

Result: 19 tests passed.

A DXP smoke with injected buf21 metadata emitted a
`matmul_operand_broadcast_backend_plan` artifact with
`logical_transfer_count=1024`, then failed closed before falling through to the
bad resident full-materialization path.

## Current Boundary

The current backend branch intentionally fails closed for this class. It does so
because the remaining physical lowering boundary is not metadata recognition; it
is schedule fusion:

> loop-scoped IFN/STCDP metadata exists, but DCC cannot yet fuse IFN movement and
> DL matmul PCFG for one consumer schedule step.

That boundary is important. It means the next step is not another Torch-side
scatter relayout and not full replicated LX allocation. The next backend task is
to connect the contract to loop-scoped input-neighbor-fetch / STCDP movement so
the matmul transfer loop fetches operand shards as it runs.

## Why The Older DLDSC-Agent Branch Is Not The Base

The branch `ah/comms-collectives-dldsc-agent` has one unique implementation
commit, `75040ee6`, which adds a guarded activation `ReStickifyOpLx` prototype.
Current `ah/comms-collectives` already has the generalized version of that idea:

- `ReStickifyOpLx` is a first-class op name.
- flash layout all-gather restickify emits `ReStickifyOpLx` rows.
- a no-H2D flash runtime probe reached hardware and emitted 32 `ReStickifyOpLx`
  rows and zero `ReStickifyOpHBM` rows for that class.

A direct branch diff from the older agent branch would remove newer flash and
runtime artifacts, so it should only be mined for historical notes.

## Next Engineering Step

Implement physical lowering for `matmul_operand_broadcast`:

1. Use the DLDSC contract and existing allocation/consumer coordinates to build
   staged source/destination operand movements.
2. Route this to the existing input-neighbor-fetch/STCDP machinery rather than
   the generic resident relayout allocator.
3. Fuse or sequence the generated IFN movement with the consumer DL matmul PCFG
   so DCC accepts one consumer schedule step.
4. Validate on the reduced `buf21` artifact first, then full Granite prefill.

Until this is done, this communication class is identified and protected from
bad fallback, but it is not yet an e2e HBM-spill removal.
