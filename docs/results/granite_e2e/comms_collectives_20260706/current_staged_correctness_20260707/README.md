# Current staged matmul-operand all-gather check

Date: 2026-07-07

## Purpose

This is a bounded correctness probe for the DLDSC LX communication substrate.
It intentionally does **not** use full flash-attention numeric correctness as
the oracle, because the baseline flash test currently has an independent
zero-stride/broadcast-view correctness issue. Flash remains useful as a
structural SDSC/lowering stress case only.

The probe uses the row-pattern matmul-operand broadcast test:

```text
mul producer output in LX, split over output chunks
  -> batchmatmul RHS operand, consumer split over M
```

That edge is classified as:

```text
kind = matmul_operand_broadcast
communication_pattern = all_gather_replicate
realization_strategy = gather_then_restickify
```

## Environment

Pod:

```text
adnan-cdx-spyre-dev-pf
```

Torch branch and SHA:

```text
gather-restickify
bced14b49acf4fae92ef4df07d2f5229806c672b
```

Deeptools branch and SHA:

```text
ah/comms-collectives
9cd9c79c3961224920b8d55710bc15501e9fa3f3
```

The Torch extension was rebuilt in place before this run because the checked-out
`torch_spyre/_C.so` was stale against the pod runtime and failed import with:

```text
undefined symbol: flex::RuntimeEntry::toPriority
```

## Result

The current branch emits the expected backend plan, but DXP/DDC rejects the
staged materialization before hardware execution:

```text
RC=1
PLAN_COUNT=1
DtException: Coordinates of transfer transfer_lds1_src:lxlu_dst:ptrow0
and allocateNode allocate-Tensor1_lx are not consistent.
```

The emitted plan is present at:

```text
backend_plans/1_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json
```

Key fields:

```text
kind = matmul_operand_broadcast
communication_pattern = all_gather_replicate
materialization_pattern = all_gather_replicate_with_layout_conversion
realization_strategy = gather_then_restickify
physical_lowering_status = lowered_gather_then_restickify
logical_transfer_count = 16
```

## Interpretation

This is a backend physicalization gap, not a flash correctness issue and not a
Torch metadata miss.

Torch emits the logical communication contract. Deeptools recognizes it and
constructs the staged gather/restickify plan. The failure happens when DDC checks
the inserted transfer coordinates against the destination LX allocation. The
transfer has loop-scoped/temporal folds that do not match the allocation
coordinate folds for `allocate-Tensor1_lx`.

The archived older row-pattern evidence remains useful:

- staged `gather_then_restickify` was value-correct through M64 in earlier
  runs;
- direct KERNEL-neighbor movement was value-wrong and should stay diagnostic;
- full flash correctness is not a valid oracle until the baseline broadcast-view
  bug is fixed separately.

The next clean step is to make staged all-gather/restickify produce coordinate
metadata that DDC accepts for a bounded resident tile, or fail closed before
claiming physical lowering. Do not build full-tensor streaming here; that belongs
to WSR/tile-scoped lowering.

## Files

- `run.log`: full compile-time failure and DDC coordinate dump.
- `env.txt`: exact flags used.
- `run_info.txt`: branch SHAs and script path.
- `backend_plans/`: emitted backend plan JSON.
