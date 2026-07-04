# Flash Attention Grouped All-Gather Checkpoint - 2026-07-04

## Result

The CDX compile/runtime smoke now passes for `test_flash.py` with the DLDSC grouped all-gather path enabled.

- Run dir: `/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/runs/flash_after_collectives_backend1_20260704_085439`
- Return code: `0`
- SUCCESS marker: `true`
- SDSC files: `550`
- Backend `matmul_operand_broadcast` plans: `32`
- Backend DXP LX fraction: `DXP_BACKEND_LX_FRAC_AVAIL=1`
- Deeptools diagnostic branch: `Adnan-Hoque1/deeptools ah/comms-collectives` at `ba981112a48c72a9e5f14720bf5fe6d537b3caa1`
- Probe mode: `no_h2d,skip_cpu_ref`; this proves compile/runtime smoke, not numeric correctness.

## What Changed In The Experiment

The important backend experiment was to stop copying producer coordinate folds into the synthetic destination LX allocation for `matmul_operand_broadcast`. For grouped all-gather, the destination allocation is consumer-sized and consumer-shaped; inheriting the producer fold makes the materialized operand too small and DDC eventually fails while distributing the matmul transfer loops.

The working diagnostic shape is:

1. Torch emits the DLDSC relayout classification and consumer/producer core maps.
2. Deeptools recognizes `matmul_operand_broadcast` / `all_gather_replicate`.
3. Deeptools seeds destination ownership, but does not force producer folds onto the destination allocation.
4. DDC derives the consumer-sized LX allocation.
5. The backend emits 32 grouped all-gather plans and the flash compile/runtime probe reaches `SUCCESS`.

## Sample Backend Plan

```json
{
  "artifact_kind": "matmul_operand_broadcast_backend_plan",
  "communication_pattern": "all_gather_replicate",
  "estimated_tensor_bytes": "0",
  "fallback_policy": "refuse resident HBM/full per-consumer operand materialization",
  "group_count": "4",
  "kind": "matmul_operand_broadcast",
  "logical_transfer_count": "256",
  "physical_lowering_status": "lowered_loop_scoped_kernel_neighbor",
  "producer_chunks_per_group": "8",
  "realization_strategy": "loop_scoped_input_fetch",
  "realized": true,
  "replication_factor": "8",
  "stages": [
    "source_operand_shards",
    "grouped_all_gather_replicate",
    "loop_scoped_input_fetch",
    "bind_matmul_kernel_operand"
  ]
}
```

## Evidence

- `summary.json` in this directory records the run metadata.
- `backend_plan_files.txt` lists the 32 generated backend plans.
- `stdout_tail.txt` contains the runtime `SUCCESS` marker and skipped-assert marker.
- `stderr_tail.txt` contains the three kernel runner launches.
- `deeptools_dest_map_backend1_experiment.diff` records the exact diagnostic Deeptools delta.

## Caveats

This is not production-ready yet. The Deeptools patch still contains diagnostic logging, a guard around a missing fold-to-loop binding, and no clean generated ring-node artifact. The result is still valuable because it shows the contract direction: destination ownership plus backend-derived consumer allocation works better than copying producer coordinates.
