# Stage 136: Descriptor-Driven Split Runtime Check

## Summary

Stage 136 tested whether the schema v3 LX endpoint contract could carry the
prototype from standalone data-op generation into the split runtime path:

```text
producer compute -> descriptor-driven LX data-op restickify -> consumer compute
```

The result is mixed:

- The split runtime path did consume the schema v3 endpoint contract.
- The generated data-op program had `HBM=0` in `senprog.txt`.
- The producer, data-op, and consumer stages all returned through the Python
  launcher.
- The device still raised a compute-control hardware error before the following
  matmul could launch.

So the blocker moved forward again: endpoint selection is no longer the issue.
The remaining issue is runtime-safe packaging/execution of the split program.

## Command

```sh
SPYRE_RESTICKIFY_RING_TELEMETRY=1 \
SPYRE_ALIGN_RESTICKIFY_CORE_MAPPING=1 \
SPYRE_ALIGN_RESTICKIFY_WORK_DISTRIBUTION=1 \
SPYRE_RESTICKIFY_LOCALITY_ASSERT=1 \
SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR=1 \
SPYRE_RESTICKIFY_LX_SPLIT_SYNC_EACH=1 \
python tools/restickify_scenario_probe.py \
  --case computed_transpose_adds_then_matmul_tuple \
  --size 2048 \
  --lx-split-dataop-prototype \
  --copy-kernel-code \
  --output-dir /tmp/stage136-split-dataop-descriptor-2048 \
  --fail-on-error
```

I also reran with:

```sh
DXP_LX_FRAC_AVAIL=1
```

and with explicit matching bases:

```sh
SPYRE_RESTICKIFY_LX_SPLIT_PRODUCER_BASE=16384
SPYRE_RESTICKIFY_LX_SPLIT_CONSUMER_BASE=8192
```

Neither changed the runtime failure.

## Evidence

The split path's data-op generation summary reported:

```json
{
  "endpoint": {
    "source": "schema-v3-lx-endpoint-contract",
    "schema_version": 3,
    "edge_id": "0:1:2",
    "memory_space": "lx"
  },
  "patched": {
    "producer": 32,
    "consumer": 32
  }
}
```

The exported data-op `senprog.txt` contained no HBM terms:

| Token | Count |
|---|---:|
| `HBM` | `0` |
| `L3LU` | `96` |
| `L3SU` | `96` |
| `LXLU` | `64` |
| `LXSU` | `64` |
| `SFP` | `0` |
| `PT` | `0` |

The kernel launch log showed the split stages completed:

```text
lx_split_dataop_before_producer
lx_split_dataop_after_producer
lx_split_dataop_before_dataop
lx_split_dataop_after_dataop
lx_split_dataop_before_consumer
lx_split_dataop_after_consumer
lx_split_dataop_launch_done
```

Then the next matmul launch failed because the stream had already entered a
hardware-error state:

```text
Compute CB hardware error detected
Cannot schedule operation on stream in error state
```

## Smaller Sizes

I also tried `512` and `1024` with the same split path. Those failed earlier,
during Deeprt data-op export:

```text
DtException: myOutPiece.second.dimToSize_.at(dimName) >=
op->outLds->dimToStickSize_.at(dimName)
restickifyOp.cpp line 333
```

That is a separate data-op shape-contract failure. At `2048`, Deeprt produces an
`init.txt` despite `dataop_export_returncode = -11`, and then runtime execution
is unsafe. Both point at the same family of issue: the data-op export/runtime
contract is still too brittle for integrated use.

## Interpretation

This stage is important because it narrows the problem:

1. Stage 134/135 gave us a real endpoint contract.
2. Stage 136 confirmed the split runtime path can consume that contract and
   produce an HBM-free data-op program.
3. The remaining failure is not "which tensor edge should this restickify use?"
   It is "how do we package and execute this LX data-op safely inside the
   Torch-Spyre runtime sequence?"

## Artifacts

Local copies:

- `artifacts/stage136_split_dataop_descriptor_runtime/restickify_scenarios_2048.jsonl`
- `artifacts/stage136_split_dataop_descriptor_runtime/kernel_launches_2048.jsonl`
- `artifacts/stage136_split_dataop_descriptor_runtime/dataop_gen_summary_2048.json`
- `artifacts/stage136_split_dataop_descriptor_runtime/senprog_2048.txt`
- `artifacts/stage136_split_dataop_descriptor_runtime/restickify_scenarios_2048_forcedbases.jsonl`

## Next Step

Do not keep tuning core ordering until this runtime issue is solved. The next
step should be a minimal three-stage runtime fixture that removes TorchDynamo and
the following matmul from the equation:

```text
producer add -> descriptor-driven data-op -> consumer add
```

Success should mean:

- Deeprt export returns `0`, not `-11`.
- The exported `senprog.txt` has `HBM=0`.
- `launch_kernel(producer)`, `launch_kernel(dataop)`, and
  `launch_kernel(consumer)` all return and synchronize without a hardware error.
- The final tensor is value-correct.
