# Granite S512 Gather-Restickify Clean Branch Smoke, 2026-07-06

This records a CDX full Granite block smoke using the clean split branches:

- Torch: `AdnanHoque/torch-spyre:gather-restickify`
- Deeptools: `Adnan-Hoque1/deeptools:gather-restickify`
- Pod: `adnan-cdx-spyre-dev-pf`
- Clean root: `/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236`

CLC was intentionally not used because Claude owned that pod during this run.

## Workload

- Probe: `spyre-granite-e2e-bench/benchmarks/granite_block_layer_probe.py`
- Case: prefill
- Shape: `B=1, S=512, hidden=4096`
- Attention: `sdpa_causal`
- Weights: empty Spyre-resident parameters
- Iterations: `1`
- Profiling: off

## Result

- Return code: `255`
- Result JSON: not produced
- Backend plans emitted: `2`
- Runtime state at termination: stuck after `[block-probe] calling block iteration 1/1`

The run was terminated manually after it remained in runtime for several
minutes. The backtrace in `stderr.log` shows the process was inside
`RuntimeScheduler::issueBarrier` during H2D scheduling when it received SIGTERM.

## What Worked

The DLDSC/backend relayout path did make it through DXP and emitted both
expected Granite attention operand plans:

| SDSC | kind | communication | strategy | lowering | logical transfers |
|---|---|---|---|---|---:|
| `8_batchmatmul` | `matmul_operand_broadcast` | `all_gather_replicate` | `gather_then_restickify` | `lowered_gather_then_restickify` | 512 |
| `16_batchmatmul` | `matmul_operand_broadcast` | `all_gather_replicate` | `gather_then_restickify` | `lowered_gather_then_restickify` | 1024 |

This confirms the clean branches include the rank-grouped/chunked
gather-restickify lowering that previously only appeared in experimental replay
artifacts.

## What Did Not Work

The full AIU runtime did not complete. This is now a runtime completion/H2D
barrier issue after backend lowering, not a DXP metadata or plan-synthesis
failure.

After the run was terminated, the reset command:

```bash
/opt/ibm/spyre/senlib/bin/aiu_dd2_hot_reset -t chip -d b0:00.0
```

found the correct CDX PCI device (`0000:b0:00.0`, `/dev/vfio/80`) but aborted
with:

```text
RISCV config not found.
```

## Archived Files

- `env.sh`: pinned environment.
- `command.txt`: exact probe command.
- `returncode.txt`: process return code.
- `stdout.log`: probe stdout.
- `stderr.log`: SIGTERM/runtime backtrace.
- `run_summary.json`: raw run summary.
- `summary_compact.json`: compact parsed summary.
- `backend_plans/*.json`: emitted Deeptools backend plans.

## Next Step

The next useful experiment is a smaller synthetic AIU value/runtime harness for
the same `matmul_operand_broadcast -> all_gather_replicate ->
gather_then_restickify` plan. The full Granite block proves the DXP/backend
plans are emitted, but it is too large to debug runtime completion directly.
