# Flash Attention DLDSC Collectives Check (2026-07-02)

Pod: `adnan-clc-spyre-dev-pf`
Workspace: `/home/adnan/codex-isolated/dldsc_flash_attention_20260702_112409`
Test script: `/home/adnan/codex-isolated/dldsc_flash_attention_20260702_112409/repos/test-spyre-scripts/test_flash.py`

## Source SHAs

- Torch main: `e1abccf7033e1a8e399b971b8cb6e25e66d47708`
- Torch collectives `ah/comms-collectives`: `b943eed4b6403266152a5cad8555b15d164a7fbf`
- Deeptools master: `0a9da5eb19d08712383312bb7dec18fbd7caf711`
- Deeptools collectives `ah/comms-collectives`: `15d39ece49e56d60e7487553116466630bdc81eb`
- Deeptools build source for `dxp_standalone`: `15d39ece49e56d60e7487553116466630bdc81eb`
- test-spyre-scripts: `afda166e58b23519d0b4ca871350b011b56d91a3`

## Runs

Latest Torch main baseline was attempted from `/home/adnan/codex-isolated/dldsc_flash_attention_20260702_112409/runs/baseline_main_noh2d_autoload_20260702_113440`. It failed before SDSC generation with rc=1 and 0 SDSCs:

```text
torch._inductor.exc.InductorError: NotImplementedError: buf10 (Pointwise): no mechanism to resolve stick incompatibility
```

Because no fresh latest-main SDSCs were emitted, the before summary below uses the prior working baseline run `/home/adnan/codex-isolated/flash-sdsc-20260701-033044/runs/baseline_noh2d_20260701_040758`.

Before baseline summary:

- rc: 0
- SDSCs: 550
- explicit HBM restickify rows: 32 `ReStickifyOpHBM`
- HBM SDSCs: 132
- JSON alloc counts from prior report: hbm 781, lx 704, pool 0
- SDSCArg counts from prior report: hbm 196, lx 704, pool 585

Optimized collectives run: `/home/adnan/codex-isolated/dldsc_flash_attention_20260702_112409/runs/optimized_collectives_noh2d_pyldfix_latest_20260702_115435`

- rc: 1
- wall seconds: 104
- SDSCs emitted before backend abort: 549
- restickify rows: {'ReStickifyOpLx': 32}
- HBM SDSCs by JSON allocate nodes: 421
- allocation nodes: {'hbm': 714, 'lx': 768}
- first backend failure: `Scheduler failed to find a suitable op mapping for sdsc: 2_ReStickifyOpLx`

## Remaining HBM Spill Classification

PR1 scatter support did not fire for the flash attention spill.

| Phase | Rows | SDSC op | Communication class | Pattern | Status |
| --- | ---: | --- | --- | --- | --- |
| Before | 32 | `ReStickifyOpHBM` | `all_gather` inferred from after metadata on same edge | `layout_allgather_restickify` | HBM restickify spill |
| After | 32 | `ReStickifyOpLx` | `all_gather` | `layout_allgather_restickify` | LX metadata emitted; DXP schedule mapping missing |

The collectives-relevant edge is form-changing restickify from `mul` into `batchmatmul`, not PR1 scatter and not the Granite AV `matmul_operand_broadcast` / matmul operand all-gather route. After collectives, the explicit `ReStickifyOpHBM` rows are gone. HBM allocations still appear in normal op SDSCs and batchmatmul storage, but no scatter or matmul-operand broadcast/all-gather classification appears in this flash run.

## Commands

The exact command lines are saved in the run directories and copied under `logs/` here:

- latest main baseline: `/home/adnan/codex-isolated/dldsc_flash_attention_20260702_112409/runs/baseline_main_noh2d_autoload_20260702_113440/command.txt`
- optimized collectives: `/home/adnan/codex-isolated/dldsc_flash_attention_20260702_112409/runs/optimized_collectives_noh2d_pyldfix_latest_20260702_115435/command.txt`

The optimized command used a wrapper for `dxp_standalone` so Python loaded installed Torch/Spyre libraries while DXP loaded the collectives Deeptools build from `/home/adnan/codex-isolated/dldsc_collectives_validate_20260702_112302/build-deeptools`.
