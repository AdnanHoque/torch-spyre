# PR2 SDPA all-gather relayout check - 2026-07-09

## What was tested

Jamie suggested the simpler SDPA microbench under `fms_granite_micro`:

```bash
python run_benchmark.py --op fms_granite_micro.mha_4h_workdiv_h4_lq8
```

This note records the 4-head explicit small shape:

```text
q = [1, 4, 512, 128]
k = [1, 4, 512, 128]
v = [1, 4, 512, 128]
```

The larger registered shape in perf-suite uses `k/v = [1, 4, 4096, 128]`; that run did not reach a useful structural comparison in this environment.

## Result

The PR2 all-gather/stick-relayout path fires structurally on the small SDPA case.

The relevant attention expression is:

```text
S = Q @ K.T
```

The targeted edge is the transposed/scaled `K` operand feeding the `QK.T` score matmul. Without PR2, Torch emits an HBM restickify before that matmul. With PR2, Torch emits an LX restickify and annotates the consumer matmul with an `all_gather` + `stick_relayout` contract.

| Variant | Relayout row | Consumer row | Consumer K input | Communication metadata |
|---|---|---|---|---|
| relayout off | `sdsc_2 ReStickifyOpHBM` | `sdsc_3 batchmatmul` | `1_hbm` | none |
| relayout on | `sdsc_2 ReStickifyOpLx` | `sdsc_3 batchmatmul` | `1_lx` | `all_gather`, `stick_relayout`, `gather_then_restickify` |

Concrete SDSC evidence:

| Variant | SDSC | Op | Tensor | Role | Location | Layout | coreIdToWkSlice |
|---|---|---|---|---|---|---|---|
| off | `sdsc_2` | `ReStickifyOpHBM` | `1_hbm` | output | HBM | `512*/8, 128, 4/4` | `{mb=0:3} {x=0:7} out=0` |
| off | `sdsc_3` | `batchmatmul` | `1_hbm` | input K.T | HBM | `512*, 128, 4/4` | `{x=0:3} {mb=0:7} out=0 in=0` |
| on | `sdsc_2` | `ReStickifyOpLx` | `1_lx` | output | LX | `512*/8, 128, 4/4` | `{mb=0:3} {x=0:7} out=0` |
| on | `sdsc_3` | `batchmatmul` | `1_lx` | input K.T | LX | `512*, 128, 4/4` | `{out=0:7} in=0 {x=0:3}` |

## Backend replay

The generated relayout-on SuperDSC was replayed through the matching broad collectives Deeptools build:

```text
/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/deeptools/build-deeptools/dxp/dxp_standalone
```

Manual DXP replay returned `rc=0` and generated `spyreCodeDir`.

System Deeptools in this pod does not yet contain this PR2 lowering; it fails on `ReStickifyOpLx` mapping for the relayout-on bundle.

## Timing status

No valid timing number was produced from this run.

Both relayout-off and relayout-on variants compile far enough to launch, then fail in the same runtime path:

```text
RuntimeError: convert_address not yet implemented - waiting for flex support
```

That failure is not specific to the PR2 relayout: it happens in both variants. The result here should therefore be treated as structural SDSC/DXP evidence, not a performance measurement.

I also tried adapting the older flash `no_h2d` compile-probe trick to `spyre-perf-suite` by intercepting `Tensor.to(device="spyre")` and creating empty Spyre tensors directly. That still hit the same `convert_address` runtime path in the perf-suite runner, so it did not produce a timing result.

## Artifacts

- `sdpa_4h_l512_relayout_off_summarize_sdsc.md`: Jamie-style SDSC summary for relayout off.
- `sdpa_4h_l512_relayout_on_summarize_sdsc.md`: Jamie-style SDSC summary for relayout on.
- `selected_sdsc/off/sdsc_2_ReStickifyOpHBM.json`: baseline HBM relayout row.
- `selected_sdsc/off/sdsc_3_QK_batchmatmul.json`: baseline QK.T consumer row.
- `selected_sdsc/on/sdsc_2_ReStickifyOpLx.json`: PR2 LX relayout row.
- `selected_sdsc/on/sdsc_3_QK_batchmatmul.json`: PR2 QK.T consumer row with all-gather metadata.
- `relayout_off_run.log`, `relayout_on_run.log`: benchmark command output and runtime failure tails.
- `selected_sdsc/dxp_replay.log`: manual DXP replay log for the relayout-on bundle.
