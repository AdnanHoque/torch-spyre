# Comms Collectives Round 2 - 2026-07-01

This checkpoint continues the Granite communication-class work across three
AIU pods:

- `adnan-spyre-dev-pf`: current `test_flash.py` baseline vs scatter-planner
  SDSC behavior.
- `adnan-cdx-spyre-dev-pf`: compact DLDSC/STCDP local partial-stick staging
  feasibility.
- `adnan-clc-spyre-dev-pf`: explicit byte-range/remap scaling and semantic
  modeled validation.

## Flash Attention Script

Workspace:

```text
/home/adnan/codex-isolated/flash_main_probe_20260701_015234
```

Script:

```text
git@github.ibm.com:aviros/test-spyre-scripts.git
main: 9cd82f0f40d019e2497b046ec00c6ca06f3d1e2f
test_flash.py blob: 2d5894a0871973da8179c93c35bacb50c36e49a8
```

The unmodified current script does not produce a runtime/perf result in this
environment:

- Real `.to(device="spyre")` hangs before compile during
  `RuntimeStream::synchronize()`.
- With host-to-device transfer skipped for compile probing, the script still
  needs singleton-stick reduction restickify to get past
  `running_max = torch.maximum(real_max, block_max)`.
- With a runtime monkey patch for singleton-stick restickify, both lanes reach
  scratchpad/SDSC and then fail in DXP:

```text
DtException: Could not find any suitable dimension mapping
```

SDSC-reaching result table:

| lane | rc | SDSCs | ReStickifyOpHBM | STCDPOpLx | alloc_pool | alloc_hbm | alloc_lx | relayout records |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline + singleton patch | 1 | 89 | 20 | 0 | 159 | 27 | 40 | 0 |
| scatter + singleton patch | 1 | 89 | 20 | 0 | 125 | 27 | 74 | 14 |

Interpretation:

- Scatter planning changes `test_flash.py` compile/SDSC behavior.
- `alloc_pool` drops from `159` to `125`; `alloc_lx` rises from `40` to `74`.
- Literal `allocation=[hbm=...]` count remains `27`.
- `ReStickifyOpHBM` count remains `20`; this means the pass is improving
  residency but has not removed all HBM restickify rows.
- No runtime/perf conclusion is valid yet because DXP aborts after SDSC
  generation.

Scatter lane relayout summary:

```text
realized:
  scatter: 7
  layout_restickify_activation: 1

unsupported:
  scatter: 4 due to backend relayout reservation did not fit in scratchpad
  layout_restickify_activation: 1 due to scratchpad reservation
  layout_restickify_activation: 1 due to loop-scoped matmul operand lowering
```

Local artifacts:

```text
flash_main/summary.txt
flash_main/summary.json
flash_main/script_version.txt
flash_main/setup_version.txt
```

## DLDSC/STCDP Compact Path

Workspace:

```text
/home/adnan-cdx/codex-isolated/dldsc_collectives_stage_local_20260701_015300
```

The CDX lane avoided the unsafe LLVM-managing CMake path and rebuilt with:

```text
MANAGE_LLVM=false
LLVM_PROJ_SRC=/home/adnan-cdx/dt-inductor-mixed/llvm-project
LLVM_PROJ_BUILD=/home/adnan-cdx/dt-inductor-mixed/build/llvm
```

Concrete test added in the isolated Deeptools tree:

```text
dcg/test/dcg_unit_test.cpp
stcdpLibtest.lxLe128LocalSubStickAssembleMetadata
```

Result:

```text
PASSED 1 test
```

What it proves:

- `SenPcfgLE128BTransferNode` can round-trip `isPartStick=true`.
- It can carry four `byteOffsetTr` entries:

```text
32 bytes at offsets 0, 32, 64, 96
```

What it does not yet solve:

- `LE128BTransferInfo` has only one `Offset` plus `bytesToTransfer` and
  `addBaseAddr`.
- A production local assemble/extract needs per-entry:

```text
srcBase
srcByteOffset
dstBase
dstByteOffset
numBytes
```

Conclusion:

- Direct partial-stick ring transfer is not available in current L3 ring
  opcode/codegen.
- Compact DLDSC can still cover gather/all-gather/broadcast/multicast using
  whole-stick ring staging plus a local LX assemble/extract primitive.
- Reduce/all-reduce also need a partial-stick reduce/accumulate primitive;
  byte copy alone is insufficient.

## Explicit Byte-Range Path

Workspace:

```text
/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212
```

New helper scripts in isolated workspaces:

```text
deeptools/scripts/explicit_lx_range_semantic_check.py
torch-spyre/tools/emit_attention_ranged_lx_remap.py
```

Validation artifact:

```text
/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/explicit_range_attention_emitter_20260701_020252
```

Results:

```text
modeled semantic byte check: pass
modeled destination bytes: 128
destination coverage: core 19 [36864, 36992), contiguous
dtTable rows checked: 4
DXP/senulator: rc=0
```

Local artifacts:

```text
explicit_range/emitter_report.json
explicit_range/semantic_check.json
explicit_range/movement_range_estimates.json
explicit_range/dxp_explicit_range_diag.txt
```

Movement-range scaling:

| edge | expanded moves | grouped moves | notes |
|---|---:|---:|---|
| `sdsc_10 Tensor1` gather, per consumer core | 65,536 | 4 | one consumer core needs four stride/count groups |
| `sdsc_10 Tensor1` gather, all 32 cores | 2,097,152 | 128 | expanded physical moves are not viable |
| `sdsc_18 buf21` all-gather remote-only | 1,015,808 stick moves | 992 groups | about 130 MB remote transfer |

Conclusion:

- Explicit remap is useful as a targeted diagnostic/research carrier.
- Expanded per-range frontend emission is not viable for full Granite.
- The path only becomes plausible if the contract supports coalesced
  count/stride groups and backend-owned loop-scoped collective lowering.
