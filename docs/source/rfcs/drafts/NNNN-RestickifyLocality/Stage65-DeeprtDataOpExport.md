# Stage 65: Deeprt Data-Op Export Probe

## Summary

Stage 65 followed the Stage 64 recommendation: avoid raw DXP import and try the
Deeprt vertical scheduler/codegen/export path for the scheduled two-step
`ReStickifyOpLx -> STCDPOpLx` data-op SDSC.

This worked.

Using a small throwaway C++ harness, we injected the scheduled data-op SDSC into
a one-node `DscSenGraph` and called Deeprt's public vertical pipeline:

```cpp
DeepRt::runSchedulerCodeGenInitPipelinePerSdsc(node);
DeepRt::printAndExport(4);
```

Result:

- `senpcfg`, `senulator`, and `sentient` modes all returned `rc=0`.
- `sentient` export produced `sdsc.json`, `senprog.txt`, `smc.txt`, and
  `init.txt`.
- `senulator` export produced `sdsc.json`, `senprog.txt`, `senprog.json`, and
  `smc.txt`.
- The exported Sentient/Senulator `senprog.txt` has zero textual `HBM` and zero
  textual `L3` hits, while retaining `LXLU`/`LXSU` activity.

This is the strongest result so far for the data-op route: Deeptools can produce
runtime-style exported program artifacts for the LX-local restickify data-op
without taking the raw DXP path that rejects `dataOpdscs_`.

## Input

Same scheduled Stage 3B data-op SDSC as Stage 64:

```text
/tmp/stage64-two-step-scheduled-tool/sdsc_stage3b_TwoStepReStickifyLxStcdp_2048.json
```

It contains:

```text
input_dataops=2 input_dldscs=0 cores=32
```

The two data ops are the scheduled two-step contract:

1. `ReStickifyOpLx`
2. `STCDPOpLx`

with `coreIdToDscSchedule` sequencing both data ops on each participating core.

## Probe Harness

The harness does not patch Deeptools. It links against the installed Deeptools
libraries and exercises public Deeprt interfaces.

High-level setup:

```cpp
auto sdsc = std::make_shared<SuperDsc>();
sdsc->importJson(sdsc_path);
sdsc->target_ = backend;

DesignSpaceConfigGlobal dsc_global(backend, true, 32);
dsc_global.dtVersion = 2;
dsc_global.doTraining = false;
dsc_global.numDevices = 1;
dsc_global.sysDef.numCoreletsPerCore = 2;
dsc_global.dataDebugMode = true;
dsc_global.parallelThreads = 1;

DeepRt deep_rt(dsc_global);
deep_rt.exportDsc = true;
deep_rt.exportPcfg = true;
deep_rt.pruneDataDsc = false;
deep_rt.noCodeGen = false;
deep_rt.outputDir = out_dir;

auto* graph = new sengraph::DscSenGraph(1);
auto* node = graph->insertNode(sdsc->name_, "SenPreparedOp");
graph->finalize();

deep_rt.dSenGraph = graph;
deep_rt.dsgNodeToSdsc_[node] = sdsc;
deep_rt.dsgNodeFold0ToAllFoldedNodeExphase_[node].emplace_back(node, 0);
deep_rt.be_usage_.dtversion = 2;
deep_rt.be_usage_.addInfo(sdsc.get(), DeepRt::CodeGenTools::DCC);

deep_rt.runSchedulerCodeGenInitPipelinePerSdsc(node);
deep_rt.printAndExport(4);
```

The first attempt crashed at `Program FramePtr Filler` because the direct
per-SDSC call bypassed fold bookkeeping normally created by
`runSchedulerCodeGenInitPipeline()`. Seeding the one-node fold map fixed it.

## Export Results

### Sentient

Command shape:

```sh
/tmp/stage65-deeprt-dataop-probe \
  /tmp/stage64-two-step-scheduled-tool/sdsc_stage3b_TwoStepReStickifyLxStcdp_2048.json \
  /tmp/stage65-deeprt-export-sentient \
  sentient
```

Result: `rc=0`.

Exported files:

| File | Size |
|---|---:|
| `execute/.../sdsc.json` | 909539 |
| `execute/.../senprog.txt` | 221409 |
| `execute/.../smc.txt` | 167834 |
| `execute/.../init.txt` | 6682 |
| `execute_dsg.txt` | 78 |
| `ldsToDciPath.json` | 29 |
| `ldsToIsgInfo.json` | 60 |
| `segment_size.json` | 390 |
| `prog_size.json` | 96 |

Tool timing:

| Phase | Time |
|---|---:|
| DCG | 15 ms |
| DCC | 244 ms |
| DIP | 7 ms |
| FillProgramFramePtr | 2 ms |

Program summary:

```text
after_progstateinfo=0
after_spb=0
prog_frame target=1 size=3328
```

Traffic-shape scan:

| File | HBM | L3 | LXLU | LXSU |
|---|---:|---:|---:|---:|
| `senprog.txt` | 0 | 0 | 128 | 128 |
| `smc.txt` | 0 | 0 | 96 | 32 |
| `init.txt` | 0 | 0 | 0 | 0 |

### Senulator

Result: `rc=0`.

Exported files:

| File | Size |
|---|---:|
| `execute/.../sdsc.json` | 909540 |
| `execute/.../senprog.txt` | 221409 |
| `execute/.../senprog.json` | 230912 |
| `execute/.../smc.txt` | 167834 |
| `execute_dsg.txt` | 78 |
| `segment_size.json` | 390 |
| `prog_size.json` | 98 |

Tool timing:

| Phase | Time |
|---|---:|
| DCG | 15 ms |
| DCC | 243 ms |
| FillProgramFramePtr | 6 ms |

Program summary:

```text
after_progstateinfo=0
after_spb=1
prog_frame target=2 size=231040
```

Traffic-shape scan:

| File | HBM | L3 | LXLU | LXSU |
|---|---:|---:|---:|---:|
| `senprog.txt` | 0 | 0 | 128 | 128 |
| `senprog.json` | 11 | 0 | 128 | 128 |
| `smc.txt` | 0 | 0 | 96 | 32 |

The `senprog.json` `HBM` strings are not direct proof of HBM traffic; the
textual program file is cleaner for this quick scan. The SDSC JSON also contains
generic `hbmStartAddress_`/`hbmSize_` fields from labeled data structures, so
the program text remains the better first-pass traffic-shape check.

### Senpcfg

Result: `rc=0`.

Exported files:

| File | Size |
|---|---:|
| `execute/.../sdsc.json` | 62594 |
| `execute/.../pcfg.json` | 830685 |
| `execute/.../senprog.json` | 594126 |
| `execute/.../senprog.txt` | 0 |
| `execute_dsg.txt` | 78 |
| `segment_size.json` | 390 |
| `prog_size.json` | 98 |

Tool timing:

| Phase | Time |
|---|---:|
| DCG | 15 ms |
| PcfgGen | 17 ms |
| FillProgramFramePtr | 16 ms |

This is useful for PCFG inspection, but Sentient/Senulator are more relevant for
runtime-style generated programs.

## What This Proves

The scheduled data-op restickify can pass through a Deeprt path that raw DXP
could not handle:

```text
scheduled data-op SDSC
  -> Deeprt one-node DscSenGraph injection
  -> DataOp scheduler
  -> DCC
  -> ProgIR verification/optimization
  -> DIP / program frame
  -> Deeprt export
  -> sentient/senulator senprog artifacts
```

That path generates a program with no textual HBM/L3 traffic in `senprog.txt`.
This is a stronger compile/export proof than Stage 64's standalone
`DataOpStandalone`/`dcc_standalone` proof.

## What This Still Does Not Prove

This still is not a full Torch-Spyre runtime execution proof.

The exported tree is Deeprt-shaped, not the exact DXP bundle layout currently
used by `SpyreSDSCKernelRunner`. Torch-Spyre today does:

```python
generate_bundle(...)
subprocess.run(["dxp_standalone", "--bundle", "-d", output_dir])
launch_kernel(output_dir, args)
```

Raw DXP bundle import still rejects `dataOpdscs_`:

```text
Datadsc not allowed, use dldsc
```

So we have a viable compiler/export route, but not yet a drop-in replacement for
the current Torch-Spyre runtime directory contract.

## Current Best Interpretation

The restickify data-op route is now technically plausible:

- Deeptools has a first-class data-op scheduler/codegen/export route.
- The generated program artifacts are LX-local in the program text.
- We do not need to force the raw DXP SDSC-tree path to accept `dataOpdscs_`.

The implementation problem has narrowed to runtime packaging:

1. Either add a Torch-Spyre compile path that invokes this Deeprt-style data-op
   export when an SDSC is pure `dataOpdscs_`; or
2. produce the DXP/Flex runtime directory shape from the Deeprt-exported
   artifacts; or
3. add/obtain a Deeptools-supported standalone binary that performs this
   one-node Deeprt data-op export directly.

## Recommended Next Step

Compare a normal DXP-compiled Torch-Spyre kernel directory against the Stage 65
Deeprt-exported directory and identify the minimal files `launch_kernel()` truly
requires.

Specifically compare:

- `execute_dsg.txt`
- `execute/<node>/sdsc.json`
- `execute/<node>/senprog.txt`
- `execute/<node>/init.txt`
- `segment_size.json`
- `prog_size.json`
- any `loadprogram_to_device*` files/folders generated by DXP but absent from
  the Deeprt probe

If `launch_kernel()` only needs the execute graph plus program/init artifacts,
then a Torch-Spyre prototype can dispatch pure data-op SDSCs through a small
Deeprt-export helper. If it needs DXP-specific autopilot/load-program artifacts,
the next prototype should synthesize those from the Deeprt export or call into
DXP after Deeprt has produced a program frame.
