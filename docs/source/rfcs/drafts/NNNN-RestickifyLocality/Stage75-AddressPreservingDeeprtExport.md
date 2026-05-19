# Stage 75: Address-Preserving Deeprt Export And Runtime Packaging

## Goal

Stage 74 proved that the address-preserving two-step data-op artifact lowers
through `DataOpStandalone`:

```text
ReStickifyOpLx -> STCDPOpLx
```

with endpoint addresses patched from the real scheduled Torch-Spyre producer and
consumer SDSCs.

Stage 75 checked the next packaging layer:

```text
address-preserving data-op SDSC
  -> Deeprt data-op export
  -> senprog/init artifacts
  -> minimal Torch-Spyre launch_kernel packaging
```

This is still a packaging/runtime smoke, not tensor correctness.

## Tooling

Added the probe C++ source used in earlier stages:

```text
tools/deeprt_dataop_export_probe.cpp
```

The harness injects one `SuperDsc` into a one-node `DscSenGraph` and calls:

```cpp
DeepRt::runSchedulerCodeGenInitPipelinePerSdsc(node);
DeepRt::printAndExport(4);
```

This avoids the raw `dxp_standalone --bundle` path, which rejects SDSCs with
`datadscs_`.

## Input

Stage74 address-preserving SDSCs:

```text
/tmp/stage74-address-preserving/sdsc_stage3b_address_preserving_2048.json
/tmp/stage74-address-preserving-baseline/sdsc_baseline_address_preserving_2048.json
```

Both preserve the same endpoint address evidence from Stage 74:

| Endpoint | Scheduled LX base |
|---|---:|
| producer output | `16384` |
| consumer input | `8192` |

## Deeprt Export Commands

Stage3B-shaped:

```sh
export SENTIENT_BASE_INSTALL_DIR=/opt/ibm/spyre
export DEEPTOOLS_INSTALL_DIR=/opt/ibm/spyre/deeptools
export PATH=/opt/ibm/spyre/deeptools/bin:/opt/ibm/spyre/runtime/bin:$PATH
export LD_LIBRARY_PATH=/opt/ibm/spyre/deeptools/lib:/opt/ibm/spyre/runtime/lib:${LD_LIBRARY_PATH:-}

/tmp/stage65-deeprt-dataop-probe \
  /tmp/stage74-address-preserving/sdsc_stage3b_address_preserving_2048.json \
  /tmp/stage75-deeprt-address-preserving-stage3b \
  sentient
```

Baseline-shaped:

```sh
/tmp/stage65-deeprt-dataop-probe \
  /tmp/stage74-address-preserving-baseline/sdsc_baseline_address_preserving_2048.json \
  /tmp/stage75-deeprt-address-preserving-baseline \
  sentient
```

## Export Results

The Stage3B-shaped export produced complete execute-node artifacts, including:

```text
execute/0_TwoStepReStickifyLxStcdp_stage3b_dataop/sdsc.json
execute/0_TwoStepReStickifyLxStcdp_stage3b_dataop/senprog.txt
execute/0_TwoStepReStickifyLxStcdp_stage3b_dataop/smc.txt
execute/0_TwoStepReStickifyLxStcdp_stage3b_dataop/init.txt
```

The old throwaway Deeprt harness segfaulted after writing the execute artifacts
for the Stage3B-shaped case, before writing the top-level `after_pipeline.json`
and DSG side files. The exported `senprog.txt` and `init.txt` were present and
usable, so this is recorded as a harness/export cleanup issue rather than a
codegen failure. The baseline-shaped export returned normally.

Traffic-shape scan:

| Mode | File | `HBM` | `L3` | `L3LU` | `L3SU` | `LXLU` | `LXSU` | bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Stage3B-shaped | `senprog.txt` | 0 | 0 | 0 | 0 | 64 | 64 | 221345 |
| Stage3B-shaped | `smc.txt` | 0 | 0 | 0 | 0 | 0 | 0 | 167770 |
| Stage3B-shaped | `init.txt` | 0 | 0 | 0 | 0 | 0 | 0 | 6682 |
| baseline-shaped | `senprog.txt` | 0 | 5312 | 96 | 96 | 64 | 64 | 513407 |
| baseline-shaped | `smc.txt` | 0 | 5120 | 0 | 0 | 0 | 0 | 348219 |
| baseline-shaped | `init.txt` | 0 | 0 | 0 | 0 | 0 | 0 | 35980 |

This preserves the Stage74 contrast at the exported program level:

```text
baseline-shaped: ring-facing L3 traffic remains
Stage3B-shaped:  L3 disappears; LXLU/LXSU remain
```

## Runtime Packaging Smoke

The Stage3B `init.txt` was staged into the minimal Torch-Spyre runtime directory
shape:

```text
/tmp/stage75-address-preserving-runtime-shape/
  bundle.mlir
  loadprogram_to_device/
    stage75-address-preserving-runtime-shape-SenProgSend/
      init.txt
```

The `init.txt` came from:

```text
/tmp/stage75-deeprt-address-preserving-stage3b/execute/0_TwoStepReStickifyLxStcdp_stage3b_dataop/init.txt
```

Launch smoke:

```python
import torch
import torch_spyre
from torch_spyre._C import launch_kernel

x = torch.empty((1,), dtype=torch.float16, device="spyre")
torch.accelerator.synchronize()

launch_kernel("/tmp/stage75-address-preserving-runtime-shape", [])
torch.accelerator.synchronize()
```

Result:

```text
initializing runtime
launching
launch returned
sync returned
```

So the address-preserving Stage3B Deeprt export can still be packaged into the
minimal no-argument Torch-Spyre launch shape.

## Important Bundle Finding

The captured Torch-Spyre fused-add bundle is not packaged as one init per SDSC.
It has one `bundle.mlir` containing three SDSCs:

```mlir
sdscbundle.sdsc_execute () {sdsc_filename="sdsc_0_add.json"}
sdscbundle.sdsc_execute () {sdsc_filename="sdsc_1_ReStickifyOpHBM.json"}
sdscbundle.sdsc_execute () {sdsc_filename="sdsc_2_add.json"}
```

but only one runtime program upload:

```text
loadprogram_to_device/sdsc_fused_add_t_0_...-SenProgSend/init.txt
```

That means there is no simple "replace only the restickify init" operation for a
real fused graph. Replacing the restickify path in production will require a
mixed bundle/export path that compiles producer, data-op restickify, and
consumer into one runtime program.

## Interpretation

This stage strengthens the address-preserving data-op route:

- Stage74 proved address-preserving data-op lowering through
  `DataOpStandalone`.
- Stage75 proves the same address-preserving Stage3B artifact can reach Deeprt
  `senprog.txt`/`init.txt` export with no textual `HBM` or `L3`.
- The exported Stage3B init can be staged into the existing Torch-Spyre
  `launch_kernel` directory contract and synchronized successfully.

But it also sharpens the remaining production blocker:

```text
current Torch-Spyre bundle:
  one fused program for producer + ReStickifyOpHBM + consumer

current data-op proof:
  standalone restickify-like data-op program
```

The missing piece is not whether the data-op can be exported. It can. The
missing piece is integrating it into the same fused runtime program as the
surrounding producer and consumer.

## Next Step

The next real integration experiment should be a mixed-graph Deeprt export:

```text
sdsc_0_add compute-op
sdsc_1 address-preserving data-op restickify
sdsc_2_add compute-op
```

Acceptance criteria:

1. Deeprt exports one runtime program for the mixed graph.
2. The exported program keeps the Stage3B restickify segment free of HBM/L3.
3. The staged bundle launches through `launch_kernel`.
4. The full graph returns numerically correct output.

If mixed Deeprt export cannot represent compute-op plus data-op nodes together,
then the fallback is to find the DLDSc equivalent that DXP accepts inside the
normal Torch-Spyre bundle ABI.
