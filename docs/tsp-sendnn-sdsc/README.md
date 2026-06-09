# TSP vs sendnn SDSC/Layout Evidence

This note captures the concrete layout evidence behind the prefill MLP fix. The question is whether sendnn gives the backend singleton-preserving layout information that torch-spyre used to drop before the fix.

Short answer: yes, structurally. sendnn does not export the same torch-spyre `sdsc_0.json` schema, so this is not a field-for-field SDSC comparison. The comparable sendnn evidence is its DeepRT host-prep DCI, which shows the prepared activation and weight tensors still carry an explicit singleton dimension. Pre-fix torch-spyre removed that singleton from the SDSC matmul iteration space; post-fix torch-spyre preserves it as `x: 1`.

## Shape

The focused projection shape is the prefill MLP-style matmul:

```text
activation: [1, 512, 4096]
weight:     [4096, 12800] or equivalent static unit-batch form
output:     [1, 512, 12800]
```

The same issue also shows up on the QO/KV projection shapes with smaller `N`.

## Main Finding

| path | backend-visible evidence | singleton visible? | relevant schedule/layout clue |
| --- | --- | --- | --- |
| torch-spyre pre-fix SDSC | `numWkSlicesPerDim_ = {"mb": 8, "out": 4, "in": 1}` | no | no `x`; input layout `[in, mb]`; output layout `[out, mb]` |
| torch-spyre post-fix SDSC | `numWkSlicesPerDim_ = {"x": 1, "mb": 4, "out": 8, "in": 1}` | yes | `x` preserved; input layout `[mb, in, x]`; output layout `[mb, out, x]` |
| sendnn DeepRT input DCI | `output_shape_ = [64, 512, 64, 1]` | yes | activation prepared with an explicit singleton dimension |
| sendnn DeepRT weight DCI | `output_shape_ = [64, 4096, 200, 1]` | yes | weight prepared with an explicit singleton dimension |

This supports the PR's direction: make torch-spyre hand DeepTools a matmul description with the same kind of singleton-preserving structure sendnn already exposes, rather than letting the size-1 batch axis disappear during graph/layout construction.

## Torch-Spyre Pre-Fix SDSC

Artifact:

```text
docs/tsp-sendnn-sdsc/artifacts/tsp/tsp_prefill_projection_pre_fix_sdsc_0.json
```

Key fields:

```json
{
  "numWkSlicesPerDim_": {
    "mb": 8,
    "out": 4,
    "in": 1
  },
  "N_": {
    "name_": "n",
    "mb_": 512,
    "out_": 4096,
    "in_": 4096
  },
  "dataStageParam_": {
    "0": {
      "ss_": {
        "name_": "core",
        "mb_": 64,
        "out_": 1024,
        "in_": 4096
      }
    }
  },
  "primaryDsInfo_": {
    "INPUT": {
      "layoutDimOrder_": ["in", "mb"],
      "stickDimOrder_": ["in"],
      "stickSize_": [64]
    },
    "KERNEL": {
      "layoutDimOrder_": ["in", "out"],
      "stickDimOrder_": ["out"],
      "stickSize_": [64]
    },
    "OUTPUT": {
      "layoutDimOrder_": ["out", "mb"],
      "stickDimOrder_": ["out"],
      "stickSize_": [64]
    }
  }
}
```

The old SDSC is still a legal matmul. It keeps the correct stick axes: activation sticks are on `in`, while weight/output sticks are on `out`. The problem is that the singleton batch/schedule identity is gone, so the backend sees a flatter `mb/out/in` matmul and chooses the poorer `mb=8, out=4` split.

## Torch-Spyre Post-Fix SDSC

Artifacts:

```text
docs/tsp-sendnn-sdsc/artifacts/tsp/tsp_prefill_projection_post_fix_sdsc_0.json
docs/tsp-sendnn-sdsc/artifacts/tsp/tsp_prefill_mlp_gate_up_post_fix_sdsc_0.json
```

Key fields from the post-fix projection SDSC:

```json
{
  "numWkSlicesPerDim_": {
    "x": 1,
    "mb": 4,
    "out": 8,
    "in": 1
  },
  "N_": {
    "name_": "n",
    "x_": 1,
    "mb_": 512,
    "out_": 4096,
    "in_": 4096
  },
  "dataStageParam_": {
    "0": {
      "ss_": {
        "name_": "core",
        "x_": 1,
        "mb_": 128,
        "out_": 512,
        "in_": 4096
      }
    }
  },
  "primaryDsInfo_": {
    "INPUT": {
      "layoutDimOrder_": ["mb", "in", "x"],
      "stickDimOrder_": ["in"],
      "stickSize_": [64]
    },
    "KERNEL": {
      "layoutDimOrder_": ["in", "out"],
      "stickDimOrder_": ["out"],
      "stickSize_": [64]
    },
    "OUTPUT": {
      "layoutDimOrder_": ["mb", "out", "x"],
      "stickDimOrder_": ["out"],
      "stickSize_": [64]
    }
  }
}
```

The post-fix MLP gate/up SDSC has the same structure, with `out_ = 12800` and per-core `out_ = 1600`. The key change is not moving the stick axis. The stick axes remain `in` for activation and `out` for weight/output. The key change is preserving the singleton batch dimension as a real schedule/layout axis, which lets the planner pick the better `mb=4, out=8` work split.

## sendnn DeepRT Evidence

Artifacts:

```text
docs/tsp-sendnn-sdsc/artifacts/sendnn/sendnn_prefill_mlp_bmm_input0_dci.json
docs/tsp-sendnn-sdsc/artifacts/sendnn/sendnn_prefill_mlp_bmm_input1_dci.json
docs/tsp-sendnn-sdsc/artifacts/sendnn/sendnn_prefill_mlp_ldsToDciPath.json
```

Activation input (`bmm-Input0`):

```json
{
  "dsName_": "bmm-Input0",
  "input_shape_": [4096, 512, 1],
  "output_shape_": [64, 512, 64, 1],
  "dcsi_": [
    {
      "size_": [64, 512, 64, 1],
      "stride_src_": [1, 4096, 64, 2097152],
      "stride_dst_": [1, 64, 32768, 2097152]
    }
  ]
}
```

Weight input (`bmm-Input1`):

```json
{
  "dsName_": "bmm-Input1",
  "input_shape_": [12800, 4096, 1],
  "output_shape_": [64, 4096, 200, 1],
  "dcsi_": [
    {
      "size_": [64, 4096, 200, 1],
      "stride_src_": [1, 12800, 64, 52428800],
      "stride_dst_": [1, 64, 262144, 52428800]
    }
  ]
}
```

This is the key sendnn-side proof: sendnn's prepared tensors still expose an explicit singleton dimension to DeepRT/DeepTools. That is the layout/schedule cue torch-spyre was losing before the PR.

## Performance Context

The same run root includes reports showing the pre/post behavior:

```text
docs/tsp-sendnn-sdsc/artifacts/reports/pr_attention_fix_four_rows_main_report.txt
docs/tsp-sendnn-sdsc/artifacts/reports/pr_attention_fix_four_rows_report.txt
```

Selected numbers:

| op | shape | pre-fix torch-spyre kernel ms | post-fix torch-spyre kernel ms | sendnn kernel ms |
| --- | --- | ---: | ---: | ---: |
| matmul KV | `[[1, 512, 4096], [4096, 1024]]` | 0.128 | 0.089 | 0.107 |
| matmul QO | `[[1, 512, 4096], [4096, 4096]]` | 0.559 | 0.321 | 0.366 |
| matmul MLP-proj | `[[1, 512, 4096], [4096, 12800]]` | 3.745 | 1.018 | 0.952 |
| mlp | `[[1, 512, 4096]]` | 21.725 | 6.952 | 5.776 |

The layout difference above is therefore not just cosmetic metadata. It lines up with the observed movement from underfilled prefill kernels to near-sendnn projection performance.

## Provenance

Source pod run roots:

```text
/home/adnan-cdx/dt-inductor-codex-clean/profiler_runs/pr_attention_fix_four_rows_20260609_024738
/home/adnan-cdx/dt-inductor-codex-clean/profiler_runs/default_no_granite_costmodel_pure_20260606_030756
/home/adnan-cdx/dt-inductor-codex-clean/profiler_runs/pr_attention_fix_lightweight_bypass_20260609_113122
```

The `default_no_granite_costmodel_pure` run root provides the sendnn DeepRT export. The `pr_attention_fix_four_rows` run root provides the paired pre/post torch-spyre SDSCs and reports. The `pr_attention_fix_lightweight_bypass` run root provides the post-fix fused MLP gate/up SDSC.
