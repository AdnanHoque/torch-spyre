# Latest Acceptance Validation - 2026-07-07

This directory records the focused acceptance validation after authorization was
restored.

## Source Heads

Torch:

```text
AdnanHoque/torch-spyre:gather-restickify
102520820da890d6a62f781e86573f38dcc6f244
```

Deeptools:

```text
Adnan-Hoque1/deeptools:ah/comms-collectives
a5ff55eee627c5c2bd4b7b0518bb0cbaad385952
```

## Results

Torch:

```text
python3 -m pytest -q tests/inductor/test_lx_relayout_dldsc.py
38 passed in 6.70s
```

Deeptools util:

```text
./build-deeptools/util/util_unit_test --gtest_filter="LayoutAllgatherRestickify.*"
32 passed
```

Deeptools DXP:

```text
./build-deeptools/dxp/dxp_unit_test --gtest_filter="DxpTestFixture.MatmulOperandBroadcast*:DxpTestFixture.PartialViewGather*:DxpTestFixture.CoreWorkDivIncomptLxRelayout*"
10 passed
```

The DXP filter covers:

- oversized matmul-operand relayout fail-closed;
- broadcast fail-closed control;
- multicast fail-closed control;
- bounded broadcast gather/restickify compile;
- bounded multicast gather/restickify compile;
- bounded partial-view gather compile;
- missing/invalid partial-view gather offset fail-closed;
- core-work-division LX relayout.

## Files

```text
torch/heads.txt
torch/torch_lx_relayout_dldsc.log
torch/torch_lx_relayout_dldsc.rc
deeptools/heads.txt
deeptools/util_layout_allgather.log
deeptools/util_layout_allgather.rc
deeptools/dxp_comm_focused.log
deeptools/dxp_comm_focused.rc
```

This validation closes the bounded-substrate evidence for gather, broadcast,
multicast, and all-gather/replicate. Full Granite activation materialization is
still intentionally WSR-scoped when the tensor is too large to keep resident.
