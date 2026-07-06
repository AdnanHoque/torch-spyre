# Generic Gather/Fan-In DLDSC Relayout Proof - 2026-07-06

This packet records the current evidence for generic LX gather/fan-in through the DLDSC relayout path.

## Branches

- Torch code branch: `gather-restickify`
- Torch SHA: `44816280a98c5cf7f67f36e99579561505ab813e`
- Deeptools code branch: `gather-restickify`
- Deeptools SHA: `e3e265d22c7283054dd36e147a7e7ec919606441`
- Artifact branch base SHA before this packet: `2eb5c0795dcc643ad0c9bf110853632dbbbcfad2`

## What This Proves

Generic gather/fan-in is the case where multiple producer LX shards feed one smaller set of consumer compute shards without arithmetic. This is distinct from reduce/all-reduce, which need arithmetic combining, and distinct from the matmul operand broadcast/kernel-neighbor path, which is a special staged matmul RHS fanout path.

The evidence is two-sided:

1. Torch emits the DLDSC contract for a generic `gather` edge:
   - top-level `lxRelayoutClassifications_` keeps `kind=gather`, `communication_class=gather`, `communication_pattern=many_to_one`, and `max_fanin=4`.
   - the LX input allocation carries producer ownership in `scheduleTree_[0].coordinates_.coreIdToWkSlice_`.

2. Deeptools consumes mismatched DLDSC coordinates through the generic relayout insertion path:
   - `DxpTestFixture.CoreWorkDivIncomptLxRelayoutCardinality` rewrites the same fixture into full->sliced, sliced->full, and sliced->replicated-full cases.
   - the sliced->full case is the generic gather/fan-in proof: producer memIds `{7,15,23,31}` become one consumer memId `{0}`.
   - DXP emits an inserted `LxRelayout` SuperDSC with one `STCDPOpLx` data-op, not an HBM relayout.

## Commands

Torch:

```bash
cd /home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/torch-spyre
source /home/adnan-cdx/dt-inductor/.venv/bin/activate
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH=$PWD:${PYTHONPATH:-}
python3 -m pytest tests/inductor/test_lx_relayout_dldsc.py -q
```

Deeptools:

```bash
cd /home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/deeptools/build-deeptools
export DEEPTOOLS_PATH=/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/deeptools
./dxp/dxp_unit_test --gtest_filter="DxpTestFixture.CoreWorkDivIncomptLxRelayout*"
```

## Results

- Torch: `19 passed`
- Deeptools: `2 passed`

Logs are archived next to this README:

- `torch_lx_relayout_dldsc_pytest.log`
- `deeptools_core_work_div_relayout_gtest.log`

## Caveats

- This proves the generic copy-only gather/fan-in substrate, not reduce/all-reduce.
- It does not prove every Granite spill is removable with generic gather; workload-level coverage still depends on whether Torch can keep the source tensor LX-resident and whether the edge is a copy-only coordinate mismatch.
- Low-LX-space fallback behavior is not exhausted here; the passing fixture proves the LX path when space is available.
