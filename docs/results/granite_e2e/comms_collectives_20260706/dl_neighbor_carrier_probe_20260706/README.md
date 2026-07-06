# DL-neighbor carrier probe for Granite matmul operand all-gather

Generated: 2026-07-06 on `adnan-cdx-spyre-dev-pf`.

## Why this probe exists

The current DLDSC metadata correctly classifies the Granite S512 down-projection operand edge as `matmul_operand_broadcast` / `all_gather_replicate`:

- producer: 32-way LX activation shards from the SwiGLU product
- consumer: matmul operand layout with 8 groups and 4 replicas per group
- expected logical transfers: 128

The current `gather_then_restickify` physical carrier uses STCDP/ReStickify data-op lowering and fails DCC with `Require larger IBUFF` (`134/128`). This probe tested whether the existing DL-neighbor ring-transfer path can be used instead.

## Diagnostic patch

`deeptools_dl_neighbor_carrier_probe.diff` is intentionally diagnostic, not production-ready. It tests four narrow changes:

1. allow `matmul_operand_broadcast` plans to target `INPUT` as well as `KERNEL` labeled DS entries;
2. preserve the original producer LX allocation as the source anchor instead of clearing it in DXP;
3. force a separate loop-scoped destination allocation for classified matmul-neighbor operands;
4. skip normal loop-offset computation for folded LX-neighbor allocations that already carry explicit coordinates.

Focused tests still pass with the diagnostic patch:

- `LayoutAllgatherRestickify.*`: 32 passed
- `DxpTestFixture.CoreWorkDivIncomptLxRelayout*`: 2 passed

## Replay ladder

All replays used the Granite S512 SDSC bundle from:

`runs/granite_s512_oneflag_no_wrapper_20260706_202931/block_prefill/cache/inductor-spyre/sdsc_fused_add_linear_mul_silu_split_with_sizes_3_clt9lx2o`

The main command shape was:

```bash
SPYRE_LX_PLANNER_RELAYOUT=1 DXP_LX_FRAC_AVAIL=1 DEEPTOOLS_MATMUL_OPERAND_BROADCAST_KERNEL_NEIGHBOR=1 DEEPTOOLS_ALLOW_DIRECT_KERNEL_NEIGHBOR_LAYOUT_BYPASS=1 DEEPTOOLS_ALLOW_MIXED_HBM_IFN_DIAGNOSTIC=1 dxp_standalone -d <granite-s512-sdsc-bundle>
```

Observed progression:

| Probe | Result |
| --- | --- |
| existing kernel-neighbor path | rejected because the classified operand is activation `INPUT`, not `KERNEL` |
| allow INPUT | reached mixed HBM + input-neighbor guard |
| diagnostic mixed HBM+IFN | raw DDC `map::at` during coordinate propagation |
| preserve source allocation only | scheduler could not find a distinct source allocation |
| preserve source + force destination allocation | reached same DDC coordinate propagation issue |
| skip folded-neighbor loop-offset computation | progressed to retry threshold on normal weight/KERNEL HBM transfer: `allocate_lds1_lx -> transfer_lds1_src:lxlu_dst:ptrow0, dim=in` |

## Current read

The DL-neighbor carrier avoids the STCDP IBUFF failure class, but it is not production-ready yet. The next backend gap is same-DSC coexistence of normal HBM matmul operand transfers and LX-neighbor activation transfers. That matters for Granite because we intentionally ignore weight-restickify/preload as out of scope, but the down-projection matmul still has a normal weight/KERNEL operand transfer in the same DSC.

The clean next backend slice is not many-to-one `ReStickifyOpLx`. It is a staged matmul-operand neighbor carrier that:

1. keeps producer LX allocation as source;
2. creates a separate loop-scoped destination/staging allocation;
3. supports mixed normal HBM transfers and LX-neighbor ring transfers in one matmul DSC;
4. implements value-safe staged layout conversion instead of relying on `DEEPTOOLS_ALLOW_DIRECT_KERNEL_NEIGHBOR_LAYOUT_BYPASS`.
