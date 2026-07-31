# Ring-native matmul handoff

Last updated: 2026-07-30

## Executive status

The ordinary stock matmul is healthy on the exact pinned stack used by the
ring-matmul experiments. The fresh normal-frontend control compiled and ran

```text
A[64, 2048] @ W[2048, 4096] -> C[64, 4096]
```

through `torch.matmul -> torch.compile -> stock BMM -> device -> D2H`, and all
262,144 outputs passed the CPU comparison in every executed profile.

The ring-native candidate is **not ready to time**. Its latest
K32-epoch/PE-LX-carry lowering:

- compiles;
- passes a complete emitted-program balance audit;
- launches on device;
- deadlocks at `device_synchronize`;
- recovers cleanly to a completing stock launch after the timed-out process is
  terminated.

There is therefore no ring-matmul speedup result yet. The immediate blocker is
candidate-specific completion, not stock matmul, the device, or the pinned
runtime stack.

This is a documentation-only handoff. The implementation, tests, prototypes,
manifests, and all unrelated worktree changes remain local and unstaged.

## Scope and non-goals

The mechanism-oracle shape is FP16 `M64 x K2048 x N4096` on 32 cores.

The paid boundary is:

- A starts in producer-native LX;
- W starts in HBM;
- C ends in its required output layout;
- all communication, conversion, computation, and final stores remain inside
  the paid device event.

The research goal is not work-division tuning. A different decomposition is
allowed only when it is part of a new distributed dataflow, and it must retain
same-grid ring-off controls so communication credit remains identifiable.

The final candidate must:

1. beat its identical-grid no-ring control;
2. beat the current best stock matmul;
3. consume core-to-core arrivals directly rather than materializing a
   standalone shuffle;
4. demonstrate useful traffic on both directions of the RIU and SFP fabrics
   when the topology and payload permit it;
5. avoid redundant link bytes and balance the critical links;
6. pass full dynamic correctness before any latency is accepted.

## First-principles target dataflow

For this shape, A is 256 KiB while W is 16 MiB. Moving W around the ring is the
wrong default. The natural distributed algorithm is output-column stationary:

1. shard W by N and keep each W shard at its owner;
2. send the smaller A panels from their producer-native LX placement to the N
   owners over RIU LX-to-LX transport;
3. compute as A arrives instead of writing a standalone relayout result and
   launching a second kernel;
4. retain C at the N owner across all K epochs;
5. use the SFP fabric for local pair sharing/reduction work that does not
   duplicate RIU traffic;
6. write the final C owner shard once.

The selected physical candidate is `M2N16K1` with:

```text
core = 2*n + m
```

Each logical N owner is an adjacent pair of physical cores. This gives the
design two separable communication opportunities:

- RIU: distribute A from LX to the N-owner pairs;
- SFP: share the complementary W work within each adjacent pair while PT
  computes and C remains output-stationary.

The intended steady-state pipeline is:

```text
RIU receives A[g+1]
    ||
SFP/HMI supplies the W work for g
    ||
PT computes g and PE/LX retains C
```

The current K32 PE/LX-carry experiment is only a compute/lifetime control for
this design. It uses HMI-local W delivery and does not yet prove the full
dual-ring algorithm.

## Difference from the incumbent

The normal stock control emits one unhinted `M4N8K1` HBM BMM. A, W, and C use
the ordinary stock path. Hardware multicast may already reduce repeated HBM
delivery, but the operation does not consume a producer-native LX A placement
through the new LX-to-LX substrate.

The target changes the dataflow, not merely the tile sizes:

| Property | Stock incumbent | Ring-native target |
|---|---|---|
| A start | ordinary HBM BMM input | live producer-native LX |
| W policy | stock HBM delivery | N-sharded/stationary owner |
| Cross-core A | none at the graph boundary | RIU LX-to-LX fan-out/stream |
| SFP role | stock internal behavior | useful adjacent-pair W sharing |
| C lifetime | stock generated schedule | retained at owner across K epochs |
| Fusion | one ordinary BMM | receive/compute/carry in one schedule |
| Attribution | incumbent | same-grid ring-off plus incumbent controls |

The theoretical advantage is removal or overlap of A materialization and
owner delivery while preserving W and C ownership. It is not a claim that
communication is free, nor that changing from `M4N8` to `M2N16` is itself a
speedup.

## Compiler substrate present in this fork

The host fork is:

```text
remote: https://github.com/AdnanHoque/torch-spyre.git
branch: ah/communication-cost-model
research base before this documentation commit:
        6952529929cdb9c9e8668e11d94d61312c5d83ca
```

The worktree is intentionally dirty. Thirteen tracked compiler/test files have
853 insertions and 35 deletions, plus the untracked research tree. Preserve
all unrelated work.

The current compiler changes make the target layout expressible by reusing
existing infrastructure:

1. `TensorPhysicalLayout` carries an explicit compound/repeated physical stick
   factorization such as `M4 x K2 x M8`.
2. `SPYRE_MATMUL_ACTIVATION_LAYOUT=preserved` selects the BMM input layout that
   preserves M in A's physical stick instead of the stock K/reduction stick.
3. work-division validation permits the logical M2 split while retaining one
   physical FP16 M64 stick and an explicit per-core tail gap.
4. equal-footprint destination geometry can select the existing
   `ReStickifyOpLx` path, avoiding an HBM restickify.
5. SuperDSC serializes the explicit physical stick sequence and validates that
   it exactly covers the flattened device geometry.
6. source and destination ownership remain represented by the existing
   relayout/STCDP machinery rather than new raw send/receive operations.

Relevant source:

- [`torch_spyre/_inductor/op_spec.py`](../torch_spyre/_inductor/op_spec.py)
- [`torch_spyre/_inductor/propagate_layouts.py`](../torch_spyre/_inductor/propagate_layouts.py)
- [`torch_spyre/_inductor/work_division.py`](../torch_spyre/_inductor/work_division.py)
- [`torch_spyre/_inductor/spyre_kernel.py`](../torch_spyre/_inductor/spyre_kernel.py)
- [`torch_spyre/_inductor/codegen/superdsc.py`](../torch_spyre/_inductor/codegen/superdsc.py)
- local research note `ring_compute_prototypes/COMPILER_CONTRACT.md`
- local research ledger `ring_compute_prototypes/RING_NATIVE_MATMUL_LEDGER.md`

This substrate is not yet a production-ready patch set. It must be reduced
into separately reviewable compiler changes only after the device algorithm
passes.

## Cross-environment path and durability registry

This registry was audited read-only on 2026-07-30 against the Mac, GitHub, and
all four live pods in namespace `a6-quantization`. No device work was launched
and no pod files were changed during the audit.

The registry records where the working state exists. It is not itself a
backup:

| Class | Meaning |
|---|---|
| GitHub | Versioned and recoverable from the named remote commit |
| Mac local | Present on the Mac, but not recoverable from GitHub unless explicitly stated |
| PVC | Survives ordinary pod recreation, but may still be detached, dirty, or unversioned |
| Pod `/tmp` | Container-overlay state; disappears when that pod is replaced |

### Published state

The first published handoff snapshot is:

```text
remote: https://github.com/AdnanHoque/torch-spyre.git
branch: ah/communication-cost-model
commit: c69d92f8689071e1162d900d66024f2d211234a3
path:   ring_compute_prototypes/RING_NATIVE_MATMUL_HANDOFF.md
```

That commit changes exactly this one documentation file. It does **not**
contain the compiler delta, tests, DDL, manifests, harnesses, ledgers, or
generated evidence described below.

The Mac Git HEAD remains:

```text
6952529929cdb9c9e8668e11d94d61312c5d83ca
```

The handoff is still an untracked file in that dirty checkout. Before this
registry was added, its blob matched `c69d92f` byte-for-byte; the registry
revision lives in the untracked handoff rather than in the Mac's Git HEAD.

### Four-pod storage map

| Pod | Durable home | Relationship | Ephemeral state |
|---|---|---|---|
| `adnan-spyre-current-pf` | `/home/adnan` | Same NFSv4-mounted GPFS PVC as CLC and DEV | ordinary `/tmp`, `/dev/shm` |
| `adnan-clc-spyre-dev-pf` | `/home/adnan` | Same NFSv4-mounted GPFS PVC as CURRENT and DEV | ordinary `/tmp`, `/dev/shm` |
| `adnan-spyre-dev-pf` | `/home/adnan` | Same NFSv4-mounted GPFS PVC as CURRENT and CLC | ordinary `/tmp`, `/dev/shm` |
| `adnan-cdx-spyre-dev-pf` | `/home/adnan-cdx` | Separate NFSv4-mounted GPFS PVC | ordinary `/tmp`, `/dev/shm` |

`/tmp/models` is a separately mounted shared-models PVC. No current
continuation artifact was identified there. Listing a `/home/adnan` path once
below covers the same durable object as seen from CURRENT, CLC, and DEV; those
are not three independent copies.

### Mac working set

Workspace root:

```text
/Users/adnan/Documents/Codex/2026-07-20-https-github-com-adnanhoque-torch-spyre
```

The 13 tracked, unstaged compiler/test modifications are:

```text
tests/inductor/test_lx_relayout_dldsc.py
tests/inductor/test_work_division_hint.py
torch_spyre/_inductor/codegen/compute_ops.py
torch_spyre/_inductor/codegen/superdsc.py
torch_spyre/_inductor/config.py
torch_spyre/_inductor/constants.py
torch_spyre/_inductor/lx_relayout.py
torch_spyre/_inductor/op_spec.py
torch_spyre/_inductor/pass_utils.py
torch_spyre/_inductor/propagate_layouts.py
torch_spyre/_inductor/spyre_kernel.py
torch_spyre/_inductor/work_division.py
torch_spyre/_inductor/wrapper.py
```

The output of `git diff --binary -- <the 13 paths above>` has SHA-256:

```text
89ed97d7004576c959bcd1a58529ed3b3000871bd7ed416eed1d531fb814262e
```

The additional activation-layout test is untracked:

```text
tests/inductor/test_matmul_activation_layout.py
```

The primary Mac research roots are below. Their contents are local-only except
for the published handoff snapshot named above:

| Root relative to the workspace | Audit inventory | Role |
|---|---:|---|
| `ring_compute_prototypes/` | 341 files, 43 MiB | contracts, models, harnesses, ledgers, evidence |
| `tmp/` excluding `fontcache` | 538 scratch/evidence files, 209 MiB total root | current DDL, manifests, generated candidates |
| `compiler_patch/` | 152 files, 11 MiB | compiler patch staging and logs |
| `tmp_deeptools_ring_streamed/` | 115 files, 28 MiB | streamed-matmul Deeptools study |
| `ring_matmul_prior_art_20260722/` | 167 files, 107 MiB | source and prior-art corpus |
| `ring_aware_implementation_gap_20260722/` | 73 files, 18 MiB | compiler/runtime gap audit |
| `e2e_harness/` | 1.7 MiB | end-to-end controls |
| `hierarchical_oracle_probe/` | 240 KiB | topology/oracle probes |
| `ring_routing_global_study/` | 28 KiB | global route model |
| `pr2939-live-807014/` | 26 MiB | pinned Torch-Spyre snapshot |
| `pr2939-live-9bea-20260729/` | 23 MiB | later PR2939 snapshot |
| `pr2939-topology-aware-placement/` | 26 MiB | placement experiment |

The exact current continuation inputs are local-only:

| Path relative to the workspace | SHA-256 |
|---|---|
| `tmp/bmm_output_stationary.k32_pe_lx_carry.ddl` | `06fa0e2f147fa906501cb2db84a7c0693949f7e70c566e89e0bf68cb82cd1792` |
| `tmp/k32_pe_lx_carry_completion_manifest.json` | `9d0a27a9b38e83fc2e5ac8a1e8bef96f12ae7d82b39158d458a1e835e8435fe3` |
| `tmp/k32_pe_lx_carry_direct_launch_contract.json` | `e5a738f8081f4095a763757975b54f9019bc9bcce7f732bec0ee9bb45dae8279` |
| `ring_compute_prototypes/RING_NATIVE_MATMUL_LEDGER.md` | `3642465449b2bca24b43dc76d1b221a959915f0dbbc1561ce3e14f998c4b0da9` |
| `ring_compute_prototypes/COMPILER_CONTRACT.md` | `a3d6ba3ab3209a34bc30ac683c4ebe78f9fd28fc8044af3b095097b5c9cb7cc4` |
| `ring_compute_prototypes/HARDWARE_ARCHITECTURE_CONTRACT.md` | `41e77280eb42303396a3918ca44246e743098df37634b72827e9740c7e9947c7` |
| `ring_compute_prototypes/DUAL_FABRIC_CEILING.md` | `607b00950bde66760c4ba0e7c398ece5142e8057640e325fbabda358df958ed1` |

The launch scripts are under:

```text
ring_compute_prototypes/prebenchmark_m2n16_os/
ring_compute_prototypes/true_fp16_os/
ring_compute_prototypes/stationary_weight_matmul_probe.py
```

The first directory also contains local-only launch contracts, ABBA scripts,
split-correction validation, and correctness checkers. The current emitted
SFP-egress investigation is under:

```text
ring_compute_prototypes/evidence/os_hmi2_sfp_egress_20260729/
```

Supporting Mac-only documents and source inputs are:

| Absolute path | SHA-256 |
|---|---|
| `/Users/adnan/torch-spyre-work/RING_MATMUL_DOSSIER.md` | `f9a53c49f58a8af588fcd6d706ad821be87e7184c7d7b005f6ebf7c3ace6819b` |
| `/Users/adnan/torch-spyre-work/R67_PROGRESS_REPORT.md` | `59eed71df24d6e7e13034a354c849695d27ab88831652f21c9c5ddbe8457a9df` |
| `/Users/adnan/torch-spyre-work/ring-native-fp16-matmul-codesign-20260722.md` | `82e9ba20b8f44dc035578b62559cf5399df395ea33fb5266fd98f68f120e5b7d` |
| `/Users/adnan/torch-spyre-work/ring-aware-matmul-and-flash.html` | `dcc91dd31da1e7f5e6d1253c89f7154eec3dd42b1c4b3bc45714f4d06cc575f5` |
| `/Users/adnan/Downloads/AIU_1_0_Rapid_Core_ISA_Spec_v1.0_260121.pdf` | `225e7ac8a83dbeed53861167708d0604265e2a862718a9e3f1a30880e09b5b05` |
| `/Users/adnan/Downloads/deeptools_aiu_lectures.pdf` | `ab3677134145a40aefac0bbe2dae2effc08cd526527753d18c5dde4428fa62e8` |

### Shared `/home/adnan` PVC working set

These paths are durable across ordinary recreation of CURRENT, CLC, and DEV:

| Absolute path | Identity or role | Audit state |
|---|---|---|
| `/home/adnan/codex-isolated/ring_matmul_true_os_torch_20260722_v1` | Torch-Spyre `80701411a151fa6402d08ce7586f671883e1e66b`, branch `codex/ring-streamed-matmul-v1` | dirty, 22 status entries |
| `/home/adnan/codex-isolated/deeptools-master-e3944781` | Deeptools `e3944781cb25b76abeb9b3e87c1f5c5879e84229`, branch `adnan/alltoall-common-refinement-4x8-32x1` | clean |
| `/home/adnan/codex-isolated/ring-native-matmul-20260722` | baseline bundles and controls | non-Git artifact root |
| `/home/adnan/codex-isolated/ring_compute_hw_20260721_v1` | first hardware probes and runs | non-Git artifact root |
| `/home/adnan/codex-isolated/ring_compute_hw_20260721_v2` | audits, replays, DXP wrapper, runs | non-Git artifact root |
| `/home/adnan/codex-isolated/ring_compute_prototypes_20260721_v1` | durable prototype bundle | non-Git artifact root |
| `/home/adnan/codex-isolated/output_stationary_audit_20260722_v1` | output-stationary audit corpus | non-Git artifact root |
| `/home/adnan/codex-isolated/r64_full8_stock_bmm_real_output_sync_device_20260725_v1` | sealed stock completion/control bundle | non-Git artifact root |
| `/home/adnan/claude-isolated/ring_r67_20260727` | complete R67 snapshot, harnesses, bundles, logs, results, scripts | non-Git artifact root |
| `/home/adnan/spyre-envs/main-ac3c7395` | pinned runtime extension tree | non-Git environment root |
| `/home/adnan/spyre-perf-suite-envfix` | perf-suite `5640b6859d09273cc814348489f68778dc88d108`, branch `adnan/run-benchmark-environment` | dirty, 5 status entries |
| `/home/adnan/dt-inductor/torch-spyre` | inherited default Torch-Spyre `cf67411d2071d0e567f4449d87ba3031a331a688`, branch `latest-main` | dirty, 29 status entries; not an accepted control |
| `/home/adnan/dt-inductor/.venv` | Python/Torch execution environment | PVC environment root, lock not yet captured |
| `/home/adnan/codex-isolated/ring_matmul_b35_coordinator_20260729/deeptools` | Deeptools `b35cece729c0ae1707ea71bf5a0d3c4451358a07` | detached, dirty |
| `/home/adnan/codex-isolated/ring_matmul_owner_w_master_b35cece_20260729_v2` | owner-W compiler arm at `b35cece7` | detached, dirty |
| `/home/adnan/codex-isolated/ring_matmul_owner_w_master_b35cece_20260729_clean_control` | owner-W clean control at `b35cece7` | detached, clean |
| `/home/adnan/codex-isolated/ring_matmul_partial_gap_probe_20260729` | partial-gap arm at `b35cece7` | detached, dirty |
| `/home/adnan/codex-isolated/ring_matmul_true_os_master_b35cece_20260729_clc` | true-OS compiler arm at `b35cece7` | detached, dirty |
| `/home/adnan/codex-isolated/ring_matmul_dual_fabric_whole_a_20260722_v1` | early whole-A dual-fabric worktree at `503f5f4d` | dirty, 50 status entries |
| `/home/adnan/codex-isolated/ring_matmul_timing_torch_20260729` | Torch-Spyre `9bea573e6f304fba5357656ce9122f6e4b587700`, branch `codex/ring-native-matmul` | dirty |
| `/home/adnan/codex-isolated/ring_matmul_torch_pr2939_9bea_20260729` | PR2939 Torch-Spyre at `9bea573e` | detached, dirty |

There are 131 broader `ring*`/`matmul*` roots on this PVC, including 41 Git
checkouts, 37 build directories, and 18 evidence/validation directories. The
family roots:

```text
/home/adnan/codex-isolated/ring_matmul_*
/home/adnan/codex-isolated/ring_compute_*
/home/adnan/claude-isolated/ring_r67_*
```

cover the historical R40-R67, marker, source-bound, SFP, K256, owner-W,
partial-gap, and dual-fabric studies without pretending they are all active
continuation inputs.

The accepted launch also depends on image-owned libraries at:

```text
/opt/ibm/spyre/runtime/lib
/opt/ibm/spyre/spyre-comms/lib
/opt/ibm/spyre/deeptools/lib
/opt/ibm/spyre/senlib/lib
```

Those paths are tied to the container image rather than the shared project PVC.

One source overlay needs special treatment:

```text
/home/adnan/codex-isolated/ring_matmul_owner_w_master_b35cece_20260729/deeptools
```

Its `.git` worktree pointer targets a missing historical path. The source is
PVC-resident, but Git cannot currently identify or preserve its delta.

### CDX `/home/adnan-cdx` PVC working set

The current compiler-side roots are:

| Absolute path | Identity or role | Audit state |
|---|---|---|
| `/home/adnan-cdx/codex-isolated/ring_matmul_rowsplit_contract_b35_20260729` | row-split contract worktree at `b35cece7` | detached, one tracked modification |
| `/home/adnan-cdx/codex-isolated/ring_matmul_rowsplit_contract_b35_20260729-build-llvm22` | matching standalone-tool build | PVC artifact |
| `/home/adnan-cdx/codex-isolated/ring_matmul_rowsplit_contract_b35_20260729-evidence` | contract/full-FX/true-OS evidence | row split not yet realized |
| `/home/adnan-cdx/codex-isolated/ring_matmul_geometry_master_20260729` | geometry worktree at `b35cece7` | detached, five tracked modifications |
| `/home/adnan-cdx/codex-isolated/ring_matmul_geometry_master_20260729-build-llvm22` | matching compiler build | PVC artifact |
| `/home/adnan-cdx/codex-isolated/ring_matmul_geometry_master_20260729-evidence` | DCC, PCFG, and MLIR outputs | PVC artifact |
| `/home/adnan-cdx/codex-isolated/dcc_lx_replication4_b35_20260729/deeptools` | most complete current compiler-substrate arm at `b35cece7` | detached, 11 tracked modifications |
| `/home/adnan-cdx/codex-isolated/dcc_lx_replication4_b35_20260729/build-clone` | RelWithDebInfo build with DDL/DCC/DDC/DXP standalone tools | PVC artifact |
| `/home/adnan-cdx/codex-isolated/m4n8_incumbent_audit_b35_20260729` | incumbent control bundle and output | non-Git PVC artifact |
| `/home/adnan-cdx/codex-isolated/ring_matmul_stage_c_route_20260723` | older route worktree at `07992243` | detached, heavily dirty |
| `/home/adnan-cdx/codex-isolated/ring_matmul_stage_c_route_build_20260723` | matching older stage-C Release build | PVC artifact |
| `/home/adnan-cdx/codex-isolated/ring_release_validation_20260723` | older release-validation worktree | detached, heavily dirty |
| `/home/adnan-cdx/codex-isolated/fused_metadata_bridge_20260723` | older metadata-bridge worktree | detached, heavily dirty |

CDX reports that name the Torch-Spyre `9bea573e` source root refer to the
shared `/home/adnan` PVC on another pod. There is no corresponding Torch-Spyre
checkout under `/home/adnan-cdx`; those reports are cross-pod provenance, not
a locally reconstructible source tree.

### Pod-local `/tmp` working set

The exact candidate, harness, and fresh controls named in this handoff exist
only on `adnan-spyre-current-pf`:

| Ephemeral CURRENT path | Role |
|---|---|
| `/tmp/os_hmi2_k32_pe_lx_carry_mscope_20260729/run` | compiled K32 candidate and emitted bundle |
| `/tmp/k32_pe_lx_carry_validation_20260729` | staged completion/correctness harness |
| `/tmp/r64_normal_frontend_stock_unhinted_20260729_v2` | fresh normal-frontend stock control |
| `/tmp/r64_stock_recovery_after_k32_carry_20260729_v1` | post-timeout stock recovery control |

Those four paths are absent from the CLC and DEV pod overlays. Their source
inputs are mirrored on the Mac, but the full generated binaries, logs, and
results are not archived on GitHub or a PVC.

The other pod-local scratch families are:

| Pod | Ephemeral paths or families | Role |
|---|---|---|
| `adnan-spyre-current-pf` | `/tmp/os_hmi2_*_20260729`, `/tmp/ring_matmul_*`, plus the four exact paths above | current completion bisections and controls |
| `adnan-clc-spyre-dev-pf` | 177 matching R40-R65, `os_hmi2_*`, `ring_matmul_*`, and `typed_tensor_*` directories; `/tmp/ring_matmul_v12_f49_integration_source_20260724_composer` is a dirty temporary Git tree at `f49f3e3c` | historical compiler/build/harness overlays; not current exact candidate |
| `adnan-spyre-dev-pf` | `/tmp/owner_w_b35cece_identity`, `/tmp/owner_w_b35cece_ownership`, `/tmp/owner_w_b35cece_tests`, `/tmp/pr2939-dxp-frontend.JvJ2Dh/repo`, `/tmp/ring-ddc-*`, `/tmp/ring_role_asym_patches_20260723`, `/tmp/tranche_stitcher_lane_compile*` | owner-W, DXP, DDC, and stitcher studies |
| `adnan-cdx-spyre-dev-pf` | `/tmp/ring_matmul_geometry_b35.patch`, `/tmp/ring_matmul_occurrence_geometry_20260729.patch`, `/tmp/diagnose_matmul_planner.py` | geometry overlays and planner diagnostic |

The two CDX geometry patches are distinct:

| Ephemeral CDX path | SHA-256 |
|---|---|
| `/tmp/ring_matmul_geometry_b35.patch` | `1e07f9e5416fae059a453038b090ed4e9d9a4a1b9a61b292a15c6e751c8fffd4` |
| `/tmp/ring_matmul_occurrence_geometry_20260729.patch` | `6f76561cc6676b893770b1cb0a665ef0f44f215149f579746188a7ee43491bad` |

### Preservation and reconstruction gaps

The path audit establishes location and identity, not reproducibility. Before
recreating any pod or consolidating the work:

1. archive the four CURRENT `/tmp` roots and the two CDX patch files to a PVC
   with a recursive SHA-256 manifest;
2. preserve the Mac's 13-file compiler diff, untracked activation-layout test,
   DDL, manifests, and harnesses without resetting or cleaning the worktree;
3. preserve patches from every detached dirty PVC checkout before attempting
   to consolidate them;
4. repair or copy the broken owner-W overlay before relying on Git metadata;
5. pin the perf-suite checkout and dirty delta, Python/Torch lock, compiler and
   runtime binaries, container image digest, exact compile command/environment,
   device/node identity, and firmware for a fully reconstructible run.

All four audited pods currently resolve their `app` container to:

```text
sha256:913f394b4b3f03740a9d35f70f273b1cb799d4cba55f7fdaff108d7749d77964
```

Still-unpinned handoff references include the adjacent `r65` static artifact,
the “earlier split control,” and hashes for the candidate final SDSC and stock
result files. Until those gaps are closed, no reader should infer that the
published documentation alone can reconstruct the current candidate.

## Correctness-gate correction

An important false lead was removed on 2026-07-29.

The sealed `r64` replay was not a valid test of stock BMM arithmetic. It
executed:

```text
LX restickify -> stock BMM
```

as one integrated artifact, but the restickify schedule had neither
`before_sync` nor `after_sync` before BMM consumption. Across historical runs,
the same inputs and ABI produced different hashes and recovered different
multiples of 64 dominant columns. That is consistent with a producer/consumer
race, not a deterministic BMM layout or arithmetic failure.

Therefore:

- the sealed `r64` completion result remains useful only for completion;
- its full-output result must not gate stock BMM correctness;
- the correct stock gate is the normal one-root frontend path;
- an integrated relayout+BMM artifact needs its own explicit ordering gate.

An adjacent `r65` static artifact adds the missing restickify `after_sync`, but
it has not been used here as a substitute for the normal stock control.

## Fresh stock control

Device: `adnan-spyre-current-pf` in namespace `a6-quantization`.

Pinned Torch-Spyre source:

```text
/home/adnan/codex-isolated/ring_matmul_true_os_torch_20260722_v1
Git HEAD: 80701411a151fa6402d08ce7586f671883e1e66b
```

Runtime extension:

```text
/home/adnan/spyre-envs/main-ac3c7395/torch-spyre/torch_spyre/_C.so
SHA-256: b681beeb640d5b8524dadfcc787d9ad2725db357adcf4e75f707d92d907487d4
```

Stock Deeptools source:

```text
/home/adnan/codex-isolated/deeptools-master-e3944781
Git HEAD: e3944781cb25b76abeb9b3e87c1f5c5879e84229
bmm.ddl SHA-256:
95db90a11bbce747e55e9e4b5ac4ab1a6034d1649e1d70d3aac4635e656f0268
```

Fresh result:

```text
/tmp/r64_normal_frontend_stock_unhinted_20260729_v2/summary.json
```

Passed gates:

- `correctness_gate=true`;
- `structural_gate=true`;
- `payload_gate=true`;
- one ordinary `batchmatmul`;
- unhinted work division emitted as stock `M4N8K1`.

Numerical results:

| Profile | Checked outputs | Tolerance mismatches | Max absolute error |
|---|---:|---:|---:|
| compile positive | 262,144 | 0 | 0.017578125 |
| dynamic poison A | 262,144 | 0 | 0.0146484375 |
| dynamic restore | 262,144 | 0 | 0.017578125 |
| warm positive | 262,144 | 0 | 0.017578125 |
| measured positive | 262,144 | 0 | 0.017578125 |

`trace_gate=false` in this run because the old harness was invoked with one
profiler iteration and then dereferenced an empty timing record. The complete
correctness/structure/payload report had already been written. This run is
accepted only as correctness and runtime evidence; it is not a latency result.

## Current custom candidate

Source:

```text
tmp/bmm_output_stationary.k32_pe_lx_carry.ddl
SHA-256:
06fa0e2f147fa906501cb2db84a7c0693949f7e70c566e89e0bf68cb82cd1792
```

Compiled artifact:

```text
/tmp/os_hmi2_k32_pe_lx_carry_mscope_20260729/run
```

Artifact hashes:

| Artifact | SHA-256 |
|---|---|
| `bundle.mlir` | `8f096dd0da33e3286ac26678488b686575b9cda4626e47aea556ed15c69ce6dd` |
| `spyreCodeDir/spyrecode.json` | `fc45f6dfd53448700f152b325b0172c9dce0e06d419492d50f530f52d1523078` |
| `spyreCodeDir/init_binary.bin` | `793756a448a5e33719454b678201215bde16ac8fc3e70053da714e3b210e1385` |
| `debug/sdsc_1/smc.txt` | `1ff84c05fee97351047b0cf0af93cfcc66a9e438d1ffd568f9926a5045f9794d` |

External ABI:

```text
(intermediate pool, A, scalar 0.5, W, C)
C = (A * 0.5) @ W
```

The bundle and external tensor/job ABI match the earlier split control:

- host program correction: 8,448 bytes at pointer `120259084288`;
- device jobs: pointers `120259092736` and `120259101952`;
- A: host `[64,2048]`, physical M64-stick layout;
- W: host `[2048,4096]`;
- C: host `[64,4096]`, device `[64,64,64]`.

### Intended K32 schedule

PT has 32 XRF entries. The candidate therefore:

1. preloads one K32 W epoch into XRF;
2. preloads the matching 32 A sticks into L0;
3. computes one K32-local C partial in PT ARF;
4. serializes all eight distinct PT-row M4 partials to PE;
5. for the first epoch, initializes PE with `partial * 1 + 0`;
6. for later epochs, reloads prior C from LX and computes
   `partial * 1 + prior_C`;
7. writes C through the stock PE -> SFP -> LX path after each epoch;
8. repeats for all 128 K32 epochs.

This avoids the illegal or deadlocking full-K PT recurrence while preserving
the desired output-stationary semantics.

### Compile-only structural evidence

The emitted audit covered all:

```text
32 cores x 2 corelets x 8 PT rows = 512 PT streams
```

It verified:

- every row forwards the upstream M strips without arithmetic mixing and
  appends its own M4 strip;
- terminal row 7 emits 32 distinct ordered M positions;
- 128 K32 epochs per PT stream;
- one zero-init plus 31 recurrent K steps per epoch;
- 4,096 XRF captures per PT stream;
- PT -> PE and PE -> SFP -> LX totals agree;
- one first block plus 63 LX reload/add carry blocks per N wave;
- LXSU signals and LXLU waits both equal 126 per core;
- no cross-row C reduction;
- external tensor/job ABI is unchanged.

This is emitted structural evidence, not numerical or runtime proof.

### Fresh device result

Completion harness:

```text
/tmp/k32_pe_lx_carry_validation_20260729
```

The corresponding durable local hash pins are:

| File | SHA-256 |
|---|---|
| `tmp/k32_pe_lx_carry_completion_manifest.json` | `9d0a27a9b38e83fc2e5ac8a1e8bef96f12ae7d82b39158d458a1e835e8435fe3` |
| `tmp/k32_pe_lx_carry_direct_launch_contract.json` | `e5a738f8081f4095a763757975b54f9019bc9bcce7f732bec0ee9bb45dae8279` |

The zero-input, no-D2H, no-timing launch completed import, initialization,
allocation, H2D, and submission. It then stopped at:

```text
device_synchronize: before
```

The 90-second hard timeout terminated it. At 60 seconds the runtime reported:

```text
in_flight_=1 device=0 - possible lost completion
```

The unchanged candidate therefore fails completion. Full correctness and
timing are forbidden.

Immediately afterward, the sealed stock completion launch passed:

```text
/tmp/r64_stock_recovery_after_k32_carry_20260729_v1
status=pass
```

This localizes the lost completion to the custom artifact rather than a wedged
device or a generally broken pinned runtime.

## Current blocker

The blocker is a candidate-specific scheduling deadlock despite mechanically
balanced aggregate counts. Aggregate equality is not sufficient: a cyclic
dependency can have equal producers and consumers while no unit can make the
first forward progress.

The highest-value next work is a single-variable completion bisection of the
PE/LX carry loop. Do not add more ring traffic until this HMI-local control
completes.

The bisection must distinguish:

1. PE waiting for PT row serialization;
2. LXLU waiting for an LXSU write that cannot retire;
3. LXSU waiting for PE/SFP output while PE waits for the LXLU reload;
4. an off-by-one first/recurrent epoch condition;
5. a fence placed at the wrong loop scope despite matching total counts.

### Strongest emitted topology discrepancy

The closest completing stock schedule exposes only the already-reduced
bottom-row result to PE:

```text
ptrow7 -> pe_ptsum
```

The candidate final SDSC instead exposes eight independently named transfers:

```text
ptrow0 -> pe_ptsum
...
ptrow7 -> pe_ptsum
```

Rows 0-6 traverse the downstream PT rows. The emitted bottom-row program
serializes upstream values and local `R0-R3`, but PE still has one `src0=pt`
consumer sequence. The aggregate loop counts can balance while the single
`pe_ptsum` port still has an unsupported eight-producer transaction topology.
The top-level JobPlan is identical to the completing control, so this
PT-row-to-PE topology is the strongest concrete candidate-specific mismatch.

Exact pod evidence:

```text
candidate:
/tmp/os_hmi2_k32_pe_lx_carry_mscope_20260729/run/debug/sdsc_1/sdsc_1.out.out.out.json
transfer region: lines 20786-22352

candidate SMC:
/tmp/os_hmi2_k32_pe_lx_carry_mscope_20260729/run/debug/sdsc_1/smc.txt
PE consumer: lines 477-483
bottom-row serialization: lines 927-942

completing stock:
/home/adnan/codex-isolated/r64_full8_stock_bmm_real_output_sync_device_20260725_v1/bundle/runtime/debug/sdsc_2/sdsc_2.out.out.out.out.json
single bottom-row transfer: lines 16775-16819
```

Do not declare it the root cause without a device bisection. The correct fix,
if confirmed, is to either:

- retain independent row-owned outputs on a native row/LX path; or
- explicitly serialize all eight row partials into one PE transaction with a
  matching temporal consumer.

Do not continue mapping eight independent row-owned tiles directly to
`pe_ptsum`.

### Smallest liveness discriminator: fence-off

Do not start by changing K depth or packet counts. The emitted counts already
balance on every corelet:

```text
PE output:       2 * (4 + 252) groups * 8 = 4096 vectors
SFP forwarding:                                4096 vectors
LXSU stores:     2 * (32 + 63*32)            = 4096 vectors
LXLU reloads:    2 * 63 * 4 * 8              = 4032 vectors
PE recurrent:    504 groups * 8               = 4032 vectors
```

The smallest discriminating arm removes only the explicit recurrent
LXSU-to-LXLU rendezvous at
`tmp/bmm_output_stationary.k32_pe_lx_carry.ddl` lines 134-141:

```ddl
ddl.if(%not_first_prefilled_w_block) {
  ddl.sync {units=["lxsu"], is_receive=false,
            signal_name="c-carry-lxsu-lxlu-sync",
            separate_corelets=true}
  ddl.sync {units=["lxlu"], is_receive=true,
            signal_name="c-carry-lxsu-lxlu-sync",
            separate_corelets=true}
}
```

Compile it as a new artifact; do not mutate the hash-pinned candidate. The
emitted fence-off gate is:

- remove LXLU sync tag 34 from both LXLUs on every core;
- remove LXSU sync tag 17 from both LXSUs on every core;
- preserve ordinary feed/completion tags 7 and 11;
- preserve every PT, PE, SFP, LXLU-load, LXSU-store instruction and loop count
  byte-for-byte otherwise.

Interpretation:

- completion means the explicit rendezvous causes the deadlock; a later
  numerical failure would then mean the fence was also providing necessary
  ordering;
- another deadlock rules the explicit fence out. The next priority is a
  completion-only control that replaces the eight named row-to-PE transfers
  with one explicitly serialized PE input. If the compiler cannot express
  that transaction, that missing row-local/serialized egress is the feature
  to implement; adding more syncs is not the answer.

An optional intermediate zero-feedback arm can remove the LX-to-PE reload and
use zero as the recurrent PE addend while retaining every forward write. It
separates feedback from the forward PE/SFP/LXSU path, but it does not clear the
eight-producer `pe_ptsum` topology and therefore cannot be the final fix.

Do not try K64: it exceeds the proven 32-entry implicit-L0/XRF epoch and
changes more than one variable.

## Acceptance ladder

Advance only in this order:

1. **Stock control**
   - normal frontend path;
   - full CPU comparison;
   - ring features disabled;
   - current stack identity recorded.
2. **Candidate completion**
   - zero inputs;
   - no D2H;
   - no profiler;
   - hard timeout;
   - must write a passing `result.json`.
3. **Candidate correctness**
   - positive;
   - poison A;
   - poison W;
   - cancellation-heavy full-K accumulation;
   - restore;
   - all 262,144 outputs checked for every case.
4. **RIU integration**
   - same compute schedule;
   - A starts in LX;
   - emitted attached route proves no standalone shuffle/HBM fallback;
   - same-grid ring-off control retained.
5. **SFP integration**
   - adjacent-pair W sharing;
   - both directions used when payload permits;
   - route/link accounting proves no redundant transit.
6. **Timing**
   - 10 warmups and 30 exact Kineto `cat=="kernel"` events;
   - serialized T-C-T-C or ABBA bracket;
   - incumbent, same-grid ring-off, RIU-only, and full dual-fabric arms;
   - no compile, materialization, or host wall time substituted for device
     kernel latency.

## Continuation commands

The candidate is already known to deadlock. Do not rerun it unchanged merely
to reproduce the same timeout. After changing exactly one scheduling variable,
recompile into a new path, update both hash-pinned manifests, and run completion
first.

The staged harness paths are:

```text
/tmp/k32_pe_lx_carry_validation_20260729/prebenchmark_m2n16_os/diagnostic_launch_only.py
/tmp/k32_pe_lx_carry_validation_20260729/prebenchmark_m2n16_os/direct_launch_matmul.py
/tmp/k32_pe_lx_carry_validation_20260729/stationary_weight_matmul_probe.py
```

The known-good runtime environment is:

```bash
export PYTHONPATH=/home/adnan/codex-isolated/ring_matmul_true_os_torch_20260722_v1:/home/adnan/spyre-perf-suite-envfix
export LD_LIBRARY_PATH=/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/spyre-comms/lib:/opt/ibm/spyre/deeptools/lib:/opt/ibm/spyre/senlib/lib:/home/adnan/dt-inductor/.venv/lib/python3.12/site-packages/torch/lib
export SENARCH=rcudd1a
export DXP_LX_FRAC_AVAIL=0.2
export DXP_BACKEND_LX_FRAC_AVAIL=0.2
export DT_OPT=psum=dataring
```

For a new candidate, use a fresh run directory and:

```bash
timeout --signal=TERM --kill-after=10s 90s \
  /home/adnan/dt-inductor/.venv/bin/python3 -u \
  /tmp/k32_pe_lx_carry_validation_20260729/prebenchmark_m2n16_os/diagnostic_launch_only.py \
  --manifest /path/to/new_completion_manifest.json \
  --run-dir /tmp/new_completion_run
```

Only if that returns zero and writes `status=pass`, run the full five-case
gate:

```bash
timeout --signal=TERM --kill-after=10s 300s \
  /home/adnan/dt-inductor/.venv/bin/python3 -u \
  /tmp/k32_pe_lx_carry_validation_20260729/prebenchmark_m2n16_os/direct_launch_matmul.py \
  --contract /path/to/new_direct_launch_contract.json \
  --code-dir /path/to/new_compiled_artifact \
  --run-dir /tmp/new_full_correctness_run
```

Do not pass `--case`, `--timing`, or `--discover-kernel-events` for the full
correctness gate.

## Runtime-path pitfall

The pod's inherited default `LD_LIBRARY_PATH` currently puts:

```text
/home/adnan/dt-inductor/sentient/runtime/lib
```

ahead of the installed `/opt/ibm/spyre` libraries. Under that inherited order,
the newer default dirty Torch-Spyre checkout at `cf67411d` fails during import
because `libspyre_comms.so.1` cannot resolve:

```text
flex::destroyP2PRdmaWaitParams
```

Do not interpret that import failure as a matmul failure. All accepted device
evidence in this handoff uses the explicit known-good `/opt/ibm/spyre` library
order above. The default `cf67411d` stack still needs a separately pinned
runtime/spyre-comms compatibility check before it is used for candidate
attribution.

## Stop conditions

Stop and reassess rather than adding complexity if either condition holds:

1. the smallest valid output-stationary control cannot complete after a
   bounded fence/carry bisection;
2. the completing/correct HMI-local control is already slower than the
   identical-grid stock compute control by more than the maximum modeled RIU
   materialization saving.

Conversely, if the HMI-local control completes and is competitive, the next
high-value step is not another PT micro-optimization. It is attaching RIU A
delivery to the same schedule, then adding SFP pair sharing and measuring the
two fabrics as one overlapped algorithm.

## 2026-07-30 output-free A/L0/PT discriminator

This is the latest completion boundary and supersedes the earlier proposal to
separate PT forwarding with a balanced synthetic output. A narrower valid
control was compiled and run.

### Why the control is valid

Deleting the result transaction in source DDL is not a valid discriminator in
the current compiler. DDC uses the ordinary output transaction to bind tensor
data-connects and preserve datastage row continuity. Making the result absent
in both HBM and LX first exposed several legitimate `isPresent()` bugs and then
failed those DDC invariants before code generation.

The successful diagnostic keeps the ordinary output transaction through DDC,
then removes its service operations before DCG/DCC. The local-only hook is
enabled by:

```bash
export DXP_DIAGNOSTIC_PRUNE_OUTPUT_STORE=1
```

It removes only:

```text
transfer_lds2_src:lx_dst:hbm
sync_send_lxsu_to_l3su
sync_receive_l3su_from_lxsu
sync_send_l3su_to_lxsu
sync_receive_lxsu_from_l3su
```

This hook is a diagnostic instrument, not proposed production functionality.
It lives in the detached Deeptools tree:

```text
/home/adnan/codex-isolated/ring_matmul_b35_coordinator_20260729/deeptools
base commit: b35cece729c0ae1707ea71bf5a0d3c4451358a07
```

That tree contains unrelated pre-existing changes. Reapply only the named
patches below; do not transfer its full diff.

### Structural proof

The K32 and K16 controls both retain the A LX-to-L0 stream and 1,024 PT FMA
instructions. K16 expresses the same K32 work as two K16 epochs. Both satisfy
all of these emitted-code gates across the 32 cores:

| Gate | K32 | K16 |
|---|---:|---:|
| `PTOP_FMA` | 1,024 | 1,024 |
| `L0_SYNC tilesize=32` | 128 | 0 |
| `L0_SYNC tilesize=16` | 0 | 128 |
| `PTOP_XRFACCESS` | 0 | 0 |
| `tgts=result` | 0 | 0 |
| PE payload | 0 | 0 |
| SFP payload | 0 | 0 |
| `LX_ST` | 0 | 0 |
| `L3_STMU` | 0 | 0 |
| `L3_SYNC synctag=163` | 0 | 0 |

Therefore neither artifact has a stranded external output consumer. W/XRF,
PT-result forwarding, PE/SFP result transfer, LX output service, and HBM
output service are all absent.

### Device result

Each arm was launched as a zero-input, no-D2H, untimed completion probe with a
hard 90-second timeout. Both reached `device_synchronize`, reported possible
lost completion after 60 seconds, and were terminated at the outer timeout.
The producer-only recovery control passed immediately after each arm through
`kernel_and_device_synchronize`.

| Arm | Completion | Immediate recovery |
|---|---|---|
| A-only/output-free K32 | timeout | pass |
| A-only/output-free K16 | timeout | pass |

No numerical correctness or performance result exists for either arm.

Shared-PVC artifacts and evidence:

```text
/home/adnan/codex-isolated/os_hmi2_focused_main_20260730/\
k32_a_only_internal_output_v16
/home/adnan/codex-isolated/os_hmi2_focused_main_20260730/\
k32_a_only_internal_output_completion_manifest_20260730.json
/home/adnan/codex-isolated/os_hmi2_focused_main_20260730/\
k32_a_only_internal_output_clc_20260730_v1.log
/home/adnan/codex-isolated/os_hmi2_focused_main_20260730/\
post_a_only_internal_output_recovery_20260730_v1/result.json

/home/adnan/codex-isolated/os_hmi2_focused_main_20260730/\
k16_a_only_internal_output_v17
/home/adnan/codex-isolated/os_hmi2_focused_main_20260730/\
k16_a_only_internal_output_completion_manifest_20260730.json
/home/adnan/codex-isolated/os_hmi2_focused_main_20260730/\
k16_a_only_internal_output_clc_20260730_v1.log
/home/adnan/codex-isolated/os_hmi2_focused_main_20260730/\
post_k16_a_only_internal_output_recovery_20260730_v1/result.json
```

Compile roots:

```text
/tmp/os_hmi2_k32_a_only_postddc_pruned_20260730_v16
/tmp/os_hmi2_k16_a_only_postddc_pruned_20260730_v17
```

Artifact hashes:

| Artifact | K32 SHA-256 | K16 SHA-256 |
|---|---|---|
| DDL | `09c15850162b1a4ed39ba016811bf7862f40f00f9209cdcd07e192f3d8fa4d43` | `84f1677238405ed25973a69d3d79ed4de99e9972be684bd4a180912788e58c3f` |
| bundle | `8f096dd0da33e3286ac26678488b686575b9cda4626e47aea556ed15c69ce6dd` | `8f096dd0da33e3286ac26678488b686575b9cda4626e47aea556ed15c69ce6dd` |
| `spyrecode.json` | `caf89f44395f02e8a8d8f7e166e637859dd097776d23fe21e3f4d3340e759338` | `81fd90cb283d0e1386bb382d7f5f054a6af7cb56f16657566195ded1ea0cc236` |
| `init_binary.bin` | `2540c3513007472aebe673303cf33fc068fb225770e550c3424227e56c29e14f` | `edef4d99a7580478a5af967ce276f0f65f1e41a3f6db370d23155c838c2704bc` |
| SMC | `252c451d96af3e1a25bdf56b747b40fca074e1813776759899cc7cff28aaa33f` | `042b037a891ddf03868d69ae1e2d7a50d4153bf59e00faa87606db4e62bcfaba` |
| timeout log | `e5d3d0650b84392d22fce0ddf137024c34e77d427e4a2f0c16303468a89d9739` | `3ad0e552ac892f0e9d7088164b23ca74cb115c168bc1f1e539fe2333cf5da2a5` |
| manifest | `bbc0de8d9c3f414ae585a480f7f52d273e1611d1ad1918b54bbffcc516ea82d9` | `58c9e0a9cc80b7d3d6a932983f0baedbfecf1f84044dea92516bbe5f04dd2767` |

Both recovery result files hash to:

```text
02f53185a5623d9d8f8e9e1551453c11cfda4c2159c40c81d8a5519b148a8d33
```

### Local diagnostic patches

These files are on the Mac under this repository's untracked `tmp/`
directory. None is committed as production code.

| Patch | Purpose | SHA-256 |
|---|---|---|
| `tmp/deeptools_lx_only_corelet_split.patch` | honor HBM presence during corelet split paging | `ae4144cfd73106b8dfc177358fb41e76d7c91b25c3a6d4ae67cc482e37f6bfe3` |
| `tmp/deeptools_lx_only_l3_paging.patch` | honor HBM presence in L3 paging | `5314df3577dc80676c21f809cf95f7c0b56b9f23d5df3b2dce1b7609d1d0472e` |
| `tmp/deeptools_lx_presence_pin.patch` | require present LX memory before treating a tensor as LX-pinned | `757512933949ae43ad42da16daabf9865c8ee815a269d438844182c854cf0771` |
| `tmp/deeptools_absent_lx_chunk_validation.patch` | skip LX chunk validation for absent/unallocated LX memory | `6a5dacdc6a27b048f972a1085dfb70b5476a2d4b04f09eda455abaddf1edf3cd` |
| `tmp/deeptools_dataconnect_diagnostic.patch` | name the consumer in missing-producer diagnostics | `db888a22101eb9cf9ec6587560d31bb9284422d3909e644e902412967b6565a2` |
| `tmp/deeptools_postddc_output_prune.patch` | remove output service after DDC for this discriminator | `06a750e94cd294c09d71920e973ffb91161a8126e6de840d5b59ef58a1c666ee` |
| `tmp/k32_direct_egress_a_only.patch` | remove W/XRF and result forwarding | `1b6a68d782a364da327eee7a14fa2e06b16c8ecfa9d5bf72b1f3a03e8709141c` |
| `tmp/k32_noegress_k16_tile.patch` | express the matched control with K16 epochs | `b3d896cc9e38099967bdf64ae8dbbff1bacbe833c28829a9ab338ddd7eaf40c4` |

### Revised blocker and next bounded pass

The blocker is now upstream of result handoff:

```text
A LX -> L0 implicit-sync -> PT execution
```

The evidence rules out W/XRF, output forwarding, both output fabrics, and K32
maximum epoch depth as necessary causes. It does not yet distinguish an
A/L0 implicit-sync protocol error from a PT compute/bundle relationship error.

The shortest next pass is deliberately bounded:

1. compile and run matched K8, K4, and K2 A-only/output-free controls;
2. if K2 also loses completion, stop depth tuning;
3. compare the same LX/L0 stream against a known-good stock PT
   micro-schedule;
4. run a matched drain control without recurrent PT FMA.

Do not add RIU/SFP ring traffic, run correctness, or begin timing until a
completion control passes. This pass localized the compiler/runtime blocker;
it did not demonstrate an algorithmic speedup.

## 2026-07-30: activation-stationary decode matmul side experiment

### Scope and decision

The ring-native output-stationary investigation above is paused, not replaced.
This bounded side experiment evaluates a different PT dataflow for Granite
decode matmuls whose logical `M` is small and whose physical PT execution is
padded to `M=64`.

For a source linear with `A[M,K]`, `W[N,K]`, and `C=A@W.T`, the candidate is:

```python
padded_a = torch.nn.functional.pad(A, (0, 0, 0, 64 - M))
C = (W @ padded_a.T).T[:M]
```

The algebra asks the existing weight-stationary BMM substrate to cast the
tensors into different hardware roles:

| PT role | Incumbent `A @ W.T` | Candidate `(W @ A.T).T` |
|---|---|---|
| West-to-East streamed input | `A` | `W` |
| XRF-stationary kernel | `W` | `A.T` |
| North-to-South reduction/output | `C` | `C.T` |

The candidate freezes the same 32-way ownership of the original `N` dimension.
It does not use a different work division to manufacture a win, and it does not
use LX-to-LX communication. This is intentionally a PT/HMI dataflow experiment
while ring-native work is paused.

### First-principles corrections

The original design note's exact-fit statement needs the physical decode
padding:

```text
M8  x K1024 x FP16 = 16 KiB per corelet, not 128 KiB
M64 x K1024 x FP16 = 128 KiB per corelet, exactly one corelet XRF
```

For `M1 x K4096 x N4096`:

- logical `A`: 8 KiB;
- physical padded `A`: 512 KiB across four K1024 corelet tiles;
- each corelet's stationary A tile: exactly 128 KiB;
- `W`: 32 MiB;
- output: 8 KiB logical;
- unique weight HBM bytes: 32 MiB in both incumbent and candidate.

Therefore this design does **not** reduce the incumbent's unique HBM weight
traffic. Its credible lever is narrower: the large weight becomes the ordinary
PT West input stream, while the small padded activation is block-loaded into
XRF. If this wins, it should be because W delivery and PT execution overlap
better than repeated W XRF block-load phases. The HMI weight floor remains the
same in both arms. No multicast or bandwidth-reduction claim is made for a
single N-sharded decode matmul.

Design B from the same note is the ordinary weight-stationary dataflow already
used by the incumbent. It is a useful architecture explanation, but it is not a
new algorithm by itself.

### Device result: correct physical-M64 realization

The exact logical Granite decode control `M1 x K4096 x N4096` passed on
`adnan-spyre-dev-pf` after explicit zero-padding to physical M64.

The emitted BMM descriptor is:

```text
logical BMM: mb=4096, in=4096, out=64
work slices: mb=32, in=1, out=1

INPUT  layout=[mb,in],  stick=[in]   # W is the West stream
KERNEL layout=[in,out], stick=[out]  # padded A.T is the XRF kernel
OUTPUT layout=[mb,out], stick=[out]  # transposed C ownership
```

The complete root inventory is:

```text
identity
identity
ReStickifyOpHBM
batchmatmul
```

The two identity roots create the 63 zero rows and copy the logical row into
the physical M64 activation. The restickify is the small activation transpose;
there is no standalone W shuffle. Original `N=4096` is `mb` in the transposed
BMM and is owned 32 ways.

All fail-closed correctness gates passed:

| Profile | allclose | finite | max absolute error |
|---|---:|---:|---:|
| positive | yes | yes | 0.015625 |
| poison activation | yes | yes | 0.0234375 |
| poison weight | yes | yes | 0.015625 |

Both poison profiles changed the device output. The final probe status is
`pass`.

No Kineto timing and no Granite end-to-end run were performed. This section
makes no latency or speedup claim.

### Compiler substrate added

Explicit padding exposed one narrow Torch-Spyre gap. Restickify insertion could
resolve graph inputs and FX-backed views, but raised `StopIteration` when the
required input was a compiler-created pointwise buffer with no FX origin.

The local patch extends `_create_restickify_node` as follows:

1. retain the existing exact FX/TensorBox lookup;
2. use an FX origin when a realized producer has one;
3. only when a compiler-created buffer has no FX value, call the same registered
   restickify lowering directly with a `TensorBox(StorageBox(buffer))`.

The targeted regression:

```text
tests/inductor/test_restickify.py::
test_matmul_with_padded_computed_kernel_restickify
```

passes on `adnan-spyre-current-pf`:

```text
1 passed in 13.15s
```

This is the only production-code change in this side experiment. It is not yet
committed as production code because the candidate has not been timed.

### Source/binary compatibility finding

The older coordinator Deeptools source at:

```text
/home/adnan/codex-isolated/ring_matmul_b35_coordinator_20260729/deeptools
```

contains a unary DDL template that binds optional `shuffle`, but the installed
`/opt/ibm/spyre/deeptools/bin/dxp_standalone` registry does not expose that
operation. The padded graph therefore aborted before device execution.

The exact generated bundle compiles successfully with the existing matched
current Deeptools pair:

```text
source:
/home/adnan/codex-isolated/deeptools-master-e3944781
HEAD e3944781cb25b76abeb9b3e87c1f5c5879e84229

binary:
/home/adnan/codex-isolated/deeptools-master-e3944781-build/dxp/dxp_standalone
SHA-256 c22185db3cbfa071dd8d261343e71a5c09543eae83c9d19e86e90d729e43a131
```

No redundant shuffle implementation was added.

### Durable paths and hashes

Mac source paths:

```text
ring_compute_prototypes/activation_stationary_decode/design_a_model.py
ring_compute_prototypes/activation_stationary_decode/test_design_a_model.py
ring_compute_prototypes/activation_stationary_decode/probe.py
torch_spyre/_inductor/insert_restickify.py
tests/inductor/test_restickify.py
```

| File | SHA-256 |
|---|---|
| `design_a_model.py` | `c37c1bd7279e345151f1b4d89bce40c6e1ccb814ddbd568c5e8f1f6736c1800f` |
| `test_design_a_model.py` | `234d61e11ea1b71afc58641be8626fb00e4fa626ff6705fc520275898624e921` |
| `probe.py` | `fd389ce033c2e05755641cb9faa99e5f1f469a95584e6cfd2674e267c42404fa` |
| `insert_restickify.py` | `11da63c12fd519fed0eeb14d6be6225260aa071e3af1270523264d480dab6a4f` |
| `test_restickify.py` | `57f509259df2cc2f161c9b4cadf4eeeaaa6b0244d2962ea37f34d9247d71dddf` |

Shared-PVC prototype root:

```text
/home/adnan/codex-isolated/activation_stationary_decode_20260730_v1
```

Passing run:

```text
/home/adnan/codex-isolated/activation_stationary_decode_20260730_v1/\
runs/m1k4096n4096_activation_stationary_padded64_v5
```

Passing summary SHA-256:

```text
fa2635dd94a5e9afed3e0f2671a96a8f7f8a7ddb51992ce0c8da0739d5dd2098
```

Bundle SHA-256:

```text
a3697974bc3b2ba20b1048d9a7cb78ff9afd8d226a3c04ca2db67dafc5225830
```

Matched DXP replay:

```text
/home/adnan/codex-isolated/activation_stationary_decode_20260730_v1/\
redxp/m1_padded64_master_e3944781_v1
```

The Torch-Spyre device checkout is:

```text
/home/adnan/codex-isolated/ring_matmul_true_os_torch_20260722_v1
HEAD 80701411a151fa6402d08ce7586f671883e1e66b
```

It remains dirty with the earlier ring-native work plus this experiment. The
compiler and test files above were copied only after verifying that their pod
versions matched the local HEAD versions exactly.

Useful failed/discovery runs are preserved:

```text
runs/m1k4096n4096_activation_stationary_v3
  singleton M folded out; DXP reuse-dimension assertion

runs/m8k4096n4096_activation_stationary_v2
  implicit M64 descriptor but non-zero/undefined padded lanes; incorrect

runs/m64k4096n4096_activation_stationary_v2
  direct physical-M64 candidate; device-correct and exact role mapping

runs/m1k4096n4096_activation_stationary_padded64_v1
  original restickify FX lookup StopIteration

runs/m1k4096n4096_activation_stationary_padded64_v2
  first computed-origin fallback still had no FX origin

runs/m1k4096n4096_activation_stationary_padded64_v3
  Torch-Spyre compile succeeded; old Deeptools source/binary mismatch

runs/m1k4096n4096_activation_stationary_padded64_v4
  device-correct; probe inventory gate was stale

runs/m1k4096n4096_activation_stationary_padded64_v5
  final passing correctness and structural control
```

### Next bounded pass

The shortest honest next step is timing, not more compiler work:

1. add a serialized fresh-process `incumbent-candidate-candidate-incumbent`
   Kineto bracket for `M1 x K4096 x N4096`;
2. require exact `cat == "kernel"` device events and include the entire paid
   four-root candidate bundle;
3. compare against a freshly compiled same-environment incumbent using the
   same N32 ownership;
4. only if the candidate wins, remove or fuse the two padding identities and
   activation restickify;
5. then repeat correctness/timing for Granite `N=1024`, `N=12800`, and
   `K=12800 -> N=4096` decode linears before graph integration.

The decision gate is strict: if the paid M64 candidate does not beat the
incumbent despite its extra roots, stop this path. If it wins, the first
production optimization is producer-native physical-M64 activation layout so
the padding and transpose do not materialize in HBM.

## 2026-07-30: Design A implemented and paid across Granite decode linears

### Outcome

The decision gate above passed decisively. Design A is now an opt-in
production-facing Torch-Spyre decomposition, not only a manually written
prototype. It is correct and faster on all four isolated Granite decode
linears tested.

The selector is intentionally default-off:

```text
SPYRE_MATMUL_DATAFLOW=weight_stationary      # existing default
SPYRE_MATMUL_DATAFLOW=activation_stationary  # Design A
```

No new backend operation, raw send/receive primitive, or special work-division
hint was added. The implementation reuses:

- the existing `aten.linear` decomposition point;
- ordinary `pad`, `matmul`, transpose, reshape, and slice operations;
- the existing BMM backend and ordinary work-division planner;
- the existing restickify lowering, with one narrow computed-producer fix.

For eligible FP16 linears, the selector:

1. flattens the logical leading dimensions into `M`;
2. requires `1 <= M <= 64`, 2D `W[N,K]`, matching K, and stick-aligned K/N;
3. zero-pads the activation to physical M64;
4. lowers `A @ W.T` as `(W @ padded_A.T).T[:M]`;
5. restores the original leading dimensions and adds bias when present.

Unsupported shapes retain the existing weight-stationary decomposition. An
unknown selector value fails closed with `Unsupported`.

### Paid matched device timing

Each row below is a fresh same-process matched comparison with:

- inputs already on the Spyre device;
- the ordinary planner for both arms (`--work-division auto`);
- 10 serialized warmup blocks;
- 30 `incumbent-candidate-candidate-incumbent` timing blocks;
- 60 exact Kineto `cat == "kernel"` events per arm;
- compile, host-to-device copy, and host wall time excluded;
- all candidate padding, identities, activation restickify, BMM, and output
  slicing included in the device event.

| Logical linear | Incumbent median | Design A median | Incumbent / Design A | Paired-block median |
|---|---:|---:|---:|---:|
| `M1 K4096 N1024` | 203.471 us | 86.2645 us | **2.3587x** | 2.4040x |
| `M1 K4096 N4096` | 821.6505 us | 237.5025 us | **3.4595x** | 3.4566x |
| `M1 K4096 N12800` | 2710.962 us | 771.686 us | **3.5130x** | 3.5093x |
| `M1 K12800 N4096` | 2999.917 us | 728.777 us | **4.1164x** | 4.1111x |

All four summaries have `status=pass`, exact output shape, finite output,
CPU-reference allclose, candidate/incumbent allclose, exact event count, and
the expected emitted root inventories.

Candidate maximum absolute error against the FP16 CPU reference was:

| Logical linear | Maximum absolute error |
|---|---:|
| `M1 K4096 N1024` | 0.013671875 |
| `M1 K4096 N4096` | 0.015625 |
| `M1 K4096 N12800` | 0.017578125 |
| `M1 K12800 N4096` | 0.0546875 |

The primary speedup is the ratio of the two arm medians. The paired-block
median is reported separately as a drift check; it agrees closely in every
case.

### Structural attribution

Every incumbent bundle has:

```text
ReStickifyOpHBM
batchmatmul
```

Every Design A bundle has:

```text
identity
identity
ReStickifyOpHBM
batchmatmul
```

Thus the candidate wins while paying for its explicit physical-M64
construction and activation restickify. This is not a hidden free-padding
oracle.

For `M1 K4096 N4096`, the automatic-selector candidate emits the exact same
bundle SHA-256 as the earlier manually written fixed-N32 candidate:

```text
a3697974bc3b2ba20b1048d9a7cb78ff9afd8d226a3c04ca2db67dafc5225830
```

That establishes that the production selector realizes the intended
activation-stationary dataflow rather than finding an unrelated graph. The
original N dimension becomes BMM `mb`, and the ordinary planner distributes it
over the 32-core budget.

The most defensible mechanism remains the PT tensor-role change:

- incumbent: small A is the West input; large W is repeatedly block-loaded as
  the XRF kernel;
- Design A: large W is the West input stream; padded A.T is the small
  XRF-stationary kernel.

Unique weight HBM bytes are unchanged. The large measured win is consistent
with replacing repeated W kernel-load phases by the ordinary streaming input
path while keeping PT fed. This pass does not yet contain counters that split
the gain into HMI service, XRF load stalls, and PT occupancy, so that causal
breakdown remains a strong architectural explanation rather than a measured
counter decomposition.

This side experiment does not use RIU, SFP, multicast, or LX-to-LX. It is not
evidence for the paused ring-native algorithm. It is a separate PT/HMI
dataflow win.

### Compiler changes and focused tests

Local and shared-PVC implementation files:

```text
torch_spyre/_inductor/config.py
torch_spyre/_inductor/decompositions.py
torch_spyre/_inductor/insert_restickify.py
tests/inductor/test_activation_stationary_linear.py
tests/inductor/test_restickify.py
ring_compute_prototypes/activation_stationary_decode/benchmark_abba.py
```

The new decomposition tests cover:

- 2D and 3D activations;
- logical M1, M8, and M64;
- bias and no-bias;
- fallback above physical M64;
- unchanged default behavior;
- invalid-selector rejection.

The computed-pad restickify regression exercises the actual
pad-transpose-BMM-slice path. The combined focused run on
`adnan-spyre-current-pf` passed:

```text
14 passed
```

Branch source hashes:

| File | SHA-256 |
|---|---|
| `torch_spyre/_inductor/config.py` | `9f8a2fdc131a2f5fe5c847e2108893e4b3279e48551db2d0b723b0a28756450c` |
| `torch_spyre/_inductor/decompositions.py` | `e7b455c5e517ee605524fc8b019875318345294b2d02f1eb6550af4e662804e1` |
| `torch_spyre/_inductor/insert_restickify.py` | `f60699513028b79d3192f3cef66341af27c442a288aba9c3a99660488157d7bb` |
| `tests/inductor/test_activation_stationary_linear.py` | `a1669874c5be6e1d43a7b09609f44050ee2723f1de0da927eaea511a4c890e4c` |
| `tests/inductor/test_restickify.py` | `57f509259df2cc2f161c9b4cadf4eeeaaa6b0244d2962ea37f34d9247d71dddf` |
| `benchmark_abba.py` | `1a2a8ae03cb8563b1e57aa9438563aeae352bd2d405ae41cde4c12b70aeee3ec` |

The accepted measurements used benchmark SHA-256
`b0ddf93dc610af4da33cb23afa4f032ece8ff2e82364bab7c98d5fd45eae085c`.
The branch version changes only its descriptive module docstring. The
decomposition and focused test hashes above match the measured checkout.
`config.py` and `insert_restickify.py` have different whole-file hashes because
the measured dirty checkout contained unrelated ring-prototype configuration
and pass changes; the Design A hunks are the same narrow implementation.

### Durable timing evidence

All accepted timing summaries are under:

```text
/home/adnan/codex-isolated/activation_stationary_decode_20260730_v1/timing
```

| Run directory | Summary SHA-256 | Trace SHA-256 |
|---|---|---|
| `m1k4096n1024_selector_auto_full_v1` | `9fd357e5f6442efbb17cda74780687c7ac71cbffede6a22711a54a4d53c3619c` | `7ed1b4929729ee18f3326935dc03a5ed2f1b263faf4ba2d381b0dae8e1baaeb7` |
| `m1k4096n4096_selector_auto_full_v1` | `fb7dce76443d8965a6707cdc6f30dbe2002d76c7d633e4fd6741a3fc7911a0e7` | `e2813d3bb6b0aee2b787b69907c1a020a2370610b84ea446840c00f26d73f1f6` |
| `m1k4096n12800_selector_auto_full_v1` | `7cf249849b6af5cb33b956d17b059ccc4d847f6109e2df67e9103afdd294d011` | `0404c82dd53d932b74b8ad8f4501ab16dfb4fa62c4c448ae78d348a3840dc526` |
| `m1k12800n4096_selector_auto_full_v1` | `3fec195fbc357473f065b5672eb15f4e60412b36c3ce10a0126ec356b610a5d7` | `af21423a5bc10e1cec817168767363533fa46c3dbbf84e07f7a2903a8797a9e0` |

Candidate bundle SHA-256 values:

| Logical linear | Candidate bundle SHA-256 |
|---|---|
| `M1 K4096 N1024` | `960df89fb145fb34b82125b639cc4bc9c6e84c5aeac97b3270b30f8a2edd8545` |
| `M1 K4096 N4096` | `a3697974bc3b2ba20b1048d9a7cb78ff9afd8d226a3c04ca2db67dafc5225830` |
| `M1 K4096 N12800` | `5c8eab2bc8d22b31249cf687b6eebcf17250f88b22660a28dfea13798f63088d` |
| `M1 K12800 N4096` | `1759af727f2a467e706e0f8c13cfd9482dab47b872691200f5740580b1fde00a` |

Pinned stack:

```text
Torch-Spyre:
  /home/adnan/codex-isolated/ring_matmul_true_os_torch_20260722_v1
  HEAD 80701411a151fa6402d08ce7586f671883e1e66b

Kineto Python:
  /home/adnan/spyre-envs/main-ac3c7395/kineto-venv/bin/python
  torch 2.11.0+aiu.kineto.1.1.2

Torch-Spyre extension:
  /home/adnan/spyre-envs/main-ac3c7395/torch-spyre/torch_spyre/_C.so
  SHA-256 b681beeb640d5b8524dadfcc787d9ad2725db357adcf4e75f707d92d907487d4

Deeptools:
  /home/adnan/codex-isolated/deeptools-master-e3944781
  HEAD e3944781cb25b76abeb9b3e87c1f5c5879e84229
  /home/adnan/codex-isolated/deeptools-master-e3944781-build/dxp/dxp_standalone
```

The Torch-Spyre checkout remains dirty with the earlier ring-native work plus
this experiment. The summaries record the full tracked status. Nothing in
this section claims that the implementation is isolated on a clean production
branch.

### Current boundary and next step

This is a strong isolated linear-kernel result, not a Granite end-to-end
result. In particular:

- no Granite graph has selected the new dataflow yet;
- the activation is ready on-device at the timing boundary, but the current
  graph still constructs/re-stickifies the physical-M64 kernel through HBM;
- token correctness, model correctness, and end-to-end decode latency are
  unmeasured;
- the implementation, focused tests, model, probe, and matched timing harness
  are tracked on `ah/communication-cost-model`;
- the matched Granite/SenDNN E2E runbooks remain in
  `Adnan-Hoque1/spyre-granite-e2e-bench` branch
  `adnan/sendnn-granite-antoni-repro-20260725`.

The shortest next step is no longer more microkernel tuning. It is a controlled
Granite integration:

1. enable `activation_stationary` only for the four eligible decode linears;
2. run token/model correctness first;
3. capture a fresh matched end-to-end decode bracket;
4. only after that, make the producer emit a physical-M64 activation kernel
   directly in LX to remove the two identities and HBM restickify.

Because the paid implementation already wins by 2.36x to 4.12x in isolation,
producer-native LX is an optimization opportunity, not a prerequisite for
proving Design A.

## 2026-07-31: Granite E2E integration result

The first full 40-layer Granite runs materially revise the conclusion above.
The isolated projection win is real, but it does not survive the current
fused decoder-layer schedule.

### Exact tested source and rollout control

```text
Torch-Spyre revision:
  2c1ab140d23f7e200ef45ef26b057229cb393727

selector:
  SPYRE_MATMUL_DATAFLOW=activation_stationary

validated scope:
  SPYRE_ACTIVATION_STATIONARY_SHAPES=12800x4096
```

Revision `2c1ab140` adds a comma-separated exact `KxN` rollout allowlist. It
does not change either matmul implementation or work division. An empty
allowlist preserves the original all-eligible behavior. The focused
decomposition and computed-pad device suite passes:

```text
16 passed in 8.23s
```

### Down-projection-only completion and correctness

Restricting Design A to the Granite MLP down projection
`M1 K12800 N4096` completed prefill and all three decode calls for the full
40-layer B1/S512 model. The same-revision stock and candidate decoded output
artifacts are byte-identical:

```text
SHA-256:
  29e7f26fed11c801d98f5a04ea00afe62f91471d9404372c66815dbf16df7888
```

One-generation Kineto device timing was:

| Device phase | Same-revision stock | Down-only Design A | Candidate change |
|---|---:|---:|---:|
| Prefill | 378.098 ms | 375.609 ms | -0.66% |
| Decode average | 161.825 ms | 226.120 ms | +39.73% |

Both traces contain exactly four phases and 40 decoder layers per phase.
Steady decode remains one fused device kernel per layer; its representative
duration increases from about `3.93-3.96 ms` to `5.49-5.50 ms`. Therefore the
regression is inside the compiled fused layer, not extra host launches.

This is a one-generation completion measurement, not a promoted stable
latency. Its magnitude is nevertheless too large to dismiss as timing noise.
The accepted isolated `4.1164x` down-projection result is insufficient to
predict fused-layer E2E performance.

Durable device run roots:

```text
stock:
  /home/adnan-cdx/dt-inductor-codex-clean/profiler_runs/\
  latest_cost_model_granite_block_20260724_202708/\
  antoni_exact_repro_20260724/runs/\
  full_40_layer_b1_s512_1x4_stock_completion_b83e5cfd_20260731_v1

down-only Design A:
  /home/adnan-cdx/dt-inductor-codex-clean/profiler_runs/\
  latest_cost_model_granite_block_20260724_202708/\
  antoni_exact_repro_20260724/runs/\
  full_40_layer_b1_s512_1x4_design_a_down_only_2c1ab140_20260731_v1
```

Their trace SHA-256 values are
`ba555f5468696f13c674a5f4ec5ee1e690adb7899fa4cfd8340c58279781f894`
and
`8eaf8dcd534cb5b052a1103b7c1bd154b10f2aaf0b059fb0a50b016fbd12caa4`
respectively.

### All-eligible compiler blocker

With an empty allowlist, prefill completes and produces token `203`. The first
decode compile then fails:

```text
DtException: out_reuse_dim.size() == 1
file /project_src/deeptools/dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp
line 803
```

The failing fused bundle contains the two `K4096 -> N12800` gate/up linears
and the `K12800 -> N4096` down linear. The same revision with
`weight_stationary` completes, isolating the failure to the new dataflow.

Exact failed-run root:

```text
/home/adnan-cdx/dt-inductor-codex-clean/profiler_runs/\
latest_cost_model_granite_block_20260724_202708/\
antoni_exact_repro_20260724/runs/\
full_40_layer_b1_s512_1x4_design_a_completion_b83e5cfd_20260731_v1
```

### Decision

Design A is now a correct isolated microkernel and a correct down-only Granite
experiment, but not an E2E optimization. Do not spend a five-run timing budget
to "prove" a speedup before explaining the fused-layer regression. The next
high-value diagnostic is to compare the stock and down-only emitted decoder
schedule around the BMM, restickify, and output-reuse boundary. Any fix must
preserve the intended PT tensor-role change; inserting an HBM materialization
only to make the graph compile is not an acceptable performance solution.

The exact three-arm E2E launcher, analyzer, evidence, and commands live in:

```text
https://github.ibm.com/Adnan-Hoque1/spyre-granite-e2e-bench
branch adnan/sendnn-granite-antoni-repro-20260725
runbook runbooks/activation_stationary_design_a_e2e.md
```
