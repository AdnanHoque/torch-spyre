# Attention QK^T on-chip all-gather — hand-off + verdict (2026-06-19)

The attention 1st-BMM "incompatible division" (Jamie's slide / the sendnn `LxRelayout`
SDSC): `mul(K,scale)` shards K, the `Q@K^T` BMM needs K broadcast to its cohort. sendnn
does this as an on-chip **multicast `STCDPOpLx` LX-relayout** (all-gather). This documents
the tsp/Inductor reproduction, the shippable wins, and the **definitive root cause** of why
the on-chip gather is not yet value-correct — plus the real fix.

## TL;DR
- **Shippable, independent win:** the deeptools **dxp is fixed on the latest base** (`a1ec02a`)
  — production-worthy, regular SDPA matches CPU eager at 1e-3. This was a master base-drift
  bug, unrelated to the all-gather.
- **All-gather substrate:** authored, P0-validated (dxp accepts the multicast, EBR inert),
  E2E-on-device, geometry/read-path/`@V`/placement all verified. Everything down to the
  cross-bundle boundary is correct.
- **Verdict:** value-correctness is blocked by the **cross-bundle boundary** — `mul(K)` is a
  separate device program, so the QK^T bundle re-reads K from HBM via a ReStickify, and the
  gather operates on that re-materialization (wrong layout model, and it can't avoid the HBM
  read it was meant to eliminate). **This is the same wall the SwiGLU substrate hit.** The
  real fix is **co-bundling `mul(K)` into the QK^T program** — a substantial redesign, not a
  patch.

## 1. The shippable win — dxp on latest deeptools (production-worthy)
The locally-built "patched" dxp computed attention ~0.02-wrong vs CPU. Root cause (device-
isolated): **deeptools master base-drift** — it was built from `6df5ad5140`, but the correct
SDK is `ibm-deeptools:2.0.0-0.main.1+932.a1ec02a` (commit `a1ec02a`, **489 commits newer**),
which carries a softmax/SFP codegen fix the local build predated. Not the gate, not the
templates, not the all-gather (all proven inert/correct by ablation).

`a1ec02a` already has `runDcgForDataOpsDlOps` + the multicast STCDP DCG, so the forward-port
collapsed to a minimal patch: the STCDPOpLx op registration + `stcdp.ddl` (the 9-file delta,
applies clean) **+** the mixed-fold gate hand-ported (`SdscTree.cpp:152` importSdsc relax,
`dxp.cpp` route mixed-fold to `runDcgForDataOpsDlOps`). Built reusing the prebuilt LLVM
(`MANAGE_LLVM=0`, `LLVM_PROJ_BUILD=build/llvm`).
- **Device-verified:** regular SDPA flag-OFF = CPU eager at **1e-3** (was 0.02 on the old base).
- Worktree `/home/adnan/dt-inductor/deeptools-allgather`, branch **`allgather-on-latest`**,
  commit **`4f7da49fc3`**. Binary `build/deeptools-allgather/dxp/dxp_standalone`; run with
  `DEEPTOOLS_PATH=/home/adnan/dt-inductor/deeptools-allgather`.

## 2. The all-gather substrate (swiglu-ws-v2, flag `SPYRE_ONCHIP_ATTN_ALLGATHER`, default off)
| piece | status | commit |
|---|---|---|
| Edge detection (`_broadcast_edge`, the broadcast dual of the reduce edge) | ✅ | `3899472` |
| Multicast relayout authoring (matches the sendnn SDSC byte-for-byte) | ✅ | `d4220c5` |
| P0: dxp accepts the multicast, builds the cohort replicate map, **EBR inert** | ✅ | — |
| E2E: codegens + executes on device | ✅ | — |
| Geometry re-derived from tsp's **committed per-core views** (not the sendnn layout) | ✅ | `596a5c7` |
| Read-path: BMM K-input flipped to the gathered LX, addr matches | ✅ verified | — |
| `@V`: V correctly stays on the HBM broadcast (not gathered) | ✅ verified | — |
| Placement: band-scope the K-input `maxDimSizes_` (clears the dxp `LX_MODLRFIMM` abort) + per-cohort-local dest | ✅ (dxp rc=0, write==read) | `05df9db` |

**tsp's actual division** (device-verified, NOT the sendnn head-grouped layout): producer
`mul(K)` splits Lk 32-way (head unsplit); consumer QK^T splits `{Lq:4, Lk:8}` (head unsplit),
so cohort `{4j..4j+3}` shares Lk-band `j`. The geometry, dest pieces, and K-input flip were
all re-derived to match this.

## 3. The verdict — cross-bundle boundary (the root cause)
Five debug layers were peeled, each fix locally correct, none reaching value-correctness:
1. dxp base-drift → fixed (§1).
2. Geometry (sendnn-hardcoded → tsp-view-derived) → fixed.
3. `lxSize_` convention → **no-op** (bit-identical; the DCG packer ignores it).
4. Placement: K-input `maxDimSizes_` unbounded → dxp abort; absolute dest coords vs band-scoped
   read → fixed (dxp rc=0, write==read all 8 bands) — but device value still ~0.02.
5. **Cross-bundle source provenance** — the conclusive layer.

**The scoped probe** (`/tmp/v2_restickify_layout.py`, device-free) settled it: the gather's
*actual* in-bundle source is the **ReStickify-of-K**, which shards on **`x` 32-way with
`out`(Lk) NOT split** (`numWkSlicesPerDim_={mb:1, x:32, out:1}`, core `c`→`x=c`). The gather
assumes the producer's `out`/Lk-band-32 model — **the split dimension itself differs** → the
source-shard model is wrong against reality → the ~0.02.

Why this is structural: `mul(K)` is a **separate device program** (bundle #0). LX does not
persist across programs, so the QK^T bundle (#1) **re-reads K from HBM** via the ReStickify
(in *its own* `x:32` work-division). The all-gather, operating on that re-materialized K:
(a) mis-models the layout, and (b) **can't avoid the HBM read it was meant to eliminate.**
This is the identical conclusion the SwiGLU substrate reached (see `SPLITK_VS_COORDINATE_REMAP.md`
and the split-K device-negative): the on-chip substrate is gated by the separate-program
boundary.

## 4. The real fix — co-bundle the producer (the genuine hard part)
For the on-chip gather to be correct *and* useful it must operate on the producer's true
LX-resident shards, which requires **co-bundling `mul(K)` into the QK^T program** so K never
round-trips HBM (no ReStickify re-read). Then the gather's source = the producer's `mul(K)`
per-core shards (the model the geometry already assumes), and the gather genuinely replaces
the HBM broadcast. This is the substantial cross-bundle redesign — the same prerequisite the
SwiGLU work identified — and it should be scoped as its own effort, not bolted onto the
ReStickify path.

Open design questions for that effort: can `spyre_fuse_nodes` co-bundle `mul(K)` + QK^T + the
softmax + `@V` within the SDSC tensor budget (the attention bundle is already large)? Does the
co-bundled producer's per-core shard layout match the consumer's `{Lq:4,Lk:8}` cohort read
(so the gather is a pure re-band + cohort-broadcast)? These mirror the SwiGLU co-bundle
analysis.

## Pointers
- Code: `swiglu-ws-v2` (`3899472`, `d4220c5`, `596a5c7`, `05df9db`), files
  `torch_spyre/_inductor/broadcast_reshard.py`, `reshard/{pieces,substrate,cells}.py`,
  `codegen/bundle.py`, `fusion.py`.
- dxp: `deeptools-allgather` branch `allgather-on-latest` (`4f7da49fc3`).
- Spec: `/tmp/attn-allgather/PARSED_SPEC.md` (the sendnn target).
- Harnesses (device-free CPU compiles): `/tmp/v2_attn_allgather_correctness.py` (real Q/K/V,
  small, no wedge), `/tmp/v2_restickify_layout.py` (the conclusive probe),
  `/tmp/v2_placement_introspect.py`, `/tmp/v2_readpath_diag4.py`.
