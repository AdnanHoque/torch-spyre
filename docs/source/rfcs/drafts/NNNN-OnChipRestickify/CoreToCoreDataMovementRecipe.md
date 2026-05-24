# Core-to-Core (LX↔LX) Data Movement on the IBM Spyre AIU — Definitive Implementation Recipe

Source-of-truth reconstruction (2026-05-23). Self-contained: read once, reimplement
the general core-to-core data-movement primitive proved on device. This is the *how*;
the on-chip RFC (`/tmp/on-chip-rfc/.../NNNN-OnChipRestickifyRFC.md`) is the *why/what*
and is not duplicated here.

Every claim below is grounded in a file that was read. Inferences are flagged
explicitly with **[INFER]**.

---

## 0. TL;DR — one-paragraph recipe

On the Spyre AIU, "core-to-core data movement" means moving an activation slice from
one core's private LX scratchpad to another core's LX **over the RIU bidirectional
ring**, with **zero HBM traffic**, using the existing deeptools data-op `STCDPOpLx`
(same-stick move) which lowers to the `l3lu`/`l3su` RIU ring units (`L3_LDU`/`L3_STU`
microcode). Because LX does **not** persist across SDSC boundaries inside a bundle,
the only way to keep a producer→consumer handoff in LX is to put the consumer's DL op
and the data-op(s) that feed it into **one mixed `SuperDsc`** (`dscs_` + `datadscs_` +
`coreIdToDscSchedule`). The five-step path: **(1)** compile the baseline graph and find
the producer/consumer SDSC files and the HBM-bridged `labeledDs`; **(2)** flip producer
output + consumer input to LX-resident; **(3)** synthesize `datadscs_` STCDP blocks whose
per-core `PieceInfo.PlacementInfo.memId` encodes the source→destination core ownership
(same `memId` = same-core copy, different `memId` = real ring transfer — the
"reversed-ownership" trick `i→31-i` forces genuine cross-core traffic while staying
value-correct); **(4)** install `datadscs_`/`coreIdToDscSchedule`/`opFuncsUsed_` and
compile through a **minimally patched dxp** (one relaxed import gate + one dispatch
branch); **(5)** verify with the three-layer methodology (microcode senprog signature,
device value-correctness, and a mandatory negative control). This was proven on real
hardware: value-correct (max_err 0.0137 = baseline), HBM-free, all 32 cores emitting
`L3_LDU`/`L3_STU` to mirror core `31-i`.

---

## 1. Hardware first principles

(Grounded in `reference_aiu_architecture.md`, `reference_dsm_psum_algos.md`,
`Stage120-InputFetchNeighborLXPrototype.md`, and the verbose senprog.)

**AIU 1.0 (IBM Spyre):** 32 cores, 75 W, ~72 TFLOPS fp16.

Per-core layout:
- 2 corelets (Corelet 0 = CW, Corelet 1 = CCW), each `8×8 PE • PT • PE • SFP`.
- **2 MB LX scratchpad**, shared between the two corelets — the private on-core memory.

Fabric bandwidths (spec):

| Fabric | Spec | Direction | Notes |
|---|---|---|---|
| HBM bus (LPDDR5) | 128 B/cyc × 1.3 GHz = **166 GB/s** | unidirectional | shared across all 32 cores; the binding bottleneck |
| **RIU BiRing** | 128 B/cyc/dir × 1.3 GHz = **166 GB/s/dir** | bidirectional | aggregate 333 GB/s/link, 33 nodes; **this is what `STCDPOpLx` uses** |
| SFP UniRing CW/CCW | 32 B/cyc × 1.1 GHz = 35.2 GB/s each | unidirectional | **intra-corelet only — NOT inter-core**; do not confuse with RIU |
| LX (per core) | 128 B/cyc × 1.1 GHz = 140 GB/s | shared port | **~4.5 TB/s aggregate across 32 cores** |

**The L3 ring units.** The RIU-facing units that move data between cores' LX over the
ring are **`l3lu`** (L3 load unit) and **`l3su`** (L3 store unit). In emitted senprog
these appear as `Program for unit l3lu` / `Program for unit l3su` blocks; the actual
ring transfers are the `L3_LDU` / `L3_STU` instructions; `L3_SYNC` are barriers. A
core that only copies to itself emits `l3lu`/`l3su` blocks containing **only** sync
ops (no `L3_LDU`/`L3_STU`) — the ring transfer is dead-code-eliminated.

**Stick.** A stick = a 128-byte aligned chunk = **64 fp16 elements**
(`STICK_SIZE = 64`, `WORD_LENGTH = 2`, `dataformat = SEN169_FP16` in
`onchip_bridge.py`).

**Ring distance / byte-hops.** For cores `i`, `j`:
`ring_distance(i, j) = min(|i-j|, 32 - |i-j|)` (ring wraps at 32);
`byte_hops = bytes_moved × ring_distance`. This is the Tier-0 cost-model metric
(Python only; not emitted in microcode).

**Why eliminate HBM round-trips and prefer the ring.** A producer→consumer activation
handoff that goes through HBM serializes through one **shared 166 GB/s** HMI pipe for
all 32 cores. The RIU BiRing aggregate (32 links × 333 GB/s ≈ **~10.6 TB/s**) is
roughly **64× the HBM bandwidth**, and LX is ~4.5 TB/s. Keeping a handoff on-chip
(LX→ring→LX) replaces a serialized HBM round trip with parallel ring transfers. For an
already-co-located slice it is a pure LX→LX copy with *no* ring traffic at all. This is
the entire economic basis of the optimization.

---

## 2. The problem — why activation handoffs go through HBM

(Grounded in `project_ring_aware_restickify.md` §"Why on-chip handoff CANNOT be
pure-inductor" and Stage 203 baseline.)

A torch-spyre **bundle** (`bundle.mlir`) is a *list* of SDSCs run sequentially:

```mlir
module {
  func.func @sdsc_bundle() {
    sdscbundle.sdsc_execute () {sdsc_filename="sdsc_0_ReStickifyOpHBM.json"}
    sdscbundle.sdsc_execute () {sdsc_filename="sdsc_1_add.json"}
    sdscbundle.sdsc_execute () {sdsc_filename="sdsc_2_add.json"}
    ...
    return
  }
}
```

Each `sdsc_execute` is a separate runtime launch (`sdsc_execute`). Two facts force HBM:

1. A bundle = multiple SDSCs (one per `OpSpec` via `compile_op_spec`), run sequentially.
2. **LX does NOT persist across an `sdsc_execute` boundary.** A value an SDSC leaves in
   LX is gone when the next SDSC starts.

Therefore the **on-chip unit is the SDSC, not the bundle.** A producer in SDSC *k* and a
consumer in SDSC *k+1* can only communicate through HBM, which is exactly what the stock
pipeline does via an explicit `ReStickifyOpHBM` SDSC between them. Stock-baseline evidence
(Stage 203): `sdsc_0_add` / `sdsc_1_ReStickifyOpHBM` / `sdsc_2_add` were all in one
`bundle.mlir` and **still** used HBM for the intermediate.

**Conclusion:** to keep a handoff in LX you must place **producer-output + the data-op
move + consumer DL op into a single `SuperDsc`.** Fusion alone does not help —
`fusion.py spyre_fuse_nodes` has no restickify barrier; matmul/bmm become
`Reduction.create` SchedulerNodes; the only fusion boundary drivers are the ~6
non-intermediate tensor-segment budget and genuine non-SchedulerNodes. None of that
moves an inter-SDSC handoff on-chip.

---

## 3. The mechanism — the mixed DL + data-op SuperDSC

(Grounded in `Stage202-MixedBundleContract.md`, `onchip_bridge.py`, the verified
spliced JSON, and the deeptools patch.)

A **mixed `SuperDsc`** carries, in one SDSC body:

- **`dscs_`** — the consumer **DL op** (e.g. the `add` or `batchmatmul` body).
- **`datadscs_`** — a list of **data-ops** (the move blocks) that run *before* the DL op.
  (In the JSON the runtime/dxp also refers to these as `dataOpdscs_`; the splice writes
  the key `datadscs_` and the patched dxp reads `dataOpdscs_` after `importJson`.)
- **`coreIdToDscSchedule`** — per-core schedule sequencing data-ops then the DL op.
- **`opFuncsUsed_`** — the list of op-function names used by the data-ops, e.g.
  `["STCDPOpLx", "STCDPOpLx"]`.

### 3.1 `coreIdToDscSchedule` row schema

`mixed_schedule(num_dataops, num_cores)` in `onchip_bridge.py` emits, for every core
`c` in `0..num_cores-1`, the **same** list of rows. Each row is a 4-tuple:

```
[datadsc_idx, dldsc_idx, after_sync, before_sync]
```

- `datadsc_idx` — index into `datadscs_`, or `-1` if this step runs a DL op.
- `dldsc_idx` — index into `dscs_`, or `-1` if this step runs a data-op.
- `after_sync` — 1 if a barrier precedes this step (0 for the first data-op).
- `before_sync` — 1 if a barrier follows this step.

The synthesized rows: each data-op `k` is `[k, -1, 1 if k>0 else 0, 1]`, then the DL op
is `[-1, 0, 1, 0]`. For the 2-STCDP round trip (verified on disk):

```json
"coreIdToDscSchedule": { "0": [[0, -1, 0, 1], [1, -1, 1, 1], [-1, 0, 1, 0]], "1": [...], ... "31": [...] }
```

Meaning: step 0 = data-op 0 (before-sync); step 1 = data-op 1 (after- and before-sync);
step 2 = DL op 0. (Matches Stage 202's `[[0,-1,0,1],[1,-1,1,1],[-1,0,1,0]]` exactly.)

### 3.2 The two data-op op types

- **`STCDPOpLx`** — pure **same-stick** LX→LX move. Op dict is minimal:
  `{"name": "STCDPOpLx"}`. Requires `stickDimOrder_` to match between source and
  destination (it cannot change the stick). **This is the data-movement primitive that
  rides the ring and runs clean on device.**
- **`ReStickifyOpWithPTLx`** — a local **stick-transpose** (the COMPUTE op). Op dict
  carries the one-corelet contract that DCG requires:

  ```json
  { "name": "ReStickifyOpWithPTLx", "numClToUse": 1, "defaultClId": 0,
    "workSplitDim": "null_ptr", "cl0ToLxOffsetLU": 0, "cl0ToLxOffsetSU": 0,
    "useARF": 1, "doInPlace": 0 }
  ```

  (Stage 195: omitting these fields fails DCG with `coreLetWorkDs.cl0ToLxOffsetLU != -1`.)
  **This op FAULTS on device with a Compute-CB hardware error** — see §10. The pure
  STCDP move does not involve it.

### 3.3 Full `datadscs_` JSON schema (from `onchip_bridge.py`, byte-verified on disk)

One data-op block (`_datadsc`) is keyed by its name (e.g. `"0_STCDPOpLx_dataop"`):

```json
{
  "0_STCDPOpLx_dataop": {
    "coreIdsUsed_": [0, 1, 2, ..., 31],
    "dimPool_": ["mb_", "out_"],
    "outDimTodimRelation_": [],
    "primaryDs_": [
      {"name_": "dataIN",  "dimNames": ["mb_", "out_"]},
      {"name_": "dataOUT", "dimNames": ["mb_", "out_"]}
    ],
    "labeledDs_": [ <dataIN_L0>, <dataOUT_L0> ],
    "op": {"name": "STCDPOpLx"}
  }
}
```

Each `labeledDs_` entry (`_labeled_ds`) — verified field-for-field against
`/tmp/spliced-roundtrip/sdsc_2_add.json`:

```json
{
  "ldsName_": "dataIN_L0",
  "pdsName_": "dataIN",
  "wordLength": 2,
  "dataformat": "SEN169_FP16",
  "isExternal_": 0,
  "segment_": "output",
  "layoutDimOrder_": ["mb_", "out_"],
  "stickDimOrder_": ["out_"],
  "dimToLayoutSize_": {"mb_": 2048, "out_": 2048},
  "dimToStickSize_": {"out_": 64},
  "validGap_": {"mb_": [[2048, 0]], "out_": [[2048, 0]]},
  "totElements": -1,
  "hbmSize_": 0,
  "hbmStartAddress_": 0,
  "lxSize_": 2097152,
  "lxStartAddress_": {},
  "PieceInfo": [ <per-core piece 0>, <per-core piece 1>, ... ]
}
```

Key field meanings:
- `hbmSize_ = 0` — declares the data-op buffer is **not** HBM-resident (it lives in LX).
- `lxSize_` = per-core LX byte span for the data-op buffer (the splices use
  `DATAOP_LX_SIZE = 2097152` = 2 MB; this is the data-op-level size, distinct from the
  DL-level sentinel — see §6f).
- `stickDimOrder_` = `[stick_dim]` — for STCDP, **must be identical** on dataIN and
  dataOUT (same-stick rule, Stage 164).

Each `PieceInfo` entry (`_piece_info`) — the heart of where data physically lives:

```json
{
  "key_": "p1",
  "dimToStartCordinate": {"mb_": 0, "out_": 0},
  "dimToSize_": {"mb_": 2048, "out_": 64},
  "validGap_": {"mb_": [[2048, 0]], "out_": [[64, 0]]},
  "PlacementInfo": [{"type": "lx", "memId": [0], "startAddr": [16384]}]
}
```

- `key_` — piece label `p{i+1}`.
- `dimToStartCordinate` — the logical offset of slice *i* along the split dim
  (`i * chunk`). **Piece *i* always covers logical slice *i*** regardless of placement.
- `dimToSize_` — slice extent; the split dim gets `chunk = iter_sizes[split_dim] //
  num_cores`, all other dims get their full size.
- `PlacementInfo` — a single-element list `[{"type": "lx", "memId": [core], "startAddr":
  [base]}]`. **`memId` is the physical core that holds logical slice *i*.** This is the
  one field that decides whether a move is same-core or cross-core (§4).

---

## 4. THE KEY INSIGHT — expressing cross-core movement via `memId`

(Grounded in `_piece_info`'s `reverse` flag, `build_roundtrip_bridge`, and the verified
disk JSON `dataIN memId=[0]` vs `dataOUT memId=[31]`.)

The data-op matches a source `PieceInfo` to a destination `PieceInfo` **by logical
coordinate** (`dimToStartCordinate` / `key_`). Slice *i* on the source side is moved to
slice *i* on the destination side. The physical movement is determined entirely by the
two `memId`s:

- **Same `memId` on src and dst → same-core copy.** No ring transfer. DCG dead-code-
  eliminates the `L3_LDU`/`L3_STU` (confirmed: the degenerate same-split STCDP senprog
  has **zero** `L3_LDU`/`L3_STU`, only `L3_SYNC`).
- **Different `memId` on src and dst → genuine cross-core ring transfer.** DCG emits
  `L3_LDU`/`L3_STU` with the remote core encoded in the node field.

### The reversed-ownership trick

`_piece_info(..., reverse=True)` places logical slice *i* on core `num_cores-1-i`
instead of core *i*:

```python
mem = (num_cores - 1 - i) if reverse else i
pieces.append({
    "key_": f"p{i + 1}",
    "dimToStartCordinate": start,         # always i*chunk  -> logical slice i
    "dimToSize_": size,
    "validGap_": gap,
    "PlacementInfo": [{"type": "lx", "memId": [mem], "startAddr": [base]}],
})
```

`build_roundtrip_bridge` uses this to build a **2-STCDP round trip** that forces every
slice across cores yet lands the data back in the consumer's native layout:

```
STCDP1: producer (linear  @producer_base, slice i on core i)
        -> scratch  (REVERSED @scratch_base,  slice i on core 31-i)   [src reverse=False, dst reverse=True]
STCDP2: scratch  (REVERSED @scratch_base,  slice i on core 31-i)
        -> consumer (linear  @consumer_base,  slice i on core i)       [src reverse=True,  dst reverse=False]
```

Verified on disk (`/tmp/spliced-roundtrip/sdsc_2_add.json`):
- STCDP1 `dataIN` `PieceInfo[0].memId=[0]@16384`, `PieceInfo[1].memId=[1]@16384`, …
- STCDP1 `dataOUT` `PieceInfo[0].memId=[31]@1048576`, `PieceInfo[1].memId=[30]@1048576`, …

So STCDP1 moves slice 0 from core 0 → core 31, slice 1 from core 1 → core 30, etc.;
STCDP2 moves it back. **Why this is value-correct without consumer-reshard surgery:**
the round trip is `i → 31-i → i`, so each slice arrives back on the consumer's
native-owning core in the consumer's native (linear) layout. The consumer DL op reads
its input exactly where it expects it; no descriptor surgery on the consumer is needed.
The reversed scratch exists *only* to force real ring traffic for the proof; a
production same-layout cross-core remap would use a single STCDP with the actual
producer→consumer ownership delta.

---

## 5. The deeptools constraint — the ONLY foundation change

(Grounded in `deeptools_onchip_foundation.patch`, `project_ring_aware_restickify.md`
§"Tier 1 realization progress", and Stage 202.)

Stock pipeline:

```
torch-spyre  ->  dxp_standalone --bundle  ->  Dxp::importSdsc  ->  Dxp::runCodegen  ->  launchDCC
```

The runtime and dcc dispatch are **already stock-ready**: `dcc.cpp` + deeprt already
have the mixed dispatch (`has_dsc_schedule -> runDcgForDataOpsDlOps`) ungated; the
runtime executes the opaque senprog and never inspects `dataOpdscs_`. The **only** thing
that blocks mixed bundles is the dxp **import gate** plus the **codegen dispatch** not
routing to the mixed path. The minimal patch (`deeptools_onchip_foundation.patch`) does
exactly two things:

**(a) Relax the import gate** (`dxp/dxp.cpp` `Dxp::importSdsc`, ~line 469 — note Stage 202
called it `SdscTree.cpp:152`; it was folded into `dxp.cpp`):

```cpp
// stock:
// DT_CHECK_MSG(mySdsc->dataOpdscs_.empty(), "Datadsc not allowed, use dldsc");

// patched: allow dataOpdscs_ iff it ALSO has dscs_ and a per-core schedule.
const bool hasMixedDataOpSchedule = !mySdsc->dataOpdscs_.empty() &&
                                    !mySdsc->dscs_.empty() &&
                                    !mySdsc->coreIdToDscSchedule.empty();
DT_CHECK_MSG(mySdsc->dataOpdscs_.empty() || hasMixedDataOpSchedule,
             "Datadsc not allowed without dldsc schedule");
DT_CHECK_MSG(!mySdsc->dscs_.empty(), "No dsc in sdsc input");
```

Pure data-op-only imports remain rejected; only the mixed shape is admitted.

**(b) Dispatch to the mixed codegen** when `coreIdToDscSchedule` covers all cores
(`dxp/dxp.cpp` `Dxp::runCodegen`, ~line 198):

```cpp
bool hasDscSchedule =
    static_cast<int>(sdsc->coreIdToDscSchedule.size()) == sdsc->numCoresUsed_ &&
    sdsc->numCoresUsed_ > 0;
if (hasDscSchedule) {
  for (auto& kv : sdsc->coreIdToDscSchedule) {
    if (kv.second.empty()) { hasDscSchedule = false; break; }
  }
}
if (hasDscSchedule) {
  dcg.runDcgForDataOpsDlOps(*sdsc);            // mixed DL + data-op path (already existed)
} else if (sdsc->dscs_.size() == 0 && sdsc->dataOpdscs_.size() > 0) {
  dcg.runDcg(*sdsc);                            // data-op only
} else {
  dcg.runDcgForDlOpsStandalone(*sdsc);         // stock DL-only
}
```

`runDcgForDataOpsDlOps` is a **pre-existing** deeptools function; the patch only *wires
it in*. This is sharply distinct from building a new primitive.

**Why a zero-deeptools-change path is impossible:** the senprog
(`loadprogram_to_device/<code_dir_name>-SenProgSend/init.txt`) — the only program file
the runtime loads besides `bundle.mlir` — is produced **only by dxp's post-DCC
orchestration**. `dcc_standalone` emits IR only. So even though the runtime and dcc
dispatch are mixed-ready, you cannot get a loadable senprog for a mixed bundle without
dxp running, and dxp's import gate rejects the mixed bundle. Hence the minimal patch is
unavoidable.

Patched binary used for all device proofs:
`/home/adnan/dt-inductor/build/deeptools-onchip/dxp/dxp_standalone`.

---

## 6. The realization recipe — step by step

(Grounded in `splice_2048_roundtrip.py`, `splice_2048_stcdp.py`, `splice_2048_bmm.py`,
and the verified producer/consumer LX-flip JSON.)

The proof harness *splices* a mixed bundle out of a real compiled baseline. The clean
production version would synthesize this in inductor (§11), but the splice is the exact
recipe.

**Worked case:** `f = (a + b.t() + c.t()) @ d`, fp16, size S=2048, 32 cores. Baseline
bundle (5 SDSCs):

```
sdsc_0_ReStickifyOpHBM.json   # graph-input restickify   (leave alone)
sdsc_1_add.json               # PRODUCER (output Tensor2-idx2 @ HBM 8388608)
sdsc_2_add.json               # CONSUMER (input  Tensor0-idx0 @ HBM 8388608)  <- becomes MIXED
sdsc_3_ReStickifyOpHBM.json   # in-graph restickify
sdsc_4_batchmatmul.json       # downstream matmul
```

### (a) Find the producer/consumer SDSCs and the bridged `labeledDs`

Locate the producer→consumer edge by **HBM-address tracing**: the producer's
`computeOp_[0].outputLabeledDs[0]` (`"Tensor2-idx2"`) and the consumer's
`computeOp_[0].inputLabeledDs[0]` (`"Tensor0-idx0"`) share the same HBM address
(8388608). The `-idx<N>` suffix is the `ldsIdx_` to patch. (For the bmm consumer the
bridged input is idx0; for an `add` consumer it was idx1 — always confirm by address.)

### (b) Flip producer output + consumer input to LX-resident (`_flip_tensor_to_lx`)

For the `labeledDs` at `ldsIdx_` in a DL DSC, rewrite both the labeledDs entry and its
`scheduleTree_` allocate node. Verified result on disk (producer output @16384):

```json
"memOrg_": {"lx": {"isPresent": 1, "allocateNode_": "allocate-Tensor2_lx"}},
"hbmStartAddress_": -1,
"hbmSize_": 0,
"lxSize_": 2147483647,
"lxBufferSize_": 2147483647,
"coreStateInit_": [ {
    "ebrInit_": -1,
    "gtr_": {"type": "multicast", "id": 18446744073709551615, "count": 0,
             "sharers": 0, "groupInfo_": {}},
    "condGtr_": [],
    "lbrInit_": [16384],                 // <- the LX base for this tensor
    "gapPerDim_": {},
    "lxSizeWithGaps_": 2147483647,
    "lbrInitForwardGap_": 0
  }, ... (one per core) ]
```

and the matching `scheduleTree_` allocate node:

```json
{ "name_": "allocate-Tensor2_lx", "nodeType_": "allocate", "ldsIdx_": 2,
  "component_": "lx",
  "startAddressCoreCorelet_": {"data_": {"[0, 0, 0]": "16384", "[1, 0, 0]": "16384", ...}} }
```

`lbrInit_` (the LX **B**ase **R**egister init) is the core's local base address for that
tensor; `2147483647` (`DL_LX_SENTINEL`) is the LX size sentinel inside the DL DSC.
`numCoreletsUsed_DSC2_ = 1` is also set on the folded DL op to mark it inside a mixed
SuperDSC.

### (c) Synthesize `datadscs_` matching the consumer's actual sharding

Read the consumer's `numWkSlicesPerDim_` to learn its work-division sharding. Verified:
the consumer add has `numWkSlicesPerDim_ = {"mb": 1, "out": 32}` — i.e. `out` is split
32 ways (one slice per core), `mb` is full. The synthesized STCDP must split the **same
dim the same way**: `split_dim = "out_"`, `chunk = iter_sizes["out_"] // num_cores =
2048 // 32 = 64`. (Our graph shards `out:32` uniformly; chunk = S/num_cores.) If the
synthesized bridge's sharding does **not** match the consumer's `numWkSlicesPerDim_`,
the per-core piece extents disagree with what the DL op reads → wrong values or DCG
failure.

Build the round-trip bridge:

```python
datadscs, opfuncs, sched = onchip_bridge.build_roundtrip_bridge(
    dim_pool=["mb_", "out_"],
    iter_sizes={"mb_": 2048, "out_": 2048},
    stick_size=64, num_cores=32, lx_size=2097152,
    producer_base=16384, scratch_base=1048576, consumer_base=8192,
    layout=["mb_", "out_"], stick_dim="out_", split_dim="out_",
)
# -> ([stcdp1, stcdp2], ["STCDPOpLx", "STCDPOpLx"], <coreIdToDscSchedule>)
```

For the degenerate same-core (HBM-elimination only, no ring) use
`build_same_layout_bridge(... src_split_dim="out_", dst_split_dim="out_")` (single
STCDP, both endpoints non-reversed → same `memId` → no `L3_LDU`/`L3_STU`).

### (d) Install the mixed-SuperDSC scaffolding

On the consumer SDSC body:

```python
body["coreIdToDscSchedule"] = sched
body["datadscs_"]           = datadscs
body["opFuncsUsed_"]        = opfuncs
```

For the STCDP-only / round-trip splices the bundle.mlir is **unchanged** (all 5 SDSCs
kept; the consumer SDSC simply becomes mixed in place). For the Tier-2 transpose splice
(`splice_2048_bmm.py`) the in-graph `ReStickifyOpHBM` and the standalone batchmatmul are
dropped and `bundle.mlir` is rewritten to end with the new mixed SDSC. Then delete stale
runtime artifacts so dxp regenerates them: `loadprogram_to_device/`, `execute/`,
`segment_size.json`, and the `*_dsg.txt` files (`clean_stale_artifacts`).

### (e) Compile with the patched dxp

Run `dxp_standalone --bundle` from the patched binary over the spliced code dir. It
regenerates `loadprogram_to_device/<dir>-SenProgSend/init.txt` (the senprog). Offline,
the 2048 mixed bundle compiles HBM-free (exit 0).

### (f) LX base contract — and the GOTCHA

Reference LX bases (Stage 195/203, used unchanged for all 2048 proofs):

| Buffer | LX base |
|---|---|
| producer output | **16384** |
| bridge scratch (round-trip reversed intermediate) | **1048576** |
| consumer input | **8192** |

Per-core data-op LX span `DATAOP_LX_SIZE = 2097152` (2 MB); DL-DSC sentinel
`DL_LX_SENTINEL = 2147483647`. **GOTCHA:** these bases are **2048-derived and fixed**;
they do *not* scale with per-core slice size. At S=4096 the per-core slice doubles
(`out_` chunk 64→128) but the bases stay at 16384/1048576/8192 with the same 2 MB span —
which is exactly why the 4096 round trip broke correctness (§9). A production version
must compute LX bases from per-core slice bytes.

---

## 7. Verification methodology — where most attempts go wrong

(Grounded in `devval_direct.py`/`.sh`, `devval_roundtrip.py`/`.sh`, `bench_onchip.py`,
and the project-memory "Runtime gotcha" notes.)

### (a) Defeat `g_artifact_cache` — redirect to a FRESH code_dir

`g_artifact_cache` (`spyre_kernel.cpp`) is keyed on `code_dir` and is **per-process**.
Swapping the on-disk senprog at the *same* `code_dir` is **shadowed** by the cache — the
device keeps running the program it first loaded (this caused an illusory "pass"). To
force a real load, monkeypatch the kernel runner to point at a **fresh** path the
process has never seen:

```python
import torch_spyre.execution.kernel_runner as kr
_orig = kr.SpyreSDSCKernelRunner.__init__
def _patched(self, name, code_dir):
    _orig(self, name, code_dir)
    if "mm" in name.lower():            # the (a+b.t+c.t)@d fused add+matmul kernel
        self.code_dir = "/tmp/spliced-roundtrip"
kr.SpyreSDSCKernelRunner.__init__ = _patched
```

### (b) ALWAYS run a negative control

After confirming the positive run, **remove the spliced senprog and re-run; the run MUST
fail.** This proves the device executed *your* spliced program, not a cached/baseline
one. From `devval_roundtrip.sh`:

```bash
SP=/tmp/spliced-roundtrip/loadprogram_to_device/spliced-roundtrip-SenProgSend/init.txt
# POSITIVE: redirect runner -> must load spliced + be VALUE-CORRECT (no Compute-CB)
# NEGATIVE: mv "$SP" "$SP.bak"; rerun  -> must FAIL (No such file / RuntimeError)
mv "$SP" "${SP}.bak"; <rerun>; mv "${SP}.bak" "$SP"
```

**Two illusory "successes" the negative control caught:** (1) a same-path senprog swap
shadowed by `g_artifact_cache` — the device ran the baseline, looked "correct"; (2) a
degenerate same-split STCDP that eliminated HBM but emitted **zero** ring traffic — it
"passed" but proved nothing about cross-core movement. Without the negative control and
the senprog inspection, both would have been mistaken for cross-core proofs.

### (c) Inspect the senprog ring signature

`DXP_VERBOSE=1` on the patched dxp dumps readable per-SDSC microcode at
`debug/<sdsc>/senprog.txt`. The differential is the proof:

**Cross-core (round trip)** — `/tmp/rt-verbose/debug/sdsc_2_add/senprog.txt`, core 0:

```text
========== Core: 0 Corelet: 0 Unit: l3lu Program START ============
L3_MVLOOPCNT | (64 << 10)
L3_LDU | (1 << 31) | (0 << 22) | (0 << 27) | (31 << 14) | (0 << 6) | (0 << 10)
L3_SYNC  | (91 << 10)
...
========== Core: 0 Corelet: 0 Unit: l3su Program START ============
L3_STU | (1 << 31) | (0 << 22) | (0 << 27) | (31 << 14) | (0 << 6) | (0 << 10)
```

The `(31 << 14)` is the **remote-core target field** — core 0 targets core 31. Core 1
emits `(30 << 14)`, core 2 `(29 << 14)`, … i.e. core *i* targets core **31-i**
(`(31-i) << 14`). All 32 cores emit both `L3_LDU` and `L3_STU`. The `reg_initial.txt`
beside it shows the LBR values `8192 / 64 / 12288` (consumer base, stick stride, …),
confirming the LX endpoints.

**Same-core (degenerate STCDP)** — `/tmp/dg-verbose/debug/sdsc_2_add/senprog.txt`:
**zero** `L3_LDU` and **zero** `L3_STU` (`grep -c` → 0); only `L3_SYNC` remain. The ring
transfers are dead-code-eliminated because src `memId` == dst `memId`. **This is the
key differential: the presence of `L3_LDU`/`L3_STU` with remote targets IS the
cross-core ring traffic, at instruction level.**

### (d) Cache-bust `torch.compile`

Otherwise the inductor passes don't re-run and you compile a stale graph:

```python
torch._dynamo.reset()
torch._inductor.config.fx_graph_cache = False
# + a fresh TORCHINDUCTOR_CACHE_DIR per run (devval_*.sh: TORCHINDUCTOR_CACHE_DIR=/tmp/<unique>)
```

### (e) Run from a NEUTRAL cwd

A stray `triton` namespace dir (e.g. `/home/adnan/dt-inductor/triton`) pollutes
`import triton` from that cwd and breaks `torch._dynamo`. Run from a neutral directory.
(Device-validation also used a process-local `sitecustomize` shim at
`PYTHONPATH=/tmp/val-boot` to import the worktree without a global venv change.)

---

## 8. The proof it's genuinely cross-core — three independent layers

(Grounded in `project_ring_aware_restickify.md` §"GENUINE CROSS-CORE RING STCDP" and the
verbose senprog.)

1. **Microcode.** All 32 cores emit `L3_LDU` *and* `L3_STU` over the RIU ring, core *i*
   targeting core *31-i* via the `((31-i) << 14)` node field (0→31, 1→30, …, verified in
   `/tmp/rt-verbose/debug/sdsc_2_add/senprog.txt`). The degenerate same-split STCDP emits
   zero such instructions. The differential **is** the ring traffic.
2. **Device.** Runs value-correct: `max_err = 0.0137` = the fp16 baseline error for the
   whole graph `(a+b.t()+c.t())@d` at 2048; **no** `Compute CB hardware error` /
   `RAS::RUNTIMESCHEDULER::ComputeHardwareError`. The remove-the-senprog negative control
   fails as required.
3. **Logical lock.** Value-correctness *requires* the path core *i → 31-i → i*: the
   consumer's `LX@8192` on core *i* is only written by STCDP2 reading core *31-i*'s
   scratch; `i ≠ 31-i` for all 32 cores. If the ring had collapsed to same-core copies
   the consumer would read uninitialized/wrong data. Correctness therefore *proves* the
   round trip happened — it cannot be a same-core illusion.

---

## 9. Performance results + what they teach

(Grounded in `bench_onchip.py`, `bench_onchip_results.txt`, `bench_onchip_multisize.txt`.)

Benchmark: `(a + b.t() + c.t()) @ d`, fp16, 32 cores, median of 60 iters (15 warmup),
one config per process (so `g_artifact_cache` only ever holds one program). `baseline_HBM`
= stock bundle (add→add handoff via HBM); `spliced-stcdp` = same-core HBM-elimination;
`spliced-roundtrip` = cross-core ring.

| Size | baseline_HBM ms | stcdp ms (speedup) | roundtrip ms (speedup) | roundtrip max_err |
|---|---|---|---|---|
| 512  | 0.0957 | 0.1006 (**0.95×**, regression) | 0.1067 | 0.004883 (correct) |
| 1024 | 0.3202 | 0.2629 (**1.22×**) | 0.2768 | 0.007812 (correct) |
| 2048 | 1.5570 | 1.3127 (**1.19×**) | 1.3294 | 0.013672 (correct) |
| 4096 | 7.9100 | 7.0151 (**1.13×**) | 7.1389 | **6.148438 (BROKEN)** |

(2048 reproduced across two reps: baseline 1.534/1.541, stcdp 1.321/1.306, roundtrip
1.339/1.340 — stable.) The round trip tracks the same-core STCDP closely (the ring
transfer is cheap relative to the matmul), and is value-correct at 512/1024/2048.

**What it teaches:**

- **(i) Size crossover.** At 512 the STCDP overhead exceeds the HBM saving → a slight
  regression (0.95×). The on-chip path should be gated on a minimum size. Note also at
  S=512 the per-core `out_` chunk is **16 < stick(64)** (verified:
  `PieceInfo[0].dimToSize_.out_ = 16`), so each core moves a sub-stick slice — small,
  awkward shapes correctly belong on HBM.
- **(ii) Relative speedup peaks mid-range.** Speedup is highest at 1024 (1.22×) and
  declines at 4096 (1.13×) because the matmul cost grows O(N³) while the handoff grows
  O(N²): the handoff saving becomes a smaller *fraction* of total time even as its
  *absolute* magnitude grows. End-to-end claims must weight by workload share.
- **(iii) The 4096 round-trip break = LX bases must scale — and a deeper capacity limit.**
  Correctness broke (`max_err 6.15`) only on the cross-core round trip at 4096. Precise
  diagnosis (verified): per-core slice = S rows × (S/32) cols × 2 B; at 4096 that is
  4096×128×2 = **1 MB**, so the producer region at base 16384 ends at 1,064,960 and
  **overlaps the reversed scratch at 1,048,576** → corruption. (At 2048 the slice is only
  256 KB, no overlap — which is why 2048 worked.) **Fix implemented**:
  `per_core_slice_bytes()` + `allocate_lx_bases()` compute stick-aligned, non-overlapping
  bases from the slice size. **But this also exposed a hard limit:** per-core LX is **2 MB**
  (`scratchpad.py`), the round trip needs **3 live regions** (producer + reversed scratch +
  consumer), and at 4096 that is 3 × 1 MB = 3 MB > 2 MB — it **cannot fit regardless of
  base placement**, so the allocator correctly *rejects* the 4096 round trip (NOFIT). The
  same-core path (2 regions) and a *real* single cross-core move (also 2 regions:
  producer + consumer, no scratch) both fit at 4096. The 3-region round trip is a *proof*
  construct; production cross-core handoffs are single moves and do not hit this wall.

---

## 10. Gotchas checklist (consolidated)

Each has a *why*, not just a *what*.

- **`import regex` (aliased `re`), never `import re`** — pre-commit hook enforces it; a
  bare `import re` fails CI.
- **Apache 14-line header on every source file** — license compliance; missing header
  fails pre-commit. (All splice scripts and `onchip_bridge.py` carry it.)
- **88-char line length (ruff)** — style gate; longer lines fail lint.
- **Mixed-bundle import gate** — stock dxp rejects any `dataOpdscs_` with
  `"Datadsc not allowed, use dldsc"`. *Why:* the import contract assumes DL-only bundles;
  the patch relaxes it only for the mixed shape (§5).
- **Fresh `code_dir` for device verification** — `g_artifact_cache` is per-process and
  keyed on `code_dir`; a same-path senprog swap is shadowed and you measure the baseline.
- **Negative control is mandatory** — removing the senprog must make the run fail; this
  is the only thing that proves the device ran your program. Two illusory passes happened
  without it.
- **Cache-bust `torch.compile`** — `torch._dynamo.reset()` +
  `_inductor.config.fx_graph_cache = False` + a fresh `TORCHINDUCTOR_CACHE_DIR`, or the
  inductor passes don't re-run and you compile a stale graph.
- **Neutral cwd** — a stray `triton` dir poisons `import triton` and breaks dynamo.
- **Same-core STCDP rings are dead-code-eliminated** — a "passing" same-split STCDP
  proves HBM elimination but **nothing** about the ring (zero `L3_LDU`/`L3_STU`). Always
  use a reversed-ownership intermediate (or a real ownership delta) to test the ring.
- **The Compute-CB fault is the `ReStickifyOpWithPTLx` TRANSPOSE, not the data move.**
  The Tier-2 transpose bridge (`splice_2048_bmm.py`) **loads and reaches the AIU** (a
  working negative control proved that) but executing it throws
  `RAS::RUNTIMESCHEDULER::ComputeHardwareError` ("Compute CB hardware error"). The pure
  STCDP move (same-core *and* cross-core round trip) runs clean. *Why it matters:* the
  ring data path is sound; the fault is confined to the compute (transpose) op — the same
  wall as Phase-B iter-3 (`sfpring` is psum-only, FMA-fused accumulation, not a pure data
  ring).
- **LX bases must scale per-size** — the 16384/1048576/8192 contract is 2048-derived;
  fixed bases corrupt the cross-core round trip at 4096 (§9 iii).
- **Synthesized sharding MUST match the consumer's `numWkSlicesPerDim_`** — read it
  (`{"mb":1,"out":32}` here) and split the same dim the same way (`chunk = S/num_cores`);
  a mismatch gives wrong per-core extents → wrong values or DCG failure.
- **Sticks are 64 fp16 / 128 B** — `dimToStickSize_ = 64`; below this the per-core slice
  is a sub-stick (S=512 → chunk 16), which is small/awkward and belongs on HBM.
- **`STCDPOpLx` requires matching `stickDimOrder_`** (same-stick only). *Why:* it cannot
  change the stick; layout-CHANGING moves need the transpose op (which faults). The
  composable layout-changing cross-core move is the genuinely-missing primitive (the open
  frontier, §11).
- **Build/run env** — use `/home/adnan/dt-inductor/.venv/bin/python3` (torch 2.11); system
  python is torch 2.10 and breaks `torch_spyre/_monkey_patch.py`. Scripts from `/tmp` need
  `PYTHONPATH=/home/adnan/dt-inductor/torch-spyre`. **[INFER]** device-event profiling
  needs `USE_SPYRE_PROFILER=1` at build; not required for the value-correctness proofs here.

---

## 11. What's proven vs what remains (frontier)

(Grounded in `project_ring_aware_restickify.md`, on-chip RFC §"two deeptools asks".)

**Proven on device (2026-05-23):**
- A genuine **cross-core same-stick ring STCDP** — every core's activation slice over the
  RIU ring to a remote core and back (`i → 31-i → i`) — runs **value-correct, HBM-free**,
  inside a mixed DL + data-op `SuperDsc`, with only the minimal dxp patch (no LD_PRELOAD).
- Same-core HBM elimination (degenerate STCDP) value-correct at all sizes 512–4096.
- The mixed DL + data-op control path, packaging, scheduling, and LX residency all work
  end-to-end on hardware.

**Open (the frontier):**
- **(a) Layout-CHANGING cross-core move.** `ReStickifyOpWithPTLx` (the local PT
  transpose) faults `Compute-CB` on device. This is the genuinely-missing composable
  primitive: the three existing ops don't compose — `STCDPOpLx` (same-stick only,
  `stickDimOrder_` must match), `ReStickifyOpWithPTLx` (transposes but emits a native
  `j_,i_,out_,mb_` descriptor ≠ the consumer's 2D LX descriptor → needs a
  consumer-endpoint adapter), `InputFetchNeighbor` (same-stick remote gather). Likely the
  same wall as `sfpring` being psum-only.
- **(b) Per-size LX allocation for the cross-core path** — *now implemented*
  (`per_core_slice_bytes()` / `allocate_lx_bases()` in `onchip_bridge.py`): fixed bases
  broke at 4096 (§9 iii); bases are now derived from per-core slice bytes, stick-aligned
  and overlap-checked against the 2 MB/core LX. Residual limit: the 3-region *round trip*
  cannot fit at 4096 (3 MB > 2 MB) — but that is a proof artifact; a production single
  cross-core move is 2 regions and fits. Validated on device at 512/1024/2048.
- **(c) Productionizing the binding** — the splice does the producer-writes-LX /
  consumer-reads-LX coordination *post hoc*. A clean version needs **scratchpad/LX
  planning in inductor**: bind the bridge LX output to the specific consumer input,
  allocate LX with correct liveness, and place sync. Currently torch-spyre cannot bind a
  bridge LX output to a specific consumer input via the supported `DscSenGraph` API (edge
  `index_` is a graph port, not an internal `labeledDs_` index).

**The two precise deeptools asks** (handoff, not inductor work — the user is on the
inductor team):
- **Foundation contract** (unlocks Tier 1 and aligned Tier 2): mixed-bundle import +
  binding hook through **stock** deeptools (no patch, no LD_PRELOAD). *Acceptance:* the
  2048 mixed bridge runs value-correct through stock deeptools.
- **Transform contract** (unlocks general Tier 2): remote-fragment-aware coordinate remap
  / consumer-endpoint adapter for the layout-changing move. *Acceptance:* the 512 case
  runs value-correct without forced descriptor overrides.

---

## 12. Real-world applicability — can it accelerate a workload today, and which ones?

This section answers three questions: *what does this buy a real model, which workloads,
and is it plug-and-play or does it need more work?* It is grounded in actual compiled
bundles found in the inductor caches; a detailed per-edge classification of **40 real
producer→consumer handoff edges** (across granite RMSNorm+linear and two SDPA attention
kernels) is at `/tmp/real_edge_analysis.md`. Headline:

| Class | Count (of 40) | Status |
|---|---|---|
| **Same-stick → STCDP today** | **27** | addressable by the proven primitive |
| ↳ same-shard (same-core, no ring) | 18 | simplest win (LX-resident, no ring) |
| ↳ different-shard (**genuine cross-core ring**) | 9 | mostly matmul-output → elementwise/softmax |
| Layout-changing → needs transpose (blocked) | 8 | cluster at matmul-input (`out→in`) + RMSNorm reshape |
| Graph-input / weight → prelayout bucket | 5 | inductor prelayout, no runtime primitive |

So the proven same-stick move covers the **majority (27/40)** of real activation handoffs —
broader than the earlier "~4% fundamental" framing, because most HBM handoffs are not
explicit restickifies; they are plain producer→consumer edges that cross an SDSC boundary
and preserve stick orientation. (Caveat from the tracer: cached SDSC JSONs lack
`hbmStartAddress_` — that field appears only post-dxp — so edges were traced via the
`scheduleTree_` allocate-node per-core HBM base with a latest-prior-producer rule;
self-consistent but inferred.)

### 12.1 The handoffs we target exist in real models

Inspecting real compiled bundles (not the `(a+b.t+c.t)@d` micro-graph) confirms the
HBM-handoff pattern is pervasive:

| Bundle (real building block) | #SDSCs | `ReStickifyOpHBM` (HBM handoffs) |
|---|---|---|
| Granite `add+linear+mul+rms_norm` block (`/tmp/granite_inductor/...rms_norm_6_*`) | 13 | **2** |
| SDPA attention (`/tmp/torchinductor_adnan/...attention_overrideable_0_*`) | 12 | **1** (+ heavy `transpose` in surrounding fusions) |

So even a single fused transformer block round-trips activations through HBM 1–2× — and
the roadmap workloads (Llama / Mistral / Granite / GPT-OSS, and MoE expert FFNs) are full
of these. Each eliminated HBM round-trip is `2 × tensor_bytes` of off-chip traffic at
~166 GB/s avoided.

### 12.2 But applicability splits sharply: same-stick vs layout-changing

What we proved on device is the **same-stick** move (`STCDPOpLx`). Real handoffs divide:

| Handoff type | Example | Status |
|---|---|---|
| **Same-stick** (same `stickDimOrder_`; only the per-core split/ownership differs) | elementwise / residual / norm chain feeding a same-orientation consumer | ✅ **proven primitive applies** (after productionization, §12.4) |
| **Layout-changing** (`stickDimOrder_` differs — a restickify that exists *to change* stick orientation, e.g. before a matmul) | most pre-matmul in-graph restickifies; Q/K/V transposes in attention | ❌ **blocked** — needs `ReStickifyOpWithPTLx`, which faults Compute-CB (§11a) |
| **Graph-input / weight restickify** (~52% bucket per codex) | initial layout of inputs/weights | ↪ better solved by **input/weight prelayout** in inductor (no runtime primitive; separate prelayout RFC) |

The uncomfortable point: an in-graph `ReStickifyOpHBM` often exists *precisely because* a
layout change is required (otherwise the compiler wouldn't have inserted it). Those are
the layout-changing bucket we cannot do yet. The handoffs we *can* do today are the ones
where producer and consumer already share a stick orientation and the restickify is pure
re-ownership (or where the handoff crosses an SDSC boundary with no layout change at all).

### 12.3 Where the win is largest

From §9: the *relative* speedup peaks mid-range (1.22× @1024) and tapers as the matmul
O(N³) dwarfs the O(N²) handoff. So the on-chip handoff matters most in
**bandwidth-bound regimes**, not compute-bound prefill:

- **Decode / autoregressive generation** — skinny matmuls (batch≈1, seq=1), activations
  dominate HBM traffic; eliminating activation round-trips is proportionally large.
- **MoE** — router → expert-FFN → combine moves a lot of activation data between cores;
  high ring-vs-HBM leverage (and the expert FFNs are the real bmm case — see
  `project_bmm_aware_split`).
- **Mid-size hidden dims (~1k–4k)** — the measured sweet spot.

Compute-bound prefill with large matmuls will see a smaller *relative* win even though the
absolute bytes saved grow.

### 12.4 Is it plug-and-play? No — the honest gap list

What exists today is a **device-proven mechanism realized by splicing** (hand-editing
compiled SDSC JSON), not an integrated compiler pass. To turn it into a model-level
speedup, in increasing order of effort:

1. **Land the deeptools gate (§5).** Today the mixed bundle compiles only via the
   *isolated patched* `dxp`. The ~1-file gate-relax + dispatch must land in production
   deeptools. Smallest blocker, but it is a deeptools handoff (Foundation contract, §11),
   not inductor work.
2. **An inductor realization pass** to replace the manual splice: detect an
   on-chip-eligible edge, flip producer-output / consumer-input to LX, allocate LX, emit
   the mixed `SuperDsc`, fold the bundle. The Tier-1 planner (`onchip_handoff.py`) already
   *detects* (fail-closed); the *binding* is the missing piece (§11c).
3. **Per-size LX allocation** (§9 iii / §11b). Mandatory for variable-size workloads — the
   fixed 2048-derived bases corrupt the cross-core path at 4096. **Now implemented**:
   `per_core_slice_bytes()` + `allocate_lx_bases()` in `onchip_bridge.py` pack
   stick-aligned, non-overlapping regions and raise if the footprint exceeds the 2 MB/core
   LX. (This also surfaced a hard limit — see the round-trip note in §9 iii / §11b.)
4. **Sharding match** — the synthesized bridge must read and match the consumer's actual
   `numWkSlicesPerDim_` (m-split, 2-D co-split for bmm/MoE), not assume `out:32`.
5. **Transpose support** for the layout-changing bucket — blocked on the Compute-CB fault
   (§11a); a hardware/deeptools question, not inductor.

**Lowest-friction first demo:** the **same-core** variant (handoff just stays in LX — no
ring, no reversed scratch; value-correct at *every* size including 4096) on a
**same-layout, same-sharding** residual/norm→linear edge at hidden dim ~1k–2k. That needs
only items 1+2+4. The cross-core ring (needed when ownerships differ) additionally needs
item 3. Layout-changing handoffs need item 5.

**Concrete best cross-core target** (from the real-edge tracer): the SDPA
`batchmatmul(QK^T) → softmax(sub/max)` edge, stick `['out']` — same-stick *and* genuinely
cross-core (producer shards `{mb:32}`, consumer `{x:32}` → real RIU-ring traffic, not a
dead-code-eliminated copy), and it recurs in **every attention layer of every roadmap
model**. Runner-up: granite `batchmatmul → mul` (MLP linear output). A real single
cross-core move there is two LX regions (producer + consumer), which fits at all sizes —
unlike the 3-region proof round trip.

### 12.5 Bottom line

The foundational hard part — a **same-stick core-to-core data-movement primitive proven
end-to-end on silicon** — is done. Same-layout handoffs are a few focused engineering
steps (gate + realization pass + per-size LX + sharding) from a real-model speedup;
layout-changing handoffs (likely the majority of high-value pre-matmul edges) are gated on
the deeptools/hardware transpose primitive. Expect the clearest early wins in
bandwidth-bound decode and MoE at mid-size hidden dimensions.

---

## Appendix: file index (all paths absolute)

| Role | Path |
|---|---|
| Synthesizer (emission core) | `/tmp/tier-up/torch_spyre/_inductor/codegen/onchip_bridge.py` |
| Splice — same-core STCDP (2048) | `/tmp/splice_2048_stcdp.py` |
| Splice — cross-core round trip (2048) | `/tmp/splice_2048_roundtrip.py` |
| Splice — Tier-2 transpose (FAULTS) | `/tmp/splice_2048_bmm.py` |
| Splice — size-parameterized (`SPLICE_SIZE`) | `/tmp/splice_onchip_stcdp.py`, `/tmp/splice_onchip_roundtrip.py` |
| Device validation — direct | `/tmp/devval_direct.py`, `/tmp/devval_direct.sh` |
| Device validation — round trip | `/tmp/devval_roundtrip.py`, `/tmp/devval_roundtrip.sh` |
| Benchmark harness + results | `/tmp/bench_onchip.py`, `/tmp/bench_onchip_results.txt`, `/tmp/bench_onchip_multisize.txt` |
| deeptools foundation patch | `/home/adnan/dt-inductor/deeptools_onchip_foundation.patch` |
| Patched dxp binary | `/home/adnan/dt-inductor/build/deeptools-onchip/dxp/dxp_standalone` |
| Cross-core senprog (proof) | `/tmp/rt-verbose/debug/sdsc_2_add/senprog.txt` |
| Same-core senprog (zero ring) | `/tmp/dg-verbose/debug/sdsc_2_add/senprog.txt` |
| Spliced bundles (on disk) | `/tmp/spliced-stcdp{,-512,-1024,-4096}`, `/tmp/spliced-roundtrip{,-512,-1024,-4096}`, `/tmp/spliced-2048` |
| On-chip RFC (the why/what) | `/tmp/on-chip-rfc/docs/source/rfcs/drafts/NNNN-OnChipRestickify/NNNN-OnChipRestickifyRFC.md` |
| Codex stage notes (senprog detail) | `/tmp/restickify-fp/docs/source/rfcs/drafts/NNNN-RestickifyLocality/Stage{120,164,195,202}*.md` |
| Project memory (running log) | `/home/adnan/.claude/projects/-home-adnan-dt-inductor-torch-spyre/memory/project_ring_aware_restickify.md` |
