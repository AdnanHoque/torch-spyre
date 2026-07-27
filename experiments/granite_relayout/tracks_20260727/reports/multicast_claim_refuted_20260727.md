## Is the 32x amplification real?

**No. There is no 32x amplification.** We inject 1.00 MiB per layer on the V edge â which is exactly the ideal figure the brief itself derives for a perfect 32-way all-gather. The "32.00 MiB" is a logging artifact, and both verifiers reproduced the correct number independently.

Here is the whole mistake, in one loop:

```
transfer_compute.cpp:477   for (const auto& [key, transfer] : op->dtTable_) {   // one entry = one INJECTION
transfer_compute.cpp:486     logicalBytes = ...                                 // computed ONCE, from the SOURCE piece
transfer_compute.cpp:488     wireBytes    = wireSticks * sysDef.bytesPerStick;  // computed ONCE, from the SOURCE piece
transfer_compute.cpp:491     for (size_t cpos = 0; cpos < transfer.cIDXs.size(); ++cpos)  // fan-out over DESTINATIONS
transfer_compute.cpp:508       std::cout << "STCDP_FINAL_TRANSFER ... wire_bytes=" << wireBytes  // SAME value, reprinted
```

`wireBytes` never reads `selectedMCMode` or `numSharers`. Summing the printed lines therefore multiplies every injection by its multicast fan-out. It yields **delivered** bytes, not injected bytes. DeepTools already prints the honest split on the line above, which nobody read:

```
runs/vedge_reinterp_5x/run.log:1123
  STCDP_FINAL_BEGIN sdsc=29_shuffle-Relayout ... entries=32 deliveries=1024
```

`entries=32` â 32 injections Ã 32768 B = **1.00 MiB injected/layer** (40.0 MiB over 40 layers).
`deliveries=1024` â 32.00 MiB delivered, 31.00 MiB of it remote.

That 31.00 MiB delivered is *numerically identical* to the SenDNN K^T figure the brief quotes ("payload 1.00 MiB/layer, 31.00 MiB DELIVERED/layer"). **We were already at byte-for-byte parity on this edge.** The claimed "~1280 MiB extra over 40 layers" is 1240 MiB of *delivered* bytes â the same quantity SenDNN's own catalog reports for P01 â not 1240 MiB of waste.

Source: DeepTools tree embedded in the binary that produced these artifacts (`strings /home/adnan/codex-isolated/device_parity_tracks_20260726/p14/deeptools-build/dcg/dcg_fe/libdcg_fe.so` â `.../p13/deeptools/...`; `diff -q` p13 vs p14 on `stcdpOp.cpp`, `transfer_compute.cpp`, `dcgbeCodegen.cpp` â all identical, so citations from either tree are valid).

### Three sub-premises that also fall

**`op.useUnicast = 0` does not mean "multicast is permitted". It means "multicast is happening."** It is an *output*:

```
stcdpOp.cpp:2733-2738
  void DcgFE::checkConvertToUnicast(STCDPOpLx* op) {
    bool useUnicast = true;
    for (auto& dtEntry : op->dtTable_)
      if (dtEntry.second.myGTR.numSharers > 1) useUnicast = false;
    op->useUnicast = useUnicast; }
```

**`selectedMCMode` is a ring *direction*, not multicast-vs-unicast.** `dataOpDsc.h:212` states it outright: `// 0--> random, 1--> CCW, 2-> CW, 3--> Replication`. Confirmed at the ISA level: in the non-unicast path `dcgbeCodegen.cpp:1648-1660` emits one `RINGDTU` with `group` = the GTR register and `node` = the mode; only the unicast path (`UNIRINGDTU`) puts a destination core id in `node`. And the brief's "selectedMCMode = 1" is half the table â the real census from `runs/vedge_reinterp_5x/stcdp_after_pcfg_29_shuffle-Relayout.json` is `{1:16, 2:16}`, because `CCWHopCWHop = (31,31)` ties on every entry and `stcdpOp.cpp:4542-4551` alternates the tie-break.

**`promoteToMode3` is a dead end twice over.** The gate is arithmetic:

```
stcdpOp.cpp:4853-4874
  if (hop_Mode3 != -1 && hop_Mode3 <= (maxNumCores / 8 * 3))   // 32/8*3 = 12
```
Our entries carry `hop_Mode3 = 16` (antipodal core is 16 hops away in *either* direction whenever you broadcast to all 32 cores). 16 > 12, so a full 32-way all-gather can **never** be promoted. And it would not help: `stcdpOp.cpp:5944-5972` models mode 3 as `tot/2` each way over CCW and CW hops â identical total ring bytes. It is a latency knob, not a bandwidth knob. Separately, the pass is not even running in this build: its caller is gated at `transfer_compute.cpp:141-144` on `psumRing == "sfpring"`, and `dscglobal.h:40` defaults it to `"dataring"`, set to `sfpring` only by sparse conv (`dsm.cpp:4788`, none in LLM prefill) or the explicit `psum=sfpring` DXP option (`dscglobal.cpp:198-210`), which appears in neither `run_integ.sh` nor either `run.log`.

**The geometry premise is also wrong.** The brief's "ratio 4, each destination core receives a distinct quarter" is not what the plan says. From `runs/vedge_reinterp_5x/relayout_plans.jsonl`, `source_name == "buf29"`: `destination_size_ratio = 32`, `destination_device_dim_splits = {"0":1,"1":1}`, and every one of the 32 cores maps to `{"0":0,"1":0}` â **one piece replicated 32 ways**, structurally the same object as SenDNN's `BatchMatMulV2_QC_3_inpLds_1_...-LxRelayout.json`. The 131072 B in the brief is buf31, the AV BMM's *output* buffer, not the relayout destination.

### So what is the +20 ms?

It is over-determined. Three candidates survive; none is proven; all point to the same action.

| # | Candidate | Magnitude | Evidence quality |
|---|---|---|---|
| **A** | **LX capacity spill** â the 1 MiB replicated destination evicts buf28 (512 KB) and buf25 (64 KB) to HBM | 53.0 MiB/layer extra HBM traffic = 2120 MiB over 40 layers â 9.3 Âµs/MiB â 107 GB/s effective | **Strongest.** The allocator says it in its own words |
| **B** | **Receive-side landing work** â L3LU landing transactions per layer 485,561 â 739,513 (+52%), +31 MiB/layer landed | +52% on the landing path | Measured from the PCFG loop nest; unattributed to time |
| **C** | **Ring link occupancy** â link-byte-hops 358.792 â 389.792 MiB/layer (+8.6%) | +8.6% ring load for +7.9% wall time | Weakest â see below |

The arithmetic for (A) is exact and it is the only one the tooling states directly:

```
runs/vedge_reinterp_5x/allocations.jsonl
  __spyre_lx_relayout_destination__:buf29   address=131072  size=1048576      (NEW)
  buf28  address=None  reject_reason="no room on scratchpad (t=33-44, size=512 KB)"
  buf25  address=None  reject_reason="no room on scratchpad (t=30-43, size=64 KB)"
runs/puremerge_5x/allocations.jsonl
  buf28  address=524288   size=524288   reject_reason=None
  buf25  address=1245184  size=65536    reject_reason=None
  buf29  address=None     reject_reason="graph output is a ReinterpretView"
```

Budget, from `allocator.py:111-115` with `contract.txt lxfrac=0.2`:
`round_up(int((2 MiB â 64 KiB) Ã 0.8), 128) = 1,625,344 B`.
Live LX at the AV BMM: baseline 655,360 B â vedge 1,310,720 B, of which 1,048,576 B (80%) is the V destination. Adding buf28: `1,310,720 + 524,288 = 1,835,008 > 1,625,344` â **overflow by 209,664 B.**

Candidate (C) is weak because the baseline already carries far more ring load than the V edge adds. On link-byte-hops â bytes Ã links traversed, the metric that actually bounds link occupancy â the pre-existing `60_shuffle` (MLP down-proj gather) is **298.438 MiB/layer** against the V edge's **31.000 MiB/layer**, a 9.6Ã ratio, and on busiest-single-link 6.250 MiB vs 0.500 MiB, 12.5Ã. That whole load lives inside the 249.6 ms baseline. If 31 MiB of link-hops cost 20 ms, `60_shuffle` alone would cost ~190 ms. (One verifier restated the reductio using injectionÃfanout and got the opposite ordering; that proxy undercounts long-hop, low-sharer edges like `60_shuffle` and is not link occupancy. Both verifiers agree on the link-byte-hop numbers themselves to three decimals.)

**The honest summary: the ring is not the problem, and it has no waste left to recover. Whatever the +20 ms is, it is on-chip capacity or landing work, and the way to shrink it is a smaller destination â or no destination at all.**

---

## V edge

**Turn it off.** Not because it is broken â it is correct and byte-optimal â but because **SenDNN does not put V on LX**, and our baseline already matched SenDNN. The edge inverted a correct arrangement.

Read SenDNN's own AV BMM descriptor, `.../sendnn_granite_antoni_20260725/runs/full_40_layer_sdsc_attribution_20260725_1615/perfdsc_debug/execute_itr0/sdsc/bmm_1-BMM_1.json`, `labeledDs_`:

| idx | name | role | placement |
|---|---|---|---|
| 0 | `bmm_1-actAttnHeadBreak-VirtualReshape_out` | INPUT (softmax probs, P) | **LX**, `lxSize_=524288` |
| 1 | `bmm_1-wtAttnHeadBreak-VirtualReshape_out` | KERNEL (**V**) | **HBM**, `hbmStartAddress_=2097152`, `hbmSize_=8192`, `lxSize_` unset |
| 2 | `bmm_1-BMM_1_out` | OUTPUT | **LX**, `lxSize_=131072` |

Invariant across layer templates: `bmm_3` (layer 1) and `bmm_79` (layer 39, `hbmStartAddress_=2736128`) are identical. And there is no fourth possibility: of the **51** `*LxRelayout*.json` files in that dump, exactly **three** have a 32-way replicated destination (32768 â 1048576) â `BatchMatMulV2_QC_{3,12,21}_inpLds_1`, sources `mul_6_out` / `mul_17_out` / `mul_435_out`. `mul_6_out` is confirmed by name as the kernel operand of `bmm` (Q@K^T). **Those are K. Nothing in the dump relayouts V.**

SenDNN's policy at the AV BMM, in one sentence: *V streams from HBM as the matmul kernel operand at 8192 B per step (which is exactly the weight-stream overlap it is good at), and the softmax probabilities stay on chip.* Our graph maps onto that 1:1 â buf28 = 524288 B/core = SenDNN's P; buf31 = 131072 = SenDNN's output; our AV BMM splits `mb` 32-way, one head per core (`origsdsc_debug_30_batchmatmul.json numWkSlicesPerDim_ = {"in":1,"out":1,"mb":32,"x":1,"y":1}`), same as SenDNN's `bmm_1`.

### Why the +20 ms is expected, not a bug

We asked the allocator to hold a 1 MiB replicated copy of V in a 1.59 MiB budget, on top of everything else the attention block needs. It did what we asked and evicted the 512 KB tensor SenDNN keeps resident. The edge also **removes nothing**: `cmp` shows `origsdsc_debug_17_ReStickifyOpHBM.json` is byte-identical between the two runs (md5 `dbcb8205e8d0b7bf0ac36f159d9a7b1e`), and the node list goes 62 â 64 with `28_identity` + `29_shuffle` inserted and everything downstream renumbered +2. It is purely additive.

### And the ceiling is below the noise floor anyway

buf29 is `is_graph_output=True`, so the HBM **write** survives via the output clone (`allocator.py:598-618`, comment: *"an output clone that still writes HBM once"*). A perfect V edge can only ever save the **read-back**: 32768 B Ã 32 cores Ã 40 layers = 40 MiB â ~0.4 ms at 9.3 Âµs/MiB, or ~1.5 ms if GQA redundancy means each of the 4 query heads sharing a KV head re-fetches. Against a Â±1.2 ms reproducibility floor and a ~4 ms trust threshold, **a perfect V edge is unmeasurable.** V is simply a small tensor. This conclusion is independent of the spill.

### The change

```diff
  # invocation, not a file edit. scripts/run_integ.sh:26  REINTERP="${12:-}"
- run_integ.sh <name> 5 1 16 2 1 0 0 1 0.2 0 buf29
+ run_integ.sh <name> 5 1 16 2 1 0 0 1 0.2 0 ""
```
This drops buf29 from `SPYRE_RELAYOUT_ORACLE_REINTERPRET_OUTPUT_CLONE_BUFFERS` (bound in `torch_spyre/_inductor/config.py`, consumed at `torch_spyre/_inductor/scratchpad/allocator.py:598-618`). buf29 reverts to `reject_reason="graph output is a ReinterpretView"`, stays in HBM, buf28/buf25 return to LX.

**This is not a forecast â `puremerge_5x` *is* that configuration**, measured at 249.574 ms.

**Do not** change `relayout_oracle_compact_gqa`'s producer-side split of buf29 (`torch_spyre/_inductor/work_division.py`, `{token_sym: 8, output_sym: 4}`) â that is the V-projection work division and is unrelated.

**Not recommended, recorded for the flash / KV-carousel track only:** a head-local destination (262,144 B/core, fan-out 8, no spill) is expressible by changing the oracle clause at `torch_spyre/_inductor/spyre_kernel.py:150-160` â but note it has **seven** conditions, including `producer_map == compact_v_planner_order`, which any rewrite must preserve. Worth ~0.4â1.5 ms. Below the floor. Don't build it.

### Bookkeeping fix (do this regardless â it is the actual bug)

Any parser of `STCDP_FINAL_TRANSFER` must dedupe before summing:

```diff
- wire = sum(int(f["wire_bytes"]) for f in transfers)          # this is DELIVERED bytes
+ injected  = sum({f["key"]: int(f["wire_bytes"]) for f in transfers}.values())
+ delivered = sum(int(f["wire_bytes"]) for f in transfers)
+ linkhops  = sum(int(f["wire_bytes"]) * hops(f) for f in dedupe_by_key(transfers))
```
Or just read `entries=` / `deliveries=` off the `STCDP_FINAL_BEGIN` line (`transfer_compute.cpp:466-475`). **The P01âP14 catalog was built by summing `wire_bytes` and therefore over-ranks high-fan-out edges by their fan-out. Re-derive it before any further edge is chosen.**

Also add `REINTERP` and `COMPACT` to `contract.txt` â it is currently byte-identical between the two runs and does not record which buffers were in the allowlist, so the A/B is not self-describing. (And the two runs are not quite single-variable: buf10 also changed verdict, from `"core div mismatch"` to `"no room on scratchpad"`. Harmless â buf10 is non-resident in both â but worth knowing.)

---

## K^T edge

**Land the dim-4 bridge as a correctness fix. Do not spend a device slot on it. It does not deliver the K^T LX edge, and three premises in the brief are contradicted by the artifacts.**

### What is actually broken

**(a) `slot` is not an operand index.** In `_map_core_id_to_wk_slice_dims` the loop is `for device_dim, slot in per_device_dim.items()` â `slot` is the *slice index* along the unmapped device dim. Core `"0"` is `{'0':0,'4':0}` and is skipped by the `if int(slot) != 0` guard; core `"1"` is `{'0':0,'4':1}`, the first with a nonzero slot, so it raises. The failing argument is **arg 0 â buf18 itself.**

**(b) The failing op is not a BMM.** Exactly 17 `sdsc_N.json` files (`sdsc_0..16`) are written before the throw. OpSpec 17 is the *new shuffle*:

```
runs/kt_restickify_lx_1x/cache/en/cengxb2usiw6dml5eflf3dty2dkccdgf5dmnrpdfxn7hz3i4cwmg.py:692
  OpSpec(op='shuffle', num_cores_override=32, dim_labels_override=['mb','x','out'],
         iteration_space {c0:(512,4), c1:(8,8), c2:(128,1)},
         args=[ buf18                                     alloc={'lx':32768} device_size=[512,2,8,64] splits={'0':8,'4':4},
                __spyre_lx_relayout_destination__:buf18   alloc={'lx':0}     device_size=[512,2,8,64] splits={'0':4,'4':8} ])
```

**(c) `SPYRE_LX_ALLOW_RESTICKIFY_READ=1` does NOT put K^T on chip.** The very next spec is unchanged:

```
:722  OpSpec(op='ReStickifyOpHBM',
         in  __spyre_lx_relayout_destination__:buf18  alloc={'lx': 0}         [512,2,8,64]
         out (unnamed)                                alloc={'pool': 6291456} [128,8,8,64])
:745  batchmatmul reads its K^T operand from {'pool': 6291456}
```
versus baseline `runs/puremerge_5x/cache/je/cjeexkgitxkz5fi6m5o5bfdc56eyodmhewg5zi3s4icjk6m5zveo.py:692-714` (`ReStickifyOpHBM` pool 6291456 â pool 5242880; BMM reads pool 5242880). **The flag moves only the restickify's input from pool to LX and inserts one extra shuffle. The K^T HBM round-trip is intact.**

### What dim 4 is

The shuffle's own `iteration_space` settles it without inference: `{c0:(512,4), c1:(8,8), c2:(128,1)}` with `dim_labels_override ['mb','x','out']` says `mb`(512) is split 4 and `x`(8) is split 8 â which is exactly `destination_device_dim_splits {'0':4,'4':8}`. Therefore **device dim 0 â mb(512), device dim 4 â x(8) = the KV-head axis**, and since `device_dim_to_sdsc_dim = {'2': x, '1': out, '0': mb}` with `device_size[2] == 8`, **dim 4 must resolve to SDSC dim `"2"`.** Byte checks close on both sides: 64 tok Ã 2 KV-heads Ã 128 Ã 2 B = 32768 B = 128 tok Ã 1 KV-head Ã 128 Ã 2 B; and 8Ã4 = 4Ã8 = 32 cores.

### The real bug: two mappers disagree about the same allocation

`_map_device_dim_splits`' compact-GQA branch in `torch_spyre/_inductor/codegen/superdsc.py` guards only on `relayout_oracle_compact_gqa and resolved_device_dim not in device_dim_to_sdsc_dim and set(device_dim_to_sdsc_dim) == {"0","1","2"}` â **no `dim_order` clause** â and therefore *already* maps 4 â `"2"` for this exact allocation's `core_splits`. Its twin in `_map_core_id_to_wk_slice_dims` adds `and [str(dim) for dim in dim_order] == ["x","out","in"]`, and our arg's `dim_order` is `["x","out","mb"]`. **Same allocation, same `device_dim_to_sdsc_dim`, opposite verdicts.** And `SPYRE_RELAYOUT_ORACLE_COMPACT_GQA=1` with `COMPACT="${13:-buf18,buf29}"` is already exported on every run (`scripts/run_integ.sh:71`, `:28`) â so the splits side is live today and the ownership side throws.

### The exact predicate, behind a flag

```diff
--- torch_spyre/_inductor/config.py   (after the relayout_oracle_compact_gqa_buffers block)
+# Test-only. The compact-GQA axis bridge in superdsc's ownership mapper is
+# guarded on the ["x","out","in"] label triple, so it does not fire for the
+# pre-transpose K activation buf18, whose normalized SDSC view is
+# ["x","out","mb"].  _map_device_dim_splits already resolves that allocation's
+# planner axis 4 to physical axis 2; this lifts the ownership mapper to agree.
+relayout_oracle_kt_restickify_axis_bridge: bool = (
+    os.environ.get("SPYRE_RELAYOUT_ORACLE_KT_RESTICKIFY_AXIS_BRIDGE", "0") == "1"
+)
```

```diff
--- torch_spyre/_inductor/codegen/superdsc.py
--- inside _map_core_id_to_wk_slice_dims, immediately AFTER the shipped compact-GQA
--- block that ends with  resolved_device_dim = {"3": "1", "4": "2"}.get(...)
--- and BEFORE the `if (` that opens the P06 qk_axis_bridge case:
+            if (
+                _spyre_config.relayout_oracle_compact_gqa
+                and _spyre_config.relayout_oracle_kt_restickify_axis_bridge
+                and allocation_name
+                in ("buf18", "__spyre_lx_relayout_destination__:buf18")
+                and resolved_device_dim == "4"
+                and resolved_device_dim not in device_dim_to_sdsc_dim
+                and set(device_dim_to_sdsc_dim) == {"0", "1", "2"}
+                and [str(dim) for dim in dim_order] == ["x", "out", "mb"]
+            ):
+                # buf18 is Granite's pre-transpose K.  The planner numbers axes
+                # on the pre-normalization device layout; SDSC normalization
+                # drops singletons and emits [512, 2, 8, 64], so the planner's
+                # KV-head axis 4 is physical device axis 2 (label ``x``).  Same
+                # 4 -> 2 identification the shipped compact-GQA bridge above
+                # makes for the ["x","out","in"] view, and the one
+                # _map_device_dim_splits already applies to these core_splits.
+                resolved_device_dim = "2"
```

No change in `_map_device_dim_splits`. With the env var unset this is a no-op, so nothing else can regress.

Why the predicate cannot collide: the three shipped bridges own `["x","out","in"]` (compact-GQA, 3 mapped dims), `["x","in","y","mb"]` (P06, 4 mapped dims), and `["in"]` pinned to buf0/buf6 (LM head). `["x","out","mb"]` is disjoint from all three, and `allocation_name` pins it to buf18. (Note: `set(device_dim_to_sdsc_dim) == {"0","1","2"}` pins **three mapped** device dims, not a 4-coordinate layout â `device_size` has four entries and entry 3, the 64-wide stick axis, is folded into `out` and never appears in the map.)

**Deliberately not proposed:** widening the shipped line to `in (["x","out","in"], ["x","out","mb"])`. Shorter, but it silently re-scopes a bridge already used by the compact-GQA V path and by buf66.

### Why the bridge still does not get us P01

Two barriers sit downstream and neither is addressed by any axis mapping:

1. `runs/kt_restickify_lx_1x/allocations.jsonl`, buf66 (the restickify output = K^T): `"reject_reason": "op not allowed"`, `"address": null`. From `allocator.py`: `if op is None or not self._op_output_good_for_lx_reuse(op): return "op not allowed"`, and `restickify` is simply absent from `OP_OUTPUT_GOOD_FOR_LX_REUSE` in `torch_spyre/_inductor/scratchpad/utils.py` (which lists: max, amax, maximum, sum, clone, exp, sub, mul, mean, add, rsqrt, neg, mm, bmm, batched_matmul, div, realdiv, expand, silu).
2. `RESTICKIFY_OP = "ReStickifyOpHBM"` is a hardcoded constant (`torch_spyre/_inductor/constants.py:19`) assigned unconditionally at `torch_spyre/_inductor/spyre_kernel.py:1298`. The DeepTools HBMâLx promotion (`dsm/workOptimizer/baseOptimizer/lxopt.cpp:3781-3803`, `if (inpLxOpted & outLxOpted) { ... ReStickifyOpHBM -> ReStickifyOpLx; }`) lives in the DSM/perfDsc pipeline. Our path is SDSC â `subprocess.run(["dxp_standalone", "-d", output_dir])` (`torch_spyre/execution/async_compile.py:85`); grep of `p13/deeptools/dxp/*.cpp` for `ReStickify` returns **zero** hits. **That pass is unreachable from torch-spyre.**

And the strongest signal of all: **SenDNN runs `lxopt` and declines the promotion.** In its dump: `ReStickifyOpHBM` Ã 12, `ReStickifyOpLx` Ã 0, `ReStickifyOpLxD` Ã 0, `ReStickifyOpWithPTLx` Ã 0, `APEOpLX` Ã 0, `STCDPOpLx` Ã 158. Its K transpose is folded into the K-projection BMM's *output descriptor* (`bmm{,_2,_78}-wtAttnHeadBreak-VirtualReshape-Output-Restickify.json`), and the on-chip movement is a **BMM-operand relayout** (`BatchMatMulV2_QC_{3,12,21}_inpLds_1_...-LxRelayout.json` = P01, an `STCDPOpLx`). If we want P01's bytes, **copy that shape**, not `ReStickifyOpLx`.

### The cost of the bridge itself, for completeness

The buf18 shuffle is a pure permutation (`ratio 1, divisor 1`): destination core `8t+h` draws 16384 B from exactly sources `8t+h//2` and `8t+4+h//2`.
- 64 transfers (vs 1024 for the V edge), all within Â±4 ring positions
- injected = delivered = **1.00 MiB/layer** â 40 MiB over 40 layers, **32Ã cheaper than the V edge**
- HBM saved: buf18's producer write + the restickify's read = 2.00 MiB/layer â 80 MiB
- HBM **not** saved: the restickify's write of K^T to pool 6291456 and the BMM's read of it â **that is the entire P01 payload, untouched**

Predicted device delta: **|Î| < 1 ms, sign uncertain.** Below the floor. Verify by compile only.

---

## Order of experiments

**The noise floor, stated once.** The gate is median device kernel ms over 5 requests, 42 kernels each, from `trace/*.pt.trace.json` (210 `cat='kernel'` events per run). Per-request values:

```
puremerge_5x  [250.259, 249.574, 242.975, 249.455, 250.082]  median 249.574
vedge_5x      [269.290, 269.772, 269.476, 268.949, 268.565]  median 269.290   delta +19.716 ms
```
The baseline's own five samples span **7.28 ms**. Two runs of the identical commit gave 246.2 and 249.6 ms. **Anything under 4 ms is a null. Pre-commit to that before running.** Every device experiment below has a predicted effect well above 4 ms, or is explicitly labelled unmeasurable.

**Every run reports both gates from the same run**: token 203 on all 6 requests including warmup, *and* the transport proof. Token 203 alone proves nothing â a silently dropped edge also yields 203.

### Tier 0 â no device time at all (do all of these first)

| # | Do | Cost | Exact observable |
|---|---|---|---|
| **T0.1** | **Revert the V edge.** `REINTERP=""`. | 0 | Not an experiment â `puremerge_5x` is already the measurement: 249.574 ms. Confirm 8 (not 9) `STCDP_FINAL_BEGIN` lines with sources `[buf40,buf43,buf45,buf52,buf56]`. |
| **T0.2** | **Fix the `wire_bytes` parser** (dedupe on `key=`, or read `entries=`/`deliveries=`), then **re-derive the P01âP14 catalog** on the injected basis. | hours | The catalog's edge ranking changes. Any edge whose rank came from fan-out rather than bytes drops. This gates *which* edge is worth building next â do not choose a target from the old catalog. |
| **T0.3** | **Read SenDNN's restickify output addressing**: `coreIdToDsc_` start addresses in `.../sdsc/bmm-wtAttnHeadBreak-VirtualReshape-Output-Restickify.json`. | minutes | Does the K^T transpose output land in LX or HBM? "HBM" in `ReStickifyOpHBM` is an addressing mode, not necessarily a buffer. **Single highest-value read for the P01 direction.** If it lands in HBM, P01's on-chip win is smaller than the catalog implies. |
| **T0.4** | **LX capacity arithmetic for a SenDNN-shaped K^T edge**, before writing any codegen. | minutes | SenDNN's K^T destination is 1,048,576 B/core replicated (`bmm-BMM_1.json` idx1 `mul_6_out`, `lxSize_=1048576`) â the *same* 1 MiB that just spilled buf28. With our 1,625,344 B budget and buf28 (524288) + buf26 (131072) + buf31 (131072) co-live, **it does not fit.** If this arithmetic says "no room", the compile-unblocking work is moot. |
| **T0.5** | **Land the dim-4 bridge** behind `SPYRE_RELAYOUT_ORACLE_KT_RESTICKIFY_AXIS_BRIDGE`, verify **by compile only**. | 0 device | See the acceptance table below. |
| **T0.6** | Add `REINTERP` and `COMPACT` to `contract.txt`. | minutes | Future A/B pairs are self-describing. |

**T0.5 acceptance â what must appear in `sdsc_17.json` / `origsdsc_debug_17_shuffle.json`:**

- `numCoresUsed_ == 32`
- source `numWkSlicesPerDim_` = `{mb:8, x:4, out:1}`; destination = `{mb:4, x:8, out:1}`. **Any `x == 1` on either side means the KV-head axis was folded away and the shuffle is a lie.**
- `coreIdToWkSlice_`: source core `c` â `{mb: c//4, x: c%4, out: 0}`; destination core `c` â `{mb: c//8, x: c%8, out: 0}` â compare element-by-element against `relayout_plans.jsonl` (`source_name=="buf18"`) after applying 0âmb, 4âx
- byte coverage: 32 Ã 32768 B = 1.00 MiB on **both** sides, every one of 512Ã8Ã128 fp16 elements owned exactly once per side
- `STCDP_FINAL_TRANSFER` for `17_shuffle-Relayout`: **~64 entries, `logical_bytes=16384`, `gtr_sharers=1`, ~1.00 MiB injected**

**Failure signatures:** 1024 entries / `gtr_sharers=31` â the ownership map degenerated to dense all-to-all; the bridge compiled but is wrong. 4â`"1"` (out, size 2): `2 % 4 != 0` â expect a DeepTools piece assert, or K split along head_dim â garbage scores. 4â`"0"`: both planner dims land on mb, last write wins, coverage non-exhaustive.
**The dangerous case:** token 203 with a *no-op* shuffle â source and destination maps agree per core, the shuffle moves nothing, and `ReStickifyOpHBM` re-reads correct data regardless. Only the ~64-nonzero-transfer proof rules this out.

### Tier 1 â the one device pair worth running (2 slots)

This exists to answer **one question that gates the entire remaining relayout program**: *can we afford a 1 MiB LX destination at all?* Both runs use `V edge ON`, `lxfrac` lowered from 0.2 so buf28 stays resident. At `DXP_LX_FRAC_AVAIL=0.05`: `round_up(int(2031616 Ã 0.95), 128) = 1,930,112 B > 1,835,008 B` â the V destination and buf28 both fit.

| # | Run | Predicted | What it discriminates |
|---|---|---|---|
| **T1.a** | **Control**: V edge OFF, `lxfrac=0.05`. Everything else = `puremerge_5x`. | ? | **The price of taking LX away from DXP's weight streaming.** This is the number nobody has. If it costs â« 4 ms, the frontend can never buy LX headroom and *every* "put another activation on chip" idea is dead â for free, without building anything. |
| **T1.b** | V edge ON, `lxfrac=0.05`. | ~250 ms if (A); ~269 ms if (B) or (C) | **Spill vs ring/landing.** `T1.b â T1.a` isolates the shuffle's own cost with the spill removed. |

Observables, from the same runs: token 203 Ã 6; median kernel ms from `trace/*.pt.trace.json`; 9 `STCDP_FINAL_BEGIN` lines in T1.b with `29_shuffle-Relayout entries=32 deliveries=1024`; and **`allocations.jsonl` must show buf28 with `address != null` and `reject_reason: null` in both** â if buf28 is still spilled, the experiment did not run and the result means nothing. Predicted effects are ~20 ms, five times the floor.

### Tier 2 â only if Tier 1 says a 1 MiB LX destination is affordable

Build the K^T edge **SenDNN's way**: fold the transpose into the K-projection BMM's output descriptor and make the on-chip movement a BMM-operand `LxRelayout` (`STCDPOpLx`), attached to the consuming node â not a standalone graph node, and not `ReStickifyOpLx`. Note SenDNN's V relayout is named `BatchMatMulV2_QC_3_inpLds_1_...` â *the relayout of input-LDS 1 of the BMM*. Ours became `29_shuffle`, its own node in the chain. That structural difference, not multicast mode, is what is left to chase.

### Never (say so out loud, so nobody re-litigates)

- **Chasing `promoteToMode3` / `STCDP_FORCE_UNICAST_SPLIT` / `forceModeMC`.** Zero bytes recoverable. `hop_Mode3=16 > 12` blocks it by construction; the pass is dead code under `psumRing="dataring"`; and mode 3 has identical total ring bytes anyway. Setting `STCDP_FORCE_UNICAST_SPLIT` can only make the edge worse.
- **Measuring the dim-4 bridge on device.** Predicted `|Î| < 1 ms` against a 4 ms floor. Also confounded: `SPYRE_LX_ALLOW_RESTICKIFY_READ=1` moves buf25/buf37 from 1245184 â 1507328 and flips buf10's verdict, so any measured delta mixes the edge with a global LX repack. (And `kt_restickify_lx_1x` has `iters=1` vs `puremerge_5x`'s `iters=5` â not a valid A/B as it stands.)
- **Building the compact-GQA V destination.** Ceiling ~0.4â1.5 ms. At the floor.
- **Any experiment chosen from the un-corrected P01âP14 catalog.** Fix T0.2 first.

---

## What would tell us to stop

Concrete, pre-committed exit conditions. Each is a single number from a single run.

**Stop the relayout-of-activations program entirely if:**

1. **T1.a (V off, `lxfrac=0.05`) is slower than 249.574 ms by more than 4 ms.** Then LX belongs to DXP's weight streaming, we cannot buy headroom, and no additional activation fits on chip at any price. This kills T1.b, Tier 2, and the whole P01-on-LX direction in one cheap run. Go straight to the ~52 ms compute/weight-stream overlap, which the independent budget analysis says is where the gap actually lives.

2. **T1.b â T1.a â +20 ms** (i.e. removing the spill did not help). Then the cost is the shuffle itself â landing work and/or ring occupancy â and it scales with the destination fan-out. A SenDNN-shaped P01 has the *same* 32-way replicated destination, so **it will cost the same ~20 ms.** Adding all-gathers is net-negative. Stop adding edges; the only remaining lever is work division (see 5).

3. **T0.4 says a 1 MiB replicated K^T destination does not fit in 1,625,344 B** and T1.a says we cannot raise the budget. Then P01 is unreachable on this stack, regardless of `superdsc.py`, the restickify barrier, or `OP_OUTPUT_GOOD_FOR_LX_REUSE`. Stop the compile-unblocking work.

4. **Cumulative measured wins from Tier 1 + Tier 2 stay under 4 ms.** The independent budget analysis puts the activation-relayout ceiling at 4â7 ms of the 56 ms gap. If we have spent the runs and are still inside the noise, the ceiling is real and the remaining 52 ms is not ours to get this way.

**Stop a specific edge if:**

5. **The transfer proof shows the ownership map degenerated** â for the K bridge, 1024 entries / `gtr_sharers=31` where 64 / `gtr_sharers=1` was predicted; for any edge, an injected-byte count that does not equal `entries Ã wire_bytes`. Revert the flag; do not debug on device.

6. **An edge is purely additive** â `cmp` shows no `ReStickifyOpHBM` removed and the node count only grows. That is what happened to the V edge (62 â 64 nodes, `origsdsc_debug_17_ReStickifyOpHBM.json` byte-identical). **Make this a standing pre-flight check: before any device run, diff the node lists and confirm the new edge *deletes* an HBM op.** If it deletes nothing, it cannot win, and no amount of transport tuning changes that.

7. **A relayout exists only to reconcile two work divisions.** The buf18 shuffle is exactly this: the producing `mul` emits 8 token-blocks Ã 4 head-pairs; the consumer wants 4 token-blocks Ã 8 heads. **If the producer were divided the consumer's way, no relayout would be planned at all and the LX residency would be free.** Paying 40 MiB of wire to fix a division mismatch after the fact is strictly worse than not creating it. When you see this shape, stop building the edge and go change the work division.

**And one thing that should never again be a stop-or-go signal:** a `wire_bytes` sum, a `selectedMCMode` value, or a `useUnicast` flag. The first is delivered bytes, the second is a compass bearing, and the third is a result. None of them measures a problem we have.