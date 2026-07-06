export const meta = {
  name: 'survivors-build-fable-orchestrated',
  description: 'Fable-authored, Opus-executed build workflow: S1 cost-model plan-selection + Granite spill audit + S3/S2 backend diffs + flash value oracle, across 3 pod lanes',
  phases: [
    { title: 'Build' },
    { title: 'Verify' },
    { title: 'Synthesize' },
  ],
}

const CONTEXT = `
You are an Opus 4.8 subagent executing precise instructions authored by the Fable orchestrator.
Your final text IS a structured data return. Cite file:line. TAG claims MEASURED/STRUCTURAL/ANALYTICAL/BLOCKED.
Do NOT inflate; a refuted result stated clearly beats a hedge.

=== NORTH STAR ===
Epic torch-spyre#3049: remove non-weight Granite HBM activation spills by making on-chip LX-to-LX (SFP ring)
communication a first-class compiler capability. Production ask: also de-spill a flash-attention prototype.
Device-grounded finding (already established by Fable this session): the two "carousel" RFCs (weight-rotation,
KV-seq-shard) are SHELVED - their motivations are refuted on hardware. We are building the SURVIVORS:
  S1 = ring cost-model correction (Bet 2, branch ah/ring-cost-term) - price per-link TRANSFER COUNT (36 vs 130
       GB/s band), no blocker, 7/7 unit-tested. THE selectability layer.
  S2 = cross-bundle co-bundling (keep producer shard LX-resident into consumer program) - unblocks BOTH flash
       value-correctness AND the attention all-gather; home of the confirmed 130 GB/s uniform-shift transport.
  S3 = P2 overlap-gate extension (overlapInpFetchWithCompute, dsmperf.cpp:~3733-3736: Conv2D/SparseConv2D ->
       matmul/PriOp consumers). Smallest exact backend edit; on S2's critical path.
  S4 = LSE ring-fold reduce lane (Bet 3, Epic Phase 5), math-validated 13/13, wire AFTER S2.

=== ENVIRONMENT (you are on the user's Mac; cwd /Users/adnan/torch-spyre-work) ===
- gh authed: github.com (AdnanHoque) + github.ibm.com (Adnan-Hoque1).
  fetch file: gh api "repos/OWNER/REPO/contents/PATH?ref=REF" --jq '.content' | base64 -d
  (github.ibm.com: prefix with GH_HOST=github.ibm.com)
- Pods (namespace a6-quantization), READ-ONLY unless a task says otherwise:
    oc exec -n a6-quantization POD -- bash -lc 'CMD'
  adnan-clc-spyre-dev-pf  /dev/vfio/25  (S1 cost-model + Granite audit lane)
  adnan-cdx-spyre-dev-pf  /dev/vfio/80  (S2/S3 backend lane; deeptools checkout DIRTY = Codex live work)
  adnan-spyre-dev-pf      /dev/vfio/31  192c (flash lane)
*** HARD SAFETY RAILS ***
  - NEVER git checkout/reset/revert/clean/stash or edit any file in a pod checkout. Read-only: status/log/show/diff/cat/ls/grep.
  - This round: NO AIU device compiles or device runs. Pure-Python and read-only analysis ONLY. Devices are precious.
  - Do not clobber Codex's isolated run roots. If you need to write scratch, use /tmp on the pod or the Mac scratchpad.

=== VERIFIED GROUND TRUTH (this session) ===
- PR1 scatter (Phase 1) MEASURED 1.065x kernel S512. Fresh CLC baseline 12.55 ms/iter clean; relayout-ON
  collectives at Granite scale currently rc=1 FAIL.
- Flash DEV run (today 19:19): 0 ReStickifyOpHBM / 97 ReStickifyOpLx / 32 matmul_operand plans, BUT
  assert_close_skipped=True (PATCH_MODE=no_h2d,skip_cpu_ref) -> value-correctness NOT established (STRUCTURAL only).
  test_flash.py CPU assert is NOT a clean oracle in-env (baseline mismatched 75.1%).
- M4 all-gather-into-KERNEL (CDX): 232 ALLCLOSE-False vs 26 True; 12 self-ring RAS::PCI::BusFence. value-correct
  XOR capacity-safe, never both. Backend wall: runDcgForInputFetchNeighbor pinned DsTypes::INPUT vs operand
  DsTypes::KERNEL; fillDataInfo empty coordinate maps (std::out_of_range map::at).
- Ring physics MEASURED: uniform p->p+1 shift 54/90/130 GB/s @ 4/8/16 MB (R^2>=0.9985); range/scatter relayout
  effective 36 GB/s (R^2>=0.9999); fixed F ~= 7.3 us per STCDP execute; lambda per-hop = 0 modeled, NOT isolable.
  HBM one shared ~170-205 GB/s pipe. Distinguishing var = per-link transfer count (1 vs 4-9), NOT burst size.
- Bet 2 (ah/ring-cost-term) = work_division.py +30/-6 + tests/tensor/test_matmul_split_cost.py (7/7). Base cf67411.

=== REAL Granite-8B S512 matmul iteration spaces (from live work_division debug dumps on CLC) ===
elems_per_stick = 64 (fp16). Cost model gets RAW sizes M_e,N_e,K_e; enumerates splits over divisors of
M_e (rows), n_sticks=N/64, k_sticks=K/64, and batch. max_cores=32.
  buf6  QKV proj:     B=1  M=512  N=6144  K=4096   (n_sticks=96, k_sticks=64)
  buf14 attn scores:  B=32 M=512  N=512   K=128    (n_sticks=8,  k_sticks=2)   [bmm; shared-weight caveat may apply]
  buf24 attn out-proj:B=1  M=512  N=4096  K=4096   (n_sticks=64, k_sticks=64)
  buf33 MLP gate+up:  B=1  M=512  N=25600 K=4096   (n_sticks=400,k_sticks=64)
  buf36 MLP down:     B=1  M=512  N=4096  K=12800  (n_sticks=64, k_sticks=200)
`

const CMP_SCHEMA = {
  type:'object', additionalProperties:false,
  required:['built','method','shapes','headline','divergences','caveats'],
  properties:{
    built:{type:'boolean'},
    method:{type:'string', description:'how the two cost fns were obtained + how enumeration was reproduced'},
    shapes:{type:'array', items:{
      type:'object', additionalProperties:false,
      required:['buf','MNK','main_split','main_cost_us','ring_split','ring_cost_us','diverged','interpretation'],
      properties:{
        buf:{type:'string'}, MNK:{type:'string'},
        main_split:{type:'string'}, main_cost_us:{type:'number'},
        ring_split:{type:'string'}, ring_cost_us:{type:'number'},
        diverged:{type:'boolean'}, interpretation:{type:'string'} } } },
    headline:{type:'string', description:'does the ring-aware term change plan selection on Granite? on which ops? toward what?'},
    divergences:{type:'array', items:{type:'string'}},
    caveats:{type:'array', items:{type:'string'}},
  },
}
const AUDIT_SCHEMA = {
  type:'object', additionalProperties:false,
  required:['spills','summary_table_md','covered','blocked','future','notes'],
  properties:{
    spills:{type:'array', items:{
      type:'object', additionalProperties:false, required:['edge','comm_class','status','note'],
      properties:{ edge:{type:'string'}, comm_class:{type:'string'}, status:{type:'string', enum:['covered','blocked','future','weight-out-of-scope']}, note:{type:'string'} } } },
    summary_table_md:{type:'string'}, covered:{type:'integer'}, blocked:{type:'integer'}, future:{type:'integer'},
    notes:{type:'array', items:{type:'string'}},
  },
}
const DIFF_SCHEMA = {
  type:'object', additionalProperties:false,
  required:['target','files','exact_change','diff_md','eligibility_or_risk','build_test','confidence'],
  properties:{
    target:{type:'string'}, files:{type:'array', items:{type:'string'}},
    exact_change:{type:'string'}, diff_md:{type:'string', description:'the concrete proposed diff (unified) or precise edit'},
    eligibility_or_risk:{type:'string'}, build_test:{type:'string', description:'how to build+test on the pod (do NOT run it)'},
    confidence:{type:'string', enum:['high','medium','low']},
  },
}
const ORACLE_SCHEMA = {
  type:'object', additionalProperties:false,
  required:['approach','harness','validity','run_cmd','expected_signal','confidence'],
  properties:{
    approach:{type:'string'}, harness:{type:'string', description:'the oracle code, inline'},
    validity:{type:'string', description:'why this oracle is trustworthy where test_flash.py CPU assert is not'},
    run_cmd:{type:'string', description:'the exact PATCH_MODE-off run to get a real value number (spec only; do NOT run)'},
    expected_signal:{type:'string'}, confidence:{type:'string', enum:['high','medium','low']},
  },
}

phase('Build')
log('Build: 5 Fable-specified tasks across the 3 pod lanes (pure-Python + read-only; zero device compiles)')

const T1 = agent(
  CONTEXT +
  '\n\n=== TASK T1 (Lane CLC) - S1 cost-model plan-selection comparison ===\n' +
  'GOAL: answer the crux of S1 - does the ring-aware cost term (ah/ring-cost-term) change WHICH work-division ' +
  'split the planner selects on Granite S512 real matmul shapes, vs main? Pure Python, NO device.\n' +
  'METHOD (faithful):\n' +
  '1. Fetch _matmul_split_cost from BOTH main and ah/ring-cost-term (gh api, work_division.py) and the shared ' +
  'constants block. main uses hbm_us = (B*M*K + B*K*N + B*M*N)*2/(204.8*1000)*max(1.0,max(m,n)/8); ring-cost-term ' +
  'uses per-operand _cohort_penalty (multicast cap 204.8/130, scatter cap 204.8/36). Everything else identical: ' +
  'compute_us (with pt_eff sqrt derate, _PEAK_MACS_US_CORE=(98.304e12/2/32)/1e6), psum_us=max(0,k-1)*B*M*N*1.4e-4, ' +
  'target_m_us, batch_penalty=b**1.4.\n' +
  '2. Reproduce the planner enumeration EXACTLY as _cost_model_matmul_planner does: for each shape enumerate ' +
  'b over divisors(batch), m over divisors(M_e), n over divisors(n_sticks), k over divisors(k_sticks), skip if ' +
  'b*m*n*k>32, cost via _matmul_split_cost((B_total,b),(M_e,mm),(N_e,nn),(K_e,kk),32); argmin is the selected split. ' +
  'N_e=n_sticks*64, K_e=k_sticks*64.\n' +
  '3. Run on ALL 5 real Granite shapes (buf6/14/24/33/36 above). For each report the selected (b,m,n,k) and cost ' +
  'under main vs ring, whether they diverge, and interpret (e.g. "ring term makes wide-N multicast plan win => ' +
  'planner now prefers the on-chip-broadcast-friendly split"). Write the script to the Mac scratchpad and run it.\n' +
  'Deliver CMP object. Be explicit if the term does NOT change selection (that is a real, publishable finding too).',
  { label:'T1:cost-plan-selection', phase:'Build', schema: CMP_SCHEMA, effort:'high' }
).then(r=>({key:'T1', r}))

const T2 = agent(
  CONTEXT +
  '\n\n=== TASK T2 (Lane CLC) - Epic Phase 2 Granite spill audit (READ-ONLY) ===\n' +
  'GOAL: inventory the non-weight HBM activation spills in the Granite S512 block and classify each by comm class ' +
  '(scatter/broadcast/multicast/gather/all-gather/reduce/all-reduce), marking covered/blocked/future/weight-out-of-scope.\n' +
  'METHOD: read EXISTING SDSC/summary artifacts on CLC (do NOT compile). Roots: ' +
  '/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/* (paired relayout on/off ' +
  'summaries, backend_plans, sdsc_*.json, sdsc_artifact_summary.py). Compare relayout OFF vs ON: OFF baseline had ' +
  '5 ReStickifyOpHBM / 15 occurrences; ON collectives had 1/3 + 1 backend plan (rc=1). Identify which HBM ' +
  'restickify rows are non-weight (in scope) vs weight (out of scope), which PR1 scatter removed, and which remain ' +
  'blocked (LX-capacity / KERNEL-placement wall / reduce-lane). Use the sdsc_artifact_summary.py tool if helpful ' +
  '(read its logic; run it read-only against existing artifacts only). Deliver AUDIT object with a markdown table.',
  { label:'T2:spill-audit', phase:'Build', schema: AUDIT_SCHEMA, effort:'high' }
).then(r=>({key:'T2', r}))

const T3 = agent(
  CONTEXT +
  '\n\n=== TASK T3 (Lane CDX) - S3 overlap-gate exact diff (READ-ONLY analysis) ===\n' +
  'GOAL: produce the exact, minimal diff to extend overlapInpFetchWithCompute from Conv2D/SparseConv2D consumers ' +
  'to matmul/PriOp, so a seam-transparent STCDPOpLx move overlaps with a matmul (the sole open perf gate for the ' +
  'collectives lane). READ (read-only) the deeptools checkout on CDX: ' +
  '/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/repos/deeptools-comms-clean ' +
  '(also mirror at /Users/adnan/torch-spyre-work/deeptools-comms-collectives). Look at dsm/dsmperf.cpp ~3720-3770 ' +
  '(the parRelayoutStcdp = opName==STCDPOpLx detection at ~3762 and the Conv-only gate at ~3733-3736), ' +
  'dsm/dsmds.cpp isPriOp (~27), dsm/graphOptimizer.cpp assignCanOverlapInpFetch (~18491) and isSrclayoutChangeStcdp. ' +
  'Confirm eligibility holds for seam-transparent moves (layoutDimOrder unchanged). Produce the DIFF object with a ' +
  'concrete unified diff, the eligibility proof, and the build+test command (do NOT build). Flag any risk that ' +
  'matmul input-fetch differs structurally from Conv such that the hook needs more than a consumer-list widening.',
  { label:'T3:overlap-gate-diff', phase:'Build', schema: DIFF_SCHEMA, effort:'high' }
).then(r=>({key:'T3', r}))

const T4 = agent(
  CONTEXT +
  '\n\n=== TASK T4 (Lane CDX) - S2 co-bundling / KERNEL-placement minimal change spec (READ-ONLY) ===\n' +
  'GOAL: pin the single highest-leverage backend change that moves BOTH the M4 all-gather-into-KERNEL and the ' +
  'attention all-gather forward, plus the frontend co-bundling prerequisite. READ (read-only) on CDX deeptools ' +
  'the DCG region: runDcgForInputFetchNeighbor (hard-pinned DsTypes::INPUT), fillDataInfo (empty coordinate maps -> ' +
  'std::out_of_range), dsc2Pcfg.cpp buildLxNeighborRingTransfers, L3DlOpsScheduler.cpp ' +
  'populateMatmulOperandBroadcastStickRingTransfers, ddc_fold.cpp. Also read the Codex status docs on the artifact ' +
  'branch (gh api AdnanHoque/torch-spyre ref ah/comms-collectives, docs/results/granite_e2e/comms_collectives_20260705/' +
  '{two_stage_matmul_operand_broadcast_plan,matmul_operand_broadcast_loop_scoped_checkpoint,backend_allgather_diagnostic_checkpoint}_20260705.md). ' +
  'State: (a) the exact INPUT->KERNEL generalization + which coordinate maps must be populated/consumed; (b) whether ' +
  'it is a patch or a redesign; (c) the frontend co-bundling requirement (mul(K) re-reads K from HBM in the consumer ' +
  'bundle -> co-bundle it into the QK^T program so LX persists). Deliver DIFF object (exact_change = the minimal ' +
  'backend change; diff_md = the precise edit sketch; build_test = spec only).',
  { label:'T4:cobundling-spec', phase:'Build', schema: DIFF_SCHEMA, effort:'high' }
).then(r=>({key:'T4', r}))

const C1 = agent(
  CONTEXT +
  '\n\n=== TASK C1 (Lane DEV) - production flash CLEAN value oracle (design only, NO device run) ===\n' +
  'GOAL: the production flash de-spill is structurally done but value-correctness is unproven because the in-env ' +
  'test_flash.py CPU assert is dirty (baseline mismatched 75.1%) and the passing runs use PATCH_MODE=skip_cpu_ref. ' +
  'Design a TRUSTWORTHY value oracle so the next device run can actually validate flash. Read test_flash.py ' +
  '(GH_HOST=github.ibm.com gh api repos/aviros/test-spyre-scripts/contents/test_flash.py). It is B=1,H=32,D=128,' +
  'Lq=Lk=4096, causal, fp16, and builds sparse real_max/denominator via amax(dim=-1). Provide: (a) a standalone ' +
  'numpy/torch fp32 reference (single-pass softmax attention with the SAME causal mask + scaling 1/sqrt(sqrt(D))) ' +
  'that is independent of the dirty in-env CPU ref, computed on host in fp32 and cast for comparison; (b) why it is ' +
  'trustworthy (fp32 carry, deterministic, matches the flash_cpu math exactly but without the device round-trip ' +
  'contamination); (c) the EXACT PATCH_MODE-off run command to capture device output tensors + compare to the ' +
  'oracle at atol/rtol matched to fp16 (do NOT run it); (d) the amax(dim=-1) reduce-wall note: this is the ' +
  'optimize_restickify blocker, tied to coarse_tile Stage-1 Lk-tiling (cf67411), NOT to the restickify-on lane. ' +
  'Deliver ORACLE object.',
  { label:'C1:flash-oracle', phase:'Build', schema: ORACLE_SCHEMA, effort:'high' }
).then(r=>({key:'C1', r}))

const built = (await parallel([()=>T1,()=>T2,()=>T3,()=>T4,()=>C1])).filter(Boolean)
const byKey = {}; for (const b of built) byKey[b.key]=b.r
log('Build complete: '+built.map(b=>b.key).join(', '))

// Verify the two load-bearing outputs: T1 (does the model really flip selection?) and T3+T4 (are the diffs correct?)
const VERIFY_SCHEMA = {
  type:'object', additionalProperties:false,
  required:['target','holds','errors_found','corrections','confidence'],
  properties:{
    target:{type:'string'}, holds:{type:'string', enum:['holds','holds-with-corrections','refuted']},
    errors_found:{type:'array', items:{type:'string'}}, corrections:{type:'array', items:{type:'string'}},
    confidence:{type:'string', enum:['high','medium','low']},
  },
}
phase('Verify')
log('Verify: adversarial re-derivation of T1 selection + T3/T4 diff correctness')
const V1 = agent(
  CONTEXT +
  '\n\n=== VERIFY T1: adversarially re-derive the cost-model plan-selection result ===\n' +
  'The claim under review:\n'+JSON.stringify(byKey['T1'],null,1)+
  '\nIndependently re-implement BOTH cost functions from the branch source (do not trust the claimant\'s transcription), ' +
  're-run the enumeration on at least buf6, buf33, buf36, and check: (1) is the enumeration faithful to ' +
  '_cost_model_matmul_planner (divisors of n_sticks/k_sticks, not raw N/K; the >max_cores skip; the never-trade-down ' +
  'rule)? (2) does the selected split actually differ between main and ring where claimed? (3) any shape where the ' +
  'claimant missed a divergence or fabricated one? A wrong faithfulness assumption REFUTES the headline.',
  { label:'V1:verify-cost', phase:'Verify', schema: VERIFY_SCHEMA, effort:'high' }
).then(r=>({key:'V1', r}))
const V2 = agent(
  CONTEXT +
  '\n\n=== VERIFY T3+T4: adversarially check the backend diffs are correct and minimal ===\n' +
  'T3 (overlap gate):\n'+JSON.stringify(byKey['T3'],null,1)+'\n\nT4 (co-bundling):\n'+JSON.stringify(byKey['T4'],null,1)+
  '\nRe-read the cited deeptools regions (read-only) and check: is the overlap-gate consumer-list widening actually ' +
  'sufficient, or does matmul input-fetch structurally differ (PT operand fetch vs Conv IFM) so the hook would ' +
  'misfire? For T4, is the INPUT->KERNEL change a real fix or does the co-bundling prerequisite dominate (i.e. the ' +
  'backend change is necessary-but-not-sufficient)? Flag over-optimism. Return VERIFY (target=both).',
  { label:'V2:verify-diffs', phase:'Verify', schema: VERIFY_SCHEMA, effort:'high' }
).then(r=>({key:'V2', r}))
const verifs = (await parallel([()=>V1,()=>V2])).filter(Boolean)
const vByKey={}; for (const v of verifs) vByKey[v.key]=v.r

phase('Synthesize')
log('Synthesize: fold results into an S1-lane status + updated next actions')
const digest =
  '## T1 cost-plan-selection\n'+JSON.stringify(byKey['T1'],null,1)+
  '\n## V1 verify\n'+JSON.stringify(vByKey['V1'],null,1)+
  '\n## T2 spill-audit\n'+JSON.stringify(byKey['T2'],null,1)+
  '\n## T3 overlap-gate\n'+JSON.stringify(byKey['T3'],null,1)+
  '\n## T4 co-bundling\n'+JSON.stringify(byKey['T4'],null,1)+
  '\n## V2 verify-diffs\n'+JSON.stringify(vByKey['V2'],null,1)+
  '\n## C1 flash-oracle\n'+JSON.stringify(byKey['C1'],null,1)
const report = await agent(
  CONTEXT +
  '\n\n=== RESULTS DIGEST ===\n'+digest+
  '\n\n=== SYNTHESIZE (markdown for the human lead) ===\n' +
  'Write a tight status update for the CLC/S1 lane and the CDX/DEV setup. Sections:\n' +
  '1. **S1 verdict**: does the ring cost term change Granite plan selection? (apply V1 corrections; if refuted or ' +
  'no-op, say so plainly and what that means for shipping S1).\n' +
  '2. **Phase-2 spill audit table** (from T2): the classified inventory, covered/blocked/future counts.\n' +
  '3. **S3 overlap-gate**: the exact diff + whether V2 confirmed it is sufficient or flagged a structural gap.\n' +
  '4. **S2 co-bundling**: the minimal backend change + the co-bundling prerequisite + patch-vs-redesign verdict.\n' +
  '5. **Flash oracle** (C1): the trustworthy oracle + the exact next device run to actually validate flash value.\n' +
  '6. **Gated device next-steps**: the specific device runs (Granite S512 plan-pick confirmation on CLC; flash ' +
  'PATCH_MODE-off value run on DEV) to trigger deliberately next, each with its command and acceptance metric ' +
  '(counter/kernel-trace, never wall alone).\n' +
  'Be decisive, cite file:line, keep the honesty ledger discipline.',
  { label:'synthesize:s1-status', phase:'Synthesize', effort:'high' }
)
return { cost: byKey['T1'], verify_cost: vByKey['V1'], audit: byKey['T2'], overlap: byKey['T3'],
         cobundle: byKey['T4'], verify_diffs: vByKey['V2'], oracle: byKey['C1'], report }
