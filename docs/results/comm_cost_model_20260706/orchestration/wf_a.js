export const meta = {
  name: 'carousel-comms-recon-gate',
  description: 'Recon + RFC analytical go/no-go for Granite HBM-spill / ring-carousel epic across all comm classes and bets',
  phases: [
    { title: 'Recon' },
    { title: 'Gate' },
    { title: 'Verify' },
    { title: 'Synthesize' },
  ],
}

// ---------------------------------------------------------------------------
// SHARED CONTEXT (agents start fresh; embed everything load-bearing)
// ---------------------------------------------------------------------------
const CONTEXT = `
YOU ARE ONE AGENT IN AN ORCHESTRATED RECON of a torch-spyre (IBM Spyre AIU) compiler effort.
Your final text IS a data return, not a human message. Be precise, cite file paths + line numbers,
and TAG every claim as MEASURED / STRUCTURAL / ANALYTICAL / BLOCKED / UNVERIFIED. Do NOT inflate.

=== THE GOAL ===
Epic github.com/torch-spyre/torch-spyre#3049 "Remove Avoidable Granite HBM Activation Spills With
On-Chip LX Communication Collectives". Remove non-weight HBM activation round-trips in Granite by
making LX-to-LX (on-chip, over the SFP core ring) communication a first-class compiler capability.
Production ask: remove HBM spills in Granite AND in a flash-attention prototype (test_flash.py),
replacing them with LX communication collectives. Weight restickifies are OUT of scope.

=== ENVIRONMENT (you are on the user's Mac, cwd /Users/adnan/torch-spyre-work) ===
- gh authed: github.com (user AdnanHoque) and github.ibm.com (user Adnan-Hoque1).
  Fetch a file: gh api "repos/AdnanHoque/torch-spyre/contents/PATH?ref=REF" --jq '.content' | base64 -d
  For github.ibm.com prefix the call with: GH_HOST=github.ibm.com gh api "repos/ORG/REPO/contents/PATH" ...
- Pods (namespace a6-quantization), read-only access via:
    oc exec -n a6-quantization POD -- bash -lc 'CMD'
  Pods & AIU devices:
    adnan-cdx-spyre-dev-pf  -> /dev/vfio/80, 128 cores, home /home/adnan-cdx/codex-isolated/
    adnan-spyre-dev-pf      -> /dev/vfio/31, 192 cores, home /home/adnan/codex-isolated/
    adnan-clc-spyre-dev-pf  -> /dev/vfio/25, 128 cores, home /home/adnan/codex-isolated/
  *** SAFETY (hard rules) ***
  - Pods have ACTIVE Codex work with intentionally-dirty git checkouts. NEVER run git checkout/reset/
    revert/clean/stash or edit any pod file. Read-only only: git status/log/show/diff, cat, ls, grep.
  - DO NOT run AIU device compiles or device runs in this phase (no torch.compile on spyre, no dxp runs).
    Reading EXISTING run artifacts/logs is fine. The AIU devices are precious (a wedge needs pod recreate).

=== BRANCHES / CHECKOUTS ===
- Torch artifact/progress branch: AdnanHoque/torch-spyre:ah/comms-collectives (github.com).
  Local mirror: /Users/adnan/torch-spyre-work/torch-spyre-comms-collectives (on ah/comms-collectives).
- Deeptools experiment branch: Adnan-Hoque1/deeptools:ah/comms-collectives (github.ibm.com).
  Local mirror: /Users/adnan/torch-spyre-work/deeptools-comms-collectives (on ah/comms-collectives, dirty).
- Torch main: /Users/adnan/torch-spyre-work/torch-spyre (on main).
- Our ring-analysis docs live at commit 89a0d4525bfb17ed119ae10f6ba740c11a9e7491, dir carousel-rfc-impl/
  (AdnanHoque/torch-spyre). Files: CAROUSEL_ROI_PLAN, RING_SPEED, UNIFORM_SHIFT_RING_SPEED, M0_CLOSURE,
  M0_GROUNDTRUTH, PRODUCTION_READINESS, RING_AWARE_MECHANISMS, RING_MECHANISMS_IMPLEMENTED, LX_TO_LX_SPEED,
  DLDSC_LX_SPEED, CONTIGUOUS_RING_MOVE, CODEX_COMMS_FOLD (.md each).
- Codex status docs at ah/comms-collectives, dir docs/results/granite_e2e/comms_collectives_20260705/:
  collective_class_status, matmul_operand_broadcast_status, backend_allgather_diagnostic_checkpoint,
  matmul_operand_broadcast_loop_scoped_checkpoint, two_stage_matmul_operand_broadcast_plan,
  m4_backend_checkpoint, cross_pod_collectives_checkpoint (_20260705*.md). Handoff:
  docs/results/granite_e2e/comms_collectives_handoff_20260705.md.
- Torch inductor relayout code (paths per handoff): torch_spyre/_inductor/{lx_relayout.py,
  layout_allgather_restickify.py, config.py, scratchpad/allocator.py, codegen/{bundle.py,compute_ops.py,
  superdsc.py}, spyre_kernel.py}. Frontend env flags: SPYRE_LX_PLANNER_RELAYOUT[_COLLECTIVES]
  [_LAYOUT_ALLGATHER_RESTICKIFY][_MATMUL_OPERAND_CONTRACT][_RESTICKIFY_OUTPUTS], DXP[_BACKEND]_LX_FRAC_AVAIL.

=== KEY GROUND-TRUTH ALREADY ESTABLISHED (verify, do not just repeat) ===
- PR1 scatter (Epic Phase 1): MEASURED ~1.065x kernel on Granite S512 (14.7258 -> 13.8213 ms/iter),
  wall 1.041x. DLDSC-metadata insertion (Approach C). Removed non-weight ReStickifyOpHBM rows.
- Flash restickify-on: STRUCTURAL only. relayout OFF=32 ReStickifyOpHBM/0 LX; ON=0 HBM/32 ReStickifyOpLx/
  32 matmul_operand_broadcast plans. NOT value-correct (successful DEV run used PATCH_MODE=no_h2d,skip_cpu_ref;
  unpatched value mismatch 31.5% on a dirty 75.1% baseline oracle).
- all-gather-into-KERNEL (matmul_operand_broadcast, Phase 4): THE shared blocker. Value-correct form is
  capacity-UNSAFE (full-resident alloc fails "unable to allocate final matmul operand LX region on core 0");
  capacity-safe loop-scoped ring-into-KERNEL is value-WRONG (949/1024 mismatch, self-ring RAS::PCI::BusFence).
  Backend region: runDcgForInputFetchNeighbor hard-pinned to DsTypes::INPUT while operand is DsTypes::KERNEL;
  fillDataInfo throws std::out_of_range map::at because coordinate maps empty (coordDims=0). Files:
  L3DlOpsScheduler.cpp, dsc2Pcfg.cpp, ddc_fold.cpp, dsc2.cpp.
- Coordinate fidelity gap: backend count-based grouping (cores 0,1,..,7) vs true strided coord map
  (0,4,8,..,28). Count/group metadata too weak; true coordinate stride must drive grouping.
- Bet 2 ring cost term: branch ah/ring-cost-term (LOCATE it). DONE + unit-tested 7/7. NOT device-measured.
  Needs: (a) edge cost carries TRUE coordinate stride not cohort count; (b) same-core-is-free-local-copy
  refinement; (c) device-measure on Granite S512.
- Bet 3 LSE ring-fold (Phase 5 reduce lane): MATH-VALIDATED 13/13, NOT compiler-wired. Neighbor
  reduce-scatter of (m,l,O) partials with LSE-combine. Reduce lane is uncontested.
- Bet 1 flash-in-a-loop: BLOCKED UPSTREAM at optimize_restickify. The wall is the amax(dim=-1) sparse
  reduction (multi-stick -> single-stick real_max/denominator), an ARITHMETIC fan-in, NOT layout relayout.
  Codex's restickify-on does NOT help (REFUTED). Real dependency = Stage-1 reduction-dim-tiling
  (coarse_tile design doc on main) surfacing Lk as a tiling level.
- MEASURED ring physics (verify vs RING_SPEED / UNIFORM_SHIFT_RING_SPEED docs): per-link TRANSFER COUNT
  (not burst size) is the variable. Uniform p->p+1 shift 54 GB/s @4MB, 90.5 @8MB, 130 @16MB (1 transfer/
  link). All-to-all scatter 34-36 GB/s (4-9 transfers/link). Streaming aggregate ~250 GB/s. R^2>=0.998.
  Same-core replicas = FREE local copy. Burst enlargement bought a measured null (1.0x).

=== GRANITE-8B DIMS ===  H=32, H_kv=8, G=4, D=128, b=2 (fp16), P=32 cores, ~40 layers, K=N~=4096.
=== test_flash.py ===  B=1,H=32,D=128,Lq=Lk=4096; q_block=Lq/4; kv_block=Lk/1 (NO Lk tiling: "current
  limitation disallows coarse tiling in Lk"); work_div {H:4,Lq:8,Lk:8}. flash_spyre builds sparse
  real_max/denominator via amax(dim=-1) (the FIXME). Value assert atol/rtol 0.1.

=== THE TWO RFCs (attached, being evaluated) ===
- WEIGHT CAROUSEL: for M-split (token-split) prefill matmul Y[S,N]=X[S,K]@W[K,N], instead of replicating
  W P times through LPDDR, partition W into P column tiles, load each once into its home core, ROTATE tiles
  around the SFP ring over P steps. Every weight byte crosses DRAM once (xP DRAM reduction). Output lands
  M-split (seam-transparent). Rotate N-tiles not K-tiles (keeps K-loop core-local, no cross-core reduce).
  Group-carousel radius r in divisors(P): r=P is today's full replication, r=1 full carousel; DRAM traffic
  r*|W|. Compute-bound gate: rho >= C*b*P/(2S). Perf model per step (full overlap):
  t_c=2*(S/P)*K*(N/P)/C, t_x=K*(N/P)*b/rho, T ~= t_prologue + P'*max(t_c,t_x) + (P'-1)*lambda + t_epi.
  Rides SHUFFLE (PR #2789). NO new backend op proposed.
- KV CAROUSEL: shard KV cache along SEQUENCE across all P cores (block-cyclic, B_kv=64), channel-affine.
  Per decode step: broadcast Q (~8KB), local flash-decode pass per core -> (A,l,m) partials, MERGE via
  LSE ring-fold (=Bet 3). BW ceiling H_kv*beta_chan -> P*beta_chan (>=4x at Granite). Merge on critical
  path, L-independent: budget n_layers*T_merge <= eps*T_step. Merge variants A(linear fold)/B(reduce-scatter
  over heads + all-gather -> head-split, chains into k_fast out-proj)/C(recursive halving). Payload/hop ~16.5KiB.
- SHARED Phase-0 SHUFFLE probes P1-P5: P1 SHUFFLE accepts arbitrary rotation delta=+/-1 (day-scale gate);
  P2 async issue + completion token (overlap); P3 LX->LX without LPDDR bounce (DOA gate); P4 bidirectional;
  P5 measured rho + per-hop lambda. NOTE Codex's measured shift/all-to-all numbers largely ANSWER P5.
`

const STATUS_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['workstream','summary','evidence','key_files','blockers','next_changes','verified','unverified_or_refuted'],
  properties: {
    workstream: { type: 'string' },
    summary: { type: 'string', description: '4-8 sentence current-state, tagged MEASURED/STRUCTURAL/ANALYTICAL/BLOCKED' },
    evidence: {
      type: 'object', additionalProperties: false,
      required: ['measured','structural','analytical','blocked'],
      properties: {
        measured: { type: 'array', items: { type: 'string' } },
        structural: { type: 'array', items: { type: 'string' } },
        analytical: { type: 'array', items: { type: 'string' } },
        blocked: { type: 'array', items: { type: 'string' } },
      },
    },
    key_files: { type: 'array', items: {
      type: 'object', additionalProperties: false, required: ['path','repo','note'],
      properties: { path:{type:'string'}, repo:{type:'string'}, note:{type:'string'} } } },
    blockers: { type: 'array', items: {
      type: 'object', additionalProperties: false, required: ['what','where','unblock_change'],
      properties: { what:{type:'string'}, where:{type:'string'}, unblock_change:{type:'string'} } } },
    next_changes: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      required: ['change','files','pod_lane','device_needed','effort','confidence'],
      properties: {
        change:{type:'string'}, files:{type:'string'},
        pod_lane:{type:'string', enum:['CDX','DEV','CLC','none/off-device']},
        device_needed:{type:'boolean'}, effort:{type:'string', enum:['S','M','L','XL']},
        confidence:{type:'string', enum:['high','medium','low']} } } },
    verified: { type:'array', items:{type:'string'}, description:'doc claims you independently confirmed by reading code/artifacts' },
    unverified_or_refuted: { type:'array', items:{type:'string'} },
  },
}

const WORKSTREAMS = [
  { key:'W1_scatter', title:'PR1 scatter (Epic Phase 1, DONE/measured)',
    focus:'Verify the 1.065x claim + the DLDSC-metadata (Approach C) lowering path. Locate the exact torch inductor code that emits dl-dsc coordinate metadata and the deeptools resident-scatter insertion. Confirm which explicit ReStickifyOpHBM rows were removed and that remaining HBM restickifies are weight-shaped (out of scope). Where is the PR1 production branch (pr-lx-relayout-scatter) vs the artifact branch?' },
  { key:'W2_flash', title:'Flash-attention de-spill (THE production ask, test_flash.py)',
    focus:'Establish the exact spill/blocker chain for test_flash.py. Confirm: the 32 ReStickifyOpHBM->32 ReStickifyOpLx conversion is STRUCTURAL only; value-correctness gate is open; the amax(dim=-1) sparse real_max/denominator reduction (optimize_restickify, multi-stick->single-stick) is the real wall and is a REDUCE not a layout relayout; and that "current limitation disallows coarse tiling in Lk" (kv_block=Lk/1). Read the flash status in Codex docs + our carousel-rfc-impl. Identify the minimal path to zero HBM spills that is also VALUE-CORRECT. Locate optimize_restickify and the coarse_tile Stage-1 reduction-dim-tiling design doc on torch main.' },
  { key:'W3_allgather_kernel', title:'All-gather-into-KERNEL / matmul_operand_broadcast (Phase 4, the shared blocker)',
    focus:'Pin down the DCG coordinate-metadata->physical-placement wall. Confirm runDcgForInputFetchNeighbor is pinned to DsTypes::INPUT while operand is DsTypes::KERNEL; confirm fillDataInfo empty-coordinate-map std::out_of_range; read the two_stage_matmul_operand_broadcast_plan + loop_scoped_checkpoint + backend_allgather_diagnostic docs. Read the deeptools local mirror files (dsc2.cpp, dsc2Pcfg.cpp, ddc_fold.cpp, L3DlOpsScheduler.cpp, inputNeighFetchOp.cpp) to confirm the exact functions/lines. State the single highest-leverage backend change and whether it is a patch or a redesign.' },
  { key:'W4_costterm', title:'Bet 2 ring cost term (Epic Phase 6 costing)',
    focus:'LOCATE branch ah/ring-cost-term (git branch -a across local checkouts; gh api to list branches on AdnanHoque/torch-spyre). Read the cost-term code + the 7/7 unit tests. Confirm it prices per-link contention using the 130-vs-36 GB/s band. Confirm it currently uses cohort COUNT not true coordinate STRIDE (the fidelity gap). Specify the exact code change to carry true stride and the same-core-free refinement, and how to device-measure whether the planner now PICKS the on-chip move on Granite S512.' },
  { key:'W5_lsefold', title:'Bet 3 LSE ring-fold (Epic Phase 5 reduce lane)',
    focus:'Find the math-validated (13/13) LSE-combine merge implementation (numpy/reference). Confirm the fold operator m=max, A=e^(m1-m)A1+e^(m2-m)A2, l=e^(m1-m)l1+e^(m2-m)l2. Determine what compiler wiring is missing to realize it as STCDPOpLx move + SFP lse_combine per hop. Confirm it depends on the same DCG coordinate plumbing as W3 for the MOVE half, but the REDUCE half (arithmetic) is net-new and uncontested. Is there an SFP elementwise op that can host lse_combine?' },
  { key:'W6_physics', title:'Ring physics substrate + SHUFFLE P1-P5 probe status',
    focus:'Read RING_SPEED, UNIFORM_SHIFT_RING_SPEED, LX_TO_LX_SPEED, CONTIGUOUS_RING_MOVE, RING_MECHANISMS_IMPLEMENTED, M0_GROUNDTRUTH, M0_CLOSURE from carousel-rfc-impl. VERIFY the measured numbers (54/90.5/130 GB/s uniform shift; 34-36 all-to-all; ~250 aggregate; R^2>=0.998; same-core free; burst-null). For each Phase-0 probe P1(rotation delta=+/-1), P2(async+token), P3(LX->LX no LPDDR bounce), P4(bidirectional), P5(rho,lambda): state whether existing artifacts already ANSWER it, and if not, the exact microbenchmark to run (which pod, which harness from PR #2789). Extract the numeric per-hop lambda if measured. Locate the SHUFFLE primitive + #2789 microbench harness on the pods/checkouts.' },
  { key:'W7_spillaudit', title:'Granite spill audit (Epic Phase 2)',
    focus:'Assemble the Granite-block non-weight HBM activation spill inventory. Find existing before/after SDSC artifacts (Granite S512) in the run roots / artifact branch. Classify each non-weight HBM spill by comm class (scatter/broadcast/multicast/gather/all-gather/reduce/all-reduce) and mark covered/blocked/future. Identify which spills PR1 already removed and which remain (and why: LX-capacity, backend gap, reduce-lane). Produce the audit as a table in your summary. Do NOT run new Granite compiles; use existing artifacts + SDSC dumps.' },
  { key:'W8_classstatus', title:'Comm-class status: broadcast/multicast/gather/reduce/all-reduce (Phases 3,5,7)',
    focus:'Read collective_class_status_20260705.md + the Epic taxonomy. For EACH of the 7 classes (scatter, broadcast, multicast, gather, all-gather/replicate, reduce, all-reduce) state: current support status, the representation-layer capability (Codex: dxp_standalone cardinality probes all pass; DLDSC can express all four cardinalities; dense-coordinate-map hard rule), the backend gap, and whether a Granite/attention edge exists that needs it. Confirm "representation layer is NOT the bottleneck; placement is."' },
]

// ---------------------------------------------------------------------------
// PHASE 1: RECON (parallel readers; barrier before analytical gates)
// ---------------------------------------------------------------------------
phase('Recon')
log('Recon: 8 parallel readers establishing verified ground truth (read-only, no device runs)')
const recon = await parallel(WORKSTREAMS.map((w) => () =>
  agent(
    CONTEXT +
    '\n\n=== YOUR WORKSTREAM: ' + w.title + ' ===\n' + w.focus +
    '\n\nMETHOD: read the real code/docs/artifacts (local mirrors first, gh api for branch-only files, ' +
    'read-only oc exec for live pod state). Confirm or refute each relevant ground-truth claim by ' +
    'actually opening the file. Tag every finding by evidence tier. For next_changes, be specific about ' +
    'files and whether an AIU device is needed and which pod lane (CDX synthetic/all-gather, DEV flash, ' +
    'CLC Granite/cost-term) fits. Return the STATUS object.',
    { label: 'recon:' + w.key, phase: 'Recon', schema: STATUS_SCHEMA, effort: 'medium', model: 'fable' }
  ).then((r) => ({ key: w.key, title: w.title, status: r }))
))
const reconOk = recon.filter(Boolean)
const reconDigest = reconOk.map((r) =>
  '### ' + r.title + '\n' + JSON.stringify(r.status, null, 1)
).join('\n\n')
log('Recon complete: ' + reconOk.length + '/' + WORKSTREAMS.length + ' workstreams returned')

// ---------------------------------------------------------------------------
// PHASE 2: ANALYTICAL GATE (needs full recon -> barrier justified)
// ---------------------------------------------------------------------------
const GATE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['question','verdict','numbers','key_finding','high_value_variation','assumptions_used','weakest_links'],
  properties: {
    question: { type:'string' },
    verdict: { type:'string', enum:['GO','NO-GO','CONDITIONAL-GO'] },
    numbers: { type:'array', items:{
      type:'object', additionalProperties:false, required:['name','value','basis'],
      properties:{ name:{type:'string'}, value:{type:'string'}, basis:{type:'string', description:'MEASURED input, formula, or assumption used'} } } },
    key_finding: { type:'string', description:'6-12 sentences, the crux' },
    high_value_variation: { type:'string', description:'the concrete near-term variation informed by the verified findings (what to actually build first)' },
    assumptions_used: { type:'array', items:{type:'string'} },
    weakest_links: { type:'array', items:{type:'string'}, description:'the assumptions/numbers most likely wrong' },
  },
}

const GATES = [
  { key:'G1_weight_carousel', q:'Weight-Carousel "afternoon check": is the perf win real for Granite-8B prefill?',
    task:'Using the VERIFIED measured ring physics (uniform-shift rho up to 130 GB/s @16MB, ~250 aggregate, per-hop lambda if measured) and Granite dims (K=N~=4096, P=32, b=2, S in {512,1024,2048,4096}), compute the RFC section-4 model: replication cost P*|W|/beta_agg vs carousel(r) cost. Fill the compute-bound gate rho >= C*b*P/(2S) with a defensible C (per-core sustained fp16 FLOP/s: derive from any measured matmul throughput you can find in the artifacts, else state the assumption and its sensitivity). Give the crossover S where carousel is compute-bound vs ring-bound. State the UNCONDITIONAL DRAM-byte win (xP) separately from the CONDITIONAL wall-clock win. Assess whether Granite prefill today even SELECTS token-split (if the cost model prunes it, the carousel value is partly indirect/seam). Verdict GO/NO-GO/CONDITIONAL with the r that wins.' },
  { key:'G2_kv_carousel', q:'KV-Carousel crossover: does seq-sharded flash-decode + LSE ring-fold beat head-split decode?',
    task:'Confirm/refute the head-split baseline presumption (H_kv=8 active of P=32 -> 25% aggregate ceiling) from recon W6/W7/W2. Compute T_carousel(L) vs T_baseline(L) with measured beta_chan and the L-independent fixed cost T_bcast+T_merge. Solve L_min (crossover). Compute the merge-latency budget n_layers*T_merge <= eps*T_step at 40 layers with measured per-hop lambda for variants A/B/C; identify which variant survives and whether the fused-hop backend ask (A5) is forced. State the asymptotic gain x(P/H_kv)/u and the capacity headroom. Verdict + the merge variant to build.' },
  { key:'G3_variations', q:'What are the highest-value RFC VARIATIONS given our findings (the DCG-KERNEL wall, same-core-free, the amax-reduce wall)?',
    task:'Synthesize across ALL recon. The clean carousels ride all-gather-into-KERNEL (W3), which is the shared BLOCKER. Identify variations that ship value WITHOUT waiting on that wall: e.g. (a) does the KV-carousel LSE-fold (Bet 3) sidestep the DsTypes::KERNEL placement wall because it reduces activation partials rather than gathering a KERNEL operand? (b) does the group-carousel r-knob let us ship the unconditional DRAM win with fewer/cheaper hops now? (c) is the flash de-spill better served by the reduce-lane (amax + LSE fold) than by layout relayout? (d) can the seam-transparent M-split property deliver a Granite MLP-chain win independent of the ring transport? Rank 3-4 concrete variations by (value x unblocked-ness), each with the exact first implementation step and pod lane.' },
  { key:'G4_flash_despill', q:'The single most direct VALUE-CORRECT path to zero HBM spills in test_flash.py?',
    task:'Lay out the exact blocker chain for test_flash.py and the minimal unblocking change. Decide among: (a) Codex restickify-on made value-correct (what value fix is needed, and does cross-bundle source provenance / mul(K) co-bundling apply here?); (b) Bet 1 flash-in-a-loop via Lk reduction-dim tiling (the coarse_tile Stage-1 path) to eliminate the amax(dim=-1) multi-stick gather; (c) the KV-carousel flash-decode LSE-fold design. Give the ordered plan with the first patch, its files, the pod lane (DEV owns flash), and a value oracle that is actually clean (test_flash.py CPU assert is NOT clean in the dirty env - specify a trustworthy oracle). Verdict on the recommended path.' },
]

phase('Gate')
log('Gate: 4 analytical RFC go/no-go models over the full recon digest')
const gatesRaw = await parallel(GATES.map((g) => () =>
  agent(
    CONTEXT +
    '\n\n=== FULL RECON DIGEST (verified ground truth from 8 workstreams) ===\n' + reconDigest +
    '\n\n=== YOUR ANALYTICAL QUESTION: ' + g.q + ' ===\n' + g.task +
    '\n\nShow the arithmetic. Separate MEASURED inputs from ASSUMPTIONS. If a needed number is missing, ' +
    'state the assumption and the sensitivity of the verdict to it. Return the GATE object.',
    { label: 'gate:' + g.key, phase: 'Gate', schema: GATE_SCHEMA, effort: 'high', model: 'fable' }
  ).then((r) => ({ key: g.key, q: g.q, gate: r }))
))
const gates = gatesRaw.filter(Boolean)

// ---------------------------------------------------------------------------
// PHASE 3: ADVERSARIAL VERIFY of each gate's numbers
// ---------------------------------------------------------------------------
const VERIFY_SCHEMA = {
  type:'object', additionalProperties:false,
  required:['target','holds','errors_found','corrections','confidence'],
  properties:{
    target:{type:'string'},
    holds:{type:'string', enum:['holds','holds-with-corrections','refuted']},
    errors_found:{type:'array', items:{type:'string'}},
    corrections:{type:'array', items:{type:'string'}},
    confidence:{type:'string', enum:['high','medium','low']},
  },
}
phase('Verify')
log('Verify: adversarial skeptic per gate (recompute the numbers, refute the verdict)')
const verified = await parallel(gates.map((g) => () =>
  agent(
    CONTEXT +
    '\n\n=== CLAIM UNDER REVIEW ('+g.key+') ===\n' + JSON.stringify(g.gate, null, 1) +
    '\n\nYou are an adversarial verifier. RE-DERIVE the key numbers independently from the measured physics ' +
    'and Granite dims. Attack the weakest links. A model error > ~20% or an unsupported assumption flips ' +
    'the verdict. Default to skepticism: if a number cannot be grounded in a MEASURED input or a stated ' +
    'formula, flag it. Return the VERIFY object.',
    { label:'verify:'+g.key, phase:'Verify', schema: VERIFY_SCHEMA, effort:'high', model: 'fable' }
  ).then((v)=>({ key:g.key, verify:v }))
))
const verifyByKey = {}
for (const v of verified.filter(Boolean)) verifyByKey[v.key] = v.verify

// ---------------------------------------------------------------------------
// PHASE 4: SYNTHESIS -> roadmap + prioritized pod-assigned target list for Phase 2
// ---------------------------------------------------------------------------
phase('Synthesize')
log('Synthesize: fold recon + gates + verifications into a prioritized, pod-assigned roadmap')
const gateDigest = gates.map((g) =>
  '### GATE '+g.key+': '+g.q+'\n'+JSON.stringify(g.gate,null,1)+
  '\nADVERSARIAL VERIFY: '+JSON.stringify(verifyByKey[g.key]||{note:'missing'},null,1)
).join('\n\n')

const roadmap = await agent(
  CONTEXT +
  '\n\n=== RECON DIGEST ===\n' + reconDigest +
  '\n\n=== ANALYTICAL GATES + ADVERSARIAL VERIFICATIONS ===\n' + gateDigest +
  '\n\n=== YOUR JOB: SYNTHESIZE THE ORCHESTRATION ROADMAP (markdown) ===\n' +
  'Write a single, self-contained markdown report for the human lead. Sections:\n' +
  '1. **Verified current-state matrix**: a table of the 7 comm classes x {support status, evidence tier, ' +
  'blocker, Epic phase}. Plus a row per our-3-bets and per RFC.\n' +
  '2. **RFC go/no-go**: Weight Carousel and KV Carousel verdicts WITH the corrected numbers (apply the ' +
  'adversarial corrections; if a gate was refuted, say so). State the unconditional vs conditional wins plainly.\n' +
  '3. **High-value variations** (from G3): the 3-4 things actually worth building first, ranked by ' +
  'value x unblocked-ness, each mapped to an Epic phase and comm class.\n' +
  '4. **The production flash de-spill plan** (from G4): the ordered, value-correct path with the first patch.\n' +
  '5. **PHASE-2 TARGET LIST (the deliverable that drives the next workflow)**: a numbered, dependency-ordered ' +
  'list of concrete work items. For each: {id, title, Epic-phase/comm-class, exact files, device_needed, ' +
  'POD LANE (CDX=synthetic/all-gather/probes, DEV=flash, CLC=Granite/cost-term), effort S/M/L/XL, ' +
  'depends-on, acceptance metric (counter-first, kernel-trace not wall), confidence}. Assign work so the ' +
  'three pod lanes run in parallel with minimal cross-dependency. Mark which items are UNBLOCKED-NOW vs ' +
  'gated on the DCG-KERNEL backend fix.\n' +
  '6. **Honesty ledger**: MEASURED vs STRUCTURAL vs ANALYTICAL vs BLOCKED, one line each, no inflation.\n' +
  'Be decisive and specific. This report is the single source of truth for the build phase.',
  { label:'synthesize:roadmap', phase:'Synthesize', effort:'high', model: 'fable' }
)

return {
  recon: reconOk.map((r)=>({ key:r.key, title:r.title, status:r.status })),
  gates: gates.map((g)=>({ key:g.key, gate:g.gate, verify: verifyByKey[g.key]||null })),
  roadmap,
}
