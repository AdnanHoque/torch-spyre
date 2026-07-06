export const meta = {
  name: 'g2-lse-reduce-lane',
  description: 'Architect + implement G2: the LSE ring-fold reduce lane (Epic Phase 5) - frontend routing + contract + LSE-combine reference, costed via the G3 comm cost model',
  phases: [{ title: 'Architect' }, { title: 'Implement' }, { title: 'Validate' }, { title: 'Synthesize' }],
}

const CTX = `
You are an Opus subagent under a Fable orchestrator building G2: the LSE ring-fold REDUCE lane for torch-spyre.
Final text is a structured data return. Cite file:line. Tag MEASURED / ANALYTICAL / STRUCTURAL / BLOCKED. No inflation.

=== WHY G2 (the uncontested lane) ===
Epic torch-spyre#3049 Phase 5 = REDUCE / all-reduce arithmetic collectives. Codex's work is ALL operand-broadcast
(all_gather_replicate); he has ZERO reduce/fold. The reduce lane is the ONLY path to (a) a value-correct online-softmax
Lk reduction once Lk is tiled, and (b) the sequence-sharded DECODE merge (KV-carousel). Bet 3 (the LSE ring-fold) is
MATH-VALIDATED 13/13 but NOT compiler-wired. G3 (the comm cost model, just landed) can now COST it.

=== THE LSE-COMBINE OPERATOR (math, MEASURED-validated 13/13) ===
Fold two flash partials over disjoint key sets. Each partial = (m=running max, l=denominator mass, A=unnormalized out):
  m  = max(m1, m2)
  A  = e^(m1-m)*A1 + e^(m2-m)*A2
  l  = e^(m1-m)*l1 + e^(m2-m)*l2
Associative + commutative up to rounding; fp32 carry; static ring order => bit-deterministic run-to-run. Normalize once
at the end (O = A/l). This is the fold-operator generalization of k_fast's '+' reduce: frontend owns the operator,
backend owns realization.

=== G3 (now available) prices the reduce lane ===
torch_spyre/_inductor/comm_cost.py (branch ah/comm-cost-model-g3): _reduce_schedules(payload, P) returns
{reduce_plain_ring: (P-1) hops, reduce_tree_fold: ceil(log2 P) hops}. cost()/comm_edge_cost_us() price a CommEdge with
is_reduction=True or comm_class in {reduce, all_reduce}. MEASURED result to reproduce: P=32, ~16.5 KiB payload ->
plain_ring ~= 230 us (98% F-fixed, F=7.3us/execute) vs tree_fold ~= 37 us. So the reduce is F-DOMINATED -> MINIMIZE HOP
COUNT (tree fold / reduce-scatter over only the split dim, not a full 32-core ring). Payload is tiny + L-INDEPENDENT.

=== FRONTEND CODE (branch ah/comm-cost-model-g3, checkout /Users/adnan/torch-spyre-work/torch-spyre-comms-collectives) ===
- lx_relayout.py: _classify_communication_class already names COMM_CLASS_REDUCE / COMM_CLASS_ALL_REDUCE (~lines 55-62,
  270-272). BUT there is a blanket producer_has_partial SKIP (~lines 440-446) that DROPS partial-reduction producers
  before they can be classified -> the reduce lane is currently UNREACHABLE. Removing/rerouting that skip is the unblock.
- layout_allgather_restickify.py has contract builders (e.g. make_matmul_operand_allgather_contract ~:79). Add a sibling
  make_lse_ring_fold_contract carrying the LSE-combine semantics + communication_class=reduce (or a reduce_scatter tag).
- comm_edge_from_plan (lx_relayout.py, just added) already maps communication_class -> a G3 CommEdge with is_reduction.
- LAYOUT DIVIDEND: choose the reduce axis = reduce-over-Lk-WITHIN-head, scatter-the-result-OVER-heads => the normalized
  output lands HEAD-SPLIT == the k_fast out-projection input layout => ZERO relayout at attention->out-proj. Bank it.

=== SCOPE / WHAT IS UNBLOCKED vs GATED ===
UNBLOCKED now (do it): the frontend classification/routing to reach the reduce lane; make_lse_ring_fold_contract; the
LSE-combine REFERENCE (pure-Python/numpy, asserting associativity + determinism + fp32 carry); wiring the COST via G3
(reduce_tree_fold, F-dominated, minimize hops); the layout-dividend axis choice. GATED (design + scope only, do NOT try
to run): the backend SFP lse_combine primitive (a new op deeptools must lower), device value-correctness, and the S2
co-bundling prerequisite for the move-half. The DEV device is DEGRADED (~60s/sync) -> NO device runs. Pure-Python +
frontend code-read + code edits only. Honor the pre-publish convention (private branch, no push/PR).

=== ENV ===
Mac, cwd /Users/adnan/torch-spyre-work. Checkout torch-spyre-comms-collectives (branch ah/comm-cost-model-g3, where G3
lives). gh api for source. system python3 for pure-Python tests (comm_cost is zero-torch; load it via importlib +
sys.modules registration to bypass the torch-importing package __init__).
`

const ARCH_SCHEMA = { type:'object', additionalProperties:true, required:['area','summary'],
  properties:{ area:{type:'string'}, summary:{type:'string'}, spec:{type:'string'}, key_points:{type:'array', items:{type:'string'}} } }
const IMPL_SCHEMA = { type:'object', additionalProperties:true, required:['built','test_results'],
  properties:{ built:{type:'boolean'}, files:{type:'array', items:{type:'string'}}, branch:{type:'string'}, test_results:{type:'string'}, api_summary:{type:'string'}, gated:{type:'array', items:{type:'string'}}, notes:{type:'array', items:{type:'string'}} } }
const VAL_SCHEMA = { type:'object', additionalProperties:true, required:['summary','verdict'],
  properties:{ summary:{type:'string'}, verdict:{type:'string'}, cases:{type:'array', items:{type:'object', additionalProperties:true}} } }

phase('Architect')
log('Architect G2: frontend routing + LSE-combine + cost/topology via G3 (parallel)')
const A1 = agent(CTX + '\n\n=== A1 — FRONTEND ROUTING (read the real code) ===\n' +
  'Read lx_relayout.py: the producer_has_partial SKIP (~440-446), _classify_communication_class (226-286, the ' +
  'COMM_CLASS_REDUCE/ALL_REDUCE branch), and the LXRelayoutPlan fields. Specify EXACTLY: (1) how to remove/reroute the ' +
  'producer_has_partial skip so partial-reduction producers reach _classify_communication_class instead of being ' +
  'dropped; (2) how the classifier tags a reduce edge (existing COMM_CLASS_REDUCE, or a new reduce_scatter tag) and ' +
  'what distinguishes reduce from gather; (3) how the new make_lse_ring_fold_contract (sibling in ' +
  'layout_allgather_restickify.py) carries the LSE-combine semantics + communication_class; (4) confirm comm_edge_from_plan ' +
  '(just added) prices it via G3 is_reduction. Cite file:line. Return ARCH (area="routing").',
  { label:'A1:routing', phase:'Architect', schema: ARCH_SCHEMA, effort:'high' }).then(r=>({k:'A1',r}))
const A2 = agent(CTX + '\n\n=== A2 — LSE-COMBINE OPERATOR + REFERENCE ===\n' +
  'Design the LSE-combine fold operator as a frontend-owned op composition (the arithmetic half of the reduce, which the ' +
  'backend must eventually lower to an SFP primitive). Specify: the op signature ((m,l,A) x (m,l,A) -> (m,l,A)); the ' +
  'exact fp32-carry arithmetic (m=max, A/l rescaled by e^(m_i-m)); how it composes STCDPOpLx move + local combine per ' +
  'hop; and a PURE-PYTHON/numpy REFERENCE that asserts: associativity, commutativity-up-to-rounding, run-to-run bit ' +
  'determinism (static ring order), fp32 carry vs fp16 drift, and equivalence to single-pass softmax attention over the ' +
  'union of key sets. This reference is the value oracle for the eventual backend op. Return ARCH (area="operator").',
  { label:'A2:operator', phase:'Architect', schema: ARCH_SCHEMA, effort:'high' }).then(r=>({k:'A2',r}))
const A3 = agent(CTX + '\n\n=== A3 — COST + TOPOLOGY via G3 + LAYOUT DIVIDEND ===\n' +
  'Using G3 (comm_cost._reduce_schedules), specify the reduce topology the frontend should emit: (1) confirm P=32 tiny ' +
  'payload -> plain_ring ~230us (98% F) vs tree_fold ~37us, so the fold MUST minimize hop count (tree / reduce-scatter ' +
  'over ONLY the actually-split dim, never a full 32-core ring); give the G3 CommEdge(s) that price the flash Lk-reduce ' +
  '(P = the Lk-split factor) and the decode merge (P = seq-shard count); (2) the LAYOUT DIVIDEND: prove that ' +
  'reduce-over-Lk-within-head + scatter-result-over-heads lands the output head-split = the k_fast out-proj input ' +
  '(zero relayout), and state where the reduce-axis choice is made; (3) the crossover: at what Lk-tile count T does ' +
  'tiling+reduce beat the untiled single-block stream (recall untiled ~15us; the reduce fixed cost sets the floor). ' +
  'Return ARCH (area="cost").',
  { label:'A3:cost', phase:'Architect', schema: ARCH_SCHEMA, effort:'high' }).then(r=>({k:'A3',r}))
const arch = (await parallel([()=>A1,()=>A2,()=>A3])).filter(Boolean)
const aByKey={}; for(const a of arch) aByKey[a.k]=a.r
const archDigest = 'A1 routing:\n'+JSON.stringify(aByKey['A1'],null,1)+'\n\nA2 operator:\n'+JSON.stringify(aByKey['A2'],null,1)+'\n\nA3 cost:\n'+JSON.stringify(aByKey['A3'],null,1)

phase('Implement')
log('Implement the UNBLOCKED frontend reduce lane + LSE reference + G3 cost wiring; scope the gated backend')
const impl = await agent(CTX + '\n\n=== ARCHITECTURE ===\n'+archDigest +
  '\n\n=== IMPLEMENT (UNBLOCKED parts only, on branch ah/comm-cost-model-g3) ===\n' +
  'In torch-spyre-comms-collectives (branch ah/comm-cost-model-g3, already checked out): implement the frontend, ' +
  'unblocked parts per the architecture:\n' +
  '1. lx_relayout.py: reroute the producer_has_partial skip so partial-reduction edges reach the reduce classification ' +
  '(guard it behind a flag if it risks regressing existing behavior, mirroring the SPYRE_COMM_COST_SEAM pattern). Add ' +
  'make_lse_ring_fold_contract in layout_allgather_restickify.py carrying communication_class=reduce + LSE semantics.\n' +
  '2. A PURE-PYTHON LSE-combine reference module (e.g. torch_spyre/_inductor/lse_fold_ref.py, stdlib+optional numpy, NO ' +
  'torch) with the fold operator + the value oracle, and tests (tests/tensor/test_lse_fold.py) asserting associativity, ' +
  'determinism, fp32-carry, and equivalence to single-pass softmax. RUN these tests with system python3 (pure-Python).\n' +
  '3. Wire the COST: show comm_edge_from_plan/comm_cost pricing a reduce edge (tree_fold vs plain_ring) for the flash ' +
  'Lk-reduce and decode merge; add a test asserting tree_fold is selected and ~37us at P=32.\n' +
  'py_compile everything. Commit on the branch (git -c commit.gpgsign=false commit -s; NO push/PR). Report files, ' +
  'test_results, api_summary, and the GATED remainder (backend SFP lse_combine primitive, device value-correctness, S2 ' +
  'co-bundling move-half). Return IMPL.',
  { label:'implement:g2', phase:'Implement', schema: IMPL_SCHEMA, effort:'high' })

phase('Validate')
log('Validate: run the reference + cost, adversarially verify')
const V1 = agent(CTX + '\n\n=== IMPL ===\n'+JSON.stringify(impl,null,1) +
  '\n\n=== V1 — RUN + VERIFY ===\n' +
  'Load the LSE reference + comm_cost standalone and verify: (a) the fold is associative + commutative-up-to-rounding + ' +
  'bit-deterministic and matches single-pass softmax on a random union of key sets (fp32); (b) G3 prices the reduce as ' +
  'F-dominated with tree_fold ~37us / plain_ring ~230us at P=32, and the frontend selects the min-hop schedule; (c) the ' +
  'layout-dividend claim (reduce-within-head/scatter-over-heads -> head-split) is internally consistent. Flag any ' +
  'reference-math error or cost mismatch as a bug. Return VAL.',
  { label:'V1:verify', phase:'Validate', schema: VAL_SCHEMA, effort:'high' }).then(r=>({k:'V1',r}))
const V2 = agent(CTX + '\n\n=== IMPL ===\n'+JSON.stringify(impl,null,1) +
  '\n\n=== V2 — ADVERSARIAL ===\n' +
  'Attack G2: is the producer_has_partial reroute safe (does it risk regressing existing non-reduce edges? is it flag- ' +
  'gated?)? Is the LSE-combine reference actually associative/deterministic, or does it hide an fp16 rounding order ' +
  'dependence that breaks determinism? Does the tree_fold really beat plain_ring, or does the reduce-SCATTER (payload/P ' +
  'per hop, 2(P-1) hops) change the answer at this payload? Is the layout-dividend head-split claim real or asserted? ' +
  'Is anything claimed as done that is actually backend/device-GATED? Return VAL (verdict=holds/holds-with-corrections/refuted).',
  { label:'V2:adversarial', phase:'Validate', schema: VAL_SCHEMA, effort:'high' }).then(r=>({k:'V2',r}))
const vals = (await parallel([()=>V1,()=>V2])).filter(Boolean)
const vByKey={}; for(const v of vals) vByKey[v.k]=v.r

phase('Synthesize')
log('Synthesize the G2 deliverable')
const report = await agent(CTX + '\n\n=== ARCH ===\n'+archDigest+'\n\n=== IMPL ===\n'+JSON.stringify(impl,null,1)+
  '\n\n=== V1 ===\n'+JSON.stringify(vByKey['V1'],null,1)+'\n\n=== V2 ===\n'+JSON.stringify(vByKey['V2'],null,1)+
  '\n\n=== SYNTHESIZE (markdown for the human lead) ===\n' +
  'Write a tight G2 deliverable: (1) what was built (frontend reduce-lane routing + make_lse_ring_fold_contract + LSE ' +
  'reference + G3 cost wiring; files, branch, tests, apply V2 corrections); (2) the reduce lane in one screen (the ' +
  'producer_has_partial unblock, the LSE-combine op, the tree-fold min-hop topology via G3, the head-split layout ' +
  'dividend); (3) validation results (math determinism, F-dominated cost, dividend); (4) the GATED remainder that needs ' +
  'the backend SFP primitive + device + S2 co-bundling; (5) honest limitations. Decisive, MEASURED-grounded.',
  { label:'synth:g2', phase:'Synthesize', effort:'high' })
return { arch: aByKey, impl, validate: vByKey, report }
