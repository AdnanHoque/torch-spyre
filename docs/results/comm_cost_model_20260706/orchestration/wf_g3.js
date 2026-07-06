export const meta = {
  name: 'g3-comm-cost-model',
  description: 'Architect + implement G3: a standalone on-chip communication cost model (separate from the matmul cost model), validated against measured ring physics',
  phases: [{ title: 'Architect' }, { title: 'Implement' }, { title: 'Validate' }, { title: 'Synthesize' }],
}

const CTX = `
You are an Opus subagent under a Fable orchestrator building G3: a SEPARATE communication cost model for torch-spyre.
Final text is a structured data return. Cite file:line. Tag MEASURED / ANALYTICAL / STRUCTURAL. No inflation.

=== WHY G3 (the whole point) ===
The planner has a matmul cost model (_matmul_split_cost) that prices compute/HBM/PSUM per work-division. It has NO
model for on-chip LX ring COLLECTIVES — so nothing prices a relayout/broadcast/reduce edge (backend plans literally
carry estimated_tensor_bytes=0). The current Bet-2 hack folds a cohort penalty INTO _matmul_split_cost's hbm_us,
conflating two resources. G3 = a STANDALONE communication cost model that prices on-chip movement, composes with the
matmul model at the seam (planner sums op-cost + edge-cost), and does NOT merge into it. It must reproduce this
session's MEASURED findings as model outputs.

=== MEASURED PHYSICS (the calibration constants — do not invent others) ===
- F = 7.3 us fixed per STCDP execute. **Execute COUNT dominates for small operands** (128 KiB chunk transfer is ~89% F).
- Uniform p->p+1 shift (1 transfer/link): 54 / 90 / 130 GB/s at 4 / 8 / 16 MB; F-corrected raw asymptote ~140-166 GB/s.
- Scatter/all-to-all (4-9 transfers/link): ~36 GB/s effective. Worse for higher occupancy (16/link is worse than 36).
- **The band variable is per-link TRANSFER COUNT (1 vs 4-9 vs 16), NOT burst size** (burst x50-100 gave 1.0x).
- **Same-core replicas = FREE local copy** (self-ring -> BusFence, must never happen).
- LX capacity ~= 2 MiB/core. HBM = ONE shared pipe ~170-205 GB/s (a spill = write+read round trip).
- Time model per SEQUENTIAL execute: t_step = F + (max bytes on any one physical link this step) / raw_rate.

=== THE REALIZED ATTENTION BROADCAST (the anchor case, MEASURED from the backend plan) ===
all_gather_replicate into a KERNEL matmul RHS (Tensor1). group_count=4, replicas/group=8, producer_chunks/group=8,
256 logical transfers. Scheduled as ALL-TO-ALL within four contiguous 8-core groups: hop distances 1-7, MAX physical
link occupancy = 16/link, 32 same-core (free) + 224 cross-core. Operand = 1 MiB/head (chunk = 128 KiB). Codex's realized
lowering gives a measured 1.053x kernel on Granite (HBM-round-trip elimination, NOT ring efficiency).

=== SEED + KEY RESULTS TO REPRODUCE (g1_ring_model.py already validated these) ===
Seed model at /private/tmp/claude-501/-Users-adnan-torch-spyre-work/8449d6fa-d2d5-4c0f-b1bb-261e1c7f35cc/scratchpad/g1_ring_model.py (READ IT).
- Attention 1 MiB/head, G=8: naive_all2all(1 execute, 16/link) = 22.3 us FASTEST; ring_carousel(7 exec, 1/link) = 57.7 us
  (2.6x SLOWER); recursive_doubling(3 exec) = 28.5 us; bidir_ring(4 exec) = 32.9 us.
- CROSSOVER: ring beats naive-1-exec only for operand > 5.2 MiB/head.
- **LX-capacity INTERCEPTION (the key structural result): crossover 5.2 MiB > LX cap 2 MiB, so any operand big enough
  to favor the ring must be tiled <= 2 MiB (below the crossover) -> the ring NEVER wins for an LX-resident broadcast.**
- Decode KV (long context) is STREAMED from HBM (sequence-sharded), NEVER ring-broadcast; ring payload (Q ~8KB, LSE
  merge ~16.5KB) is L-INDEPENDENT tiny. MLP: weights out-of-scope, activations M-split/seam-transparent (edge cost 0).
- LSE reduce fold: (P-1)=31 sequential hops x F = ~226 us, 98% fixed-cost -> minimize hop count, tiny payload.

=== ENV / CODE / CONVENTIONS ===
- Mac, cwd /Users/adnan/torch-spyre-work. Local torch checkout on the working branch:
  /Users/adnan/torch-spyre-work/torch-spyre-comms-collectives  (branch ah/comms-collectives).
- Relevant code (READ, do not depend on the torch runtime): torch_spyre/_inductor/work_division.py
  (_matmul_split_cost with the Bet-2 _cohort_penalty folded into hbm_us -> the thing to SEPARATE; _cost_model_matmul_planner),
  torch_spyre/_inductor/lx_relayout.py (the classifications, _classify_coordinate_topology, plan_lx_relayouts; the
  estimated_tensor_bytes=0 costing gap). The ah/ring-cost-term branch's _cohort_penalty (multicast cap peak/130, scatter
  cap peak/36) is the seed to REFACTOR OUT of the matmul model into G3.
- gh api available (repos/AdnanHoque/torch-spyre). Degraded DEV device -> NO device runs; this is pure-Python + code-read.
- PRE-PUBLISH CONVENTION: keep exploratory work in a PRIVATE LOCAL BRANCH; NO PRs, NO pushes, NO main writes.
`

const ARCH_SCHEMA = { type:'object', additionalProperties:true, required:['area','summary'],
  properties:{ area:{type:'string'}, summary:{type:'string'}, spec:{type:'string'}, key_points:{type:'array', items:{type:'string'}} } }
const IMPL_SCHEMA = { type:'object', additionalProperties:true, required:['built','test_results'],
  properties:{ built:{type:'boolean'}, module_path:{type:'string'}, test_path:{type:'string'}, branch:{type:'string'}, test_results:{type:'string'}, api_summary:{type:'string'}, notes:{type:'array', items:{type:'string'}} } }
const VAL_SCHEMA = { type:'object', additionalProperties:true, required:['summary','verdict'],
  properties:{ summary:{type:'string'}, verdict:{type:'string'}, cases:{type:'array', items:{type:'object', additionalProperties:true}} } }

phase('Architect')
log('Architect G3: code-integration map + cost-model API + validation spec (parallel)')
const A1 = agent(CTX +
  '\n\n=== A1 — CODE-INTEGRATION MAP (read the real code) ===\n' +
  'Read work_division.py (_matmul_split_cost, _cohort_penalty, _cost_model_matmul_planner) and lx_relayout.py ' +
  '(classifications, _classify_coordinate_topology, plan_lx_relayouts). Produce: (1) exactly what to REMOVE from ' +
  '_matmul_split_cost (the cohort penalty folded into hbm_us) and how the matmul model calls G3 at the seam instead; ' +
  '(2) the CommEdge descriptor G3 consumes — derived from the lx_relayout classification + backend plan fields ' +
  '(comm_class, group_count, replicas/group, producer_chunks, operand bytes, the coordinate/source->dest map); ' +
  '(3) where G3 lives (new module torch_spyre/_inductor/comm_cost.py) and how the planner composes op-cost + edge-cost ' +
  '(sum at the seam, NOT merge). Cite file:line. Return ARCH (area="integration").',
  { label:'A1:integration', phase:'Architect', schema: ARCH_SCHEMA, effort:'high' }).then(r=>({k:'A1',r}))
const A2 = agent(CTX +
  '\n\n=== A2 — COST-MODEL ARCHITECTURE (API + terms) ===\n' +
  'Design the standalone CommCostModel (pure-Python, stdlib only, NO torch import, so it is testable anywhere). Read the ' +
  'seed g1_ring_model.py. Specify: (1) the CommEdge input {comm_class, operand_bytes, dtype, group structure OR ' +
  'coordinate source->dest map, payload_per_hop}; (2) the RESOURCE VECTOR output {time_us, executes, dram_bytes_saved_vs_' +
  'spill, max_link_occupancy, lx_highwater} — NOT a scalar; (3) the term functions: link_occupancy(coord_map) (route ' +
  'transfers on the P-ring shortest arc, count per physical link), band(occupancy, transfer_bytes) (1/link->~140 raw, ' +
  'derated), schedule_cost(schedule) = sum_steps(F + max_link_bytes/rate), the schedule set {naive_all2all(1 exec), ' +
  'ring(G-1 exec), recursive_doubling(log2 G), bidir}, best_schedule respecting the LX_CAP tiling (operand>2MiB -> ' +
  'n_tiles=ceil(bytes/LX_CAP), per-tile cost x n_tiles), same-core-free, and the reduce/LSE-fold as an F-dominated ' +
  'linear chain. State the equations precisely. Return ARCH (area="architecture").',
  { label:'A2:architecture', phase:'Architect', schema: ARCH_SCHEMA, effort:'high' }).then(r=>({k:'A2',r}))
const A3 = agent(CTX +
  '\n\n=== A3 — VALIDATION SPEC ===\n' +
  'Define the validation cases the implemented G3 MUST reproduce (with expected outputs from the MEASURED findings), so ' +
  'the model is checked against reality: (C1) attention operand broadcast 1 MiB G=8 -> best=naive_all2all ~22us, ring ' +
  '~58us (2.6x worse); (C2) crossover: ring beats naive only >5.2 MiB; (C3) LX-cap interception: a 16 MiB operand tiled ' +
  'to 2 MiB chunks -> ring still loses per-tile (the general "ring never wins for LX-resident broadcast"); (C4) decode ' +
  'KV: model returns HBM-stream cost (L-dependent) for the KV and a tiny L-INDEPENDENT ring cost for Q+merge; (C5) MLP ' +
  'M-split activation handoff -> edge cost 0 (seam-transparent); (C6) LSE reduce over P=32 -> ~226us F-dominated, and ' +
  'recursive/bidir schedules beat plain ring at fewer hops; (C7) the realized 16/link broadcast -> the model predicts ' +
  'its slow band and that the 1.053x is HBM-elimination not ring-efficiency. For each: the exact input + the expected ' +
  'output/relationship. Return ARCH (area="validation").',
  { label:'A3:validation', phase:'Architect', schema: ARCH_SCHEMA, effort:'high' }).then(r=>({k:'A3',r}))
const arch = (await parallel([()=>A1,()=>A2,()=>A3])).filter(Boolean)
const aByKey={}; for(const a of arch) aByKey[a.k]=a.r
const archDigest = 'A1 integration:\n'+JSON.stringify(aByKey['A1'],null,1)+'\n\nA2 architecture:\n'+JSON.stringify(aByKey['A2'],null,1)+'\n\nA3 validation:\n'+JSON.stringify(aByKey['A3'],null,1)

phase('Implement')
log('Implement comm_cost.py + tests on a private branch; run tests')
const impl = await agent(CTX +
  '\n\n=== ARCHITECTURE (from Phase 1) ===\n'+archDigest+
  '\n\n=== IMPLEMENT G3 ===\n' +
  'In the local checkout /Users/adnan/torch-spyre-work/torch-spyre-comms-collectives, create a PRIVATE branch ' +
  'ah/comm-cost-model-g3 (git checkout -b; NO push, NO PR). Implement per A1+A2:\n' +
  '- torch_spyre/_inductor/comm_cost.py — the standalone CommCostModel, PURE PYTHON (stdlib only, NO torch import), ' +
  'with the CommEdge dataclass, the resource-vector output, link_occupancy/band/schedule_cost/best_schedule, the ' +
  'LX_CAP tiling term, same-core-free, and the reduce/LSE-fold cost. Calibrate to the MEASURED constants exactly. ' +
  'Reuse the g1_ring_model.py logic. Docstring it clearly as SEPARATE from the matmul cost model.\n' +
  '- tests/tensor/test_comm_cost.py — unit tests pinning: F/band/LX-cap/same-core invariants; naive beats ring at 1 MiB; ' +
  'the 5.2 MiB crossover; the LX-cap interception (ring never wins for <=2 MiB resident); the reduce F-domination; ' +
  'and at least the C1/C2/C3/C6 validation cases from A3.\n' +
  'RUN the tests with the system python3 (they must be pure-Python, no torch): report pass/fail counts and any output. ' +
  'Then `git add` + `git commit` on the private branch (do NOT push). Report the module_path, test_path, branch, ' +
  'test_results, api_summary, and `git diff --stat`. Return IMPL.',
  { label:'implement:comm_cost', phase:'Implement', schema: IMPL_SCHEMA, effort:'high' })

phase('Validate')
log('Validate: run the real cases through it + adversarially verify')
const V1 = agent(CTX +
  '\n\n=== IMPLEMENTATION ===\n'+JSON.stringify(impl,null,1)+
  '\n\n=== V1 — RUN THE VALIDATION CASES ===\n' +
  'Import/exec the implemented comm_cost.py (module_path in the IMPL result) and RUN each A3 validation case (C1-C7). ' +
  'For each, report the model output vs the expected MEASURED relationship and whether it MATCHES. Especially confirm: ' +
  'attention -> naive wins (~22us); the crossover ~5.2 MiB; the LX-cap interception (ring loses for any <=2 MiB ' +
  'resident); decode KV streamed not ring; MLP M-split edge cost 0; reduce ~226us F-dominated. If a case does NOT ' +
  'reproduce the finding, that is a model bug -> report it precisely. Return VAL.',
  { label:'V1:run-cases', phase:'Validate', schema: VAL_SCHEMA, effort:'high' }).then(r=>({k:'V1',r}))
const V2 = agent(CTX +
  '\n\n=== IMPLEMENTATION ===\n'+JSON.stringify(impl,null,1)+
  '\n\n=== V2 — ADVERSARIAL VERIFY ===\n' +
  'Independently re-derive the load-bearing numbers (the 22us naive, 58us ring, 5.2 MiB crossover, 226us reduce) from ' +
  'the MEASURED constants WITHOUT trusting the implementation. Attack: are the calibration constants right? does the ' +
  'band()/occupancy routing double-count? does the LX-cap tiling term actually intercept the crossover, or is it hard- ' +
  'coded? does best_schedule pick correctly at the boundaries? does the model over/under-count executes (F)? Is the ' +
  'comm model TRULY decoupled from the matmul model (no torch/matmul import)? Flag any number that cannot be grounded ' +
  'in a MEASURED input. Return VAL (verdict = holds / holds-with-corrections / refuted).',
  { label:'V2:adversarial', phase:'Validate', schema: VAL_SCHEMA, effort:'high' }).then(r=>({k:'V2',r}))
const vals = (await parallel([()=>V1,()=>V2])).filter(Boolean)
const vByKey={}; for(const v of vals) vByKey[v.k]=v.r

phase('Synthesize')
log('Synthesize the G3 deliverable')
const report = await agent(CTX +
  '\n\n=== ARCH ===\n'+archDigest+'\n\n=== IMPL ===\n'+JSON.stringify(impl,null,1)+
  '\n\n=== V1 run ===\n'+JSON.stringify(vByKey['V1'],null,1)+'\n\n=== V2 adversarial ===\n'+JSON.stringify(vByKey['V2'],null,1)+
  '\n\n=== SYNTHESIZE (markdown for the human lead) ===\n' +
  'Write a tight G3 deliverable report: (1) what was built (module + tests + branch, with paths); (2) the cost-model ' +
  'architecture in one screen (CommEdge in, resource-vector out, the terms, WHY separate from the matmul model); ' +
  '(3) the validation results table — each case, model output, expected, match (apply V2 corrections; if a case failed ' +
  'or a number was refuted, say so); (4) the integration next step (what to remove from _matmul_split_cost, how the ' +
  'planner composes op+edge cost, the estimated_tensor_bytes=0 gap it fills); (5) honest limitations. Be decisive, ' +
  'MEASURED-grounded, no wall-time-only claims.',
  { label:'synth:g3', phase:'Synthesize', effort:'high' })
return { arch: aByKey, impl, validate: vByKey, report }
