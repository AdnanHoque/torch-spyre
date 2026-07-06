export const meta = {
  name: 'flash-collectives-perf',
  description: 'Flash-attention on-chip LX communication collectives + performance: map spills->collectives, cost vs HBM, S2/S4/decode plan',
  phases: [{ title: 'Analyze' }, { title: 'Synthesize' }],
}

const CTX = `
You are an Opus subagent under a Fable orchestrator. Final text is a structured data return. Cite file:line / artifact
paths. Tag MEASURED / STRUCTURAL / ANALYTICAL / BLOCKED. No inflation.

=== SCOPE (hard) ===
We are NOT working on flash VALUE-CORRECTNESS — that vanilla bug (causal-boundary numerator leak from a broadcast
unsqueeze losing stride_map: op_spec.py:40 / spyre_kernel.py:497 / superdsc.py:231) is Jamie's lane. IGNORE it.
OUR job = PERFORMANCE + the on-chip LX communication collectives the flash kernel needs to remove its non-weight HBM
activation spills (Epic torch-spyre#3049 production ask). Analyze the CURRENT de-spilled flash AND the eventual
value-correct + Granite-integrated + sequence-sharded (decode) versions.

=== FLASH KERNEL ===
test_flash.py: prefill flash, B=1, H=32, H_kv (Granite decode)=8, D=128, Lq=Lk=4096, causal, fp16 (b=2), scale=1/sqrt(sqrt(D)).
Ops per attention: scaled_keys = K*scale; scores = (Q*scale) @ Kt ; scores += causal_mask ; block_max=amax(scores,-1);
exp_scores=exp(scores-max); denom=sum(exp_scores,-1); out = exp_scores @ V ; return out/denom.
Work-div hints: {H:4, Lq:8, Lk:8} but Lk:8 is SKIPPED (H*Lq*Lk=256 > SENCORES=32) -> Lk is NOT tiled (kv_block=Lk/1).

=== MEASURED STATE (this session) ===
- Flash relayout-ON compiles to 550 SDSC with **32 ReStickifyOpHBM -> 32 ReStickifyOpLx + 32 matmul_operand_broadcast
  backend plans** (STRUCTURAL; value skipped). So the flash de-spill target = 32 operand-broadcast handoffs.
- The all_gather_replicate-into-KERNEL-operand class is the shared BLOCKER (S2). Granite attention hit it: backend plan
  pattern=all_gather_replicate, groups=2, 16 producer_chunks x 16 consumer_replicas, 512 logical transfers; dies at
  L3DlOpsScheduler.cpp:1701 "initial chunk parameters must fit in LX for SuperDSC" (capacity wall).
- S2 co-bundling: LX does NOT persist across separate device programs; mul(K)/scaled_keys is a separate program so the
  consumer re-reads K from HBM. bundle.py has ZERO co-bundling wiring today (grep-empty). Fix = co-bundle the producer
  (scaled_keys / exp_scores) into the consumer (QK^T / AV) program so LX persists; then the broadcast rides true shards.
- S4 LSE ring-fold (reduce lane, Epic Phase 5): math-validated 13/13; the online-softmax Lk reduction and the eventual
  sequence-sharded decode merge need it. Cost it as a LINEAR-CHAIN CONTENTION pattern (NOT the uniform shift).

=== MEASURED RING PHYSICS (use these; do NOT use ideal link BW) ===
- Uniform p->p+1 shift: 54 / 90 / 130 GB/s at 4 / 8 / 16 MB (R^2>=0.9985). 1 transfer/link.
- Range/scatter/multicast relayout EFFECTIVE: ~36 GB/s (R^2>=0.9999). 4-9 transfers/link.
- Fixed cost F ~= 7.3 us per STCDP execute. lambda per-hop = 0 modeled, NOT isolable (do not lean on it).
- HBM: one shared pipe ~170-205 GB/s across all 32 cores. Same-core replicas = FREE local copy.
- Distinguishing variable = per-link TRANSFER COUNT (1 vs 4-9), not burst size.
- P=32 cores. LX budget ~2 MiB/core.

=== ENV (no device this round; DEV device is degraded ~60s/sync; READ-ONLY only) ===
- oc exec -n a6-quantization POD -- bash -lc 'CMD'. DEV=adnan-spyre-dev-pf, CLC=adnan-clc, CDX=adnan-cdx.
- Flash artifacts (READ-ONLY) on DEV: /home/adnan/codex-isolated/flash_attention_devpf_verify_20260705_070129/runs/relayout_on_current_main_backend1_20260705_191932/
  has artifact_files.txt, backend_plan_files.txt, backend_plans/ (the 32 operand-broadcast plan JSONs), cache/ (sdsc_*.json).
  NEVER edit/checkout Codex files. No device compiles/runs.
- gh api for source (repos/AdnanHoque/torch-spyre ref ah/comms-collectives).
`

const RECON_SCHEMA = { type:'object', additionalProperties:true, required:['summary','handoffs'],
  properties:{ summary:{type:'string'},
    handoffs:{type:'array', items:{ type:'object', additionalProperties:true, required:['operand','comm_class','layout_change','note'],
      properties:{ operand:{type:'string'}, comm_class:{type:'string'}, layout_change:{type:'string'}, group_struct:{type:'string'}, note:{type:'string'} } } } } }
const PERF_SCHEMA = { type:'object', additionalProperties:true, required:['summary','rows'],
  properties:{ summary:{type:'string'},
    rows:{type:'array', items:{ type:'object', additionalProperties:true, required:['item','hbm_cost','lx_collective_cost','net','basis'],
      properties:{ item:{type:'string'}, hbm_cost:{type:'string'}, lx_collective_cost:{type:'string'}, net:{type:'string'}, basis:{type:'string'} } } } } }
const PLAN_SCHEMA = { type:'object', additionalProperties:true, required:['summary','collectives','sequence'],
  properties:{ summary:{type:'string'}, collectives:{type:'array', items:{type:'string'}}, sequence:{type:'array', items:{type:'string'}} } }

phase('Analyze')
log('Flash collectives: SDSC recon + perf model + integration/decode plan (parallel, no device)')
const FA = agent(CTX +
  '\n\n=== TASK FA — flash communication STRUCTURE from the real SDSC (READ-ONLY) ===\n' +
  'Read the flash run artifacts on DEV (backend_plans/*.json = the 32 matmul_operand_broadcast plans; ' +
  'artifact_files.txt / cache sdsc_*.json for the op graph). For the flash attention, enumerate the on-chip ' +
  'communication handoffs that currently spill to HBM (the ~32 restickifies): for EACH, identify which operand ' +
  '(scaled_keys/K into QK^T, exp_scores or V into AV, Q, the softmax reduce), the SOURCE->DEST layout change ' +
  '(stick swap etc.), the comm class (broadcast / multicast / all_gather_replicate / reduce), and the group/replica/ ' +
  'stride structure from the backend plan JSON (group_count, producer_chunks, consumer_replicas, logical transfers). ' +
  'State which are the S2 KERNEL-operand-wall class. Return RECON.',
  { label:'FA:sdsc-recon', phase:'Analyze', schema: RECON_SCHEMA, effort:'high' }).then(r=>({k:'FA',r}))

const FB = agent(CTX +
  '\n\n=== TASK FB — PERFORMANCE model: LX collective vs HBM spill (ANALYTICAL) ===\n' +
  'Using flash dims (Lq=Lk=4096, H=32, D=128, b=2, P=32) and the MEASURED ring physics, quantify: ' +
  '(1) the HBM traffic the current flash spills cost (the operand restickifies: bytes moved to/from HBM and the ' +
  '~170 GB/s shared-pipe time); (2) the LX collective cost that replaces each (broadcast/all-gather of the operand: ' +
  'bytes on the ring at the honest 36 GB/s scatter or 130 GB/s uniform-shift band + F~7.3us/execute), and whether ' +
  'the operand move is a uniform-shift (fast) or a scatter/multicast (slow-band); (3) the net win per handoff and ' +
  'chip-wide; (4) the ONLINE-SOFTMAX REDUCE cost if Lk were tiled into T blocks (the LSE fold as a linear-chain ' +
  'contention over the (m,l,O) partials, ~16.5 KiB/hop, P-1 hops, F/hop) and the crossover T where tiling+reduce ' +
  'beats the single-untiled-block streaming; (5) decode (BS=1) attention: sequence-sharded KV across P vs head-split ' +
  'H_kv=8 — the x(P/H_kv) bandwidth ceiling and the L-independent merge overhead. Show arithmetic; separate MEASURED ' +
  'inputs from assumptions. Return PERF.',
  { label:'FB:perf-model', phase:'Analyze', schema: PERF_SCHEMA, effort:'high' }).then(r=>({k:'FB',r}))

const FC = agent(CTX +
  '\n\n=== TASK FC — the collectives to BUILD (implementation plan, ANALYTICAL) ===\n' +
  'Define the concrete on-chip collectives the flash kernel needs and the build sequence, tied to the survivors: ' +
  'S2 (cross-bundle co-bundling so the operand-broadcast rides LX-resident true shards instead of re-reading HBM — ' +
  'the bundle.py + KERNEL-operand-carrier work), and S4 (LSE ring-fold reduce lane for the Lk online-softmax reduction ' +
  'and the eventual sequence-sharded decode merge). For each: the exact frontend/backend touchpoints, the capacity- ' +
  'safe loop/tile-scoped shape (avoid the L3DlOpsScheduler.cpp:1701 LX-fit wall; same-core = free local copy; never ' +
  'self-ring -> BusFence), the layout dividend (reduce-scatter-over-heads leaves output head-split = the k_fast ' +
  'out-projection input, zero relayout), and the decode KV-carousel Q-broadcast + block-cyclic placement. Rank by ' +
  '(perf value x unblocked-ness) and give the first concrete step for each. Return PLAN.',
  { label:'FC:build-plan', phase:'Analyze', schema: PLAN_SCHEMA, effort:'high' }).then(r=>({k:'FC',r}))

const outs = (await parallel([()=>FA,()=>FB,()=>FC])).filter(Boolean)
const by={}; for(const o of outs) by[o.k]=o.r

phase('Synthesize')
log('Synthesize flash collectives + performance plan')
const report = await agent(CTX +
  '\n\n=== RESULTS ===\nFA (structure):\n'+JSON.stringify(by['FA'],null,1)+
  '\n\nFB (perf):\n'+JSON.stringify(by['FB'],null,1)+
  '\n\nFC (build plan):\n'+JSON.stringify(by['FC'],null,1)+
  '\n\n=== SYNTHESIZE (markdown for the human lead) ===\n' +
  'Write a tight flash PERFORMANCE + COMMUNICATION-COLLECTIVES plan. Sections: ' +
  '(1) the flash de-spill communication map (each HBM spill -> the LX collective that removes it, comm class, S2 or not); ' +
  '(2) the performance model table (HBM cost vs LX collective cost vs net, per handoff and chip-wide, MEASURED-grounded; ' +
  'call out uniform-shift-fast vs scatter-slow-band); (3) the reduce lane (Lk-tiling LSE fold) cost + crossover; ' +
  '(4) the decode collectives (sequence-sharded KV + Q broadcast + LSE merge; x(P/H_kv) ceiling); (5) the prioritized ' +
  'build sequence (what to implement first for the biggest perf win, unblocked vs gated on S2). Be decisive, ' +
  'MEASURED-grounded, counter-first (no wall-time-only claims). This drives the flash collectives workstream.',
  { label:'synth:flash-collectives', phase:'Synthesize', effort:'high' })
return { recon: by['FA'], perf: by['FB'], plan: by['FC'], report }
