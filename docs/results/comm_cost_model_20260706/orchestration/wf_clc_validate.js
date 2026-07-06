export const meta = {
  name: 'clc-validate-comm-cost',
  description: 'Validate the G3/seam/G2 branch on adnan-clc: real-env test suites + Granite S512 plan-pick seam OFF-vs-ON selection comparison',
  phases: [{ title: 'SetupEnvTests' }, { title: 'DeviceSelection' }],
}

const CTX = `
You are an Opus subagent under a Fable orchestrator, validating a private torch-spyre branch on the adnan-clc AIU pod.
Final text is a structured data return. Cite exact commands/paths. Tag MEASURED / BLOCKED. No inflation.

=== WHAT WE ARE VALIDATING ===
Branch ah/comm-cost-model-g3 (3 commits off base 005b5af): G3 comm_cost.py (standalone cost model), a flag-gated
planner SEAM in work_division.py (env SPYRE_COMM_COST_SEAM, default OFF), and the G2 LSE reduce lane (config gate
SPYRE_LX_PLANNER_RELAYOUT_REDUCE, default OFF). Everything default-OFF = zero change. The device-gated question: does
turning the SEAM ON reproduce the device-verified Granite work-division selections?

=== HARD SAFETY RAILS ===
- adnan-clc pod (namespace a6-quantization), device /dev/vfio/25. oc exec -n a6-quantization adnan-clc-spyre-dev-pf -- bash -lc 'CMD'.
- NEVER modify Codex's real checkouts. Work ONLY in fresh /tmp dirs you create. Wrap device runs in 'timeout 900'.
- If a device op hangs past ~60s per sync (the known "possible lost completion" degraded state) or hits a fence/RAS/
  SIGABRT: STOP, capture, report as BLOCKED. Do not retry on the device.

=== PATHS ===
- Git bundle of our branch (on THIS Mac): /private/tmp/claude-501/-Users-adnan-torch-spyre-work/8449d6fa-d2d5-4c0f-b1bb-261e1c7f35cc/scratchpad/g3.bundle
  It contains ref ah/comm-cost-model-g3 (tip d73c6c7) and requires base 005b5af (which the pod's repo HAS).
- Pod torch-spyre checkout (has _C.so build artifacts + base 005b5af):
  POD_TS=/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/torch-spyre  (branch ah/comms-collectives @ 55bb656)
- Granite harness root: R=/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404
  Bench: $R/spyre-granite-e2e-bench ; FMS: $R/foundation-model-stack ; venv python: /home/adnan/dt-inductor/.venv/bin/python3
  The FULL proven env block is in $R/runs/granite_s512_collectives_archive_20260705_081840/run_pair.sh (source its exports).
- Granite plan-pick command template (from run_pair.sh):
    cd $BENCH && $PY benchmarks/granite_block_layer_probe.py --fms-root $FMS --run-root <FRESH_DIR> \\
      --case prefill --seq-len 512 --batch 1 --hidden 4096 --compile-block --attn-name sdpa_causal --iters 5 --warmups 1

=== KNOWN BASELINE (from the earlier f32392e plan-pick; cross-check only, NOT the primary comparison) ===
buf6 QKV m=4 n=8 ; buf14 attn-scores m=16 n=2 ; buf22 attn-AV m=32 n=1 ; buf24 out-proj m=4 n=8 ; buf33 MLP-gate+up
m=4 n=8 ; buf36 MLP-down m=8 n=4. The PRIMARY comparison is our-branch SEAM-OFF vs SEAM-ON (same base = isolates the
seam's effect); the f32392e baseline is a cross-check since our base (005b5af) differs.
`

const SETUP_SCHEMA = { type:'object', additionalProperties:true, required:['setup_ok','summary'],
  properties:{ setup_ok:{type:'boolean'}, checkout_path:{type:'string'}, our_commits_present:{type:'boolean'}, cso_present:{type:'boolean'}, test_results:{type:'string'}, summary:{type:'string'}, notes:{type:'array', items:{type:'string'}} } }
const DEV_SCHEMA = { type:'object', additionalProperties:true, required:['device_ok','summary'],
  properties:{ device_ok:{type:'boolean'}, seam_off:{type:'array', items:{type:'object', additionalProperties:true}}, seam_on:{type:'array', items:{type:'object', additionalProperties:true}}, selections_match:{type:'boolean'}, headline:{type:'string'}, summary:{type:'string'}, logs:{type:'array', items:{type:'string'}} } }

phase('SetupEnvTests')
log('CLC: transfer bundle, fresh checkout (+_C.so), run real-env test suites')
const setup = await agent(CTX +
  '\n\n=== STEPS ===\n' +
  '1. Copy the bundle to the pod: `oc cp /private/tmp/claude-501/-Users-adnan-torch-spyre-work/8449d6fa-d2d5-4c0f-b1bb-261e1c7f35cc/scratchpad/g3.bundle a6-quantization/adnan-clc-spyre-dev-pf:/tmp/g3.bundle` (or the `-n` form; verify it landed with ls).\n' +
  '2. On the pod, build a FRESH working copy that has our source AND a usable _C.so: `cp -r $POD_TS /tmp/ts-g3-validate` ' +
  '(this brings Codex build artifacts incl. torch_spyre/_C*.so). Then `cd /tmp/ts-g3-validate && git fetch /tmp/g3.bundle ' +
  'ah/comm-cost-model-g3:ah/comm-cost-model-g3 && git checkout -f ah/comm-cost-model-g3`. The git checkout switches TRACKED ' +
  'source files to our branch (005b5af base + our 3 commits); the untracked/ignored _C*.so stays. Verify: our files exist ' +
  '(torch_spyre/_inductor/comm_cost.py, lse_fold_ref.py), `git log --oneline -3` shows d73c6c7/fb07bb4/31e76c7, and ' +
  '`ls torch_spyre/_C*.so` (or find torch_spyre -name "_C*.so") shows the built extension is present.\n' +
  '3. TIER-1 real-env test suites (device-free): run the pure-Python suites against OUR checkout in the pod Python: ' +
  '`cd /tmp/ts-g3-validate && PYTHONPATH=/tmp/ts-g3-validate /home/adnan/dt-inductor/.venv/bin/python3 -m pytest ' +
  'tests/tensor/test_comm_cost.py tests/tensor/test_lse_fold.py --noconftest -q`. Report the pass/fail counts (expect ' +
  '20/20 + 16/16). If --noconftest breaks a fixture, try loading the modules directly, but report exactly what ran.\n' +
  'Return SETUP {setup_ok, checkout_path, our_commits_present, cso_present, test_results, summary, notes}.',
  { label:'setup+tier1', phase:'SetupEnvTests', schema: SETUP_SCHEMA, effort:'high' })

phase('DeviceSelection')
log('CLC: device health + Granite S512 plan-pick SEAM OFF vs ON, compare selections')
const dev = await agent(CTX +
  '\n\n=== SETUP RESULT (from phase 1) ===\n'+JSON.stringify(setup,null,1)+
  '\n\n=== STEPS (only proceed if setup_ok and cso_present) ===\n' +
  '1. DEVICE HEALTH (bounded): source the run_pair.sh env block from $R/runs/granite_s512_collectives_archive_20260705_081840/' +
  'run_pair.sh (its exports up to record_repo_state give the full LD_LIBRARY_PATH/DEEPTOOLS_PATH/SENTIENT_*/PATH). Then ' +
  '`timeout 90 $PY -c "import torch,torch_spyre;torch_spyre._autoload();import torch as T;print(float((T.ones(8,dtype=T.float16).to(\\"spyre\\")+1).cpu().sum()))"`. ' +
  'If it returns ~16.0 in a few seconds -> HEALTHY. If it takes ~60s+ per sync or hangs to timeout -> DEGRADED like DEV: ' +
  'report device_ok=false and STOP (do not run the compiles).\n' +
  '2. If HEALTHY: run the Granite plan-pick TWICE, using OUR checkout (prepend PYTHONPATH=/tmp/ts-g3-validate BEFORE the ' +
  'run_pair.sh PYTHONPATH), relayout OFF (set the 5 SPYRE_LX_PLANNER_RELAYOUT* = 0 for a clean baseline compile), ' +
  'SPYRE_INDUCTOR_LOG=1 SPYRE_INDUCTOR_LOG_LEVEL=DEBUG, into fresh run dirs:\n' +
  '   RUN A: SPYRE_COMM_COST_SEAM=0  -> /tmp/planpick_off_$(date +%s)\n' +
  '   RUN B: SPYRE_COMM_COST_SEAM=1  -> /tmp/planpick_on_$(date +%s)\n' +
  '   Each: `cd $BENCH && timeout 900 env <PYTHONPATH override + flags> $PY benchmarks/granite_block_layer_probe.py ' +
  '--fms-root $FMS --run-root $RUN --case prefill --seq-len 512 --batch 1 --hidden 4096 --compile-block --attn-name ' +
  'sdpa_causal --iters 5 --warmups 1 2>&1 | tee $RUN/log.txt`.\n' +
  '3. From EACH log grep `cost_model work_division` and extract the selected split (b/m/n/k) per matmul buf (buf6/14/22/24/33/36).\n' +
  '4. COMPARE seam-OFF vs seam-ON per buf. Report the two selection tables, whether they MATCH per buf, and cross-check ' +
  'against the known baseline. Note rc of each run. If a compile fails but the cost_model selection lines were already ' +
  'logged before the failure, still extract+report them (selection is a compile-time decision).\n' +
  'INTERPRET: match => the separated comm-cost seam reproduces the tuned selections (safe to promote). Divergence => the ' +
  'edge-cost calibration shifts selection; report exactly which bufs and how (this is a real finding, not a failure).\n' +
  'Return DEV {device_ok, seam_off:[{buf,split}], seam_on:[{buf,split}], selections_match, headline, summary, logs}.',
  { label:'device-selection', phase:'DeviceSelection', schema: DEV_SCHEMA, effort:'high' })

return { setup, device: dev }
