export const meta = {
  name: 'device-gated-runs',
  description: 'Fire the two gated AIU device runs: (A) Granite S512 plan-pick on CLC, (B) flash value vs clean oracle on DEV',
  phases: [{ title: 'Device' }],
}

const SAFETY = `
*** HARD DEVICE SAFETY RAILS (read first) ***
- The AIU devices are PRECIOUS: a wedge needs a pod recreate we cannot do. Be conservative.
- NEVER git checkout/reset/revert/clean/stash or edit ANY file inside a Codex checkout. You only RUN, you do not modify code.
- Write ONLY to fresh dirs you create under /tmp on the pod. Do not write into Codex run roots.
- Wrap every device invocation in 'timeout 900'. If a run hangs past that, let timeout kill it and report.
- If you see a device fence / BusFence / RAS:: / StreamInErrorState / 'hardware error' / SIGABRT: STOP IMMEDIATELY,
  do NOT retry on the device, capture the log, and report it as the result. Do not loop.
- Do exactly the runs specified. Do not improvise extra device workloads.
- oc access: oc exec -n a6-quantization POD -- bash -lc 'CMD'. You are on the user's Mac.
Your final text is a structured data return.`

const CLC = `${SAFETY}

=== RUN (A) — Granite S512 plan-pick confirmation, pod adnan-clc-spyre-dev-pf (/dev/vfio/25) ===
GOAL: capture which work-division split the REAL in-compiler planner selects for the 5 Granite S512 matmuls, to
confirm (or refute) the analytical finding that selection is (b=1, m=4, n=8, k=1) on every one. This is a COMPILE-time
selection; run the vetted baseline probe (relayout OFF, which compiles clean rc=0) with inductor DEBUG logging on.

HARNESS (already discovered — do not re-discover):
- ROOT=/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404
- The proven env + exact command live in \$ROOT/runs/granite_s512_collectives_archive_20260705_081840/run_pair.sh.
  Reuse its environment block verbatim (venv activate, DEEPTOOLS_PATH, LD_LIBRARY_PATH, PYTHONPATH, PATH, DTI_*,
  SENTIENT_*, RUNTIME_*, SENDNN_*, DXP_LX_FRAC_AVAIL=0, DXP_BACKEND_LX_FRAC_AVAIL=1).
- BENCH=\$ROOT/spyre-granite-e2e-bench ; FMS=\$ROOT/foundation-model-stack ; PY=/home/adnan/dt-inductor/.venv/bin/python3
- Command template (from run_pair.sh command.txt):
    cd \$BENCH && \$PY benchmarks/granite_block_layer_probe.py --fms-root \$FMS --run-root <FRESH_TMP_DIR> \\
      --case prefill --seq-len 512 --batch 1 --hidden 4096 --compile-block --attn-name sdpa_causal --iters 5 --warmups 1
- torch-spyre on this pod is branch ah/comms-collectives and HAS _cost_model_matmul_planner engaged (verified).

STEPS:
1. Make a fresh run dir: RUN=/tmp/granite_planpick_\$(date +%s); mkdir -p \$RUN.
2. Build a run script that: sources the run_pair.sh env block, then additionally exports:
     SPYRE_LX_PLANNING=1  SPYRE_LX_PLANNER_RELAYOUT=0 (+ all 4 other RELAYOUT_* flags =0 -> clean baseline)
     SPYRE_INDUCTOR_LOG=1  SPYRE_INDUCTOR_LOG_LEVEL=DEBUG  TORCHINDUCTOR_CACHE_DIR=\$RUN/cache
   then runs the command template with --run-root \$RUN, teeing stdout+stderr to \$RUN/log.txt.
   Wrap the python invocation in 'timeout 900'.
3. After it finishes, grep the log for the planner selection lines:
     grep -E 'cost_model work_division' \$RUN/log.txt        # format: cost_model work_division NAME: b=.. m=.. n=.. k=.. cost=..us [B=.. M=.. K=.. N=..]
   Also grep 'work_division buf' for the default-distributor context. Extract, for each matmul buf (QKV N=6144,
   attn-scores B=32/K=128, out-proj N=4096/K=4096, MLP gate+up N=25600, MLP down K=12800), the SELECTED (b,m,n,k).
4. Report rc, and per-matmul the selected split; state whether every matmul selected (b=1,m=4,n=8,k=1). If cost_model
   lines are absent (planner declined / logging off), say so and fall back to the 'work_division ... cores=' lines.

Return: { run:'A_granite_planpick', rc:<int>, device_ok:<bool>, selections:[{buf,MNK,selected_split,cost_us}],
          matches_analytical:<bool>, headline:<str>, log_path:<str>, anomalies:[..] }`

const DEV = `${SAFETY}

=== RUN (B) — Flash value vs CLEAN oracle, pod adnan-spyre-dev-pf (/dev/vfio/31, 192c) ===
GOAL: establish flash-attention VALUE-correctness on device with a trustworthy fp32 oracle (the in-env test_flash.py
CPU assert is dirty: baseline mismatched 75.1%). Prior runs were structural-only because PATCH_MODE=no_h2d makes
Tensor.to(device=spyre) return UNINITIALIZED empty_strided (no real data / no real execution). So you MUST run with
PATCH_MODE UNSET (real h2d + real device execution).

HARNESS (discovered — do not re-discover):
- venv: /home/adnan/dt-inductor/.venv ; python /home/adnan/dt-inductor/.venv/bin/python3
- test_flash.py at /tmp/test-spyre-scripts/test_flash.py (module exposes B=1,H=32,D=128,Lq=Lk=4096 and the compiled
  fn flash_spyre). Importing it as a module does NOT run its __main__ (guarded), so you can call tf.flash_spyre yourself.
- PROVEN ENV TEMPLATE: /home/adnan/codex-isolated/flash_attention_devpf_verify_20260705_070129/runs/relayout_on_current_main_backend1_20260705_191932/env.txt
  (a 'env|sort' dump). Re-export it robustly: while IFS= read -r L; do case "\$L" in PWD=*|SHLVL=*|_=*|OLDPWD=*|PS1=*|
  TERM=*|container=*) ;; *) export "\$L";; esac; done < env.txt  -- then UNSET PATCH_MODE.
- torch-spyre root: /home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/torch-spyre (ah/comms-collectives).

STEPS:
1. Write /tmp/flash_valrun.py with EXACTLY this driver (it embeds the clean fp32 oracle; do not change the math):
-----
import os, math, sys, traceback, torch
import torch_spyre
torch_spyre._autoload()
sys.path.insert(0, "/tmp/test-spyre-scripts")
import test_flash as tf
from torch_spyre._inductor.propagate_named_dims import declare_tensor_dim, name_tensor_dims
B,H,D,Lq,Lk = tf.B, tf.H, tf.D, tf.Lq, tf.Lk
torch.manual_seed(42)
q = torch.randn(B,H,Lq,D, dtype=torch.float16)
k = torch.randn(B,H,Lk,D, dtype=torch.float16)
v = torch.randn(B,H,Lk,D, dtype=torch.float16)
causal = torch.tril(torch.ones(Lq,Lk, dtype=torch.bool))
mask = torch.zeros(1,1,Lq,Lk, dtype=torch.float16); mask.masked_fill_(~causal, float("-inf"))
def oracle(q,k,v,mask):                     # clean fp32 single-pass softmax, EXACT test_flash math
    scale = 1.0/math.sqrt(math.sqrt(D))     # scale**2 == 1/sqrt(D); applied to BOTH q and k
    qf,kf,vf,mf = q.float(),k.float(),v.float(),mask.float()
    s = torch.matmul(qf*scale,(kf*scale).transpose(-1,-2)) + mf
    rm = torch.amax(s, dim=-1, keepdim=True); e = torch.exp(s-rm); den = e.sum(dim=-1, keepdim=True)
    return torch.matmul(e, vf)/den
ref = oracle(q,k,v,mask)                     # B,H,Lq,D fp32
tag = os.environ.get("RUN_TAG","run")
try:
    qs,ks,vs,ms = q.to("spyre"),k.to("spyre"),v.to("spyre"),mask.to("spyre")
    for nm,val in [("B",B),("H",H),("Lq",Lq),("Lk",Lk),("D",D)]: declare_tensor_dim(nm,val)
    name_tensor_dims(qs,["B","H","Lq","D"]); name_tensor_dims(ks,["B","H","Lk","D"])
    name_tensor_dims(vs,["B","H","Lk","D"]); name_tensor_dims(ms,["B","H","Lq","Lk"])
    out = torch.compile(tf.flash_spyre)(qs,ks,vs,ms).cpu().float()
    torch.save(out, f"/tmp/flash_out_{tag}.pt")
    diff = (out-ref).abs(); tol = 0.1 + 0.1*ref.abs()
    frac = (diff>tol).float().mean().item()
    print(f"RESULT tag={tag} maxabs={diff.max().item():.4e} meanabs={diff.mean().item():.4e} beyond_tol_frac={frac:.4f}")
    print(f"RESULT tag={tag} FLASH_ORACLE " + ("PASS" if torch.allclose(out,ref,atol=0.1,rtol=0.1) else "FAIL"))
except Exception as e:
    print(f"RESULT tag={tag} RUN_ERROR {type(e).__name__}: {e}"); traceback.print_exc()
-----
2. Run VARIANT 1 = BASELINE flash first (isolates flash-kernel correctness from collectives):
   export the proven env (UNSET PATCH_MODE), then set the 5 SPYRE_LX_PLANNER_RELAYOUT* flags = 0, RUN_TAG=baseline,
   fresh TORCHINDUCTOR_CACHE_DIR=/tmp/flash_base_cache, and run: timeout 900 \$PY /tmp/flash_valrun.py 2>&1 | tee /tmp/flash_baseline.log
3. Only if VARIANT 1 finished without a device fence, run VARIANT 2 = RELAYOUT-ON (the de-spilled path):
   same env but set the 5 RELAYOUT* flags = 1 and DEEPTOOLS_ALLOW_MIXED_HBM_IFN_DIAGNOSTIC=1,
   DEEPTOOLS_MATMUL_OPERAND_BROADCAST_KERNEL_NEIGHBOR=1, RUN_TAG=relayout_on, fresh cache, and run the driver again
   -> /tmp/flash_relayout_on.log. (This path may fail at COMPILE with an LX-chunk-fit DtException; that is an expected,
   safe, informative outcome. If instead it triggers a device fence, STOP per the rails.)
4. Parse both logs for the 'RESULT tag=..' lines.

INTERPRETATION KEY: baseline PASS => flash kernel is device-correct and the 75.1% was purely the dirty CPU ref (big
result). baseline FAIL small-fraction => genuine kernel numeric bug. relayout_on PASS => de-spill preserves value
(the win). relayout_on FAIL large-fraction or compile-fail => the KERNEL-operand wall still blocks value (expected).

Return: { run:'B_flash_value', variants:[{tag, status:'PASS'|'FAIL'|'RUN_ERROR'|'DEVICE_FENCE'|'COMPILE_FAIL',
          maxabs, meanabs, beyond_tol_frac, note}], device_ok:<bool>, headline:<str>, logs:[..], anomalies:[..] }`

const RES_SCHEMA = {
  type:'object', additionalProperties:true,
  required:['run','headline'],
  properties:{ run:{type:'string'}, headline:{type:'string'} },
}

phase('Device')
log('Firing gated device runs: (A) Granite plan-pick on CLC + (B) flash value on DEV, in parallel')
const [a, b] = await parallel([
  () => agent(CLC, { label:'runA:granite-planpick-CLC', phase:'Device', schema: RES_SCHEMA, effort:'high' }),
  () => agent(DEV, { label:'runB:flash-value-DEV', phase:'Device', schema: RES_SCHEMA, effort:'high' }),
])
return { granite_planpick: a, flash_value: b }
