export const meta = {
  name: 'flash-clean-rerun',
  description: 'Clean flash value re-run on DEV: warmup call to dodge the known synchronize stall, then compare device output to fp32 oracle',
  phases: [{ title: 'FlashRerun' }],
}

const DEV = `
*** DEVICE SAFETY RAILS ***
- AIU device is precious (wedge = pod recreate). Never edit/checkout/reset any Codex file; write only under /tmp.
- If you see BusFence / RAS:: / SIGABRT / a real hardware error (NOT the benign 'possible lost completion' warmup warning): STOP, capture log, report. Do not retry on device.
- oc exec -n a6-quantization adnan-spyre-dev-pf -- bash -lc 'CMD'.
Your final text is a structured data return.

=== GOAL ===
The previous flash value run produced a NaN device output (74% beyond tol) BUT its log shows the known benign warmup
stall: 'RuntimeStream::synchronize() still waiting ... possible lost completion' (8x60s) then 'completed'. Per prior
findings this thread-lock fires on WARMUP ONLY. So: re-run flash with a WARMUP call first (compiles + fires the stall),
then a SECOND real call (uses the cached kernel, should be clean), save + compare the SECOND output to the fp32 oracle.
This definitively answers: is BASELINE flash (relayout OFF) value-correct on device, or is the NaN a genuine kernel bug
(the sparse amax(dim=-1) real_max FIXME) independent of the warmup stall?

=== HARNESS (discovered; do not re-discover) ===
- venv python: /home/adnan/dt-inductor/.venv/bin/python3
- test_flash.py: /tmp/test-spyre-scripts/test_flash.py (module: B=1,H=32,D=128,Lq=Lk=4096; fn flash_spyre). Import as
  a module -> __main__ does NOT run.
- PROVEN ENV TEMPLATE (env|sort dump): /home/adnan/codex-isolated/flash_attention_devpf_verify_20260705_070129/runs/relayout_on_current_main_backend1_20260705_191932/env.txt
  Re-export robustly: while IFS= read -r L; do case "$L" in PWD=*|SHLVL=*|_=*|OLDPWD=*|PS1=*|TERM=*|container=*) ;; *) export "$L";; esac; done < env.txt
- Device /dev/vfio/31.

=== STEPS ===
1. Confirm device free: no stale flash python proc (ps -eo pid,cmd | grep flash_valrun | grep -v grep). If a stale one
   from a prior kill is running, do NOT kill Codex procs; only proceed if the device is idle. ls /dev/vfio (expect 31).
2. Write /tmp/flash_clean.py EXACTLY (warmup + real; explicit relayout-OFF is set via env in step 3, this script just
   runs flash_spyre twice and compares the 2nd output to the oracle):
-----
import os, math, sys, traceback, torch
import torch_spyre; torch_spyre._autoload()
sys.path.insert(0, "/tmp/test-spyre-scripts")
import test_flash as tf
from torch_spyre._inductor.propagate_named_dims import declare_tensor_dim, name_tensor_dims
print("RELAYOUT_FLAGS", {k:os.environ.get(k) for k in ["SPYRE_LX_PLANNER_RELAYOUT","SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES","PATCH_MODE"]}, flush=True)
B,H,D,Lq,Lk = tf.B,tf.H,tf.D,tf.Lq,tf.Lk
torch.manual_seed(42)
q=torch.randn(B,H,Lq,D,dtype=torch.float16); k=torch.randn(B,H,Lk,D,dtype=torch.float16); v=torch.randn(B,H,Lk,D,dtype=torch.float16)
causal=torch.tril(torch.ones(Lq,Lk,dtype=torch.bool)); mask=torch.zeros(1,1,Lq,Lk,dtype=torch.float16); mask.masked_fill_(~causal,float("-inf"))
def oracle(q,k,v,mask):
    s=1.0/math.sqrt(math.sqrt(D)); qf,kf,vf,mf=q.float(),k.float(),v.float(),mask.float()
    sc=torch.matmul(qf*s,(kf*s).transpose(-1,-2))+mf; rm=torch.amax(sc,dim=-1,keepdim=True)
    e=torch.exp(sc-rm); den=e.sum(dim=-1,keepdim=True); return torch.matmul(e,vf)/den
ref=oracle(q,k,v,mask)
try:
    qs,ks,vs,ms=q.to("spyre"),k.to("spyre"),v.to("spyre"),mask.to("spyre")
    for nm,val in [("B",B),("H",H),("Lq",Lq),("Lk",Lk),("D",D)]: declare_tensor_dim(nm,val)
    name_tensor_dims(qs,["B","H","Lq","D"]); name_tensor_dims(ks,["B","H","Lk","D"]); name_tensor_dims(vs,["B","H","Lk","D"]); name_tensor_dims(ms,["B","H","Lq","Lk"])
    c=torch.compile(tf.flash_spyre)
    print("WARMUP call (expect the benign synchronize stall here)...", flush=True)
    w=c(qs,ks,vs,ms); wc=w.cpu().float(); print("WARMUP done finite=",bool(torch.isfinite(wc).all()),"absmax=",float(wc.abs().max()), flush=True)
    print("REAL call (cached kernel, should be clean)...", flush=True)
    out=c(qs,ks,vs,ms).cpu().float()
    torch.save(out,"/tmp/flash_out_clean.pt")
    dif=(out-ref).abs(); tol=0.1+0.1*ref.abs()
    print("REAL finite=",bool(torch.isfinite(out).all()),"absmax=",float(out.abs().max()),"zeros_frac=",round(float((out==0).float().mean()),5), flush=True)
    print("RESULT maxabs=%.4e meanabs=%.4e beyond_tol_frac=%.4f"%(dif.max().item(),dif.mean().item(),(dif>tol).float().mean().item()), flush=True)
    print("RESULT FLASH_ORACLE", "PASS" if torch.allclose(out,ref,atol=0.1,rtol=0.1) else "FAIL", flush=True)
    # also compare WARMUP output, to see if only the first call is bad
    dw=(wc-ref).abs(); print("WARMUP_CMP beyond_tol_frac=%.4f finite=%s"%((dw>tol).float().mean().item(), bool(torch.isfinite(wc).all())), flush=True)
except Exception as e:
    print("RESULT RUN_ERROR", type(e).__name__, str(e), flush=True); traceback.print_exc()
-----
3. Launch it DETACHED with a long timeout so it survives (compile+warmup can take ~15 min):
     RUN dir /tmp/flash_clean_$(date +%s); mkdir it.
     In one oc exec: cd there; source the proven env (re-export env.txt per above); UNSET PATCH_MODE; set the 5
     SPYRE_LX_PLANNER_RELAYOUT* = 0 (baseline, relayout OFF); export TORCHINDUCTOR_CACHE_DIR=$RUN/cache; then
     nohup timeout 2400 /home/adnan/dt-inductor/.venv/bin/python3 /tmp/flash_clean.py > $RUN/clean.log 2>&1 &  ; echo started pid $!
4. POLL: loop up to ~28 minutes: every iteration do a single oc exec that 'sleep 90; tail -3 $RUN/clean.log; grep -c RESULT $RUN/clean.log'. Stop polling as soon as a line matching 'RESULT ' (FLASH_ORACLE/RUN_ERROR) appears, or on a real fence, or on timeout-kill.
5. When done, grep $RUN/clean.log for RELAYOUT_FLAGS, WARMUP done, REAL finite, RESULT lines. Report them.

INTERPRETATION: REAL PASS => baseline flash IS device-correct; the earlier NaN was the warmup stall; 75% was the dirty
oracle after all -> flash unblocked. REAL FAIL/NaN but WARMUP also NaN => genuine kernel bug (sparse amax real_max FIXME
/ Lk-not-tiled), independent of the stall -> flash blocked on the reduce/coarse-tile lane, not the collective lane.

Return {run:'flash_clean', relayout:'off', device_ok:<bool>, warmup_finite:<bool>, real_finite:<bool>,
        real_beyond_tol_frac:<num or null>, oracle:'PASS'|'FAIL'|'RUN_ERROR'|'TIMEOUT', headline:<str>, log_path:<str>}`

const SCHEMA = { type:'object', additionalProperties:true, required:['run','headline'],
  properties:{ run:{type:'string'}, headline:{type:'string'} } }

phase('FlashRerun')
log('Clean flash re-run on DEV (warmup-dodge, relayout OFF), detached + polled')
const res = await agent(DEV, { label:'flash-clean-DEV', phase:'FlashRerun', schema: SCHEMA, effort:'high' })
return { flash_clean: res }
