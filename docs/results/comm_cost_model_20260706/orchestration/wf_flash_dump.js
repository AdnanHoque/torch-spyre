export const meta = {
  name: 'flash-intermediate-dump',
  description: 'Dump device pre-divide numerator + denominator from flash to localize the causal-boundary bug to num vs den',
  phases: [{ title: 'Dump' }],
}

const DEV = `
*** DEVICE SAFETY RAILS *** AIU precious (wedge=pod recreate). Never edit/checkout Codex files; write only /tmp.
Benign 'possible lost completion' warmup warning is OK; a real BusFence/RAS::/SIGABRT => STOP, capture, report, no retry.
oc exec -n a6-quantization adnan-spyre-dev-pf -- bash -lc 'CMD'. Final text is a structured data return.

=== GOAL ===
The flash device output is wrong at small causal-boundary query positions (lq<128), error decaying ~1/lq, output norm
EXCEEDS the softmax bound. Ruled out (via CPU analysis of the final tensor): mask value/leak, online-softmax init,
num/den mask-inconsistency, wrong max, and stale-memory contamination (residuals decorrelated). Need the device's
PRE-DIVIDE numerator and denominator SEPARATELY to localize the bug. flash_spyre returns 'output / denominator'; we run
a byte-identical copy that instead returns (output_pre_divide, denominator), so the upstream compute is unchanged and
the bug reproduces. Then compare each to the fp32 reference at small lq.

=== HARNESS (discovered) ===
- venv python /home/adnan/dt-inductor/.venv/bin/python3 ; test_flash.py at /tmp/test-spyre-scripts/test_flash.py
- PROVEN ENV: /home/adnan/codex-isolated/flash_attention_devpf_verify_20260705_070129/runs/relayout_on_current_main_backend1_20260705_191932/env.txt
  Re-export: while IFS= read -r L; do case "$L" in PWD=*|SHLVL=*|_=*|OLDPWD=*|PS1=*|TERM=*|container=*) ;; *) export "$L";; esac; done < env.txt
  Then UNSET PATCH_MODE and set the 5 SPYRE_LX_PLANNER_RELAYOUT* = 0 (baseline). Device /dev/vfio/31.

=== STEPS ===
1. Confirm device idle (no stale flash python proc; ls /dev/vfio shows 31).
2. Write /tmp/flash_dump.py EXACTLY (it copies flash_spyre from the test module but returns pre-divide (output, denom);
   all constants + spyre_hint come from the imported test_flash module so it is byte-identical upstream):
-----
import os, math, sys, traceback, torch
import torch_spyre; torch_spyre._autoload()
sys.path.insert(0,"/tmp/test-spyre-scripts")
import test_flash as tf
from torch_spyre._inductor import spyre_hint
from torch_spyre._inductor.propagate_named_dims import declare_tensor_dim, name_tensor_dims
B,H,D,Lq,Lk = tf.B,tf.H,tf.D,tf.Lq,tf.Lk
def flash_dbg(queries, keys, values, mask):
    scale=1.0/math.sqrt(math.sqrt(D))
    output=torch.zeros_like(queries)
    real_max=torch.full((B,H,Lq,64),float("-inf"),device=queries.device,dtype=torch.float16).amax(dim=-1)
    denominator=torch.zeros((B,H,Lq,64),device=queries.device,dtype=torch.float16).amax(dim=-1)
    with spyre_hint(tiles={"B":B//tf.b_block_size}):
      with spyre_hint(tiles={"H":H//tf.h_block_size}):
        with spyre_hint(tiles={"Lq":Lq//tf.q_block_size}):
          with spyre_hint(tiles={"Lk":Lk//tf.kv_block_size}):
            with spyre_hint(work_div={"H":4,"Lq":8,"Lk":8}):
              scaled_keys=keys*scale; keys_T=scaled_keys.transpose(-1,-2)
              scores=torch.matmul(queries*scale,keys_T); scores=scores+mask
              block_max=torch.amax(scores,dim=-1); running_max=torch.maximum(real_max,block_max)
              exp_scores=torch.exp(scores-running_max.unsqueeze(-1)); correction=torch.exp(real_max-running_max)
              denominator.copy_(denominator*correction+exp_scores.sum(dim=-1))
              output.copy_(output*correction.unsqueeze(-1)+torch.matmul(exp_scores,values))
              real_max.copy_(running_max)
    return output, denominator      # PRE-DIVIDE
torch.manual_seed(42)
q=torch.randn(B,H,Lq,D,dtype=torch.float16); k=torch.randn(B,H,Lk,D,dtype=torch.float16); v=torch.randn(B,H,Lk,D,dtype=torch.float16)
causal=torch.tril(torch.ones(Lq,Lk,dtype=torch.bool)); mask=torch.zeros(1,1,Lq,Lk,dtype=torch.float16); mask.masked_fill_(~causal,float("-inf"))
# fp32 reference numerator (pre-divide) + denominator
scale=1.0/math.sqrt(math.sqrt(D)); s=torch.matmul(q.float()*scale,(k.float()*scale).transpose(-1,-2))+mask.float()
rm=torch.amax(s,-1,keepdim=True); e=torch.exp(s-rm); ref_num=torch.matmul(e,v.float()); ref_den=e.sum(-1)  # (1,H,Lq,D),(1,H,Lq)
try:
    qs,ks,vs,ms=q.to("spyre"),k.to("spyre"),v.to("spyre"),mask.to("spyre")
    for nm,val in [("B",B),("H",H),("Lq",Lq),("Lk",Lk),("D",D)]: declare_tensor_dim(nm,val)
    name_tensor_dims(qs,["B","H","Lq","D"]); name_tensor_dims(ks,["B","H","Lk","D"]); name_tensor_dims(vs,["B","H","Lk","D"]); name_tensor_dims(ms,["B","H","Lq","Lk"])
    c=torch.compile(flash_dbg)
    print("WARMUP...",flush=True); _=c(qs,ks,vs,ms)
    print("REAL...",flush=True)
    dev_num,dev_den=c(qs,ks,vs,ms); dev_num=dev_num.cpu().float(); dev_den=dev_den.cpu().float()
    torch.save({"num":dev_num,"den":dev_den},"/tmp/flash_dump.pt")
    print("=== DENOMINATOR at small lq (head0). ref_den[lq]=# valid-key softmax mass ===",flush=True)
    for lq in [0,1,3,7,15,31,63,127]:
        print(f"  lq={lq:4d} dev_den={dev_den[0,0,lq].item():.4f}  ref_den={ref_den[0,0,lq].item():.4f}  ratio={ (dev_den[0,0,lq]/ref_den[0,0,lq]).item():.4f}",flush=True)
    print("=== NUMERATOR at small lq (head0): ||dev_num|| vs ||ref_num|| and their cos ===",flush=True)
    for lq in [0,1,3,7,15,31,63,127]:
        dn=dev_num[0,0,lq]; rn=ref_num[0,0,lq]; cs=torch.dot(dn,rn).item()/(dn.norm()*rn.norm()+1e-9).item()
        print(f"  lq={lq:4d} ||dev_num||={dn.norm():.3f} ||ref_num||={rn.norm():.3f} cos={cs:+.4f}",flush=True)
    print("RESULT_DONE",flush=True)
except Exception as ex:
    print("RESULT RUN_ERROR",type(ex).__name__,str(ex),flush=True); traceback.print_exc()
-----
3. Launch DETACHED: RUN=/tmp/flash_dump_$(date +%s); mkdir it; source env (unset PATCH_MODE; relayout*=0);
   export TORCHINDUCTOR_CACHE_DIR=$RUN/cache; nohup timeout 2400 $PY /tmp/flash_dump.py > $RUN/dump.log 2>&1 & ; echo pid $!.
4. POLL every ~90s (in-pod sleep) up to ~28 min until 'RESULT_DONE' or 'RUN_ERROR' or a real fence appears.
5. Report the DENOMINATOR and NUMERATOR comparison tables verbatim, plus your verdict: is the boundary bug in the
   DENOMINATOR (dev_den/ref_den ratio != 1 at small lq) or the NUMERATOR (cos < 1 / wrong norm) or BOTH?

INTERPRETATION: at lq=0 the correct denominator is ~1.0 (one valid key) and numerator ~ V[0]. If dev_den is too small
=> normalization/sum bug; if dev_num direction wrong (cos<1) => the exp_scores@V accumulation is picking up masked keys
or a tiling artifact. This isolates the defect to one op.

Return {run:'flash_dump', device_ok:<bool>, denom_table:[...], num_table:[...], verdict:<str>, headline:<str>, log_path:<str>}`

const SCHEMA={type:'object',additionalProperties:true,required:['run','headline'],properties:{run:{type:'string'},headline:{type:'string'}}}
phase('Dump')
log('Flash intermediate dump on DEV: localize causal-boundary bug to numerator vs denominator')
const res = await agent(DEV,{label:'flash-dump-DEV',phase:'Dump',schema:SCHEMA,effort:'high'})
return { flash_dump: res }
