"""Replay a foreign program dir on THIS device. Detects the internal program
hash, stages the dir under that name, and launch_kernel + times.
Usage: replay_foreign.py <program_dir> B M N K"""
import sys, os, glob, shutil, torch
import torch_spyre; torch_spyre._autoload()
from torch_spyre._C import launch_kernel
from torch.profiler import profile, ProfilerActivity
src = sys.argv[1].rstrip("/")
B,M,N,K = (int(x) for x in sys.argv[2:6])
# detect internal hash from loadprogram_to_device/<hash>-SenProgSend
lp = glob.glob(src + "/loadprogram_to_device/*-SenProgSend")
if not lp:
    print("NO_PROGRAM (no loadprogram_to_device/*-SenProgSend)"); sys.exit(0)
hashname = os.path.basename(lp[0]).replace("-SenProgSend", "")
stage = "/tmp/replay_stage/" + hashname
shutil.rmtree("/tmp/replay_stage", ignore_errors=True); os.makedirs("/tmp/replay_stage")
shutil.copytree(src, stage)
a=torch.randn(B,M,K,dtype=torch.float16).to('spyre'); b=torch.randn(B,K,N,dtype=torch.float16).to('spyre')
out=torch.empty(B,M,N,dtype=torch.float16).to('spyre')
try:
    launch_kernel(stage,[a,b,out]); torch.spyre.synchronize()
except Exception as e:
    print("LAUNCH_ERROR:", repr(e)[:200]); sys.exit(0)
with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.PrivateUse1]) as prof:
    for _ in range(20): launch_kernel(stage,[a,b,out])
    torch.spyre.synchronize()
tot=sum((getattr(e,'self_device_time_total',0) or 0) for e in prof.key_averages())
print(f"REPLAY-FOREIGN {os.path.basename(src)} (hash {hashname}) device_us/rep={tot/20:.1f}")
