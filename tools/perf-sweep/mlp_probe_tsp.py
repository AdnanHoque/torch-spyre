"""TSP (torch-spyre proxy) device kernel-time probe for the MLP/FFN study.

Compiles a named variant lambda on the "spyre" device, profiles
ProfilerActivity.PrivateUse1, and prints the FULL per-event device list
(every kernel/Memset/Memcpy with its per-iter ms) plus kernel_ms =
sum of COMPUTE kernels excluding Memset/Memcpy and any .sum reduction.

Run ONE variant per process (serial) so the device is never shared.

usage: mlp_probe_tsp.py <variant> <M>
  variant in {ffn, front_real, front_nosilu, silu_only, mm1, ffn_nosilu}
  M = sequence length (512 prefill, 4 decode)

Shapes (SwiGLU FFN): emb=4096, intermediate=16384, fp16.
  Wg,Wu = [4096,16384]; Wd = [16384,4096]
  x = [M,4096]; a,b = [M,16384]
"""
from __future__ import annotations
import os
import sys

# Match the perf-suite tsp env: import torch_spyre from the shim worktree.
# PYTHONPATH=/tmp/cost_model_unified_shim is set by the caller; the shim's
# sitecustomize.py redirects torch_spyre to /tmp/cost-model-unified.
# Also strip any EditableFinder that points torch_spyre at the main repo.
sys.meta_path = [
    f for f in sys.meta_path
    if not (type(f).__name__.endswith("EditableFinder") and "torch_spyre" in repr(f))
]

EMB = 4096
INTER = 16384


def build(variant: str, torch):
    silu = torch.nn.functional.silu
    if variant == "ffn":
        # full FFN: ((silu(x@Wg))*(x@Wu)) @ Wd
        return (lambda x, Wg, Wu, Wd: (silu(x @ Wg) * (x @ Wu)) @ Wd), "x,Wg,Wu,Wd"
    if variant == "ffn_nosilu":
        return (lambda x, Wg, Wu, Wd: ((x @ Wg) * (x @ Wu)) @ Wd), "x,Wg,Wu,Wd"
    if variant == "front_real":
        # silu(x@Wg) * (x@Wu)  -> [M,16384]
        return (lambda x, Wg, Wu: silu(x @ Wg) * (x @ Wu)), "x,Wg,Wu"
    if variant == "front_nosilu":
        return (lambda x, Wg, Wu: (x @ Wg) * (x @ Wu)), "x,Wg,Wu"
    if variant == "silu_only":
        # SiLU on [M,16384] alone
        return (lambda a: silu(a)), "a"
    if variant == "mm1":
        return (lambda x, Wg: x @ Wg), "x,Wg"
    raise ValueError(f"unknown variant {variant}")


def make_inputs(variant: str, M: int, torch, d):
    x = torch.randn(M, EMB, dtype=torch.float16).to(d)
    Wg = torch.randn(EMB, INTER, dtype=torch.float16).to(d)
    Wu = torch.randn(EMB, INTER, dtype=torch.float16).to(d)
    Wd = torch.randn(INTER, EMB, dtype=torch.float16).to(d)
    a = torch.randn(M, INTER, dtype=torch.float16).to(d)
    if variant in ("ffn", "ffn_nosilu"):
        return (x, Wg, Wu, Wd)
    if variant in ("front_real", "front_nosilu"):
        return (x, Wg, Wu)
    if variant == "silu_only":
        return (a,)
    if variant == "mm1":
        return (x, Wg)
    raise ValueError(variant)


def run(variant: str, M: int, runs: int = 10):
    os.environ.setdefault("USE_SPYRE_PROFILER", "1")
    os.system("rm -rf /tmp/torchinductor_adnan")

    import torch
    import torch._inductor.config as _icfg
    _icfg.compile_threads = 1
    _icfg.worker_start_method = "fork"
    _icfg.fx_graph_cache = False
    _icfg.fx_graph_remote_cache = False
    from torch.profiler import ProfilerActivity, profile

    fn, argdesc = build(variant, torch)
    d = torch.device("spyre")
    inputs = make_inputs(variant, M, torch, d)

    compiled = torch.compile(fn, fullgraph=True)

    # warm up (compile + a few runs), result blocked via .sum().item()
    for _ in range(3):
        _ = compiled(*inputs).sum().item()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]) as p:
        for _ in range(runs):
            _ = compiled(*inputs).sum().item()

    _MEM = ("Memcpy", "Memset", "memcpy", "memset")
    _SUM = ("_sum_", "fused_sum")
    kern_ms = mem_ms = sum_ms = 0.0
    events = []  # (key, per_iter_ms, category)
    for e in p.key_averages():
        if e.device_time_total > 0:
            per_iter = (e.device_time_total / runs) / 1000.0  # us->ms
            if any(s in e.key for s in _MEM):
                cat = "MEM"
                mem_ms += per_iter
            elif any(s in e.key for s in _SUM):
                cat = "SUM"
                sum_ms += per_iter
            else:
                cat = "COMPUTE"
                kern_ms += per_iter
            events.append((e.key, per_iter, cat, e.count))

    print(f"PROBE_TSP variant={variant} M={M} args=({argdesc}) runs={runs}",
          flush=True)
    print(f"  kernel_ms(COMPUTE)={kern_ms:.5f}  mem_ms={mem_ms:.5f}  "
          f"sum_ms={sum_ms:.5f}", flush=True)
    print("  PER-EVENT (per-iter ms | category | count):", flush=True)
    for key, t, cat, cnt in sorted(events, key=lambda z: -z[1]):
        short = key
        if "/" in short:
            short = short.split("/")[-2] if short.endswith("bundle.mlir") else short.split("/")[-1]
        print(f"    [{cat:7s}] {t:9.5f} ms  x{cnt:<3d}  {short}", flush=True)
    print(f"RESULT variant={variant} M={M} kernel_ms={kern_ms:.5f}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: mlp_probe_tsp.py <variant> <M>", file=sys.stderr)
        sys.exit(1)
    run(sys.argv[1], int(sys.argv[2]))
