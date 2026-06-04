"""sendnn (torch_sendnn baseline) device kernel-time probe for the MLP study.

Mirrors spyre-perf-suite benchmark.py run_curr_stack: registers the sendnn
backend on PrivateUse1 ("aiu"), torch.compiles with backend="sendnn",
keeps inputs on CPU (the backend moves them), blocks via result.cpu().

Profiles ProfilerActivity.PrivateUse1, prints the FULL per-event device list
and kernel_ms = sum of COMPUTE kernels excluding Memset/Memcpy and any .sum.

Run ONE variant per process (serial).

usage: mlp_probe_sendnn.py <variant> <M>
  variant in {ffn, ffn_nosilu, front_real, front_nosilu, silu_only, mm1}
"""
from __future__ import annotations
import os
import sys

EMB = 4096
INTER = 16384


def build(variant: str, torch):
    silu = torch.nn.functional.silu
    if variant == "ffn":
        return (lambda x, Wg, Wu, Wd: (silu(x @ Wg) * (x @ Wu)) @ Wd), "x,Wg,Wu,Wd"
    if variant == "ffn_nosilu":
        return (lambda x, Wg, Wu, Wd: ((x @ Wg) * (x @ Wu)) @ Wd), "x,Wg,Wu,Wd"
    if variant == "front_real":
        return (lambda x, Wg, Wu: silu(x @ Wg) * (x @ Wu)), "x,Wg,Wu"
    if variant == "front_nosilu":
        return (lambda x, Wg, Wu: (x @ Wg) * (x @ Wu)), "x,Wg,Wu"
    if variant == "silu_only":
        return (lambda a: silu(a)), "a"
    if variant == "mm1":
        return (lambda x, Wg: x @ Wg), "x,Wg"
    raise ValueError(f"unknown variant {variant}")


def make_inputs(variant: str, M: int, torch):
    x = torch.randn(M, EMB, dtype=torch.float16)
    Wg = torch.randn(EMB, INTER, dtype=torch.float16)
    Wu = torch.randn(EMB, INTER, dtype=torch.float16)
    Wd = torch.randn(INTER, EMB, dtype=torch.float16)
    a = torch.randn(M, INTER, dtype=torch.float16)
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
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

    import torch
    from torch_sendnn import torch_sendnn
    from torch.profiler import ProfilerActivity, profile

    torch_sendnn.sendnn_backend.is_available = lambda: False
    try:
        torch.utils.rename_privateuse1_backend("aiu")
        torch._register_device_module("aiu", torch_sendnn.sendnn_backend)
        torch.utils.generate_methods_for_privateuse1_backend()
    except RuntimeError as e:
        if "already been set" not in str(e):
            raise

    fn, argdesc = build(variant, torch)
    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="sendnn")
    tensors = make_inputs(variant, M, torch)

    # warm up + compile (run 1 not profiled), block via .cpu()
    for _ in range(3):
        _ = compiled(*tensors).cpu()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]) as p:
        for _ in range(runs):
            _ = compiled(*tensors).cpu()

    _MEM = ("Memcpy", "Memset", "memcpy", "memset")
    _SUM = ("_sum_", "fused_sum", "Sum")
    kern_ms = mem_ms = sum_ms = 0.0
    events = []
    for e in p.key_averages():
        if e.device_time_total > 0:
            per_iter = (e.device_time_total / runs) / 1000.0
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

    print(f"PROBE_SENDNN variant={variant} M={M} args=({argdesc}) runs={runs}",
          flush=True)
    print(f"  kernel_ms(COMPUTE)={kern_ms:.5f}  mem_ms={mem_ms:.5f}  "
          f"sum_ms={sum_ms:.5f}", flush=True)
    print("  PER-EVENT (per-iter ms | category | count):", flush=True)
    for key, t, cat, cnt in sorted(events, key=lambda z: -z[1]):
        print(f"    [{cat:7s}] {t:9.5f} ms  x{cnt:<3d}  {key}", flush=True)
    print(f"RESULT variant={variant} M={M} kernel_ms={kern_ms:.5f}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: mlp_probe_sendnn.py <variant> <M>", file=sys.stderr)
        sys.exit(1)
    run(sys.argv[1], int(sys.argv[2]))
