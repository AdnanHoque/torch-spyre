"""Pure data-movement HBM bandwidth sweep on Spyre AIU.

Compiles a trivial memory-bound pointwise op (a * 2.0) on-device and measures
the device COMPUTE kernel time (excluding Memcpy/Memset) for a single fp16
tensor of N elements. Traffic model: 1 read of x + 1 write of y => 2*N*2 bytes.

Run ONE size per process invocation (serial) so the device is never shared.

usage: hbm_bw_sweep.py <size_MB> [op]
   op in {mul2, addx, add1}  (default mul2)
"""

from __future__ import annotations
import os
import sys


sys.path.insert(0, "/tmp/pr-mixed-splits-cost-model")
sys.meta_path = [
    f for f in sys.meta_path
    if not (type(f).__name__.endswith("EditableFinder") and "torch_spyre" in repr(f))
]


def build_fn(op: str):
    if op == "mul2":
        return lambda a: a * 2.0, "y = x * 2.0  (1 read + 1 write)"
    if op == "addx":
        return lambda a: a + a, "y = x + x    (1 read + 1 write)"
    if op == "add1":
        return lambda a: a + 1.0, "y = x + 1.0  (1 read + 1 write)"
    raise ValueError(f"unknown op {op}")


def run(size_mb: float, op: str, runs: int = 10):
    os.environ.setdefault("USE_SPYRE_PROFILER", "1")
    os.system("rm -rf /tmp/torchinductor_adnan")

    import torch
    import torch._inductor.config as _icfg
    _icfg.compile_threads = 1
    _icfg.worker_start_method = "fork"
    _icfg.fx_graph_cache = False
    _icfg.fx_graph_remote_cache = False
    from torch.profiler import ProfilerActivity, profile

    fn, traffic_desc = build_fn(op)

    # N = size_MB * 1e6 / 2 (fp16 = 2 bytes). Round to multiple of 64 (stick).
    N = int(round(size_mb * 1e6 / 2))
    N = (N // 64) * 64
    bytes_per_tensor = N * 2

    d = torch.device("spyre")
    x = torch.randn(N, dtype=torch.float16).to(d)

    compiled = torch.compile(fn, fullgraph=True)

    # warm up
    for _ in range(3):
        _ = compiled(x).sum().item()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]) as p:
        for _ in range(runs):
            _ = compiled(x).sum().item()

    # The blocking .sum().item() spawns its OWN device reduction kernel
    # (sdsc_fused_sum_...). Exclude it: count ONLY the target pointwise kernel.
    _MEM = ("Memcpy", "Memset", "memcpy", "memset")
    _SUM = ("_sum_", "fused_sum")
    kern_ms = mem_ms = sum_ms = 0.0
    kernel_keys = []
    for e in p.key_averages():
        if e.device_time_total > 0:
            per_iter = (e.device_time_total / runs) / 1000.0  # us->ms
            if any(s in e.key for s in _MEM):
                mem_ms += per_iter
            elif any(s in e.key for s in _SUM):
                sum_ms += per_iter  # blocking reduction, NOT the streamed op
            else:
                kern_ms += per_iter
                kernel_keys.append((e.key, per_iter))

    traffic_bytes = 2 * N * 2  # 1 read + 1 write, fp16
    gbps = traffic_bytes / (kern_ms * 1e6) if kern_ms > 0 else 0.0
    pct = gbps / 204.8 * 100.0

    real_kernel = kern_ms > 0 and len(kernel_keys) > 0
    print(
        f"BWROW op={op} size_MB={size_mb:g} N={N} "
        f"tensor_MB={bytes_per_tensor/1e6:.3f} traffic_MB={traffic_bytes/1e6:.3f} "
        f"t_ms={kern_ms:.5f} mem_ms={mem_ms:.5f} sum_ms={sum_ms:.5f} "
        f"GBps={gbps:.2f} pct_spec={pct:.1f} "
        f"real_kernel={real_kernel}",
        flush=True,
    )
    print(f"  traffic_model: {traffic_desc} -> traffic_bytes = 2*N*2", flush=True)
    print("  kernel_keys (per-iter ms):", flush=True)
    for k, t in kernel_keys:
        print(f"    {k}: {t:.5f} ms", flush=True)
    if not real_kernel:
        print("  WARNING: no real device kernel detected (op may be a no-op)",
              flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: hbm_bw_sweep.py <size_MB> [op]", file=sys.stderr)
        sys.exit(1)
    size = float(sys.argv[1])
    op = sys.argv[2] if len(sys.argv) > 2 else "mul2"
    run(size, op)
