"""Force a specific (b,m,n,k) split via monkeypatch and time the kernel.

Used to validate the cost-model's TIED predictions: does (m=8, n=4) really run
in the same time as (m=4, n=8) for QO? Is (1,2,8,2) actually faster than
(1,4,8,1) for MoE?

The monkeypatch overrides ``multi_dim_iteration_space_split`` to write the
requested split for the matmul op (identified by symbol names d0..d3 in the
iteration space). No files are modified; main repo stays untouched.

Usage:
    python3 force_split_timing.py <shape> <split_spec>
where <split_spec> is e.g. "d0=1,d1=8,d2=4,d3=1".
"""

from __future__ import annotations
import os
import sys
import time
import statistics


def parse_split_spec(spec: str) -> dict[str, int]:
    return {k: int(v) for k, v in (p.split("=") for p in spec.split(",") if p)}


def run(shape: str, force_spec: str, runs: int = 8):
    forced = parse_split_spec(force_spec)
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")
    os.environ.pop("SPYRE_2D_MN_SPLIT", None)
    os.environ.setdefault("DXP_LX_FRAC_AVAIL", "0.8")
    os.system("rm -rf /tmp/torchinductor_adnan")

    import torch
    import torch_spyre._inductor.work_division as wd
    from torch.profiler import ProfilerActivity, profile

    orig_split = wd.multi_dim_iteration_space_split

    from torch_spyre._inductor.pass_utils import concretize_expr
    def patched_split(iteration_space, max_cores, output_dims, reduction_dims, min_splits=None):
        splits = orig_split(iteration_space, max_cores, output_dims, reduction_dims, min_splits)
        # Only fire when the iteration space matches the target shape's dim count
        # *and* all forced names are present (else this is a different op, e.g.
        # the trailing .sum()).
        names = {str(s) for s in iteration_space}
        if forced and names == set(forced) and len(iteration_space) == len(forced):
            for sym in list(splits.keys()):
                splits[sym] = forced[str(sym)]
            print(
                f"FORCED on it_space="
                + str({str(s): int(concretize_expr(e)) for s, e in iteration_space.items()})
                + " -> " + str({str(s): v for s, v in splits.items()})
            )
        return splits

    wd.multi_dim_iteration_space_split = patched_split

    d = torch.device("spyre")
    if shape == "QO":
        M, K, N = 512, 4096, 4096
        x = torch.randn(1, M, K, dtype=torch.float16).to(d)
        W = torch.empty(K, N, dtype=torch.float16); torch.nn.init.kaiming_uniform_(W); W = W.to(d)
        fn = lambda a, b: torch.nn.functional.linear(a, b.T)
        args = (x, W)
    elif shape == "KV":
        M, K, N = 512, 4096, 1024
        x = torch.randn(1, M, K, dtype=torch.float16).to(d)
        W = torch.empty(K, N, dtype=torch.float16); torch.nn.init.kaiming_uniform_(W); W = W.to(d)
        fn = lambda a, b: torch.nn.functional.linear(a, b.T)
        args = (x, W)
    elif shape == "MLP":
        M, K, N = 512, 4096, 12800
        x = torch.randn(1, M, K, dtype=torch.float16).to(d)
        W = torch.empty(K, N, dtype=torch.float16); torch.nn.init.kaiming_uniform_(W); W = W.to(d)
        fn = lambda a, b: torch.nn.functional.linear(a, b.T)
        args = (x, W)
    elif shape == "MoE_gateup":
        B, M, K, N = 8, 128, 2048, 8192
        A = torch.randn(B, M, K, dtype=torch.float16).to(d)
        Bm = torch.randn(B, K, N, dtype=torch.float16).to(d)
        fn = lambda a, b: torch.bmm(a, b)
        args = (A, Bm)
    elif shape == "bmm_largeK":
        B, M, K, N = 8, 512, 4096, 512
        A = torch.randn(B, M, K, dtype=torch.float16).to(d)
        Bm = torch.randn(B, K, N, dtype=torch.float16).to(d)
        fn = lambda a, b: torch.bmm(a, b)
        args = (A, Bm)
    else:
        sys.exit(f"unknown shape {shape}")

    compiled = torch.compile(fn, fullgraph=True)
    # Warmup (also triggers the monkeypatched planner).
    for _ in range(3):
        _ = compiled(*args).sum().item()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]) as p:
        for _ in range(runs):
            _ = compiled(*args).sum().item()

    _MEM = ("Memcpy", "Memset", "memcpy", "memset", "Copy", "copy")
    kern = mem = 0.0
    for e in p.key_averages():
        if e.device_time_total > 0:
            per_iter = (e.device_time_total / runs) / 1000
            if any(s in e.key for s in _MEM):
                mem += per_iter
            else:
                kern += per_iter
    print(f"RESULT shape={shape} force={force_spec}  kernel_ms={kern:.4f}  mem_ms={mem:.4f}  spyre_ms={kern+mem:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: force_split_timing.py <shape> <d0=...,d1=...,d2=...,d3=...>", file=sys.stderr)
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])
