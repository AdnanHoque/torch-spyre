"""Force-split harness for matmuls where M is folded out (M=1 decode).

Iteration space is 2D: (N_sticks, K_sticks). Spec is N=<n>,K=<k>.
"""

from __future__ import annotations
import os
import sys


sys.path.insert(0, "/tmp/pr-mixed-splits-cost-model")
sys.meta_path = [
    f for f in sys.meta_path
    if not (type(f).__name__.endswith("EditableFinder") and "torch_spyre" in repr(f))
]


def parse_split_spec(spec: str) -> dict[str, int]:
    return {k: int(v) for k, v in (p.split("=") for p in spec.split(",") if p)}


def run(M: int, K: int, N: int, force_spec: str, runs: int = 8):
    forced = parse_split_spec(force_spec)  # {"N": ..., "K": ...}
    os.environ.setdefault("USE_SPYRE_PROFILER", "1")
    os.system("rm -rf /tmp/torchinductor_adnan")

    import torch
    import torch._inductor.config as _icfg
    _icfg.compile_threads = 1
    _icfg.worker_start_method = "fork"
    _icfg.fx_graph_cache = False
    _icfg.fx_graph_remote_cache = False
    import torch_spyre._inductor.work_division as wd
    from torch.profiler import ProfilerActivity, profile
    from torch_spyre._inductor.pass_utils import concretize_expr

    orig_split = wd.multi_dim_iteration_space_split
    elems_per_stick = 64
    N_sticks = N // elems_per_stick
    K_sticks = K // elems_per_stick

    def patched_split(it_space, max_cores, output_dims, reduction_dims, min_splits=None):
        splits = orig_split(it_space, max_cores, output_dims, reduction_dims, min_splits)
        sizes = {s: int(concretize_expr(e)) for s, e in it_space.items()}
        print(f"DBG patched_split called sizes={ {str(k):v for k,v in sizes.items()} } "
              f"N_sticks={N_sticks} K_sticks={K_sticks} forced={forced} "
              f"orig_splits={ {str(k):v for k,v in splits.items()} }", flush=True)
        labels = {}
        for sym in reduction_dims:
            if sym in sizes and sizes[sym] == K_sticks:
                labels[sym] = "K"
        for sym in output_dims:
            if sym in labels:
                continue
            if sizes.get(sym) == N_sticks:
                labels[sym] = "N"
        if len(labels) >= len(forced):
            applied = {}
            for sym, lbl in labels.items():
                if lbl in forced:
                    splits[sym] = forced[lbl]
                    applied[str(sym)] = (lbl, forced[lbl])
            cores = 1
            for v in splits.values():
                cores *= v
            print(
                f"FORCED2D on it_space_sizes={sizes}  "
                f"reduction_dims={[str(x) for x in reduction_dims]} "
                f"output_dims={[str(x) for x in output_dims]} -> "
                f"applied={applied}  cores_used={cores}"
            )
        return splits

    wd.multi_dim_iteration_space_split = patched_split

    d = torch.device("spyre")
    x = torch.randn(1, M, K, dtype=torch.float16).to(d)
    W = torch.empty(K, N, dtype=torch.float16)
    torch.nn.init.kaiming_uniform_(W)
    W = W.to(d)
    fn = lambda a, b: torch.nn.functional.linear(a, b.T)
    args = (x, W)

    compiled = torch.compile(fn, fullgraph=True)
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
    print(f"RESULT shape=[{M}x{K}x{N}] force={force_spec}  "
          f"kernel_ms={kern:.4f}  mem_ms={mem:.4f}  spyre_ms={kern+mem:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("usage: force_split_timing_2d.py <M> <K> <N> N=<n>,K=<k>", file=sys.stderr)
        sys.exit(1)
    run(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
