"""Time a matmul under a forced (m,n,k) split, with the dbg config knobs
that make the work_division patch actually fire (compile_threads=1,
fx_graph_cache off, dynamo.reset). 3D forcing: M=<m>,N=<n>,K=<k>.

Pass force_spec 'DEFAULT' to measure the planner's unforced choice
(prints the chosen split).

Usage: python3 force_split_mnk.py <M> <K> <N> M=<m>,N=<n>,K=<k>|DEFAULT [runs]
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
    is_default = force_spec.upper() == "DEFAULT"
    forced = {} if is_default else parse_split_spec(force_spec)
    os.environ.setdefault("USE_SPYRE_PROFILER", "1")
    os.system("rm -rf /tmp/torchinductor_adnan")

    import torch
    import torch._inductor.config as _icfg
    _icfg.compile_threads = 1
    _icfg.worker_start_method = "fork"
    _icfg.fx_graph_cache = False
    _icfg.fx_graph_remote_cache = False
    import torch._dynamo as _dynamo
    import torch_spyre._inductor.work_division as wd
    from torch.profiler import ProfilerActivity, profile
    from torch_spyre._inductor.pass_utils import concretize_expr

    print("torch_spyre.wd from:", wd.__file__, flush=True)

    orig_split = wd.multi_dim_iteration_space_split
    elems_per_stick = 64
    M_size = M
    N_size_sticks = N // elems_per_stick
    K_size_sticks = K // elems_per_stick

    seen = {"chosen": None}

    def patched_split(it_space, max_cores, output_dims, reduction_dims, min_splits=None):
        splits = orig_split(it_space, max_cores, output_dims, reduction_dims, min_splits)
        sizes = {s: int(concretize_expr(e)) for s, e in it_space.items()}
        labels = {}
        for sym in reduction_dims:
            if sym in sizes and sizes[sym] == K_size_sticks:
                labels[sym] = "K"
        for sym in output_dims:
            if sym in labels:
                continue
            if sizes.get(sym) == M_size:
                labels[sym] = "M"
            elif sizes.get(sym) == N_size_sticks:
                labels[sym] = "N"
        # Report planner default ONLY for the matmul iteration space: it must
        # contain M, N-sticks AND K-sticks (3 dims). Avoids capturing the
        # .sum() reduction op (has M and N but no K-sticks dim).
        is_matmul_space = (
            any(v == M_size for v in sizes.values())
            and any(v == N_size_sticks for v in sizes.values())
            and any(v == K_size_sticks for v in sizes.values())
        )
        if is_matmul_space:
            default_named = {labels.get(s, str(s)): int(v) for s, v in splits.items()}
            seen["chosen"] = default_named
            print(f"DEFAULT_SPLIT[matmul] it_space_sizes={sizes} -> "
                  f"orig_splits_named={default_named}", flush=True)
        if not is_default and len(labels) >= len(forced):
            applied = {}
            for sym, lbl in labels.items():
                if lbl in forced:
                    splits[sym] = forced[lbl]
                    applied[str(sym)] = (lbl, forced[lbl])
            cores = 1
            for v in splits.values():
                cores *= v
            print(f"FORCED on it_space_sizes={sizes} -> applied={applied} "
                  f"cores_used={cores}", flush=True)
        return splits

    wd.multi_dim_iteration_space_split = patched_split

    _dynamo.reset()
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
    print(f"RESULT shape=[{M}x{K}x{N}] force={force_spec} chosen={seen['chosen']} "
          f"kernel_ms={kern:.4f} mem_ms={mem:.4f} spyre_ms={kern+mem:.4f}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) not in (5, 6):
        print("usage: force_split_mnk.py <M> <K> <N> M=<m>,N=<n>,K=<k>|DEFAULT [runs]",
              file=sys.stderr)
        sys.exit(1)
    rr = int(sys.argv[5]) if len(sys.argv) == 6 else 8
    run(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], rr)
