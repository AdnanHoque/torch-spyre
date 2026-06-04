"""Harness B (production benchmark.py) with a FULL per-event device dump.

Reuses spyre-perf-suite benchmark.py's run_tsp_stack / run_curr_stack exactly
(same compile path, same per-call .to(device) weight reload), but additionally
prints the full key_averages device breakdown split into COMPUTE / Memcpy
(HtoD) / Memcpy (DtoH) / Memset, plus the benchmark's own kernel_ms & spyre_ms.

usage: mlp_harnessB_probe.py <stack> <M>
  stack in {tsp, sendnn}; M = 512 (prefill) or 4 (decode)
"""
import os
import sys

sys.path.insert(0, "/tmp/spyre-perf-suite")
EMB = 4096


def main(stack: str, M: int, runs: int = 5):
    import torch
    from torch.profiler import ProfilerActivity, profile
    import benchmark as B

    input_shapes = [(M, EMB)]
    op = "mlp"

    if stack == "tsp":
        device = torch.device("spyre")
        torch._dynamo.reset()
        compiled = torch.compile(B.get_function(op, torch, "tsp"))
        cpu_tensors = B.create_tensors(torch, input_shapes, op, "tsp")
        # warmup / compile (matches without_compilation=True, start=1)
        compiled(*tuple(t.to(device) for t in cpu_tensors))
        run_tensors_each = lambda: tuple(t.to(device) for t in cpu_tensors)
        block = lambda r: r.cpu()
    else:
        os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
        from torch_sendnn import torch_sendnn
        torch_sendnn.sendnn_backend.is_available = lambda: False
        try:
            torch.utils.rename_privateuse1_backend("aiu")
            torch._register_device_module("aiu", torch_sendnn.sendnn_backend)
            torch.utils.generate_methods_for_privateuse1_backend()
        except RuntimeError as e:
            if "already been set" not in str(e):
                raise
        torch._dynamo.reset()
        compiled = torch.compile(B.get_function(op, torch, "sendnn"), backend="sendnn")
        cpu_tensors = B.create_tensors(torch, input_shapes, op, "sendnn")
        compiled(*cpu_tensors)
        run_tensors_each = lambda: cpu_tensors
        block = lambda r: r.cpu()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
                 record_shapes=True, profile_memory=True) as p:
        for _ in range(runs):
            r = compiled(*run_tensors_each())
            block(r)

    # benchmark.py-style spyre_ms: sum of ALL raw device events / runs
    spyre_ms = sum(e.device_time_total for e in p.events()) / 1000.0 / runs

    _MEM = ("Memcpy", "Memset", "memcpy", "memset")
    _SUM = ("_sum_", "fused_sum", "Sum")
    cats = {"COMPUTE": 0.0, "HtoD": 0.0, "DtoH": 0.0, "Memset": 0.0, "SUM": 0.0}
    rows = []
    for e in p.key_averages():
        if e.device_time_total <= 0:
            continue
        per_iter = (e.device_time_total / runs) / 1000.0
        k = e.key
        if any(s in k for s in _MEM):
            if "HtoD" in k:
                cat = "HtoD"
            elif "DtoH" in k:
                cat = "DtoH"
            else:
                cat = "Memset"
        elif any(s in k for s in _SUM):
            cat = "SUM"
        else:
            cat = "COMPUTE"
        cats[cat] += per_iter
        rows.append((cat, per_iter, e.count, k))

    print(f"HARNESS_B stack={stack} M={M} runs={runs}", flush=True)
    print(f"  spyre_ms(all-events sum/run)={spyre_ms:.4f}", flush=True)
    print(f"  COMPUTE={cats['COMPUTE']:.4f}  HtoD={cats['HtoD']:.4f}  "
          f"DtoH={cats['DtoH']:.4f}  Memset={cats['Memset']:.4f}  "
          f"SUM={cats['SUM']:.4f}", flush=True)
    kern_total = cats["COMPUTE"]
    print(f"  kernel_ms(COMPUTE sum)={kern_total:.4f}", flush=True)
    print("  PER-EVENT (per-iter ms | cat | count):", flush=True)
    for cat, t, cnt, k in sorted(rows, key=lambda z: -z[1]):
        short = k
        if "/" in short and short.endswith("bundle.mlir"):
            short = short.split("/")[-2]
        print(f"    [{cat:7s}] {t:10.5f} ms  x{cnt:<4d} {short}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
