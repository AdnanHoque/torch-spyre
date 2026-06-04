"""MLP prefill timeline probe: export a Chrome trace and inspect it.

Adapts /tmp/mlp_probe_tsp.py. Runs the SwiGLU FFN (prefill M=512, emb=4096,
inter=16384, fp16) on the spyre device under torch.profiler with
[CPU, PrivateUse1], then:
  1. exports a chrome trace to /tmp/mlp_trace.json
  2. prints the key_averages() device table (the "screenshot" table format)
  3. scans the chrome trace JSON for device kernel / Memcpy / Memset events
"""
from __future__ import annotations
import json
import os
import sys
from collections import Counter

# Match the perf-suite tsp env: resolve torch_spyre from the shim worktree.
sys.meta_path = [
    f for f in sys.meta_path
    if not (type(f).__name__.endswith("EditableFinder") and "torch_spyre" in repr(f))
]

EMB = 4096
INTER = 16384
M = 512
TRACE = "/tmp/mlp_trace.json"


def main():
    os.environ.setdefault("USE_SPYRE_PROFILER", "1")
    os.system("rm -rf /tmp/torchinductor_adnan")

    import torch
    import torch._inductor.config as _icfg
    _icfg.compile_threads = 1
    _icfg.worker_start_method = "fork"
    _icfg.fx_graph_cache = False
    _icfg.fx_graph_remote_cache = False
    from torch.profiler import ProfilerActivity, profile

    print("torch", torch.__version__, flush=True)
    from torch.autograd import _supported_activities
    print("supported activities:", _supported_activities(), flush=True)

    silu = torch.nn.functional.silu
    fn = lambda x, Wg, Wu, Wd: (silu(x @ Wg) * (x @ Wu)) @ Wd

    d = torch.device("spyre")
    x = torch.randn(M, EMB, dtype=torch.float16).to(d)
    Wg = torch.randn(EMB, INTER, dtype=torch.float16).to(d)
    Wu = torch.randn(EMB, INTER, dtype=torch.float16).to(d)
    Wd = torch.randn(INTER, EMB, dtype=torch.float16).to(d)

    compiled = torch.compile(fn, fullgraph=True)

    for _ in range(3):
        _ = compiled(x, Wg, Wu, Wd).sum().item()

    runs = 10
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
        record_shapes=True,
        profile_memory=True,
    ) as p:
        for _ in range(runs):
            _ = compiled(x, Wg, Wu, Wd).sum().item()

    # --- (A) the device key_averages() TABLE (screenshot format) ---
    print("\n========== key_averages() device table ==========", flush=True)
    print(
        p.key_averages().table(
            sort_by="self_device_time_total", row_limit=25
        ).replace("CUDA", "AIU"),
        flush=True,
    )

    # --- (B) export chrome trace and inspect for DEVICE events ---
    p.export_chrome_trace(TRACE)
    with open(TRACE) as f:
        data = json.load(f)
    events = data.get("traceEvents", [])
    print(f"\n========== chrome trace: {TRACE} ==========", flush=True)
    print(f"total traceEvents: {len(events)}", flush=True)

    cats = Counter(e.get("cat", "<none>") for e in events if e.get("ph") == "X")
    print("event categories (ph=X):", flush=True)
    for c, n in cats.most_common():
        print(f"  {c:24s} {n}", flush=True)

    # Device-side categories per the research doc: kernel, gpu_memset,
    # gpu_memcpy, privateuse1_runtime/driver.
    DEVICE_CATS = {
        "kernel", "gpu_memset", "gpu_memcpy", "gpu_user_annotation",
        "privateuse1_runtime", "privateuse1_driver",
    }
    dev_events = [e for e in events if e.get("cat") in DEVICE_CATS]
    print(f"\nDEVICE-side events (kernel/memset/memcpy/runtime/driver): "
          f"{len(dev_events)}", flush=True)

    # Show sample kernel-cat events (the matmul/sdsc bundle rows)
    kern = [e for e in events if e.get("cat") == "kernel"]
    print(f"\ncat=kernel events: {len(kern)}", flush=True)
    seen = set()
    for e in kern:
        name = e.get("name", "")
        short = name.split("/")[-2] if name.endswith("bundle.mlir") else name
        key = short
        if key in seen:
            continue
        seen.add(key)
        print(f"  dur={e.get('dur')}us  pid={e.get('pid')} tid={e.get('tid')}  "
              f"name={short}", flush=True)
        if len(seen) >= 15:
            break

    # Memset/Memcpy
    for cat in ("gpu_memset", "gpu_memcpy"):
        evs = [e for e in events if e.get("cat") == cat]
        if evs:
            print(f"\ncat={cat}: {len(evs)} events, e.g. "
                  f"name={evs[0].get('name')} dur={evs[0].get('dur')}us",
                  flush=True)

    # Device tracks (process_name / thread_name metadata for AIU)
    print("\ndevice track metadata (process/thread names):", flush=True)
    for e in events:
        if e.get("ph") == "M" and e.get("name") in ("process_name", "thread_name"):
            args = e.get("args", {})
            nm = args.get("name", "")
            if any(k in str(nm) for k in ("AIU", "Spyre", "spyre", "Device", "stream")):
                print(f"  {e['name']}: pid={e.get('pid')} tid={e.get('tid')} "
                      f"-> {nm}", flush=True)

    print(f"\nVERDICT device_events_in_chrome_trace="
          f"{len(dev_events) > 0}", flush=True)


if __name__ == "__main__":
    main()
