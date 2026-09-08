"""Measure unchanged spyre-inference attention: one sequence, FP16,
8 KV heads, 32 query heads, head size 128, page size 128.
Prefill: 512 query tokens / KV=512. Decode: one query token / KV=513.

Run OFF and ON in separate processes on the same idle device:
  python paged.py --phase prefill --relayout off --out prefill-off
  python paged.py --phase prefill --relayout on  --out prefill-on
For decode, replace "prefill" with "decode" in both commands.
Read device_median_ms; OFF / ON is the speedup. Do not use wall_median_us
as device time. Compilation and warmup are excluded.
Compare output.pt tensors before attributing a timing difference to relayout.

Use a working Spyre environment with the LX stack (#4283, #4284, #3440,
#4152, #4153, #3955) plus #4347 installed. Measured compiler:
d42522b7da19d06dd03c67adb021203d78023d9b (#4347 was later rebased).
Measured spyre-inference: caad95117cb2f338fa72191105916e685b29efb6,
main af188385 + #796 dd41afe9. This is the slot-major path, NOT PR #783.
Software: Torch 2.13.0+cpu, vLLM 0.28.0, Transformers 5.14.1, Python 3.12.
Keep your usual SDK, device profiler and model-cache setup. Run serially.

Self-contained adaptation of our measured harness; no application source edits.
Device-replayed OFF/ON on 2026-09-08; results and setup are in README.md.
Different source/runtime versions may give different timings.
"""

# Compiler settings must be established before importing Torch/Spyre.
# ruff: noqa: E402
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("--phase", choices=("prefill", "decode"), required=True)
parser.add_argument("--relayout", choices=("off", "on"), required=True)
parser.add_argument(
    "--out", type=Path, required=True, help="New result directory; never overwritten"
)
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show settings without imports or device access",
)
args = parser.parse_args()
OUT = args.out.resolve()

# Set the measured settings before importing Torch or the compiler.
settings = dict(
    PYTHONHASHSEED="0",
    SENCORES="32",
    LX_PLANNING="1",
    LAYOUT_SOLVER="greedy",
    DXP_LX_FRAC_AVAIL="0.2",
    SPYRE_HAZARD_TRACKER="1",
    SPYRE_ATTN_RECORD="1",
    SPYRE_NUM_CPUS="8",
    SPYRE_LX_PLANNER_RELAYOUT=str(int(args.relayout == "on")),
    RANK="0",
    LOCAL_RANK="0",
    WORLD_SIZE="1",
    LOCAL_WORLD_SIZE="1",
    AIU_WORLD_SIZE="1",
    MASTER_ADDR="127.0.0.1",
)
for name in (
    "SPYRE_LX_FUSED_SPLIT_VIEWS",
    "SPYRE_LX_CONSUMER_ANCHORED_ORDERING",
    "SPYRE_LX_RESTICKIFY_RESIDENCY",
    "SPYRE_LX_KV_LAYOUT",
    "SPYRE_BUCKETED_DECODE",
    "SPYRE_ATTN_MAX_CORES",
    "SPYRE_BATCHED_DECODE",
    "SPYRE_INDEXED_SELECTION_CONSUMER_LAYOUT",
    "CO_OPTIMIZING_LX_PLANNING",
):
    os.environ.pop(name, None)
restart_for_hash_seed = os.environ.get("PYTHONHASHSEED") != "0"
os.environ.update(settings)
os.environ.setdefault("MASTER_PORT", "29687")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(OUT / "cache")
if args.dry_run:
    print("DRY RUN ONLY: no files, imports, compilation or device access.")
    print(
        json.dumps(dict(arguments=vars(args), settings=settings), indent=2, default=str)
    )
    raise SystemExit(0)
if restart_for_hash_seed:
    os.execv(sys.executable, [sys.executable, *sys.argv])
if OUT.exists():
    parser.error(f"Result directory already exists: {OUT}")
OUT.mkdir(parents=True)
print(f"Run serially on an idle Spyre device. Results: {OUT}", flush=True)


def metadata():
    import torch
    import torch_spyre
    import spyre_inference
    from torch_spyre._inductor import config

    sources = {}
    for name, module in (
        ("torch-spyre", torch_spyre),
        ("spyre-inference", spyre_inference),
    ):
        root = Path(module.__file__).resolve().parent.parent
        item = {"location": str(root)}
        try:
            item["version"] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            item["version"] = None
        if (root / ".git").exists():
            item["head"] = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip()
            item["tracked_edits"] = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(root),
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                ],
                text=True,
            ).strip()
        sources[name] = item
    if config.lx_planner_relayout != (args.relayout == "on"):
        raise RuntimeError("The imported compiler did not pick up the relayout switch")
    return dict(
        arm=args.relayout,
        sources=sources,
        torch=torch.__version__,
        packages={
            name: importlib.metadata.version(name)
            for name in ("vllm", "transformers", "huggingface-hub")
        },
        settings={
            name: getattr(config, name)
            for name in (
                "sencores",
                "layout_solver",
                "lx_planning",
                "lx_planner_relayout",
                "allow_all_ops_in_lx_planning",
                "dxp_lx_frac_avail",
            )
        },
        environment=settings,
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )


def save(record):
    with (OUT / "result.json").open("x") as file:
        json.dump(record, file, indent=2, default=str)
    summary = {
        key: record[key]
        for key in (
            "arm",
            "phase",
            "status",
            "max_abs_error",
            "device_median_ms",
            "wall_median_us",
            "prefill_median_ms",
            "decode_median_ms",
            "model_init_seconds",
        )
        if key in record
    }
    print("RESULT=" + json.dumps(summary), flush=True)


import math
import statistics
import time
import traceback
from unittest.mock import patch
import torch
import torch_spyre
from torch_spyre.execution import async_compile
from spyre_inference.v1.attention.backends import spyre_attn

record = metadata()
mode = args.phase
qlen, kvlen, pages = (512, 512, 4) if mode == "prefill" else (1, 513, 8)
# The upstream attention test's fixed numerical policy. Keep our stricter
# exploratory check as a separately reported diagnostic, not an adjustable gate.
atol, rtol = (0.3, 0.2) if qlen >= 32 else (0.005, 0.01)
# Match the application's max-batched-token staging extent, larger than the
# selected query. Selecting every row of the gather source is not its contract.
staging = 513
torch.manual_seed(783)
q = torch.randn(staging, 32, 128, dtype=torch.float16)
k = torch.randn(16, 128, 8, 128, dtype=torch.float16)
v = torch.randn_like(k)
indices = torch.zeros(pages, 32, dtype=torch.int32)
indices[: math.ceil(kvlen / 128), 0] = torch.arange(
    math.ceil(kvlen / 128), dtype=torch.int32
)
rows = torch.zeros(max(32, qlen), dtype=torch.int32)
rows[:qlen] = torch.arange(qlen, dtype=torch.int32)
masks = []
for page in range(pages):
    positions = torch.arange(128) + page * 128
    visible = positions[None, :] <= torch.arange(qlen)[:, None] + kvlen - qlen
    masks.append(torch.where(visible, 0.0, torch.finfo(torch.float16).min).half())
# Independent, dense FP32 reference from the same stored half-precision inputs.
keys = k[: math.ceil(kvlen / 128)].reshape(-1, 8, 128)[:kvlen].float()
values = v[: math.ceil(kvlen / 128)].reshape(-1, 8, 128)[:kvlen].float()
queries = q[:qlen].float().reshape(qlen, 8, 4, 128)
scores = torch.einsum("qhgd,thd->hgqt", queries, keys) / math.sqrt(128)
causal = torch.arange(kvlen)[None, :] <= torch.arange(qlen)[:, None] + kvlen - qlen
scores.masked_fill_(~causal, -torch.inf)
reference = torch.einsum("hgqt,thd->qhgd", scores.softmax(-1), values).reshape(
    qlen, 32, 128
)
torch.save(
    dict(query=q, key=k, value=v, rows=rows, pages=indices, reference=reference),
    OUT / "inputs.pt",
)
record.update(
    phase=mode,
    query_shape=list(q.shape),
    actual_query_tokens=qlen,
    kv_shape=list(k.shape),
    kv_length=kvlen,
    page_iterations=pages,
    logical_pages=math.ceil(kvlen / 128),
    kernel="_create_compilable_page_attn",
    tolerance=dict(atol=atol, rtol=rtol),
    tolerance_source="tests/attention/test_spyre_attn.py:537-549"
    if qlen >= 32
    else "strict decode check",
    scope="attention kernel only, not model latency",
)
dirs = set()
original = async_compile.get_output_dir


def capture(*args, **kwargs):
    directory = original(*args, **kwargs)
    dirs.add(str(directory))
    return directory


try:
    torch_spyre._autoload()
    cache_layout = spyre_attn.slot_major_kv_layout(
        k.shape[0] * 128, 8, 128, torch.float16
    )
    args = [
        q.to("spyre"),
        rows.to("spyre"),
        k.to("spyre", device_layout=cache_layout),
        v.to("spyre", device_layout=cache_layout),
        indices.to("spyre"),
    ]
    masks_device = [x.to("spyre") for x in masks]
    out = torch.zeros_like(q).to("spyre")
    fn = spyre_attn._create_compilable_page_attn(
        pages, qlen, 32, 8, 128, store_out=True
    )
    compiled = torch.compile(fn, dynamic=False)

    def run():
        return compiled(*args, masks_device, 1 / math.sqrt(128), out=out)

    with torch.inference_mode(), patch.object(async_compile, "get_output_dir", capture):
        begin = time.perf_counter()
        actual = run().cpu()[:qlen]
        record["compile_and_first_seconds"] = time.perf_counter() - begin
        record["strict_outliers"] = int(
            ((actual.float() - reference).abs() > 0.005 + 0.01 * reference.abs()).sum()
        )
        record["max_abs_error"] = (actual.float() - reference).abs().max().item()
        torch.save(actual, OUT / "output.pt")
        torch.testing.assert_close(actual.float(), reference, atol=atol, rtol=rtol)
        torch.testing.assert_close(actual, out.cpu()[:qlen], atol=0, rtol=0)
        for _ in range(3):
            run()
        torch.spyre.synchronize()
        samples = []
        for _ in range(10):
            begin = time.perf_counter()
            run()
            torch.spyre.synchronize()
            samples.append((time.perf_counter() - begin) * 1e6)
        record.update(
            wall_samples_us=samples, wall_median_us=statistics.median(samples)
        )
        from torch.profiler import profile, ProfilerActivity

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]
        ) as prof:
            for _ in range(5):
                run()
                torch.spyre.synchronize()
        prof.export_chrome_trace(str(OUT / "trace.json"))
        # Profiled device time is distinct from the synchronized wall samples.
        events = json.loads((OUT / "trace.json").read_text())["traceEvents"]
        kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel"]
        if len(kernels) != 5 or len({e["name"] for e in kernels}) != 1:
            raise RuntimeError(
                "Expected five device kernel events; check your profiler setup and trace.json"
            )
        device_ms = [e["dur"] / 1000 for e in kernels]
        if not all(math.isfinite(value) and value > 0 for value in device_ms):
            raise RuntimeError("Invalid device timing in trace.json")
        record.update(
            device_samples_ms=device_ms, device_median_ms=statistics.median(device_ms)
        )
        torch.testing.assert_close(
            out.cpu()[:qlen].float(), reference, atol=atol, rtol=rtol
        )
        record["status"] = "passed"
except Exception:
    record.update(status="failed", traceback=traceback.format_exc())
    print(record["traceback"], flush=True)
finally:
    record["bundle_directories"] = sorted(dirs)
    save(record)
raise SystemExit(0 if record["status"] == "passed" else 1)
