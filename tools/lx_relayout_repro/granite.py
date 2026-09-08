"""Measure Granite 3.3 8B through unchanged vLLM/spyre-inference.
One 512-token prompt, two generated tokens: prefill selects the first;
one decode step at KV=513 selects the second.

  python granite.py --relayout off --out granite-off
  python granite.py --relayout on  --out granite-on
Read prefill_median_ms and decode_median_ms; OFF / ON is the speedup.
These are synchronized engine-step wall times, not attention-only device
times or HTTP latency. Initialization and warmup are excluded.
Compare the tokens and logprobs in both result.json files as well as timing.

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


import contextlib
import math
import statistics
import time
import traceback
from unittest.mock import patch
import torch
from vllm import LLM, SamplingParams
from vllm.config import AttentionConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from torch_spyre.execution import async_compile

record = metadata()
record.update(
    model="ibm-granite/granite-3.3-8b-instruct",
    revision="51dd4bc2ade4059a6bd87649d68aa11e4fb2529b",
    batch=1,
    prompt_tokens=512,
    generated_tokens=2,
    decode_kv_length=513,
    max_model_len=1024,
    max_num_batched_tokens=512,
    scope="vLLM engine-step wall time; not HTTP request latency or serving throughput",
)
dirs = set()
original = async_compile.get_output_dir


def capture(*args, **kwargs):
    directory = original(*args, **kwargs)
    dirs.add(str(directory))
    return directory


try:
    with patch.object(async_compile, "get_output_dir", capture):
        begin = time.perf_counter()
        llm = LLM(
            model=record["model"],
            revision=record["revision"],
            dtype="float16",
            max_model_len=1024,
            max_num_seqs=1,
            max_num_batched_tokens=512,
            num_gpu_blocks_override=64,
            block_size=128,
            enable_prefix_caching=False,
            compilation_config={"compile_sizes": [1, 512]},
            attention_config=AttentionConfig(backend=AttentionBackendEnum.CUSTOM),
            distributed_executor_backend="external_launcher",
        )
        record["model_init_seconds"] = time.perf_counter() - begin
        tokens = llm.get_tokenizer().encode(
            "Explain compiler optimization number 0 in simple terms. ",
            add_special_tokens=False,
        )
        prompt = {
            "prompt_token_ids": (tokens * ((512 + len(tokens) - 1) // len(tokens)))[
                :512
            ]
        }
        (OUT / "prompt.json").write_text(json.dumps(prompt))
        params = SamplingParams(
            max_tokens=2, min_tokens=2, temperature=0.0, ignore_eos=True, logprobs=1
        )
        engine = llm.llm_engine
        original_step = engine.step

        def request(profile_name=None):
            steps = []

            def observed(*args, **kwargs):
                torch.spyre.synchronize()
                start = time.perf_counter()
                outputs = original_step(*args, **kwargs)
                torch.spyre.synchronize()
                steps.append(
                    dict(
                        seconds=time.perf_counter() - start,
                        tokens=[len(x.token_ids) for o in outputs for x in o.outputs],
                    )
                )
                return outputs

            if profile_name:
                from torch.profiler import profile, ProfilerActivity

                cm = profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]
                )
            else:
                cm = contextlib.nullcontext()
            with cm as prof, patch.object(engine, "step", observed):
                outputs = llm.generate([prompt], params, use_tqdm=False)
            if profile_name:
                prof.export_chrome_trace(str(OUT / profile_name))
            final = outputs[0].outputs[0]
            assert (
                len(outputs) == 1 and len(final.token_ids) == 2 and len(steps) == 2
            ), steps
            lp = [
                entry[token].logprob
                for token, entry in zip(final.token_ids, final.logprobs, strict=True)
            ]
            assert all(math.isfinite(v) for v in lp)
            return dict(steps=steps, tokens=list(final.token_ids), logprobs=lp)

        record["warmup"] = request()
        print("WARMUP=" + json.dumps(record["warmup"]), flush=True)
        request()
        record["samples"] = []
        for i in range(5):
            sample = request()
            assert sample["tokens"] == record["warmup"]["tokens"]
            record["samples"].append(sample)
            print("SAMPLE=" + json.dumps(sample), flush=True)
        record["profiled"] = [request(f"trace-{i}.json") for i in range(3)]
        record["prefill_median_ms"] = (
            statistics.median(x["steps"][0]["seconds"] for x in record["samples"])
            * 1000
        )
        record["decode_median_ms"] = (
            statistics.median(x["steps"][1]["seconds"] for x in record["samples"])
            * 1000
        )
        record["status"] = "passed"
except Exception:
    record.update(status="failed", traceback=traceback.format_exc())
    print(record["traceback"], flush=True)
finally:
    record["bundle_directories"] = sorted(dirs)
    save(record)
raise SystemExit(0 if record["status"] == "passed" else 1)
