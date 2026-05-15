# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end Granite-3.3-8B profile with and without LX_PLANNING=1.

Adapted from docs/source/user_guide/profiling/end_to_end_example.md, with
prefill kwarg setup (selected_freqs, mask, position_ids) replicated from
fms.utils.generation.generate(). The doc's torch.randint example does not
include those kwargs and so trips on the FMS attention forward.

Loads granite-3.3-8b-instruct via FMS, compiles via model.compile(),
profiles a warmup + N steady-state forward passes with torch.profiler +
kineto-spyre PrivateUse1 activities, and aggregates per-op device time.

The ring-aware-restickify project wants the device-time share spent in
ReStickifyOpHBM. This script reports it directly from the trace.
"""

import argparse
import json
import math
import os
import sys
import time
from statistics import mean, median

# torch_spyre repo on path so it imports from source.
sys.path.insert(0, "/home/adnan/dt-inductor/torch-spyre")

import torch
import torch_spyre  # noqa: F401
from torch.profiler import ProfilerActivity, profile
from fms.models import get_model
from fms.utils.generation import pad_input_ids
from transformers import AutoTokenizer

from torch_spyre._inductor import config as ts_config

DEVICE = torch.device("spyre")
DTYPE = torch.float16
MODEL_PATH = "/tmp/models/granite-3.3-8b-instruct"
PROMPT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\nProvide a list of instructions for preparing chicken soup.\n\n"
    "### Response:"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lx-planning", action="store_true")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--n-runs", type=int, default=3)
    p.add_argument("--log-dir", default="/tmp/granite_profile")
    return p.parse_args()


def load_and_compile():
    print("=== loading granite-3.3-8b via fms ===", flush=True)
    t0 = time.time()
    model = get_model(
        architecture="hf_pretrained",
        model_path=MODEL_PATH,
        device_type="spyre",
        data_type=DTYPE,
        unfuse_weights=True,
    ).eval().to(DEVICE)
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("=== compiling ===", flush=True)
    model.compile()
    return model, tokenizer


def build_prefill_inputs(model, tokenizer, seq_len):
    """Tokenize the prompt, pad to seq_len, build mask/position_ids and the
    spyre-side selected_freqs the same way fms.utils.generation.generate does
    for the prefill iteration."""
    encoded = tokenizer.encode(PROMPT, return_tensors="pt").squeeze(0)
    input_ids, padding_kwargs = pad_input_ids(
        [encoded],
        min_pad_length=max(seq_len, math.ceil(encoded.size(0) / 64) * 64),
    )

    max_seq_len = model.config.max_expected_seq_len
    alpha = model.base_model.rot_emb.compute_freqs_cis(DEVICE, max_seq_len)
    selected_freqs = (
        model.base_model.rot_emb.cached_freqs[0][alpha][padding_kwargs["position_ids"]]
        .contiguous()
        .to(DEVICE)
    )
    mask = padding_kwargs["mask"].to(dtype=DTYPE).to(DEVICE)

    kwargs = {
        "mask": mask,
        "position_ids": padding_kwargs["position_ids"].to(DEVICE),
        "selected_freqs": selected_freqs,
    }
    return input_ids.to(DEVICE), kwargs


def warm_then_profile(model, input_ids, kwargs, n_runs, log_dir, label):
    print("=== warmup forward (compile + codegen happens here) ===", flush=True)
    t0 = time.time()
    with torch.no_grad():
        model(input_ids, **kwargs)
    print(f"  warmup wall-clock: {time.time()-t0:.1f}s", flush=True)

    print(f"=== profiling {n_runs} steady-state iterations ===", flush=True)
    wall = []
    out_dir = os.path.join(log_dir, label)
    os.makedirs(out_dir, exist_ok=True)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
        record_shapes=True,
        profile_memory=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(out_dir),
    ) as prof:
        for i in range(n_runs):
            t = time.perf_counter()
            with torch.no_grad():
                model(input_ids, **kwargs)
            wall.append((time.perf_counter() - t) * 1000)
            prof.step()
    return prof, wall


def summarize(prof, wall, label, out_path):
    print(f"\n############ {label} ############", flush=True)
    print(
        f"  wall-clock ms: mean={mean(wall):.3f}  median={median(wall):.3f}  "
        f"min={min(wall):.3f}  max={max(wall):.3f}", flush=True,
    )
    print("\n  top 25 ops by device_time_total:", flush=True)
    print(
        prof.key_averages().table(sort_by="device_time_total", row_limit=25),
        flush=True,
    )

    by_name = {}
    for e in prof.key_averages():
        v = (
            getattr(e, "device_time_total", 0)
            or getattr(e, "self_device_time_total", 0)
            or 0
        )
        by_name[e.key] = by_name.get(e.key, 0) + v
    total = sum(by_name.values())

    rs_total = 0
    print("\n  per-op summary (device_time us):", flush=True)
    for n in sorted(by_name, key=lambda k: -by_name[k])[:40]:
        is_rs = "restickify" in n.lower() or "stickify" in n.lower()
        if is_rs:
            rs_total += by_name[n]
        if total:
            print(
                f"    {n[:60]:<60} {by_name[n]:12.1f}  "
                f"({100*by_name[n]/total:5.2f}%){' [RS]' if is_rs else ''}",
                flush=True,
            )

    if total:
        print(
            f"\n  ** restickify share of device time: "
            f"{100*rs_total/total:.2f}% **", flush=True,
        )

    with open(out_path, "w") as f:
        json.dump(
            {
                "label": label,
                "wall_clock_ms": wall,
                "restickify_us": rs_total,
                "total_us": total,
                "restickify_share": rs_total / total if total else 0,
                "by_op": by_name,
            },
            f,
            indent=2,
        )
    print(f"\nWrote summary to {out_path}", flush=True)


def main():
    args = parse_args()
    label = "lx_planning" if args.lx_planning else "baseline"
    print(
        f"=== run config: {label}  seq_len={args.seq_len}  "
        f"n_runs={args.n_runs} ===", flush=True,
    )
    with ts_config.patch(
        lx_planning=args.lx_planning,
        allow_all_ops_in_lx_planning=args.lx_planning,
        sencores=int(os.environ.get("SENCORES", "32")),
    ):
        torch.compiler.reset()
        model, tokenizer = load_and_compile()
        input_ids, kwargs = build_prefill_inputs(model, tokenizer, args.seq_len)
        prof, wall = warm_then_profile(
            model, input_ids, kwargs, args.n_runs, args.log_dir, label
        )

    os.makedirs(args.log_dir, exist_ok=True)
    summarize(
        prof, wall, label, os.path.join(args.log_dir, f"{label}_summary.json")
    )


if __name__ == "__main__":
    main()
