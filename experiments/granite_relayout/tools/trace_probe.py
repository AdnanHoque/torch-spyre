#!/usr/bin/env python3
import json
import sys


with open(sys.argv[1], encoding="utf-8") as handle:
    trace = json.load(handle)

kernels = [
    event
    for event in trace["traceEvents"]
    if event.get("cat") == "kernel" and "dur" in event
]

phases = []
start = 0
for index, event in enumerate(kernels):
    if "sdsc_fused_div_0_" in event["name"]:
        phases.append(kernels[start : index + 1])
        start = index + 1

print(f"kernels={len(kernels)} phases={len(phases)} trailing={len(kernels) - start}")
for phase_index, phase in enumerate(phases):
    total_ms = sum(event["dur"] for event in phase) / 1000.0
    print(f"PHASE {phase_index} kernels={len(phase)} total_ms={total_ms:.6f}")
    if "--verbose" in sys.argv[2:]:
        for kernel_index, event in enumerate(phase):
            print(
                f"{kernel_index:03d} {event['dur'] / 1000.0:.6f} "
                f"{event['name']}"
            )
