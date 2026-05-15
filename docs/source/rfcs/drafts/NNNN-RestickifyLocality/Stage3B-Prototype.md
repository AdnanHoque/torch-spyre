# Stage 3B Prototype

This note packages the Stage 3B experiment as a small, demoable prototype for
upstream discussion. It is intentionally narrow and default-off.

## Claim

For eligible in-graph producer-to-restickify edges, the compiler can steer
restickify work distribution so the same physical cores keep ownership of the
same logical tensor regions. This can reduce ring byte-hops without changing:

- tensor semantics
- restickify placement
- restickify count
- bytes moved

This prototype does not change `optimize_restickify_locations`. The placement
cost model remains element-count based. Stage 3B acts later, during
`work_distribution`, and Stage 2 applies the compatible physical core mapping
override after work distribution.

## Flags

All prototype behavior is disabled by default:

```sh
SPYRE_RESTICKIFY_RING_TELEMETRY=1
SPYRE_ALIGN_RESTICKIFY_WORK_DISTRIBUTION=1
SPYRE_ALIGN_RESTICKIFY_CORE_MAPPING=1
```

`SPYRE_RESTICKIFY_RING_TELEMETRY=1` only measures. The two alignment flags enable
the Stage 3B prototype path.

## Eligibility

A restickify is optimized only when all of these are true:

- the source is classified as `in_graph_computed`
- the restickify source has exactly one producer
- stride/symbol correspondence between producer output and restickify output is
  unambiguous
- the producer has one dominant split dimension
- the restickify output can legally split on the corresponding dimension

If any condition is not met, the prototype preserves existing behavior.

## Demo Command

The high-signal demo case is:

```python
(a + b.t() + c.t()) @ d
```

Run baseline and Stage 3B in separate Python processes:

```sh
export HOME=/home/adnan-cdx
export DTI_PROJECT_ROOT=$HOME/dt-inductor
source $DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh
export SENTIENT_BASE_INSTALL_DIR=/opt/ibm/spyre
export DEEPTOOLS_INSTALL_DIR=/opt/ibm/spyre/deeptools
export DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share
export PATH=/opt/ibm/spyre/deeptools/bin:/opt/ibm/spyre/runtime/bin:$PATH
export LD_LIBRARY_PATH=/opt/ibm/spyre/deeptools/lib:/opt/ibm/spyre/runtime/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/tmp/torch-spyre-stage2:${PYTHONPATH:-}
export SENCORES=32
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export LX_PLANNING=0
cd /tmp/torch-spyre-stage2

rm -rf /tmp/restickify-stage3b-prototype
mkdir -p /tmp/restickify-stage3b-prototype/base
mkdir -p /tmp/restickify-stage3b-prototype/stage3b

SPYRE_ALIGN_RESTICKIFY_CORE_MAPPING=0 \
SPYRE_ALIGN_RESTICKIFY_WORK_DISTRIBUTION=0 \
python tools/restickify_scenario_probe.py \
  --case adds_then_matmul \
  --size 2048 \
  --ring-telemetry \
  --skip-correctness \
  --output-dir /tmp/restickify-stage3b-prototype/base

SPYRE_ALIGN_RESTICKIFY_CORE_MAPPING=1 \
SPYRE_ALIGN_RESTICKIFY_WORK_DISTRIBUTION=1 \
python tools/restickify_scenario_probe.py \
  --case adds_then_matmul \
  --size 2048 \
  --ring-telemetry \
  --skip-correctness \
  --output-dir /tmp/restickify-stage3b-prototype/stage3b
```

## Expected Telemetry

The expected summary for `size=2048` is:

| Mode | Restickifies | Bytes moved | Source kinds | Exact rows | Byte-hops |
|---|---:|---:|---|---:|---:|
| Baseline | 2 | 16,777,216 | `graph_input_or_weight:1`, `in_graph_computed:1` | 1 | 67,108,864 |
| Stage 3B | 2 | 16,777,216 | `graph_input_or_weight:1`, `in_graph_computed:1` | 1 | 0 |

The exact in-graph row changes from:

| Mode | Producer split | Restickify split | Byte-hops |
|---|---|---|---:|
| Baseline | `d1:32` | `d0:32` | 67,108,864 |
| Stage 3B | `d1:32` | `d1:32` | 0 |

This is the prototype's core result: restickify count and bytes moved remain
unchanged, but logical ownership is preserved across physical cores.

## Validation Check

The demo can be checked with:

```sh
python - <<'PY'
import json
from pathlib import Path

root = Path("/tmp/restickify-stage3b-prototype")

def load(mode):
    path = root / mode / "restickify_scenarios.jsonl"
    return json.loads(path.read_text().splitlines()[0])

base = load("base")
stage3b = load("stage3b")

print("base", base["restickify_count"], base["total_bytes"], base["ring_total_byte_hops"])
print("stage3b", stage3b["restickify_count"], stage3b["total_bytes"], stage3b["ring_total_byte_hops"])

assert base["status"] == stage3b["status"] == "ok"
assert base["restickify_count"] == stage3b["restickify_count"] == 2
assert base["total_bytes"] == stage3b["total_bytes"] == 16_777_216
assert base["ring_source_kinds"] == stage3b["ring_source_kinds"]
assert base["ring_source_kinds"] == {"graph_input_or_weight": 1, "in_graph_computed": 1}
assert base["ring_total_byte_hops"] == 67_108_864
assert stage3b["ring_total_byte_hops"] == 0

base_exact = [e for e in base["ring_entries"] if e["source_kind"] == "in_graph_computed"]
stage3b_exact = [e for e in stage3b["ring_entries"] if e["source_kind"] == "in_graph_computed"]
assert len(base_exact) == len(stage3b_exact) == 1
assert base_exact[0]["byte_hops"] == 67_108_864
assert stage3b_exact[0]["byte_hops"] == 0
assert base_exact[0]["producer_splits"] == {"d1": 32}
assert base_exact[0]["restickify_splits"] == {"d0": 32}
assert stage3b_exact[0]["restickify_splits"] == {"d1": 32}

print("prototype validation ok")
PY
```

## Out Of Scope

Graph-input, weight, constant, extern, mutation-target, and persistent-state
restickifies are attribution-only in this prototype. They can still move bytes,
but Stage 3B does not know a prior in-graph physical owner to preserve. Those
cases point to a different optimization family: input layout selection, weight
prepacking, and persistent-state layout management.

## Suggested Prototype Tests

```sh
python3 -m py_compile \
  torch_spyre/_inductor/restickify_ring.py \
  torch_spyre/_inductor/restickify_telemetry.py \
  tools/restickify_scenario_probe.py

python -m pytest tests/inductor/test_restickify_mapping_alignment.py -q
python -m pytest tests/inductor/test_restickify.py -q
```
