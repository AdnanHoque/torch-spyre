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

"""Score the LX relayout law against its measured records, offline.

``relayout_records.json`` (beside this file) holds single LX relayouts timed on
the card, each as its own program, in the record format of
``parse_sweep_logs.py``: ``feats`` holds the features ``relayout_ns`` reads and
``kernel_us`` the measured time. Each record also says what kind of move it is:

* ``permutation``: a complete partition of the cores onto a complete partition
  of the same cores. The law was fitted on these.
* ``gather``: a move into a matmul's replicated operand view.

The script prices every record with the shipped law (``relayout_ns`` and
``CostParams``) and reports the error by kind and governing split. It then
applies the rule that decides which splits the solver may be offered a
permutation at: a split qualifies when its records come from at least two
geometries and the law's mean error there is within 25 % and its RMS error
within 30 %. ``--refit`` also refits the law's three constants (least squares
on relative error, as the original fit) and reports leave-one-geometry-out
error. The basis of the refit is evaluated through ``relayout_ns`` itself, so
the script cannot drift from the law's form.

No hardware is needed. The script imports torch-spyre's cost model, as the
compiler does.

    python3 tools/cost_model/fit_relayout_law.py
    python3 tools/cost_model/fit_relayout_law.py --refit
"""

import argparse
import collections
import dataclasses
import json
import math
import os

import numpy as np

from torch_spyre._inductor.cost_model import CostParams, op_from_dict, relayout_ns

_HERE = os.path.dirname(os.path.abspath(__file__))
RECORDS = os.path.join(_HERE, "relayout_records.json")

#: The accuracy rule for a governing split: records from this many geometries,
#: and the law's mean and RMS relative error within these bounds.
MIN_GEOMETRIES = 2
MAX_ABS_MEAN = 0.25
MAX_RMS = 0.30


def load(path):
    with open(path, encoding="utf-8") as f:
        return [r for r in json.load(f)["records"] if not r.get("failed")]


def predicted_us(rec, params):
    return relayout_ns(op_from_dict(rec["feats"][0]), params) / 1000.0


def error(rec, params):
    """Relative error of the law on one record: (predicted - measured) / measured."""
    return predicted_us(rec, params) / rec["kernel_us"] - 1.0


def summary(errors):
    mean = sum(errors) / len(errors)
    rms = math.sqrt(sum(e * e for e in errors) / len(errors))
    return mean, rms


def qualifies(records, params):
    """Whether these records (one kind, one split) meet the accuracy rule."""
    mean, rms = summary([error(r, params) for r in records])
    geometries = {r["variant"] for r in records}
    return (
        len(geometries) >= MIN_GEOMETRIES
        and abs(mean) <= MAX_ABS_MEAN
        and rms <= MAX_RMS
    )


def by_kind_and_split(records):
    groups = collections.defaultdict(list)
    for r in records:
        groups[(r["kind"], r["feats"][0]["relayout_split"])].append(r)
    return groups


def basis(rec):
    """The law's three terms for one record, evaluated through ``relayout_ns``.

    ``relayout_ns`` is linear in (a, b, 1/K): runs * (a + b * log2(split)) +
    span / K. Unit values of one constant, with the others zero, give each term.
    """
    unit = CostParams()

    def term(a, b, inv_k):
        params = dataclasses.replace(
            unit,
            relayout_run_a_ns=a,
            relayout_run_b_ns=b,
            relayout_span_gbps=1.0 / inv_k if inv_k else math.inf,
        )
        return relayout_ns(op_from_dict(rec["feats"][0]), params)

    return [term(1.0, 0.0, 0.0), term(0.0, 1.0, 0.0), term(0.0, 0.0, 1.0)]


def refit(records):
    """(a, b, K) minimizing the relative error over ``records``."""
    x = np.array([basis(r) for r in records]) / np.array(
        [[r["kernel_us"] * 1000.0] for r in records]
    )
    (a, b, inv_k), *_ = np.linalg.lstsq(x, np.ones(len(records)), rcond=None)
    return dataclasses.replace(
        CostParams(),
        relayout_run_a_ns=float(a),
        relayout_run_b_ns=float(b),
        relayout_span_gbps=float(1.0 / inv_k) if inv_k > 0 else math.inf,
    )


def report_refit(name, records):
    params = refit(records)
    mean, rms = summary([error(r, params) for r in records])
    held_out = []
    for geometry in sorted({r["variant"] for r in records}):
        train = [r for r in records if r["variant"] != geometry]
        held = refit(train)
        held_out += [error(r, held) for r in records if r["variant"] == geometry]
    _, lovo = summary(held_out)
    print(
        f"refit on {name} ({len(records)} records): a = {params.relayout_run_a_ns:.3f} "
        f"ns/run, b = {params.relayout_run_b_ns:.3f} ns/run per log2(split), "
        f"K = {params.relayout_span_gbps:.1f} GB/s per core; mean {mean:+.1%}, "
        f"RMS {rms:.1%}, leave-one-geometry-out RMS {lovo:.1%}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default=RECORDS)
    ap.add_argument("--refit", action="store_true", help="also refit (a, b, K)")
    args = ap.parse_args()

    records = load(args.records)
    params = CostParams()
    builds = sorted({r.get("model_sha") or "?" for r in records})
    print(
        f"{len(records)} records (build {', '.join(builds)}); law: a = "
        f"{params.relayout_run_a_ns} ns/run, b = {params.relayout_run_b_ns} ns/run "
        f"per log2(split), K = {params.relayout_span_gbps} GB/s per core"
    )
    print(
        f"\n{'kind':>12} {'split':>5} {'n':>3} {'geom':>4} {'mean':>7} {'RMS':>6}  rule"
    )
    groups = by_kind_and_split(records)
    for kind, split in sorted(groups):
        rs = groups[(kind, split)]
        mean, rms = summary([error(r, params) for r in rs])
        geometries = len({r["variant"] for r in rs})
        verdict = "meets" if qualifies(rs, params) else "fails"
        if kind != "permutation":
            ratios = [r["kernel_us"] / predicted_us(r, params) for r in rs]
            verdict = f"measured {min(ratios):.1f}-{max(ratios):.1f}x the law"
        print(
            f"{kind:>12} {split:>5} {len(rs):>3} {geometries:>4} {mean:>+7.1%} "
            f"{rms:>6.1%}  {verdict}"
        )
    perms = [r for r in records if r["kind"] == "permutation"]
    mean, rms = summary([error(r, params) for r in perms])
    print(f"\nall permutations: mean {mean:+.1%}, RMS {rms:.1%}")
    meets = sorted(
        split
        for (kind, split), rs in groups.items()
        if kind == "permutation" and qualifies(rs, params)
    )
    print(
        f"permutation splits that meet the rule: {meets}; offered: "
        f"[{params.relayout_split_min}, {params.relayout_split_max}] "
        "(CostParams.relayout_split_min/max). The floor of 2 is the law's original "
        "calibration range (splits 2-8)."
    )
    if args.refit:
        print()
        report_refit("permutations", perms)
        report_refit("every record", records)


if __name__ == "__main__":
    main()
