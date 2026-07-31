#!/usr/bin/env python3
# Copyright 2026 IBM Corporation
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

"""Install the private FP8 bridge, then execute the Antoni FMS runner."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

from fms_mo_spyre_fp8_bridge import install_fms_mo_spyre_fp8_bridge


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate and install the bridge without loading or running Granite",
    )
    args, runner_args = parser.parse_known_args()
    if runner_args[:1] == ["--"]:
        runner_args = runner_args[1:]
    return args, runner_args


def main() -> None:
    args, runner_args = parse_args()
    versions = install_fms_mo_spyre_fp8_bridge()
    print("FMS-MO Spyre FP8 bridge installed: " + json.dumps(versions, sort_keys=True))

    if args.check_only:
        return
    if not args.runner.is_file():
        raise FileNotFoundError(f"Antoni runner does not exist: {args.runner}")

    sys.argv = [str(args.runner), *runner_args]
    runpy.run_path(str(args.runner), run_name="__main__")


if __name__ == "__main__":
    main()
