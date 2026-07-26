#!/usr/bin/env bash
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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="${GRANITE_PARITY_ROOT:-/home/adnan/codex-isolated/device_parity_pr2939_20260725}"

if [[ "${1:-}" != "--apply" ]]; then
  echo "dry run: overlay sources are below ${BUNDLE_ROOT}/overlays"
  echo "target project root: ${PROJECT_ROOT}"
  echo "rerun with --apply only on the recorded base commits"
  exit 0
fi

if [[ "$(git -C "${PROJECT_ROOT}/deeptools" rev-parse HEAD)" != "406142afb9f080b9271e7c565a757ab8d8b5ed8f" ]]; then
  echo "DeepTools base mismatch; refusing to overwrite files" >&2
  exit 1
fi
if [[ "$(git -C "${PROJECT_ROOT}/foundation-model-stack" rev-parse HEAD)" != "61bc991b175103e80cb8202b24a66ba7dbe79d1b" ]]; then
  echo "FMS base mismatch; refusing to overwrite files" >&2
  exit 1
fi

cp -R "${BUNDLE_ROOT}/overlays/deeptools/." "${PROJECT_ROOT}/deeptools/"
cp -R "${BUNDLE_ROOT}/overlays/foundation_model_stack/." \
  "${PROJECT_ROOT}/foundation-model-stack/"
echo "external overlays applied; rebuild DeepTools before device execution"

