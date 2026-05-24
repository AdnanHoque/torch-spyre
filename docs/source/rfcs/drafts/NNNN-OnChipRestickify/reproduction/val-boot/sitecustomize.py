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

"""Process-local shim: import torch_spyre from the ONCHIP_SRC worktree.

Drops torch_spyre* entries from the editable-install finder so the normal path
importer resolves torch_spyre to the worktree (which carries the copied .so).
Only active when this dir is on PYTHONPATH, so it never affects other processes.
The worktree path comes from $ONCHIP_SRC (default /tmp/tier-up).
"""

import os
import sys

_ONCHIP_SRC = os.environ.get("ONCHIP_SRC", "/tmp/tier-up")

try:
    import __editable___torch_spyre_0_0_1_finder as _ef

    for _k in [
        k
        for k in list(_ef.MAPPING)
        if k == "torch_spyre" or k.startswith("torch_spyre.")
    ]:
        _ef.MAPPING.pop(_k, None)
    for _k in [
        k
        for k in list(_ef.NAMESPACES)
        if k == "torch_spyre" or k.startswith("torch_spyre.")
    ]:
        _ef.NAMESPACES.pop(_k, None)
except Exception as _e:  # pragma: no cover
    print("val-boot: could not patch editable finder:", _e, file=sys.stderr)

sys.path.insert(0, _ONCHIP_SRC)
