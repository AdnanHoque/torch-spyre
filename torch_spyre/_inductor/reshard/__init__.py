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

"""On-chip core-to-core reduction reshard substrate (torch-free authoring).

The 2-D ``mul -> down_proj`` reduction-input edge that flash-ws fail-closes on:
a ``{mb:4, out:8}`` co-split producer feeding a ``K=12800`` reduction consumer,
moved LX -> RIU ring -> LX instead of round-tripping HBM. ``pieces`` builds the
native per-core tiles; ``substrate`` synthesizes the ``STCDPOpLx`` program and
the standalone pure-data-op SDSC splice.
"""

from .pieces import (
    Band,
    Piece,
    build_consumer_pieces,
    build_producer_pieces,
    build_swiglu_edge,
    build_swiglu_perband_edges,
    pieces_to_pieceinfo,
    swiglu_consumer_owner,
    swiglu_producer_owner,
    swiglu_reshard_sources,
)
from .substrate import (
    DATAOP_LX_SIZE,
    LX_CAPACITY_BYTES,
    LxFlip,
    allocate_lx_bases,
    apply_lx_flip,
    build_asymmetric_reshard_bridge,
    build_perband_reshard_bridge,
    build_standalone_dataop_sdsc,
    splice_reshard,
    splice_reshard_standalone,
)

__all__ = [
    "Band",
    "Piece",
    "build_consumer_pieces",
    "build_producer_pieces",
    "build_swiglu_edge",
    "build_swiglu_perband_edges",
    "pieces_to_pieceinfo",
    "swiglu_consumer_owner",
    "swiglu_producer_owner",
    "swiglu_reshard_sources",
    "DATAOP_LX_SIZE",
    "LX_CAPACITY_BYTES",
    "LxFlip",
    "allocate_lx_bases",
    "apply_lx_flip",
    "build_asymmetric_reshard_bridge",
    "build_perband_reshard_bridge",
    "build_standalone_dataop_sdsc",
    "splice_reshard",
    "splice_reshard_standalone",
]
