import json
import os
import sys
from pathlib import Path

from torch_spyre._inductor.scratchpad.allocator import _lx_planning_size
from torch_spyre._inductor.scratchpad.firstfit_bestfit_solver import (
    BestFitLayoutSolver,
    FirstFitLayoutSolver,
)
from torch_spyre._inductor.scratchpad.plan_solver import LifetimeBoundBuffer


records = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
record = max(records, key=lambda item: len(item["buffers"]))


def fresh_buffers():
    disabled = {
        name.strip()
        for name in os.environ.get("OFFLINE_DISABLE_LX_BUFFERS", "").split(",")
        if name.strip()
    }
    buffers = [
        LifetimeBoundBuffer(
            name=item["name"],
            size=item["size"],
            uses=item["uses"],
            first_use_is_read=item["first_use_is_read"],
            address=None,
            in_place_parents=item["in_place_parents"],
            residency_reason=(
                "disabled by offline probe"
                if item["name"] in disabled
                else item["residency_reason"]
            ),
        )
        for item in record["buffers"]
    ]
    alias_source = os.environ.get("OFFLINE_QK_OUTPUT_ALIAS", "")
    if alias_source:
        by_name = {item.name: item for item in buffers}
        destination = by_name.get(
            f"__spyre_lx_relayout_destination__:{alias_source}"
        )
        output = by_name.get("buf20")
        if destination is not None and output is not None:
            output.in_place_parents.append(destination.name)
    return buffers


targets = {
    "buf14",
    "__spyre_lx_relayout_destination__:buf14",
    "buf66",
    "__spyre_lx_relayout_destination__:buf66",
}
for solver_type in (FirstFitLayoutSolver, BestFitLayoutSolver):
    allocation = solver_type(_lx_planning_size()).plan_layout(fresh_buffers())
    print(solver_type.__name__, "limit", _lx_planning_size())
    for item in allocation:
        if item.name in targets:
            print(item.name, item.address, item.size, item.uses)
    for tick in (23, 24):
        print("LIVE", tick)
        for item in sorted(
            (
                item
                for item in allocation
                if item.address is not None
                and item.start_time <= tick < item.end_time
            ),
            key=lambda item: (item.address, item.name),
        ):
            print(
                item.name,
                item.address,
                item.address + item.size,
                item.size,
                item.uses,
            )
