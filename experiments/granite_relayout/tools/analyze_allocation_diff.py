import json
import sys
from pathlib import Path


def load_main(path: str) -> dict:
    records = [json.loads(line) for line in Path(path).read_text().splitlines()]
    return max(records, key=lambda record: len(record["buffers"]))


left = load_main(sys.argv[1])
right = load_main(sys.argv[2])
left_by_name = {item["name"]: item for item in left["buffers"]}
right_by_name = {item["name"]: item for item in right["buffers"]}

print("LEFT_RELAYOUT_SOURCES", left["relayout_sources"])
print("RIGHT_RELAYOUT_SOURCES", right["relayout_sources"])
print("ONLY_LEFT")
for name in sorted(left_by_name.keys() - right_by_name.keys()):
    print(json.dumps(left_by_name[name], sort_keys=True))
print("ONLY_RIGHT")
for name in sorted(right_by_name.keys() - left_by_name.keys()):
    print(json.dumps(right_by_name[name], sort_keys=True))

fields = (
    "allocation_index",
    "size",
    "uses",
    "first_use_is_read",
    "address",
    "in_place_parents",
    "residency_reason",
    "reject_reason",
    "is_graph_input",
    "is_graph_output",
    "is_relayout_destination",
)
print("CHANGED_COMMON")
for name in sorted(left_by_name.keys() & right_by_name.keys()):
    a = left_by_name[name]
    b = right_by_name[name]
    changed = {field: [a[field], b[field]] for field in fields if a[field] != b[field]}
    if changed:
        print(name, json.dumps(changed, sort_keys=True))


def overlap_audit(label: str, record: dict) -> None:
    addressed = [b for b in record["buffers"] if b["address"] is not None]
    conflicts = []
    for i, a in enumerate(addressed):
        for b in addressed[i + 1 :]:
            address_overlap = (
                a["address"] < b["address"] + b["size"]
                and b["address"] < a["address"] + a["size"]
            )
            time_overlap = (
                min(a["uses"]) < max(b["uses"]) + 1
                and min(b["uses"]) < max(a["uses"]) + 1
            )
            if not address_overlap or not time_overlap:
                continue
            in_place = (
                a["name"] in b["in_place_parents"]
                or b["name"] in a["in_place_parents"]
                or bool(set(a["in_place_parents"]) & set(b["in_place_parents"]))
            )
            conflicts.append((a["name"], b["name"], in_place, a["address"], b["address"], a["size"], b["size"], a["uses"], b["uses"]))
    print(label + "_OVERLAPS")
    for conflict in conflicts:
        print(json.dumps(conflict))


overlap_audit("LEFT", left)
overlap_audit("RIGHT", right)


def live_at(label: str, record: dict, tick: int) -> None:
    print(f"{label}_LIVE_AT_{tick}")
    live = []
    for b in record["buffers"]:
        if b["address"] is None:
            continue
        if min(b["uses"]) <= tick <= max(b["uses"]):
            live.append(b)
    for b in sorted(live, key=lambda item: (item["address"], item["name"])):
        print(
            b["name"],
            f"addr={b['address']}",
            f"end={b['address'] + b['size']}",
            f"size={b['size']}",
            f"uses={b['uses']}",
        )


for tick in range(17, 24):
    live_at("LEFT", left, tick)
live_at("RIGHT", right, 23)
