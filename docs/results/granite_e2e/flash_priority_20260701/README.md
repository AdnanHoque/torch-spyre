# Flash Priority Checkpoint - 2026-07-01

Priority shifted to the latest `test_flash.py` attention spill.

Key finding: PR1 scatter removes 0 flash HBM spills because the remaining repeated edge is not scatter. It is an activation layout/restickify plus grouped all-gather/broadcast into the following `batchmatmul` KERNEL operand.

Representative edge:

```text
mul -> ReStickifyOpHBM -> batchmatmul
sdsc_1.json -> sdsc_2.json -> sdsc_3.json
```

Producer split:

```text
mul: {mb:4, x:8, out:1}
output: LX OUTPUT [out,x,mb], stick out
```

Consumer split:

```text
batchmatmul: {x:4, mb:8, out:1, in:1}
KERNEL operand expects renamed/layout-transformed view
```

Why scatter is insufficient:

- each value does not have one destination owner;
- each batch-local group needs producer chunks split over BMM `out` to be visible to consumer cores split over BMM `mb`;
- the handoff also changes stick/layout from pointwise output form into BMM KERNEL form.

Artifacts:

- `layout_restickify_gap/flash_layout_restickify_gap.md`
- `layout_restickify_gap/representative_edge.json`
- `layout_restickify_gap/sdsc_triplet_snippets.json`

## DLDSC Path Assessment

Artifact directory:

```text
dldsc_layout_restickify/
```

CDX conclusion: Deeptools master / PR4408 cannot express this flash edge from plain DLDSC tensor-vs-compute distribution metadata alone. The right long-term direction is still metadata-driven backend-generated movement, but the serialized contract has to include the layout/restickify edge explicitly:

- producer and consumer layouts;
- source and destination stick dimensions;
- producer and consumer core mappings;
- dimension rename between pointwise output and BMM KERNEL operand;
- replication/all-gather policy;
- fail-closed behavior when backend cannot realize the handoff.

Key files:

- `dldsc_layout_restickify/analysis.md`
- `dldsc_layout_restickify/deeptools.diff`
- `dldsc_layout_restickify/focused_refs.txt`
- `dldsc_layout_restickify/deeptools_status_short.txt`

## Explicit/Grouped-Remap Path Assessment

Artifact directory:

```text
explicit_layout_restickify/
```

CLC conclusion: grouped explicit physical movement can express the communication shape in principle, but the current carrier schema is not sufficient. It needs source/destination layout views, dimension rename, replication groups, and consumer operand lifetime. A proposed schema was written, but no semantic mutation of the latest flash SDSCs was attempted because that would invent backend semantics before the contract is agreed.

Tiny executable carrier result:

- normal DXP replay of the focused bundle: `rc=0`;
- senulator replay: `rc=134`;
- first senulator blocker: `ProgCorrectionScatter0 op=ScatterOpHBM ... pieceSize=1 layoutSize=0`.

Key files:

- `explicit_layout_restickify/analysis.md`
- `explicit_layout_restickify/results.txt`
- `explicit_layout_restickify/metadata/latest_flash_proposed_explicit_schema.json`
- `explicit_layout_restickify/replay_senulator/explicit_range_failure_dump.txt`
- `explicit_layout_restickify/diffs/deeptools_workspace.diff`
- `explicit_layout_restickify/diffs/deeptools_status.txt`

## 2026-07-01 DLDSC backend checker update

The latest CDX checkpoint adds a Deeptools-side fail-closed contract checker for the flash `layout_allgather_restickify` class. It validates the full logical contract for the `mul -> ReStickifyOpHBM -> batchmatmul KERNEL` edge and rejects scatter-shaped or partial metadata. Focused Deeptools test result: `LayoutAllgatherRestickify.*`, 4 tests passed.

Artifacts:

- `dldsc_layout_restickify/backend_contract_checker_20260701.md`
- `dldsc_layout_restickify/deeptools_layout_allgather_restickify_checker.patch`


Deeptools fork checkpoint:

- Repository: `git@github.ibm.com:Adnan-Hoque1/deeptools.git`
- Branch: `ah/comms-collectives`
- Commit: `4afc4d9f5` (`[DXP] Add flash layout-allgather contract checker`)

