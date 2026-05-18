# Stage 31: GTR Multicast Contract Probe

## Question

Stage 30 made multicast-aware constant and weight fanout look like a promising
ring-aware compiler project. Before proposing a torch-spyre codegen change, this
stage checked the backend contract:

- Does the SDSC / Deeptools schedule path already have a way to express GTR
  multicast?
- Do torch-spyre-generated SDSCs trigger that path today?
- If not, is the missing work likely to be "emit a new GTR field" or "shape
  the existing HBM-pinned fanout so Deeptools can identify sharing"?

## Tool

Added a local probe:

```sh
python tools/sdsc_gtr_probe.py --run-scheduler --output-dir /tmp/stage31-gtr-contract-probe <sdsc json...>
```

The tool can:

1. Run `L3DlOpsScheduler_standalone` on one or more SDSC JSON files.
2. Walk the scheduled JSON.
3. Summarize schedule-tree nodes carrying `coreIdToGTRInfo_`.
4. Report whether any per-core entries have `groupId_ >= 0` and
   `numSharers_ > 1`, which is the observable scheduled-JSON signal for
   multicast rather than unicast.

Artifacts from the first run live under:

```text
artifacts/stage31_gtr_contract_probe/raw
```

## Backend Facts

The Deeptools source already exposes the relevant schedule-tree field:

```cpp
std::map<int, dsc2::GroupTagRegInfo> coreIdToGTRInfo_;  // L3
```

`GroupTagRegInfo` carries:

- `groupId_`
- `numSharers_`

`L3DlOpsScheduler::fillTransferMulticastInfo()` already computes this field for
HBM-pinned tensors transferred through L3LU. It scans HBM-resident labeled data,
finds HBM-to-LX or HBM-to-L3LUIBR transfer nodes, computes sharing with
`getSharesAndGroupName(...)`, and writes `coreIdToGTRInfo_` onto the transfer
node.

The scheduler explicitly says there is no multicast support on L3SU in this
path, so this is an inbound HBM-to-core fanout mechanism.

The Spyre knowledge base matches this contract:

- `wiki/artifacts/designs/schedule-ir-spec.md` lists `gtr` as optional
  per-core group-tag-register metadata for multicast transfers.
- `wiki/entities/aiu.md` describes GTR multicast as producer-to-multi-consumer
  ring delivery.
- `wiki/foundations/hardware/microarchitecture.md` confirms that all
  off-chip-memory to core data passes through L3LU/L3SU, the ring-facing
  interfaces.

## Probe Input

The probe ran over 40 recently generated torch-spyre SDSCs from the pod's
Inductor cache, after L3 scheduling.

Observed operation roots:

| Root op | Rows | GTR Nodes | Multicast Nodes |
|---|---:|---:|---:|
| `ReStickifyOpHBM` | 14 | 14 | 0 |
| `add` | 22 | 44 | 0 |
| `batchmatmul` | 3 | 6 | 3 |
| `sumnonstick` | 1 | 1 | 0 |
| **Total** | **40** | **65** | **3** |

Across the full run:

| Metric | Value |
|---|---:|
| SDSC rows | 40 |
| Nodes with `coreIdToGTRInfo_` | 65 |
| Rows with multicast | 3 |
| Multicast nodes | 3 |
| Multicast core entries | 96 |
| Max sharers | 32 |

## Concrete Examples

Scheduled `batchmatmul`:

| Transfer node | GTR summary |
|---|---|
| `transfer_lds0_src:hbm_dst:lx` | `groupId=-1`, `numSharers=1` on 32 cores |
| `transfer_lds1_src:hbm_dst:lx` | `groupId=0`, `numSharers=32` on 32 cores |

Scheduled `ReStickifyOpHBM`:

| Transfer node | GTR summary |
|---|---|
| `transfer_lds0_src:hbm_dst:lx` | `groupId=-1`, `numSharers=1` |

Scheduled `add`:

| Transfer node | GTR summary |
|---|---|
| `transfer_lds0_src:hbm_dst:lx` | `groupId=-1`, `numSharers=1` |
| `transfer_lds1_src:hbm_dst:lx` | `groupId=-1`, `numSharers=1` |

## Interpretation

This changes the shape of the multicast project.

The first hypothesis was: torch-spyre may need to emit `coreIdToGTRInfo_`
directly. The probe suggests a better hypothesis:

> Deeptools already owns the low-level GTR assignment for HBM-pinned inbound
> transfers. Torch-spyre should first make sure its SDSCs expose read-only
> sharing in the form Deeptools already recognizes.

In other words, the near-term compiler work is probably not a new public SDSC
field. It is attribution and shaping:

1. Identify graph-input, constant, weight, and shared activation fanout edges
   where multiple cores consume the same HBM-resident logical data.
2. Verify whether the resulting SDSC marks that data HBM-pinned with compatible
   ownership metadata.
3. Run the scheduler and check whether `numSharers_ > 1`.
4. If sharing is hidden from Deeptools, adjust torch-spyre's layout/staging
   choices so the existing GTR pass can see it.
5. Only if that fails, consider explicit GTR emission from torch-spyre.

## Relationship To Restickify

The sampled `ReStickifyOpHBM` SDSCs did not use GTR multicast. That is expected:
restickify is usually a permutation/rewrite, not a many-consumer broadcast of
the same HBM tile. GTR is therefore a sibling ring-aware project, not a direct
replacement for Stage 3B.

The clean opportunity is weight or constant fanout. Batchmatmul already shows
one 32-sharer multicast in scheduled torch-spyre output, so the backend path is
not hypothetical.

## Next Steps

1. Add a probe case that deliberately creates shared read-only fanout from one
   HBM input or constant to many cores.
2. Compare generated SDSCs where GTR appears against ones where it does not.
3. Add telemetry around HBM-pinned logical data sharing before scheduling.
4. Build a small torch-spyre-side diagnostic that reports "GTR-eligible but not
   scheduled as multicast" rows.
5. Keep this separate from Stage 3B. Stage 3B is about preserving producer to
   restickify locality; GTR is about reducing repeated inbound fanout traffic.
