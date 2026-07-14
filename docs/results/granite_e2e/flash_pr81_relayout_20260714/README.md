# Flash PR81 LX Relayout Experiment

## Question

Measure the benefit of replacing the HBM restickify before the attention score
matmul with grouped LX all-gather plus stick relayout for the three flash
attention variants in `spyre-perf-suite` PR81.

The experiment is structural and performance-oriented. Flash numeric
correctness is not an acceptance criterion because the baseline kernel has a
separate known correctness issue.

## Source State

| Component | Branch | SHA |
|---|---|---|
| Torch | `pr-lx-relayout-scatter` | `832aa67a28f47345e074d802251562924c286fc3` |
| Deeptools | `adnan/lx-relayout-scatter-sizing` | `704c19f8fb7f0cc972f20404f9dd0010895a35e2` |
| spyre-perf-suite | `jamie/flash_attn` | `6f4e54127f825ed2370f866e80292422f97d6b0` |
| Pod image | `torch-spyre-sshd:latest` | `sha256:913f394b4b3f03740a9d35f70f273b1cb799d4cba55f7fdaff108d7749d77964` |

The three benchmark operations are:

- `experimental.flash_attn_online_softmax`
- `experimental.flash_attn_softmax`
- `experimental.flash_attn_stable_softmax`

The tested shapes use `B=1`, `H=4`, `D=128`, `Lk=4096`, and `Lq` in
`{512, 1024}`. Masked rows add a `[1, 1, Lq, 4096]` mask.

## Variants

| Variant | Torch LX fraction | DXP LX fraction | Relayout |
|---|---:|---:|---|
| `off0p2` | `0.2` | `0.2` | disabled; production-capacity control |
| `off0p6` | `0.6` | `0.6` | disabled; same backend-capacity control |
| `split` | `0` | `0.6` | enabled; benchmark-only split accounting |

`split` is deliberately not production-safe. Torch sees full LX while the DXP
subprocess sees 60% backend workspace. The small patch in
`split_lx_benchmark_only.patch` lets the Torch planner validate against the
backend fraction, and `scripts/dxp_standalone_wrapper.sh` rewrites the DXP
subprocess environment. Production needs one allocator to account for both the
resident source and backend staging; it must not overcommit LX this way.

Using `DXP_LX_FRAC_AVAIL=0.6` directly is not sufficient. It gives DXP staging
space but removes enough frontend capacity that the 1 MB source cannot remain
LX-resident, so the relayout never fires.

## Structural Result

The same edge changes in every successful `split` row:

```text
K_scaled = K * scale
S = Q @ K_scaled.T
```

The tracked edge is `K_scaled` into the score matmul.

| Stage | Control | Split relayout |
|---|---|---|
| Producer shard | LX, 128 KB/core | LX, 128 KB/core |
| Layout conversion | `ReStickifyOpHBM` | `ReStickifyOpLx` |
| Consumer K operand | HBM, assembled 1 MB/core view | LX, assembled 1 MB/core view |
| Explicit allocation distribution | empty | populated `coordinates_.coreIdToWkSlice_` |

For the representative `flash_attn_softmax`, `Lq=512` row:

| Variant | Op | Input | Output | Per-core tile |
|---|---|---|---|---|
| `off0p2` | `ReStickifyOpHBM` | LX | HBM | 128 KB |
| `split` | `ReStickifyOpLx` | LX | LX | 128 KB |
| `off0p2` | score `batchmatmul` K input | HBM | n/a | 1 MB |
| `split` | score `batchmatmul` K input | LX | n/a | 1 MB |

The frontend bundle alone does not prove physical ring movement. A manual replay
of the representative split bundle against the custom DXP succeeded, and a GDB
breakpoint at `SdscRelayoutInsertion.cpp:159` hit while DXP constructed the
inserted `STCDPOpLx`. The stack was:

```text
Dxp::insertRelayoutSdsc
Dxp::runDsmRelayout
Dxp::runDxp
```

The allocation map contained all 32 cores. This proves that the backend
materialized movement rather than only accepting a renamed LX restickify row.

## Performance Result

Representative `Lq=512`, unmasked rows:

| Operation | Control kernel ms | Split kernel ms | Kernel speedup | Control wall ms | Split wall ms | Wall speedup |
|---|---:|---:|---:|---:|---:|---:|
| online softmax | 0.306 | 0.232 | 1.319x | 15.760 | 15.669 | 1.006x |
| materialized softmax | 0.296 | 0.247 | 1.198x | 15.378 | 14.551 | 1.057x |
| stable softmax | 0.245 | 0.191 | 1.283x | 16.570 | 17.017 | 0.974x |

The bounded `Lq=512` cases consistently improve kernel time by 1.18x-1.32x.
Wall time is dominated by host/runtime overhead and ranges from a small loss to
about a 6% win.

At `Lq=1024`, the result reverses versus the production-capacity control:

| Operation | `off0p2` kernel ms | `split` kernel ms | Relative result |
|---|---:|---:|---:|
| online softmax | 0.439 | 1.296 | 0.339x |
| materialized softmax | 0.430 | 1.279 | 0.336x |
| stable softmax | 0.330 | 1.180 | 0.280x |

The relayout still fires and remains faster than the pathological direct-0.6
control, but full-tensor materialization is not a viable production strategy at
this size. The result supports bounded/chunked lowering, coordinated with WSR,
instead of globally expanding each core's resident operand.

The masked online-softmax baseline hung before any relayout-specific behavior,
so that operation/shape pair was not rerun. The other masked variants completed
and are included in `summary.md` and `summary.csv`.

## Reproduction

The shared pod root used for these runs is:

```text
/home/adnan/codex-isolated/flash_pr81_relayout_20260714
```

All healthy pods use:

```bash
source /home/adnan/spyre-envs/image-913f394b4b3f/activate.sh
```

Run one operation on one pod:

```bash
LQ_VALUES="512 1024" MASK_VALUES="0 1" \
  bash /home/adnan/codex-isolated/flash_pr81_relayout_20260714/bin/run_flash_pr81_matrix.sh \
  flash_attn_softmax dev
```

The runner always executes `off0p2`, `off0p6`, and `split`. Filters can limit
the matrix:

```bash
LQ_VALUES=1024 MASK_VALUES=0 VARIANTS="off0p2 off0p6 split" \
  bash /home/adnan/codex-isolated/flash_pr81_relayout_20260714/bin/run_flash_pr81_matrix.sh \
  flash_attn_online_softmax current-recovered
```

Regenerate the aggregate report:

```bash
python3 scripts/summarize_flash_pr81_matrix.py \
  /home/adnan/codex-isolated/flash_pr81_relayout_20260714/runs/pr81_matrix \
  --csv /home/adnan/codex-isolated/flash_pr81_relayout_20260714/runs/pr81_matrix/summary.csv \
  --markdown /home/adnan/codex-isolated/flash_pr81_relayout_20260714/runs/pr81_matrix/summary.md
```

## Pod Recovery

An interrupted runtime can leave VFIO busy. Hot reset did not recover this
state. Recreating the pod did:

```bash
kubectl get pod POD -o json \
  | jq 'del(.metadata.creationTimestamp,
            .metadata.resourceVersion,
            .metadata.uid,
            .metadata.managedFields,
            .metadata.annotations["kubectl.kubernetes.io/last-applied-configuration"],
            .spec.nodeName,
            .status)' > /tmp/POD.json
kubectl delete pod POD --wait=true
kubectl create -f /tmp/POD.json
kubectl wait --for=condition=Ready pod/POD --timeout=300s
```

After recreation, validate the card with a real allocation:

```bash
source /home/adnan/spyre-envs/image-913f394b4b3f/activate.sh
python -c 'import torch; print(torch.ones(8, device="spyre").cpu())'
```

`adnan-clc-spyre-dev-pf` initially remained unusable after a same-node
recreation because the card on `p1-worker-23` reported
`DdrInitRetryLimitExceeded`. Recreating it with a required node-affinity
expression excluding `p1-worker-23` placed it on `p1-worker-4`; the real tensor
allocation then passed. This confirmed a hardware/node issue rather than an
image or Python environment mismatch.

## Artifact Index

- `summary.md`: complete timing and structural matrix.
- `summary.csv`: machine-readable matrix.
- `analysis/`: Jamie-style SDSC summaries for the bounded win and large-shape
  regression, plus the saved DXP GDB breakpoint proof.
- `selected_sdsc/`: original generated SDSC bundles for those four cases.
- `run_metadata/`: per-run environment, version, timing, and profiler reports.
- `scripts/`: exact matrix runner, parser, and DXP wrapper.
- `split_lx_benchmark_only.patch`: the isolated eight-line Torch benchmark
  patch; not proposed production code.
