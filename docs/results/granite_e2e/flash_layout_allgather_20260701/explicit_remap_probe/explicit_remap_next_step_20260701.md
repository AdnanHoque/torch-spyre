# Explicit Remap Next Prototype Step - Flash Layout AllGather Restickify

## Scope

Pod-local work only on CLC: `adnan-clc-spyre-dev-pf`.
Workspace: `/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212`.

Only explicit-remap/deeptools prototype code was changed. Torch and DLDSC backend files were not edited.

## Files Changed

- `/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/deeptools/scripts/explicit_lx_range_semantic_check.py`
  - added `synthesize_layout_allgather_ranged_remap(...)`
  - added `make_synthetic_ranged_sdsc(...)`
  - added CLI flags `--emit-ranged-sdsc`, `--emit-ranged-remap-json`, `--source-lx-base`, `--destination-lx-base`
  - generated concrete explicit `movementRanges` for the latest flash `layout_allgather_restickify` edge

Generated artifacts:

- latest probe pointer: `/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/metadata/latest_concrete_layout_allgather_range_probe.txt`
- latest probe dir: `/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/concrete_layout_allgather_range_probe_20260701_083445`
- concrete remap JSON: `/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/concrete_layout_allgather_range_probe_20260701_083445/layout_allgather_concrete_ranged_remap.json`
- synthetic SDSC: `/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/concrete_layout_allgather_range_probe_20260701_083445/layout_allgather_concrete_range_sdsc.json`
- semantic checks: `/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/concrete_layout_allgather_range_probe_20260701_083445/layout_schema_to_ranges_check.*`, `/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/concrete_layout_allgather_range_probe_20260701_083445/concrete_range_semantic_check.*`
- normal DXP replay: `/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/concrete_layout_allgather_range_probe_20260701_083445/dxp_bundle_replay`
- senulator replay: `/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/concrete_layout_allgather_range_probe_20260701_083445/dxp_bundle_replay_senulator`
- script patch: `/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/concrete_layout_allgather_range_probe_20260701_083445/explicit_lx_range_semantic_check.patch`

## Result

The latest flash edge can now be represented as a concrete explicit-remap carrier:

- communication class: `layout_allgather_restickify`
- producer: `mul.OUTPUT` in LX
- consumer: `batchmatmul.KERNEL` in LX
- groups: 4
- producer chunks per group: 8
- consumer replicas per group: 8
- concrete range count: 256
- bytes per range: 131072
- modeled total movement: 33554432 bytes
- each destination core receives contiguous 1048576 bytes at `[0x100000, 0x200000)`

## Commands Run

```bash
cd /home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/deeptools
python3 -m py_compile scripts/explicit_lx_range_semantic_check.py

python3 scripts/explicit_lx_range_semantic_check.py \
  --layout-schema /home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/metadata/latest_flash_proposed_explicit_schema.json \
  --emit-ranged-sdsc /home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/concrete_layout_allgather_range_probe_20260701_083445/layout_allgather_concrete_range_sdsc.json \
  --emit-ranged-remap-json /home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/concrete_layout_allgather_range_probe_20260701_083445/layout_allgather_concrete_ranged_remap.json \
  --text-out /home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/concrete_layout_allgather_range_probe_20260701_083445/layout_schema_to_ranges_check.txt \
  --json-out /home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/concrete_layout_allgather_range_probe_20260701_083445/layout_schema_to_ranges_check.json

python3 scripts/explicit_lx_range_semantic_check.py \
  --sdsc /home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/concrete_layout_allgather_range_probe_20260701_083445/layout_allgather_concrete_range_sdsc.json \
  --text-out /home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/concrete_layout_allgather_range_probe_20260701_083445/concrete_range_semantic_check.txt \
  --json-out /home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/concrete_layout_allgather_range_probe_20260701_083445/concrete_range_semantic_check.json

DEEPTOOLS_PATH=/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/deeptools \
DEEPTOOLS_EXPLICIT_LX_RANGE_PROTO=1 \
DEEPTOOLS_EXPLICIT_LX_RANGE_PROTO_DUMP=/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/concrete_layout_allgather_range_probe_20260701_083445/dxp_bundle_replay/explicit_range_diag.txt \
/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/deeptools/build-explicit-range-agent-nollvm/dxp/dxp_standalone \
  --bundle -d /home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/concrete_layout_allgather_range_probe_20260701_083445/dxp_bundle_replay/bundle_input

DEEPTOOLS_PATH=/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/deeptools \
DEEPTOOLS_EXPLICIT_LX_RANGE_PROTO=1 \
DEEPTOOLS_EXPLICIT_LX_RANGE_PROTO_DUMP=/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/concrete_layout_allgather_range_probe_20260701_083445/dxp_bundle_replay_senulator/explicit_range_diag.txt \
/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/deeptools/build-explicit-range-agent-nollvm/dxp/dxp_standalone \
  -b senulator --bundle -d /home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/flash_explicit_layout_restickify_20260701_062020/concrete_layout_allgather_range_probe_20260701_083445/dxp_bundle_replay_senulator/bundle_input
```

## Pass/Fail

- Python compile: PASS
- layout schema validation: PASS
- generated concrete range semantic validation: PASS
- normal DXP bundle replay/import/routing: PASS, rc=0
- senulator backend replay: FAIL, rc=134

Senulator failure:

```text
DtException: skv.second <= layoutSize, file .../dcg/dcg_fe/transfer_compute/transfer_compute.cpp line 639
pieceVerificationFailure dataOpDsc=ProgCorrectionScatter0 op=ScatterOpHBM lds=ProgCorrectionFlit piece=p0 dim=d1 pieceSize=1 layoutSize=0
```

## Concrete Remaining Runtime/Backend Gap

The explicit-remap lane can now represent the latest flash all-gather/restickify edge as concrete `STCDPOpLx.rangedLxRemap.movementRanges`, and normal DXP accepts/routes that carrier.

The remaining blocker is backend/senulator realization, not schema representation: senulator compile-time correction still synthesizes a legacy `ProgCorrectionScatter0` / `ProgCorrectionFlit` path with a zero-sized LDS dimension. That path must either be taught to preserve the explicit range LDS layout or bypass the legacy program-correction scatter path for explicit LX range transfers.

A second still-open e2e gap remains after that: the copied-bundle replay proves carrier import/routing, but a real flash lowering must also bind the resulting LX-resident KERNEL view to the downstream PT `batchmatmul` operand and remove the original HBM `ReStickifyOpHBM` row.
