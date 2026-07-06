# Flash staged gather-restickify replay at Deeptools a4930be14

This replay checks that the Granite 4-slot chunking patch does not regress the flash attention staged relayout bundle that previously lowered 32 matmul operand broadcasts.

## Environment

- Pod: adnan-spyre-dev-pf
- Torch artifact branch SHA: 66eea3e04deb5d6b0ca7450dba2a7202f72fab42
- Deeptools SHA: a4930be14b6e7d01f7447b7692a79a20487c09c3
- Source run: /home/adnan/codex-isolated/flash_attention_comms_backend2162_20260706_005751/staged_gather_restickify_replay_20260706_071930_a4930be14

## Result

- returncode: 0
- backend plans: 32
- logical transfers total: 8192
- kind counts: {'matmul_operand_broadcast': 32}
- strategy counts: {'gather_then_restickify': 32}
- status counts: {'lowered_gather_then_restickify': 32}

All 32 plans are matmul_operand_broadcast / all_gather_replicate and lower as gather_then_restickify.

## Scope

This is DXP/DCC replay evidence, not value correctness. CDX could not run the fresh script from the current source tree because its Torch extension import is ABI-broken.
