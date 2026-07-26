# Within-device LX broadcast status

This checks the LX relayout collective from epic torch-spyre#3049, not the
separate multi-card `torch.distributed.broadcast` implementation.

## Verdict

Broadcast is not end-to-end supported by PR #2939 yet.

- A natural one-source-core to 32-consumer-core map is rejected because the
  producer and consumer maps have different participant keys.
- Padding the producer map to 32 identical owners is also rejected because
  neither side is a complete partition.
- `_destination_size_ratio(...)` returns `None` for both representations, so
  the planner records no LX relayout plan and emits no broadcast SHUFFLE.
- A manually constructed one-source-to-32-consumer SHUFFLE bundle does compile
  through unpatched DeepTools `e3944781`, demonstrating useful backend
  substrate but not frontend support or device correctness.

The remaining work is a distinct broadcast classification and allocation
contract: represent one physical source owner separately from 32 replicated
destination owners, allocate one destination copy per participating core, and
validate patterned device output without an HBM materialization.
