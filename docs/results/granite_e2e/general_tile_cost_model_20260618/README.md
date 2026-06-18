# General Tile Cost Model Probe, 2026-06-18

This directory records the Granite block investigation around the large-M tile-shape refinement on top of `cost-model-physics`.

Files:

- `large_m_tile_cost_model.md`: what changed in the cost model and why it helps prefill.
- `granite_block_vs_antoni_trace.md`: launch-name and fusion comparison between Antoni's trace and the local Granite block probe.
- `raw_probe_results.txt`: raw `RESULT` and `SDSC` lines from the pod runs.

Implementation:

- `torch_spyre/_inductor/work_division.py`
- `tests/inductor/test_work_division_cost_model.py`
- `benchmarks/granite_block_probe.py`

