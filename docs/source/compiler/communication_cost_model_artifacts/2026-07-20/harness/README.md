# Harness inventory

These are byte-for-byte copies of the scripts used by the corrected campaign,
plus the exact base probes and QC/closure helpers that its wrappers import.
They are evidence, not a portable standalone benchmark: the runner paths refer
to the pinned source and runtime layout recorded by the preregistration.

`analyze_coherent_oracle_factorial_v2.py` is the filename used by the campaign
runner. `analyze_joint_oracle_factorial.py` is the identical local-source alias;
both hashes are retained to make that name handoff auditable.

The `joint_*` scripts reproduce the quarantined incoherent negative control.
Their historical comments must not be read as a performance conclusion. The
authoritative disposition is `../structural/failed_joint/STATUS.json`: the
high-contrast correctness gate failed and no timing was collected.

The package intentionally omits generated environment captures, traces, tensor
outputs, compiler caches, and console logs. `HARNESS_SHA256SUMS` covers every
file in this directory except the manifest itself.
