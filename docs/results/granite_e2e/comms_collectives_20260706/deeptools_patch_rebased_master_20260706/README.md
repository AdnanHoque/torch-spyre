# Deeptools comms collectives patch rebased on master

Generated: 2026-07-07
Base: 0a9da5eb19d08712383312bb7dec18fbd7caf711
Head: 64dfa6e8d24698e2372cd9a1ef6ca9f326bae541
Branch: Adnan-Hoque1/deeptools ah/comms-collectives-onegate-artifact

This patch includes the gather/restickify comms-collectives prototype plus the one-gate alias update.
The public feature flag is SPYRE_LX_PLANNER_RELAYOUT=1. Legacy DEEPTOOLS_* knobs remain as debug overrides, but are not required for the normal path in this patch.

Regenerate with:

    git fetch origin master ah/comms-collectives-onegate-artifact
    git diff --binary origin/master..origin/ah/comms-collectives-onegate-artifact > deeptools_ah_comms_collectives_rebased_on_master.patch
