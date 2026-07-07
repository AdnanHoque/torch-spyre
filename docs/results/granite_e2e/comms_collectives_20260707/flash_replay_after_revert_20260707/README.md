# Flash replay after source-chunk revert - 2026-07-07

This is a DXP-only replay of the saved full flash SuperDSC bundle after
reverting the diagnostic source-core-aware chunking patch from Deeptools.

Result:

- return code: `0`
- backend plans: `64`
- communication pattern: `all_gather_replicate`
- realization strategy: `gather_then_restickify`
- physical lowering status: `lowered_gather_then_restickify`

This confirms that the current Deeptools `ah/comms-collectives` head keeps the
saved bounded flash replay working.  The reverted source-core chunking patch is
kept only as a diagnostic artifact in the Granite fail-closed checkpoint because
it caused an IBuff failure on this same saved flash replay.
