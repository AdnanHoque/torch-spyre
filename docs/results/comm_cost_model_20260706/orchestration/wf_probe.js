export const meta = {
  name: 'model-probe',
  description: 'Probe which model alias yields Fable subagents',
  phases: [{ title: 'Probe' }],
}
phase('Probe')
// Three variants to find the one that routes subagents to Fable.
const results = await parallel([
  () => agent('Reply with exactly: ok', { label: 'alias-fable', model: 'fable' }),
  () => agent('Reply with exactly: ok', { label: 'full-id', model: 'claude-fable-5' }),
  () => agent('Reply with exactly: ok', { label: 'no-override' }),
])
return { results }
