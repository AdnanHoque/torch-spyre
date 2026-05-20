# Stage 144: Deeptools Graph Port Finding

## Summary

After the Stage 143 mixed-graph retry, I inspected the installed Deeptools graph
headers instead of trying more runtime launches.

The important finding is that `sengraph::Edge::Pair.index_` is not an internal
SDSC `labeledDs_` index. It is the number of the graph input/output edge used by
the graph edge.

From the installed header:

```cpp
/*
 * pair of node/index
 * NOTE: Index is only meaningful for data edges, it captures
 * the number of input or output used by the edge
 * index is not defined for control edges
 */
struct Pair {
  Node *node_ = nullptr;
  size_t index_ = -1;
};
```

The convenience API simply passes those graph edge indices through:

```cpp
Edge *insertDataEdge(Node *src, size_t sidx, Node *tgt, size_t tidx) {
  return insertDataEdge(Edge::Pair(src, sidx), Edge::Pair(tgt, tidx));
}
```

This explains why the Stage 143 attempt with `consumer_input_index=1` crashed
during graph construction. We were treating a graph edge index as if it were
`sdsc_2_add`'s internal `labeledDs_` index.

## Additional Constraint

`DscSenGraph::finalizeDscSenGraph` is also very linear. It follows the first
output edge list:

```cpp
if (currNode->outputs().size() > 0) {
  if (currNode->outputs()[0].size() > 0) {
    for (auto &edge : currNode->outputs()[0]) {
      ...
      nextNode = edge->target_node();
    }
  }
}
```

So the simple `DscSenGraph` path is not a rich internal-SDSC wiring mechanism.
It is a graph of prepared ops with graph-level ports, while the restickify
problem needs a binding to a specific consumer `labeledDs_` / schedule-tree
allocation.

## Interpretation

This rules out a second shortcut:

```text
Use insertDataEdge(..., consumer_input_index=1) to bind the data-op output to
consumer Tensor1 / ldsIdx=1.
```

That is not what the API means.

The API may still have a deeper prepared-op binding mechanism elsewhere, but it
is not exposed by this basic `insertDataEdge` interface. Without that deeper
hook, a mixed graph built from already-prepared Torch-Spyre SDSCs cannot express
the internal LX alias we need.

## Updated Direction

The most promising path is now the DXP/fused-bundle or compound-SDSC route:

1. Keep the normal Torch-Spyre fused bundle ABI so runtime segments remain
   valid.
2. Change the internal producer/restickify/consumer contract inside that bundle,
   not through external Deeprt graph edges.
3. Make DDC/DCC see one coherent internal LX edge:

```text
producer output LX
  -> LX restickify movement
  -> consumer input LX
```

The next offline experiment should inspect whether the existing fused bundle can
be transformed into a compound SDSC/DLDSc-shaped artifact, or whether we need to
generate a single custom bridge-aware compute/data schedule for the fixture.

## Device Safety

No generated kernels were launched for this stage. This was header/source
inspection only.
