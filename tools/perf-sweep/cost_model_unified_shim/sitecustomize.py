"""Make `import torch_spyre` resolve to the cost-model-unified worktree
(/tmp/cost-model-unified) rather than the main venv's editable install
OR the current working directory.

Identical strategy to /tmp/cost_model_shim, but points at the unified
branch where matmul + pointwise + reduction sibling planners all share
the same Option-C scaffolding.
"""
import sys

_WORKTREE = "/tmp/cost-model-unified"
_MAIN_REPO = "/home/adnan/dt-inductor/torch-spyre"

if _WORKTREE not in sys.path:
    sys.path.insert(0, _WORKTREE)

# Also patch the editable finder's MAPPING in case it runs before
# PathFinder somehow.
try:
    import __editable___torch_spyre_0_0_1_finder as _ef
    for _k in list(_ef.MAPPING):
        if _k == "torch_spyre" or _k.startswith("torch_spyre."):
            _ef.MAPPING[_k] = _ef.MAPPING[_k].replace(_MAIN_REPO, _WORKTREE)
    for _k in list(_ef.NAMESPACES):
        if _k == "torch_spyre" or _k.startswith("torch_spyre."):
            _ef.NAMESPACES[_k] = [
                p.replace(_MAIN_REPO, _WORKTREE) for p in _ef.NAMESPACES[_k]
            ]
except ImportError:
    pass
