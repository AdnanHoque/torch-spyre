Torch gather-restickify validation
date: 2026-07-07T18:40:50+00:00
pod: adnan-spyre-dev-pf
namespace: a6-quantization
checkout: /tmp/codex_validate_gather_restickify_20260707_183447/torch-spyre
branch: gather-restickify
sha: 7a188395295947e7cfe51619f958df712e676c6f
python: /home/adnan/dt-inductor/.venv/bin/python (Python 3.12.13)
pytest: /home/adnan/dt-inductor/.venv/bin/pytest

commands:
  git clone git@github.com:AdnanHoque/torch-spyre.git torch-spyre
  git fetch origin gather-restickify
  git checkout gather-restickify
  git reset --hard origin/gather-restickify
  python -m pytest tests/inductor/test_lx_relayout_dldsc.py -q
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest tests/inductor/test_lx_relayout_dldsc.py -q
  python -m py_compile tests/inductor/test_lx_relayout_dldsc.py
  python -m compileall -q torch_spyre/_inductor tests/inductor/test_lx_relayout_dldsc.py
  PYTHONPATH=$PWD:${PYTHONPATH-} TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest tests/inductor/test_layout_allgather_restickify_import_light.py -q

results:
  pytest_lx_relayout_dldsc.rc: 4
  pytest_lx_relayout_dldsc_no_autoload.rc: 2
  py_compile_test_lx_relayout_dldsc.rc: 0
  compileall_inductor.rc: 0
  pytest_layout_allgather_restickify_import_light.rc: 0

blocker: focused pytest collection requires torch_spyre._C, but this pod resolves it to /home/adnan/dt-inductor/torch-spyre/torch_spyre/_C.so and fails against /opt/ibm/spyre/spyre-comms/lib/libspyre_comms.so.1 with undefined symbol _ZN4flex19AllocationDirectiveC1ENS_15PlacementPolicyESt6vectorIjSaIjEESt8optionalINS_16CompositeAddressEENS_10MemoryTypeE.

logs:
/tmp/codex_validate_gather_restickify_20260707_183447/VALIDATION_SUMMARY.md
/tmp/codex_validate_gather_restickify_20260707_183447/checkout.log
/tmp/codex_validate_gather_restickify_20260707_183447/clone.log
/tmp/codex_validate_gather_restickify_20260707_183447/compileall_inductor.log
/tmp/codex_validate_gather_restickify_20260707_183447/compileall_inductor.rc
/tmp/codex_validate_gather_restickify_20260707_183447/fetch.log
/tmp/codex_validate_gather_restickify_20260707_183447/import_spec_torch_spyre_C.log
/tmp/codex_validate_gather_restickify_20260707_183447/import_spec_torch_spyre_C.rc
/tmp/codex_validate_gather_restickify_20260707_183447/narrow_import_validation.log
/tmp/codex_validate_gather_restickify_20260707_183447/narrow_import_validation.py
/tmp/codex_validate_gather_restickify_20260707_183447/narrow_import_validation.rc
/tmp/codex_validate_gather_restickify_20260707_183447/narrow_import_validation_repo_path.log
/tmp/codex_validate_gather_restickify_20260707_183447/narrow_import_validation_repo_path.rc
/tmp/codex_validate_gather_restickify_20260707_183447/py_compile_test_lx_relayout_dldsc.log
/tmp/codex_validate_gather_restickify_20260707_183447/py_compile_test_lx_relayout_dldsc.rc
/tmp/codex_validate_gather_restickify_20260707_183447/pytest_layout_allgather_restickify_import_light.log
/tmp/codex_validate_gather_restickify_20260707_183447/pytest_layout_allgather_restickify_import_light.rc
/tmp/codex_validate_gather_restickify_20260707_183447/pytest_lx_relayout_dldsc.log
/tmp/codex_validate_gather_restickify_20260707_183447/pytest_lx_relayout_dldsc.rc
/tmp/codex_validate_gather_restickify_20260707_183447/pytest_lx_relayout_dldsc_no_autoload.log
/tmp/codex_validate_gather_restickify_20260707_183447/pytest_lx_relayout_dldsc_no_autoload.rc
/tmp/codex_validate_gather_restickify_20260707_183447/reset.log
/tmp/codex_validate_gather_restickify_20260707_183447/sha.txt
