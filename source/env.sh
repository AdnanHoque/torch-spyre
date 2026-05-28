# Environment for reproducing the SDSC dump in sdsc_dumps/mlp_M512_K4096_N12800/.
# Source this file (`source env.sh`) before running compile_mlp_matmul.py.

# Use the cost-model branch's torch_spyre (via shim).
export PYTHONPATH=/tmp/cost_model_shim

# Enable the cost-model matmul planner (gates _cost_model_matmul_planner in
# work_division.py). With this OFF, the heuristic picks (32, 1, 1).
export SPYRE_COST_MODEL_MATMUL_PLANNER=1

# Standard Spyre runtime env.
export DXP_LX_FRAC_AVAIL=1
export SENCORES=32
export USE_SPYRE_PROFILER=1

export LD_LIBRARY_PATH=/home/adnan/dt-inductor/.venv/lib/python3.12/site-packages/torch/lib:/home/adnan/dt-inductor/sentient/libaiupti/lib:/home/adnan/dt-inductor/sentient/runtime/lib:/home/adnan/dt-inductor/sentient/deeptools/lib:/opt/ibm/spyre/tvm/lib:/opt/ibm/spyre/spyre-comms/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib:/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/sentinyexec/lib

# Optional: capture Inductor debug artifacts (output_code.py, fx_graph_*.py,
# ir_pre_fusion.txt, ir_post_fusion.txt). Files land in /tmp/torchinductor_<user>/.
# export TORCH_COMPILE_DEBUG=1
# export TORCH_LOGS="+inductor"
