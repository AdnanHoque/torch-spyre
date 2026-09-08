# Reproduce LX relayout attention and Granite measurements

Two standalone scripts call **unchanged spyre-inference code**. Run each with
relayout OFF and ON on the same idle Spyre device; no helper files are required.
These are custom measurement scripts, not an upstream benchmark command.

## Results from the standalone scripts

Device replay on 2026-09-08, one sequence, FP16, 512 prompt tokens:

| Measurement | OFF | ON | Faster |
| --- | ---: | ---: | ---: |
| Prefill attention, device kernel | 2.363 ms | 1.062 ms | **2.22x** |
| Decode attention, device kernel | 0.751 ms | 0.492 ms | **1.53x** |
| Granite prefill, vLLM engine step | 367.255 ms | 312.472 ms | **1.18x** |
| Granite next decode, vLLM engine step | 205.342 ms | 193.717 ms | **1.06x** |

All six runs passed. Attention inputs and outputs match bit-for-bit between
arms. Granite produces the same tokens `[328, 4281]` and identical selected-token
log probabilities in corresponding OFF/ON requests, including warmup and profiles.
The first warmup's decode log probability differs slightly from the warmed
requests in **both** arms; warmup is not part of the timing comparison.

These are short warmed comparisons, not guaranteed timings on other systems.
Startup is excluded: Granite initialization took 172 s OFF / 226 s ON.
Engine-step time is not HTTP latency or serving throughput.

## Install the matching sources

Use an existing working Spyre environment: Python 3.12, Torch 2.13.0+cpu,
vLLM 0.28.0, Transformers 5.14.1, huggingface-hub 1.26.0, and your usual SDK,
device-profiler and model-cache configuration. The replay used DeepTools
`2245.85f9432` / Spyre comms `120.503dab8` on one device.

**This fork branch already contains the tested compiler stack.** It is frozen
for reproduction, not based on today's latest main. Its compiler code is
`d42522b7da19d06dd03c67adb021203d78023d9b`; this folder is the only addition.
That composition contains #4283, #4284, #3440, #4152, #4153, #3955 and #4347.
#3955 is included, but these attention matmuls do not split their sum. No smaller
dependency set is claimed tested. Other compiler PR branches are unchanged.

```bash
git clone --branch ah/lx-relayout-repro \
  https://github.com/AdnanHoque/torch-spyre.git torch-spyre-lx-repro
python -m pip install --no-build-isolation --no-deps -e ./torch-spyre-lx-repro

git clone https://github.com/torch-spyre/spyre-inference.git spyre-inference-lx-repro
cd spyre-inference-lx-repro
git fetch origin af188385334fe84541a797ad0e5779cfff3b0af6
git fetch origin dd41afe99a6dd2cd9fa002422fe191de7335ecd0
git switch -c lx-repro af188385334fe84541a797ad0e5779cfff3b0af6
git merge --no-ff --no-edit dd41afe99a6dd2cd9fa002422fe191de7335ecd0
git rev-parse 'HEAD^{tree}'
# Expected: 4da6d61d2945ad8db089cdb906745c328da15944
python -m pip install --no-deps -e .
cd ../torch-spyre-lx-repro/tools/lx_relayout_repro
```

The application merge is the measured main + PR #796 composition; there are no
manual application edits. Its original merge commit was `caad9511`; your merge
commit can differ, but the file-tree hash above must match. Use the qualified
environment's Python directly: a later dependency sync can replace the compiler.

This is the standard **slot-major** cache path, not Tom's head-major PR #783
configuration. OFF and ON use the **same patched compiler and application**;
this is neither clean-main-versus-PR nor an additional gain over #783.

## Run the attention kernel

```bash
python paged.py --phase prefill --relayout off --out prefill-off
python paged.py --phase prefill --relayout on  --out prefill-on

python paged.py --phase decode --relayout off --out decode-off
python paged.py --phase decode --relayout on  --out decode-on
```

Read `device_median_ms` in the final output or `result.json`.
**Speedup = OFF time / ON time.** `wall_median_us` is reported separately and
must not be substituted for device time. Missing device events fail the run.

The script calls `_create_compilable_page_attn` in
`spyre_inference/v1/attention/backends/spyre_attn.py`: 8 KV heads, 32 query heads,
head size 128, 128-token pages. Prefill uses 512 query tokens, KV=512 and four
pages. Decode uses one query token, KV=513 and an eight-page bucket (five logical
pages). Query staging is `[513,32,128]`; K and V storage are each
`[16,128,8,128]`. Storage capacity is not the active context length.

Each run compiles once, checks a dense FP32 reference, warms three calls, times
ten synchronized calls, then profiles five separate calls. Inputs and outputs
are saved as `inputs.pt` and `output.pt`; compare both arms before claiming a
gain. Prefill retains upstream tolerance 0.3 absolute / 0.2 relative, with stricter
outliers reported separately; decode uses 0.005 / 0.01. No tolerance was changed.

## Run Granite through vLLM

```bash
python granite.py --relayout off --out granite-off
python granite.py --relayout on  --out granite-on
```

Read `prefill_median_ms` and `decode_median_ms`. The script calls `LLM.generate`
for `ibm-granite/granite-3.3-8b-instruct`, revision
`51dd4bc2ade4059a6bd87649d68aa11e4fb2529b`, FP16. One 512-token prompt generates
two tokens: prefill selects the first, then one decode step at KV=513 selects
the second. Two requests warm up, five measure engine-step wall time, and three
separate requests record profiles. Check the saved tokens and log probabilities;
this is not a full-logit or model-quality test.

## Settings and limits

Both scripts set the measured profile before importing Torch: 32 cores,
`LX_PLANNING=1`, `LAYOUT_SOLVER=greedy`, `DXP_LX_FRAC_AVAIL=0.2`, hazard tracking
and attention recording enabled, and eight compiler CPUs. They clear old
relayout/cache/work-division overrides. Only `SPYRE_LX_PLANNER_RELAYOUT` changes
between OFF and ON. There are no manual work-division hints. These are the
measured settings, not a claim about untouched production defaults.

Every output directory must be new, giving each process a fresh compile cache.
`--help` explains usage; `--dry-run` needs no Torch import or device. Imported
package versions, source locations, available Git heads and settings are saved.
Keep the same environment for both arms and run serially on an idle device.

The earlier emitted prefill bundles showed all eight matmul outputs and the four
probability inputs to value matmuls in LX with relayout ON. Initial cache/page/query
gathering, K before its layout conversion, and final output still used HBM.
This standalone replay checks answers and timing; it does not claim all spills
are removed. The saved `bundle_directories` identify artifacts for further inspection.
