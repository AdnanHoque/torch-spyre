# Ring-native matmul handoff

Last updated: 2026-07-29

## Executive status

The ordinary stock matmul is healthy on the exact pinned stack used by the
ring-matmul experiments. The fresh normal-frontend control compiled and ran

```text
A[64, 2048] @ W[2048, 4096] -> C[64, 4096]
```

through `torch.matmul -> torch.compile -> stock BMM -> device -> D2H`, and all
262,144 outputs passed the CPU comparison in every executed profile.

The ring-native candidate is **not ready to time**. Its latest
K32-epoch/PE-LX-carry lowering:

- compiles;
- passes a complete emitted-program balance audit;
- launches on device;
- deadlocks at `device_synchronize`;
- recovers cleanly to a completing stock launch after the timed-out process is
  terminated.

There is therefore no ring-matmul speedup result yet. The immediate blocker is
candidate-specific completion, not stock matmul, the device, or the pinned
runtime stack.

This is a documentation-only handoff. The implementation, tests, prototypes,
manifests, and all unrelated worktree changes remain local and unstaged.

## Scope and non-goals

The mechanism-oracle shape is FP16 `M64 x K2048 x N4096` on 32 cores.

The paid boundary is:

- A starts in producer-native LX;
- W starts in HBM;
- C ends in its required output layout;
- all communication, conversion, computation, and final stores remain inside
  the paid device event.

The research goal is not work-division tuning. A different decomposition is
allowed only when it is part of a new distributed dataflow, and it must retain
same-grid ring-off controls so communication credit remains identifiable.

The final candidate must:

1. beat its identical-grid no-ring control;
2. beat the current best stock matmul;
3. consume core-to-core arrivals directly rather than materializing a
   standalone shuffle;
4. demonstrate useful traffic on both directions of the RIU and SFP fabrics
   when the topology and payload permit it;
5. avoid redundant link bytes and balance the critical links;
6. pass full dynamic correctness before any latency is accepted.

## First-principles target dataflow

For this shape, A is 256 KiB while W is 16 MiB. Moving W around the ring is the
wrong default. The natural distributed algorithm is output-column stationary:

1. shard W by N and keep each W shard at its owner;
2. send the smaller A panels from their producer-native LX placement to the N
   owners over RIU LX-to-LX transport;
3. compute as A arrives instead of writing a standalone relayout result and
   launching a second kernel;
4. retain C at the N owner across all K epochs;
5. use the SFP fabric for local pair sharing/reduction work that does not
   duplicate RIU traffic;
6. write the final C owner shard once.

The selected physical candidate is `M2N16K1` with:

```text
core = 2*n + m
```

Each logical N owner is an adjacent pair of physical cores. This gives the
design two separable communication opportunities:

- RIU: distribute A from LX to the N-owner pairs;
- SFP: share the complementary W work within each adjacent pair while PT
  computes and C remains output-stationary.

The intended steady-state pipeline is:

```text
RIU receives A[g+1]
    ||
SFP/HMI supplies the W work for g
    ||
PT computes g and PE/LX retains C
```

The current K32 PE/LX-carry experiment is only a compute/lifetime control for
this design. It uses HMI-local W delivery and does not yet prove the full
dual-ring algorithm.

## Difference from the incumbent

The normal stock control emits one unhinted `M4N8K1` HBM BMM. A, W, and C use
the ordinary stock path. Hardware multicast may already reduce repeated HBM
delivery, but the operation does not consume a producer-native LX A placement
through the new LX-to-LX substrate.

The target changes the dataflow, not merely the tile sizes:

| Property | Stock incumbent | Ring-native target |
|---|---|---|
| A start | ordinary HBM BMM input | live producer-native LX |
| W policy | stock HBM delivery | N-sharded/stationary owner |
| Cross-core A | none at the graph boundary | RIU LX-to-LX fan-out/stream |
| SFP role | stock internal behavior | useful adjacent-pair W sharing |
| C lifetime | stock generated schedule | retained at owner across K epochs |
| Fusion | one ordinary BMM | receive/compute/carry in one schedule |
| Attribution | incumbent | same-grid ring-off plus incumbent controls |

The theoretical advantage is removal or overlap of A materialization and
owner delivery while preserving W and C ownership. It is not a claim that
communication is free, nor that changing from `M4N8` to `M2N16` is itself a
speedup.

## Compiler substrate present in this fork

The host fork is:

```text
remote: https://github.com/AdnanHoque/torch-spyre.git
branch: ah/communication-cost-model
research base before this documentation commit:
        6952529929cdb9c9e8668e11d94d61312c5d83ca
```

The worktree is intentionally dirty. Thirteen tracked compiler/test files have
853 insertions and 35 deletions, plus the untracked research tree. Preserve
all unrelated work.

The current compiler changes make the target layout expressible by reusing
existing infrastructure:

1. `TensorPhysicalLayout` carries an explicit compound/repeated physical stick
   factorization such as `M4 x K2 x M8`.
2. `SPYRE_MATMUL_ACTIVATION_LAYOUT=preserved` selects the BMM input layout that
   preserves M in A's physical stick instead of the stock K/reduction stick.
3. work-division validation permits the logical M2 split while retaining one
   physical FP16 M64 stick and an explicit per-core tail gap.
4. equal-footprint destination geometry can select the existing
   `ReStickifyOpLx` path, avoiding an HBM restickify.
5. SuperDSC serializes the explicit physical stick sequence and validates that
   it exactly covers the flattened device geometry.
6. source and destination ownership remain represented by the existing
   relayout/STCDP machinery rather than new raw send/receive operations.

Relevant source:

- [`torch_spyre/_inductor/op_spec.py`](../torch_spyre/_inductor/op_spec.py)
- [`torch_spyre/_inductor/propagate_layouts.py`](../torch_spyre/_inductor/propagate_layouts.py)
- [`torch_spyre/_inductor/work_division.py`](../torch_spyre/_inductor/work_division.py)
- [`torch_spyre/_inductor/spyre_kernel.py`](../torch_spyre/_inductor/spyre_kernel.py)
- [`torch_spyre/_inductor/codegen/superdsc.py`](../torch_spyre/_inductor/codegen/superdsc.py)
- local research note `ring_compute_prototypes/COMPILER_CONTRACT.md`
- local research ledger `ring_compute_prototypes/RING_NATIVE_MATMUL_LEDGER.md`

This substrate is not yet a production-ready patch set. It must be reduced
into separately reviewable compiler changes only after the device algorithm
passes.

## Correctness-gate correction

An important false lead was removed on 2026-07-29.

The sealed `r64` replay was not a valid test of stock BMM arithmetic. It
executed:

```text
LX restickify -> stock BMM
```

as one integrated artifact, but the restickify schedule had neither
`before_sync` nor `after_sync` before BMM consumption. Across historical runs,
the same inputs and ABI produced different hashes and recovered different
multiples of 64 dominant columns. That is consistent with a producer/consumer
race, not a deterministic BMM layout or arithmetic failure.

Therefore:

- the sealed `r64` completion result remains useful only for completion;
- its full-output result must not gate stock BMM correctness;
- the correct stock gate is the normal one-root frontend path;
- an integrated relayout+BMM artifact needs its own explicit ordering gate.

An adjacent `r65` static artifact adds the missing restickify `after_sync`, but
it has not been used here as a substitute for the normal stock control.

## Fresh stock control

Device: `adnan-spyre-current-pf` in namespace `a6-quantization`.

Pinned Torch-Spyre source:

```text
/home/adnan/codex-isolated/ring_matmul_true_os_torch_20260722_v1
Git HEAD: 80701411a151fa6402d08ce7586f671883e1e66b
```

Runtime extension:

```text
/home/adnan/spyre-envs/main-ac3c7395/torch-spyre/torch_spyre/_C.so
SHA-256: b681beeb640d5b8524dadfcc787d9ad2725db357adcf4e75f707d92d907487d4
```

Stock Deeptools source:

```text
/home/adnan/codex-isolated/deeptools-master-e3944781
Git HEAD: e3944781cb25b76abeb9b3e87c1f5c5879e84229
bmm.ddl SHA-256:
95db90a11bbce747e55e9e4b5ac4ab1a6034d1649e1d70d3aac4635e656f0268
```

Fresh result:

```text
/tmp/r64_normal_frontend_stock_unhinted_20260729_v2/summary.json
```

Passed gates:

- `correctness_gate=true`;
- `structural_gate=true`;
- `payload_gate=true`;
- one ordinary `batchmatmul`;
- unhinted work division emitted as stock `M4N8K1`.

Numerical results:

| Profile | Checked outputs | Tolerance mismatches | Max absolute error |
|---|---:|---:|---:|
| compile positive | 262,144 | 0 | 0.017578125 |
| dynamic poison A | 262,144 | 0 | 0.0146484375 |
| dynamic restore | 262,144 | 0 | 0.017578125 |
| warm positive | 262,144 | 0 | 0.017578125 |
| measured positive | 262,144 | 0 | 0.017578125 |

`trace_gate=false` in this run because the old harness was invoked with one
profiler iteration and then dereferenced an empty timing record. The complete
correctness/structure/payload report had already been written. This run is
accepted only as correctness and runtime evidence; it is not a latency result.

## Current custom candidate

Source:

```text
tmp/bmm_output_stationary.k32_pe_lx_carry.ddl
SHA-256:
06fa0e2f147fa906501cb2db84a7c0693949f7e70c566e89e0bf68cb82cd1792
```

Compiled artifact:

```text
/tmp/os_hmi2_k32_pe_lx_carry_mscope_20260729/run
```

Artifact hashes:

| Artifact | SHA-256 |
|---|---|
| `bundle.mlir` | `8f096dd0da33e3286ac26678488b686575b9cda4626e47aea556ed15c69ce6dd` |
| `spyreCodeDir/spyrecode.json` | `fc45f6dfd53448700f152b325b0172c9dce0e06d419492d50f530f52d1523078` |
| `spyreCodeDir/init_binary.bin` | `793756a448a5e33719454b678201215bde16ac8fc3e70053da714e3b210e1385` |
| `debug/sdsc_1/smc.txt` | `1ff84c05fee97351047b0cf0af93cfcc66a9e438d1ffd568f9926a5045f9794d` |

External ABI:

```text
(intermediate pool, A, scalar 0.5, W, C)
C = (A * 0.5) @ W
```

The bundle and external tensor/job ABI match the earlier split control:

- host program correction: 8,448 bytes at pointer `120259084288`;
- device jobs: pointers `120259092736` and `120259101952`;
- A: host `[64,2048]`, physical M64-stick layout;
- W: host `[2048,4096]`;
- C: host `[64,4096]`, device `[64,64,64]`.

### Intended K32 schedule

PT has 32 XRF entries. The candidate therefore:

1. preloads one K32 W epoch into XRF;
2. preloads the matching 32 A sticks into L0;
3. computes one K32-local C partial in PT ARF;
4. serializes all eight distinct PT-row M4 partials to PE;
5. for the first epoch, initializes PE with `partial * 1 + 0`;
6. for later epochs, reloads prior C from LX and computes
   `partial * 1 + prior_C`;
7. writes C through the stock PE -> SFP -> LX path after each epoch;
8. repeats for all 128 K32 epochs.

This avoids the illegal or deadlocking full-K PT recurrence while preserving
the desired output-stationary semantics.

### Compile-only structural evidence

The emitted audit covered all:

```text
32 cores x 2 corelets x 8 PT rows = 512 PT streams
```

It verified:

- every row forwards the upstream M strips without arithmetic mixing and
  appends its own M4 strip;
- terminal row 7 emits 32 distinct ordered M positions;
- 128 K32 epochs per PT stream;
- one zero-init plus 31 recurrent K steps per epoch;
- 4,096 XRF captures per PT stream;
- PT -> PE and PE -> SFP -> LX totals agree;
- one first block plus 63 LX reload/add carry blocks per N wave;
- LXSU signals and LXLU waits both equal 126 per core;
- no cross-row C reduction;
- external tensor/job ABI is unchanged.

This is emitted structural evidence, not numerical or runtime proof.

### Fresh device result

Completion harness:

```text
/tmp/k32_pe_lx_carry_validation_20260729
```

The corresponding durable local hash pins are:

| File | SHA-256 |
|---|---|
| `tmp/k32_pe_lx_carry_completion_manifest.json` | `9d0a27a9b38e83fc2e5ac8a1e8bef96f12ae7d82b39158d458a1e835e8435fe3` |
| `tmp/k32_pe_lx_carry_direct_launch_contract.json` | `e5a738f8081f4095a763757975b54f9019bc9bcce7f732bec0ee9bb45dae8279` |

The zero-input, no-D2H, no-timing launch completed import, initialization,
allocation, H2D, and submission. It then stopped at:

```text
device_synchronize: before
```

The 90-second hard timeout terminated it. At 60 seconds the runtime reported:

```text
in_flight_=1 device=0 - possible lost completion
```

The unchanged candidate therefore fails completion. Full correctness and
timing are forbidden.

Immediately afterward, the sealed stock completion launch passed:

```text
/tmp/r64_stock_recovery_after_k32_carry_20260729_v1
status=pass
```

This localizes the lost completion to the custom artifact rather than a wedged
device or a generally broken pinned runtime.

## Current blocker

The blocker is a candidate-specific scheduling deadlock despite mechanically
balanced aggregate counts. Aggregate equality is not sufficient: a cyclic
dependency can have equal producers and consumers while no unit can make the
first forward progress.

The highest-value next work is a single-variable completion bisection of the
PE/LX carry loop. Do not add more ring traffic until this HMI-local control
completes.

The bisection must distinguish:

1. PE waiting for PT row serialization;
2. LXLU waiting for an LXSU write that cannot retire;
3. LXSU waiting for PE/SFP output while PE waits for the LXLU reload;
4. an off-by-one first/recurrent epoch condition;
5. a fence placed at the wrong loop scope despite matching total counts.

### Strongest emitted topology discrepancy

The closest completing stock schedule exposes only the already-reduced
bottom-row result to PE:

```text
ptrow7 -> pe_ptsum
```

The candidate final SDSC instead exposes eight independently named transfers:

```text
ptrow0 -> pe_ptsum
...
ptrow7 -> pe_ptsum
```

Rows 0-6 traverse the downstream PT rows. The emitted bottom-row program
serializes upstream values and local `R0-R3`, but PE still has one `src0=pt`
consumer sequence. The aggregate loop counts can balance while the single
`pe_ptsum` port still has an unsupported eight-producer transaction topology.
The top-level JobPlan is identical to the completing control, so this
PT-row-to-PE topology is the strongest concrete candidate-specific mismatch.

Exact pod evidence:

```text
candidate:
/tmp/os_hmi2_k32_pe_lx_carry_mscope_20260729/run/debug/sdsc_1/sdsc_1.out.out.out.json
transfer region: lines 20786-22352

candidate SMC:
/tmp/os_hmi2_k32_pe_lx_carry_mscope_20260729/run/debug/sdsc_1/smc.txt
PE consumer: lines 477-483
bottom-row serialization: lines 927-942

completing stock:
/home/adnan/codex-isolated/r64_full8_stock_bmm_real_output_sync_device_20260725_v1/bundle/runtime/debug/sdsc_2/sdsc_2.out.out.out.out.json
single bottom-row transfer: lines 16775-16819
```

Do not declare it the root cause without a device bisection. The correct fix,
if confirmed, is to either:

- retain independent row-owned outputs on a native row/LX path; or
- explicitly serialize all eight row partials into one PE transaction with a
  matching temporal consumer.

Do not continue mapping eight independent row-owned tiles directly to
`pe_ptsum`.

### Smallest liveness discriminator: fence-off

Do not start by changing K depth or packet counts. The emitted counts already
balance on every corelet:

```text
PE output:       2 * (4 + 252) groups * 8 = 4096 vectors
SFP forwarding:                                4096 vectors
LXSU stores:     2 * (32 + 63*32)            = 4096 vectors
LXLU reloads:    2 * 63 * 4 * 8              = 4032 vectors
PE recurrent:    504 groups * 8               = 4032 vectors
```

The smallest discriminating arm removes only the explicit recurrent
LXSU-to-LXLU rendezvous at
`tmp/bmm_output_stationary.k32_pe_lx_carry.ddl` lines 134-141:

```ddl
ddl.if(%not_first_prefilled_w_block) {
  ddl.sync {units=["lxsu"], is_receive=false,
            signal_name="c-carry-lxsu-lxlu-sync",
            separate_corelets=true}
  ddl.sync {units=["lxlu"], is_receive=true,
            signal_name="c-carry-lxsu-lxlu-sync",
            separate_corelets=true}
}
```

Compile it as a new artifact; do not mutate the hash-pinned candidate. The
emitted fence-off gate is:

- remove LXLU sync tag 34 from both LXLUs on every core;
- remove LXSU sync tag 17 from both LXSUs on every core;
- preserve ordinary feed/completion tags 7 and 11;
- preserve every PT, PE, SFP, LXLU-load, LXSU-store instruction and loop count
  byte-for-byte otherwise.

Interpretation:

- completion means the explicit rendezvous causes the deadlock; a later
  numerical failure would then mean the fence was also providing necessary
  ordering;
- another deadlock rules the explicit fence out. The next priority is a
  completion-only control that replaces the eight named row-to-PE transfers
  with one explicitly serialized PE input. If the compiler cannot express
  that transaction, that missing row-local/serialized egress is the feature
  to implement; adding more syncs is not the answer.

An optional intermediate zero-feedback arm can remove the LX-to-PE reload and
use zero as the recurrent PE addend while retaining every forward write. It
separates feedback from the forward PE/SFP/LXSU path, but it does not clear the
eight-producer `pe_ptsum` topology and therefore cannot be the final fix.

Do not try K64: it exceeds the proven 32-entry implicit-L0/XRF epoch and
changes more than one variable.

## Acceptance ladder

Advance only in this order:

1. **Stock control**
   - normal frontend path;
   - full CPU comparison;
   - ring features disabled;
   - current stack identity recorded.
2. **Candidate completion**
   - zero inputs;
   - no D2H;
   - no profiler;
   - hard timeout;
   - must write a passing `result.json`.
3. **Candidate correctness**
   - positive;
   - poison A;
   - poison W;
   - cancellation-heavy full-K accumulation;
   - restore;
   - all 262,144 outputs checked for every case.
4. **RIU integration**
   - same compute schedule;
   - A starts in LX;
   - emitted attached route proves no standalone shuffle/HBM fallback;
   - same-grid ring-off control retained.
5. **SFP integration**
   - adjacent-pair W sharing;
   - both directions used when payload permits;
   - route/link accounting proves no redundant transit.
6. **Timing**
   - 10 warmups and 30 exact Kineto `cat=="kernel"` events;
   - serialized T-C-T-C or ABBA bracket;
   - incumbent, same-grid ring-off, RIU-only, and full dual-fabric arms;
   - no compile, materialization, or host wall time substituted for device
     kernel latency.

## Continuation commands

The candidate is already known to deadlock. Do not rerun it unchanged merely
to reproduce the same timeout. After changing exactly one scheduling variable,
recompile into a new path, update both hash-pinned manifests, and run completion
first.

The staged harness paths are:

```text
/tmp/k32_pe_lx_carry_validation_20260729/prebenchmark_m2n16_os/diagnostic_launch_only.py
/tmp/k32_pe_lx_carry_validation_20260729/prebenchmark_m2n16_os/direct_launch_matmul.py
/tmp/k32_pe_lx_carry_validation_20260729/stationary_weight_matmul_probe.py
```

The known-good runtime environment is:

```bash
export PYTHONPATH=/home/adnan/codex-isolated/ring_matmul_true_os_torch_20260722_v1:/home/adnan/spyre-perf-suite-envfix
export LD_LIBRARY_PATH=/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/spyre-comms/lib:/opt/ibm/spyre/deeptools/lib:/opt/ibm/spyre/senlib/lib:/home/adnan/dt-inductor/.venv/lib/python3.12/site-packages/torch/lib
export SENARCH=rcudd1a
export DXP_LX_FRAC_AVAIL=0.2
export DXP_BACKEND_LX_FRAC_AVAIL=0.2
export DT_OPT=psum=dataring
```

For a new candidate, use a fresh run directory and:

```bash
timeout --signal=TERM --kill-after=10s 90s \
  /home/adnan/dt-inductor/.venv/bin/python3 -u \
  /tmp/k32_pe_lx_carry_validation_20260729/prebenchmark_m2n16_os/diagnostic_launch_only.py \
  --manifest /path/to/new_completion_manifest.json \
  --run-dir /tmp/new_completion_run
```

Only if that returns zero and writes `status=pass`, run the full five-case
gate:

```bash
timeout --signal=TERM --kill-after=10s 300s \
  /home/adnan/dt-inductor/.venv/bin/python3 -u \
  /tmp/k32_pe_lx_carry_validation_20260729/prebenchmark_m2n16_os/direct_launch_matmul.py \
  --contract /path/to/new_direct_launch_contract.json \
  --code-dir /path/to/new_compiled_artifact \
  --run-dir /tmp/new_full_correctness_run
```

Do not pass `--case`, `--timing`, or `--discover-kernel-events` for the full
correctness gate.

## Runtime-path pitfall

The pod's inherited default `LD_LIBRARY_PATH` currently puts:

```text
/home/adnan/dt-inductor/sentient/runtime/lib
```

ahead of the installed `/opt/ibm/spyre` libraries. Under that inherited order,
the newer default dirty Torch-Spyre checkout at `cf67411d` fails during import
because `libspyre_comms.so.1` cannot resolve:

```text
flex::destroyP2PRdmaWaitParams
```

Do not interpret that import failure as a matmul failure. All accepted device
evidence in this handoff uses the explicit known-good `/opt/ibm/spyre` library
order above. The default `cf67411d` stack still needs a separately pinned
runtime/spyre-comms compatibility check before it is used for candidate
attribution.

## Stop conditions

Stop and reassess rather than adding complexity if either condition holds:

1. the smallest valid output-stationary control cannot complete after a
   bounded fence/carry bisection;
2. the completing/correct HMI-local control is already slower than the
   identical-grid stock compute control by more than the maximum modeled RIU
   materialization saving.

Conversely, if the HMI-local control completes and is competitive, the next
high-value step is not another PT micro-optimization. It is attaching RIU A
delivery to the same schedule, then adding SFP pair sharing and measuring the
two fabrics as one overlapped algorithm.
