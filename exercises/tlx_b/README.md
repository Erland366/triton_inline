# TLX-B: Intra-Kernel Profiling

**Module:** TLX-B | **Focus:** Proton traces for TLX async-task kernels |
**GPU:** Hopper+

This module is separate from `exercises/tlx/` on purpose. Exercise 0 teaches
how to write the TLX kernel. This module teaches how to profile the completed
kernel without turning the exercise file into a profiling sandbox.

The target is the same kind of view as an "intra-kernel profile": a timeline
showing named regions inside one GPU kernel, split by profiled warp or warp
group.

## What You Are Building

You will create a profiled variant of your Exercise 0 TLX dual-add kernel:

- one Proton scope around the whole kernel
- one Proton scope inside the `tlx.async_task("default")` body
- one Proton scope inside the non-default `tlx.async_task(...)` body
- a host-side Proton instrumentation session that writes a `.chrome_trace`
  file for Perfetto

The profiling script is:

```bash
exercises/tlx_b/profile_vector_add2.py
```

That file is intentionally a scaffold. Copy the body of your completed TLX
kernel from `exercises/tlx/exercise.py`, then add the profiling scopes where
the TODOs ask for them. Do not edit anything under `compiled_resources/`.

## Why Proton, Not Torch Profiler

`torch.profiler` can tell you that a kernel launch happened and how long it
took. It cannot show which part of the kernel was running on the default task
versus the specialized task.

For this module, use Proton's instrumentation backend:

```python
proton.start(
    "tlx-add2",
    data="trace",
    backend="instrumentation",
    mode=mode,
)
```

`data="trace"` writes a Chrome trace file, normally:

```text
tlx-add2.chrome_trace
```

Open that file in:

```text
https://ui.perfetto.dev/
```

## Important Constraints

### Triton DSL instrumentation must be enabled

Proton enables Gluon kernel instrumentation by default. For Triton/TLX kernels,
you must explicitly call:

```python
pl.enable_semantic("triton")
```

Do that before `proton.start(...)`.

### Keep scopes coarse at first

Start with only these scopes:

```python
pl.enter_scope("add2_ws_kernel")

with pl.scope("default_task_x_plus_y"):
    ...

with pl.scope("specialized_task_a_plus_b"):
    ...

pl.exit_scope("add2_ws_kernel")
```

Do not put Proton scopes inside a loop until the coarse trace works. Too many
events can overflow the trace buffer and make the first profile hard to read.

### Profile one launch first

This is not a benchmark. Run one kernel launch first, inspect the trace, then
increase size or repetitions only after you understand the output.

## Step-by-Step

### 1. Finish Exercise 0

Run:

```bash
source .venv/bin/activate && python exercises/tlx/exercise.py
```

Do not profile a kernel that is not correct yet.

### 2. Fill the profiling scaffold

Open:

```bash
exercises/tlx_b/profile_vector_add2.py
```

Find `PROFILE_KERNEL_READY = False`.

Then:

1. Copy your completed `add2_warp_specialized_kernel` indexing and async-task
   structure into the profiled kernel.
2. Put the first addition inside `with pl.scope("default_task_x_plus_y")`.
3. Put the second addition inside `with pl.scope("specialized_task_a_plus_b")`.
4. Keep the same launch signature and output order.
5. Change `PROFILE_KERNEL_READY = True`.

The profile script will fail loudly until that flag is changed.

### 3. Run on a local Hopper+ GPU

If the current machine has a Hopper+ GPU:

```bash
source .venv/bin/activate && python exercises/tlx_b/profile_vector_add2.py
```

Expected artifact:

```text
tlx-add2.chrome_trace
```

### 4. Run on Modal H100

Use the same Modal runner as the plot workflow:

```bash
source .venv/bin/activate && modal run run_modal.py --action run --script exercises/tlx_b/profile_vector_add2.py
```

The runner now collects these artifacts by default:

```text
plots/vector-add-performance.png
*.chrome_trace
*.hatchet
```

If the profile script writes `tlx-add2.chrome_trace`, the local runner saves it
back to:

```text
tlx-add2.chrome_trace
```

Open it in Perfetto.

### 5. If the trace is too large

Use warp sampling:

```bash
source .venv/bin/activate && python exercises/tlx_b/profile_vector_add2.py --warp-ids 0,4,8
```

On Modal:

```bash
source .venv/bin/activate && modal run run_modal.py --action run --script "exercises/tlx_b/profile_vector_add2.py --warp-ids 0,4,8"
```

Interpretation of the warp IDs is empirical at this stage:

- `0` usually samples a trunk/default warp
- `4` is a useful first probe for a non-default task
- `8` is useful when the non-default task is replicated

If you later use explicit `warp_group_start_id`, update the sampled IDs to match
the assignments you requested.

### 6. Generate a tree profile instead of a timeline

Timeline mode is the default. To get operation measurements:

```bash
source .venv/bin/activate && python exercises/tlx_b/profile_vector_add2.py --op-measure
```

Expected artifact:

```text
tlx-add2.hatchet
```

View it with:

```bash
source .venv/bin/activate && proton-viewer -m normalized_cycles tlx-add2.hatchet
```

Use the trace first. Use the Hatchet tree after you know the scopes are
recording the regions you meant to record.

## Modal Artifact Collection

`run_modal.py` now has a generic artifact collector. The default is:

```text
plots/vector-add-performance.png,*.chrome_trace,*.hatchet
```

For a custom output directory, pass a comma-separated glob list:

```bash
source .venv/bin/activate && modal run run_modal.py \
  --action run \
  --script exercises/tlx_b/profile_vector_add2.py \
  --artifact-globs "*.chrome_trace,*.hatchet,profiles/*.chrome_trace"
```

Artifacts are saved locally with the same relative path used on the remote
machine. Large files are skipped loudly with an `artifact_errors` entry. Raise
the cap if needed:

```bash
source .venv/bin/activate && modal run run_modal.py \
  --action run \
  --script exercises/tlx_b/profile_vector_add2.py \
  --max-artifact-bytes 200000000
```

## What To Look For In Perfetto

Start by checking these questions:

1. Do you see `default_task_x_plus_y`?
2. Do you see `specialized_task_a_plus_b`?
3. Are they on different profiled warp lanes?
4. Does one region start much earlier than the other?
5. Does one region disappear when you sample a different warp ID?

For this toy vector-add kernel, do not expect a performance win. The useful
learning signal is whether the task split is visible and whether your mental
model of warp assignment matches the trace.

## When Proton Is Not Enough

If Proton scopes are moved by compiler rewrites or the trace is too coarse, the
next step is manual instrumentation:

- read `tlx.clock64()` inside the kernel
- write compact event records to a global tensor
- copy the tensor back to CPU
- emit Chrome trace JSON yourself

Do not start there. Proton is the lower-friction first pass.
