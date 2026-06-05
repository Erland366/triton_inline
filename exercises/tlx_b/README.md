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

pl.enter_scope("default_task_x_plus_y")
...
pl.exit_scope("default_task_x_plus_y")

pl.enter_scope("specialized_task_a_plus_b")
...
pl.exit_scope("specialized_task_a_plus_b")

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
2. Put the first addition between `pl.enter_scope("default_task_x_plus_y")`
   and `pl.exit_scope("default_task_x_plus_y")`.
3. Put the second addition between
   `pl.enter_scope("specialized_task_a_plus_b")` and
   `pl.exit_scope("specialized_task_a_plus_b")`.
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
**/*.png
**/*.chrome_trace
**/*.hatchet
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
source .venv/bin/activate && python exercises/tlx_b/profile_vector_add2.py --warp-ids 0,4,8,11
```

On Modal:

```bash
source .venv/bin/activate && modal run run_modal.py --action run --script "exercises/tlx_b/profile_vector_add2.py --warp-ids 0,4,8,11"
```

Interpretation of the warp IDs is empirical at this stage:

- `0` usually samples a trunk/default warp
- `4` is a useful first probe for a non-default task
- use a power-of-two number of sampled IDs, such as `0,4` or `0,4,8,11`

Proton's current instrumentation pass requires the number of selected warp IDs
to be a power of two. A list like `0,4,8` fails in `ConvertProtonToProtonGPU`
because it creates three profiling segments.

The profiling script prints the effective warp IDs before launching the kernel.
Check that stdout line first when the trace does not show the lanes you expect.

If you later use explicit `warp_group_start_id`, update the sampled IDs to match
the assignments you requested.

### 6. Why Not `--granularity warp_group`

The Proton Python mode exposes `warp_group`, but this TLX/NVIDIA lowering path
currently fails later with:

```text
granularity must be warp for now
```

So this module intentionally uses:

```text
--granularity warp
```

and samples representative warp IDs instead. For this toy kernel, use:

```text
--warp-ids 0,4,8,11
```

That gives you the default task on warp 0 and sampled specialized-task warps
from the replicated async task.

You may also see this lower-level error while experimenting:

```text
buffer-size per segment(12) must be power of 2
```

This TLX toy kernel has 12 total warps: 4 default/trunk warps plus two
specialized replicas of 4 warps each. With all-warps profiling, Proton splits a
nonzero buffer size across 12 profiling segments and requires each segment to be
a power of two. For example, this is invalid:

```text
--granularity warp_group --all-warps --buffer-size 4096
```

because `4096 / 12` is not a power of two. Prefer the script default:

```text
--buffer-size 0
```

which lets Proton choose its default global buffer size. If you need an explicit
size, use `12 * power_of_two`, for example:

```text
--buffer-size 49152
```

This buffer-size fix does not make `warp_group` work on the current NVIDIA
lowering path; it only avoids the earlier buffer-size error.

### 7. Generate a tree profile instead of a timeline

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
**/*.png,**/*.chrome_trace,**/*.hatchet
```

Only artifacts created or changed during the remote run are returned. That keeps
stale files uploaded from the local project from being copied back by accident.

For a custom output pattern, pass a comma-separated glob list:

```bash
source .venv/bin/activate && modal run run_modal.py \
  --action run \
  --script exercises/tlx_b/profile_vector_add2.py \
  --artifact-globs "**/*.chrome_trace,**/*.hatchet,profiles/*.chrome_trace"
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
