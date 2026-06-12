# Exercise Track: TLX Basics

**Module:** TLX | **Phase:** Low-Level Triton Extensions | **GPU:** Hopper+

## Objective

Learn TLX from the smallest useful surface upward: first warp-specialized task
regions, then explicit local-memory staging, then barriers and TMA-driven
producer/consumer kernels.

## Study Material

Before starting, read:

- `compiled_resources/TLX/README.md` - local index for the imported TLX source
- `compiled_resources/TLX/triton/README.md` - TLX API overview
- `compiled_resources/TLX/triton/third_party/tlx/tutorials/vector-add2.py` - minimal warp-specialization example
- `compiled_resources/TLX/triton/third_party/tlx/tutorials/hopper_gemm_pipelined.py` - first serious Hopper GEMM example
- `compiled_resources/TLX/triton/third_party/tlx/doc/tlx_barriers.md` - barrier model for later exercises

## Exercise 0: Vector Add With Async Tasks

**Status:** prepared now.

Start from `exercises/tlx/exercise.py`.

You will build three paths:

1. `torch`: PyTorch reference for two independent vector additions
2. `triton`: normal Triton kernel that performs both additions in one program
3. `triton_ws`: TLX warp-specialized kernel that splits the additions across
   `tlx.async_task` regions

### Acceptance Criteria

1. Correctness passes against PyTorch for both output vectors.
2. `exercise.py` contains the implementation; `compiled_resources/` remains
   read-only reference material.
3. `benchmark.py` produces `plots/vector-add-performance.png` after correctness
   passes.
4. The benchmark table shows all three providers: `Triton`, `Triton_WS`, and
   `Torch`.

### Commands

Correctness:

```bash
source .venv/bin/activate && python exercises/tlx/exercise.py
```

Benchmark locally, if the active machine has a Hopper+ GPU:

```bash
source .venv/bin/activate && python exercises/tlx/benchmark.py
```

Benchmark through Modal H100:

```bash
source .venv/bin/activate && modal run run_modal.py --action run --script exercises/tlx/benchmark.py
```

## Exercise 1: Pipelined Hopper GEMM

**Status:** roadmap only. Do not start until Exercise 0 is completed and
documented.

Study `compiled_resources/TLX/triton/third_party/tlx/tutorials/hopper_gemm_pipelined.py`.

Focus on these concepts before editing anything:

- `tlx.local_alloc` and `tlx.local_view`
- `tlx.async_load`
- `tlx.async_load_commit_group`
- `tlx.async_load_wait_group`
- `tlx.async_dot`
- `tlx.async_dot_wait`

The goal is to explain the pipeline from global memory to SMEM to tensor cores
without involving warp-specialized producers yet.

## Exercise 2: Warp-Specialized Hopper Kernels

**Status:** prepared as a worksheet module.

Start from:

- `exercises/tlx_hopper_ws/README.md`
- `exercises/tlx_hopper_ws/exercise.py`

This module studies `compiled_resources/TLX/triton/third_party/tlx/tutorials/hopper_gemm_ws.py`
as a producer/consumer state machine before any full rewrite. Keep the first
pass fixed to `NUM_CTAS=1`, `USE_WARP_BARRIER=False`, and
`EPILOGUE_SUBTILE=False`; the advanced variants are follow-up exercises.

## Reporting

Write run results under `benchmark_results/` using the existing report template.
For Exercise 0, fill in `benchmark_results/tlx-vector-add-2026-05-27.md` after
your benchmark run completes.

## Next Module: TLX-B Profiling

After Exercise 0 correctness works, move to `exercises/tlx_b/README.md`.
That module is focused only on profiling the completed TLX dual-add kernel with
Proton intra-kernel traces. It keeps profiling notes and scaffolds out of this
exercise so this file can stay focused on implementation and benchmarking.
