# Exercise: Matmul Optimization

**Module:** 1.1 | **Phase:** Core Kernel Mastery | **GPU:** Ampere+

## Objective

Write a high-performance FP16 tiled matrix multiplication kernel in Triton from scratch.
You will implement the block-tiled algorithm with L2 cache-friendly super-grouping,
autotuning across multiple configurations, and correct boundary masking for arbitrary
matrix dimensions.

## Study Material

Before starting, read:
- `tutorials/03-matrix-multiplication.py` — core algorithm and patterns
- `tutorials/09-persistent-matmul.py` — modernized boundary handling and `tl.assume`
- `exercises/1.1/study-notes.md` — your study notes from the study phase

All tutorial paths relative to `~/dotfiles/compiled_resources/triton_learning/`.

## What to Implement

1. Define at least 4 `triton.Config` autotuning configurations varying `BLOCK_SIZE_M`,
   `BLOCK_SIZE_N`, `BLOCK_SIZE_K`, `num_stages`, and `num_warps`
2. Write the kernel with L2 cache-friendly super-grouping (`GROUP_SIZE_M`)
3. Implement multi-dimensional pointer arithmetic for loading A and B tiles
4. Implement the K-dimension accumulation loop with FP32 accumulator
5. Handle boundary conditions with proper masking for non-aligned M, N, K
6. Write the launch wrapper that allocates output and computes the grid

## Acceptance Criteria

1. Numerically correct: `atol=1e-2` vs `torch.matmul` for FP16
2. Autotuning selects config automatically via `@triton.autotune`
3. Benchmarked across M=N=K in [1024, 2048, 4096, 8192]

## Benchmark Target

> **>80% of cuBLAS throughput at M=N=K=4096**

## Bridge to Production

Production matmul in `triton_kernels/matmul.py` uses closure-based specialization
(`specialize.py`) instead of `@triton.autotune`. Study how `_matmul.py` handles
the same tiling but with fused epilogues and dynamic dispatch.

## Files

| File | Purpose |
|------|---------|
| `exercise.py` | Skeleton — implement the TODOs |
| `benchmark.py` | Performance benchmark — run after correctness passes |
| `study-notes.md` | Study notes from the study phase |
| `solution_notes.md` | Your reflection notes (created after completion) |
