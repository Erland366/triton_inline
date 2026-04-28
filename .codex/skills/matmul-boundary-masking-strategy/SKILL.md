---
name: matmul-boundary-masking-strategy
description: >
  Three-mechanism boundary handling for tiled GPU kernels: M/N offset wrapping
  keeps masks out of the inner loop, K load masks with other=0.0, store mask
  for final write. Use when: writing tiled matmul or GEMM kernels in Triton.
metadata:
  short-description: "Move boundary checks out of the inner loop"
  tags:
    - triton
    - matmul
    - boundary-handling
    - performance
  domain: cuda
  created: 2026-02-13
  author: triton-teacher
---

# Boundary Masking Strategy for Tiled Kernels

## General Description

In tiled GPU kernels (matmul, GEMM, attention), boundary handling for
non-aligned dimensions is a critical performance concern. The naive approach
--- masking every load inside the inner loop --- adds branch overhead to the
hottest code path. The optimized approach separates boundary handling into
three mechanisms across three dimensions, keeping the inner K-loop mask-free
for M and N.

## When to Apply

Use this knowledge when:
- Writing a tiled matmul or GEMM kernel in Triton
- Implementing any kernel with a 2D output tile and an inner reduction loop
- Optimizing boundary handling for non-power-of-2 dimensions
- Working with `tl.dot` which requires zero-padding for correctness

Do NOT use when:
- Kernel has no inner loop (e.g., elementwise ops) — use direct `mask=offs < N` instead;
  wrapping adds complexity (two offset sets) with zero performance gain when there's
  no hot loop to protect
- All dimensions are guaranteed to be aligned to block sizes
- Using `tl.make_block_ptr` with `boundary_check` (handles this automatically)

**Why not for non-tiled kernels?** In a non-tiled kernel (vector add, softmax),
load and store happen at the same level — no loop. Both wrapping and direct masking
execute once, so they have identical cost. Direct masking is simpler:

```python
# Non-tiled: direct masking — simple, runs once
x = tl.load(x_ptr + offs, mask=offs < N, other=0.0)
tl.store(out_ptr + offs, y, mask=offs < N)

# Non-tiled with wrapping — same cost, more complexity, still needs store mask
offs_wrapped = (pid * BLOCK + tl.arange(0, BLOCK)) % N
x = tl.load(x_ptr + offs_wrapped)
offs_real = pid * BLOCK + tl.arange(0, BLOCK)
tl.store(out_ptr + offs_real, y, mask=offs_real < N)
```

**The rule:** Wrapping pays off only when it removes masks from an inner loop
that runs hundreds of iterations.

## Results Summary

| Mechanism | Dimension | Where | Cost |
|-----------|-----------|-------|------|
| Offset wrapping (`% M`, `% N`) | M, N | Pointer setup (before loop) | Zero inner-loop cost |
| Load mask (`offs_k < K - k * BLOCK_K`) | K | Inside inner loop | Per-iteration branch |
| Store mask (`offs < M` and `offs < N`) | M, N | After loop (store) | Once at the end |

## Recommended Practice

### Step 1: Wrap M/N offsets at pointer creation

Use modular arithmetic to wrap out-of-bounds M/N indices back into valid
memory. This happens once, before the loop, so it costs nothing per iteration.

```python
# Offsets wrap around --- out-of-bounds indices read valid (but irrelevant) memory
offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
```

The wrapped values read "garbage" from valid addresses, but it doesn't matter
because the store mask (Step 3) prevents writing those positions.

### Step 2: Mask only K inside the inner loop

K is the only dimension that changes each iteration, so it's the only one
that needs a per-iteration mask. Use `other=0.0` so out-of-bounds loads
contribute zero to the dot product.

```python
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
    accumulator = tl.dot(a, b, accumulator)
    a_ptrs += BLOCK_SIZE_K * stride_ak
    b_ptrs += BLOCK_SIZE_K * stride_bk
```

### Step 3: Apply a strict store mask for M/N

After the loop, use un-wrapped offsets to create a boolean mask that prevents
writing to out-of-bounds output positions.

```python
# NO modular wrapping here --- these are the real positions
offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
tl.store(c_ptrs, c, mask=c_mask)
```

## Failure Modes

| What Failed | Why | Lesson |
|-------------|-----|--------|
| Masking M/N inside the inner loop | Adds per-iteration branch overhead; K iterates hundreds of times | Move M/N checks out of the loop |
| No `other=0.0` on K mask | Garbage values enter `tl.dot`, corrupt accumulator | Always zero-fill masked loads in reduction loops |
| Using `% M` on store offsets | Wraps out-of-bounds stores to valid positions, overwrites correct results | Store offsets must be un-wrapped with explicit `< M` mask |
| Skipping store mask entirely | Works for aligned dims, silently corrupts for non-aligned | Always include store mask for correctness |

## Configuration

```python
# Pattern: 3-mechanism boundary handling for tiled matmul
# 1. Pointer setup --- wrap M/N
offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

# 2. Inner loop --- mask only K
mask_k = offs_k[None, :] < K - k * BLOCK_K  # for A loads
mask_k = offs_k[:, None] < K - k * BLOCK_K  # for B loads

# 3. Store --- strict M/N mask, NO wrapping
offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # no % M
offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # no % N
c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
```

## Debugging: Is It a Bug or Precision?

When a kernel test fails after implementing boundary masking, determine whether
it's a logic bug or FP16 numerical precision before changing the kernel.

### Fastest test: run in FP32

Re-run with FP32 inputs and accumulation. If it passes in FP32 but fails in FP16,
the kernel logic is correct and the issue is FP16 rounding.

### Error pattern diagnosis

| Pattern | Cause |
|---------|-------|
| Max diff is a power of 2 (e.g., 0.0625 = 2^-4), mean diff is tiny | FP16 ULP rounding — not a bug |
| Errors concentrated at matrix edges | Boundary handling bug |
| Errors in stripes or periodic | Stride bug |
| Max diff is large and chaotic, most elements wrong | Logic bug |

### Why correct kernels disagree with cuBLAS

FP addition is not associative: `(a + b) + c != a + (b + c)`. Different tiling
means different K-dimension summation order → different FP32 rounding → different
FP16 output at rounding boundaries. Expected for `torch.randn` inputs at sizes
>=1024 where outputs reach magnitudes where FP16 ULP > test tolerance.

### Tolerance guidance for FP16 matmul tests

- `atol=1e-2, rtol=0`: only works for small inputs/outputs (tutorial-style)
- `atol=5e-2, rtol=1e-2`: robust for `torch.randn` inputs at any matrix size

## References

- Study notes: `exercises/1.1/study-notes.md` section 6 (Boundary Handling)
- Tutorial source: `~/dotfiles/compiled_resources/triton_learning/tutorials/03-matrix-multiplication.py`
- Modern alternative: `tl.make_block_ptr` with `boundary_check` parameter (handles all three automatically)
