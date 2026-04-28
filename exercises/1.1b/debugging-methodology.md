# Debugging Methodology

> Permanent reference for diagnosing correctness and precision issues in Triton kernels.
> Used during Module 1.1b study phase and referenced in all subsequent modules.

---

## Diagnostic Flowchart

When a kernel produces incorrect results, follow this sequence:

```
1. FP32 Verification
   Run kernel with FP32 inputs + FP32 accumulation.
   ├── PASS → Precision issue (go to §Expected Tolerances)
   └── FAIL → Logic bug (go to §Isolation Testing)
   └── N/A  → Skip if kernel uses tl.dot on Ampere (see §Ampere tl.dot Caveat)

2. Error Pattern Analysis (§Error Patterns)
   Examine the error tensor: ref - yours
   ├── Uniform small error everywhere → Precision (accumulation order, dtype)
   ├── Errors at boundaries only → Masking bug (boundary condition)
   ├── Striped/periodic errors → Stride bug (pointer arithmetic)
   └── Large errors in specific tiles → Tile logic bug (go to §Isolation Testing)

3. Isolation Testing (§Isolation Matrix)
   Narrow down the failure with controlled inputs.

4. Fix and Regression Test
   Fix the bug, add the failing case to your test suite.
```

---

## Expected Tolerances by Dtype

These are **expected** tolerances for correctly implemented kernels. If your kernel
exceeds these, you have a bug — not a precision problem.

### Matmul (C = A @ B)

| Dtype (A, B → Acc → Output) | atol | rtol | Notes |
|------------------------------|------|------|-------|
| FP32 → FP32 → FP32 | 1e-5 | 1e-5 | Near-exact |
| FP16 → FP32 → FP16 | 1e-2 | 1e-2 | Standard Triton matmul |
| BF16 → FP32 → BF16 | 1e-1 | 1e-1 | BF16 has 8-bit mantissa |
| FP8e4 → FP32 → FP16 | 1e-1 | 5e-2 | Wide tolerance expected |
| FP8e5 → FP32 → FP16 | 5e-1 | 1e-1 | Wider dynamic range, less precision |

### Reductions (softmax, layernorm, sum)

| Dtype | atol | rtol | Notes |
|-------|------|------|-------|
| FP32 | 1e-5 | 1e-5 | Near-exact |
| FP16 | 1e-3 | 1e-3 | Accumulation order matters |
| BF16 | 5e-3 | 5e-3 | Reduction over large rows amplifies error |

### Flash Attention

| Dtype | atol | rtol | Notes |
|-------|------|------|-------|
| FP16 → FP32 acc | 1e-2 | 1e-2 | Online softmax introduces rounding |
| BF16 → FP32 acc | 5e-2 | 5e-2 | Acceptable for training |
| FP8 | 5e-1 | 1e-1 | Inference-only use case |

### General Rule

If max error is **10x above the expected atol**, you have a logic bug, not a precision issue.

---

## Tolerance Sweep Technique

When you're unsure if an error is precision or a bug, sweep tolerances systematically:

```python
import torch

def tolerance_sweep(ref, actual, name="kernel"):
    """Sweep tolerances to characterize error distribution."""
    diff = torch.abs(ref - actual)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    std_diff = diff.std().item()

    print(f"--- {name} Error Analysis ---")
    print(f"  Max:  {max_diff:.6e}")
    print(f"  Mean: {mean_diff:.6e}")
    print(f"  Std:  {std_diff:.6e}")
    print()

    # Sweep atol to find the passing threshold
    for atol in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 5e-1, 1.0]:
        matches = torch.isclose(ref, actual, atol=atol, rtol=0).float().mean()
        status = "PASS" if matches == 1.0 else f"{matches*100:.1f}%"
        print(f"  atol={atol:.0e}: {status}")

    # Check error spatial distribution
    print()
    error_mask = diff > mean_diff + 3 * std_diff  # outliers
    n_outliers = error_mask.sum().item()
    n_total = diff.numel()
    print(f"  Outliers (>3σ): {n_outliers}/{n_total} ({n_outliers/n_total*100:.2f}%)")

    if n_outliers > 0 and diff.dim() >= 2:
        # Show where outliers cluster
        rows_with_errors = error_mask.any(dim=-1).sum().item()
        total_rows = error_mask.shape[0]
        print(f"  Rows with outliers: {rows_with_errors}/{total_rows}")
```

**How to interpret:**
- **Gradual falloff** (e.g., 95% pass at 1e-3, 100% at 1e-2): precision issue, expected behavior
- **Sharp cliff** (e.g., 60% pass at 1e-1, 100% at 1.0): bug affecting subset of elements
- **Outliers concentrated in specific rows/tiles**: masking or stride bug at boundaries

---

## Error Patterns

| Pattern | Visual (error heatmap) | Likely Cause | Where to Look |
|---------|----------------------|--------------|---------------|
| Uniform small error | Even noise across output | Accumulation precision | Accumulator dtype, reduction order |
| Boundary errors | Errors only at M/N/K edges | Masking bug | `tl.load(..., mask=..., other=0.0)` |
| Striped/periodic errors | Regular pattern every BLOCK_SIZE | Stride bug | Pointer arithmetic, `stride_*` params |
| Block-aligned errors | Entire tiles wrong | Tile index bug | `pid_m`, `pid_n` computation, super-grouping |
| Last-tile errors | Only final block row/col wrong | Off-by-one in grid | Grid size rounding, `tl.cdiv` |
| Random scattered errors | No spatial pattern | Race condition | Atomic operations, shared memory sync |
| All zeros in output | Entire output is zero | Kernel not writing | Store mask, output pointer offsets |
| Correct shape, wrong values | Consistent wrong answer | Algorithm bug | Core computation logic |

### Visualizing Error Patterns

```python
import torch
import matplotlib.pyplot as plt

def visualize_errors(ref, actual, title="Error Heatmap"):
    """Visualize spatial distribution of errors."""
    diff = torch.abs(ref - actual).cpu().float()
    if diff.dim() > 2:
        diff = diff.reshape(-1, diff.shape[-1])

    plt.figure(figsize=(10, 6))
    plt.imshow(diff.numpy(), aspect='auto', cmap='hot')
    plt.colorbar(label='|ref - actual|')
    plt.title(title)
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.savefig('error_heatmap.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved error heatmap to error_heatmap.png")
```

---

## Isolation Testing Matrix

Use controlled inputs to isolate the failure mode. Run each test case independently.

### Test Cases

| Test Case | Input | Purpose | What It Catches |
|-----------|-------|---------|-----------------|
| **Identity** | A=I, B=X (or X) | Verify pass-through | Basic pointer/store bugs |
| **Single tile** | M=N=K=BLOCK_SIZE | One tile, no boundaries | Core algorithm correctness |
| **Boundary + 1** | M=BLOCK_M+1, N=BLOCK_N+1 | Two tiles with partial last tile | Boundary masking |
| **K not aligned** | K not divisible by BLOCK_K | Partial K tile | K-loop masking |
| **Single row** | M=1, N=large | Degenerate dimension | Row pointer arithmetic |
| **Single column** | M=large, N=1 | Degenerate dimension | Column pointer arithmetic |
| **Power of 2** | M=N=K=1024 | Aligned, no boundaries | Baseline (should always pass) |
| **Prime sizes** | M=127, N=131, K=137 | Maximum boundary stress | All masking paths |
| **Large K** | M=N=64, K=8192 | Many K iterations | Accumulation correctness |

### Implementation Pattern

```python
def isolation_test(matmul_fn, reference_fn, atol=1e-2):
    """Run isolation test matrix for a matmul kernel."""
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32  # match your kernel config

    test_cases = [
        ("identity",      BLOCK_M,     BLOCK_N,     BLOCK_K,     "identity"),
        ("single_tile",   BLOCK_M,     BLOCK_N,     BLOCK_K,     "random"),
        ("boundary+1",    BLOCK_M + 1, BLOCK_N + 1, BLOCK_K,     "random"),
        ("K_not_aligned", BLOCK_M,     BLOCK_N,     BLOCK_K + 7, "random"),
        ("single_row",    1,           BLOCK_N * 4, BLOCK_K,     "random"),
        ("single_col",    BLOCK_M * 4, 1,           BLOCK_K,     "random"),
        ("power_of_2",    1024,        1024,        1024,        "random"),
        ("prime_sizes",   127,         131,         137,         "random"),
        ("large_K",       64,          64,          8192,        "random"),
    ]

    for name, M, N, K, mode in test_cases:
        if mode == "identity":
            A = torch.eye(M, K, device="cuda", dtype=torch.float16)
            B = torch.randn(K, N, device="cuda", dtype=torch.float16)
        else:
            A = torch.randn(M, K, device="cuda", dtype=torch.float16)
            B = torch.randn(K, N, device="cuda", dtype=torch.float16)

        ref = reference_fn(A, B)
        out = matmul_fn(A, B)

        if torch.allclose(ref, out, atol=atol, rtol=0):
            print(f"  [PASS] {name:20s}  M={M:5d} N={N:5d} K={K:5d}")
        else:
            diff = torch.abs(ref - out)
            print(f"  [FAIL] {name:20s}  M={M:5d} N={N:5d} K={K:5d}"
                  f"  max_err={diff.max().item():.4e}")
```

---

## triton-viz Tools

**Requires:** `pip install triton-viz` — runs on CPU via Triton interpreter (no GPU needed)

### triton-viz Sanitizer

Symbolic OOB (out-of-bounds) memory access detection using the Z3 solver. Catches
bugs **before** they produce wrong results — detects when a load or store would
access memory outside tensor bounds.

```python
import triton_viz
from triton_viz.clients import Sanitizer

@triton_viz.trace(client=Sanitizer(abort_on_error=True))
@triton.jit
def my_kernel(X_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Bug: no mask — will OOB when N is not divisible by BLOCK_SIZE
    x = tl.load(X_ptr + offs)
    tl.store(X_ptr + offs, x * 2)

# Sanitizer halts here with detailed OOB report
my_kernel[(4,)](x, N=100, BLOCK_SIZE=32)
```

Output includes:
- Exact operation that triggered OOB (load or store)
- Source file and line number
- Symbolic expression of the offending offset
- Tensor metadata (shape, dtype, base pointer)

**CLI wrapper** (patches ALL `@triton.jit` kernels in a script):
```bash
triton-sanitizer my_script.py
```

**When to use the sanitizer:**
- After writing a new kernel, before correctness testing
- When you suspect a masking bug (boundary conditions)
- When pointer arithmetic is complex (multi-dimensional, strided)
- Insert into the diagnostic flowchart between Step 1 (FP32 verification) and
  Step 2 (error pattern analysis) — OOB access is a common root cause

**Limitations:**
- Slow (symbolic execution via Z3) — don't use on large inputs
- Only detects OOB, not logic bugs (wrong values from valid addresses)

### triton-viz Visualizer

Interactive web UI that shows memory access patterns per operation. Useful for
**understanding** how your kernel accesses memory, not just debugging.

```python
import triton_viz
from triton_viz.clients import Tracer

@triton_viz.trace(client=Tracer())
@triton.jit
def my_kernel(...):
    ...

my_kernel[grid](...)
triton_viz.launch(share=False, port=5001)
```

Shows:
- Which memory locations each `tl.load` and `tl.store` accesses
- Mask patterns (which elements are masked out)
- `tl.dot` operand shapes and access patterns
- `tl.sum`/`tl.max` reduction patterns

**When to use the visualizer:**
- During study phases to see access patterns (e.g., matmul tiling)
- When you suspect coalescing issues (scattered vs contiguous access)
- To verify mask correctness visually (are the right elements masked?)

### triton-viz Profiler

See `profiling-guide.md` §triton-viz for mask efficiency and loop unrolling checks.

---

## Triton-Specific Debugging Tools

### `tl.device_print`

Print values from inside a Triton kernel. Limited to scalar values per thread.

```python
@triton.jit
def debug_kernel(X_ptr, N: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * N + tl.arange(0, N)

    # Print the program ID (scalar)
    tl.device_print("pid", pid)

    x = tl.load(X_ptr + offs)

    # Print first element of loaded block (reduce to scalar first)
    first_elem = tl.sum(x * (tl.arange(0, N) == 0).to(tl.float32))
    tl.device_print("first_x", first_elem)
```

**Limitations:**
- Only prints scalars (not tensors/blocks)
- Output is unordered across program instances
- Slows execution significantly — use only for debugging
- Filter by `pid == 0` to limit output volume

**Filtering pattern:**

```python
@triton.jit
def debug_kernel(X_ptr, ...):
    pid = tl.program_id(0)
    # Only print from program 0
    if pid == 0:
        tl.device_print("value", some_scalar)
```

### Step-by-Step NumPy Comparison

The most reliable debugging technique: rewrite the kernel logic in NumPy/PyTorch
and compare intermediate values tile-by-tile.

```python
def debug_matmul_tile(A, B, pid_m, pid_n, BLOCK_M, BLOCK_N, BLOCK_K):
    """Simulate one tile of the matmul kernel in PyTorch for comparison."""
    M, K = A.shape
    K2, N = B.shape

    # Replicate the kernel's pointer arithmetic
    offs_m = pid_m * BLOCK_M + torch.arange(BLOCK_M)
    offs_n = pid_n * BLOCK_N + torch.arange(BLOCK_N)

    acc = torch.zeros(BLOCK_M, BLOCK_N, dtype=torch.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + torch.arange(BLOCK_K)

        # Replicate masking
        mask_m = offs_m[:, None] < M
        mask_k_a = offs_k[None, :] < K
        mask_a = mask_m & mask_k_a

        mask_k_b = offs_k[:, None] < K
        mask_n = offs_n[None, :] < N
        mask_b = mask_k_b & mask_n

        # Load with masking (other=0.0)
        a_tile = torch.where(
            mask_a,
            A[offs_m.clamp(max=M-1)][:, offs_k.clamp(max=K-1)],
            torch.zeros(1)
        ).float()
        b_tile = torch.where(
            mask_b,
            B[offs_k.clamp(max=K-1)][:, offs_n.clamp(max=N-1)],
            torch.zeros(1)
        ).float()

        acc += a_tile @ b_tile

        print(f"  k_start={k_start}: "
              f"a_tile norm={a_tile.norm():.4f}, "
              f"b_tile norm={b_tile.norm():.4f}, "
              f"acc norm={acc.norm():.4f}")

    return acc
```

**When to use this:**
1. Isolation test identifies a failing tile configuration
2. You suspect pointer arithmetic is wrong
3. You want to verify masking logic for boundary tiles

### Debugging Checklist

When a kernel fails, work through this checklist:

1. **Run FP32 verification** — is it precision or a bug?
2. **Run tolerance sweep** — characterize the error distribution
3. **Visualize errors** — look for spatial patterns
4. **Run isolation matrix** — which configurations fail?
5. **Add `tl.device_print`** — inspect values at the failing tile
6. **Write NumPy equivalent** — compare intermediate values step by step
7. **Check pointer arithmetic** — verify strides match `tensor.stride()`
8. **Check masks** — verify `other=0.0` for loads, mask for stores
9. **Check accumulator dtype** — should be FP32 for FP16/BF16 operands
10. **Check grid dimensions** — `tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N)`

---

## Ampere `tl.dot` Caveat

On Ampere GPUs (compute capability 8.x), `tl.dot` always uses tensor cores, which
**only support FP16/BF16 operands**. If you pass FP32 tensors, Triton silently downcasts
them to FP16 before the tensor core operation, then upcasts the result back to FP32.

**Impact on debugging:**
- Step 1 of the diagnostic flowchart (FP32 verification) does NOT work for `tl.dot`-based
  kernels on Ampere. Even a correct FP32 kernel will show precision errors of ~0.02-0.13
  because the underlying computation is still FP16.
- **Workaround:** Skip FP32 verification for `tl.dot` kernels on Ampere. Go directly to
  error pattern analysis (Step 2) and isolation testing (Step 3).
- On Hopper (compute capability 9.x), `tl.dot` supports native FP32 via TF32 tensor cores,
  so FP32 verification works there (with appropriate tolerances for TF32).

**Tolerance implications:**
- For FP16 matmul on Ampere, use `atol=2e-1` for correctness thresholds to avoid
  false failures from FP16 precision. The table in §Expected Tolerances shows `atol=1e-2`
  as standard, but that assumes ideal conditions — real kernels with large K dimensions
  accumulate more rounding error, and the `other=0.0` masking pattern can amplify this.
