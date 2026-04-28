"""
Exercise: Fused Softmax & LayerNorm
Module: 1.2 — Core Kernel Mastery

Objectives:
1. Write a fused row-wise softmax kernel
2. Write a fused layernorm forward kernel
3. Write a fused layernorm backward kernel (dx only — dw/db is bonus)

Acceptance Criteria:
1. Softmax matches torch.softmax(x, dim=-1) within atol=1e-3
2. LayerNorm forward matches torch.nn.functional.layer_norm within atol=1e-3
3. LayerNorm backward dx matches PyTorch autograd within atol=1e-3

Instructions:
1. Complete all TODO sections below
2. Run this file to verify correctness: python exercises/1.2/exercise.py
3. Once correct, write your benchmark: exercises/1.2/benchmark.py
"""

import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# =============================================================================
# REFERENCE IMPLEMENTATIONS (do not modify)
# =============================================================================

def reference_softmax(x):
    """PyTorch reference for row-wise softmax."""
    return torch.softmax(x, dim=-1)


def reference_layernorm(x, weight, bias, eps=1e-5):
    """PyTorch reference for layernorm forward."""
    return torch.nn.functional.layer_norm(x, x.shape[-1:], weight, bias, eps)


def reference_layernorm_backward(x, weight, bias, dy, eps=1e-5):
    """PyTorch reference for layernorm backward (returns dx)."""
    x = x.clone().requires_grad_(True)
    y = torch.nn.functional.layer_norm(x, x.shape[-1:], weight, bias, eps)
    y.backward(dy)
    return x.grad


# =============================================================================
# PART 1: FUSED SOFTMAX
# =============================================================================

# TODO 1: Write the softmax kernel
# Key decisions:
#   - One program per row (or persistent loop over rows)
#   - BLOCK_SIZE must be >= n_cols (entire row in one block)
#   - Use other=-float('inf') for masked loads (why?)
#   - Numerical stability: subtract max before exp
#   - Reductions: tl.max(..., axis=0) and tl.sum(..., axis=0)
@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    tl.assume(pid >= 0)
    tl.assume(n_rows >= 0)
    tl.assume(n_cols >= 0)

    row_ptr = input_ptr + pid * input_row_stride
    col_offs = tl.arange(0, BLOCK_SIZE)
    col_offs = tl.multiple_of(col_offs, 16)
    mask = col_offs < n_cols
    row = tl.load(row_ptr + col_offs, mask=mask, other=-float("inf"))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator)

    out = numerator / denominator
    out_ptr = output_ptr + pid * output_row_stride
    tl.store(out_ptr + col_offs, out, mask=mask)



def fused_softmax(x):
    """Launch wrapper for the softmax kernel."""
    n_rows, n_cols = x.shape
    # TODO 2: Compute BLOCK_SIZE, allocate output, set grid, launch kernel
    # Hint: BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # Hint: grid = (n_rows,) for one-program-per-row, or fewer for persistent
    y = torch.empty_like(x)
    grid = lambda META: (n_rows, )
    softmax_kernel[grid](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=triton.next_power_of_2(n_cols))
    return y


# =============================================================================
# PART 2: FUSED LAYERNORM FORWARD
# =============================================================================

# TODO 3: Write the layernorm forward kernel
# Key decisions:
#   - One program per row
#   - Accumulate mean and variance in FP32
#   - If row > BLOCK_SIZE, loop over chunks (for off in range(0, N, BLOCK_SIZE))
#   - Apply affine transform: y = (x - mean) * rstd * weight + bias
#   - Save mean and rstd for backward pass
@triton.jit
def layernorm_fwd_kernel(
    X, Y, W, B, Mean, Rstd,
    stride, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    x_ptr = X + pid * stride
    tl.assume(pid > 0)
    offs = tl.arange(0, BLOCK_SIZE)
    offs = tl.multiple_of(offs, 16)
    _mean = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + offs
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        _mean += x
    mean = tl.sum(_mean, axis=0) / N


    _variance = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + offs
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x_centered = x - mean
        x_centered = tl.where(mask, x_centered, 0)
        _variance += x_centered * x_centered
    variance = tl.sum(_variance, axis=0) / N
    rstd = 1 / tl.sqrt(variance + eps)

    # TODO: normalize and apply affine transform
    for off in range(0, N, BLOCK_SIZE):
        pass



def fused_layernorm_fwd(x, weight, bias, eps=1e-5):
    """Launch wrapper for layernorm forward."""
    x_flat = x.reshape(-1, x.shape[-1])
    M, N = x_flat.shape
    y = torch.empty_like(x_flat)
    mean = torch.empty(M, dtype=torch.float32, device=x.device)
    rstd = torch.empty(M, dtype=torch.float32, device=x.device)
    # TODO 4: Compute BLOCK_SIZE, set grid, launch kernel
    # Hint: BLOCK_SIZE = triton.next_power_of_2(N), capped at 65536 // x.element_size()
    # Hint: grid = (M,) — one program per row
    return y.reshape_as(x), mean, rstd


# =============================================================================
# PART 3: FUSED LAYERNORM BACKWARD (dx only)
# =============================================================================

# TODO 5: Write the layernorm backward kernel for dx
# The VJP formula for dx is:
#   x_hat = (x - mean) * rstd
#   wdy = weight * dy
#   c1 = (1/N) * sum(x_hat * wdy)
#   c2 = (1/N) * sum(wdy)
#   dx = (wdy - x_hat * c1 - c2) * rstd
#
# Key decisions:
#   - One program per row (same as forward)
#   - Load x, dy, weight; use saved mean/rstd from forward
#   - Accumulate c1 and c2 as FP32 scalars via tl.sum
@triton.jit
def layernorm_bwd_dx_kernel(
    DX, DY, X, W, Mean, Rstd,
    stride, N,
    BLOCK_SIZE: tl.constexpr,
):
    pass  # TODO: implement


def fused_layernorm_bwd_dx(dy, x, weight, mean, rstd):
    """Launch wrapper for layernorm backward (dx only)."""
    x_flat = x.reshape(-1, x.shape[-1])
    dy_flat = dy.reshape(-1, dy.shape[-1])
    M, N = x_flat.shape
    dx = torch.empty_like(x_flat)
    # TODO 6: Compute BLOCK_SIZE, set grid, launch kernel
    # Same BLOCK_SIZE and grid as forward
    return dx.reshape_as(x)


# =============================================================================
# CORRECTNESS VERIFICATION
# =============================================================================

def verify():
    """Verify all implementations against references."""
    torch.manual_seed(42)
    all_pass = True

    # --- Softmax ---
    print("=" * 50)
    print("  Softmax Verification")
    print("=" * 50)
    for M, N in [(1823, 781), (4096, 4096), (128, 8192), (8192, 1024)]:
        x = torch.randn(M, N, device=DEVICE, dtype=torch.float16)
        ref = reference_softmax(x)
        out = fused_softmax(x)
        diff = torch.abs(ref - out)
        max_err = diff.max().item()
        if torch.allclose(ref, out, atol=1e-3, rtol=0):
            print(f"  [PASS] M={M:5d} N={N:5d}  max_err={max_err:.4e}")
        else:
            print(f"  [FAIL] M={M:5d} N={N:5d}  max_err={max_err:.4e}")
            all_pass = False

    # --- LayerNorm Forward ---
    print()
    print("=" * 50)
    print("  LayerNorm Forward Verification")
    print("=" * 50)
    for M, N in [(4096, 1024), (2048, 4096), (1151, 8192), (512, 768)]:
        x = torch.randn(M, N, device=DEVICE, dtype=torch.float16)
        w = torch.randn(N, device=DEVICE, dtype=torch.float16)
        b = torch.randn(N, device=DEVICE, dtype=torch.float16)
        ref = reference_layernorm(x, w, b)
        out, mean, rstd = fused_layernorm_fwd(x, w, b)
        diff = torch.abs(ref - out)
        max_err = diff.max().item()
        if torch.allclose(ref, out, atol=1e-3, rtol=0):
            print(f"  [PASS] M={M:5d} N={N:5d}  max_err={max_err:.4e}")
        else:
            print(f"  [FAIL] M={M:5d} N={N:5d}  max_err={max_err:.4e}")
            all_pass = False

    # --- LayerNorm Backward (dx) ---
    print()
    print("=" * 50)
    print("  LayerNorm Backward (dx) Verification")
    print("=" * 50)
    for M, N in [(4096, 1024), (2048, 4096), (1151, 8192)]:
        x = torch.randn(M, N, device=DEVICE, dtype=torch.float16)
        w = torch.randn(N, device=DEVICE, dtype=torch.float16)
        b = torch.randn(N, device=DEVICE, dtype=torch.float16)
        dy = torch.randn(M, N, device=DEVICE, dtype=torch.float16)
        ref_dx = reference_layernorm_backward(x, w, b, dy)
        # Run forward to get mean/rstd, then backward
        _, mean, rstd = fused_layernorm_fwd(x, w, b)
        out_dx = fused_layernorm_bwd_dx(dy, x, w, mean, rstd)
        diff = torch.abs(ref_dx - out_dx)
        max_err = diff.max().item()
        if torch.allclose(ref_dx, out_dx, atol=1e-3, rtol=0):
            print(f"  [PASS] M={M:5d} N={N:5d}  max_err={max_err:.4e}")
        else:
            print(f"  [FAIL] M={M:5d} N={N:5d}  max_err={max_err:.4e}")
            all_pass = False

    return all_pass


# =============================================================================
# HINTS (read progressively — try without hints first!)
# =============================================================================

# --- Hint 1 (Direction) ---
# Softmax: each program loads one row, computes max, subtracts, exp, sum, divides.
# LayerNorm fwd: each program loads one row in chunks, computes mean in first loop,
# variance in second loop, then normalizes in third loop.
# LayerNorm bwd: load x, dy, w for the row. Compute x_hat, wdy, then c1 and c2 via
# tl.sum, then compute dx = (wdy - x_hat*c1 - c2) * rstd.

# --- Hint 2 (Approach) ---
# For softmax, the trick is BLOCK_SIZE >= n_cols so the whole row fits.
# Use: col_offsets = tl.arange(0, BLOCK_SIZE), mask = col_offsets < n_cols
# For layernorm with large N, loop: for off in range(0, N, BLOCK_SIZE)
# and accumulate partial sums into a tl.zeros([BLOCK_SIZE], dtype=tl.float32) buffer,
# then tl.sum at the end.

# --- Hint 3 (Near-solution) ---
# Softmax kernel body:
#   row_idx = tl.program_id(0)
#   row_ptr = input_ptr + row_idx * input_row_stride
#   cols = tl.arange(0, BLOCK_SIZE)
#   mask = cols < n_cols
#   row = tl.load(row_ptr + cols, mask=mask, other=-float('inf'))
#   row = row - tl.max(row, axis=0)
#   num = tl.exp(row)
#   den = tl.sum(num, axis=0)
#   out = num / den
#   out_ptr = output_ptr + row_idx * output_row_stride
#   tl.store(out_ptr + cols, out, mask=mask)


if __name__ == "__main__":
    if verify():
        print("\nAll correctness checks passed! Write your benchmark.py next.")
    else:
        print("\nFix the implementation before benchmarking.")
