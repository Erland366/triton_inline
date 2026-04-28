"""
Sub-exercise 4: Debug Seeded Bugs
Module: 1.1b — Benchmarking, Profiling & Debugging Toolkit

Three BROKEN variants of your Module 1.1 matmul kernel. Each contains exactly
ONE bug that produces incorrect results.

Your task (for each variant):
1. Run the tests — observe which test cases pass and which fail
2. Follow the diagnostic flowchart from debugging-methodology.md
3. Use the tolerance sweep to characterize the error
4. Identify the bug and document your process in debug_report.md

DO NOT look at the diff between these and your working kernel.
The goal is to practice the debugging METHODOLOGY, not just find the bug.
"""

import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

ATOL = 2e-1  # FP16 matmul precision threshold (avoids false failures from FP16 rounding)


# =============================================================================
# REFERENCE IMPLEMENTATION
# =============================================================================

def reference_matmul(a, b):
    return torch.matmul(a, b)


# =============================================================================
# SHARED CONFIG
# =============================================================================

def get_autotune_configs():
    return [
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 64,  "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64,  "BLOCK_SIZE_N": 64,  "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=4),
    ]


# =============================================================================
# BUG 1: K-loop bug
# =============================================================================

@triton.autotune(configs=get_autotune_configs(), key=["M", "N", "K"])
@triton.jit
def buggy_kernel_v1(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    position = pid % num_pid_in_group
    pid_m = first_pid_m + (position % group_size_m)
    pid_n = position // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K // BLOCK_SIZE_K):
        A = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        B = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(A, B, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def buggy_matmul_v1(a, b):
    assert a.shape[1] == b.shape[0] and a.is_contiguous()
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), dtype=torch.float16, device=a.device)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
    buggy_kernel_v1[grid](a, b, c, M, N, K, *a.stride(), *b.stride(), *c.stride())
    return c


# =============================================================================
# BUG 2: Stride bug
# =============================================================================

@triton.autotune(configs=get_autotune_configs(), key=["M", "N", "K"])
@triton.jit
def buggy_kernel_v2(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    position = pid % num_pid_in_group
    pid_m = first_pid_m + (position % group_size_m)
    pid_n = position // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        A = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        B = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(A, B, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def buggy_matmul_v2(a, b):
    assert a.shape[1] == b.shape[0] and a.is_contiguous()
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), dtype=torch.float16, device=a.device)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
    buggy_kernel_v2[grid](a, b, c, M, N, K, *a.stride(), *b.stride(), *c.stride())
    return c


# =============================================================================
# BUG 3: Accumulator bug
# =============================================================================

@triton.autotune(configs=get_autotune_configs(), key=["M", "N", "K"])
@triton.jit
def buggy_kernel_v3(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    position = pid % num_pid_in_group
    pid_m = first_pid_m + (position % group_size_m)
    pid_n = position // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        A = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        B = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(A, B, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def buggy_matmul_v3(a, b):
    assert a.shape[1] == b.shape[0] and a.is_contiguous()
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), dtype=torch.float16, device=a.device)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
    buggy_kernel_v3[grid](a, b, c, M, N, K, *a.stride(), *b.stride(), *c.stride())
    return c


# =============================================================================
# TEST INFRASTRUCTURE
# =============================================================================

test_cases = [
    (512, 512, 512, "square aligned"),
    (1024, 1024, 1024, "larger square"),
    (768, 512, 1024, "rectangular"),
    (1024, 1024, 1023, "K off by one"),
    (513, 517, 519, "non-aligned"),
    (64, 64, 64, "single-tile size"),
    (64, 64, 65, "K boundary + 1"),
    (65, 65, 65, "all boundary + 1"),
    (127, 131, 137, "prime sizes"),
    (64, 64, 32, "K = BLOCK_SIZE_K"),
]


def tolerance_sweep(ref, actual, name="kernel"):
    """Sweep tolerances to characterize error distribution."""
    diff = torch.abs(ref - actual)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    std_diff = diff.std().item()

    print(f"\n--- {name} Error Analysis ---")
    print(f"  Max:  {max_diff:.6e}")
    print(f"  Mean: {mean_diff:.6e}")
    print(f"  Std:  {std_diff:.6e}")
    print()

    for atol in [1e-4, 1e-3, 1e-2, 1e-1, 5e-1, 1.0, 5.0, 10.0]:
        matches = torch.isclose(ref, actual, atol=atol, rtol=0).float().mean()
        status = "PASS" if matches == 1.0 else f"{matches*100:.1f}%"
        print(f"  atol={atol:.0e}: {status}")

    print()
    error_mask = diff > mean_diff + 3 * std_diff
    n_outliers = error_mask.sum().item()
    n_total = diff.numel()
    print(f"  Outliers (>3sigma): {n_outliers}/{n_total} ({n_outliers/n_total*100:.2f}%)")

    if n_outliers > 0 and diff.dim() >= 2:
        rows_with_errors = error_mask.any(dim=-1).sum().item()
        total_rows = error_mask.shape[0]
        print(f"  Rows with outliers: {rows_with_errors}/{total_rows}")


def run_tests(matmul_fn, name):
    """Run test suite for a buggy matmul variant."""
    torch.manual_seed(42)
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    for M, N, K, desc in test_cases:
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
        b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)

        ref = reference_matmul(a, b)
        out = matmul_fn(a, b)

        diff = torch.abs(ref - out)
        max_err = diff.max().item()
        if torch.allclose(ref, out, atol=ATOL, rtol=0):
            print(f"  [PASS] {desc:25s}  M={M:5d} N={N:5d} K={K:5d}  max_err={max_err:.4e}")
        else:
            print(f"  [FAIL] {desc:25s}  M={M:5d} N={N:5d} K={K:5d}  max_err={max_err:.4e}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys

    # Select which bug to debug: python exercise_debug.py [1|2|3|all]
    choice = sys.argv[1] if len(sys.argv) > 1 else "all"

    variants = {
        "1": (buggy_matmul_v1, "Bug 1"),
        "2": (buggy_matmul_v2, "Bug 2"),
        "3": (buggy_matmul_v3, "Bug 3"),
    }

    if choice == "all":
        for key in ["1", "2", "3"]:
            fn, name = variants[key]
            run_tests(fn, name)
    elif choice in variants:
        fn, name = variants[choice]
        run_tests(fn, name)

        # Run tolerance sweep on first failing case
        print(f"\n{'=' * 60}")
        print(f"  Tolerance Sweep")
        print(f"{'=' * 60}")
        torch.manual_seed(42)
        for M, N, K, desc in test_cases:
            a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
            b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
            ref = reference_matmul(a, b)
            out = fn(a, b)
            if not torch.allclose(ref, out, atol=ATOL, rtol=0):
                tolerance_sweep(ref, out, f"{name} [{desc}] M={M} N={N} K={K}")
                break
    else:
        print(f"Usage: python exercise_debug.py [1|2|3|all]")
