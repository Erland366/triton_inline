"""
Exercise: Matmul Optimization
Module: 1.1 — Core Kernel Mastery

Objectives:
1. Implement a tiled FP16 matmul kernel with L2 cache-friendly super-grouping
2. Use @triton.autotune with at least 4 configurations
3. Handle boundary conditions for arbitrary M, N, K
4. Accumulate in FP32, output in FP16

Acceptance Criteria:
1. Numerically correct (atol=1e-2 vs torch.matmul for FP16)
2. Autotuning selects config automatically
3. Works for non-square and non-aligned dimensions

Instructions:
1. Complete all TODO sections below
2. Run this file to verify correctness: python exercises/1.1/exercise.py
3. Once correct, run the benchmark: python exercises/1.1/benchmark.py
"""

import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# =============================================================================
# REFERENCE IMPLEMENTATION (do not modify)
# =============================================================================

def reference_matmul(a, b):
    """PyTorch reference implementation for correctness checking."""
    return torch.matmul(a, b)


# =============================================================================
# YOUR IMPLEMENTATION
# =============================================================================

# TODO 1: Define autotune configurations
# Create a list of at least 4 triton.Config objects. Each config specifies:
#   - BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K: tile dimensions (try powers of 2: 32, 64, 128, 256)
#   - GROUP_SIZE_M: super-grouping factor (8 is a good default)
#   - num_stages: software pipelining depth (2-5)
#   - num_warps: warp count (4 or 8)
#
# Think about which combinations make sense:
#   - Larger blocks = more compute per program, but more register pressure
#   - More stages = better latency hiding, but more shared memory
#   - More warps = more parallelism within a block

def get_autotune_configs():
    return [
        # Your configs here. Example structure (replace values):
        # triton.Config({'BLOCK_SIZE_M': ?, 'BLOCK_SIZE_N': ?, 'BLOCK_SIZE_K': ?, 'GROUP_SIZE_M': 8},
        #               num_stages=?, num_warps=?),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 64,  "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64,  "BLOCK_SIZE_N": 64,  "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=4),
    ]


# TODO 2: Add the @triton.autotune decorator and @triton.jit decorator
# The autotune decorator needs:
#   - configs: your list of configs from get_autotune_configs()
#   - key: which kernel arguments trigger re-tuning (hint: the matrix dimensions)

@triton.autotune(
    configs=get_autotune_configs(),
    key=["M", "N", "K"]
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides (how many elements to skip to move one position in each dimension)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters (block sizes, set by autotune)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Kernel for computing C = A @ B.
    A has shape (M, K), B has shape (K, N), C has shape (M, N).
    """
    # TODO 3: Compute program ID and map to (pid_m, pid_n) using super-grouping
    #
    # Steps:
    #   a. Get the linear program id: pid = tl.program_id(axis=0)
    #   b. Compute num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    #   c. Compute num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    #   d. Apply the super-grouping formula to get pid_m and pid_n
    #      (see study-notes.md "Deep Dive: Super-Grouping Walkthrough")
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

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # TODO 4: Create initial pointer blocks for A and B
    #
    # Steps:
    #   a. Compute row offsets for A: offs_am (BLOCK_SIZE_M elements starting at pid_m * BLOCK_SIZE_M)
    #   b. Compute col offsets for B: offs_bn (BLOCK_SIZE_N elements starting at pid_n * BLOCK_SIZE_N)
    #   c. Compute K offsets: offs_k (BLOCK_SIZE_K elements starting at 0)
    #   d. Build 2D pointer blocks using broadcasting:
    #      a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    #      b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    #
    # For boundary handling on M and N, you can use either:
    #   - Modular wrapping: offs_am = (...) % M  (tutorial 03 style)
    #   - Explicit clamping: tl.where(offs_am < M, offs_am, 0)  (tutorial 09 style)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # TODO 5: Accumulation loop over K dimension
    #
    # Steps:
    #   a. Initialize accumulator: tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    #   b. Loop: for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    #      - Load A block with K-dimension mask: mask = offs_k[None, :] < K - k * BLOCK_SIZE_K
    #      - Load B block with K-dimension mask: mask = offs_k[:, None] < K - k * BLOCK_SIZE_K
    #      - Accumulate: accumulator = tl.dot(a, b, accumulator)
    #      - Advance pointers: a_ptrs += BLOCK_SIZE_K * stride_ak
    #                          b_ptrs += BLOCK_SIZE_K * stride_bk
    #   c. Cast result: c = accumulator.to(tl.float16)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        A = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        B = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(A, B, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)

    # TODO 6: Store the result with boundary masking
    #
    # Steps:
    #   a. Compute output offsets: offs_cm, offs_cn (no modular wrapping here!)
    #   b. Build output pointer block: c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    #   c. Create mask: (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    #   d. Store: tl.store(c_ptrs, c, mask=c_mask)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm   + offs_cn[None, :] * stride_cn
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def matmul(a, b):
    """Launch wrapper for the Triton matmul kernel."""
    # TODO 7: Allocate output, compute grid, launch kernel
    #
    # Steps:
    #   a. Assert a.shape[1] == b.shape[0] (compatible dimensions)
    #   b. Assert a.is_contiguous() (required for stride assumptions)
    #   c. Get M, K from a.shape and K, N from b.shape
    #   d. Allocate output: c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    #   e. Compute grid as a lambda that uses META dict:
    #      grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    #   f. Launch: matmul_kernel[grid](a, b, c, M, N, K, ...strides..., )
    #   g. Return c
    assert a.shape[1] == b.shape[0]
    assert a.is_contiguous()
    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), dtype=torch.float16, device=a.device)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    matmul_kernel[grid](a, b, c, M, N, K, *a.stride(), *b.stride(), *c.stride())
    return c



# =============================================================================
# CORRECTNESS VERIFICATION
# =============================================================================

def verify():
    """Verify your implementation against the reference."""
    torch.manual_seed(42)

    test_cases = [
        (512, 512, 512, "square aligned"),
        (1024, 1024, 1024, "larger square"),
        (768, 512, 1024, "rectangular"),
        (513, 517, 519, "non-aligned (boundary test)"),
    ]

    all_passed = True
    for M, N, K, desc in test_cases:
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
        b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)

        ref_output = reference_matmul(a, b)
        your_output = matmul(a, b)

        if torch.allclose(ref_output, your_output, atol=5e-2, rtol=1e-2):
            diff = torch.abs(ref_output - your_output)
            print(f"[PASS] {desc} ({M}x{K} @ {K}x{N})")
            print(f"  Max diff: {diff.max().item():.6f}, Mean diff: {diff.mean().item():.6f}")
        else:
            diff = torch.abs(ref_output - your_output)
            print(f"[FAIL] {desc} ({M}x{K} @ {K}x{N})")
            print(f"  Max diff: {diff.max().item():.6f}, Mean diff: {diff.mean().item():.6f}")
            all_passed = False

    return all_passed


# =============================================================================
# HINTS (read progressively — try without hints first!)
# =============================================================================

# --- Hint 1 (Direction) ---
# The kernel is a direct implementation of the block-tiled algorithm from the study
# notes. Each program computes one BLOCK_SIZE_M x BLOCK_SIZE_N tile of the output.
# The main pieces are: (1) map pid to tile coordinates, (2) set up pointer blocks,
# (3) loop over K, (4) store the result. Focus on getting the pointer arithmetic
# right — that's where most bugs come from.

# --- Hint 2 (Approach) ---
# For the super-grouping (TODO 3), the key formula is:
#   num_pid_in_group = GROUP_SIZE_M * num_pid_n
#   group_id = pid // num_pid_in_group
#   first_pid_m = group_id * GROUP_SIZE_M
#   group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
#   pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
#   pid_n = (pid % num_pid_in_group) // group_size_m
#
# For pointers (TODO 4), remember that A is [M, K] and B is [K, N]:
#   a_ptrs shape: [BLOCK_SIZE_M, BLOCK_SIZE_K]
#   b_ptrs shape: [BLOCK_SIZE_K, BLOCK_SIZE_N]

# --- Hint 3 (Near-solution) ---
# The accumulation loop (TODO 5) in full:
#
#   accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
#   for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
#       a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
#       b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
#       accumulator = tl.dot(a, b, accumulator)
#       a_ptrs += BLOCK_SIZE_K * stride_ak
#       b_ptrs += BLOCK_SIZE_K * stride_bk
#   c = accumulator.to(tl.float16)
#
# For the store (TODO 6), use *unmasked* offsets (no % M):
#   offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#   offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)


if __name__ == "__main__":
    if verify():
        print("\nCorrectness verified! Run benchmark.py next.")
    else:
        print("\nFix the implementation before benchmarking.")
