"""
Exercise: Pointer Arithmetic & Block Pointers
Module: 1.1c - Core Kernel Mastery

Objectives:
1. Practice manual pointer arithmetic for row-wise kernels.
2. Practice tl.make_block_ptr for standard 2D rectangular tiles.
3. Write PyTorch-index comments for every tl.load and tl.store.

Run with:
    source .venv/bin/activate && \
    CC=/usr/bin/gcc CXX=/usr/bin/g++ PATH="$VIRTUAL_ENV/bin:/usr/bin:$PATH" \
    python exercises/1.1c/exercise.py
"""

import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def reference_row_affine(x, weight, bias, row_scale):
    """Reference for Y[row, col] = X[row, col] * W[col] + B[col] + RowScale[row]."""
    return x * weight[None, :] + bias[None, :] + row_scale[:, None]


def reference_block_copy_2d(x):
    """Reference for the 2D block pointer copy drill."""
    return x.clone()


@triton.jit
def row_affine_kernel(
    X,
    W,
    B,
    RowScale,
    Y,
    stride_xm,
    stride_ym,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # TODO: X[row, cols]
    # x = tl.load(...)

    # TODO: W[cols]
    # w = tl.load(...)

    # TODO: B[cols]
    # b = tl.load(...)

    # TODO: RowScale[row]
    # row_scale = tl.load(...)

    # TODO: Y[row, cols]
    # y = x * w + b + row_scale
    # tl.store(...)
    return


def row_affine(x, weight, bias, row_scale):
    """Launch wrapper for row_affine_kernel."""
    M, N = x.shape
    y = torch.empty_like(x)
    block_size = triton.next_power_of_2(N)

    # TODO: Launch one program per row.
    # row_affine_kernel[(M,)](...)
    raise NotImplementedError("Implement row_affine_kernel and its launch wrapper.")


@triton.jit
def block_copy_2d_kernel(
    X,
    Y,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N

    # TODO: X[start_m:start_m + BLOCK_M, start_n:start_n + BLOCK_N]
    # x_block = tl.make_block_ptr(...)
    # tile = tl.load(x_block, boundary_check=(0, 1), padding_option="zero")

    # TODO: Y[start_m:start_m + BLOCK_M, start_n:start_n + BLOCK_N]
    # y_block = tl.make_block_ptr(...)
    # tl.store(y_block, tile, boundary_check=(0, 1))
    return


def block_copy_2d(x, block_m=16, block_n=32):
    """Launch wrapper for block_copy_2d_kernel."""
    M, N = x.shape
    y = torch.empty_like(x)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

    # TODO: Launch block_copy_2d_kernel with explicit strides and block sizes.
    # block_copy_2d_kernel[grid](...)
    raise NotImplementedError("Implement block_copy_2d_kernel and its launch wrapper.")


def verify():
    """Run correctness checks after filling in the TODOs."""
    torch.manual_seed(0)
    all_pass = True

    print("=" * 50)
    print("  Row Affine Pointer Drill")
    print("=" * 50)
    for M, N in [(3, 7), (128, 781), (1024, 4096)]:
        x = torch.randn(M, N, device=DEVICE, dtype=torch.float16)
        weight = torch.randn(N, device=DEVICE, dtype=torch.float16)
        bias = torch.randn(N, device=DEVICE, dtype=torch.float16)
        row_scale = torch.randn(M, device=DEVICE, dtype=torch.float16)
        ref = reference_row_affine(x, weight, bias, row_scale)
        try:
            out = row_affine(x, weight, bias, row_scale)
        except NotImplementedError as exc:
            print(f"  [TODO] {exc}")
            all_pass = False
            break
        diff = torch.abs(ref - out)
        passed = torch.allclose(ref, out, atol=1e-3, rtol=0)
        print(f"  [{'PASS' if passed else 'FAIL'}] M={M:4d} N={N:4d} max_err={diff.max().item():.4e}")
        all_pass &= passed

    print()
    print("=" * 50)
    print("  2D Block Pointer Copy Drill")
    print("=" * 50)
    for M, N in [(17, 31), (129, 257), (1025, 769)]:
        x = torch.randn(M, N, device=DEVICE, dtype=torch.float16)
        ref = reference_block_copy_2d(x)
        try:
            out = block_copy_2d(x)
        except NotImplementedError as exc:
            print(f"  [TODO] {exc}")
            all_pass = False
            break
        diff = torch.abs(ref - out)
        passed = torch.equal(ref, out)
        print(f"  [{'PASS' if passed else 'FAIL'}] M={M:4d} N={N:4d} max_err={diff.max().item():.4e}")
        all_pass &= passed

    return all_pass


if __name__ == "__main__":
    if verify():
        print("\nAll pointer arithmetic drills passed.")
    else:
        print("\nFill in the TODOs, keeping PyTorch-index comments above every pointer.")

