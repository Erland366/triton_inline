"""
Exercise: TLX Vector Add With Async Tasks
Module: TLX — Low-Level Triton Extensions

Objectives:
1. Write a normal Triton kernel for two independent vector additions
2. Write a TLX warp-specialized version using async task regions
3. Verify both implementations against PyTorch

Acceptance Criteria:
1. Both outputs from the normal Triton kernel match PyTorch
2. Both outputs from the TLX warp-specialized kernel match PyTorch
3. The TLX kernel uses `tlx.async_tasks()` and at least two `tlx.async_task`
   regions

Instructions:
1. Complete the TODO sections below
2. Run this file to verify correctness:
   python exercises/tlx/exercise.py
3. Once correct, run the benchmark:
   python exercises/tlx/benchmark.py
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton._internal_testing import is_hopper_or_newer

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# =============================================================================
# REFERENCE IMPLEMENTATION (do not modify)
# =============================================================================


def reference_dual_add(x, y, a, b):
    """PyTorch reference for two independent vector additions."""
    return x + y, a + b


# =============================================================================
# PART 1: NORMAL TRITON DUAL ADD
# =============================================================================


# TODO 1: Write a normal Triton kernel that computes:
#   z = x + y
#   c = a + b
#
# Keep the indexing simple:
#   - one program handles BLOCK_SIZE contiguous elements
#   - use `tl.program_id(0)` for the program id
#   - use `tl.arange(0, BLOCK_SIZE)` for element offsets
#   - mask loads and stores with `offsets < n_elements`
@triton.jit
def add2_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    output1 = x + y
    output2 = a + b
    tl.store(z_ptr + offsets, output1, mask=mask)
    tl.store(c_ptr + offsets, output2, mask=mask)

def add2(x: torch.Tensor, y: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    """Launch `add2_kernel` and return both outputs."""
    # TODO 2: Allocate outputs, compute grid, launch add2_kernel, return outputs.
    output1 = torch.empty_like(x)
    output2 = torch.empty_like(a)
    assert (
        x.device == DEVICE 
        and y.device == DEVICE 
        and output1.device == DEVICE 
        and output2.device == DEVICE
    )
    n_elements = output1.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    add2_kernel[grid](x, y, output1, a, b, output2, n_elements, BLOCK_SIZE=1024)
    return output1, output2


# =============================================================================
# PART 2: TLX WARP-SPECIALIZED DUAL ADD
# =============================================================================


# TODO 3: Write a TLX version of the dual-add kernel.
#
# Goal:
#   - Use `with tlx.async_tasks():`
#   - Put one addition in `with tlx.async_task("default"):`
#   - Put the other addition in a second `tlx.async_task(...)` region
#
# This is intentionally a toy kernel. The learning target is task partitioning,
# not performance.
@triton.jit
def add2_warp_specialized_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = BLOCK_SIZE * pid
    with tlx.async_tasks():
        with tlx.async_task("default"):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y
            tl.store(z_ptr + offsets, output, mask=mask)
        with tlx.async_task(num_warps=4, replicate=2):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            a = tl.load(a_ptr + offsets, mask=mask)
            b = tl.load(b_ptr + offsets, mask=mask)
            output = a + b
            tl.store(c_ptr + offsets, output, mask=mask)


def add2_warp_specialized(
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
):
    """Launch `add_warp_specialized` and return both outputs."""
    # TODO 2: Allocate outputs, compute grid, launch add2_kernel, return outputs.
    output1 = torch.empty_like(x)
    output2 = torch.empty_like(a)
    assert (
        x.device == DEVICE 
        and y.device == DEVICE 
        and output1.device == DEVICE 
        and output2.device == DEVICE
    )
    n_elements = output1.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    add2_warp_specialized_kernel[grid](x, y, output1, a, b, output2, n_elements, BLOCK_SIZE=1024)
    return output1, output2


# =============================================================================
# CORRECTNESS CHECKS
# =============================================================================


def verify_correctness(size: int = 98_432):
    if not is_hopper_or_newer():
        raise RuntimeError("TLX warp specialization requires Hopper or newer.")

    torch.manual_seed(0)
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)
    a = torch.rand(size, device=DEVICE)
    b = torch.rand(size, device=DEVICE)

    ref1, ref2 = reference_dual_add(x, y, a, b)

    triton1, triton2 = add2(x, y, a, b)
    torch.testing.assert_close(triton1, ref1)
    torch.testing.assert_close(triton2, ref2)

    tlx1, tlx2 = add2_warp_specialized(x, y, a, b)
    torch.testing.assert_close(tlx1, ref1)
    torch.testing.assert_close(tlx2, ref2)

    print("Correctness OK.")


if __name__ == "__main__":
    verify_correctness()
