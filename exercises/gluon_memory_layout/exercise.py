
# type: ignore
"""
Exercise: Gluon Memory Layouts
Module: Gluon - explicit tensor layout control

Objectives:
1. Learn how `gl.BlockedLayout` maps logical tile elements to GPU threads.
2. Write a 1D Gluon memcpy where `gl.arange(..., layout=layout)` drives the
   layout of every derived tensor.
3. Extend the same idea to 2D tensors with `gl.SliceLayout` and broadcasting.
4. Handle input/output tensors whose best global-memory layouts differ.

Run locally, if this machine has a CUDA GPU:
    source .venv/bin/activate && python exercises/gluon_memory_layout/exercise.py

Run on Modal H100:
    source .venv/bin/activate && modal run run_modal.py \
      --action pytest \
      --script "exercises/gluon_memory_layout/exercise.py -q"
"""
from __future__ import annotations

from dataclasses import dataclass
import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


def active_device():
    return triton.runtime.driver.active.get_active_torch_device()


@dataclass(frozen=True)
class LayoutSpec:
    """Small Python-side description for BlockedLayout arithmetic drills."""

    size_per_thread: tuple[int, ...]
    threads_per_warp: tuple[int, ...]
    warps_per_cta: tuple[int, ...]
    order: tuple[int, ...]


def block_shape(spec: LayoutSpec) -> tuple[int, ...]:
    """
    Return the logical block shape represented by a Gluon BlockedLayout.

    TODO 0:
    - Multiply `size_per_thread`, `threads_per_warp`, and `warps_per_cta`
      elementwise.
    - Keep this as plain Python. It is a warm-up for reading layout configs.
    """
    raise NotImplementedError("TODO 0: compute the elementwise block shape.")


def elements_per_thread(shape: tuple[int, ...], spec: LayoutSpec) -> int:
    """
    Estimate how many logical elements one thread owns for a tensor shape.

    TODO 1:
    - Start from `block_shape(spec)`.
    - Account for tensors larger than one block by counting how many logical
      blocks tile each dimension.
    - Use ceil division per dimension.

    This is intentionally approximate. The point is to notice when a layout
    increases register pressure.
    """
    raise NotImplementedError("TODO 1: estimate per-thread logical element count.")


def reference_copy_1d(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


def reference_copy_2d(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


def make_1d_layout(r: int, num_warps: int):
    """
    Build the 1D BlockedLayout used by the first memcpy drill.

    TODO 2:
    - Return `gl.BlockedLayout(...)`.
    - The layout should have one dimension.
    - `r` is the number of elements per thread.
    - NVIDIA has 32 threads per warp.
    """
    raise NotImplementedError("TODO 2: build the 1D BlockedLayout.")


def make_2d_row_contiguous_layout(num_warps: int):
    """
    Layout for a 2D tile whose inner / column dimension is contiguous.

    TODO 3:
    - Return a 2D `gl.BlockedLayout`.
    - Choose the layout so adjacent lanes walk the contiguous column dimension.
    """
    raise NotImplementedError("TODO 3: build the row-contiguous 2D layout.")


def make_2d_col_contiguous_layout(num_warps: int):
    """
    Layout for a 2D tile whose outer / row dimension is contiguous.

    TODO 4:
    - Return a 2D `gl.BlockedLayout`.
    - This should be the transposed counterpart of TODO 3.
    """
    raise NotImplementedError("TODO 4: build the column-contiguous 2D layout.")


def layout_for_gmem_access(x: torch.Tensor, num_warps: int):
    """
    Pick a layout based on the contiguous dimension of a 1D or 2D tensor.

    TODO 5:
    - For 1D tensors, use the 1D layout from TODO 2 with r=1.
    - For 2D tensors, inspect `x.stride()`.
    - If the second dimension is contiguous, use TODO 3.
    - Otherwise use TODO 4.
    """
    raise NotImplementedError("TODO 5: select a layout from tensor strides.")


@gluon.jit
def memcpy_1d_kernel(
    in_ptr,
    out_ptr,
    xnumel,
    XBLOCK: gl.constexpr,
    layout: gl.constexpr,
):
    pid = gl.program_id(0)
    start = pid * XBLOCK

    # TODO 6:
    # - Create a 1D index tensor using `gl.arange` with the supplied layout.
    # - Build offsets from `start`.
    # - Mask by `xnumel`.
    # - Use `gl.load` and `gl.store`.
    return


def memcpy_1d(x: torch.Tensor, xblock: int = 256, r: int = 1, num_warps: int = 4):
    """Launch the 1D layout-aware memcpy kernel."""
    out = torch.empty_like(x)
    layout = make_1d_layout(r, num_warps)
    grid = (triton.cdiv(x.numel(), xblock),)
    memcpy_1d_kernel[grid](x, out, x.numel(), xblock, layout, num_warps=num_warps)
    return out


@gluon.jit
def memcpy_2d_kernel(
    in_ptr,
    out_ptr,
    xnumel,
    ynumel,
    xstride_in,
    ystride_in,
    xstride_out,
    ystride_out,
    layout: gl.constexpr,
    XBLOCK: gl.constexpr,
    YBLOCK: gl.constexpr,
):
    pid_x = gl.program_id(0)
    pid_y = gl.program_id(1)

    start_x = pid_x * XBLOCK
    start_y = pid_y * YBLOCK

    # TODO 7:
    # - Create x indices with a SliceLayout that drops the y dimension.
    # - Create y indices with a SliceLayout that drops the x dimension.
    # - Broadcast them into a 2D offset tensor.
    # - Build input/output offsets from the explicit strides.
    # - Load and store with a 2D mask.
    return


def memcpy_2d(
    x: torch.Tensor,
    xblock: int = 1,
    yblock: int = 256,
    num_warps: int = 4,
    layout=None,
):
    """Launch the 2D layout-aware memcpy kernel."""
    out = torch.empty_like(x)
    if layout is None:
        layout = layout_for_gmem_access(x, num_warps)

    grid = (triton.cdiv(x.shape[0], xblock), triton.cdiv(x.shape[1], yblock))
    memcpy_2d_kernel[grid](
        x,
        out,
        x.shape[0],
        x.shape[1],
        *x.stride(),
        *out.stride(),
        layout,
        xblock,
        yblock,
        num_warps=num_warps,
    )
    return out


@gluon.jit
def memcpy_2d_inout_kernel(
    in_ptr,
    out_ptr,
    xnumel,
    ynumel,
    xstride_in,
    ystride_in,
    xstride_out,
    ystride_out,
    layout_in: gl.constexpr,
    layout_out: gl.constexpr,
    XBLOCK: gl.constexpr,
    YBLOCK: gl.constexpr,
):
    pid_x = gl.program_id(0)
    pid_y = gl.program_id(1)

    start_x = pid_x * XBLOCK
    start_y = pid_y * YBLOCK

    # TODO 8:
    # - Build input offsets using `layout_in`.
    # - Build output offsets using `layout_out`.
    # - Load with the input layout.
    # - Convert the loaded value to `layout_out`.
    # - Store with the output layout.
    return


def memcpy_2d_inout(
    x: torch.Tensor,
    out: torch.Tensor,
    xblock: int = 128,
    yblock: int = 128,
    num_warps: int = 4,
):
    """Copy between tensors whose best global-memory layouts may differ."""
    assert x.shape == out.shape, "input and output must have the same logical shape"
    layout_in = layout_for_gmem_access(x, num_warps)
    layout_out = layout_for_gmem_access(out, num_warps)
    grid = (triton.cdiv(x.shape[0], xblock), triton.cdiv(x.shape[1], yblock))
    memcpy_2d_inout_kernel[grid](
        x,
        out,
        x.shape[0],
        x.shape[1],
        *x.stride(),
        *out.stride(),
        layout_in,
        layout_out,
        xblock,
        yblock,
        num_warps=num_warps,
    )
    return out


def _cuda_available() -> bool:
    return torch.cuda.is_available()


def _skip_without_cuda():
    if not _cuda_available():
        pytest.skip("Gluon exercises require a CUDA GPU.")


def test_layout_arithmetic():
    spec = LayoutSpec(
        size_per_thread=(2, 4),
        threads_per_warp=(16, 2),
        warps_per_cta=(2, 2),
        order=(1, 0),
    )
    assert block_shape(spec) == (64, 16)
    assert elements_per_thread((128, 128), spec) >= 8


@pytest.mark.parametrize("xnumel", [200, 1000, 4099])
@pytest.mark.parametrize("r", [1, 2])
def test_memcpy_1d(xnumel, r):
    _skip_without_cuda()
    torch.manual_seed(0)
    x = torch.randn(xnumel, device=active_device())
    out = memcpy_1d(x, xblock=256, r=r)
    torch.testing.assert_close(out, reference_copy_1d(x), atol=0, rtol=0)


@pytest.mark.parametrize("shape", [(17, 257), (100, 2000)])
@pytest.mark.parametrize("transposed", [False, True])
def test_memcpy_2d(shape, transposed):
    _skip_without_cuda()
    torch.manual_seed(0)
    x = torch.randn(shape, device=active_device())
    x = x.T if transposed else x
    out = memcpy_2d(x)
    torch.testing.assert_close(out, reference_copy_2d(x), atol=0, rtol=0)


@pytest.mark.parametrize("transpose_in, transpose_out", [(True, False), (False, True)])
def test_memcpy_2d_inout(transpose_in, transpose_out):
    _skip_without_cuda()
    torch.manual_seed(0)
    xnumel, ynumel = 128, 256
    if transpose_in:
        x = torch.randn((ynumel, xnumel), device=active_device()).T
    else:
        x = torch.randn((xnumel, ynumel), device=active_device())

    if transpose_out:
        out = torch.empty((ynumel, xnumel), device=active_device()).T
    else:
        out = torch.empty((xnumel, ynumel), device=active_device())

    result = memcpy_2d_inout(x, out)
    torch.testing.assert_close(result, x, atol=0, rtol=0)


def bench_memcpy_1d_layouts(size: int = 1 << 24):
    """
    Optional benchmark after correctness passes.

    TODO 9:
    - Sweep R values such as 1, 2, 4, 8.
    - Use `triton.testing.do_bench`.
    - Return rows that include R and GB/s or TB/s.
    """
    raise NotImplementedError("TODO 9: add the optional layout throughput sweep.")


def main():
    print("Gluon memory-layout curriculum scaffold.")
    print("Run with pytest after filling each TODO:")
    print("  source .venv/bin/activate && python -m pytest exercises/gluon_memory_layout/exercise.py -q")
    print("On Modal H100:")
    print("  source .venv/bin/activate && modal run run_modal.py --action pytest --script \"exercises/gluon_memory_layout/exercise.py -q\"")


if __name__ == "__main__":
    main()
