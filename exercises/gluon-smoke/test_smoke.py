# type: ignore
"""Minimal correctness gate for running Gluon kernels on the Modal H100."""

from __future__ import annotations

import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def elementwise_add_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    xnumel,
    ynumel,
    xstride_a,
    ystride_a,
    xstride_b,
    ystride_b,
    xstride_output,
    ystride_output,
    xblock: gl.constexpr,
    yblock: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1],
        threads_per_warp=[1, 32],
        warps_per_cta=[1, 4],
        order=[1, 0],
    )
    xoffsets = gl.program_id(0) * xblock + gl.arange(
        0,
        xblock,
        layout=gl.SliceLayout(dim=1, parent=layout),
    )

    a_rows = a_ptr + xstride_a * xoffsets[:, None]
    b_rows = b_ptr + xstride_b * xoffsets[:, None]
    output_rows = output_ptr + xstride_output * xoffsets[:, None]

    for ystart in range(0, ynumel, yblock):
        yoffsets = ystart + gl.arange(
            0,
            yblock,
            layout=gl.SliceLayout(dim=0, parent=layout),
        )
        mask = (xoffsets < xnumel)[:, None] & (yoffsets < ynumel)[None, :]
        a = gl.load(a_rows + ystride_a * yoffsets[None, :], mask=mask)
        b = gl.load(b_rows + ystride_b * yoffsets[None, :], mask=mask)
        gl.store(
            output_rows + ystride_output * yoffsets[None, :],
            a + b,
            mask=mask,
        )


def elementwise_add(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    xblock: int = 32,
    yblock: int = 128,
) -> torch.Tensor:
    """Add two contiguous 2D CUDA tensors with an explicit Gluon layout."""
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("elementwise_add requires CUDA tensors")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"inputs must be 2D, got {a.ndim}D and {b.ndim}D")
    if a.shape != b.shape:
        raise ValueError(f"input shapes must match, got {a.shape} and {b.shape}")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("elementwise_add requires contiguous inputs")
    for name, value in (("xblock", xblock), ("yblock", yblock)):
        if value <= 0 or value & (value - 1):
            raise ValueError(f"{name} must be a positive power of two, got {value}")

    output = torch.empty_like(a)
    grid = (triton.cdiv(a.shape[0], xblock),)
    elementwise_add_kernel[grid](
        a,
        b,
        output,
        a.shape[0],
        a.shape[1],
        *a.stride(),
        *b.stride(),
        *output.stride(),
        xblock,
        yblock,
        num_warps=4,
    )
    return output


@pytest.mark.parametrize("shape", [(1, 1), (17, 257), (129, 4099)])
def test_elementwise_add(shape: tuple[int, int]) -> None:
    if not torch.cuda.is_available():
        pytest.skip("The Gluon smoke test requires a CUDA GPU")

    device = triton.runtime.driver.active.get_active_torch_device()
    torch.manual_seed(0)
    a = torch.randn(shape, device=device)
    b = torch.randn(shape, device=device)

    actual = elementwise_add(a, b)

    torch.testing.assert_close(actual, a + b, atol=0, rtol=0)


def test_elementwise_add_rejects_invalid_block_size() -> None:
    if not torch.cuda.is_available():
        pytest.skip("The Gluon smoke test requires a CUDA GPU")

    device = triton.runtime.driver.active.get_active_torch_device()
    a = torch.randn((8, 32), device=device)

    with pytest.raises(ValueError, match="positive power of two"):
        elementwise_add(a, a, yblock=96)
