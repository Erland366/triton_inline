# type: ignore
import pytest
import torch
import triton
from functools import partial
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

@gluon.jit
def memcpy_1d_kernel(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr, layout: gl.constexpr):
    pid = gl.program_id(0)
    start = pid * XBLOCK
    indices = gl.arange(0, XBLOCK, layout=layout)
    offsets = start + indices
    in_ptrs = in_ptr + offsets
    mask = offsets < xnumel
    value = gl.load(in_ptrs, mask=mask)
    out_ptrs = out_ptr + offsets
    gl.store(out_ptrs, value, mask=mask)



def get_throughput(inp: torch.Tensor, ms: int | float) -> float:
    tbytes = (2 * inp.numel() * inp.element_size() >> 30) / 1024
    return (tbytes / (ms * 1e-3))

def bench_memcpy_impl(inp, out, impl):
    compiled_kernel = impl(inp, out)
    fn = lambda: impl(inp, out)
    ms = triton.testing.do_bench(fn)
    return compiled_kernel, get_throughput(inp, ms)

def memcpy_1d_impl(inp, out, XBLOCK, layout, num_warps):
    xnumel = inp.numel()
    grid = (triton.cdiv(xnumel, XBLOCK), )
    compiled_kernel = memcpy_1d_kernel[grid](inp, out, xnumel, XBLOCK, layout, num_warps=num_warps)
    return compiled_kernel

@gluon.jit
def copy_scalar_kernel(in_ptr, out_ptr):
    value = gl.load(in_ptr)
    gl.store(out_ptr, value)

@gluon.jit
def memcpy_kernel(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr):
    pid = gl.program_id(0)
    start = pid * XBLOCK
    end = min(start + XBLOCK, xnumel)
    for i in range(start, end):
        value = gl.load(in_ptr + i)
        gl.store(out_ptr + i, value)

def memcpy(inp, out, XBLOCK: gl.constexpr):
    xnumel = inp.numel()
    grid = (triton.cdiv(xnumel, XBLOCK), )
    memcpy_kernel[grid](inp, out, xnumel, XBLOCK, num_warps=1)

@triton.autotune(
    configs=[triton.Config({"XBLOCK" : 2**i}, num_warps=1) for i in range(8, 14)],
    key=["xnumel"]
)
@gluon.jit
def memcpy_kernel_autotune(inp, out, xnumel, XBLOCK: gl.constexpr):
    memcpy_kernel(inp, out, xnumel, XBLOCK)

def memcpy_autotune(inp, out):
    xnumel = inp.numel()
    grid = lambda meta: (triton.cdiv(xnumel, meta["XBLOCK"]), )
    memcpy_kernel_autotune[grid](inp, out, xnumel)

def test_memcpy():
    a = torch.randn(2, 48, device="cuda", dtype=torch.bfloat16)
    b = torch.empty_like(a)
    memcpy_autotune(a, b)
    torch.testing.assert_close(a, b)

def copy_scalar(inp, out):
    grid = (1, )
    copy_scalar_kernel[grid](inp, out, num_warps=1)

def test_copy_scalar():
    a = torch.randn(1, device="cuda", dtype=torch.bfloat16)
    b = torch.empty_like(a)
    copy_scalar(a, b)
    torch.testing.assert_close(a, b)

def main():
    test_memcpy()

if __name__ == "__main__":
    main()
