import torch
import triton
import triton.language as tl
import lovely_tensors as lt; lt.monkey_patch()


@triton.jit
def normal_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    N,
    BLOCK_SIZE: tl.constexpr = 1024,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=1.0)

    c = a / b
    tl.store(c_ptr + offs, c, mask=mask)

@triton.jit
def _reciprocal_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    N,
    BLOCK_SIZE: tl.constexpr = 1024,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=1.0)

    (multiplier, ) = tl.inline_asm_elementwise(
        asm="rcp.approx.ftz.f32 $0, $1;",
        constraints="=r,r",
        args=[b],
        dtype=[tl.float32],
        is_pure=True,
        pack=1
    )

    c = a * multiplier
    tl.store(c_ptr + offs, c, mask=mask)

def div (a, b, reciprocal: bool = True):
    N = a.numel()
    out = torch.empty_like(a)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]), )
    if reciprocal:
        K = _reciprocal_kernel[grid](a, b, out, N)
    else:
        K = normal_kernel[grid](a, b, out, N)
    return out, K

def main():
    import torch

    # Create two large random matrices on the GPU
    size = 10000
    a = torch.randn(size, device="cuda")
    b = torch.randn(size, device="cuda")

    # Perform element-wise division using the Triton kernel
    result_triton, K = div(a, b, reciprocal=False)
    print(f"{result_triton = }")

    result_reciprocal, K = div(a, b, reciprocal=True)
    print(f"{result_reciprocal = }")

    result_torch = torch.div(a, b)
    print(f"{result_torch = }")

    torch.testing.assert_close(result_torch, result_triton)
    torch.testing.assert_close(result_torch, result_reciprocal)

    # print("K assembly")
    # print(K.asm["ptx"])


if __name__ == "__main__":
    main()
