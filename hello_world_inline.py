import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = tl.inline_asm_elementwise(
        asm="add.v2.f16 $0, $1, $2;",
        constraints="=f,f,f",
        args=[x, y],
        dtype=tl.float16,
        is_pure=True,
        pack=2
    )
    tl.store(output_ptr + offsets, output, mask=mask)

size = 128
x = torch.randn(size, device='cuda', dtype=torch.float16)
y = torch.randn(size, device='cuda', dtype=torch.float16)
output = torch.empty_like(x)

grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
add_kernel[grid](x, y, output, x.numel(), BLOCK_SIZE=32)

# Verification
assert torch.allclose(output, x + y, atol=1e-3)
print("Inline assembly addition successful!")
print("Input X:", x[:5].cpu().numpy())
print("Input Y:", y[:5].cpu().numpy())
print("Output:", output[:5].cpu().numpy())