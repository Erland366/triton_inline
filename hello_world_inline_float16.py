import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel_f16(
    x_ptr,
    y_ptr,
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    """
    Adds two float16 tensors using packed inline PTX assembly.
    """
    pid = tl.program_id(axis=0)
    # Create offsets for a block of data
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle the last block if N is not a multiple of BLOCK_SIZE
    mask = offsets < N

    # Load a block of data from x and y
    # Because dtype is float16 and pack=2, this will load two f16 values
    # at a time into a single 32-bit memory space for each lane.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # --- The Fix is Here ---
    # We change the constraint from "f" to "r".
    # 'f' specifies a 32-bit FLOAT register (.f32).
    # 'r' specifies a 32-bit general-purpose/INTEGER register (.u32/.s32).
    # The 'add.v2.f16' instruction expects its packed data to be in
    # general-purpose registers, not float registers. This resolves the
    # "Arguments mismatch" error from the PTX assembler.
    output = tl.inline_asm_elementwise(
        asm="add.f16 $0, $1, $2;",
        constraints="=r,r,r",
        args=[x, y],
        dtype=tl.float16,
        is_pure=True,
        # pack=2 tells Triton to handle two f16 elements per lane.
        pack=2
    )

    # Store the result back to the output tensor
    tl.store(output_ptr + offsets, output, mask=mask)

# --- Python Host Code ---
size = 128
# Ensure tensors are on the GPU and are float16
x = torch.randn(size, device='cuda', dtype=torch.float16)
y = torch.randn(size, device='cuda', dtype=torch.float16)
output = torch.empty_like(x)

# The grid launch is the same
grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
# The block size determines how many lanes (threads) operate in parallel.
# With pack=2, each lane processes two f16 elements, so a BLOCK_SIZE of 32
# will process 64 elements from the tensors per step.
add_kernel_f16[grid](x, y, output, x.numel(), BLOCK_SIZE=32)

# Verification
# Use a slightly higher tolerance for float16 arithmetic
assert torch.allclose(output, x + y, atol=1e-3, rtol=1e-3)

print("Inline assembly f16 addition successful!")
print("Input X:", x[:5].cpu().numpy())
print("Input Y:", y[:5].cpu().numpy())
print("Output: ", output[:5].cpu().numpy())
print("Expected:", (x + y)[:5].cpu().numpy())
