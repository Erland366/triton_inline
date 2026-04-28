import torch
import triton
import triton.language as tl

# Write the correct triton inline asm elementwise for converting from float to half
@triton.jit
def float_to_half(
    x_ptr,
    y_ptr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)

    y = tl.inline_asm_elementwise(
        asm="cvt.rn.f16.f32 $0, $1;",
        constraints="=h,f",
        args=[x],
        dtype=tl.float16,
        is_pure=True,
        pack=1
    )
    tl.store(y_ptr + offsets, y)

@triton.jit
def kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    a = tl.load(A_ptr + offsets) # uint8
    b = tl.load(B_ptr + offsets)

    # For each (a, b) in zip(a, b), perform the following:
    # - Let ai be `a` converted to int32
    # - Let af be `a` converted to float
    # - Let m be the max of `ai` and b
    # - Return ai and mi
    # Do the above 4 elements at a time
    (c, d) = tl.inline_asm_elementwise(
        asm="""
        {
            // Unpack `a` into `ai`
            .reg .b8 tmp<4>;
            mov.b32 {tmp0, tmp1, tmp2, tmp3}, $8;
            cvt.u32.u8 $0, tmp0;
            cvt.u32.u8 $1, tmp1;
            cvt.u32.u8 $2, tmp2;
            cvt.u32.u8 $3, tmp3;
        }
        // Convert `ai` to float.
        cvt.rn.f32.s32 $4, $0;
        cvt.rn.f32.s32 $5, $1;
        cvt.rn.f32.s32 $6, $2;
        cvt.rn.f32.s32 $7, $3;

        // Take max of `ai` and `b`
        max.f32 $4, $4, $9;
        max.f32 $5, $5, $10;
        max.f32 $6, $6, $11;
        max.f32 $7, $7, $12;
        """,
        constraints=(
            # 8 Output registers, namely
            # $0=ai0, $1=ai1, $2=ai2, $3=ai3,
            # $4=m0, $5=m1, $6=m2, $7=m3
            "=r,=r,=r,=r,=r,=r,=r,=r"
            # 5 input registers, mainly
            # $8=ai,
            # $9=b0, $10=b1, $11=b2, $12=b3
            # The four elements from `a` are all packed into one register
            "r,r,r,r,r"
        ),
        args=(a, b),
        dtype=(tl.int32, tl.float32),
        is_pure=True,
        pack=4
    )


def test_float_to_half():
    size = 64
    x = torch.randn(size, device='cuda', dtype=torch.float32)
    # put x[0] as the maximum value of float32 to test the conversion
    x[0] = torch.finfo(torch.float32).max
    y = torch.empty(size, device='cuda', dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
    float_to_half[grid](x, y, BLOCK_SIZE=32)
    # Verification
    print(f"{x[:10] = }")
    print(f"{y[:10] = }")
    assert torch.allclose(y, x.to(torch.bfloat16), atol=1e-3)

@triton.jit
def unpack_3_input_of_float32_into_bf16(
    A_ptr, # float32
    B_ptr, # float32
    C_ptr, # float32
    D_ptr, # bf16
    E_ptr, # bf16
    F_ptr, # bf16
    G_ptr, # bf16
    H_ptr, # bf16
    I_ptr, # bf16
    N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(A_ptr + offsets, mask=mask)
    b = tl.load(B_ptr + offsets, mask=mask)
    c = tl.load(C_ptr + offsets, mask=mask)
    # Unpack the 3 input of float32 into bf16
    (d, e, f, g, h, i) = tl.inline_asm_elementwise(
        asm="""
        {
            .reg .b16 tmp<6>;
            mov.b32 {tmp0, tmp1}, $6;
            cvt.rn.bf16.f32 $0, tmp0;
            cvt.rn.bf16.f32 $1, tmp1;

            mov.b32 {tmp2, tmp3}, $7;
            cvt.rn.bf16.f32 $2, tmp2;
            cvt.rn.bf16.f32 $3, tmp3;

            mov.b32 {tmp4, tmp5}, $8;
            cvt.rn.bf16.f32 $4, tmp4;
            cvt.rn.bf16.f32 $5, tmp5;
        }
        """,
        constraints="=h,=h,=h,=h,=h,=h,f,f,f",
        args=(a, b, c),
        dtype=(tl.bfloat16, tl.bfloat16, tl.bfloat16,
               tl.bfloat16, tl.bfloat16, tl.bfloat16),
        is_pure=True,
        pack=1
    )

    tl.store(D_ptr + offsets, d, mask=mask)
    tl.store(E_ptr + offsets, e, mask=mask)
    tl.store(F_ptr + offsets, f, mask=mask)
    tl.store(G_ptr + offsets, g, mask=mask)
    tl.store(H_ptr + offsets, h, mask=mask)
    tl.store(I_ptr + offsets, i, mask=mask)

def test_unpack_3_input_of_float32_into_bf16():
    size = 64
    a = torch.randn(size, device='cuda', dtype=torch.float32)
    b = torch.randn(size, device='cuda', dtype=torch.float32)
    c = torch.randn(size, device='cuda', dtype=torch.float32)
    d = torch.empty(size, device='cuda', dtype=torch.bfloat16)
    e = torch.empty(size, device='cuda', dtype=torch.bfloat16)
    f = torch.empty(size, device='cuda', dtype=torch.bfloat16)
    g = torch.empty(size, device='cuda', dtype=torch.bfloat16)
    h = torch.empty(size, device='cuda', dtype=torch.bfloat16)
    i = torch.empty(size, device='cuda', dtype=torch.bfloat16)

    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
    unpack_3_input_of_float32_into_bf16[grid](
        a, b, c, d, e, f, g, h, i, a.numel(),
        BLOCK_SIZE=32
    )

    # Verification
    print(f"{a[:10] = }")
    print(f"{b[:10] = }")
    print(f"{c[:10] = }")
    print(f"{d[:10] = }")
    print(f"{e[:10] = }")
    print(f"{f[:10] = }")
    print(f"{g[:10] = }")
    print(f"{h[:10] = }")
    print(f"{i[:10] = }")

test_unpack_3_input_of_float32_into_bf16()