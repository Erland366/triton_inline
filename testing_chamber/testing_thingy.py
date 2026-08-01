import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_dm, stride_dn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        accumulator = tl.dot(a, b, accumulator, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = tl.load(c_ptrs)
    d = -accumulator + c
    d_ptrs = d_ptr + offs_m[:, None] * stride_dm + offs_n[None, :] * stride_dn
    tl.store(d_ptrs, d)

def run(M, N, K):
    A = torch.zeros((M, K), dtype=torch.float16).cuda()
    B = torch.zeros((K, N), dtype=torch.float16).cuda()
    C = torch.full((M, N), 2**20, dtype=torch.float32).cuda()
    D = torch.zeros((M, N), dtype=torch.float32).cuda()

    # Same values as the paper
    A[0, 0] = 2**10;  A[0, 1] = 2**-2
    A[1, 0] = 2**10;  A[1, 1] = 2**-2
    B[0, 0] = 2**10;  B[0, 1] = 2**-3
    B[1, 0] = 2**-3;  B[1, 1] = 2**-3

    grid = lambda META: (triton.cdiv(M, 16), triton.cdiv(N, 16))
    matmul_kernel[grid](A, B, C, D, M, N, K,
                        A.stride(0), A.stride(1),
                        B.stride(0), B.stride(1),
                        C.stride(0), C.stride(1),
                        D.stride(0), D.stride(1),
                        BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=16)
    print(f"D[0,0] = {D[0,0].item()}")

run(2, 2, 2)
