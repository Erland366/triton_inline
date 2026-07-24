# type: ignore
import math
from typing import Optional

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.language.extra.tlx.warp_spec import get_bufidx_phase
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def alloc_fn(size: int, align: int, stream: Optional[int]):
    assert align == 128
    assert stream == 0
    return torch.empty(size, dtype=torch.int8, device=DEVICE)

def matmul_tma_set_block_size_hook(nargs):
    BLOCK_M = nargs["BM"]
    BLOCK_N = nargs["BN"]
    BLOCK_K = nargs["BK"]
    NUM_MMA_GROUPS = nargs["NUM_MMA_GROUPS"]
    BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS
    NUM_CTAS = nargs.get("NUM_CTAS", 1)
    # For column-major inputs, TMA descriptor block shape matches the transposed view
    if nargs.get("A_ROW_MAJOR", True):
        nargs["a_desc"].block_shape = [BLOCK_M_SPLIT, BLOCK_K]
    else:
        nargs["a_desc"].block_shape = [BLOCK_K, BLOCK_M_SPLIT]
    if nargs.get("B_ROW_MAJOR", True):
        nargs["b_desc"].block_shape = [BLOCK_K, BLOCK_N // NUM_CTAS]
    else:
        nargs["b_desc"].block_shape = [BLOCK_N // NUM_CTAS, BLOCK_K]

    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", False)
    if EPILOGUE_SUBTILE:
        nargs["c_desc"].block_shape = [BLOCK_M_SPLIT, BLOCK_N // 2] # This is hardcoded since the code is designed for 2 stage
    else:
        nargs["c_desc"].block_shape = [BLOCK_M_SPLIT, BLOCK_N]

    # Add NUM_SMS
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    nargs["NUM_SMS"] = NUM_SMS

def _skinny_zero_c_hook(nargs):
    pass

def _get_skinny_autotune_configs():
    configs = []
    for bm, bn, bk in [
        (128, 64, 64),
        (128, 128, 64),
        (128, 256, 64),
        (64, 256, 64),
        (128, 256, 128),
        (128, 64, 128),
        (128, 128, 128),
        (64, 256, 128),
        (64, 64, 64),
        (64, 128, 64),
        (64, 64, 128),
        (64, 128, 128),
    ]:
        for s in [2, 3, 4]:
            if bk == 128 and s > 2:
                continue
            for gsm in [4, 8]:
                for nw in [4, 8]:
                    configs.append(
                        triton.Config(
                            dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_SIZE_M=gsm, NUM_STAGES=s),
                            num_warps=nw,
                            num_stages=1,
                            pre_hook=_skinny_zero_c_hook
                        )
                    )
    return configs

@triton.autotune(
    configs=_get_skinny_autotune_configs(),
    key=["M", "N", "K_LEN"]
)
@triton.jit
def _skinny_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K_LEN,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_ck, # Because we use split-K
    stride_cm,
    stride_cn,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_STAGES: tl.constexpr
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    k_start = pid_k * K_LEN
    a_ptr += k_start * stride_ak
    b_ptr += k_start * stride_bk

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    buffers_A = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), NUM_STAGES)
    buffers_B = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_ptr), NUM_STAGES)

    for i in tl.range(0, NUM_STAGES - 1, loop_unroll_factor=NUM_STAGES - 1):
        a_buf = tlx.local_view(buffers_A, i)
        b_buf = tlx.local_view(buffers_B, i)
        token_a = tlx.async_load(a_ptrs, a_buf, mask=offs_k[None, :] < K_LEN - i * BLOCK_K)
        token_b = tlx.async_load(b_ptrs, b_buf, mask=offs_k[:, None] < K_LEN - i * BLOCK_K)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        tlx.async_load_commit_group([token_a, token_b])

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K_LEN, BLOCK_K), num_stages=0):
        buf = k % NUM_STAGES
        a_k = tlx.local_view(buffers_A, buf)
        b_k = tlx.local_view(buffers_B, buf)

        tlx.async_load_wait_group(NUM_STAGES - 2)
        acc = tlx.async_dot(a_k, b_k, acc)

        i = k + NUM_STAGES - 1
        a_next = tlx.local_view(buffers_A, i % NUM_STAGES)
        b_next = tlx.local_view(buffers_B, i % NUM_STAGES)
        acc = tlx.async_dot_wait(1, acc)
        token_a = tlx.async_load(a_ptrs, a_next, mask=offs_k[None, :] < K_LEN - i * BLOCK_K)
        token_b = tlx.async_load(b_ptrs, b_next, mask=offs_k[:, None] < K_LEN - i * BLOCK_K)
        tlx.async_load_commit_group([token_a, token_b])
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    acc = tlx.async_dot_wait(0, acc)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    c = acc.to(tl.float16)
    if SPLIT_K > 1:
        c_ptrs = c_ptr + pid_k * stride_ck + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    else:
        c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=c_mask)


def _skinny_matmul(a, b, M, N, K):
    triton.set_allocator(alloc_fn)
    NUM_SMS = torch.cuda.get_device_properties(a.device).multi_processor_count

    tiles = math.ceil(M / 128) * math.ceil(N / 64)

    # I need to study more on this sections ngl
    split_k = 1
    k_blocks = K // 64
    target_sk = max(1, 2 * NUM_SMS // tiles)
    for sk in [128, 64, 32, 16, 8, 4, 2]:
        if sk <= target_sk and k_blocks // sk >= 16 and tiles * sk >= 8:
            split_k = sk
            break

    k_per_split = K // split_k
    if split_k > 1:
        c = torch.empty((split_k, M, N), dtype=torch.float16, device=a.device)
        stride_ck = M * N
    else:
        c = torch.empty((M, N), dtype=torch.float16, device=a.device)
        stride_ck = 0
        
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]), split_k)
    _skinny_matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        k_per_split,
        *a.stride(),
        *b.stride(),
        stride_ck,
        c.stride(-2),
        c.stride(-1),
        SPLIT_K=split_k
    )

    if split_k > 1:
        c = c.sum(dim=0)
    return c

def _skinny_tma_set_block_hook(nargs):
    BM = nargs["BLOCK_M"]
    BN = nargs["BLOCK_N"]
    BK = nargs["BLOCK_K"]
    nargs["a_desc"].block_shape = [BM, BK]
    nargs["b_desc"].block_shape = [BK, BN]

def _get_skinny_tma_configs():
    configs = []
    for bm, bn, bk in [
        (128, 64, 64),
        (128, 128, 64),
        (128, 256, 64),
        (64, 256, 64),
        (64, 64, 64),
        (64, 128, 64),
        (128, 64, 128),
        (128, 128, 128),
        (64, 64, 128),
        (64, 128, 128),
    ]:
        for s in [2, 3, 4]:
            if bk == 128 and s > 2:
                continue
            for gsm in [4, 8]:
                for nw in [4, 8]:
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_SIZE_M": gsm, "NUM_STAGES": s},
                            num_warps=nw,
                            num_stages=1,
                            pre_hook=_skinny_tma_set_block_hook,
                        ))
    return configs

@triton.autotune(
    configs=_get_skinny_tma_configs(),
    key=["M", "N", "K_LEN"]
)
@triton.jit
def _skinny_tma_kernel(
    a_desc,
    b_desc,
    c_ptr,
    M,
    N,
    K_LEN,
    stride_ck, # Because we use split-K
    stride_cm,
    stride_cn,
    K_START,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_STAGES: tl.constexpr
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    k_start = pid_k * K_LEN + K_START
    offset_am = pid_m * BLOCK_M
    offset_bn = pid_n * BLOCK_N

    buffers_A = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_desc), NUM_STAGES)
    buffers_B = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_desc), NUM_STAGES)

    bars_full_a = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    bars_full_b = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)

    num_k_iters = tl.cdiv(K_LEN, BLOCK_K)

    for i in tl.range(0, NUM_STAGES - 1, loop_unroll_factor=NUM_STAGES - 1):
        buf_a = tlx.local_view(buffers_A, i)
        buf_b = tlx.local_view(buffers_B, i)
        bar_a = tlx.local_view(bars_full_a, i)
        bar_b = tlx.local_view(bars_full_b, i)

        tlx.barrier_expect_bytes(bar_a, BLOCK_M * BLOCK_K * tlx.size_of(tlx.dtype_of(a_desc)))
        tlx.barrier_expect_bytes(bar_b, BLOCK_K * BLOCK_N * tlx.size_of(tlx.dtype_of(b_desc)))

        offset_k = k_start + i * BLOCK_K
        tlx.async_descriptor_load(a_desc, buf_a, [offset_am, offset_k], bar_a)
        tlx.async_descriptor_load(b_desc, buf_b, [offset_k, offset_bn], bar_b)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, num_k_iters, num_stages=0):
        buf = k % NUM_STAGES
        phase = (k // NUM_STAGES) & 1

        bar_a = tlx.local_view(bars_full_a, buf)
        bar_b = tlx.local_view(bars_full_b, buf)

        tlx.barrier_wait(bar=bar_a, phase=phase)
        tlx.barrier_wait(bar=bar_b, phase=phase)

        a_k = tlx.local_view(buffers_A, buf)
        b_k = tlx.local_view(buffers_B, buf)
        acc = tlx.async_dot(a_k, b_k, acc)

        next_i = k + NUM_STAGES - 1

        acc = tlx.async_dot_wait(1, acc)
        if next_i < num_k_iters:
            next_buf = next_i % NUM_STAGES
            buf_a_next = tlx.local_view(buffers_A, next_buf)
            buf_b_next = tlx.local_view(buffers_B, next_buf)
            bar_a_next = tlx.local_view(bars_full_a, next_buf)
            bar_b_next = tlx.local_view(bars_full_b, next_buf)


            tlx.barrier_expect_bytes(bar_a_next, BLOCK_M * BLOCK_K * tlx.size_of(tlx.dtype_of(a_desc)))
            tlx.barrier_expect_bytes(bar_b_next, BLOCK_K * BLOCK_N * tlx.size_of(tlx.dtype_of(b_desc)))

            offset_k = k_start + next_i * BLOCK_K
            tlx.async_descriptor_load(a_desc, buf_a_next, [offset_am, offset_k], bar_a_next)
            tlx.async_descriptor_load(b_desc, buf_b_next, [offset_k, offset_bn], bar_b_next)

    acc = tlx.async_dot_wait(0, acc)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    c = acc.to(tl.float16)
    if SPLIT_K > 1:
        c_ptrs = c_ptr + pid_k * stride_ck + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    else:
        c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, c, c_mask)

def _skinny_matmul_tma(a, b, M, N, K):
    triton.set_allocator(alloc_fn)
    NUM_SMS = torch.cuda.get_device_properties(a.device).multi_processor_count

    tiles = math.ceil(M / 128) * math.ceil(N / 64)

    # I need to study more on this sections ngl
    split_k = 1
    k_blocks = K // 64
    target_sk = max(1, 2 * NUM_SMS // tiles)
    for sk in [128, 64, 32, 16, 8, 4, 2]:
        if sk <= target_sk and k_blocks // sk >= 16 and tiles * sk >= 8:
            split_k = sk
            break

    k_per_split = K // split_k
    if split_k > 1:
        c = torch.empty((split_k, M, N), dtype=torch.float16, device=a.device)
        stride_ck = M * N
    else:
        c = torch.empty((M, N), dtype=torch.float16, device=a.device)
        stride_ck = 0
    
    dummy_block = [1, 1]
    desc_a = TensorDescriptor(a, shape=[M, K], strides=[K, 1], block_shape=dummy_block)
    desc_b = TensorDescriptor(b, shape=[K, N], strides=[N, 1], block_shape=dummy_block)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]), split_k)
    _skinny_tma_kernel[grid](
        desc_a,
        desc_b,
        c,
        M,
        N,
        k_per_split,
        stride_ck,
        c.stride(-2),
        c.stride(-1),
        K_START=0,
        SPLIT_K=split_k
    )
    if split_k > 1:
        c = c.sum(dim=0)
    return c



def main():
    configs= _get_skinny_autotune_configs()
    print(f"{configs = }")

if __name__ == "__main__":
    main()