# type: ignore
import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

import pytest
from pathlib import Path

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def get_cuda_autotune_config():
    return [
        triton.Config(
            {"BLOCK_SIZE_M" : 128, "BLOCK_SIZE_N" : 256, "BLOCK_SIZE_K" : 64, "GROUP_SIZE_M" : 8, "NUM_STAGES" : 3},
            num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_M" : 64, "BLOCK_SIZE_N" : 64, "BLOCK_SIZE_K" : 64, "GROUP_SIZE_M" : 8, "NUM_STAGES" : 3},
            num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M" : 64, "BLOCK_SIZE_N" : 128, "BLOCK_SIZE_K" : 64, "GROUP_SIZE_M" : 8, "NUM_STAGES" : 3},
            num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M" : 128, "BLOCK_SIZE_N" : 128, "BLOCK_SIZE_K" : 64, "GROUP_SIZE_M" : 8, "NUM_STAGES" : 3},
            num_warps=8
        ),
    ]

@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=["M", "N", "K"]
)
@triton.jit
def matmul_kernel_pipelined_hopper(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_STAGES: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    where_in_group = pid % num_pid_in_group
    which_row = where_in_group % group_size_m
    pid_m = first_pid_m + which_row
    pid_n = where_in_group // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Allocate NUM_STAGES buffers
    buffers_A = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_K), tlx.dtype_of(a_ptr), NUM_STAGES)
    buffers_B = tlx.local_alloc((BLOCK_SIZE_K, BLOCK_SIZE_N), tlx.dtype_of(b_ptr), NUM_STAGES)

    for i in tl.range(0, NUM_STAGES - 1, loop_unroll_factor=NUM_STAGES - 1):
        a = tlx.local_view(buffers_A, i)
        b = tlx.local_view(buffers_B, i)
        token_a = tlx.async_load(a_ptrs, a, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K)
        token_b = tlx.async_load(b_ptrs, b, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        tlx.async_load_commit_group([token_a, token_b])

    # main K loop
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Disable auto-pipelining with num_stages=0
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), num_stages=0):
        # Select the buffer for the current iteration
        buf = k % NUM_STAGES
        a_k = tlx.local_view(buffers_A, buf)
        b_k = tlx.local_view(buffers_B, buf)

        # wait for buffers to be ready.
        tlx.async_load_wait_group(NUM_STAGES - 2)

        # Run MMA
        acc = tlx.async_dot(a_k, b_k, acc)

        # Prefetch the iteration NUM_STAGES - 1 blocks ahead
        i = k + NUM_STAGES - 1
        a_next = tlx.local_view(buffers_A, i % NUM_STAGES)
        b_next = tlx.local_view(buffers_B, i % NUM_STAGES)

        # Wait for the previous MMA using this buffer to complete
        acc = tlx.async_dot_wait(1, acc)

        # Prefetch
        token_a = tlx.async_load(a_ptrs, a_next, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K)
        token_b = tlx.async_load(b_ptrs, b_next, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K)

        tlx.async_load_commit_group([token_a, token_b])
        # Advance ptrs to the next block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Wait for the last MMA to complete
    acc = tlx.async_dot_wait(0, acc)
    c = acc.to(tlx.dtype_of(c_ptr))
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def matmul_hopper(a, b, config=None):
    assert a.shape[1] == b.shape[0], "Incompatible alignment"
    assert a.is_contiguous() and b.is_contiguous()
    M, K = a.shape
    K, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=a.dtype)

    if config is not None:
        grid = (triton.cdiv(M, config["BLOCK_SIZE_M"]) 
                * triton.cdiv(N, config["BLOCK_SIZE_N"]), )
        matmul_kernel_pipelined_hopper[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            *a.stride(),
            *b.stride(),
            *c.stride(),
            **config
        )
    else:
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]) 
                             * triton.cdiv(N, meta["BLOCK_SIZE_N"]), )
        matmul_kernel_pipelined_hopper[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            *a.stride(),
            *b.stride(),
            *c.stride(),
        )
    return c

def verify_before_benchmark():
    torch.manual_seed(42)

    kwargs = dict(device=DEVICE, dtype=torch.bfloat16)
    a = torch.randn(513, 1024, **kwargs)
    b = torch.randn(1024, 2048, **kwargs)

    tlx_result = matmul_hopper(a, b)
    torch_result = a @ b

    torch.testing.assert_close(tlx_result, torch_result)

configs = [
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[512, 1024, 2048, 4096, 8192],
        x_log=True,
        line_arg="provider",
        line_vals=["torch", "TLX"],
        line_names=["Torch", "TLX"],
        ylabel="TFLOP/s",
        plot_name="matmul_performance_tlx",
        args={}
    )
]

@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.bfloat16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.bfloat16)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        result = a @ b
        torch.cuda.synchronize()

        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: a @ b, quantiles=quantiles, warmup=2000, rep=2000,
        )
    elif provider == "TLX":
        result = matmul_hopper(a, b)
        torch.cuda.synchronize()

        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matmul_hopper(a, b), quantiles=quantiles, warmup=2000, rep=2000,
        )
    
    tflops = lambda ms: 2 * M * K * N * 1e-12 / (ms * 1e-3)
    return tflops(ms), tflops(max_ms), tflops(min_ms)

@pytest.mark.parametrize("M", [1024, 2048])
@pytest.mark.parametrize("K", [1024])
@pytest.mark.parametrize("N", [1024, 2048])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_matmul_hopper(dtype, N, K, M):
    torch.manual_seed(42)

    kwargs = dict(device=DEVICE, dtype=dtype)
    a = torch.randn(M, K, **kwargs)
    b = torch.randn(K, N, **kwargs)

    tlx_result = matmul_hopper(a, b)
    torch_result = a @ b

    torch.testing.assert_close(tlx_result, torch_result)


def blocked_matmul_pseudo(A, B, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M):
    M, K = A.shape
    K, N = B.shape
    C = torch.zeros(M, N)

    assert M % BLOCK_SIZE_M == 0, "Assuming M divisible by BLOCK_SIZE_M"
    assert N % BLOCK_SIZE_N == 0, "Assuming M divisible by BLOCK_SIZE_M"

    num_block_rows = M // BLOCK_SIZE_M
    num_block_cols = N // BLOCK_SIZE_N

    num_pids = list(range(num_block_rows * num_block_cols))

    for i in num_pids:
        block_m = i // num_block_cols
        block_n = i % num_block_cols

        m = block_m * BLOCK_SIZE_M
        n = block_n * BLOCK_SIZE_N

        block_A = A[m:m+BLOCK_SIZE_M, :]
        block_B = B[:, n:n+BLOCK_SIZE_N]
        C[m:m+BLOCK_SIZE_M, n:n+BLOCK_SIZE_N] = block_A @ block_B
    
    return C

def grouped_blocked_matmul_pseudo(A, B, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M):
    M, K = A.shape
    K, N = B.shape
    C = torch.zeros(M, N)

    assert M % BLOCK_SIZE_M == 0, "Assuming M divisible by BLOCK_SIZE_M"
    assert N % BLOCK_SIZE_N == 0, "Assuming M divisible by BLOCK_SIZE_M"

    num_block_rows = M // BLOCK_SIZE_M
    num_block_cols = N // BLOCK_SIZE_N

    num_pids = list(range(num_block_rows * num_block_cols))

    pid_visual = torch.zeros(num_block_rows, num_block_cols)

    for i in num_pids:
        num_pid_in_group = GROUP_SIZE_M * num_block_cols
        
        # Which group are we in
        group_id = i // num_pid_in_group

        # starting row in m dimension for this specific group
        first_pid_m = group_id * GROUP_SIZE_M

        # what is the true group size incase not divisible
        group_size_m = min(num_block_rows - first_pid_m, GROUP_SIZE_M)

        where_in_group = i % num_pid_in_group
        which_row = where_in_group % GROUP_SIZE_M

        # This is the row within the group
        pid_m = first_pid_m + which_row
        pid_n = where_in_group // group_size_m

        pid_visual[pid_m, pid_n] = i

        m = pid_m * BLOCK_SIZE_M
        n = pid_n * BLOCK_SIZE_N

        block_A = A[m:m+BLOCK_SIZE_M, :]
        block_B = B[:, n:n+BLOCK_SIZE_N]

        C[m:m+BLOCK_SIZE_M, n:n+BLOCK_SIZE_N] = block_A @ block_B

    return C, pid_visual    

def main():
    # A = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    # B = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)

    # C, pid_visual = grouped_blocked_matmul_pseudo(A, B, 4, 4, 2)
    # print(f"{pid_visual = }")
    # print(f"{C = }")
    verify_before_benchmark()
    save_path = Path("plots")
    save_path.mkdir(exist_ok=True)
    benchmark.run(
        print_data=True,
        show_plots=False,
        save_path=save_path
    )


if __name__ == "__main__":
    main()
