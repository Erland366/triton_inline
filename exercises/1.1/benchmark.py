"""
Benchmark: Matmul Optimization
Module: 1.1 — Core Kernel Mastery

Target: >80% of cuBLAS throughput at M=N=K=4096
Sweep: M=N=K in [1024, 2048, 4096, 8192]

Usage: python exercises/1.1/benchmark.py
"""

import os
import torch
import triton
from exercise import matmul, reference_matmul

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# =============================================================================
# CORRECTNESS CHECK (fail fast)
# =============================================================================

def verify_before_benchmark():
    torch.manual_seed(42)
    a = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
    b = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
    ref = reference_matmul(a, b)
    ours = matmul(a, b)
    if not torch.allclose(ref, ours, atol=5e-2, rtol=1e-2):
        diff = torch.abs(ref - ours)
        print(f"CORRECTNESS FAILED — max diff: {diff.max().item():.6f}")
        print("Fix your kernel before benchmarking.")
        raise SystemExit(1)
    print("Correctness OK — starting benchmark.\n")


# =============================================================================
# BENCHMARK
# =============================================================================

configs = [
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[1024, 2048, 4096, 8192],
        line_arg="provider",
        line_vals=["cublas", "triton"],
        line_names=["cuBLAS", "Triton"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="TFLOPS",
        plot_name="matmul-performance-fp16",
        args={},
    )
]


@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "cublas":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.matmul(a, b), quantiles=quantiles
        )
    elif provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matmul(a, b), quantiles=quantiles
        )

    # 2 * M * N * K FLOPs for matmul (multiply + add per output element per K)
    tflops = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return tflops(ms), tflops(max_ms), tflops(min_ms)


# =============================================================================
# TARGET EVALUATION
# =============================================================================

def evaluate_target():
    """Check if we hit >80% of cuBLAS at M=N=K=4096."""
    M = N = K = 4096
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)

    cublas_ms = triton.testing.do_bench(lambda: torch.matmul(a, b))
    triton_ms = triton.testing.do_bench(lambda: matmul(a, b))

    tflops = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    cublas_tflops = tflops(cublas_ms)
    triton_tflops = tflops(triton_ms)
    ratio = triton_tflops / cublas_tflops * 100

    print(f"\n{'='*60}")
    print(f"TARGET EVALUATION (M=N=K=4096)")
    print(f"{'='*60}")
    print(f"cuBLAS:  {cublas_tflops:7.1f} TFLOPS ({cublas_ms:.3f} ms)")
    print(f"Triton:  {triton_tflops:7.1f} TFLOPS ({triton_ms:.3f} ms)")
    print(f"Ratio:   {ratio:.1f}% of cuBLAS")
    print(f"Target:  >80% of cuBLAS")
    print(f"Result:  {'PASS' if ratio > 80 else 'MISS'}")
    print(f"{'='*60}")

    return ratio, cublas_tflops, triton_tflops


if __name__ == "__main__":
    verify_before_benchmark()

    # Save results
    save_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "benchmark_results", "1.1",
    )
    os.makedirs(save_dir, exist_ok=True)

    benchmark.run(
        show_plots=False,
        print_data=True,
        save_path=save_dir,
    )

    ratio, cublas_tflops, triton_tflops = evaluate_target()
