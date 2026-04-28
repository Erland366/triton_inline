"""
Sub-exercise 1: Write a Benchmark from Scratch
Module: 1.1b — Benchmarking, Profiling & Debugging Toolkit

Objectives:
1. Create a triton.testing.Benchmark config that sweeps M=N=K
2. Use do_bench to measure kernel execution time
3. Convert milliseconds to TFLOPS
4. Compare your Module 1.1 matmul against cuBLAS

Acceptance Criteria:
1. Benchmark runs without errors
2. Produces TFLOPS comparison chart (triton vs cuBLAS)
3. Results saved to benchmark_results/1.1b/

Instructions:
1. Complete all TODO sections below
2. Run: python exercises/1.1b/exercise.py
3. Results will be saved to benchmark_results/1.1b/
"""

import os
import sys
import torch
import triton
import triton.testing

# Import YOUR matmul from Module 1.1
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "1.1"))
from exercise import matmul

DEVICE = triton.runtime.driver.active.get_active_torch_device()

SAVE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "benchmark_results", "1.1b",
)
os.makedirs(SAVE_DIR, exist_ok=True)

torch.manual_seed(0)


# =============================================================================
# TODO 1: Correctness gate
# =============================================================================
# Before benchmarking, verify your kernel still works.
# Create two random FP16 matrices (e.g., 512x512), run both matmul() and
# torch.matmul(), and check with torch.allclose(atol=1e-2).
# If incorrect, print an error and sys.exit(1).
#
# Why: Never benchmark a broken kernel — the numbers are meaningless.
def correctness():
    config = dict(dtype=torch.float16, device=DEVICE)
    a = torch.randn((512, 512), **config)
    b = torch.randn((512, 512), **config)

    torch_result = a @ b
    triton_result = matmul(a, b)

    if not torch.allclose(torch_result, triton_result, atol=1e-2):
        print("Fail")
        sys.exit(1)
    print("Correct!")


# =============================================================================
# TODO 2: Define the Benchmark configuration
# =============================================================================
# Create a triton.testing.Benchmark with:
#   x_names=["M", "N", "K"]          — sweep all three dimensions together
#   x_vals=[512, 1024, 2048, 4096]   — four sizes
#   line_arg="provider"               — selects which kernel to run
#   line_vals=["triton", "torch"]     — your kernel vs cuBLAS
#   line_names=["Your Matmul", "cuBLAS"]
#   styles=[("blue", "-"), ("red", "-")]
#   ylabel="TFLOPS"
#   plot_name="1.1b-matmul-benchmark"
#   args={}                           — no fixed args needed
#
# Wrap it in @triton.testing.perf_report(...)
perf_report = triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[512, 1024, 2048, 4096],
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton Matmul", "cuBLAS"],
        styles=[("blue", "-"), ("red", "-")],
        ylabel="TFLOPS",
        plot_name="1.1b-matmul-benchmark",
        args={}
    )
)


# =============================================================================
# TODO 3: Implement the benchmark function
# =============================================================================
# The function signature must match the Benchmark config:
#   def benchmark(M, N, K, provider):
#
# Inside:
#   1. Create random FP16 input matrices a (M, K) and b (K, N) on DEVICE
#   2. Branch on provider:
#      - "triton": measure with do_bench(lambda: matmul(a, b))
#      - "torch":  measure with do_bench(lambda: torch.matmul(a, b))
#   3. Convert ms to TFLOPS: 2 * M * N * K / (ms * 1e-3) / 1e12
#   4. Return the TFLOPS value
#
# Hint: do_bench returns milliseconds. Use quantiles=[0.5, 0.2, 0.8]
# to get median/min/max, and return (perf(ms), perf(max_ms), perf(min_ms)).
@perf_report
def benchmark(M, N, K, provider):
    config = dict(dtype=torch.float16, device=DEVICE)
    a = torch.randn((M, K), **config)
    b = torch.randn((K, N), **config)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: a @ b, quantiles=quantiles)
    elif provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K / (ms * 1e-3) / 1e12
    return perf(ms), perf(max_ms), perf(min_ms)

# =============================================================================
# TODO 4: Run and save
# =============================================================================
if __name__ == "__main__":
    correctness()
    benchmark.run(save_path=SAVE_DIR, print_data=True)
    print(f"\nResults saved to: {SAVE_DIR}")


# =============================================================================
# HINTS
# =============================================================================

# --- Hint 1 (Direction) ---
# The Benchmark config is declarative — you describe WHAT to sweep, not HOW.
# The perf_report decorator handles the loop, timing, and chart generation.
# Your function just needs to: create data, time the kernel, return TFLOPS.

# --- Hint 2 (Approach) ---
# Look at tutorial 03 lines 404-442 for a complete working example.
# The key pattern is:
#   @triton.testing.perf_report([triton.testing.Benchmark(...)])
#   def benchmark(M, N, K, provider):
#       ...
#       ms = triton.testing.do_bench(lambda: kernel(a, b))
#       return 2 * M * N * K * 1e-12 / (ms * 1e-3)

# --- Hint 3 (Near-solution) ---
# @triton.testing.perf_report([
#     triton.testing.Benchmark(
#         x_names=["M", "N", "K"],
#         x_vals=[512, 1024, 2048, 4096],
#         line_arg="provider",
#         line_vals=["triton", "torch"],
#         line_names=["Your Matmul", "cuBLAS"],
#         styles=[("blue", "-"), ("red", "-")],
#         ylabel="TFLOPS",
#         plot_name="1.1b-matmul-benchmark",
#         args={},
#     )
# ])
# def benchmark(M, N, K, provider):
#     a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
#     b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
#     quantiles = [0.5, 0.2, 0.8]
#     if provider == "triton":
#         ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
#     elif provider == "torch":
#         ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
#     perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
#     return perf(ms), perf(max_ms), perf(min_ms)
#
# if __name__ == "__main__":
#     benchmark.run(save_path=SAVE_DIR, print_data=True)
