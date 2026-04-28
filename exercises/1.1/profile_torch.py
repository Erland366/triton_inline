"""
Hardware Profile: Matmul Optimization (torch.profiler)
Module: 1.1

Uses torch.profiler with CUDA activity tracing to get per-kernel timing
and derive TFLOPS. This is an alternative to ncu when hardware counter
access is restricted (ERR_NVGPUCTRPERM).

Key Metrics: TFLOPS (derived from kernel timing), kernel duration
Provides: per-kernel timing comparison between Triton matmul and cuBLAS

Usage:
  python exercises/1.1/profile_nsight.py
"""

import os
import sys
import csv
import torch
from torch.profiler import profile, ProfilerActivity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exercise import matmul

DEVICE = "cuda"

SAVE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "benchmark_results", "1.1",
)
os.makedirs(SAVE_DIR, exist_ok=True)

SIZES = [1024, 2048, 4096, 8192]
NUM_RUNS = 10


def compute_tflops(time_us, n):
    """Derive TFLOPS from kernel time in microseconds."""
    flops = 2 * n * n * n
    return flops / (time_us * 1e-6) / 1e12


def profile_kernel(n, num_runs=NUM_RUNS):
    """Profile triton matmul vs cuBLAS at size n using torch.profiler."""
    a = torch.randn(n, n, device=DEVICE, dtype=torch.float16)
    b = torch.randn(n, n, device=DEVICE, dtype=torch.float16)

    # Warmup
    for _ in range(3):
        matmul(a, b)
        torch.matmul(a, b)
    torch.cuda.synchronize()

    # Profile Triton
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as triton_prof:
        for _ in range(num_runs):
            matmul(a, b)
            torch.cuda.synchronize()

    # Profile cuBLAS
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as cublas_prof:
        for _ in range(num_runs):
            torch.matmul(a, b)
            torch.cuda.synchronize()

    # Extract CUDA kernel events
    triton_events = [e for e in triton_prof.key_averages()
                     if e.device_type == torch.autograd.DeviceType.CUDA]
    cublas_events = [e for e in cublas_prof.key_averages()
                     if e.device_type == torch.autograd.DeviceType.CUDA]

    # Sum up CUDA kernel time (in microseconds)
    # device_time_total is the GPU-side time for CUDA events
    triton_total_us = sum(e.device_time_total for e in triton_events) / num_runs
    cublas_total_us = sum(e.device_time_total for e in cublas_events) / num_runs

    triton_tflops = compute_tflops(triton_total_us, n)
    cublas_tflops = compute_tflops(cublas_total_us, n)

    return {
        "n": n,
        "triton_time_us": triton_total_us,
        "cublas_time_us": cublas_total_us,
        "triton_TFLOPS": triton_tflops,
        "cublas_TFLOPS": cublas_tflops,
        "triton_pct_cublas": (triton_tflops / cublas_tflops * 100) if cublas_tflops > 0 else 0,
        "triton_kernels": [(e.key, e.device_time_total / num_runs) for e in triton_events],
        "cublas_kernels": [(e.key, e.device_time_total / num_runs) for e in cublas_events],
    }


def main():
    print("Matmul Profiling: Triton vs cuBLAS (torch.profiler)")
    print(f"Sizes: {SIZES}")
    print(f"Runs per size: {NUM_RUNS}")
    print(f"Output: {SAVE_DIR}\n")

    all_results = []

    for n in SIZES:
        print(f"{'='*60}")
        print(f"M=N=K={n}")
        print(f"{'='*60}")

        result = profile_kernel(n)
        all_results.append(result)

        print(f"  Triton:  {result['triton_TFLOPS']:.1f} TFLOPS  ({result['triton_time_us']:.0f} us/iter)")
        print(f"  cuBLAS:  {result['cublas_TFLOPS']:.1f} TFLOPS  ({result['cublas_time_us']:.0f} us/iter)")
        print(f"  Ratio:   {result['triton_pct_cublas']:.1f}% of cuBLAS")

        # Show kernel breakdown
        print(f"\n  Triton kernel breakdown:")
        for name, time_us in result["triton_kernels"]:
            print(f"    {name[:60]:60s}  {time_us:8.1f} us")
        print(f"\n  cuBLAS kernel breakdown:")
        for name, time_us in result["cublas_kernels"]:
            print(f"    {name[:60]:60s}  {time_us:8.1f} us")
        print()

    # Save CSV summary
    csv_path = os.path.join(SAVE_DIR, "torch_profiler_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "n", "triton_time_us", "cublas_time_us",
            "triton_TFLOPS", "cublas_TFLOPS", "triton_pct_cublas",
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: v for k, v in r.items()
                             if k not in ("triton_kernels", "cublas_kernels")})
    print(f"Summary saved to: {csv_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"{'n':>6}  {'Triton TFLOPS':>14}  {'cuBLAS TFLOPS':>14}  {'% cuBLAS':>10}")
    print(f"{'='*60}")
    for r in all_results:
        print(f"{r['n']:>6}  {r['triton_TFLOPS']:>14.1f}  {r['cublas_TFLOPS']:>14.1f}  {r['triton_pct_cublas']:>9.1f}%")

    print("\n--- Interpretation ---")
    print("This uses torch.profiler (CUDA event timing), not hardware counters.")
    print("TFLOPS derived from: 2*M*N*K / kernel_time")
    print("For hardware counters (DRAM%, SM%), ncu access is needed.")
    print(f"  ncu status: ERR_NVGPUCTRPERM (perf_event_paranoid=4, no CAP_SYS_ADMIN)")


if __name__ == "__main__":
    main()
