"""
Hardware Profile: Matmul Optimization
Module: 1.1
Profiling Tier: Proton (triton.profiler)

Key Metrics (from curriculum): TFLOPS, DRAM throughput %, arithmetic intensity
Diagnosis: compute-bound kernel; high DRAM% indicates poor tiling.

Collects TFLOPS and bandwidth via Proton scope annotations.
See profiling-guide.md for metric interpretation.
"""

import os
import torch
import triton.profiler as proton
import triton.profiler.viewer as proton_viewer
from contextlib import contextmanager

from exercise import matmul, reference_matmul

DEVICE = torch.device("cuda")


@contextmanager
def proton_context():
    proton.activate(0)
    try:
        yield
    finally:
        proton.deactivate(0)


# =============================================================================
# PROFILING
# =============================================================================

def profile_kernel():
    """Profile using Proton scope annotations with FLOPS/bytes metadata."""
    save_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "benchmark_results", "1.1",
    )
    os.makedirs(save_dir, exist_ok=True)
    profile_name = os.path.join(save_dir, "profile")

    sizes = [1024, 2048, 4096, 8192]
    bytes_per_elem = 2  # FP16
    reps = 10

    # Warm up autotuner for all sizes BEFORE profiling.
    # The autotuner tries all configs on the first call per (M, N, K),
    # which pollutes the profile with overhead.
    print("Warming up autotuner for all sizes...")
    for size in sizes:
        M = N = K = size
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
        b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
        matmul(a, b)  # triggers autotuning
        torch.cuda.synchronize()
    print("Warmup done. Starting profiled runs.\n")

    proton.start(profile_name, hook="triton")
    proton.deactivate(0)

    for size in sizes:
        M = N = K = size
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
        b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)

        flops = 2.0 * M * N * K
        bytes_ = bytes_per_elem * (M * K + N * K + M * N)

        with proton_context():
            for _ in range(reps):
                with proton.scope(
                    f"cublas [M=N=K={size}]",
                    {"bytes": bytes_, "flops16": flops},
                ):
                    reference_matmul(a, b)

                with proton.scope(
                    f"triton [M=N=K={size}]",
                    {"bytes": bytes_, "flops16": flops},
                ):
                    matmul(a, b)

    proton.finalize()
    return profile_name


def show_profile(profile_name):
    """Display profiling results as a tree."""
    metric_names = ["tflop16/s", "time/ms"]
    tree, metrics = proton_viewer.parse(metric_names, f"{profile_name}.hatchet")
    proton_viewer.print_tree(tree, metrics)


def main():
    profile_name = profile_kernel()
    print("\n" + "=" * 70)
    print("PROTON PROFILE RESULTS")
    print("=" * 70)
    show_profile(profile_name)

    print("\n--- Interpretation ---")
    print("Module 1.1 key metrics: TFLOPS, DRAM throughput %, arithmetic intensity")
    print("This is a compute-bound kernel.")
    print("  - High tflop16/s near cuBLAS = well-optimized tiling")
    print("  - If tflop16/s is much lower = check tiling, L2 super-grouping")
    print("  - Compare triton vs cublas lines at each size")
    print(f"\nProfile saved to: {profile_name}.hatchet")


if __name__ == "__main__":
    main()
