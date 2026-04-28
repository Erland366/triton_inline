"""
Benchmark: Fused Softmax & LayerNorm
Module: 1.2
Mode: Student-written (post-1.1b)

Write your benchmark following the triton.testing.Benchmark pattern.
See profiling-guide.md and your Module 1.1b work for reference.

Target: Beat PyTorch eager by >2x on softmax for (8192, 4096) input
"""

import torch
import triton
import triton.testing

from exercise import fused_softmax, fused_layernorm_fwd, reference_softmax, reference_layernorm

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# TODO 1: Correctness gate
# Before benchmarking, verify your kernel is correct.
# Call reference and your wrapper, compare with torch.allclose.
# If incorrect, print an error and exit — never benchmark a broken kernel.


# TODO 2: Define Benchmark configuration for SOFTMAX
# Create a triton.testing.Benchmark with:
#   - x_names: ["N"] (sweep column width)
#   - x_vals: [128 * i for i in range(2, 100)]  (or similar range)
#   - line_arg: "provider"
#   - line_vals: ["triton", "torch"]
#   - line_names: ["Your Softmax", "torch.softmax"]
#   - ylabel: "GB/s"  (bandwidth-bound — use GB/s not TFLOPS)
#   - args: {"M": 4096}  (fixed rows)


# TODO 3: Implement the perf_report function for softmax
# @triton.testing.perf_report([<your Benchmark config>])
# def bench_softmax(M, N, provider):
#     - Create input tensor (M, N) FP16 on DEVICE
#     - Select kernel based on provider
#     - Use triton.testing.do_bench to measure
#     - Convert ms to GB/s: gbps = 2 * M * N * element_size * 1e-9 / (ms * 1e-3)
#       (2x because read input + write output)
#     - Return the metric value


# TODO 4: (Optional) Define a second Benchmark for LAYERNORM
# Similar to softmax but with weight and bias tensors.
# For backward pass: gbps = 3 * M * N * element_size * 1e-9 / (ms * 1e-3)
# (3x because read input + read dy + write dx)


# TODO 5: Run and save
# if __name__ == "__main__":
#     bench_softmax.run(save_path="./benchmark_results/1.2/", print_data=True)
#     print("Target: >2x over PyTorch for softmax at (8192, 4096)")
