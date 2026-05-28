# type: ignore
"""
Benchmark: TLX Vector Add With Async Tasks
Module: TLX
Mode: Student-written

Write this benchmark after `exercise.py` passes correctness.

Target:
- Compare PyTorch, normal Triton, and TLX warp-specialized dual-add throughput.
- Use this benchmark to observe whether async task partitioning helps or hurts
  this toy bandwidth-bound kernel.
"""

import torch
import triton
import triton.testing

from exercise import add2, add2_warp_specialized, reference_dual_add, verify_correctness

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# TODO 1: Correctness gate
# Before benchmarking, call `verify_correctness()`.
# Never benchmark a kernel that is still incorrect.
def verify_before_benchmark():
  torch.manual_seed(42)
  kwargs = dict(device=DEVICE, dtype=torch.bfloat16)

  x = torch.randn((512, 512), **kwargs)
  y = torch.randn((512, 512), **kwargs)
  a = torch.randn((512, 512), **kwargs)
  b = torch.randn((512, 512), **kwargs)

  ref1, ref2 = reference_dual_add(x, y, a, b)
  our1_tl, our2_tl = add2(x, y, a, b)
  our1_tlx, our2_tlx = add2_warp_specialized(x, y, a, b)
  if not torch.allclose(ref1, our1, atol=5e-2, rtol=1e-2) and :
      diff = torch.abs(ref - ours)
      print(f"CORRECTNESS FAILED — max diff: {diff.max().item():.6f}")
      print("Fix your kernel before benchmarking.")
      raise SystemExit(1)
  print("Correctness OK — starting benchmark.\n")


# TODO 2: Define a triton.testing.Benchmark config.
# Suggested shape:
#   - x_names: ["size"]
#   - x_vals: [2**i for i in range(12, 28)]
#   - x_log: True
#   - line_arg: "provider"
#   - line_vals: ["triton", "triton_ws", "torch"]
#   - line_names: ["Triton", "Triton_WS", "Torch"]
#   - ylabel: "GB/s"
#   - plot_name: "vector-add-performance"
configs = [
   triton.testing.Benchmark(
      x_names=["size"],
      x_vals=[2**i for i in range(12, 28)],
      x_log=True,
      line_arg="provider",
      line_val=["triton", "triton_ws", "torch"],
      line_names=["Triton", "Triton_WS", "Torch"],
      ylabel="GB/s",
      plot_name=""
   )
]


# TODO 3: Implement the perf_report function.
# For each provider:
#   - allocate x, y, a, b on DEVICE
#   - benchmark the selected implementation with `triton.testing.do_bench`
#   - convert milliseconds to GB/s
#
# Memory-traffic question:
#   How many element-sized reads and writes should this dual-add count?
#   Decide this before filling in the GB/s formula.


# TODO 4: Run and save.
# if __name__ == "__main__":
#     verify_correctness()
#     benchmark.run(
#         print_data=True,
#         show_plots=False,
#         save_path="plots",
#     )
