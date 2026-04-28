"""
Hardware Profile: Fused Softmax & LayerNorm
Module: 1.2
Profiling Tier: Proton (triton.profiler)
Mode: Student-written (post-1.1b)

Write your Proton profile following the patterns from Module 1.1b.
See profiling-guide.md §Tier 2 for API reference and metric interpretation.

NOTE: File is named proton_profile.py (not profile.py) to avoid shadowing
Python's stdlib profile module.
"""

import torch
import triton.profiler as proton
import triton.profiler.viewer as proton_viewer
from contextlib import contextmanager

from exercise import fused_softmax, fused_layernorm_fwd, reference_softmax, reference_layernorm

DEVICE = "cuda"


# TODO 1: Implement proton_context manager
# Use the activate/deactivate pattern from profiling-guide.md


# TODO 2: Setup — define sweep sizes
# sweep_sizes = [1024, 2048, 4096, 8192]  (column widths)
# M = 4096  (fixed rows)


# TODO 3: Compute FLOPS and bytes metadata for scope annotations
# Softmax is bandwidth-bound, so bandwidth is the key metric:
#   bytes_per_elem = torch.float16.itemsize  (= 2)
#   flops_key = f"flops{bytes_per_elem * 8}"  (= "flops16")
#   flops = 5 * M * N   (max, sub, exp, sum, div — 5 ops per element)
#   bytes_ = bytes_per_elem * 2 * M * N  (read input + write output)
# Note: Proton cannot measure SM-level metrics. Those require nsight-python (Tier 1).


# TODO 4: Start Proton session
# proton.start(<profile_name>, hook="triton")
# proton.deactivate(0)


# TODO 5: Profile loop with scope annotations
# for N in sweep_sizes:
#     <create tensors>
#     with proton_context():
#         for _ in range(reps):
#             with proton.scope(f"softmax [N={N}]", {<metadata>}):
#                 fused_softmax(<args>)
#             with proton.scope(f"torch_softmax [N={N}]", {<metadata>}):
#                 reference_softmax(<args>)


# TODO 6: Finalize and display
# proton.finalize()
# metric_names = ["tflop16/s", "gbyte/s", "time/ms"]
# tree, metrics = proton_viewer.parse(metric_names, f"{profile_name}.hatchet")
# proton_viewer.print_tree(tree, metrics)


# TODO 7: Write interpretation
# Based on the profile results, answer:
#   - Is the kernel compute-bound or memory-bound at each size?
#   - How does throughput scale with problem size?
#   - What is the bottleneck?
