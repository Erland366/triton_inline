"""
Sub-exercise 2: Write a Proton Profile from Scratch
Module: 1.1b — Benchmarking, Profiling & Debugging Toolkit
Profiling Tier: Proton (triton.profiler)

Objectives:
1. Set up the full Proton lifecycle (start, deactivate/activate, scope, finalize)
2. Annotate kernel launches with FLOPS and bytes metadata
3. Sweep at least 3 matrix sizes
4. Display the call tree with proton_viewer

Acceptance Criteria:
1. Profile runs without errors
2. Produces .hatchet file at benchmark_results/1.1b/profile.hatchet
3. Call tree shows TFLOPS and GB/s for both triton and cuBLAS at each size

Instructions:
1. Complete all TODO sections below
2. Run: python exercises/1.1b/profile.py
"""

import os
import sys
import torch

# Import YOUR matmul from Module 1.1
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "1.1"))
from exercise import matmul

DEVICE = "cuda"

SAVE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "benchmark_results", "1.1b",
)
os.makedirs(SAVE_DIR, exist_ok=True)


# =============================================================================
# TODO 1: Import Proton and implement the proton_context manager
# =============================================================================
# Import:
#   import triton.profiler as proton
#   import triton.profiler.viewer as proton_viewer
#   from contextlib import contextmanager
#
# Then implement proton_context():
#   @contextmanager
#   def proton_context():
#       proton.activate(0)
#       try:
#           yield
#       finally:
#           proton.deactivate(0)
#
# This helper cleanly toggles profiling for a block of code.
import triton.profiler as proton
import triton.profiler.viewer as proton_viewer
from contextlib import contextmanager

@contextmanager
def proton_context():
    proton.activate(0)
    try:
        yield
    finally:
        proton.deactivate(0)


# =============================================================================
# TODO 2: Define sweep sizes and number of reps
# =============================================================================
# sweep_sizes = [1024, 2048, 4096]  # M=N=K values to profile
# reps = 10  # repetitions per size (more = more accurate, but slower)
sweep_sizes = [1024, 2048, 4096]
reps = 10


# =============================================================================
# TODO 3: Compute FLOPS and bytes metadata
# =============================================================================
# For FP16 matmul (M=N=K=n):
#   bytes_per_elem = 2  (FP16)
#   flops_key = "flops16"  (FP16 operations)
#   flops = 2 * n * n * n  (multiply-accumulate for each output element)
#   bytes_ = bytes_per_elem * (n*n + n*n + n*n)  (read A + read B + write C)
#
# These go into the proton.scope metadata dict:
#   {"bytes": bytes_, flops_key: float(flops)}
#
# Proton uses these to compute tflop16/s and gbps automatically.
bytes_per_elem = 2
for n in sweep_sizes:
    flops_key = "flops16"
    flops = 2 * n * n * n
    bytes_ = bytes_per_elem * (n*n + n*n + n*n)
    bytes_dict = {"bytes" : bytes_, flops_key: float(flops)}

# =============================================================================
# TODO 4: Start Proton session
# =============================================================================
# profile_name = os.path.join(SAVE_DIR, "profile")
# proton.start(profile_name, hook="triton")
# proton.deactivate(0)
#
# The hook="triton" tells Proton to instrument Triton kernel launches.
# deactivate(0) pauses profiling during setup code.
    profile_name = os.path.join(SAVE_DIR, "proton")
    proton.start(profile_name, hook="triton")
    proton.deactivate(0)



# =============================================================================
# TODO 5: Profile loop with scope annotations
# =============================================================================
# For each size n in sweep_sizes:
#   1. Create random FP16 matrices a, b on DEVICE
#   2. Warmup both kernels (outside proton_context)
#   3. Inside proton_context():
#      for _ in range(reps):
#          with proton.scope(f"cublas [n={n}]", {"bytes": bytes_, flops_key: float(flops)}):
#              torch.matmul(a, b)
#          with proton.scope(f"triton [n={n}]", {"bytes": bytes_, flops_key: float(flops)}):
#              matmul(a, b)
#
# Key points:
# - Each scope gets its OWN metadata dict (recompute for each size)
# - The scope name appears in the call tree
# - Warmup runs should be OUTSIDE proton_context (not profiled)
    kwargs = dict(device=DEVICE, dtype=torch.float16)
    a = torch.randn((n, n), **kwargs)
    b = torch.randn((n, n), **kwargs)

    for _ in range(10):
        cublas_result = a @ b
        triton_result = matmul(a, b)

    with proton_context():
        for _ in range(reps):
            with proton.scope(f"cublas [n={n}]", bytes_dict):
                a @ b
            with proton.scope(f"triton [n={n}]", bytes_dict):
                matmul(a, b)



# =============================================================================
# TODO 6: Finalize and display the call tree
# =============================================================================
# proton.finalize()
#
# metric_names = ["tflop16/s", "gbps", "time/ms"]
# tree, metrics = proton_viewer.parse(metric_names, f"{profile_name}.hatchet")
# proton_viewer.print_tree(tree, metrics)
#
# This prints a tree showing each scope with:
#   tflop16/s — FP16 throughput (TFLOPS)
#   gbps — memory bandwidth (GB/s)
#   time/ms — wall-clock time
proton.finalize()
metric_names = ["tflop16/s", "gbyte/s", "time/ms"]
tree, metrics = proton_viewer.parse(metric_names, f"{profile_name}.hatchet")
proton_viewer.print_tree(tree, metrics)


# =============================================================================
# TODO 7: Write interpretation (printed summary)
# =============================================================================
# After displaying the tree, print a brief analysis:
#   - At each size, is triton faster or slower than cublas?
#   - What's the peak TFLOPS your kernel achieves?
#   - Is the kernel compute-bound or memory-bound?
#     (Hint: for matmul, arithmetic intensity = 2*n*n*n / (2*(n*n+n*n+n*n) bytes)
#      = n/3. At n=1024, AI=341 — well above roofline knee, so compute-bound.)


# =============================================================================
# HINTS
# =============================================================================

# --- Hint 1 (Direction) ---
# The Proton lifecycle is: start → deactivate → [activate → scope → deactivate] → finalize
# Think of activate/deactivate as "recording on/off" buttons.
# Scopes tag individual kernel calls with metadata.

# --- Hint 2 (Approach) ---
# Look at tutorial 09 lines 618-758 for the production pattern.
# The key insight: proton.scope() is a context manager that wraps a kernel call.
# The metadata dict tells Proton how many FLOPS and bytes that call involves.

# --- Hint 3 (Near-solution) ---
# def main():
#     profile_name = os.path.join(SAVE_DIR, "profile")
#     proton.start(profile_name, hook="triton")
#     proton.deactivate(0)
#
#     for n in [1024, 2048, 4096]:
#         a = torch.randn(n, n, device=DEVICE, dtype=torch.float16)
#         b = torch.randn(n, n, device=DEVICE, dtype=torch.float16)
#         # warmup
#         matmul(a, b); torch.matmul(a, b); torch.cuda.synchronize()
#
#         flops = 2.0 * n * n * n
#         bytes_ = 2 * 3 * n * n  # FP16: 2 bytes * (A + B + C)
#         with proton_context():
#             for _ in range(10):
#                 with proton.scope(f"cublas [n={n}]", {"bytes": bytes_, "flops16": flops}):
#                     torch.matmul(a, b)
#                 with proton.scope(f"triton [n={n}]", {"bytes": bytes_, "flops16": flops}):
#                     matmul(a, b)
#
#     proton.finalize()
#     tree, metrics = proton_viewer.parse(["tflop16/s", "gbps", "time/ms"],
#                                          f"{profile_name}.hatchet")
#     proton_viewer.print_tree(tree, metrics)
