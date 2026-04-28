"""
triton-viz demo for the softmax kernel.

Three tools:
1. Visualizer — see memory access patterns interactively
2. Profiler — check mask efficiency and load/store byte counts
3. Sanitizer — detect OOB memory accesses symbolically

Usage:
    python triton_viz_demo.py visualizer   # opens browser UI
    python triton_viz_demo.py profiler     # text output
    python triton_viz_demo.py sanitizer    # OOB check
    python triton_viz_demo.py sanitizer_oob  # intentionally buggy kernel (demo)

Key concepts:
- triton_viz.trace(client=...) must go ABOVE @triton.jit
- Wrapping a kernel requires a factory function (decorator applied at definition time)
- triton-viz runs kernels on CPU — use device="cpu" and small inputs
- Three client types: Tracer (visualizer), Profiler (mask stats), Sanitizer (OOB check)
"""

import sys
import torch
import triton
import triton.language as tl
import triton_viz
from triton_viz.clients import Tracer, Profiler, Sanitizer


# =============================================================================
# TODO 1: Create a traced softmax kernel
# =============================================================================
# Write a factory function make_softmax_kernel(client) that:
#   - Takes a triton-viz client (Tracer, Profiler, or Sanitizer)
#   - Defines a softmax kernel decorated with BOTH @triton_viz.trace and @triton.jit
#   - The kernel should be your working softmax from exercise.py
#   - Returns the kernel function
#
# Decorator order matters: @triton_viz.trace(client=client) goes ABOVE @triton.jit
#
# Hint: You can copy your softmax kernel from exercise.py, but remove
#       tl.assume and tl.multiple_of hints (triton-viz doesn't support them)

def make_softmax_kernel(client):
    pass  # TODO: implement


# =============================================================================
# TODO 2: Create a buggy softmax kernel (for sanitizer demo)
# =============================================================================
# Same as TODO 1, but intentionally remove the mask from tl.load and tl.store.
# This should cause OOB accesses when n_cols < BLOCK_SIZE.

def make_buggy_softmax_kernel(client):
    pass  # TODO: implement


# =============================================================================
# TODO 3: Write the kernel launcher
# =============================================================================
# Write run_kernel(kernel, M, N) that:
#   - Creates a random (M, N) input on CPU (device="cpu", dtype=float32)
#   - Allocates output tensor
#   - Computes BLOCK_SIZE = triton.next_power_of_2(N)
#   - Launches the kernel with grid = (M,)
#   - Returns (input, output)

def run_kernel(kernel, M, N):
    pass  # TODO: implement


# =============================================================================
# TODO 4: Write the four demo functions
# =============================================================================

def demo_visualizer():
    """Visualize memory access patterns of the softmax kernel.

    TODO:
    - Create kernel with make_softmax_kernel(Tracer())
    - Run on small input (e.g. M=4, N=13) so the pattern is readable
    - Launch the browser UI with triton_viz.launch()
    """
    pass  # TODO: implement


def demo_profiler():
    """Run the profiler to check mask efficiency.

    TODO:
    - Create kernel with make_softmax_kernel(Profiler())
    - Run on input where N is not a power of 2 (e.g. M=8, N=100)
    - The profiler prints mask stats automatically
    - Calculate expected mask efficiency: N / next_power_of_2(N)
    """
    pass  # TODO: implement


def demo_sanitizer():
    """Run the sanitizer on a correct kernel — should find no OOB.

    TODO:
    - Create kernel with make_softmax_kernel(Sanitizer(abort_on_error=True))
    - Run on small input
    - If no exception, the kernel is safe
    """
    pass  # TODO: implement


def demo_sanitizer_oob():
    """Run the sanitizer on a BUGGY kernel — should detect OOB.

    TODO:
    - Create kernel with make_buggy_softmax_kernel(Sanitizer(abort_on_error=False))
    - Run on input where N < BLOCK_SIZE (e.g. M=4, N=13 → BLOCK_SIZE=16)
    - Wrap in try/except to catch sanitizer errors
    """
    pass  # TODO: implement


# =============================================================================
# CLI dispatch (provided — do not modify)
# =============================================================================

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "profiler"

    demos = {
        "visualizer": demo_visualizer,
        "profiler": demo_profiler,
        "sanitizer": demo_sanitizer,
        "sanitizer_oob": demo_sanitizer_oob,
    }

    if mode in demos:
        demos[mode]()
    else:
        print(f"Usage: python triton_viz_demo.py [{' | '.join(demos.keys())}]")
