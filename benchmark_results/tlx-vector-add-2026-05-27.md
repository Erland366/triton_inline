# Report: TLX Vector Add Exercise 0

**Date:** 2026-05-27
**Author:** Erland
**Status:** In Progress

## Objective

Establish the first TLX study baseline by comparing normal Triton vector
addition, TLX warp-specialized vector addition, and PyTorch for two independent
elementwise additions.

## Setup

### Environment

- Hardware: Modal H100
- Software: Modal CUDA 12.8.1 image, PyTorch 2.11.0 CUDA 12.8 wheel, local build of `facebookexperimental/triton`
- Dataset: Synthetic FP32 vectors generated inside `vector_add_tlx.py`

### Configuration

```yaml
exercise_script: exercises/tlx/exercise.py
benchmark_script: exercises/tlx/benchmark.py
remote_plot_path: plots/vector-add-performance.png
local_plot_path: plot.png
providers:
  - torch
  - triton
  - triton_ws
sizes: 2**12 through 2**27
metric: GB/s
```

## Experiments

### Run 1: Modal H100 TLX Vector Add

**Command:**

```bash
source .venv/bin/activate && modal run run_modal.py --action run --script exercises/tlx/benchmark.py
```

**Results:**

| Metric | Value |
|--------|-------|
| Correctness | TODO |
| Best Torch GB/s | TODO |
| Best Triton GB/s | TODO |
| Best Triton_WS GB/s | TODO |
| Local plot artifact | TODO |

**Observations:**

TODO: Paste the main benchmark result trends and whether the TLX split-task
version helps or hurts this bandwidth-bound toy kernel.

## Analysis

### What Worked

- TODO

### What Failed

- TODO

### Key Insights

1. TODO

## Next Steps

- [ ] Move to `hopper_gemm_pipelined.py` after the vector-add artifact path is reliable.
- [ ] Record whether TLX async-task overhead is visible on small vector sizes.

## Appendix

- Local exercise guide: `exercises/tlx/README.md`
- Student implementation file: `exercises/tlx/exercise.py`
- Student benchmark file: `exercises/tlx/benchmark.py`
- Imported TLX reference: `compiled_resources/TLX/triton/third_party/tlx/tutorials/vector-add2.py`
