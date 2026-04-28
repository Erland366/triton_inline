# Exercise: Benchmarking, Profiling & Debugging Toolkit

**Module:** 1.1b | **Phase:** Core Kernel Mastery | **GPU:** Ampere

## Objective

Learn to measure, profile, and debug Triton kernels using the same tools used in
production. You'll use your completed Module 1.1 matmul kernel as the subject for
all four sub-exercises.

## Study Material

Before starting, review:
- `exercises/1.1b/study-notes.md` (your study notes from this module)
- `profiling-guide.md` §Teaching Sequence
- `debugging-methodology.md`

## Sub-Exercises

### Sub-exercise 1: Write a Benchmark (`exercise.py`)

Write a `triton.testing.Benchmark` from scratch that:
- Sweeps M=N=K over [512, 1024, 2048, 4096]
- Compares your matmul kernel against `torch.matmul` (cuBLAS)
- Computes TFLOPS from measured milliseconds
- Saves results to `benchmark_results/1.1b/`

### Sub-exercise 2: Write a Proton Profile (`profile.py`)

Write a Proton profiling script that:
- Sets up the full Proton lifecycle (start, deactivate/activate, scope, finalize)
- Annotates kernels with FLOPS and bytes metadata
- Sweeps at least 3 matrix sizes
- Displays the call tree with `proton_viewer`
- Saves to `benchmark_results/1.1b/profile.hatchet`

### Sub-exercise 3: Write an Interpretation (`interpretation.md`)

After running your benchmark and profile, analyze:
- Is your kernel compute-bound or memory-bound at each size?
- Calculate arithmetic intensity and compare against roofline knee
- Identify the bottleneck at each matrix size

### Sub-exercise 4: Debug a Seeded Bug (`exercise_debug.py`)

A deliberately broken matmul variant is provided. Use the diagnostic flowchart
from `debugging-methodology.md` to:
- Identify the bug type (precision vs logic)
- Run the tolerance sweep
- Use the isolation testing matrix
- Document your process in `debug_report.md`

## Acceptance Criteria

1. Sub-exercise 1: Benchmark runs, produces TFLOPS comparison, results saved to CSV
2. Sub-exercise 2: Profile runs, shows call tree with TFLOPS and GB/s per scope
3. Sub-exercise 3: Written interpretation correctly identifies compute-bound vs memory-bound
4. Sub-exercise 4: Bug correctly identified with supporting evidence

## Files

| File | Purpose |
|------|---------|
| `exercise.py` | Sub-exercise 1 — benchmark skeleton with TODOs |
| `profile.py` | Sub-exercise 2 — Proton profile skeleton with TODOs |
| `interpretation.md` | Sub-exercise 3 — template for your analysis |
| `exercise_debug.py` | Sub-exercise 4 — broken kernel + debugging scaffold |
| `debug_report.md` | Sub-exercise 4 — template for your debugging report |
| `study-notes.md` | Your study notes from the study phase |
