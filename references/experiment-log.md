# Experiment Log

This file tracks experiment plans, decisions, and retrospectives in chronological order.

## Format

Each entry should include:
- **Date**: YYYY-MM-DD
- **Type**: Plan | Observation | Retrospective
- **General description**: One sentence for non-technical context
- **Details**: What was planned/observed/learned

---

## 2026-02-13 — Retrospective: Boundary Masking in Tiled Matmul

- **Type**: Retrospective
- **General description**: Captured the three-mechanism boundary handling pattern for tiled GPU kernels as a reusable skill.
- **Details**: While implementing Module 1.1 (tiled FP16 matmul), identified that boundary handling splits into three mechanisms across M, N, K dimensions. M/N use offset wrapping (`% M`, `% N`) at pointer creation to avoid inner-loop masks. K uses per-iteration load masks with `other=0.0`. Store uses strict un-wrapped masks. This keeps the hot inner loop free of M/N branches.
- **Skill created**: `.codex/skills/matmul-boundary-masking-strategy/SKILL.md`

## 2026-02-13 — Benchmark: Module 1.1 Matmul

- **Type**: Observation
- **General description**: Tiled FP16 matmul kernel achieves 99-100% of cuBLAS throughput across all tested sizes.
- **Details**:
  - Target: >80% of cuBLAS at M=N=K=4096
  - Achieved: 99.8% of cuBLAS (166.8 vs 167.7 TFLOP/s) — **PASS**
  - Peak: ~167 TFLOP/s (near RTX 4090 FP16 theoretical peak)
  - Proton profile confirms compute-bound execution (no memory bottleneck)
  - Autotuner selects 128x128x32 (4 warps, 4 stages) for large sizes
  - Test tolerance fixed from `atol=1e-2, rtol=0` to `atol=5e-2, rtol=1e-2` for FP16 numerical accuracy at large output magnitudes
- **Artifacts**: `benchmark_results/1.1/` (CSV, PNG, profile.hatchet)

<!-- New entries go above this line -->
