# Exercise: Fused Softmax & LayerNorm

**Module:** 1.2 | **Phase:** Core Kernel Mastery | **GPU:** Ampere+

## Objective

Write fused row-wise reduction kernels that beat PyTorch by minimizing memory traffic.
These are bandwidth-bound operations — the goal is to read/write DRAM as few times as possible.

## Study Material

Before starting, read:
- `tutorials/02-fused-softmax.py` — softmax kernel with persistent row loop
- `tutorials/05-layer-norm.py` — layernorm fwd/bwd with parallel reduction
- `triton_kernels/reduce.py` — production generic reduction with PostprocessFn
- `exercises/1.2/study_notes.md` — your study notes from the study phase

## What to Implement

1. **Softmax kernel** (`softmax_kernel`): load row, subtract max, exp, divide by sum, store
2. **LayerNorm forward** (`layernorm_fwd_kernel`): compute mean, variance, normalize, apply affine
3. **LayerNorm backward dx** (`layernorm_bwd_dx_kernel`): compute dx using saved mean/rstd

## Acceptance Criteria

1. Softmax matches `torch.softmax(x, dim=-1)` within atol=1e-3
2. LayerNorm forward matches `torch.nn.functional.layer_norm` within atol=1e-3
3. LayerNorm backward dx matches PyTorch autograd within atol=1e-3

## Benchmark Target

Beat PyTorch eager by >2x on softmax for (8192, 4096) input

## Bridge to Production

`triton_kernels/reduce.py` generalizes the same pattern with `PostprocessFn` epilogues.
Softmax and layernorm are both row-wise reductions with different epilogues.

## Files

| File | Purpose |
|------|---------|
| `exercise.py` | Skeleton — implement the TODOs |
| `benchmark.py` | TODO skeleton — write your benchmark (Mode B) |
| `proton_profile.py` | TODO skeleton — write your Proton profile (Mode B) |
| `study_notes.md` | Study phase notes |
