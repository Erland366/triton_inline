# Exercise: Pointer Arithmetic & Block Pointers

**Module:** 1.1c | **Phase:** Core Kernel Mastery | **GPU:** Ampere+

## Objective

Practice Triton pointer indexing deliberately before continuing with more complex
kernels. This module exists because most early Triton bugs are not math bugs;
they are indexing bugs.

The goal is to make every pointer expression traceable back to a PyTorch-style
index such as `X[row, col]`, `W[col]`, or `Mean[row]`.

## Study Material

Before starting, read:

- `references/triton-pointer-arithmetic.md` - the permanent pointer arithmetic note
- `exercises/1.1/study-notes.md` - matmul indexing and boundary masking context
- `exercises/1.2/study_notes.md` - row-wise softmax and LayerNorm context

## What to Practice

1. **Row-wise manual indexing**
   - Load `X[row, cols]`
   - Load `W[cols]` and `B[cols]`
   - Load/store `RowScale[row]`
   - Store `Y[row, cols]`

2. **Standard 2D rectangular tiles with `tl.make_block_ptr`**
   - Copy a 2D matrix with boundary checks
   - Use `shape`, `strides`, `offsets`, and `block_shape` explicitly
   - Avoid manual `rows[:, None] * stride + cols[None, :]` for this standard case

3. **Manual offset reading drills**
   - For each pointer expression, write the PyTorch expression above it
   - Explain whether each index comes from `tl.program_id` or `tl.arange`

## Policy for This Module

- Use manual pointer arithmetic for 1D and row-wise kernels.
- Use `tl.make_block_ptr` for standard 2D rectangular tiles.
- Do not optimize for speed. Optimize for clear indexing.
- Every `tl.load` and `tl.store` should have a comment naming the logical tensor access.

## Acceptance Criteria

1. `row_affine` matches the PyTorch reference for non-power-of-two `N`
2. `block_copy_2d` matches the PyTorch reference for non-multiple `M` and `N`
3. Every pointer expression in `exercise.py` has a PyTorch-index comment above it

## Files

| File | Purpose |
|------|---------|
| `exercise.py` | TODO skeleton for pointer indexing drills |
| `study-notes.md` | Local study notes and reminders |

