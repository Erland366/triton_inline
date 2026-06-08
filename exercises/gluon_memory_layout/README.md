# Exercise Track: Gluon Memory Layouts

**Module:** Gluon | **Phase:** Explicit tensor layout control | **GPU:** CUDA

## Objective

Learn Gluon from the layout surface upward. TLX made async work, local buffers,
and warp-specialized tasks explicit. Gluon makes a different layer explicit:
how a logical tensor tile is distributed across CTA threads, warps, lanes, and
per-lane registers.

The immediate goal is not to write GEMM. The goal is to build enough intuition
that a layout such as:

```python
gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
```

stops looking like magic numbers.

## Study Material

Read these in order:

- `compiled_resources/Tensor Layouts_ The Complete Guide to Gluon Layouts.pdf`
- `compiled_resources/TLX/triton/python/tutorials/gluon/01-intro.py`
- `compiled_resources/TLX/triton/python/tutorials/gluon/02-layouts.py`
- `compiled_resources/TLX/triton/python/triton/experimental/gluon/language/_layouts.py`

The upstream Gluon source under `compiled_resources/TLX/triton/python/triton/experimental/gluon/`
is reference material only. Do not edit it.

## Curriculum

| id | description | key_differences | notes |
|----|-------------|-----------------|-------|
| G0 | Layout arithmetic drill | Pure Python `LayoutSpec`, no GPU kernel | Compute block shape and reason about register pressure before writing Gluon |
| G1 | 1D layout-aware memcpy | `gl.arange(..., layout=layout)` controls derived tensor layout | First real Gluon kernel; compare with `tl.arange` intuition from Triton |
| G2 | R sweep and SASS reading | Vary `size_per_thread=[R]` | Learn that larger per-thread work can change LDG/STG shape and throughput |
| G3 | 2D memcpy with `SliceLayout` | 1D row/column indices come from slices of a 2D parent layout | Learn why 2D offsets need broadcasting from layout-compatible index tensors |
| G4 | In/out layout conversion | Pick different load/store layouts and use `gl.convert_layout` | Learn when conversion cost is better than bad global-memory coalescing |

## Exercise File

Use:

```text
exercises/gluon_memory_layout/exercise.py
```

The file is intentionally TODO-first. It should define behavior and tests
without handing you the implementation.

## Commands

Run the local scaffold:

```bash
source .venv/bin/activate && python exercises/gluon_memory_layout/exercise.py
```

Run tests locally if this machine has CUDA:

```bash
source .venv/bin/activate && python -m pytest exercises/gluon_memory_layout/exercise.py -q
```

Run tests on Modal H100:

```bash
source .venv/bin/activate && modal run run_modal.py \
  --action pytest \
  --script "exercises/gluon_memory_layout/exercise.py -q"
```

If you need environment variables:

```bash
source .venv/bin/activate && modal run run_modal.py \
  --action pytest \
  --script "exercises/gluon_memory_layout/exercise.py -q" \
  --env-vars TRITON_PRINT_AUTOTUNING=1
```

## Acceptance Criteria

1. `test_layout_arithmetic` passes without GPU execution.
2. `test_memcpy_1d` passes for non-power-of-two vector lengths.
3. `test_memcpy_2d` passes for contiguous and transposed inputs.
4. `test_memcpy_2d_inout` passes when input/output tensors prefer different
   global-memory layouts.
5. Optional benchmark rows explain whether changing `R` helped, hurt, or did
   nothing measurable on the target GPU.

## Notes

- `layout` is not memory storage. It is the mapping from logical tensor elements
  to hardware participants.
- In Gluon, layout usually enters the program through `gl.arange`.
- `SliceLayout` is for rank-lowering index tensors that will later be expanded
  and broadcast into a parent layout.
- `gl.convert_layout` may require cross-thread or cross-warp movement. Treat it
  as a real cost unless you can prove the conversion is trivial.
- Prefer correctness and layout reasoning first. Do not tune `R`, `XBLOCK`, or
  `YBLOCK` until the copy kernels are correct.
