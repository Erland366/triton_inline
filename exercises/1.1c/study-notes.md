# Module 1.1c Study Notes: Pointer Arithmetic & Block Pointers

## Why This Module Exists

Pointer arithmetic is the source of many early Triton bugs. The fix is to stop
guessing whether to add `pid`, `offsets`, or strides, and instead write the
logical PyTorch index first.

## Rule

```text
pointer = base + index_0 * stride_0 + index_1 * stride_1 + ...
```

`tl.program_id` picks the outer work item. `tl.arange` picks lanes inside that
work item.

## Row-Wise Kernels

For tensors shaped like `X[M, N]`, `Y[M, N]`, `W[N]`, `B[N]`, `Mean[M]`:

```python
row = tl.program_id(0)
cols = tl.arange(0, BLOCK_SIZE)

# X[row, cols]
x = tl.load(X + row * stride + cols, mask=cols < N, other=0.0)

# W[cols]
w = tl.load(W + cols, mask=cols < N, other=0.0)

# Mean[row]
tl.store(Mean + row, mean)
```

The tensor shape decides which dimensions are legal. Do not add `row * stride`
to a 1D tensor such as `W` or `B`.

## 2D Tiles

Manual 2D address math is:

```python
# X[rows, cols]
ptrs = X + rows[:, None] * stride_m + cols[None, :] * stride_n
```

For standard 2D rectangular tiles, prefer `tl.make_block_ptr`:

```python
block = tl.make_block_ptr(
    base=X,
    shape=(M, N),
    strides=(stride_m, stride_n),
    offsets=(start_m, start_n),
    block_shape=(BLOCK_M, BLOCK_N),
    order=(1, 0),
)
```

This is usually clearer than manually constructing the full address matrix.

## Practice Rule

For every `tl.load` and `tl.store`, write one comment above it:

```python
# X[row, cols]
x = tl.load(...)
```

If the comment is hard to write, the pointer expression is probably not ready.

