# Triton Pointer Arithmetic Field Guide

Use this note when a kernel feels confusing because you are unsure whether to add
`pid`, `tl.arange`, a stride, or some combination of them.

## Core Rule

Write the PyTorch indexing expression first, then translate it mechanically:

```text
pointer = base + index_0 * stride_0 + index_1 * stride_1 + ...
```

`tl.program_id(...)` chooses the outer chunk this program owns. `tl.arange(...)`
chooses the lanes inside that chunk.

For a row-wise kernel:

```python
row = tl.program_id(0)
cols = tl.arange(0, BLOCK_SIZE)

# X[row, cols]
x = tl.load(X + row * stride + cols, mask=cols < N, other=0.0)
```

## Shape Decides the Pointer

Only include dimensions that the tensor actually has.

For LayerNorm:

| Tensor | Logical shape | PyTorch access | Triton pointer |
|---|---:|---|---|
| `X` | `(M, N)` | `X[row, col]` | `X + row * stride + cols` |
| `Y` | `(M, N)` | `Y[row, col]` | `Y + row * stride + cols` |
| `W` | `(N,)` | `W[col]` | `W + cols` |
| `B` | `(N,)` | `B[col]` | `B + cols` |
| `Mean` | `(M,)` | `Mean[row]` | `Mean + row` |
| `Rstd` | `(M,)` | `Rstd[row]` | `Rstd + row` |

The common mistake is treating `W` or `B` like `(M, N)` tensors:

```python
# Wrong for LayerNorm. W has no row dimension.
w_ptr = W + row * stride
```

The correct pointer is:

```python
# W[cols]
w = tl.load(W + cols, mask=cols < N, other=0.0)
```

## Manual 2D Formula

For a general 2D tensor:

```python
# X[rows, cols]
ptrs = X + rows[:, None] * stride_m + cols[None, :] * stride_n
```

For a contiguous row-major tensor, `stride_n == 1`, so this often becomes:

```python
ptrs = X + rows[:, None] * stride_m + cols[None, :]
```

Broadcasting makes a 2D tile of addresses:

```text
rows[:, None] -> BLOCK_M x 1
cols[None, :] -> 1 x BLOCK_N
sum           -> BLOCK_M x BLOCK_N
```

This manual formula is worth understanding because it explains what every Triton
kernel is doing. For standard rectangular 2D tiles, prefer `tl.make_block_ptr`
once the indexing concept is clear.

## Prefer `tl.make_block_ptr` for Standard 2D Tiles

For normal rectangular loads/stores from a 2D tensor, use Triton's block pointer
API instead of manually assembling a matrix of addresses:

```python
block_ptr = tl.make_block_ptr(
    base=X,
    shape=(M, N),
    strides=(stride_m, stride_n),
    offsets=(start_m, start_n),
    block_shape=(BLOCK_M, BLOCK_N),
    order=(1, 0),
)

tile = tl.load(block_ptr, boundary_check=(0, 1), padding_option="zero")
```

The arguments map directly to PyTorch indexing:

| Argument | Meaning |
|---|---|
| `shape=(M, N)` | Logical tensor shape |
| `strides=(stride_m, stride_n)` | Pointer step for each logical dimension |
| `offsets=(start_m, start_n)` | Top-left logical index of this tile |
| `block_shape=(BLOCK_M, BLOCK_N)` | Tile size this program handles |
| `boundary_check=(0, 1)` | Mask both dimensions against `shape` |

For stores:

```python
out_ptr = tl.make_block_ptr(
    base=Y,
    shape=(M, N),
    strides=(stride_m, stride_n),
    offsets=(start_m, start_n),
    block_shape=(BLOCK_M, BLOCK_N),
    order=(1, 0),
)

tl.store(out_ptr, tile, boundary_check=(0, 1))
```

Use `tl.make_block_ptr` for:

- rectangular 2D copies
- tiled transpose or layout conversion
- matmul tiles when the access pattern is standard
- kernels where boundary checks should be explicit and readable

Use manual pointers for:

- row-wise vectors like softmax and LayerNorm
- 1D elementwise kernels
- irregular gathers/scatters
- custom broadcasting patterns
- learning and debugging the address math

## Matmul Indexing

Manual matmul pointers follow the same rule.

```python
pid_m = tl.program_id(0)
pid_n = tl.program_id(1)

offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
offs_k = tl.arange(0, BLOCK_K)

# A[offs_m, offs_k]
a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak

# B[offs_k, offs_n]
b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

# C[offs_m, offs_n]
c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
```

With `tl.make_block_ptr`, the same matmul tile intent is:

```python
a_block = tl.make_block_ptr(
    base=A,
    shape=(M, K),
    strides=(stride_am, stride_ak),
    offsets=(pid_m * BLOCK_M, k_start),
    block_shape=(BLOCK_M, BLOCK_K),
    order=(1, 0),
)

b_block = tl.make_block_ptr(
    base=B,
    shape=(K, N),
    strides=(stride_bk, stride_bn),
    offsets=(k_start, pid_n * BLOCK_N),
    block_shape=(BLOCK_K, BLOCK_N),
    order=(1, 0),
)
```

## Pointer Checklist

Before writing `tl.load` or `tl.store`, answer these in order:

1. What is the PyTorch expression? Example: `X[row, cols]`.
2. What is the tensor shape? Example: `X` is `(M, N)`, `W` is `(N,)`.
3. Which index comes from `tl.program_id`?
4. Which index comes from `tl.arange`?
5. Which strides correspond to the tensor dimensions?
6. What mask is based on the logical indices?

If you cannot write the PyTorch expression, do not write the pointer yet.

## Error Patterns

| Symptom | Likely cause |
|---|---|
| Row 0 wrong or missing | False assumption like `tl.assume(pid > 0)` |
| Every row reads the same vector | Missing `row * stride` on a 2D tensor |
| Weight/bias changes by row | Added `row * stride` to a 1D tensor |
| Only edge columns wrong | Missing `cols < N` mask or boundary check |
| Only edge rows wrong | Missing `rows < M` mask or boundary check |
| Periodic stripes | Wrong stride or swapped dimensions |
| Large chaotic error | Pointer formula does not match logical indexing |

