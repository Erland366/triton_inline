# Module 1.1: Matmul Optimization — Study Notes

## Study Material

- `tutorials/03-matrix-multiplication.py` — the classic Triton matmul tutorial
- `tutorials/09-persistent-matmul.py` — persistent & TMA variants (non-persistent section relevant here)
- `triton_kernels/matmul.py` — production matmul interface
- `triton_kernels/matmul_details/_matmul.py` — production kernel implementation

All paths relative to `~/dotfiles/compiled_resources/triton_learning/`.

---

## Key Concepts

### 1. Block-Tiled Matrix Multiplication

The core algorithm decomposes `C = A @ B` into tiles computed in parallel. Each Triton
program computes one `[BLOCK_SIZE_M, BLOCK_SIZE_N]` output tile by iterating over K in
`BLOCK_SIZE_K` chunks:

```python
# tutorials/03-matrix-multiplication.py:297-307
accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
    accumulator = tl.dot(a, b, accumulator)
    a_ptrs += BLOCK_SIZE_K * stride_ak
    b_ptrs += BLOCK_SIZE_K * stride_bk
```

The accumulator is **FP32** even though inputs are FP16. This prevents precision loss
during the K-reduction. The final result is cast back: `c = accumulator.to(tl.float16)`.

### 2. Why FP32 Accumulation Matters

FP16 has only ~3.3 decimal digits of mantissa precision. When K=8192 and
`BLOCK_SIZE_K=64`, you do 128 `tl.dot` accumulations. If the accumulated sum reaches
500.0 in FP16, adding a small dot product result like 0.05 gets **rounded away** —
FP16 can't represent 500.05. Over 128 iterations, these rounding errors compound
catastrophically.

FP32 has ~7.2 digits of precision, so 500.0 + 0.05 = 500.05 is representable. The
final cast `accumulator.to(tl.float16)` only rounds once, at the end.

### 3. Multi-Dimensional Pointer Arithmetic

Triton doesn't have indexing syntax — you compute pointer blocks manually. For a 2D
row-major tensor, `X[i, j]` lives at `X + i*stride_0 + j*stride_1`:

```python
# tutorials/03-matrix-multiplication.py:286-290
offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
offs_k = tl.arange(0, BLOCK_SIZE_K)
a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
```

The broadcasting (`[:, None]` and `[None, :]`) creates a 2D block of pointers from 1D
index vectors. This is the fundamental memory access pattern in Triton.

### 4. L2 Cache Optimization via Super-Grouping

Naive row-major tile ordering causes poor cache reuse. **Super-grouping** orders tiles
so that `GROUP_SIZE_M` rows are processed together before moving to the next column
group. This keeps B-tiles in L2:

```python
# tutorials/03-matrix-multiplication.py:256-264
pid = tl.program_id(axis=0)
num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
num_pid_in_group = GROUP_SIZE_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

The tutorial reports >10% improvement from this on A100 (220 -> 245 TFLOPS).

### 5. `@triton.autotune` with `triton.Config`

Autotuning searches over block sizes, warp counts, and pipeline stages at runtime:

```python
# tutorials/03-matrix-multiplication.py:228-231
@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
```

Each `triton.Config` specifies `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`,
`GROUP_SIZE_M`, plus `num_stages` (software pipelining depth) and `num_warps`. The
`key` parameter triggers re-evaluation when matrix dimensions change.

### 6. Boundary Handling — Three Mechanisms for Three Dimensions

When M, N, or K aren't multiples of block sizes, the last tile in that dimension
extends past the matrix. Each dimension is handled differently:

#### M and N Boundaries (two-part handling)

**Part 1 — Make loads safe** (offset wrapping or clamping):

The last tile's indices go out of bounds (e.g., M=513, BLOCK_M=128 → last tile
reads rows 512-639, but only 512 exists). Prevent illegal memory access:

```python
# Option A: Modular wrapping (Tutorial 03) — wraps to valid but wrong rows
offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
# [512, 0, 1, 2, ...] — row 513 wraps to row 0

# Option B: Explicit clamping (Tutorial 09) — clamps to row 0
offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
offs_am = tl.where(offs_am < M, offs_am, 0)
# [512, 0, 0, 0, ...] — row 513+ clamp to row 0
```

Both load garbage data for out-of-bounds positions — but that's fine because...

**Part 2 — Make stores correct** (output mask):

```python
offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # NO wrapping
offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
tl.store(c_ptrs, c, mask=c_mask)
```

The garbage results from wrapped/clamped rows are **never written** to C.

#### K Boundary (single-part handling)

```python
# Load mask with zero-fill:
a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
```

Out-of-bounds K positions load as `0.0`. Zero times anything is zero, so they
contribute nothing to the accumulator. No second fix needed.

#### Summary Table

| Dimension | Problem | Fix at load time | Fix at store time |
|-----------|---------|------------------|-------------------|
| **M** | Last tile rows > M | Wrap/clamp offsets | Store mask |
| **N** | Last tile cols > N | Wrap/clamp offsets | Store mask |
| **K** | Last K-block > K | Load mask `other=0.0` | Not needed |

#### Why not mask M/N loads too?

You could add masks to M/N loads in the inner loop, but it's slower. The K-mask
runs every iteration and is unavoidable. Adding M/N masks would double the masking
overhead in the hot loop. The wrapping trick is free (just `% M`) and the store mask
runs only once after the loop. This is an important optimization: **move boundary
checks out of the inner loop whenever possible.**

#### Why non-tiled kernels don't use wrapping

In non-tiled kernels (vector add, softmax, elementwise ops), there's no inner loop.
The load and store happen at the same level, so both execute exactly once:

```python
# Non-tiled: direct masking — runs once, simple and clear
x = tl.load(x_ptr + offs, mask=offs < N, other=0.0)
y = x * 2
tl.store(out_ptr + offs, y, mask=offs < N)
```

Using wrapping here would be strictly worse — more complex with no payoff:

```python
# Non-tiled with wrapping — same cost, more complexity, easy to mix up
offs_wrapped = (pid * BLOCK + tl.arange(0, BLOCK)) % N
x = tl.load(x_ptr + offs_wrapped)          # no mask needed
y = x * 2
offs_real = pid * BLOCK + tl.arange(0, BLOCK)  # un-wrapped for store
tl.store(out_ptr + offs_real, y, mask=offs_real < N)
```

Both versions execute the same number of operations. The wrapping version adds
two offset sets (easy to confuse) and still needs a store mask.

**The rule:** Wrapping is an optimization for **tiled kernels with an inner loop**,
where it removes M/N masks from hundreds of loop iterations. No inner loop = no hot
path to protect = direct masking is simpler and equally fast.

### 7. 1D Grid for Custom Tile Ordering

The kernel uses a **1D grid** instead of 2D:

```python
grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
```

A 2D grid `(cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))` would give `pid_m = tl.program_id(0)`
and `pid_n = tl.program_id(1)` in a hardware-determined order you can't control. With a
1D grid, you get a single `pid = tl.program_id(0)` and can implement any custom mapping
from pid to (pid_m, pid_n), including the super-grouping pattern.

### 8. Debugging Kernel Correctness: Is It a Bug or Precision?

When a kernel test fails, you need to determine whether it's a logic bug or
a floating-point precision issue. Here's a systematic decision tree.

#### Step 0: The fastest test — run in FP32

Before any analysis, re-run with FP32 inputs and accumulation:

```python
a = torch.randn((M, K), device=DEVICE, dtype=torch.float32)
b = torch.randn((K, N), device=DEVICE, dtype=torch.float32)
ref = torch.matmul(a, b)
ours = matmul_fp32(a, b)  # same kernel logic, but FP32 throughout
```

- **Passes in FP32, fails in FP16** → precision issue, not a logic bug
- **Fails in both** → real bug in your kernel logic

This is definitive and takes 30 seconds. Do this first.

#### Step 1: Look at the error shape

```
Max diff: 0.062500, Mean diff: 0.003691
```

- **Mean diff << max diff** (orders of magnitude gap) → few outlier elements,
  probably precision
- **Mean diff ≈ max diff** → widespread errors, probably a logic bug
- **Max diff is a power of 2** (0.0625 = 2^-4) → FP16 ULP rounding, not a bug
- **Max diff is chaotic** (e.g., 147.3) → logic bug

#### Step 2: Check where errors are located

```python
bad = (diff > tolerance)
rows, cols = torch.where(bad)
```

| Error pattern | Likely cause |
|---------------|-------------|
| Concentrated at matrix edges | Boundary handling bug |
| In stripes or periodic | Stride bug |
| Spread evenly everywhere | Numerical precision |
| Entire output is wrong | Fundamental logic bug |

#### Step 3: Check output magnitude vs FP16 ULP

FP16 has 10 mantissa bits. The ULP (unit in last place) scales with magnitude:

| Output magnitude | FP16 ULP |
|-----------------|----------|
| ~1 | 0.001 |
| ~32 | 0.03125 |
| ~64 | 0.0625 |
| ~128 | 0.125 |

If your max diff ≈ ULP at your output's magnitude → precision, not a bug.

```python
import math
max_val = ref.abs().max().item()
ulp = 2 ** (math.floor(math.log2(max_val)) - 10)
print(f"Output max: {max_val:.1f}, ULP: {ulp}, Max diff: {diff.max().item()}")
```

#### Step 4: Validate the reference test

Check what inputs/sizes the reference test was designed for. A tolerance of
`atol=1e-2, rtol=0` works for small inputs (`torch.rand() - 0.5`, range [-0.5, 0.5])
but fails for `torch.randn()` (range ~[-3, 3]) at large matrix sizes because
outputs can reach 100+ where FP16 ULP exceeds 0.01.

#### Step 5: Sweep tolerances (if still unsure)

```python
for atol_v, rtol_v in [(1e-2, 0), (1e-2, 1e-2), (5e-2, 1e-2), (1e-1, 0)]:
    tolerance = atol_v + rtol_v * torch.abs(ours)
    n_fail = (diff > tolerance).sum().item()
    print(f"atol={atol_v}, rtol={rtol_v}: {n_fail} failures")
```

- **Smooth decrease** in failures as tolerance increases → precision issue
- **Cliff** (0 failures or thousands, nothing in between) → logic bug with
  a fixed error magnitude

#### Why two correct implementations differ

Floating-point addition is not associative: `(a + b) + c ≠ a + (b + c)`.
Your kernel and cuBLAS both accumulate in FP32, but they tile K differently.
Different summation order → different FP32 rounding → different final FP16
value when near a rounding boundary. This is expected, not a bug.

---

## Deep Dive: Super-Grouping from First Principles

### Prerequisite: How `//` and `%` Create 2D Indexing

Before super-grouping, make sure the two simplest mappings are clear. We have a
flat sequence of program IDs and need to map each to a (row, col) coordinate.

**Row-major** — sweep left to right, then next row:

```python
pid_m = pid // num_cols    # // counts how many full rows fit → row index
pid_n = pid % num_cols     # % gives remainder → column index
```

For a 4-row, 3-column grid:

```
        col0  col1  col2
row0: [  0     1     2  ]   pid 0→(0,0), pid 1→(0,1), pid 2→(0,2)
row1: [  3     4     5  ]   pid 3→(1,0), pid 4→(1,1), pid 5→(1,2)
row2: [  6     7     8  ]
row3: [  9    10    11  ]
```

**Column-major** — sweep top to bottom, then next column:

```python
pid_m = pid % num_rows     # % cycles through rows → row index
pid_n = pid // num_rows    # // counts complete columns → column index
```

```
        col0  col1  col2
row0: [  0     4     8  ]   pid 0→(0,0), pid 1→(1,0), pid 2→(2,0)
row1: [  1     5     9  ]   pid 3→(3,0), pid 4→(0,1), pid 5→(1,1)
row2: [  2     6    10  ]
row3: [  3     7    11  ]
```

The key insight: **`%` cycles, `//` advances.**

---

### Super-Grouping = Divide into Bands + Column-Major Within Each Band

Super-grouping is a three-step process. Let's build it on a small example:
4 rows, 3 columns, `GROUP_SIZE_M = 2`.

#### Step 1: Which band am I in?

Each band has `GROUP_SIZE_M = 2` rows and all 3 columns = `2 * 3 = 6` tiles per band.

```python
num_pid_in_group = GROUP_SIZE_M * num_pid_n    # = 2 * 3 = 6
group_id = pid // num_pid_in_group             # = pid // 6
```

```
pid  0-5  → group_id = 0  (band 0: rows 0-1)
pid  6-11 → group_id = 1  (band 1: rows 2-3)
```

That's it. `// 6` tells you which band.

#### Step 2: Where am I within my band?

```python
position = pid % num_pid_in_group    # = pid % 6, gives 0-5
```

```
pid 0 → position 0     pid 6 → position 0
pid 1 → position 1     pid 7 → position 1
pid 2 → position 2     pid 8 → position 2
pid 3 → position 3     pid 9 → position 3
pid 4 → position 4     pid 10 → position 4
pid 5 → position 5     pid 11 → position 5
```

Now we have a small sub-problem: map position (0-5) to (row, col) within a
2-row, 3-column sub-grid. We want **column-major** within the sub-grid.

#### Step 3: Column-major within the band

A band has `group_size_m = 2` rows and `num_pid_n = 3` columns:

```python
local_row = position % group_size_m     # cycles through 0, 1, 0, 1, 0, 1
local_col = position // group_size_m    # advances:      0, 0, 1, 1, 2, 2
```

```
position:  0  1  2  3  4  5
local_row: 0  1  0  1  0  1    ← % 2 cycles rows
local_col: 0  0  1  1  2  2    ← // 2 advances columns
```

Then convert local row to global row:

```python
pid_m = first_pid_m + local_row    # first_pid_m = group_id * GROUP_SIZE_M
pid_n = local_col
```

#### Result: Full 4x3 Grid

```
        col0  col1  col2
row0: [  0     2     4  ]  ← Band 0 (group_id=0)
row1: [  1     3     5  ]     Column-major: down, then right

row2: [  6     8    10  ]  ← Band 1 (group_id=1)
row3: [  7     9    11  ]     Same pattern
```

Trace to verify:

```
pid=0: group_id=0, position=0, local_row=0%2=0, local_col=0//2=0 → (0,0) ✓
pid=1: group_id=0, position=1, local_row=1%2=1, local_col=1//2=0 → (1,0) ✓
pid=2: group_id=0, position=2, local_row=2%2=0, local_col=2//2=1 → (0,1) ✓
pid=5: group_id=0, position=5, local_row=5%2=1, local_col=5//2=2 → (1,2) ✓
pid=7: group_id=1, position=1, local_row=1%2=1, local_col=1//2=0 → (3,0) ✓
```

---

### The Actual Code, Component by Component

```python
pid = tl.program_id(axis=0)
num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

# How many tiles in one horizontal band
num_pid_in_group = GROUP_SIZE_M * num_pid_n

# Step 1: which band?
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M

# Handle last band possibly being smaller
# (e.g., 9 rows with GROUP_SIZE_M=4 → last band has 1 row, not 4)
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

# Step 2: position within band
# Step 3: column-major within band
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)  # % cycles rows
pid_n = (pid % num_pid_in_group) // group_size_m                  # // advances cols
```

Every component:

| Variable | What it computes | Why it's needed |
|----------|------------------|-----------------|
| `num_pid_in_group` | Tiles per horizontal band | To split pids into bands |
| `group_id` | Which band (0, 1, 2, ...) | `pid // band_size` |
| `first_pid_m` | First row of this band | Starting row offset |
| `group_size_m` | Rows in this band | Handles last band being smaller |
| `pid % num_pid_in_group` | Position within band (0 to band_size-1) | Reduces to sub-problem |
| `... % group_size_m` | Local row within band | `%` cycles rows (column-major) |
| `... // group_size_m` | Column within band | `//` advances columns (column-major) |

---

### Why Column-Major Within the Band? (Not Row-Major?)

This is the core of why super-grouping works. Let's compare both options on our
4x3 grid (GROUP_SIZE_M = 2, Band 0 = rows 0-1):

**Option A: Row-major within band** (right then down):

```
pid 0→(0,0)  pid 1→(0,1)  pid 2→(0,2)  pid 3→(1,0)  pid 4→(1,1)  pid 5→(1,2)
```

What L2 sees over time (GPU runs nearby pids together):

```
pid 0,1,2 run together:  need A-row 0 + B-col 0, B-col 1, B-col 2
                          → 1 A-row, 3 B-cols in L2

pid 3,4,5 run together:  need A-row 1 + B-col 0, B-col 1, B-col 2
                          → 1 A-row, 3 B-cols in L2 (B reloaded!)
```

Problem: at any moment, L2 holds **1 A-row + ALL B-columns**. Each program in a row
needs a different B-column. This is exactly the same as no grouping — row-major
within the band defeats the purpose.

**Option B: Column-major within band** (down then right):

```
pid 0→(0,0)  pid 1→(1,0)  pid 2→(0,1)  pid 3→(1,1)  pid 4→(0,2)  pid 5→(1,2)
```

What L2 sees over time:

```
pid 0,1 run together:    need A-row 0, A-row 1 + B-col 0
                          → 2 A-rows, 1 B-col in L2

pid 2,3 run together:    need A-row 0, A-row 1 + B-col 1
                          → 2 A-rows (STILL CACHED!), 1 new B-col

pid 4,5 run together:    need A-row 0, A-row 1 + B-col 2
                          → 2 A-rows (STILL CACHED!), 1 new B-col
```

At any moment, L2 holds **2 A-rows + only 1 B-column**. The A-rows stay cached
across the entire band as we sweep through columns one at a time.

**The rule:** Column-major means consecutive pids share the **same B-column** (only
GROUP_SIZE_M pids before moving to the next column). Row-major means consecutive pids
need **all B-columns at once**. Column-major keeps B-data small in L2 at any instant.

---

### Why No Grouping Along N?

There is no `GROUP_SIZE_N`. Each band spans **ALL** columns:

```
Band 0:  rows 0-1  ×  ALL columns (col 0, col 1, col 2)
Band 1:  rows 2-3  ×  ALL columns (col 0, col 1, col 2)
```

The only partitioning is along M. That's why `num_pid_in_group` includes the full
column count:

```
num_pid_in_group = GROUP_SIZE_M * num_pid_n
                   ^^^^^^^^^^^^   ^^^^^^^^^
                   rows in band   ALL columns (no grouping here)
```

We don't need to group N because the column-major ordering within the band already
handles N efficiently — it processes one B-column at a time, shared by GROUP_SIZE_M
programs. Grouping N would add complexity for no benefit.

---

### Why local_col = Global Column (No Offset Needed)

Bands only slice **rows**, not columns. Every band covers the full column range:

```
                col 0   col 1   col 2       ← same columns for every band
              +-------+-------+-------+
Band 0 row 0 |       |       |       |     rows change per band
       row 1 |       |       |       |
              +-------+-------+-------+
Band 1 row 2 |       |       |       |     columns are always 0..num_pid_n-1
       row 3 |       |       |       |
              +-------+-------+-------+
```

For **rows**, local and global differ because each band starts at a different row:

```
Band 0: local row 0 = global row 0,  local row 1 = global row 1
Band 1: local row 0 = global row 2,  local row 1 = global row 3
                                  ↑ different!
```

That's why rows need the offset: `pid_m = first_pid_m + local_row`

For **columns**, local and global are always identical:

```
Band 0: local col 0 = global col 0,  local col 1 = global col 1,  local col 2 = global col 2
Band 1: local col 0 = global col 0,  local col 1 = global col 1,  local col 2 = global col 2
                                  ↑ same!
```

Every band starts at column 0. There is no column offset. So:

```python
pid_m = first_pid_m + local_row    # needs offset (depends on which band)
pid_n = local_col                  # no offset (all bands start at column 0)
```

---

### Why Group Along M? (And Not N?)

The choice is a **convention**, not a requirement. You could group along N and get the
same cache benefit for square matrices.

In typical ML workloads:

| Dimension | Represents | Size |
|-----------|------------|------|
| M | batch * seq_len (tokens) | Variable, often large |
| N | hidden_dim, FFN width | Fixed, e.g. 4096 |

When M >> N, grouping along M creates horizontal bands that sweep all N columns.
Since N is moderate, each band's B-data fits in L2 well. The convention is to group
along the larger/variable dimension, which in ML is usually M.

---

### Scaling Up: 9x9 Grid, GROUP_SIZE_M = 3

### Setup: 9x9 Grid, GROUP_SIZE_M = 3

```
num_pid_m = 9       (rows of output tiles)
num_pid_n = 9       (columns of output tiles)
GROUP_SIZE_M = 3
num_pid_in_group = GROUP_SIZE_M * num_pid_n = 3 * 9 = 27
```

81 total programs split into 3 groups of 27 programs each.

### Tracing the Mapping

```
group_id = pid // num_pid_in_group        → which horizontal band
first_pid_m = group_id * GROUP_SIZE_M     → first row of this band
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)  → handle remainder

position = pid % num_pid_in_group         → index within the group (0-26)
pid_m = first_pid_m + (position % 3)      → cycles rows:  0, 1, 2, 0, 1, 2, ...
pid_n = position // 3                     → advances col:  0, 0, 0, 1, 1, 1, ...
```

This is **column-major ordering** within each group.

### Group 0 Trace (pid 0-26)

```
pid  | position | pid_m | pid_n | tile
-----|----------|-------|-------|------
  0  |    0     |   0   |   0   | (0,0)
  1  |    1     |   1   |   0   | (1,0)
  2  |    2     |   2   |   0   | (2,0)
  3  |    3     |   0   |   1   | (0,1)
  4  |    4     |   1   |   1   | (1,1)
  5  |    5     |   2   |   1   | (2,1)
  6  |    6     |   0   |   2   | (0,2)
  7  |    7     |   1   |   2   | (1,2)
  8  |    8     |   2   |   2   | (2,2)
  9  |    9     |   0   |   3   | (0,3)
 10  |   10     |   1   |   3   | (1,3)
 11  |   11     |   2   |   3   | (2,3)
 ...          ...continues through column 8...
 26  |   26     |   2   |   8   | (2,8)
```

### Full 9x9 Grid — pid Assignment

```
          col 0   col 1   col 2   col 3   col 4   col 5   col 6   col 7   col 8
        +-------+-------+-------+-------+-------+-------+-------+-------+-------+
row 0   |   0   |   3   |   6   |   9   |  12   |  15   |  18   |  21   |  24   |
row 1   |   1   |   4   |   7   |  10   |  13   |  16   |  19   |  22   |  25   | Group 0
row 2   |   2   |   5   |   8   |  11   |  14   |  17   |  20   |  23   |  26   |
        +-------+-------+-------+-------+-------+-------+-------+-------+-------+
row 3   |  27   |  30   |  33   |  36   |  39   |  42   |  45   |  48   |  51   |
row 4   |  28   |  31   |  34   |  37   |  40   |  43   |  46   |  49   |  52   | Group 1
row 5   |  29   |  32   |  35   |  38   |  41   |  44   |  47   |  50   |  53   |
        +-------+-------+-------+-------+-------+-------+-------+-------+-------+
row 6   |  54   |  57   |  60   |  63   |  66   |  69   |  72   |  75   |  78   |
row 7   |  55   |  58   |  61   |  64   |  67   |  70   |  73   |  76   |  79   | Group 2
row 8   |  56   |  59   |  62   |  65   |  68   |  71   |  74   |  77   |  80   |
        +-------+-------+-------+-------+-------+-------+-------+-------+-------+
```

### Why This Helps: L2 Cache Behavior

The GPU runs ~16 programs concurrently. Programs with **nearby pids** execute around
the same time.

**Programs 0, 1, 2** (tiles `(0,0)`, `(1,0)`, `(2,0)`):
- A tiles: rows 0, 1, 2 (each with all K blocks) = 3 rows of A
- B tiles: column 0 (all K blocks) = 1 column of B

**Programs 3, 4, 5** (tiles `(0,1)`, `(1,1)`, `(2,1)`):
- A tiles: rows 0, 1, 2 = **same 3 rows, still in L2!**
- B tiles: column 1 = 1 new column of B

**Programs 6, 7, 8** (tiles `(0,2)`, `(1,2)`, `(2,2)`):
- A tiles: same rows 0-2, **still cached**
- B tiles: column 2, 1 new column

The A-tiles for rows 0-2 stay in L2 across all 27 programs in Group 0.

### Comparison: Row-Major vs Super-Grouped

For the first 9 output tiles (K/BLOCK_K = 9):

**Row-major** (pid 0-8 compute row 0, all 9 columns):

```
A tiles loaded:  1 row  x 9 K-blocks =  9 tiles
B tiles loaded:  9 cols x 9 K-blocks = 81 tiles
Total: 90 tiles through L2
```

**Super-grouped** (pid 0-8 compute 3x3 block):

```
A tiles loaded:  3 rows x 9 K-blocks = 27 tiles
B tiles loaded:  3 cols x 9 K-blocks = 27 tiles
Total: 54 tiles through L2
```

Row-major loads fewer A-tiles (1 vs 3) but far more B-tiles (81 vs 27).
Super-grouping trades some A-row reuse for massive B-column reuse.

### The Tradeoff

Increasing `GROUP_SIZE_M` increases B-reuse but decreases A-reuse (more A-rows loaded).
Sweet spot is typically `GROUP_SIZE_M = 8`.

---

## Modernization Notes

### 1. Manual Pointer Arithmetic vs. `tl.make_block_ptr`

The tutorial builds pointer blocks by hand. Modern Triton offers a higher-level API:

```python
# Modern alternative (not in tutorial):
a_block_ptr = tl.make_block_ptr(
    base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
    offsets=(pid_m * BLOCK_SIZE_M, 0),
    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1, 0)
)
a = tl.load(a_block_ptr, boundary_check=(0, 1))
a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
```

When to use which:
- **Manual pointers**: maximum control, needed for non-standard access patterns
- **`tl.make_block_ptr`**: cleaner for standard tiled access, enables compiler optimizations
- **TMA descriptors**: Hopper+ only, hardware-accelerated bulk loads (see Tutorial 09)

For Module 1.1 exercise: use manual pointers to learn the memory model.

### 2. Modular Wrapping `% M` vs. Explicit Clamping

Tutorial 03 uses `offs_am = (...) % M` which wraps to valid indices for garbage data
that gets masked out. Tutorial 09 uses the clearer pattern:

```python
# tutorials/09-persistent-matmul.py:130-136
offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
offs_am = tl.where(offs_am < M, offs_am, 0)
offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
```

The `tl.max_contiguous` / `tl.multiple_of` hints help the compiler generate more
efficient loads.

### 3. `tl.assume` Hints

Absent in Tutorial 03, present in Tutorial 09:

```python
# tutorials/09-persistent-matmul.py (matmul_kernel):270-277
tl.assume(pid_m >= 0)
tl.assume(pid_n >= 0)
tl.assume(stride_am > 0)
```

These enable the compiler to optimize address calculations and remove unnecessary
bounds checks.

---

## Bridge to Production

Production matmul in `triton_kernels/matmul.py` differs architecturally:

1. **Closure-based specialization** replaces `@triton.autotune`. The
   `SpecializationModule` at `matmul.py:69` binds kernel arguments at definition time,
   eliminating runtime autotuning overhead.

2. **Fused epilogues** — the production kernel (`_matmul.py`) accepts `ACTIVATION_FN`
   and `EPILOGUE_FN` as `tl.constexpr` parameters, allowing matmul + activation +
   quantization in one kernel launch.

3. **FlexCtx numerics** — `FlexCtx` wraps scale factors for FP8/flexpoint quantized
   matmul, abstracting the scale handling away from the core tiling logic.

4. **The same tiling logic persists** — the pid computation, accumulation loop, and
   store pattern in `_matmul.py` are structurally identical to the tutorial. Production
   adds complexity around the core, not inside it.
