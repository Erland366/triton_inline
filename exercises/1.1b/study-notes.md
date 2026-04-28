# Module 1.1b Study Notes: Benchmarking, Profiling & Debugging Toolkit

## Lesson 1: `triton.testing.Benchmark` and `do_bench`

### Benchmark Config Anatomy

```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],              # sweep variable names (x-axis)
        x_vals=[128 * i for i in range(2, 33)],  # values to sweep
        line_arg="provider",                   # argument selecting which kernel
        line_vals=["cublas", "triton"],        # implementations to compare
        line_names=["cuBLAS", "Triton"],       # legend labels
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",
        plot_name="matmul-performance-fp16",
        args={"fp8_inputs": False},            # fixed args
    )
)
def benchmark(M, N, K, provider, fp8_inputs):
    ...
```

Key fields:
- `x_names` + `x_vals` = x-axis parameter sweep
- `line_arg` + `line_vals` = compare multiple implementations in one plot
- `args` = parameters that stay fixed across all sweep points

### `do_bench` Semantics

```python
ms, min_ms, max_ms = triton.testing.do_bench(
    lambda: my_kernel(a, b, c),
    warmup=25,       # warm-up iterations (excluded)
    rep=100,         # timed repetitions
    quantiles=[0.5, 0.2, 0.8]  # median, p20, p80
)
```

- Returns wall-clock time in **milliseconds**
- Lambda captures the kernel call; do_bench handles CUDA sync internally
- With `quantiles`, returns (median, min_quantile, max_quantile)

### TFLOPS Formulas

| Operation | FLOPS Formula |
|-----------|--------------|
| Matmul (C = A @ B) | `2 * M * N * K` |
| Softmax | `5 * M * N` |
| LayerNorm | `5 * M * N` |
| Attention | `4 * B * H * S * S * D` |
| SwiGLU | `3 * M * N` |

### ms-to-TFLOPS Conversion

```python
tflops = 2 * M * N * K / (ms * 1e-3) / 1e12
```

- `ms * 1e-3` converts milliseconds to seconds
- `/ 1e12` converts FLOPS to TFLOPS
- Example: M=N=K=4096, ms=0.84 -> `2 * 4096^3 / 0.00084 / 1e12 = 163.6 TFLOPS`

---

## Lesson 2: Proton Profiling (our tier)

### Proton Lifecycle

```
proton.start("matmul", hook="triton")   # Start session
proton.deactivate()                     # Pause during setup
  for each_size:
    proton.activate(0)                  # Resume for benchmark loop
      with proton.scope(name, meta):    # Tag kernels with FLOPS/bytes
        kernel(...)
    proton.deactivate(0)               # Pause between sizes
proton.finalize()                       # Write .hatchet file
```

### Scope Metadata Keys

| Key | Meaning | Example (FP16 matmul) |
|-----|---------|----------------------|
| `"bytes"` | Total memory traffic | `2 * (M*K + N*K + M*N)` |
| `"flops16"` | FP16 operations | `2 * M * N * K` |
| `"flops32"` | FP32 operations | Same formula, different key |
| `"flops8"` | FP8 operations | For quantized kernels |

Rule: use `flopsN` where N = `dtype.itemsize * 8`. FP16 -> `flops16`.

### The proton_context Helper

```python
@contextmanager
def proton_context():
    proton.activate(0)
    try:
        yield
    finally:
        proton.deactivate(0)
```

### Viewing Results

```python
import triton.profiler.viewer as proton_viewer
tree, metrics = proton_viewer.parse(["tflop16/s", "time/ms"], "matmul.hatchet")
proton_viewer.print_tree(tree, metrics)
```

### What Proton Gives You That do_bench Doesn't

1. **Per-scope breakdown** — see each annotated kernel's throughput separately
2. **Bandwidth alongside compute** — reports GB/s from bytes metadata alongside TFLOPS
3. **Automatic accounting** — declare FLOPS/bytes once, Proton derives rates

---

## Lesson 3: ncu Conceptual Overview

ncu reads **hardware performance counters** that Proton cannot access.

### When to Use ncu Over Proton

| Need | Proton | ncu |
|------|--------|-----|
| TFLOPS throughput | Yes | Yes |
| Bandwidth (GB/s) | Yes | Yes |
| SM throughput % | No | Yes |
| Warp occupancy | No | Yes |
| Tensor core utilization | No | Yes |
| Cache hit rates | No | Yes |

Rule of thumb: Start with Proton for TFLOPS/bandwidth. Switch to ncu when you need
to understand *why* performance is what it is.

Key ncu metrics for matmul:
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` — DRAM utilization
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` — SM utilization
- SM% should be high (>70%), DRAM% moderate (high = poor tiling)

**Note:** On our system, ncu is blocked by ERR_NVGPUCTRPERM (perf_event_paranoid=4).

### Production Pattern: roofline.py

```python
def parse_profile(profile_path, useful_op_regex):
    from triton.profiler import viewer
    gf, _, _, _ = viewer.read(profile_path)
    useful = gf.filter(f"MATCH ('*', c) WHERE c.'name' =~ '{useful_op_regex}' AND c IS LEAF").dataframe
    bytes = int(useful["bytes"].sum())
    flops = int(sum(useful[[c for c in ["flops8", "flops16"] if c in useful.columns]].sum()))
    allops = gf.filter("MATCH ('*', c) WHERE c IS LEAF").dataframe
    time_ns = allops["time (ns)"].sum()
    return PerfRecord(time_ns=time_ns, flops=flops, bytes=bytes)
```

---

## Lesson 4: Debugging Methodology

### Diagnostic Flowchart

```
1. Run kernel in FP32
   +-- PASS -> Precision issue (check tolerance table)
   +-- FAIL -> Logic bug (go to isolation testing)

2. Examine error pattern (ref - yours):
   +-- Uniform small errors -> Precision (accumulation order)
   +-- Boundary-only errors -> Masking bug
   +-- Striped/periodic     -> Stride bug
   +-- Entire tiles wrong   -> Tile index bug

3. Isolation testing: controlled inputs to narrow down failure
```

### Expected Tolerances

| Dtype (in -> acc -> out) | atol | rtol |
|--------------------------|------|------|
| FP32 -> FP32 -> FP32 | 1e-5 | 1e-5 |
| FP16 -> FP32 -> FP16 | 1e-2 | 1e-2 |
| BF16 -> FP32 -> BF16 | 1e-1 | 1e-1 |

If max error is **10x above expected atol**, it's a logic bug, not precision.

### Tolerance Sweep

```python
for atol in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 5e-1, 1.0]:
    matches = torch.isclose(ref, actual, atol=atol, rtol=0).float().mean()
    print(f"atol={atol:.0e}: {matches*100:.1f}%")
```

Interpretation:
- **Gradual falloff** (95% at 1e-3, 100% at 1e-2) -> precision, expected
- **Sharp cliff** (60% at 1e-1, 100% at 1.0) -> bug affecting subset of elements
- **Outliers in specific rows/tiles** -> masking or stride bug

### Error Pattern Table

| Pattern | Likely Cause | Where to Look |
|---------|-------------|---------------|
| Uniform small error | Accumulation precision | Accumulator dtype, reduction order |
| Boundary errors | Masking bug | `tl.load(..., mask=..., other=0.0)` |
| Striped/periodic | Stride bug | Pointer arithmetic, stride params |
| Block-aligned errors | Tile index bug | `pid_m`, `pid_n`, super-grouping |
| Last-tile errors | Off-by-one in grid | `tl.cdiv` rounding |
| All zeros | Kernel not writing | Store mask, output pointer offsets |

### Isolation Testing Matrix

| Test Case | Input | What It Catches |
|-----------|-------|-----------------|
| Identity | A=I, B=X | Basic pointer/store bugs |
| Single tile | M=N=K=BLOCK_SIZE | Core algorithm |
| Boundary+1 | M=BLOCK_M+1 | Boundary masking |
| K not aligned | K % BLOCK_K != 0 | K-loop masking |
| Prime sizes | M=127, N=131, K=137 | All masking paths |
| Large K | M=N=64, K=8192 | Accumulation correctness |

### Triton-Specific Tools

- `tl.device_print("pid", pid)` — scalars only, filter with `if pid == 0:`
- NumPy/PyTorch side-by-side — rewrite one tile in Python, compare intermediates

### Debugging Checklist

1. Run FP32 verification
2. Run tolerance sweep
3. Visualize errors (heatmap)
4. Run isolation matrix
5. Add `tl.device_print` at failing tile
6. Write NumPy equivalent for comparison
7. Check pointer arithmetic (strides match `tensor.stride()`)
8. Check masks (`other=0.0` for loads, mask for stores)
9. Check accumulator dtype (FP32 for FP16/BF16 operands)
10. Check grid dimensions (`tl.cdiv`)

---

## Comprehension Answers

**Q1: TFLOPS from do_bench**
- Formula: `2 * M * N * K / (ms * 1e-3) / 1e12`
- For M=N=K=4096, ms=0.84: `2 * 4096^3 / 0.00084 / 1e12 = 163.6 TFLOPS`
- The `1e-3` converts ms to seconds, `1e12` converts to TFLOPS

**Q2: Proton vs do_bench**
- Both compute TFLOPS the same way (FLOPS / time)
- Proton adds: per-scope breakdown, bandwidth alongside compute, automatic rate computation from declared metadata

**Q3: Masking bug diagnosis**
- Sharp cliff at atol (100% at 1e-1, 40% at 1e-2) = not precision
- Errors at last row of every tile = boundary masking bug
- Next step: check `tl.load` mask and `tl.store` mask, then run isolation tests at boundary+1 sizes
