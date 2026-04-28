# Kernel Benchmark: <short-name>

- **Date**: YYYY-MM-DD
- **Author**: your-name / Claude-assisted
- **Goal**: one-paragraph description
- **General description**: short, non-technical summary
- **Kernel type**: matmul / attention / reduction / custom
- **Implementation**: CUDA / Triton / both

---

## 1. Setup

### 1.1 Problem Description

- **Operation**:
- **Input shapes**:
- **Output shape**:
- **Data type**: fp32 / fp16 / bf16 / int8

### 1.2 Hardware

- **GPU**: e.g., NVIDIA A100-80GB
- **CUDA version**:
- **Triton version** (if applicable):
- **Theoretical peak**:
  - FP32: X TFLOPS
  - FP16: X TFLOPS
  - Memory bandwidth: X GB/s

### 1.3 Baseline Implementations

| baseline | description | expected_performance |
|----------|-------------|---------------------|
| PyTorch native | `torch.matmul()` | |
| cuBLAS | via PyTorch | |
| Triton tutorial | reference impl | |

---

## 2. Implementation Details

### 2.1 Algorithm

<description of the algorithm and any optimizations>

### 2.2 Thread/Block Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Block size (x, y, z) | | |
| Grid size | | |
| Threads per block | | |
| Shared memory per block | | |
| Registers per thread | | |

### 2.3 Memory Access Pattern

- **Global memory**: coalesced / strided / random
- **Shared memory**: used / not used / bank conflicts?
- **L1/L2 cache**: expected hit rate
- **Tiling strategy**: BLOCK_M x BLOCK_N x BLOCK_K = ...

### 2.4 Triton-Specific (if applicable)

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': , 'BLOCK_N': , 'BLOCK_K': }, num_stages=, num_warps=),
        # ... more configs
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def kernel_name(...):
    # key implementation details
    pass
```

---

## 3. Runs

### 3.1 Run Table

| run_id | description | key_change | block_size | tiling | shared_mem |
|--------|-------------|------------|------------|--------|------------|
| r1 | baseline | - | | | |
| r2 | | | | | |

### 3.2 Notes per run

- **r1**:
- **r2**:

---

## 4. Results

### 4.1 Performance Metrics

| run_id | time_ms | TFLOPS | bandwidth_GB/s | % peak_compute | % peak_bandwidth |
|--------|---------|--------|----------------|----------------|------------------|
| baseline | | | | | |
| r1 | | | | | |
| r2 | | | | | |

### 4.2 Profiler Metrics (from Nsight/NCU)

| run_id | occupancy | registers | shared_mem | L1_hit | L2_hit | DRAM_throughput |
|--------|-----------|-----------|------------|--------|--------|-----------------|
| | | | | | | |

### 4.3 Roofline Analysis

- **Arithmetic intensity**: X FLOPS/byte
- **Memory bound / Compute bound**:
- **Bottleneck**:

### 4.4 Scaling Behavior

| M | N | K | time_ms | TFLOPS | notes |
|---|---|---|---------|--------|-------|
| | | | | | |

---

## 5. Analysis

### 5.1 What worked

-

### 5.2 What didn't work

-

### 5.3 Key insights

- Tiling strategy:
- Memory access:
- Occupancy vs ILP tradeoff:

### 5.4 Comparison to Baselines

| implementation | speedup_vs_pytorch | speedup_vs_cublas | notes |
|----------------|-------------------|-------------------|-------|
| | | | |

---

## 6. Lessons Learned -> Candidate Skills

List patterns that should become skills:

- **Candidate skill 1**:
- **Candidate skill 2**:

### Code to preserve

```python
# Best performing kernel configuration
@triton.jit
def optimized_kernel(...):
    # ... implementation
    pass
```

---

## 7. References

- This report: `benchmark_results/<filename>.md`
- Source code: `kernels/...`
- Profiling data: `profiles/...`
- Related benchmarks:
