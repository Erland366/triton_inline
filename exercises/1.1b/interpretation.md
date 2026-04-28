# Sub-exercise 3: Performance Interpretation

**Module:** 1.1b | **Kernel:** Module 1.1 Matmul

After running your benchmark (exercise.py) and profile (profile.py), answer
the following questions. Use actual numbers from your results.

---

## 1. Compute-bound or Memory-bound?

**Arithmetic intensity** = FLOPS / bytes transferred.

For FP16 matmul at M=N=K=n:
- FLOPS = 2 * n^3
- Bytes = 2 * (n^2 + n^2 + n^2) = 6 * n^2  (read A + read B + write C, FP16 = 2 bytes)
- AI = 2n^3 / (6n^2) = n/3

| Size (n) | Arithmetic Intensity | Classification |
|----------|---------------------|----------------|
| 512 | 170.6 | compute bound (almost memory bound) |
| 1024 | 341.3 | compute bound |
| 2048 | 682.6 | compute bound |
| 4096 | 1365.3 | compute bound |

The RTX 4090 roofline knee is roughly at AI ~100 (peak compute ~165 TFLOPS,
peak bandwidth ~1 TB/s, knee = 165e12 / 1e12 = 165).

**Your conclusion:** All of them is compute bound

---

## 2. Throughput Analysis

From your benchmark results:

| Size | Triton TFLOPS | cuBLAS TFLOPS | % of cuBLAS |
|------|--------------|--------------|-------------|
| 512 | 23.6 | 36.4 | 64% |
| 1024 | 126.2 | 109.4 | 115% |
| 2048 | 157.4 | 159.3 | 98% |
| 4096 | 166.0 | 165.6 | 100% |

**Observations:**
- How does throughput scale with size?
    - Jumps sharply from 512→1024 (23.6→126.2 TFLOPS), then plateaus toward peak (~165 TFLOPS). The jump is because n=512 with BLOCK=128 tiles gives only 4×4=16 tiles for 128 SMs — most SMs sit idle (occupancy problem, not memory-bound).
- At which size does your kernel perform best relative to cuBLAS?
    - At 1024 (115%). Likely because autotune found a config that happens to beat cuBLAS's heuristic at this size.
- Why might smaller sizes show different behavior than larger ones?
    - Small sizes have too few tiles to fill all 128 SMs (n=512 → 16 tiles). The GPU is underutilized regardless of how good the kernel is. Large sizes have thousands of tiles, so SM occupancy is high and the kernel can approach peak compute.

---

## 3. Bottleneck Identification

From your Proton profile (tflop16/s and gbps):

| Size | tflop16/s (triton) | gbps (triton) | Bottleneck |
|------|-------------------|---------------|------------|
| 1024 | 125.3 (76% peak) | 367.0 (36% peak) | Compute-bound |
| 2048 | 157.9 (96% peak) | 231.4 (23% peak) | Compute-bound |
| 4096 | 165.0 (100% peak) | 120.9 (12% peak) | Compute-bound |

Peak references: FP16 compute ~165 TFLOPS, bandwidth ~1008 GB/s.

**Your analysis:**
- What limits performance at the smallest size (1024)?
    - Compute is the bottleneck (76% of peak compute vs 36% of peak bandwidth). The gap from 100% is due to insufficient tile parallelism — 8×8=64 tiles for 128 SMs leaves some SMs underloaded. Wave quantization wastes cycles.
- What limits performance at the largest size (4096)?
    - Still compute-bound, but now at 100% of peak. Bandwidth drops to 12% of peak because arithmetic intensity is very high (AI=1365). The kernel is doing maximum useful compute per byte transferred.
- If you wanted to improve performance, what would you try?
    - For small sizes: use smaller block sizes (e.g., 32×32) to create more tiles and fill all SMs. Trade per-tile efficiency for better occupancy.
    - For large sizes: already at peak — no room to improve without changing the algorithm (e.g., persistent kernels that reduce tile launch overhead).

---

## 4. Summary

All tested matrix sizes are compute-bound (AI >> roofline knee of ~165), but small sizes suffer from poor SM occupancy — 16 tiles can't fill 128 SMs, so throughput drops to 14% of peak at n=512. As size increases, tile parallelism saturates the GPU and throughput approaches 100% of peak FP16 compute at n=4096. The main lever for improving small-size performance is creating more tiles (smaller blocks), while large sizes are already at the hardware ceiling.
