# Module 1.2 Study Notes: Fused Softmax & LayerNorm

## Key Concept: Fusion for Bandwidth-Bound Operations

Naive PyTorch softmax reads 5MN+2M elements, writes 3MN+2M. Fused kernel reads MN, writes MN.
~4x reduction in memory traffic. Since these are bandwidth-bound (low AI), reducing traffic = direct speedup.

**Opposite of matmul:** matmul is compute-bound (optimize TFLOPS), reductions are memory-bound (optimize DRAM throughput %).

## Softmax Kernel Pattern

- One program per row (or persistent: stride across rows)
- `BLOCK_SIZE = triton.next_power_of_2(n_cols)` — entire row in one block (SRAM constraint)
- `other=-float('inf')` for masked loads — `exp(-inf) = 0`, doesn't affect sum
  - If you used `other=0.0`, `exp(0) = 1` would corrupt the denominator
- Numerical stability: subtract `tl.max(row)` before `tl.exp`
- Reductions: `tl.max(row, axis=0)`, `tl.sum(numerator, axis=0)`

## LayerNorm Forward Pattern

- Two-pass (mean, variance) + normalize pass — three reads of X
- Not 3x slower because after first read, data is in L2 cache (~5 TB/s) not DRAM (~1 TB/s)
  - RTX 4090 has 72MB L2. Typical row = 4096 elements * 2 bytes = 8KB. Tiny vs 72MB.
  - LRU eviction policy keeps recently-read rows resident
- Accumulate in FP32: `.to(tl.float32)` even for FP16 inputs
- `other=0.0` for masked loads — zeros don't affect sum (mean/variance)

## LayerNorm Backward — Parallel Reduction

- **`dx`**: per-row gradient, one program per row, no synchronization needed
- **`dw`/`db`**: shared weights, ALL rows contribute → race condition
- Solution: `GROUP_SIZE_M` partial buffers + atomic locks
  - Stage 1: each program locks a buffer, adds its partial sum, releases
  - Stage 2: separate kernel reduces partial buffers → final `dw`/`db`
- `tl.atomic_cas(Lock, 0, 1)` — compare-and-swap for spin lock
- `tl.debug_barrier()` before releasing lock — ensure stores complete

## SRAM Constraint

Row must fit in SRAM: `MAX_FUSED_SIZE = 65536 // element_size`
- FP16: 32K elements max (covers hidden dims up to 32768)
- FP32: 16K elements max

## Modernization Notes

- Manual SRAM/occupancy query (Tutorial 02) → use autotuning instead
- Apex comparison (Tutorial 05) → compare against `torch.nn.functional.layer_norm`
- `torch.softmax(x, dim=-1)` is the real baseline (already fused via cuDNN)

## Bridge to Production

`triton_kernels/reduce.py`:
- Generic reduction + `PostprocessFn` epilogue → softmax = reduction + exp/normalize postprocess
- `SpecializationModule` + `ClosureArg` for closure-based dispatch
- Same pattern handles layernorm, softmax, attention reductions
