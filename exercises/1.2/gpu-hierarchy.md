# GPU Compute & Memory Hierarchy (Triton Perspective)

> Reference for understanding where data lives and what you control in Triton.
> Numbers are for RTX 4090 (Ampere, compute capability 8.9).

---

## Compute Hierarchy

```
GPU (1 device)
└── SMs (128 on RTX 4090) — Streaming Multiprocessors
    └── Warp Schedulers (4 per SM on Ampere)
        └── Warps (32 threads each, execute in lockstep)
            └── Threads (the actual CUDA cores)
```

One Triton **program** = one **CTA** (thread block) = multiple warps on one SM.

| Concept | What it is | You control it? |
|---------|-----------|-----------------|
| Grid | How many programs to launch | Yes — `kernel[grid]()` |
| Program/CTA | One instance of your kernel | Implicitly — one per grid element |
| `num_warps` | Warps per program (e.g., 4 = 128 threads) | Yes |
| Warp | 32 threads in lockstep | No — hardware unit |
| Thread | Executes one element's worth of work | No — invisible in Triton |

---

## Memory Hierarchy (fast to slow)

```
Registers          ~255 per thread     │ Per-thread, fastest
  ↕                                    │
Shared Memory      ~128KB per SM       │ Per-SM, on-chip SRAM
(SRAM)                                 │ Shares physical space with L1
  ↕                                    │
L1 Cache           ~128KB per SM       │ Per-SM, hardware-managed
                                       │ Same physical SRAM as shared memory
  ↕                                    │
L2 Cache           72MB                │ Shared across ALL SMs
  ↕                                    │
DRAM (GDDR6X)      24GB, ~1 TB/s      │ Global memory, off-chip
```

**Shared memory and L1 share the same 128KB of physical SRAM per SM.**
The split is configurable (e.g., 64KB shared + 64KB L1). Triton handles this automatically.

---

## What You Control in Triton

| Parameter | What it affects | Memory impact |
|-----------|----------------|---------------|
| `BLOCK_SIZE` | Elements per program | Register usage (bigger = more registers per thread) |
| `num_warps` | Threads per program | Registers split across more threads (less per thread) |
| `num_stages` | Pipeline depth | Shared memory usage (more stages = more SRAM buffers) |
| Grid size | Number of programs | SM occupancy (too few = idle SMs) |

**That's it.** You do NOT control:
- Which SM runs your program
- What goes in registers vs shared memory vs spills
- L1 or L2 cache behavior
- Shared memory allocation

The Triton compiler decides all of that.
In CUDA you'd write `__shared__ float smem[1024]` — in Triton there's no equivalent.

---

## Where Data Lives During Execution

```python
row = tl.load(X + cols, mask=mask, other=0.)  # DRAM → L2 → L1 → Registers
_mean += row                                    # Registers (computation)
mean = tl.sum(_mean, axis=0) / N               # Registers → scalar register
tl.store(Y + cols, y, mask=mask)               # Registers → L1 → L2 → DRAM
```

- `tl.load`: data travels **DRAM → L2 → L1 → Registers**
- All computation: **Registers**
- `tl.store`: data travels **Registers → L1 → L2 → DRAM**
- `tl.dot`: compiler may stage operands through **shared memory** (invisible to you)
- Re-reading same address: might hit **L1** or **L2** instead of DRAM (no guarantee)

---

## Quick Reference Table

| Level | You Control | Size (4090) | Bandwidth | Scope |
|-------|-------------|-------------|-----------|-------|
| Registers | Indirectly* | ~255/thread | Fastest | Per-thread |
| Shared Mem (SRAM) | No** | ~128KB/SM | ~19 TB/s | Per-SM |
| L1 Cache | No | ~128KB/SM | ~19 TB/s | Per-SM |
| L2 Cache | No | 72MB | ~5 TB/s | All SMs |
| DRAM | No | 24GB | ~1 TB/s | All SMs |

\* `BLOCK_SIZE` and `num_warps` determine register pressure
\*\* `num_stages` affects shared memory usage via the compiler

---

## Register Budget Example

Ampere: max 255 registers per thread.

With `num_warps=4` (128 threads) and `BLOCK_SIZE=1024`:
- Each thread handles `1024 / 128 = 8` elements
- One FP32 variable of BLOCK_SIZE = 8 registers per thread
- Kernel with 5 block-sized variables = 40 registers per thread → fine
- Kernel with 30 block-sized variables = 240 registers per thread → near limit, may spill

With `BLOCK_SIZE=8192` and `num_warps=4`:
- Each thread handles `8192 / 128 = 64` elements
- One variable = 64 registers per thread
- 4 variables = 256 registers → exceeds limit, compiler spills to local memory (slow)

**Rule of thumb:** keep `BLOCK_SIZE / (num_warps * 32)` small enough that your total
register usage stays under ~255 per thread. Use fewer variables or more warps if needed.

---

## How Triton Maps to This Hierarchy

| Triton Code | Hardware Reality |
|-------------|-----------------|
| `tl.program_id(0)` | CTA index in the grid |
| `tl.arange(0, BLOCK_SIZE)` | Distributed across warps/threads in registers |
| `tl.load(ptr + offs)` | Coalesced DRAM read → L2 → L1 → registers |
| `tl.store(ptr + offs, val)` | Registers → L1 → L2 → coalesced DRAM write |
| `tl.dot(A, B, acc)` | Tensor core HMMA instructions, operands staged via shared memory |
| `tl.sum(x, axis=0)` | Warp shuffle reductions (no shared memory) |
| `tl.max(x, axis=0)` | Warp shuffle reductions (no shared memory) |
| `for ... in range(...)` | Sequential loop on one SM |
| `num_stages=3` | Compiler inserts cp.async + triple-buffer in shared memory |
