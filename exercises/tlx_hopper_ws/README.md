# TLX Hopper WS: Warp-Specialized GEMM Study

**Module:** TLX Hopper WS | **Focus:** producer/consumer warp specialization |
**GPU:** Hopper+

## Objective

Study `hopper_gemm_ws.py` as a synchronization and ownership problem before
trying to rewrite the kernel.

The pipelined GEMM exercise used:

- local SMEM buffers
- `tlx.async_load`
- async copy commit/wait groups
- `tlx.async_dot`
- `tlx.async_dot_wait`

The warp-specialized GEMM changes the model:

- a default producer task issues TMA descriptor loads
- replicated consumer tasks issue WGMMA
- full and empty mbarriers transfer buffer ownership
- a persistent grid assigns output tiles to SM workers
- optional features add 2-CTA multicast, warp barriers, epilogue subtiles, and
  row/column-major descriptor variants

Do not start from those optional features. Start with the single-CTA,
non-subtiled, normal-mbarrier case.

## Reference Files

- `compiled_resources/TLX/triton/third_party/tlx/tutorials/hopper_gemm_ws.py`
- `compiled_resources/TLX/triton/third_party/tlx/tutorials/hopper_gemm_pipelined.py`
- `compiled_resources/TLX/triton/third_party/tlx/doc/tlx_barriers.md`
- `exercises/tlx_hopper_pipelined/hopper_gemm_pipelined.py`

## Exercise Files

| file | purpose |
|------|---------|
| `exercise.py` | Worksheet-style state-machine scaffold. Fill the blanks by reading the upstream WS kernel. |
| `exercise_me.py` | Student-written skinny and TMA GEMM implementation under active development. |
| `test_exercise_me.py` | Hopper correctness regression for the skinny TMA wrapper and delayed-MMA buffer reuse. |
| `benchmark.py` | Benchmark scaffold for later comparison against Torch and the pipelined exercise. |

## Study Order

### 1. Fixed Configuration

Use this mental config first:

```python
{
    "BM": 128,
    "BN": 256,
    "BK": 64,
    "GROUP_SIZE_M": 8,
    "NUM_STAGES": 3,
    "NUM_MMA_WARPS": 8,
    "NUM_MMA_GROUPS": 2,
    "EPILOGUE_SUBTILE": False,
    "NUM_CTAS": 1,
    "USE_WARP_BARRIER": False,
}
```

This removes 2-CTA multicast, warp-barrier variants, and epilogue subtile
stores from the first pass.

### 2. Producer Task

Read the default task region in the upstream kernel. Identify:

- which buffers it writes
- which empty barriers it waits on
- which full barriers it signals
- why A has one slot per MMA group but B has one shared slot per stage
- where phase tracking comes from

### 3. Consumer Tasks

Read the replicated consumer task region. Identify:

- how the replica id selects the A split
- why both consumers wait on the same B full barrier
- when each consumer releases A
- when B becomes reusable
- why the accumulator shape is half of `BM` by full `BN`

### 4. Persistent Scheduling

Explain why the launch grid is based on SM workers rather than output tiles.

Compare:

```python
grid = (num_pid_m * num_pid_n,)
```

against:

```python
grid = (min(NUM_SMS, total_tiles),)
```

## Commands

The worksheet is CPU-only and should run locally:

```bash
source .venv/bin/activate && python exercises/tlx_hopper_ws/exercise.py
```

The benchmark scaffold is intentionally incomplete:

```bash
source .venv/bin/activate && python exercises/tlx_hopper_ws/benchmark.py
```

Fill the worksheet first. Only then complete the benchmark.

Run the student TMA correctness regression on the Modal H100 environment:

```bash
source .venv/bin/activate
modal run run_modal.py \
  --action pytest \
  --script "exercises/tlx_hopper_ws/test_exercise_me.py -q"
```

The regression uses FP16 `(M, N, K) = (128, 256, 4096)`. On the target
Hopper configuration, the wrapper selects split-K while leaving enough K
iterations per program to wrap and reuse the circular TMA buffers multiple
times. This makes stale or prematurely overwritten buffer contents observable
against the `torch.matmul` reference.

## Acceptance Criteria

1. `exercise.py` explains the producer/consumer roles without copying the full
   upstream kernel.
2. The fixed configuration is understood before enabling autotune.
3. `benchmark.py` compares at least Torch, pipelined TLX, and upstream WS TLX
   using TFLOP/s.
4. All tested GEMM cases use fp16 or bf16 inputs and `K % 64 == 0`.
5. Any benchmark result is recorded in `benchmark_results/` or summarized in a
   short report.
6. `test_exercise_me.py` passes on Hopper before benchmarking the student TMA
   implementation.

## Boundaries

Do not implement these in the first pass:

- K-tail handling
- `NUM_CTAS=2`
- `USE_WARP_BARRIER=True`
- `EPILOGUE_SUBTILE=True`
- column-major input variants
- custom autotune pruning

Those are separate follow-up exercises after the basic state machine is clear.
