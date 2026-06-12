# type: ignore
"""
Benchmark scaffold: Hopper GEMM pipelined vs warp-specialized TLX.
Module: TLX Hopper WS

This file is intentionally incomplete. Fill it only after `exercise.py` is
complete enough that the producer/consumer state machine is clear.

Rules:
- Use fp16 or bf16 inputs only.
- Keep K divisible by 64.
- Report TFLOP/s, not GB/s.
- Start with one fixed WS config before enabling upstream autotune.
"""

from __future__ import annotations

from pathlib import Path

import torch
import triton
import triton.testing


DEVICE = triton.runtime.driver.active.get_active_torch_device()


FIXED_WS_CONFIG = {
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


def load_pipelined_matmul():
    """
    TODO 1: Import your completed pipelined GEMM wrapper.

    Expected return value: a callable like `matmul_hopper(a, b)`.
    """
    raise NotImplementedError("Import matmul_hopper from exercises/tlx_hopper_pipelined.")


def load_ws_matmul():
    """
    TODO 2: Import the upstream WS GEMM wrapper.

    Expected return value: a callable like `matmul(a, b, config=FIXED_WS_CONFIG)`.

    Hint: first confirm the import path used by your Modal TLX environment.
    """
    raise NotImplementedError("Import hopper_gemm_ws.matmul from the TLX tutorial package.")


def verify_before_benchmark():
    """
    TODO 3: Compare Torch, pipelined TLX, and WS TLX on one small valid shape.

    Use:
    - dtype torch.bfloat16 or torch.float16
    - K divisible by 64
    - explicit tolerances suitable for tensor-core matmul
    """
    raise NotImplementedError("Add a correctness gate before benchmarking.")


configs = [
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[1024, 2048, 4096, 8192],
        x_log=True,
        line_arg="provider",
        line_vals=["torch", "pipelined_tlx", "ws_tlx"],
        line_names=["Torch", "Pipelined TLX", "WS TLX"],
        ylabel="TFLOP/s",
        plot_name="hopper-ws-vs-pipelined",
        args={},
    )
]


@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    """
    TODO 4: Allocate inputs, run the selected provider, and return TFLOP/s.

    Keep dimension variables named M/N/K. Name tensors `a` and `b` so the
    TFLOP/s formula stays pure Python numeric math.
    """
    raise NotImplementedError("Fill benchmark provider branches and TFLOP/s formula.")


def main():
    verify_before_benchmark()
    save_path = Path("plots")
    save_path.mkdir(exist_ok=True)
    benchmark.run(print_data=True, show_plots=False, save_path=save_path)


if __name__ == "__main__":
    main()
