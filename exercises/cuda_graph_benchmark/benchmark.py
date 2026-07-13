"""
Benchmark Triton kernel timing with and without CUDA graph replay.

This intentionally benchmarks a tiny memory-bandwidth style kernel so the
difference between normal launch timing and CUDA graph replay is easy to see.

Usage:
    python exercises/cuda_graph_benchmark/benchmark.py
    python exercises/cuda_graph_benchmark/benchmark.py --sizes 1048576 16777216
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import triton
import triton.language as tl
import triton.testing


DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor, block_size: int) -> None:
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, block_size),)
    _add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=block_size)


def verify_once(size: int, block_size: int) -> None:
    x = torch.randn(size, device=DEVICE, dtype=torch.float32)
    y = torch.randn(size, device=DEVICE, dtype=torch.float32)
    out = torch.empty_like(x)
    add(x, y, out, block_size)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x + y, atol=0.0, rtol=0.0)


def bandwidth_gbps(size: int, ms: float) -> float:
    # fp32 add reads x and y and writes out: 3 tensors * 4 bytes.
    bytes_moved = size * 3 * 4
    return bytes_moved / (ms * 1e-3) / 1e9


def make_benchmark(sizes: list[int], warmup_ms: int, rep_ms: int, block_size: int):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=sizes,
            x_log=True,
            line_arg="mode",
            line_vals=["normal", "cuda_graph"],
            line_names=["do_bench", "do_bench_cudagraph"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="GB/s",
            plot_name="triton-cuda-graph-vs-normal",
            args={
                "warmup_ms": warmup_ms,
                "rep_ms": rep_ms,
                "block_size": block_size,
            },
        )
    )
    def benchmark(N, mode, warmup_ms, rep_ms, block_size):  # noqa: N803
        x = torch.randn(N, device=DEVICE, dtype=torch.float32)
        y = torch.randn(N, device=DEVICE, dtype=torch.float32)
        out = torch.empty_like(x)

        # Compile before timing; CUDA graph capture also needs stable allocations.
        add(x, y, out, block_size)
        torch.cuda.synchronize()

        fn = lambda: add(x, y, out, block_size)  # noqa: E731
        quantiles = [0.5, 0.2, 0.8]
        if mode == "normal":
            ms, min_ms, max_ms = triton.testing.do_bench(
                fn,
                warmup=warmup_ms,
                rep=rep_ms,
                quantiles=quantiles,
            )
        elif mode == "cuda_graph":
            ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
                fn,
                rep=rep_ms,
                quantiles=quantiles,
            )
        else:
            raise ValueError(f"unknown benchmark mode: {mode}")

        return (
            bandwidth_gbps(N, ms),
            bandwidth_gbps(N, max_ms),
            bandwidth_gbps(N, min_ms),
        )

    return benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[2**12, 2**16, 2**20, 2**24, 2**26],
        help="Vector sizes to benchmark.",
    )
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--save-path", type=Path, default=Path("plots"))
    parser.add_argument("--show-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_once(min(args.sizes), args.block_size)
    args.save_path.mkdir(parents=True, exist_ok=True)
    benchmark = make_benchmark(
        sizes=args.sizes,
        warmup_ms=args.warmup_ms,
        rep_ms=args.rep_ms,
        block_size=args.block_size,
    )
    benchmark.run(
        print_data=True,
        show_plots=args.show_plots,
        save_path=str(args.save_path),
    )
    print(f"\nSaved plot and CSV under: {args.save_path}")


if __name__ == "__main__":
    main()
