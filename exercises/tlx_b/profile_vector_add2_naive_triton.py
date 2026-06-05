# type: ignore
"""
Profile scaffold: TLX Vector Add With Proton Intra-Kernel Tracing
Module: TLX-B

This file is intentionally not a completed kernel solution.

Use it after `exercises/tlx/exercise.py` is correct:
1. Copy your completed TLX warp-specialized kernel structure into the TODOs.
2. Add the Proton scopes exactly around the async-task bodies.
3. Set PROFILE_KERNEL_READY = True.
4. Run this script to generate `tlx-add2.chrome_trace`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
import triton.profiler as proton
import triton.profiler.language as pl
import triton.profiler.mode as pmode
from triton._internal_testing import is_hopper_or_newer 

DEVICE = triton.runtime.driver.active.get_active_torch_device()
PROFILE_KERNEL_READY = True
PROFILED_TOTAL_WARPS = 12

@triton.jit
def add2_profiled_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    pl.enter_scope("add2_kernel") 

    pl.enter_scope("default_task_x_plus_y")
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = x + y
    tl.store(z_ptr + offsets, z, mask=mask)
    pl.exit_scope("default_task_x_plus_y")

    pl.enter_scope("default_task_a_plus_b")
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)
    pl.exit_scope("default_task_a_plus_b")

    pl.exit_scope("add2_kernel")

def add2_profiled(
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    block_size: int
):
    z = torch.empty_like(x)
    c = torch.empty_like(a)
    assert x.device == DEVICE and y.device == DEVICE
    assert a.device == DEVICE and b.device == DEVICE
    assert z.device == DEVICE and c.device == DEVICE
    assert x.numel() == y.numel() == z.numel()
    assert a.numel() == b.numel() == c.numel()
    assert x.numel() == a.numel()

    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    add2_profiled_kernel[grid](
        x,
        y,
        z,
        a,
        b,
        c,
        n_elements,
        BLOCK_SIZE=block_size
    )
    return z, c

def reference_dual_add(
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
):
    return x + y, a + b

def build_proton_mode(args: argparse.Namespace):
    optimizations = "clock32,time_shift" if args.increase_accuracy else "clock32"

    if args.granularity != "warp":
        raise ValueError(
            "This Proton/TLX path currently supports only --granularity warp. "
            f"--granularity {args.granularity!r} reaches the NVIDIA lowering error "
            "'granularity must be warp for now'. Use --granularity warp with "
            "--warp-ids to sample the warp groups you care about."
        )

    if args.all_warps:
        args.effective_warp_ids = "all"
        return pmode.Default(
            granularity=args.granularity,
            buffer_type=args.buffer_type,
            buffer_size=0,
            optimizations=optimizations
        )

    warp_ids = [warp_id.strip() for warp_id in args.warp_ids.split(",") if warp_id.strip()]
    if len(warp_ids) & (len(warp_ids) - 1):
        raise ValueError(
            "Proton selective sampling requires the number of sampled warp ids to be a power of two. "
            f"Got {len(warp_ids)} ids from --warp-ids={args.warp_ids!r}."
        )
    sampling_options = ",".join(warp_ids)
    args.effective_warp_ids = sampling_options
    return pmode.Default(
        granularity=args.granularity,
        buffer_type=args.buffer_type,
        buffer_size=0,
        optimizations=optimizations
    )

def run_profile(args: argparse.Namespace):
    output_base = Path(args.output)
    mode = build_proton_mode(args)

    print(f"Problem size: {args.size} elements")
    print(f"Block size: {args.block_size} elements")
    print(f"Expected CTAs: {triton.cdiv(args.size, args.block_size)}")
    print(f"Sampled warp ids: {args.effective_warp_ids}")
    print(f"Proton granularity: {args.granularity}")
    print(f"Proton buffer size: {args.buffer_size} bytes")

    pl.enable_semantic("triton")
    if args.op_measure:
        proton.start(str(output_base), backend="instrumentation", mode=mode)
        expected_artifact = output_base.with_suffix(".hatchet")
    else:
        proton.start(str(output_base), data="trace", backend="instrumentation", mode=mode)
        expected_artifact = output_base.with_suffix(".chrome_trace")

    torch.manual_seed(args.seed)
    x = torch.rand(args.size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(args.size, device=DEVICE, dtype=torch.float32)
    a = torch.rand(args.size, device=DEVICE, dtype=torch.float32)
    b = torch.rand(args.size, device=DEVICE, dtype=torch.float32)

    z, c = add2_profiled(x, y, a, b, args.block_size)
    torch.cuda.synchronize()
    proton.finalize()

    ref_z, ref_c = reference_dual_add(x, y, a, b)
    torch.testing.assert_close(z, ref_z)
    torch.testing.assert_close(c, ref_c)

    print("Correctness OK.")
    print(f"Profile mode: {'hatchet tree' if args.op_measure else 'chrome trace'}")
    print(f"Expected artifact: {expected_artifact}")
    if not args.op_measure:
        print("Open the .chrome_trace file in https://ui.perfetto.dev/")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="tlx-add2", help="Output path without profile suffix.")
    parser.add_argument("--size", type=int, default=8192, help="Number of elements per vector.")
    parser.add_argument("--block-size", type=int, default=1024, help="Elements per Triton program.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--op-measure",
        action="store_true",
        help="Write a .hatchet tree profile instead of a .chrome_trace timeline.",
    )
    parser.add_argument(
        "--all-warps",
        action="store_true",
        help="Profile all warps. Start with selective sampling unless the trace is too sparse.",
    )
    parser.add_argument(
        "--warp-ids",
        default="0,4,8,11",
        help="Comma-separated warp IDs for selective sampling.",
    )
    parser.add_argument(
        "--granularity",
        default="warp",
        choices=[
            "cta",
            "warp",
            "warp_2",
            "warp_4",
            "warp_8",
            "warp_group",
            "warp_group_2",
            "warp_group_4",
            "warp_group_8",
        ],
    )
    parser.add_argument(
        "--buffer-type",
        default="global",
        choices=["shared", "global"],
        help="Use global for first traces to avoid shared-memory capacity surprises.",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=0,
        help=(
            "Total Proton buffer size in bytes. Use 0 for Proton's default "
            "16KB per profiling segment."
        ),
    )
    parser.add_argument(
        "--increase-accuracy",
        action="store_true",
        help="Enable Proton's time_shift optimization in addition to clock32.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_profile(parse_args())