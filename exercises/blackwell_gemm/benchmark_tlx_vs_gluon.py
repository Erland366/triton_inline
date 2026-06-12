# type: ignore
"""
Benchmark TLX Blackwell GEMM against Gluon Blackwell GEMM variants.

Run on a Blackwell CUDA GPU:

    python exercises/blackwell_gemm/benchmark_tlx_vs_gluon.py

The TLX side uses the optimized Blackwell tutorial wrappers. The Gluon side
keeps the newer CLC and multi-CTA kernels directly in this file so the
benchmark has one runnable script and does not depend on copying upstream
tutorial files into the exercise directory.
"""

import argparse
import importlib
import os
import sys
from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl
import triton.testing
from triton._internal_testing import is_blackwell
from triton.language.core import _aggregate as aggregate
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.blackwell import TensorDescriptor as GluonTensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared as gluon_fence_async_shared,
    mbarrier as gluon_mbarrier,
    tma as gluon_tma,
)
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    get_tmem_reg_layout,
    tensor_memory_descriptor,
    tcgen05_commit,
    tcgen05_mma,
    tcgen05_mma_barrier_count,
)

try:
    from triton.experimental.gluon.language.nvidia.blackwell import clc as blackwell_clc
except ImportError:
    blackwell_clc = None

gluon_tma_async_load = getattr(gluon_tma, "async_load", gluon_tma.async_copy_global_to_shared)

from triton.language.extra.tlx.tutorials.blackwell_gemm_2cta import (
    matmul as tlx_blackwell_2cta_matmul,
)
from triton.language.extra.tlx.tutorials.blackwell_gemm_clc import (
    matmul as tlx_blackwell_clc_matmul,
)
from triton.language.extra.tlx.tutorials.blackwell_gemm_ws import (
    matmul as tlx_blackwell_ws_matmul,
)


DEVICE = triton.runtime.driver.active.get_active_torch_device()
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIZES = [2048, 4096, 8192]
DEFAULT_PROVIDERS = [
    "torch",
    "tlx_ws",
    "tlx_clc",
    "tlx_2cta",
    "gluon_ws",
    "gluon_clc_static",
]


def _gluon_tutorial_dirs() -> list[Path]:
    """Return candidate directories that contain the Gluon tutorial scripts."""
    installed_triton_root = Path(triton.__file__).resolve().parents[1]
    return [
        installed_triton_root / "tutorials" / "gluon",
        REPO_ROOT / "compiled_resources" / "TLX" / "triton" / "python" / "tutorials" / "gluon",
    ]


def _patch_tlx_gluon_tensor_descriptor() -> None:
    """Patch a TLX snapshot TensorDescriptor typo used by the Gluon tutorials."""
    from triton.experimental.gluon.nvidia import hopper

    tensor_descriptor = hopper.TensorDescriptor
    if not hasattr(tensor_descriptor, "round_f32_to_tf32"):
        tensor_descriptor.round_f32_to_tf32 = False


@lru_cache(maxsize=1)
def _load_gluon_warp_specialization():
    _patch_tlx_gluon_tensor_descriptor()

    for tutorial_dir in _gluon_tutorial_dirs():
        if not (tutorial_dir / "08-warp-specialization.py").exists():
            continue

        tutorial_dir_str = str(tutorial_dir)
        if tutorial_dir_str not in sys.path:
            sys.path.insert(0, tutorial_dir_str)
        return importlib.import_module("08-warp-specialization")

    searched = "\n".join(f"  - {path}" for path in _gluon_tutorial_dirs())
    raise FileNotFoundError(
        "Could not find Gluon tutorial 08-warp-specialization.py. Searched:\n"
        f"{searched}"
    )


@lru_cache(maxsize=1)
def _gluon_scheduler_impl():
    gluon_ws = _load_gluon_warp_specialization()
    return gluon_ws.t7.GroupedPersistentTileScheduler(8)


def _require_gluon_clc_api() -> None:
    if blackwell_clc is None:
        raise RuntimeError(
            "This installed Triton/Gluon build does not expose "
            "`triton.experimental.gluon.language.nvidia.blackwell.clc`. "
            "The inline CLC and multi-CTA Gluon providers need a newer upstream "
            "Gluon runtime than the older TLX snapshot."
        )


@aggregate
class InlineMMAv5:
    use_acc: gl.tensor
    acc_tmem: tensor_memory_descriptor
    barrier: gl.shared_memory_descriptor
    issued: gl.tensor
    reg_layout: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, use_acc, acc_tmem, barrier, issued, reg_layout):
        self.use_acc = use_acc
        self.acc_tmem = acc_tmem
        self.barrier = barrier
        self.issued = issued
        self.reg_layout = gl.constexpr(reg_layout)

    @gluon.jit
    def initialize(dtype: gl.constexpr, block_m: gl.constexpr, block_n: gl.constexpr, num_warps: gl.constexpr):
        layout: gl.constexpr = TensorMemoryLayout([block_m, block_n], col_stride=1)
        acc_tmem = allocate_tensor_memory(gl.float32, [block_m, block_n], layout)
        barrier = gl.allocate_shared_memory(gl.int64, [1], gluon_mbarrier.MBarrierLayout())
        gluon_mbarrier.init(barrier, count=1)
        reg_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (block_m, block_n), layout, num_warps)
        return InlineMMAv5(gl.to_tensor(False), acc_tmem, barrier, gl.to_tensor(0), reg_layout)

    @gluon.jit
    def issue_async_mma(self, a_smem, b_smem):
        tcgen05_mma(a_smem, b_smem, self.acc_tmem, use_acc=self.use_acc)
        tcgen05_commit(self.barrier)
        return InlineMMAv5(gl.to_tensor(True), self.acc_tmem, self.barrier, self.issued + 1, self.reg_layout)

    @gluon.jit
    def wait_num_outstanding(self, num_outstanding: gl.constexpr):
        gluon_mbarrier.wait(self.barrier, (self.issued - 1 - num_outstanding) & 1)
        return self

    @gluon.jit
    def take_result(self):
        next_mma = InlineMMAv5(gl.to_tensor(False), self.acc_tmem, self.barrier, self.issued, self.reg_layout)
        return self.acc_tmem.load(self.reg_layout), next_mma


@gluon.jit
def inline_issue_loads(producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs, num_buffers: gl.constexpr,
                       pred=True):
    index = producer % num_buffers
    producer += 1
    bar = bars.index(index)
    gluon_mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes, pred=pred)
    gluon_tma_async_load(a_desc, [off_m, k], bar, a_bufs.index(index), pred)
    gluon_tma_async_load(b_desc, [k, off_n], bar, b_bufs.index(index), pred)
    return producer


@gluon.jit
def inline_issue_mma(consumer, mma, bars, a_bufs, b_bufs, num_buffers: gl.constexpr):
    index = consumer % num_buffers
    phase = consumer // num_buffers & 1
    consumer += 1
    gluon_mbarrier.wait(bars.index(index), phase)
    mma = mma.wait_num_outstanding(0)
    mma = mma.issue_async_mma(a_bufs.index(index), b_bufs.index(index))
    return consumer, mma


@triton.jit
def inline_grouped_pid(tile_id, num_pid_in_group, num_pid_m, group_size_m: tl.constexpr):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * group_size_m
    actual_group_size_m = min(num_pid_m - first_pid_m, group_size_m)
    pid_m = first_pid_m + (tile_id % actual_group_size_m)
    pid_n = (tile_id % num_pid_in_group) // actual_group_size_m
    return pid_m, pid_n


@aggregate
class InlineClcTileScheduler:
    has_work: gl.tensor
    tile_id: gl.tensor
    clc_result_buf: gl.shared_memory_descriptor
    barrier: gl.shared_memory_descriptor
    phase: gl.tensor

    @gluon.constexpr_function
    def __init__(self, has_work, tile_id, clc_result_buf, barrier, phase):
        self.has_work = has_work
        self.tile_id = tile_id
        self.clc_result_buf = clc_result_buf
        self.barrier = barrier
        self.phase = phase

    @gluon.jit
    def initialize(M, N, block_m, block_n):
        has_work = gl.to_tensor(True)
        tile_id = gl.program_id(0)
        barrier = gluon_mbarrier.allocate_mbarrier()
        result_buf = gl.allocate_shared_memory(gl.int64, [2], gl.SwizzledSharedLayout(1, 1, 1, [0]))
        gluon_mbarrier.init(barrier, count=1)
        return InlineClcTileScheduler(has_work, tile_id, result_buf, barrier, gl.to_tensor(0))

    @gluon.jit
    def try_cancel(self):
        blackwell_clc.try_cancel(self.clc_result_buf, self.barrier)
        gluon_mbarrier.expect(self.barrier, 16)

    @gluon.jit
    def advance(self):
        gluon_mbarrier.wait(self.barrier, self.phase)
        result = blackwell_clc.load_result(self.clc_result_buf)
        has_work = result.is_canceled()
        next_tile = result.program_id(0)
        return InlineClcTileScheduler(has_work, next_tile, self.clc_result_buf, self.barrier, self.phase ^ 1)


@aggregate
class InlineStaticTileScheduler:
    has_work: gl.tensor
    tile_id: gl.tensor
    num_tiles: gl.tensor

    @gluon.constexpr_function
    def __init__(self, has_work, tile_id, num_tiles):
        self.has_work = has_work
        self.tile_id = tile_id
        self.num_tiles = num_tiles

    @gluon.jit
    def initialize(M, N, block_m, block_n):
        tile_id = gl.program_id(0)
        num_tiles = gl.cdiv(M, block_m) * gl.cdiv(N, block_n)
        return InlineStaticTileScheduler(tile_id < num_tiles, tile_id, num_tiles)

    @gluon.jit
    def try_cancel(self):
        pass

    @gluon.jit
    def advance(self):
        next_tile = self.tile_id + gl.num_programs(0)
        return InlineStaticTileScheduler(next_tile < self.num_tiles, next_tile, self.num_tiles)


@gluon.jit
def inline_clc_persistent_matmul_kernel(a_desc, b_desc, c_desc, SchedulerImpl: gl.constexpr,
                                        num_buffers: gl.constexpr, num_warps: gl.constexpr):
    block_m: gl.constexpr = c_desc.block_type.shape[0]
    block_n: gl.constexpr = c_desc.block_type.shape[1]
    block_k: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]
    M = c_desc.shape[0]
    N = c_desc.shape[1]

    group_size_m: gl.constexpr = 8
    num_pid_m = gl.cdiv(M, block_m)
    num_pid_n = gl.cdiv(N, block_n)
    num_pid_in_group = group_size_m * num_pid_n

    bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], gluon_mbarrier.MBarrierLayout())
    for i in gl.static_range(num_buffers):
        gluon_mbarrier.init(bars.index(i), count=1)

    producer = 0
    consumer = 0
    mma = InlineMMAv5.initialize(dtype, block_m, block_n, num_warps)
    scheduler = SchedulerImpl.initialize(M, N, block_m, block_n)

    while scheduler.has_work:
        pid_m, pid_n = inline_grouped_pid(scheduler.tile_id, num_pid_in_group, num_pid_m, group_size_m)
        off_m = pid_m * block_m
        off_n = pid_n * block_n

        scheduler.try_cancel()

        a_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout)
        b_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + b_desc.block_type.shape, b_desc.layout)
        for k in gl.static_range(0, block_k * (num_buffers - 2), block_k):
            producer = inline_issue_loads(producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs,
                                          num_buffers)

        for k in range(block_k * (num_buffers - 2), K, block_k):
            producer = inline_issue_loads(producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs,
                                          num_buffers)
            consumer, mma = inline_issue_mma(consumer, mma, bars, a_bufs, b_bufs, num_buffers)

        for _ in gl.static_range(num_buffers - 2):
            consumer, mma = inline_issue_mma(consumer, mma, bars, a_bufs, b_bufs, num_buffers)

        mma = mma.wait_num_outstanding(0)
        c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
        c, mma = mma.take_result()
        c_smem.store(c.to(dtype))
        gluon_fence_async_shared()
        gluon_tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
        gluon_tma.store_wait(pendings=0)
        scheduler = scheduler.advance()


def gluon_blackwell_clc_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor | None = None,
    *,
    use_clc: bool = True,
) -> torch.Tensor:
    _patch_tlx_gluon_tensor_descriptor()
    if use_clc:
        _require_gluon_clc_api()
    if c is None:
        c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)

    block_m = 128
    block_n = 256
    block_k = 64
    num_buffers = 3
    num_warps = 4

    a_layout = gl.NVMMASharedLayout.get_default_for([block_m, block_k], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([block_k, block_n], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([block_m, block_n], gl.float16)
    a_desc = GluonTensorDescriptor.from_tensor(a, [block_m, block_k], a_layout)
    b_desc = GluonTensorDescriptor.from_tensor(b, [block_k, block_n], b_layout)
    c_desc = GluonTensorDescriptor.from_tensor(c, [block_m, block_n], c_layout)

    if use_clc:
        grid = triton.cdiv(c.shape[0], block_m) * triton.cdiv(c.shape[1], block_n)
        scheduler = InlineClcTileScheduler
    else:
        grid = torch.cuda.get_device_properties(a.device).multi_processor_count
        scheduler = InlineStaticTileScheduler

    inline_clc_persistent_matmul_kernel[(grid, )](
        a_desc,
        b_desc,
        c_desc,
        scheduler,
        num_buffers,
        num_warps=num_warps,
    )
    return c


def gluon_blackwell_clc_static_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor | None = None,
) -> torch.Tensor:
    return gluon_blackwell_clc_matmul(a, b, c, use_clc=False)


@aggregate
class InlineCounter:
    index: gl.tensor
    phase: gl.tensor
    count: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, index, phase, count):
        self.index = index
        self.phase = phase
        self.count = gl.constexpr(count)

    @gluon.jit
    def create(phase, count: gl.constexpr):
        return InlineCounter(gl.to_tensor(0), gl.to_tensor(phase), count)

    @gluon.must_use_result
    @gluon.jit
    def next(self, pred=True):
        raw = self.index + gl.where(pred, 1, 0)
        rollover = raw == self.count
        return InlineCounter(gl.where(rollover, 0, raw), gl.where(rollover, self.phase ^ 1, self.phase), self.count)


@gluon.constexpr_function
def inline_split_dim(cga_layout, dim):
    return 1 << sum(b[dim] != 0 for b in cga_layout)


def inline_operand_cga_layout(cga_layout, op_idx, two_ctas):
    assert op_idx in (0, 1)
    if not cga_layout:
        return cga_layout

    def broadcast(base):
        mul = 2 if two_ctas else 1
        return (base[0], 0) if op_idx == 0 else (0, mul * base[1])

    if not two_ctas:
        return tuple(map(broadcast, cga_layout))

    assert cga_layout[0] == (1, 0)
    first = (1, 0) if op_idx == 0 else (0, 1)
    return (first, *map(broadcast, cga_layout[1:]))


@gluon.jit
def inline_planar_snake(tile_id, m_tiles, n_tiles, minor_dim: gl.constexpr, tile_width: gl.constexpr):
    major_size = n_tiles if minor_dim == 0 else m_tiles
    minor_size = m_tiles if minor_dim == 0 else n_tiles

    full_minor_tiles = minor_size // tile_width
    full_minor_size = full_minor_tiles * tile_width
    full_elements = full_minor_tiles * tile_width * major_size
    minor_tile_id = tile_id // (tile_width * major_size)

    full_minor_in_tile = tile_id % tile_width
    full_major_in_tile = (tile_id // tile_width) % major_size
    full_minor = minor_tile_id * tile_width + full_minor_in_tile
    full_major = gl.where((minor_tile_id % 2) == 0, full_major_in_tile, major_size - 1 - full_major_in_tile)

    partial_width = minor_size - full_minor_size
    partial_width = gl.where(partial_width > 0, partial_width, 1)
    partial_id = tile_id - full_elements
    partial_minor_in_tile = partial_id % partial_width
    partial_major_in_tile = (partial_id // partial_width) % major_size
    partial_minor = minor_tile_id * tile_width + partial_minor_in_tile
    partial_major = gl.where((minor_tile_id % 2) == 0, partial_major_in_tile,
                             major_size - 1 - partial_major_in_tile)

    in_full = tile_id < full_elements
    minor = gl.where(in_full, full_minor, partial_minor)
    major = gl.where(in_full, full_major, partial_major)
    if minor_dim == 0:
        return minor, major
    return major, minor


@aggregate
class InlineMultictaScheduler:
    has_work: gl.tensor
    tile_id: gl.tensor
    pid_m: gl.tensor
    pid_n: gl.tensor
    num_pid_m: gl.tensor
    num_pid_n: gl.tensor
    tile_m: gl.constexpr
    tile_n: gl.constexpr
    minor_dim: gl.constexpr
    tile_width: gl.constexpr
    clc_result_buffers: gl.shared_memory_descriptor
    clc_barriers: gl.shared_memory_descriptor
    planar_buffers: gl.shared_memory_descriptor
    planar_ready_bars: gl.shared_memory_descriptor
    consumed_bars: gl.shared_memory_descriptor
    counter: InlineCounter
    consumed_counter: InlineCounter

    @gluon.constexpr_function
    def __init__(self, has_work, tile_id, pid_m, pid_n, num_pid_m, num_pid_n, tile_m, tile_n, minor_dim,
                 tile_width, clc_result_buffers, clc_barriers, planar_buffers, planar_ready_bars, consumed_bars,
                 counter, consumed_counter):
        self.has_work = has_work
        self.tile_id = tile_id
        self.pid_m = pid_m
        self.pid_n = pid_n
        self.num_pid_m = num_pid_m
        self.num_pid_n = num_pid_n
        self.tile_m = gl.constexpr(tile_m)
        self.tile_n = gl.constexpr(tile_n)
        self.minor_dim = gl.constexpr(minor_dim)
        self.tile_width = gl.constexpr(tile_width)
        self.clc_result_buffers = clc_result_buffers
        self.clc_barriers = clc_barriers
        self.planar_buffers = planar_buffers
        self.planar_ready_bars = planar_ready_bars
        self.consumed_bars = consumed_bars
        self.counter = counter
        self.consumed_counter = consumed_counter

    @gluon.jit
    def initialize(M, N, tile_m: gl.constexpr, tile_n: gl.constexpr, minor_dim: gl.constexpr,
                   tile_width: gl.constexpr, clc_result_buffers, clc_barriers, planar_buffers, planar_ready_bars,
                   consumed_bars):
        tile_id = gl.program_id(axis=0)
        num_pid_m = gl.cdiv(M, tile_m)
        num_pid_n = gl.cdiv(N, tile_n)
        pid_m, pid_n = inline_planar_snake(tile_id, num_pid_m, num_pid_n, minor_dim, tile_width)
        return InlineMultictaScheduler(
            gl.to_tensor(True),
            tile_id,
            pid_m,
            pid_n,
            num_pid_m,
            num_pid_n,
            tile_m,
            tile_n,
            minor_dim,
            tile_width,
            clc_result_buffers,
            clc_barriers,
            planar_buffers,
            planar_ready_bars,
            consumed_bars,
            InlineCounter.create(0, clc_barriers.shape[0]),
            InlineCounter.create(0, clc_barriers.shape[0]),
        )

    @gluon.jit
    def offsets(self):
        return self.pid_m * self.tile_m, self.pid_n * self.tile_n

    @gluon.jit
    def step(self, iteration):
        consumed = self.consumed_counter
        if iteration > 0:
            gluon_mbarrier.arrive(self.consumed_bars.index(consumed.index))
            consumed = consumed.next()

        counter = self.counter
        barrier = self.clc_barriers.index(counter.index)
        result_buf = self.clc_result_buffers.index(counter.index)
        gluon_mbarrier.wait(barrier, counter.phase)
        result = blackwell_clc.load_result(result_buf)
        gluon_mbarrier.wait(self.planar_ready_bars.index(counter.index), counter.phase)

        planar_layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0],
                                                       [[0]] * (gl.num_ctas().bit_length() - 1))
        packed = self.planar_buffers.index(counter.index).load(planar_layout).reshape([])
        pid_m = ((packed >> 32) & 0xFFFFFFFF).to(gl.int32)
        pid_n = (packed & 0xFFFFFFFF).to(gl.int32)
        has_work = result.is_canceled()
        tile_id = self.tile_id
        if has_work:
            tile_id = result.program_id(0)

        return InlineMultictaScheduler(
            has_work,
            tile_id,
            pid_m,
            pid_n,
            self.num_pid_m,
            self.num_pid_n,
            self.tile_m,
            self.tile_n,
            self.minor_dim,
            self.tile_width,
            self.clc_result_buffers,
            self.clc_barriers,
            self.planar_buffers,
            self.planar_ready_bars,
            self.consumed_bars,
            counter.next(),
            consumed,
        )


@aggregate
class InlineMultictaArgs:
    a_desc: gluon_tma.tensor_descriptor
    b_desc: gluon_tma.tensor_descriptor
    c_desc: gluon_tma.tensor_descriptor
    a_bufs: gl.shared_memory_descriptor
    b_bufs: gl.shared_memory_descriptor
    load_empty_bars: gl.shared_memory_descriptor
    load_ready_bars: gl.shared_memory_descriptor
    acc_bufs: tensor_memory_descriptor
    acc_empty_bars: gl.shared_memory_descriptor
    acc_ready_bars: gl.shared_memory_descriptor
    clc_result_buffers: gl.shared_memory_descriptor
    clc_barriers: gl.shared_memory_descriptor
    planar_buffers: gl.shared_memory_descriptor
    planar_ready_bars: gl.shared_memory_descriptor
    consumed_bars: gl.shared_memory_descriptor
    minor_dim: gl.constexpr
    tile_width: gl.constexpr
    subtile_stages: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, a_desc, b_desc, c_desc, a_bufs, b_bufs, load_empty_bars, load_ready_bars, acc_bufs,
                 acc_empty_bars, acc_ready_bars, clc_result_buffers, clc_barriers, planar_buffers,
                 planar_ready_bars, consumed_bars, minor_dim, tile_width, subtile_stages):
        self.a_desc = a_desc
        self.b_desc = b_desc
        self.c_desc = c_desc
        self.a_bufs = a_bufs
        self.b_bufs = b_bufs
        self.load_empty_bars = load_empty_bars
        self.load_ready_bars = load_ready_bars
        self.acc_bufs = acc_bufs
        self.acc_empty_bars = acc_empty_bars
        self.acc_ready_bars = acc_ready_bars
        self.clc_result_buffers = clc_result_buffers
        self.clc_barriers = clc_barriers
        self.planar_buffers = planar_buffers
        self.planar_ready_bars = planar_ready_bars
        self.consumed_bars = consumed_bars
        self.minor_dim = gl.constexpr(minor_dim)
        self.tile_width = gl.constexpr(tile_width)
        self.subtile_stages = gl.constexpr(subtile_stages)

    @gluon.jit
    def scheduler(self):
        return InlineMultictaScheduler.initialize(
            self.c_desc.shape[0],
            self.c_desc.shape[1],
            self.a_desc.block_shape[0],
            self.b_desc.block_shape[1],
            self.minor_dim,
            self.tile_width,
            self.clc_result_buffers,
            self.clc_barriers,
            self.planar_buffers,
            self.planar_ready_bars,
            self.consumed_bars,
        )


@gluon.jit
def inline_multicta_clc_partition(args):
    tile_m: gl.constexpr = args.a_desc.block_shape[0]
    tile_n: gl.constexpr = args.b_desc.block_shape[1]
    has_work = gl.to_tensor(True)
    num_pid_m = gl.cdiv(args.c_desc.shape[0], tile_m)
    num_pid_n = gl.cdiv(args.c_desc.shape[1], tile_n)
    state = InlineCounter.create(0, args.clc_barriers.shape[0])
    consumed = InlineCounter.create(1, args.clc_barriers.shape[0])
    stages: gl.constexpr = args.clc_barriers.shape[0]
    i = 0
    while has_work:
        gluon_mbarrier.wait(args.consumed_bars.index(consumed.index), consumed.phase, pred=(i >= stages))
        barrier = args.clc_barriers.index(state.index)
        result_buf = args.clc_result_buffers.index(state.index)
        gluon_mbarrier.expect(barrier, 16)
        blackwell_clc.try_cancel(result_buf, barrier)
        gluon_mbarrier.wait(barrier, state.phase)
        result = blackwell_clc.load_result(result_buf)
        has_work = result.is_canceled()
        pid_m = gl.to_tensor(0)
        pid_n = gl.to_tensor(0)
        if has_work:
            tile_id = result.program_id(0)
            pid_m, pid_n = inline_planar_snake(tile_id, num_pid_m, num_pid_n, args.minor_dim, args.tile_width)
        packed = (pid_m.to(gl.int64) << 32) | (pid_n.to(gl.int64) & 0xFFFFFFFF)
        planar_layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0],
                                                       [[0]] * (gl.num_ctas().bit_length() - 1))
        args.planar_buffers.index(state.index).store(gl.full([1], packed, gl.int64, layout=planar_layout))
        gluon_mbarrier.arrive(args.planar_ready_bars.index(state.index))
        state = state.next()
        consumed = consumed.next()
        i += 1


@gluon.jit
def inline_multicta_load_partition(args):
    block_k: gl.constexpr = args.a_desc.block_shape[1]
    K = args.a_desc.shape[1]
    state = InlineCounter.create(1, args.load_ready_bars.shape[0])
    scheduler = args.scheduler()

    i = 0
    while scheduler.has_work:
        off_m, off_n = scheduler.offsets()
        for k in range(0, K, block_k):
            pred = (i > 0) or (k >= block_k * args.load_ready_bars.shape[0])
            gluon_mbarrier.wait(args.load_empty_bars.index(state.index), state.phase, pred=pred)
            barrier = args.load_ready_bars.index(state.index)
            gluon_mbarrier.expect(barrier, args.a_desc.nbytes_per_cta + args.b_desc.nbytes_per_cta)
            gluon_tma_async_load(args.a_desc, [off_m, k], barrier, args.a_bufs.index(state.index), multicast=True)
            gluon_tma_async_load(args.b_desc, [k, off_n], barrier, args.b_bufs.index(state.index), multicast=True)
            state = state.next()
        scheduler = scheduler.step(i)
        i += 1


@gluon.jit
def inline_multicta_mma_partition(args):
    block_k: gl.constexpr = args.a_desc.block_shape[1]
    K = args.a_desc.shape[1]
    load_state = InlineCounter.create(0, args.load_empty_bars.shape[0])
    acc_state = InlineCounter.create(1, args.acc_empty_bars.shape[0])
    scheduler = args.scheduler()

    i = 0
    while scheduler.has_work:
        acc = args.acc_bufs.index(acc_state.index)
        gluon_mbarrier.wait(args.acc_empty_bars.index(acc_state.index), acc_state.phase,
                            pred=(i >= args.acc_empty_bars.shape[0]))
        use_acc = False
        for _ in range(0, K, block_k):
            gluon_mbarrier.wait(args.load_ready_bars.index(load_state.index), load_state.phase)
            tcgen05_mma(
                args.a_bufs.index(load_state.index),
                args.b_bufs.index(load_state.index),
                acc,
                use_acc=use_acc,
                multicast=True,
                mbarriers=[args.load_empty_bars.index(load_state.index)],
            )
            load_state = load_state.next()
            use_acc = True
        tcgen05_commit(args.acc_ready_bars.index(acc_state.index), descs=[args.a_bufs.index(0), args.b_bufs.index(0)])
        acc_state = acc_state.next()
        scheduler = scheduler.step(i)
        i += 1


@gluon.jit
def inline_multicta_epilogue_partition(args):
    tile_m: gl.constexpr = args.a_desc.block_shape[0]
    tile_n: gl.constexpr = args.b_desc.block_shape[1]
    split_n: gl.constexpr = args.c_desc.block_shape[1]
    subtile_factor: gl.constexpr = tile_n // split_n
    dtype: gl.constexpr = args.c_desc.dtype

    acc_state = InlineCounter.create(0, args.acc_empty_bars.shape[0])
    smem_outs = gl.allocate_shared_memory(dtype, [args.subtile_stages, tile_m, split_n], args.c_desc.layout)
    subtile_state = InlineCounter.create(0, args.subtile_stages)
    scheduler = args.scheduler()

    i = 0
    while scheduler.has_work:
        off_m, off_n = scheduler.offsets()
        gluon_mbarrier.wait(args.acc_ready_bars.index(acc_state.index), acc_state.phase)
        acc = args.acc_bufs.index(acc_state.index)
        for s in gl.static_range(subtile_factor):
            acc_sub = acc.slice(split_n * s, split_n)
            smem_out = smem_outs.index(subtile_state.index)
            gluon_tma.store_wait(pendings=args.subtile_stages - 1)
            smem_out.store(acc_sub.load().to(dtype))
            gluon_tma.async_copy_shared_to_global(args.c_desc, [off_m, off_n + split_n * s], smem_out)
            subtile_state = subtile_state.next()
        gluon_mbarrier.arrive(args.acc_empty_bars.index(acc_state.index))
        acc_state = acc_state.next()
        scheduler = scheduler.step(i)
        i += 1


@gluon.jit
def inline_multicta_matmul_kernel(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    block_size_m: gl.constexpr,
    block_size_n: gl.constexpr,
    block_size_k: gl.constexpr,
    grid_minor_dim: gl.constexpr,
    grid_tile_width: gl.constexpr,
    stages: gl.constexpr,
    acc_stages: gl.constexpr,
    cga_layout: gl.constexpr,
    epilogue_size_n: gl.constexpr,
    subtile_stages: gl.constexpr,
):
    block_m: gl.constexpr = a_desc.block_shape[0]
    block_n: gl.constexpr = b_desc.block_shape[1]
    two_ctas: gl.constexpr = gl.num_ctas() > 1
    dtype: gl.constexpr = a_desc.dtype

    a_bufs = gl.allocate_shared_memory(dtype, [stages] + a_desc.block_shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(dtype, [stages] + b_desc.block_shape, b_desc.layout)
    tmem_layout: gl.constexpr = TensorMemoryLayout(
        [block_size_m, block_n // inline_split_dim(cga_layout, 1)],
        col_stride=1,
        cga_layout=cga_layout,
        two_ctas=two_ctas,
    )
    acc_bufs = allocate_tensor_memory(gl.float32, [acc_stages, block_m, block_n], tmem_layout)
    mma_count: gl.constexpr = tcgen05_mma_barrier_count([a_bufs.index(0), b_bufs.index(0)], multicast=True,
                                                        two_ctas=acc_bufs.index(0).type.layout.two_ctas)

    load_empty_bars = gluon_mbarrier.allocate_mbarrier(batch=stages)
    load_ready_bars = gluon_mbarrier.allocate_mbarrier(batch=stages, two_ctas=two_ctas)
    for i in gl.static_range(stages):
        gluon_mbarrier.init(load_empty_bars.index(i), count=mma_count)
        gluon_mbarrier.init(load_ready_bars.index(i), count=1)

    acc_empty_bars = gluon_mbarrier.allocate_mbarrier(batch=acc_stages, two_ctas=two_ctas)
    acc_ready_bars = gluon_mbarrier.allocate_mbarrier(batch=acc_stages)
    for i in gl.static_range(acc_stages):
        gluon_mbarrier.init(acc_empty_bars.index(i), count=1)
        gluon_mbarrier.init(acc_ready_bars.index(i), count=mma_count)

    clc_barriers = gluon_mbarrier.allocate_mbarrier(batch=acc_stages)
    planar_ready_bars = gluon_mbarrier.allocate_mbarrier(batch=acc_stages)
    consumed_bars = gluon_mbarrier.allocate_mbarrier(batch=acc_stages, two_ctas=two_ctas)
    for i in gl.static_range(acc_stages):
        gluon_mbarrier.init(clc_barriers.index(i), count=1)
        gluon_mbarrier.init(planar_ready_bars.index(i), count=1)
        gluon_mbarrier.init(consumed_bars.index(i), count=3)

    clc_cga_layout: gl.constexpr = [[0]] * (gl.num_ctas().bit_length() - 1)
    clc_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0], cga_layout=clc_cga_layout)
    clc_result_buffers = gl.allocate_shared_memory(gl.int64, [clc_barriers.shape[0], 2], clc_layout)
    planar_buffers = gl.allocate_shared_memory(gl.int64, [clc_barriers.shape[0], 1], clc_layout)

    args = InlineMultictaArgs(
        a_desc,
        b_desc,
        c_desc,
        a_bufs,
        b_bufs,
        load_empty_bars,
        load_ready_bars,
        acc_bufs,
        acc_empty_bars,
        acc_ready_bars,
        clc_result_buffers,
        clc_barriers,
        planar_buffers,
        planar_ready_bars,
        consumed_bars,
        grid_minor_dim,
        grid_tile_width,
        subtile_stages,
    )

    gl.warp_specialize([
        (inline_multicta_epilogue_partition, (args, )),
        (inline_multicta_load_partition, (args, )),
        (inline_multicta_mma_partition, (args, )),
        (inline_multicta_clc_partition, (args, )),
    ], [1, 1, 1], [24, 24, 24])


def gluon_blackwell_multicta_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor | None = None,
) -> torch.Tensor:
    _patch_tlx_gluon_tensor_descriptor()
    _require_gluon_clc_api()
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("inline Gluon multi-CTA provider only supports fp16 inputs")
    if c is None:
        c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)

    block_m = 128
    block_n = 256
    block_k = 64
    cga_layout = ((1, 0), )
    epilogue_size_n = 32
    tile_m = block_m * inline_split_dim(cga_layout, 0)
    two_ctas = bool(cga_layout)

    a_layout = gl.NVMMASharedLayout.get_default_for([tile_m, block_k], gl.float16,
                                                    cga_layout=inline_operand_cga_layout(cga_layout, 0, two_ctas))
    b_layout = gl.NVMMASharedLayout.get_default_for([block_k, block_n], gl.float16,
                                                    cga_layout=inline_operand_cga_layout(cga_layout, 1, two_ctas))
    c_layout = gl.NVMMASharedLayout.get_default_for([tile_m, epilogue_size_n], gl.float16, cga_layout=cga_layout)

    a_desc = GluonTensorDescriptor.from_tensor(a, [tile_m, block_k], a_layout)
    b_desc = GluonTensorDescriptor.from_tensor(b, [block_k, block_n], b_layout)
    c_desc = GluonTensorDescriptor.from_tensor(c, [tile_m, epilogue_size_n], c_layout)

    grid = (triton.cdiv(a.shape[0], tile_m) * triton.cdiv(b.shape[1], block_n), )

    inline_multicta_matmul_kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        a.shape[0],
        b.shape[1],
        a.shape[1],
        block_m,
        block_n,
        block_k,
        0,
        16,
        6,
        2,
        cga_layout,
        epilogue_size_n,
        4,
        num_warps=4,
        num_ctas=2**len(cga_layout),
    )
    return c


def gluon_blackwell_ws_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor | None = None,
) -> torch.Tensor:
    """Call the Gluon Blackwell warp-specialized GEMM with tutorial defaults."""
    gluon_ws = _load_gluon_warp_specialization()
    if c is None:
        c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)

    gluon_ws.matmul_warp_specialized(
        a,
        b,
        c,
        BLOCK_M=128,
        BLOCK_N=256,
        BLOCK_K=64,
        num_buffers=4,
        SUBTILE_FACTOR=4,
        num_warps=4,
        SchedulerImpl=_gluon_scheduler_impl(),
    )
    return c


MATMUL_METHODS = {
    "torch": torch.matmul,
    "tlx_ws": tlx_blackwell_ws_matmul,
    "tlx_clc": tlx_blackwell_clc_matmul,
    "tlx_2cta": tlx_blackwell_2cta_matmul,
    "gluon_ws": gluon_blackwell_ws_matmul,
    "gluon_clc_static": gluon_blackwell_clc_static_matmul,
    "gluon_clc": gluon_blackwell_clc_matmul,
    "gluon_multicta": gluon_blackwell_multicta_matmul,
}


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype for this benchmark: {name}")


def _tflops(ms: float, M: int, N: int, K: int) -> float:
    return 2 * M * N * K * 1e-12 / (ms * 1e-3)


def _make_provider_callable(provider: str, a: torch.Tensor, b: torch.Tensor):
    if provider.startswith("gluon_"):
        c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
        if provider == "gluon_ws":
            _load_gluon_warp_specialization()
            _gluon_scheduler_impl()
        return lambda: MATMUL_METHODS[provider](a, b, c)

    return lambda: MATMUL_METHODS[provider](a, b)


def verify_correctness(dtype: torch.dtype, providers: list[str]) -> None:
    torch.manual_seed(0)
    M = N = K = 512
    a = torch.randn((M, K), device=DEVICE, dtype=dtype)
    b = torch.randn((K, N), device=DEVICE, dtype=dtype)
    reference = torch.matmul(a, b)

    for name in providers:
        if name == "torch":
            continue
        actual = MATMUL_METHODS[name](a, b)
        torch.testing.assert_close(actual, reference, rtol=1e-2, atol=1e-1)

    checked = ", ".join(provider for provider in providers if provider != "torch")
    print(f"Correctness OK on 512x512x512 fp16 GEMM: {checked}.")


def create_benchmark(
    sizes: list[int],
    dtype: torch.dtype,
    warmup: int,
    rep: int,
    providers: list[str],
):
    line_names = {
        "torch": "Torch / cuBLAS",
        "tlx_ws": "TLX WS",
        "tlx_clc": "TLX CLC",
        "tlx_2cta": "TLX 2CTA",
        "gluon_ws": "Gluon WS",
        "gluon_clc_static": "Gluon static CLC kernel",
        "gluon_clc": "Gluon CLC",
        "gluon_multicta": "Gluon multiCTA",
    }
    dtype_name = {torch.float16: "fp16"}[dtype]

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],
            x_vals=sizes,
            line_arg="provider",
            line_vals=providers,
            line_names=[line_names[provider] for provider in providers],
            ylabel="TFLOPS",
            plot_name=f"blackwell-gemm-tlx-vs-gluon-{dtype_name}",
            args={},
        )
    )
    def benchmark(M, N, K, provider):
        torch.manual_seed(0)
        a = torch.randn((M, K), device=DEVICE, dtype=dtype)
        b = torch.randn((K, N), device=DEVICE, dtype=dtype)
        quantiles = [0.5, 0.2, 0.8]
        provider_fn = _make_provider_callable(provider, a, b)

        ms, min_ms, max_ms = triton.testing.do_bench(
            provider_fn,
            quantiles=quantiles,
            warmup=warmup,
            rep=rep,
        )

        return (
            _tflops(ms, M, N, K),
            _tflops(max_ms, M, N, K),
            _tflops(min_ms, M, N, K),
        )

    return benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TLX vs Gluon Blackwell GEMM")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
        help="Square GEMM sizes to benchmark. Each value means M=N=K=size.",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=DEFAULT_PROVIDERS,
        choices=list(MATMUL_METHODS),
        help="Providers to include in the benchmark plot.",
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=["fp16"],
        help="Input dtype. The current Gluon tutorial wrapper is fp16-only.",
    )
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--rep", type=int, default=500)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--save-path", default="plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = _dtype_from_name(args.dtype)

    test_variable = 42

    if not is_blackwell():
        raise RuntimeError("This benchmark requires a Blackwell CUDA GPU.")

    # The TLX tutorial wrapper checks this at call time.
    os.environ.setdefault("TLX_GEMM_USE_HEURISTIC", "1")

    if not args.skip_correctness:
        verify_correctness(dtype, args.providers)

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    benchmark = create_benchmark(
        sizes=args.sizes,
        dtype=dtype,
        warmup=args.warmup,
        rep=args.rep,
        providers=args.providers,
    )
    benchmark.run(print_data=True, show_plots=False, save_path=save_path)


if __name__ == "__main__":
    main()
