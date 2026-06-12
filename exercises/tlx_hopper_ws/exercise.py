# type: ignore
"""
Exercise: Hopper GEMM Warp-Specialized State Machine
Module: TLX Hopper WS

This is intentionally a worksheet, not a completed kernel.

Fill the TODO functions by reading:
compiled_resources/TLX/triton/third_party/tlx/tutorials/hopper_gemm_ws.py

Keep this file focused on the fixed first-pass configuration:
- NUM_CTAS = 1
- USE_WARP_BARRIER = False
- EPILOGUE_SUBTILE = False
- NUM_MMA_GROUPS = 2
- NUM_STAGES = 3
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


REFERENCE = (
    "compiled_resources/TLX/triton/third_party/tlx/tutorials/"
    "hopper_gemm_ws.py"
)


FIXED_CONFIG = {
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


@dataclass(frozen=True)
class TaskRegion:
    name: str
    role: str
    warp_count: int | str
    replica_count: int | str
    waits_on: tuple[str, ...]
    writes_or_reads: tuple[str, ...]
    signals: tuple[str, ...]


@dataclass(frozen=True)
class BarrierSet:
    name: str
    count_expression: str
    owner_that_waits: str
    owner_that_arrives_or_signals: str
    meaning: str


@dataclass(frozen=True)
class PipelineStep:
    k_iter: int
    smem_accum_count: int
    buffer_index: int
    phase: int
    producer_summary: str
    consumer_summary: str


def todo(name: str):
    raise NotImplementedError(
        f"Fill {name} by reading {REFERENCE}. Keep the answer short and "
        "specific to the fixed first-pass configuration."
    )


def compute_bufidx_phase(smem_accum_count: int, num_stages: int) -> tuple[int, int]:
    """TODO 1: Implement the Python equivalent of get_bufidx_phase."""
    todo("compute_bufidx_phase")


def describe_task_regions() -> tuple[TaskRegion, ...]:
    """
    TODO 2: Describe the two task regions in matmul_kernel_tlx_ws.

    Include only the fixed first-pass case:
    - one default producer task
    - one replicated consumer task with replica_count=2
    """
    todo("describe_task_regions")


def describe_barrier_sets() -> tuple[BarrierSet, ...]:
    """
    TODO 3: Describe bars_empty_a, bars_empty_b, bars_full_a, bars_full_b.

    Do not include cta_bars yet; NUM_CTAS is fixed to 1 for this exercise.
    """
    todo("describe_barrier_sets")


def trace_single_output_tile(num_k_iters: int = 5) -> tuple[PipelineStep, ...]:
    """
    TODO 4: Produce a compact table for one output tile.

    For each K iteration, show:
    - smem_accum_count
    - buffer index
    - phase
    - what the producer does
    - what each consumer replica does
    """
    todo("trace_single_output_tile")


def explain_a_split_and_b_sharing() -> str:
    """
    TODO 5: Explain why A has NUM_STAGES * NUM_MMA_GROUPS slots but B has
    NUM_STAGES slots.
    """
    todo("explain_a_split_and_b_sharing")


def explain_persistent_grid() -> str:
    """
    TODO 6: Explain why WS launches persistent SM workers instead of one CTA
    per output tile.
    """
    todo("explain_persistent_grid")


def explain_release_order() -> str:
    """
    TODO 7: Explain when an A slot and a B slot become reusable.

    Be precise: A is owned by one consumer replica; B is shared by both
    consumer replicas.
    """
    todo("explain_release_order")


def _run_todo(name: str, fn: Callable[[], object]) -> bool:
    try:
        value = fn()
    except NotImplementedError as exc:
        print(f"[TODO] {name}: {exc}")
        return False

    print(f"[OK] {name}")
    print(value)
    return True


def main():
    print("TLX Hopper WS worksheet")
    print(f"Reference: {REFERENCE}")
    print(f"Fixed config: {FIXED_CONFIG}")
    print()

    checks = [
        ("compute_bufidx_phase", lambda: [compute_bufidx_phase(i, 3) for i in range(8)]),
        ("describe_task_regions", describe_task_regions),
        ("describe_barrier_sets", describe_barrier_sets),
        ("trace_single_output_tile", trace_single_output_tile),
        ("explain_a_split_and_b_sharing", explain_a_split_and_b_sharing),
        ("explain_persistent_grid", explain_persistent_grid),
        ("explain_release_order", explain_release_order),
    ]

    completed = sum(_run_todo(name, fn) for name, fn in checks)
    print()
    print(f"Completed {completed}/{len(checks)} worksheet sections.")
    if completed != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
