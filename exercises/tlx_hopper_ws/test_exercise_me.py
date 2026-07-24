"""Correctness regression for the non-WS skinny TMA GEMM exercise.

Run on Hopper or newer with:

    source .venv/bin/activate && python -m pytest \
        exercises/tlx_hopper_ws/test_exercise_me.py -q

The shape deliberately gives each split-K program enough K iterations to wrap
the circular shared-memory pipeline several times.  That makes the test cover
the delayed ``async_dot_wait(1)`` reuse path instead of only the initial
preloaded stages.
"""

import pytest


torch = pytest.importorskip("torch")


def _require_hopper_or_newer() -> None:
    if not torch.cuda.is_available():
        pytest.skip("The TLX TMA correctness test requires a CUDA GPU.")

    major, minor = torch.cuda.get_device_capability()
    if major < 9:
        pytest.skip(
            "The TLX TMA correctness test requires Hopper or newer "
            f"(found compute capability {major}.{minor})."
        )


def test_skinny_matmul_tma_delayed_mma_wait_reuses_pipeline_buffers() -> None:
    _require_hopper_or_newer()

    # Import only after the hardware check: exercise_me initializes Triton's
    # active device at module-import time.
    from exercises.tlx_hopper_ws.exercise_me import _skinny_matmul_tma

    torch.manual_seed(0)
    m, n, k = 128, 256, 4096
    a = torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.5
    b = torch.randn((k, n), device="cuda", dtype=torch.float16) * 0.5

    actual = _skinny_matmul_tma(a, b, m, n, k)
    expected = torch.matmul(a, b)

    assert isinstance(actual, torch.Tensor), (
        "_skinny_matmul_tma must return its output tensor; got "
        f"{type(actual).__name__} instead"
    )
    assert actual.shape == (m, n)
    assert actual.dtype == torch.float16
    assert actual.device == a.device

    # Split-K rounds each partial output to fp16 before the wrapper reduction,
    # so use a tolerance that admits expected rounding but still catches stale
    # or prematurely overwritten pipeline buffers.
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2.5e-1)

