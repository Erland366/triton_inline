# Gluon H100 Smoke Test

This exercise verifies that the project's Modal environment can import, compile,
launch, and validate a Gluon kernel before the Gluon learning track begins.
It is deliberately separate from `exercises/gluon_memory_layout/`, whose TODOs
are student work.

The smoke kernel performs two-dimensional elementwise addition with:

- an explicit two-dimensional `gl.BlockedLayout`
- layout-carrying row and column indices from `gl.arange` and `gl.SliceLayout`
- masked global-memory loads and stores
- non-power-of-two matrix shapes that exercise both boundary masks

## Run on Modal H100

```bash
source .venv/bin/activate
modal run run_modal.py \
  --action pytest \
  --script "exercises/gluon-smoke/test_smoke.py -q"
```

The expected result is four passing tests. Three cover boundary shapes; the
fourth verifies that invalid launch parameters fail with an actionable error
before compilation.

## Validated Baseline

Validated on 2026-08-01 with Modal H100 and Triton 3.7.1:

```text
4 passed in 6.57s
```

This confirms the Gluon import, JIT compiler, explicit distributed layouts,
masked memory operations, arithmetic, launch path, and CUDA result validation.

## Environment Boundary

`run_modal.py` currently pins the upstream Triton `3.7.1` wheel. This is the
active Gluon study environment. It intentionally replaces the earlier
`facebookexperimental/triton` source image after that fork failed to preserve
layouts for tutorial-style elementwise arithmetic on H100.

TLX source and exercises remain in the repository, but TLX kernels will not run
in this upstream-only image. Returning to TLX requires an explicit, documented
environment change rather than a compatibility fallback.

## Next Step

After this smoke test passes, continue with G0 through G2 in
`exercises/gluon_memory_layout/README.md`: layout arithmetic, layout-aware 1D
memcpy, and the per-thread work/SASS sweep.
