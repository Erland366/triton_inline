# Blackwell GEMM Benchmark

This folder contains a quick benchmark for comparing Blackwell GEMM tutorial
kernels:

- `TLX WS`: `triton.language.extra.tlx.tutorials.blackwell_gemm_ws.matmul`
- `TLX CLC`: `triton.language.extra.tlx.tutorials.blackwell_gemm_clc.matmul`
- `TLX 2CTA`: `triton.language.extra.tlx.tutorials.blackwell_gemm_2cta.matmul`
- `Gluon WS`: the installed or local `python/tutorials/gluon/08-warp-specialization.py`
- `Gluon static CLC kernel`: inline static scheduler version of the upstream
  CLC persistent GEMM
- `Gluon CLC`: inline cluster-launch-control persistent GEMM
- `Gluon multiCTA`: inline 2CTA, warp-specialized, CLC-driven GEMM
- `Torch / cuBLAS`: `torch.matmul`

The benchmark uses `triton.testing.Benchmark`, matching the style used in
`exercises/tlx/benchmark.py`. The newer Gluon CLC and multiCTA providers are
defined directly in `benchmark_tlx_vs_gluon.py` so the script is self-contained;
it does not copy tutorial files into this exercise at runtime.

The default provider set is the one expected to work on the current
`run_modal.py` image, which installs `facebookexperimental/triton`:

- `torch`
- `tlx_ws`
- `tlx_clc`
- `tlx_2cta`
- `gluon_ws`
- `gluon_clc_static`

The inline `gluon_clc` and `gluon_multicta` providers are available by name, but
they need a Triton/Gluon build that exposes
`triton.experimental.gluon.language.nvidia.blackwell.clc`.

## Run

This requires a Blackwell CUDA GPU. The default run benchmarks square
`2048`, `4096`, and `8192` fp16 GEMMs:

```bash
source .venv/bin/activate
python exercises/blackwell_gemm/benchmark_tlx_vs_gluon.py
```

To make a faster smoke run:

```bash
source .venv/bin/activate
python exercises/blackwell_gemm/benchmark_tlx_vs_gluon.py --sizes 2048 --providers torch tlx_ws gluon_ws --warmup 20 --rep 50
```

With `run_modal.py`, pass benchmark arguments inside `--script`:

```bash
modal run run_modal.py --action run --script "exercises/blackwell_gemm/benchmark_tlx_vs_gluon.py --sizes 2048 --providers torch tlx_ws gluon_ws --warmup 20 --rep 50"
```

For `breakpoint()` debugging, use the `debug` action with Modal interactive
mode. This runs the script in-process with `runpy` instead of as a captured
subprocess, so the debugger can attach to your terminal:

```bash
modal run -i run_modal.py --action debug --script "exercises/blackwell_gemm/benchmark_tlx_vs_gluon.py --sizes 512 --providers gluon_ws --warmup 1 --rep 1"
```

To focus on the newer Blackwell scheduling paths:

```bash
modal run run_modal.py --action run --script "exercises/blackwell_gemm/benchmark_tlx_vs_gluon.py --sizes 4096 8192 --providers torch tlx_clc tlx_2cta gluon_clc gluon_multicta --warmup 100 --rep 300"
```

That command will fail fast on the current `facebookexperimental/triton` Modal
image because the image does not currently expose Gluon `blackwell.clc`.

The script saves the plot under `plots/`.

## Notes

- The benchmark is fp16-only.
- The script sets `TLX_GEMM_USE_HEURISTIC=1` by default so the TLX wrapper uses
  its shape heuristic instead of relying on a slower setup path.
- Some TLX snapshots have a `TensorDescriptor` compatibility issue where
  NVIDIA Gluon descriptors read `round_f32_to_tf32` without defining it. The
  benchmark patches that attribute to `False` for this fp16 path before loading
  the Gluon tutorial.
- The Gluon scheduler class is cached once and the output tensor is preallocated
  for each benchmark shape. Otherwise host-side setup between CUDA event records
  can distort the small-GEMM timings.
- The inline Gluon CLC and multiCTA providers require a Triton/Gluon build that
  exposes `triton.experimental.gluon.language.nvidia.blackwell.clc`. Older TLX
  snapshots have only the earlier Gluon tutorials and will fail loudly for those
  providers.
- `run_modal.py` must request a Blackwell GPU for this benchmark.
