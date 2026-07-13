# CUDA Graph Benchmark

This folder contains a small benchmark comparing Triton's normal benchmark path
against CUDA graph replay:

- `triton.testing.do_bench`: normal repeated kernel launches, with Triton's
  cache-flush benchmark behavior.
- `triton.testing.do_bench_cudagraph`: captures repeated calls into a CUDA graph
  and times graph replay to reduce Python/kernel-launch overhead.

Run:

```bash
source .venv/bin/activate
python exercises/cuda_graph_benchmark/benchmark.py
```

The script uses a preallocated Triton vector-add kernel so CUDA graph capture
does not include tensor allocation. Results are reported as GB/s and saved under
`plots/` by default.
