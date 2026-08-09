# Task: cuVS (GPU-accelerated ANN)

## Status: scaffolded and researched, NOT executed (no GPU in this environment)

This machine has no NVIDIA GPU / CUDA toolkit (`nvidia-smi` and `nvcc` both
absent), so nothing here has been run -- unlike FAISS and ParlayANN, which
were both actually built and executed. Don't treat anything below as
verified until it's been run on real GPU hardware. This is the honest state
to walk into a conversation with Chen with -- "verified on CPU-only tasks,
scaffolded but untested on the GPU task" is a real, credible status update.

## Real structure (confirmed by cloning `rapidsai/cuvs`)

- Python bindings exist and are real: `python/cuvs/cuvs/neighbors/ivf_flat/`
  (Cython `.pyx`/`.pxd`), plus `cagra`, `hnsw`, and others under
  `python/cuvs/cuvs/neighbors/`.
- `cuvs_bench` (`python/cuvs_bench/`) has ready-made YAML configs for
  IVF-Flat, including CPU-comparison configs (`faiss_cpu_ivf_flat.yaml`),
  which is useful context for how this project already thinks about
  cross-library comparison.
- `examples/c/src/ivf_flat_c_example.c` is a minimal end-to-end C example --
  the closest analog to FAISS's `perf_script.py` pattern once you have GPU
  access.

## What you'll need to actually run this (on a GPU machine)

1. An NVIDIA GPU + matching CUDA toolkit (check `cuvs`'s `dependencies.yaml`
   for the exact supported CUDA version at clone time -- this moves).
2. Either build from source (`build.sh` at repo root) or install via conda
   (`rapidsai` channel) -- building cuVS from source is a substantial
   compile (it's a full RAPIDS library); conda install is much faster to get
   started if the goal is just running the benchmark, not developing cuVS
   itself.
3. Adapt the FAISS `perf_script.py` pattern: build an `ivf_flat` (or
   `cagra`) index over a synthetic or real dataset, time search, and record
   recall the same way `check_recall.py` does for FAISS.

## Where to actually do this

Given the GPU requirement, this task is the natural one to run on your
Rockfish access (once the allocation question with Arora is resolved) or
whatever GPU resource Chen points you to -- not locally. Worth raising this
explicitly with Chen rather than assuming.

## Suggested first task once GPU access is available

Mirror the FAISS task directly for comparability: build an IVF-Flat index in
cuVS on the same shape of synthetic data (e.g. 200k x 64), and give the
agent the equivalent of `perf_script.py` + `check_recall.py`. This makes the
FAISS (CPU, exact + approximate) and cuVS (GPU, approximate) results
directly comparable, and lets you speak to all three libraries in the same
vocabulary in the eventual writeup.
