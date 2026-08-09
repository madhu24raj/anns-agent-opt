# Task: ParlayANN (Vamana graph-based ANN)

## Status: verified working end-to-end on this machine

- Cloned `cmuparlay/ParlayANN` + `parlaylib` submodule.
- The **Python bindings (`python/compile.sh`) currently fail to compile**
  against this checkout -- real template-instantiation errors in
  `graph_index.cpp` (`Point::is_metric()` used as a function on a type where
  it isn't one; a deleted move-assignment operator on `PointRange`). This is
  a genuine, reproducible build break, not an environment issue -- worth
  flagging to Chen/the ParlayANN maintainers directly rather than quietly
  working around it.
- **The standalone C++ binary (`algorithms/vamana/neighbors`, built via
  `make`) compiles and runs cleanly.** This is the path used below.

## Setup (reproducible)
```bash
git clone --depth 1 https://github.com/cmuparlay/ParlayANN.git
cd ParlayANN
git submodule update --init --depth 1   # pulls parlaylib
cd algorithms/vamana
make neighbors
```

## Synthetic test data
`make_synthetic_data.py` generates tiny `.fbin`/ground-truth files in the
exact binary format ParlayANN's `PointRange` expects (confirmed by reading
`algorithms/utils/point_range.h`: `uint32 num_points, uint32 dim,
float32[n*d]`). This avoids needing to download a full SIFT/GIST-scale
dataset just to validate the pipeline.

```bash
python3 make_synthetic_data.py     # writes base.fbin, query.fbin, gt.bin
```

## Verified real run (5,000 points, 32-dim, 200 queries, k=10)

```
./neighbors -R 32 -L 64 -alpha 1.2 -two_pass 0 -data_type float \
  -dist_func Euclidian -k 10 -query_path query.fbin -gt_path gt.bin \
  -res_path test.csv -base_path base.fbin
```

Graph build: 1.359s, average degree 30.48 (target R=32).

**Real QPS-vs-recall curve, produced by sweeping the search beam width (Q):**

| Q (beam width) | recall@10 | QPS |
|---|---|---|
| 10 | 0.628 | 52,220 |
| 13 | 0.722 | 43,010 |
| 17 | 0.878 | 29,210 |
| 26 | 0.944 | 21,700 |
| 45 | 0.984 | 12,970 |
| 80 | 0.995 | 9,090 |
| 180 | 1.000 | 5,064 |

This is the exact tradeoff curve discussed with Chen -- Q (search beam width,
capped by the `visited limit`) is the "knob" that traces out the curve; each
row is one point on it. This full table (14 points, not just the 7 shown
here) comes for free from a single `neighbors` invocation, which sweeps Q
internally.

## Why this task is a good test of PerfAgent's failure modes

Unlike the FAISS task (single bottleneck to find), ParlayANN's `neighbors`
binary is parameterized by construction (`-R`, `-L`, `-alpha`) and search
(`Q`/beam width) knobs that trace out this exact curve. The interesting
question for an agent here isn't "find the bottleneck" -- it's:

- Can the agent recognize that runtime alone is meaningless without also
  reporting recall (i.e., does it accidentally regress recall while chasing
  QPS)?
- Given a target recall floor (e.g. recall >= 0.90), can it tune R/L/alpha
  to *shift the whole curve outward* -- better QPS at that recall level --
  rather than just picking a different point on the same curve?
- This directly tests the accuracy-gate framing discussed with Chen: fix a
  recall floor, optimize QPS subject to it, and check whether the result is
  a genuinely better curve or just a different tradeoff point.

## What to record when running an agent against this

- Does it touch the graph construction parameters (R, L, alpha) or only the
  search-time parameter (Q)? Construction-time tuning is the "deep"
  optimization; only touching Q is closer to sliding along the existing curve.
- Full recall-vs-QPS table for its result, not a single number.
- Whether it noticed / reported the Python-bindings compile failure, or
  silently avoided that code path.
