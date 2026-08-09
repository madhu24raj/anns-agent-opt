# Task: FAISS brute-force search + re-ranking

## Setup
`perf_script.py` builds a 200k x 64-dim `IndexFlatL2`, searches 2,000 queries
for k=20 neighbors, then re-ranks each candidate set with a naive Python
double-loop that redundantly recomputes L2 distances FAISS already computed
internally.

## Verified baseline (measured on this machine, 3 runs)
```
avg wall time: 1.1287s
```

## Real profiling breakdown (measured, not assumed)
```
index.search():   1.1078s  (~91% of total)
rerank_naive():    0.1095s  (~9% of total)
```

## Why this task is designed the way it is

This has **two distinct, real optimization opportunities**, deliberately layered
to test which failure mode (if any) the agent falls into, per PerfAgent's own
taxonomy (arXiv:2607.19653, Section III):

1. **The shallow, easy-to-find win (~9% of runtime):** `rerank_naive` is an
   obvious, un-vectorized Python double-loop. Any agent that looks at the code
   at all will likely find and fix this. Vectorizing it should yield a small,
   real speedup -- but stopping here is a "premature termination" in
   PerfAgent's vocabulary.

2. **The real, dominant bottleneck (~91% of runtime), across an abstraction
   boundary:** `IndexFlatL2.search()` is *exhaustive brute-force* search --
   it is correct but algorithmically the slowest index type FAISS offers.
   A genuine optimization requires recognizing that the bottleneck lives
   inside a native (C++) FAISS internal, not in the visible Python script,
   and switching to an approximate index (e.g. `IndexIVFFlat` or
   `IndexHNSWFlat`) -- which requires understanding FAISS's index API, not
   just the script in front of the agent.

**This is intentional.** PerfAgent's central finding is that general-purpose
agents miss bottlenecks "hidden behind abstraction layers and native
extensions" and "stop after shallow speedups." This task is built specifically
to surface that failure mode with ANNS code: does the agent stop at the easy
9% win, or does it go looking for -- and find -- the 91%?

## What to record when running an agent against this

- Did it profile at all, or just read the code?
- Did it fix `rerank_naive` only, or also touch the index type / search call?
- If it changed the index type: did it preserve **recall**, not just speed?
  (An approximate index trades recall for speed -- a correctness-blind agent
  could "optimize" this into a much faster but much worse search.)
- Final measured speedup, and where the remaining time (if any) is spent.

## Correctness / recall check

Any change must be validated against recall, not just wall-clock time, since
approximate indexes are a legitimate-but-different tradeoff, not a free win.
See `check_recall.py` for a script that compares an index's top-k results
against the brute-force ground truth on a held-out query set.
