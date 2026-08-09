"""
Benchmark workload for FAISS brute-force (IndexFlatL2) search, followed by
a Python-side re-ranking step.

This mirrors the pattern used by PerfAgent's perf_script.py (arXiv:2607.19653):
a small, self-contained script that exercises one API path in the repo, whose
wall-clock time is what the coding agent (OpenHands / Codex) is asked to reduce
without changing the script's observable behavior (same top-k neighbors for
the same query set, same recall).

Usage:
    python perf_script.py            # single timed run
    python perf_script.py --loop 10  # repeated timed runs (for stable timing)
"""
import argparse
import time

import numpy as np
import faiss


def build_index(d: int, n: int, seed: int = 1234):
    rng = np.random.default_rng(seed)
    xb = rng.random((n, d), dtype=np.float32)
    index = faiss.IndexFlatL2(d)
    index.add(xb)
    return index, xb


def make_queries(d: int, nq: int, seed: int = 5678):
    rng = np.random.default_rng(seed)
    return rng.random((nq, d), dtype=np.float32)


def rerank_naive(xb: np.ndarray, xq: np.ndarray, I: np.ndarray) -> np.ndarray:
    """
    Deliberately naive re-ranking step: for each query, re-derive exact L2
    distances to its candidate set one-by-one in a Python loop, instead of
    vectorizing. This is representative of the kind of "shallow" bottleneck
    PerfAgent's case studies describe -- correct, but far from optimal, and
    easy for a profiler (not just eyeballing) to catch.
    """
    nq, k = I.shape
    out = np.empty((nq, k), dtype=np.float32)
    for i in range(nq):
        q = xq[i]
        for j in range(k):
            cand = xb[I[i, j]]
            diff = q - cand
            out[i, j] = float(np.dot(diff, diff))
    return out


def run(n=200_000, d=64, nq=2_000, k=20, loops=1):
    index, xb = build_index(d, n)
    xq = make_queries(d, nq)

    durations = []
    for _ in range(loops):
        t0 = time.perf_counter()
        D, I = index.search(xq, k)
        _ = rerank_naive(xb, xq, I)
        t1 = time.perf_counter()
        durations.append(t1 - t0)

    avg = sum(durations) / len(durations)
    print(f"n={n} d={d} nq={nq} k={k} loops={loops}")
    print(f"avg wall time: {avg:.4f}s (runs: {[f'{x:.4f}' for x in durations]})")
    return avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", type=int, default=1)
    args = parser.parse_args()
    run(loops=args.loop)
