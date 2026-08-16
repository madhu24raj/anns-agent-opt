"""
Benchmark workload for FAISS brute-force (IndexFlatL2) search, followed by
a Python-side re-ranking step.

Supports two data sources:
  --data synthetic (default)  -- random uniform vectors, no download needed
  --data sift1m --data_dir X  -- real SIFT1M data, run setup/download_sift1m.sh
                                  first to populate X with sift_base.fvecs /
                                  sift_query.fvecs

Usage:
    python perf_script.py                                  # synthetic
    python perf_script.py --loop 10                         # synthetic, repeated
    python perf_script.py --data sift1m --data_dir ./sift1m # real data
"""
import argparse
import time
from pathlib import Path

import numpy as np
import faiss

from vecs_io import fvecs_read


def build_index_synthetic(d: int, n: int, seed: int = 1234):
    rng = np.random.default_rng(seed)
    xb = rng.random((n, d), dtype=np.float32)
    index = faiss.IndexFlatL2(d)
    index.add(xb)
    return index, xb


def make_queries_synthetic(d: int, nq: int, seed: int = 5678):
    rng = np.random.default_rng(seed)
    return rng.random((nq, d), dtype=np.float32)


def load_sift1m(data_dir: str):
    """
    Loads real SIFT1M data. Expects sift_base.fvecs and sift_query.fvecs in
    data_dir (produced by setup/download_sift1m.sh -- run that on your own
    machine first, this environment can't reach the dataset host).
    """
    base_path = Path(data_dir) / "sift_base.fvecs"
    query_path = Path(data_dir) / "sift_query.fvecs"
    if not base_path.exists() or not query_path.exists():
        raise FileNotFoundError(
            f"Expected {base_path} and {query_path}. "
            f"Run setup/download_sift1m.sh first, or use --data synthetic."
        )
    xb = fvecs_read(str(base_path))
    xq = fvecs_read(str(query_path))
    d = xb.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(xb)
    return index, xb, xq


def rerank_naive(xb: np.ndarray, xq: np.ndarray, I: np.ndarray) -> np.ndarray:
    """
    Deliberately naive re-ranking step: for each query, re-derive exact L2
    distances to its candidate set one-by-one in a Python loop, instead of
    vectorizing. Correct but far from optimal -- the "shallow" bottleneck.
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


def run(data="synthetic", data_dir="./sift1m", n=200_000, d=64, nq=2_000, k=20, loops=1):
    if data == "sift1m":
        index, xb, xq = load_sift1m(data_dir)
        n, d = xb.shape
        nq = xq.shape[0]
    else:
        index, xb = build_index_synthetic(d, n)
        xq = make_queries_synthetic(d, nq)

    durations = []
    for _ in range(loops):
        t0 = time.perf_counter()
        D, I = index.search(xq, k)
        _ = rerank_naive(xb, xq, I)
        t1 = time.perf_counter()
        durations.append(t1 - t0)

    avg = sum(durations) / len(durations)
    print(f"data={data} n={n} d={d} nq={nq} k={k} loops={loops}")
    print(f"avg wall time: {avg:.4f}s (runs: {[f'{x:.4f}' for x in durations]})")
    return avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", type=int, default=1)
    parser.add_argument("--data", choices=["synthetic", "sift1m"], default="synthetic")
    parser.add_argument("--data_dir", default="./sift1m")
    args = parser.parse_args()
    run(data=args.data, data_dir=args.data_dir, loops=args.loop)
