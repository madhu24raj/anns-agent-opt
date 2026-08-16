"""
Recall check: compares a candidate index's search results against exact
brute-force ground truth on a held-out query set.

Supports --data synthetic (default) or --data sift1m --data_dir X, matching
perf_script.py.
"""
import argparse
import numpy as np
import faiss

from perf_script import build_index_synthetic, make_queries_synthetic, load_sift1m


def recall_at_k(I_approx: np.ndarray, I_exact: np.ndarray, k: int) -> float:
    nq = I_exact.shape[0]
    hits = 0
    total = 0
    for i in range(nq):
        true_set = set(I_exact[i, :k].tolist())
        approx_set = set(I_approx[i, :k].tolist())
        hits += len(true_set & approx_set)
        total += k
    return hits / total


def main(data="synthetic", data_dir="./sift1m", n=200_000, d=64, nq=2_000, k=20):
    if data == "sift1m":
        _, xb, xq = load_sift1m(data_dir)
        d = xb.shape[1]
    else:
        _, xb = build_index_synthetic(d, n)
        xq = make_queries_synthetic(d, nq)

    # ground truth: exact brute-force
    flat = faiss.IndexFlatL2(d)
    flat.add(xb)
    _, I_exact = flat.search(xq, k)

    # example approximate index -- swap this for whatever the agent produces
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, 100)
    ivf.train(xb)
    ivf.add(xb)
    ivf.nprobe = 8
    _, I_approx = ivf.search(xq, k)

    r = recall_at_k(I_approx, I_exact, k)
    print(f"data={data} recall@{k}: {r:.4f}")
    return r


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["synthetic", "sift1m"], default="synthetic")
    parser.add_argument("--data_dir", default="./sift1m")
    args = parser.parse_args()
    main(data=args.data, data_dir=args.data_dir)
