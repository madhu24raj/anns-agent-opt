"""
Recall check: compares a candidate index's search results against exact
brute-force ground truth on a held-out query set.

This exists specifically to catch the failure mode PerfAgent's own
correctness-testing component targets: a patch that is fast but silently
wrong. Swapping IndexFlatL2 for an approximate index is a legitimate
optimization ONLY if recall stays acceptable -- this script makes that
measurable instead of assumed.

Usage:
    python check_recall.py            # uses perf_script.py's data generation
"""
import numpy as np
import faiss

from perf_script import build_index, make_queries


def recall_at_k(I_approx: np.ndarray, I_exact: np.ndarray, k: int) -> float:
    """Fraction of true top-k neighbors recovered, averaged over queries."""
    nq = I_exact.shape[0]
    hits = 0
    total = 0
    for i in range(nq):
        true_set = set(I_exact[i, :k].tolist())
        approx_set = set(I_approx[i, :k].tolist())
        hits += len(true_set & approx_set)
        total += k
    return hits / total


def main(n=200_000, d=64, nq=2_000, k=20):
    candidate, xb = build_index(d, n)
    xq = make_queries(d, nq)

    # ground truth: exact brute-force
    flat = faiss.IndexFlatL2(d)
    flat.add(xb)
    _, I_exact = flat.search(xq, k)

    # Candidate under test: use the same index configuration as perf_script.
    _, I_approx = candidate.search(xq, k)

    r = recall_at_k(I_approx, I_exact, k)
    print(f"recall@{k}: {r:.4f}")
    return r


if __name__ == "__main__":
    main()
