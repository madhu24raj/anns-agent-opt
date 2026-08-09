"""
Generates tiny synthetic base/query/ground-truth files in the binary format
ParlayANN's PointRange expects (confirmed by reading algorithms/utils/point_range.h):

    uint32 num_points
    uint32 dim
    float32[num_points * dim]   (row-major, data_type=float)

This lets the vamana `neighbors` binary run end-to-end without needing to
download a real large-scale ANN dataset (SIFT/GIST/etc.) first.
"""
import struct
import numpy as np


def write_fbin(path, X: np.ndarray):
    n, d = X.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<II", n, d))
        f.write(X.astype(np.float32).tobytes())


def write_gt(path, I: np.ndarray):
    """Ground-truth neighbor indices, same header convention (n, k)."""
    n, k = I.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<II", n, k))
        f.write(I.astype(np.uint32).tobytes())


def main(n=5000, d=32, nq=200, k=10, seed=42):
    rng = np.random.default_rng(seed)
    xb = rng.random((n, d), dtype=np.float32)
    xq = rng.random((nq, d), dtype=np.float32)

    # brute-force ground truth
    d2 = ((xq[:, None, :] - xb[None, :, :]) ** 2).sum(-1)
    I_gt = np.argsort(d2, axis=1)[:, :k]

    write_fbin("base.fbin", xb)
    write_fbin("query.fbin", xq)
    write_gt("gt.bin", I_gt)
    print(f"wrote base.fbin ({n}x{d}), query.fbin ({nq}x{d}), gt.bin (k={k})")


if __name__ == "__main__":
    main()
