"""
Reader for the .fvecs / .ivecs binary format used by SIFT1M and the other
texmex-corpus / BigANN-family datasets (this is the same format FAISS's own
demo_sift1M.cpp reads -- see facebookresearch/faiss/demos/demo_sift1M.cpp).

Format (repeated per vector, no separate header):
    int32   dim
    float32[dim]   (fvecs)   OR   int32[dim]   (ivecs, used for ground truth)

This is deliberately dependency-light (just numpy) so it drops into the
existing perf_script.py / check_recall.py without new requirements.
"""
import numpy as np


def fvecs_read(path: str) -> np.ndarray:
    a = np.fromfile(path, dtype="int32")
    if a.size == 0:
        raise ValueError(f"{path}: empty or unreadable file")
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy().view("float32")


def ivecs_read(path: str) -> np.ndarray:
    a = np.fromfile(path, dtype="int32")
    if a.size == 0:
        raise ValueError(f"{path}: empty or unreadable file")
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_write(path: str, X: np.ndarray) -> None:
    """Inverse of fvecs_read -- used here only to generate a synthetic file
    that exercises the real reader, since the real SIFT1M files aren't
    reachable from this environment."""
    n, d = X.shape
    header = np.full((n, 1), d, dtype="int32")
    body = X.astype("float32").view("int32")
    out = np.hstack([header, body])
    out.astype("int32").tofile(path)


if __name__ == "__main__":
    # Self-test: write a small synthetic .fvecs in the REAL format, read it
    # back, confirm round-trip correctness. This validates the parser logic
    # against the actual binary layout SIFT1M uses -- it does not validate
    # against the real dataset itself, since that file isn't downloadable
    # from this sandbox. Run download_sift1m.sh on your own machine to get
    # the real thing, then point perf_script.py at it.
    rng = np.random.default_rng(0)
    X = rng.random((37, 128), dtype=np.float32)
    fvecs_write("/tmp/_selftest.fvecs", X)
    Y = fvecs_read("/tmp/_selftest.fvecs")
    assert Y.shape == X.shape, (Y.shape, X.shape)
    assert np.allclose(X, Y), "round-trip mismatch"
    print(f"fvecs round-trip OK: shape={Y.shape}, dtype={Y.dtype}")
