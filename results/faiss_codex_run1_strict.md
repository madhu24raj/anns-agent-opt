# Result: faiss / Codex / August 12, 2026

**Library:** faiss  
**Agent:** Codex  
**Model:** gpt-5.6-sol  
**Task:** [benchmarks/faiss/README.md](benchmarks/faiss/README.md)  

---

## Raw outcome
- **Baseline runtime / QPS:** 0.4946s average wall time (local M-series Mac benchmark)
- **Agent's final runtime / QPS:** 0.4630s average wall time
- **Speedup:** 1.068x (~6.4% improvement)
- **Recall before / after:** 1.0000 (100%) / 1.0000 (100%) exact match

---

## PerfAgent's three failure modes (arXiv:2607.19653, Section III) -- did we see them?

**1. Missing the real bottleneck**
* **Observed?** Yes.
* **Details:** The agent used `cProfile` and correctly identified the performance split: ~90% in FAISS's native `index.search()` and ~10% in the Python `rerank_naive` function. However, it bypassed optimizing the core 90% search bottleneck altogether, opting for a shallow Python-level edit (removing the redundant `rerank_naive` call and directly reusing `IndexFlatL2.search()`'s returned distances `D`).

**2. Premature termination**
* **Observed?** Yes.
* **Details:** The agent stopped immediately after achieving a minor 6.4% speedup on its first attempt and emitting a completion signal, leaving substantial performance headroom unaddressed.

**3. Insufficient testing / correctness regressions**
* **Observed?** No.
* **Details:** The agent ran rigorous verification scripts checking for exact top-k neighbor ID equality and floating-point distance tolerances (max diff `1.71e-05`), successfully preserving 100% exact correctness.

---

## ANNS-specific failure mode (not in PerfAgent, since PerfAgent doesn't cover ANNS specifically -- this is the preliminary-evidence angle)

The agent recognized that swapping to an approximate index (e.g., IVF or HNSW) would alter distance values/rankings slightly. Because the initial prompt constrained it from altering output behavior, the agent explicitly declined to navigate the recall-versus-QPS tradeoff curve, choosing to optimize only superficial Python wrapper overhead while keeping the search exact.

---

## Notes / surprises

* **Prompt Engineering Insight:** The agent explicitly noted in its trace: *"Because exact top-k is required, switching to IVF/HNSW would violate the request."*
* **Experimental Methodology Takeaway:** Requesting "no change in output behavior" causes coding agents to avoid approximate indexing techniques (IVF/HNSW/CAGRA). Future ANNS optimization benchmarks must explicitly specify an acceptable recall threshold (e.g., `Recall@20 >= 0.95`) so agents are authorized to optimize search indices within a valid accuracy envelope.