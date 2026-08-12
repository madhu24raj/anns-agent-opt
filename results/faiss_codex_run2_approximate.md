# Result: faiss / Codex / August 12, 2026

**Library:** faiss  
**Agent:** Codex  
**Model:** gpt-5.6-sol  
**Task:** [benchmarks/faiss/README.md](benchmarks/faiss/README.md)  

---

## Raw outcome
- **Baseline runtime / QPS:** 0.5442s average wall time (local M-series Mac benchmark)
- **Agent's final runtime / QPS:** 0.1993s average wall time
- **Speedup:** ~2.7x improvement
- **Recall before / after:** 1.0000 (100%) / 0.9621 (96.21%)

---

## PerfAgent's three failure modes (arXiv:2607.19653, Section III) -- did we see them?

**1. Missing the real bottleneck**
* **Observed?** No. 
* **Details:** Unlike the exact-match prompt, the agent successfully targeted both the deep 90% bottleneck (by swapping `IndexFlatL2` for `IndexHNSWFlat`) and the shallow 10% bottleneck (by vectorizing the `rerank_naive` Python loop using `np.einsum`).

**2. Premature termination**
* **Observed?** No. 
* **Details:** The agent iterated significantly. It ran an internal parameter sweep comparing IVF and HNSW configurations to find the optimal speed/accuracy tradeoff before settling on its final patch.

**3. Insufficient testing / correctness regressions**
* **Observed?** No. 
* **Details:** The agent modified `check_recall.py` to test the actual generated index, validating that its tuned HNSW parameters (`efSearch=288`, `M=32`) achieved a recall of 0.9621, safely clearing the 0.95 threshold.

---

## ANNS-specific failure mode (not in PerfAgent, since PerfAgent doesn't cover ANNS specifically -- this is the preliminary-evidence angle)

The agent successfully recognized that runtime is meaningless without recall. When given permission via the prompt to use approximate methods with a recall floor (`>= 0.95`), the agent expertly navigated the tradeoff curve rather than picking a faster-but-worse point. It tuned the HNSW hyperparameters to maximize latency reduction while strictly honoring the recall constraint.

---

## Notes / surprises

* **The Power of the Recall Floor:** The difference between Run 1 (6% speedup) and Run 2 (2.7x speedup) was entirely dictated by the prompt's definition of "correctness." Standard SWE benchmarks demand exact outputs; ANNS optimization requires a recall floor.
* **Agent Sophistication:** The agent was sophisticated enough to recognize that index construction (`index.add()`) occurred outside the timed loop, allowing it to select HNSW (which has slow build times but incredibly fast search times) as the optimal index for this specific benchmark structure.
