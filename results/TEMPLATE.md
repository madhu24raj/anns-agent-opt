# Result: [library] / [agent] / [date]

**Library:** faiss | parlayann | cuvs
**Agent:** OpenHands | Codex
**Model:** (e.g. GPT-5.1, Kimi-K2)
**Task:** (link to benchmarks/<library>/README.md)

## Raw outcome
- Baseline runtime / QPS:
- Agent's final runtime / QPS:
- Speedup:
- Recall before / after (NOT just "did it pass tests" -- actual recall number):

## PerfAgent's three failure modes (arXiv:2607.19653, Section III) -- did we see them?

**1. Missing the real bottleneck**
Did the agent profile at all? If so, with what tool? Did it find the
dominant cost (e.g. FAISS's index.search() at ~91% of runtime) or only the
shallow one (e.g. the un-vectorized rerank at ~9%)?

**2. Premature termination**
Did the agent stop after the first passing/faster patch, or keep iterating?
Roughly how much headroom (if any) was left on the table vs. what a
human/PerfAgent achieved in the paper for a comparable task?

**3. Insufficient testing / correctness regressions**
Did the agent check recall (or equivalent correctness measure), or only
wall-clock time? Did it silently trade away correctness for speed?

## ANNS-specific failure mode (not in PerfAgent, since PerfAgent doesn't
## cover ANNS specifically -- this is the preliminary-evidence angle)

Did the agent recognize that runtime alone is meaningless without recall?
Did it just pick a faster-but-worse point on an existing tradeoff curve
(e.g. swap in an approximate index without tuning it, like the check_recall.py
0.39 recall example), or did it find something that's a genuinely better
tradeoff?

## Notes / surprises
(free text)
