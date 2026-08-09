# ANNS Agent Optimization -- Preliminary Evidence

Assignment from Xuhao Chen (Aug 2026): use OpenHands and Codex to optimize
ANNS code in FAISS, ParlayANN, and cuVS, as preliminary evidence motivating
extending PerfAgent (arXiv:2607.19653) to vector-search systems specifically.

## What's actually verified vs. what's scaffolded

| Library | Benchmark script | Runs? | Agent run yet? |
|---|---|---|---|
| **FAISS** | `benchmarks/faiss/perf_script.py` | Yes -- real baseline: 1.13s avg, two-layer bottleneck confirmed by profiling (91% index.search / 9% rerank) | Not yet -- needs your API key |
| **ParlayANN** | `benchmarks/parlayann/` (compiled C++ binary) | Yes -- real end-to-end run on synthetic data, full recall-vs-QPS curve captured | Not yet |
| **cuVS** | `benchmarks/cuvs/` | Not run -- no GPU in this environment. Structure researched and documented. | Not yet |

Nothing here has been run through OpenHands or Codex yet -- that step needs
your own API credentials, which is intentionally not something this
scaffolding includes. Everything else (getting real code cloned, compiled,
benchmarked, and profiled) is done and verified.

## Why FAISS and ParlayANN are designed the way they are

Both tasks are deliberately built to test PerfAgent's own failure-mode
taxonomy (missing bottlenecks across abstraction boundaries, premature
termination, insufficient testing) specifically in ANNS code, where there's
an extra wrinkle PerfAgent doesn't have to deal with: **runtime alone is a
meaningless metric without an accuracy/recall constraint.** A do-nothing
"optimization" or a naive index swap can look like a huge speedup while
silently destroying result quality. See each benchmark's README for details.

## Repo layout

```
benchmarks/
  faiss/        verified, real baseline + profiling breakdown
  parlayann/    verified, real end-to-end run + QPS-recall curve
  cuvs/         researched, not executable here (needs GPU)
setup/
  setup_codex.sh       verified: installs + confirms Codex CLI
  setup_openhands.sh   documents current (post-restructure) OpenHands setup
results/
  TEMPLATE.md    fill one of these out per (library, agent) run
external/
  faiss/         actual clone (gitignored -- see below)
  parlayann/     actual clone + compiled binary (gitignored)
  cuvs/          actual clone (gitignored)
```

## Next steps to actually run the agents

1. `bash setup/setup_codex.sh` then `codex auth` (needs your OpenAI
   credentials -- not something to automate here).
2. For OpenHands: see `setup/setup_openhands.sh` -- note this needs your own
   read of their current docs, since it's restructured recently (see that
   file for details and an open question worth confirming with Chen).
3. Point the agent at `benchmarks/faiss/` first (smallest, fastest
   iteration loop). Fill in `results/TEMPLATE.md` for each run.
4. Move to `benchmarks/parlayann/` once FAISS results look reasonable.
5. cuVS last, and only once you have GPU access (Rockfish, once the
   allocation question with Arora is resolved, or wherever Chen points you).

## Pushing this to your own GitHub

This was built locally, not pushed anywhere -- create an empty repo on
GitHub yourself first (I can't authenticate as you), then:

```bash
cd anns-agent-opt
git init
echo "external/" > .gitignore   # the cloned libraries are large; don't commit them
git add .
git commit -m "Initial scaffold: FAISS + ParlayANN verified, cuVS researched"
git remote add origin <your-new-repo-url>
git push -u origin main
```
