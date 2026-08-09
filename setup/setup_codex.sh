#!/bin/bash
# Codex CLI setup. Verified working (installed + `codex --version` confirmed
# 0.147.0) as of this repo's creation date.
set -e

echo "Installing Codex CLI..."
npm install -g @openai/codex

echo "Verifying install..."
codex --version

echo ""
echo "Next step (not automated -- needs your credentials, not run here):"
echo "  codex auth"
echo "This requires either an OpenAI API key or a ChatGPT Plus/Pro/Business login."
echo ""
echo "Then, to run against a task directory:"
echo "  cd benchmarks/faiss && codex \"optimize perf_script.py's runtime, do not change its output behavior\""
