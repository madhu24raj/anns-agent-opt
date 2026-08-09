#!/bin/bash
# OpenHands setup.
#
# IMPORTANT: OpenHands has restructured significantly since PerfAgent's paper
# (arXiv:2607.19653) was written. It is now "Agent Canvas" -- a self-hosted
# control center that can run the OpenHands agent OR Claude Code, Codex, or
# any Agent-Client-Protocol (ACP) compatible agent as its backend. Verified
# against github.com/OpenHands/OpenHands's current README at repo-creation
# time; not run end-to-end here since it needs an LLM API key and (for the
# Docker path) a Docker daemon, neither of which this scaffolding step has.
set -e

echo "== Option 1: npm install (runs agent-server directly on this machine) =="
echo "Prerequisites: Node.js 22.12.x+, uv"
echo ""
echo "  npm install -g @openhands/agent-canvas"
echo "  agent-canvas"
echo ""
echo "UI at http://localhost:8000"
echo ""
echo "== Option 2: Docker sandbox (safer, recommended if unsure) =="
echo '  export PROJECTS_PATH="$HOME/projects"'
echo '  mkdir -p "$PROJECTS_PATH" "$HOME/.openhands"'
echo "  docker run -it --rm -p 8000:8000 \\"
echo '    -v "$HOME/.openhands:/home/openhands/.openhands" \\'
echo '    -v "${PROJECTS_PATH}:/projects" \\'
echo "    ghcr.io/openhands/agent-canvas:1.12.0"
echo ""
echo "== For headless / scriptable use (matching PerfAgent's usage pattern) =="
echo "The programmatic interface is the OpenHands Agent Server / software-agent-sdk,"
echo "not the Agent Canvas UI. See: https://github.com/OpenHands/software-agent-sdk"
echo "This is worth confirming directly with Chen -- PerfAgent's paper describes"
echo "using an older, simpler CLI-style OpenHands as one of its two baselines"
echo "(Table II), and it's not yet confirmed whether the current Agent Canvas"
echo "restructuring changes how that comparison should be run."
