#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible LIBERO entrypoint.
exec "$(dirname "${BASH_SOURCE[0]}")/run_score_flow_benchmark.sh" "$@"
