#!/usr/bin/env bash
# Run every fast test suite. No TF, no GPU, no Cosmos clone needed.
# Self-locating: works from any cwd.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."          # → DreamLoop/pipeline/
PY="${PYTHON:-python3}"
echo "=== test_pipeline.py ==="
"$PY" -m tests.test_pipeline
echo
echo "=== test_edge_cases.py ==="
"$PY" -m tests.test_edge_cases
echo
echo "=== test_primitives.py ==="
"$PY" -m tests.test_primitives
echo
echo "✓ all suites green"
