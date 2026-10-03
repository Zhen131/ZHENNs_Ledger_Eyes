#!/usr/bin/env bash
# Regenerate notebook/ledger_eyes.ipynb from notebook/ledger_eyes.py.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
exec "$EYES_PYTHON" "$REPO_DIR/scripts/notebook_build.py" "$@"
