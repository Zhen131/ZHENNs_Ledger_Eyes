#!/usr/bin/env bash
# Project checks: scripts/check.sh quick|full [-p NAME VALUE ...]  (see scripts/check.py)
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
exec "$EYES_PYTHON" "$REPO_DIR/scripts/check.py" "$@"
