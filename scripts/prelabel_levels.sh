#!/usr/bin/env bash
# Pre-label stock tweets with the level of their reason through the local Ollama server (see scripts/prelabel_levels.py).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
exec "$EYES_PYTHON" "$REPO_DIR/scripts/prelabel_levels.py" "$@"
