#!/usr/bin/env bash
# Pre-label posts with emotions through the local Ollama server (see scripts/prelabel_emotions.py).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
exec "$EYES_PYTHON" "$REPO_DIR/scripts/prelabel_emotions.py" "$@"
