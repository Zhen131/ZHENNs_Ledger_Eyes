#!/usr/bin/env bash
# Download the public dataset into data/raw/ (see scripts/download_data.py).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
exec "$EYES_PYTHON" "$REPO_DIR/scripts/download_data.py" "$@"
