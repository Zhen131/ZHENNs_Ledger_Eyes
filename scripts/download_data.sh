#!/usr/bin/env bash
# Download the public datasets into data/raw/: no argument for StockTwits, --dated for the two dated datasets (see scripts/download_data.py).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
exec "$EYES_PYTHON" "$REPO_DIR/scripts/download_data.py" "$@"
