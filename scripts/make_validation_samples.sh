#!/usr/bin/env bash
# Write the made-up price file and the Fear and Greed Index excerpt into samples/ (see scripts/make_validation_samples.py).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
exec "$EYES_PYTHON" "$REPO_DIR/scripts/make_validation_samples.py" "$@"
