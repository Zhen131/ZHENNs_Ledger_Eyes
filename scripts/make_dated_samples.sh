#!/usr/bin/env bash
# Draw the dated samples into samples/ from data/raw/dlt-tweets/ and data/raw/dlt-news/ (see scripts/make_dated_samples.py).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
exec "$EYES_PYTHON" "$REPO_DIR/scripts/make_dated_samples.py" "$@"
