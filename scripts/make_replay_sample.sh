#!/usr/bin/env bash
# Draw the replay file samples/replay-tweets.parquet from StephanAkkerman/financial-tweets (see scripts/make_replay_sample.py).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
exec "$EYES_PYTHON" "$REPO_DIR/scripts/make_replay_sample.py" "$@"
