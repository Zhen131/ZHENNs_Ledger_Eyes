"""Download the public dataset into data/raw/.

    scripts/download_data.sh   TimKoornstra/financial-tweets-sentiment (about 2.6 MB), pinned revision,
                               into data/raw/tim-financial-tweets-bba60269.parquet

Run it through scripts/download_data.sh so the environment from
scripts/env.sh is in place (cache under data/, no token, no telemetry).

The file goes to the same path, and is written the same way, as the copy that
scripts/prelabel_levels.py makes: this script uses that script's own functions
(importing it connects to nothing). The file is written to a temporary name
first and renamed when complete, so an interrupted download never leaves half
a file. A file that already exists is never overwritten: it is kept, and the
script says so.
"""

import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR / "scripts"))

import prelabel_levels as levels  # noqa: E402


def main() -> int:
    if len(sys.argv) > 1:
        print("usage: scripts/download_data.sh   (no arguments)", file=sys.stderr)
        return 2
    try:
        levels.check_environment()  # no mirror, no token: the settings of scripts/env.sh
    except levels.Stop as exc:
        print(f"stopped: {exc}", file=sys.stderr)
        return 1
    copy = REPO_DIR / levels.DATASET_COPY
    if copy.is_file():
        print(f"{levels.DATASET_COPY} already exists ({copy.stat().st_size:,} bytes); kept as it is")
        return 0
    levels.download_copy(copy)
    print(f"{levels.DATASET} at revision {levels.DATASET_REVISION}: {copy.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
