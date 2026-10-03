"""Download the public StockTwits crypto dataset into data/raw/.

Run it through scripts/download_data.sh so the environment from
scripts/env.sh is in place (cache under data/, no token, no telemetry).
"""

import hashlib
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "ElKulako/stocktwits-crypto"  # public dataset, MIT license
FILES = ["st-data-full.xlsx", "README.md"]  # the data file and the dataset card


def sha256_of(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    repo_dir = Path(__file__).resolve().parent.parent
    target = repo_dir / "data" / "raw"
    target.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        # token=False: never send a token, even if one is stored on this machine
        local = Path(hf_hub_download(repo_id=REPO_ID, filename=name, repo_type="dataset",
                                     local_dir=target, token=False))
        print(f"{local.relative_to(repo_dir)}  {local.stat().st_size:,} bytes  sha256={sha256_of(local)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
