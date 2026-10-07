"""Download the public datasets into data/raw/.

    scripts/download_data.sh           the StockTwits crypto dataset (69.6 MB) into data/raw/
    scripts/download_data.sh --dated   the two dated datasets (about 3.6 GB) into data/raw/dlt-tweets/ and
                                       data/raw/dlt-news/, at their pinned revisions

Run it through scripts/download_data.sh so the environment from
scripts/env.sh is in place (cache under data/, no token, no telemetry).

A file that already exists is never overwritten: it is kept, and for the
dated datasets its size and SHA-256 are compared with the dataset's record.
An interrupted download resumes from its partial file on the next run, and a
failed file is tried again a few times.
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "ElKulako/stocktwits-crypto"  # public dataset, MIT license
FILES = ["st-data-full.xlsx", "README.md"]  # the data file and the dataset card

# The dated datasets (CC BY-NC 4.0), each pinned to a revision and kept in a folder of its own under data/raw/,
# with the dataset's own layout (data/*.parquet) and its card (README.md) inside that folder.
DATED = [
    {"repo_id": "ExponentialScience/DLT-Tweets", "revision": "c7d979394488bbd94bd6f493af1775d6b6159176",
     "folder": "dlt-tweets"},
    {"repo_id": "ExponentialScience/DLT-Sentiment-News", "revision": "b52bfb8b3ec099924a52f6f4bfd1a8ac6c89c9e1",
     "folder": "dlt-news"},
]
ATTEMPTS = 5  # tries per file before giving up
WAIT_SECONDS = 10  # pause between tries


def sha256_of(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(repo_id, filename, local_dir, revision=None, size=None):
    """Download one file into local_dir unless it is already there; returns (path, how)."""
    target = local_dir / filename
    if target.exists():
        if size is not None and target.stat().st_size != size:
            raise RuntimeError(f"{target} exists with {target.stat().st_size:,} bytes, the dataset records {size:,}; "
                               "it is not overwritten: move it away and run again")
        return target, "kept (already there)"
    for attempt in range(1, ATTEMPTS + 1):
        try:
            # token=False: never send a token, even if one is stored on this machine
            local = Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", revision=revision,
                                         local_dir=local_dir, token=False))
            return local, "downloaded"
        except Exception as exc:  # network trouble: the partial file stays and the next try resumes it
            if attempt == ATTEMPTS:
                raise
            print(f"  {filename}: attempt {attempt} failed ({type(exc).__name__}: {exc}); trying again", flush=True)
            time.sleep(WAIT_SECONDS)
    raise AssertionError("unreachable")


def download_stocktwits(repo_dir):
    """The StockTwits crypto dataset and its card into data/raw/."""
    target = repo_dir / "data" / "raw"
    target.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        local, how = fetch(REPO_ID, name, target)
        print(f"{local.relative_to(repo_dir)}  {local.stat().st_size:,} bytes  sha256={sha256_of(local)}  {how}",
              flush=True)


def download_dated(repo_dir):
    """The two dated datasets, at their pinned revisions, each into its own folder under data/raw/."""
    api = HfApi()
    for dataset in DATED:
        target = repo_dir / "data" / "raw" / dataset["folder"]
        target.mkdir(parents=True, exist_ok=True)
        info = api.dataset_info(dataset["repo_id"], revision=dataset["revision"], files_metadata=True, token=False)
        files = [item for item in info.siblings
                 if item.rfilename == "README.md" or (item.rfilename.startswith("data/")
                                                      and item.rfilename.endswith(".parquet"))]
        print(f"{dataset['repo_id']}@{dataset['revision']}: {len(files)} files, "
              f"{sum(item.size or 0 for item in files):,} bytes -> {target.relative_to(repo_dir)}/", flush=True)
        for item in sorted(files, key=lambda item: item.rfilename):
            started = time.perf_counter()
            local, how = fetch(dataset["repo_id"], item.rfilename, target, dataset["revision"], item.size)
            expected = item.lfs.sha256 if item.lfs else None
            digest = sha256_of(local) if expected else ""
            if expected and digest != expected:
                raise RuntimeError(f"{local} has sha256 {digest}, the dataset records {expected}; it is not "
                                   "overwritten: move it away and run again")
            print(f"  {local.relative_to(repo_dir)}  {local.stat().st_size:,} bytes  "
                  f"{'sha256 matches the record' if expected else 'no sha256 recorded'}  {how}  "
                  f"{time.perf_counter() - started:.1f} s", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the public datasets into data/raw/.")
    parser.add_argument("--dated", action="store_true",
                        help="download the two dated datasets (about 3.6 GB) instead of the StockTwits dataset")
    args = parser.parse_args()
    repo_dir = Path(__file__).resolve().parent.parent
    if args.dated:
        download_dated(repo_dir)
    else:
        download_stocktwits(repo_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
