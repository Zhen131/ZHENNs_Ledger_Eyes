"""Draw the replay file samples/replay-tweets.parquet from StephanAkkerman/financial-tweets.

Run through scripts/make_replay_sample.sh (it sets up the environment from scripts/env.sh):

    scripts/make_replay_sample.sh [--output PATH] [--max-per-ticker N] [--seed N]

What it does, in order (see "The replay sample" in README.md):

1. Reads the local copy data/raw/stephan-financial-tweets-ad27fe2d.parquet. When it is missing, it downloads the
   dataset's file at the pinned revision from Hugging Face (no token), through a temporary file and a rename; a copy
   that is already there is never overwritten.
2. Keeps the posts of the last 365 days of the copy (from the latest created_at minus 365 days to the latest
   created_at, both included) whose tickers name NVDA, IBM or SPY.
3. Removes the authors: only the six columns post_id, created_at, tickers, text, likes and views are written (no
   tweet_id, no author, no picture, no link column); in the text every link becomes "http" and every account name
   "@user" (links first, then account names).
4. Leaves out texts that are empty after stripping, and keeps one post per text (the text after step 3, compared
   exactly): the earliest one, and of two posts at the same time the one with the smaller row in the copy.
5. Caps every ticker at --max-per-ticker posts (default 4000; a post counts for every ticker it names): a ticker
   above the cap is drawn down evenly over time, every week in proportion, with --seed.
6. post_id is "stephan:" and the first 8 hexadecimal digits of the SHA-256 of the written text (UTF-8).

It prints the copy's size and SHA-256, the posts kept per step and per ticker, and the first and last time.
"""

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent

DATASET = "StephanAkkerman/financial-tweets"  # Hugging Face dataset, MIT License
DATASET_REVISION = "ad27fe2d13415186a552a332cbff917530c69a29"  # pinned revision
DATASET_FILE = "data/train.parquet"
DATASET_ADDRESS = f"hf://datasets/{DATASET}@{DATASET_REVISION}/{DATASET_FILE}"
DATASET_COPY = f"data/raw/stephan-financial-tweets-{DATASET_REVISION[:8]}.parquet"  # local copy, written once
OUTPUT = "samples/replay-tweets.parquet"
TICKERS = ["NVDA", "IBM", "SPY"]  # one hyped stock, one dividend stock, one index fund
WINDOW_DAYS = 365
DEFAULT_MAX_PER_TICKER = 4000
DEFAULT_SEED = 42
READ_COLUMNS = ["created_at", "tickers", "text", "likes", "views"]  # nothing else is read from the copy
COLUMNS = ["post_id", "created_at", "tickers", "text", "likes", "views"]  # the replay file, in order
ID_PREFIX = "stephan:"

# The two masking rules. The quick check holds its own copy of both, character for character.
LINK = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
HANDLE = re.compile(r"(?<![A-Za-z0-9_])@[A-Za-z0-9_]+")


def mask(text):
    """Every link becomes "http", then every account name "@user"."""
    return HANDLE.sub("@user", LINK.sub("http", text))


def post_id(text):
    """stephan: and the first 8 hexadecimal digits of the SHA-256 of the text (UTF-8)."""
    return ID_PREFIX + hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def sha256_of(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shown(path):
    try:
        return str(Path(path).resolve().relative_to(REPO_DIR))
    except ValueError:
        return str(path)


def download_copy(copy):
    """Copy the pinned parquet file from Hugging Face byte for byte, through a temporary file and a rename."""
    if os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN") != "1":
        sys.exit("HF_HUB_DISABLE_IMPLICIT_TOKEN is not 1: run through scripts/make_replay_sample.sh, which sets up "
                 "the environment (no token, no telemetry) from scripts/env.sh")
    import fsspec  # noqa: F401  (huggingface_hub provides the hf:// protocol to fsspec)

    copy.parent.mkdir(parents=True, exist_ok=True)
    temporary = copy.with_name(f".{copy.name}.download")
    print(f"downloading {DATASET_ADDRESS} (about 119 MB) ...", flush=True)
    with fsspec.open(DATASET_ADDRESS, "rb") as source, temporary.open("wb") as target:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            target.write(chunk)
        target.flush()
        os.fsync(target.fileno())
    if copy.exists():  # another run finished first: keep its file
        temporary.unlink()
    else:
        os.replace(temporary, copy)
    print(f"saved a copy to {shown(copy)}", flush=True)


def week_start(times):
    """The Monday (UTC, 00:00) of the week of every time."""
    import pandas as pd

    days = times.dt.floor("D")
    return days - pd.to_timedelta(days.dt.weekday, unit="D")


def cap_ticker(table, ticker, cap, seed):
    """The row labels to keep of the posts naming ticker, at most cap, drawn evenly over the weeks."""
    import numpy as np
    import pandas as pd

    mine = table[table["tickers"].map(lambda names: ticker in names)]
    if len(mine) <= cap:
        return set(mine.index)
    weeks = week_start(mine["created_at"])
    sizes = weeks.value_counts().sort_index()
    quota = (sizes * cap / len(mine)).apply(np.floor).astype(int)
    rest = cap - int(quota.sum())
    remainders = (sizes * cap / len(mine)) - quota
    for week in remainders.sort_values(ascending=False, kind="stable").index[:rest]:
        quota[week] += 1
    random = np.random.default_rng(seed)
    kept = []
    for week, size in quota.items():
        rows = mine.index[weeks == week].to_numpy()
        kept += list(random.choice(rows, size=size, replace=False)) if size < len(rows) else list(rows)
    return set(pd.Index(kept))


def main():
    parser = argparse.ArgumentParser(description="Draw the replay file from StephanAkkerman/financial-tweets.")
    parser.add_argument("--output", default=OUTPUT, help=f"where to write the file (default {OUTPUT})")
    parser.add_argument("--max-per-ticker", type=int, default=DEFAULT_MAX_PER_TICKER,
                        help=f"most posts per ticker (default {DEFAULT_MAX_PER_TICKER})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"seed of the cap (default {DEFAULT_SEED})")
    args = parser.parse_args()
    if args.max_per_ticker < 1:
        sys.exit("--max-per-ticker must be a positive integer")

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    copy = REPO_DIR / DATASET_COPY
    if not copy.exists():
        download_copy(copy)
    print(f"local copy {shown(copy)}: {copy.stat().st_size:,} bytes, sha256 {sha256_of(copy)}")
    table = pq.read_table(copy, columns=READ_COLUMNS).to_pandas()  # index = row in the copy, from 0
    print(f"rows in the copy: {len(table):,}")

    latest = table["created_at"].max()
    first = latest - pd.Timedelta(days=WINDOW_DAYS)
    table = table[(table["created_at"] >= first) & (table["created_at"] <= latest)]
    print(f"window: {first.isoformat()} to {latest.isoformat()} (both included): {len(table):,} rows")
    table = table[table["tickers"].map(lambda names: names is not None and any(name in TICKERS for name in names))]
    print(f"naming {', '.join(TICKERS)}: {len(table):,} rows")

    table = table.assign(text=table["text"].fillna("").map(mask))
    with_handles = int(table["text"].str.contains("@user", regex=False).sum())
    with_links = int(table["text"].str.contains("http", regex=False).sum())
    table = table[table["text"].str.strip() != ""]
    print(f"not empty after masking: {len(table):,} rows ({with_handles:,} with an account name, {with_links:,} with "
          "a link, both masked)")
    table = table.assign(row=table.index).sort_values(["created_at", "row"], kind="stable")
    before = len(table)
    table = table[~table["text"].duplicated(keep="first")]
    print(f"one post per text: {len(table):,} rows ({before - len(table):,} repeats left out)")

    keep = set()
    for ticker in TICKERS:
        keep |= cap_ticker(table, ticker, args.max_per_ticker, args.seed)
    capped = len(table) - len(keep)
    table = table.loc[sorted(keep, key=lambda label: (table.at[label, "created_at"], label))]
    print(f"cap of {args.max_per_ticker:,} per ticker: {capped:,} rows left out")

    ids = table["text"].map(post_id)
    if ids.duplicated().any():
        sys.exit(f"two texts share the post_id {ids[ids.duplicated()].iloc[0]}: stop (the ids would not be unique)")
    out = pd.DataFrame({
        "post_id": ids.to_numpy(),
        "created_at": table["created_at"].to_numpy(),
        "tickers": [list(names) for names in table["tickers"]],
        "text": table["text"].to_numpy(),
        "likes": table["likes"].astype("Int64").to_numpy(),
        "views": table["views"].astype("Int64").to_numpy(),
    })
    schema = pa.schema([("post_id", pa.string()), ("created_at", pa.timestamp("us", tz="UTC")),
                        ("tickers", pa.list_(pa.string())), ("text", pa.string()), ("likes", pa.int64()),
                        ("views", pa.int64())])
    arrow = pa.Table.from_pandas(out, schema=schema, preserve_index=False)
    output = Path(args.output)
    output = output if output.is_absolute() else REPO_DIR / output
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(arrow, output, compression="zstd")
    per_ticker = {ticker: int(out["tickers"].map(lambda names: ticker in names).sum()) for ticker in TICKERS}
    print(f"wrote {shown(output)}: {len(out):,} posts, {output.stat().st_size:,} bytes")
    print(f"posts per ticker (a post counts for every ticker it names): {per_ticker}")
    print(f"first post {out['created_at'].min().isoformat()}, last post {out['created_at'].max().isoformat()}")
    print(f"fingerprint of the ids in file order: {hashlib.sha256(chr(10).join(out['post_id']).encode()).hexdigest()}")


if __name__ == "__main__":
    main()
