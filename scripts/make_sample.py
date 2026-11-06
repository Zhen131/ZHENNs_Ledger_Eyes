"""Draw the small sample that the quick check runs on.

Reads the local copy of the dataset (data/raw/tim-financial-tweets-bba60269.parquet,
see scripts/download_data.sh) and writes samples/tim-financial-tweets-sample.parquet
with three columns: tweet and sentiment exactly as in the dataset (sentiment in
the dataset's own coding, 0 neutral, 1 bullish, 2 bearish, not converted), and
row, the post's row in the full file (from 0), so that the notebook gives every
sampled post the same id, tim:<row>, as in the full file. The column url is
not copied. Rows are sorted by row.

Only posts that can go into a public repository are drawn: posts whose text is
empty after stripping, or that contain an account name (@ followed by a letter,
digit or underscore) or a web link (http://, https:// or www.), are never
drawn. The posts that are drawn keep their text exactly as in the dataset.

The draw, with a fixed seed (each step draws at random among the posts not yet
taken, and only as many as are still missing):
  1. 100 posts of each of the six levels of labels/levels-prelabel-v2.csv
     (assets_dividends, earnings, revenue_business, price_chart, story_macro,
     emotion_only), so a chapter on the levels has enough posts in the quick
     check;
  2. until every class (bearish, neutral, bullish, after the notebook's
     conversion) has 400 posts;
  3. until the bearish and the bullish class each have 150 posts with a ticker,
     so the run with TICKER_ONLY = True and two classes can be split;
  4. until 60 posts contain emoji and 60 contain a negation, so chapter 4 has
     examples of both.
Whether a post has a ticker is decided by the notebook's own cell (chapter 1),
executed from notebook/ledger_eyes.py, never by a second copy of the rule.

Usage: .venv/bin/python scripts/make_sample.py   (after sourcing scripts/env.sh)
"""

import contextlib
import csv
import hashlib
import io
import random
import re
import sys
from pathlib import Path

import emoji
import jupytext
import pandas as pd
import pyarrow.parquet as pq

REPO_DIR = Path(__file__).resolve().parent.parent
SOURCE = REPO_DIR / "data" / "raw" / "tim-financial-tweets-bba60269.parquet"
TARGET = REPO_DIR / "samples" / "tim-financial-tweets-sample.parquet"
LEVEL_FILE = REPO_DIR / "labels" / "levels-prelabel-v2.csv"
NOTEBOOK_SOURCE = REPO_DIR / "notebook" / "ledger_eyes.py"
SEED = 42
LEVELS = ["assets_dividends", "earnings", "revenue_business", "price_chart", "story_macro", "emotion_only"]
PER_LEVEL = 100  # posts per level (step 1)
PER_CLASS = 400  # posts per class at least (step 2)
TICKER_PER_SIDE = 150  # bearish and bullish posts with a ticker at least (step 3)
EMOJI_POSTS = 60  # posts with emoji at least (step 4)
NEGATION_POSTS = 60  # posts with a negation at least (step 4)
LABEL_OF_SENTIMENT = {0: 1, 1: 2, 2: 0}  # the dataset's sentiment -> the notebook's label (0 bearish, 1 neutral, 2 bullish)

HANDLE = re.compile(r"@[A-Za-z0-9_]")  # an account name
LINK = re.compile(r"https?://|www\.", flags=re.IGNORECASE)  # a web link
NEGATION = re.compile(r"\b(?:not|no|never|nor)\b|n['’]t\b", flags=re.IGNORECASE)


def notebook_has_ticker(texts):
    """Run the notebook's own ticker cell (chapter 1) on texts; returns a list of True/False."""
    notebook = jupytext.read(NOTEBOOK_SOURCE, fmt="py:percent")
    cells = [cell.source for cell in notebook.cells
             if cell.cell_type == "code" and re.search(r"^TICKER_PATTERN = ", cell.source, flags=re.MULTILINE)]
    if len(cells) != 1:
        raise SystemExit(f"expected one cell that defines TICKER_PATTERN in {NOTEBOOK_SOURCE}, found {len(cells)}")
    namespace = {"pd": pd, "df_all": pd.DataFrame({"text": list(texts)})}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(cells[0], "the notebook's ticker cell", "exec"), namespace)
    return namespace["df_all"]["has_ticker"].tolist()


def main() -> int:
    table = pq.read_table(SOURCE, columns=["tweet", "sentiment"]).to_pandas()
    table["row"] = range(len(table))
    text = table["tweet"].fillna("")
    eligible = (text.str.strip() != "") & ~text.str.contains(HANDLE) & ~text.str.contains(LINK)
    table["label"] = table["sentiment"].map(LABEL_OF_SENTIMENT)
    table["has_ticker"] = notebook_has_ticker(text)
    table["has_emoji"] = [emoji.emoji_count(value) > 0 for value in text]
    table["has_negation"] = text.str.contains(NEGATION)
    with LEVEL_FILE.open(encoding="utf-8", newline="") as handle:
        level_of_row = {int(row["post_id"].split(":", 1)[1]): row["level"] for row in csv.DictReader(handle)}
    table["level"] = table["row"].map(level_of_row)
    pool = table[eligible]
    rng = random.Random(SEED)
    taken = set()

    def take(candidates, missing):
        """Add `missing` posts drawn at random from candidates (rows not yet taken), in row order before the draw."""
        rows = sorted(set(candidates) - taken)
        chosen = rng.sample(rows, min(max(missing, 0), len(rows)))
        taken.update(chosen)

    for level in LEVELS:  # step 1
        take(pool.loc[pool["level"] == level, "row"], PER_LEVEL)
    for label in (0, 1, 2):  # step 2
        have = int((table.loc[sorted(taken), "label"] == label).sum())
        take(pool.loc[pool["label"] == label, "row"], PER_CLASS - have)
    for label in (0, 2):  # step 3
        picked = table.loc[sorted(taken)]
        have = int(((picked["label"] == label) & picked["has_ticker"]).sum())
        take(pool.loc[(pool["label"] == label) & pool["has_ticker"], "row"], TICKER_PER_SIDE - have)
    for column, wanted in (("has_emoji", EMOJI_POSTS), ("has_negation", NEGATION_POSTS)):  # step 4
        have = int(table.loc[sorted(taken), column].sum())
        take(pool.loc[pool[column], "row"], wanted - have)

    sample = table.loc[sorted(taken)]
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    sample[["tweet", "sentiment", "row"]].to_parquet(TARGET, index=False)
    print(f"Wrote {TARGET.relative_to(REPO_DIR)}: {len(sample)} posts (seed {SEED})")
    print("  per class (0 bearish, 1 neutral, 2 bullish):", sample["label"].value_counts().sort_index().to_dict())
    print("  per level:", {level: int((sample["level"] == level).sum()) for level in LEVELS})
    print("  with a ticker per class:", sample[sample["has_ticker"]]["label"].value_counts().sort_index().to_dict())
    print(f"  with emoji: {int(sample['has_emoji'].sum())}, with a negation: {int(sample['has_negation'].sum())}")
    row_list = ",".join(map(str, sample["row"]))
    print(f"  rows {sample['row'].iloc[0]} to {sample['row'].iloc[-1]}; "
          f"SHA-256 of the comma-separated row list: {hashlib.sha256(row_list.encode()).hexdigest()[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
