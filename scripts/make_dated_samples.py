"""Draw the two small dated samples that the quick check runs on.

Reads the downloaded dated datasets (data/raw/dlt-tweets/data/ and
data/raw/dlt-news/data/, see scripts/download_data.sh --dated) one file at a
time, in batches, and writes

    samples/dlt-tweets-sample.parquet   (columns and types of the tweet dataset)
    samples/dlt-news-sample.parquet     (columns and types of the news dataset)

so that chapter 12 reads them with the same code as the full datasets.

Both samples cover the same window of UTC days, WINDOW_START to WINDOW_END
(62 days around the crash of May 2022). With a fixed seed, every row gets a
draw key (the SHA-256 of "<seed>:<time text><unit separator><text>") and the
rows with the smallest keys are taken:

  tweets: per day 40 that mention bitcoin or btc (120 on the BUSY_DAYS, so
          that some days have more posts than the quick check's daily cap)
          and 10 that do not; 6 tweets dated before 2021-01-01 (the default
          switches must drop them); and, for every way of writing the time
          that the full dataset uses, at least 5 tweets (taken from the rest
          of 2022 when the window has fewer).
  news:   per day 6 whose headline mentions bitcoin or btc and 3 whose
          headline does not, and up to 10 rows of the window whose source
          address is empty.

The day, the Bitcoin rule and the replacements are those of the notebook:
parse_times and mask_texts are executed from notebook/ledger_eyes.py, and
the Bitcoin rule is chapter 1's. Before a text is saved, its links become
"http" and its account names "@user" (tweet; news title, description and
text); every other value is kept as it is. The data is published under
CC BY-NC 4.0 (see README.md).

Usage: scripts/make_dated_samples.sh
"""

import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

import jupytext
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_DIR = Path(__file__).resolve().parent.parent
NOTEBOOK_SOURCE = REPO_DIR / "notebook" / "ledger_eyes.py"
TWEETS = REPO_DIR / "data" / "raw" / "dlt-tweets" / "data"
NEWS = REPO_DIR / "data" / "raw" / "dlt-news" / "data"
TWEETS_SAMPLE = REPO_DIR / "samples" / "dlt-tweets-sample.parquet"
NEWS_SAMPLE = REPO_DIR / "samples" / "dlt-news-sample.parquet"
SEED = 42
WINDOW_START, WINDOW_END = "2022-04-15", "2022-06-15"  # 62 UTC days, May 2022 inside
BUSY_DAYS = {"2022-05-09", "2022-05-10", "2022-05-11", "2022-05-12"}
TWEETS_PER_DAY = {"btc": 40, "other": 10}
TWEETS_PER_BUSY_DAY = {"btc": 120, "other": 10}
TWEETS_BEFORE_2021 = 6
TWEETS_PER_FORM = 5
NEWS_PER_DAY = {"btc": 6, "other": 3}
NEWS_EMPTY_SOURCE = 10
BATCH_ROWS = 200_000
OUTSIDE_YEAR = "2022"  # rows for a missing way of writing the time come from the rest of this year
POSITION_BASE = 10 ** 9  # position = file number * POSITION_BASE + row number in the file
BTC_PATTERN = re.compile(r"\b(?:bitcoin|btc)\b", flags=re.IGNORECASE)  # chapter 1's rule


def notebook_functions():
    """parse_times and mask_texts, executed from the notebook source (with the cells they need)."""
    namespace = {"__name__": "notebook_cells", "pd": pd, "re": re}
    cells = [cell.source for cell in jupytext.read(NOTEBOOK_SOURCE, fmt="py:percent").cells if cell.cell_type == "code"]
    for name in ("MASK_LINK_PATTERN", "TIME_KINDS"):
        source = next(cell for cell in cells if re.search(rf"^{name} = ", cell, flags=re.MULTILINE))
        exec(compile(source, f"cell defining {name}", "exec"), namespace)
    return namespace["parse_times"], namespace["mask_texts"]


def draw_keys(times, texts):
    return [hashlib.sha256(f"{SEED}:{time}\x1f{text}".encode("utf-8")).hexdigest() for time, text in zip(times, texts)]


def smallest(frame, groups, limits):
    """The rows with the smallest keys in every group, at most limits(group) each."""
    ordered = frame.sort_values(["key", "position"], kind="stable")
    return pd.concat([rows.head(limits(group)) for group, rows in ordered.groupby(groups, sort=True)]
                     or [frame.iloc[0:0]])


def form_of(texts):
    """How a time is written: every digit replaced by d."""
    return texts.fillna("").astype(str).str.replace(r"[0-9]", "d", regex=True)


def draw_tweets(parse_times, mask_texts):
    files = sorted(TWEETS.glob("*.parquet"))
    if not files:
        raise SystemExit(f"{TWEETS.relative_to(REPO_DIR)} holds no .parquet file: run scripts/download_data.sh --dated")
    window, early, forms = [], [], []
    all_forms = Counter()
    for number, path in enumerate(files):
        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_ROWS, columns=["timestamp", "tweet"]):
            frame = batch.to_pandas()
            frame["position"] = [number * POSITION_BASE + offset + row for row in range(len(frame))]  # file and row
            offset += len(frame)
            kind, _, day = parse_times(frame["timestamp"])
            text = mask_texts(frame["tweet"])
            frame = frame.assign(day=day, form=form_of(frame["timestamp"]), btc=text.str.contains(BTC_PATTERN),
                                 key=draw_keys(frame["timestamp"], text))
            all_forms.update(frame["form"].value_counts().to_dict())
            frame = frame[kind != "unparseable"]
            inside = frame[(frame["day"] >= WINDOW_START) & (frame["day"] <= WINDOW_END)]
            inside = inside.assign(group=inside["btc"].map({True: "btc", False: "other"}))
            window.append(smallest(inside, ["day", "group"], lambda group: (
                TWEETS_PER_BUSY_DAY if group[0] in BUSY_DAYS else TWEETS_PER_DAY)[group[1]]))
            early.append(smallest(frame[frame["day"] < "2021-01-01"].assign(group="early"), ["group"],
                                  lambda group: TWEETS_BEFORE_2021))
            outside = frame[frame["day"].str.startswith(OUTSIDE_YEAR)
                            & ((frame["day"] < WINDOW_START) | (frame["day"] > WINDOW_END))]
            forms.append(smallest(outside, ["form"], lambda group: TWEETS_PER_FORM))
        print(f"  {path.name}: {offset:,} rows", flush=True)
    window = pd.concat(window)
    window = smallest(window, ["day", "group"], lambda group: (
        TWEETS_PER_BUSY_DAY if group[0] in BUSY_DAYS else TWEETS_PER_DAY)[group[1]])
    early = smallest(pd.concat(early), ["group"], lambda group: TWEETS_BEFORE_2021)
    chosen = pd.concat([window, early])
    forms = smallest(pd.concat(forms), ["form"], lambda group: TWEETS_PER_FORM)
    have = chosen["form"].value_counts()
    added = forms[forms["form"].map(lambda form: have.get(form, 0) < TWEETS_PER_FORM)]
    added = smallest(added, ["form"], lambda form: TWEETS_PER_FORM - have.get(form[0] if isinstance(form, tuple) else form, 0))
    chosen = pd.concat([chosen, added]).drop_duplicates("position")
    print(f"  forms in the full dataset: {dict(all_forms)}")
    print(f"  added from outside the window for missing forms: {added['form'].value_counts().to_dict()}")

    # Second pass: the chosen rows with every column, in file order.
    wanted = {}
    for position in chosen["position"]:
        wanted.setdefault(position // POSITION_BASE, set()).add(position % POSITION_BASE)
    pieces = []
    schema = None
    for number, path in enumerate(files):
        parquet = pq.ParquetFile(path)
        schema = parquet.schema_arrow
        offset = 0
        for batch in parquet.iter_batches(batch_size=BATCH_ROWS):
            rows = sorted(row - offset for row in wanted.get(number, ()) if offset <= row < offset + batch.num_rows)
            if rows:
                pieces.append(pa.Table.from_batches([batch]).take(rows))
            offset += batch.num_rows
    table = pa.concat_tables(pieces)
    masked = mask_texts(pd.Series(table.column("tweet").to_pylist(), dtype=object))
    table = table.set_column(table.schema.get_field_index("tweet"), schema.field("tweet"),
                             pa.array(masked.tolist(), type=schema.field("tweet").type))
    return table.replace_schema_metadata(schema.metadata), chosen


def draw_news(parse_times, mask_texts):
    files = sorted(NEWS.glob("*.parquet"))
    if not files:
        raise SystemExit(f"{NEWS.relative_to(REPO_DIR)} holds no .parquet file: run scripts/download_data.sh --dated")
    table = pa.concat_tables([pq.read_table(path) for path in files])
    metadata = pq.ParquetFile(files[0]).schema_arrow.metadata
    frame = table.select(["timestamp", "title", "source_url"]).to_pandas()
    frame["position"] = range(len(frame))
    _, _, day = parse_times(frame["timestamp"])
    title = mask_texts(frame["title"])
    frame = frame.assign(day=day, btc=title.str.contains(BTC_PATTERN), key=draw_keys(frame["timestamp"], title))
    inside = frame[(frame["day"] >= WINDOW_START) & (frame["day"] <= WINDOW_END)]
    inside = inside.assign(group=inside["btc"].map({True: "btc", False: "other"}))
    chosen = smallest(inside, ["day", "group"], lambda group: NEWS_PER_DAY[group[1]])
    empty = inside[inside["source_url"].fillna("").str.strip() == ""].assign(group="empty_source")
    chosen = pd.concat([chosen, smallest(empty, ["group"], lambda group: NEWS_EMPTY_SOURCE)]).drop_duplicates("position")
    table = table.take(sorted(chosen["position"]))
    for column in ["title", "description", "text"]:
        masked = mask_texts(pd.Series(table.column(column).to_pylist(), dtype=object))
        index = table.schema.get_field_index(column)
        table = table.set_column(index, table.schema.field(column), pa.array(masked.tolist(), type=table.schema.field(column).type))
    return table.replace_schema_metadata(metadata), chosen


def describe(name, chosen, path):
    per_day = chosen[(chosen["day"] >= WINDOW_START) & (chosen["day"] <= WINDOW_END)]["day"].value_counts()
    print(f"{path.relative_to(REPO_DIR)}: {len(chosen):,} rows, days {chosen['day'].min()} to {chosen['day'].max()}, {path.stat().st_size:,} bytes; window days with rows "
          f"{len(per_day)} ({per_day.index.min()} to {per_day.index.max()}), rows per day min {per_day.min()}, median "
          f"{per_day.median():g}, max {per_day.max()}; groups {chosen['group'].value_counts().to_dict()}")


def main() -> int:
    parse_times, mask_texts = notebook_functions()
    print("tweets:")
    tweets, chosen_tweets = draw_tweets(parse_times, mask_texts)
    pq.write_table(tweets, TWEETS_SAMPLE)
    describe("tweets", chosen_tweets, TWEETS_SAMPLE)
    print(f"  time forms in the sample: {chosen_tweets['form'].value_counts().to_dict()}")
    news, chosen_news = draw_news(parse_times, mask_texts)
    pq.write_table(news, NEWS_SAMPLE)
    describe("news", chosen_news, NEWS_SAMPLE)
    print(f"  rows with an empty source address: {int((chosen_news['source_url'].fillna('').str.strip() == '').sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
