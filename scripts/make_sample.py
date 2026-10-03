"""Draw the small sample that the quick check runs on.

Reads the downloaded dataset (data/raw/st-data-full.xlsx, see
scripts/download_data.sh) and writes samples/stocktwits-crypto-sample.xlsx:
the same two sheets and the same two columns (text, label) as the original
file, so the notebook reads it with the same code.

Per class (0 bearish, 1 neutral, 2 bullish), with a fixed seed:
  80 posts that mention bitcoin or btc,
  20 other posts that contain emoji,
  20 other posts that contain a negation,
  80 other posts with none of the three.
Rows keep their original sheet and their original order inside the sheet.
Posts that are empty, start with "=" (a spreadsheet would read them as a
formula) or contain characters a spreadsheet cannot store are skipped.

Usage: .venv/bin/python scripts/make_sample.py   (after sourcing scripts/env.sh)
"""

import re
import sys
from pathlib import Path

import emoji
import pandas as pd
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

REPO_DIR = Path(__file__).resolve().parent.parent
SOURCE = REPO_DIR / "data" / "raw" / "st-data-full.xlsx"
TARGET = REPO_DIR / "samples" / "stocktwits-crypto-sample.xlsx"
SEED = 42
PER_CLASS = {"btc": 80, "emoji": 20, "negation": 20, "other": 80}  # posts drawn per class from each group

BTC_PATTERN = re.compile(r"\b(?:bitcoin|btc)\b", flags=re.IGNORECASE)  # same rule as the notebook
NEGATION_PATTERN = re.compile(r"\b(?:not|no|never|nor)\b|n['’]t\b|\b(?:dont|cant|wont|isnt|didnt|doesnt)\b",
                              flags=re.IGNORECASE)


def main() -> int:
    sheets = pd.read_excel(SOURCE, sheet_name=None, dtype={"text": str})
    frames = []
    for sheet_name, sheet in sheets.items():
        frames.append(pd.DataFrame({
            "sheet": sheet_name,
            "row": range(2, len(sheet) + 2),  # spreadsheet row number (row 1 holds the column names)
            "text": sheet["text"].fillna("").astype(str),
            "label": sheet["label"].astype("int64"),
        }))
    posts = pd.concat(frames, ignore_index=True)

    usable = (posts["text"].str.strip() != "") & ~posts["text"].str.startswith("=") \
        & ~posts["text"].map(lambda text: bool(ILLEGAL_CHARACTERS_RE.search(text)))
    posts = posts[usable]
    btc = posts["text"].str.contains(BTC_PATTERN)
    has_emoji = posts["text"].map(emoji.emoji_count) > 0
    negation = posts["text"].str.contains(NEGATION_PATTERN)
    groups = {
        "btc": btc,
        "emoji": ~btc & has_emoji,
        "negation": ~btc & ~has_emoji & negation,
        "other": ~btc & ~has_emoji & ~negation,
    }

    picked = []
    for label in (0, 1, 2):
        in_class = posts["label"] == label
        for group, mask in groups.items():
            picked.append(posts[in_class & mask].sample(n=PER_CLASS[group], random_state=SEED))
    sample = pd.concat(picked)
    if sample.duplicated(["sheet", "row"]).any():
        raise SystemExit("a post was drawn twice")

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(TARGET, engine="openpyxl") as writer:
        for sheet_name in sheets:  # same sheets, same order as the original file
            part = sample[sample["sheet"] == sheet_name].sort_values("row")
            part[["text", "label"]].to_excel(writer, sheet_name=sheet_name, index=False)

    # Read the file back the way the notebook does and compare.
    back = pd.concat(pd.read_excel(TARGET, sheet_name=None, dtype={"text": str}).values(), ignore_index=True)
    expected = pd.concat([sample[sample["sheet"] == name].sort_values("row") for name in sheets], ignore_index=True)
    if not (back["text"].fillna("").tolist() == expected["text"].tolist()
            and back["label"].tolist() == expected["label"].tolist()):
        raise SystemExit("the written sample does not read back identically")

    print(f"wrote {TARGET.relative_to(REPO_DIR)}: {len(sample)} posts")
    print("per sheet:", sample["sheet"].value_counts().sort_index().to_dict())
    print("per class:", sample["label"].value_counts().sort_index().to_dict())
    print("mention bitcoin/btc per class:", sample[sample["text"].str.contains(BTC_PATTERN)]["label"].value_counts().sort_index().to_dict())
    print("with emoji:", int(sample["text"].map(emoji.emoji_count).gt(0).sum()),
          "with a negation:", int(sample["text"].str.contains(NEGATION_PATTERN).sum()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
