"""Write the two small files that chapter 15 reads in the quick check.

    samples/btc-usd-made-up-closes.csv
        MADE-UP daily closes, not real prices: a random walk that starts at
        100 and moves every day by a factor exp(g), g drawn from a normal
        distribution with mean 0 and standard deviation 0.03 (random.Random
        with SEED). Real prices are not committed, because none of the three
        price sources says that its data may be passed on. Columns date and
        close; the first lines start with "#" and say that the prices are
        made up.

    samples/fear-greed-alternative-me-sample.json
        A real part of the Crypto Fear and Greed Index by alternative.me
        (https://alternative.me/crypto/fear-and-greed-index/), in the shape of
        its reply (name, data with value, value_classification and timestamp,
        newest first, and metadata), plus a field "attribution" that names
        the source. alternative.me allows the use of its data when it is
        credited next to the data.

Both cover the days of the dated samples' periods (WINDOW_START to WINDOW_END;
the price one day more, for the last day's change).

Usage: scripts/make_validation_samples.sh [--fear-greed-json PATH]
  Without --fear-greed-json the index is read once from
  https://api.alternative.me/fng/?limit=0&format=json (no key).
"""

import argparse
import json
import math
import random
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
WINDOW_START = date(2022, 4, 15)  # first day of the dated samples' periods
WINDOW_END = date(2022, 10, 10)  # last day of the dated samples' periods (the tweets' latest added day)
SEED = 42
PRICE_FILE = REPO_DIR / "samples" / "btc-usd-made-up-closes.csv"
INDEX_FILE = REPO_DIR / "samples" / "fear-greed-alternative-me-sample.json"
INDEX_ADDRESS = "https://api.alternative.me/fng/?limit=0&format=json"
ATTRIBUTION = "Crypto Fear and Greed Index, data from alternative.me (https://alternative.me/crypto/fear-and-greed-index/)"


def made_up_closes():
    """(day, close) for WINDOW_START to WINDOW_END + 1 day: a seeded random walk from 100."""
    generator, close, rows = random.Random(SEED), 100.0, []
    day = WINDOW_START
    while day <= WINDOW_END + timedelta(days=1):
        rows.append((day.isoformat(), round(close, 2)))
        close *= math.exp(generator.gauss(0.0, 0.03))
        day += timedelta(days=1)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fear-greed-json", help="a saved reply of the alternative.me address, instead of reading it")
    args = parser.parse_args()

    lines = ["# MADE-UP prices for the quick check only. These are NOT real BTC-USD prices.",
             f"# A random walk from 100 with daily factor exp(N(0, 0.03)), seed {SEED} (scripts/make_validation_samples.py).",
             "date,close"]
    lines += [f"{day},{close:.2f}" for day, close in made_up_closes()]
    PRICE_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.fear_greed_json:
        reply = json.loads(Path(args.fear_greed_json).read_text(encoding="utf-8"))
    else:
        request = urllib.request.Request(INDEX_ADDRESS, headers={"User-Agent": "ledger-eyes-samples (no key)"})
        with urllib.request.urlopen(request, timeout=30) as answer:
            reply = json.loads(answer.read())
    kept = []
    for item in reply["data"]:
        day = datetime.fromtimestamp(int(item["timestamp"]), timezone.utc).date()
        if WINDOW_START <= day <= WINDOW_END:
            kept.append({"value": item["value"], "value_classification": item["value_classification"],
                         "timestamp": item["timestamp"]})
    sample = {"name": reply.get("name", "Fear and Greed Index"), "attribution": ATTRIBUTION, "data": kept,
              "metadata": {"error": None}}
    INDEX_FILE.write_text(json.dumps(sample, indent=1) + "\n", encoding="utf-8")
    print(f"{PRICE_FILE.relative_to(REPO_DIR)}: {len(lines) - 3} made-up closes; "
          f"{INDEX_FILE.relative_to(REPO_DIR)}: {len(kept)} days of the index")


if __name__ == "__main__":
    main()
