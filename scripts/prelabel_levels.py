"""Pre-label stock tweets with the level of what they look at, through the local Ollama server.

The ladder goes from what is closest to cash to what is farthest from it: assets_dividends, earnings,
revenue_business, price_chart, story_macro, emotion_only; none is for posts that do not talk about stocks or markets
at all. It measures the appetite for risk, not whether a post is bullish or bearish.

Run it through scripts/prelabel_levels.sh, which sets up the environment from scripts/env.sh and uses the virtual
environment's Python:

    scripts/prelabel_levels.sh label [--sample] [options]
        Draw the posts, ask the model several times per post, append one row per post to a CSV file. Resumable:
        posts already in the file are skipped, so a stopped run continues where it stopped.
    scripts/prelabel_levels.sh show-selection [options]
        Read the dataset and draw the posts without contacting Ollama; print the pools, the keyword hits, the
        first ids of the processing order and the selection fingerprint (no text).
    scripts/prelabel_levels.sh review-template [options]
        Draw the review table (post id, pre-label, empty human label) from the full label file.
    scripts/prelabel_levels.sh export-review [options]
        Write the same review rows with the original texts, under data/ only.
    scripts/prelabel_levels.sh show-prompt
        Print the fixed prompt text, the answer format and their fingerprint.
    scripts/prelabel_levels.sh check-file <csv file> [--samples N] [--min-votes K] [--labels FILE]
        Check the format of a label file or a review table (standard library only; no dataset, no Ollama).

The labels come from a language model, not from people (silver labels). The label files hold only post ids, labels,
vote counts, the model name, the prompt version and an 8-digit fingerprint of each text; never any text.

Only the local Ollama server is contacted, on a loopback address, without any proxy, key or model download. The
dataset is read from Hugging Face at a pinned revision, without a token. Relative paths are relative to the
repository.
"""

import argparse
import csv
import hashlib
import http.client
import io
import json
import os
import random
import re
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- the dataset

DATASET = "TimKoornstra/financial-tweets-sentiment"  # Hugging Face dataset, MIT License
DATASET_REVISION = "bba602698fd6f18226f6404fd8fb637ef67fa1eb"  # pinned revision
DATASET_FILE = "data/train-00000-of-00001.parquet"
DATASET_ADDRESS = f"hf://datasets/{DATASET}@{DATASET_REVISION}/{DATASET_FILE}"
DATASET_COPY = f"data/raw/tim-financial-tweets-{DATASET_REVISION[:8]}.parquet"  # local copy, written once
DATASET_COLUMNS = ["tweet", "sentiment", "url"]  # must all be there; only "tweet" is used
TEXT_COLUMN = "tweet"
ID_PREFIX = "tim"  # post_id = tim:<row>, the row's position (from 0) in the parquet file
OFFICIAL_ENDPOINT = "https://huggingface.co"

# ---------------------------------------------------------------- defaults

DEFAULT_HOST = "http://127.0.0.1:11434"  # the local Ollama server
LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost")  # the only host names the script talks to
DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_COUNT = 8000  # posts drawn (the full run must use the default)
COUNT_RANGE = (6000, 10000)  # the range the default count must stay in
DEFAULT_POOL_CAP = 1500  # most posts taken from one keyword pool
DEFAULT_SEED = 42  # draw, processing order, review draw; the model's seed of sample i is seed + i
DEFAULT_SAMPLES = 5  # questions (samples) per post, each with its own seed
DEFAULT_TEMPERATURE = 0.7  # the model's temperature, so that the samples can differ
DEFAULT_MIN_VOTES = 3  # votes the most frequent label needs; must be more than half of the samples
DEFAULT_MAX_ATTEMPTS = 3  # attempts per sample before the sample counts as a failed vote
DEFAULT_TIMEOUT = 120.0  # seconds one request may take before it counts as a failed attempt
DEFAULT_CONNECT_RETRIES = 3  # extra tries when the server cannot be reached, before the run stops
DEFAULT_CONNECT_WAIT = 5.0  # seconds between those tries
NUM_PREDICT = 64  # most tokens of an answer; the longest answer, {"level": "revenue_business"}, needs about 10
DEFAULT_REVIEW_SIZE = 200  # rows of the review table
PROGRESS_EVERY = 25  # print a progress line after this many posts
RUN_LOG = "data/prelabel/levels-runs.jsonl"  # one line per run (no text); always under data/
SETTINGS_DIR = "data/prelabel"  # the settings record of every output file
REVIEW_EXPORT = "data/review/levels-review-with-text.csv"  # the review rows with the original texts

COLUMNS = ["post_id", "level", "votes", "model", "prompt_version", "text_hash"]  # columns of a label file, in order
REVIEW_COLUMNS = ["post_id", "prelabel", "human_label"]  # columns of the review table, in order
UNPARSED = "unparsed"  # no sample gave one of the labels; means "failed", not a level
DISPUTED = "disputed"  # the most frequent label has fewer votes than --min-votes; not a level
NONE = "none"  # not about stocks or markets at all; a label, but not a level
POST_ID_PATTERN = re.compile(r"tim:(0|[1-9][0-9]*)")
VOTES_PATTERN = re.compile(r"(0|[1-9][0-9]*)/(0|[1-9][0-9]*)")
HASH_PATTERN = re.compile(r"[0-9a-f]{8}")
BOM = b"\xef\xbb\xbf"

# ---------------------------------------------------------------- keywords
# Posts that seem to talk about the three rare levels are drawn first. A word or phrase matches when no letter or
# digit touches it on either side, in any case ("p/e" matches "P/E", "dividend" does not match "dividendo"). A post
# that matches several pools belongs to the first of them (closest to cash, as in the prompt's rule).

KEYWORD_POOLS = [
    ("assets_dividends", ["dividend", "dividends", "yield", "payout", "book value", "p/b", "price to book", "nav",
                          "balance sheet", "net asset", "net assets"]),
    ("earnings", ["p/e", "pe ratio", "earnings", "eps", "profit", "profits", "margin", "margins", "guidance",
                  "net income"]),
    ("revenue_business", ["revenue", "revenues", "sales", "p/s", "growth", "subscribers", "users", "deliveries",
                          "units"]),
]
RANDOM_STEP = "random"  # the step that fills the remaining places from every post not yet taken


def keyword_regex(word):
    return re.compile(r"(?<![a-z0-9])" + re.escape(word) + r"(?![a-z0-9])", flags=re.IGNORECASE)


KEYWORD_REGEXES = [(pool, [(word, keyword_regex(word)) for word in words]) for pool, words in KEYWORD_POOLS]


def keywords_fingerprint():
    """SHA-256 (first 16 hex digits) of the keyword list: the settings record notes which list drew the posts."""
    text = json.dumps(KEYWORD_POOLS, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("ascii")).hexdigest()[:16]


# ---------------------------------------------------------------- prompt, version v2
# Everything sent to the model except the post itself is fixed by the prompt version. The checks pin the
# fingerprint of version v2 (prompt_fingerprint); changing one character of the texts below, the examples or the
# answer format needs a new version, a new label file and a new pinned fingerprint. The prompt lives only here,
# never in a separate file, so that line endings cannot change it. Version v1 asked for the reason the writer gives;
# it was never used for a committed file and is gone. Version v2 asks what the post pays attention to.

PROMPT_VERSION = "v2"
LEVELS_V2 = [
    {"name": "assets_dividends",
     "meaning": "net assets and dividends: price to book (P/B), book value, trading below book value, net assets, "
                "the balance sheet, dividends, dividend yield, payouts"},
    {"name": "earnings",
     "meaning": "profits: price to earnings (P/E), earnings per share (EPS), profit, profit margins, net income, "
                "guidance, earnings that beat or miss expectations"},
    {"name": "revenue_business",
     "meaning": "revenue and the business: price to sales (P/S), revenue, revenue growth, sales, deliveries, units "
                "sold, users or subscribers, orders, products, how the business is going"},
    {"name": "price_chart",
     "meaning": "the chart: the price itself or a price level, its trend, breakouts, support, resistance, moving "
                "averages, RSI, volume, turnover, candlestick patterns"},
    {"name": "story_macro",
     "meaning": "a story: news, analyst ratings and price targets, policy, interest rates, inflation, big industry "
                "trends, narratives such as a new technology that will change the world"},
    {"name": "emotion_only",
     "meaning": "emotion only: the post talks about a stock or the market but only takes a position, shows a "
                "feeling or waits, without anything of levels 1 to 5, such as to the moon, it is over, just buy, "
                "holding on"},
    {"name": "none",
     "meaning": "not about stocks or markets at all: small talk, adverts for airdrops or giveaways, only mentions "
                "of people and links, or a post that cannot be read"},
]
LEVEL_NAMES = [item["name"] for item in LEVELS_V2]  # the seven labels, closest to cash first, none last
SIX_LEVELS = LEVEL_NAMES[:6]  # the levels of the ladder (none is not a level)
# Hand-written examples, at least one per label; none of them is a post of the dataset (checked before labelling).
EXAMPLES_V2 = [
    {"post": "$KLMN pays a 5% dividend and trades below book value, I keep adding on dips", "level": "assets_dividends"},
    {"post": "balance sheet has more cash than the whole market cap, net assets alone justify the price",
     "level": "assets_dividends"},
    {"post": "$QRST eps beat by 12 cents and margins widened again this quarter", "level": "earnings"},
    {"post": "trading at 8 times earnings with profits still growing, way too cheap", "level": "earnings"},
    {"post": "$VWXY deliveries up 40% this quarter and they just opened orders in Europe",
     "level": "revenue_business"},
    {"post": "$VWXY revenue grew 30% and the new chip will change the whole industry, this is the future",
     "level": "revenue_business"},
    {"post": "$PLMO broke out above the 200 day moving average on heavy volume, next resistance at 52",
     "level": "price_chart"},
    {"post": "rates are about to come down and inflation is cooling, banks will love this", "level": "story_macro"},
    {"post": "analyst upgrades $ZRTX to buy with a 150 target after the merger news", "level": "story_macro"},
    {"post": "$PLMO to the moon, loading up more tomorrow", "level": "emotion_only"},
    {"post": "sold everything, this market is finished, never again", "level": "emotion_only"},
    {"post": "@marketfriend haha see you at lunch", "level": "none"},
    {"post": "who is watching the game tonight?", "level": "none"},
    {"post": "$HJKL q3 earnings call tomorrow after the close", "level": "earnings"},
    {"post": "$NBVC patience, just waiting for it to get back to .15", "level": "price_chart"},
    {"post": "free airdrop for the first 500 wallets, follow and retweet to join #giveaway", "level": "none"},
]
USER_PREFIX = "Post:\n"  # put in front of every post (and every example post) in the user message
POST_PLACEHOLDER = "<post>"  # stands for the post when the fingerprint is computed


def system_prompt():
    """The system message: what the posts are, the ladder with its labels and meanings, and the rules."""
    ladder = "\n".join(f"{number}. {item['name']}: {item['meaning']}."
                       for number, item in enumerate(LEVELS_V2[:6], start=1))
    other = LEVELS_V2[6]
    return (
        "You read short posts from Twitter about stocks and markets. Each user message is one post.\n\n"
        "Decide what the post pays attention to when it talks about stocks or markets, and how close to cash that "
        "is. The post does not need to take a position or give a reason: a headline, an earnings calendar or an "
        "analyst rating is labelled by what it talks about (an earnings call or an earnings date is earnings, a "
        "rating or a price target is story_macro, a net asset value announcement is assets_dividends). You do not "
        "judge whether the writer is bullish or bearish: a bullish post and a bearish post can stand on the same "
        "level.\n\n"
        "The levels, from what is closest to cash (1) to what is farthest from it (6):\n\n"
        f"{ladder}\n\n"
        "And one label that is not a level:\n\n"
        f"{other['name']}: {other['meaning']}.\n\n"
        "Rules:\n"
        "1. Choose exactly one label.\n"
        "2. If the post touches two or more levels, choose the level closest to cash, the one with the "
        "lowest number. For example, a post about revenue and about a story is revenue_business.\n"
        "3. If the post talks about a stock or the market but only takes a position, shows a feeling or waits, "
        "without anything of levels 1 to 5, choose emotion_only.\n"
        f"4. If the post does not talk about stocks or markets at all, or cannot be read, choose {NONE}.\n"
        '5. Answer with JSON only, in this form: {"level": "<label>"}'
    )


def retry_reminder():
    """The fixed sentence added to the system message when a sample is asked again."""
    return ('Your previous answer could not be used. Answer with JSON only, exactly in the form '
            f'{{"level": "<label>"}}, where <label> is one of: {", ".join(LEVEL_NAMES)}.')


def answer_format():
    """JSON Schema for Ollama's structured output: an object whose only field is one of the seven labels."""
    return {"type": "object",
            "properties": {"level": {"type": "string", "enum": list(LEVEL_NAMES)}},
            "required": ["level"]}


def build_payload(text, *, retry, model, seed, temperature):
    """The request body for Ollama's /api/chat: fixed messages, the examples, then the post; no dataset label."""
    system = system_prompt()
    if retry:
        system += "\n\n" + retry_reminder()
    messages = [{"role": "system", "content": system}]
    for example in EXAMPLES_V2:
        messages.append({"role": "user", "content": USER_PREFIX + example["post"]})
        messages.append({"role": "assistant", "content": json.dumps({"level": example["level"]})})
    messages.append({"role": "user", "content": USER_PREFIX + text})
    return {"model": model, "messages": messages, "stream": False, "format": answer_format(),
            "options": {"temperature": temperature, "seed": seed, "num_predict": NUM_PREDICT}}


def prompt_fingerprint():
    """SHA-256 (hex) of everything fixed that is sent: the messages of a first request and of a retry, with the
    placeholder in place of the post, and the answer format."""
    first = build_payload(POST_PLACEHOLDER, retry=False, model="", seed=0, temperature=0)
    again = build_payload(POST_PLACEHOLDER, retry=True, model="", seed=0, temperature=0)
    text = json.dumps({"first": first["messages"], "retry": again["messages"], "format": first["format"]},
                      ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("ascii")).hexdigest()


class Stop(Exception):
    """A problem that stops the script with a message (nothing is guessed or overwritten)."""


def default_output(kind, version=PROMPT_VERSION):
    """Default file paths; the prompt version is part of the full file's name."""
    if kind == "full":
        return f"labels/levels-prelabel-{version}.csv"
    suffix = "" if version == PROMPT_VERSION else f"-{version}"
    if kind == "sample":
        return f"samples/levels-prelabel-sample{suffix}.csv"
    return f"labels/levels-review-template{suffix}.csv"


def repo_path(path):
    """A path given on the command line: absolute as given, relative to the repository otherwise."""
    path = Path(path)
    return path if path.is_absolute() else REPO_DIR / path


def shown(path):
    """A path for messages: relative to the repository when inside it."""
    try:
        return Path(path).resolve().relative_to(REPO_DIR.resolve()).as_posix()
    except ValueError:
        return str(path)


# ---------------------------------------------------------------- reading the dataset

def check_environment():
    """The commands that read the dataset refuse a mirror and need the settings of scripts/env.sh (no token)."""
    endpoint = os.environ.get("HF_ENDPOINT", "")
    if endpoint and endpoint.rstrip("/") != OFFICIAL_ENDPOINT:
        raise Stop(f"HF_ENDPOINT is set to {endpoint!r}; this script reads the dataset only from {OFFICIAL_ENDPOINT}. "
                   "Unset HF_ENDPOINT and run again.")
    if os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN") != "1":
        raise Stop("HF_HUB_DISABLE_IMPLICIT_TOKEN is not 1: please run through scripts/prelabel_levels.sh, which "
                   "sets up the environment (no token, no telemetry) from scripts/env.sh")


def download_copy(copy):
    """Copy the pinned parquet file from Hugging Face byte for byte to copy, through a temporary file and a rename,
    so that an interrupted download never leaves half a file."""
    import fsspec  # noqa: F401  (huggingface_hub provides the hf:// protocol to fsspec)

    copy.parent.mkdir(parents=True, exist_ok=True)
    temporary = copy.with_name(f".{copy.name}.download")
    print(f"downloading {DATASET_ADDRESS} ...", flush=True)
    with fsspec.open(DATASET_ADDRESS, "rb") as source, temporary.open("wb") as target:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            target.write(chunk)
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, copy)
    print(f"saved a copy to {shown(copy)}", flush=True)


def load_table(source=None):
    """The dataset as a table with its rows in file order (index 0, 1, ...). source None: the local copy of the
    pinned revision, downloaded first if it is not there; otherwise a local parquet file. Returns (table, where)."""
    if source is None:
        path = REPO_DIR / DATASET_COPY
        if not path.is_file():
            download_copy(path)
        where = f"{DATASET}@{DATASET_REVISION} ({shown(path)})"
    else:
        path = repo_path(source)
        if not path.is_file():
            raise Stop(f"{source} not found")
        where = shown(path)
    try:
        table = pd.read_parquet(path)
    except Exception as exc:  # a broken or partial file
        if source is None:
            raise Stop(f"cannot read {shown(path)} ({type(exc).__name__}: {exc}); delete {shown(path)} and run "
                       "again to download it anew") from exc
        raise Stop(f"cannot read {source}: {type(exc).__name__}: {exc}") from exc
    missing = [name for name in DATASET_COLUMNS if name not in table.columns]
    if missing:
        raise Stop(f"{where} lacks the columns {missing}; columns found: {list(table.columns)}")
    return table.reset_index(drop=True), where


def posts_from_table(table):
    """[{post_id, text}, ...] in row order: post_id tim:<row> (the position from 0), text the tweet as it is (an
    empty value becomes ""). Only these two go on: the dataset's sentiment and url are never passed to the draw or
    to the model."""
    texts = table[TEXT_COLUMN].reset_index(drop=True)
    return [{"post_id": f"{ID_PREFIX}:{row}", "text": "" if pd.isna(text) else str(text)}
            for row, text in enumerate(texts.tolist())]


def row_of(post_id):
    match = POST_ID_PATTERN.fullmatch(post_id)
    if not match:
        raise Stop(f"post id {post_id!r} is not {ID_PREFIX}:<row>")
    return int(match.group(1))


def text_hash(text):
    """First 8 hexadecimal digits of the SHA-256 of the text (UTF-8): identifies the text, cannot give it back."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def selection_fingerprint(ids):
    """SHA-256 (hex) of the ids in processing order, joined by newlines: the same on every machine that draws the
    same posts in the same order."""
    return hashlib.sha256("\n".join(ids).encode("ascii")).hexdigest()


def first_pool(text):
    """The first keyword pool that matches the text, or None."""
    for pool, words in KEYWORD_REGEXES:
        if any(regex.search(text) for _, regex in words):
            return pool
    return None


def select_posts(posts, count=DEFAULT_COUNT, pool_cap=DEFAULT_POOL_CAP, seed=DEFAULT_SEED):
    """Draw the posts to label and put them in processing order.

    posts: [{post_id, text}, ...] in any order; they are sorted by row first, so the draw depends only on the rows.
    1. Leave out texts that are empty after stripping, and repeated texts (the same after stripping, case counts),
       keeping the smallest row.
    2. Every remaining post belongs to the first keyword pool it matches (assets_dividends, earnings,
       revenue_business) or to none. Every pool is taken whole, or drawn down to pool_cap when it is larger.
    3. The remaining places up to count are drawn from every post not taken yet (pool posts above the cap
       included). When the pools alone hold more than count posts, the draw stops instead of cutting.
    4. All the posts taken are shuffled: that is the processing order.
    All randomness comes from one random.Random(seed), used in this order: the pools above the cap (pool order),
    the fill, the shuffle; never from numpy or pandas, so that every machine draws the same posts.
    Returns (selection [{post_id, text}, ...] in processing order, summary, step_by_id)."""
    if count < 1 or pool_cap < 0:
        raise Stop("the count must be at least 1 and the pool cap at least 0")
    ordered = sorted(posts, key=lambda post: row_of(post["post_id"]))
    empty = [post for post in ordered if post["text"].strip() == ""]
    seen, unique, repeated = set(), [], 0
    for post in ordered:
        key = post["text"].strip()
        if key == "":
            continue
        if key in seen:
            repeated += 1
            continue
        seen.add(key)
        unique.append(post)
    keyword_hits = {word: 0 for _, words in KEYWORD_POOLS for word in words}
    for post in unique:
        for _, words in KEYWORD_REGEXES:
            for word, regex in words:
                if regex.search(post["text"]):
                    keyword_hits[word] += 1
    members = {pool: [] for pool, _ in KEYWORD_POOLS}
    for post in unique:
        pool = first_pool(post["text"])
        if pool is not None:
            members[pool].append(post)
    rng = random.Random(seed)
    taken, step_by_id = [], {}
    for pool, _ in KEYWORD_POOLS:
        chosen = members[pool] if len(members[pool]) <= pool_cap else rng.sample(members[pool], pool_cap)
        taken += chosen
        step_by_id.update((post["post_id"], pool) for post in chosen)
    if len(taken) > count:
        raise Stop(f"the keyword pools alone hold {len(taken)} posts, more than the count {count}; raise the count "
                   "or lower the pool cap")
    rest = [post for post in unique if post["post_id"] not in step_by_id]
    fill_wanted = count - len(taken)
    fill = rest if fill_wanted >= len(rest) else rng.sample(rest, fill_wanted)
    step_by_id.update((post["post_id"], RANDOM_STEP) for post in fill)
    chosen = taken + fill
    rng.shuffle(chosen)
    selection = [{"post_id": post["post_id"], "text": post["text"]} for post in chosen]
    steps = Counter(step_by_id.values())
    summary = {
        "posts_in_source": len(posts), "empty_texts_left_out": len(empty), "repeated_texts_left_out": repeated,
        "usable_posts": len(unique), "count": count, "pool_cap": pool_cap, "seed": seed,
        "pool_posts": {pool: len(members[pool]) for pool, _ in KEYWORD_POOLS},
        "taken_per_step": {name: steps.get(name, 0) for name in [pool for pool, _ in KEYWORD_POOLS] + [RANDOM_STEP]},
        "fill_wanted": fill_wanted, "selected": len(selection), "keyword_hits": keyword_hits,
        "first_ids": [post["post_id"] for post in selection[:5]],
        "fingerprint": selection_fingerprint([post["post_id"] for post in selection]),
    }
    return selection, summary, step_by_id


def examples_in_dataset(texts):
    """The hand-written examples whose text (stripped, lower case) is also a post of the dataset."""
    known = {text.strip().lower() for text in texts}
    return sorted(example["post"] for example in EXAMPLES_V2 if example["post"].strip().lower() in known)


# ---------------------------------------------------------------- talking to Ollama

class AnswerTimeout(Exception):
    """One request took longer than the timeout: a failed attempt of this sample."""


class ServerUnavailable(Exception):
    """The server cannot be reached (connection refused, reset) or answered with a server error (5xx)."""


def loopback_base(host):
    """The server address as http://<loopback host>:<port>; anything that is not a loopback address is refused."""
    try:
        parts = urllib.parse.urlsplit(host)
        port = parts.port
    except ValueError as exc:
        raise Stop(f"refusing {host!r}: {exc}") from exc
    if (parts.scheme != "http" or parts.hostname not in LOOPBACK_HOSTS or parts.username or parts.password
            or parts.path not in ("", "/") or parts.query or parts.fragment):
        raise Stop(f"refusing {host!r}: only the local Ollama server is allowed, as http:// with one of the "
                   f"loopback host names {', '.join(LOOPBACK_HOSTS)}")
    name = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
    return f"http://{name}:{port or 11434}"


def no_proxy_opener():
    """A URL opener that ignores every proxy setting (environment or system), so requests go straight to the
    loopback address."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


class OllamaClient:
    """Minimal client for the local Ollama server (standard library only)."""

    def __init__(self, host, timeout):
        self.base = loopback_base(host)
        self.timeout = timeout
        self.opener = no_proxy_opener()

    def _request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(self.base + path, data=data, method="GET" if data is None else "POST",
                                         headers={"Content-Type": "application/json"})
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # before URLError: HTTPError is a kind of URLError
            detail = exc.read()[:300].decode("utf-8", "replace")
            if exc.code >= 500:
                raise ServerUnavailable(f"HTTP {exc.code} {detail}") from exc
            raise Stop(f"Ollama answered HTTP {exc.code} on {path}: {detail}") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise AnswerTimeout(str(exc.reason)) from exc
            raise ServerUnavailable(str(exc.reason)) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise AnswerTimeout(str(exc)) from exc
        except (ConnectionError, http.client.HTTPException) as exc:
            raise ServerUnavailable(f"{type(exc).__name__}: {exc}") from exc
        except ValueError as exc:  # not JSON: something other than Ollama answered
            raise Stop(f"the answer on {path} is not JSON: {exc}") from exc

    def version(self):
        return self._request("/api/version").get("version")

    def installed_models(self):
        """Names of the installed models (never downloads one)."""
        models = self._request("/api/tags").get("models", [])
        names = {item.get(key) for item in models if isinstance(item, dict) for key in ("name", "model")}
        return sorted(name for name in names if isinstance(name, str))

    def ask(self, payload):
        """Send one chat request; return the text of the model's answer."""
        body = self._request("/api/chat", payload)
        message = body.get("message") if isinstance(body, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        return content if isinstance(content, str) else ""


def parse_answer(content):
    """The label in an answer: JSON with "level", stripped and lower-cased, one of the seven; otherwise None."""
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        return None
    value = data.get("level") if isinstance(data, dict) else None
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    return value if value in LEVEL_NAMES else None


def ask_with_reconnect(ask, payload, connect_retries, connect_wait, sleep):
    """Ask once; when the server cannot be reached, wait and try again, and give up after connect_retries more tries
    (ServerUnavailable is raised then)."""
    for attempt in range(connect_retries + 1):
        try:
            return ask(payload)
        except ServerUnavailable:
            if attempt == connect_retries:
                raise
            sleep(connect_wait)
    raise AssertionError("unreachable")


def check_votes(samples, min_votes):
    """--samples at least 1, and more than half of them but not more than all needed: a label that is recorded
    always has a majority, so there is never a tie for first place."""
    if not (isinstance(samples, int) and isinstance(min_votes, int) and samples >= 1
            and samples / 2 < min_votes <= samples):
        raise Stop(f"--samples {samples} with --min-votes {min_votes}: --samples must be at least 1 and --min-votes "
                   "more than half of --samples and at most --samples (5 samples with 3 votes, 3 samples with 2)")


def ask_sample(text, ask, *, model, seed, temperature, max_attempts, connect_retries, connect_wait, sleep):
    """One sample (one vote): (label or None, attempts used, what went wrong in the failed attempts).

    A bad answer or a timeout is a failed attempt; the next attempt adds the fixed reminder to the system message
    and keeps the same seed. ServerUnavailable is passed on: it stops the whole run."""
    failures = []
    for attempt in range(1, max_attempts + 1):
        payload = build_payload(text, retry=attempt > 1, model=model, seed=seed, temperature=temperature)
        try:
            content = ask_with_reconnect(ask, payload, connect_retries, connect_wait, sleep)
        except AnswerTimeout:
            failures.append("timeout")
            continue
        label = parse_answer(content)
        if label is not None:
            return label, attempt, failures
        failures.append("bad answer")
    return None, max_attempts, failures


def tally(votes, min_votes):
    """(level, votes text) from the labels of the samples (None for a failed sample). The failed samples do not
    count; the most frequent label with at least min_votes votes is recorded, else disputed; no vote: unparsed."""
    valid = [vote for vote in votes if vote is not None]
    if not valid:
        return UNPARSED, "0/0"
    label, top = Counter(valid).most_common(1)[0]
    return (label if top >= min_votes else DISPUTED), f"{top}/{len(valid)}"


def label_one(text, ask, *, samples, min_votes, model, seed, temperature, max_attempts, connect_retries,
              connect_wait, sleep):
    """Ask one post samples times, sample i with seed + i; returns (level, votes text, attempts, failures)."""
    votes, attempts, failures = [], 0, []
    for index in range(samples):
        label, used, failed = ask_sample(text, ask, model=model, seed=seed + index, temperature=temperature,
                                         max_attempts=max_attempts, connect_retries=connect_retries,
                                         connect_wait=connect_wait, sleep=sleep)
        votes.append(label)
        attempts += used
        failures += failed
    level, votes_text = tally(votes, min_votes)
    return level, votes_text, attempts, failures


# ---------------------------------------------------------------- label files

def row_line(row):
    """One CSV line (with its LF newline, on every system) for a row of strings."""
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerow(row)
    return buffer.getvalue()


def csv_bytes(rows):
    """A whole CSV file (header included) as UTF-8 bytes with LF newlines."""
    return "".join(row_line(row) for row in rows).encode("utf-8")


def write_atomic(path, data):
    """Write bytes to a temporary file next to path, then rename it over path: never half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_rows(raw, columns, path):
    """Rows of a complete CSV file (bytes ending with a newline) with the given header."""
    lines = raw.decode("utf-8").split("\n")[:-1]
    if not lines or lines[0] != ",".join(columns):
        raise Stop(f"{shown(path)}: the first line must be the header {','.join(columns)}")
    rows = list(csv.reader(lines[1:]))
    wrong = [number for number, row in enumerate(rows, start=2) if len(row) != len(columns)]
    if wrong:
        raise Stop(f"{shown(path)}: lines {wrong[:5]} do not have {len(columns)} fields")
    return rows


def open_output(path, log):
    """Prepare a label file for a run: create it with its header, or read it, dropping a partial last line left by
    a run that was stopped while writing. Returns (rows, bytes dropped)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.read_bytes() if path.exists() else b""
    dropped = 0
    if raw and not raw.endswith(b"\n"):
        keep = raw[:raw.rfind(b"\n") + 1]
        dropped = len(raw) - len(keep)
        log(f"dropped a partial last line ({dropped} bytes) of {path.name}; that post is labelled again")
        raw = keep
        write_atomic(path, raw)
    if not raw:
        raw = csv_bytes([COLUMNS])
        write_atomic(path, raw)
    return parse_rows(raw, COLUMNS, path), dropped


def check_existing(rows, *, model, samples, text_by_id, path):
    """Rows already in the file must come from the same model, prompt version and data, with no more samples."""
    models = sorted({row[3] for row in rows})
    versions = sorted({row[4] for row in rows})
    if models and models != [model]:
        raise Stop(f"{shown(path)} holds labels from model(s) {models}; this run uses {model!r}. Use another "
                   "output file.")
    if versions and versions != [PROMPT_VERSION]:
        raise Stop(f"{shown(path)} holds labels from prompt version(s) {versions}; this run uses "
                   f"{PROMPT_VERSION!r}. Use another output file.")
    twice = sorted(post_id for post_id, times in Counter(row[0] for row in rows).items() if times > 1)
    if twice:
        raise Stop(f"{shown(path)}: post ids appear more than once: {twice[:5]}")
    unknown = sorted({row[1] for row in rows} - set(LEVEL_NAMES) - {UNPARSED, DISPUTED})
    if unknown:
        raise Stop(f"{shown(path)}: labels that are not in the list: {unknown[:5]}")
    larger = [row[0] for row in rows
              if not VOTES_PATTERN.fullmatch(row[2]) or int(row[2].split("/")[1]) > samples]
    if larger:
        raise Stop(f"{shown(path)}: {len(larger)} rows have a vote count that is malformed or counts more votes "
                   f"than --samples {samples} ({larger[:3]}); this file was labelled with more samples. Run with the "
                   "same --samples and --min-votes as before, or use another output file.")
    missing = [row[0] for row in rows if row[0] not in text_by_id]
    changed = [row[0] for row in rows if row[0] in text_by_id and text_hash(text_by_id[row[0]]) != row[5]]
    if missing or changed:
        raise Stop(f"{shown(path)} does not belong to this data: {len(missing)} ids missing from it "
                   f"{missing[:3]}, {len(changed)} texts with another fingerprint {changed[:3]}")


def settings_record_path(output):
    """Where the settings record of an output file lives: data/prelabel/<path in the repository with / as __>
    .settings.json, or, for a file outside the repository, the first 16 hex digits of the SHA-256 of its full path."""
    resolved = Path(output).resolve()
    try:
        name = resolved.relative_to(REPO_DIR.resolve()).as_posix().replace("/", "__")
    except ValueError:
        name = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16]
    return REPO_DIR / SETTINGS_DIR / f"{name}.settings.json"


def check_settings(settings, settings_path, rows, path, log):
    """Compare this run's settings with the record of the output file; write the record when the file has no rows
    yet, or when it has rows but no record (a warning). A difference stops the run and names the setting."""
    settings_path = Path(settings_path)
    if rows and settings_path.is_file():
        try:
            recorded = json.loads(settings_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise Stop(f"cannot read the settings record {shown(settings_path)}: {exc}") from exc
        differing = [f"{key} (recorded {recorded.get(key)!r}, now {value!r})" for key, value in settings.items()
                     if recorded.get(key) != value]
        if differing:
            raise Stop(f"{shown(path)} was started with other settings: {'; '.join(differing)}. Run with the recorded "
                       "settings, or use another output file (changing --samples and --min-votes is only possible "
                       "on an empty file).")
        return
    if rows:
        log(f"warning: {shown(path)} has {len(rows)} rows but no settings record ({shown(settings_path)}); going on "
            "with this run's settings and recording them")
    write_atomic(settings_path, (json.dumps(settings, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def label_posts(selection, output, ask, *, settings_path, selection_settings=None, model=DEFAULT_MODEL,
                samples=DEFAULT_SAMPLES, min_votes=DEFAULT_MIN_VOTES, seed=DEFAULT_SEED,
                temperature=DEFAULT_TEMPERATURE, max_attempts=DEFAULT_MAX_ATTEMPTS, limit=None, retry_unparsed=False,
                text_by_id=None, connect_retries=DEFAULT_CONNECT_RETRIES, connect_wait=DEFAULT_CONNECT_WAIT,
                log=print, sleep=time.sleep, clock=time.perf_counter, progress_every=PROGRESS_EVERY):
    """Label the selected posts that are not yet in the output file, appending and flushing one row per post.

    ask(payload) -> answer text is the only contact with the model, so the checks can replace it. Every post is
    asked samples times (sample i with seed + i) and the votes are counted (tally). With retry_unparsed, only the
    rows recorded as unparsed are asked again; a new result that is not unparsed (disputed too) replaces its row in
    place (whole file rewritten through a temporary file; order and row count unchanged). limit caps the posts asked
    in this run. When the server cannot be reached the run stops: the current post is not written. The settings
    (selection_settings plus the voting settings) are compared with the record at settings_path. Returns a summary.
    """
    check_votes(samples, min_votes)
    output = Path(output)
    if text_by_id is None:
        text_by_id = {post["post_id"]: post["text"] for post in selection}
    settings = {**(selection_settings or {}), "seed": seed, "samples": samples, "temperature": temperature,
                "min_votes": min_votes, "max_attempts": max_attempts, "model": model, "prompt_version": PROMPT_VERSION}
    rows, dropped = open_output(output, log)
    check_existing(rows, model=model, samples=samples, text_by_id=text_by_id, path=output)
    check_settings(settings, settings_path, rows, output, log)
    done = {row[0] for row in rows}
    if retry_unparsed:
        todo = [{"post_id": row[0], "text": text_by_id[row[0]]} for row in rows if row[1] == UNPARSED]
    else:
        todo = [post for post in selection if post["post_id"] not in done]
    skipped = 0 if retry_unparsed else len(selection) - len(todo)
    if limit is not None:
        todo = todo[:limit]
    summary = {"mode": "retry-unparsed" if retry_unparsed else "label", "to_label": len(todo),
               "already_in_file": skipped, "dropped_partial_line_bytes": dropped, "labelled": 0,
               "new_unparsed": 0, "new_disputed": 0, "replaced": 0, "requests": 0, "failures": Counter(),
               "post_seconds": 0.0, "stopped": None}
    log(f"{len(todo)} posts to label in this run ({summary['mode']}, {samples} samples per post, {min_votes} votes "
        f"needed); {skipped} already in {output.name}")
    position = {row[0]: index for index, row in enumerate(rows)}
    started = clock()
    handle = None if retry_unparsed else output.open("a", encoding="utf-8", newline="")
    try:
        for number, post in enumerate(todo, start=1):
            post_started = clock()
            try:
                level, votes, attempts, failures = label_one(
                    post["text"], ask, samples=samples, min_votes=min_votes, model=model, seed=seed,
                    temperature=temperature, max_attempts=max_attempts, connect_retries=connect_retries,
                    connect_wait=connect_wait, sleep=sleep)
            except ServerUnavailable as exc:
                summary["stopped"] = f"the Ollama server cannot be reached ({exc}); post {post['post_id']} not written"
                log(f"stopped: {summary['stopped']}. Start the server and run the same command again to continue.")
                break
            summary["post_seconds"] += clock() - post_started
            summary["requests"] += attempts
            summary["failures"].update(failures)
            summary["labelled"] += 1
            summary["new_unparsed"] += level == UNPARSED
            summary["new_disputed"] += level == DISPUTED
            row = [post["post_id"], level, votes, model, PROMPT_VERSION, text_hash(post["text"])]
            if retry_unparsed:
                if level != UNPARSED:
                    rows[position[post["post_id"]]] = row
                    write_atomic(output, csv_bytes([COLUMNS] + rows))
                    summary["replaced"] += 1
            else:
                handle.write(row_line(row))
                handle.flush()
                os.fsync(handle.fileno())
            if number % progress_every == 0 or number == len(todo):
                elapsed = clock() - started
                left = (len(todo) - number) * elapsed / number
                log(f"[{number}/{len(todo)}] {elapsed:.0f} s, {elapsed / number:.2f} s per post, about "
                    f"{left / 60:.0f} min left; disputed {summary['new_disputed']}, unparsed "
                    f"{summary['new_unparsed']} in this run")
    finally:
        if handle is not None:
            handle.close()
    summary["run_seconds"] = clock() - started
    return summary


def label_counts(rows):
    """Rows per label in ladder order, then none, unparsed and disputed (column level)."""
    counts = Counter(row[1] for row in rows)
    return {name: counts.get(name, 0) for name in LEVEL_NAMES + [UNPARSED, DISPUTED]}


# ---------------------------------------------------------------- checking a file (check-file)
# Only the standard library: this runs on a machine with nothing but requirements-prelabel.txt installed, before a
# label file or a filled-in review table is committed.

def read_file_lines(path):
    """(lines without their newlines, problems) of a CSV file, with the byte-level problems: a BOM, not UTF-8,
    carriage returns, no newline at the end."""
    raw = Path(path).read_bytes()
    problems = []
    if raw.startswith(BOM):
        problems.append("line 1: the file starts with a BOM (byte order mark); most likely it was saved with Excel. "
                        "Save it as UTF-8 without BOM, or write it again with the script")
        raw = raw[len(BOM):]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return None, problems + [f"not UTF-8 ({exc})"]
    if not text:
        return None, problems + ["the file is empty"]
    lines = text.split("\n")
    if text.endswith("\n"):
        lines = lines[:-1]
    else:
        problems.append(f"line {len(lines)}: the file does not end with a newline")
    carriage = [number for number, line in enumerate(lines, start=1) if "\r" in line]
    if carriage:
        problems.append(f"line {carriage[0]}: carriage return (\\r) in {len(carriage)} lines; the file must use LF "
                        "line endings only")
    return lines, problems


def prelabel_rows_problems(rows, samples=None, min_votes=None):
    """Problems of the rows (header excluded, line numbers from 2) of a label file."""
    problems = []
    for number, row in enumerate(rows, start=2):
        if len(row) != len(COLUMNS):
            problems.append(f"line {number}: {len(row)} fields instead of {len(COLUMNS)}")
            continue
        post_id, level, votes, model, version, digest = row
        if not POST_ID_PATTERN.fullmatch(post_id):
            problems.append(f"line {number}, column post_id: {post_id!r} is not tim:<row number>")
        if level not in LEVEL_NAMES + [UNPARSED, DISPUTED]:
            problems.append(f"line {number}, column level: {level!r} is not one of the seven labels, {UNPARSED} or "
                            f"{DISPUTED}")
        match = VOTES_PATTERN.fullmatch(votes)
        if not match:
            problems.append(f"line {number}, column votes: {votes!r} is not N/M")
        else:
            top, valid = int(match.group(1)), int(match.group(2))
            if top > valid:
                problems.append(f"line {number}, column votes: {votes} has more votes than valid samples")
            elif level == UNPARSED and (top, valid) != (0, 0):
                problems.append(f"line {number}, column votes: {UNPARSED} must have 0/0, not {votes}")
            elif level in LEVEL_NAMES and not (top >= 1 and 2 * top > valid):
                problems.append(f"line {number}, column votes: a label needs a majority of the valid samples, not "
                                f"{votes}")
            elif level == DISPUTED and not (top >= 1 and valid >= 1):
                problems.append(f"line {number}, column votes: {DISPUTED} needs at least one vote, not {votes}")
            if samples is not None and valid > samples:
                problems.append(f"line {number}, column votes: {votes} counts more than --samples {samples}")
            if min_votes is not None and level in LEVEL_NAMES and top < min_votes:
                problems.append(f"line {number}, column votes: {level} with {votes} has fewer than --min-votes "
                                f"{min_votes}")
            if min_votes is not None and level == DISPUTED and top >= min_votes:
                problems.append(f"line {number}, column votes: {DISPUTED} with {votes} reaches --min-votes "
                                f"{min_votes}")
        if not model:
            problems.append(f"line {number}, column model: empty")
        if not re.fullmatch(r"v[0-9]+", version):
            problems.append(f"line {number}, column prompt_version: {version!r} is not v<number>")
        if not HASH_PATTERN.fullmatch(digest):
            problems.append(f"line {number}, column text_hash: {digest!r} is not 8 lower-case hexadecimal digits")
    good = [row for row in rows if len(row) == len(COLUMNS)]
    first_line = {}
    for number, row in enumerate(rows, start=2):
        if len(row) == len(COLUMNS):
            first_line.setdefault(row[0], number)
    seen = set()
    for number, row in enumerate(rows, start=2):
        if len(row) == len(COLUMNS):
            if row[0] in seen:
                problems.append(f"line {number}, column post_id: {row[0]} appears again (first on line "
                                f"{first_line[row[0]]})")
            seen.add(row[0])
    for column, index in (("model", 3), ("prompt_version", 4)):
        values = sorted({row[index] for row in good})
        if len(values) > 1:
            other = next(number for number, row in enumerate(rows, start=2)
                         if len(row) == len(COLUMNS) and row[index] != good[0][index])
            problems.append(f"line {other}, column {column}: the file holds {len(values)} values {values}; a label "
                            "file holds one")
    return problems


def review_rows_problems(rows, labels=None):
    """Problems of the rows (header excluded, line numbers from 2) of a review table; labels: {post_id: level} of
    the label file it was drawn from, or None."""
    problems, seen = [], {}
    for number, row in enumerate(rows, start=2):
        if len(row) != len(REVIEW_COLUMNS):
            problems.append(f"line {number}: {len(row)} fields instead of {len(REVIEW_COLUMNS)}")
            continue
        post_id, prelabel, human = row
        if not POST_ID_PATTERN.fullmatch(post_id):
            problems.append(f"line {number}, column post_id: {post_id!r} is not tim:<row number>")
        elif post_id in seen:
            problems.append(f"line {number}, column post_id: {post_id} appears again (first on line {seen[post_id]})")
        seen.setdefault(post_id, number)
        if prelabel not in SIX_LEVELS:
            problems.append(f"line {number}, column prelabel: {prelabel!r} is not one of the six levels")
        if human != "" and human not in SIX_LEVELS + [NONE]:
            problems.append(f"line {number}, column human_label: {human!r} is neither empty nor one of the six levels "
                            f"or {NONE}")
        if labels is not None:
            if post_id not in labels:
                problems.append(f"line {number}, column post_id: {post_id} is not in the label file")
            elif labels[post_id] != prelabel:
                problems.append(f"line {number}, column prelabel: {prelabel!r}, but the label file says "
                                f"{labels[post_id]!r}")
    return problems


def file_report(path, samples=None, min_votes=None, labels_path=None):
    """(kind, rows, problems, summary line) of a label file or a review table, told apart by the header."""
    lines, problems = read_file_lines(path)
    if lines is None or not lines:
        return None, [], problems + ([] if lines is None else ["the file has no header"]), ""
    try:
        parsed = list(csv.reader(lines))
    except csv.Error as exc:
        return None, [], problems + [f"not a readable CSV file: {exc}"], ""
    header, rows = parsed[0], parsed[1:]
    if header == COLUMNS:
        kind = "label file"
        problems += prelabel_rows_problems(rows, samples, min_votes)
    elif header == REVIEW_COLUMNS:
        kind = "review table"
        labels = None
        if labels_path is not None:
            _, label_rows, label_problems, _ = file_report(labels_path)
            if label_problems:
                problems.append(f"the label file {labels_path} given with --labels has problems: {label_problems[0]}")
            labels = {row[0]: row[1] for row in label_rows if len(row) == len(COLUMNS)}
        problems += review_rows_problems(rows, labels)
    else:
        return None, rows, problems + [f"line 1: cannot tell what kind of file this is from the header {header}; a "
                                       f"label file starts with {','.join(COLUMNS)}, a review table with "
                                       f"{','.join(REVIEW_COLUMNS)}"], ""
    if not rows:
        problems.append("the file has no rows below the header")
    if kind == "label file":
        good = [row for row in rows if len(row) == len(COLUMNS)]
        counts = Counter(row[1] for row in good)
        votes = Counter(row[2] for row in good)
        summary = (f"{len(rows)} rows; per label "
                   + json.dumps({name: counts.get(name, 0) for name in LEVEL_NAMES + [UNPARSED, DISPUTED]})
                   + "; votes " + json.dumps(dict(sorted(votes.items(), key=lambda item: (-item[1], item[0]))))
                   + f"; model {sorted({row[3] for row in good})}; prompt version {sorted({row[4] for row in good})}")
    else:
        good = [row for row in rows if len(row) == len(REVIEW_COLUMNS)]
        counts = Counter(row[1] for row in good)
        summary = (f"{len(rows)} rows; per level "
                   + json.dumps({name: counts.get(name, 0) for name in SIX_LEVELS})
                   + f"; human_label filled in {sum(row[2] != '' for row in good)}")
    return kind, rows, problems, summary


SHOWN_PROBLEMS = 10  # problems printed by check-file, at most


def command_check_file(args):
    path = repo_path(args.file)
    if not path.is_file():
        raise Stop(f"{args.file} not found")
    labels_path = repo_path(args.labels) if args.labels else None
    if labels_path is not None and not labels_path.is_file():
        raise Stop(f"{args.labels} not found")
    kind, _, problems, summary = file_report(path, args.samples, args.min_votes, labels_path)
    if problems:
        print(f"error: {args.file}" + (f" ({kind})" if kind else "") + f": {len(problems)} problems", file=sys.stderr)
        for problem in problems[:SHOWN_PROBLEMS]:
            print(f"  {problem}", file=sys.stderr)
        if len(problems) > SHOWN_PROBLEMS:
            print(f"  ... and {len(problems) - SHOWN_PROBLEMS} more", file=sys.stderr)
        return 1
    print(f"ok: {args.file} ({kind}): {summary}")
    return 0


# ---------------------------------------------------------------- review table

def review_rows(rows, size=DEFAULT_REVIEW_SIZE, seed=DEFAULT_SEED):
    """Draw the review rows from a label file's rows, taking the six levels in turn.

    none, unparsed and disputed rows are left out. The rows of every level (file order) are shuffled with their own
    random.Random(seed); then, round after round, every level in ladder order gives its next row, a level with no
    rows left being skipped, until size rows are drawn or none are left. Returns [(post_id, level), ...] in drawing
    order."""
    queues = []
    for name in SIX_LEVELS:
        ids = [row[0] for row in rows if row[1] == name]
        random.Random(seed).shuffle(ids)
        queues.append((name, ids))
    picked, position = [], 0
    while len(picked) < size and any(position < len(ids) for _, ids in queues):
        for name, ids in queues:
            if position < len(ids) and len(picked) < size:
                picked.append((ids[position], name))
        position += 1
    return picked


def write_review_template(labels_path, output, size=DEFAULT_REVIEW_SIZE, seed=DEFAULT_SEED):
    """Check the label file (as check-file does), draw the review rows and write the table; returns the summary."""
    kind, rows, problems, _ = file_report(labels_path)
    if kind != "label file" or problems:
        raise Stop(f"{shown(labels_path)} does not pass check-file: " + "; ".join(problems[:3] or [f"it is a {kind}"]))
    picked = review_rows(rows, size, seed)
    write_atomic(output, csv_bytes([REVIEW_COLUMNS] + [[post_id, level, ""] for post_id, level in picked]))
    available = label_counts(rows)
    drawn = Counter(level for _, level in picked)
    return {"rows": len(picked), "usable": sum(available[name] for name in SIX_LEVELS),
            "per_level": {name: drawn.get(name, 0) for name in SIX_LEVELS}, "in_label_file": available}


def inside_data_or_outside_repository(path):
    """True when path is under the repository's data/ folder (ignored by Git) or outside the repository."""
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(REPO_DIR.resolve())
    except ValueError:
        return True
    return bool(relative.parts) and relative.parts[0] == "data"


# ---------------------------------------------------------------- commands

def log_run(record):
    """Append one line about a run to data/prelabel/levels-runs.jsonl (no text of any post)."""
    path = REPO_DIR / RUN_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def read_and_select(args):
    """Environment check, the dataset, the posts and the draw (shared by label, show-selection, export-review)."""
    check_environment()
    print(f"reading {args.source or DATASET_ADDRESS} ...", flush=True)
    table, where = load_table(args.source)
    print(f"{where}: {len(table)} rows, columns {list(table.columns)}", flush=True)
    posts = posts_from_table(table)
    del table  # the sentiment and url columns go no further
    selection, summary, steps = select_posts(posts, args.count, args.pool_cap, args.seed)
    return posts, selection, summary, steps, where


def print_selection(summary):
    print(f"posts in the dataset:      {summary['posts_in_source']}")
    print(f"left out, empty text:      {summary['empty_texts_left_out']}")
    print(f"left out, repeated text:   {summary['repeated_texts_left_out']}  (same text after stripping, case counts)")
    print(f"usable posts:              {summary['usable_posts']}")
    print(f"count {summary['count']}, pool cap {summary['pool_cap']}, seed {summary['seed']}")
    print("pool               posts in pool   taken")
    for pool, _ in KEYWORD_POOLS:
        print(f"  {pool:17s} {summary['pool_posts'][pool]:13d}   {summary['taken_per_step'][pool]:5d}")
    print(f"  {RANDOM_STEP:17s} {'':13s}   {summary['taken_per_step'][RANDOM_STEP]:5d}  "
          f"(wanted {summary['fill_wanted']})")
    print(f"selected:                  {summary['selected']}")
    print("keyword hits (usable posts that contain the word or phrase; a post can count for several):")
    for pool, words in KEYWORD_POOLS:
        print(f"  {pool}: " + ", ".join(f"{word} {summary['keyword_hits'][word]}" for word in words))
    print(f"first ids of the processing order: {', '.join(summary['first_ids'])}")
    print(f"selection fingerprint: {summary['fingerprint']}")


def command_show_selection(args):
    _, _, summary, _, where = read_and_select(args)
    if args.source is None:
        copy = REPO_DIR / DATASET_COPY
        print(f"dataset copy: {shown(copy)}, {copy.stat().st_size} bytes, sha256 "
              f"{hashlib.sha256(copy.read_bytes()).hexdigest()}")
    print_selection(summary)
    print(f"keyword list fingerprint: {keywords_fingerprint()}")
    return 0


def command_label(args):
    base = loopback_base(args.host)  # refuse any other address before anything else happens
    check_votes(args.samples, args.min_votes)
    if (args.max_attempts < 1 or args.timeout <= 0 or args.connect_retries < 0 or args.connect_wait < 0
            or args.temperature < 0):
        raise Stop("--max-attempts must be at least 1, --timeout above 0, --connect-retries, --connect-wait and "
                   "--temperature at least 0")
    output = repo_path(args.output or default_output("sample" if args.sample else "full"))
    check_environment()

    client = OllamaClient(base, args.timeout)
    try:
        server_version = client.version()
        installed = client.installed_models()
    except (ServerUnavailable, AnswerTimeout) as exc:
        raise Stop(f"the Ollama server at {client.base} cannot be reached ({exc}); start it (ollama serve) and "
                   "try again") from exc
    if args.model not in installed:
        raise Stop(f"model {args.model!r} is not installed in Ollama (installed: {installed}). This script never "
                   f"downloads models; install it yourself (ollama pull {args.model}) if you want it.")
    print(f"Ollama {server_version} at {client.base}, model {args.model}, prompt {PROMPT_VERSION} "
          f"(fingerprint {prompt_fingerprint()[:16]}...)", flush=True)

    posts, selection, drawn, steps, where = read_and_select(args)
    found = examples_in_dataset(post["text"] for post in posts)
    if found:
        raise Stop(f"hand-written examples of the prompt are posts of the dataset: {found}")
    text_by_id = {post["post_id"]: post["text"] for post in posts}
    print(f"selection: {drawn['selected']} posts, per step {json.dumps(drawn['taken_per_step'])}, fingerprint "
          f"{drawn['fingerprint']}", flush=True)
    selection_settings = {"count": args.count, "pool_cap": args.pool_cap,
                          "data": DATASET_REVISION if args.source is None else f"file {args.source}",
                          "keywords": keywords_fingerprint(), "selection_fingerprint": drawn["fingerprint"]}
    settings_path = settings_record_path(output)

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = label_posts(selection, output, client.ask, settings_path=settings_path,
                          selection_settings=selection_settings, model=args.model, samples=args.samples,
                          min_votes=args.min_votes, seed=args.seed, temperature=args.temperature,
                          max_attempts=args.max_attempts, limit=args.limit, retry_unparsed=args.retry_unparsed,
                          text_by_id=text_by_id, connect_retries=args.connect_retries,
                          connect_wait=args.connect_wait, log=lambda text: print(text, flush=True))
    raw = output.read_bytes()
    rows = parse_rows(raw, COLUMNS, output)
    counts = label_counts(rows)
    per_step = Counter(steps.get(row[0], "not in this selection") for row in rows)
    labelled = summary["labelled"]
    per_post = summary["post_seconds"] / labelled if labelled else None
    per_request = summary["post_seconds"] / summary["requests"] if summary["requests"] else None
    print(f"\n{shown(output)}: {len(rows)} rows ({sum(row[0] in steps for row in rows)} of the {len(selection)} "
          f"selected posts); settings record {shown(settings_path)}")
    print("rows per label:", json.dumps(counts))
    print("rows per votes:", json.dumps(dict(sorted(Counter(row[2] for row in rows).items(),
                                                     key=lambda item: (-item[1], item[0])))))
    print("rows per step of the draw:", json.dumps(dict(per_step)))
    print(f"this run: {labelled} posts, {summary['requests']} requests in {summary['run_seconds']:.1f} s"
          + (f", {per_post:.2f} s per post, {per_request:.2f} s per request" if per_post is not None else "")
          + f"; failed attempts {dict(summary['failures'])}"
          + (f"; replaced {summary['replaced']} unparsed rows" if args.retry_unparsed else ""), flush=True)
    log_run({"started": started_at, "ended": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "output": shown(output), "data": where, "ollama": server_version, "model": args.model,
             "prompt_version": PROMPT_VERSION, "settings": {**selection_settings, "seed": args.seed,
                                                             "samples": args.samples, "min_votes": args.min_votes,
                                                             "temperature": args.temperature,
                                                             "max_attempts": args.max_attempts},
             "mode": summary["mode"], "to_label": summary["to_label"], "labelled": labelled,
             "requests": summary["requests"], "new_unparsed": summary["new_unparsed"],
             "new_disputed": summary["new_disputed"], "replaced": summary["replaced"],
             "post_seconds": round(summary["post_seconds"], 3), "run_seconds": round(summary["run_seconds"], 3),
             "failures": dict(summary["failures"]), "dropped_partial_line_bytes": summary["dropped_partial_line_bytes"],
             "stopped": summary["stopped"], "rows_in_file": len(rows), "rows_per_label": counts})
    return 3 if summary["stopped"] else 0


def command_review_template(args):
    labels_path = repo_path(args.labels or default_output("full"))
    output = repo_path(args.output or default_output("review"))
    if not labels_path.is_file():
        raise Stop(f"{shown(labels_path)} not found")
    result = write_review_template(labels_path, output, args.size, args.seed)
    print(f"{shown(output)}: {result['rows']} rows drawn from {result['usable']} rows with one of the six levels in "
          f"{shown(labels_path)}")
    print("rows per level in the table:", json.dumps(result["per_level"]))
    print("rows per label in the label file:", json.dumps(result["in_label_file"]))
    return 0


def command_export_review(args):
    template = repo_path(args.template or default_output("review"))
    labels_path = repo_path(args.labels or default_output("full"))
    output = repo_path(args.output)
    if not inside_data_or_outside_repository(output):
        raise Stop(f"refusing {output}: the table with the original texts may only be written under data/ "
                   "(ignored by Git) or outside the repository")
    for path in (template, labels_path):
        kind, _, problems, _ = file_report(path)
        if problems:
            raise Stop(f"{shown(path)} does not pass check-file: {problems[0]}")
    review = parse_rows(template.read_bytes(), REVIEW_COLUMNS, template)
    hashes = {row[0]: row[5] for row in parse_rows(labels_path.read_bytes(), COLUMNS, labels_path)}
    check_environment()
    table, where = load_table(args.source)
    text_by_id = {post["post_id"]: post["text"] for post in posts_from_table(table)}
    lines = [row_line(REVIEW_COLUMNS + ["text"])]
    for post_id, prelabel, human_label in review:
        text = text_by_id.get(post_id)
        if text is None or hashes.get(post_id) != text_hash(text):
            raise Stop(f"{post_id}: not in {where}, or its text does not match the label file's fingerprint")
        lines.append(row_line([post_id, prelabel, human_label, text]))
    write_atomic(output, "".join(lines).encode("utf-8"))
    print(f"wrote {len(review)} rows with their texts to {output}")
    return 0


def command_show_prompt(args):
    payload = build_payload(POST_PLACEHOLDER, retry=False, model=DEFAULT_MODEL, seed=DEFAULT_SEED,
                            temperature=DEFAULT_TEMPERATURE)
    print(f"prompt version {PROMPT_VERSION}, fingerprint {prompt_fingerprint()}\n")
    for message in payload["messages"]:
        print(f"--- {message['role']}\n{message['content']}")
    print(f"\n--- added to the system message when a sample is asked again\n{retry_reminder()}")
    print(f"\n--- answer format\n{json.dumps(payload['format'])}\n--- options of sample 0 (sample i: seed + i)\n"
          f"{json.dumps(payload['options'])}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description="Pre-label stock tweets with the level of what they look at through the "
                                                 "local Ollama server.")
    commands = parser.add_subparsers(dest="command", required=True)

    def add_draw(command):
        command.add_argument("--source", help=f"a local parquet file with the dataset's columns (default: the "
                                              f"pinned dataset, copied once to {DATASET_COPY})")
        command.add_argument("--count", type=int, default=DEFAULT_COUNT, help=f"posts to draw (default {DEFAULT_COUNT})")
        command.add_argument("--pool-cap", type=int, default=DEFAULT_POOL_CAP,
                             help=f"most posts from one keyword pool (default {DEFAULT_POOL_CAP})")
        command.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"seed (default {DEFAULT_SEED})")

    label = commands.add_parser("label", help="label posts (resumable)")
    label.add_argument("--sample", action="store_true",
                       help=f"write to {default_output('sample')} instead of {default_output('full')} (same posts, "
                            "same order)")
    label.add_argument("--output", help="label file (default as above)")
    add_draw(label)
    label.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                       help=f"questions per post, sample i with seed + i (default {DEFAULT_SAMPLES})")
    label.add_argument("--min-votes", type=int, default=DEFAULT_MIN_VOTES,
                       help=f"votes the label needs, more than half of --samples (default {DEFAULT_MIN_VOTES})")
    label.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                       help=f"the model's temperature (default {DEFAULT_TEMPERATURE:g})")
    label.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default {DEFAULT_MODEL})")
    label.add_argument("--host", default=DEFAULT_HOST, help=f"Ollama server, loopback only (default {DEFAULT_HOST})")
    label.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                       help=f"seconds per request (default {DEFAULT_TIMEOUT:g})")
    label.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS,
                       help=f"attempts per sample before it counts as failed (default {DEFAULT_MAX_ATTEMPTS})")
    label.add_argument("--connect-retries", type=int, default=DEFAULT_CONNECT_RETRIES,
                       help=f"extra tries when the server cannot be reached (default {DEFAULT_CONNECT_RETRIES})")
    label.add_argument("--connect-wait", type=float, default=DEFAULT_CONNECT_WAIT,
                       help=f"seconds between those tries (default {DEFAULT_CONNECT_WAIT:g})")
    label.add_argument("--limit", type=int, help="label at most this many posts in this run")
    label.add_argument("--retry-unparsed", action="store_true",
                       help=f"ask again only for the rows recorded as {UNPARSED} ({DISPUTED} rows are kept)")

    show_selection = commands.add_parser("show-selection", help="draw the posts without Ollama and print the draw")
    add_draw(show_selection)

    review = commands.add_parser("review-template", help="draw the review table from a label file")
    review.add_argument("--labels", help=f"label file (default {default_output('full')})")
    review.add_argument("--output", help=f"review table (default {default_output('review')})")
    review.add_argument("--size", type=int, default=DEFAULT_REVIEW_SIZE, help=f"rows (default {DEFAULT_REVIEW_SIZE})")
    review.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"seed (default {DEFAULT_SEED})")

    export = commands.add_parser("export-review", help="the review rows with their texts, under data/ only")
    export.add_argument("--template", help=f"review table (default {default_output('review')})")
    export.add_argument("--labels", help=f"label file (default {default_output('full')})")
    export.add_argument("--source", help="a local parquet file with the dataset's columns (default: the pinned dataset)")
    export.add_argument("--output", default=REVIEW_EXPORT, help=f"output (default {REVIEW_EXPORT})")

    commands.add_parser("show-prompt", help="print the fixed prompt, the answer format and the fingerprint")

    check = commands.add_parser("check-file", help="check the format of a label file or a review table")
    check.add_argument("file", help="the CSV file")
    check.add_argument("--samples", type=int, help="also check that no row counts more valid samples than this")
    check.add_argument("--min-votes", type=int,
                       help=f"also check that every label has at least this many votes and every {DISPUTED} fewer")
    check.add_argument("--labels", help="with a review table: the label file it was drawn from")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    commands = {"label": command_label, "show-selection": command_show_selection,
                "review-template": command_review_template, "export-review": command_export_review,
                "show-prompt": command_show_prompt, "check-file": command_check_file}
    try:
        return commands[args.command](args)
    except Stop as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def utf8_line_output():
    """UTF-8 output flushed line by line: Windows terminals are not UTF-8 by default (a tweet with an emoji would
    stop the run), and Python's output into a pipe (Git Bash) is otherwise held back in large blocks."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


if __name__ == "__main__":
    utf8_line_output()
    sys.exit(main())
