"""Pre-label StockTwits posts with one of nine emotions through the local Ollama server.

Run it through scripts/prelabel_emotions.sh, which sets up the environment from
scripts/env.sh and uses the virtual environment's Python:

    scripts/prelabel_emotions.sh label [--sample] [options]
        Label posts and append one row per post to a CSV file. Resumable: posts
        already in the file are skipped, so a stopped run continues where it
        stopped when it is started again.
    scripts/prelabel_emotions.sh review-template [options]
        Draw the review table (post id, pre-label, empty human label) from the
        labelled file.
    scripts/prelabel_emotions.sh export-review [options]
        Write the same review rows with the original texts, under data/ only.
    scripts/prelabel_emotions.sh show-prompt
        Print the fixed prompt text, the answer format and their fingerprint.

The labels come from a 7-billion-parameter language model, not from people
(silver labels). The label files hold only post ids, labels, the model name,
the prompt version and an 8-digit fingerprint of each text; never any text.

Only the local Ollama server is contacted, on a loopback address, without any
proxy, key or model download. Relative paths are relative to the repository.
"""

import argparse
import csv
import hashlib
import http.client
import io
import json
import math
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

# ---------------------------------------------------------------- defaults

FULL_SOURCE = "data/raw/st-data-full.xlsx"  # the full dataset (scripts/download_data.sh)
SAMPLE_SOURCE = "samples/stocktwits-crypto-sample.xlsx"  # the 600-post sample of the quick check
DEFAULT_HOST = "http://127.0.0.1:11434"  # the local Ollama server
LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost")  # the only host names the script talks to
DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_COUNT = 4000  # posts drawn from the full dataset ("all" = every post with text)
DEFAULT_BTC_SHARE = 0.75  # share of the drawn posts that mention Bitcoin
DEFAULT_SEED = 42  # draw, processing order, review draw, and the model's own seed
DEFAULT_MAX_ATTEMPTS = 3  # questions per post before it is recorded as unparsed
DEFAULT_TIMEOUT = 120.0  # seconds one request may take before it counts as a failed attempt
DEFAULT_CONNECT_RETRIES = 3  # extra tries when the server cannot be reached, before the run stops
DEFAULT_CONNECT_WAIT = 5.0  # seconds between those tries
NUM_PREDICT = 64  # most tokens of an answer; the longest answer, {"emotion": "wait_and_see"}, needs about 10
DEFAULT_REVIEW_SIZE = 200  # rows of the review table
PROGRESS_EVERY = 25  # print a progress line after this many posts
RUN_LOG = "data/prelabel/runs.jsonl"  # one line per run (no text); always under data/
REVIEW_EXPORT = "data/review/emotions-review-with-text.csv"  # the review rows with the original texts

COLUMNS = ["post_id", "emotion", "model", "prompt_version", "text_hash"]  # columns of a label file, in order
REVIEW_COLUMNS = ["post_id", "prelabel", "human_label"]  # columns of the review table, in order
UNPARSED = "unparsed"  # recorded when no attempt gave one of the labels; means "failed", not an emotion
FALLBACK = "none"  # the label for posts without a clear emotion; every label list must have it
BTC_PATTERN = re.compile(r"\b(?:bitcoin|btc)\b", flags=re.IGNORECASE)  # same rule as the notebook's mentions_btc
POST_ID_PATTERN = re.compile(r"[A-Za-z0-9_]+:[0-9]+")  # <sheet name>:<row number>
LABEL_NAME_PATTERN = re.compile(r"[a-z][a-z_]*")

# ---------------------------------------------------------------- prompt, version v1
# Everything sent to the model except the post itself is fixed by the prompt version. The checks pin the
# fingerprint of version v1 (prompt_fingerprint); changing one character of the texts below, the examples or the
# answer format needs a new version, a new label file and a new pinned fingerprint.

PROMPT_VERSION = "v1"
LABELS_V1 = [
    {"name": "panic", "meaning": "afraid of losing money, rushing to sell, shouting that the price will crash"},
    {"name": "greed", "meaning": "wanting to get rich fast, fear of missing out, going all in, gambling with leverage"},
    {"name": "excitement", "meaning": "thrilled by a price rise or good news, cheering the price on"},
    {"name": "anger", "meaning": "angry at someone or something, such as an exchange, regulators, whales or other posters"},
    {"name": "mockery", "meaning": "sarcasm or ridicule, laughing at others for losing money or for believing the wrong thing"},
    {"name": "despair", "meaning": "hopeless after a loss, giving up, feeling that it is all over"},
    {"name": "wait_and_see", "meaning": "undecided or cautious, waiting for confirmation before acting"},
    {"name": "confidence", "meaning": "calm certainty, holding for the long term, sure of one's own judgement"},
    {"name": "none", "meaning": "no emotion: news, a neutral fact, a plain question, or only a link"},
]
# Hand-written examples, one or two per label; none of them is a post of the dataset.
EXAMPLES_V1 = [
    {"post": "dumping everything before it goes to zero, this is the end", "emotion": "panic"},
    {"post": "all in with 50x leverage, lambo by friday or bust", "emotion": "greed"},
    {"post": "we just smashed through the high, let's goooo", "emotion": "excitement"},
    {"post": "this exchange froze withdrawals again, they should all be in jail", "emotion": "anger"},
    {"post": "imagine buying the top and calling it a long term investment lol", "emotion": "mockery"},
    {"post": "down 85% on everything, i'm done, deleting the app", "emotion": "despair"},
    {"post": "not touching anything until we get a daily close above resistance", "emotion": "wait_and_see"},
    {"post": "held through three crashes, dips don't scare me, see you in five years", "emotion": "confidence"},
    {"post": "the fed announces its rate decision on wednesday", "emotion": "none"},
    {"post": "which wallet do you use for cold storage?", "emotion": "none"},
]
USER_PREFIX = "Post:\n"  # put in front of every post (and every example post) in the user message
POST_PLACEHOLDER = "<post>"  # stands for the post when the fingerprint is computed


def system_prompt(labels):
    """The system message: what the posts are, the labels and their meanings, and the rules."""
    label_lines = "\n".join(f"{item['name']}: {item['meaning']}." for item in labels)
    return (
        "You label the emotion in posts from StockTwits, a social platform where people write short posts about "
        "crypto coins and crypto markets. Each user message is one post.\n\n"
        f"Choose the one emotion that the writer of the post shows most clearly. Choose exactly one of these "
        f"{len(labels)} labels:\n\n"
        f"{label_lines}\n\n"
        "Rules:\n"
        "1. Judge the feeling of the writer, not only whether the post is bullish or bearish.\n"
        "2. Choose exactly one label.\n"
        f"3. If the post shows no clear emotion, or you are not sure, choose {FALLBACK}.\n"
        '4. Answer with JSON only, in this form: {"emotion": "<label>"}'
    )


def retry_reminder(labels):
    """The fixed sentence added to the system message when a post is asked again."""
    names = ", ".join(item["name"] for item in labels)
    return ('Your previous answer could not be used. Answer with JSON only, exactly in the form '
            f'{{"emotion": "<label>"}}, where <label> is one of: {names}.')


def answer_format(labels):
    """JSON Schema for Ollama's structured output: an object whose only field is one of the labels."""
    return {"type": "object",
            "properties": {"emotion": {"type": "string", "enum": [item["name"] for item in labels]}},
            "required": ["emotion"]}


def build_payload(text, *, retry, label_set, model, seed):
    """The request body for Ollama's /api/chat: fixed messages, the examples, then the post; no dataset label."""
    labels = label_set["labels"]
    system = system_prompt(labels)
    if retry:
        system += "\n\n" + retry_reminder(labels)
    messages = [{"role": "system", "content": system}]
    for example in label_set["examples"]:
        messages.append({"role": "user", "content": USER_PREFIX + example["post"]})
        messages.append({"role": "assistant", "content": json.dumps({"emotion": example["emotion"]})})
    messages.append({"role": "user", "content": USER_PREFIX + text})
    return {"model": model, "messages": messages, "stream": False, "format": answer_format(labels),
            "options": {"temperature": 0, "seed": seed, "num_predict": NUM_PREDICT}}


def prompt_fingerprint(label_set=None):
    """SHA-256 (hex) of everything fixed that is sent: the messages of a first request and of a retry, with the
    placeholder in place of the post, and the answer format."""
    label_set = label_set or BUILT_IN
    first = build_payload(POST_PLACEHOLDER, retry=False, label_set=label_set, model="", seed=0)
    again = build_payload(POST_PLACEHOLDER, retry=True, label_set=label_set, model="", seed=0)
    text = json.dumps({"first": first["messages"], "retry": again["messages"], "format": first["format"]},
                      ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("ascii")).hexdigest()


BUILT_IN = {"prompt_version": PROMPT_VERSION, "labels": LABELS_V1, "examples": EXAMPLES_V1}


class Stop(Exception):
    """A problem that stops the script with a message (nothing is guessed or overwritten)."""


def label_set_problems(data):
    """Every way in which a label list read from a file does not fit (empty when it fits)."""
    problems = []
    if not isinstance(data, dict) or sorted(data) != ["examples", "labels", "prompt_version"]:
        return ["needs exactly the fields prompt_version, labels and examples"]
    version = data["prompt_version"]
    if not isinstance(version, str) or not re.fullmatch(r"v[0-9]+", version):
        problems.append(f"prompt_version {version!r} is not v<number>")
    elif version == PROMPT_VERSION:
        problems.append(f"{PROMPT_VERSION} is the built-in prompt; another label list needs another prompt version")
    labels = data["labels"]
    names = []
    if not isinstance(labels, list) or not labels:
        problems.append("labels must be a non-empty list")
        labels = []
    for index, item in enumerate(labels):
        if not (isinstance(item, dict) and sorted(item) == ["meaning", "name"] and isinstance(item["name"], str)
                and LABEL_NAME_PATTERN.fullmatch(item["name"]) and isinstance(item["meaning"], str)
                and item["meaning"].strip()):
            problems.append(f"label {index}: needs a name (lower-case letters and _) and a meaning")
            continue
        names.append(item["name"])
    if len(set(names)) != len(names):
        problems.append("a label name is used twice")
    if UNPARSED in names:
        problems.append(f"{UNPARSED!r} is reserved for failed answers")
    if FALLBACK not in names:
        problems.append(f"the list needs the label {FALLBACK!r} for posts without a clear emotion")
    examples = data["examples"]
    if not isinstance(examples, list):
        problems.append("examples must be a list (it may be empty)")
        examples = []
    for index, item in enumerate(examples):
        if not (isinstance(item, dict) and sorted(item) == ["emotion", "post"] and isinstance(item["post"], str)
                and item["post"].strip() and item["emotion"] in names):
            problems.append(f"example {index}: needs a post and an emotion from the list")
    return problems


def load_label_set(path=None):
    """The built-in label list and prompt (version v1), or a list read from a JSON file with another version."""
    if path is None:
        return BUILT_IN
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Stop(f"cannot read the label list {path}: {exc}") from exc
    problems = label_set_problems(data)
    if problems:
        raise Stop(f"label list {path}: " + "; ".join(problems))
    return data


def label_names(label_set):
    return [item["name"] for item in label_set["labels"]]


def default_output(kind, version):
    """Default file paths; the version is part of the name, so another prompt version never writes into v1's files."""
    if kind == "full":
        return f"labels/emotions-prelabel-{version}.csv"
    suffix = "" if version == PROMPT_VERSION else f"-{version}"
    if kind == "sample":
        return f"samples/emotions-prelabel-sample{suffix}.csv"
    return f"labels/emotions-review-template{suffix}.csv"


# ---------------------------------------------------------------- posts

def repo_path(path):
    """A path given on the command line: absolute as given, relative to the repository otherwise."""
    path = Path(path)
    return path if path.is_absolute() else REPO_DIR / path


def read_posts(source):
    """Read every sheet and stack them, with the same ids and texts as chapter 1 of the notebook (read_posts):
    post_id = <sheet name>:<row number> (row 1 holds the column names), text read as text, empty cells as "".
    The dataset's own label is not read: the model never sees it."""
    sheets = pd.read_excel(source, sheet_name=None, dtype={"text": str})  # dict: sheet name -> table
    frames = []
    for sheet_name, sheet in sheets.items():
        if "text" not in sheet.columns:
            raise Stop(f"sheet {sheet_name!r} of {source} has no column 'text'")
        frames.append(pd.DataFrame({
            "post_id": [f"{sheet_name}:{row}" for row in range(2, len(sheet) + 2)],
            "text": sheet["text"].fillna("").astype(str),
        }))
    posts = pd.concat(frames, ignore_index=True)
    if not posts["post_id"].is_unique:
        raise Stop("post_id is not unique")
    return posts


def text_hash(text):
    """First 8 hexadecimal digits of the SHA-256 of the text (UTF-8): identifies the text, cannot give it back."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def mentions_btc(text):
    return bool(BTC_PATTERN.search(text))


def parse_count(value):
    """--count: a positive integer, or "all" (returned as None)."""
    if str(value).lower() == "all":
        return None
    try:
        count = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is neither a positive integer nor 'all'") from None
    if count < 1:
        raise argparse.ArgumentTypeError("the count must be at least 1")
    return count


def select_posts(posts, count, btc_share, seed):
    """Draw the posts to label and put them in processing order.

    Posts whose text is empty after stripping are left out (as in the notebook's n_empty_text). With a count, the
    posts that mention Bitcoin give round(count x btc_share) posts and the others the rest, each drawn with the seed
    (a pool that is too small is taken whole); with count None every post is taken. The posts drawn are then
    shuffled with the same seed, so that a run stopped half-way has labelled a random part with about the same
    share of Bitcoin posts. Returns (list of {post_id, text, mentions_btc}, summary)."""
    if not 0 <= btc_share <= 1:
        raise Stop(f"the Bitcoin share must be between 0 and 1, not {btc_share}")
    usable = posts[posts["text"].str.strip() != ""]
    is_btc = usable["text"].str.contains(BTC_PATTERN)
    pools = {"btc": usable[is_btc], "other": usable[~is_btc]}
    if count is None:
        wanted = {name: len(pool) for name, pool in pools.items()}
    else:
        wanted_btc = math.floor(count * btc_share + 0.5)  # rounded half up
        wanted = {"btc": wanted_btc, "other": count - wanted_btc}
    taken = {name: pools[name].sample(n=min(wanted[name], len(pools[name])), random_state=seed) for name in pools}
    chosen = pd.concat([taken["btc"], taken["other"]]).sample(frac=1, random_state=seed)
    selection = [{"post_id": post_id, "text": text, "mentions_btc": mentions_btc(text)}
                 for post_id, text in zip(chosen["post_id"], chosen["text"])]
    summary = {"posts_in_source": len(posts), "empty_texts_left_out": len(posts) - len(usable),
               "pool_btc": len(pools["btc"]), "pool_other": len(pools["other"]),
               "wanted_btc": wanted["btc"], "wanted_other": wanted["other"],
               "taken_btc": len(taken["btc"]), "taken_other": len(taken["other"]), "selected": len(selection)}
    return selection, summary


# ---------------------------------------------------------------- talking to Ollama

class AnswerTimeout(Exception):
    """One request took longer than the timeout: a failed attempt for this post."""


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


def parse_answer(content, names):
    """The label in an answer: JSON with "emotion", stripped and lower-cased, one of the names; otherwise None."""
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        return None
    value = data.get("emotion") if isinstance(data, dict) else None
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    return value if value in names else None


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


def label_one(text, ask, *, label_set, model, seed, max_attempts, connect_retries, connect_wait, sleep):
    """Label one post: (label or UNPARSED, attempts used, what went wrong in the failed attempts).

    A bad answer or a timeout is a failed attempt; the next attempt adds the fixed reminder to the system message
    (at temperature 0 the same question would get the same answer). ServerUnavailable is passed on: it stops the
    whole run instead of marking posts as unparsed."""
    names = label_names(label_set)
    failures = []
    for attempt in range(1, max_attempts + 1):
        payload = build_payload(text, retry=attempt > 1, label_set=label_set, model=model, seed=seed)
        try:
            content = ask_with_reconnect(ask, payload, connect_retries, connect_wait, sleep)
        except AnswerTimeout:
            failures.append("timeout")
            continue
        label = parse_answer(content, names)
        if label is not None:
            return label, attempt, failures
        failures.append("bad answer")
    return UNPARSED, max_attempts, failures


# ---------------------------------------------------------------- label files

def row_line(row):
    """One CSV line (with its newline) for a row of strings."""
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerow(row)
    return buffer.getvalue()


def write_atomic(path, data):
    """Write bytes to a temporary file next to path, then rename it over path: never half a file."""
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
        raise Stop(f"{path}: the first line must be the header {','.join(columns)}")
    rows = list(csv.reader(lines[1:]))
    wrong = [number for number, row in enumerate(rows, start=2) if len(row) != len(columns)]
    if wrong:
        raise Stop(f"{path}: lines {wrong[:5]} do not have {len(columns)} fields")
    return rows


def read_label_file(path, columns=COLUMNS):
    """Read a complete label file without changing it (a partial last line is an error here)."""
    if not path.is_file():
        raise Stop(f"{path} not found")
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise Stop(f"{path} ends with a partial line; run the label command again first, it repairs the file")
    return parse_rows(raw, columns, path)


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
        raw = row_line(COLUMNS).encode("utf-8")
        write_atomic(path, raw)
    return parse_rows(raw, COLUMNS, path), dropped


def check_existing(rows, *, model, prompt_version, names, text_by_id, path):
    """Rows already in the file must come from the same model, prompt version and data source; never mix."""
    models = sorted({row[2] for row in rows})
    versions = sorted({row[3] for row in rows})
    if models and models != [model]:
        raise Stop(f"{path} holds labels from model(s) {models}; this run uses {model!r}. Use another output file.")
    if versions and versions != [prompt_version]:
        raise Stop(f"{path} holds labels from prompt version(s) {versions}; this run uses {prompt_version!r}. "
                   "Use another output file.")
    ids = Counter(row[0] for row in rows)
    twice = sorted(post_id for post_id, times in ids.items() if times > 1)
    if twice:
        raise Stop(f"{path}: post ids appear more than once: {twice[:5]}")
    unknown = [row[1] for row in rows if row[1] not in names and row[1] != UNPARSED]
    if unknown:
        raise Stop(f"{path}: labels that are not in the list: {sorted(set(unknown))[:5]}")
    missing = [row[0] for row in rows if row[0] not in text_by_id]
    changed = [row[0] for row in rows if row[0] in text_by_id and text_hash(text_by_id[row[0]]) != row[4]]
    if missing or changed:
        raise Stop(f"{path} does not belong to this data source: {len(missing)} ids missing from it "
                   f"{missing[:3]}, {len(changed)} texts with another fingerprint {changed[:3]}")


def label_posts(selection, output, ask, *, label_set=BUILT_IN, model=DEFAULT_MODEL, seed=DEFAULT_SEED,
                max_attempts=DEFAULT_MAX_ATTEMPTS, limit=None, retry_unparsed=False, text_by_id=None,
                connect_retries=DEFAULT_CONNECT_RETRIES, connect_wait=DEFAULT_CONNECT_WAIT,
                log=print, sleep=time.sleep, clock=time.perf_counter, progress_every=PROGRESS_EVERY):
    """Label the selected posts that are not yet in the output file, appending and flushing one row per post.

    ask(payload) -> answer text is the only contact with the model, so the checks can replace it. With
    retry_unparsed, only the rows recorded as unparsed are asked again, and each new label replaces its row in place
    (whole file rewritten through a temporary file; order and row count unchanged). limit caps the posts asked in
    this run. When the server cannot be reached the run stops: the current post is not written. Returns a summary.
    """
    output = Path(output)
    names = label_names(label_set)
    version = label_set["prompt_version"]
    if text_by_id is None:
        text_by_id = {post["post_id"]: post["text"] for post in selection}
    rows, dropped = open_output(output, log)
    check_existing(rows, model=model, prompt_version=version, names=names, text_by_id=text_by_id, path=output)
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
               "new_unparsed": 0, "replaced": 0, "attempts": Counter(), "failures": Counter(),
               "post_seconds": 0.0, "stopped": None}
    log(f"{len(todo)} posts to label in this run ({summary['mode']}); {skipped} already in {output.name}")
    position = {row[0]: index for index, row in enumerate(rows)}
    started = clock()
    handle = None if retry_unparsed else output.open("a", encoding="utf-8", newline="")
    try:
        for number, post in enumerate(todo, start=1):
            post_started = clock()
            try:
                emotion, attempts, failures = label_one(
                    post["text"], ask, label_set=label_set, model=model, seed=seed, max_attempts=max_attempts,
                    connect_retries=connect_retries, connect_wait=connect_wait, sleep=sleep)
            except ServerUnavailable as exc:
                summary["stopped"] = f"the Ollama server cannot be reached ({exc}); post {post['post_id']} not written"
                log(f"stopped: {summary['stopped']}. Start the server and run the same command again to continue.")
                break
            summary["post_seconds"] += clock() - post_started
            summary["attempts"][attempts] += 1
            summary["failures"].update(failures)
            summary["labelled"] += 1
            summary["new_unparsed"] += emotion == UNPARSED
            row = [post["post_id"], emotion, model, version, text_hash(post["text"])]
            if retry_unparsed:
                if emotion != UNPARSED:
                    rows[position[post["post_id"]]] = row
                    write_atomic(output, b"".join(row_line(item).encode("utf-8") for item in [COLUMNS] + rows))
                    summary["replaced"] += 1
            else:
                handle.write(row_line(row))
                handle.flush()
                os.fsync(handle.fileno())
            if number % progress_every == 0 or number == len(todo):
                elapsed = clock() - started
                left = (len(todo) - number) * elapsed / number
                log(f"[{number}/{len(todo)}] {elapsed:.0f} s, {elapsed / number:.2f} s per post, about "
                    f"{left / 60:.0f} min left; unparsed in this run: {summary['new_unparsed']}")
    finally:
        if handle is not None:
            handle.close()
    summary["run_seconds"] = clock() - started
    return summary


def label_counts(rows, names):
    """Rows per label in label-list order, then unparsed."""
    counts = Counter(row[1] for row in rows)
    return {name: counts.get(name, 0) for name in names + [UNPARSED]}


# ---------------------------------------------------------------- review table

def review_rows(rows, names, size, seed):
    """Draw the review rows from a label file's rows, taking the labels in turn.

    Unparsed rows are left out. The rows of every label (file order) are shuffled with their own random.Random(seed);
    then, round after round, every label in list order gives its next row, a label with no rows left being skipped,
    until size rows are drawn or none are left. Returns [(post_id, label), ...] in drawing order."""
    queues = []
    for name in names:
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


def inside_data_or_outside_repository(path):
    """True when path is under the repository's data/ folder (ignored by Git) or outside the repository."""
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(REPO_DIR.resolve())
    except ValueError:
        return True
    return bool(relative.parts) and relative.parts[0] == "data"


# ---------------------------------------------------------------- commands

def log_run(record):
    """Append one line about a run to data/prelabel/runs.jsonl (no text of any post)."""
    path = REPO_DIR / RUN_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def command_label(args):
    base = loopback_base(args.host)  # refuse any other address before anything else happens
    label_set = load_label_set(args.labels_file)
    version = label_set["prompt_version"]
    kind = "sample" if args.sample else "full"
    source = args.source or (SAMPLE_SOURCE if args.sample else FULL_SOURCE)
    output = repo_path(args.output or default_output(kind, version))
    count = args.count if args.count is not False else (None if args.sample else DEFAULT_COUNT)
    if args.max_attempts < 1 or args.timeout <= 0 or args.connect_retries < 0 or args.connect_wait < 0:
        raise Stop("--max-attempts must be at least 1, --timeout above 0, --connect-retries and --connect-wait "
                   "at least 0")

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
    print(f"Ollama {server_version} at {client.base}, model {args.model}, prompt {version} "
          f"(fingerprint {prompt_fingerprint(label_set)[:16]}...)")

    source_path = repo_path(source)
    print(f"reading {source} ...", flush=True)
    posts = read_posts(source_path)
    text_by_id = dict(zip(posts["post_id"], posts["text"]))
    selection, drawn = select_posts(posts, count, args.btc_share, args.seed)
    print(f"selection: {drawn}")

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = label_posts(selection, output, client.ask, label_set=label_set, model=args.model, seed=args.seed,
                          max_attempts=args.max_attempts, limit=args.limit, retry_unparsed=args.retry_unparsed,
                          text_by_id=text_by_id, connect_retries=args.connect_retries,
                          connect_wait=args.connect_wait, log=lambda text: print(text, flush=True))
    rows = read_label_file(output)
    counts = label_counts(rows, label_names(label_set))
    selected_ids = {post["post_id"] for post in selection}
    btc = Counter("btc" if mentions_btc(text_by_id[row[0]]) else "other" for row in rows)
    average = summary["post_seconds"] / summary["labelled"] if summary["labelled"] else None
    print(f"\n{output.relative_to(REPO_DIR) if output.is_relative_to(REPO_DIR) else output}: {len(rows)} rows "
          f"({sum(row[0] in selected_ids for row in rows)} of the {len(selection)} selected posts)")
    print("rows per label:", json.dumps(counts))
    print(f"posts that mention Bitcoin: {btc['btc']}, other posts: {btc['other']}")
    print(f"this run: {summary['labelled']} posts in {summary['run_seconds']:.1f} s"
          + (f", {average:.2f} s per post" if average is not None else "")
          + f"; attempts {dict(sorted(summary['attempts'].items()))}; failed attempts {dict(summary['failures'])}"
          + (f"; replaced {summary['replaced']} unparsed rows" if args.retry_unparsed else ""))
    log_run({"started": started_at, "ended": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "output": str(output.name), "source": source, "ollama": server_version, "model": args.model,
             "prompt_version": version, "selection": drawn, "mode": summary["mode"], "to_label": summary["to_label"],
             "labelled": summary["labelled"], "new_unparsed": summary["new_unparsed"],
             "replaced": summary["replaced"], "post_seconds": round(summary["post_seconds"], 3),
             "run_seconds": round(summary["run_seconds"], 3),
             "attempts": {str(key): value for key, value in summary["attempts"].items()},
             "failures": dict(summary["failures"]), "dropped_partial_line_bytes": summary["dropped_partial_line_bytes"],
             "stopped": summary["stopped"], "rows_in_file": len(rows), "rows_per_label": counts})
    return 3 if summary["stopped"] else 0


def command_review_template(args):
    label_set = load_label_set(args.labels_file)
    version = label_set["prompt_version"]
    labels_path = repo_path(args.labels or default_output("full", version))
    output = repo_path(args.output or default_output("review", version))
    rows = read_label_file(labels_path)
    if {row[3] for row in rows} - {version}:
        raise Stop(f"{labels_path} was not labelled with prompt version {version}")
    names = label_names(label_set)
    picked = review_rows(rows, names, args.size, args.seed)
    write_atomic(output, b"".join(row_line(row).encode("utf-8")
                                  for row in [REVIEW_COLUMNS] + [[post_id, label, ""] for post_id, label in picked]))
    available = label_counts(rows, names)
    drawn = Counter(label for _, label in picked)
    print(f"{output.name}: {len(picked)} rows drawn from {len(rows) - available[UNPARSED]} labelled rows of "
          f"{labels_path.name} ({available[UNPARSED]} unparsed left out)")
    print("rows per label in the table:", json.dumps({name: drawn.get(name, 0) for name in names}))
    print("rows per label in the label file:", json.dumps(available))
    return 0


def command_export_review(args):
    label_set = load_label_set(args.labels_file)
    version = label_set["prompt_version"]
    template = repo_path(args.template or default_output("review", version))
    labels_path = repo_path(args.labels or default_output("full", version))
    output = repo_path(args.output)
    if not inside_data_or_outside_repository(output):
        raise Stop(f"refusing {output}: the table with the original texts may only be written under data/ "
                   "(ignored by Git) or outside the repository")
    review = read_label_file(template, REVIEW_COLUMNS)
    hashes = {row[0]: row[4] for row in read_label_file(labels_path)}
    posts = read_posts(repo_path(args.source))
    text_by_id = dict(zip(posts["post_id"], posts["text"]))
    lines = [row_line(REVIEW_COLUMNS + ["text"])]
    for post_id, prelabel, human_label in review:
        text = text_by_id.get(post_id)
        if text is None or hashes.get(post_id) != text_hash(text):
            raise Stop(f"{post_id}: not in {args.source}, or its text does not match the label file's fingerprint")
        lines.append(row_line([post_id, prelabel, human_label, text]))
    output.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(output, "".join(lines).encode("utf-8"))
    print(f"wrote {len(review)} rows with their texts to {output}")
    return 0


def command_show_prompt(args):
    label_set = load_label_set(args.labels_file)
    payload = build_payload(POST_PLACEHOLDER, retry=False, label_set=label_set, model=DEFAULT_MODEL, seed=DEFAULT_SEED)
    print(f"prompt version {label_set['prompt_version']}, fingerprint {prompt_fingerprint(label_set)}\n")
    for message in payload["messages"]:
        print(f"--- {message['role']}\n{message['content']}")
    print(f"\n--- added to the system message when a post is asked again\n{retry_reminder(label_set['labels'])}")
    print(f"\n--- answer format\n{json.dumps(payload['format'])}\n--- options\n{json.dumps(payload['options'])}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description="Pre-label posts with emotions through the local Ollama server.")
    commands = parser.add_subparsers(dest="command", required=True)

    label = commands.add_parser("label", help="label posts (resumable)")
    label.add_argument("--sample", action="store_true",
                       help=f"label every post of the sample ({SAMPLE_SOURCE}) into {default_output('sample', PROMPT_VERSION)}")
    label.add_argument("--source", help=f"data file (default {FULL_SOURCE}, or the sample with --sample)")
    label.add_argument("--output", help="label file (default labels/emotions-prelabel-<version>.csv, or the sample's)")
    label.add_argument("--count", type=parse_count, default=False,
                       help=f"posts to draw, or 'all' (default {DEFAULT_COUNT}; 'all' with --sample)")
    label.add_argument("--btc-share", type=float, default=DEFAULT_BTC_SHARE,
                       help=f"share of drawn posts that mention Bitcoin (default {DEFAULT_BTC_SHARE})")
    label.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"seed (default {DEFAULT_SEED})")
    label.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default {DEFAULT_MODEL})")
    label.add_argument("--host", default=DEFAULT_HOST, help=f"Ollama server, loopback only (default {DEFAULT_HOST})")
    label.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                       help=f"seconds per request (default {DEFAULT_TIMEOUT:g})")
    label.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS,
                       help=f"questions per post before it is recorded as unparsed (default {DEFAULT_MAX_ATTEMPTS})")
    label.add_argument("--connect-retries", type=int, default=DEFAULT_CONNECT_RETRIES,
                       help=f"extra tries when the server cannot be reached (default {DEFAULT_CONNECT_RETRIES})")
    label.add_argument("--connect-wait", type=float, default=DEFAULT_CONNECT_WAIT,
                       help=f"seconds between those tries (default {DEFAULT_CONNECT_WAIT:g})")
    label.add_argument("--limit", type=int, help="label at most this many posts in this run")
    label.add_argument("--retry-unparsed", action="store_true", help="ask again only for the rows recorded as unparsed")
    label.add_argument("--labels-file", help="JSON file with another label list and prompt version")

    review = commands.add_parser("review-template", help="draw the review table from a label file")
    review.add_argument("--labels", help="label file (default labels/emotions-prelabel-<version>.csv)")
    review.add_argument("--output", help="review table (default labels/emotions-review-template.csv)")
    review.add_argument("--size", type=int, default=DEFAULT_REVIEW_SIZE, help=f"rows (default {DEFAULT_REVIEW_SIZE})")
    review.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"seed (default {DEFAULT_SEED})")
    review.add_argument("--labels-file", help="JSON file with another label list and prompt version")

    export = commands.add_parser("export-review", help="the review rows with their texts, under data/ only")
    export.add_argument("--template", help="review table (default labels/emotions-review-template.csv)")
    export.add_argument("--labels", help="label file (default labels/emotions-prelabel-<version>.csv)")
    export.add_argument("--source", default=FULL_SOURCE, help=f"data file (default {FULL_SOURCE})")
    export.add_argument("--output", default=REVIEW_EXPORT, help=f"output (default {REVIEW_EXPORT})")
    export.add_argument("--labels-file", help="JSON file with another label list and prompt version")

    show = commands.add_parser("show-prompt", help="print the fixed prompt, the answer format and the fingerprint")
    show.add_argument("--labels-file", help="JSON file with another label list and prompt version")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    commands = {"label": command_label, "review-template": command_review_template,
                "export-review": command_export_review, "show-prompt": command_show_prompt}
    try:
        return commands[args.command](args)
    except Stop as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
