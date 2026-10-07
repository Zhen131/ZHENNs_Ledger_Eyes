"""Project checks. Run through scripts/check.sh so the environment from scripts/env.sh is set.

    scripts/check.sh quick [-p NAME VALUE ...]
    scripts/check.sh full  [-p NAME VALUE ...]

quick: static checks, then the notebook runs three times on the bundled
       sample (default switches, every two-state switch flipped, default
       switches again); the first two saved briefs are checked, and the
       topics sections of the two default runs must be identical. A fourth
       run sets both thresholds of the why chapter to 0.5 and the too-few bar
       above the topic post table; the why section of every brief is checked.
       The fourth run also sets the too-few-bullish-or-bearish bar above the
       topic post table, and the calibration of chapter 10 is checked in
       every brief.
       Every brief must pass the format rules (docs/brief.schema.json); the
       first two are also broken on purpose (locked counterexamples, all of
       which must be rejected) and shown on the panel, built but not started.
       Before any run, the built-in drawer word list is checked against its
       rules and its pinned fingerprint, and drawer_matches against fixed
       sentences; the drawers section and the drawer rows of every brief are
       checked, and the two default runs must give the same drawers.
       The emotion pre-label script (scripts/prelabel_emotions.py) is checked
       without Ollama: its interface and pinned prompt, its requests, that it
       only talks to a loopback address, that it reads the sample like
       chapter 1, and its labelling loop driven by a fake model; the label
       files and the review table in the repository are checked statically.
       The emotion chapter (11) reads the sample's pre-labels in every run;
       its emotions section is checked in every brief and compared between
       the two default runs. Before any run, its switches, its explanation
       and its reading cells are checked, the latter on the sample with
       counterexamples (a changed fingerprint, the full pre-labels with the
       sample, a missing file) and a hand-written review table.
       The dated data chapter (12) reads the dated samples in samples/ with a
       small daily cap in every run; its dated_data section is checked in
       every brief (steps, time kinds, every layer's days, the cap, the
       Bitcoin rule, the news labels against a second count from the news
       sample) and compared between the runs. Before any run, its switches,
       the shape of the dated post table against docs/dated-posts.md and
       against broken tables, the way it reads, and its tweet cells on the
       sample (shuffled, split, cached, changed) are checked, and the samples
       are scanned for account names and links.
       The day by day chapter (13) scores every dated post and counts it per
       layer and UTC day, and finds the new words of every day; its daily and
       new_words sections are checked in every brief (every pile's
       identities, the sums over the days, the calibration, the event days,
       every new word against the rules) and compared between the two default
       runs. A fifth run reads a copy of the tweet sample with made-up words
       added on one day, which the new words must report (the spike run).
       The two layers and validation chapters (14 and 15) read made-up prices
       and a part of the Fear and Greed Index from samples/ (or nothing) in
       every run, so the quick check connects to nothing; their layers and
       validation sections are checked in every brief (the overlap, the
       opposite days, the model on the news headlines, where the price and the
       index come from, every coefficient recomputed from the files the run
       read) and compared between the two default runs. Before any run, their
       switches, where chapter 15 may connect, and its parsers and caches with
       fake sources are checked. A sixth run reads no prices at all, which the
       notebook must take as "not obtained".
full:  static checks, then one run on the full dataset with the default
       switches; prints the key numbers from the brief. The calibrated
       bullish share must narrow the gap to the labels in the overall fair
       comparison. The drawers are checked and printed as well. The full
       emotion label file is compared with the full dataset (ids and text
       fingerprints) and its rows per label are printed. The emotion chapter
       reads the full pre-labels; every emotion network must beat both
       baselines, and its numbers, emotions per group and example posts are
       printed. The dated data chapter reads the downloaded copies in
       data/raw/dlt-tweets/ and data/raw/dlt-news/; its dated_data section
       is checked and its numbers are printed. The daily and new_words
       sections are checked, and their numbers, every new word day by day and
       the event days are printed. Chapter 15 reads the BTC daily closes and the
       Fear and Greed Index from the web (Yahoo Finance through yfinance, then
       Coinbase, then Stooq; alternative.me) the first time and from its caches
       in WORK_DIR/cache/ afterwards; the layers and validation sections are
       checked (every coefficient recomputed from the cache) and printed.

Both checks always set LAUNCH_PANEL to False: the panel is built, never started.

-p NAME VALUE overrides a switch of the notebook's parameters cell for every
run (VALUE is read as a Python literal when possible, otherwise as text).
Exit code 0 means every check passed.
"""

import argparse
import ast
import builtins
import contextlib
import copy
import csv
import hashlib
import io
import itertools
import json
import os
import random
import re
import shutil
import subprocess
import symtable
import sys
import time
from collections import Counter
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR / "scripts"))

import jupytext  # noqa: E402
import prelabel_emotions as prelabel  # noqa: E402
from notebook_build import NOTEBOOK, SOURCE, build_notebook_text  # noqa: E402

REQUIREMENTS = REPO_DIR / "requirements.txt"
SAMPLE_SOURCE = "samples/stocktwits-crypto-sample.xlsx"  # relative to the repository, as the notebook sees it
FULL_SOURCE = "data/raw/st-data-full.xlsx"
TOP_LEVEL_FIELDS = ["format_version", "generated_at", "scope", "dataset", "sentiment", "topics", "why", "drawers",
                    "emotions", "dated_data", "daily", "new_words", "layers", "validation"]
SCOPE_FIELDS = ["source", "granularity", "coins", "n_classes"]
REQUIRED_SWITCHES = ["DATA_SOURCE", "WORK_DIR", "RANDOM_SEED", "SAMPLE_PER_CLASS",
                     "N_CLASSES", "BTC_ONLY", "EMOJI_TO_WORDS", "KEEP_NEGATIONS"]
SENTIMENT_SWITCHES = ["BPE_VOCAB_SIZE", "VOCAB_SIZE", "EPOCHS", "BATCH_SIZE"]
SENTIMENT_MODELS = ["count_network", "tfidf_network"]  # the two networks of chapter 7; more may be added later
SENTIMENT_BASELINES = ["always_majority", "uniform_random"]
SENTIMENT_SUBSETS = ["all_test", "test_mentions_btc"]
SPLIT_PARTS = ["train", "validation", "test"]
TOPIC_SWITCHES = ["TOPIC_SCOPE", "TOPIC_MAX_POSTS", "N_TOPICS"]
TOPIC_CLASSES = ["0", "1", "2"]  # the topic table keeps all three classes, whatever N_CLASSES says
SELECTION_WORDS = {"bitcoin", "btc"}  # with TOPIC_SCOPE "btc" every topic post has one; never part of a topic name
WHY_SWITCHES = ["BULLISH_THRESHOLD", "BEARISH_THRESHOLD", "MIN_TOPIC_POSTS"]
MODEL_CALLS = ["bullish", "bearish", "unsure"]  # the model's calls in the why section
FAIR_CLASSES = ["0", "2"]  # the fair comparison uses posts labelled bearish or bullish only
THRESHOLD_SETTINGS = {"BULLISH_THRESHOLD": 0.5, "BEARISH_THRESHOLD": 0.5}  # fourth quick run; plus MIN_TOPIC_POSTS
THRESHOLD_LABEL = "threshold run"
# Chapter 10's calibration: calls at 0.5, the recognition rates of chapter 8 and the too-few-bullish-or-bearish mark.
CALIBRATION_SWITCHES = ["MIN_RATE_GAP", "MIN_BULLISH_BEARISH_POSTS"]
HALF_CALLS = ["bullish", "bearish"]  # the calls at 0.5: every post on one side
CALIBRATION_STATUSES = ["applied", "rates_missing", "rate_gap_too_small", "three_classes"]
RATE_SUBSETS = {"btc": "test_mentions_btc", "all": "all_test"}  # TOPIC_SCOPE -> evaluation subset of the rates
# Chapter 9's drawers: the word list, the rule that puts posts into drawers, and the drawers in the brief.
DRAWER_SWITCHES = ["DRAWER_WORDS_FILE", "N_OTHER_GROUPS"]
BUILT_IN_SOURCE = "built-in"  # drawers.word_list_source when DRAWER_WORDS_FILE is ""
# Locked: names of single events that the built-in word list must never contain, as a whole word anywhere.
DRAWER_BANNED_NAMES = ["luna", "terra", "ust", "celsius", "3ac", "ftx"]
# Locked: the fingerprint of every released version of the built-in word list. A new version adds a line and no
# line ever changes, so a word of a released version cannot change without a new version number.
DRAWER_FINGERPRINTS = {
    "v1": "610a3dbfa89f30af",
}
# Locked (only add): sentences and the drawers they must go into with the built-in word list.
DRAWER_PROBES = [
    ("SEC sues another exchange", ["regulation", "exchanges"]),
    ("wait a sec, btc is pumping", []),
    ("The Fed will hike again", ["central_banks"]),
    ("I'm fed up with this chop", []),
    ("$ETH.X looking strong", ["other_coins"]),
    ("my method works every time", []),
    ("what a steal at this price", []),
    ("exchange got hacked, withdrawals frozen", ["hacks_scams", "exchanges"]),
    ("new all time high incoming", ["charts"]),
    ("customer support never answers", []),
    ("new  all\ttime   high", ["charts"]),  # a space in a phrase matches any whitespace
    ("RUG PULL", ["hacks_scams"]),  # case is ignored
    ("STAR WARS marathon tonight", []),  # an excluded phrase, in capitals
    ("", []),  # an empty text goes into no drawer
]
# A small word list read through DRAWER_WORDS_FILE, and the drawers its sentences must go into.
DRAWER_FILE_PROBE = {"version": "probe-1", "drawers": [
    {"drawer": "moon", "name": "Moon shots", "words": ["moon shot"], "exclude": ["no moon shot"]}]}
DRAWER_FILE_PROBES = [("moon shot soon", ["moon"]), ("no moon shot today", [])]
DRAWER_QUOTES = 5  # spot-check quotes per drawer, at most
DRAWER_STABLE_ROW_FIELDS = ["drawer", "name", "n_posts", "too_few_posts", "label_counts", "label_bullish_share",
                            "label_vs_overall_pp", "too_few_bullish_bearish"]  # why.drawers fields that ignore training
# Chapter 16: the brief's format rules, its documentation and the panel.
SCHEMA_FILE = REPO_DIR / "docs" / "brief.schema.json"  # the format rules; the notebook carries an identical copy
FORMAT_DOC = REPO_DIR / "docs" / "brief-format.md"  # the field-by-field documentation of the format
DOC_TABLE_HEADER = "| Field | Type | Required | Null | Meaning |"  # header of every field table in FORMAT_DOC
PANEL_SWITCH = "LAUNCH_PANEL"
CHECK_SETTINGS = {PANEL_SWITCH: False}  # every run of both checks: the panel is built, never started
PANEL_TABLES = ["scope", "evaluation", "topics", "why"]  # the panel's tables, in tab order
PANEL_MORE_TABLES = ["calibration", "calibrated", "drawers", "emotions", "new_words", "validation",
                     "sources"]  # the panel's tables after the first four, in tab order
PANEL_CHART_LABELS = ["Bullish share by day", "The two layers side by side"]  # the panel's two charts, in tab order
PANEL_EMOTION_COLUMNS = ["row", "name", "posts"]  # the emotion table's first columns; one column per class follows
PANEL_DASH = "—"  # what the panel shows where the brief has null
LOOPBACK_HOST = "127.0.0.1"  # the only address the panel may listen on

# Text scan. Each pattern is written so that it does not match its own source line
# (for example "c[l]aude" matches the word but not this file).
# Private numbering is caught in the forms the log uses: file-name prefixes (two digits,
# optionally a letter A-D or Q, optionally a round suffix "-R" or "-<number>", then "_W"
# and a week), review codes (two digits, a letter A-D, "-R"), week codes and question codes.
# A bare "two digits and a letter" is not flagged on purpose, because it collides with
# normal notation such as 50D (50 days) or 10B (10 billion).
NUMBERING_PATTERNS = [
    (r"\b[0-9]{2}[A-DQ]?(?:-(?:R|[0-9]+))?_W[0-9]{1,2}\b", "private log file prefix (two digits, optional letter and round, _W and a week)"),
    (r"\b[0-9]{2}[A-D]-R\b", "private review code (two digits, a letter A-D and -R)"),
    (r"\bW[0-9]{2}\b", "private week code (W and two digits)"),
    (r"\bQ[0-9]{2}\b", "private question code (Q and two digits)"),
    ("\u7b2c\\s*[0-9]+\\s*\u5468", "private week number written in Chinese"),
]
FORBIDDEN_PATTERNS = NUMBERING_PATTERNS + [
    (r"(?i)\bc[l]aude\b", "assistant name"),
    (r"(?i)\banthrop[i]c\b", "assistant vendor name"),
    (r"(?i)\bchat[g]pt\b", "assistant name"),
    (r"(?i)\bopen[a]i\b", "assistant vendor name"),
    (r"(?i)\bcopi[l]ot\b", "assistant name"),
    (r"(?i)\bg[p]t-?[0-9]", "model name"),
    (r"\bA[I]\b", "the abbreviation for artificial intelligence"),
    (r"(?i)co-authored-[b]y", "co-author trailer"),
    (r"/User[s]/|zhuzhen[0-9]", "local home path or user name"),
]
# Self-test of the numbering patterns. Codes that must be caught are written in pieces
# (joined at run time) so that this file does not trip its own scan.
NUMBERING_MUST_MATCH = [
    ("31B", "_W20"),         # contract file prefix
    ("31B-", "R_W20"),       # review file prefix
    ("31C", "_W20"),         # report file prefix
    ("31D", "_W20"),         # acceptance file prefix
    ("09D-3", "_W15"),       # acceptance file prefix with a round number
    ("01D-1", "_W11"),       # same, single-digit week
    ("06", "_W19"),          # file or folder prefix without a letter
    ("31B-", "R"),           # review code on its own
    ("see W", "20 log"),     # week code
    ("Q", "01 list"),        # question code
    ("\u7b2c 20 \u5468",),  # week number written in Chinese
]
NUMBERING_MUST_NOT_MATCH = [
    "50D moving average", "30D change", "7D volume", "200D", "10B market cap", "a 3D chart",
    "W2 form", "Q4 2021", "MA50", "batch size 32", "RSI_14", "top_W",
]


def check_numbering_rules(report):
    """The numbering patterns must catch the log's code forms and must not catch normal notation."""
    def caught(text):
        return any(re.search(pattern, text) for pattern, _ in NUMBERING_PATTERNS)
    missed = ["".join(parts) for parts in NUMBERING_MUST_MATCH if not caught("".join(parts))]
    wrongly = [text for text in NUMBERING_MUST_NOT_MATCH if caught(text)]
    detail = "; ".join(([f"not caught: {missed}"] if missed else []) + ([f"wrongly caught: {wrongly}"] if wrongly else []))
    report.check(not missed and not wrongly,
                 f"numbering scan self-test ({len(NUMBERING_MUST_MATCH)} codes must be caught, "
                 f"{len(NUMBERING_MUST_NOT_MATCH)} normal notations must not)", detail)


CJK = re.compile("[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uff00-\uffef]")
SKIP_DIRS = {".git", ".venv", "data", "__pycache__", ".ipynb_checkpoints"}
BINARY_SUFFIXES = {".xlsx", ".parquet", ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip"}

# Cleaning examples: the sentences the notebook must clean and put into the brief,
# with the words that must survive (with repeats) under each product rule.
NEGATION_TOKENS = {
    "I don't think btc will pump, not bullish at all 😭": ["not", "not"],
    "Never selling my bitcoin 🚀🚀": ["never"],
    "No way it can't go higher, I won't sell 😂": ["no", "not", "not"],
    "Nor do I care about this dip 💎🙌": ["nor"],
}
EMOJI_TOKENS = {
    "I don't think btc will pump, not bullish at all 😭": ["loudly", "cry", "face"],
    "Never selling my bitcoin 🚀🚀": ["rocket", "rocket"],
    "No way it can't go higher, I won't sell 😂": ["face", "tear", "joy"],
    "Nor do I care about this dip 💎🙌": ["gem", "stone", "raising", "hand"],
}
EMOJI_ONLY_WORDS = ["loudly", "rocket", "joy", "gem", "stone", "raising"]  # words that can only come from the emoji


class Report:
    """Collects PASS and FAIL lines and prints them as it goes."""

    def __init__(self):
        self.failures = []

    def check(self, ok, label, detail=""):
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {label}" + (f": {detail}" if detail else ""), flush=True)
        if not ok:
            self.failures.append(label)
        return ok

    def info(self, text):
        print(f"       {text}", flush=True)


# ---------------------------------------------------------------- static checks

def normalise(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def read_requirements():
    pins = {}
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, _, pinned = line.partition("==")
            pins[normalise(name)] = pinned.strip()
    return pins


def check_requirements(report):
    pins = read_requirements()
    wrong = []
    for name, pinned in pins.items():
        try:
            installed = version(name)
        except PackageNotFoundError:
            installed = None
        if installed != pinned:
            wrong.append(f"{name} pinned {pinned}, installed {installed}")
    report.check(not wrong and bool(pins), "requirements.txt matches the installed packages", "; ".join(wrong))

    source = SOURCE.read_text(encoding="utf-8")
    notebook_pins = dict((normalise(name), pinned) for name, pinned in
                         re.findall(r'"([A-Za-z0-9_.\-]+)==([^"]+)"', source))
    mismatched = [f"{name}=={pinned} (requirements.txt: {pins.get(name)})"
                  for name, pinned in notebook_pins.items() if pins.get(name) != pinned]
    report.check(bool(notebook_pins) and not mismatched,
                 "notebook install pins match requirements.txt", "; ".join(mismatched))


def check_notebook_files(report):
    expected = build_notebook_text()
    actual = NOTEBOOK.read_text(encoding="utf-8") if NOTEBOOK.exists() else ""
    detail = ""
    if expected != actual:
        detail = "regenerate with scripts/build_notebook.sh"
        try:
            built = json.loads(expected)["cells"]
            stored = json.loads(actual)["cells"]
            for index, (left, right) in enumerate(zip(built, stored)):
                if (left["cell_type"], left["source"]) != (right["cell_type"], right["source"]):
                    detail += f"; first differing cell: {index}"
                    break
            else:
                if len(built) != len(stored):
                    detail += f"; cell count {len(built)} in source, {len(stored)} in .ipynb"
                else:
                    detail += "; cells match, metadata or outputs differ"
        except (ValueError, KeyError):
            detail += "; .ipynb is not valid JSON"
    report.check(expected == actual, "source file and .ipynb are consistent", detail)

    try:
        cells = json.loads(actual)["cells"]
    except (ValueError, KeyError):
        cells = None
    with_results = [] if cells is None else [
        index for index, cell in enumerate(cells)
        if cell.get("outputs") or cell.get("execution_count") is not None]
    report.check(cells is not None and not with_results, ".ipynb carries no run results",
                 f"cells with outputs or execution counts: {with_results}" if with_results else "")


def read_switches(report):
    """Parse the parameters cell; return (defaults, other values of the two-state switches)."""
    notebook = jupytext.read(SOURCE, fmt="py:percent")
    cells = [cell for cell in notebook.cells if "parameters" in cell.metadata.get("tags", [])]
    if not report.check(len(cells) == 1, "exactly one cell is tagged parameters", f"found {len(cells)}"):
        return {}, {}
    defaults, others, bad = {}, {}, []
    for line in cells[0].source.splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([A-Z][A-Z0-9_]*) = (.+?)  # default: (.+)", line)
        if not match:
            bad.append(line)
            continue
        name, value, comment = match.groups()
        try:
            defaults[name] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            bad.append(line)
            continue
        two_state = re.search(r"two-state, other: (\S+)", comment)
        if two_state:
            others[name] = ast.literal_eval(two_state.group(1))
    report.check(not bad, "every switch is a plain assignment with a '# default: ...' comment",
                 "; ".join(bad))
    absent = [name for name in REQUIRED_SWITCHES if name not in defaults]
    report.check(not absent, "all switches of the interface are present", ", ".join(absent))
    report.info("two-state switches and their other values: " + ", ".join(f"{k}={v!r}" for k, v in others.items()))
    return defaults, others


def text_files():
    for path in sorted(REPO_DIR.rglob("*")):
        relative = path.relative_to(REPO_DIR)
        if path.is_file() and not (set(relative.parts) & SKIP_DIRS) and path.suffix.lower() not in BINARY_SUFFIXES:
            yield path, relative


def check_text(report):
    hits = []
    for path, relative in text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern, meaning in FORBIDDEN_PATTERNS:
            for number, line in enumerate(text.splitlines(), start=1):
                if re.search(pattern, line):
                    hits.append(f"{relative}:{number} ({meaning})")
    report.check(not hits, "no private codes, assistant names or local paths in the repository text",
                 "; ".join(hits[:10]))

    cjk_outside = []
    for path, relative in text_files():
        if relative.parts[0] == "notebook":
            continue  # notebook text is checked cell by cell below
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        cjk_outside += [f"{relative}:{number}" for number, line in enumerate(text.splitlines(), start=1)
                        if CJK.search(line)]
    report.check(not cjk_outside, "no Chinese text outside the notebook's explanation cells",
                 "; ".join(cjk_outside[:10]))

    notebook = jupytext.read(SOURCE, fmt="py:percent")
    code_cjk, layout = [], []
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            if CJK.search(cell.source):
                code_cjk.append(index)
        elif cell.cell_type == "markdown":
            lines = cell.source.splitlines()
            if lines.count("***") != 1:
                layout.append(f"cell {index}: needs exactly one '***' line")
                continue
            english = "\n".join(lines[:lines.index("***")])
            chinese = "\n".join(lines[lines.index("***") + 1:])
            if CJK.search(english) or not re.search(r"[A-Za-z]{3}", english) or not CJK.search(chinese):
                layout.append(f"cell {index}: English before '***', Chinese after")
    report.check(not code_cjk, "code cells (and their comments) contain no Chinese", f"cells {code_cjk}")
    report.check(not layout, "every explanation cell is English, a '***' line, then Chinese", "; ".join(layout))


# ---------------------------------------------------------------- running the notebook

def run_notebook(report, label, parameters, output):
    """Execute the .ipynb with papermill; return (ok, seconds, '[load]' lines)."""
    import papermill
    from papermill.exceptions import PapermillExecutionError

    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    error = ""
    try:
        papermill.execute_notebook(str(NOTEBOOK), str(output), parameters=parameters, kernel_name="python3",
                                   cwd=str(REPO_DIR), progress_bar=False, log_output=False)
    except PapermillExecutionError as exc:
        error = f"{exc.ename}: {exc.evalue}"
    except Exception as exc:  # kernel start-up problems and the like
        error = f"{type(exc).__name__}: {exc}"
    seconds = time.perf_counter() - started
    load_lines = []
    if output.exists():
        executed = json.loads(output.read_text(encoding="utf-8"))
        for cell in executed["cells"]:
            for out in cell.get("outputs", []):
                text = out.get("text", "")
                text = "".join(text) if isinstance(text, list) else text
                load_lines += [line for line in text.splitlines() if line.startswith("[load]")]
    report.check(not error, f"{label}: notebook runs from top to bottom", error or f"{seconds:.1f} s")
    for line in load_lines:
        report.info(line)
    return not error, seconds, load_lines


def shown(path):
    """Path for messages: relative to the repository when inside it, absolute otherwise."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO_DIR))
    except ValueError:
        return str(path)


def reject_constant(name):
    raise ValueError(f"non-standard JSON constant {name}")


def read_brief(report, label, path):
    if not path.exists():
        report.check(False, f"{label}: brief saved", f"{shown(path)} not found")
        return None
    try:
        brief = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except ValueError as exc:
        report.check(False, f"{label}: brief is valid JSON without NaN or infinity", str(exc))
        return None
    report.check(True, f"{label}: brief saved as valid JSON without NaN or infinity", shown(path))
    return brief


def check_brief(report, label, brief, switches):
    absent = [field for field in TOP_LEVEL_FIELDS if field not in brief]
    report.check(not absent, f"{label}: all top-level fields present", ", ".join(absent))
    dataset = brief.get("dataset")
    report.check(isinstance(dataset, dict) and bool(dataset), f"{label}: dataset section has content")
    dataset = dataset if isinstance(dataset, dict) else {}

    generated = brief.get("generated_at")
    try:
        ok_time = isinstance(generated, str) and generated.endswith("Z") and bool(datetime.fromisoformat(generated))
    except ValueError:
        ok_time = False
    report.check(ok_time and isinstance(brief.get("format_version"), str) and bool(brief.get("format_version")),
                 f"{label}: format_version is set and generated_at is a UTC ISO 8601 time", f"{generated!r}")

    scope = brief.get("scope") if isinstance(brief.get("scope"), dict) else {}
    expected_scope = {
        "source": switches["DATA_SOURCE"],
        "granularity": "batch",
        "coins": "btc_only" if switches["BTC_ONLY"] else "all",
        "n_classes": switches["N_CLASSES"],
    }
    wrong = [f"{key}={scope.get(key)!r} (expected {value!r})" for key, value in expected_scope.items()
             if scope.get(key) != value]
    report.check(not wrong, f"{label}: scope matches the switches", "; ".join(wrong))

    # Stable ids: unique in df_all.
    n_posts, n_ids = dataset.get("n_posts"), dataset.get("n_unique_post_ids")
    report.check(isinstance(n_posts, int) and n_posts > 0 and n_ids == n_posts,
                 f"{label}: post_id is unique in df_all", f"{n_ids} ids for {n_posts} posts")
    counts = dataset.get("class_counts") or {}
    report.check(sum(counts.values()) == n_posts, f"{label}: class counts add up to the number of posts",
                 f"{counts}")

    sample = dataset.get("sample") or {}
    sample_counts = sample.get("class_counts") or {}
    expected_labels = ["0", "2"] if switches["N_CLASSES"] == 2 else ["0", "1", "2"]
    report.check(sorted(sample_counts) == expected_labels and len(set(sample_counts.values())) == 1
                 and sum(sample_counts.values()) == sample.get("n_posts"),
                 f"{label}: working sample has the same number of posts in every class in use", f"{sample_counts}")
    if switches["BTC_ONLY"]:
        report.check(sample.get("n_mentions_btc") == sample.get("n_posts"),
                     f"{label}: with BTC_ONLY every sampled post mentions bitcoin or btc")

    cleaning = dataset.get("cleaning") or {}
    report.check(cleaning.get("clean_texts_matches_steps") is True,
                 f"{label}: clean_texts gives exactly the step-by-step clean_text")
    report.check(cleaning.get("emoji_to_words") == switches["EMOJI_TO_WORDS"]
                 and cleaning.get("keep_negations") == switches["KEEP_NEGATIONS"],
                 f"{label}: cleaning switches recorded as set")
    check_examples(report, label, cleaning.get("examples") or [], switches)


def teacher_clean(text):
    """The course recipe, step by step, with the plain NLTK stop word list and no emoji handling."""
    from bs4 import BeautifulSoup
    import contractions
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    text = text.lower()
    text = " ".join(text.strip().split())
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = contractions.fix(text)
    stop_words = set(stopwords.words("english"))
    text = " ".join(word for word in word_tokenize(text) if word.lower() not in stop_words)
    lemmatizer = WordNetLemmatizer()
    return " ".join(lemmatizer.lemmatize(word) for word in word_tokenize(text))


def check_examples(report, label, examples, switches):
    by_raw = {example.get("raw"): example.get("clean", "") for example in examples if isinstance(example, dict)}
    absent = [raw for raw in NEGATION_TOKENS if raw not in by_raw]
    if not report.check(not absent, f"{label}: cleaning examples present in the brief", f"missing {absent}"):
        return
    problems = []
    for raw, clean in by_raw.items():
        if raw not in NEGATION_TOKENS:
            continue
        tokens = Counter(clean.split())
        if switches["KEEP_NEGATIONS"]:
            need = Counter(NEGATION_TOKENS[raw])
            if need - tokens:
                problems.append(f"negation lost in {clean!r} (needs {sorted((need - tokens).elements())})")
        else:
            kept = [word for word in ("not", "no", "nor") if tokens[word]]
            if kept:
                problems.append(f"stop-listed negation {kept} kept in {clean!r}")
        if switches["EMOJI_TO_WORDS"]:
            need = Counter(EMOJI_TOKENS[raw])
            if need - tokens:
                problems.append(f"emoji not turned into words in {clean!r} (needs {sorted((need - tokens).elements())})")
        else:
            leaked = [word for word in EMOJI_ONLY_WORDS if tokens[word]]
            if leaked:
                problems.append(f"emoji words {leaked} present in {clean!r}")
    rules = (f"KEEP_NEGATIONS={switches['KEEP_NEGATIONS']}, EMOJI_TO_WORDS={switches['EMOJI_TO_WORDS']}")
    report.check(not problems, f"{label}: cleaning examples follow the switches ({rules})", "; ".join(problems))

    if not switches["KEEP_NEGATIONS"] and not switches["EMOJI_TO_WORDS"]:
        differ = [f"{clean!r} vs course recipe {teacher_clean(raw)!r}"
                  for raw, clean in by_raw.items() if raw in NEGATION_TOKENS and clean != teacher_clean(raw)]
        report.check(not differ, f"{label}: with both switches off the examples equal the course recipe",
                     "; ".join(differ))
        for raw, clean in by_raw.items():
            if raw in NEGATION_TOKENS:
                kept = [word for word in ("never", "no", "not", "nor") if word in clean.split()]
                report.info(f"switches off: {raw!r} -> {clean!r}; negation words left: {kept}")


# ---------------------------------------------------------------- sentiment chapters (4 to 8)

def check_sentiment_switches(report, defaults):
    absent = [name for name in SENTIMENT_SWITCHES if name not in defaults]
    report.check(not absent, "switches of the sentiment chapters are present", ", ".join(absent))


def is_score(value):
    """A score in the brief is a number between 0 and 1, or null when it cannot be computed."""
    return value is None or (isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1)


def is_count(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def check_sentiment(report, label, brief, switches, full_run):
    """Structure of the sentiment section; on the full run also the bar: every model beats both baselines."""
    sentiment = brief.get("sentiment")
    if not report.check(isinstance(sentiment, dict) and bool(sentiment), f"{label}: sentiment section has content"):
        return
    classes = ["0", "2"] if switches["N_CLASSES"] == 2 else ["0", "1", "2"]
    report.check(sentiment.get("classes") == classes, f"{label}: sentiment uses the classes of N_CLASSES",
                 f"{sentiment.get('classes')} (expected {classes})")

    # Chapter 4: tokenization, shown only.
    tokenization = sentiment.get("tokenization") if isinstance(sentiment.get("tokenization"), dict) else {}
    nltk_part = tokenization.get("nltk_word_tokenize") if isinstance(tokenization.get("nltk_word_tokenize"), dict) else {}
    bpe_part = tokenization.get("bpe") if isinstance(tokenization.get("bpe"), dict) else {}
    sizes = [nltk_part.get("vocab_size"), bpe_part.get("vocab_size")]
    lengths = [nltk_part.get("avg_tokens_per_post"), bpe_part.get("avg_tokens_per_post")]
    # The BPE vocabulary always keeps every single character seen in training and the special tokens,
    # and adds merged pieces up to BPE_VOCAB_SIZE. So it holds at most BPE_VOCAB_SIZE pieces, except when
    # the single characters and special tokens alone are more: then it holds exactly those, nothing merged.
    limit = switches.get("BPE_VOCAB_SIZE")
    parts = [bpe_part.get("n_special_tokens"), bpe_part.get("n_single_characters"), bpe_part.get("n_merged_pieces")]
    bpe_ok = (isinstance(limit, int) and not isinstance(limit, bool) and is_count(sizes[1])
              and all(is_count(part) for part in parts) and parts[0] > 0 and parts[1] > 0
              and sum(parts) == sizes[1]
              and sizes[1] <= max(limit, parts[0] + parts[1])
              and (sizes[1] <= limit or parts[2] == 0))
    report.check(tokenization.get("used_by_models") is False
                 and all(is_count(size) and size > 0 for size in sizes)
                 and all(isinstance(length, (int, float)) and not isinstance(length, bool) and length > 0
                         for length in lengths)
                 and bpe_ok,
                 f"{label}: tokenization comparison recorded (vocabulary sizes and tokens per post, not used by models; "
                 "BPE vocabulary at most BPE_VOCAB_SIZE, or exactly the single characters and special tokens "
                 "when those alone are more)",
                 f"vocabulary sizes {sizes}, tokens per post {lengths}, BPE_VOCAB_SIZE {limit!r}, "
                 f"BPE special / single characters / merged {parts}")

    # Chapter 5: the split.
    split = sentiment.get("split") if isinstance(sentiment.get("split"), dict) else {}
    sizes = split.get("n_posts") if isinstance(split.get("n_posts"), dict) else {}
    class_counts = split.get("class_counts") if isinstance(split.get("class_counts"), dict) else {}
    sample = (brief.get("dataset") or {}).get("sample") or {}
    split_ok = sorted(sizes) == sorted(SPLIT_PARTS) and all(is_count(sizes[part]) for part in SPLIT_PARTS)
    split_ok = split_ok and sum(sizes.values()) == sample.get("n_posts")
    for part in SPLIT_PARTS if split_ok else []:
        counts = class_counts.get(part) if isinstance(class_counts.get(part), dict) else {}
        split_ok = split_ok and sorted(counts) == classes and all(is_count(count) and count > 0 for count in counts.values())
        split_ok = split_ok and sum(counts.values()) == sizes[part]
    if split_ok:  # 80 / 10 / 10, up to rounding inside each class
        split_ok = abs(sizes["train"] - 0.8 * sample["n_posts"]) <= len(classes) and sizes["train"] > sizes["test"] > 0
    report.check(split_ok, f"{label}: training, validation and test sets add up to the working sample, "
                 "about 80/10/10, every class in every part", f"{sizes} {class_counts}")
    n_test = sizes.get("test")
    btc_counts = split.get("n_mentions_btc") if isinstance(split.get("n_mentions_btc"), dict) else {}
    n_test_btc = btc_counts.get("test")
    report.check(is_count(n_test_btc) and n_test_btc <= (n_test if is_count(n_test) else -1),
                 f"{label}: test posts that mention bitcoin are counted", f"{n_test_btc} of {n_test}")

    # Chapter 6: vectorizers learn on the training set only.
    vectors = sentiment.get("vectorizers") if isinstance(sentiment.get("vectorizers"), dict) else {}
    vocab_sizes = [(vectors.get(name) or {}).get("vocab_size") if isinstance(vectors.get(name), dict) else None
                   for name in ("count", "tfidf")]
    report.check(vectors.get("fitted_on") == "train" and vectors.get("fitted_on_n_posts") == sizes.get("train")
                 and isinstance(switches.get("VOCAB_SIZE"), int)
                 and all(is_count(size) and 0 < size <= switches["VOCAB_SIZE"] for size in vocab_sizes),
                 f"{label}: both vectorizers learned at most VOCAB_SIZE words from the training set only",
                 f"fitted on {vectors.get('fitted_on')!r} ({vectors.get('fitted_on_n_posts')} posts), "
                 f"vocabulary sizes {vocab_sizes}")

    # Chapters 7 and 8: models, evaluation table, confusion matrices, choice and scoring function.
    models = sentiment.get("models") if isinstance(sentiment.get("models"), list) else []
    report.check(all(name in models for name in SENTIMENT_MODELS),
                 f"{label}: the word-count and TF-IDF networks are both listed", f"{models}")
    evaluation = sentiment.get("evaluation") if isinstance(sentiment.get("evaluation"), dict) else {}
    rows = [row for row in evaluation.get("rows") or [] if isinstance(row, dict)]
    found = Counter((row.get("model"), row.get("subset")) for row in rows)
    expected = [(name, subset) for name in models + SENTIMENT_BASELINES for subset in SENTIMENT_SUBSETS]
    wrong = [pair for pair in expected if found[pair] != 1]
    report.check(not wrong and len(rows) == len(expected),
                 f"{label}: evaluation table has one row per model and per baseline (always majority, uniform random) "
                 "for all test posts and for test posts that mention bitcoin",
                 f"missing or repeated: {wrong}; {len(rows)} rows for {len(expected)} expected")

    problems = []
    for row in rows:
        name = f"{row.get('model')}/{row.get('subset')}"
        n = row.get("n")
        expected_n = n_test if row.get("subset") == "all_test" else n_test_btc
        if not is_count(n) or n != expected_n:
            problems.append(f"{name}: {n} posts, expected {expected_n}")
            continue
        per_class = row.get("per_class") if isinstance(row.get("per_class"), dict) else {}
        if sorted(per_class) != classes or not all(isinstance(scores, dict) for scores in per_class.values()):
            problems.append(f"{name}: classes {sorted(per_class)}")
            continue
        supports = [scores.get("support") for scores in per_class.values()]
        if not all(is_count(support) for support in supports) or sum(supports) != n:
            problems.append(f"{name}: class supports {supports} do not add up to {n}")
        scores = [row.get("accuracy"), row.get("macro_f1")] + [
            scores.get(measure) for scores in per_class.values() for measure in ("precision", "recall", "f1")]
        if not all(is_score(score) for score in scores):
            problems.append(f"{name}: a score is neither a number in [0, 1] nor null: {scores}")
        if n == 0 and (row.get("accuracy") is not None or row.get("macro_f1") is not None):
            problems.append(f"{name}: no posts, so accuracy and macro-F1 must be null")
        if n > 0 and (row.get("accuracy") is None or row.get("macro_f1") is None):
            problems.append(f"{name}: {n} posts, so accuracy and macro-F1 must be numbers")
    report.check(bool(rows) and not problems,
                 f"{label}: every evaluation row has the right number of posts, class supports that add up, "
                 "and scores that are numbers in [0, 1] or null", "; ".join(problems[:5]))

    accuracy_all = {row.get("model"): row.get("accuracy") for row in rows if row.get("subset") == "all_test"}
    matrices = sentiment.get("confusion_matrices") if isinstance(sentiment.get("confusion_matrices"), dict) else {}
    problems = []
    for name in models:
        matrix = matrices.get(name) if isinstance(matrices.get(name), dict) else {}
        counts = matrix.get("counts")
        if not (matrix.get("labels") == classes and isinstance(counts, list) and len(counts) == len(classes)
                and all(isinstance(line, list) and len(line) == len(classes) and all(is_count(cell) for cell in line)
                        for line in counts)):
            problems.append(f"{name}: not a {len(classes)}x{len(classes)} table of counts with labels {classes}")
            continue
        total = sum(sum(line) for line in counts)
        if total != n_test or matrix.get("n") != total:
            problems.append(f"{name}: cells add up to {total}, test posts {n_test}")
        elif is_count(n_test) and n_test > 0 and isinstance(accuracy_all.get(name), float) and \
                abs(sum(counts[i][i] for i in range(len(classes))) / n_test - accuracy_all[name]) > 1e-9:
            problems.append(f"{name}: diagonal does not match the accuracy in the evaluation table")
    report.check(bool(models) and not problems,
                 f"{label}: one confusion matrix per model, adding up to the number of test posts", "; ".join(problems))

    selection = sentiment.get("selection") if isinstance(sentiment.get("selection"), dict) else {}
    validation = selection.get("validation_macro_f1") if isinstance(selection.get("validation_macro_f1"), dict) else {}
    report.check(selection.get("selected") in models and sorted(validation) == sorted(models)
                 and all(is_score(score) for score in validation.values()),
                 f"{label}: one model chosen by validation macro-F1, with the validation score of every model",
                 f"selected {selection.get('selected')!r}, validation scores {validation}")

    scoring = sentiment.get("scoring") if isinstance(sentiment.get("scoring"), dict) else {}
    scoring_check = scoring.get("check") if isinstance(scoring.get("check"), dict) else {}
    report.check(scoring.get("function") == "sentiment_proba" and scoring.get("column_labels") == classes
                 and scoring.get("bullish_column") == len(classes) - 1
                 and scoring.get("model") == selection.get("selected")
                 and scoring.get("train_ids") == "train_post_ids" and scoring.get("n_train_ids") == sizes.get("train")
                 and scoring_check.get("n_texts") == n_test and scoring_check.get("n_columns") == len(classes)
                 and scoring_check.get("rows_sum_to_one") is True
                 and scoring_check.get("matches_evaluated_predictions") is True,
                 f"{label}: scoring function gives one column per class (bullish last), rows adding up to 1, "
                 "and the chosen model's evaluated test labels",
                 f"columns {scoring.get('column_labels')}, bullish column {scoring.get('bullish_column')}, "
                 f"model {scoring.get('model')!r}, check {scoring_check}")

    duplicates = sentiment.get("test_texts_also_in_train")
    duplicates = duplicates if isinstance(duplicates, dict) else {}
    report.check(all(is_count(duplicates.get(key)) and duplicates[key] <= (n_test if is_count(n_test) else -1)
                     for key in ("same_text", "same_clean_text")),
                 f"{label}: test posts with the same text as a training post are counted", f"{duplicates}")

    macro_all = {row.get("model"): row.get("macro_f1") for row in rows if row.get("subset") == "all_test"}

    def beats_both(name):
        score = macro_all.get(name)
        return score is not None and all(macro_all.get(baseline) is not None and score > macro_all[baseline]
                                         for baseline in SENTIMENT_BASELINES)

    report.check(sentiment.get("beats_baselines") == {name: beats_both(name) for name in models},
                 f"{label}: beats_baselines agrees with the evaluation rows", f"{sentiment.get('beats_baselines')}")

    if not full_run:
        return  # the sample is too small for the scores to mean anything
    btc_rows = [row for row in rows if row.get("subset") == "test_mentions_btc"]
    report.check(bool(btc_rows) and all(is_count(row.get("n")) and row["n"] > 0 for row in btc_rows),
                 f"{label}: the rows for test posts that mention bitcoin have posts",
                 f"{[row.get('n') for row in btc_rows]}")
    for name in models:
        report.check(beats_both(name),
                     f"{label}: {name} macro-F1 on all test posts beats always majority and uniform random",
                     f"{macro_all.get(name)} vs " + ", ".join(f"{baseline} {macro_all.get(baseline)}"
                                                               for baseline in SENTIMENT_BASELINES))


def print_sentiment(brief):
    """Print the key numbers of the sentiment section (full run)."""
    sentiment = brief.get("sentiment") if isinstance(brief.get("sentiment"), dict) else {}
    if not sentiment:
        return
    tokenization = sentiment.get("tokenization") or {}
    split = sentiment.get("split") or {}
    selection = sentiment.get("selection") or {}
    print("\nSentiment (full run)")
    for name in ("nltk_word_tokenize", "bpe"):
        part = tokenization.get(name) or {}
        print(f"  tokenization {name + ':':20} vocabulary {part.get('vocab_size')}, "
              f"tokens per post {part.get('avg_tokens_per_post')}")
    print(f"  split (posts):                 {split.get('n_posts')}   mention bitcoin: {split.get('n_mentions_btc')}")
    print(f"  vectorizers:                   {sentiment.get('vectorizers')}")
    print(f"  validation macro-F1:           {selection.get('validation_macro_f1')}   chosen: {selection.get('selected')}")
    print(f"  test texts also in training:   {sentiment.get('test_texts_also_in_train')}")
    classes = sentiment.get("classes") or []
    header = f"  {'model':16} {'subset':18} {'posts':>6} {'accuracy':>8} " + " ".join(
        f"{'P' + c:>6} {'R' + c:>6} {'F1-' + c:>6}" for c in classes) + f" {'macro-F1':>8}"
    print("  evaluation table (P precision, R recall, F1 per class; - = null):")
    print(header)

    def number(value):
        return f"{value:.4f}" if isinstance(value, (int, float)) and not isinstance(value, bool) else "-"

    for row in (sentiment.get("evaluation") or {}).get("rows") or []:
        per_class = row.get("per_class") or {}
        cells = " ".join(f"{number((per_class.get(c) or {}).get('precision')):>6.6} "
                         f"{number((per_class.get(c) or {}).get('recall')):>6.6} "
                         f"{number((per_class.get(c) or {}).get('f1')):>6.6}" for c in classes)
        print(f"  {str(row.get('model')):16} {str(row.get('subset')):18} {str(row.get('n')):>6} "
              f"{number(row.get('accuracy')):>8} {cells} {number(row.get('macro_f1')):>8}")
    for name, matrix in (sentiment.get("confusion_matrices") or {}).items():
        print(f"  confusion matrix {name} (rows true, columns predicted, labels {matrix.get('labels')}): "
              f"{matrix.get('counts')}")
    print(f"  beats both baselines:          {sentiment.get('beats_baselines')}")


# ---------------------------------------------------------------- topic chapter (9)

def check_topic_switches(report, defaults):
    absent = [name for name in TOPIC_SWITCHES if name not in defaults]
    report.check(not absent, "switches of the topic chapter are present", ", ".join(absent))


def as_dict(value):
    return value if isinstance(value, dict) else {}


def check_topics(report, label, brief, switches, full_run):
    """The topic table follows its switches; every topic has a name, posts, a share and quotes; sizes add up."""
    topics = brief.get("topics")
    if not report.check(isinstance(topics, dict) and bool(topics), f"{label}: topics section has content"):
        return
    dataset = as_dict(brief.get("dataset"))
    btc = as_dict(dataset.get("btc_mentions"))
    scope, cap, wanted = switches.get("TOPIC_SCOPE"), switches.get("TOPIC_MAX_POSTS"), switches.get("N_TOPICS")
    if scope == "btc":  # every post that mentions bitcoin or btc, as counted in chapter 2
        available, natural = btc.get("n_posts"), as_dict(btc.get("by_class"))
    else:  # every post of the data source
        available, natural = dataset.get("n_posts"), as_dict(dataset.get("class_counts"))

    # The topic post table: scope, cap, all three classes, stable ids.
    n_posts = topics.get("n_posts")
    counts = as_dict(topics.get("class_counts"))
    table_ok = (topics.get("post_table") == "topic_posts" and topics.get("topic_column") == "topic"
                and topics.get("scope") == scope and topics.get("max_posts") == cap
                and is_count(available) and is_count(cap) and is_count(n_posts)
                and topics.get("n_available") == available
                and n_posts == min(available, cap) and topics.get("sampled") == (available > cap)
                and sorted(counts) == TOPIC_CLASSES and all(is_count(count) for count in counts.values())
                and sum(counts.values()) == n_posts
                and topics.get("n_unique_post_ids") == n_posts
                and is_count(topics.get("n_mentions_btc")) and topics["n_mentions_btc"] <= n_posts
                and (scope != "btc" or topics["n_mentions_btc"] == n_posts))
    report.check(table_ok, f"{label}: topic post table follows TOPIC_SCOPE and TOPIC_MAX_POSTS "
                 "(posts in scope up to the cap, classes 0, 1 and 2 adding up, unique post_id)",
                 f"scope {topics.get('scope')!r} (switch {scope!r}), cap {topics.get('max_posts')!r} (switch {cap!r}), "
                 f"{n_posts} of {topics.get('n_available')} posts (expected of {available}), "
                 f"sampled {topics.get('sampled')}, classes {counts}, unique ids {topics.get('n_unique_post_ids')}, "
                 f"mention bitcoin {topics.get('n_mentions_btc')}")
    if table_ok and not topics.get("sampled"):
        report.check(counts == natural,
                     f"{label}: without a draw the topic post table keeps every post in scope with its natural "
                     "class counts (not filtered by N_CLASSES or BTC_ONLY)", f"{counts} vs {natural}")
    if scope == "btc" and is_count(cap) and is_count(btc.get("n_posts")) and cap >= btc["n_posts"]:
        report.check(n_posts == btc["n_posts"],
                     f"{label}: the topic post table holds exactly the posts that chapter 2 counted as mentioning "
                     "bitcoin or btc", f"{n_posts} vs {btc.get('n_posts')}")
    if full_run and table_ok and not topics.get("sampled"):
        report.check(all(counts[label_value] > 0 for label_value in TOPIC_CLASSES),
                     f"{label}: the topic post table has posts of all three classes", f"{counts}")

    cleaning = as_dict(topics.get("cleaning"))
    report.check(cleaning.get("function") == "clean_texts" and cleaning.get("same_as_working_sample") is True
                 and is_count(cleaning.get("n_also_in_working_sample"))
                 and is_count(cleaning.get("n_empty_after")) and is_count(n_posts)
                 and cleaning["n_empty_after"] <= n_posts,
                 f"{label}: topic posts cleaned with clean_texts (posts also in the working sample get the same "
                 "clean_text)", f"{cleaning}")

    vectorizer = as_dict(topics.get("vectorizer"))
    blocked = vectorizer.get("blocked_words")
    vocab = vectorizer.get("vocab_size")
    report.check(vectorizer.get("kind") == "tfidf" and vectorizer.get("fitted_on") == "topic_posts"
                 and vectorizer.get("fitted_on_n_posts") == n_posts and is_count(vocab)
                 and is_count(vectorizer.get("vocab_size_requested")) and vocab <= vectorizer["vocab_size_requested"]
                 and isinstance(blocked, list) and (scope != "btc" or SELECTION_WORDS <= set(blocked))
                 and is_count(vectorizer.get("n_posts_without_words")) and is_count(n_posts)
                 and vectorizer["n_posts_without_words"] <= n_posts,
                 f"{label}: topic vectorizer of its own learned on the topic table "
                 "(with TOPIC_SCOPE btc, bitcoin and btc left out)",
                 f"fitted on {vectorizer.get('fitted_on')!r} ({vectorizer.get('fitted_on_n_posts')} posts), "
                 f"vocabulary {vocab} of {vectorizer.get('vocab_size_requested')}, "
                 f"posts without words {vectorizer.get('n_posts_without_words')}")

    # The topics: as many as asked for (fewer only with fewer distinct post vectors), numbered 0 to n-1.
    items = topics.get("topics") if isinstance(topics.get("topics"), list) else []
    items = [item for item in items if isinstance(item, dict)]
    n_topics, distinct = topics.get("n_topics"), topics.get("n_distinct_vectors")
    count_ok = (topics.get("method") == "kmeans" and topics.get("n_topics_requested") == wanted
                and is_count(wanted) and is_count(distinct) and is_count(n_posts) and distinct <= n_posts
                and (n_posts == 0 or distinct >= 1) and n_topics == min(wanted, distinct)
                and len(items) == n_topics and isinstance(topics.get("topics"), list)
                and len(items) == len(topics["topics"])
                and [item.get("topic") for item in items] == list(range(n_topics)))
    report.check(count_ok, f"{label}: N_TOPICS topics numbered from 0 (fewer only when the table has fewer distinct "
                 "post vectors)", f"method {topics.get('method')!r}, {n_topics} topics, asked for "
                 f"{topics.get('n_topics_requested')} (switch {wanted!r}), {distinct} distinct vectors")

    name_words, quotes_per_topic = topics.get("name_words"), topics.get("quotes_per_topic")
    problems = []
    for item in items:
        topic, size = item.get("topic"), item.get("n_posts")
        words, quotes, without = item.get("top_words"), item.get("quotes"), item.get("n_posts_without_words")
        if not is_count(size) or size == 0:
            problems.append(f"topic {topic}: {size!r} posts")
            continue
        if not (isinstance(item.get("share"), float) and is_count(n_posts) and n_posts > 0
                and abs(item["share"] - size / n_posts) <= 1e-9):
            problems.append(f"topic {topic}: share {item.get('share')!r} is not {size} / {n_posts}")
        if not (isinstance(words, list) and all(isinstance(word, str) and word for word in words)
                and is_count(name_words) and len(words) <= name_words and item.get("name") == " / ".join(words)):
            problems.append(f"topic {topic}: name {item.get('name')!r} / words {words!r}")
        elif not words and not (vocab == 0 or without == size):
            problems.append(f"topic {topic}: no name although its posts have words")
        elif scope == "btc" and SELECTION_WORDS & set(words):
            problems.append(f"topic {topic}: name {item.get('name')!r} contains bitcoin or btc")
        if not (isinstance(quotes, list) and is_count(quotes_per_topic) and 1 <= len(quotes) <= min(quotes_per_topic, size)
                and all(isinstance(quote, dict) and isinstance(quote.get("post_id"), str)
                        and quote.get("label") in (0, 1, 2) and isinstance(quote.get("text"), str) for quote in quotes)
                and len({quote["post_id"] for quote in quotes}) == len(quotes)
                and len({quote["text"] for quote in quotes}) == len(quotes)):
            problems.append(f"topic {topic}: quotes {quotes!r}")
        if not (is_count(without) and without <= size):
            problems.append(f"topic {topic}: posts without words {without!r}")
    report.check(not problems and count_ok,
                 f"{label}: every topic has a name (a few words), posts, a share and quotes (different posts)",
                 "; ".join(problems[:5]))

    sizes = [item.get("n_posts") for item in items]
    shares = [item.get("share") for item in items]
    sums_ok = (all(is_count(size) for size in sizes) and is_count(n_posts) and sum(sizes) == n_posts
               and topics.get("n_posts_with_topic") == n_posts
               and (n_posts == 0 or (all(isinstance(share, float) for share in shares)
                                     and abs(sum(shares) - 1) <= 1e-6)))
    report.check(sums_ok, f"{label}: every topic post has a topic; topic sizes add up to the table and shares to 1",
                 f"sizes {sizes} (sum {sum(size for size in sizes if is_count(size))}) for {n_posts} posts, "
                 f"posts with a topic {topics.get('n_posts_with_topic')}, shares sum "
                 f"{sum(share for share in shares if isinstance(share, float))}")


def print_topics(brief):
    """Print the topic post table and the full topic table (full run)."""
    topics = as_dict(brief.get("topics"))
    if not topics:
        return
    btc = as_dict(as_dict(brief.get("dataset")).get("btc_mentions"))
    print("\nTopics (full run)")
    print(f"  topic post table:              {topics.get('n_posts')} posts (scope {topics.get('scope')!r}, "
          f"{topics.get('n_available')} in scope, cap {topics.get('max_posts')}, drawn: {topics.get('sampled')})")
    print(f"  posts per class:               {topics.get('class_counts')}   (0 bearish, 1 neutral, 2 bullish)")
    print(f"  chapter 2, mention bitcoin:    {btc.get('n_posts')}")
    print(f"  cleaning:                      {topics.get('cleaning')}")
    vectorizer = as_dict(topics.get("vectorizer"))
    print(f"  vectorizer:                    vocabulary {vectorizer.get('vocab_size')} of "
          f"{vectorizer.get('vocab_size_requested')}, posts without words {vectorizer.get('n_posts_without_words')}")
    print(f"  blocked words:                 {vectorizer.get('blocked_words')}")
    print(f"  method:                        {topics.get('method')} {topics.get('method_settings')}, "
          f"{topics.get('n_topics')} topics (asked for {topics.get('n_topics_requested')}, "
          f"{topics.get('n_distinct_vectors')} distinct vectors)")
    items = [item for item in topics.get("topics") or [] if isinstance(item, dict)]
    print(f"  topic table ({len(items)} topics, largest first; share of the topic post table):")
    for item in sorted(items, key=lambda item: (-(item.get("n_posts") or 0), item.get("topic") or 0)):
        share = item.get("share")
        share_text = f"{share:.4f}" if isinstance(share, float) else "-"
        print(f"    topic {item.get('topic')}: {item.get('n_posts')} posts, share {share_text} | {item.get('name')}")
        for quote in item.get("quotes") or []:
            print(f"        [{quote.get('label')}, {quote.get('post_id')}] {quote.get('text')}")


# ---------------------------------------------------------------- why chapter (10)

def check_why_switches(report, defaults):
    absent = [name for name in WHY_SWITCHES if name not in defaults]
    report.check(not absent, "switches of the why chapter are present", ", ".join(absent))


INVALID = object()  # marks a share that cannot be derived because its counts are malformed


def share_of(numerator, denominator):
    """The share the notebook must have written: None when the denominator is 0."""
    return None if denominator == 0 else numerator / denominator


def points(share, baseline):
    """Difference in percentage points: None when either share is None, INVALID when either is INVALID."""
    if share is INVALID or baseline is INVALID:
        return INVALID
    return None if share is None or baseline is None else 100 * (share - baseline)


def same_value(value, expected):
    """value is None where expected is None, otherwise a number equal to expected up to rounding."""
    if expected is INVALID:
        return False
    if expected is None:
        return value is None
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and abs(value - expected) <= 1e-9 * max(1.0, abs(expected)))


def counts_of(value, keys):
    """The counts stored under exactly these keys, or None when the keys or the counts are wrong."""
    if not isinstance(value, dict) or sorted(value) != sorted(keys) or not all(is_count(value[key]) for key in keys):
        return None
    return {key: value[key] for key in keys}


def label_share_from(row):
    """Bullish share by label that follows from the row's label counts (INVALID when they are malformed)."""
    labels = counts_of(as_dict(row).get("label_counts"), TOPIC_CLASSES)
    return INVALID if labels is None else share_of(labels["2"], labels["0"] + labels["2"])


def why_row_problems(row, where):
    """Problems of one why row: (identities, shares, fair comparison)."""
    identity, shares, fair_problems = [], [], []
    n = row.get("n_posts")
    labels = counts_of(row.get("label_counts"), TOPIC_CLASSES)
    calls = counts_of(row.get("model_counts"), MODEL_CALLS)
    if not is_count(n) or labels is None or calls is None:
        identity.append(f"{where}: posts {n!r}, label counts {row.get('label_counts')!r}, "
                        f"model calls {row.get('model_counts')!r}")
    else:
        if sum(labels.values()) != n:
            identity.append(f"{where}: label classes {labels} do not add up to {n} posts")
        if sum(calls.values()) != n:
            identity.append(f"{where}: model calls {calls} do not add up to {n} posts")

    expected_label = INVALID if labels is None else share_of(labels["2"], labels["0"] + labels["2"])
    expected_model = INVALID if calls is None else share_of(calls["bullish"], calls["bullish"] + calls["bearish"])
    expected_unsure = INVALID if calls is None or not is_count(n) else share_of(calls["unsure"], n)
    for key, expected in [("label_bullish_share", expected_label), ("model_bullish_share", expected_model),
                          ("model_unsure_share", expected_unsure)]:
        if not (is_score(row.get(key)) and same_value(row.get(key), expected)):
            shares.append(f"{where}: {key} {row.get(key)!r}")

    fair = as_dict(row.get("fair"))
    fair_n = fair.get("n_posts")
    fair_labels = counts_of(fair.get("label_counts"), FAIR_CLASSES)
    fair_calls = counts_of(fair.get("model_counts"), MODEL_CALLS)
    if not is_count(fair_n) or fair_labels is None or fair_calls is None:
        fair_problems.append(f"{where}: fair comparison {fair!r}")
        return identity, shares, fair_problems
    if sum(fair_labels.values()) != fair_n or sum(fair_calls.values()) != fair_n:
        fair_problems.append(f"{where}: fair labels {fair_labels} or calls {fair_calls} do not add up to {fair_n}")
    if labels is None or fair_n > labels["0"] + labels["2"] or any(fair_labels[c] > labels[c] for c in FAIR_CLASSES):
        fair_problems.append(f"{where}: {fair_n} fair posts {fair_labels} but labels {labels}")
    fair_label = share_of(fair_labels["2"], fair_labels["0"] + fair_labels["2"])
    fair_model = share_of(fair_calls["bullish"], fair_calls["bullish"] + fair_calls["bearish"])
    for key, expected in [("label_bullish_share", fair_label), ("model_bullish_share", fair_model)]:
        if not (is_score(fair.get(key)) and same_value(fair.get(key), expected)):
            shares.append(f"{where}: fair {key} {fair.get(key)!r}")
    if not same_value(fair.get("model_minus_label_pp"), points(fair_model, fair_label)):
        shares.append(f"{where}: fair model_minus_label_pp {fair.get('model_minus_label_pp')!r}")
    return identity, shares, fair_problems


def why_rows(why):
    """The topic rows of a why section (dicts only)."""
    rows = why.get("topics") if isinstance(why.get("topics"), list) else []
    return [row for row in rows if isinstance(row, dict)]


def check_why(report, label, brief, switches):
    """The topic x bullish table: one row per topic plus the overall row; counts add up; shares, marks and order."""
    why = brief.get("why")
    if not report.check(isinstance(why, dict) and bool(why), f"{label}: why section has content"):
        return
    sentiment, topics = as_dict(brief.get("sentiment")), as_dict(brief.get("topics"))
    selected = as_dict(sentiment.get("selection")).get("selected")
    scoring, thresholds = as_dict(why.get("scoring")), as_dict(why.get("thresholds"))
    high, low, min_posts = (switches.get("BULLISH_THRESHOLD"), switches.get("BEARISH_THRESHOLD"),
                            switches.get("MIN_TOPIC_POSTS"))
    report.check(why.get("post_table") == "topic_posts" and why.get("topic_column") == "topic"
                 and isinstance(selected, str) and why.get("model") == selected
                 and scoring.get("function") == "sentiment_proba" and is_count(topics.get("n_posts"))
                 and scoring.get("n_scored") == topics["n_posts"]
                 and same_value(thresholds.get("bullish_at_or_above"), high)
                 and same_value(thresholds.get("bearish_at_or_below"), low)
                 and is_count(min_posts) and why.get("min_topic_posts") == min_posts,
                 f"{label}: why scores every topic post with the chosen model and records the thresholds and "
                 "MIN_TOPIC_POSTS as set",
                 f"model {why.get('model')!r} (chosen {selected!r}), scored {scoring.get('n_scored')!r} of "
                 f"{topics.get('n_posts')!r} topic posts, thresholds {thresholds}, min posts {why.get('min_topic_posts')!r}")

    # One row per topic of the topics section, plus the overall row for the whole topic post table.
    overall = as_dict(why.get("overall"))
    rows = why_rows(why)
    topic_items = [item for item in topics.get("topics") or [] if isinstance(item, dict)] \
        if isinstance(topics.get("topics"), list) else []
    expected = {item.get("topic"): item for item in topic_items if is_count(item.get("topic"))}
    numbers = [row.get("topic") for row in rows]
    one_to_one = (isinstance(why.get("topics"), list) and len(rows) == len(why["topics"])
                  and all(is_count(number) for number in numbers) and len(set(numbers)) == len(numbers)
                  and sorted(numbers) == sorted(expected) and len(expected) == len(topic_items)
                  and all(row.get("name") == expected[row["topic"]].get("name")
                          and row.get("n_posts") == expected[row["topic"]].get("n_posts") for row in rows))
    overall_ok = (bool(overall) and is_count(overall.get("n_posts")) and overall["n_posts"] == topics.get("n_posts")
                  and counts_of(overall.get("label_counts"), TOPIC_CLASSES) is not None
                  and overall.get("label_counts") == topics.get("class_counts"))
    report.check(one_to_one and overall_ok,
                 f"{label}: why has one row per topic of the topics section (same numbers, names and sizes) and an "
                 "overall row for the whole topic post table",
                 f"why topics {numbers}, topics section {sorted(expected)}; overall {overall.get('n_posts')!r} posts "
                 f"{overall.get('label_counts')!r} vs topic table {topics.get('n_posts')!r} {topics.get('class_counts')!r}")

    identity, shares, fair_problems = [], [], []
    for where, row in [("all", overall)] + [(f"topic {row.get('topic')}", row) for row in rows]:
        row_identity, row_shares, row_fair = why_row_problems(row, where)
        identity += row_identity
        shares += row_shares
        fair_problems += row_fair
    overall_share = label_share_from(overall)
    for row in rows:
        expected_points = points(label_share_from(row), overall_share)
        if not same_value(row.get("label_vs_overall_pp"), expected_points):
            shares.append(f"topic {row.get('topic')}: label_vs_overall_pp {row.get('label_vs_overall_pp')!r}")
    report.check(not identity, f"{label}: why rows add up: model calls bullish + bearish + unsure = posts and label "
                 "classes = posts (every topic and the overall row)", "; ".join(identity[:5]))
    report.check(not shares, f"{label}: why shares follow from the counts (null when the denominator is 0, "
                 "otherwise in [0, 1]) and differences from the shares (percentage points)", "; ".join(shares[:5]))

    # Topic rows add up to the overall row.
    def total(part, key, keys):
        values = [counts_of(as_dict(row.get(part) if part else row).get(key), keys) for row in rows]
        if any(value is None for value in values):
            return None
        return {item: sum(value[item] for value in values) for item in keys}

    overall_fair = as_dict(overall.get("fair"))
    sums_ok = (total(None, "label_counts", TOPIC_CLASSES) == counts_of(overall.get("label_counts"), TOPIC_CLASSES)
               and total(None, "model_counts", MODEL_CALLS) == counts_of(overall.get("model_counts"), MODEL_CALLS)
               and total("fair", "label_counts", FAIR_CLASSES) == counts_of(overall_fair.get("label_counts"), FAIR_CLASSES)
               and total("fair", "model_counts", MODEL_CALLS) == counts_of(overall_fair.get("model_counts"), MODEL_CALLS)
               and counts_of(overall.get("model_counts"), MODEL_CALLS) is not None)
    report.check(sums_ok, f"{label}: why topic rows add up to the overall row (label classes, model calls, fair "
                 "comparison labels and calls)")

    # The fair comparison: unseen posts labelled bearish or bullish.
    info = as_dict(why.get("fair_comparison"))
    unseen, in_training = info.get("n_unseen_bullish_or_bearish"), info.get("n_topic_posts_in_training_set")
    topic_fair = [as_dict(row.get("fair")).get("n_posts") for row in rows]
    overall_fair_n = overall_fair.get("n_posts")
    n_train_ids = as_dict(sentiment.get("scoring")).get("n_train_ids")
    fair_ok = (not fair_problems and is_count(unseen) and is_count(overall_fair_n)
               and all(is_count(size) for size in topic_fair) and sum(topic_fair) == overall_fair_n == unseen
               and is_count(in_training) and is_count(n_train_ids) and in_training <= n_train_ids
               and is_count(overall.get("n_posts")) and in_training <= overall["n_posts"]
               and all(is_count(info.get(key)) and info[key] <= overall_fair_n
                       for key in ("n_same_text_as_a_training_post", "n_same_clean_text_as_a_training_post")))
    report.check(fair_ok, f"{label}: fair comparison: each topic has at most its bullish + bearish posts in it; the "
                 "topics add up to the overall row and to the unseen bullish or bearish posts of the topic table",
                 f"topics {topic_fair} (sum {sum(size for size in topic_fair if is_count(size))}), overall "
                 f"{overall_fair_n!r}, unseen bullish or bearish {unseen!r}, topic posts in training {in_training!r} "
                 f"of {n_train_ids!r} training ids; " + "; ".join(fair_problems[:3]))

    marks_ok = is_count(min_posts) and all(
        row.get("too_few_posts") is (is_count(row.get("n_posts")) and row["n_posts"] < min_posts) for row in rows)
    report.check(marks_ok, f"{label}: too-few marks follow MIN_TOPIC_POSTS",
                 f"{[(row.get('topic'), row.get('n_posts'), row.get('too_few_posts')) for row in rows]}")

    def order_key(row):
        share = row.get("label_bullish_share")
        return (share is None, -share if isinstance(share, (int, float)) else 0.0, row.get("topic"))

    order_ok = (all(row.get("label_bullish_share") is None or isinstance(row.get("label_bullish_share"), (int, float))
                    for row in rows) and all(is_count(number) for number in numbers)
                and numbers == [row.get("topic") for row in sorted(rows, key=order_key)])
    report.check(order_ok, f"{label}: why topics sorted by bullish share by label, highest first, null last, ties by "
                 "topic number", f"order {numbers}")


WHY_STABLE_ROW_FIELDS = ["topic", "name", "n_posts", "too_few_posts", "label_counts", "label_bullish_share",
                         "label_vs_overall_pp", "too_few_bullish_bearish"]
WHY_STABLE_FAIR_FIELDS = ["n_posts", "label_counts", "label_bullish_share"]
WHY_STABLE_COUNTS = ["n_topic_posts_in_training_set", "n_unseen_bullish_or_bearish",
                     "n_same_text_as_a_training_post", "n_same_clean_text_as_a_training_post"]
# Calibration fields that do not depend on training (the rates and the chosen model may).
WHY_STABLE_CALIBRATION = ["rate_posts", "min_rate_gap", "n_rate_posts_in_fair_comparison"]


def stable_why(brief):
    """The parts of the why section that do not depend on training, or None when there is no why section."""
    why = as_dict(brief).get("why") if isinstance(brief, dict) else None
    if not isinstance(why, dict) or not why:
        return None

    def part(row):
        row = as_dict(row)
        return {**{key: row.get(key) for key in WHY_STABLE_ROW_FIELDS},
                "fair": {key: as_dict(row.get("fair")).get(key) for key in WHY_STABLE_FAIR_FIELDS}}

    calibration = as_dict(why.get("calibration"))
    return {"overall": part(why.get("overall")), "topics": [part(row) for row in why_rows(why)],
            "fair_comparison": {key: as_dict(why.get("fair_comparison")).get(key) for key in WHY_STABLE_COUNTS},
            "calibration": {**{key: calibration.get(key) for key in WHY_STABLE_CALIBRATION},
                            "subset": as_dict(calibration.get("rates_from")).get("subset"),
                            "min_bullish_bearish_posts": why.get("min_bullish_bearish_posts")}}


def compare_why(report, first, again):
    """Two runs with the same switches: everything in the why section that does not depend on training is identical."""
    left, right = stable_why(first), stable_why(again)
    detail = ""
    if left is None or right is None:
        detail = "a brief or its why section is missing"
    elif left != right:
        detail = "differing parts: " + ", ".join(key for key in left if left[key] != right[key])
    report.check(left is not None and left == right,
                 "default switches run twice: the why sections agree on everything that does not depend on training "
                 "(label counts and shares, differences from all, too-few marks, order, fair comparison sizes and "
                 "label shares; too-few-bullish-or-bearish marks, the calibration's test subset and test posts)", detail)


def threshold_run(report, defaults, overrides, default_brief, rules=None, built_in=None):
    """Both thresholds at 0.5 and MIN_TOPIC_POSTS above the topic post table: no post unsure, every topic too small."""
    table = as_dict(as_dict(default_brief).get("topics")).get("n_posts")
    settings = {**THRESHOLD_SETTINGS, "MIN_TOPIC_POSTS": table + 1 if is_count(table) else 10 ** 9}
    settings["MIN_BULLISH_BEARISH_POSTS"] = settings["MIN_TOPIC_POSTS"]  # above every row's bullish + bearish posts
    parameters = {"DATA_SOURCE": SAMPLE_SOURCE, "EMOTION_LABELS_SOURCE": PRELABEL_SAMPLE_FILE,
                  **DATED_SAMPLE_SETTINGS, **VALIDATION_SAMPLE_SETTINGS, "WORK_DIR": "data/checks/quick-thresholds",
                  **overrides, **settings}
    parameters = panel_settings(report, THRESHOLD_LABEL, parameters, overrides)
    switches = {**defaults, **parameters}
    report.info(f"{THRESHOLD_LABEL}: " + ", ".join(f"{k}={v!r}" for k, v in parameters.items()))
    kept = [name for name in settings if name in overrides]
    if kept:
        report.info(f"{THRESHOLD_LABEL}: keeps its own values of {', '.join(kept)} instead of the -p values")
    check_offline_sources(report, THRESHOLD_LABEL, parameters, defaults)
    brief_path = REPO_DIR / switches["WORK_DIR"] / "brief.json"
    brief_path.unlink(missing_ok=True)  # a stale brief must never pass
    ok, _, _ = run_notebook(report, THRESHOLD_LABEL, parameters, REPO_DIR / switches["WORK_DIR"] / "executed.ipynb")
    brief = read_brief(report, THRESHOLD_LABEL, brief_path) if ok else None
    if brief is None:
        return
    check_why(report, THRESHOLD_LABEL, brief, switches)
    check_calibration(report, THRESHOLD_LABEL, brief, switches)
    check_drawers(report, THRESHOLD_LABEL, brief, switches, built_in)
    check_emotions(report, THRESHOLD_LABEL, brief, switches)
    check_dated_data(report, THRESHOLD_LABEL, brief, switches, quick_default=True)
    check_daily(report, THRESHOLD_LABEL, brief, switches)
    check_new_words(report, THRESHOLD_LABEL, brief, switches)
    check_layers(report, THRESHOLD_LABEL, brief, switches)
    check_validation(report, THRESHOLD_LABEL, brief, switches)
    check_format(report, THRESHOLD_LABEL, brief, rules)
    check_panel_output(report, THRESHOLD_LABEL, REPO_DIR / switches["WORK_DIR"] / "executed.ipynb")
    why = as_dict(brief.get("why"))
    rows, overall = why_rows(why), as_dict(why.get("overall"))
    n_table = as_dict(brief.get("topics")).get("n_posts")
    report.check(is_count(n_table) and switches["MIN_TOPIC_POSTS"] > n_table,
                 f"{THRESHOLD_LABEL}: MIN_TOPIC_POSTS is larger than the whole topic post table",
                 f"{switches['MIN_TOPIC_POSTS']} vs {n_table!r} posts")
    unsure = []
    for where, row in [("all", overall)] + [(f"topic {row.get('topic')}", row) for row in rows]:
        calls, n = counts_of(row.get("model_counts"), MODEL_CALLS), row.get("n_posts")
        expected = (0.0 if n > 0 else None) if is_count(n) else INVALID
        if calls is None or calls["unsure"] != 0 or not same_value(row.get("model_unsure_share"), expected):
            unsure.append(f"{where}: calls {row.get('model_counts')!r}, unsure share {row.get('model_unsure_share')!r}")
    report.check(bool(overall) and not unsure,
                 f"{THRESHOLD_LABEL}: with both thresholds at 0.5 no post is unsure (unsure share 0 in every topic and "
                 "the overall row)", "; ".join(unsure[:5]))
    report.check(all(row.get("too_few_posts") is True for row in rows),
                 f"{THRESHOLD_LABEL}: every topic is marked as having too few posts",
                 f"{sum(row.get('too_few_posts') is True for row in rows)} of {len(rows)} topics marked")
    marked_rows = [overall] + rows
    report.check(is_count(n_table) and switches["MIN_BULLISH_BEARISH_POSTS"] > n_table and bool(overall)
                 and all(row.get("too_few_bullish_bearish") is True for row in marked_rows),
                 f"{THRESHOLD_LABEL}: MIN_BULLISH_BEARISH_POSTS is larger than the whole topic post table and every row, "
                 "the overall row included, is marked as having too few bullish or bearish posts",
                 f"{switches['MIN_BULLISH_BEARISH_POSTS']} vs {n_table!r} posts; "
                 f"{sum(row.get('too_few_bullish_bearish') is True for row in marked_rows)} of {len(marked_rows)} rows marked")


def print_why(brief):
    """Print the topic x bullish table (full run)."""
    why = as_dict(brief.get("why"))
    if not why:
        return

    def share(value):
        return f"{value:.4f}" if isinstance(value, (int, float)) and not isinstance(value, bool) else "-"

    def pp(value):
        return f"{value:+.2f}" if isinstance(value, (int, float)) and not isinstance(value, bool) else "-"

    def line(name, row, few, versus):
        labels, calls = as_dict(row.get("label_counts")), as_dict(row.get("model_counts"))
        fair = as_dict(row.get("fair"))
        return (f"  {name:>5} {str(row.get('n_posts')):>6} {few:>3} "
                f"{str(labels.get('0')):>6} {str(labels.get('1')):>6} {str(labels.get('2')):>6} "
                f"{share(row.get('label_bullish_share')):>7} {versus:>7} "
                f"{str(calls.get('bullish')):>6} {str(calls.get('bearish')):>6} {str(calls.get('unsure')):>6} "
                f"{share(row.get('model_bullish_share')):>7} {share(row.get('model_unsure_share')):>7} "
                f"{str(fair.get('n_posts')):>6} {share(fair.get('label_bullish_share')):>7} "
                f"{share(fair.get('model_bullish_share')):>7} {pp(fair.get('model_minus_label_pp')):>7}")

    scoring, info = as_dict(why.get("scoring")), as_dict(why.get("fair_comparison"))
    print("\nWhy: topic x bullish (full run)")
    print(f"  model:                         {why.get('model')}   bullish probability: {scoring.get('bullish_probability')}")
    print(f"  thresholds:                    {why.get('thresholds')}   too few below {why.get('min_topic_posts')} posts")
    seconds = scoring.get("seconds")
    seconds_text = f"{seconds:.2f} s" if isinstance(seconds, (int, float)) and not isinstance(seconds, bool) else "-"
    print(f"  scored:                        {scoring.get('n_scored')} topic posts in {seconds_text}")
    print(f"  fair comparison:               {info}")
    print("  columns: topic, posts, too few, label bearish / neutral / bullish, bullish share by label, vs all (pp),")
    print("           model bullish / bearish / unsure, bullish share by model, unsure share,")
    print("           fair posts, fair share by label, fair share by model, model minus label (pp); - = null")
    print(f"  {'topic':>5} {'posts':>6} {'few':>3} {'bear':>6} {'neut':>6} {'bull':>6} {'label':>7} {'vs all':>7} "
          f"{'m-bull':>6} {'m-bear':>6} {'unsure':>6} {'model':>7} {'unsure%':>7} "
          f"{'fair':>6} {'f-label':>7} {'f-model':>7} {'m-l pp':>7}")
    print(line("all", as_dict(why.get("overall")), "", ""))
    for row in why_rows(why):
        print(line(str(row.get("topic")), row, "yes" if row.get("too_few_posts") else "no",
                   pp(row.get("label_vs_overall_pp"))))
    print("  topic names:")
    for row in why_rows(why):
        print(f"    topic {row.get('topic')}: {row.get('name')}")


# ---------------------------------------------------------------- calibration (chapter 10)

def check_calibration_switches(report, defaults):
    absent = [name for name in CALIBRATION_SWITCHES if name not in defaults]
    report.check(not absent, "switches of the calibration are present", ", ".join(absent))


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def calibrated_from(share, calibration):
    """The calibrated share the notebook must have written for a share at 0.5 (None unless status is applied)."""
    if share is INVALID:
        return INVALID
    if share is None or calibration.get("status") != "applied":
        return None
    true_rate, false_rate = calibration.get("true_bullish_rate"), calibration.get("false_bullish_rate")
    if not (is_number(true_rate) and is_number(false_rate)) or true_rate - false_rate <= 0:
        return INVALID
    return min(1.0, max(0.0, (share - false_rate) / (true_rate - false_rate)))


def stored_share(value):
    """A share as stored in the brief: the value when it is a number in [0, 1] or None, otherwise INVALID."""
    return value if is_score(value) else INVALID


def half_row_problems(row, where, calibration, min_bullish_bearish_posts):
    """Problems of the fields a row gets from the calibration, for any row with them (topic, overall or other).

    Returns (problems with the calls at 0.5, the shares and differences; problems with the too-few mark).
    """
    problems, marks = [], []
    row = as_dict(row)
    n, labels = row.get("n_posts"), counts_of(row.get("label_counts"), TOPIC_CLASSES)
    halves = counts_of(row.get("half_counts"), HALF_CALLS)
    if halves is None or not is_count(n) or sum(halves.values()) != n:
        problems.append(f"{where}: calls at 0.5 {row.get('half_counts')!r} do not add up to {n!r} posts")
    expected_half = INVALID if halves is None else share_of(halves["bullish"], halves["bullish"] + halves["bearish"])
    if not (is_score(row.get("half_bullish_share")) and same_value(row.get("half_bullish_share"), expected_half)):
        problems.append(f"{where}: half_bullish_share {row.get('half_bullish_share')!r}")
    expected_calibrated = calibrated_from(stored_share(row.get("half_bullish_share")), calibration)
    if not (is_score(row.get("calibrated_bullish_share"))
            and same_value(row.get("calibrated_bullish_share"), expected_calibrated)):
        problems.append(f"{where}: calibrated_bullish_share {row.get('calibrated_bullish_share')!r} "
                        f"(expected {expected_calibrated if expected_calibrated is not INVALID else 'not computable'!r})")
    expected_mark = (labels["0"] + labels["2"] < min_bullish_bearish_posts
                     if labels is not None and is_count(min_bullish_bearish_posts) else INVALID)
    if expected_mark is INVALID or row.get("too_few_bullish_bearish") is not expected_mark:
        marks.append(f"{where}: too_few_bullish_bearish {row.get('too_few_bullish_bearish')!r} with label counts "
                     f"{row.get('label_counts')!r} and bar {min_bullish_bearish_posts!r}")

    fair = as_dict(row.get("fair"))
    fair_n, fair_labels = fair.get("n_posts"), counts_of(fair.get("label_counts"), FAIR_CLASSES)
    fair_halves = counts_of(fair.get("half_counts"), HALF_CALLS)
    if fair_halves is None or not is_count(fair_n) or sum(fair_halves.values()) != fair_n:
        problems.append(f"{where}: fair calls at 0.5 {fair.get('half_counts')!r} do not add up to {fair_n!r} posts")
    fair_half = INVALID if fair_halves is None else share_of(fair_halves["bullish"],
                                                             fair_halves["bullish"] + fair_halves["bearish"])
    if not (is_score(fair.get("half_bullish_share")) and same_value(fair.get("half_bullish_share"), fair_half)):
        problems.append(f"{where}: fair half_bullish_share {fair.get('half_bullish_share')!r}")
    fair_calibrated = calibrated_from(stored_share(fair.get("half_bullish_share")), calibration)
    if not (is_score(fair.get("calibrated_bullish_share"))
            and same_value(fair.get("calibrated_bullish_share"), fair_calibrated)):
        problems.append(f"{where}: fair calibrated_bullish_share {fair.get('calibrated_bullish_share')!r}")
    fair_label = INVALID if fair_labels is None else stored_share(fair.get("label_bullish_share"))
    for key, share in [("half_minus_label_pp", stored_share(fair.get("half_bullish_share"))),
                       ("calibrated_minus_label_pp", stored_share(fair.get("calibrated_bullish_share")))]:
        if not same_value(fair.get(key), points(share, fair_label)):
            problems.append(f"{where}: fair {key} {fair.get(key)!r}")
    return problems, marks


def evaluation_row(brief, model, subset):
    """The one row of sentiment.evaluation.rows for model and subset, or None."""
    rows = as_dict(as_dict(brief.get("sentiment")).get("evaluation")).get("rows")
    found = [row for row in rows or [] if isinstance(row, dict) and row.get("model") == model
             and row.get("subset") == subset] if isinstance(rows, list) else []
    return found[0] if len(found) == 1 else None


def check_calibration(report, label, brief, switches):
    """Chapter 10's calibration: where the rates come from, the status, every row's new fields and marks."""
    why = as_dict(brief.get("why"))
    calibration = as_dict(why.get("calibration"))
    if not report.check(bool(calibration), f"{label}: why has a calibration block"):
        return
    selected = as_dict(as_dict(brief.get("sentiment")).get("selection")).get("selected")
    subset = RATE_SUBSETS.get(switches.get("TOPIC_SCOPE"))
    source = evaluation_row(brief, selected, subset)
    per_class = as_dict(as_dict(source).get("per_class"))
    supports = {key: as_dict(per_class.get(key)).get("support") for key in FAIR_CLASSES}
    min_gap, min_posts = switches.get("MIN_RATE_GAP"), switches.get("MIN_BULLISH_BEARISH_POSTS")
    report.check(source is not None and calibration.get("rates_from") == {"model": selected, "subset": subset}
                 and all(is_count(value) for value in supports.values())
                 and calibration.get("rate_posts") == supports
                 and is_number(min_gap) and same_value(calibration.get("min_rate_gap"), min_gap)
                 and is_count(min_posts) and why.get("min_bullish_bearish_posts") == min_posts
                 and all(isinstance(calibration.get(key), str) and calibration[key] for key in ("half_rule", "formula")),
                 f"{label}: calibration reads the chosen model's row for the matching test posts (rates_from), records "
                 "its bearish and bullish test posts (rate_posts) and MIN_RATE_GAP and MIN_BULLISH_BEARISH_POSTS as set",
                 f"rates_from {calibration.get('rates_from')!r} (expected model {selected!r}, subset {subset!r}, "
                 f"row found {source is not None}); rate_posts {calibration.get('rate_posts')!r} (supports {supports}); "
                 f"min_rate_gap {calibration.get('min_rate_gap')!r} ({min_gap!r}); "
                 f"min_bullish_bearish_posts {why.get('min_bullish_bearish_posts')!r} ({min_posts!r})")

    # The rates and the status, from chapter 8's recalls, N_CLASSES and MIN_RATE_GAP.
    recalls = {key: as_dict(per_class.get(key)).get("recall") for key in FAIR_CLASSES}
    if switches.get("N_CLASSES") == 3:
        expected = {"true_bullish_rate": None, "false_bullish_rate": None, "rate_gap": None}
        status = "three_classes"
    else:
        true_rate = recalls["2"] if is_number(recalls["2"]) else None
        false_rate = 1.0 - recalls["0"] if is_number(recalls["0"]) else None
        gap = None if true_rate is None or false_rate is None else true_rate - false_rate
        expected = {"true_bullish_rate": true_rate, "false_bullish_rate": false_rate, "rate_gap": gap}
        if source is None:
            status = INVALID
        elif gap is None:
            status = "rates_missing"
        elif not is_number(min_gap) or gap < min_gap or gap <= 0:
            status = "rate_gap_too_small"
        else:
            status = "applied"
    wrong = [f"{key} {calibration.get(key)!r} (expected {value!r})" for key, value in expected.items()
             if not same_value(calibration.get(key), value)]
    report.check(source is not None and not wrong and calibration.get("status") == status
                 and calibration.get("status") in CALIBRATION_STATUSES,
                 f"{label}: the two rates follow chapter 8's recalls (true = bullish recall, false = 1 - bearish recall, "
                 "both null with three classes) and the status follows them, N_CLASSES and MIN_RATE_GAP",
                 f"status {calibration.get('status')!r} (expected {status if status is not INVALID else 'no source row'!r}); "
                 + "; ".join(wrong))

    # Every row and every fair comparison.
    overall, rows = as_dict(why.get("overall")), why_rows(why)
    problems, marks = [], []
    for where, row in [("all", overall)] + [(f"topic {row.get('topic')}", row) for row in rows]:
        row_problems, row_marks = half_row_problems(row, where, calibration, min_posts)
        problems += row_problems
        marks += row_marks
    report.check(bool(overall) and not problems,
                 f"{label}: every row and every fair comparison: calls at 0.5 add up to the posts, the share at 0.5 "
                 "follows from them, the calibrated share follows from it and the two rates (null unless the status is "
                 "applied, otherwise in [0, 1]), and the two new differences follow from the shares",
                 "; ".join(problems[:5]))
    report.check(bool(overall) and not marks,
                 f"{label}: too_few_bullish_bearish marks follow MIN_BULLISH_BEARISH_POSTS (bearish + bullish posts by "
                 "label below the bar), in every topic and in the overall row", "; ".join(marks[:5]))

    calibrated = [as_dict(row).get("calibrated_bullish_share") for row in [overall] + rows] + [
        as_dict(as_dict(row).get("fair")).get("calibrated_bullish_share") for row in [overall] + rows]
    report.check(calibration.get("status") == "applied" or all(value is None for value in calibrated),
                 f"{label}: when the status is not applied, every calibrated share is null",
                 f"status {calibration.get('status')!r}, calibrated shares {calibrated[:6]}")

    def total(part):
        values = [counts_of(as_dict(row.get(part) if part else row).get("half_counts"), HALF_CALLS) for row in rows]
        return None if any(value is None for value in values) else {
            call: sum(value[call] for value in values) for call in HALF_CALLS}

    report.check(bool(rows) and total(None) == counts_of(overall.get("half_counts"), HALF_CALLS)
                 and total("fair") == counts_of(as_dict(overall.get("fair")).get("half_counts"), HALF_CALLS),
                 f"{label}: topic rows add up to the overall row at 0.5 (calls, and calls in the fair comparison)",
                 f"topics {total(None)} vs overall {overall.get('half_counts')!r}; fair topics {total('fair')} vs "
                 f"overall {as_dict(overall.get('fair')).get('half_counts')!r}")

    shared = calibration.get("n_rate_posts_in_fair_comparison")
    fair_n = as_dict(overall.get("fair")).get("n_posts")
    report.check(is_count(shared) and is_count(fair_n) and shared <= fair_n
                 and all(is_count(value) for value in supports.values()) and shared <= sum(supports.values()),
                 f"{label}: the test posts behind the rates that are also in the fair comparison are counted (at most "
                 "the rates' bearish and bullish test posts and the fair comparison)",
                 f"{shared!r} of {supports} test posts, fair comparison {fair_n!r}")


def check_calibration_narrows(report, label, brief):
    """Full run: in the overall fair comparison the calibrated share is closer to the labels than both counted shares."""
    why = as_dict(brief.get("why"))
    fair = as_dict(as_dict(why.get("overall")).get("fair"))
    status = as_dict(why.get("calibration")).get("status")
    differences = {key: fair.get(key) for key in ("calibrated_minus_label_pp", "half_minus_label_pp", "model_minus_label_pp")}
    numbers = all(is_number(value) for value in differences.values())
    report.check(status == "applied" and numbers
                 and abs(differences["calibrated_minus_label_pp"]) < abs(differences["half_minus_label_pp"])
                 and abs(differences["calibrated_minus_label_pp"]) < abs(differences["model_minus_label_pp"]),
                 f"{label}: calibration narrows the gap to the labels in the overall fair comparison: "
                 "|calibrated - label| is smaller than |at 0.5 - label| and |with the thresholds - label|, status applied",
                 f"status {status!r}; " + ", ".join(f"{key} {value:+.2f}" if is_number(value) else f"{key} {value!r}"
                                                    for key, value in differences.items()))


def print_calibration(brief):
    """Print the calibration and the new columns of the topic x bullish table (full run)."""
    why = as_dict(brief.get("why"))
    calibration = as_dict(why.get("calibration"))
    if not calibration:
        return

    def share(value):
        return f"{value:.4f}" if is_number(value) else "-"

    def pp(value):
        return f"{value:+.2f}" if is_number(value) else "-"

    rates_from = as_dict(calibration.get("rates_from"))
    rate_posts = as_dict(calibration.get("rate_posts"))
    print("\nCalibration (full run)")
    print(f"  rates from:                    {rates_from.get('model')} / {rates_from.get('subset')}   test posts: "
          f"{rate_posts.get('0')} bearish, {rate_posts.get('2')} bullish")
    print(f"  true bullish rate:             {share(calibration.get('true_bullish_rate'))}   false bullish rate: "
          f"{share(calibration.get('false_bullish_rate'))}   gap: {share(calibration.get('rate_gap'))} "
          f"(at least {calibration.get('min_rate_gap')})")
    print(f"  status:                        {calibration.get('status')}")
    print(f"  test posts also in the fair comparison: {calibration.get('n_rate_posts_in_fair_comparison')}")
    print(f"  too few bullish + bearish below: {why.get('min_bullish_bearish_posts')} posts")
    fair = as_dict(as_dict(why.get("overall")).get("fair"))
    print(f"  overall fair comparison, minus the labels (pp): with the thresholds {pp(fair.get('model_minus_label_pp'))}, "
          f"at 0.5 {pp(fair.get('half_minus_label_pp'))}, calibrated {pp(fair.get('calibrated_minus_label_pp'))}")
    true_rate, false_rate = calibration.get("true_bullish_rate"), calibration.get("false_bullish_rate")
    clipped = []
    rows = [("all", as_dict(why.get("overall")))] + [(str(row.get("topic")), row) for row in why_rows(why)]
    if calibration.get("status") == "applied" and is_number(true_rate) and is_number(false_rate):
        for name, row in rows:
            for where, part in (("row", row), ("fair", as_dict(row.get("fair")))):
                half = part.get("half_bullish_share")
                if is_number(half):
                    raw = (half - false_rate) / (true_rate - false_rate)
                    if raw < 0 or raw > 1:
                        clipped.append(f"{name} {where} {raw:.4f} -> {min(1.0, max(0.0, raw)):.0f}")
    print(f"  clipped to 0 or 1:             {', '.join(clipped) or 'none'}")
    print("  columns: topic, too few bullish + bearish, calls at 0.5 bullish / bearish, share at 0.5, calibrated share,")
    print("           fair posts, fair share by label, fair share at 0.5, fair calibrated share, at 0.5 minus label (pp),")
    print("           calibrated minus label (pp); - = null")
    print(f"  {'topic':>5} {'few':>3} {'h-bull':>6} {'h-bear':>6} {'half':>7} {'calib':>7} "
          f"{'fair':>6} {'f-label':>7} {'f-half':>7} {'f-calib':>7} {'h-l pp':>7} {'c-l pp':>7}")
    for name, row in rows:
        halves, part = as_dict(row.get("half_counts")), as_dict(row.get("fair"))
        print(f"  {name:>5} {'yes' if row.get('too_few_bullish_bearish') else 'no':>3} "
              f"{str(halves.get('bullish')):>6} {str(halves.get('bearish')):>6} "
              f"{share(row.get('half_bullish_share')):>7} {share(row.get('calibrated_bullish_share')):>7} "
              f"{str(part.get('n_posts')):>6} {share(part.get('label_bullish_share')):>7} "
              f"{share(part.get('half_bullish_share')):>7} {share(part.get('calibrated_bullish_share')):>7} "
              f"{pp(part.get('half_minus_label_pp')):>7} {pp(part.get('calibrated_minus_label_pp')):>7}")


# ---------------------------------------------------------------- drawers (chapters 9 and 10)

def check_drawer_switches(report, defaults, others):
    absent = [name for name in DRAWER_SWITCHES if name not in defaults]
    groups = defaults.get("N_OTHER_GROUPS")
    report.check(not absent and defaults.get("DRAWER_WORDS_FILE") == "" and is_count(groups) and groups >= 1
                 and not set(DRAWER_SWITCHES) & set(others),
                 "switches of the drawers are present (DRAWER_WORDS_FILE defaults to \"\", the built-in word list; "
                 "N_OTHER_GROUPS is a positive integer; neither is two-state)",
                 f"missing {absent}; DRAWER_WORDS_FILE {defaults.get('DRAWER_WORDS_FILE')!r}, N_OTHER_GROUPS {groups!r}, "
                 f"two-state: {sorted(set(DRAWER_SWITCHES) & set(others))}")


def drawer_fingerprint_of(drawers):
    """The documented fingerprint: SHA-256 of [[drawer, name, words, exclude], ...] as compact ASCII JSON, 16 digits."""
    text = json.dumps([[item.get("drawer"), item.get("name"), item.get("words"), item.get("exclude")] for item in drawers],
                      ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("ascii")).hexdigest()[:16]


def phrase_regex(phrases):
    """Whole words and phrases, ignoring case, a space matching any whitespace, longer phrases first."""
    parts = [r"\s+".join(re.escape(piece) for piece in phrase.split()) for phrase in sorted(phrases, key=len, reverse=True)]
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.IGNORECASE)


def rule_matches(text, words, excluded):
    """The words of one drawer found in a text by the documented rule, in word-list order (written apart from the
    notebook): the drawer's excluded phrases are blanked out first, then every word is looked for on its own."""
    if not isinstance(text, str) or not text:
        return []
    left = phrase_regex(excluded).sub(" ", text) if excluded else text
    return [word for word in words if phrase_regex([word]).search(left)]


def load_drawer_cells(extra):
    """Run the notebook's built-in word list cell and the cell defining drawer_matches; (namespace, problems)."""
    namespace, problems = {"__name__": "notebook_cells", **extra}, []
    for name in ("DRAWER_WORDS_V1", "drawer_matches"):
        source = cell_defining(name)
        if source is None:
            problems.append(f"no cell defines {name}")
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(source, f"cell defining {name}", "exec"), namespace)
        except Exception as exc:  # the cell itself is broken
            problems.append(f"cell defining {name}: {type(exc).__name__}: {exc}")
    return namespace, problems


def word_list_rule_problems(word_list):
    """Problems of the built-in word list with the drafting rules (written apart from the notebook's own check)."""
    problems = []
    if not isinstance(word_list, dict) or not isinstance(word_list.get("version"), str) or not word_list["version"].strip():
        problems.append("the version is missing or empty")
    drawers = as_dict(word_list).get("drawers")
    if not isinstance(drawers, list) or not drawers:
        return problems + ["no drawers"]
    ids, owner = [], {}
    for index, item in enumerate(drawers):
        if not isinstance(item, dict) or sorted(item) != ["drawer", "exclude", "name", "words"]:
            problems.append(f"drawer {index}: fields {sorted(item) if isinstance(item, dict) else item!r}")
            continue
        ids.append(item["drawer"])
        if not (isinstance(item["words"], list) and item["words"]):
            problems.append(f"{item['drawer']}: no word")
        for key in ("words", "exclude"):
            for entry in item[key] if isinstance(item[key], list) else [item[key]]:
                if not (isinstance(entry, str) and entry and entry == entry.lower() and entry.isascii()
                        and entry == entry.strip()):
                    problems.append(f"{item['drawer']}.{key}: {entry!r} is not lower-case ASCII")
        for word in item["words"] if isinstance(item["words"], list) else []:
            if word in owner and owner[word] != item["drawer"]:
                problems.append(f"{word!r} is in two drawers ({owner[word]}, {item['drawer']})")
            owner.setdefault(word, item["drawer"])
    duplicates = sorted({drawer for drawer in ids if ids.count(drawer) > 1})
    if duplicates:
        problems.append(f"drawer ids used twice: {duplicates}")
    if not all(isinstance(drawer, str) and re.fullmatch(r"[a-z][a-z0-9_]*", drawer) for drawer in ids):
        problems.append(f"drawer ids {ids}")
    return problems


def check_drawer_words(report):
    """The built-in word list follows the rules and its pinned fingerprint; drawer_matches follows the rule on fixed
    sentences, also with a word list read from a file. Returns the built-in word list, or None."""
    namespace, problems = load_drawer_cells({"DRAWER_WORDS_FILE": ""})
    if not report.check(not problems, "the notebook's drawer word list and drawer_matches load from its source",
                        "; ".join(problems)):
        return None
    word_list = namespace.get("DRAWER_WORDS_V1")
    rule_problems = word_list_rule_problems(word_list)
    report.check(not rule_problems, "built-in drawer word list: a version, unique drawer ids, at least one word per "
                 "drawer, words and excluded phrases in lower-case ASCII, no word in two drawers", "; ".join(rule_problems[:5]))
    if rule_problems:
        return None
    drawers = word_list["drawers"]
    banned = [f"{item['drawer']}: {entry!r}" for item in drawers for entry in item["words"] + item["exclude"]
              if any(re.search(rf"\b{re.escape(name)}\b", entry) for name in DRAWER_BANNED_NAMES)]
    report.check(not banned, f"built-in drawer word list holds none of the event names {DRAWER_BANNED_NAMES} (as a whole "
                 "word, also inside phrases and excluded phrases)", "; ".join(banned[:5]))
    version, fingerprint = word_list["version"], drawer_fingerprint_of(drawers)
    report.check(DRAWER_FINGERPRINTS.get(version) == fingerprint and namespace.get("DRAWER_WORDS_FINGERPRINT") == fingerprint
                 and namespace.get("DRAWER_WORDS_VERSION") == version and namespace.get("DRAWER_WORDS_SOURCE") == BUILT_IN_SOURCE,
                 f"built-in drawer word list {version}: its fingerprint is the one pinned for this version (a changed "
                 "word needs a new version and a new pinned fingerprint)",
                 f"computed {fingerprint}, notebook {namespace.get('DRAWER_WORDS_FINGERPRINT')!r}, pinned "
                 f"{DRAWER_FINGERPRINTS.get(version)!r}")
    report.info(f"built-in drawer word list {version}: {len(drawers)} drawers, fingerprint {fingerprint}")

    def probe(matches, cases, items):
        wrong = []
        for sentence, expected in cases:
            got = matches(sentence)
            rule = {item["drawer"]: rule_matches(sentence, item["words"], item["exclude"]) for item in items}
            rule = {drawer: words for drawer, words in rule.items() if words}
            if not isinstance(got, dict) or sorted(got) != sorted(expected) or got != rule:
                wrong.append(f"{sentence!r}: got {got!r}, expected drawers {expected}, rule {rule!r}")
        return wrong

    wrong = probe(namespace["drawer_matches"], DRAWER_PROBES, drawers)
    report.check(not wrong, f"drawer_matches puts {len(DRAWER_PROBES)} fixed sentences into the expected drawers, with the "
                 "words the rule finds", "; ".join(wrong[:5]))

    folder = REPO_DIR / "data" / "checks" / "drawer-probe"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    probe_file = folder / "word-list.json"
    probe_file.write_text(json.dumps(DRAWER_FILE_PROBE), encoding="utf-8")
    from_file, problems = load_drawer_cells({"DRAWER_WORDS_FILE": str(probe_file)})
    wrong = problems or probe(from_file["drawer_matches"], DRAWER_FILE_PROBES, DRAWER_FILE_PROBE["drawers"])
    report.check(not wrong and from_file.get("DRAWER_WORDS_VERSION") == DRAWER_FILE_PROBE["version"]
                 and from_file.get("DRAWER_WORDS_SOURCE") == str(probe_file)
                 and from_file.get("DRAWER_WORDS_FINGERPRINT") == drawer_fingerprint_of(DRAWER_FILE_PROBE["drawers"]),
                 "a word list read through DRAWER_WORDS_FILE replaces the built-in one (version, source, fingerprint) and "
                 "follows the same rule", "; ".join(wrong[:3]))
    shutil.rmtree(folder, ignore_errors=True)
    return word_list


def check_drawers(report, label, brief, switches, built_in):
    """The drawers section and the drawer rows of why: the word list as used, counts that add up, spot checks that
    follow the rule, the groups of the posts in no drawer, and rows that pass the checks of every why row."""
    drawers = brief.get("drawers")
    if not report.check(isinstance(drawers, dict) and isinstance(drawers.get("drawers"), list) and bool(drawers["drawers"]),
                        f"{label}: drawers section has content"):
        return
    topics, why = as_dict(brief.get("topics")), as_dict(brief.get("why"))
    items = [item for item in drawers["drawers"] if isinstance(item, dict)]
    n, any_n, none_n = drawers.get("n_posts"), drawers.get("n_in_any_drawer"), drawers.get("n_in_no_drawer")

    # The word list as used: source, version and a fingerprint that follows from the drawers in the brief.
    file_value = switches.get("DRAWER_WORDS_FILE")
    listed = [{"drawer": item.get("drawer"), "name": item.get("name"), "words": item.get("words"),
               "exclude": item.get("excluded_phrases")} for item in items]
    fingerprint = drawer_fingerprint_of(listed)
    list_ok = (drawers.get("post_table") == "topic_posts" and drawers.get("text") == "text"
               and len(items) == len(drawers["drawers"]) and drawers.get("word_list_fingerprint") == fingerprint
               and drawers.get("word_list_source") == (BUILT_IN_SOURCE if file_value == "" else file_value))
    if file_value == "":
        list_ok = (list_ok and isinstance(built_in, dict) and drawers.get("word_list_version") == built_in.get("version")
                   and listed == built_in.get("drawers") and DRAWER_FINGERPRINTS.get(drawers.get("word_list_version")) == fingerprint)
    report.check(list_ok, f"{label}: drawers record the word list as used (source, version, and a fingerprint that "
                 "follows from the drawers listed; with DRAWER_WORDS_FILE \"\" exactly the built-in list and its pinned "
                 "fingerprint)", f"source {drawers.get('word_list_source')!r}, version {drawers.get('word_list_version')!r}, "
                 f"fingerprint {drawers.get('word_list_fingerprint')!r} (from the drawers: {fingerprint})")

    # Counts: every post is in 0 or more drawers; in a drawer + in none = all posts; drawer sizes overlap.
    memberships = drawers.get("memberships_per_post")
    keys = sorted(int(key) for key in memberships if re.fullmatch(r"0|[1-9][0-9]*", key)) \
        if isinstance(memberships, dict) else []
    memberships_ok = (isinstance(memberships, dict) and len(keys) == len(memberships) and keys == list(range(len(keys)))
                      and bool(keys) and all(is_count(value) for value in memberships.values()))
    sizes = [item.get("n_posts") for item in items]
    counts_ok = (is_count(n) and n == topics.get("n_posts") and is_count(any_n) and is_count(none_n)
                 and any_n + none_n == n and memberships_ok and sum(memberships.values()) == n
                 and memberships["0"] == none_n and all(is_count(size) and size <= any_n for size in sizes)
                 and sum(int(key) * value for key, value in memberships.items()) == sum(sizes)
                 and same_value(drawers.get("share_in_any_drawer"), share_of(any_n, n))
                 and same_value(drawers.get("share_in_no_drawer"), share_of(none_n, n))
                 and all(same_value(item.get("share"), share_of(item.get("n_posts"), n)) for item in items))
    report.check(counts_ok, f"{label}: drawer counts add up: posts in a drawer + posts in none = all posts = the topic "
                 "post table; posts by number of drawers add up to all posts and \"0\" equals the posts in none; the "
                 "drawer sizes add up to the sum of k x posts in k drawers; shares follow from the counts",
                 f"{n!r} posts (topic table {topics.get('n_posts')!r}), {any_n!r} in a drawer, {none_n!r} in none, "
                 f"by number of drawers {memberships!r}, drawer sizes {sizes} (sum {sum(s for s in sizes if is_count(s))})")

    # Spot checks: at most DRAWER_QUOTES different posts of the drawer, with the words the rule finds in them.
    problems = []
    if drawers.get("quotes_per_drawer") != DRAWER_QUOTES:
        problems.append(f"quotes_per_drawer {drawers.get('quotes_per_drawer')!r}")
    for item in items:
        quotes, size = item.get("quotes"), item.get("n_posts")
        quotes = quotes if isinstance(quotes, list) else None
        if quotes is None or not is_count(size) or not (min(1, size) <= len(quotes) <= min(DRAWER_QUOTES, size)):
            problems.append(f"{item.get('drawer')}: {len(quotes) if quotes is not None else quotes!r} quotes for {size!r} posts")
            continue
        if not all(isinstance(quote, dict) for quote in quotes) or \
                len({quote.get("post_id") for quote in quotes}) != len(quotes) or \
                len({quote.get("text") for quote in quotes}) != len(quotes):
            problems.append(f"{item.get('drawer')}: quotes are not different posts with different texts")
            continue
        for quote in quotes:
            words = quote.get("matched_words")
            found = rule_matches(quote.get("text"), item.get("words") or [], item.get("excluded_phrases") or [])
            if not (isinstance(quote.get("post_id"), str) and quote.get("label") in (0, 1, 2) and isinstance(words, list)
                    and words and set(words) <= set(item.get("words") or []) and words == found):
                problems.append(f"{item.get('drawer')}: {quote.get('post_id')!r} matched words {words!r}, the rule "
                                f"finds {found!r} in {quote.get('text')!r}")
    report.check(not problems, f"{label}: every drawer has up to {DRAWER_QUOTES} spot checks: different posts with "
                 "different texts that belong to the drawer, each with the non-empty list of words the rule finds in it "
                 "(words of that drawer, in word-list order)", "; ".join(problems[:5]))

    # The posts in no drawer, grouped like the topics.
    other = as_dict(drawers.get("other"))
    groups = other.get("groups") if isinstance(other.get("groups"), list) else None
    wanted, distinct, made, other_n = (switches.get("N_OTHER_GROUPS"), other.get("n_distinct_vectors"),
                                       other.get("n_groups"), other.get("n_posts"))
    group_problems = []
    if not (other_n == none_n and is_count(other_n) and other.get("method") == "kmeans" and is_count(wanted)
            and other.get("n_groups_requested") == wanted and is_count(distinct) and distinct <= other_n
            and (other_n == 0 or distinct >= 1) and made == min(wanted, distinct) and groups is not None
            and len(groups) == made and other.get("quotes_per_group") == topics.get("quotes_per_topic")):
        group_problems.append(f"{other_n!r} posts (in none: {none_n!r}), method {other.get('method')!r}, groups "
                              f"{made!r} of {other.get('n_groups_requested')!r} (switch {wanted!r}), {distinct!r} "
                              f"distinct vectors, {len(groups) if groups is not None else None} listed")
    groups = [group for group in groups or [] if isinstance(group, dict)]
    if [group.get("group") for group in groups] != list(range(len(groups))):
        group_problems.append(f"group numbers {[group.get('group') for group in groups]}")
    for group in groups:
        size, words, quotes = group.get("n_posts"), group.get("top_words"), group.get("quotes")
        if not (is_count(size) and size > 0 and is_count(other_n) and same_value(group.get("share"), share_of(size, other_n))
                and isinstance(words, list) and all(isinstance(word, str) and word for word in words)
                and is_count(topics.get("name_words")) and len(words) <= topics["name_words"]
                and group.get("name") == " / ".join(words)
                and not (switches.get("TOPIC_SCOPE") == "btc" and SELECTION_WORDS & set(words))
                and isinstance(quotes, list) and 1 <= len(quotes) <= min(other.get("quotes_per_group") or 0, size)
                and all(isinstance(quote, dict) and isinstance(quote.get("post_id"), str) and quote.get("label") in (0, 1, 2)
                        and isinstance(quote.get("text"), str) for quote in quotes)
                and len({quote["post_id"] for quote in quotes}) == len(quotes)
                and len({quote["text"] for quote in quotes}) == len(quotes)):
            group_problems.append(f"group {group.get('group')}: {size!r} posts, share {group.get('share')!r}, name "
                                  f"{group.get('name')!r}, {len(quotes) if isinstance(quotes, list) else quotes!r} quotes")
    if sum(group.get("n_posts") for group in groups if is_count(group.get("n_posts"))) != other_n:
        group_problems.append(f"group sizes do not add up to {other_n!r}")
    report.check(not group_problems, f"{label}: the posts in no drawer are grouped like the topics: N_OTHER_GROUPS groups "
                 "(fewer only with fewer distinct vectors, none without posts) numbered from 0, sizes adding up to the posts "
                 "in no drawer, shares from the sizes, names from the top words, different quotes",
                 "; ".join(group_problems[:5]))

    # The why rows: one per drawer, plus the posts in a drawer and in none, which add up to the overall row.
    rows = [row for row in why.get("drawers") or [] if isinstance(row, dict)] if isinstance(why.get("drawers"), list) else []
    any_row, no_row, overall = as_dict(why.get("any_drawer")), as_dict(why.get("no_drawer")), as_dict(why.get("overall"))
    report.check(len(rows) == len(items) == len(why.get("drawers") or [])
                 and [(row.get("drawer"), row.get("name"), row.get("n_posts")) for row in rows]
                 == [(item.get("drawer"), item.get("name"), item.get("n_posts")) for item in items]
                 and any_row.get("n_posts") == any_n and no_row.get("n_posts") == none_n and bool(overall),
                 f"{label}: why has one row per drawer of the drawers section (same order, ids, names and sizes), and "
                 "any_drawer and no_drawer hold the posts in a drawer and in none",
                 f"why drawers {[(row.get('drawer'), row.get('n_posts')) for row in rows]}; any {any_row.get('n_posts')!r} "
                 f"({any_n!r}), none {no_row.get('n_posts')!r} ({none_n!r})")
    calibration = as_dict(why.get("calibration"))
    min_posts, min_bb = switches.get("MIN_TOPIC_POSTS"), switches.get("MIN_BULLISH_BEARISH_POSTS")
    identity, marks = [], []
    overall_share = label_share_from(overall)
    for where, row in [("any drawer", any_row), ("no drawer", no_row)] + [(f"drawer {row.get('drawer')}", row) for row in rows]:
        row_identity, row_shares, row_fair = why_row_problems(row, where)
        row_half, row_marks = half_row_problems(row, where, calibration, min_bb)
        identity += row_identity + row_shares + row_fair + row_half
        marks += row_marks
    for row in rows:
        if not same_value(row.get("label_vs_overall_pp"), points(label_share_from(row), overall_share)):
            identity.append(f"drawer {row.get('drawer')}: label_vs_overall_pp {row.get('label_vs_overall_pp')!r}")
        if row.get("too_few_posts") is not (is_count(min_posts) and is_count(row.get("n_posts")) and row["n_posts"] < min_posts):
            marks.append(f"drawer {row.get('drawer')}: too_few_posts {row.get('too_few_posts')!r} with {row.get('n_posts')!r} posts")
    report.check(not identity, f"{label}: every drawer row and the rows any_drawer and no_drawer pass the checks of a why "
                 "row (counts add up, shares and differences follow from them, calls at 0.5 and the calibrated share "
                 "follow the calibration; label_vs_overall_pp from the shares)", "; ".join(identity[:5]))
    report.check(not marks, f"{label}: drawer rows are marked too few by MIN_TOPIC_POSTS (posts) and every row by "
                 "MIN_BULLISH_BEARISH_POSTS (bearish + bullish posts by label)", "; ".join(marks[:5]))

    def adds_up(part, key, keys):
        parts = [counts_of(as_dict(row.get(part) if part else row).get(key), keys) for row in (any_row, no_row)]
        whole = counts_of(as_dict(overall.get(part) if part else overall).get(key), keys)
        return None not in parts and whole is not None and {k: parts[0][k] + parts[1][k] for k in keys} == whole

    sums_ok = (all(adds_up(None, key, keys) for key, keys in
                   (("label_counts", TOPIC_CLASSES), ("model_counts", MODEL_CALLS), ("half_counts", HALF_CALLS)))
               and all(adds_up("fair", key, keys) for key, keys in
                       (("label_counts", FAIR_CLASSES), ("model_counts", MODEL_CALLS), ("half_counts", HALF_CALLS)))
               and is_count(any_row.get("n_posts")) and is_count(no_row.get("n_posts"))
               and any_row["n_posts"] + no_row["n_posts"] == overall.get("n_posts")
               and is_count(as_dict(any_row.get("fair")).get("n_posts")) and is_count(as_dict(no_row.get("fair")).get("n_posts"))
               and as_dict(any_row.get("fair"))["n_posts"] + as_dict(no_row.get("fair"))["n_posts"]
               == as_dict(overall.get("fair")).get("n_posts"))
    report.check(sums_ok, f"{label}: any_drawer + no_drawer = the overall row (posts, label classes, model calls, calls at "
                 "0.5, and the fair comparison's posts, labels, calls and calls at 0.5)")


def stable_drawers(brief):
    """The drawers section and the parts of the drawer rows of why that do not depend on training, or None."""
    if not isinstance(brief, dict) or not isinstance(brief.get("drawers"), dict):
        return None
    why = as_dict(brief.get("why"))

    def part(row, keys):
        row = as_dict(row)
        return {**{key: row.get(key) for key in keys},
                "fair": {key: as_dict(row.get("fair")).get(key) for key in WHY_STABLE_FAIR_FIELDS}}

    rows = why.get("drawers") if isinstance(why.get("drawers"), list) else []
    totals = ["n_posts", "label_counts", "label_bullish_share", "too_few_bullish_bearish"]
    return {"drawers": brief["drawers"], "rows": [part(row, DRAWER_STABLE_ROW_FIELDS) for row in rows],
            "any_drawer": part(why.get("any_drawer"), totals), "no_drawer": part(why.get("no_drawer"), totals)}


def compare_drawers(report, first, again):
    """Two runs with the same switches: identical drawers sections and drawer rows apart from what training changes."""
    left, right = stable_drawers(first), stable_drawers(again)
    detail = ""
    if left is None or right is None:
        detail = "a brief or its drawers section is missing"
    elif left != right:
        detail = "differing parts: " + ", ".join(key for key in left if left[key] != right[key])
    report.check(left is not None and left == right,
                 "default switches run twice: identical drawers sections (counts, spot checks, groups of the posts in no "
                 "drawer), and the drawer rows of why agree on everything that does not depend on training", detail)


def print_drawers(brief):
    """Print the word list, the drawer table with its three bullish shares and marks, the spot checks and the groups
    of the posts in no drawer (full run)."""
    drawers, why = as_dict(brief.get("drawers")), as_dict(brief.get("why"))
    if not drawers:
        return

    def share(value):
        return f"{value:.4f}" if is_number(value) else "-"

    def pp(value):
        return f"{value:+.2f}" if is_number(value) else "-"

    print("\nDrawers (full run)")
    print(f"  word list:                     {drawers.get('word_list_version')} ({drawers.get('word_list_source')}), "
          f"fingerprint {drawers.get('word_list_fingerprint')}")
    print(f"  posts:                         {drawers.get('n_posts')}   in at least one drawer: "
          f"{drawers.get('n_in_any_drawer')} ({share(drawers.get('share_in_any_drawer'))})   in none: "
          f"{drawers.get('n_in_no_drawer')} ({share(drawers.get('share_in_no_drawer'))})")
    print(f"  posts by number of drawers:    {drawers.get('memberships_per_post')}")
    print("  columns: drawer, posts, share of all posts, too few posts, too few bullish + bearish, label bearish / neutral /")
    print("           bullish, bullish share by label, at 0.5, calibrated, label minus all (pp), fair posts, fair share by")
    print("           label, fair at 0.5, fair calibrated, fair calibrated minus label (pp); - = null; word-list order")
    print(f"  {'drawer':>15} {'posts':>6} {'share':>7} {'few':>3} {'fbb':>3} {'bear':>6} {'neut':>6} {'bull':>6} "
          f"{'label':>7} {'half':>7} {'calib':>7} {'vs all':>7} {'fair':>6} {'f-label':>7} {'f-half':>7} {'f-calib':>7} "
          f"{'c-l pp':>7}")
    sizes = {item.get("drawer"): item.get("share") for item in drawers.get("drawers") or [] if isinstance(item, dict)}
    overall_share = as_dict(why.get("overall")).get("label_bullish_share")
    rows = [("all", as_dict(why.get("overall")), "", 1.0), ("any drawer", as_dict(why.get("any_drawer")), "",
                                                           drawers.get("share_in_any_drawer")),
            ("no drawer", as_dict(why.get("no_drawer")), "", drawers.get("share_in_no_drawer"))]
    rows += [(row.get("drawer"), row, "yes" if row.get("too_few_posts") else "no", sizes.get(row.get("drawer")))
             for row in why.get("drawers") or [] if isinstance(row, dict)]
    for name, row, few, size in rows:
        labels, fair = as_dict(row.get("label_counts")), as_dict(row.get("fair"))
        label_share = row.get("label_bullish_share")
        versus = row.get("label_vs_overall_pp") if "label_vs_overall_pp" in row else (
            None if name == "all" or not (is_number(label_share) and is_number(overall_share))
            else 100 * (label_share - overall_share))
        print(f"  {name:>15} {str(row.get('n_posts')):>6} {share(size):>7} {few:>3} "
              f"{'yes' if row.get('too_few_bullish_bearish') else 'no':>3} "
              f"{str(labels.get('0')):>6} {str(labels.get('1')):>6} {str(labels.get('2')):>6} "
              f"{share(label_share):>7} {share(row.get('half_bullish_share')):>7} "
              f"{share(row.get('calibrated_bullish_share')):>7} {pp(versus):>7} {str(fair.get('n_posts')):>6} "
              f"{share(fair.get('label_bullish_share')):>7} {share(fair.get('half_bullish_share')):>7} "
              f"{share(fair.get('calibrated_bullish_share')):>7} {pp(fair.get('calibrated_minus_label_pp')):>7}")
    calibration = as_dict(why.get("calibration"))
    true_rate, false_rate = calibration.get("true_bullish_rate"), calibration.get("false_bullish_rate")
    clipped = []
    if calibration.get("status") == "applied" and is_number(true_rate) and is_number(false_rate):
        for name, row, _, _ in rows[1:]:
            for where, part in (("row", row), ("fair", as_dict(row.get("fair")))):
                half = part.get("half_bullish_share")
                if is_number(half):
                    raw = (half - false_rate) / (true_rate - false_rate)
                    if raw < 0 or raw > 1:
                        clipped.append(f"{name} {where} {raw:.4f} -> {min(1.0, max(0.0, raw)):.0f}")
    print(f"  calibrated share clipped to 0 or 1: {', '.join(clipped) or 'none'}")
    print("  spot checks per drawer ([label, post_id; words found] text):")
    for item in drawers.get("drawers") or []:
        print(f"    {item.get('drawer')}: {item.get('name')} ({item.get('n_posts')} posts)")
        for quote in item.get("quotes") or []:
            print(f"        [{quote.get('label')}, {quote.get('post_id')}; {', '.join(quote.get('matched_words') or [])}] "
                  f"{quote.get('text')}")
    other = as_dict(drawers.get("other"))
    print(f"  posts in no drawer:            {other.get('n_posts')}, {other.get('n_groups')} groups (asked for "
          f"{other.get('n_groups_requested')}, {other.get('n_distinct_vectors')} distinct vectors), largest first; "
          "share of the posts in no drawer:")
    groups = [group for group in other.get("groups") or [] if isinstance(group, dict)]
    for group in sorted(groups, key=lambda group: (-(group.get("n_posts") or 0), group.get("group") or 0)):
        print(f"    group {group.get('group')}: {group.get('n_posts')} posts, share {share(group.get('share'))} | "
              f"{group.get('name')}")
        for quote in group.get("quotes") or []:
            print(f"        [{quote.get('label')}, {quote.get('post_id')}] {quote.get('text')}")


# ---------------------------------------------------------------- the brief's format and the panel (chapter 16)

def code_cells():
    """The notebook source's code cells, in order."""
    return [cell.source for cell in jupytext.read(SOURCE, fmt="py:percent").cells if cell.cell_type == "code"]


def cell_defining(name):
    """Source of the code cell whose top level defines name (assignment, def or import), or None."""
    for source in code_cells():
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
                return source
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name
                                                    for target in node.targets):
                return source
    return None


def schema_field_rows(schema):
    """{path: (type, required, null)} for every field the rules allow, $refs expanded; and problems found."""
    definitions = schema.get("$defs", {}) if isinstance(schema, dict) else {}
    rows, problems = {}, []

    def resolve(node):
        node = dict(node) if isinstance(node, dict) else {}
        while "$ref" in node:
            reference = node.pop("$ref")
            name = reference.rsplit("/", 1)[-1]
            if not reference.startswith("#/$defs/") or name not in definitions:
                problems.append(f"unknown reference {reference}")
                return node
            node = {**definitions[name], **node}
        return node

    def walk(node, path, required):
        node = resolve(node)
        types = node.get("type")
        types = [types] if isinstance(types, str) else list(types or [])
        kinds = [kind for kind in types if kind != "null"]
        if len(kinds) != 1:
            problems.append(f"{path or 'brief'}: needs exactly one type besides null, has {types}")
            return
        kind = kinds[0]
        if path:
            rows[path] = (kind, required, "yes" if "null" in types else "no")
        if kind == "object":
            if "properties" in node:
                if node.get("additionalProperties") is not False:
                    problems.append(f"{path or 'brief'}: an object with fields must set additionalProperties to false")
                for name, child in node["properties"].items():
                    walk(child, f"{path}.{name}" if path else name,
                         "yes" if name in node.get("required", []) else "no")
            elif isinstance(node.get("additionalProperties"), dict):
                key = resolve(node.get("propertyNames", {})).get("title")
                if not key:
                    problems.append(f"{path}: an object with varying keys needs propertyNames with a title")
                    return
                walk(node["additionalProperties"], f"{path}.<{key}>", "—")
            else:
                problems.append(f"{path}: an object must list its fields or describe its keys")
        elif kind == "array":
            if not isinstance(node.get("items"), dict):
                problems.append(f"{path}: an array must describe its items")
                return
            walk(node["items"], f"{path}[]", "—")

    walk(schema, "", "yes")
    return rows, problems


def doc_field_rows(text):
    """{path: (type, required, null)} from every field table of the format document; and problems found."""
    rows, problems, inside = {}, [], False
    for number, line in enumerate(text.splitlines(), start=1):
        if line.strip() == DOC_TABLE_HEADER:
            inside = True
            continue
        if inside and not line.startswith("|"):
            inside = False
        if not inside:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(set(cell) <= {"-", " ", ":"} for cell in cells):
            continue  # the line under the header
        if len(cells) != 5 or not re.fullmatch(r"`[^`]+`", cells[0]):
            problems.append(f"line {number}: not a field row with five cells")
            continue
        path = cells[0].strip("`")
        if path in rows:
            problems.append(f"line {number}: {path} listed twice")
        rows[path] = tuple(cells[1:4])
    return rows, problems


def outside_names(source):
    """Names a cell reads that it neither defines nor imports itself (builtins excluded)."""
    table = symtable.symtable(source, "cell", "exec")
    bound = {symbol.get_name() for symbol in table.get_symbols() if symbol.is_assigned() or symbol.is_imported()}
    used = set()

    def walk(scope, top):
        for symbol in scope.get_symbols():
            if symbol.is_referenced() and (top or symbol.is_global()):
                used.add(symbol.get_name())
        for child in scope.get_children():
            walk(child, False)

    walk(table, True)
    return sorted(used - bound - set(dir(builtins)))


def load_format_rules(report):
    """Static checks of the format rules, their notebook copy and their documentation.

    Returns the notebook's own rules and check functions (BRIEF_SCHEMA, brief_problems, save_brief, ...),
    executed from its source, or None when they cannot be loaded.
    """
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import SchemaError

    try:
        schema_text = SCHEMA_FILE.read_text(encoding="utf-8")
        schema = json.loads(schema_text)
        Draft202012Validator.check_schema(schema)
        detail = ""
    except (OSError, ValueError, SchemaError) as exc:
        schema_text, schema, detail = None, None, f"{type(exc).__name__}: {str(exc)[:200]}"
    if not report.check(isinstance(schema, dict), f"{shown(SCHEMA_FILE)} is a valid JSON Schema (draft 2020-12)", detail):
        return None
    version = as_dict(as_dict(schema.get("properties")).get("format_version")).get("const")
    report.info(f"format rules: {schema.get('title')!r}, format_version {version!r}")

    # The notebook's copy of the rules and its check functions, executed from the source.
    namespace, problems = {"__name__": "notebook_cells"}, []
    for name in ("BRIEF_SCHEMA_TEXT", "brief_problems", "save_brief"):
        source = cell_defining(name)
        if source is None:
            problems.append(f"no cell defines {name}")
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(source, f"cell defining {name}", "exec"), namespace)
        except Exception as exc:  # the cell itself is broken
            problems.append(f"cell defining {name}: {type(exc).__name__}: {exc}")
    if not report.check(not problems, "the notebook's rules and check functions load from its source", "; ".join(problems)):
        return None
    same = namespace.get("BRIEF_SCHEMA_TEXT") == schema_text and namespace.get("BRIEF_SCHEMA") == schema
    report.check(same, f"the notebook's copy of the format rules is identical to {shown(SCHEMA_FILE)} (one set of rules)",
                 "" if same else "copy the file's text into BRIEF_SCHEMA_TEXT unchanged")

    # The version: the same in the rules, the notebook's brief and the document.
    source = SOURCE.read_text(encoding="utf-8")
    written = re.search(r'"format_version": "([^"]*)"', source)
    documented = re.search(r"describes format \*\*([^*]+)\*\*", FORMAT_DOC.read_text(encoding="utf-8")
                           if FORMAT_DOC.exists() else "")
    versions = [version, written.group(1) if written else None, documented.group(1) if documented else None]
    report.check(isinstance(version, str) and len(set(versions)) == 1,
                 "format_version is the same in the rules, in the brief the notebook writes and in the format document",
                 f"rules {versions[0]!r}, notebook {versions[1]!r}, document {versions[2]!r}")

    # The document lists exactly the fields of the rules, and the rules allow no undocumented field.
    expected, schema_problems = schema_field_rows(schema)
    report.check(not schema_problems, "every object of the format rules lists its fields or names its keys, and allows "
                 "no other field (so every field of a passing brief is documented)", "; ".join(schema_problems[:5]))
    listed, doc_problems = doc_field_rows(FORMAT_DOC.read_text(encoding="utf-8") if FORMAT_DOC.exists() else "")
    missing = [path for path in expected if path not in listed]
    extra = [path for path in listed if path not in expected]
    differ = [f"{path}: document {listed[path]}, rules {expected[path]} (type, required, null)"
              for path in expected if path in listed and listed[path] != expected[path]]
    report.check(bool(expected) and not (missing or extra or differ or doc_problems),
                 f"{shown(FORMAT_DOC)} lists every field of the rules with the same type, required and null "
                 f"({len(listed)} of {len(expected)} fields)",
                 "; ".join(doc_problems[:3] + [f"missing {missing[:5]}"] * bool(missing)
                           + [f"not in the rules {extra[:5]}"] * bool(extra) + differ[:3]))

    # Saving: only through save_brief, which checks before it writes and refuses NaN.
    save_source = cell_defining("save_brief") or ""
    function = next((node for node in ast.walk(ast.parse(save_source))
                     if isinstance(node, ast.FunctionDef) and node.name == "save_brief"), None)
    order = {}
    for node in ast.walk(function) if function else []:
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            order.setdefault(name, node.lineno)
            if name == "dumps":
                order["allow_nan=False"] = any(keyword.arg == "allow_nan" and isinstance(keyword.value, ast.Constant)
                                               and keyword.value.value is False for keyword in node.keywords)
    save_cell = cell_defining("brief_path") or ""
    cells = code_cells()
    writers = [index for index, cell in enumerate(cells) if re.search(r"\.write_text\(|json\.dump\(", cell)]
    report.check(function is not None and "validate_brief" in order and "write_text" in order
                 and order["validate_brief"] < order["write_text"] and order.get("allow_nan=False") is True
                 and re.search(r"\bsave_brief\(brief, brief_path\)", save_cell) is not None
                 and writers == [cells.index(save_source) if save_source in cells else -1],
                 "the notebook saves the brief only through save_brief, which runs validate_brief before writing and "
                 "refuses NaN (allow_nan=False)",
                 f"calls in save_brief {order}, cells that write files {writers}")
    return namespace


def check_panel_source(report, defaults, others):
    """The panel switch, and the panel cell: reads only the brief file, no public link, no usage statistics."""
    report.check(defaults.get(PANEL_SWITCH) is True and PANEL_SWITCH not in others,
                 f"{PANEL_SWITCH} exists, defaults to True and is not a two-state switch (the checks always set it to False)",
                 f"default {defaults.get(PANEL_SWITCH)!r}, two-state: {PANEL_SWITCH in others}")

    panel_cell = cell_defining("build_panel") or ""
    outside = outside_names(panel_cell) if panel_cell else ["(no cell defines build_panel)"]
    functions = {node.name: node for node in ast.walk(ast.parse(panel_cell)) if isinstance(node, ast.FunctionDef)}
    parameters = {name: [argument.arg for argument in functions[name].args.args] if name in functions else None
                  for name in ("panel_tables", "build_panel")}
    cells = code_cells()
    last = cells[-1] if cells else ""
    report.check(panel_cell != "" and not outside and all(value == ["brief_path"] for value in parameters.values())
                 and re.search(r"\bbuild_panel\(brief_path\)", last) is not None,
                 "the panel cell uses no other variable of the notebook, panel_tables and build_panel take only the "
                 "brief file's path, and the last cell builds the panel from the saved file",
                 f"names from outside the cell {outside}, parameters {parameters}")

    # No public link, no server beyond this machine, Gradio's usage statistics off before the import.
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]

    def keyword(call, name):
        found = [item.value for item in call.keywords if item.arg == name]
        return found[0].value if found and isinstance(found[0], ast.Constant) else ("missing" if not found else "not a constant")

    launches = [node for node in calls if node.func.attr == "launch"]
    blocks = [node for node in calls if node.func.attr == "Blocks"]
    every_share = [keyword(node, "share") for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and any(item.arg == "share" for item in node.keywords)]
    every_server = [keyword(node, "server_name") for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and any(item.arg == "server_name" for item in node.keywords)]
    report.check(bool(launches) and all(keyword(node, "share") is False and keyword(node, "server_name") == LOOPBACK_HOST
                                        for node in launches)
                 and all(value is False for value in every_share) and all(value == LOOPBACK_HOST for value in every_server),
                 f"the panel is started with share=False and server_name=\"{LOOPBACK_HOST}\" only (no public link, "
                 "nothing served beyond this machine)",
                 f"launch calls: {[(keyword(node, 'share'), keyword(node, 'server_name')) for node in launches]}; "
                 f"every share= {every_share}, every server_name= {every_server}")

    lines = source.splitlines()
    setting = re.compile(r'^\s*os\.environ\["GRADIO_ANALYTICS_ENABLED"\] = "False"')
    importing = re.compile(r"^\s*(?:import gradio\b|from gradio\b)")
    first_setting = next((number for number, line in enumerate(lines) if setting.match(line)), None)
    first_import = next((number for number, line in enumerate(lines) if importing.match(line)), None)
    cell_lines = panel_cell.splitlines()
    cell_setting = next((number for number, line in enumerate(cell_lines) if setting.match(line)), None)
    cell_import = next((number for number, line in enumerate(cell_lines) if importing.match(line)), None)
    setup_cell = cell_defining("REQUIRED_PACKAGES") or ""  # the install cell, before anything else is imported
    in_setup = any(setting.match(line) for line in setup_cell.splitlines())
    environment = (REPO_DIR / "scripts" / "env.sh").read_text(encoding="utf-8")
    report.check(first_setting is not None and first_import is not None and first_setting < first_import
                 and in_setup and cell_setting is not None and cell_import is not None and cell_setting < cell_import
                 and bool(blocks) and all(keyword(node, "analytics_enabled") is False for node in blocks)
                 and re.search(r"^export GRADIO_ANALYTICS_ENABLED=False$", environment, flags=re.MULTILINE) is not None
                 and os.environ.get("GRADIO_ANALYTICS_ENABLED") == "False",
                 "Gradio's usage statistics are off: GRADIO_ANALYTICS_ENABLED=False is set in the setup cell and again "
                 "in the panel cell, both before any import of gradio (notebook), and in scripts/env.sh (checks); the "
                 "page is built with analytics_enabled=False",
                 f"setup cell sets it: {in_setup}; notebook lines: setting {first_setting}, first import {first_import}; "
                 f"panel cell: setting "
                 f"{cell_setting}, import {cell_import}; Blocks calls "
                 f"{[keyword(node, 'analytics_enabled') for node in blocks]}; this process: "
                 f"{os.environ.get('GRADIO_ANALYTICS_ENABLED')!r}")


def check_format(report, label, brief, rules):
    """The brief passes the notebook's own check against the format rules."""
    if rules is None:
        report.check(False, f"{label}: brief passes the format rules", "the rules could not be loaded")
        return
    problems = rules["brief_problems"](brief)
    report.check(not problems, f"{label}: brief passes the format rules ({shown(SCHEMA_FILE)}, no NaN or infinity)",
                 "; ".join(problems[:5]))


DELETE = object()  # a counterexample that removes the field
NAN, INFINITY = float("nan"), float("inf")


def changed_copy(document, path, change):
    """A deep copy of document with the field at path removed (DELETE) or replaced by change(old value)."""
    result = copy.deepcopy(document)
    target = result
    for key in path[:-1]:
        target = target[key]
    if change is DELETE:
        del target[path[-1]]
    else:
        target[path[-1]] = change(target.get(path[-1]) if isinstance(target, dict) else target[path[-1]])
    return result


PROBE_DAY = 7  # a day of new_words.layers.retail.days that the counterexamples give a new word (judged by default)


def with_new_word(day, **change):
    """A copy of a day of new_words whose first new word (its own, or a valid made-up one) is changed."""
    day = copy.deepcopy(day)
    words = day.get("new_words") or [{"word": "probeword", "n_posts": 1, "past_daily_mean": 0.0, "ratio": None,
                                      "quotes": [{"post_id": "probe:0", "text": "probeword"}]}]
    words[0] = {**words[0], **change}
    day["new_words"] = words
    return day


def day_before(day):
    from datetime import date, timedelta
    return (date.fromisoformat(day) - timedelta(days=1)).isoformat()


# Locked counterexamples: each breaks a valid brief in one place, and the format check must reject it with a
# problem at the given path that mentions the given word. Kinds 1 to 5 are the five kinds the format check must
# catch at least; kind 6 adds a null where the format does not allow one, and a field the format does not know.
BROKEN_BRIEFS = [
    *[("1 top-level field missing", (field,), DELETE, "brief", field) for field in TOP_LEVEL_FIELDS],
    ("2 required field missing in a section", ("scope", "granularity"), DELETE, "scope", "granularity"),
    ("2 required field missing in a section", ("dataset", "n_posts"), DELETE, "dataset", "n_posts"),
    ("2 required field missing in a section", ("sentiment", "evaluation"), DELETE, "sentiment", "evaluation"),
    ("2 required field missing in a section", ("sentiment", "evaluation", "rows", 0, "macro_f1"), DELETE,
     "sentiment.evaluation.rows[0]", "macro_f1"),
    ("2 required field missing in a section", ("topics", "topics"), DELETE, "topics", "topics"),
    ("2 required field missing in a section", ("why", "overall"), DELETE, "why", "overall"),
    ("2 required field missing in a section", ("why", "topics", 0, "fair", "label_bullish_share"), DELETE,
     "why.topics[0].fair", "label_bullish_share"),
    ("2 required field missing in a section", ("why", "calibration"), DELETE, "why", "calibration"),
    ("3 wrong type", ("dataset", "n_posts"), str, "dataset.n_posts", "integer"),  # a count written as text
    ("3 wrong type", ("why", "overall", "label_bullish_share"), lambda old: "0.5", "why.overall.label_bullish_share",
     "number"),  # a share written as text
    ("3 wrong type", ("topics", "topics"), lambda old: {}, "topics.topics", "array"),  # a list written as an object
    ("3 wrong type", ("scope",), lambda old: [], "scope", "object"),  # an object written as a list
    ("3 wrong type", ("dataset", "cleaning", "keep_negations"), lambda old: "true", "dataset.cleaning.keep_negations",
     "boolean"),
    ("3 wrong type", ("topics", "topics", 0, "n_posts"), lambda old: 1.5, "topics.topics[0].n_posts", "integer"),
    ("4 granularity not allowed", ("scope", "granularity"), lambda old: "weekly", "scope.granularity", "weekly"),
    ("4 granularity not allowed", ("scope", "granularity"), lambda old: "Batch", "scope.granularity", "Batch"),
    ("4 granularity not allowed", ("scope", "granularity"), lambda old: "", "scope.granularity", "''"),
    ("4 granularity not allowed", ("scope", "granularity"), lambda old: None, "scope.granularity", "None"),
    ("4 granularity not allowed", ("scope", "granularity"), lambda old: 1, "scope.granularity", "1"),
    ("5 NaN or infinity", ("dataset", "class_shares", "0"), lambda old: NAN, "dataset.class_shares.0", "nan"),
    ("5 NaN or infinity", ("why", "overall", "fair", "model_minus_label_pp"), lambda old: NAN,
     "why.overall.fair.model_minus_label_pp", "nan"),
    ("5 NaN or infinity", ("dataset", "n_posts"), lambda old: NAN, "dataset.n_posts", "nan"),
    ("5 NaN or infinity", ("why", "scoring", "seconds"), lambda old: INFINITY, "why.scoring.seconds", "inf"),
    ("5 NaN or infinity", ("sentiment", "evaluation", "rows", 0, "accuracy"), lambda old: -INFINITY,
     "sentiment.evaluation.rows[0].accuracy", "-inf"),
    ("6 null not allowed, or an unknown field", ("dataset", "n_posts"), lambda old: None, "dataset.n_posts", "None"),
    ("6 null not allowed, or an unknown field", ("topics", "n_topics"), lambda old: None, "topics.n_topics", "None"),
    ("6 null not allowed, or an unknown field", ("why", "topics", 0, "name"), lambda old: None, "why.topics[0].name",
     "None"),
    ("6 null not allowed, or an unknown field", ("why", "extra_field"), lambda old: 1, "why", "extra_field"),
    ("6 null not allowed, or an unknown field", ("why", "overall", "too_few_bullish_bearish"), lambda old: None,
     "why.overall.too_few_bullish_bearish", "None"),
    ("2 required field missing in a section", ("drawers", "other"), DELETE, "drawers", "other"),
    ("2 required field missing in a section", ("why", "no_drawer"), DELETE, "why", "no_drawer"),
    ("2 required field missing in a section", ("why", "drawers", 0, "fair", "calibrated_bullish_share"), DELETE,
     "why.drawers[0].fair", "calibrated_bullish_share"),
    ("3 wrong type", ("drawers", "memberships_per_post"), lambda old: [], "drawers.memberships_per_post", "object"),
    ("6 null not allowed, or an unknown field", ("drawers", "drawers", 0, "n_posts"), lambda old: None,
     "drawers.drawers[0].n_posts", "None"),
    ("2 required field missing in a section", ("emotions", "evaluation"), DELETE, "emotions", "evaluation"),
    ("2 required field missing in a section", ("emotions", "overall", "quotes"), DELETE, "emotions.overall", "quotes"),
    ("2 required field missing in a section", ("emotions", "human_review", "by_prelabel", "panic"), DELETE,
     "emotions.human_review.by_prelabel", "panic"),
    ("3 wrong type", ("emotions", "classes"), lambda old: {}, "emotions.classes", "array"),
    ("3 wrong type", ("emotions", "n_posts"), lambda old: 1.5, "emotions.n_posts", "integer"),
    ("3 wrong type", ("emotions", "keep_none"), lambda old: "true", "emotions.keep_none", "boolean"),
    ("5 NaN or infinity", ("emotions", "evaluation", "rows", 0, "macro_f1"), lambda old: NAN,
     "emotions.evaluation.rows[0].macro_f1", "nan"),
    ("6 null not allowed, or an unknown field", ("emotions", "n_posts"), lambda old: None, "emotions.n_posts", "None"),
    ("6 null not allowed, or an unknown field", ("emotions", "labels_file", "n_matched"), lambda old: None,
     "emotions.labels_file.n_matched", "None"),
    ("6 null not allowed, or an unknown field", ("emotions", "extra_field"), lambda old: 1, "emotions", "extra_field"),
    ("2 required field missing in a section", ("dated_data", "date_from"), DELETE, "dated_data", "date_from"),
    ("7 value the format does not allow", ("dated_data", "layers", "retail", "date_from"), lambda old: "2022/05/01",
     "dated_data.layers.retail.date_from", "2022/05/01"),
    ("7 value the format does not allow", ("dated_data", "layers"),
     lambda old: {("whale" if key == "retail" else key): value for key, value in old.items()}, "dated_data.layers", "whale"),
    ("3 wrong type", ("dated_data", "layers", "retail", "days", 0, "n_posts"), lambda old: 1.5,
     "dated_data.layers.retail.days[0].n_posts", "integer"),
    ("3 wrong type", ("dated_data", "layers", "retail", "days", 0, "n_posts"), lambda old: "12",
     "dated_data.layers.retail.days[0].n_posts", "integer"),
    ("6 null not allowed, or an unknown field", ("dated_data", "extra_field"), lambda old: 1, "dated_data", "extra_field"),
    ("2 required field missing in a section", ("daily", "layers", "retail", "days", 0, "date"), DELETE,
     "daily.layers.retail.days[0]", "date"),
    ("2 required field missing in a section", ("new_words", "layers", "retail", "days", 0, "date"), DELETE,
     "new_words.layers.retail.days[0]", "date"),
    ("2 required field missing in a section", ("daily", "layers", "retail", "date_from"), DELETE, "daily.layers.retail",
     "date_from"),
    ("2 required field missing in a section", ("new_words", "date_from"), DELETE, "new_words", "date_from"),
    ("5 NaN or infinity", ("new_words", "layers", "retail", "days", PROBE_DAY), lambda old: with_new_word(old, ratio=INFINITY),
     f"new_words.layers.retail.days[{PROBE_DAY}].new_words[0].ratio", "inf"),
    ("7 value the format does not allow", ("new_words", "layers", "retail", "days", PROBE_DAY),
     lambda old: with_new_word(old, quotes=[{"post_id": f"probe:{number}", "text": f"probe {number}"} for number in range(4)]),
     f"new_words.layers.retail.days[{PROBE_DAY}].new_words[0].quotes", "too long"),
    ("3 wrong type", ("daily", "layers", "retail", "overall", "emotion_counts"), lambda old: {**old, "none": "12"},
     "daily.layers.retail.overall.emotion_counts.none", "integer"),
    ("7 value the format does not allow", ("daily", "layers"),
     lambda old: {("whale" if key == "retail" else key): value for key, value in old.items()}, "daily.layers", "whale"),
    ("7 value the format does not allow", ("new_words", "layers"),
     lambda old: {("whale" if key == "retail" else key): value for key, value in old.items()}, "new_words.layers", "whale"),
    ("7 value the format does not allow", ("daily", "layers", "retail", "days", 0, "date"), day_before,
     "daily.layers.retail.days[0].date", "outside"),
    ("7 value the format does not allow", ("new_words", "layers", "retail", "days", 0, "date"), day_before,
     "new_words.layers.retail.days[0].date", "outside"),
    ("7 value the format does not allow", ("validation", "layers", "retail", "fear_greed", "spearman"), lambda old: 1.5,
     "validation.layers.retail.fear_greed.spearman", "1.5"),
    ("7 value the format does not allow", ("validation", "price", "source"), lambda old: "binance",
     "validation.price.source", "binance"),
    ("3 wrong type", ("layers", "days", 0, "opposite"), lambda old: "yes", "layers.days[0].opposite", "boolean"),
    ("7 value the format does not allow", ("validation", "layers"),
     lambda old: {("whale" if key == "retail" else key): value for key, value in old.items()}, "validation.layers", "whale"),
    ("7 value the format does not allow", ("layers", "n_days_with_share"),
     lambda old: {("whale" if key == "retail" else key): value for key, value in old.items()}, "layers.n_days_with_share",
     "whale"),
    ("7 value the format does not allow", ("layers", "n_days_opposite"), lambda old: -1, "layers.n_days_opposite", "minimum"),
    ("7 value the format does not allow", ("validation", "layers", "retail", "next_day_change", "n_days"), lambda old: -3,
     "validation.layers.retail.next_day_change.n_days", "minimum"),
    ("2 required field missing in a section", ("layers", "date_from"), DELETE, "layers", "date_from"),
]
# Locked: fields the format allows to be null. Set to null one at a time, the brief must still pass.
NULL_ALLOWED = [
    ("dataset", "class_shares", "0"),
    ("dataset", "length_words", "p90"),
    ("sentiment", "evaluation", "rows", 0, "macro_f1"),
    ("why", "overall", "fair", "model_minus_label_pp"),
    ("why", "topics", 0, "label_vs_overall_pp"),
    ("why", "overall", "fair", "calibrated_minus_label_pp"),
    ("drawers", "share_in_any_drawer"),
    ("why", "drawers", 0, "label_vs_overall_pp"),
    ("emotions", "evaluation", "rows", 0, "macro_f1"),
    ("emotions", "human_review", "agreement"),
    ("emotions", "selection", "validation_macro_f1", "count_network"),
    ("dated_data", "switches", "date_end"),
    ("dated_data", "switches", "language"),
    ("daily", "layers", "retail", "overall", "calibrated_bullish_share"),
    ("daily", "layers", "retail", "days", 0, "half_bullish_share"),
    ("daily", "calibration", "rate_gap"),
    ("layers", "date_from"),
    ("layers", "days", 0, "opposite"),
    ("layers", "media_recognition", "true_bullish_rate"),
    ("validation", "layers", "retail", "fear_greed", "spearman"),
    ("validation", "price", "fetched_on"),
]


def path_text(path):
    return "".join(f"[{part}]" if isinstance(part, int) else (f".{part}" if index else str(part))
                   for index, part in enumerate(path))


def check_counterexamples(report, label, brief, rules):
    """Every locked counterexample is rejected for the right reason; null is accepted where the format allows it."""
    if rules is None:
        report.check(False, f"{label}: counterexamples", "the rules could not be loaded")
        return
    kinds = {}
    for kind, path, change, where, word in BROKEN_BRIEFS:
        try:
            broken = changed_copy(brief, path, change)
        except (KeyError, IndexError, TypeError) as exc:
            kinds.setdefault(kind, []).append((path, False, f"could not be applied: {type(exc).__name__} {exc}"))
            continue
        problems = rules["brief_problems"](broken)
        reason = [problem for problem in problems if problem.startswith(f"{where}: ") and word in problem]
        kinds.setdefault(kind, []).append((path, bool(reason), (reason or problems or ["no problem found"])[0]))
    for kind, cases in kinds.items():
        rejected = sum(ok for _, ok, _ in cases)
        report.check(rejected == len(cases), f"{label}: the format check rejects every counterexample of kind "
                     f"'{kind}' ({rejected} of {len(cases)})",
                     "; ".join(f"{path_text(path)}: {message}" for path, ok, message in cases if not ok))
        for path, ok, message in cases:
            report.info(f"{'rejected' if ok else 'NOT rejected'}: {path_text(path)} -> {message}")
    accepted, refused = 0, []
    for path in NULL_ALLOWED:
        try:
            problems = rules["brief_problems"](changed_copy(brief, path, lambda old: None))
        except (KeyError, IndexError, TypeError) as exc:
            problems = [f"could not be applied: {exc}"]
        accepted += not problems
        refused += [f"{path_text(path)}: {problems[0]}"] if problems else []
    report.check(not refused, f"{label}: the format check accepts null in fields that may be null "
                 f"({accepted} of {len(NULL_ALLOWED)})", "; ".join(refused))


def check_save_refuses(report, label, brief, rules):
    """save_brief writes nothing for a brief that breaks the rules, and never writes NaN."""
    if rules is None:
        report.check(False, f"{label}: save_brief", "the rules could not be loaded")
        return
    folder = REPO_DIR / "data" / "checks" / "save-probe"
    shutil.rmtree(folder, ignore_errors=True)
    target = folder / "brief.json"

    def attempt(document):
        target.unlink(missing_ok=True)
        try:
            rules["save_brief"](document, target)
            return None
        except ValueError as exc:
            lines = str(exc).splitlines()
            return lines[1].strip().removeprefix("- ") if len(lines) > 1 else str(exc)

    results = []
    for description, path, change in [("a missing section", ("why",), DELETE),
                                      ("NaN in a field that may not be null", ("dataset", "n_posts"), lambda old: NAN)]:
        message = attempt(changed_copy(brief, path, change))
        results.append(message is not None and not target.exists())
        report.info(f"save_brief with {description}: {'refused, nothing written' if results[-1] else 'NOT refused'}"
                    f" ({message})")
    message = attempt(changed_copy(brief, ("why", "overall", "fair", "model_minus_label_pp"), lambda old: NAN))
    try:
        saved = json.loads(target.read_text(encoding="utf-8"), parse_constant=reject_constant)
        nan_as_null = message is None and saved["why"]["overall"]["fair"]["model_minus_label_pp"] is None
    except (OSError, ValueError, KeyError, TypeError):
        nan_as_null = False
    message = attempt(brief)
    try:
        round_trip = message is None and json.loads(target.read_text(encoding="utf-8"), parse_constant=reject_constant) == brief
    except (OSError, ValueError):
        round_trip = False
    report.check(all(results) and nan_as_null and round_trip,
                 f"{label}: save_brief writes nothing when the check fails, saves a NaN in a field that may be null "
                 "as null, and writes a valid brief unchanged", f"refused {results}, NaN saved as null {nan_as_null}, "
                 f"round trip {round_trip}")
    shutil.rmtree(folder, ignore_errors=True)


# Run in a separate Python with an audit hook that records every network connection and name lookup to a
# host other than this machine. It executes the notebook's panel cell, then builds the panel from a brief file
# and from a copy with marked values, and prints the tables and the page's table components as JSON.
# It also reports every image on the page (label, size, colours, a hash of its pixels) for both, and builds the
# panel from two more copies next to the marked one, each with one number of one chart changed.
PANEL_PROBE = r'''
import json, os, sys
LOOPBACK = {"127.0.0.1", "::1", "localhost", b"127.0.0.1", b"::1", b"localhost"}
network = []

def audit(event, args):
    if event == "socket.connect" and isinstance(args[1], tuple) and args[1][0] not in LOOPBACK:
        network.append(f"connect {args[1]!r}")
    elif event == "socket.getaddrinfo" and args[0] is not None and args[0] not in LOOPBACK:
        network.append(f"getaddrinfo {args[0]!r}")
    elif event in ("socket.sendto", "socket.sendmsg"):
        network.append(f"{event}")

sys.addaudithook(audit)
cell_file, brief_file, marked_file = sys.argv[1:4]
namespace = {"__name__": "panel_cell"}
exec(compile(open(cell_file, encoding="utf-8").read(), "panel cell", "exec"), namespace)


def chart_images(page):  # every image on the page: label, file size, pixel size, colours and a hash of the pixels
    import hashlib
    from PIL import Image
    found = []
    for block in page.blocks.values():
        if type(block).__name__ != "Image":
            continue
        value = block.value if isinstance(block.value, dict) else {}
        item = {"label": block.label, "sha256": None, "bytes": 0, "size": None, "colours": 0}
        if value.get("path") and os.path.isfile(value["path"]):
            with Image.open(value["path"]) as picture:
                pixels = picture.convert("RGB")
            colours = pixels.getcolors(maxcolors=4096)
            item.update(sha256=hashlib.sha256(repr(pixels.size).encode() + pixels.tobytes()).hexdigest(),
                        bytes=os.path.getsize(value["path"]), size=list(pixels.size),
                        colours=4097 if colours is None else len(colours))
        found.append(item)
    return found


result = {"environment": os.environ.get("GRADIO_ANALYTICS_ENABLED")}
for key, path in (("brief", brief_file), ("marked", marked_file)):
    tables = namespace["panel_tables"](path)
    page = namespace["build_panel"](path)
    components = [block.value for block in page.blocks.values() if type(block).__name__ == "Dataframe"]
    result[key] = {
        "page": type(page).__name__,
        "analytics_enabled": page.analytics_enabled,
        "tables": {name: {"columns": [str(column) for column in table.columns],
                          "rows": [[str(cell) for cell in row] for row in table.values.tolist()]}
                   for name, table in tables.items()},
        "components": [{"columns": [str(column) for column in value.get("headers", [])],
                        "rows": [[str(cell) for cell in row] for row in value.get("data", [])]}
                       for value in components],
    }
    result[key]["images"] = chart_images(page)
for key in ("daily_changed", "layers_changed"):  # copies with one number of a chart changed, next to the marked copy
    path = os.path.join(os.path.dirname(marked_file), key + "-brief.json")
    if os.path.isfile(path):
        result[key] = {"images": chart_images(namespace["build_panel"](path))}
result["network"] = network
print("PANEL-PROBE " + json.dumps(result))
'''
PANEL_MARKS = [  # (path in the brief, marker, table that must show it)
    (("scope", "source"), "PANEL-MARK-SOURCE", "scope"),
    (("sentiment", "evaluation", "rows", 0, "model"), "PANEL-MARK-MODEL", "evaluation"),
    (("topics", "topics", 0, "name"), "PANEL-MARK-TOPIC", "topics"),
    (("why", "topics", 0, "name"), "PANEL-MARK-WHY", "why"),
    (("why", "calibration", "rates_from", "subset"), "PANEL-MARK-RATES", "calibration"),
    (("why", "topics", 0, "topic"), "PANEL-MARK-CALIBRATED", "calibrated"),
    (("drawers", "drawers", 0, "name"), "PANEL-MARK-DRAWER", "drawers"),
    (("emotions", "drawers", 0, "name"), "PANEL-MARK-EMOTION", "emotions"),
    (("validation", "layers", "retail", "date_from"), "PANEL-MARK-LAYER", "validation"),
    (("validation", "price", "source"), "PANEL-MARK-PRICE", "sources"),
]
PANEL_NEW_WORD_PATH = ("new_words", "layers", "retail", "days", 0, "new_words")  # a day that every brief has
PANEL_NEW_WORD = {"word": "PANEL-MARK-NEW-WORD", "n_posts": 1, "past_daily_mean": 0.0, "ratio": None,
                  "quotes": [{"post_id": "panel-probe", "text": "PANEL-MARK-NEW-WORD"}]}  # added to it in the copy
PANEL_CHART_CHANGES = [  # (copy, the number changed in it, the chart that must change: position in PANEL_CHART_LABELS)
    ("daily_changed", ("daily", "layers", "retail", "days", 0, "calibrated_bullish_share"), 0),
    ("layers_changed", ("layers", "days", 0, "calibrated_bullish_share", "media"), 1),
]


def panel_chart_value(old):
    """A share far from the old one (or from nothing): what a copy of the brief gets for one number of a chart."""
    return 0.95 if old is None or old < 0.5 else 0.05


def check_panel(report, label, brief_path, brief):
    """Build the panel from the brief file in a separate Python; tables from the file; no network connection."""
    folder = REPO_DIR / "data" / "checks" / "panel-probe"
    folder.mkdir(parents=True, exist_ok=True)
    cell_file, marked_file = folder / "panel_cell.py", folder / "marked-brief.json"
    cell_file.write_text(cell_defining("build_panel") or "", encoding="utf-8")
    marked = brief
    try:
        for path, marker, _ in PANEL_MARKS:
            marked = changed_copy(marked, path, lambda old, marker=marker: marker)
    except (KeyError, IndexError, TypeError) as exc:
        report.check(False, f"{label}: panel probe", f"cannot mark the brief: {exc}")
        return
    try:  # a made-up new word on the first day of the retail layer: a brief may have no new word at all
        marked = changed_copy(marked, PANEL_NEW_WORD_PATH, lambda old: [PANEL_NEW_WORD] + list(old))
    except (KeyError, IndexError, TypeError) as exc:
        report.check(False, f"{label}: panel probe", f"cannot add a new word to the brief: {exc}")
    for key, path, _ in PANEL_CHART_CHANGES:  # read by the probe from the marked copy's folder
        chart_file = folder / f"{key}-brief.json"
        chart_file.unlink(missing_ok=True)
        try:
            chart_file.write_text(json.dumps(changed_copy(brief, path, panel_chart_value), ensure_ascii=False),
                                  encoding="utf-8")
        except (KeyError, IndexError, TypeError) as exc:
            report.check(False, f"{label}: panel probe", f"cannot change {path} in a copy of the brief: {exc}")
    marked_file.write_text(json.dumps(marked, ensure_ascii=False), encoding="utf-8")
    started = time.perf_counter()
    completed = subprocess.run([sys.executable, "-c", PANEL_PROBE, str(cell_file), str(brief_path), str(marked_file)],
                               capture_output=True, text=True, cwd=REPO_DIR, timeout=900)
    seconds = time.perf_counter() - started
    lines = [line for line in completed.stdout.splitlines() if line.startswith("PANEL-PROBE ")]
    built_ok = completed.returncode == 0 and len(lines) == 1
    if not report.check(built_ok, f"{label}: the panel builds from the brief file without being started ({seconds:.1f} s)",
                        "" if built_ok else (completed.stderr.strip().splitlines() or ["no output"])[-1][:300]):
        return
    probe = json.loads(lines[0][len("PANEL-PROBE "):])
    built = probe["brief"]
    tables = built["tables"]
    sentiment, topics, why = as_dict(brief.get("sentiment")), as_dict(brief.get("topics")), as_dict(brief.get("why"))
    expected_rows = {"evaluation": len(as_dict(sentiment.get("evaluation")).get("rows") or []),
                     "topics": len(topics.get("topics") or []), "why": 1 + len(why.get("topics") or [])}
    rows = {name: len(table["rows"]) for name, table in tables.items()}
    scope_values = [row[-1] for row in as_dict(tables.get("scope")).get("rows", [])]
    report.check(list(tables) == PANEL_TABLES + PANEL_MORE_TABLES
                 and all(rows[name] == count for name, count in expected_rows.items())
                 and brief.get("format_version") in scope_values and as_dict(brief.get("scope")).get("source") in scope_values
                 and built["page"] == "Blocks"
                 and built["components"] == [tables[name] for name in PANEL_TABLES + PANEL_MORE_TABLES],
                 f"{label}: the panel shows the scope, evaluation, topic and topic x bullish tables with one row per "
                 "row of the brief, on one tab each",
                 f"tables {rows}, expected {expected_rows}; page {built['page']}; "
                 f"{len(built['components'])} table components matching the tables: "
                 f"{built['components'] == [tables[name] for name in PANEL_TABLES + PANEL_MORE_TABLES]}")
    marks = []
    for _, marker, table in PANEL_MARKS:
        shown_marked = any(marker in cell for row in probe["marked"]["tables"][table]["rows"] for cell in row)
        on_page = any(marker in cell for component in probe["marked"]["components"] for row in component["rows"]
                      for cell in row)
        before = any(marker in cell for table_value in tables.values() for row in table_value["rows"] for cell in row)
        marks.append(shown_marked and on_page and not before)
        report.info(f"panel marker {marker}: in the {table} table {shown_marked}, on the page {on_page}, "
                    f"before marking {before}")
    report.check(all(marks), f"{label}: the panel's tables come from the brief file (values changed in a copy of the "
                 "file show up on the panel built from that copy)", f"{marks}")
    report.check(probe["network"] == [] and probe["environment"] == "False"
                 and built["analytics_enabled"] is False and probe["marked"]["analytics_enabled"] is False,
                 f"{label}: importing Gradio and building the panel made no network connection beyond this machine "
                 "(usage statistics off)",
                 f"connections {probe['network'][:5]}, GRADIO_ANALYTICS_ENABLED {probe['environment']!r}, "
                 f"analytics_enabled {built['analytics_enabled']!r}")
    report.info(f"panel probe output: {len(completed.stdout):,} characters")
    check_panel_more(report, label, brief, probe)


def panel_more_expected(brief):
    """For the panel's tables after the first four, computed from the brief: their rows; per checked column, the
    rows whose value is null in the brief (True) or not (False); and the emotion classes."""
    why, drawers, emotions = (as_dict(brief.get(name)) for name in ("why", "drawers", "emotions"))
    daily, new_words, validation = (as_dict(brief.get(name)) for name in ("daily", "new_words", "validation"))
    shares = [("label share", "label_bullish_share"), ("share at 0.5", "half_bullish_share"),
              ("calibrated share", "calibrated_bullish_share")]
    calibrations = [as_dict(why.get("calibration")), as_dict(daily.get("calibration"))]
    calibrated = [as_dict(why.get("overall"))] + [as_dict(row) for row in why.get("topics") or []]
    by_drawer = {as_dict(row).get("drawer"): as_dict(row) for row in why.get("drawers") or []}
    drawer_rows = [(as_dict(item).get("share"), by_drawer.get(as_dict(item).get("drawer"), {}))
                   for item in drawers.get("drawers") or []]
    drawer_rows.append((drawers.get("share_in_no_drawer"), as_dict(why.get("no_drawer"))))
    groups = [as_dict(emotions.get("overall"))] + [as_dict(item) for item in emotions.get("drawers") or []]
    classes = emotions.get("classes") or []
    word_layers = as_dict(new_words.get("layers"))
    order = list(word_layers)
    words = sorted(((day.get("date"), order.index(layer), position, as_dict(item))
                    for layer, record in word_layers.items() for day in as_dict(record).get("days") or []
                    for position, item in enumerate(day.get("new_words") or [])), key=lambda entry: entry[:3])
    layers = [as_dict(item) for item in as_dict(validation.get("layers")).values()]
    sources = [as_dict(validation.get("price")), as_dict(validation.get("fear_greed"))]
    fair = [("fair label", "label_bullish_share"), ("fair at 0.5", "half_bullish_share"),
            ("fair calibrated", "calibrated_bullish_share"), ("fair at 0.5 − label (pp)", "half_minus_label_pp"),
            ("fair calibrated − label (pp)", "calibrated_minus_label_pp")]
    nulls = {
        "calibration": {column: [item.get(field) is None for item in calibrations]
                        for column, field in [("true bullish rate", "true_bullish_rate"),
                                              ("false bullish rate", "false_bullish_rate"), ("rate gap", "rate_gap")]},
        "calibrated": {**{column: [row.get(field) is None for row in calibrated] for column, field in shares},
                       **{column: [as_dict(row.get("fair")).get(field) is None for row in calibrated]
                          for column, field in fair}},
        "drawers": {"share of topic posts": [share is None for share, _ in drawer_rows],
                    **{column: [row.get(field) is None for _, row in drawer_rows] for column, field in shares}},
        "emotions": {label: [as_dict(group.get("shares")).get(label) is None for group in groups] for label in classes},
        "new_words": {"ratio": [entry[3].get("ratio") is None for entry in words]},
        "validation": {"Fear and Greed Index (alternative.me): Spearman":
                       [as_dict(item.get("fear_greed")).get("spearman") is None for item in layers],
                       "next day's BTC change: Spearman":
                       [as_dict(item.get("next_day_change")).get("spearman") is None for item in layers]},
        "sources": {column: [item.get(field) is None for item in sources]
                    for column, field in [("address", "address"), ("first day", "date_from"), ("last day", "date_to"),
                                          ("fetched on", "fetched_on"), ("why not obtained", "reason")]},
    }
    rows = {"calibration": len(calibrations), "calibrated": len(calibrated), "drawers": len(drawer_rows),
            "emotions": len(groups), "new_words": len(words), "validation": len(layers), "sources": len(sources)}
    return rows, nulls, classes


def check_panel_more(report, label, brief, probe):
    """The panel after its first four tables: the new tables with their rows from the brief and "—" for null, the
    two charts, drawn from the file's numbers only, and a new word added in a copy of the file."""
    built = probe["brief"]
    tables = built["tables"]
    expected_rows, nulls, classes = panel_more_expected(brief)
    rows = {name: len(as_dict(tables.get(name)).get("rows") or []) for name in PANEL_MORE_TABLES}
    emotion_columns = as_dict(tables.get("emotions")).get("columns") or []
    after_four = list(tables)[len(PANEL_TABLES):]
    report.check(after_four == PANEL_MORE_TABLES and rows == expected_rows
                 and emotion_columns == PANEL_EMOTION_COLUMNS + classes,
                 f"{label}: after the first four, the panel shows the tables {', '.join(PANEL_MORE_TABLES)} with one "
                 "row per row of the brief (the calibrations of chapters 10 and 13; all topic posts and every topic; "
                 "every drawer and the posts in no drawer; all topic posts and every drawer, with one column per "
                 "emotion class learned; every new word; every layer; the prices and the index)",
                 f"tables {after_four}; rows {rows}, expected {expected_rows}; emotion columns "
                 f"{emotion_columns[len(PANEL_EMOTION_COLUMNS):]}, classes {classes}")

    wrong, checked, null_cells = [], 0, 0
    for name, columns in nulls.items():
        table = as_dict(tables.get(name))
        header = table.get("columns") or []
        for column, expected in columns.items():
            if column not in header:
                wrong.append(f"{name}: no column {column!r}")
                continue
            shown = [row[header.index(column)] == PANEL_DASH for row in table.get("rows") or []]
            checked, null_cells = checked + len(expected), null_cells + expected.count(True)
            if shown != expected:
                wrong.append(f"{name} / {column}: {shown.count(True)} dashes where the brief has "
                             f"{expected.count(True)} nulls (rows differ: {shown != expected})")
    report.check(not wrong, f"{label}: in the new tables a value that is null in the brief shows as \"{PANEL_DASH}\" "
                 f"and no other value does ({checked:,} cells checked, {null_cells:,} of them null)",
                 "; ".join(wrong[:5]))

    images = built.get("images") or []
    for image in images:
        report.info(f"panel chart {image.get('label')!r}: {image.get('size')} pixels, {image.get('bytes'):,} bytes, "
                    f"{image.get('colours')} colours (4097: more than 4096), pixels sha256 {image.get('sha256')}")
    report.check([image.get("label") for image in images] == PANEL_CHART_LABELS
                 and all(image.get("sha256") and image.get("bytes", 0) > 0 and image.get("colours", 0) > 1
                         for image in images),
                 f"{label}: the panel shows its two charts ({'; '.join(PANEL_CHART_LABELS)}) as pictures that are not "
                 "empty (a file with more than one colour)",
                 f"{[(image.get('label'), image.get('size'), image.get('bytes'), image.get('colours')) for image in images]}")

    def hashes(key):
        return [image.get("sha256") for image in as_dict(probe.get(key)).get("images") or []]

    original = hashes("brief")
    outcome = {"marked": len(original) == len(PANEL_CHART_LABELS) and hashes("marked") == original}
    for key, _, changed_chart in PANEL_CHART_CHANGES:
        changed = hashes(key)
        outcome[key] = (len(changed) == len(original) == len(PANEL_CHART_LABELS)
                        and all((before != after) == (position == changed_chart)
                                for position, (before, after) in enumerate(zip(original, changed))))
    report.check(all(outcome.values()),
                 f"{label}: the charts are drawn from the brief file: a daily share changed in a copy changes only the "
                 "chart by day, a share of the two layers changed in a copy changes only the chart of the two layers, "
                 "and the copy with the marked texts changes neither", f"{outcome}")

    marker = PANEL_NEW_WORD["word"]
    marked_rows = as_dict(probe["marked"]["tables"].get("new_words")).get("rows") or []
    shown = any(marker in cell for row in marked_rows for cell in row)
    on_page = any(marker in cell for component in probe["marked"]["components"] for row in component["rows"]
                  for cell in row)
    before = any(marker in cell for table in tables.values() for row in table["rows"] for cell in row)
    report.info(f"panel new word {marker}: in the new_words table {shown}, on the page {on_page}, before {before}")
    report.check(shown and on_page and not before,
                 f"{label}: a new word added in a copy of the file shows up in the new word table of the panel built "
                 "from that copy", f"shown {shown}, on the page {on_page}, before {before}")


def check_panel_charts_source(report):
    """The panel cell draws its charts with Matplotlib in memory and writes no file."""
    panel_cell = cell_defining("build_panel") or ""
    tree = ast.parse(panel_cell)
    imports = ({alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
               | {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)})
    calls = {node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
             for node in ast.walk(tree) if isinstance(node, ast.Call)}
    functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    writing = sorted(calls & {"savefig", "imsave", "write_text", "write_bytes", "to_csv", "to_json", "dump", "open",
                              "mkdir", "show", "print_png"})
    pyplot = sorted(name for name in imports if name and "pyplot" in name)
    report.check(panel_cell != "" and {"matplotlib.figure", "matplotlib.backends.backend_agg"} <= imports
                 and not pyplot and not writing and {"panel_charts", "Image"} <= calls | functions,
                 "the panel cell draws its two charts with Matplotlib in memory (a Figure on the Agg canvas, no pyplot) "
                 "and writes no file (no savefig, imsave, write_text, write_bytes, to_csv, to_json, dump, open, mkdir, "
                 "show or print_png); the pictures go to Gradio as arrays of pixels",
                 f"imports {sorted(name for name in imports if name and name.startswith('matplotlib'))}; pyplot {pyplot}; "
                 f"writing calls {writing}")


def check_panel_output(report, label, executed):
    """The executed notebook saved the brief through the format check and built the panel without starting it."""
    lines = []
    if executed.exists():
        for cell in json.loads(executed.read_text(encoding="utf-8"))["cells"]:
            for out in cell.get("outputs", []):
                text = out.get("text", "")
                lines += ("".join(text) if isinstance(text, list) else text).splitlines()
    saved = [line for line in lines if line.startswith("[brief] saved") and "passed the format check" in line]
    built = [line for line in lines if line.startswith("[panel] built from")]
    idle = [line for line in lines if line.startswith(f"[panel] not started ({PANEL_SWITCH} = False)")]
    report.check(len(saved) == 1 and len(built) == 1 and len(idle) == 1,
                 f"{label}: the notebook saved the brief after the format check and built the panel without starting it",
                 " | ".join(saved + built + idle) or "no [brief] or [panel] lines")
    for line in saved + built + idle:
        report.info(line)


def null_fields(value, path=""):
    """Generalised paths ([] for list items) of every null in a brief."""
    if value is None:
        return {path}
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            found |= null_fields(item, f"{path}.{key}" if path else key)
    elif isinstance(value, list):
        for item in value:
            found |= null_fields(item, f"{path}[]")
    return found


def panel_settings(report, label, parameters, overrides):
    """The checks never start the panel, whatever -p says."""
    if PANEL_SWITCH in overrides and overrides[PANEL_SWITCH] != CHECK_SETTINGS[PANEL_SWITCH]:
        report.info(f"{label}: keeps {PANEL_SWITCH}={CHECK_SETTINGS[PANEL_SWITCH]!r} instead of the -p value")
    return {**parameters, **CHECK_SETTINGS}


# ---------------------------------------------------------------- emotion pre-labels (scripts/prelabel_emotions.py)
# The checks never contact Ollama: the labelling loop is driven by a fake model that answers from a script.

EMOTION_LABELS = ["panic", "greed", "excitement", "anger", "mockery", "despair", "wait_and_see", "confidence",
                  "none"]  # the nine labels, in order (interface of the label files)
PRELABEL_UNPARSED = "unparsed"  # the tenth value of the emotion column: no attempt gave a label
PRELABEL_COLUMNS = ["post_id", "emotion", "model", "prompt_version", "text_hash"]  # label files, in order
REVIEW_TEMPLATE_COLUMNS = ["post_id", "prelabel", "human_label"]  # review table, in order
PRELABEL_DEFAULT_MODEL = "qwen2.5:7b"
PRELABEL_SAMPLE_FILE = "samples/emotions-prelabel-sample.csv"
PRELABEL_FULL_FILE = "labels/emotions-prelabel-v1.csv"
REVIEW_TEMPLATE_FILE = "labels/emotions-review-template.csv"
REVIEW_TEMPLATE_SIZE = 200
REVIEW_TEMPLATE_SEED = 42
# SHA-256 of everything fixed that is sent to the model, per released prompt version (see prompt_fingerprint_of):
# the system message with the label meanings and the rules, the hand-written examples, the retry reminder and the
# answer format. A changed character needs a new prompt version, new label files and a new line here.
PROMPT_FINGERPRINTS = {
    "v1": "f5b471650c79b801daa219376d245f545819db75cd574a961454666e3faa0bf8",
}
REFUSED_HOSTS = ["http://8.8.8.8:11434", "http://example.com:11434", "https://127.0.0.1:11434",
                 "http://127.0.0.1.example.com:11434", "http://127.0.0.1@example.com:11434", "http://0.0.0.0:11434",
                 "http://192.168.1.10:11434", "ftp://localhost:11434", "127.0.0.1:11434",
                 "http://localhost:11434/api"]
ACCEPTED_HOSTS = ["http://127.0.0.1:11434", "http://localhost:11434", "http://[::1]:11434", "http://127.0.0.1"]
PRELABEL_PROBE_DIR = REPO_DIR / "data" / "checks" / "prelabel-probe"
FAKE_GOOD = '{"emotion": "panic"}'


def own_text_hash(text):
    """The documented text fingerprint, computed here: first 8 hex digits of the SHA-256 of the UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def prompt_fingerprint_of(label_set):
    """The documented prompt fingerprint, computed here from what the script sends: SHA-256 of the compact ASCII JSON
    (sorted keys) of {"first": the messages of a first request, "retry": the messages of a retry, "format": the answer
    format}, with the script's placeholder in place of the post."""
    first = prelabel.build_payload(prelabel.POST_PLACEHOLDER, retry=False, label_set=label_set, model="", seed=0)
    again = prelabel.build_payload(prelabel.POST_PLACEHOLDER, retry=True, label_set=label_set, model="", seed=0)
    text = json.dumps({"first": first["messages"], "retry": again["messages"], "format": first["format"]},
                      ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def raises_stop(function, *args):
    try:
        function(*args)
    except prelabel.Stop:
        return True
    return False


def check_prelabel_script(report):
    """The label script's interface, its pinned prompt, its requests, and that it talks only to a loopback address."""
    import urllib.parse
    import urllib.request

    built_in = prelabel.BUILT_IN
    version = prelabel.PROMPT_VERSION
    interface = {
        "labels": (prelabel.label_names(built_in), EMOTION_LABELS),
        "columns": (prelabel.COLUMNS, PRELABEL_COLUMNS),
        "review columns": (prelabel.REVIEW_COLUMNS, REVIEW_TEMPLATE_COLUMNS),
        "unparsed": (prelabel.UNPARSED, PRELABEL_UNPARSED),
        "default model": (prelabel.DEFAULT_MODEL, PRELABEL_DEFAULT_MODEL),
        "full label file": (prelabel.default_output("full", version), PRELABEL_FULL_FILE),
        "sample label file": (prelabel.default_output("sample", version), PRELABEL_SAMPLE_FILE),
        "review table": (prelabel.default_output("review", version), REVIEW_TEMPLATE_FILE),
        "review size": (prelabel.DEFAULT_REVIEW_SIZE, REVIEW_TEMPLATE_SIZE),
        "seed": (prelabel.DEFAULT_SEED, REVIEW_TEMPLATE_SEED),
    }
    wrong = [f"{name}: {got!r}, expected {expected!r}" for name, (got, expected) in interface.items() if got != expected]
    report.check(not wrong, "emotion pre-labels: the script's nine labels (in order), columns, default model, file "
                 "paths, review size and seed are the documented interface", "; ".join(wrong))

    computed = prompt_fingerprint_of(built_in)
    report.check(version in PROMPT_FINGERPRINTS and computed == PROMPT_FINGERPRINTS[version]
                 and prelabel.prompt_fingerprint() == computed,
                 f"emotion pre-labels: prompt {version} (system message with the label meanings and rules, "
                 "hand-written examples, retry reminder, answer format) has the pinned SHA-256; a changed character "
                 "needs a new version",
                 f"computed {computed}, script {prelabel.prompt_fingerprint()}, pinned {PROMPT_FINGERPRINTS.get(version)!r}")

    first = prelabel.build_payload("a post", retry=False, label_set=built_in, model=PRELABEL_DEFAULT_MODEL, seed=42)
    again = prelabel.build_payload("a post", retry=True, label_set=built_in, model=PRELABEL_DEFAULT_MODEL, seed=42)
    reminder = prelabel.retry_reminder(built_in["labels"])
    answer_rules = {"type": "object", "properties": {"emotion": {"type": "string", "enum": EMOTION_LABELS}},
                    "required": ["emotion"]}
    problems = []
    for name, payload in (("first", first), ("retry", again)):
        options = payload.get("options", {})
        if sorted(payload) != ["format", "messages", "model", "options", "stream"]:
            problems.append(f"{name}: fields {sorted(payload)}")
        if payload.get("stream") is not False or options.get("temperature") != 0 or options.get("seed") != 42:
            problems.append(f"{name}: stream {payload.get('stream')!r}, options {options}")
        if not is_count(options.get("num_predict")) or options["num_predict"] < 16:
            problems.append(f"{name}: num_predict {options.get('num_predict')!r}")
        if payload.get("format") != answer_rules:
            problems.append(f"{name}: format {payload.get('format')}")
        if payload["messages"][-1] != {"role": "user", "content": prelabel.USER_PREFIX + "a post"}:
            problems.append(f"{name}: the post is not the last message")
    if reminder in first["messages"][0]["content"] or not again["messages"][0]["content"].endswith("\n\n" + reminder) \
            or again["messages"][0]["content"][:-len(reminder) - 2] != first["messages"][0]["content"] \
            or first["messages"][1:] != again["messages"][1:]:
        problems.append("a retry must differ from the first request only by the reminder at the end of the system message")
    examples = [item for item in built_in["examples"] if item.get("emotion") not in EMOTION_LABELS or not item.get("post")]
    if examples:
        problems.append(f"examples without a post or with an unknown label: {examples}")
    report.check(not problems, "emotion pre-labels: a request is not streamed, has temperature 0, the seed, room for "
                 "the longest answer and an answer format that allows only the nine labels; the post is the last "
                 "message; a retry adds only the fixed reminder", "; ".join(problems))

    folder = PRELABEL_PROBE_DIR / "label-list"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    same_version = folder / "v1.json"
    same_version.write_text(json.dumps({"prompt_version": version, "labels": built_in["labels"],
                                        "examples": built_in["examples"]}), encoding="utf-8")
    other = {"prompt_version": "v2", "labels": built_in["labels"][:2] + [{"name": "none", "meaning": "no emotion"}],
             "examples": []}
    other_file = folder / "v2.json"
    other_file.write_text(json.dumps(other), encoding="utf-8")
    loaded = None if raises_stop(prelabel.load_label_set, str(other_file)) else prelabel.load_label_set(str(other_file))
    paths = [prelabel.default_output(kind, "v2") for kind in ("full", "sample", "review")]
    report.check(raises_stop(prelabel.load_label_set, str(same_version)) and loaded == other
                 and prompt_fingerprint_of(other) != computed
                 and not set(paths) & {PRELABEL_FULL_FILE, PRELABEL_SAMPLE_FILE, REVIEW_TEMPLATE_FILE},
                 "emotion pre-labels: another label list needs another prompt version (a list file claiming "
                 f"{version} is refused), which changes the fingerprint and every default file name", f"v2 paths {paths}")
    shutil.rmtree(folder, ignore_errors=True)

    refused = [host for host in REFUSED_HOSTS if not raises_stop(prelabel.loopback_base, host)]
    accepted = [host for host in ACCEPTED_HOSTS if raises_stop(prelabel.loopback_base, host)]
    default_host = prelabel.build_parser().parse_args(["label"]).host
    default_ok = default_host == prelabel.DEFAULT_HOST and urllib.parse.urlsplit(default_host).hostname == "127.0.0.1"
    report.check(not refused and not accepted and default_ok,
                 "emotion pre-labels: the script refuses every address that is not http:// on a loopback host "
                 f"({len(REFUSED_HOSTS)} cases), accepts 127.0.0.1, localhost and ::1, and defaults to 127.0.0.1",
                 f"not refused {refused}; not accepted {accepted}; default {default_host!r}")
    result = subprocess.run([sys.executable, str(REPO_DIR / "scripts" / "prelabel_emotions.py"), "label",
                             "--host", "http://example.com:11434"], capture_output=True, text=True, timeout=120)
    report.check(result.returncode == 1 and "refusing" in result.stderr and not result.stdout,
                 "emotion pre-labels: the label command with an outside address stops before doing anything",
                 f"exit {result.returncode}, stderr {result.stderr.strip()[:160]!r}, stdout {result.stdout.strip()[:80]!r}")

    names = ["http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY", "no_proxy", "NO_PROXY"]
    saved = {name: os.environ.get(name) for name in names}
    try:
        for name in names[:6]:
            os.environ[name] = "http://127.0.0.1:9"
        for name in names[6:]:
            os.environ.pop(name, None)
        openers = [urllib.request.build_opener(), prelabel.no_proxy_opener(),
                   prelabel.OllamaClient(prelabel.DEFAULT_HOST, 1).opener]
        proxies = [[handler.proxies for handler in opener.handlers if isinstance(handler, urllib.request.ProxyHandler)]
                   for opener in openers]
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    default_uses_proxy = any(proxy.get("http") for proxy in proxies[0])
    report.check(default_uses_proxy and not any(proxy for proxy in proxies[1] + proxies[2]),
                 "emotion pre-labels: requests ignore proxy settings (with proxy variables set, a default opener would "
                 "use the proxy; the script's opener has no proxy at all)",
                 f"default opener {proxies[0]}, script {proxies[1]}, client {proxies[2]}")


def check_prelabel_reading(report):
    """The script reads the sample with the same ids and texts as chapter 1 of the notebook (its read_posts, executed
    from the source), and finds the same Bitcoin posts and empty texts as the notebook's cells."""
    import pandas as pd

    namespace, problems = {"__name__": "notebook_cells", "brief": {}}, []
    for name in ("LABEL_NAMES", "read_posts"):
        source = cell_defining(name)
        if source is None:
            problems.append(f"no cell defines {name}")
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(source, f"cell defining {name}", "exec"), namespace)
        except Exception as exc:  # the cell itself is broken
            problems.append(f"cell defining {name}: {type(exc).__name__}: {exc}")
    if not report.check(not problems, "emotion pre-labels: chapter 1's read_posts loads from the notebook source",
                        "; ".join(problems)):
        return
    sample = REPO_DIR / SAMPLE_SOURCE
    notebook_posts = namespace["read_posts"](str(sample))
    script_posts = prelabel.read_posts(sample)
    notebook_pairs = list(zip(notebook_posts["post_id"], notebook_posts["text"]))
    script_pairs = list(zip(script_posts["post_id"], script_posts["text"]))
    differing = [index for index, (left, right) in enumerate(zip(notebook_pairs, script_pairs)) if left != right]
    report.check(notebook_pairs == script_pairs, f"emotion pre-labels: the script gives the sample the same "
                 f"(post_id, text) pairs as chapter 1, in the same order ({len(script_pairs)} posts)",
                 f"{len(notebook_pairs)} vs {len(script_pairs)} posts; first differing rows {differing[:3]}")

    probes = pd.DataFrame({"post_id": [f"probe:{row}" for row in range(2, 11)],
                           "text": ["", "   ", "btc to 50k", "bitcoin's dominance", "bitcoins", "gbtc", "btc_usd",
                                    "BTC pump", " just a post "]})
    tables = {"sample": notebook_posts[["post_id", "text"]], "probes": probes}
    wrong = []
    for name, table in tables.items():
        cells = {"__name__": "notebook_cells", "df_all": table.copy(), "brief": {"dataset": {}}}
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(cell_defining("BTC_PATTERN"), "cell defining BTC_PATTERN", "exec"), cells)
                exec(compile(cell_defining("sheet_sizes"), "cell defining sheet_sizes", "exec"), cells)
        except Exception as exc:
            wrong.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        script_btc = [prelabel.mentions_btc(text) for text in table["text"]]
        if cells["df_all"]["mentions_btc"].tolist() != script_btc:
            wrong.append(f"{name}: Bitcoin posts differ")
        _, drawn = prelabel.select_posts(table, None, prelabel.DEFAULT_BTC_SHARE, prelabel.DEFAULT_SEED)
        if drawn["empty_texts_left_out"] != cells["brief"]["dataset"]["n_empty_text"]:
            wrong.append(f"{name}: {drawn['empty_texts_left_out']} empty texts left out, "
                         f"n_empty_text {cells['brief']['dataset']['n_empty_text']}")
    report.check(not wrong, "emotion pre-labels: the script finds the same Bitcoin posts as chapter 1's mentions_btc "
                 "and leaves out exactly the texts counted by n_empty_text (on the sample and on probe texts)",
                 "; ".join(wrong))


class FakeModel:
    """Stands in for Ollama: answers every post from a script {post text: [answer, ...]}, the last answer repeating.
    An answer is the reply text, or an exception class to raise (AnswerTimeout, ServerUnavailable). Records every
    request."""

    def __init__(self, script):
        self.script = {text: list(answers) for text, answers in script.items()}
        self.requests = []

    @staticmethod
    def post_of(payload):
        content = payload["messages"][-1]["content"]
        return content[len(prelabel.USER_PREFIX):] if content.startswith(prelabel.USER_PREFIX) else content

    def __call__(self, payload):
        self.requests.append(payload)
        answers = self.script.get(self.post_of(payload)) or ['{"emotion": "none"}']
        answer = answers.pop(0) if len(answers) > 1 else answers[0]
        if isinstance(answer, type) and issubclass(answer, Exception):
            raise answer("fake model")
        return answer

    def asked(self):
        """[(post text, the request carries the retry reminder), ...] in request order."""
        reminder = prelabel.retry_reminder(prelabel.BUILT_IN["labels"])
        return [(self.post_of(payload), reminder in payload["messages"][0]["content"]) for payload in self.requests]


def probe_rows(path):
    return [row for row in csv.reader(path.read_text(encoding="utf-8").splitlines())][1:]


def check_prelabel_fake_model(report):
    """Parsing, retries, stopping and resuming of the label script, driven by a fake model on posts of the sample;
    every file is written under data/checks/prelabel-probe/ and removed afterwards."""
    folder = PRELABEL_PROBE_DIR
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    sample_posts = prelabel.read_posts(REPO_DIR / SAMPLE_SOURCE)
    text_by_id = dict(zip(sample_posts["post_id"], sample_posts["text"]))
    selection, _ = prelabel.select_posts(sample_posts, None, prelabel.DEFAULT_BTC_SHARE, prelabel.DEFAULT_SEED)
    posts = selection[:6]
    texts = [post["text"] for post in posts]
    ids = [post["post_id"] for post in posts]
    if not report.check(len(set(texts)) == 6, "fake model: six sample posts with different texts to label"):
        return
    waits = []
    common = {"text_by_id": text_by_id, "connect_retries": 2, "connect_wait": 0.25, "log": lambda text: None,
              "sleep": waits.append}

    def row(index, emotion):
        return [ids[index], emotion, PRELABEL_DEFAULT_MODEL, prelabel.PROMPT_VERSION, own_text_hash(texts[index])]

    # Good answers, odd case and whitespace, bad answers, a late good answer, and timeouts.
    fake = FakeModel({texts[0]: [FAKE_GOOD], texts[1]: ['{"emotion": "  CONFIDENCE \\n"}'],
                      texts[2]: ["not json", '{"emotion": "joy"}', '{"feeling": "panic"}'],
                      texts[3]: ['{"emotion": 5}', '["mockery"]', '{"emotion": " Mockery"}'],
                      texts[4]: [prelabel.AnswerTimeout], texts[5]: ['{"emotion": "greed"}']})
    first_file = folder / "first.csv"
    summary = prelabel.label_posts(posts, first_file, fake, **common)
    expected = [row(0, "panic"), row(1, "confidence"), row(2, PRELABEL_UNPARSED), row(3, "mockery"),
                row(4, PRELABEL_UNPARSED), row(5, "greed")]
    rows = probe_rows(first_file)
    report.check(rows == expected and summary["stopped"] is None and summary["labelled"] == 6,
                 "fake model: a good answer gives its label, also in another case or with spaces around it; three bad "
                 "answers (not JSON, a label not in the list, no emotion field) give unparsed; two bad answers and a "
                 "good one give the good label; a request that times out three times gives unparsed and the next post "
                 "is labelled; every row has the model, the prompt version and the text fingerprint",
                 f"rows {rows}")
    expected_asked = [(texts[0], False), (texts[1], False), (texts[2], False), (texts[2], True), (texts[2], True),
                      (texts[3], False), (texts[3], True), (texts[3], True), (texts[4], False), (texts[4], True),
                      (texts[4], True), (texts[5], False)]
    settings = [index for index, payload in enumerate(fake.requests)
                if payload["model"] != PRELABEL_DEFAULT_MODEL or payload["stream"] is not False
                or payload["options"].get("temperature") != 0 or payload["options"].get("seed") != prelabel.DEFAULT_SEED
                or payload["format"]["properties"]["emotion"]["enum"] != EMOTION_LABELS]
    report.check(fake.asked() == expected_asked and not settings,
                 "fake model: a post is asked at most 3 times; the first question has no reminder, every retry (after "
                 "a bad answer or a timeout) carries the fixed reminder; every request has the model, temperature 0, "
                 "the seed and the nine-label format",
                 f"asked {[(ids[texts.index(text)], retry) for text, retry in fake.asked()]}; requests with other settings {settings}")

    # The server cannot be reached: stop without writing; a short outage is ridden out.
    fake = FakeModel({texts[0]: [FAKE_GOOD], texts[1]: [prelabel.ServerUnavailable], texts[2]: [FAKE_GOOD]})
    waits.clear()
    stop_file = folder / "stop.csv"
    summary = prelabel.label_posts(posts[:3], stop_file, fake, **common)
    asked = [text for text, _ in fake.asked()]
    report.check(probe_rows(stop_file) == [row(0, "panic")] and bool(summary["stopped"]) and waits == [0.25, 0.25]
                 and asked == [texts[0]] + [texts[1]] * 3,
                 "fake model: when the server cannot be reached the run waits and tries twice more, then stops the "
                 "whole run; that post is not written (not even as unparsed) and no later post is asked",
                 f"rows {probe_rows(stop_file)}, stopped {summary['stopped']!r}, waits {waits}, asked {len(asked)}")
    fake = FakeModel({texts[0]: [FAKE_GOOD], texts[1]: [prelabel.ServerUnavailable, '{"emotion": "anger"}'],
                      texts[2]: [FAKE_GOOD]})
    waits.clear()
    outage_file = folder / "outage.csv"
    summary = prelabel.label_posts(posts[:3], outage_file, fake, **common)
    report.check(probe_rows(outage_file) == [row(0, "panic"), row(1, "anger"), row(2, "panic")]
                 and summary["stopped"] is None and waits == [0.25],
                 "fake model: after a short outage the same post is asked again and the run goes on",
                 f"rows {probe_rows(outage_file)}, waits {waits}")

    # Resuming: --limit, a second run, and a partial last line.
    resume_file = folder / "resume.csv"
    prelabel.label_posts(posts, resume_file, FakeModel({text: [FAKE_GOOD] for text in texts}), limit=2, **common)
    after_limit = [item[0] for item in probe_rows(resume_file)]
    fake = FakeModel({text: [FAKE_GOOD] for text in texts})
    summary = prelabel.label_posts(posts, resume_file, fake, **common)
    resumed = probe_rows(resume_file)
    report.check(after_limit == ids[:2] and resumed == [row(index, "panic") for index in range(6)]
                 and [text for text, _ in fake.asked()] == texts[2:] and summary["already_in_file"] == 2,
                 "resume: --limit 2 labels two posts; the next run skips them (they are not asked again) and appends "
                 "the other four, each id once",
                 f"after --limit {after_limit}; then {[item[0] for item in resumed]}; asked again {len(fake.asked())}")
    raw = resume_file.read_bytes()
    resume_file.write_bytes(raw[:-7])  # a run stopped while writing: the last line loses its end
    fake = FakeModel({text: [FAKE_GOOD] for text in texts})
    summary = prelabel.label_posts(posts, resume_file, fake, **common)
    report.check(probe_rows(resume_file) == resumed and summary["dropped_partial_line_bytes"] == len(raw.split(b"\n")[-2]) - 6
                 and [text for text, _ in fake.asked()] == [texts[5]] and resume_file.read_bytes() == raw,
                 "resume: a partial last line is dropped and that post labelled again; the file ends as it would have",
                 f"dropped {summary['dropped_partial_line_bytes']} bytes, asked {len(fake.asked())}")

    # --retry-unparsed: in place.
    retry_file = folder / "retry.csv"
    shutil.copyfile(first_file, retry_file)
    fake = FakeModel({texts[2]: ['{"emotion": "despair"}'], texts[4]: ['{"emotion": "anger"}']})
    summary = prelabel.label_posts(posts, retry_file, fake, retry_unparsed=True, **common)
    expected = [row(0, "panic"), row(1, "confidence"), row(2, "despair"), row(3, "mockery"), row(4, "anger"),
                row(5, "greed")]
    report.check(probe_rows(retry_file) == expected and [text for text, _ in fake.asked()] == [texts[2], texts[4]]
                 and summary["replaced"] == 2 and not list(folder.glob(".*.tmp")),
                 "--retry-unparsed asks only the unparsed rows and replaces each in place: same row count, same order, "
                 "each id once, no temporary file left", f"rows {probe_rows(retry_file)}")

    # A file from another prompt version, model or data source is never added to.
    wrong = []
    for what, column, value, every in (("prompt version", 3, "v0", True), ("model", 2, "another:1b", True),
                                       ("text fingerprint", 4, "00000000", False)):
        mixed = folder / "mixed.csv"
        rows = probe_rows(first_file)
        for item in rows if every else rows[:1]:
            item[column] = value
        mixed.write_text("".join(prelabel.row_line(item) for item in [PRELABEL_COLUMNS] + rows), encoding="utf-8")
        before = mixed.read_bytes()
        fake = FakeModel({})
        stopped = raises_stop(lambda: prelabel.label_posts(posts, mixed, fake, **common))
        if not (stopped and mixed.read_bytes() == before and not fake.requests):
            wrong.append(what)
    report.check(not wrong, "emotion pre-labels: an existing file from another prompt version, another model or another "
                 "data source stops the run before anything is asked or written", f"not stopped: {wrong}")
    shutil.rmtree(folder, ignore_errors=True)


def read_csv_file(path):
    """(header, rows, problems) of a CSV file; a file that does not end with a newline is a problem."""
    raw = path.read_bytes()
    problems = [] if raw.endswith(b"\n") else ["does not end with a complete line"]
    try:
        lines = list(csv.reader(raw.decode("utf-8").splitlines()))
    except (UnicodeDecodeError, csv.Error) as exc:
        return None, [], problems + [str(exc)]
    if not lines:
        return None, [], problems + ["empty"]
    return lines[0], lines[1:], problems


def prelabel_file_problems(path):
    """(rows with five fields, problems) of a label file: header, one value format per column, one model and one
    prompt version (the script's), each id once."""
    header, rows, problems = read_csv_file(path)
    if header != PRELABEL_COLUMNS:
        problems.append(f"header {header}")
    good = []
    for number, item in enumerate(rows, start=2):
        if len(item) != len(PRELABEL_COLUMNS):
            problems.append(f"line {number}: {len(item)} fields")
            continue
        post_id, emotion, _, _, digest = item
        match = re.fullmatch(r"[A-Za-z0-9_]+:([0-9]+)", post_id)
        if not match or int(match.group(1)) < 2:
            problems.append(f"line {number}: id {post_id!r}")
        if emotion not in EMOTION_LABELS + [PRELABEL_UNPARSED]:
            problems.append(f"line {number}: emotion {emotion!r}")
        if not re.fullmatch(r"[0-9a-f]{8}", digest):
            problems.append(f"line {number}: text_hash {digest!r}")
        good.append(item)
    models, versions = sorted({item[2] for item in good}), sorted({item[3] for item in good})
    if models != [prelabel.DEFAULT_MODEL] or models != [PRELABEL_DEFAULT_MODEL]:
        problems.append(f"models {models}")
    if versions != [prelabel.PROMPT_VERSION]:
        problems.append(f"prompt versions {versions}")
    repeated = sorted(post_id for post_id, times in Counter(item[0] for item in good).items() if times > 1)
    if repeated:
        problems.append(f"ids more than once: {repeated[:3]}")
    if not good:
        problems.append("no rows")
    return good, problems


def per_label(rows, column=1):
    counts = Counter(item[column] for item in rows)
    return {label: counts.get(label, 0) for label in EMOTION_LABELS + [PRELABEL_UNPARSED]}


def review_draw(rows, size, seed):
    """The documented review draw, written here apart from the script: for every label (list order) its ids in file
    order shuffled with random.Random(seed); then the labels in turn, one id each per round, a label with none left
    skipped, until size ids are drawn or none are left. Unparsed rows never take part."""
    queues = []
    for label in EMOTION_LABELS:
        ids = [item[0] for item in rows if item[1] == label]
        random.Random(seed).shuffle(ids)
        queues.append([(post_id, label) for post_id in ids])
    in_turn = [pair for round_ in itertools.zip_longest(*queues) for pair in round_ if pair is not None]
    return in_turn[:size]


def check_prelabel_files(report):
    """The two label files and the review table in the repository (static)."""
    parsed = {}
    for name, relative in (("sample", PRELABEL_SAMPLE_FILE), ("full", PRELABEL_FULL_FILE)):
        path = REPO_DIR / relative
        if not report.check(path.is_file(), f"emotion pre-labels: {relative} exists"):
            continue
        rows, problems = prelabel_file_problems(path)
        report.check(not problems, f"emotion pre-labels: {relative} has exactly the columns "
                     f"{', '.join(PRELABEL_COLUMNS)} and every cell fits its column (id <sheet>:<row>, one of the nine "
                     "labels or unparsed, the script's model and prompt version as the only values, 8 lower-case "
                     "hexadecimal digits), each id once", "; ".join(problems[:5]))
        parsed[name] = rows
        report.info(f"{relative}: {len(rows)} rows; per label {json.dumps(per_label(rows))}")

    if "sample" in parsed:
        sample = prelabel.read_posts(REPO_DIR / SAMPLE_SOURCE)
        with_text = {post_id: text for post_id, text in zip(sample["post_id"], sample["text"]) if text.strip()}
        ids = [item[0] for item in parsed["sample"]]
        changed = [item[0] for item in parsed["sample"] if item[0] in with_text and item[4] != own_text_hash(with_text[item[0]])]
        report.check(set(ids) == set(with_text) and len(ids) == len(with_text) and not changed,
                     f"emotion pre-labels: {PRELABEL_SAMPLE_FILE} labels every post of the sample "
                     f"({len(with_text)}), each once, and every fingerprint is that of the sample post's text",
                     f"{len(ids)} rows, {len(set(ids) - set(with_text))} ids not in the sample, "
                     f"{len(set(with_text) - set(ids))} sample posts missing, fingerprints differing {changed[:3]}")

    template = REPO_DIR / REVIEW_TEMPLATE_FILE
    if not report.check(template.is_file(), f"emotion pre-labels: {REVIEW_TEMPLATE_FILE} exists") or "full" not in parsed:
        return
    header, rows, problems = read_csv_file(template)
    full = {item[0]: item[1] for item in parsed["full"]}
    if header != REVIEW_TEMPLATE_COLUMNS:
        problems.append(f"header {header}")
    problems += [f"line {number}: {len(item)} fields" for number, item in enumerate(rows, start=2) if len(item) != 3]
    rows = [item for item in rows if len(item) == 3]
    filled = [item[0] for item in rows if item[2] != ""]
    repeated = sorted(post_id for post_id, times in Counter(item[0] for item in rows).items() if times > 1)
    unknown = [item[0] for item in rows if item[0] not in full]
    differing = [item[0] for item in rows if item[0] in full and full[item[0]] != item[1]]
    unparsed = [item[0] for item in rows if item[1] == PRELABEL_UNPARSED]
    available = sum(label != PRELABEL_UNPARSED for label in full.values())
    expected = review_draw(parsed["full"], REVIEW_TEMPLATE_SIZE, REVIEW_TEMPLATE_SEED)
    same_draw = [(item[0], item[1]) for item in rows] == expected
    report.check(not problems and not filled and not repeated and not unknown and not differing and not unparsed
                 and same_draw and len(rows) == min(REVIEW_TEMPLATE_SIZE, available),
                 f"emotion pre-labels: {REVIEW_TEMPLATE_FILE} has exactly the columns "
                 f"{', '.join(REVIEW_TEMPLATE_COLUMNS)}, an empty human_label everywhere, each id once and in "
                 f"{PRELABEL_FULL_FILE} with its label there (never unparsed), and is exactly the documented draw "
                 f"(labels in turn, seed {REVIEW_TEMPLATE_SEED}), {REVIEW_TEMPLATE_SIZE} rows or every usable row",
                 f"problems {problems[:3]}; filled {filled[:3]}; repeated {repeated[:3]}; not in the label file "
                 f"{unknown[:3]}; other label {differing[:3]}; unparsed {unparsed[:3]}; same draw {same_draw}; "
                 f"{len(rows)} rows of {available} usable")
    report.info(f"{REVIEW_TEMPLATE_FILE}: {len(rows)} rows; per label {json.dumps(per_label(rows))}")


def check_prelabel_full(report, switches):
    """The full label file against the full dataset (the posts the full run read, from its cache), and the numbers."""
    import pandas as pd

    path = REPO_DIR / PRELABEL_FULL_FILE
    if not report.check(path.is_file(), f"emotion pre-labels: {PRELABEL_FULL_FILE} exists"):
        return
    rows, _ = prelabel_file_problems(path)
    source = switches["DATA_SOURCE"]
    namespace = {"__name__": "notebook_cells"}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(cell_defining("source_fingerprint"), "cell defining source_fingerprint", "exec"), namespace)
    fingerprint = namespace["source_fingerprint"](source if "://" in source else str(REPO_DIR / source))
    cache = REPO_DIR / switches["WORK_DIR"] / "cache" / f"posts-{fingerprint}.parquet"
    if cache.is_file():
        posts, origin = pd.read_parquet(cache, columns=["post_id", "text"]), f"the full run's cache {shown(cache)}"
    elif "://" not in source and (REPO_DIR / source).is_file():
        posts, origin = prelabel.read_posts(REPO_DIR / source), source
    else:
        report.check(False, "emotion pre-labels: the full dataset is available to compare the label file with",
                     f"no cache {shown(cache)} and no local file {source}")
        return
    text_by_id = dict(zip(posts["post_id"], posts["text"]))
    missing = [item[0] for item in rows if item[0] not in text_by_id]
    changed = [item[0] for item in rows if item[0] in text_by_id and own_text_hash(text_by_id[item[0]]) != item[4]]
    report.check(bool(rows) and not missing and not changed,
                 f"emotion pre-labels: every id of {PRELABEL_FULL_FILE} is a post of the full dataset and its "
                 "fingerprint is that of that post's text",
                 f"{len(rows)} rows; {len(missing)} ids missing {missing[:3]}; {len(changed)} other texts {changed[:3]}")
    examples = {item["post"].strip().lower() for item in prelabel.EXAMPLES_V1}
    found = sorted(examples & set(posts["text"].str.strip().str.lower()))
    report.check(not found, "emotion pre-labels: none of the prompt's hand-written examples is a post of the full "
                 "dataset", f"found {found}")
    btc = Counter("btc" if re.search(r"\b(?:bitcoin|btc)\b", text_by_id.get(item[0], ""), re.IGNORECASE) else "other"
                  for item in rows)
    by_label_btc = Counter((item[1], "btc" if re.search(r"\b(?:bitcoin|btc)\b", text_by_id.get(item[0], ""),
                                                        re.IGNORECASE) else "other") for item in rows)
    print(f"\nEmotion pre-labels ({PRELABEL_FULL_FILE}, compared with {origin})")
    print(f"  rows:                          {len(rows)}   model {sorted({item[2] for item in rows})}   prompt "
          f"{sorted({item[3] for item in rows})}")
    print(f"  mention Bitcoin / other:       {btc['btc']} / {btc['other']}")
    print("  label             rows   share   Bitcoin   other")
    for label in EMOTION_LABELS + [PRELABEL_UNPARSED]:
        count = sum(item[1] == label for item in rows)
        print(f"  {label:16s} {count:5d}   {count / len(rows) if rows else 0:5.1%}   {by_label_btc[(label, 'btc')]:7d}"
              f"   {by_label_btc[(label, 'other')]:5d}")


# ---------------------------------------------------------------- emotions (chapter 11)
# Chapter 11 learns the emotions of the pre-label files (silver labels) and gives every topic post an emotion. The
# checks always read the pre-labels from the repository (the sample's with the sample, the full file with the full
# dataset) and never from the default web address.

EMOTION_SWITCHES = ["EMOTION_LABELS_SOURCE", "EMOTION_NAMES", "EMOTION_KEEP_NONE", "EMOTION_MIN_CLASS_POSTS",
                    "EMOTION_REVIEW_SOURCE"]
EMOTION_DEFAULT_SOURCE = ("https://raw.githubusercontent.com/Zhen131/ZHENNs_Ledger_Eyes/main/"
                          "labels/emotions-prelabel-v1.csv")  # the default of EMOTION_LABELS_SOURCE; never read by a check
EMOTION_EIGHT = EMOTION_LABELS[:8]  # the emotions EMOTION_NAMES may list (none is EMOTION_KEEP_NONE's)
EMOTION_DROP_STEPS = ["unparsed", "none", "not_in_emotion_names", "too_few_posts"]  # in this order
EMOTION_QUOTES_OVERALL, EMOTION_QUOTES_PER_GROUP = 3, 1  # example posts per class: all topic posts; other groups
EMOTION_PROBE_DIR = REPO_DIR / "data" / "checks" / "emotion-probe"
SILVER_WORDS = ("silver labels", "\u94f6\u6807\u7b7e")  # English and Chinese, in the chapter's explanation cells
# Locked (only add): switch values the switch check cell must accept (True) or stop at (False).
EMOTION_SWITCH_PROBES = [
    ({}, True),
    ({"EMOTION_NAMES": ["greed"]}, True),
    ({"EMOTION_NAMES": ["confidence", "panic"]}, True),
    ({"EMOTION_MIN_CLASS_POSTS": 25}, True),
    ({"EMOTION_REVIEW_SOURCE": "labels/emotions-review-template.csv"}, True),
    ({"EMOTION_NAMES": []}, False),
    ({"EMOTION_NAMES": ["panic", "panic"]}, False),
    ({"EMOTION_NAMES": ["panic", "joy"]}, False),
    ({"EMOTION_NAMES": ["panic", "none"]}, False),
    ({"EMOTION_NAMES": "panic"}, False),
    ({"EMOTION_MIN_CLASS_POSTS": 9}, False),
    ({"EMOTION_MIN_CLASS_POSTS": 10.0}, False),
    ({"EMOTION_MIN_CLASS_POSTS": True}, False),
    ({"EMOTION_KEEP_NONE": "yes"}, False),
    ({"EMOTION_LABELS_SOURCE": ""}, False),
    ({"EMOTION_LABELS_SOURCE": 5}, False),
    ({"EMOTION_REVIEW_SOURCE": None}, False),
]
# Locked: a hand-written review table (post_id, prelabel, human_label) and what emotion_review_agreement must give.
EMOTION_REVIEW_PROBE = [
    ("probe:2", "panic", "panic"),
    ("probe:3", "panic", "greed"),
    ("probe:4", "panic", ""),
    ("probe:5", "none", " NONE "),  # spaces around it and case are ignored
    ("probe:6", "anger", ""),
    ("probe:7", "mockery", "mockery"),
    ("probe:8", "despair", "panic"),
]
EMOTION_REVIEW_EXPECTED = {
    "n_reviewed": 5, "agreement": 0.6,
    "by_prelabel": {**{label: {"n_reviewed": 0, "agreement": None} for label in EMOTION_LABELS},
                    "panic": {"n_reviewed": 2, "agreement": 0.5}, "none": {"n_reviewed": 1, "agreement": 1.0},
                    "mockery": {"n_reviewed": 1, "agreement": 1.0}, "despair": {"n_reviewed": 1, "agreement": 0.0}},
}


def run_cells(names, namespace):
    """Execute, in order, the code cells that define names; returns the problems (empty when every cell ran)."""
    problems = []
    for name in names:
        source = cell_defining(name)
        if source is None:
            problems.append(f"no cell defines {name}")
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(source, f"cell defining {name}", "exec"), namespace)
        except Exception as exc:  # the cell stopped; the caller decides whether that was expected
            problems.append(f"cell defining {name}: {type(exc).__name__}: {exc}")
    return problems


def check_emotion_switches(report, defaults, others):
    """The five switches of chapter 11 with their defaults, and the switch check cell on good and bad values."""
    absent = [name for name in EMOTION_SWITCHES if name not in defaults]
    expected = {"EMOTION_LABELS_SOURCE": EMOTION_DEFAULT_SOURCE, "EMOTION_NAMES": EMOTION_EIGHT,
                "EMOTION_KEEP_NONE": True, "EMOTION_MIN_CLASS_POSTS": 10, "EMOTION_REVIEW_SOURCE": ""}
    wrong = [f"{name}={defaults.get(name)!r}" for name, value in expected.items() if defaults.get(name) != value]
    two_state = sorted(set(EMOTION_SWITCHES) & set(others))
    report.check(not absent and not wrong and two_state == ["EMOTION_KEEP_NONE"] and others.get("EMOTION_KEEP_NONE") is False,
                 "switches of the emotion chapter are present with their defaults (the GitHub address of the full "
                 "pre-label file, the eight emotions in order, none kept, at least 10 posts per class, no review "
                 "table); EMOTION_KEEP_NONE is the only two-state one, other value False",
                 f"missing {absent}; other defaults {wrong}; two-state {two_state}")

    wrong = []
    for overrides, accepted in EMOTION_SWITCH_PROBES:
        namespace = {"__name__": "notebook_cells"}
        problems = run_cells(["DATA_SOURCE"], namespace)  # the parameters cell
        namespace.update(overrides)
        problems += run_cells(["EMOTION_CHOICES"], namespace)  # the switch check cell
        stopped = [problem for problem in problems if problem.startswith("cell defining EMOTION_CHOICES: ValueError")]
        if accepted and problems:
            wrong.append(f"{overrides} refused: {problems[0][:120]}")
        elif not accepted and not (stopped and any(name in stopped[0] for name in overrides)):
            wrong.append(f"{overrides} not refused with its name: {problems[:1]}")
    report.check(not wrong, f"the switch check cell accepts good emotion switches and stops at bad ones with the "
                 f"switch's name ({len(EMOTION_SWITCH_PROBES)} cases: EMOTION_NAMES empty, repeated, unknown, none or not "
                 "a list; EMOTION_MIN_CLASS_POSTS below 10 or not an integer; EMOTION_KEEP_NONE not a boolean; an empty "
                 "or non-text source)", "; ".join(wrong[:3]))


def chapter_cells():
    """{chapter number: [cells]} of the notebook source, a chapter starting at its '## <n>. ' heading."""
    chapters, current = {}, None
    for cell in jupytext.read(SOURCE, fmt="py:percent").cells:
        heading = re.match(r"## ([0-9]+)\. ", cell.source) if cell.cell_type == "markdown" else None
        if heading:
            current = int(heading.group(1))
            chapters.setdefault(current, [])
        if current is not None:
            chapters[current].append(cell)
    return chapters


def check_emotion_chapter(report):
    """Chapter numbers 0 to 16 in order with the brief last; chapter 11 explains silver labels in both languages and
    reads nothing but its two sources."""
    notebook = jupytext.read(SOURCE, fmt="py:percent")
    numbers = [int(match.group(1)) for cell in notebook.cells if cell.cell_type == "markdown"
               for match in [re.match(r"## ([0-9]+)\. ", cell.source)] if match]
    chapters = chapter_cells()
    heading = {number: cells[0].source.splitlines()[0] for number, cells in chapters.items()}
    report.check(numbers == list(range(17)) and heading.get(11, "").startswith("## 11. Emotions")
                 and heading.get(12, "").startswith("## 12. Dated data")
                 and heading.get(13, "").startswith("## 13. Day by day")
                 and heading.get(14, "").startswith("## 14. Two layers")
                 and heading.get(15, "").startswith("## 15. Against the market")
                 and heading.get(16, "").startswith("## 16. The brief"),
                 "chapters 0 to 16 in order: chapter 11 is the emotions chapter, chapter 12 the dated data, chapter 13 "
                 "day by day, chapter 14 the two layers, chapter 15 against the market, chapter 16 the brief, which stays "
                 "last",
                 f"chapter headings {numbers}; 11: {heading.get(11)!r}; 12: {heading.get(12)!r}; 13: {heading.get(13)!r}; "
                 f"14: {heading.get(14)!r}; 15: {heading.get(15)!r}; 16: {heading.get(16)!r}")

    explained = []
    for cell in chapters.get(11, []):
        if cell.cell_type != "markdown" or "***" not in cell.source.splitlines():
            continue
        lines = cell.source.splitlines()
        english, chinese = "\n".join(lines[:lines.index("***")]), "\n".join(lines[lines.index("***") + 1:])
        explained.append(SILVER_WORDS[0] in english.lower() and SILVER_WORDS[1] in chinese)
    report.check(any(explained), "chapter 11 says in an explanation cell that its labels are silver labels (English "
                 "part) and \u94f6\u6807\u7b7e (Chinese part)", f"{sum(explained)} of {len(explained)} explanation cells")

    code = "\n".join(cell.source for cell in chapters.get(11, []) if cell.cell_type == "code")
    tree = ast.parse(code)
    addresses = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)
                 and "://" in node.value]
    imports = sorted({alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
                      for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(node.module or "")])}
                     & {"urllib", "requests", "http", "socket", "httpx", "urllib3", "aiohttp"})
    reads = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and node.func.attr.startswith("read_")]
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == "read_emotion_table"]
    sources = sorted({ast.unparse(node.args[0]) for node in calls if node.args})
    report.check(not addresses and not imports and [ast.unparse(node.func) for node in reads] == ["pd.read_csv"]
                 and sources == ["EMOTION_LABELS_SOURCE", "EMOTION_REVIEW_SOURCE"],
                 "chapter 11 reads only EMOTION_LABELS_SOURCE and EMOTION_REVIEW_SOURCE (through read_emotion_table, "
                 "its only pd.read_csv), holds no web address and imports no network package",
                 f"addresses {addresses}; network imports {imports}; read calls {[ast.unparse(node.func) for node in reads]}; "
                 f"read_emotion_table sources {sources}")


def check_emotion_probes(report):
    """The cells that read and check the pre-labels, on the sample: the sample's labels fit; a changed fingerprint,
    the full file and an unknown emotion stop with 'not the same set'; a missing file stops with its path and what to
    do; and the review cell's agreement on a hand-written table (all locked)."""
    import pandas as pd
    import numpy as np

    shutil.rmtree(EMOTION_PROBE_DIR, ignore_errors=True)
    EMOTION_PROBE_DIR.mkdir(parents=True)
    base = {"__name__": "notebook_cells", "brief": {}, "pd": pd, "np": np}
    problems = run_cells(["LABEL_NAMES", "read_posts"], base)
    if not report.check(not problems, "emotion probes: chapter 1's read_posts loads from the notebook source",
                        "; ".join(problems)):
        return
    posts = base["read_posts"](str(REPO_DIR / SAMPLE_SOURCE))[["post_id", "text", "label"]]

    def read(source):
        """Run the two reading cells on the sample with EMOTION_LABELS_SOURCE = source: (namespace, problems)."""
        namespace = {**base, "df_all": posts.copy(), "brief": {}, "DATA_SOURCE": SAMPLE_SOURCE,
                     "EMOTION_LABELS_SOURCE": source}
        return namespace, run_cells(["read_emotion_table", "emotion_labels"], namespace)

    namespace, problems = read(PRELABEL_SAMPLE_FILE)
    labels_file = as_dict(as_dict(namespace["brief"].get("emotions")).get("labels_file"))
    rows = len(read_csv_file(REPO_DIR / PRELABEL_SAMPLE_FILE)[1])
    report.check(not problems and labels_file.get("n_rows") == rows and labels_file.get("n_matched") == rows
                 and sum(as_dict(labels_file.get("label_counts")).values()) == rows,
                 f"emotion probes: the sample's pre-labels ({PRELABEL_SAMPLE_FILE}) fit the sample: every row matches "
                 "a post by id and text fingerprint", f"{problems[:1]}; labels file {labels_file}")

    header, sample_rows, _ = read_csv_file(REPO_DIR / PRELABEL_SAMPLE_FILE)
    changed = [list(item) for item in sample_rows]
    changed[3][4] = "0" * 8 if changed[3][4] != "0" * 8 else "f" * 8  # one text fingerprint changed
    unknown = [list(item) for item in sample_rows]
    unknown[5][1] = "joy"  # an emotion that is not one of the nine labels
    cases = []
    for name, file_rows in (("changed-fingerprint", changed), ("unknown-emotion", unknown)):
        path = EMOTION_PROBE_DIR / f"{name}.csv"
        path.write_text("".join(prelabel.row_line(item) for item in [header] + file_rows), encoding="utf-8")
        cases.append((name, str(path.relative_to(REPO_DIR)), changed[3][0] if name == "changed-fingerprint" else "joy"))
    cases.append(("full pre-labels with the sample", PRELABEL_FULL_FILE, "ids are not posts of this data source"))
    wrong = []
    for name, source, detail in cases:
        _, problems = read(source)
        stop = problems[0] if problems else ""
        if not (stop.startswith("cell defining emotion_labels: ValueError") and "not the same set" in stop
                and source in stop and detail in stop):
            wrong.append(f"{name}: {stop[:200] or 'not stopped'}")
        report.info(f"emotion probe {name}: {stop[:260] or 'NOT stopped'}")
    report.check(not wrong, "emotion probes: a pre-label file with a changed text fingerprint, with an unknown emotion, "
                 "and the full pre-label file read with the sample all stop the notebook, saying that the label file "
                 "and the data source are not the same set", "; ".join(wrong))

    missing = str((EMOTION_PROBE_DIR / "no-such-prelabels.csv").relative_to(REPO_DIR))
    _, problems = read(missing)
    stop = problems[0] if problems else ""
    report.info(f"emotion probe missing file: {stop[:400] or 'NOT stopped'}")
    report.check(stop.startswith("cell defining emotion_labels: FileNotFoundError")
                 and all(part in stop for part in ("EMOTION_LABELS_SOURCE", missing, "GitHub", "pushed",
                                                   "labels/emotions-prelabel-v1.csv")),
                 "emotion probes: a pre-label file that cannot be read stops the notebook with its path, says that the "
                 "default address can be read only after the file has been pushed to GitHub, and names the local copy "
                 "labels/emotions-prelabel-v1.csv", stop[:300])

    # The review table: a hand-written table through EMOTION_REVIEW_SOURCE, no table, and a table that cannot be read.
    review = EMOTION_PROBE_DIR / "review.csv"
    with review.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerows([REVIEW_TEMPLATE_COLUMNS] + [list(row) for row in EMOTION_REVIEW_PROBE])
    results, wrong = {}, []
    for name, source in (("filled-in table", str(review.relative_to(REPO_DIR))), ("no table", ""),
                         ("missing table", str((EMOTION_PROBE_DIR / "no-such-review.csv").relative_to(REPO_DIR)))):
        namespace, problems = read(PRELABEL_SAMPLE_FILE)
        namespace["EMOTION_REVIEW_SOURCE"] = source
        problems += run_cells(["emotion_review_agreement"], namespace)
        results[name] = (as_dict(namespace["brief"].get("emotions")).get("human_review"), problems)
    expected = {"source": str(review.relative_to(REPO_DIR)), **EMOTION_REVIEW_EXPECTED}
    got, problems = results["filled-in table"]
    report.info(f"emotion probe review table {EMOTION_REVIEW_PROBE} -> {json.dumps(got)}")
    if problems or not isinstance(got, dict) or {key: got.get(key) for key in ("source", "n_reviewed", "by_prelabel")} != \
            {key: expected[key] for key in ("source", "n_reviewed", "by_prelabel")} \
            or not same_value(got.get("agreement"), expected["agreement"]):
        wrong.append(f"filled-in table: {problems[:1]} {got}")
    got, problems = results["no table"]
    if problems or got != {"source": "", "n_reviewed": 0, "agreement": None,
                           "by_prelabel": {label: {"n_reviewed": 0, "agreement": None} for label in EMOTION_LABELS}}:
        wrong.append(f"no table: {problems[:1]} {got}")
    got, problems = results["missing table"]
    stop = problems[-1] if problems else ""
    if not (stop.startswith("cell defining emotion_review_agreement: FileNotFoundError") and "EMOTION_REVIEW_SOURCE" in stop
            and "no-such-review.csv" in stop):
        wrong.append(f"missing table: {stop[:200] or 'not stopped'}")
    report.check(not wrong, "emotion probes: the review cell gives the precomputed agreement for a hand-written table "
                 f"({len(EMOTION_REVIEW_PROBE)} rows, {EMOTION_REVIEW_EXPECTED['n_reviewed']} filled in, agreement "
                 f"{EMOTION_REVIEW_EXPECTED['agreement']}, per pre-label), 0 rows and null without a table, and stops "
                 "with the path when EMOTION_REVIEW_SOURCE cannot be read", "; ".join(wrong))
    shutil.rmtree(EMOTION_PROBE_DIR, ignore_errors=True)


EMOTION_FILE_CACHE = {}


def emotion_file_rows(source):
    """[(post_id, emotion), ...] of the pre-label file a run read, when it is a local file in the label format."""
    if "://" in source:
        return None
    path = Path(source) if Path(source).is_absolute() else REPO_DIR / source
    if not path.is_file():
        return None
    if path not in EMOTION_FILE_CACHE:
        header, rows, _ = read_csv_file(path)
        EMOTION_FILE_CACHE[path] = ([(item[0], item[1]) for item in rows if len(item) == len(PRELABEL_COLUMNS)]
                                    if header == PRELABEL_COLUMNS else None)
    return EMOTION_FILE_CACHE[path]


def first_maximum(probabilities, classes):
    """The class with the largest probability, the earlier class on a tie (the notebook's argmax)."""
    values = [probabilities.get(label) for label in classes]
    if not all(is_number(value) for value in values):
        return None
    return classes[values.index(max(values))]


def emotion_quote_problems(where, group, classes, limit, emotion_ids, belongs):
    """Problems of a group's example posts: per class at most the limit and at most the class's posts; the network's
    call (the largest probability, earlier class on a tie) is the class; probabilities add up to 1; highest first;
    different posts and texts; not in the emotion data; and in the group (belongs(text))."""
    problems = []
    quotes, counts = group.get("quotes"), as_dict(group.get("counts"))
    if not isinstance(quotes, dict) or sorted(quotes) != sorted(classes) or group.get("quotes_per_class") != limit:
        return [f"{where}: quotes keyed {sorted(quotes) if isinstance(quotes, dict) else quotes!r}, quotes_per_class "
                f"{group.get('quotes_per_class')!r} (expected {limit})"]
    ids, texts = [], []
    for label in classes:
        items = quotes[label]
        if not isinstance(items, list) or len(items) > limit or len(items) > (counts.get(label) or 0):
            problems.append(f"{where} {label}: {len(items) if isinstance(items, list) else items!r} quotes for "
                            f"{counts.get(label)!r} posts, limit {limit}")
            continue
        previous = None
        for quote in items:
            quote = as_dict(quote)
            probability, probabilities = quote.get("probability"), as_dict(quote.get("probabilities"))
            post_id, text = quote.get("post_id"), quote.get("text")
            if not (isinstance(post_id, str) and isinstance(text, str) and is_number(probability) and 0 <= probability <= 1
                    and sorted(probabilities) == sorted(classes)
                    and all(is_number(value) and 0 <= value <= 1 for value in probabilities.values())):
                problems.append(f"{where} {label}: quote {quote!r}")
                continue
            if abs(sum(probabilities.values()) - 1) > 1e-4 or first_maximum(probabilities, classes) != label \
                    or probabilities[label] != probability:
                problems.append(f"{where} {label}: {post_id} is not called {label} (probabilities {probabilities})")
            if previous is not None and probability > previous:
                problems.append(f"{where} {label}: quotes not sorted by probability, highest first")
            previous = probability
            if emotion_ids is not None and post_id in emotion_ids:
                problems.append(f"{where} {label}: {post_id} is in the emotion data")
            if not belongs(text):
                problems.append(f"{where} {label}: {post_id} does not belong to the group ({text[:60]!r})")
            ids.append(post_id)
            texts.append(text)
    if len(set(ids)) != len(ids) or len(set(texts)) != len(texts):
        problems.append(f"{where}: a post or a text is quoted twice")
    return problems


def check_emotions(report, label, brief, switches, full_run=False):
    """The emotions section: the pre-label file as read, the leave-out steps and classes (recomputed from its label
    counts), the split, the evaluation and its bar, the emotion calls of the topic posts per group (adding up with
    the drawers section), the example posts and the review agreement."""
    emotions = brief.get("emotions")
    if not report.check(isinstance(emotions, dict) and bool(emotions), f"{label}: emotions section has content"):
        return
    topics, drawers = as_dict(brief.get("topics")), as_dict(brief.get("drawers"))
    source = switches["EMOTION_LABELS_SOURCE"]
    file_rows = emotion_file_rows(source)
    if file_rows is None:
        report.info(f"{label}: {source} is not a local pre-label file; the comparisons with the file are skipped")

    # The pre-label file as read.
    labels_file = as_dict(emotions.get("labels_file"))
    counts = counts_of(labels_file.get("label_counts"), EMOTION_LABELS + [PRELABEL_UNPARSED])
    n_rows = labels_file.get("n_rows")
    file_counts = None if file_rows is None else per_label(file_rows)
    report.check(emotions.get("labels_source") == source and "silver labels" in str(emotions.get("label_kind"))
                 and counts is not None and is_count(n_rows) and sum(counts.values()) == n_rows
                 and labels_file.get("n_matched") == n_rows
                 and isinstance(labels_file.get("models"), list) and isinstance(labels_file.get("prompt_versions"), list)
                 and (file_rows is None or (len(file_rows) == n_rows and counts == file_counts)),
                 f"{label}: the pre-label file as read: source as set, silver labels named, rows per label adding up to "
                 "the rows, every row matched, and the same rows per label as the file itself",
                 f"source {emotions.get('labels_source')!r} (switch {source!r}), {n_rows} rows, matched "
                 f"{labels_file.get('n_matched')}, per label {labels_file.get('label_counts')}, file {file_counts}")
    if counts is None or not is_count(n_rows):
        return

    # The switches, the leave-out steps and the classes, recomputed from the rows per label.
    names, keep_none, minimum = list(switches["EMOTION_NAMES"]), switches["EMOTION_KEEP_NONE"], switches["EMOTION_MIN_CLASS_POSTS"]
    report.check(emotions.get("emotion_names") == names and emotions.get("keep_none") is keep_none
                 and emotions.get("min_class_posts") == minimum,
                 f"{label}: EMOTION_NAMES, EMOTION_KEEP_NONE and EMOTION_MIN_CLASS_POSTS recorded as set",
                 f"{emotions.get('emotion_names')}, {emotions.get('keep_none')!r}, {emotions.get('min_class_posts')!r}")
    not_named = [item for item in EMOTION_EIGHT if item not in names]
    candidates = names + (["none"] if keep_none else [])
    too_few = [item for item in candidates if counts[item] < minimum]
    expected_steps = [("unparsed", [PRELABEL_UNPARSED], counts[PRELABEL_UNPARSED]),
                      ("none", [] if keep_none else ["none"], 0 if keep_none else counts["none"]),
                      ("not_in_emotion_names", not_named, sum(counts[item] for item in not_named)),
                      ("too_few_posts", too_few, sum(counts[item] for item in too_few))]
    steps = [as_dict(step) for step in emotions.get("dropped") or []]
    got_steps = [(step.get("step"), step.get("labels"), step.get("n_posts")) for step in steps]
    classes = emotions.get("classes")
    expected_classes = [item for item in candidates if item not in too_few]
    n_posts = emotions.get("n_posts")
    class_counts = emotions.get("class_counts")
    report.check(got_steps == expected_steps and all(isinstance(step.get("rule"), str) and step["rule"] for step in steps)
                 and classes == expected_classes and len(expected_classes) >= 2
                 and class_counts == {item: counts[item] for item in expected_classes}
                 and is_count(n_posts) and n_posts == sum(class_counts.values())
                 and sum(step[2] for step in got_steps if is_count(step[2])) + n_posts == labels_file.get("n_matched"),
                 f"{label}: the four leave-out steps (unparsed, none, not in EMOTION_NAMES, fewer than "
                 "EMOTION_MIN_CLASS_POSTS posts) leave out the right labels and posts, and add up with the emotion data "
                 "to the matched rows; the classes are EMOTION_NAMES in order with none last, minus the classes left out",
                 f"steps {got_steps} (expected {expected_steps}); classes {classes} (expected {expected_classes}); "
                 f"{n_posts} posts {class_counts}")
    report.check(emotions.get("keep_none") is keep_none and (keep_none or ("none" not in (classes or [])
                                                                          and steps[1:2] and steps[1].get("labels") == ["none"])),
                 f"{label}: with EMOTION_KEEP_NONE {keep_none} the classes {'may include' if keep_none else 'leave out'} "
                 "none", f"classes {classes}")
    if classes != expected_classes or not is_count(n_posts):
        return

    # The split.
    split = as_dict(emotions.get("split"))
    sizes = counts_of(split.get("n_posts"), SPLIT_PARTS)
    per_part = as_dict(split.get("class_counts"))
    split_ok = (sizes is not None and sum(sizes.values()) == n_posts and split.get("seed") == switches["RANDOM_SEED"]
                and split.get("stratified_by") == "emotion"
                and split.get("fractions") == {"train": 0.8, "validation": 0.1, "test": 0.1})
    for part in SPLIT_PARTS if split_ok else []:
        part_counts = counts_of(per_part.get(part), classes)
        split_ok = split_ok and part_counts is not None and all(count > 0 for count in part_counts.values()) \
            and sum(part_counts.values()) == sizes[part]
    split_ok = split_ok and all(sum(per_part[part][item] for part in SPLIT_PARTS) == class_counts[item] for item in classes) \
        and abs(sizes["train"] - 0.8 * n_posts) <= len(classes) and sizes["train"] > sizes["test"] > 0
    report.check(split_ok, f"{label}: emotion training, validation and test parts add up to the emotion data, about "
                 "80/10/10, every class in every part, class by class", f"{split.get('n_posts')} {per_part}")
    if not split_ok:
        return
    n_test, test_counts = sizes["test"], per_part["test"]

    # Vectorizers, networks, training and the choice.
    vectors = as_dict(emotions.get("vectorizers"))
    vocab = [as_dict(vectors.get(name)).get("vocab_size") for name in ("count", "tfidf")]
    report.check(vectors.get("fitted_on_n_posts") == sizes["train"] and vectors.get("vocab_size_requested") == switches["VOCAB_SIZE"]
                 and all(is_count(size) and 0 < size <= switches["VOCAB_SIZE"] for size in vocab),
                 f"{label}: both emotion vectorizers learned at most VOCAB_SIZE words from the emotion training part only",
                 f"fitted on {vectors.get('fitted_on_n_posts')} posts, vocabulary sizes {vocab}")
    models = emotions.get("models") if isinstance(emotions.get("models"), list) else []
    training = as_dict(emotions.get("training"))
    report.check(models == SENTIMENT_MODELS and training.get("epochs") == switches["EPOCHS"]
                 and training.get("batch_size") == switches["BATCH_SIZE"] and training.get("seed") == switches["RANDOM_SEED"]
                 and training.get("output_layer") == f"{len(classes)} units, softmax"
                 and training.get("loss") == "sparse_categorical_crossentropy"
                 and training.get("target_of_class") == {item: position for position, item in enumerate(classes)}
                 and sorted(as_dict(training.get("history"))) == sorted(models),
                 f"{label}: a word-count and a TF-IDF network, one softmax unit per class, EPOCHS, BATCH_SIZE and the seed",
                 f"models {models}, output {training.get('output_layer')!r}, targets {training.get('target_of_class')}")
    selection = as_dict(emotions.get("selection"))
    validation = as_dict(selection.get("validation_macro_f1"))
    scores = {name: (validation.get(name) if is_number(validation.get(name)) else -1.0) for name in models}
    report.check(sorted(validation) == sorted(models) and all(is_score(score) for score in validation.values())
                 and bool(models) and selection.get("selected") == max(models, key=lambda name: scores[name]),
                 f"{label}: one emotion network chosen by validation macro-F1 (a tie goes to the word-count network), "
                 "with the validation score of every network",
                 f"selected {selection.get('selected')!r}, validation {validation}")

    # The evaluation on the emotion test part.
    evaluation = as_dict(emotions.get("evaluation"))
    rows = [row for row in evaluation.get("rows") or [] if isinstance(row, dict)]
    train_counts = per_part["train"]
    majority = evaluation.get("majority_label")
    problems = []
    if [row.get("model") for row in rows] != models + SENTIMENT_BASELINES or evaluation.get("baselines") != SENTIMENT_BASELINES:
        problems.append(f"rows {[row.get('model') for row in rows]}, baselines {evaluation.get('baselines')}")
    if majority not in classes or train_counts[majority] != max(train_counts.values()):
        problems.append(f"majority label {majority!r} is not a most frequent class of the training part {train_counts}")
    for row in rows:
        name = row.get("model")
        per_class = as_dict(row.get("per_class"))
        if row.get("kind") != ("baseline" if name in SENTIMENT_BASELINES else "model") or row.get("n") != n_test \
                or sorted(per_class) != sorted(classes):
            problems.append(f"{name}: kind {row.get('kind')!r}, {row.get('n')} posts (test part {n_test}), classes {sorted(per_class)}")
            continue
        supports = {item: as_dict(per_class[item]).get("support") for item in classes}
        if supports != test_counts:
            problems.append(f"{name}: supports {supports} are not the test part's classes {test_counts}")
        values = [row.get("accuracy"), row.get("macro_f1")] + [as_dict(per_class[item]).get(measure) for item in classes
                                                                for measure in ("precision", "recall", "f1")]
        if not all(is_score(value) for value in values) or row.get("accuracy") is None or row.get("macro_f1") is None:
            problems.append(f"{name}: a score is neither a number in [0, 1] nor null (accuracy and macro-F1 are numbers)")
        if name == "always_majority" and majority in classes and not same_value(row.get("accuracy"), test_counts[majority] / n_test):
            problems.append(f"always_majority: accuracy {row.get('accuracy')!r} is not the majority class's share of the test part")
    report.check(bool(rows) and not problems,
                 f"{label}: the emotion evaluation table has one row per network and per baseline on the emotion test "
                 "part, class supports equal to the test part's classes, scores in [0, 1] or null, and the majority "
                 "baseline guessing a most frequent class of the training part", "; ".join(problems[:4]))
    accuracy = {row.get("model"): row.get("accuracy") for row in rows}
    matrices = as_dict(emotions.get("confusion_matrices"))
    problems = []
    for name in models:
        matrix = as_dict(matrices.get(name))
        cells = matrix.get("counts")
        if not (matrix.get("labels") == classes and isinstance(cells, list) and len(cells) == len(classes)
                and all(isinstance(line, list) and len(line) == len(classes) and all(is_count(cell) for cell in line)
                        for line in cells)):
            problems.append(f"{name}: not a {len(classes)}x{len(classes)} table of counts with labels {classes}")
            continue
        if sum(map(sum, cells)) != n_test or matrix.get("n") != n_test \
                or [sum(line) for line in cells] != [test_counts[item] for item in classes]:
            problems.append(f"{name}: cells add up to {sum(map(sum, cells))}, rows {[sum(line) for line in cells]}")
        elif not same_value(accuracy.get(name), sum(cells[index][index] for index in range(len(classes))) / n_test):
            problems.append(f"{name}: the diagonal does not give the accuracy")
    report.check(bool(models) and not problems, f"{label}: one emotion confusion matrix per network, adding up to the test "
                 "part, its rows to the test part's classes and its diagonal to the accuracy", "; ".join(problems))
    macro = {row.get("model"): row.get("macro_f1") for row in rows}

    def beats_both(name):
        return is_number(macro.get(name)) and all(is_number(macro.get(baseline)) and macro[name] > macro[baseline]
                                                  for baseline in SENTIMENT_BASELINES)

    report.check(emotions.get("beats_baselines") == {name: beats_both(name) for name in models},
                 f"{label}: emotion beats_baselines agrees with the evaluation rows", f"{emotions.get('beats_baselines')}")
    if full_run:
        for name in models:
            report.check(beats_both(name), f"{label}: emotion {name} macro-F1 on the emotion test part beats always "
                         "majority and uniform random",
                         f"{macro.get(name)} vs " + ", ".join(f"{baseline} {macro.get(baseline)}" for baseline in SENTIMENT_BASELINES))

    # The emotion calls of the topic posts, per group.
    scoring = as_dict(emotions.get("scoring"))
    n_scored = scoring.get("n_scored")
    report.check(scoring.get("function") == "emotion_proba" and scoring.get("model") == selection.get("selected")
                 and scoring.get("post_table") == "topic_posts" and scoring.get("table") == "emotion_posts"
                 and is_count(n_scored) and n_scored == topics.get("n_posts") and is_count(scoring.get("n_in_emotion_data"))
                 and scoring["n_in_emotion_data"] <= min(n_scored, n_posts),
                 f"{label}: every topic post got an emotion from the chosen network (posts in the emotion data counted)",
                 f"{scoring}; topic posts {topics.get('n_posts')}")
    drawer_items = [item for item in drawers.get("drawers") or [] if isinstance(item, dict)]
    rows = [item for item in emotions.get("drawers") or [] if isinstance(item, dict)]
    groups = [("all", as_dict(emotions.get("overall")), EMOTION_QUOTES_OVERALL, n_scored)]
    groups += [(f"drawer {item.get('drawer')}", row, EMOTION_QUOTES_PER_GROUP, item.get("n_posts"))
               for item, row in zip(drawer_items, rows)]
    groups += [("any drawer", as_dict(emotions.get("any_drawer")), EMOTION_QUOTES_PER_GROUP, drawers.get("n_in_any_drawer")),
               ("no drawer", as_dict(emotions.get("no_drawer")), EMOTION_QUOTES_PER_GROUP, drawers.get("n_in_no_drawer"))]
    problems = []
    if [(row.get("drawer"), row.get("name")) for row in rows] != [(item.get("drawer"), item.get("name")) for item in drawer_items]:
        problems.append("the drawer rows are not the drawers of the drawers section, in its order")
    for where, group, _, expected_n in groups:
        n, got = group.get("n_posts"), counts_of(group.get("counts"), classes)
        shares = as_dict(group.get("shares"))
        if not is_count(n) or n != expected_n or got is None or sum(got.values()) != n:
            problems.append(f"{where}: {n!r} posts (expected {expected_n!r}), counts {group.get('counts')}")
            continue
        if sorted(shares) != sorted(classes) or not all(same_value(shares[item], None if n == 0 else got[item] / n)
                                                        for item in classes):
            problems.append(f"{where}: shares {shares} do not follow from the counts")
    every, anywhere, nowhere = (as_dict(groups[0][1].get("counts")), as_dict(groups[-2][1].get("counts")),
                                as_dict(groups[-1][1].get("counts")))
    if not all(is_count(anywhere.get(item)) and is_count(nowhere.get(item))
               and anywhere[item] + nowhere[item] == every.get(item) for item in classes):
        problems.append(f"any drawer {anywhere} + no drawer {nowhere} is not all topic posts {every}")
    report.check(not problems, f"{label}: emotions per group: all topic posts, every drawer (same order, names and sizes as "
                 "the drawers section), any drawer and no drawer; counts add up to the group's posts, shares follow "
                 "from the counts, any drawer + no drawer = all topic posts, class by class", "; ".join(problems[:4]))

    # The example posts.
    emotion_ids = None if file_rows is None else {post_id for post_id, emotion in file_rows if emotion in classes}
    words = {item.get("drawer"): (item.get("words") or [], item.get("excluded_phrases") or []) for item in drawer_items}

    def in_drawer(text, drawer):
        found_words, excluded = words.get(drawer, ([], []))
        return bool(rule_matches(text, found_words, excluded))

    def in_topics(text):
        return switches["TOPIC_SCOPE"] != "btc" or re.search(r"\b(?:bitcoin|btc)\b", text, re.IGNORECASE) is not None

    checks = [("all", groups[0][1], EMOTION_QUOTES_OVERALL, in_topics)]
    checks += [(f"drawer {row.get('drawer')}", row, EMOTION_QUOTES_PER_GROUP,
                lambda text, drawer=row.get("drawer"): in_drawer(text, drawer)) for row in rows]
    checks += [("any drawer", groups[-2][1], EMOTION_QUOTES_PER_GROUP, lambda text: any(in_drawer(text, d) for d in words)),
               ("no drawer", groups[-1][1], EMOTION_QUOTES_PER_GROUP, lambda text: not any(in_drawer(text, d) for d in words))]
    problems, n_quotes = [], 0
    for where, group, limit, belongs in checks:
        problems += emotion_quote_problems(where, group, classes, limit, emotion_ids, belongs)
        n_quotes += sum(len(items) for items in as_dict(group.get("quotes")).values() if isinstance(items, list))
    report.check(not problems, f"{label}: example posts ({n_quotes}): at most {EMOTION_QUOTES_OVERALL} per class for all "
                 f"topic posts and {EMOTION_QUOTES_PER_GROUP} per group otherwise, called that class by the network, "
                 "highest probability first, different posts and texts, none in the emotion data, each in its group "
                 "(also when there are none)", "; ".join(problems[:4]))

    # The review interface.
    review = as_dict(emotions.get("human_review"))
    by_prelabel = as_dict(review.get("by_prelabel"))
    reviewed = review.get("n_reviewed")
    parts = [(as_dict(by_prelabel.get(item)).get("n_reviewed"), as_dict(by_prelabel.get(item)).get("agreement"))
             for item in EMOTION_LABELS]
    review_ok = (review.get("source") == switches["EMOTION_REVIEW_SOURCE"] and sorted(by_prelabel) == sorted(EMOTION_LABELS)
                 and is_count(reviewed) and all(is_count(n) and is_score(share) and (share is None) == (n == 0) for n, share in parts)
                 and sum(n for n, _ in parts) == reviewed and (review.get("agreement") is None) == (reviewed == 0)
                 and (reviewed == 0 or same_value(review.get("agreement"), sum(n * share for n, share in parts if n) / reviewed))
                 and (switches["EMOTION_REVIEW_SOURCE"] != "" or (reviewed == 0 and review.get("agreement") is None)))
    report.check(review_ok, f"{label}: the review agreement: source as set, reviewed rows per pre-label adding up, "
                 "agreement null exactly when nothing was reviewed (always without a review table)", f"{review}")


def stable_emotions(brief):
    """The parts of the emotions section that do not depend on training, or None."""
    emotions = as_dict(brief).get("emotions") if isinstance(brief, dict) else None
    if not isinstance(emotions, dict) or not emotions:
        return None
    evaluation = as_dict(emotions.get("evaluation"))
    keys = ["labels_source", "labels_file", "emotion_names", "keep_none", "min_class_posts", "dropped", "classes",
            "n_posts", "class_counts", "split", "vectorizers", "human_review"]
    return {**{key: emotions.get(key) for key in keys},
            "baselines": [row for row in evaluation.get("rows") or [] if as_dict(row).get("kind") == "baseline"],
            "majority_label": evaluation.get("majority_label"),
            "scoring": {key: as_dict(emotions.get("scoring")).get(key) for key in ("n_scored", "n_in_emotion_data")},
            "group_sizes": [as_dict(emotions.get(key)).get("n_posts") for key in ("overall", "any_drawer", "no_drawer")]
            + [as_dict(row).get("n_posts") for row in emotions.get("drawers") or []]}


def compare_emotions(report, first, again):
    """Two runs with the same switches: the parts of the emotions section that do not depend on training agree."""
    left, right = stable_emotions(first), stable_emotions(again)
    detail = ""
    if left is None or right is None:
        detail = "a brief or its emotions section is missing"
    elif left != right:
        detail = "differing parts: " + ", ".join(key for key in left if left[key] != right[key])
    report.check(left is not None and left == right,
                 "default switches run twice: the emotions sections agree on everything that does not depend on training "
                 "(the pre-label file as read, the leave-out steps, classes, posts per class, the split per part and "
                 "class, the vocabularies, both baseline rows, the posts scored and the group sizes)", detail)


def print_emotions(brief):
    """Print the emotion chapter's numbers (full run): the pre-labels, the leave-out steps, the split, the evaluation
    table and confusion matrices, the choice, the emotions per group and every example post."""
    emotions = as_dict(brief.get("emotions"))
    if not emotions:
        return
    classes = emotions.get("classes") or []

    def number(value, digits=4):
        return f"{value:.{digits}f}" if is_number(value) else "-"

    labels_file = as_dict(emotions.get("labels_file"))
    print("\nEmotions (full run)")
    print(f"  pre-label file:                {emotions.get('labels_source')}: {labels_file.get('n_rows')} rows, "
          f"{labels_file.get('n_matched')} matched; models {labels_file.get('models')}, prompts {labels_file.get('prompt_versions')}")
    print(f"  rows per label:                {labels_file.get('label_counts')}")
    for step in emotions.get("dropped") or []:
        print(f"  left out ({step.get('step')}):{' ' * max(1, 20 - len(str(step.get('step'))))}{step.get('n_posts')} posts, "
              f"labels {step.get('labels')}")
    print(f"  classes ({len(classes)}):                  {classes}")
    print(f"  emotion data:                  {emotions.get('n_posts')} posts   {emotions.get('class_counts')}")
    split = as_dict(emotions.get("split"))
    print(f"  split (posts):                 {split.get('n_posts')}")
    for part, part_counts in as_dict(split.get("class_counts")).items():
        print(f"    {part:11}                 {part_counts}")
    vectors = as_dict(emotions.get("vectorizers"))
    print(f"  vectorizers:                   count {as_dict(vectors.get('count')).get('vocab_size')}, tfidf "
          f"{as_dict(vectors.get('tfidf')).get('vocab_size')} words, fitted on {vectors.get('fitted_on_n_posts')} posts")
    selection = as_dict(emotions.get("selection"))
    print(f"  validation macro-F1:           {selection.get('validation_macro_f1')}   chosen: {selection.get('selected')}")
    evaluation = as_dict(emotions.get("evaluation"))
    print(f"  evaluation on the emotion test part (majority label {evaluation.get('majority_label')}; P precision, "
          "R recall, F1, n support; - = null):")
    print(f"    {'model':16} {'posts':>5} {'accuracy':>8} {'macro-F1':>8}")
    for row in evaluation.get("rows") or []:
        print(f"    {str(row.get('model')):16} {str(row.get('n')):>5} {number(row.get('accuracy')):>8} "
              f"{number(row.get('macro_f1')):>8}")
    print(f"    {'class':14} " + " ".join(f"{str(row.get('model'))[:15]:>24}" for row in evaluation.get("rows") or []))
    print(f"    {'':14} " + " ".join(f"{'P':>6}{'R':>6}{'F1':>6}{'n':>6}" for _ in evaluation.get("rows") or []))
    for item in classes:
        cells = []
        for row in evaluation.get("rows") or []:
            scores = as_dict(as_dict(row.get("per_class")).get(item))
            cells.append(f"{number(scores.get('precision'), 3):>6}{number(scores.get('recall'), 3):>6}"
                         f"{number(scores.get('f1'), 3):>6}{str(scores.get('support')):>6}")
        print(f"    {item:14} " + " ".join(cells))
    print(f"  beats both baselines:          {emotions.get('beats_baselines')}")
    for name, matrix in as_dict(emotions.get("confusion_matrices")).items():
        print(f"  confusion matrix {name} (rows true, columns predicted, {matrix.get('n')} posts):")
        print(f"    {'':14} " + " ".join(f"{item[:6]:>6}" for item in matrix.get("labels") or []))
        for item, line in zip(matrix.get("labels") or [], matrix.get("counts") or []):
            print(f"    {item:14} " + " ".join(f"{cell:>6}" for cell in line))
    scoring = as_dict(emotions.get("scoring"))
    print(f"  topic posts given an emotion:  {scoring.get('n_scored')} by {scoring.get('model')}; in the emotion data: "
          f"{scoring.get('n_in_emotion_data')}")
    groups = [("all", "all topic posts", as_dict(emotions.get("overall"))),
              ("any drawer", "in at least one drawer", as_dict(emotions.get("any_drawer"))),
              ("no drawer", "in no drawer", as_dict(emotions.get("no_drawer")))]
    groups += [(row.get("drawer"), row.get("name"), row) for row in emotions.get("drawers") or [] if isinstance(row, dict)]
    print("  emotions per group (posts; then share of each class, count in brackets):")
    print(f"    {'group':15} {'posts':>6} " + " ".join(f"{item[:12]:>18}" for item in classes))
    for key, _, group in groups:
        counts, shares = as_dict(group.get("counts")), as_dict(group.get("shares"))
        print(f"    {str(key):15} {str(group.get('n_posts')):>6} "
              + " ".join(f"{number(shares.get(item), 3):>9} ({str(counts.get(item)):>6})" for item in classes))
    print("  example posts ([emotion probability, post_id] text):")
    for key, name, group in groups:
        print(f"    {key}: {name} ({group.get('n_posts')} posts)")
        quotes = as_dict(group.get("quotes"))
        for item in classes:
            for quote in quotes.get(item) or []:
                print(f"        [{item} {number(quote.get('probability'), 3)}, {quote.get('post_id')}] {quote.get('text')}")
        if not any(quotes.get(item) for item in classes):
            print("        (none)")
    review = as_dict(emotions.get("human_review"))
    print(f"  review table:                  source {review.get('source')!r}, reviewed {review.get('n_reviewed')}, "
          f"agreement {review.get('agreement')}")


# ---------------------------------------------------------------- dated data (chapter 12)
# Two dated datasets (tweets: retail layer, news: media layer) read into one dated post table and counted per layer
# and UTC day. The quick check reads the samples in samples/, the full check the copies in data/raw/; neither reads
# Hugging Face.

DATED_SWITCH_DEFAULTS = {
    "DATED_TWEETS_SOURCE": "hf://datasets/ExponentialScience/DLT-Tweets@c7d979394488bbd94bd6f493af1775d6b6159176/data",
    "DATED_NEWS_SOURCE": ("hf://datasets/ExponentialScience/DLT-Sentiment-News@b52bfb8b3ec099924a52f6f4bfd1a8ac6c89c9e1/"
                          "data"),
    "DATED_START": "2021-01-01", "DATED_END": None, "DATED_DAILY_CAP": 2000, "DATED_BTC_ONLY": True,
    "DATED_LANGUAGE": "en", "NEWS_TEXT": "title", "NEWS_LABEL": "market_direction",
}
DATED_OTHER_VALUES = {"DATED_BTC_ONLY": False, "DATED_LANGUAGE": None, "NEWS_TEXT": "text", "NEWS_LABEL": "votes"}
DATED_TWEETS_SAMPLE = "samples/dlt-tweets-sample.parquet"
DATED_NEWS_SAMPLE = "samples/dlt-news-sample.parquet"
DATED_TWEETS_FULL = "data/raw/dlt-tweets/data"  # written by scripts/download_data.sh --dated
DATED_NEWS_FULL = "data/raw/dlt-news/data"
DATED_QUICK_CAP = 80  # below the busiest sample day after the default filters (120 tweets that mention Bitcoin)
DATED_SAMPLE_SETTINGS = {"DATED_TWEETS_SOURCE": DATED_TWEETS_SAMPLE, "DATED_NEWS_SOURCE": DATED_NEWS_SAMPLE,
                         "DATED_DAILY_CAP": DATED_QUICK_CAP}
DATED_FULL_SETTINGS = {"DATED_TWEETS_SOURCE": DATED_TWEETS_FULL, "DATED_NEWS_SOURCE": DATED_NEWS_FULL}
DATED_DOC = REPO_DIR / "docs" / "dated-posts.md"
DATED_DOC_HEADER = "| Column | Type | Null | Meaning |"
DATED_LAYER_NAMES = ["media", "kol", "retail"]
DATED_STEP_NAMES = ["read", "time_parsed", "in_dates", "language", "mentions_btc", "unique", "daily_cap"]
DATED_TIME_KINDS = ["with_zone", "without_zone", "date_only", "unparseable"]
DATED_DATASET_RECORDS = {  # what the brief must say about each dataset
    "tweets": {"name": "ExponentialScience/DLT-Tweets", "revision": "c7d979394488bbd94bd6f493af1775d6b6159176",
               "license": "CC BY-NC 4.0", "layer": "retail"},
    "news": {"name": "ExponentialScience/DLT-Sentiment-News", "revision": "b52bfb8b3ec099924a52f6f4bfd1a8ac6c89c9e1",
             "license": "CC BY-NC 4.0", "layer": "media"},
}
DATED_PROBE_DIR = REPO_DIR / "data" / "checks" / "dated-probe"
# Locked (only add): switch values the switch check cell must accept (True) or stop at with the switch's name (False).
DATED_SWITCH_PROBES = [
    ({"DATED_END": "2022-06-30"}, True), ({"DATED_LANGUAGE": None}, True), ({"DATED_DAILY_CAP": 1}, True),
    ({"NEWS_TEXT": "text", "NEWS_LABEL": "votes", "DATED_BTC_ONLY": False}, True),
    ({"DATED_START": "2021/01/01"}, False), ({"DATED_START": "2021-02-30"}, False), ({"DATED_START": 20210101}, False),
    ({"DATED_END": "2020-12-31"}, False), ({"DATED_END": "end"}, False), ({"DATED_DAILY_CAP": 0}, False),
    ({"DATED_DAILY_CAP": 2.5}, False), ({"DATED_DAILY_CAP": True}, False), ({"DATED_BTC_ONLY": "yes"}, False),
    ({"DATED_LANGUAGE": ""}, False), ({"DATED_LANGUAGE": 5}, False), ({"NEWS_TEXT": "body"}, False),
    ({"NEWS_LABEL": "label"}, False), ({"DATED_TWEETS_SOURCE": ""}, False), ({"DATED_NEWS_SOURCE": None}, False),
]
# Locked (only add): ways to break a valid dated post table, and the column the problem must name.
DATED_SHAPE_PROBES = [
    ("a column missing", lambda table: table.drop(columns="source"), "source"),
    ("an extra column", lambda table: table.assign(author="x"), "author"),
    ("an empty value where none is allowed", lambda table: table.assign(text=[None] + list(table["text"][1:])), "text"),
    ("a layer that is not one of the three", lambda table: table.assign(layer=["whale"] + list(table["layer"][1:])),
     "layer"),
    ("a repeated post_id", lambda table: table.assign(post_id=[table["post_id"][0]] * len(table)), "post_id"),
    ("a date not written YYYY-MM-DD", lambda table: table.assign(date=["2022/05/01"] + list(table["date"][1:])), "date"),
    ("a date that does not exist", lambda table: table.assign(date=["2022-02-30"] + list(table["date"][1:])), "date"),
    ("label as text", lambda table: table.assign(label=["1"] * len(table)), "label"),
    ("mentions_btc as text", lambda table: table.assign(mentions_btc=["yes"] * len(table)), "mentions_btc"),
    ("time without the UTC time zone", lambda table: table.assign(time=table["time"].dt.tz_localize(None)), "time"),
    ("a date that is not the UTC day of the time", lambda table: table.assign(date=["2022-05-03"] + list(table["date"][1:])),
     "date"),
]
# A second implementation of the two replacements (links, then account names), for scanning the samples.
DATED_LINK = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
DATED_HANDLE = re.compile(r"(?<![A-Za-z0-9_])@[A-Za-z0-9_]+")
DATED_BTC = re.compile(r"\b(?:bitcoin|btc)\b", flags=re.IGNORECASE)
NEWS_FROM_DATASET = {0: 1, 1: 0, 2: 2}  # the news dataset's market_direction -> this project's label


def dated_mask(text):
    return DATED_HANDLE.sub("@user", DATED_LINK.sub("http", "" if text is None else str(text)))


def check_dated_switches(report, defaults, others):
    """The nine switches of chapter 12 with their defaults, the four two-state ones, and the switch check cell."""
    wrong = [f"{name}={defaults.get(name, 'missing')!r}" for name, value in DATED_SWITCH_DEFAULTS.items()
             if name not in defaults or defaults[name] != value]
    two_state = {name: others[name] for name in DATED_SWITCH_DEFAULTS if name in others}
    report.check(not wrong and two_state == DATED_OTHER_VALUES,
                 "switches of the dated data chapter are present with their defaults (both datasets on Hugging Face at "
                 "their pinned revisions, from 2021-01-01 to the end, at most 2000 posts per layer and day, Bitcoin only, "
                 "English tweets, news headlines, the dataset's news label); DATED_BTC_ONLY, DATED_LANGUAGE, NEWS_TEXT and "
                 "NEWS_LABEL are two-state (other values False, None, \"text\", \"votes\")",
                 f"other defaults {wrong}; two-state {two_state}")
    wrong = []
    for overrides, accepted in DATED_SWITCH_PROBES:
        namespace = {"__name__": "notebook_cells"}
        problems = run_cells(["DATA_SOURCE"], namespace)  # the parameters cell
        namespace.update(overrides)
        problems += run_cells(["EMOTION_CHOICES"], namespace)  # the switch check cell
        stopped = [problem for problem in problems if problem.startswith("cell defining EMOTION_CHOICES: ValueError")]
        if accepted and problems:
            wrong.append(f"{overrides} refused: {problems[0][:120]}")
        elif not accepted and not (stopped and any(name in stopped[0] for name in overrides)):
            wrong.append(f"{overrides} not refused with its name: {problems[:1]}")
    report.check(not wrong, f"the switch check cell accepts good dated data switches and stops at bad ones with the "
                 f"switch's name ({len(DATED_SWITCH_PROBES)} cases: dates not written YYYY-MM-DD or not existing, an end "
                 "before the start, a cap that is not a positive integer, a wrong true/false, language or news choice, "
                 "an empty source)", "; ".join(wrong[:3]))


def dated_doc_columns():
    """[(column, type, may be empty)] from the column table of docs/dated-posts.md; and problems found."""
    rows, problems, inside = [], [], False
    text = DATED_DOC.read_text(encoding="utf-8") if DATED_DOC.exists() else ""
    for number, line in enumerate(text.splitlines(), start=1):
        if line.strip() == DATED_DOC_HEADER:
            inside = True
            continue
        if inside and not line.startswith("|"):
            inside = False
        if not inside:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(set(cell) <= {"-", " ", ":"} for cell in cells):
            continue
        if len(cells) < 4 or not re.fullmatch(r"`[a-z_]+`", cells[0]) or cells[2] not in ("yes", "no"):
            problems.append(f"line {number}: not a column row")
            continue
        rows.append((cells[0].strip("`"), cells[1], cells[2] == "yes"))
    return rows, problems


def dated_shape_cells():
    """DATED_POST_SHAPE, DATED_LAYERS and dated_table_problems, executed from the notebook source; (namespace, problems)."""
    import pandas as pd
    namespace = {"__name__": "notebook_cells", "pd": pd}
    return namespace, run_cells(["DATED_POST_SHAPE"], namespace)


def dated_probe_table():
    """A small dated post table that fits the shape."""
    import pandas as pd
    return pd.DataFrame({
        "post_id": ["dlt-news:0000000000000001", "dlt-tweets:0000000000000002", "dlt-tweets:0000000000000003"],
        "date": ["2022-05-01", "2022-05-01", "2022-05-02"],
        "time": pd.to_datetime(["2022-05-01 08:00:00", "2022-05-01 23:59:59", None], utc=True),
        "text": ["Bitcoin falls", "@user btc http", ""], "layer": ["media", "retail", "retail"],
        "source": ["example.com", "twitter", "twitter"], "language": [None, "en", "en"],
        "mentions_btc": [True, True, False], "label": pd.array([0, None, None], dtype="Int64"),
        "votes_bullish": pd.array([1, None, None], dtype="Int64"), "votes_bearish": pd.array([4, None, None], dtype="Int64"),
    })


def check_dated_shape(report):
    """The shape constant and docs/dated-posts.md list the same columns; dated_table_problems rejects broken tables."""
    namespace, problems = dated_shape_cells()
    if not report.check(not problems, "dated data: the shape of the dated post table and its check load from the notebook "
                        "source", "; ".join(problems)):
        return
    shape = [tuple(row) for row in namespace["DATED_POST_SHAPE"]]
    documented, doc_problems = dated_doc_columns()
    doc_text = DATED_DOC.read_text(encoding="utf-8") if DATED_DOC.exists() else ""
    layers_named = all(f"`{layer}`" in doc_text for layer in DATED_LAYER_NAMES)
    report.check(shape == documented and not doc_problems and namespace["DATED_LAYERS"] == DATED_LAYER_NAMES
                 and layers_named,
                 f"{shown(DATED_DOC)} lists exactly the columns of DATED_POST_SHAPE, in the same order, with the same type "
                 f"and the same null ({len(documented)} of {len(shape)} columns), and the layers are media, kol and retail",
                 f"notebook {shape}; document {documented}; {doc_problems}; layers {namespace['DATED_LAYERS']}, named in "
                 f"the document: {layers_named}")
    problems_of = namespace["dated_table_problems"]
    good = dated_probe_table()
    wrong = [] if not problems_of(good) else [f"the valid table is refused: {problems_of(good)}"]
    for description, change, column in DATED_SHAPE_PROBES:
        found = problems_of(change(good.copy()))
        if not any(problem.startswith(f"{column}: ") for problem in found):
            wrong.append(f"{description}: {found or 'not refused'}")
        report.info(f"dated table probe, {description}: {found[:1]}")
    report.check(not wrong, f"dated_table_problems accepts a valid dated post table and refuses each of "
                 f"{len(DATED_SHAPE_PROBES)} broken ones naming the column (a missing or extra column, an empty value, a "
                 "layer whale, a repeated id, a date not YYYY-MM-DD or not existing, a wrong type, a time without UTC, a "
                 "date that is not the day of its time)", "; ".join(wrong[:3]))


def check_dated_chapter(report):
    """Chapter 12 reads both datasets through one function each, which read files only through parquet_files and
    parquet_batches; it has no way of reading meant for the samples, reads the tweets in batches of three columns,
    imports no network package and never loads the datasets library."""
    chapters = chapter_cells()
    code = "\n".join(cell.source for cell in chapters.get(12, []) if cell.cell_type == "code")
    tree = ast.parse(code)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    names = [ast.unparse(node.func) for node in calls]
    readers = sorted(ast.unparse(node) for node in calls
                     if ast.unparse(node.func) in ("read_dated_tweets", "read_dated_news"))
    reads = sorted({name for name in names if re.search(r"(?:^|\.)(?:read_parquet|read_table|read_csv|read_excel|"
                                                      r"ParquetFile|ParquetDataset|load_dataset|url_to_fs)$", name)})
    imports = sorted({alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
                      for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(node.module or "")])})
    network = [name for name in imports if name.split(".")[0] in ("requests", "http", "socket", "httpx", "urllib3",
                                                                  "aiohttp", "datasets")
               or name.startswith("urllib.request")]
    samples = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)
               and "sample" in node.value.lower()]
    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    tweet_reader = ast.unparse(functions["read_dated_tweets"]) if "read_dated_tweets" in functions else ""
    columns = re.search(r'TWEET_COLUMNS = (\[[^\]]*\])', code)
    report.check(readers == ["read_dated_news(DATED_NEWS_SOURCE)", "read_dated_tweets(DATED_TWEETS_SOURCE, WORK_DIR)"]
                 and reads == ["fsspec.core.url_to_fs", "pq.ParquetFile", "pq.read_table"]
                 and not network and not samples
                 and "parquet_batches(fs, file, TWEET_COLUMNS, DATED_BATCH_ROWS)" in tweet_reader
                 and columns is not None and ast.literal_eval(columns.group(1)) == ["timestamp", "tweet", "language"],
                 "chapter 12 reads the tweets only through read_dated_tweets(DATED_TWEETS_SOURCE, WORK_DIR) and the news "
                 "only through read_dated_news(DATED_NEWS_SOURCE), both through parquet_files and parquet_batches (the "
                 "cache is read with pq.read_table), with no way of reading meant for the samples; the tweets are read "
                 "in batches of three columns (timestamp, tweet, language); no network package and no datasets library",
                 f"reader calls {readers}; reading calls {reads}; network imports {network}; strings naming a sample "
                 f"{samples}; tweet columns {columns.group(1) if columns else None}")


def dated_notebook_cells(namespace_extra):
    """Chapter 12's code cells up to the tweets, executed from the source with the given switches; (namespace, problems)."""
    import matplotlib
    matplotlib.use("Agg")
    import pandas as pd
    namespace = {"__name__": "notebook_cells", "pd": pd, "re": re, "brief": {},
                 "BTC_PATTERN": DATED_BTC, "RANDOM_SEED": 42, **DATED_SWITCH_DEFAULTS, **namespace_extra}
    problems = run_cells(["DATED_POST_SHAPE", "MASK_LINK_PATTERN", "TIME_KINDS", "post_ids", "parquet_files",
                          "TWEETS_DATASET"], namespace)
    return namespace, problems


def check_dated_probes(report):
    """Chapter 12 on the tweet sample, in data/checks/dated-probe/: its self-checks pass; the same tweets are kept, with
    the same counts, when the sample's rows are shuffled or split into two files read in the other order; a second
    read comes from the cache; a changed file is read again. The samples hold no account name and no link."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    shutil.rmtree(DATED_PROBE_DIR, ignore_errors=True)
    DATED_PROBE_DIR.mkdir(parents=True)
    work = DATED_PROBE_DIR / "work"
    settings = {"DATED_TWEETS_SOURCE": str(REPO_DIR / DATED_TWEETS_SAMPLE), "DATED_DAILY_CAP": DATED_QUICK_CAP,
                "WORK_DIR": str(work)}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        namespace, problems = dated_notebook_cells(settings)
    if not report.check(not problems, "dated probes: chapter 12's cells run from the source on the tweet sample, and their "
                        "self-checks pass (replacements, times, the daily cap in either row order)", "; ".join(problems)):
        return
    examples = namespace["MASK_EXAMPLES"]
    email_kept = any("@" in text and text == expected and re.search(r"[A-Za-z0-9]@", text) for text, expected in examples)
    report.check(email_kept, "the replacement examples include an e-mail address that stays as it is",
                 f"{[text for text, _ in examples]}")
    read = namespace["read_dated_tweets"]
    first_rows, first_info = namespace["tweet_rows"], namespace["tweet_info"]
    table = pq.read_table(REPO_DIR / DATED_TWEETS_SAMPLE)
    order = list(range(table.num_rows))
    random.Random(7).shuffle(order)
    shuffled = DATED_PROBE_DIR / "shuffled" / "part.parquet"
    shuffled.parent.mkdir()
    pq.write_table(table.take(order), shuffled)
    split = DATED_PROBE_DIR / "split"
    split.mkdir()
    half = table.num_rows // 2
    pq.write_table(table.slice(half), split / "a.parquet")  # the second half first in name order
    pq.write_table(table.slice(0, half), split / "b.parquet")
    results = {}
    with contextlib.redirect_stdout(output):
        for name, source in [("shuffled rows", shuffled.parent), ("two files, halves swapped", split)]:
            rows, info = read(str(source), str(work))
            results[name] = (rows, info)
    def same(rows, info):
        return (sorted(rows["post_id"]) == sorted(first_rows["post_id"]) and info == first_info)
    differing = [name for name, (rows, info) in results.items() if not same(rows, info)]
    report.check(not differing and first_info["steps"]["daily_cap"] > 0,
                 "dated probes: shuffling the tweet sample's rows, or splitting it into two files read in the other "
                 "order, keeps exactly the same tweets with the same counts (ids and the daily cap do not depend on the "
                 "position of a row)", f"differ: {differing}; kept {first_info['steps']['daily_cap']}")
    # The cache: a second read of the same source comes from the cache; a changed file is read again.
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        again_rows, again_info = read(str(shuffled.parent), str(work))
    hit = "cache hit" in log.getvalue()
    grown = DATED_PROBE_DIR / "grown" / "part.parquet"
    grown.parent.mkdir()
    pq.write_table(pa.concat_tables([table, table.slice(0, 1).set_column(
        table.schema.get_field_index("tweet"), table.schema.field("tweet"),
        pa.array(["bitcoin probe row"], type=table.schema.field("tweet").type))]), grown)
    log2 = io.StringIO()
    with contextlib.redirect_stdout(log2):
        grown_rows, grown_info = read(str(grown.parent), str(work))
        grown_again_rows, grown_again_info = read(str(grown.parent), str(work))
    report.check(hit and same(again_rows, again_info) and "cache hit" not in log2.getvalue().splitlines()[0]
                 and grown_info["steps"]["read"] == first_info["steps"]["read"] + 1
                 and "cache hit" in log2.getvalue().splitlines()[-1] and grown_again_info == grown_info,
                 "dated probes: reading the same tweets again comes from the cache in WORK_DIR/cache (no file opened); "
                 "a copy with one more row is read again and counts it, and is then cached in turn",
                 f"second read: {log.getvalue().strip()[:160]}; changed copy: {log2.getvalue().strip()[:300]}; rows read "
                 f"{first_info['steps']['read']} -> {grown_info['steps']['read']}")
    # The samples hold no account name (other than @user) and no link, by a second implementation of the rule.
    leftovers = []
    for path, columns in [(DATED_TWEETS_SAMPLE, ["tweet"]), (DATED_NEWS_SAMPLE, ["title", "description", "text"])]:
        frame = pd.read_parquet(REPO_DIR / path, columns=columns)
        for column in columns:
            for text in frame[column].fillna(""):
                if DATED_LINK.search(text) or any(found != "@user" for found in DATED_HANDLE.findall(text)) \
                        or dated_mask(text) != text:
                    leftovers.append(f"{path} {column}: {text[:60]!r}")
    report.check(not leftovers, "the dated samples hold no link and no account name other than @user (texts already "
                 "replaced; replacing again changes nothing)", "; ".join(leftovers[:3]))
    shutil.rmtree(DATED_PROBE_DIR, ignore_errors=True)


def expected_news(path, switches):
    """A second implementation of chapter 12 for the news: the news kept and their market_direction and labels."""
    import pandas as pd
    news = pd.read_parquet(REPO_DIR / path)
    times = pd.to_datetime(news["timestamp"], format="ISO8601", utc=True, errors="coerce")
    news = news.assign(day=times.dt.strftime("%Y-%m-%d"))[times.notna()]
    news = news[news["day"] >= switches["DATED_START"]]
    if switches["DATED_END"] is not None:
        news = news[news["day"] <= switches["DATED_END"]]
    text = news[switches["NEWS_TEXT"]].map(dated_mask)
    if switches["DATED_BTC_ONLY"]:
        news = news[text.map(lambda value: DATED_BTC.search(value) is not None)]
    news = news.assign(post_id=[hashlib.sha256((str(time) + "\x1f" + str(url)).encode("utf-8")).hexdigest()[:16]
                                for time, url in zip(news["timestamp"], news["url"])])
    news = news.drop_duplicates("post_id")  # rows with the same id are the same item (the counts below agree)
    news = news.assign(key=[hashlib.sha256(f"{switches['RANDOM_SEED']}:dlt-news:{post_id}".encode("utf-8")).hexdigest()[:16]
                            for post_id in news["post_id"]])
    news = news.sort_values(["day", "key", "post_id"]).groupby("day").head(switches["DATED_DAILY_CAP"])
    bullish = news["vote_counts"].map(lambda votes: (votes or {}).get("bullish") or 0)
    bearish = news["vote_counts"].map(lambda votes: (votes or {}).get("bearish") or 0)
    if switches["NEWS_LABEL"] == "market_direction":
        labels = news["market_direction"].map(NEWS_FROM_DATASET)
    else:
        labels = [2 if up > down else 0 if down > up else 1 for up, down in zip(bullish, bearish)]
    count = lambda values: {str(label): int(sum(1 for value in values if value == label)) for label in (0, 1, 2)}
    return len(news), count(list(news["market_direction"])), count(list(labels))


def day_after(day):
    from datetime import date, timedelta
    return (date.fromisoformat(day) + timedelta(days=1)).isoformat()


def check_dated_data(report, label, brief, switches, quick_default=False):
    """The dated_data section: the switches as set, both datasets' steps and time kinds, every layer's period and days,
    the daily cap, the Bitcoin rule and the news labels."""
    dated = as_dict(brief.get("dated_data"))
    if not report.check(bool(dated), f"{label}: dated_data section has content"):
        return
    problems = []
    recorded = as_dict(dated.get("switches"))
    expected = {"date_start": switches["DATED_START"], "date_end": switches["DATED_END"],
                "language": switches["DATED_LANGUAGE"], "btc_only": switches["DATED_BTC_ONLY"],
                "daily_cap": switches["DATED_DAILY_CAP"], "seed": switches["RANDOM_SEED"]}
    if recorded != expected:
        problems.append(f"switches {recorded} (expected {expected})")
    cap = switches["DATED_DAILY_CAP"]
    layers = as_dict(dated.get("layers"))
    if set(layers) - set(DATED_LAYER_NAMES):
        problems.append(f"layer keys {sorted(layers)}")
    datasets = as_dict(dated.get("datasets"))
    for key, fixed in DATED_DATASET_RECORDS.items():
        record = as_dict(datasets.get(key))
        source = switches["DATED_TWEETS_SOURCE" if key == "tweets" else "DATED_NEWS_SOURCE"]
        column = "tweet" if key == "tweets" else switches["NEWS_TEXT"]
        wrong = {name: record.get(name) for name, value in {**fixed, "source": source, "text_column": column}.items()
                 if record.get(name) != value}
        if wrong:
            problems.append(f"datasets.{key}: {wrong}")
        steps = record.get("steps") or []
        names = [as_dict(step).get("step") for step in steps]
        counts = [as_dict(step).get("n_posts") for step in steps]
        kinds = as_dict(record.get("time_kinds"))
        if names != DATED_STEP_NAMES or not all(is_count(count) for count in counts):
            problems.append(f"datasets.{key}.steps {names}")
            continue
        step = dict(zip(names, counts))
        if any(later > earlier for earlier, later in zip(counts, counts[1:])):
            problems.append(f"datasets.{key}: a step has more posts than the one before it {counts}")
        if sorted(kinds) != sorted(DATED_TIME_KINDS) or sum(kinds.values()) != step["read"] \
                or step["time_parsed"] != step["read"] - kinds.get("unparseable", -1):
            problems.append(f"datasets.{key}: time kinds {kinds} do not add up to the rows read {step['read']} and "
                            f"the parsed rows {step['time_parsed']}")
        if record.get("n_duplicates") != step["mentions_btc"] - step["unique"]:
            problems.append(f"datasets.{key}: n_duplicates {record.get('n_duplicates')} is not mentions_btc - unique")
        if not switches["DATED_BTC_ONLY"] and step["mentions_btc"] != step["language"]:
            problems.append(f"datasets.{key}: with DATED_BTC_ONLY False the Bitcoin step dropped posts")
        if (key == "news" or switches["DATED_LANGUAGE"] is None) and step["language"] != step["in_dates"]:
            problems.append(f"datasets.{key}: the language step dropped posts although it filters nothing here")
        layer = as_dict(layers.get(fixed["layer"]))
        if step["daily_cap"] != (layer.get("n_posts") if layer else 0):
            problems.append(f"datasets.{key}: daily_cap {step['daily_cap']} is not the {fixed['layer']} layer's posts "
                            f"{layer.get('n_posts')}")
        if quick_default and key == "tweets" and not step["time_parsed"] > step["in_dates"]:
            problems.append("the date step dropped no tweet of the sample (it has tweets from before 2021)")
    news = as_dict(datasets.get("news"))
    direction, labels = as_dict(news.get("market_direction_counts")), as_dict(news.get("label_counts"))
    media_posts = as_dict(layers.get("media")).get("n_posts", 0)
    if news.get("label_rule") != switches["NEWS_LABEL"] or sum(direction.values()) != media_posts \
            or sum(labels.values()) != media_posts:
        problems.append(f"news labels: rule {news.get('label_rule')}, counts {direction} / {labels}, media posts "
                        f"{media_posts}")
    elif switches["NEWS_LABEL"] == "market_direction" and not (
            labels.get("1") == direction.get("0") and labels.get("0") == direction.get("1")
            and labels.get("2") == direction.get("2")):
        problems.append(f"news labels not converted one by one: market_direction {direction}, labels {labels}")
    for name, layer in layers.items():
        layer = as_dict(layer)
        days = layer.get("days") or []
        dates = [as_dict(day).get("date") for day in days]
        posts = [as_dict(day).get("n_posts") for day in days]
        if not days or not all(is_count(n) for n in posts) or not all(isinstance(day, str) for day in dates):
            problems.append(f"layers.{name}: days malformed")
            continue
        try:
            calendar = [dates[0]]
            while calendar[-1] < dates[-1] and len(calendar) <= len(dates):
                calendar.append(day_after(calendar[-1]))
        except ValueError:
            calendar = []
        if dates != calendar or dates[0] != layer.get("date_from") or dates[-1] != layer.get("date_to"):
            problems.append(f"layers.{name}: days are not every calendar day from {layer.get('date_from')} to "
                            f"{layer.get('date_to')} in order")
        if posts[0] == 0 or posts[-1] == 0:
            problems.append(f"layers.{name}: the first or last day has no post")
        if max(posts) > cap:
            problems.append(f"layers.{name}: a day has {max(posts)} posts, more than the cap {cap}")
        if sum(posts) != layer.get("n_posts") or max(posts) != layer.get("max_posts_per_day") \
                or sum(1 for n in posts if n > 0) != layer.get("n_days_with_posts"):
            problems.append(f"layers.{name}: n_posts, max_posts_per_day or n_days_with_posts do not follow from days")
        capped = layer.get("n_days_capped")
        if not is_count(capped) or capped > sum(1 for n in posts if n == cap):
            problems.append(f"layers.{name}: n_days_capped {capped} is more than the days with exactly the cap")
        mentions = layer.get("n_mentions_btc")
        if not is_count(mentions) or mentions > layer.get("n_posts", 0) \
                or (switches["DATED_BTC_ONLY"] and mentions != layer.get("n_posts")):
            problems.append(f"layers.{name}: n_mentions_btc {mentions} for {layer.get('n_posts')} posts")
        top = layer.get("top_sources") or []
        top_posts = [as_dict(item).get("n_posts") for item in top]
        if not top or len(top) > 10 or len(top) != min(10, layer.get("n_sources", 0)) \
                or top_posts != sorted(top_posts, reverse=True) or sum(top_posts) > layer.get("n_posts", 0):
            problems.append(f"layers.{name}: top_sources {top[:3]}")
        if not is_count(layer.get("n_unknown_source")) or layer["n_unknown_source"] > layer.get("n_posts", 0):
            problems.append(f"layers.{name}: n_unknown_source {layer.get('n_unknown_source')}")
        if name == "retail" and (layer.get("n_sources") != 1 or layer.get("n_unknown_source") != 0
                                 or [as_dict(item).get("source") for item in top] != ["twitter"]):
            problems.append("layers.retail: the tweets' only source must be twitter")
    report.check(not problems, f"{label}: dated_data follows the switches; every dataset's steps never grow and end at its "
                 "layer's posts, its time kinds add up to the rows read, its repeated posts are mentions_btc - unique; "
                 "every layer lists every calendar day from its date_from to its date_to, no day above the cap, the days "
                 "add up to its posts; with DATED_BTC_ONLY every post mentions Bitcoin; the news labels are the dataset's "
                 "classes converted one by one (0 -> 1, 1 -> 0, 2 -> 2)", "; ".join(problems[:5]))
    starts = [as_dict(layer).get("date_from") for layer in layers.values()]
    ends = [as_dict(layer).get("date_to") for layer in layers.values()]
    period_ok = (bool(layers) and dated.get("date_from") == min(starts) and dated.get("date_to") == max(ends)
                 and dated["date_from"] >= switches["DATED_START"]
                 and (switches["DATED_END"] is None or dated["date_to"] <= switches["DATED_END"])
                 and dated.get("n_posts") == sum(as_dict(layer).get("n_posts", 0) for layer in layers.values()))
    report.check(period_ok, f"{label}: dated_data's date_from and date_to are the earliest and latest of its layers, "
                 "within DATED_START and DATED_END, and its posts are the layers' posts",
                 f"{dated.get('date_from')} to {dated.get('date_to')}; layers {list(zip(layers, starts, ends))}")
    default_dates = switches["DATED_START"] == DATED_SWITCH_DEFAULTS["DATED_START"] and switches["DATED_END"] is None
    if default_dates:
        report.check(sorted(layers) == ["media", "retail"],
                     f"{label}: with the default dates both the media and the retail layer have posts (kol has no data)",
                     f"layers {sorted(layers)}")
    if quick_default:
        report.check(any(as_dict(layer).get("n_days_capped", 0) > 0 for layer in layers.values()),
                     f"{label}: on the sample the daily cap ({cap}) left posts out on at least one day",
                     f"days capped {[(name, as_dict(layer).get('n_days_capped')) for name, layer in layers.items()]}")
    if switches["DATED_NEWS_SOURCE"] == DATED_NEWS_SAMPLE:
        n_kept, direction_expected, labels_expected = expected_news(DATED_NEWS_SAMPLE, switches)
        report.check(media_posts == n_kept and direction == direction_expected and labels == labels_expected,
                     f"{label}: the news kept, counted again by the check from {DATED_NEWS_SAMPLE} (a second "
                     "implementation of the steps), have the same market_direction counts and the same labels as the "
                     "brief", f"brief {media_posts} posts, market_direction {direction}, labels {labels}; check {n_kept}, "
                     f"{direction_expected}, {labels_expected}")


def compare_dated(report, first, again, other):
    """Two runs with the default switches: identical dated_data. The run with the other switches: more posts in a
    layer, and not more Bitcoin posts than posts."""
    left = as_dict(as_dict(first).get("dated_data"))
    right = as_dict(as_dict(again).get("dated_data"))
    report.check(bool(left) and left == right, "default switches run twice: the two dated_data sections are identical",
                 "" if left == right else "differing parts: " + ", ".join(key for key in left if left.get(key) != right.get(key)))
    flipped = as_dict(as_dict(other).get("dated_data"))
    layers, flipped_layers = as_dict(left.get("layers")), as_dict(flipped.get("layers"))
    more = [name for name in flipped_layers
            if as_dict(flipped_layers[name]).get("n_posts", 0) > as_dict(layers.get(name)).get("n_posts", 0)]
    within = all(as_dict(layer).get("n_mentions_btc", 0) <= as_dict(layer).get("n_posts", 0)
                 for layer in flipped_layers.values())
    report.check(bool(flipped) and bool(more) and within,
                 "other switches (DATED_BTC_ONLY False and the rest): at least one layer has more posts than with the "
                 "defaults, and no layer has more Bitcoin posts than posts",
                 f"layers with more posts: {more}; " + ", ".join(
                     f"{name} {as_dict(flipped_layers[name]).get('n_mentions_btc')}/{as_dict(flipped_layers[name]).get('n_posts')}"
                     for name in flipped_layers))


def check_dated_copies(report, switches):
    """The full check reads the downloaded copies; they must be there."""
    missing = [switches[name] for name in ("DATED_TWEETS_SOURCE", "DATED_NEWS_SOURCE")
               if "://" not in switches[name] and not (REPO_DIR / switches[name]).exists()]
    return report.check(not missing, "dated datasets available locally",
                        f"{missing} not found; run scripts/download_data.sh --dated" if missing else
                        f"{switches['DATED_TWEETS_SOURCE']}, {switches['DATED_NEWS_SOURCE']}")


def print_dated(brief, brief_path):
    """Print chapter 12's numbers (full run)."""
    import statistics
    dated = as_dict(brief.get("dated_data"))
    if not dated:
        return
    print("\nDated data (full run)")
    print(f"  switches:                      {dated.get('switches')}")
    print(f"  whole table:                   {dated.get('n_posts')} posts, {dated.get('date_from')} to {dated.get('date_to')}")
    for key, record in as_dict(dated.get("datasets")).items():
        print(f"  {key} ({record.get('name')}@{record.get('revision')}, layer {record.get('layer')}, text column "
              f"{record.get('text_column')}):")
        print("    posts left after each step:  " + ", ".join(f"{as_dict(step).get('step')} {as_dict(step).get('n_posts')}"
                                                         for step in record.get("steps") or []))
        print(f"    time kinds:                  {record.get('time_kinds')}")
        print(f"    repeated posts dropped:      {record.get('n_duplicates')}")
        if key == "news":
            print(f"    label rule:                  {record.get('label_rule')}; class names found in the file: "
                  f"{record.get('class_names_found')}")
            print(f"    market_direction (dataset):  {record.get('market_direction_counts')}  (0 neutral, 1 bearish, 2 bullish)")
            print(f"    labels (this project):       {record.get('label_counts')}  (0 bearish, 1 neutral, 2 bullish)")
    for name, layer in as_dict(dated.get("layers")).items():
        posts = [as_dict(day).get("n_posts", 0) for day in layer.get("days") or []]
        with_posts = [n for n in posts if n > 0]
        print(f"  layer {name}: {layer.get('date_from')} to {layer.get('date_to')} ({len(posts)} calendar days), "
              f"{layer.get('n_posts')} posts, {layer.get('n_mentions_btc')} mention Bitcoin")
        print(f"    days with posts {layer.get('n_days_with_posts')}, days capped {layer.get('n_days_capped')}, most on "
              f"one day {layer.get('max_posts_per_day')}")
        if posts:
            print(f"    posts per calendar day: min {min(posts)}, median {statistics.median(posts):g}, max {max(posts)}; "
                  f"per day with posts: min {min(with_posts)}, median {statistics.median(with_posts):g}, max {max(with_posts)}")
        print(f"    sources {layer.get('n_sources')}, unknown {layer.get('n_unknown_source')}; top: " + ", ".join(
            f"{as_dict(item).get('source')} {as_dict(item).get('n_posts')}" for item in layer.get("top_sources") or []))
    print(f"  brief file size:               {brief_path.stat().st_size:,} bytes")


# ---------------------------------------------------------------- day by day and new words (chapter 13)
# Every dated post scored and counted per layer and UTC day (the daily section), and the new words of every day (the
# new_words section). The quick check also runs the notebook once on a copy of the tweet sample with made-up words
# added on one day (the spike run): the new words must find them.

DAILY_SWITCH_DEFAULTS = {"DAILY_MIN_POSTS": 100, "NEW_WORD_WINDOW": 7, "NEW_WORD_MIN_COUNT": 10, "NEW_WORD_RATIO": 3.0,
                         "NEW_WORD_MAX_PER_DAY": 5, "EVENT_DAYS": ["2022-05-09", "2022-11-09"], "EVENT_EXAMPLES": 3}
# Locked (only add): switch values the switch check cell must accept (True) or stop at with the switch's name (False).
DAILY_SWITCH_PROBES = [
    ({"DAILY_MIN_POSTS": 1, "NEW_WORD_RATIO": 1, "EVENT_DAYS": []}, True),
    ({"NEW_WORD_RATIO": 0.5, "NEW_WORD_WINDOW": 30, "EVENT_DAYS": ("2023-01-01",)}, True),
    ({"DAILY_MIN_POSTS": 0}, False), ({"DAILY_MIN_POSTS": 2.5}, False), ({"NEW_WORD_WINDOW": True}, False),
    ({"NEW_WORD_MIN_COUNT": -1}, False), ({"NEW_WORD_MAX_PER_DAY": "5"}, False), ({"EVENT_EXAMPLES": 0}, False),
    ({"NEW_WORD_RATIO": 0}, False), ({"NEW_WORD_RATIO": -2.0}, False), ({"NEW_WORD_RATIO": "3"}, False),
    ({"NEW_WORD_RATIO": float("inf")}, False), ({"NEW_WORD_RATIO": float("nan")}, False),
    ({"EVENT_DAYS": "2022-05-09"}, False), ({"EVENT_DAYS": ["2022/05/09"]}, False), ({"EVENT_DAYS": ["2022-02-30"]}, False),
    ({"EVENT_DAYS": ["2022-05-09", "2022-05-09"]}, False), ({"EVENT_DAYS": [20220509]}, False),
]
NEW_WORD_MUST_IGNORE = ["bitcoin", "btc", "user", "http", "not", "no", "nor", "never"]  # besides NLTK's stop words
DAILY_SECTIONS = ["daily", "new_words"]
# The spike run: one more run of the quick check on a copy of the tweet sample with made-up words added.
SPIKE_LABEL = "new word spike run"
SPIKE_DIR = REPO_DIR / "data" / "checks" / "spike-probe"  # the copy of the tweet sample
SPIKE_DAY = "2022-05-20"  # a day of the sample's tweet window, well after the first NEW_WORD_WINDOW days
SPIKE_WORDS = {"sudden": "zorvexa", "rising": "quelbrin", "short": "tavmoru"}  # made-up words, unknown to the cleaning
SPIKE_POSTS = {"sudden": 12, "rising": 12, "short": 9}  # posts of SPIKE_DAY that get the word
SPIKE_BEFORE = 1  # posts per day with the rising word on the NEW_WORD_WINDOW days before SPIKE_DAY
# The settings of the spike run: no daily cap (adding a word changes a post's id, and so what a cap keeps), the default
# new word rules written out (so that -p values cannot move them), no limit per day (so that the made-up words are
# reported whatever else the day has), and a too-few bar of 1 (so that every day with posts has its shares checked).
SPIKE_SETTINGS = {"DATED_DAILY_CAP": 10 ** 9, "NEW_WORD_WINDOW": 7, "NEW_WORD_MIN_COUNT": 10, "NEW_WORD_RATIO": 3.0,
                  "NEW_WORD_MAX_PER_DAY": 1000, "DAILY_MIN_POSTS": 1}


def check_daily_switches(report, defaults, others):
    """The seven switches of chapter 13 with their defaults, none two-state, and the switch check cell."""
    wrong = [f"{name}={defaults.get(name, 'missing')!r}" for name, value in DAILY_SWITCH_DEFAULTS.items()
             if name not in defaults or defaults[name] != value or type(defaults[name]) is not type(value)]
    two_state = sorted(set(DAILY_SWITCH_DEFAULTS) & set(others))
    report.check(not wrong and not two_state,
                 "switches of the day by day chapter are present with their defaults (at least 100 posts for a share; "
                 "new words: 7 days before, at least 10 posts, at least 3 times the past daily mean, at most 5 a day; "
                 "event days 2022-05-09 and 2022-11-09 with 3 example posts per side), none of them two-state",
                 f"other defaults {wrong}; two-state {two_state}")
    wrong = []
    for overrides, accepted in DAILY_SWITCH_PROBES:
        namespace = {"__name__": "notebook_cells"}
        problems = run_cells(["DATA_SOURCE"], namespace)  # the parameters cell
        namespace.update(overrides)
        problems += run_cells(["EMOTION_CHOICES"], namespace)  # the switch check cell
        stopped = [problem for problem in problems if problem.startswith("cell defining EMOTION_CHOICES: ValueError")]
        if accepted and problems:
            wrong.append(f"{overrides} refused: {problems[0][:120]}")
        elif not accepted and not (stopped and any(name in stopped[0] for name in overrides)):
            wrong.append(f"{overrides} not refused with its name: {problems[:1]}")
    report.check(not wrong, f"the switch check cell accepts good day by day switches and stops at bad ones with the "
                 f"switch's name ({len(DAILY_SWITCH_PROBES)} cases: counts that are not positive integers, a ratio that "
                 "is not a number above 0 or is infinite, event days that are not a list of different real days written "
                 "YYYY-MM-DD)", "; ".join(wrong[:3]))


def daily_share_problems(pile, where, min_posts, calibration, checked):
    """The too-few mark and both shares of one pile (posts n, bullish b): mark = n < min_posts; counted = b / n and
    calibrated = the counted share calibrated with the section's rates, both null when marked. checked counts the
    calibrated shares that are not null and were recomputed."""
    problems = []
    n, bullish = pile.get("n_posts"), pile.get("bullish")
    if not is_count(n) or not is_count(bullish) or bullish > n:
        return [f"{where}: posts {n!r} and bullish calls {bullish!r}"]
    too_few = n < min_posts
    if pile.get("too_few_posts") is not too_few:
        problems.append(f"{where}: too_few_posts {pile.get('too_few_posts')!r} with {n} posts and a bar of {min_posts}")
    counted = None if too_few else share_of(bullish, n)
    if not (is_score(pile.get("half_bullish_share")) and same_value(pile.get("half_bullish_share"), counted)):
        problems.append(f"{where}: half_bullish_share {pile.get('half_bullish_share')!r} (expected {counted!r})")
    calibrated = calibrated_from(counted, calibration)
    if not (is_score(pile.get("calibrated_bullish_share")) and same_value(pile.get("calibrated_bullish_share"), calibrated)):
        problems.append(f"{where}: calibrated_bullish_share {pile.get('calibrated_bullish_share')!r} (expected "
                        f"{calibrated if calibrated is not INVALID else 'not computable'!r})")
    elif pile.get("calibrated_bullish_share") is not None:
        checked.append(where)
    return problems


def daily_row_problems(row, where, min_posts, calibration, classes, drawer_ids, checked):
    """Every identity of one row of the daily section (a day or a layer's whole period)."""
    row = as_dict(row)
    n, halves = row.get("n_posts"), counts_of(row.get("half_counts"), HALF_CALLS)
    if not is_count(n) or halves is None or halves["bullish"] + halves["bearish"] != n:
        return [f"{where}: calls at 0.5 {row.get('half_counts')!r} do not add up to {n!r} posts"]
    problems = daily_share_problems({"n_posts": n, "bullish": halves["bullish"], **row}, where, min_posts, calibration,
                                    checked)
    emotions = row.get("emotion_counts")
    if not isinstance(emotions, dict) or list(emotions) != classes or not all(is_count(value) for value in emotions.values()) \
            or sum(emotions.values()) != n:
        problems.append(f"{where}: emotion_counts {emotions!r} are not the classes {classes} adding up to {n} posts")
    any_drawer, no_drawer = row.get("n_in_any_drawer"), row.get("n_in_no_drawer")
    if not (is_count(any_drawer) and is_count(no_drawer) and any_drawer + no_drawer == n):
        problems.append(f"{where}: posts in any drawer {any_drawer!r} + in no drawer {no_drawer!r} are not {n} posts")
    drawers = row.get("drawers")
    if not isinstance(drawers, dict) or list(drawers) != drawer_ids:
        problems.append(f"{where}: drawers {list(drawers) if isinstance(drawers, dict) else drawers!r} are not "
                        f"{drawer_ids}")
        return problems
    for drawer, pile in drawers.items():
        pile = as_dict(pile)
        size, bullish = pile.get("n_posts"), pile.get("n_half_bullish")
        if not (is_count(size) and is_count(bullish) and bullish <= size <= n) or \
                (is_count(any_drawer) and is_count(size) and size > any_drawer):
            problems.append(f"{where}.drawers.{drawer}: {size!r} posts, {bullish!r} bullish, of {n} posts")
            continue
        problems += daily_share_problems({"n_posts": size, "bullish": bullish, **pile}, f"{where}.drawers.{drawer}",
                                         min_posts, calibration, checked)
    return problems


def daily_flat_counts(row):
    """Every count of a daily row, as {name: count}, for adding the days up."""
    row = as_dict(row)
    counts = {"n_posts": row.get("n_posts"), "n_in_any_drawer": row.get("n_in_any_drawer"),
              "n_in_no_drawer": row.get("n_in_no_drawer"),
              **{f"half {key}": value for key, value in as_dict(row.get("half_counts")).items()},
              **{f"emotion {key}": value for key, value in as_dict(row.get("emotion_counts")).items()}}
    for drawer, pile in as_dict(row.get("drawers")).items():
        counts[f"drawer {drawer} posts"] = as_dict(pile).get("n_posts")
        counts[f"drawer {drawer} bullish"] = as_dict(pile).get("n_half_bullish")
    return counts


def dated_days(brief, layer):
    """[(date, posts)] of a layer of dated_data."""
    days = as_dict(as_dict(as_dict(brief.get("dated_data")).get("layers")).get(layer)).get("days") or []
    return [(as_dict(day).get("date"), as_dict(day).get("n_posts")) for day in days]


def section_period_problems(brief, section):
    """A daily section covers the period of dated_data: the same layers, periods and days, in the same order."""
    problems = []
    content, dated = as_dict(brief.get(section)), as_dict(brief.get("dated_data"))
    if (content.get("date_from"), content.get("date_to")) != (dated.get("date_from"), dated.get("date_to")):
        problems.append(f"{section}: period {content.get('date_from')} to {content.get('date_to')}, dated_data "
                        f"{dated.get('date_from')} to {dated.get('date_to')}")
    layers, dated_layers = as_dict(content.get("layers")), as_dict(dated.get("layers"))
    if list(layers) != list(dated_layers):
        problems.append(f"{section}: layers {list(layers)}, dated_data {list(dated_layers)}")
    for name, layer in layers.items():
        layer, dated_layer = as_dict(layer), as_dict(dated_layers.get(name))
        if (layer.get("date_from"), layer.get("date_to")) != (dated_layer.get("date_from"), dated_layer.get("date_to")):
            problems.append(f"{section}.layers.{name}: period {layer.get('date_from')} to {layer.get('date_to')}")
        days = [(as_dict(day).get("date"), as_dict(day).get("n_posts")) for day in layer.get("days") or []]
        if days != dated_days(brief, name):
            problems.append(f"{section}.layers.{name}: the days (dates and posts) are not those of dated_data")
    return problems


def check_daily(report, label, brief, switches):
    """The daily section: every pile's identities, sums over the days, the calibration, the names and the event days."""
    daily = as_dict(brief.get("daily"))
    layers = as_dict(daily.get("layers"))
    if not report.check(bool(daily) and bool(layers), f"{label}: daily section has content"):
        return
    min_posts = switches["DAILY_MIN_POSTS"]
    classes = as_dict(brief.get("emotions")).get("classes") or []
    drawer_ids = [as_dict(item).get("drawer") for item in as_dict(brief.get("drawers")).get("drawers") or []]
    calibration = as_dict(daily.get("calibration"))
    problems, checked = [], []
    if daily.get("min_posts") != min_posts:
        problems.append(f"min_posts {daily.get('min_posts')!r} is not DAILY_MIN_POSTS {min_posts!r}")
    for name, layer in layers.items():
        layer = as_dict(layer)
        where = f"daily.layers.{name}"
        problems += daily_row_problems(layer.get("overall"), f"{where}.overall", min_posts, calibration, classes,
                                       drawer_ids, checked)
        total = {}
        for index, day in enumerate(layer.get("days") or []):
            problems += daily_row_problems(day, f"{where}.days[{index}]", min_posts, calibration, classes, drawer_ids,
                                           checked)
            for key, value in daily_flat_counts(day).items():
                total[key] = total.get(key, 0) + (value if is_count(value) else 0)
        overall = daily_flat_counts(layer.get("overall"))
        if total != overall:
            problems.append(f"{where}: the days do not add up to the whole period in "
                            f"{[key for key in overall if total.get(key) != overall.get(key)][:5]}")
        if overall.get("n_posts") != as_dict(as_dict(as_dict(brief.get("dated_data")).get("layers")).get(name)).get("n_posts"):
            problems.append(f"{where}: {overall.get('n_posts')!r} posts, not those of dated_data")
    report.check(not problems, f"{label}: daily: in every layer's whole period and every day, and in every drawer of "
                 "them, the calls at 0.5 add up to the posts, too_few_posts follows DAILY_MIN_POSTS and then both shares "
                 "are null, the counted share is bullish / posts, the calibrated share is null or in [0, 1] and is the "
                 "counted share calibrated with the section's rates (recomputed), the emotions add up to the posts, any "
                 "drawer + no drawer = posts, every drawer has at most the posts and its bullish calls at most its "
                 "posts; the days add up to the whole period in every count, which has the posts of dated_data",
                 "; ".join(problems[:5]))
    problems = section_period_problems(brief, "daily")
    report.check(not problems, f"{label}: daily covers the period of dated_data: the same period, the same layers with "
                 "the same periods, and every layer's days are those of dated_data, one by one, with the same posts",
                 "; ".join(problems[:5]))
    # The calibration: the rates of the evaluation row chosen by DATED_BTC_ONLY, and the status of chapter 10's rules.
    selected = as_dict(as_dict(brief.get("sentiment")).get("selection")).get("selected")
    subset = "test_mentions_btc" if switches["DATED_BTC_ONLY"] else "all_test"
    row = evaluation_row(brief, selected, subset)
    problems = []
    if as_dict(calibration.get("rates_from")) != {"model": selected, "subset": subset} or row is None:
        problems.append(f"rates_from {calibration.get('rates_from')!r} (expected {selected} / {subset}, row found: "
                        f"{row is not None})")
    else:
        per_class = as_dict(row.get("per_class"))
        supports = {label_: as_dict(per_class.get(label_)).get("support") for label_ in ("0", "2")}
        if calibration.get("rate_posts") != supports:
            problems.append(f"rate_posts {calibration.get('rate_posts')!r} (the row's supports {supports})")
        if switches["N_CLASSES"] == 3:
            expected = {"true_bullish_rate": None, "false_bullish_rate": None, "rate_gap": None}
            status = "three_classes"
        else:
            true_rate = as_dict(per_class.get("2")).get("recall")
            bearish = as_dict(per_class.get("0")).get("recall")
            false_rate = None if bearish is None else 1.0 - bearish
            gap = None if true_rate is None or false_rate is None else true_rate - false_rate
            expected = {"true_bullish_rate": true_rate, "false_bullish_rate": false_rate, "rate_gap": gap}
            status = ("rates_missing" if gap is None else "rate_gap_too_small"
                      if gap < switches["MIN_RATE_GAP"] or gap <= 0 else "applied")
        for key, value in expected.items():
            if not same_value(calibration.get(key), value):
                problems.append(f"{key} {calibration.get(key)!r} (expected {value!r})")
        if calibration.get("status") != status:
            problems.append(f"status {calibration.get('status')!r} (expected {status})")
    if calibration.get("min_rate_gap") != switches["MIN_RATE_GAP"]:
        problems.append(f"min_rate_gap {calibration.get('min_rate_gap')!r}")
    report.check(not problems, f"{label}: daily.calibration reads the chosen model's row of the evaluation table for the "
                 f"test posts that match DATED_BTC_ONLY ({subset}): its supports, the bullish recall, 1 - the bearish "
                 "recall and their gap (all null with three classes), and the status of chapter 10's rules; "
                 f"{len(checked)} calibrated shares were not null and were recomputed (status "
                 f"{calibration.get('status')})", "; ".join(problems[:5]))
    scoring = as_dict(daily.get("scoring"))
    names_ok = (daily.get("emotion_classes") == classes and daily.get("drawer_ids") == drawer_ids
                and daily.get("word_list_version") == as_dict(brief.get("drawers")).get("word_list_version")
                and scoring.get("sentiment_model") == selected
                and scoring.get("emotion_model") == as_dict(as_dict(brief.get("emotions")).get("selection")).get("selected")
                and scoring.get("n_scored") == as_dict(brief.get("dated_data")).get("n_posts")
                and is_count(scoring.get("n_distinct_clean_texts")) and scoring["n_distinct_clean_texts"] <= scoring["n_scored"])
    report.check(names_ok, f"{label}: daily names the emotions the network actually learned (emotions.classes, in "
                 "order) and the drawers of the word list (in order, same version), the chosen sentiment model and "
                 "emotion network, and scored every dated post",
                 f"emotions {daily.get('emotion_classes')} vs {classes}; drawers {daily.get('drawer_ids')} vs {drawer_ids}; "
                 f"scoring {scoring}")
    check_events(report, label, brief, switches)


def check_events(report, label, brief, switches):
    """The event days: one item per day of EVENT_DAYS; a day outside a layer's period is marked; examples on the right
    side of their threshold, most confident first, every text once, at most EVENT_EXAMPLES; that day's new words."""
    daily = as_dict(brief.get("daily"))
    events = as_dict(daily.get("events"))
    layers, word_layers = as_dict(daily.get("layers")), as_dict(as_dict(brief.get("new_words")).get("layers"))
    problems = []
    expected = {"days": list(switches["EVENT_DAYS"]), "examples_per_side": switches["EVENT_EXAMPLES"],
                "bullish_at_or_above": switches["BULLISH_THRESHOLD"], "bearish_at_or_below": switches["BEARISH_THRESHOLD"]}
    wrong = {key: events.get(key) for key, value in expected.items() if events.get(key) != value}
    if wrong:
        problems.append(f"events {wrong} (switches {expected})")
    items = events.get("items") or []
    if [as_dict(item).get("date") for item in items] != list(switches["EVENT_DAYS"]):
        problems.append(f"event items {[as_dict(item).get('date') for item in items]}")
    in_period = outside = 0
    for item in items:
        item = as_dict(item)
        day = item.get("date")
        if list(as_dict(item.get("layers"))) != list(layers):
            problems.append(f"{day}: layers {list(as_dict(item.get('layers')))}")
        for name, entry in as_dict(item.get("layers")).items():
            entry, layer = as_dict(entry), as_dict(layers.get(name))
            inside = isinstance(day, str) and layer.get("date_from", "") <= day <= layer.get("date_to", "")
            if entry.get("in_period") is not inside:
                problems.append(f"{day} {name}: in_period {entry.get('in_period')!r} (expected {inside})")
                continue
            if not inside:
                outside += 1
                if (entry.get("n_posts"), entry.get("bullish"), entry.get("bearish"), entry.get("new_words")) != (0, [], [], []):
                    problems.append(f"{day} {name}: outside the period but with posts, examples or words")
                continue
            in_period += 1
            posts = next((as_dict(row).get("n_posts") for row in layer.get("days") or [] if as_dict(row).get("date") == day),
                         None)
            words = next(([as_dict(word).get("word") for word in as_dict(row).get("new_words") or []]
                          for row in as_dict(word_layers.get(name)).get("days") or [] if as_dict(row).get("date") == day), None)
            if entry.get("n_posts") != posts or entry.get("new_words") != words:
                problems.append(f"{day} {name}: posts {entry.get('n_posts')!r} / {posts!r}, new words "
                                f"{entry.get('new_words')!r} / {words!r}")
            for side, keep, reverse in [("bullish", lambda p: p >= switches["BULLISH_THRESHOLD"], True),
                                        ("bearish", lambda p: p <= switches["BEARISH_THRESHOLD"], False)]:
                quotes = entry.get(side) if isinstance(entry.get(side), list) else None
                if quotes is None or len(quotes) > switches["EVENT_EXAMPLES"]:
                    problems.append(f"{day} {name}: {side} examples {quotes!r}")
                    continue
                probabilities = [as_dict(quote).get("bullish_probability") for quote in quotes]
                texts = [as_dict(quote).get("text") for quote in quotes]
                if not all(is_score(value) and value is not None and keep(value) for value in probabilities) \
                        or probabilities != sorted(probabilities, reverse=reverse) or len(set(texts)) != len(texts) \
                        or not all(isinstance(as_dict(quote).get("post_id"), str) for quote in quotes):
                    problems.append(f"{day} {name}: {side} examples {probabilities}")
    report.check(not problems, f"{label}: daily.events has one item per day of EVENT_DAYS, in order, with every layer; "
                 "a day outside a layer's period is marked there (no posts, examples or words), a day inside has that "
                 "day's posts and new words, at most EVENT_EXAMPLES bullish examples at or above BULLISH_THRESHOLD "
                 "(highest first) and bearish ones at or below BEARISH_THRESHOLD (lowest first), every text once "
                 f"({in_period} layer-days in the period, {outside} outside)", "; ".join(problems[:5]))


def new_word_key(item):
    """The order of a day's new words: most posts, then the higher ratio (null highest), then the word."""
    ratio = item.get("ratio")
    return (-item.get("n_posts", 0), 0 if ratio is None else 1, -(ratio or 0.0), item.get("word", ""))


def check_new_words(report, label, brief, switches):
    """The new_words section: its switches, its period and days, and every reported word against the rules."""
    section = as_dict(brief.get("new_words"))
    layers = as_dict(section.get("layers"))
    if not report.check(bool(section) and bool(layers), f"{label}: new_words section has content"):
        return
    problems = []
    expected = {"window_days": switches["NEW_WORD_WINDOW"], "min_count": switches["NEW_WORD_MIN_COUNT"],
                "ratio": switches["NEW_WORD_RATIO"], "max_per_day": switches["NEW_WORD_MAX_PER_DAY"]}
    wrong = {key: section.get(key) for key, value in expected.items() if section.get(key) != value}
    if wrong:
        problems.append(f"switches {wrong} (expected {expected})")
    ignored = section.get("ignored_words") if isinstance(section.get("ignored_words"), list) else []
    try:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words("english"))
    except LookupError:  # NLTK's data is missing: the notebook would have downloaded it
        stop_words = set()
    missing = sorted((set(NEW_WORD_MUST_IGNORE) | stop_words) - set(ignored))
    if missing or not stop_words:
        problems.append(f"ignored_words lacks {missing[:10]} (NLTK stop words found: {len(stop_words)})")
    quotes_limit = section.get("quotes_per_word")
    if not (is_count(quotes_limit) and 1 <= quotes_limit <= 3):
        problems.append(f"quotes_per_word {quotes_limit!r}")
        quotes_limit = 3
    window, minimum, ratio_bar = expected["window_days"], expected["min_count"], expected["ratio"]
    n_words = 0
    for name, layer in layers.items():
        layer = as_dict(layer)
        days = layer.get("days") or []
        judged = [as_dict(day).get("judged") for day in days]
        if judged != [index >= window for index in range(len(days))]:
            problems.append(f"layers.{name}: the days judged are not every day but the first {window}")
        counted = {"n_days_judged": sum(value is True for value in judged),
                   "n_days_with_new_words": sum(bool(as_dict(day).get("new_words")) for day in days),
                   "n_new_words": sum(len(as_dict(day).get("new_words") or []) for day in days)}
        if any(layer.get(key) != value for key, value in counted.items()):
            problems.append(f"layers.{name}: counts { {key: layer.get(key) for key in counted} } (from the days {counted})")
        for day in days:
            day = as_dict(day)
            where = f"layers.{name} {day.get('date')}"
            words = day.get("new_words") if isinstance(day.get("new_words"), list) else None
            if words is None or (not day.get("judged") and words) or len(words) > expected["max_per_day"]:
                problems.append(f"{where}: {len(words) if words is not None else words!r} new words (judged "
                                f"{day.get('judged')!r}, at most {expected['max_per_day']})")
                continue
            n_words += len(words)
            if [as_dict(item).get("word") for item in words] != list(dict.fromkeys(as_dict(item).get("word") for item in words)) \
                    or [as_dict(item) for item in words] != sorted((as_dict(item) for item in words), key=new_word_key):
                problems.append(f"{where}: new words repeated or not in the order of the rule")
            for item in words:
                item = as_dict(item)
                word, count, mean, ratio = item.get("word"), item.get("n_posts"), item.get("past_daily_mean"), item.get("ratio")
                if not isinstance(word, str) or word in ignored or len(word) < 2 or not any(ch.isalpha() for ch in word):
                    problems.append(f"{where}: {word!r} may not be a new word")
                if not is_count(count) or count < minimum or not is_count(day.get("n_posts")) or count > day["n_posts"]:
                    problems.append(f"{where} {word}: {count!r} posts (at least {minimum}, at most the day's "
                                    f"{day.get('n_posts')!r})")
                    continue
                if not is_number(mean) or mean < 0 or abs(mean * window - round(mean * window)) > 1e-6:
                    problems.append(f"{where} {word}: past daily mean {mean!r} is not a count of posts / {window}")
                elif mean == 0:
                    if ratio is not None:
                        problems.append(f"{where} {word}: a past daily mean of 0 needs a ratio of null, not {ratio!r}")
                elif not (is_number(ratio) and same_value(ratio, count / mean) and ratio >= ratio_bar * (1 - 1e-9)):
                    problems.append(f"{where} {word}: ratio {ratio!r} (posts / past daily mean = {count / mean}, at least "
                                    f"{ratio_bar})")
                quotes = item.get("quotes") if isinstance(item.get("quotes"), list) else []
                texts = [as_dict(quote).get("text") for quote in quotes]
                ids = [as_dict(quote).get("post_id") for quote in quotes]
                if not 1 <= len(quotes) <= quotes_limit or len(set(texts)) != len(texts) \
                        or not all(isinstance(text, str) for text in texts) \
                        or not all(isinstance(post_id, str) for post_id in ids) or ids != sorted(ids):
                    problems.append(f"{where} {word}: {len(quotes)} example posts (1 to {quotes_limit}, different texts, "
                                    "in post_id order)")
    report.check(not problems, f"{label}: new_words follows its switches and ignores at least NLTK's stop words and "
                 f"{', '.join(NEW_WORD_MUST_IGNORE)}; in every layer the first NEW_WORD_WINDOW days are not judged and "
                 "have no new words; every new word is a word with a letter and two characters or more, outside the "
                 "ignored words, in at least NEW_WORD_MIN_COUNT posts, with a ratio of null when its past daily mean is "
                 "0 and otherwise posts / past daily mean at least NEW_WORD_RATIO, at most NEW_WORD_MAX_PER_DAY a day in "
                 f"the order of the rule, with 1 to 3 example posts of different texts ({n_words} new words checked)",
                 "; ".join(problems[:5]))
    problems = section_period_problems(brief, "new_words")
    report.check(not problems, f"{label}: new_words covers the period of dated_data: the same period, the same layers "
                 "with the same periods, and every layer's days are those of dated_data, one by one, with the same posts",
                 "; ".join(problems[:5]))


DAILY_STABLE_ROW_FIELDS = ["n_posts", "too_few_posts", "n_in_any_drawer", "n_in_no_drawer"]  # do not depend on training


def stable_daily(brief):
    """The parts of the daily section that do not depend on training, or None when there is no daily section."""
    daily = as_dict(brief).get("daily") if isinstance(brief, dict) else None
    if not isinstance(daily, dict) or not daily:
        return None

    def part(row):
        row = as_dict(row)
        return {**{key: row.get(key) for key in ["date"] + DAILY_STABLE_ROW_FIELDS if key in row},
                "drawers": {drawer: {key: as_dict(pile).get(key) for key in ("n_posts", "too_few_posts")}
                            for drawer, pile in as_dict(row.get("drawers")).items()}}

    calibration = as_dict(daily.get("calibration"))
    return {
        "settings": {key: daily.get(key) for key in ("date_from", "date_to", "min_posts", "emotion_classes", "drawer_ids",
                                                     "word_list_version")},
        "scoring": {key: as_dict(daily.get("scoring")).get(key) for key in ("n_scored", "n_distinct_clean_texts")},
        "calibration": {"subset": as_dict(calibration.get("rates_from")).get("subset"),
                        "rate_posts": calibration.get("rate_posts")},
        "layers": {name: {"period": (as_dict(layer).get("date_from"), as_dict(layer).get("date_to")),
                          "overall": part(as_dict(layer).get("overall")),
                          "days": [part(day) for day in as_dict(layer).get("days") or []]}
                   for name, layer in as_dict(daily.get("layers")).items()},
        "events": [{"date": as_dict(item).get("date"),
                    "layers": {name: {key: as_dict(entry).get(key) for key in ("in_period", "n_posts", "new_words")}
                               for name, entry in as_dict(as_dict(item).get("layers")).items()}}
                   for item in as_dict(daily.get("events")).get("items") or []],
    }


def compare_daily(report, first, again):
    """Two runs with the default switches: identical new_words sections, and the same daily counts that do not depend
    on training (posts, drawer posts, any and no drawer, too-few marks, the event days in and out of the periods)."""
    left, right = as_dict(as_dict(first).get("new_words")), as_dict(as_dict(again).get("new_words"))
    report.check(bool(left) and left == right, "default switches run twice: the two new_words sections are identical",
                 "" if left == right else "differing parts: " + ", ".join(key for key in left if left.get(key) != right.get(key)))
    left, right = stable_daily(first), stable_daily(again)
    report.check(left is not None and left == right,
                 "default switches run twice: the daily sections agree on everything that does not depend on training "
                 "(every layer's and day's posts, posts in every drawer, in any drawer and in none, the too-few marks, the "
                 "calibration's test subset and test posts, the event days in and out of the periods, their posts and "
                 "new words)", "" if left == right or left is None or right is None else
                 "differing parts: " + ", ".join(key for key in left if left[key] != right[key]))


def make_spike_copy():
    """A copy of the tweet sample with made-up words added: on SPIKE_DAY, SPIKE_POSTS posts get each word; the rising
    word is also in SPIKE_BEFORE post a day on the NEW_WORD_WINDOW days before. Returns (path, what was added)."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datetime import date, timedelta

    table = pq.read_table(REPO_DIR / DATED_TWEETS_SAMPLE)
    frame = table.to_pandas()
    times = pd.to_datetime(frame["timestamp"], format="ISO8601", utc=True, errors="coerce")
    days = times.dt.strftime("%Y-%m-%d")
    texts = frame["tweet"].map(dated_mask)
    kept = (times.notna() & (frame["language"] == "en") & (days >= DATED_SWITCH_DEFAULTS["DATED_START"])
            & texts.map(lambda text: DATED_BTC.search(text) is not None)
            & ~frame.duplicated(["timestamp", "tweet"], keep=False))  # posts the notebook keeps, each once

    def rows_on(day):
        return list(frame.index[kept & (days == day)])

    tweets = frame["tweet"].astype(object).copy()
    added = {key: {} for key in SPIKE_WORDS}
    on_day = rows_on(SPIKE_DAY)
    start = 0
    for key, word in SPIKE_WORDS.items():
        chosen = on_day[start:start + SPIKE_POSTS[key]]
        if len(chosen) != SPIKE_POSTS[key]:
            raise ValueError(f"the tweet sample has too few posts on {SPIKE_DAY} for the spike run")
        for row in chosen:
            tweets[row] = f"{tweets[row]} {word}"
        added[key][SPIKE_DAY] = len(chosen)
        start += SPIKE_POSTS[key]
    for offset in range(1, SPIKE_SETTINGS["NEW_WORD_WINDOW"] + 1):
        day = (date.fromisoformat(SPIKE_DAY) - timedelta(days=offset)).isoformat()
        chosen = rows_on(day)[:SPIKE_BEFORE]
        if len(chosen) != SPIKE_BEFORE:
            raise ValueError(f"the tweet sample has too few posts on {day} for the spike run")
        for row in chosen:
            tweets[row] = f"{tweets[row]} {SPIKE_WORDS['rising']}"
        added["rising"][day] = len(chosen)
    shutil.rmtree(SPIKE_DIR, ignore_errors=True)
    SPIKE_DIR.mkdir(parents=True)
    path = SPIKE_DIR / "tweets.parquet"
    position = table.schema.get_field_index("tweet")
    pq.write_table(table.set_column(position, table.schema.field("tweet"),
                                    pa.array(tweets.tolist(), type=table.schema.field("tweet").type)), path)
    return path, added


def spike_run(report, defaults, overrides, rules=None, built_in=None):
    """The notebook on a copy of the tweet sample with made-up words: a word new on SPIKE_DAY (never seen before), a
    word rising on SPIKE_DAY (a little on the days before), and a word one post short of NEW_WORD_MIN_COUNT. The first
    two must be reported on SPIKE_DAY with example posts that contain them, the third never; the first on no other day,
    the second on no day before."""
    try:
        path, added = make_spike_copy()
    except ValueError as exc:
        report.check(False, f"{SPIKE_LABEL}: copy of the tweet sample with made-up words", str(exc))
        return
    report.info(f"{SPIKE_LABEL}: words added to {shown(path)} (word: posts per day): "
                + "; ".join(f"{SPIKE_WORDS[key]}: {added[key]}" for key in SPIKE_WORDS))
    parameters = {"DATA_SOURCE": SAMPLE_SOURCE, "EMOTION_LABELS_SOURCE": PRELABEL_SAMPLE_FILE, **DATED_SAMPLE_SETTINGS,
                  **VALIDATION_SAMPLE_SETTINGS, "WORK_DIR": "data/checks/quick-spike", **overrides, **SPIKE_SETTINGS,
                  "DATED_TWEETS_SOURCE": shown(path)}
    parameters = panel_settings(report, SPIKE_LABEL, parameters, overrides)
    switches = {**defaults, **parameters}
    report.info(f"{SPIKE_LABEL}: " + ", ".join(f"{k}={v!r}" for k, v in parameters.items()))
    kept = [name for name in list(SPIKE_SETTINGS) + ["DATED_TWEETS_SOURCE"] if name in overrides]
    if kept:
        report.info(f"{SPIKE_LABEL}: keeps its own values of {', '.join(kept)} instead of the -p values")
    check_offline_sources(report, SPIKE_LABEL, parameters, defaults)
    brief_path = REPO_DIR / switches["WORK_DIR"] / "brief.json"
    brief_path.unlink(missing_ok=True)  # a stale brief must never pass
    ok, _, _ = run_notebook(report, SPIKE_LABEL, parameters, REPO_DIR / switches["WORK_DIR"] / "executed.ipynb")
    brief = read_brief(report, SPIKE_LABEL, brief_path) if ok else None
    if brief is None:
        return
    check_dated_data(report, SPIKE_LABEL, brief, switches)
    check_daily(report, SPIKE_LABEL, brief, switches)
    check_new_words(report, SPIKE_LABEL, brief, switches)
    check_layers(report, SPIKE_LABEL, brief, switches)
    index_coefficients, change_coefficients = check_validation(report, SPIKE_LABEL, brief, switches)
    report.check(index_coefficients > 0 and change_coefficients > 0,
                 f"{SPIKE_LABEL}: with both small files of chapter 15 and a too-few bar of 1, coefficients can be computed: "
                 f"{index_coefficients} against the index and {change_coefficients} against the next day's change are not "
                 "null and agree with the check's own computation")
    check_format(report, SPIKE_LABEL, brief, rules)
    reported = {}  # word -> {day: (posts, ratio, example texts)}
    for name, layer in as_dict(as_dict(brief.get("new_words")).get("layers")).items():
        for day in as_dict(layer).get("days") or []:
            for item in as_dict(day).get("new_words") or []:
                item = as_dict(item)
                if item.get("word") in SPIKE_WORDS.values():
                    reported.setdefault(item["word"], {})[f"{name} {as_dict(day).get('date')}"] = (
                        item.get("n_posts"), item.get("ratio"), [as_dict(quote).get("text") for quote in item.get("quotes") or []])
    sudden, rising, short = (SPIKE_WORDS[key] for key in ("sudden", "rising", "short"))
    on_day = f"retail {SPIKE_DAY}"
    found = {word: reported.get(word, {}) for word in SPIKE_WORDS.values()}
    sudden_ok = list(found[sudden]) == [on_day] and found[sudden][on_day][1] is None \
        and all(sudden in text for text in found[sudden][on_day][2])
    rising_ok = on_day in found[rising] and all(key >= on_day for key in found[rising] if key.startswith("retail ")) \
        and all(not key.startswith("media ") for key in found[rising]) \
        and all(rising in text for text in found[rising][on_day][2])
    report.check(sudden_ok and rising_ok and not found[short],
                 f"{SPIKE_LABEL}: on {SPIKE_DAY} the new words of the retail layer contain {sudden} (in "
                 f"{SPIKE_POSTS['sudden']} posts, never before: ratio null) and {rising} (in {SPIKE_POSTS['rising']} posts, "
                 f"{SPIKE_BEFORE} a day before), each with example posts that contain it; {sudden} on no other day, "
                 f"{rising} on no day before, and {short} (in {SPIKE_POSTS['short']} posts, one below "
                 f"NEW_WORD_MIN_COUNT {SPIKE_SETTINGS['NEW_WORD_MIN_COUNT']}) on no day",
                 "; ".join(f"{word}: " + (", ".join(f"{key} posts {value[0]} ratio {value[1]}" for key, value in days.items())
                                          or "not reported") for word, days in found.items()))
    shutil.rmtree(SPIKE_DIR, ignore_errors=True)


def daily_lines(executed):
    """The [daily] lines of an executed notebook (how long the steps of chapter 13 took)."""
    lines = []
    if executed.exists():
        for cell in json.loads(executed.read_text(encoding="utf-8"))["cells"]:
            for out in cell.get("outputs", []):
                text = out.get("text", "")
                lines += [line for line in ("".join(text) if isinstance(text, list) else text).splitlines()
                          if line.startswith("[daily]")]
    return lines


def print_daily(brief):
    """Print chapter 13's numbers (full run): shares per day, too-few days, the rates, emotions, drawers, event days."""
    import statistics
    daily = as_dict(brief.get("daily"))
    if not daily:
        return
    calibration = as_dict(daily.get("calibration"))
    print("\nDay by day (full run)")
    print(f"  scoring:                       {daily.get('scoring')}")
    print(f"  too few below:                 {daily.get('min_posts')} posts")
    print(f"  calibration:                   status {calibration.get('status')}; rates from "
          f"{as_dict(calibration.get('rates_from')).get('model')} / {as_dict(calibration.get('rates_from')).get('subset')} "
          f"(test posts {calibration.get('rate_posts')}); true bullish rate {calibration.get('true_bullish_rate')}, false "
          f"bullish rate {calibration.get('false_bullish_rate')}, gap {calibration.get('rate_gap')}")
    classes = daily.get("emotion_classes") or []
    for name, layer in as_dict(daily.get("layers")).items():
        days = layer.get("days") or []
        print(f"  layer {name}: {layer.get('date_from')} to {layer.get('date_to')} ({len(days)} days)")
        for field, title in [("half_bullish_share", "counted"), ("calibrated_bullish_share", "calibrated")]:
            values = [day[field] for day in days if day.get(field) is not None]
            low = min(values, default=None)
            high = max(values, default=None)
            print(f"    {title:10} share: {len(values)} days with a share; lowest {low} on "
                  f"{[day['date'] for day in days if day.get(field) == low][:3] if values else '-'}, median "
                  f"{statistics.median(values) if values else None}, highest {high} on "
                  f"{[day['date'] for day in days if day.get(field) == high][:3] if values else '-'}")
        print(f"    days too few: {sum(bool(day.get('too_few_posts')) for day in days)} (of them without posts: "
              f"{sum(day.get('n_posts') == 0 for day in days)})")
        overall = as_dict(layer.get("overall"))
        print(f"    whole period: {overall.get('n_posts')} posts, at 0.5 {overall.get('half_counts')}, counted "
              f"{overall.get('half_bullish_share')}, calibrated {overall.get('calibrated_bullish_share')}")
        n = overall.get("n_posts") or 0
        print("    emotions over the whole period: " + ", ".join(
            f"{label} {as_dict(overall.get('emotion_counts')).get(label)} ({as_dict(overall.get('emotion_counts')).get(label, 0) / n:.3f})"
            for label in classes) if n else "    emotions: no posts")
        for label in classes:
            shares = [day["emotion_counts"][label] / day["n_posts"] for day in days if day.get("n_posts")]
            print(f"      {label:12} share per day with posts: lowest {min(shares, default=0):.3f}, median "
                  f"{statistics.median(shares) if shares else 0:.3f}, highest {max(shares, default=0):.3f}")
        print(f"    drawers over the whole period (any {overall.get('n_in_any_drawer')}, none {overall.get('n_in_no_drawer')}):")
        for drawer, pile in as_dict(overall.get("drawers")).items():
            shares = [as_dict(day["drawers"]).get(drawer, {}).get("n_posts", 0) / day["n_posts"] for day in days
                      if day.get("n_posts")]
            calibrated = [as_dict(day["drawers"]).get(drawer, {}).get("calibrated_bullish_share") for day in days]
            calibrated = [value for value in calibrated if value is not None]
            print(f"      {drawer:16} {pile.get('n_posts'):>8} posts ({pile.get('n_posts', 0) / n if n else 0:.3f}); "
                  f"counted {pile.get('half_bullish_share')}, calibrated {pile.get('calibrated_bullish_share')}; share of "
                  f"the day's posts: median {statistics.median(shares) if shares else 0:.3f}, highest "
                  f"{max(shares, default=0):.3f}; days with a calibrated share {len(calibrated)}"
                  + (f" (lowest {min(calibrated):.3f}, highest {max(calibrated):.3f})" if calibrated else ""))
    print("  event days:")
    for item in as_dict(daily.get("events")).get("items") or []:
        for name, entry in as_dict(item.get("layers")).items():
            if not entry.get("in_period"):
                print(f"    {item.get('date')} {name}: not in this layer's period")
                continue
            print(f"    {item.get('date')} {name}: {entry.get('n_posts')} posts; new words: "
                  f"{', '.join(entry.get('new_words') or []) or '(none)'}")
            for side in ("bullish", "bearish"):
                for quote in entry.get(side) or []:
                    print(f"      [{side} {quote.get('bullish_probability'):.3f}] {quote.get('text')}")


def print_new_words(brief):
    """Print every new word, one line per layer and day with new words (full run)."""
    section = as_dict(brief.get("new_words"))
    if not section:
        return
    print("\nNew words (full run)")
    print(f"  rules: {section.get('window_days')} days before, at least {section.get('min_count')} posts, at least "
          f"{section.get('ratio')} x the past daily mean, at most {section.get('max_per_day')} a day")
    for name, layer in as_dict(section.get("layers")).items():
        print(f"  layer {name}: {layer.get('n_days_judged')} days judged, {layer.get('n_days_with_new_words')} with new "
              f"words, {layer.get('n_new_words')} new words; days without new words: "
              f"{len(layer.get('days') or []) - layer.get('n_days_with_new_words', 0)}")
    print("  every new word, day by day (date layer: word (posts / past daily mean / ratio)):")
    for name, layer in as_dict(section.get("layers")).items():
        for day in layer.get("days") or []:
            if day.get("new_words"):
                print(f"    {day.get('date')} {name}: " + ", ".join(
                    f"{item['word']} ({item['n_posts']} / {item['past_daily_mean']:.2f} / "
                    f"{'null' if item['ratio'] is None else round(item['ratio'], 2)})" for item in day["new_words"]))


# ---------------------------------------------------------------- two layers and validation (chapters 14 and 15)

VALIDATION_SWITCH_DEFAULTS = {"LAYER_GAP": 0.1, "PRICE_SOURCE": "auto",
                              "FEAR_GREED_SOURCE": "https://api.alternative.me/fng/?limit=0&format=json",
                              "VALIDATION_MIN_DAYS": 30}
# Locked (only add): switch values the switch check cell must accept (True) or stop at with the switch's name (False).
VALIDATION_SWITCH_PROBES = [
    ({"LAYER_GAP": 0.5, "PRICE_SOURCE": "none", "FEAR_GREED_SOURCE": "none", "VALIDATION_MIN_DAYS": 1}, True),
    ({"LAYER_GAP": 0.05, "PRICE_SOURCE": "samples/btc-usd-made-up-closes.csv",
      "FEAR_GREED_SOURCE": "https://api.alternative.me/fng/?limit=10&format=json"}, True),
    ({"LAYER_GAP": 0}, False), ({"LAYER_GAP": -0.1}, False), ({"LAYER_GAP": 0.6}, False), ({"LAYER_GAP": "0.1"}, False),
    ({"LAYER_GAP": True}, False), ({"LAYER_GAP": float("nan")}, False),
    ({"PRICE_SOURCE": ""}, False), ({"PRICE_SOURCE": None}, False), ({"PRICE_SOURCE": "https://stooq.com/q/d/l/?s=btcusd&i=d"}, False),
    ({"FEAR_GREED_SOURCE": ""}, False), ({"FEAR_GREED_SOURCE": "https://example.com/fng/"}, False),
    ({"FEAR_GREED_SOURCE": "http://api.alternative.me/fng/"}, False), ({"FEAR_GREED_SOURCE": 7}, False),
    ({"VALIDATION_MIN_DAYS": 0}, False), ({"VALIDATION_MIN_DAYS": 2.5}, False), ({"VALIDATION_MIN_DAYS": False}, False),
]
PRICE_SAMPLE = "samples/btc-usd-made-up-closes.csv"  # made-up closes, not real prices
FEAR_GREED_SAMPLE = "samples/fear-greed-alternative-me-sample.json"  # a part of the index, data from alternative.me
# Every run of the quick check reads the price and the index from these files (or not at all), so it connects to nothing.
VALIDATION_SAMPLE_SETTINGS = {"PRICE_SOURCE": PRICE_SAMPLE, "FEAR_GREED_SOURCE": FEAR_GREED_SAMPLE}
OTHER_RUN_SOURCES = {"FEAR_GREED_SOURCE": "none"}  # the run with the other switches also takes the index's "none" path
# One more run of the quick check: no prices at all ("not obtained"); a too-few bar of 1 gives the index days to pair.
PRICE_NONE_LABEL = "price not obtained run"
PRICE_NONE_SETTINGS = {"PRICE_SOURCE": "none", "FEAR_GREED_SOURCE": FEAR_GREED_SAMPLE, "DAILY_MIN_POSTS": 1}
LAYER_PAIR = ["media", "retail"]  # the two layers chapter 14 compares, in this order
PRICE_WEB_SOURCES = ["yfinance", "coinbase", "stooq"]  # in the order "auto" tries them
WEB_ADDRESS_PREFIXES = ["https://api.exchange.coinbase.com/", "https://stooq.com/", "https://api.alternative.me/",
                        "https://alternative.me/"]  # the only web addresses chapter 15 may hold
WEB_ADDRESS = re.compile(r"[a-z]+://[^\s)\"']+")  # a web address inside a text
NETWORK_MODULES = ("urllib", "requests", "http", "socket", "httpx", "urllib3", "aiohttp", "yfinance", "curl_cffi")
VALIDATION_HOST_WORDS = ["yfinance", "coinbase", "stooq", "alternative.me", "yahoo"]  # only chapter 15 talks to them
CORRELATION_TOLERANCE = 1e-9
VALIDATION_PROBE_DIR = REPO_DIR / "data" / "checks" / "validation-probe"


def check_validation_switches(report, defaults, others):
    """The four switches of chapters 14 and 15 with their defaults, none two-state, and the switch check cell."""
    wrong = [f"{name}={defaults.get(name, 'missing')!r}" for name, value in VALIDATION_SWITCH_DEFAULTS.items()
             if name not in defaults or defaults[name] != value or type(defaults[name]) is not type(value)]
    two_state = sorted(set(VALIDATION_SWITCH_DEFAULTS) & set(others))
    report.check(not wrong and not two_state,
                 "switches of the two layers and validation chapters are present with their defaults (a lean 0.1 away "
                 "from 0.5; prices \"auto\"; the index from the alternative.me address; at least 30 days for a "
                 "correlation), none of them two-state", f"other defaults {wrong}; two-state {two_state}")
    wrong = []
    for overrides, accepted in VALIDATION_SWITCH_PROBES:
        namespace = {"__name__": "notebook_cells"}
        problems = run_cells(["DATA_SOURCE"], namespace)  # the parameters cell
        namespace.update(overrides)
        problems += run_cells(["EMOTION_CHOICES"], namespace)  # the switch check cell
        stopped = [problem for problem in problems if problem.startswith("cell defining EMOTION_CHOICES: ValueError")]
        if accepted and problems:
            wrong.append(f"{overrides} refused: {problems[0][:120]}")
        elif not accepted and not (stopped and any(name in stopped[0] for name in overrides)):
            wrong.append(f"{overrides} not refused with its name: {problems[:1]}")
    report.check(not wrong, f"the switch check cell accepts good switches of chapters 14 and 15 and stops at bad ones "
                 f"with the switch's name ({len(VALIDATION_SWITCH_PROBES)} cases: a gap that is not a number above 0 and "
                 "at most 0.5, a price source that is empty or a web address, an index source that is empty or an "
                 "address outside https://api.alternative.me/, a day count that is not a positive integer)",
                 "; ".join(wrong[:3]))


def string_constants(tree):
    return [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)]


def imported_modules(tree):
    return sorted({alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
                   for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(node.module or "")])})


def check_validation_chapters(report):
    """Chapter 14 holds no web address and imports no network package; chapter 15 is the only chapter that talks to
    the price sources and alternative.me, only through web_get (urllib) and yfinance, only at their addresses, and
    keeps its caches (yfinance's too) under WORK_DIR."""
    chapters = chapter_cells()
    code14 = "\n".join(cell.source for cell in chapters.get(14, []) if cell.cell_type == "code")
    code15 = "\n".join(cell.source for cell in chapters.get(15, []) if cell.cell_type == "code")
    tree14, tree15 = ast.parse(code14), ast.parse(code15)
    addresses14 = [value for value in string_constants(tree14) if "://" in value]
    network14 = [name for name in imported_modules(tree14) if name.split(".")[0] in NETWORK_MODULES]
    report.check(bool(code14) and not addresses14 and not network14,
                 "chapter 14 (two layers) holds no web address and imports no network package",
                 f"addresses {addresses14}; network imports {network14}")
    addresses15 = [address for value in string_constants(tree15) for address in WEB_ADDRESS.findall(value)]
    outside = [value for value in addresses15 if not any(value.startswith(prefix) for prefix in WEB_ADDRESS_PREFIXES)]
    network15 = [name for name in imported_modules(tree15) if name.split(".")[0] in NETWORK_MODULES]
    functions = {node.name: node for node in ast.walk(tree15) if isinstance(node, ast.FunctionDef)}
    opened_in = sorted({name for name, node in functions.items() for call in ast.walk(node)
                        if isinstance(call, ast.Call) and ast.unparse(call.func).endswith("urlopen")})
    all_opens = [call for call in ast.walk(tree15) if isinstance(call, ast.Call) and ast.unparse(call.func).endswith("urlopen")]
    cache_calls = [ast.unparse(call) for call in ast.walk(tree15) if isinstance(call, ast.Call)
                   and ast.unparse(call.func).endswith("set_tz_cache_location")]
    cache_files = [ast.unparse(node) for name in ("price_cache_file", "fear_greed_index") if name in functions
                   for node in ast.walk(functions[name]) if isinstance(node, ast.BinOp) and "cache" in ast.unparse(node)]
    report.check(bool(addresses15) and not outside and sorted(network15) == ["urllib.error", "urllib.request", "yfinance",
                                                                              "yfinance.data"]
                 and opened_in == ["web_get"] and len(all_opens) == 1
                 and len(cache_calls) == 1 and "WORK_DIR" in cache_calls[0]
                 and cache_files and all("WORK_DIR" in text for text in cache_files),
                 "chapter 15 reads the web only through web_get (its one urlopen, Python's urllib) and yfinance, holds "
                 f"web addresses only under {', '.join(WEB_ADDRESS_PREFIXES)}, and keeps its caches and yfinance's "
                 "cookie and time zone files under WORK_DIR/cache/",
                 f"addresses outside {outside}; network imports {network15}; urlopen in {opened_in} ({len(all_opens)} "
                 f"calls); yfinance cache {cache_calls}; cache paths {cache_files}")
    elsewhere = []
    for number, cells in chapters.items():
        if number == 15:
            continue
        for cell in cells:
            if cell.cell_type != "code" or "BRIEF_SCHEMA_TEXT = " in cell.source:  # the format rules name the sources
                continue
            lowered = cell.source.lower()
            elsewhere += [f"chapter {number}: {word}" for word in VALIDATION_HOST_WORDS if word in lowered]
    report.check(not elsewhere, "no chapter but chapter 15 names yfinance, Yahoo, Coinbase, Stooq or alternative.me in its "
                 "code (the parameters, install and switch check cells come before chapter 0)", "; ".join(elsewhere[:5]))


def check_validation_probes(report):
    """Chapter 15's functions, executed from the source, without any connection (in data/checks/validation-probe/):
    the parsers on replies written here, and the caches with fake sources: a cache that covers the days is read
    without asking any source; one that does not is fetched again and replaced; a source that fails or misses days
    is passed over with its reason; when every source fails the prices are not obtained."""
    import pandas as pd
    shutil.rmtree(VALIDATION_PROBE_DIR, ignore_errors=True)
    namespace = {"__name__": "notebook_cells", "WORK_DIR": str(VALIDATION_PROBE_DIR), "PRICE_SOURCE": "auto",
                 "FEAR_GREED_SOURCE": VALIDATION_SWITCH_DEFAULTS["FEAR_GREED_SOURCE"]}
    problems = run_cells(["btc_closes"], namespace)
    if not report.check(not problems, "chapter 15's function cell runs from the source", "; ".join(problems)):
        return
    not_obtained = namespace["NotObtained"]
    wrong = []

    def refused(function, *args):
        try:
            function(*args)
        except not_obtained as problem:
            return str(problem)
        return None

    candles = json.dumps([[1650153600, 1.0, 2.0, 1.5, 40000.5, 10.0], [1650067200, 1.0, 2.0, 1.5, 39000.25, 9.0]])
    closes = namespace["closes_from_coinbase"]([candles.encode("utf-8")])
    if closes.to_dict() != {"2022-04-16": 39000.25, "2022-04-17": 40000.5}:
        wrong.append(f"Coinbase candles read as {closes.to_dict()}")
    if "not a list" not in (refused(namespace["closes_from_coinbase"], [b'{"message": "Unauthorized"}']) or ""):
        wrong.append("a Coinbase error reply was not refused")
    page = "<!DOCTYPE html><html><noscript>This site requires JavaScript to verify your browser.</noscript></html>"
    if "web page" not in (refused(namespace["closes_from_stooq"], page) or ""):
        wrong.append("a Stooq web page was not refused")
    if "key" not in (refused(namespace["closes_from_stooq"], "Get your apikey: https://stooq.com/q/d/?s=btcusd") or ""):
        wrong.append("a Stooq request for a key was not refused")
    stooq = namespace["closes_from_stooq"]("Date,Open,High,Low,Close,Volume\n2022-04-15,1,2,0.5,1.5,10\n")
    if stooq.to_dict() != {"2022-04-15": 1.5}:
        wrong.append(f"a Stooq CSV read as {stooq.to_dict()}")
    sample = namespace["fear_greed_from_json"]((REPO_DIR / FEAR_GREED_SAMPLE).read_text(encoding="utf-8"))
    if (len(sample), sample.index[0], sample.index[-1]) != (179, "2022-04-15", "2022-10-10"):
        wrong.append(f"the index sample read as {len(sample)} days {sample.index[0]} to {sample.index[-1]}")
    try:
        namespace["fear_greed_from_json"]('{"data": [], "metadata": {"error": "probe error"}}')
        wrong.append("an index reply with an error was not refused")
    except ValueError:
        pass
    report.check(not wrong, "chapter 15's parsers: Coinbase candles become UTC days and closes, an error reply is refused; "
                 "a Stooq web page (one that wants JavaScript) and a request for a key are refused, a Stooq CSV is read; "
                 "the index sample gives 179 days, a reply with an error is refused", "; ".join(wrong))

    calls = []

    def fake(source, first_day, last_day, fail=None):
        def fetch(first, last):
            calls.append(source)
            if fail:
                raise not_obtained(fail)
            days = pd.date_range(first_day, last_day, freq="D").strftime("%Y-%m-%d")
            return pd.Series(range(1, len(days) + 1), index=list(days), dtype="float64")
        return fetch

    namespace["PRICE_FETCHERS"] = {"yfinance": fake("yfinance", None, None, fail="probe: refused"),
                                   "coinbase": fake("coinbase", "2022-01-01", "2022-12-31"),
                                   "stooq": fake("stooq", None, None, fail="probe: a web page")}
    steps = []
    with contextlib.redirect_stdout(io.StringIO()):
        for first, last in [("2022-03-01", "2022-03-31"), ("2022-03-10", "2022-03-20"), ("2022-02-01", "2022-03-31"),
                            ("2021-06-01", "2022-03-31")]:
            calls.clear()
            closes, record = namespace["btc_closes"](first, last)
            steps.append((list(calls), record["source"], record["from_cache"], [item["source"] for item in record["failed"]],
                          record["reason"], None if closes is None else (closes.index[0], closes.index[-1])))
    expected = [(["yfinance", "coinbase"], "coinbase", False, ["yfinance"], None, ("2022-01-01", "2022-12-31")),
                ([], "coinbase", True, ["yfinance"], None, ("2022-01-01", "2022-12-31")),
                (["yfinance", "coinbase"], "coinbase", False, ["yfinance"], None, ("2022-01-01", "2022-12-31")),
                (["yfinance", "coinbase", "stooq"], "not_obtained", False, ["yfinance", "coinbase", "stooq"], "reason", None)]
    shape = [step[:4] + ("reason" if step[4] else None,) + step[5:] for step in steps]
    missed = steps[3][4] or ""
    report.check(shape == expected and "misses needed days" in missed and "probe: refused" in missed,
                 "chapter 15's price sources with fake sources and no connection: the first that has every needed day is "
                 "used and the ones before it are recorded with their reasons; a cache that covers the days is read without "
                 "asking any source; wider days are fetched again (the cache is replaced); a source that misses needed days "
                 "is passed over, and when every source fails the prices are not obtained, with every reason",
                 f"steps {steps}")

    replies = []

    def fake_get(address):
        replies.append(address)
        return (REPO_DIR / FEAR_GREED_SAMPLE).read_bytes()

    def broken_get(address):
        raise not_obtained("probe: no reply")

    namespace["web_get"] = fake_get
    steps = []
    with contextlib.redirect_stdout(io.StringIO()):
        for address in [VALIDATION_SWITCH_DEFAULTS["FEAR_GREED_SOURCE"], VALIDATION_SWITCH_DEFAULTS["FEAR_GREED_SOURCE"],
                        "https://api.alternative.me/fng/?limit=200&format=json"]:
            namespace["FEAR_GREED_SOURCE"] = address
            replies.clear()
            values, record = namespace["fear_greed_index"]("2022-05-01", "2022-05-31")
            steps.append((len(replies), record["source"], record["from_cache"], None if values is None else len(values)))
        namespace["web_get"] = broken_get
        values, record = namespace["fear_greed_index"]("2022-01-01", "2022-05-31")
        steps.append((None, record["source"], record["from_cache"], record["reason"]))
    report.check(steps == [(1, "alternative.me", False, 179), (0, "alternative.me", True, 179), (1, "alternative.me", False, 179),
                           (None, "not_obtained", False, "probe: no reply")],
                 "chapter 15's index with a fake reply and no connection: fetched once and cached; read from the cache "
                 "next time; fetched again for another address; not obtained, with the reason, when there is no reply",
                 f"steps {steps}")
    shutil.rmtree(VALIDATION_PROBE_DIR, ignore_errors=True)


def check_layers_probe(report):
    """Chapter 14's cell, executed from the source on made-up daily layers: periods that do not meet, a missing layer
    and a small overlap; the period, the days, the leans and the counts must follow the rules."""
    import pandas as pd

    def layer(first, shares):
        days = [{"date": (pd.Timestamp(first) + pd.Timedelta(days=offset)).strftime("%Y-%m-%d"),
                 "calibrated_bullish_share": share} for offset, share in enumerate(shares)]
        return {"date_from": days[0]["date"], "date_to": days[-1]["date"], "days": days}

    cases = {
        "periods that do not meet": {"media": layer("2022-01-01", [0.7, 0.2]), "retail": layer("2022-02-01", [0.3, 0.8])},
        "no media layer": {"retail": layer("2022-02-01", [0.3, 0.8])},
        "a small overlap": {"media": layer("2022-01-01", [0.9, 0.65, 0.5, None]),
                            "retail": layer("2022-01-02", [0.3, 0.3, 0.2, 0.7])},
    }
    expected = {
        "periods that do not meet": (None, None, [], 0, 0, []),
        "no media layer": (None, None, [], 0, 0, []),
        "a small overlap": ("2022-01-02", "2022-01-04", [True, False, None], 3, 2, ["2022-01-02"]),
    }
    wrong = []
    for name, daily_layers in cases.items():
        namespace = {"__name__": "notebook_cells", "pd": pd, "brief": {}, "daily_layers": daily_layers,
                     "LAYER_GAP": VALIDATION_SWITCH_DEFAULTS["LAYER_GAP"]}
        problems = run_cells(["LAYER_PAIR"], namespace)
        section = as_dict(namespace["brief"].get("layers"))
        found = (section.get("date_from"), section.get("date_to"), [day.get("opposite") for day in section.get("days") or []],
                 section.get("n_days"), section.get("n_days_both"), section.get("opposite_dates"))
        if problems or found != expected[name]:
            wrong.append(f"{name}: {problems[:1] or found}")
    report.check(not wrong, "chapter 14's cell on made-up layers: periods that do not meet and a missing media layer give a "
                 "null period, no days and 0 counts without an error; a small overlap gives its days, a lean only beyond "
                 "0.5 ± LAYER_GAP, null where a layer takes no part, and the opposite days", "; ".join(wrong))


def check_offline_sources(report, label, parameters, defaults):
    """In the quick check every run reads the price and the index from a local file or not at all."""
    price = parameters.get("PRICE_SOURCE", defaults.get("PRICE_SOURCE"))
    index = parameters.get("FEAR_GREED_SOURCE", defaults.get("FEAR_GREED_SOURCE"))
    report.check(isinstance(price, str) and price != "auto" and "://" not in price
                 and isinstance(index, str) and "://" not in index,
                 f"{label}: the price and the index come from local files or none, so the run connects to no price source "
                 "and not to alternative.me "
                 f"(PRICE_SOURCE {price!r}, FEAR_GREED_SOURCE {index!r})")


def check_lean(share, gap):
    """'bullish', 'bearish' or 'neither' by the rule of chapter 14; None without a share."""
    if share is None:
        return None
    return "bullish" if share >= 0.5 + gap else ("bearish" if share <= 0.5 - gap else "neither")


def calendar_days(first, last):
    """Every calendar day from first to last, YYYY-MM-DD; empty when either is None."""
    from datetime import date, timedelta
    if first is None or last is None:
        return []
    start, end = date.fromisoformat(first), date.fromisoformat(last)
    return [(start + timedelta(days=offset)).isoformat() for offset in range((end - start).days + 1)]


def check_layers(report, label, brief, switches):
    """The layers section: the overlap of the two layers' periods, every day of it with the two shares of daily, the
    leans and opposite days recomputed, the counts; and the model on the news headlines."""
    layers = as_dict(brief.get("layers"))
    if not report.check(bool(layers), f"{label}: layers section has content"):
        return
    daily_layers = as_dict(as_dict(brief.get("daily")).get("layers"))
    gap = switches["LAYER_GAP"]
    problems = []
    if layers.get("compared") != LAYER_PAIR:
        problems.append(f"compared {layers.get('compared')!r}")
    if not (layers.get("gap") == gap and same_value(layers.get("bullish_at_or_above"), 0.5 + gap)
            and same_value(layers.get("bearish_at_or_below"), 0.5 - gap)):
        problems.append(f"gap {layers.get('gap')!r} and bars {layers.get('bullish_at_or_above')!r}, "
                        f"{layers.get('bearish_at_or_below')!r} (LAYER_GAP {gap!r})")
    present = all(layer in daily_layers for layer in LAYER_PAIR)
    first = max(as_dict(daily_layers[layer]).get("date_from") for layer in LAYER_PAIR) if present else None
    last = min(as_dict(daily_layers[layer]).get("date_to") for layer in LAYER_PAIR) if present else None
    if first is not None and first > last:
        first = last = None
    if (layers.get("date_from"), layers.get("date_to")) != (first, last):
        problems.append(f"period {layers.get('date_from')} to {layers.get('date_to')} (expected {first} to {last})")
    days = layers.get("days") if isinstance(layers.get("days"), list) else []
    if [as_dict(day).get("date") for day in days] != calendar_days(first, last):
        problems.append(f"days {len(days)}, not every calendar day of the overlap ({len(calendar_days(first, last))})")
    daily_shares = {layer: {as_dict(day).get("date"): as_dict(day).get("calibrated_bullish_share")
                            for day in as_dict(daily_layers.get(layer)).get("days") or []} for layer in LAYER_PAIR}
    with_share, both, opposite_dates = dict.fromkeys(LAYER_PAIR, 0), 0, []
    for day in days:
        day = as_dict(day)
        shares = as_dict(day.get("calibrated_bullish_share"))
        if list(shares) != LAYER_PAIR:
            problems.append(f"{day.get('date')}: layers {list(shares)}")
            continue
        for layer in LAYER_PAIR:
            if shares[layer] != daily_shares[layer].get(day.get("date"), "missing"):
                problems.append(f"{day.get('date')} {layer}: {shares[layer]!r}, daily has "
                                f"{daily_shares[layer].get(day.get('date'), 'no such day')!r}")
            with_share[layer] += shares[layer] is not None
        leans = [check_lean(shares[layer], gap) for layer in LAYER_PAIR]
        expected = None if None in leans else sorted(leans) == ["bearish", "bullish"]
        if day.get("opposite") is not expected:
            problems.append(f"{day.get('date')}: opposite {day.get('opposite')!r} with shares {shares} (expected {expected})")
        both += expected is not None
        if expected:
            opposite_dates.append(day.get("date"))
    counted = {"n_days": len(days), "n_days_with_share": with_share, "n_days_both": both,
               "n_days_opposite": len(opposite_dates), "opposite_dates": opposite_dates}
    problems += [f"{key} {layers.get(key)!r} (counted {value!r})" for key, value in counted.items() if layers.get(key) != value]
    report.check(not problems, f"{label}: layers covers the overlap of the two layers' periods ({first} to {last}), "
                 "every calendar day of it in order, each with the calibrated shares of daily for media and retail; the "
                 "leans (LAYER_GAP) and the opposite days recomputed agree, and so do the days with a share, the days both "
                 f"take part ({both}), the opposite days ({len(opposite_dates)}) and their dates", "; ".join(problems[:5]))

    recognition = as_dict(layers.get("media_recognition"))
    news = as_dict(as_dict(as_dict(brief.get("dated_data")).get("datasets")).get("news"))
    labels = as_dict(news.get("label_counts"))
    calibration = as_dict(as_dict(brief.get("daily")).get("calibration"))
    media = as_dict(daily_layers.get("media"))
    problems = []
    n_bullish, n_bearish = recognition.get("n_labelled_bullish"), recognition.get("n_labelled_bearish")
    called_bullish, called_bearish = recognition.get("n_bullish_called_bullish"), recognition.get("n_bearish_called_bullish")
    if not all(is_count(value) for value in (n_bullish, n_bearish, called_bullish, called_bearish)) \
            or called_bullish > n_bullish or called_bearish > n_bearish:
        problems.append(f"counts {n_bullish!r}, {n_bearish!r}, {called_bullish!r}, {called_bearish!r}")
    else:
        if (n_bullish, n_bearish) != (labels.get("2"), labels.get("0")):
            problems.append(f"labelled {n_bullish} bullish and {n_bearish} bearish, dated_data has {labels}")
        expected = {
            "true_bullish_rate": share_of(called_bullish, n_bullish),
            "false_bullish_rate": share_of(called_bearish, n_bearish),
            "stocktwits_true_bullish_rate": calibration.get("true_bullish_rate"),
            "stocktwits_false_bullish_rate": calibration.get("false_bullish_rate"),
            "label_bullish_share": share_of(n_bullish, n_bullish + n_bearish),
            "half_bullish_share": share_of(called_bullish + called_bearish, n_bullish + n_bearish),
        }
        expected["calibrated_bullish_share"] = calibrated_from(expected["half_bullish_share"], calibration)
        problems += [f"{key} {recognition.get(key)!r} (expected {value!r})" for key, value in expected.items()
                     if not (is_score(recognition.get(key)) and same_value(recognition.get(key), value))]
    if (recognition.get("layer"), recognition.get("date_from"), recognition.get("date_to")) != \
            ("media", media.get("date_from"), media.get("date_to")):
        problems.append(f"layer and period {recognition.get('layer')} {recognition.get('date_from')} to "
                        f"{recognition.get('date_to')}")
    if (recognition.get("label_rule"), recognition.get("text_column")) != (news.get("label_rule"), news.get("text_column")):
        problems.append(f"label rule {recognition.get('label_rule')!r} and text {recognition.get('text_column')!r}")
    report.check(not problems, f"{label}: layers.media_recognition: the headlines labelled bullish and bearish are those "
                 "of dated_data, the called ones at most as many, both rates and the three shares follow from the counts "
                 "(null for an empty denominator), the calibrated one with the rates of daily.calibration, and the "
                 "StockTwits rates beside them are those of daily.calibration; the media layer's whole period",
                 "; ".join(problems[:5]))


def average_ranks(values):
    """Ranks from 1, equal values at their average rank (a second implementation, not pandas')."""
    order = sorted(range(len(values)), key=lambda position: values[position])
    ranks, start = [0.0] * len(values), 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for position in order[start:end + 1]:
            ranks[position] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def rank_correlation(pairs, min_days):
    """Spearman's rank correlation of (x, y) pairs: the Pearson correlation of the average ranks; None with fewer than
    min_days pairs or when either side has a single value."""
    if len(pairs) < min_days or len({x for x, _ in pairs}) < 2 or len({y for _, y in pairs}) < 2:
        return None
    left, right = average_ranks([x for x, _ in pairs]), average_ranks([y for _, y in pairs])
    mean_left, mean_right = sum(left) / len(left), sum(right) / len(right)
    covariance = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    spread = (sum((a - mean_left) ** 2 for a in left) * sum((b - mean_right) ** 2 for b in right)) ** 0.5
    return covariance / spread


def read_validation_inputs(brief, switches):
    """The closes and the index the run used, read by the check itself: a local file from its path, a web source from
    its cache in WORK_DIR/cache/; (closes or None, index or None, problems)."""
    from datetime import datetime, timezone
    import pyarrow.parquet as pq
    validation = as_dict(brief.get("validation"))
    price, index = as_dict(validation.get("price")), as_dict(validation.get("fear_greed"))
    cache = REPO_DIR / switches["WORK_DIR"] / "cache"
    closes, values, problems = None, None, []
    if price.get("source") == "local_file":
        lines = [line for line in (REPO_DIR / price.get("address", "")).read_text(encoding="utf-8").splitlines()
                 if line and not line.startswith("#")]
        if lines[0] != "date,close":
            problems.append(f"price file header {lines[0]!r}")
        closes = {day: float(close) for day, close in (line.split(",") for line in lines[1:])}
    elif price.get("source") in PRICE_WEB_SOURCES:
        file = cache / f"price-btc-usd-{price['source']}.parquet"
        if not file.exists():
            problems.append(f"no price cache {shown(file)}")
        else:
            table = pq.read_table(file)
            saved = json.loads(table.schema.metadata[b"ledger_eyes"])
            if (saved.get("source"), saved.get("fetched_on")) != (price.get("source"), price.get("fetched_on")):
                problems.append(f"price cache from {saved.get('source')} fetched on {saved.get('fetched_on')}")
            closes = dict(zip(table.column("date").to_pylist(), table.column("value").to_pylist()))
    if index.get("source") == "local_file":
        content = json.loads((REPO_DIR / index.get("address", "")).read_text(encoding="utf-8"))
        values = {datetime.fromtimestamp(int(item["timestamp"]), timezone.utc).strftime("%Y-%m-%d"): int(item["value"])
                  for item in content["data"]}
    elif index.get("source") == "alternative.me":
        file = cache / "fear-greed.parquet"
        if not file.exists():
            problems.append(f"no index cache {shown(file)}")
        else:
            table = pq.read_table(file)
            saved = json.loads(table.schema.metadata[b"ledger_eyes"])
            if (saved.get("address"), saved.get("fetched_on")) != (index.get("address"), index.get("fetched_on")):
                problems.append(f"index cache of {saved.get('address')} fetched on {saved.get('fetched_on')}")
            values = dict(zip(table.column("date").to_pylist(), table.column("value").to_pylist()))
    return closes, values, problems


def coverage_problems(record, series, name):
    """The first and last day and the number of days of series within the record's requested days."""
    inside = sorted(day for day in (series or {}) if record.get("requested_from") <= day <= record.get("requested_to"))
    expected = (inside[0] if inside else None, inside[-1] if inside else None, len(inside))
    found = (record.get("date_from"), record.get("date_to"), record.get("n_days"))
    return [] if found == expected else [f"{name} covers {found}, the data {expected}"]


def check_validation(report, label, brief, switches):
    """The validation section: where the price and the index come from, the periods, and every coefficient and its days
    recomputed by the check from the files the run read (a local file or the cache). Returns the coefficients against
    the index and against the next day's change that are not null."""
    from datetime import date, timedelta
    validation = as_dict(brief.get("validation"))
    if not report.check(bool(validation), f"{label}: validation section has content"):
        return 0, 0
    daily_layers = as_dict(as_dict(brief.get("daily")).get("layers"))
    first = min((as_dict(layer).get("date_from") for layer in daily_layers.values()), default=None)
    last = max((as_dict(layer).get("date_to") for layer in daily_layers.values()), default=None)
    price, index = as_dict(validation.get("price")), as_dict(validation.get("fear_greed"))
    problems = []
    if (validation.get("date_from"), validation.get("date_to"), validation.get("min_days")) != \
            (first, last, switches["VALIDATION_MIN_DAYS"]):
        problems.append(f"period {validation.get('date_from')} to {validation.get('date_to')}, min_days "
                        f"{validation.get('min_days')!r}")
    after = (date.fromisoformat(last) + timedelta(days=1)).isoformat() if last else None
    if (price.get("requested_from"), price.get("requested_to"), index.get("requested_from"), index.get("requested_to")) \
            != (first, after, first, last):
        problems.append(f"requested days: price {price.get('requested_from')} to {price.get('requested_to')}, index "
                        f"{index.get('requested_from')} to {index.get('requested_to')}")
    price_switch, index_switch = switches["PRICE_SOURCE"], switches["FEAR_GREED_SOURCE"]
    expected_price = (["not_obtained"] if price_switch == "none" else ["local_file"] if price_switch != "auto"
                      else PRICE_WEB_SOURCES + ["not_obtained"])
    expected_index = (["not_obtained"] if index_switch == "none" else ["local_file"] if "://" not in index_switch
                      else ["alternative.me", "not_obtained"])
    if price.get("source") not in expected_price or index.get("source") not in expected_index:
        problems.append(f"sources {price.get('source')!r} and {index.get('source')!r} with PRICE_SOURCE {price_switch!r} "
                        f"and FEAR_GREED_SOURCE {index_switch!r}")
    for name, record, local in (("price", price, price_switch), ("index", index, index_switch)):
        failed = record.get("failed", [])
        if record.get("source") == "not_obtained":
            if not (isinstance(record.get("reason"), str) and record["reason"].strip()) or record.get("address") is not None \
                    or record.get("n_days") != 0 or record.get("fetched_on") is not None or record.get("from_cache") is not False:
                problems.append(f"{name} not obtained: reason {record.get('reason')!r}, address {record.get('address')!r}, "
                                f"days {record.get('n_days')!r}, fetched on {record.get('fetched_on')!r}")
            if name == "price" and price_switch == "auto" and [item.get("source") for item in failed] != PRICE_WEB_SOURCES:
                problems.append(f"price not obtained after trying {[item.get('source') for item in failed]}")
        else:
            if record.get("reason") is not None:
                problems.append(f"{name} obtained with a reason {record.get('reason')!r}")
            if record.get("source") == "local_file":
                if record.get("address") != local or record.get("fetched_on") is not None or record.get("from_cache") is not False:
                    problems.append(f"{name} from the local file {record.get('address')!r}, fetched on "
                                    f"{record.get('fetched_on')!r}, from cache {record.get('from_cache')!r}")
            elif not (isinstance(record.get("fetched_on"), str) and isinstance(record.get("from_cache"), bool)):
                problems.append(f"{name} from the web: fetched on {record.get('fetched_on')!r}")
            if name == "price":
                tried = [item.get("source") for item in failed]
                if record.get("source") in PRICE_WEB_SOURCES and tried != PRICE_WEB_SOURCES[:PRICE_WEB_SOURCES.index(record["source"])]:
                    problems.append(f"price from {record.get('source')} after trying {tried}")
                if not (isinstance(record.get("day_boundary"), str) and record["day_boundary"].strip()):
                    problems.append("price without its day boundary")
        if any(not (isinstance(item.get("reason"), str) and item["reason"].strip()) for item in failed):
            problems.append(f"{name}: a failed source without a reason")
    if "alternative.me" not in str(index.get("attribution")):
        problems.append(f"index attribution {index.get('attribution')!r}")
    closes, values, read_problems = read_validation_inputs(brief, switches)
    problems += read_problems
    problems += coverage_problems(price, closes, "price") + coverage_problems(index, values, "index")
    report.check(not problems, f"{label}: validation records where the price ({price.get('source')}) and the index "
                 f"({index.get('source')}) come from as the switches say, a reason when not obtained and the sources tried "
                 "with theirs, the day boundary, the days requested (every layer's period, the price one day more) and "
                 "the days covered, recounted from the files the run read", "; ".join(problems[:5]))

    following = {}
    for day, close in (closes or {}).items():
        after_day = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
        if after_day in closes:
            following[day] = closes[after_day] / close - 1
    problems, checked = [], {"fear_greed": 0, "next_day_change": 0}
    layers = as_dict(validation.get("layers"))
    if list(layers) != list(daily_layers):
        problems.append(f"layers {list(layers)}, daily {list(daily_layers)}")
    for name, layer in daily_layers.items():
        entry = as_dict(layers.get(name))
        shares = {as_dict(day).get("date"): as_dict(day).get("calibrated_bullish_share")
                  for day in as_dict(layer).get("days") or [] if as_dict(day).get("calibrated_bullish_share") is not None}
        if (entry.get("date_from"), entry.get("date_to"), entry.get("n_days_with_share")) != \
                (as_dict(layer).get("date_from"), as_dict(layer).get("date_to"), len(shares)):
            problems.append(f"{name}: period {entry.get('date_from')} to {entry.get('date_to')}, days with a share "
                            f"{entry.get('n_days_with_share')!r} ({len(shares)})")
        for key, other in (("fear_greed", values or {}), ("next_day_change", following)):
            pairs = [(share, other[day]) for day, share in shares.items() if day in other]
            expected = rank_correlation(pairs, switches["VALIDATION_MIN_DAYS"])
            found = as_dict(entry.get(key))
            value = found.get("spearman")
            close = (value is None and expected is None) or (is_number(value) and expected is not None
                                                             and -1 <= value <= 1
                                                             and abs(value - expected) <= CORRELATION_TOLERANCE)
            if found.get("n_days") != len(pairs) or not close:
                problems.append(f"{name} {key}: {found.get('n_days')!r} days, {value!r} (recomputed {len(pairs)} days, "
                                f"{expected!r})")
            checked[key] += value is not None and close
    if price.get("source") == "not_obtained" and any(as_dict(as_dict(entry).get("next_day_change")) !=
                                                     {"n_days": 0, "spearman": None} for entry in layers.values()):
        problems.append("prices not obtained, but a layer has days or a coefficient against the next day's change")
    if index.get("source") == "not_obtained" and any(as_dict(as_dict(entry).get("fear_greed")) !=
                                                     {"n_days": 0, "spearman": None} for entry in layers.values()):
        problems.append("index not obtained, but a layer has days or a coefficient against it")
    report.check(not problems, f"{label}: validation, recomputed by the check (average ranks, Pearson of the ranks) from "
                 "the shares of daily and the files the run read: every layer's period and days with a share, and for the "
                 "index and the next day's change the days that take part and the coefficient (within "
                 f"{CORRELATION_TOLERANCE}, in [-1, 1], null below VALIDATION_MIN_DAYS or for a single value); "
                 f"{checked['fear_greed']} coefficients against the index and {checked['next_day_change']} against the next "
                 "day's change were not null and agree", "; ".join(problems[:5]))
    return checked["fear_greed"], checked["next_day_change"]


def stable_layers_validation(brief):
    """What two runs with the same switches must agree on: the overlap, the labelled headlines, where the price and the
    index come from and which days they cover."""
    layers, validation = as_dict(as_dict(brief).get("layers")), as_dict(as_dict(brief).get("validation"))
    recognition = as_dict(layers.get("media_recognition"))
    keep = ("source", "address", "reason", "requested_from", "requested_to", "date_from", "date_to", "n_days", "fetched_on")
    return {
        "layers": {key: layers.get(key) for key in ("compared", "date_from", "date_to", "n_days", "gap")},
        "days": [as_dict(day).get("date") for day in layers.get("days") or []],
        "labelled": {key: recognition.get(key) for key in ("n_labelled_bullish", "n_labelled_bearish", "label_rule")},
        "period": {key: validation.get(key) for key in ("date_from", "date_to", "min_days")},
        "price": {key: as_dict(validation.get("price")).get(key) for key in keep},
        "index": {key: as_dict(validation.get("fear_greed")).get(key) for key in keep},
    }


def compare_layers_validation(report, first, again):
    """Two runs with the default switches: the same overlap, labelled headlines, price and index sources and days."""
    left, right = stable_layers_validation(first), stable_layers_validation(again)
    report.check(bool(as_dict(first).get("layers")) and left == right,
                 "default switches run twice: the same period of layers and its days, the same headlines labelled bullish "
                 "and bearish, and the price and the index from the same place covering the same days",
                 "" if left == right else "differing parts: " + ", ".join(key for key in left if left[key] != right[key]))


def price_none_run(report, defaults, overrides, rules=None, built_in=None):
    """One more run of the quick check without prices (PRICE_SOURCE "none"): the notebook runs to the end, the prices
    are not obtained with a reason, every coefficient against the next day's change is null with 0 days, and the index
    half is still computed (a too-few bar of 1 gives it days to pair)."""
    parameters = {"DATA_SOURCE": SAMPLE_SOURCE, "EMOTION_LABELS_SOURCE": PRELABEL_SAMPLE_FILE, **DATED_SAMPLE_SETTINGS,
                  **VALIDATION_SAMPLE_SETTINGS, "WORK_DIR": "data/checks/quick-price-none", **overrides,
                  **PRICE_NONE_SETTINGS}
    parameters = panel_settings(report, PRICE_NONE_LABEL, parameters, overrides)
    switches = {**defaults, **parameters}
    report.info(f"{PRICE_NONE_LABEL}: " + ", ".join(f"{k}={v!r}" for k, v in parameters.items()))
    kept = [name for name in PRICE_NONE_SETTINGS if name in overrides]
    if kept:
        report.info(f"{PRICE_NONE_LABEL}: keeps its own values of {', '.join(kept)} instead of the -p values")
    check_offline_sources(report, PRICE_NONE_LABEL, parameters, defaults)
    brief_path = REPO_DIR / switches["WORK_DIR"] / "brief.json"
    brief_path.unlink(missing_ok=True)  # a stale brief must never pass
    ok, _, _ = run_notebook(report, PRICE_NONE_LABEL, parameters, REPO_DIR / switches["WORK_DIR"] / "executed.ipynb")
    brief = read_brief(report, PRICE_NONE_LABEL, brief_path) if ok else None
    if brief is None:
        return
    check_daily(report, PRICE_NONE_LABEL, brief, switches)
    check_layers(report, PRICE_NONE_LABEL, brief, switches)
    index_coefficients, _ = check_validation(report, PRICE_NONE_LABEL, brief, switches)
    check_format(report, PRICE_NONE_LABEL, brief, rules)
    check_panel_output(report, PRICE_NONE_LABEL, REPO_DIR / switches["WORK_DIR"] / "executed.ipynb")
    validation = as_dict(brief.get("validation"))
    price, layers = as_dict(validation.get("price")), as_dict(validation.get("layers"))
    report.check(price.get("source") == "not_obtained" and bool(str(price.get("reason") or "").strip())
                 and bool(layers) and all(as_dict(entry.get("next_day_change")) == {"n_days": 0, "spearman": None}
                                          for entry in layers.values())
                 and as_dict(validation.get("fear_greed")).get("source") == "local_file"
                 and any(as_dict(entry.get("fear_greed")).get("n_days", 0) > 0 for entry in layers.values())
                 and index_coefficients > 0,
                 f"{PRICE_NONE_LABEL}: the notebook ran to the end; the prices are not obtained with a reason, every "
                 "layer's correlation with the next day's change is null with 0 days, and the index half is computed "
                 f"({index_coefficients} coefficients against the index not null)",
                 f"price {price.get('source')!r} ({price.get('reason')!r}); layers "
                 + "; ".join(f"{name}: {entry.get('fear_greed')}, {entry.get('next_day_change')}" for name, entry in layers.items()))


def print_layers(brief):
    """Print chapter 14's numbers (full run)."""
    layers = as_dict(brief.get("layers"))
    if not layers:
        return
    recognition = as_dict(layers.get("media_recognition"))
    print("\nTwo layers (full run)")
    print(f"  compared:                      {layers.get('compared')}; lean bullish at or above "
          f"{layers.get('bullish_at_or_above')}, bearish at or below {layers.get('bearish_at_or_below')} (gap {layers.get('gap')})")
    print(f"  period both cover:             {layers.get('date_from')} to {layers.get('date_to')} ({layers.get('n_days')} days)")
    print(f"  days with a share:             {layers.get('n_days_with_share')}")
    print(f"  days both take part:           {layers.get('n_days_both')}")
    print(f"  days leaning opposite ways:    {layers.get('n_days_opposite')}")
    for day in layers.get("days") or []:
        if day.get("opposite"):
            print(f"    {day.get('date')}: {day.get('calibrated_bullish_share')}")
    print(f"  model on news headlines:       media {recognition.get('date_from')} to {recognition.get('date_to')}, label rule "
          f"{recognition.get('label_rule')}, text {recognition.get('text_column')}")
    print(f"    true bullish rate:           {recognition.get('true_bullish_rate')} ({recognition.get('n_bullish_called_bullish')} "
          f"of {recognition.get('n_labelled_bullish')} labelled bullish); on StockTwits {recognition.get('stocktwits_true_bullish_rate')}")
    print(f"    false bullish rate:          {recognition.get('false_bullish_rate')} ({recognition.get('n_bearish_called_bullish')} "
          f"of {recognition.get('n_labelled_bearish')} labelled bearish); on StockTwits {recognition.get('stocktwits_false_bullish_rate')}")
    print(f"    bullish share:               by the readers' labels {recognition.get('label_bullish_share')}, counted at 0.5 "
          f"{recognition.get('half_bullish_share')}, calibrated with the StockTwits rates {recognition.get('calibrated_bullish_share')}")


def print_validation(brief):
    """Print chapter 15's numbers (full run)."""
    validation = as_dict(brief.get("validation"))
    if not validation:
        return
    price, index = as_dict(validation.get("price")), as_dict(validation.get("fear_greed"))
    print("\nAgainst the market (full run)")
    print(f"  period:                        {validation.get('date_from')} to {validation.get('date_to')}; at least "
          f"{validation.get('min_days')} days for a coefficient")
    print(f"  price source:                  {price.get('source')} ({price.get('address')}); reason {price.get('reason')}; "
          f"fetched on {price.get('fetched_on')}, from the cache {price.get('from_cache')}")
    for item in price.get("failed") or []:
        print(f"    tried, not obtained:         {item.get('source')}: {item.get('reason')}")
    print(f"  price days:                    needed {price.get('requested_from')} to {price.get('requested_to')}, covered "
          f"{price.get('date_from')} to {price.get('date_to')} ({price.get('n_days')} days); {price.get('day_boundary')}")
    print(f"  fear and greed index:          {index.get('source')} ({index.get('address')}); reason {index.get('reason')}; "
          f"fetched on {index.get('fetched_on')}, from the cache {index.get('from_cache')}")
    print(f"  index days:                    needed {index.get('requested_from')} to {index.get('requested_to')}, covered "
          f"{index.get('date_from')} to {index.get('date_to')} ({index.get('n_days')} days); {index.get('attribution')}")
    for name, entry in as_dict(validation.get("layers")).items():
        print(f"  layer {name}: {entry.get('date_from')} to {entry.get('date_to')}, {entry.get('n_days_with_share')} days "
              f"with a share; against the index: {as_dict(entry.get('fear_greed')).get('spearman')} "
              f"({as_dict(entry.get('fear_greed')).get('n_days')} days); against the next day's change: "
              f"{as_dict(entry.get('next_day_change')).get('spearman')} ({as_dict(entry.get('next_day_change')).get('n_days')} days)")


# ---------------------------------------------------------------- modes

def parse_overrides(pairs):
    overrides = {}
    for name, value in pairs:
        try:
            overrides[name] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            overrides[name] = value
    return overrides


def sha256_of(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


REPEAT_LABEL = "default switches again"  # third quick run: same switches as the first, for the topics comparison


def compare_topics(report, first, again):
    """Two runs with the same switches must give identical topics sections."""
    first_topics = first.get("topics") if isinstance(first, dict) else None
    again_topics = again.get("topics") if isinstance(again, dict) else None
    detail = ""
    if isinstance(first_topics, dict) and isinstance(again_topics, dict) and first_topics != again_topics:
        keys = sorted(key for key in set(first_topics) | set(again_topics)
                      if first_topics.get(key) != again_topics.get(key))
        detail = f"differing fields: {keys}"
        if "topics" in keys and isinstance(first_topics.get("topics"), list) and isinstance(again_topics.get("topics"), list):
            changed = [index for index, (left, right) in enumerate(zip(first_topics["topics"], again_topics["topics"]))
                       if left != right]
            detail += f"; differing topics: {changed}"
    elif first_topics is None or again_topics is None:
        detail = "a brief or its topics section is missing"
    report.check(isinstance(first_topics, dict) and bool(first_topics) and first_topics == again_topics,
                 "default switches run twice: the two briefs have identical topics sections", detail)


def quick(report, defaults, others, overrides, rules=None, built_in=None):
    runs = [("default switches", {}, "quick-default"),
            ("other switches", {**dict(others), **OTHER_RUN_SOURCES}, "quick-other"),
            (REPEAT_LABEL, {}, "quick-default-again")]
    briefs = {}
    for label, flipped, folder in runs:
        parameters = {"DATA_SOURCE": SAMPLE_SOURCE, "EMOTION_LABELS_SOURCE": PRELABEL_SAMPLE_FILE,
                      **DATED_SAMPLE_SETTINGS, **VALIDATION_SAMPLE_SETTINGS, "WORK_DIR": f"data/checks/{folder}", **flipped,
                      **overrides}
        parameters = panel_settings(report, label, parameters, overrides)
        switches = {**defaults, **parameters}
        report.info(f"{label}: " + ", ".join(f"{k}={v!r}" for k, v in parameters.items()))
        check_offline_sources(report, label, parameters, defaults)
        brief_path = REPO_DIR / switches["WORK_DIR"] / "brief.json"
        brief_path.unlink(missing_ok=True)  # a stale brief must never pass
        ok, _, _ = run_notebook(report, label, parameters, REPO_DIR / switches["WORK_DIR"] / "executed.ipynb")
        brief = read_brief(report, label, brief_path) if ok else None
        briefs[label] = brief
        if brief is not None and label != REPEAT_LABEL:
            check_brief(report, label, brief, switches)
            check_sentiment(report, label, brief, switches, full_run=False)
            check_topics(report, label, brief, switches, full_run=False)
        if brief is not None:
            check_why(report, label, brief, switches)
            check_calibration(report, label, brief, switches)
            check_drawers(report, label, brief, switches, built_in)
            check_emotions(report, label, brief, switches)
            check_dated_data(report, label, brief, switches, quick_default=not flipped)
            check_daily(report, label, brief, switches)
            check_new_words(report, label, brief, switches)
            check_layers(report, label, brief, switches)
            check_validation(report, label, brief, switches)
            check_format(report, label, brief, rules)
            check_panel_output(report, label, REPO_DIR / switches["WORK_DIR"] / "executed.ipynb")
        if brief is not None and label != REPEAT_LABEL:  # the default run and the run with the other switches
            check_counterexamples(report, label, brief, rules)
            check_panel(report, label, brief_path, brief)
        if brief is not None and label == "default switches":
            check_save_refuses(report, label, brief, rules)
    compare_topics(report, briefs.get("default switches"), briefs.get(REPEAT_LABEL))
    compare_why(report, briefs.get("default switches"), briefs.get(REPEAT_LABEL))
    compare_drawers(report, briefs.get("default switches"), briefs.get(REPEAT_LABEL))
    compare_emotions(report, briefs.get("default switches"), briefs.get(REPEAT_LABEL))
    compare_dated(report, briefs.get("default switches"), briefs.get(REPEAT_LABEL), briefs.get("other switches"))
    compare_daily(report, briefs.get("default switches"), briefs.get(REPEAT_LABEL))
    compare_layers_validation(report, briefs.get("default switches"), briefs.get(REPEAT_LABEL))
    threshold_run(report, defaults, overrides, briefs.get("default switches"), rules, built_in)
    spike_run(report, defaults, overrides, rules, built_in)
    price_none_run(report, defaults, overrides, rules, built_in)


def full(report, defaults, overrides, rules=None, built_in=None):
    parameters = {"DATA_SOURCE": FULL_SOURCE, "EMOTION_LABELS_SOURCE": PRELABEL_FULL_FILE, **DATED_FULL_SETTINGS,
                  **overrides}
    parameters = panel_settings(report, "full run", parameters, overrides)
    switches = {**defaults, **parameters}
    source = switches["DATA_SOURCE"]
    if "://" not in source:
        path = REPO_DIR / source
        if not report.check(path.is_file(), "full dataset available locally",
                            "" if path.is_file() else
                            f"{source} not found; run scripts/download_data.sh or pass -p DATA_SOURCE <address>"):
            return
        report.info(f"data file {source}: {path.stat().st_size:,} bytes, sha256 {sha256_of(path)}")
    if not check_dated_copies(report, switches):
        return
    report.info("full run: " + ", ".join(f"{k}={v!r}" for k, v in parameters.items()))
    brief_path = REPO_DIR / switches["WORK_DIR"] / "brief.json"
    brief_path.unlink(missing_ok=True)
    ok, seconds, _ = run_notebook(report, "full run", parameters, REPO_DIR / switches["WORK_DIR"] / "runs" / "full.ipynb")
    brief = read_brief(report, "full run", brief_path) if ok else None
    check_prelabel_full(report, switches)
    if brief is None:
        return
    check_brief(report, "full run", brief, switches)
    check_sentiment(report, "full run", brief, switches, full_run=True)
    check_topics(report, "full run", brief, switches, full_run=True)
    check_why(report, "full run", brief, switches)
    check_calibration(report, "full run", brief, switches)
    check_calibration_narrows(report, "full run", brief)
    check_drawers(report, "full run", brief, switches, built_in)
    check_emotions(report, "full run", brief, switches, full_run=True)
    check_dated_data(report, "full run", brief, switches)
    check_daily(report, "full run", brief, switches)
    check_new_words(report, "full run", brief, switches)
    check_layers(report, "full run", brief, switches)
    check_validation(report, "full run", brief, switches)
    format_started = time.perf_counter()
    check_format(report, "full run", brief, rules)
    format_seconds = time.perf_counter() - format_started
    check_panel_output(report, "full run", REPO_DIR / switches["WORK_DIR"] / "runs" / "full.ipynb")
    check_panel(report, "full run", brief_path, brief)
    dataset = brief.get("dataset") or {}
    sample, cleaning = dataset.get("sample") or {}, dataset.get("cleaning") or {}
    btc = dataset.get("btc_mentions") or {}
    print("\nKey numbers (full run)")
    print(f"  posts in total:                {dataset.get('n_posts')}   sheets: {dataset.get('sheets')}")
    print(f"  posts per class:               {dataset.get('class_counts')}   (0 bearish, 1 neutral, 2 bullish)")
    print(f"  class shares:                  " + json.dumps({k: round(v, 4) for k, v in (dataset.get('class_shares') or {}).items() if v is not None}))
    print(f"  mention bitcoin/btc:           {btc.get('n_posts')}   per class: {btc.get('by_class')}")
    print(f"  words per post:                {dataset.get('length_words')}")
    print(f"  repeated texts:                {dataset.get('duplicates')}   posts with links: {dataset.get('n_posts_with_links')}")
    print(f"  empty texts:                   {dataset.get('n_empty_text')}")
    print(f"  working sample per class:      {sample.get('class_counts')}   (requested {sample.get('per_class_requested')})")
    print(f"  average words before / after:  {cleaning.get('avg_words_before')} / {cleaning.get('avg_words_after')}")
    print(f"  empty after cleaning:          {cleaning.get('n_empty_after')}")
    print(f"  run time:                      {seconds:.1f} s")
    print_sentiment(brief)
    print_topics(brief)
    print_why(brief)
    print_calibration(brief)
    print_drawers(brief)
    print_emotions(brief)
    print_dated(brief, brief_path)
    print_daily(brief)
    print_new_words(brief)
    print_layers(brief)
    print_validation(brief)
    print("\nTimes (full run)")
    for line in daily_lines(REPO_DIR / switches["WORK_DIR"] / "runs" / "full.ipynb"):
        print(f"  {line}")
    print(f"  format check of the brief:     {format_seconds:.1f} s (the check the notebook runs before saving)")
    print(f"  brief file size:               {brief_path.stat().st_size:,} bytes")
    print("\nBrief format (full run)")
    print(f"  format_version:                {brief.get('format_version')}")
    print(f"  fields that are null here:     {sorted(null_fields(brief)) or 'none'}")


def main():
    parser = argparse.ArgumentParser(description="Project checks.")
    parser.add_argument("mode", choices=["quick", "full"])
    parser.add_argument("-p", nargs=2, action="append", default=[], metavar=("NAME", "VALUE"),
                        help="override a notebook switch for every run")
    args = parser.parse_args()
    overrides = parse_overrides(args.p)

    report = Report()
    print(f"== {args.mode} check ==")
    check_requirements(report)
    check_notebook_files(report)
    defaults, others = read_switches(report)
    check_sentiment_switches(report, defaults)
    check_topic_switches(report, defaults)
    check_why_switches(report, defaults)
    check_calibration_switches(report, defaults)
    check_drawer_switches(report, defaults, others)
    check_numbering_rules(report)
    check_text(report)
    rules = load_format_rules(report)
    built_in_drawers = check_drawer_words(report)
    check_panel_source(report, defaults, others)
    check_panel_charts_source(report)
    check_prelabel_script(report)
    check_prelabel_reading(report)
    check_prelabel_fake_model(report)
    check_prelabel_files(report)
    check_emotion_switches(report, defaults, others)
    check_emotion_chapter(report)
    check_emotion_probes(report)
    check_dated_switches(report, defaults, others)
    check_dated_shape(report)
    check_dated_chapter(report)
    check_dated_probes(report)
    check_daily_switches(report, defaults, others)
    check_validation_switches(report, defaults, others)
    check_validation_chapters(report)
    check_validation_probes(report)
    check_layers_probe(report)
    unknown = [name for name in overrides if name not in defaults]
    report.check(not unknown, "overrides name existing switches", ", ".join(unknown))
    if defaults and not unknown:
        if args.mode == "quick":
            quick(report, defaults, others, overrides, rules, built_in_drawers)
        else:
            full(report, defaults, overrides, rules, built_in_drawers)

    print()
    if report.failures:
        print(f"{args.mode} check FAILED ({len(report.failures)}):")
        for label in report.failures:
            print(f"  - {label}")
        return 1
    print(f"{args.mode} check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
