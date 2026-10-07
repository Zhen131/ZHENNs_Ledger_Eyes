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
full:  static checks, then one run on the full dataset with the default
       switches; prints the key numbers from the brief. The calibrated
       bullish share must narrow the gap to the labels in the overall fair
       comparison.

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
import hashlib
import io
import json
import os
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
from notebook_build import NOTEBOOK, SOURCE, build_notebook_text  # noqa: E402

REQUIREMENTS = REPO_DIR / "requirements.txt"
SAMPLE_SOURCE = "samples/stocktwits-crypto-sample.xlsx"  # relative to the repository, as the notebook sees it
FULL_SOURCE = "data/raw/st-data-full.xlsx"
TOP_LEVEL_FIELDS = ["format_version", "generated_at", "scope", "dataset", "sentiment", "topics", "why", "drawers"]
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
# Chapter 11: the brief's format rules, its documentation and the panel.
SCHEMA_FILE = REPO_DIR / "docs" / "brief.schema.json"  # the format rules; the notebook carries an identical copy
FORMAT_DOC = REPO_DIR / "docs" / "brief-format.md"  # the field-by-field documentation of the format
DOC_TABLE_HEADER = "| Field | Type | Required | Null | Meaning |"  # header of every field table in FORMAT_DOC
PANEL_SWITCH = "LAUNCH_PANEL"
CHECK_SETTINGS = {PANEL_SWITCH: False}  # every run of both checks: the panel is built, never started
PANEL_TABLES = ["scope", "evaluation", "topics", "why"]  # the panel's tables, in tab order
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
    parameters = {"DATA_SOURCE": SAMPLE_SOURCE, "WORK_DIR": "data/checks/quick-thresholds", **overrides, **settings}
    parameters = panel_settings(report, THRESHOLD_LABEL, parameters, overrides)
    switches = {**defaults, **parameters}
    report.info(f"{THRESHOLD_LABEL}: " + ", ".join(f"{k}={v!r}" for k, v in parameters.items()))
    kept = [name for name in settings if name in overrides]
    if kept:
        report.info(f"{THRESHOLD_LABEL}: keeps its own values of {', '.join(kept)} instead of the -p values")
    brief_path = REPO_DIR / switches["WORK_DIR"] / "brief.json"
    brief_path.unlink(missing_ok=True)  # a stale brief must never pass
    ok, _, _ = run_notebook(report, THRESHOLD_LABEL, parameters, REPO_DIR / switches["WORK_DIR"] / "executed.ipynb")
    brief = read_brief(report, THRESHOLD_LABEL, brief_path) if ok else None
    if brief is None:
        return
    check_why(report, THRESHOLD_LABEL, brief, switches)
    check_calibration(report, THRESHOLD_LABEL, brief, switches)
    check_drawers(report, THRESHOLD_LABEL, brief, switches, built_in)
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


# ---------------------------------------------------------------- the brief's format and the panel (chapter 11)

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
result["network"] = network
print("PANEL-PROBE " + json.dumps(result))
'''
PANEL_MARKS = [  # (path in the brief, marker, table that must show it)
    (("scope", "source"), "PANEL-MARK-SOURCE", "scope"),
    (("sentiment", "evaluation", "rows", 0, "model"), "PANEL-MARK-MODEL", "evaluation"),
    (("topics", "topics", 0, "name"), "PANEL-MARK-TOPIC", "topics"),
    (("why", "topics", 0, "name"), "PANEL-MARK-WHY", "why"),
]


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
    report.check(list(tables) == PANEL_TABLES and all(rows[name] == count for name, count in expected_rows.items())
                 and brief.get("format_version") in scope_values and as_dict(brief.get("scope")).get("source") in scope_values
                 and built["page"] == "Blocks"
                 and built["components"] == [tables[name] for name in PANEL_TABLES],
                 f"{label}: the panel shows the scope, evaluation, topic and topic x bullish tables with one row per "
                 "row of the brief, on one tab each",
                 f"tables {rows}, expected {expected_rows}; page {built['page']}; "
                 f"{len(built['components'])} table components matching the tables: "
                 f"{built['components'] == [tables[name] for name in PANEL_TABLES]}")
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
            ("other switches", dict(others), "quick-other"),
            (REPEAT_LABEL, {}, "quick-default-again")]
    briefs = {}
    for label, flipped, folder in runs:
        parameters = {"DATA_SOURCE": SAMPLE_SOURCE, "WORK_DIR": f"data/checks/{folder}", **flipped, **overrides}
        parameters = panel_settings(report, label, parameters, overrides)
        switches = {**defaults, **parameters}
        report.info(f"{label}: " + ", ".join(f"{k}={v!r}" for k, v in parameters.items()))
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
    threshold_run(report, defaults, overrides, briefs.get("default switches"), rules, built_in)


def full(report, defaults, overrides, rules=None, built_in=None):
    parameters = {"DATA_SOURCE": FULL_SOURCE, **overrides}
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
    report.info("full run: " + ", ".join(f"{k}={v!r}" for k, v in parameters.items()))
    brief_path = REPO_DIR / switches["WORK_DIR"] / "brief.json"
    brief_path.unlink(missing_ok=True)
    ok, seconds, _ = run_notebook(report, "full run", parameters, REPO_DIR / switches["WORK_DIR"] / "runs" / "full.ipynb")
    brief = read_brief(report, "full run", brief_path) if ok else None
    if brief is None:
        return
    check_brief(report, "full run", brief, switches)
    check_sentiment(report, "full run", brief, switches, full_run=True)
    check_topics(report, "full run", brief, switches, full_run=True)
    check_why(report, "full run", brief, switches)
    check_calibration(report, "full run", brief, switches)
    check_calibration_narrows(report, "full run", brief)
    check_drawers(report, "full run", brief, switches, built_in)
    check_format(report, "full run", brief, rules)
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
