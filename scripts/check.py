"""Project checks. Run through scripts/check.sh so the environment from scripts/env.sh is set.

    scripts/check.sh quick [-p NAME VALUE ...]
    scripts/check.sh full  [-p NAME VALUE ...]

quick: static checks, then the notebook runs three times on the bundled
       sample (default switches, every two-state switch flipped, default
       switches again). The first two saved briefs are checked chapter by
       chapter (dataset, posts with a ticker, cleaning examples, sentiment:
       split, vectorizers, models, evaluation table, confusion matrices,
       scoring function). Every
       brief must pass the format rules (docs/brief.schema.json); the first
       two are also broken on purpose (locked counterexamples, all of which
       must be rejected). The two default runs must agree on every number that
       does not depend on training (the whole dataset section, the posts per
       class in every part of the split and the posts in every test subset).
       Static checks: the pinned packages, the source file and the .ipynb,
       the switch cell (defaults, other values, and the switch check cell run
       from the source on good and bad values), the text scan, the format
       rules with their notebook copy and their document, and the chapter
       headings (0 to 9, the brief last, with no network package in its
       cells). Chapter 1's reading cells run from the source on small
       hand-written parquet files (ids, empty texts, the label conversion,
       missing columns, an unknown label, a sample with its row column), and
       its ticker cell on sentences that must and must not count. The bundled
       sample is checked: its columns and rows, no empty text, no account name
       and no link, enough posts per class, per level and with a ticker, and
       its texts against the fingerprints of the level pre-label file.
       Chapter 5's split runs from the source on six hand-written classes,
       and chapter 7's score functions (a cell of definitions only) on six
       classes against scikit-learn. The names later chapters use (the
       interface) must be defined with their locked parameters, and chapter
       6 must train its word vectors itself, downloading none.
       The level pre-label script (scripts/prelabel_levels.py) is checked
       without Ollama and without the dataset: its interface and pinned
       prompt, its requests, that it only talks to a loopback address, its
       draw on a hand-written table, its voting, retries, resuming and
       settings record driven by a fake model, its check-file command on
       good and broken files, and that its files get LF line endings only;
       the committed sample label file is checked statically; scripts/env.sh
       is checked for the Python of a Windows virtual environment, and
       requirements-prelabel.txt and .gitattributes statically.
full:  static checks, then one run on the full dataset (the local copy in
       data/raw/, see scripts/download_data.sh) with the default switches;
       prints the key numbers and the evaluation table from the brief, and
       every model must beat both baselines on all test posts. The notebook's
       own reading cell reads the local copy, and every id of the level
       pre-label file, the review table and the sample must be a post of it
       with the same text fingerprint (the sample: the same text and label).
       The level pre-label files are compared with the dataset (read from data/raw/, downloaded
       from Hugging Face at the pinned revision the first time): the sample,
       and the full label file and the review table when they are there
       (they are labelled on another machine; when missing, a line says so).

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
import prelabel_levels as levels  # noqa: E402
from notebook_build import NOTEBOOK, SOURCE, build_notebook_text  # noqa: E402

REQUIREMENTS = REPO_DIR / "requirements.txt"
SAMPLE_SOURCE = "samples/tim-financial-tweets-sample.parquet"  # relative to the repository, as the notebook sees it
FULL_SOURCE = "data/raw/tim-financial-tweets-bba60269.parquet"  # the local copy (scripts/download_data.sh)
DEFAULT_DATA_SOURCE = ("hf://datasets/TimKoornstra/financial-tweets-sentiment@bba602698fd6f18226f6404fd8fb637ef67fa1eb/"
                       "data/train-00000-of-00001.parquet")  # the pinned file on Hugging Face
TOP_LEVEL_FIELDS = ["format_version", "generated_at", "scope", "dataset", "sentiment", "levels"]
SCOPE_FIELDS = ["source", "granularity", "tickers", "n_classes"]
REQUIRED_SWITCHES = ["DATA_SOURCE", "WORK_DIR", "RANDOM_SEED", "SAMPLE_PER_CLASS",
                     "N_CLASSES", "TICKER_ONLY", "EMOJI_TO_WORDS", "KEEP_NEGATIONS"]
SENTIMENT_SWITCHES = ["BPE_VOCAB_SIZE", "VOCAB_SIZE", "EPOCHS", "BATCH_SIZE"]
WORD2VEC_SWITCHES = ["WORD2VEC_DIM", "WORD2VEC_WINDOW", "WORD2VEC_MIN_COUNT", "WORD2VEC_EPOCHS"]  # chapter 6, not two-state
SENTIMENT_MODELS = ["count_network", "tfidf_network", "word2vec_network"]  # the networks of chapter 7; more may be added later
SENTIMENT_BASELINES = ["always_majority", "uniform_random"]
SENTIMENT_SUBSETS = ["all_test", "test_has_ticker"]
SPLIT_PARTS = ["train", "validation", "test"]
# Every switch of the notebook, in order: the switch cell holds exactly these (so no switch of a removed chapter is left).
ALL_SWITCHES = ["DATA_SOURCE", "WORK_DIR", "RANDOM_SEED", "SAMPLE_PER_CLASS", "N_CLASSES", "TICKER_ONLY", "EMOJI_TO_WORDS",
                "KEEP_NEGATIONS", "BPE_VOCAB_SIZE", "VOCAB_SIZE", "EPOCHS", "BATCH_SIZE", "WORD2VEC_DIM", "WORD2VEC_WINDOW",
                "WORD2VEC_MIN_COUNT", "WORD2VEC_EPOCHS", "LEVEL_LABELS_SOURCE", "LEVEL_REVIEW_SOURCE",
                "LEVEL_SECOND_REVIEW_SOURCE", "LEVEL_BALANCE", "LEVEL_EPOCHS"]
# Chapter 1: what a ticker is. Locked (only add): texts that must count and texts that must not.
TICKER_MUST_MATCH = ["$TSLA", "$tsla", "$F", "$BRK.B", "($AAPL)", "$NVDA's", "buying more $SPY."]
TICKER_MUST_NOT_MATCH = ["$100", "$5.20", "$TOOLONG", "abc$TSLA", "a lone $ sign", "$"]
RAW_TO_LABEL = {0: 1, 1: 2, 2: 0}  # the dataset's sentiment (0 neutral, 1 bullish, 2 bearish) -> the notebook's label
READING_PROBE_DIR = REPO_DIR / "data" / "checks" / "reading-probe"
# The bundled sample: what it must hold (all lower bounds).
SAMPLE_COLUMNS = ["tweet", "sentiment", "row"]
SAMPLE_MIN_PER_CLASS = 200
SAMPLE_MIN_PER_LEVEL = 80
SAMPLE_MIN_NONE = 80  # posts labelled none (the levels chapter learns none as a seventh class)
SAMPLE_MIN_TICKER_PER_SIDE = 100  # bearish and bullish posts with a ticker: TICKER_ONLY with two classes must split
SAMPLE_HANDLE = re.compile(r"@[A-Za-z0-9_]")  # an account name
SAMPLE_LINK = re.compile(r"https?://|www\.", flags=re.IGNORECASE)  # a web link
# Names later chapters use (the interface): each must be defined at the top level of a code cell.
INTERFACE_NAMES = ["df_all", "has_ticker", "clean_texts", "split_train_val_test", "train_df", "val_df", "test_df",
                   "count_vectorizer", "tfidf_vectorizer", "word2vec_vectorizer", "vectorizers", "fit_vectorizers",
                   "vectorize_texts", "to_targets", "build_network", "predict_proba", "predict_labels", "models",
                   "selected_model_name", "selected_model", "selected_vectorizer", "sentiment_proba", "score_texts",
                   "class_scores", "macro_f1", "evaluate", "brief_path", "save_brief"]
# Locked parameters of the interface functions: name -> [(parameter, default as written in the source or None)].
INTERFACE_PARAMETERS = {
    "split_train_val_test": [("posts", None), ("seed", None), ("label_column", "'label'"), ("labels", "None")],
    "fit_vectorizers": [("cleaned_texts", None)],
    "vectorize_texts": [("cleaned_texts", None), ("fitted", "vectorizers")],
    "to_targets": [("labels_of_posts", None), ("labels", "CLASS_LABELS")],
    "build_network": [("n_features", None), ("labels", "CLASS_LABELS")],
    "predict_proba": [("model", None), ("matrix", None), ("labels", "CLASS_LABELS")],
    "predict_labels": [("model", None), ("matrix", None), ("labels", "CLASS_LABELS")],
    "class_scores": [("true_labels", None), ("predicted_labels", None), ("labels", "CLASS_LABELS")],
    "macro_f1": [("true_labels", None), ("predicted_labels", None), ("labels", "CLASS_LABELS")],
    "evaluate": [("true_labels", None), ("predicted_labels", None), ("labels", "CLASS_LABELS")],
    "score_texts": [("texts", None)],
    "sentiment_proba": [("cleaned_texts", None), ("model", "selected_model"), ("vectorizer", "selected_vectorizer")],
    "save_brief": [("brief_dict", None), ("path", None)],
}
NO_DOWNLOAD_WORDS = ["gensim.downloader", "api.load", "://"]  # chapter 6 must contain none of them
# Chapter 9: the brief's format rules and their documentation.
SCHEMA_FILE = REPO_DIR / "docs" / "brief.schema.json"  # the format rules; the notebook carries an identical copy
FORMAT_DOC = REPO_DIR / "docs" / "brief-format.md"  # the field-by-field documentation of the format
DOC_TABLE_HEADER = "| Field | Type | Required | Null | Meaning |"  # header of every field table in FORMAT_DOC
# The chapters, in order; the last one checks and saves the brief and must not use the network.
CHAPTER_COUNT = 11  # chapters 0 to 10
BRIEF_CHAPTER = 10
LEVEL_CHAPTERS = {9: "Levels: what does a post pay attention to?"}  # chapter -> title; they import no Gradio
BRIEF_CHAPTER_TITLE = "The brief: check and save"
BRIEF_CHAPTER_IMPORTS = {"json", "math", "pathlib", "datetime", "numpy", "pandas", "jsonschema"}  # all it may import

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
    "I don't think $TSLA will pump, not bullish at all 😭": ["not", "not"],
    "Never selling my Nvidia shares 🚀🚀": ["never"],
    "No way it can't go higher, I won't sell 😂": ["no", "not", "not"],
    "Nor do I care about this dip 💎🙌": ["nor"],
}
EMOJI_TOKENS = {
    "I don't think $TSLA will pump, not bullish at all 😭": ["loudly", "cry", "face"],
    "Never selling my Nvidia shares 🚀🚀": ["rocket", "rocket"],
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
        "tickers": "has_ticker_only" if switches["TICKER_ONLY"] else "all",
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
    if switches["TICKER_ONLY"]:
        report.check(sample.get("n_has_ticker") == sample.get("n_posts"),
                     f"{label}: with TICKER_ONLY every sampled post names a ticker",
                     f"{sample.get('n_has_ticker')} of {sample.get('n_posts')}")
    tickers = dataset.get("ticker_mentions") if isinstance(dataset.get("ticker_mentions"), dict) else {}
    by_class = tickers.get("by_class") if isinstance(tickers.get("by_class"), dict) else {}
    report.check(sorted(by_class) == ["0", "1", "2"] and all(is_count(count) for count in by_class.values())
                 and sum(by_class.values()) == tickers.get("n_posts")
                 and is_count(tickers.get("n_posts")) and tickers["n_posts"] <= (n_posts if is_count(n_posts) else -1),
                 f"{label}: posts with a ticker per class add up to the posts with a ticker, at most every post",
                 f"by class {by_class}, total {tickers.get('n_posts')}, posts {n_posts}")

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
    ticker_counts = split.get("n_has_ticker") if isinstance(split.get("n_has_ticker"), dict) else {}
    n_test_ticker = ticker_counts.get("test")
    report.check(is_count(n_test_ticker) and n_test_ticker <= (n_test if is_count(n_test) else -1),
                 f"{label}: test posts that name a ticker are counted", f"{n_test_ticker} of {n_test}")

    # Chapter 6: vectorizers learn on the training set only.
    vectors = sentiment.get("vectorizers") if isinstance(sentiment.get("vectorizers"), dict) else {}
    vocab_sizes = [(vectors.get(name) or {}).get("vocab_size") if isinstance(vectors.get(name), dict) else None
                   for name in ("count", "tfidf")]
    word2vec = vectors.get("word2vec") if isinstance(vectors.get("word2vec"), dict) else {}
    zeros = word2vec.get("n_posts_all_zero") if isinstance(word2vec.get("n_posts_all_zero"), dict) else {}
    report.check(word2vec.get("fitted_on_n_posts") == sizes.get("train") and is_count(word2vec.get("vocab_size"))
                 and word2vec["vocab_size"] > 0
                 and all(word2vec.get(key) == switches.get(name) for key, name in
                         (("vector_size", "WORD2VEC_DIM"), ("window", "WORD2VEC_WINDOW"),
                          ("min_count", "WORD2VEC_MIN_COUNT"), ("epochs", "WORD2VEC_EPOCHS")))
                 and sorted(zeros) == sorted(SPLIT_PARTS)
                 and all(is_count(zeros[part]) and zeros[part] <= sizes.get(part, -1) for part in SPLIT_PARTS),
                 f"{label}: the averaged word vectors learned from the training set only, with the WORD2VEC switches, "
                 "and count the posts left with a vector of zeros",
                 f"{word2vec}")
    report.check(vectors.get("fitted_on") == "train" and vectors.get("fitted_on_n_posts") == sizes.get("train")
                 and isinstance(switches.get("VOCAB_SIZE"), int)
                 and all(is_count(size) and 0 < size <= switches["VOCAB_SIZE"] for size in vocab_sizes),
                 f"{label}: both vectorizers learned at most VOCAB_SIZE words from the training set only",
                 f"fitted on {vectors.get('fitted_on')!r} ({vectors.get('fitted_on_n_posts')} posts), "
                 f"vocabulary sizes {vocab_sizes}")

    # Chapters 7 and 8: models, evaluation table, confusion matrices, choice and scoring function.
    models = sentiment.get("models") if isinstance(sentiment.get("models"), list) else []
    report.check(all(name in models for name in SENTIMENT_MODELS),
                 f"{label}: the word-count, TF-IDF and averaged-word-vector networks are all listed", f"{models}")
    evaluation = sentiment.get("evaluation") if isinstance(sentiment.get("evaluation"), dict) else {}
    rows = [row for row in evaluation.get("rows") or [] if isinstance(row, dict)]
    found = Counter((row.get("model"), row.get("subset")) for row in rows)
    expected = [(name, subset) for name in models + SENTIMENT_BASELINES for subset in SENTIMENT_SUBSETS]
    wrong = [pair for pair in expected if found[pair] != 1]
    report.check(not wrong and len(rows) == len(expected),
                 f"{label}: evaluation table has one row per model and per baseline (always majority, uniform random) "
                 "for all test posts and for test posts that name a ticker",
                 f"missing or repeated: {wrong}; {len(rows)} rows for {len(expected)} expected")

    problems = []
    for row in rows:
        name = f"{row.get('model')}/{row.get('subset')}"
        n = row.get("n")
        expected_n = n_test if row.get("subset") == "all_test" else n_test_ticker
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

    training = sentiment.get("training") if isinstance(sentiment.get("training"), dict) else {}
    expected_layer = "1 unit, sigmoid" if len(classes) == 2 else f"{len(classes)} units, softmax"
    expected_loss = "binary_crossentropy" if len(classes) == 2 else "sparse_categorical_crossentropy"
    report.check(training.get("output_layer") == expected_layer and training.get("loss") == expected_loss
                 and training.get("target_of_label") == {label: position for position, label in enumerate(classes)},
                 f"{label}: the output layer, the loss and the targets follow the classes in use ({expected_layer})",
                 f"output layer {training.get('output_layer')!r}, loss {training.get('loss')!r}, "
                 f"targets {training.get('target_of_label')}")

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

    score_table = scoring.get("score_texts") if isinstance(scoring.get("score_texts"), dict) else {}
    demo = score_table.get("demo") if isinstance(score_table.get("demo"), dict) else {}
    names = {"0": "bearish", "1": "neutral", "2": "bullish"}
    report.check(score_table.get("columns") == [f"p_{names[c]}" for c in classes] + ["call"]
                 and is_count(demo.get("n_texts")) and demo["n_texts"] > 0 and demo.get("n_rows") == demo["n_texts"]
                 and demo.get("rows_sum_to_one") is True and demo.get("call_is_highest") is True,
                 f"{label}: score_texts gives one row per demo sentence, a probability column per class in use adding up "
                 "to 1, and a call that is the class of the highest probability", f"{score_table}")

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
    ticker_rows = [row for row in rows if row.get("subset") == "test_has_ticker"]
    report.check(bool(ticker_rows) and all(is_count(row.get("n")) and row["n"] > 0 for row in ticker_rows),
                 f"{label}: the rows for test posts that name a ticker have posts",
                 f"{[row.get('n') for row in ticker_rows]}")
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
    print(f"  split (posts):                 {split.get('n_posts')}   with a ticker: {split.get('n_has_ticker')}")
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


def print_levels(brief):
    """Print the key numbers of the levels section (full run)."""
    section = as_dict(brief.get("levels"))
    if not section:
        return
    print("\nLevels (full run)")
    labels_file = as_dict(section.get("labels_file"))
    print(f"  label file:                    {labels_file.get('n_rows')} rows, sha256 {labels_file.get('sha256')}")
    print(f"  per value:                     {labels_file.get('label_counts')}")
    print(f"  models, prompt versions:       {labels_file.get('models')}, {labels_file.get('prompt_versions')}")
    print(f"  matching:                      {section.get('matching')}")
    print(f"  left out:                      {section.get('dropped')}")
    balance = as_dict(section.get("balance"))
    print(f"  per class before the draw:     {balance.get('counts_before')}")
    print(f"  per class after the draw:      {balance.get('counts_after')}   (LEVEL_BALANCE {balance.get('enabled')})")
    print(f"  split (posts):                 {as_dict(section.get('split')).get('n_posts')}")
    selection = as_dict(section.get("selection"))
    print(f"  validation macro-F1 (six):     {selection.get('validation_macro_f1')}   chosen: {selection.get('selected')}")

    def number(value):
        return f"{value:.4f}" if isinstance(value, (int, float)) and not isinstance(value, bool) else "-"

    short = {"assets_dividends": "assets", "earnings": "earn", "revenue_business": "revenue", "price_chart": "chart",
             "story_macro": "story", "emotion_only": "emotion", "none": "none"}
    print("  evaluation table, seven classes (F1 per class; macro-F1 over the six levels, then over all seven):")
    print(f"  {'model':16} {'posts':>5} {'accuracy':>8} " + " ".join(f"{short[c]:>7}" for c in LEVEL_LABELS)
          + f" {'six':>7} {'seven':>7}")
    for row in as_dict(section.get("evaluation")).get("rows") or []:
        per_class = as_dict(row.get("per_class"))
        print(f"  {str(row.get('model')):16} {str(row.get('n')):>5} {number(row.get('accuracy')):>8} "
              + " ".join(f"{number(as_dict(per_class.get(c)).get('f1')):>7}" for c in LEVEL_LABELS)
              + f" {number(row.get('macro_f1')):>7} {number(row.get('macro_f1_seven')):>7}")
    print("  precision and recall per class:")
    for row in as_dict(section.get("evaluation")).get("rows") or []:
        per_class = as_dict(row.get("per_class"))
        print(f"    {str(row.get('model')):16} " + " ".join(
            f"{short[c]} {number(as_dict(per_class.get(c)).get('precision'))}/"
            f"{number(as_dict(per_class.get(c)).get('recall'))}" for c in LEVEL_LABELS))
    for name, matrix in as_dict(section.get("confusion_matrices")).items():
        print(f"  7x7 confusion matrix {name} (rows true, columns predicted, {LEVEL_LABELS}): {matrix.get('counts')}")
    print(f"  beats both baselines:          {section.get('beats_baselines')}")
    groups = as_dict(section.get("groups"))
    print(f"  groups: true none {groups.get('n_true_none')}, called none {groups.get('n_called_none')}")
    print(f"  {'model':16} {'posts':>5} {'accuracy':>8} " + " ".join(f"{g:>11}" for g in LEVEL_GROUP_ORDER)
          + f" {'macro-F1':>8}")
    for row in as_dict(groups.get("evaluation")).get("rows") or []:
        per_class = as_dict(row.get("per_class"))
        print(f"  {str(row.get('model')):16} {str(row.get('n')):>5} {number(row.get('accuracy')):>8} "
              + " ".join(f"{number(as_dict(per_class.get(g)).get('f1')):>11}" for g in LEVEL_GROUP_ORDER)
              + f" {number(row.get('macro_f1')):>8}")
    for name, matrix in as_dict(groups.get("confusion_matrices")).items():
        print(f"  3x3 confusion matrix {name} (rows true, columns predicted, {LEVEL_GROUP_ORDER}): {matrix.get('counts')}")
    review = as_dict(section.get("review"))
    for who in ("human", "second_model"):
        part = as_dict(review.get(who))
        if part:
            print(f"  review {who}: {part.get('n_reviewed')} rows, agreement {part.get('agreement')}, weighted "
                  f"{part.get('weighted_agreement')}")
            print(f"    per pre-label: {part.get('by_prelabel')}")
            print(f"    confusion (rows review label {LEVEL_LABELS}, columns pre-label {LEVEL_SIX}): "
                  f"{as_dict(part.get('confusion')).get('counts')}")


# ---------------------------------------------------------------- the brief's format (chapter 9)

def as_dict(value):
    return value if isinstance(value, dict) else {}


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
# catch at least; kind 6 adds a null where the format does not allow one, and a field the format does not know;
# kind 7 a value the format does not allow.
BROKEN_BRIEFS = [
    *[("1 top-level field missing", (field,), DELETE, "brief", field) for field in TOP_LEVEL_FIELDS],
    ("2 required field missing in a section", ("scope", "granularity"), DELETE, "scope", "granularity"),
    ("2 required field missing in a section", ("dataset", "n_posts"), DELETE, "dataset", "n_posts"),
    ("2 required field missing in a section", ("sentiment", "evaluation"), DELETE, "sentiment", "evaluation"),
    ("2 required field missing in a section", ("sentiment", "evaluation", "rows", 0, "macro_f1"), DELETE,
     "sentiment.evaluation.rows[0]", "macro_f1"),
    ("2 required field missing in a section", ("levels", "evaluation"), DELETE, "levels", "evaluation"),
    ("3 wrong type", ("dataset", "n_posts"), str, "dataset.n_posts", "integer"),  # a count written as text
    ("3 wrong type", ("scope",), lambda old: [], "scope", "object"),  # an object written as a list
    ("3 wrong type", ("dataset", "cleaning", "keep_negations"), lambda old: "true", "dataset.cleaning.keep_negations",
     "boolean"),
    ("3 wrong type", ("levels", "labels_file", "n_rows"), str, "levels.labels_file.n_rows", "integer"),
    ("4 granularity not allowed", ("scope", "granularity"), lambda old: "weekly", "scope.granularity", "weekly"),
    ("4 granularity not allowed", ("scope", "granularity"), lambda old: "Batch", "scope.granularity", "Batch"),
    ("4 granularity not allowed", ("scope", "granularity"), lambda old: "", "scope.granularity", "''"),
    ("4 granularity not allowed", ("scope", "granularity"), lambda old: None, "scope.granularity", "None"),
    ("4 granularity not allowed", ("scope", "granularity"), lambda old: 1, "scope.granularity", "1"),
    ("5 NaN or infinity", ("dataset", "class_shares", "0"), lambda old: NAN, "dataset.class_shares.0", "nan"),
    ("5 NaN or infinity", ("dataset", "n_posts"), lambda old: NAN, "dataset.n_posts", "nan"),
    ("5 NaN or infinity", ("sentiment", "evaluation", "rows", 0, "accuracy"), lambda old: -INFINITY,
     "sentiment.evaluation.rows[0].accuracy", "-inf"),
    ("5 NaN or infinity", ("levels", "review", "second_model", "agreement"), lambda old: NAN,
     "levels.review.second_model.agreement", "nan"),
    ("6 null not allowed, or an unknown field", ("dataset", "n_posts"), lambda old: None, "dataset.n_posts", "None"),
    ("6 null not allowed, or an unknown field", ("sentiment", "extra_field"), lambda old: 1, "sentiment", "extra_field"),
    ("6 null not allowed, or an unknown field", ("scope", "extra_field"), lambda old: 1, "scope", "extra_field"),
    ("6 null not allowed, or an unknown field", ("levels", "matching", "n_matched"), lambda old: None,
     "levels.matching.n_matched", "None"),
    ("6 null not allowed, or an unknown field", ("levels", "extra_field"), lambda old: 1, "levels", "extra_field"),
    ("7 value the format does not allow", ("scope", "n_classes"), lambda old: 4, "scope.n_classes", "4"),
    ("7 value the format does not allow", ("levels", "classes"), lambda old: [value for value in old if value != "none"],
     "levels.classes", "expected"),
]
# Locked: fields the format allows to be null. Set to null one at a time, the brief must still pass.
NULL_ALLOWED = [
    ("dataset", "class_shares", "0"),
    ("dataset", "length_words", "p90"),
    ("sentiment", "evaluation", "rows", 0, "macro_f1"),
    ("dataset", "cleaning", "avg_words_after"),
    ("sentiment", "selection", "validation_macro_f1", "count_network"),
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
    for description, path, change in [("a missing section", ("sentiment",), DELETE),
                                      ("NaN in a field that may not be null", ("dataset", "n_posts"), lambda old: NAN)]:
        message = attempt(changed_copy(brief, path, change))
        results.append(message is not None and not target.exists())
        report.info(f"save_brief with {description}: {'refused, nothing written' if results[-1] else 'NOT refused'}"
                    f" ({message})")
    message = attempt(changed_copy(brief, ("dataset", "class_shares", "0"), lambda old: NAN))
    try:
        saved = json.loads(target.read_text(encoding="utf-8"), parse_constant=reject_constant)
        nan_as_null = message is None and saved["dataset"]["class_shares"]["0"] is None
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


def check_panel_output(report, label, executed):
    """The executed notebook saved the brief through the format check (the line the save cell prints)."""
    lines = []
    if executed.exists():
        for cell in json.loads(executed.read_text(encoding="utf-8"))["cells"]:
            for out in cell.get("outputs", []):
                text = out.get("text", "")
                lines += ("".join(text) if isinstance(text, list) else text).splitlines()
    saved = [line for line in lines if line.startswith("[brief] saved") and "passed the format check" in line]
    report.check(len(saved) == 1,
                 f"{label}: the notebook saved the brief after the format check",
                 " | ".join(saved) or "no [brief] line")
    for line in saved:
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

# ---------------------------------------------------------------- shared helpers
# Kept from the checks of removed chapters because the checks above and below use them.

REFUSED_HOSTS = ["http://8.8.8.8:11434", "http://example.com:11434", "https://127.0.0.1:11434",
                 "http://127.0.0.1.example.com:11434", "http://127.0.0.1@example.com:11434", "http://0.0.0.0:11434",
                 "http://192.168.1.10:11434", "ftp://localhost:11434", "127.0.0.1:11434",
                 "http://localhost:11434/api"]
ACCEPTED_HOSTS = ["http://127.0.0.1:11434", "http://localhost:11434", "http://[::1]:11434", "http://127.0.0.1"]


def own_text_hash(text):
    """The documented text fingerprint, computed here: first 8 hex digits of the SHA-256 of the UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


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


def function_from_cell(name):
    """The source of the top-level function name, taken alone from the code cell that defines it (None when absent)."""
    source = cell_defining(name) or ""
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    return None


def imported_modules(tree):
    return sorted({alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
                   for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(node.module or "")])})


NETWORK_MODULES = ("urllib", "requests", "http", "socket", "httpx", "urllib3", "aiohttp", "yfinance", "curl_cffi")

# ---------------------------------------------------------------- chapters

def check_chapters(report):
    """The chapter headings are 0 to 9 in order; the last one is the brief and imports no network package."""
    chapters = chapter_cells()
    numbers = list(chapters)
    titles = {}
    for number, cells in chapters.items():
        heading = re.match(r"## [0-9]+\. (.+)", cells[0].source)
        titles[number] = heading.group(1).strip() if heading else ""
    report.check(numbers == list(range(CHAPTER_COUNT)),
                 f"the chapter headings are 0 to {CHAPTER_COUNT - 1}, in order", f"found {numbers}")
    last = chapters.get(BRIEF_CHAPTER, [])
    report.check(bool(numbers) and numbers[-1] == BRIEF_CHAPTER and titles.get(BRIEF_CHAPTER) == BRIEF_CHAPTER_TITLE,
                 f"chapter {BRIEF_CHAPTER} is the last chapter and is the brief ({BRIEF_CHAPTER_TITLE!r})",
                 f"last chapter {numbers[-1] if numbers else None}, title {titles.get(BRIEF_CHAPTER)!r}")
    modules = set()
    for cell in last:
        if cell.cell_type == "code":
            modules |= set(imported_modules(ast.parse(cell.source)))
    outside = sorted(name for name in modules if name.split(".")[0] in NETWORK_MODULES
                     or name.split(".")[0] not in BRIEF_CHAPTER_IMPORTS)
    report.check(bool(last) and not outside,
                 f"chapter {BRIEF_CHAPTER} imports no network package and nothing outside "
                 f"{sorted(BRIEF_CHAPTER_IMPORTS)} (no web page, no panel)",
                 f"imports {sorted(modules)}; not allowed {outside}")
    gradio = sorted(number for number in LEVEL_CHAPTERS for cell in chapters.get(number, [])
                    if cell.cell_type == "code" and "gradio" in {name.split(".")[0] for name in
                                                                 imported_modules(ast.parse(cell.source))})
    wrong_titles = {number: titles.get(number) for number, title in LEVEL_CHAPTERS.items() if titles.get(number) != title}
    report.check(not gradio and not wrong_titles,
                 "the level chapters are there with their titles ("
                 + ", ".join(f"{number}. {title}" for number, title in LEVEL_CHAPTERS.items())
                 + ") and import no Gradio (no panel)", f"titles differing {wrong_titles}; chapters importing gradio {gradio}")


# ---------------------------------------------------------------- switches, reading, tickers and the sample (chapters 1 and 2)

def check_switch_cell(report, defaults, others):
    """The switch cell's defaults and other values, and the switch check cell run from the source."""
    removed = sorted(name for name in defaults if name not in ALL_SWITCHES)  # switches of removed chapters, or unknown
    report.check(defaults.get("DATA_SOURCE") == DEFAULT_DATA_SOURCE and defaults.get("TICKER_ONLY") is False
                 and others.get("TICKER_ONLY") is True and not removed and list(defaults) == ALL_SWITCHES,
                 "switch cell: DATA_SOURCE defaults to the pinned Hugging Face file, TICKER_ONLY defaults to False with "
                 "True as its other value, and the switch cell holds exactly the switches of ALL_SWITCHES, in order (so "
                 "no switch of a removed chapter is left)",
                 f"DATA_SOURCE {defaults.get('DATA_SOURCE')!r}, TICKER_ONLY {defaults.get('TICKER_ONLY')!r} / "
                 f"{others.get('TICKER_ONLY')!r}, switches not in ALL_SWITCHES {removed}, order {list(defaults)}")
    report.check(defaults.get("N_CLASSES") == 3 and others.get("N_CLASSES") == 2,
                 "switch cell: N_CLASSES defaults to 3 (bearish, neutral, bullish), with 2 as its other value",
                 f"default {defaults.get('N_CLASSES')!r}, other {others.get('N_CLASSES')!r}")
    source = cell_defining_text("Switches look fine.")
    problems = []

    def run(values):
        namespace = {**defaults, **values}
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(source, "the switch check cell", "exec"), namespace)
            return None
        except ValueError as exc:
            return str(exc)

    if source is None:
        problems.append("no cell prints 'Switches look fine.'")
    else:
        message = run({})
        if message is not None:
            problems.append(f"default values refused: {message}")
        message = run(dict(others))
        if message is not None:
            problems.append(f"other values refused: {message}")
        for name, value in SWITCH_PROBES:
            message = run({name: value})
            if message is None or name not in message:
                problems.append(f"{name}={value!r} {'passed' if message is None else 'refused without naming it'}")
    report.check(not problems, "switch check cell (run from the source): the defaults and the other values pass, and "
                 "every bad value stops it with the switch's name ("
                 + ", ".join(f"{name}={value!r}" for name, value in SWITCH_PROBES) + ")", "; ".join(problems))


SWITCH_PROBES = [("N_CLASSES", 4), ("TICKER_ONLY", "yes"), ("EMOJI_TO_WORDS", 1), ("VOCAB_SIZE", 0), ("WORD2VEC_DIM", 0),
                 ("WORD2VEC_WINDOW", -1), ("WORD2VEC_MIN_COUNT", 0), ("WORD2VEC_EPOCHS", -3),
                 ("LEVEL_LABELS_SOURCE", ""), ("LEVEL_LABELS_SOURCE", 5), ("LEVEL_REVIEW_SOURCE", None),
                 ("LEVEL_SECOND_REVIEW_SOURCE", 1.5), ("LEVEL_BALANCE", "yes"), ("LEVEL_EPOCHS", 0), ("LEVEL_EPOCHS", -2)]


def cell_defining_text(marker):
    """Source of the first code cell that contains marker, or None."""
    return next((source for source in code_cells() if marker in source), None)


def reading_namespace():
    """The notebook's reading cells (chapter 1), executed from the source; returns (namespace, problems)."""
    namespace = {"__name__": "notebook_cells"}
    return namespace, run_cells(["read_posts"], namespace)


def write_parquet(path, columns):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path)
    return path


def stops_with(function, *args, words=()):
    """The ValueError message when function(*args) stops, or None when it does not stop."""
    try:
        function(*args)
    except ValueError as exc:
        message = str(exc)
        return message if all(word in message for word in words) else f"(without {words}) {message}"
    return None


def check_reading(report):
    """Chapter 1's reading cells, run from the source on small hand-written parquet files."""
    shutil.rmtree(READING_PROBE_DIR, ignore_errors=True)
    namespace, problems = reading_namespace()
    if not report.check(not problems, "chapter 1: the reading cell runs from the source", "; ".join(problems)):
        return
    read_posts, posts_from_raw = namespace["read_posts"], namespace["posts_from_raw"]
    texts = ["$TSLA to the moon", None, "  spaces around  ", "", "plain text, no ticker", "second $tsla post"]
    sentiments = [1, 0, 2, 0, 1, 2]
    whole = write_parquet(READING_PROBE_DIR / "whole.parquet",
                          {"tweet": texts, "sentiment": sentiments, "url": [f"https://example.com/{i}" for i in range(6)]})
    posts = read_posts(str(whole))
    import pandas as pd
    expected = levels.posts_from_table(pd.read_parquet(whole))
    same_as_script = ([{"post_id": row.post_id, "text": row.text} for row in posts.itertuples()] == expected)
    report.check(list(posts["post_id"]) == [f"tim:{row}" for row in range(len(texts))] and same_as_script,
                 "chapter 1 on a file without a row column: ids tim:0, tim:1, ... by position, and ids and texts equal "
                 "those of scripts/prelabel_levels.py (posts_from_table)",
                 f"ids {list(posts['post_id'])}; same as the pre-label script: {same_as_script}")
    report.check(list(posts["text"]) == ["" if text is None else text for text in texts]
                 and list(posts["label"]) == [RAW_TO_LABEL[value] for value in sentiments]
                 and list(posts.columns) == ["post_id", "text", "label"],
                 "chapter 1: a missing text becomes an empty string, spaces are kept, the labels are converted "
                 "(sentiment 0 -> 1, 1 -> 2, 2 -> 0) and the url column is not read",
                 f"texts {list(posts['text'])}, labels {list(posts['label'])}, columns {list(posts.columns)}")
    # The sample route: rows drawn from the whole file keep the id and the text of the whole file.
    picked = [4, 0, 5]
    part = write_parquet(READING_PROBE_DIR / "part.parquet",
                         {"tweet": [texts[row] for row in picked], "sentiment": [sentiments[row] for row in picked],
                          "row": picked})
    sample = read_posts(str(part))
    by_id = {row.post_id: (row.text, row.label) for row in posts.itertuples()}
    mismatched = [row.post_id for row in sample.itertuples() if by_id.get(row.post_id) != (row.text, row.label)]
    report.check(list(sample["post_id"]) == [f"tim:{row}" for row in picked] and not mismatched,
                 "chapter 1 on a file with a row column: ids tim:<row> from that column, and every post has the id, the "
                 "text and the label it has in the whole file", f"ids {list(sample['post_id'])}; mismatched {mismatched}")
    stops = {
        "sentiment 3": stops_with(read_posts, str(write_parquet(READING_PROBE_DIR / "three.parquet",
                                                                {"tweet": ["a", "b"], "sentiment": [1, 3]})),
                                  words=("sentiment", "3")),
        "no tweet column": stops_with(read_posts, str(write_parquet(READING_PROBE_DIR / "no-tweet.parquet",
                                                                    {"text": ["a"], "sentiment": [1]})),
                                      words=("tweet",)),
        "no sentiment column": stops_with(read_posts, str(write_parquet(READING_PROBE_DIR / "no-sentiment.parquet",
                                                                        {"tweet": ["a"], "label": [1]})),
                                          words=("sentiment",)),
    }
    report.check(all(message is not None and not message.startswith("(without") for message in stops.values()),
                 "chapter 1 stops on an unknown sentiment value and on a missing tweet or sentiment column, naming it",
                 "; ".join(f"{case}: {message}" for case, message in stops.items()))
    # The loading cell and the ticker cell, from the source, on the whole probe file: df_all has exactly four columns.
    namespace.update({"DATA_SOURCE": str(whole), "WORK_DIR": str(READING_PROBE_DIR / "work")})
    problems = run_cells(["load_posts", "TICKER_PATTERN"], namespace)
    df_all = namespace.get("df_all")
    columns = list(df_all.columns) if df_all is not None else None
    report.check(not problems and columns == ["post_id", "text", "label", "has_ticker"]
                 and list(df_all["has_ticker"]) == [True, False, False, False, False, True],
                 "chapter 1: loading and the ticker cell, run from the source, give df_all exactly the columns post_id, "
                 "text, label and has_ticker", f"{'; '.join(problems)} columns {columns}")
    shutil.rmtree(READING_PROBE_DIR, ignore_errors=True)


def check_split_function(report):
    """Chapter 5's split, run from the source: about 80/10/10 per class for any number of classes; stops on a class
    in use without posts and on a class that is too small."""
    import pandas as pd
    posts = pd.DataFrame({"label": [label for label in range(6) for _ in range(30)], "post_id": range(180)})
    namespace = {"N_CLASSES": 3, "RANDOM_SEED": 42, "df": posts[posts["label"] < 3].copy()}
    problems = run_cells(["split_train_val_test"], namespace)
    if not report.check(not problems, "chapter 5: the split cell runs from the source", "; ".join(problems)):
        return
    split = namespace["split_train_val_test"]
    train, validation, test = split(posts, 42, labels=list(range(6)))
    sizes = {name: part["label"].value_counts().sort_index().tolist()
             for name, part in (("train", train), ("validation", validation), ("test", test))}
    stops = {
        "a class without posts": stops_with(split, posts[posts["label"] != 4], 42, "label", list(range(6)), words=("4",)),
        "a class of 5 posts": stops_with(split, posts.iloc[list(range(150)) + list(range(150, 155))], 42, words=("5",)),
    }
    report.check(sizes == {"train": [24] * 6, "validation": [3] * 6, "test": [3] * 6}
                 and all(message is not None and not message.startswith("(without") for message in stops.values()),
                 "chapter 5: split_train_val_test splits six classes of 30 posts into 24 / 3 / 3 each, and stops on a "
                 "class in use without posts and on a class of fewer than MIN_POSTS_PER_CLASS posts",
                 f"sizes per class {sizes}; " + "; ".join(f"{case}: {message}" for case, message in stops.items()))


def check_score_functions(report):
    """The cell of class_scores, macro_f1 and evaluate holds definitions only; run from the source on six classes,
    it gives the scores scikit-learn gives."""
    import numpy as np
    from sklearn.metrics import f1_score, precision_recall_fscore_support
    source = cell_defining("evaluate") or ""
    tree = ast.parse(source) if source else None
    kinds = sorted({type(node).__name__ for node in (tree.body if tree else [])})
    defined = {node.name for node in (tree.body if tree else []) if isinstance(node, ast.FunctionDef)}
    only_definitions = bool(tree) and set(kinds) <= {"Import", "ImportFrom", "FunctionDef"} \
        and {"class_scores", "macro_f1", "evaluate"} <= defined
    report.check(only_definitions, "chapter 7: class_scores, macro_f1 and evaluate sit together in a cell that holds "
                 "only imports and function definitions", f"statements {kinds}, functions {sorted(defined)}")
    namespace = {"CLASS_LABELS": [0, 1, 2]}
    problems = run_cells(["evaluate"], namespace)
    if problems:
        report.check(False, "chapter 7: the score functions run from the source", "; ".join(problems))
        return
    labels = list(range(6))
    true = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 2])
    predicted = np.array([0, 1, 2, 3, 4, 5, 1, 1, 2, 4, 4, 5, 0, 2, 2, 3, 5, 5, 0, 0])
    expected = precision_recall_fscore_support(true, predicted, labels=labels, zero_division=np.nan)
    found = namespace["class_scores"](true, predicted, labels)
    row = namespace["evaluate"](true, predicted, labels)
    macro = namespace["macro_f1"](true, predicted, labels)
    same = all(np.allclose(np.asarray(left, dtype=float), np.asarray(right, dtype=float)) for left, right in zip(found, expected))
    reference = f1_score(true, predicted, labels=labels, average="macro")
    report.check(same and abs(macro - reference) < 1e-12 and abs(row["macro_f1"] - reference) < 1e-12
                 and sorted(row["per_class"]) == [str(label) for label in labels]
                 and sum(item["support"] for item in row["per_class"].values()) == len(true)
                 and abs(row["accuracy"] - float(np.mean(true == predicted))) < 1e-12,
                 "chapter 7: on six classes (labels 0 to 5) the score functions, run from the source, give "
                 "scikit-learn's per-class scores and macro-F1, so they do not depend on the number of classes",
                 f"macro-F1 {macro} (scikit-learn {reference}); per-class keys {sorted(row['per_class'])}")


def top_level_names():
    """Every name bound at the top level of a code cell of the source (assignments, tuples included, defs, classes)."""
    names = set()
    for source in code_cells():
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    names |= {item.id for item in ast.walk(target) if isinstance(item, ast.Name)}
    return names


def check_interface(report, defaults, others):
    """The names later chapters use are defined, with their locked parameters; chapter 6 downloads nothing; the
    word vector switches are not two-state."""
    defined = top_level_names()
    missing = [name for name in INTERFACE_NAMES if name not in defined]
    report.check(not missing, f"interface: all {len(INTERFACE_NAMES)} names later chapters use are defined in the "
                 "notebook", f"missing {missing}")
    wrong = []
    for name, expected in INTERFACE_PARAMETERS.items():
        source = cell_defining(name) or ""
        function = next((node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
                         and node.name == name), None) if source else None
        if function is None:
            wrong.append(f"{name}: not found")
            continue
        arguments = function.args.args
        defaults_written = [None] * (len(arguments) - len(function.args.defaults)) + \
            [ast.unparse(value) for value in function.args.defaults]
        found = list(zip([argument.arg for argument in arguments], defaults_written))
        if found != expected:
            wrong.append(f"{name}: {found}")
    report.check(not wrong, "interface: the functions take the locked parameters (labels with the classes in use as "
                 "default, so two and three classes behave as before)", "; ".join(wrong))
    chapter_six = "\n".join(cell.source for cell in chapter_cells().get(6, []) if cell.cell_type == "code")
    found = [word for word in NO_DOWNLOAD_WORDS if word in chapter_six]
    report.check(bool(chapter_six) and not found and "Word2Vec(" in chapter_six,
                 "chapter 6 trains its word vectors itself and downloads none (no gensim.downloader, api.load or web "
                 "address in its cells)", f"found {found}")
    word2vec = {name: defaults.get(name) for name in WORD2VEC_SWITCHES}
    report.check(all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in word2vec.values())
                 and not any(name in others for name in WORD2VEC_SWITCHES),
                 "switch cell: the word vector switches are positive integers and not two-state",
                 f"{word2vec}; two-state: {[name for name in WORD2VEC_SWITCHES if name in others]}")


def notebook_ticker(texts):
    """has_ticker for texts, from the notebook's own ticker cell run from the source; None when it cannot run."""
    import pandas as pd
    namespace = {"pd": pd, "df_all": pd.DataFrame({"text": list(texts)})}
    if run_cells(["TICKER_PATTERN"], namespace):
        return None
    return [bool(value) for value in namespace["df_all"]["has_ticker"]]


def check_ticker_rule(report):
    """The notebook's ticker cell, run from the source, on texts that must and must not count."""
    found = notebook_ticker(TICKER_MUST_MATCH + TICKER_MUST_NOT_MATCH)
    if found is None:
        report.check(False, "chapter 1: the ticker cell runs from the source")
        return
    missed = [text for text, value in zip(TICKER_MUST_MATCH, found) if not value]
    wrong = [text for text, value in zip(TICKER_MUST_NOT_MATCH, found[len(TICKER_MUST_MATCH):]) if value]
    report.check(not missed and not wrong, f"chapter 1: the ticker rule counts {TICKER_MUST_MATCH} and does not count "
                 f"{TICKER_MUST_NOT_MATCH}", f"missed {missed}; wrongly counted {wrong}")


def level_file_rows(relative):
    """The rows of a level file in the repository (lists, as scripts/prelabel_levels.py reads them), or None."""
    path = REPO_DIR / relative
    return levels.file_report(path)[1] if path.is_file() else None


def check_sample_file(report):
    """The bundled sample: columns, rows, what it holds, nothing private, and its texts against the level file."""
    import pandas as pd
    import pyarrow.parquet as pq
    path = REPO_DIR / SAMPLE_SOURCE
    if not report.check(path.is_file(), f"{SAMPLE_SOURCE} exists"):
        return
    columns = pq.ParquetFile(path).schema_arrow.names
    sample = pd.read_parquet(path)
    rows = list(sample["row"]) if "row" in sample.columns else []
    # The rows drawn first are in order; the posts labelled none that make_sample.py appends follow them, in order
    # among themselves. So the rows are unique and fall into at most two parts that are each in order.
    cut = next((index for index in range(1, len(rows)) if rows[index] < rows[index - 1]), len(rows))
    first_part, appended = rows[:cut], rows[cut:]
    report.check(columns == SAMPLE_COLUMNS and len(rows) == len(set(rows)) and bool(rows)
                 and first_part == sorted(first_part) and appended == sorted(appended),
                 f"{SAMPLE_SOURCE}: columns exactly {SAMPLE_COLUMNS}, rows unique; the rows drawn first in order, and "
                 "the rows appended after them (posts labelled none) in order",
                 f"columns {columns}, {len(rows)} rows, {len(set(rows))} distinct; {len(first_part)} rows in the first "
                 f"part, {len(appended)} appended")
    if columns != SAMPLE_COLUMNS:
        return
    texts = sample["tweet"].fillna("")
    empty = int((texts.str.strip() == "").sum())
    handles = int(texts.str.contains(SAMPLE_HANDLE).sum())
    links = int(texts.str.contains(SAMPLE_LINK).sum())
    report.check(empty == 0 and handles == 0 and links == 0,
                 f"{SAMPLE_SOURCE}: no empty text, no account name (@...), no link (http://, https://, www.)",
                 f"empty {empty}, account names {handles}, links {links}")
    labels = sample["sentiment"].map(RAW_TO_LABEL)
    per_class = {label: int((labels == label).sum()) for label in (0, 1, 2)}
    ticker = notebook_ticker(texts)
    with_ticker = {label: int(((labels == label) & pd.Series(ticker or [False] * len(sample))).sum()) for label in (0, 2)}
    level_rows = level_file_rows(LEVEL_FULL_FILE) or []
    level_of = {item[0]: item[1] for item in level_rows}
    ids = [f"tim:{row}" for row in rows]
    per_level = {level: sum(level_of.get(post_id) == level for post_id in ids) for level in LEVEL_SIX}
    report.check(sample["sentiment"].isin(list(RAW_TO_LABEL)).all() and min(per_class.values()) >= SAMPLE_MIN_PER_CLASS,
                 f"{SAMPLE_SOURCE}: every class has at least {SAMPLE_MIN_PER_CLASS} posts after the conversion",
                 f"{per_class} (0 bearish, 1 neutral, 2 bullish)")
    report.check(bool(level_rows) and min(per_level.values()) >= SAMPLE_MIN_PER_LEVEL,
                 f"{SAMPLE_SOURCE}: every level of {LEVEL_FULL_FILE} has at least {SAMPLE_MIN_PER_LEVEL} posts in it",
                 f"{per_level}")
    report.check(ticker is not None and min(with_ticker.values()) >= SAMPLE_MIN_TICKER_PER_SIDE,
                 f"{SAMPLE_SOURCE}: the bearish and the bullish class each have at least {SAMPLE_MIN_TICKER_PER_SIDE} "
                 "posts with a ticker (the notebook's own rule), so the run with TICKER_ONLY and two classes can split",
                 f"{with_ticker}")
    hash_of = {item[0]: item[5] for item in level_rows}
    compared = [(post_id, text) for post_id, text in zip(ids, sample["tweet"]) if post_id in hash_of]
    differ = [post_id for post_id, text in compared if own_text_hash(text) != hash_of[post_id]]
    report.check(bool(compared) and not differ,
                 f"{SAMPLE_SOURCE}: every post that is also in {LEVEL_FULL_FILE} has that file's text fingerprint",
                 f"{len(compared)} compared, {len(differ)} differ {differ[:3]}")
    n_none = sum(level_of.get(post_id) == "none" for post_id in ids)
    appended_levels = sorted({level_of.get(f"tim:{row}") for row in appended}, key=str)
    report.check(bool(level_rows) and n_none >= SAMPLE_MIN_NONE and appended_levels in ([], ["none"]),
                 f"{SAMPLE_SOURCE}: at least {SAMPLE_MIN_NONE} posts labelled none in {LEVEL_FULL_FILE}, and every row "
                 "appended after the first part is labelled none",
                 f"{n_none} labelled none; {len(appended)} appended, their labels {appended_levels}")
    report.info(f"sample: {len(sample)} posts; per class {per_class}; per level {per_level}; labelled none {n_none} "
                f"({len(appended)} appended); with a ticker (bearish, bullish) {with_ticker}")


def check_fingerprints_full(report, source):
    """Full check: the notebook's own reading cell reads the local copy; the level file, the review table and the
    sample are compared with it."""
    import pandas as pd
    namespace, problems = reading_namespace()
    path = REPO_DIR / source
    if not report.check(not problems and path.is_file(), "the notebook's reading cell reads the local copy of the dataset",
                        "; ".join(problems) or f"{source} exists: {path.is_file()}"):
        return
    posts = namespace["read_posts"](str(path))
    text_of = dict(zip(posts["post_id"], posts["text"]))
    label_of = dict(zip(posts["post_id"], posts["label"]))
    for relative in (LEVEL_FULL_FILE, LEVEL_REVIEW_FILE):
        rows = level_file_rows(relative)
        if rows is None:
            report.check(False, f"{relative} is there to compare with the dataset")
            continue
        missing = [item[0] for item in rows if item[0] not in text_of]
        changed = [item[0] for item in rows if len(item) > 5 and item[0] in text_of and own_text_hash(text_of[item[0]]) != item[5]]
        hashed = sum(len(item) > 5 for item in rows)
        report.check(not missing and not changed,
                     f"{relative}: every id is a post of the dataset read by the notebook"
                     + (", with the same text fingerprint" if hashed else ""),
                     f"{len(rows)} ids, {hashed} fingerprints compared; {len(missing)} missing {missing[:3]}; "
                     f"{len(changed)} differ {changed[:3]}")
    sample = namespace["read_posts"](str(REPO_DIR / SAMPLE_SOURCE))
    differ = [row.post_id for row in sample.itertuples()
              if text_of.get(row.post_id) != row.text or label_of.get(row.post_id) != row.label]
    report.check(len(sample) > 0 and not differ,
                 f"{SAMPLE_SOURCE}: every post has the id, the text and the label it has in the dataset",
                 f"{len(sample)} posts compared, {len(differ)} differ {differ[:3]}")


# ---------------------------------------------------------------- level pre-labels (scripts/prelabel_levels.py)
# The checks never contact Ollama, and the quick check never reads the dataset: the draw is checked on a
# hand-written table and the labelling loop with a fake model. The full check compares the files with the dataset.

LEVEL_LABELS = ["assets_dividends", "earnings", "revenue_business", "price_chart", "story_macro", "emotion_only",
                "none"]  # the seven labels, closest to cash first (interface of the label files)
LEVEL_SIX = LEVEL_LABELS[:6]  # the levels of the ladder; none is not one
LEVEL_FAILURES = ["unparsed", "disputed"]  # no vote at all; too few votes for the most frequent label
LEVEL_COLUMNS = ["post_id", "level", "votes", "model", "prompt_version", "text_hash"]  # label files, in order
LEVEL_REVIEW_COLUMNS = ["post_id", "prelabel", "human_label"]  # review table, in order
LEVEL_DEFAULT_MODEL = "qwen2.5:7b"
LEVEL_FULL_FILE = "labels/levels-prelabel-v2.csv"  # the prompt version is part of the name
LEVEL_SAMPLE_FILE = "samples/levels-prelabel-sample.csv"
LEVEL_REVIEW_FILE = "labels/levels-review-template.csv"
LEVEL_DEFAULTS = {"DEFAULT_REVIEW_SIZE": 200, "DEFAULT_SEED": 42, "DEFAULT_SAMPLES": 5, "DEFAULT_TEMPERATURE": 0.7,
                  "DEFAULT_MIN_VOTES": 3, "DEFAULT_COUNT": 8000, "DEFAULT_POOL_CAP": 1500}
LEVEL_COUNT_RANGE = (6000, 10000)  # the default count must stay in this range
LEVEL_DATASET = "TimKoornstra/financial-tweets-sentiment"
LEVEL_DATASET_REVISION = "bba602698fd6f18226f6404fd8fb637ef67fa1eb"
LEVEL_CORE_KEYWORDS = {"assets_dividends": ["dividend", "p/b", "book value"], "earnings": ["p/e", "pe ratio"],
                       "revenue_business": ["revenue"]}  # may never be removed from the keyword pools
# SHA-256 of everything fixed that is sent to the model, per released prompt version (see level_fingerprint_of).
# v1 (the reason the writer gives) was never used for a committed file; v2 (what the post pays attention to) is
# the first released version.
LEVEL_PROMPT_FINGERPRINTS = {
    "v2": "3f8c06a7371553ff57b3894fbfabf2308743542c5089b226d0a40ab8ea0ce7fa",
}
LEVEL_SAMPLE_SAMPLES = 3  # the committed sample was labelled with 3 samples and 2 votes
LEVEL_SAMPLE_MIN_VOTES = 2
LEVEL_PRELABEL_PACKAGES = ["pandas", "numpy", "pyarrow", "huggingface-hub", "fsspec"]  # requirements-prelabel.txt
LEVEL_LF_PATTERNS = ["*.sh", "*.csv", "*.py", "*.json", "*.txt", "*.md"]  # .gitattributes: text eol=lf
LEVEL_PROBE_DIR = REPO_DIR / "data" / "checks" / "levels-probe"
ENV_PROBE_DIR = REPO_DIR / "data" / "checks" / "env-probe"
LEVEL_FULL_SKIPPED = ("level pre-labels: labels/levels-prelabel-v2.csv not obtained (labelled on another machine); "
                      "skipped")
LEVEL_REVIEW_SKIPPED = ("level pre-labels: labels/levels-review-template.csv not obtained (labelled on another "
                        "machine); skipped")
LEVEL_MARK = "SENTIMENT-MARK-7319"  # a conspicuous sentiment value that must never reach a request


def level_answer(label):
    return json.dumps({"level": label})


def level_fingerprint_of():
    """The documented prompt fingerprint, computed here from what the script sends: SHA-256 of the compact ASCII JSON
    (sorted keys) of {"first": the messages of a first request, "retry": the messages of a retry, "format": the answer
    format}, with the script's placeholder in place of the post."""
    first = levels.build_payload(levels.POST_PLACEHOLDER, retry=False, model="", seed=0, temperature=0)
    again = levels.build_payload(levels.POST_PLACEHOLDER, retry=True, model="", seed=0, temperature=0)
    text = json.dumps({"first": first["messages"], "retry": again["messages"], "format": first["format"]},
                      ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def level_stops(function, *args, **kwargs):
    """The message of the Stop raised by function, or None when it does not stop."""
    try:
        function(*args, **kwargs)
    except levels.Stop as exc:
        return str(exc)
    return None


def check_levels_interface(report):
    """The script's interface: labels, failure values, columns, defaults, file paths, dataset and core keywords."""
    parsed = levels.build_parser().parse_args(["label"])
    review = levels.build_parser().parse_args(["review-template"])
    pools = dict(levels.KEYWORD_POOLS)
    interface = {
        "labels": (levels.LEVEL_NAMES, LEVEL_LABELS),
        "six levels": (levels.SIX_LEVELS, LEVEL_SIX),
        "failure values": ([levels.UNPARSED, levels.DISPUTED], LEVEL_FAILURES),
        "columns": (levels.COLUMNS, LEVEL_COLUMNS),
        "review columns": (levels.REVIEW_COLUMNS, LEVEL_REVIEW_COLUMNS),
        "default model": ((levels.DEFAULT_MODEL, parsed.model), (LEVEL_DEFAULT_MODEL, LEVEL_DEFAULT_MODEL)),
        "full label file": (levels.default_output("full"), LEVEL_FULL_FILE),
        "sample label file": (levels.default_output("sample"), LEVEL_SAMPLE_FILE),
        "review table": (levels.default_output("review"), LEVEL_REVIEW_FILE),
        "dataset": ((levels.DATASET, levels.DATASET_REVISION), (LEVEL_DATASET, LEVEL_DATASET_REVISION)),
        "pool order": ([pool for pool, _ in levels.KEYWORD_POOLS], LEVEL_SIX[:3]),
        "count range": (levels.COUNT_RANGE, LEVEL_COUNT_RANGE),
    }
    interface.update({name: (getattr(levels, name), value) for name, value in LEVEL_DEFAULTS.items()})
    parser_defaults = {"DEFAULT_SAMPLES": parsed.samples, "DEFAULT_TEMPERATURE": parsed.temperature,
                       "DEFAULT_MIN_VOTES": parsed.min_votes, "DEFAULT_COUNT": parsed.count,
                       "DEFAULT_POOL_CAP": parsed.pool_cap, "DEFAULT_SEED": parsed.seed,
                       "DEFAULT_REVIEW_SIZE": review.size}
    interface.update({f"parser {name}": (value, LEVEL_DEFAULTS[name]) for name, value in parser_defaults.items()})
    interface["review seed"] = (review.seed, LEVEL_DEFAULTS["DEFAULT_SEED"])
    wrong = [f"{name}: {got!r}, expected {expected!r}" for name, (got, expected) in interface.items() if got != expected]
    low, high = LEVEL_COUNT_RANGE
    if not low <= levels.DEFAULT_COUNT <= high:
        wrong.append(f"default count {levels.DEFAULT_COUNT} outside {LEVEL_COUNT_RANGE}")
    missing = [f"{pool}: {word}" for pool, words in LEVEL_CORE_KEYWORDS.items() for word in words
               if word not in pools.get(pool, [])]
    report.check(not wrong and not missing,
                 "level pre-labels: the script's seven labels (in order), unparsed and disputed, the six columns, the "
                 "review table's three columns, the default model, the three file paths, the review size and seed, "
                 "five samples, temperature 0.7, three votes, 8,000 posts (within 6,000 to 10,000), a pool cap of "
                 "1,500, the dataset and its pinned revision, and the four core keyword groups are the documented "
                 "interface", "; ".join(wrong + [f"core keyword missing: {item}" for item in missing]))


def check_levels_prompt(report):
    """Prompt v1 is pinned; a request carries the system message, the examples and the post, the seed and the
    temperature it is given, and an answer format that allows only the seven labels; a retry adds the reminder."""
    computed = level_fingerprint_of()
    version = levels.PROMPT_VERSION
    report.check(version in LEVEL_PROMPT_FINGERPRINTS and computed == LEVEL_PROMPT_FINGERPRINTS[version]
                 and levels.prompt_fingerprint() == computed,
                 f"level pre-labels: prompt {version} (system message with the ladder and the rules, hand-written "
                 "examples, retry reminder, answer format) has the pinned SHA-256; a changed character needs a new "
                 "version", f"computed {computed}, script {levels.prompt_fingerprint()}, pinned "
                 f"{LEVEL_PROMPT_FINGERPRINTS.get(version)!r}")
    answer_rules = {"type": "object", "properties": {"level": {"type": "string", "enum": LEVEL_LABELS}},
                    "required": ["level"]}
    problems = []
    for temperature, seed in ((0.7, 42), (0.25, 107)):
        first = levels.build_payload("a post", retry=False, model=LEVEL_DEFAULT_MODEL, seed=seed,
                                     temperature=temperature)
        again = levels.build_payload("a post", retry=True, model=LEVEL_DEFAULT_MODEL, seed=seed,
                                     temperature=temperature)
        for name, payload in (("first", first), ("retry", again)):
            options = payload.get("options", {})
            if sorted(payload) != ["format", "messages", "model", "options", "stream"]:
                problems.append(f"{name}: fields {sorted(payload)}")
            if payload.get("stream") is not False or options.get("temperature") != temperature \
                    or options.get("seed") != seed or payload.get("model") != LEVEL_DEFAULT_MODEL:
                problems.append(f"{name}: stream {payload.get('stream')!r}, options {options}")
            if not is_count(options.get("num_predict")) or options["num_predict"] < 16:
                problems.append(f"{name}: num_predict {options.get('num_predict')!r}")
            if payload.get("format") != answer_rules:
                problems.append(f"{name}: format {payload.get('format')}")
            messages = payload["messages"]
            examples = [{"role": role, "content": content} for item in levels.EXAMPLES_V2
                        for role, content in (("user", levels.USER_PREFIX + item["post"]),
                                              ("assistant", level_answer(item["level"])))]
            if messages[0]["role"] != "system" or not messages[0]["content"].startswith(levels.system_prompt()):
                problems.append(f"{name}: the first message is not the system message")
            if messages[1:-1] != examples:
                problems.append(f"{name}: the hand-written examples are not the messages between the system message "
                                "and the post")
            if messages[-1] != {"role": "user", "content": levels.USER_PREFIX + "a post"}:
                problems.append(f"{name}: the post is not the last message")
        reminder = levels.retry_reminder()
        if reminder in first["messages"][0]["content"] \
                or first["messages"][0]["content"] + "\n\n" + reminder != again["messages"][0]["content"] \
                or first["messages"][1:] != again["messages"][1:]:
            problems.append("a retry must differ from the first request only by the reminder at the end of the system "
                            "message")
    system = levels.system_prompt()
    missing = [label for label in LEVEL_LABELS if label not in system]
    covered = {item["level"] for item in levels.EXAMPLES_V2}
    if missing or set(LEVEL_LABELS) - covered or covered - set(LEVEL_LABELS):
        problems.append(f"labels missing from the system message {missing}; labels without an example "
                        f"{sorted(set(LEVEL_LABELS) - covered)}; unknown example labels {sorted(covered - set(LEVEL_LABELS))}")
    entries = [f"\n{number}. {label}: " for number, label in enumerate(LEVEL_SIX, start=1)] + [f"\n{LEVEL_LABELS[6]}: "]
    places = [system.find(entry) for entry in entries]
    if -1 in places or places != sorted(places):
        problems.append("the system message does not list the labels in ladder order (1. to 6., then none)")
    report.check(not problems, "level pre-labels: a request is not streamed and has the model, the seed and the "
                 "temperature it is given, room for the longest answer and an answer format that allows only the "
                 "seven labels; the system message lists the seven labels in ladder order, then come the hand-written "
                 "examples (at least one per label), then the post; a retry adds only the fixed reminder",
                 "; ".join(problems))


def check_levels_hosts(report):
    """The label script talks only to a loopback address, without a proxy."""
    import urllib.parse
    import urllib.request

    refused = [host for host in REFUSED_HOSTS if level_stops(levels.loopback_base, host) is None]
    accepted = [host for host in ACCEPTED_HOSTS if level_stops(levels.loopback_base, host) is not None]
    default_host = levels.build_parser().parse_args(["label"]).host
    default_ok = default_host == levels.DEFAULT_HOST and urllib.parse.urlsplit(default_host).hostname == "127.0.0.1"
    report.check(not refused and not accepted and default_ok,
                 "level pre-labels: the script refuses every address that is not http:// on a loopback host "
                 f"({len(REFUSED_HOSTS)} cases), accepts 127.0.0.1, localhost and ::1, and defaults to 127.0.0.1",
                 f"not refused {refused}; not accepted {accepted}; default {default_host!r}")
    result = subprocess.run([sys.executable, str(REPO_DIR / "scripts" / "prelabel_levels.py"), "label",
                             "--host", "http://example.com:11434"], capture_output=True, text=True, timeout=120)
    report.check(result.returncode == 1 and "refusing" in result.stderr and not result.stdout,
                 "level pre-labels: the label command with an outside address stops before doing anything",
                 f"exit {result.returncode}, stderr {result.stderr.strip()[:160]!r}, stdout {result.stdout.strip()[:80]!r}")
    names = ["http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY", "no_proxy", "NO_PROXY"]
    saved = {name: os.environ.get(name) for name in names}
    try:
        for name in names[:6]:
            os.environ[name] = "http://127.0.0.1:9"
        for name in names[6:]:
            os.environ.pop(name, None)
        openers = [urllib.request.build_opener(), levels.no_proxy_opener(),
                   levels.OllamaClient(levels.DEFAULT_HOST, 1).opener]
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
                 "level pre-labels: requests ignore proxy settings (with proxy variables set, a default opener would "
                 "use the proxy; the script's opener has no proxy at all)",
                 f"default opener {proxies[0]}, script {proxies[1]}, client {proxies[2]}")


LEVEL_DRAW_TEXTS = [  # a hand-written table; the comment says where each row must end up
    "dividend raised again, nice",           # 0: pool assets_dividends
    None,                                    # 1: empty (no value)
    "   ",                                   # 2: empty after stripping
    "P/B under 1 here, cheap",               # 3: assets_dividends (upper case, slash)
    "p/e of 12 and revenue growing",         # 4: earnings and revenue_business -> earnings
    "dividend, earnings and revenue all up",  # 5: three pools -> assets_dividends
    "dividendo alto en este banco",          # 6: no pool (dividendo is not dividend)
    "  dividend raised again, nice  ",       # 7: repeats row 0 after stripping -> left out
    "Dividend raised again, nice",           # 8: case differs from row 0 -> kept, assets_dividends
    "earnings beat on the top line",         # 9: earnings
    "eps came in light",                     # 10: earnings
    "profit margins under pressure",         # 11: earnings
    "guidance raised for the year",          # 12: earnings
    "sales were flat this quarter",          # 13: revenue_business
    "users keep growing every month",        # 14: revenue_business
] + [f"chart talk number {number}, looking good" for number in range(15, 30)]  # 15-29: no pool
LEVEL_DRAW_POOLS = {"assets_dividends": [0, 3, 5, 8], "earnings": [4, 9, 10, 11, 12], "revenue_business": [13, 14]}


def check_levels_draw(report):
    """The draw on a hand-written table: ids, empty and repeated texts, pools, cap, fill, seed and row order."""
    import pandas as pd

    table = pd.DataFrame({"tweet": LEVEL_DRAW_TEXTS, "sentiment": [LEVEL_MARK] * len(LEVEL_DRAW_TEXTS),
                          "url": ["https://example.com/source"] * len(LEVEL_DRAW_TEXTS)})
    posts = levels.posts_from_table(table)
    problems = []
    if [post["post_id"] for post in posts] != [f"tim:{row}" for row in range(len(LEVEL_DRAW_TEXTS))]:
        problems.append("ids are not tim:<row from 0>")
    if any(sorted(post) != ["post_id", "text"] for post in posts) or posts[1]["text"] != "":
        problems.append("posts must carry only post_id and text, an empty value as ''")
    pools = {pool: [row for row, text in enumerate(LEVEL_DRAW_TEXTS)
                    if text and text.strip() and row != 7 and levels.first_pool(text) == pool]  # row 7 repeats row 0
             for pool in LEVEL_SIX[:3]}
    if pools != LEVEL_DRAW_POOLS:
        problems.append(f"pools {pools}, expected {LEVEL_DRAW_POOLS} (rows with several pools go to the first, P/B "
                        "and p/e match, dividendo does not)")
    selection, summary, steps = levels.select_posts(posts, count=20, pool_cap=3, seed=42)
    ids = [post["post_id"] for post in selection]
    rows = [int(post_id.split(":")[1]) for post_id in ids]
    if (summary["empty_texts_left_out"], summary["repeated_texts_left_out"], summary["usable_posts"]) != (2, 1, 27):
        problems.append(f"empty {summary['empty_texts_left_out']}, repeated {summary['repeated_texts_left_out']}, usable "
                        f"{summary['usable_posts']} (expected 2, 1, 27)")
    if {1, 2, 7} & set(rows) or len(set(ids)) != len(ids) or len(ids) != 20:
        problems.append(f"selected rows {sorted(rows)}: 20 different posts without rows 1, 2 and 7")
    if summary["pool_posts"] != {pool: len(members) for pool, members in LEVEL_DRAW_POOLS.items()}:
        problems.append(f"pool posts {summary['pool_posts']}")
    expected_steps = {"assets_dividends": 3, "earnings": 3, "revenue_business": 2, "random": 12}
    if summary["taken_per_step"] != expected_steps or Counter(steps.values()) != Counter(expected_steps):
        problems.append(f"taken per step {summary['taken_per_step']}, expected {expected_steps} (cap 3)")
    for post_id, step in steps.items():
        row = int(post_id.split(":")[1])
        if step != "random" and row not in LEVEL_DRAW_POOLS[step]:
            problems.append(f"{post_id} taken for {step} without being in that pool")
    fill = [int(post_id.split(":")[1]) for post_id, step in steps.items() if step == "random"]
    pool_taken = {int(post_id.split(":")[1]) for post_id, step in steps.items() if step != "random"}
    if pool_taken & set(fill):
        problems.append("the fill took a post that a pool had already taken")
    left_over = [row for members in LEVEL_DRAW_POOLS.values() for row in members if row not in pool_taken]
    if len(left_over) != 3:
        problems.append(f"posts above the cap {left_over} (expected 3)")
    if summary["fingerprint"] != hashlib.sha256("\n".join(ids).encode("ascii")).hexdigest() \
            or summary["first_ids"] != ids[:5]:
        problems.append("the selection fingerprint is not the SHA-256 of the ids in processing order joined by "
                        "newlines")
    if any(sorted(post) != ["post_id", "text"] for post in selection):
        problems.append("the selection must carry only post_id and text")
    again, again_summary, _ = levels.select_posts(posts, count=20, pool_cap=3, seed=42)
    other, other_summary, _ = levels.select_posts(posts, count=20, pool_cap=3, seed=43)
    if again != selection or again_summary["fingerprint"] != summary["fingerprint"]:
        problems.append("the same seed gave another draw")
    if other_summary["fingerprint"] == summary["fingerprint"]:
        problems.append("another seed gave the same draw")
    shuffled = list(posts)
    random.Random(7).shuffle(shuffled)
    mixed, mixed_summary, mixed_steps = levels.select_posts(shuffled, count=20, pool_cap=3, seed=42)
    if mixed != selection or mixed_summary["fingerprint"] != summary["fingerprint"] or mixed_steps != steps:
        problems.append("shuffling the rows of the table changed the draw (it must depend on the rows only)")
    whole, whole_summary, _ = levels.select_posts(posts, count=100, pool_cap=10, seed=42)
    if len(whole) != 27 or whole_summary["taken_per_step"]["random"] != 16:
        problems.append(f"with room for every post, all 27 usable posts are taken ({len(whole)})")
    message = level_stops(levels.select_posts, posts, count=7, pool_cap=3, seed=42)
    if message is None or "pools" not in message:
        problems.append(f"pools holding more than the count must stop the draw ({message!r})")
    report.check(not problems, "level pre-labels: the draw on a hand-written table: ids tim:<row from 0>; empty and "
                 "repeated texts (same after stripping, case counts) left out, the smallest row kept; keywords match "
                 "whole words in any case (P/B, p/e; not dividendo); a post in several pools belongs to the first; a "
                 "pool above the cap is drawn down to it; the fill comes from every post not taken yet; pools above "
                 "the count stop the draw; the same seed gives the same draw and another seed another; shuffled rows "
                 "give the same draw and fingerprint; posts carry only id and text", "; ".join(problems))


class LevelFake:
    """Stands in for Ollama: answers every post from a script {post text: [answer, ...]}, the last answer repeating.
    An answer is the reply text, or an exception class to raise. Records every request."""

    def __init__(self, script):
        self.script = {text: list(answers) for text, answers in script.items()}
        self.requests = []

    @staticmethod
    def post_of(payload):
        content = payload["messages"][-1]["content"]
        return content[len(levels.USER_PREFIX):] if content.startswith(levels.USER_PREFIX) else content

    def __call__(self, payload):
        self.requests.append(payload)
        answers = self.script.get(self.post_of(payload)) or [level_answer("none")]
        answer = answers.pop(0) if len(answers) > 1 else answers[0]
        if isinstance(answer, type) and issubclass(answer, Exception):
            raise answer("fake model")
        return answer

    def asked(self, text=None):
        """[(post text, seed, retry reminder present), ...] in request order (of one post when text is given)."""
        reminder = levels.retry_reminder()
        return [(self.post_of(payload), payload["options"]["seed"], reminder in payload["messages"][0]["content"])
                for payload in self.requests if text is None or self.post_of(payload) == text]


def level_probe_rows(path):
    return [row for row in csv.reader(path.read_text(encoding="utf-8").splitlines())][1:]


LEVEL_FAKE_POSTS = [  # hand-written posts for the fake model (never posts of the dataset)
    "probe post zero: five votes alike",
    "probe post one: four against one",
    "probe post two: three, one and one",
    "probe post three: two, two and one",
    "probe post four: one sample fails, then three against one",
    "probe post five: every sample fails",
    "probe post six: labels in another case and with spaces",
    "probe post seven: two bad attempts, then a good one",
    "probe post eight: one sample times out three times",
]


def check_levels_fake_model(report):
    """Voting, retries, stopping, resuming and the settings record of the label script, driven by a fake model on
    hand-written posts; every file is written under data/checks/levels-probe/ and removed afterwards."""
    import pandas as pd

    folder = LEVEL_PROBE_DIR / "fake"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    records = REPO_DIR / levels.SETTINGS_DIR
    records_before = sorted(path.name for path in records.glob("*")) if records.is_dir() else []
    table = pd.DataFrame({"tweet": LEVEL_FAKE_POSTS, "sentiment": [LEVEL_MARK] * len(LEVEL_FAKE_POSTS),
                          "url": [f"https://example.com/{LEVEL_MARK}"] * len(LEVEL_FAKE_POSTS)})
    posts = levels.posts_from_table(table)
    texts = [post["text"] for post in posts]
    ids = [post["post_id"] for post in posts]
    text_by_id = dict(zip(ids, texts))
    waits, logged = [], []
    common = {"text_by_id": text_by_id, "connect_retries": 2, "connect_wait": 0.25, "log": logged.append,
              "sleep": waits.append, "seed": 100, "temperature": 0.55}
    good, bad = level_answer, "not json at all"
    E, P, S = "earnings", "price_chart", "story_macro"

    def row(index, level, votes, model=LEVEL_DEFAULT_MODEL):
        return [ids[index], level, votes, model, levels.PROMPT_VERSION, own_text_hash(texts[index])]

    def record(name):
        return folder / f"{name}.settings.json"

    fake = LevelFake({texts[0]: [good(E)], texts[1]: [good(E)] * 4 + [good(P)],
                      texts[2]: [good(E)] * 3 + [good(P), good(S)], texts[3]: [good(E), good(E), good(P), good(P), good(S)],
                      texts[4]: [bad, '{"level": 5}', '{"label": "earnings"}', good(E), good(E), good(E), good(P)],
                      texts[5]: [bad], texts[6]: ['{"level": "  PRICE_CHART \\n"}', '{"level": "Price_Chart"}'],
                      texts[7]: [bad, '{"level": "growth"}', good(E), good(E)],
                      texts[8]: [levels.AnswerTimeout] * 3 + [good(S)]})
    first_file = folder / "first.csv"
    summary = levels.label_posts(posts, first_file, fake, settings_path=record("first"), **common)
    expected = [row(0, E, "5/5"), row(1, E, "4/5"), row(2, E, "3/5"), row(3, "disputed", "2/5"), row(4, E, "3/4"),
                row(5, "unparsed", "0/0"), row(6, P, "5/5"), row(7, E, "5/5"), row(8, S, "4/4")]
    rows = level_probe_rows(first_file)
    report.check(rows == expected and summary["stopped"] is None and summary["labelled"] == 9,
                 "level pre-labels, fake model: five votes alike give 5/5, four to one 4/5, three, one and one 3/5; "
                 "two, two and one give disputed 2/5; a sample whose three attempts fail is not a vote (three to one "
                 "of four: 3/4); every sample failing gives unparsed 0/0; another case and spaces count; two bad "
                 "attempts and a good one make a vote; a sample that times out three times fails alone; every row "
                 "has the model, the prompt version and the fingerprint", f"rows {rows}")
    asked7 = fake.asked(texts[7])
    asked0 = fake.asked(texts[0])
    seeds_ok = [seed for _, seed, _ in asked0] == [100, 101, 102, 103, 104] \
        and [(seed, retry) for _, seed, retry in asked7] == [(100, False), (100, True), (100, True), (101, False),
                                                             (102, False), (103, False), (104, False)]
    counts = Counter(text for text, _, _ in fake.asked())
    settings = [index for index, payload in enumerate(fake.requests)
                if payload["model"] != LEVEL_DEFAULT_MODEL or payload["stream"] is not False
                or payload["options"].get("temperature") != 0.55
                or payload["format"]["properties"]["level"]["enum"] != LEVEL_LABELS]
    leaked = [index for index, payload in enumerate(fake.requests) if LEVEL_MARK in json.dumps(payload)]
    report.check(seeds_ok and counts[texts[5]] == 15 and counts[texts[8]] == 7 and not settings and not leaked
                 and summary["requests"] == len(fake.requests),
                 "level pre-labels, fake model: sample i asks with the seed + i, its retries keep that seed and carry "
                 "the reminder (first attempts do not); at most 3 attempts per sample (15 for a post whose samples all "
                 "fail); every request has the model, the temperature given and the seven-label format; the "
                 "dataset's sentiment (a conspicuous value in the table) is in no request",
                 f"seeds of post 0 {[seed for _, seed, _ in asked0]}, post 7 {asked7}; requests per post "
                 f"{[counts[text] for text in texts]}; other settings {settings}; sentiment in requests {leaked}")

    # The server cannot be reached: stop without writing; a short outage is ridden out.
    fake = LevelFake({texts[0]: [good(E)], texts[1]: [good(E), good(E), levels.ServerUnavailable],
                      texts[2]: [good(E)]})
    waits.clear()
    stop_file = folder / "stop.csv"
    summary = levels.label_posts(posts[:3], stop_file, fake, settings_path=record("stop"), **common)
    asked = [text for text, _, _ in fake.asked()]
    report.check(level_probe_rows(stop_file) == [row(0, E, "5/5")] and bool(summary["stopped"])
                 and waits == [0.25, 0.25] and asked == [texts[0]] * 5 + [texts[1]] * 5 and texts[2] not in asked,
                 "level pre-labels, fake model: when the server cannot be reached the run waits and tries twice more, "
                 "then stops the whole run; that post is not written (although two samples were answered) and no "
                 "later post is asked", f"rows {level_probe_rows(stop_file)}, stopped {summary['stopped']!r}, waits "
                 f"{waits}, asked {len(asked)}")
    fake = LevelFake({texts[0]: [good(E)], texts[1]: [good(E), levels.ServerUnavailable, good(E)], texts[2]: [good(E)]})
    waits.clear()
    outage_file = folder / "outage.csv"
    summary = levels.label_posts(posts[:3], outage_file, fake, settings_path=record("outage"), **common)
    report.check(level_probe_rows(outage_file) == [row(0, E, "5/5"), row(1, E, "5/5"), row(2, E, "5/5")]
                 and summary["stopped"] is None and waits == [0.25],
                 "level pre-labels, fake model: after a short outage the same sample is asked again and the run goes on",
                 f"rows {level_probe_rows(outage_file)}, waits {waits}")

    # Resuming: --limit, a second run, a partial last line.
    alike = {text: [good(E)] for text in texts}
    resume_file = folder / "resume.csv"
    levels.label_posts(posts, resume_file, LevelFake(alike), settings_path=record("resume"), limit=2, **common)
    after_limit = [item[0] for item in level_probe_rows(resume_file)]
    fake = LevelFake(alike)
    summary = levels.label_posts(posts, resume_file, fake, settings_path=record("resume"), **common)
    resumed = level_probe_rows(resume_file)
    report.check(after_limit == ids[:2] and resumed == [row(index, E, "5/5") for index in range(9)]
                 and sorted(set(text for text, _, _ in fake.asked())) == sorted(texts[2:])
                 and summary["already_in_file"] == 2,
                 "level pre-labels, resume: --limit 2 labels two posts; the next run skips them (they are not asked "
                 "again) and appends the others, each id once",
                 f"after --limit {after_limit}; then {[item[0] for item in resumed]}")
    raw = resume_file.read_bytes()
    resume_file.write_bytes(raw[:-7])  # a run stopped while writing: the last line loses its end
    fake = LevelFake(alike)
    summary = levels.label_posts(posts, resume_file, fake, settings_path=record("resume"), **common)
    report.check(level_probe_rows(resume_file) == resumed and resume_file.read_bytes() == raw
                 and summary["dropped_partial_line_bytes"] == len(raw.split(b"\n")[-2]) - 6
                 and [text for text, _, _ in fake.asked()] == [texts[8]] * 5,
                 "level pre-labels, resume: a partial last line is dropped and that post labelled again; the file ends "
                 "as it would have", f"dropped {summary['dropped_partial_line_bytes']} bytes, asked {len(fake.requests)}")

    # --retry-unparsed: only unparsed rows, in place; a disputed result replaces, unparsed again stays.
    retry_file = folder / "retry.csv"
    fake = LevelFake({texts[0]: [bad], texts[1]: [good(E), good(E), good(P), good(P), good(S)], texts[2]: [bad],
                      texts[3]: [good(E)]})
    levels.label_posts(posts[:4], retry_file, fake, settings_path=record("retry"), **common)
    before = level_probe_rows(retry_file)
    fake = LevelFake({texts[0]: [good(P), good(P), good(S), good(S), good(E)], texts[2]: [bad]})
    summary = levels.label_posts(posts[:4], retry_file, fake, settings_path=record("retry"), retry_unparsed=True,
                                 **common)
    after = level_probe_rows(retry_file)
    report.check(before == [row(0, "unparsed", "0/0"), row(1, "disputed", "2/5"), row(2, "unparsed", "0/0"),
                            row(3, E, "5/5")]
                 and after == [row(0, "disputed", "2/5"), row(1, "disputed", "2/5"), row(2, "unparsed", "0/0"),
                               row(3, E, "5/5")]
                 and sorted(set(text for text, _, _ in fake.asked())) == sorted([texts[0], texts[2]])
                 and summary["replaced"] == 1 and not list(folder.glob(".*.tmp")),
                 "level pre-labels: --retry-unparsed asks only the unparsed rows (never disputed ones); a new result "
                 "that is not unparsed (disputed too) replaces its row in place, unparsed again leaves it; same row "
                 "count and order, no temporary file left", f"before {before}; after {after}")

    # Files that must stop the run before anything is asked or written.
    wrong = []
    for what, column, value in (("prompt version", 4, "v0"), ("model", 3, "another:1b")):
        mixed = folder / "mixed.csv"
        changed = [item[:column] + [value] + item[column + 1:] for item in level_probe_rows(first_file)]
        mixed.write_bytes(levels.csv_bytes([LEVEL_COLUMNS] + changed))
        before_bytes = mixed.read_bytes()
        fake = LevelFake({})
        message = level_stops(levels.label_posts, posts, mixed, fake, settings_path=record("mixed"), **common)
        if message is None or mixed.read_bytes() != before_bytes or fake.requests:
            wrong.append(what)
        record("mixed").unlink(missing_ok=True)
    report.check(not wrong, "level pre-labels: an existing file from another prompt version or another model stops "
                 "the run before anything is asked or written", f"not stopped: {wrong}")

    # The settings record.
    problems = []
    settings_file = folder / "settings.csv"
    levels.label_posts(posts[:2], settings_file, LevelFake(alike), settings_path=record("settings"), samples=3,
                       min_votes=2, **common)
    stored = json.loads(record("settings").read_text(encoding="utf-8"))
    if stored.get("samples") != 3 or stored.get("seed") != 100 or stored.get("min_votes") != 2:
        problems.append(f"the record holds {stored}")
    seeded = {**common, "seed": 7}
    for what, options, word in (("seed", {**seeded, "samples": 3, "min_votes": 2}, "seed"),
                                ("samples", {**common, "samples": 5, "min_votes": 3}, "samples")):
        before_bytes = settings_file.read_bytes()
        fake = LevelFake(alike)
        message = level_stops(levels.label_posts, posts, settings_file, fake, settings_path=record("settings"),
                              **options)
        if message is None or word not in message or fake.requests or settings_file.read_bytes() != before_bytes:
            problems.append(f"another {what} did not stop the run naming it ({message!r})")
    record("five").unlink(missing_ok=True)
    five_file = folder / "five.csv"
    levels.label_posts(posts[:2], five_file, LevelFake(alike), settings_path=record("five"), **common)
    record("five").unlink()
    fake = LevelFake(alike)
    message = level_stops(levels.label_posts, posts, five_file, fake, settings_path=record("five"), samples=3,
                          min_votes=2, **common)
    if message is None or "samples" not in message or fake.requests:
        problems.append(f"rows with more votes than --samples did not stop the run without a record ({message!r})")
    logged.clear()
    fake = LevelFake(alike)
    levels.label_posts(posts[:3], five_file, fake, settings_path=record("five"), **common)
    if not any(text.startswith("warning") for text in logged) or not record("five").is_file() \
            or [text for text, _, _ in fake.asked()] != [texts[2]] * 5:
        problems.append("rows without a settings record must give a warning, go on and write the record")
    fresh = folder / "fresh.csv"
    levels.label_posts(posts[:1], fresh, LevelFake(alike), settings_path=record("settings"), samples=5, min_votes=3,
                       **common)
    if json.loads(record("settings").read_text(encoding="utf-8")).get("samples") != 5:
        problems.append("a new, empty output file must take this run's settings")
    paths = {"a": levels.settings_record_path(LEVEL_PROBE_DIR / "a" / "out.csv"),
             "b": levels.settings_record_path(LEVEL_PROBE_DIR / "b" / "out.csv"),
             "full": levels.settings_record_path(REPO_DIR / LEVEL_FULL_FILE),
             "outside": levels.settings_record_path(REPO_DIR.parent / "elsewhere" / "out.csv")}
    if paths["a"] == paths["b"] or paths["full"] != REPO_DIR / levels.SETTINGS_DIR / \
            "labels__levels-prelabel-v2.csv.settings.json" \
            or not re.fullmatch(r"[0-9a-f]{16}\.settings\.json", paths["outside"].name) \
            or any(path.parent != REPO_DIR / levels.SETTINGS_DIR for path in paths.values()):
        problems.append(f"settings record paths {paths}")
    for samples, min_votes in ((5, 2), (5, 6), (3, 1), (0, 0), (4, 2), (2, 1)):
        target = folder / "never.csv"
        if level_stops(levels.label_posts, posts, target, LevelFake(alike), settings_path=record("never"),
                       samples=samples, min_votes=min_votes, **common) is None or target.exists():
            problems.append(f"--samples {samples} with --min-votes {min_votes} was accepted")
    for samples, min_votes in ((5, 3), (5, 5), (3, 2), (1, 1), (4, 3)):
        if level_stops(levels.check_votes, samples, min_votes) is not None:
            problems.append(f"--samples {samples} with --min-votes {min_votes} was refused")
    records_after = sorted(path.name for path in records.glob("*")) if records.is_dir() else []
    if records_after != records_before:
        problems.append(f"{levels.SETTINGS_DIR} changed: {sorted(set(records_after) ^ set(records_before))}")
    report.check(not problems, "level pre-labels: the settings record of a file holds the seed, the samples and the "
                 "votes; another seed or number of samples on a file with rows stops the run and names it; rows with "
                 "more votes than --samples stop the run even without a record; rows without a record give a warning "
                 "and a new record; an empty file takes the new settings; two files with the same name in two folders "
                 "have two records, under data/prelabel/ (outside the repository: a hash of the path); --min-votes must "
                 "be more than half of --samples and at most --samples; the probes leave data/prelabel/ unchanged",
                 "; ".join(problems))
    shutil.rmtree(folder, ignore_errors=True)


def run_level_check_file(*args):
    """check-file in this process: (exit code, standard output, standard error)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = levels.main(["check-file", *[str(arg) for arg in args]])
    return code, out.getvalue(), err.getvalue()


LEVEL_GOOD_ROWS = [  # a label file that passes check-file with --samples 5 --min-votes 3
    ["tim:10", "assets_dividends", "5/5", "qwen2.5:7b", "v1", "0a1b2c3d"],
    ["tim:11", "earnings", "4/5", "qwen2.5:7b", "v1", "1a1b2c3d"],
    ["tim:12", "revenue_business", "3/4", "qwen2.5:7b", "v1", "2a1b2c3d"],
    ["tim:13", "price_chart", "3/5", "qwen2.5:7b", "v1", "3a1b2c3d"],
    ["tim:14", "story_macro", "5/5", "qwen2.5:7b", "v1", "4a1b2c3d"],
    ["tim:15", "emotion_only", "5/5", "qwen2.5:7b", "v1", "5a1b2c3d"],
    ["tim:16", "none", "5/5", "qwen2.5:7b", "v1", "6a1b2c3d"],
    ["tim:17", "disputed", "2/5", "qwen2.5:7b", "v1", "7a1b2c3d"],
    ["tim:18", "unparsed", "0/0", "qwen2.5:7b", "v1", "8a1b2c3d"],
    ["tim:19", "earnings", "5/5", "qwen2.5:7b", "v1", "9a1b2c3d"],
]
LEVEL_THREE_ROWS = [  # labelled with 3 samples, except the last row, which counts 4 valid votes
    ["tim:20", "earnings", "3/3", "qwen2.5:7b", "v1", "aa1b2c3d"],
    ["tim:21", "story_macro", "2/3", "qwen2.5:7b", "v1", "ab1b2c3d"],
    ["tim:22", "disputed", "1/3", "qwen2.5:7b", "v1", "ac1b2c3d"],
    ["tim:23", "unparsed", "0/0", "qwen2.5:7b", "v1", "ad1b2c3d"],
    ["tim:24", "earnings", "3/4", "qwen2.5:7b", "v1", "ae1b2c3d"],
]
LEVEL_GOOD_REVIEW = [["tim:10", "assets_dividends", "assets_dividends"], ["tim:11", "earnings", ""],
                     ["tim:12", "revenue_business", "none"], ["tim:13", "price_chart", ""],
                     ["tim:14", "story_macro", "emotion_only"], ["tim:15", "emotion_only", ""]]


def changed_rows(rows, line, column, value):
    """A copy of the rows with one cell changed; line counts as in the file (the first row is line 2)."""
    copy_rows = [list(item) for item in rows]
    copy_rows[line - 2][column] = value
    return copy_rows


def check_levels_check_file(report):
    """check-file on the sample, on good files and on one broken file per way of breaking; and that nothing written
    under labels/ or samples/ can get a carriage return."""
    folder = LEVEL_PROBE_DIR / "check-file"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    script = REPO_DIR / "scripts" / "prelabel_levels.py"
    result = subprocess.run([sys.executable, str(script), "check-file", LEVEL_SAMPLE_FILE, "--samples",
                             str(LEVEL_SAMPLE_SAMPLES), "--min-votes", str(LEVEL_SAMPLE_MIN_VOTES)],
                            capture_output=True, text=True, timeout=120, cwd=REPO_DIR)
    report.check(result.returncode == 0 and result.stdout.startswith("ok: ") and "rows; per label" in result.stdout,
                 f"level pre-labels: check-file {LEVEL_SAMPLE_FILE} --samples {LEVEL_SAMPLE_SAMPLES} --min-votes "
                 f"{LEVEL_SAMPLE_MIN_VOTES} (in its own process) passes and prints the summary",
                 f"exit {result.returncode}, {result.stdout.strip()[:300]!r} {result.stderr.strip()[:300]!r}")
    if result.returncode == 0:
        report.info(result.stdout.strip())

    good_labels = folder / "labels.csv"
    good_labels.write_bytes(levels.csv_bytes([LEVEL_COLUMNS] + LEVEL_GOOD_ROWS))
    good_review = folder / "review.csv"
    good_review.write_bytes(levels.csv_bytes([LEVEL_REVIEW_COLUMNS] + LEVEL_GOOD_REVIEW))
    plain = run_level_check_file(good_labels, "--samples", 5, "--min-votes", 3)
    review_alone = subprocess.run([sys.executable, str(script), "check-file", str(good_review)], capture_output=True,
                                  text=True, timeout=120)
    review_with = run_level_check_file(good_review, "--labels", good_labels)
    report.check(plain[0] == 0 and review_alone.returncode == 0 and "human_label filled in 3" in review_alone.stdout
                 and review_with[0] == 0,
                 "level pre-labels: check-file passes a good label file (every label, disputed and unparsed) with "
                 "--samples 5 --min-votes 3, and a good review table (filled and empty human_label) alone and with "
                 "--labels", f"{plain}; {review_alone.returncode} {review_alone.stdout.strip()!r} "
                 f"{review_alone.stderr.strip()!r}; {review_with}")

    header = LEVEL_COLUMNS
    good_bytes = levels.csv_bytes([header] + LEVEL_GOOD_ROWS)
    broken_labels = [  # (what, file bytes, options, text the error must contain)
        ("a column missing", levels.csv_bytes([header[:-1]] + [item[:-1] for item in LEVEL_GOOD_ROWS]), [], "line 1"),
        ("columns in another order", levels.csv_bytes([header[1:2] + header[:1] + header[2:]] + LEVEL_GOOD_ROWS), [],
         "line 1"),
        ("a label that is not one of the nine values",
         levels.csv_bytes([header] + changed_rows(LEVEL_GOOD_ROWS, 4, 1, "growth")), [], "line 4, column level"),
        ("votes 5", levels.csv_bytes([header] + changed_rows(LEVEL_GOOD_ROWS, 5, 2, "5")), [], "line 5, column votes"),
        ("votes 6/5", levels.csv_bytes([header] + changed_rows(LEVEL_GOOD_ROWS, 6, 2, "6/5")), [], "line 6, column votes"),
        ("0/0 with a label", levels.csv_bytes([header] + changed_rows(LEVEL_GOOD_ROWS, 7, 2, "0/0")), [],
         "line 7, column votes"),
        ("2/5 with a label", levels.csv_bytes([header] + changed_rows(LEVEL_GOOD_ROWS, 3, 2, "2/5")), [],
         "line 3, column votes"),
        ("1/0", levels.csv_bytes([header] + changed_rows(LEVEL_GOOD_ROWS, 8, 2, "1/0")), [], "line 8, column votes"),
        ("unparsed with 1/3", levels.csv_bytes([header] + changed_rows(LEVEL_GOOD_ROWS, 10, 2, "1/3")), [],
         "line 10, column votes"),
        ("disputed with 0/3", levels.csv_bytes([header] + changed_rows(LEVEL_GOOD_ROWS, 9, 2, "0/3")), [],
         "line 9, column votes"),
        ("an id that is not tim:<number>", levels.csv_bytes([header] + changed_rows(LEVEL_GOOD_ROWS, 2, 0, "tim:x1")),
         [], "line 2, column post_id"),
        ("an id twice", levels.csv_bytes([header] + changed_rows(LEVEL_GOOD_ROWS, 11, 0, "tim:12")), [],
         "line 11, column post_id"),
        ("a fingerprint of 7 digits", levels.csv_bytes([header] + changed_rows(LEVEL_GOOD_ROWS, 5, 5, "0a1b2c3")), [],
         "line 5, column text_hash"),
        ("a fingerprint in upper case", levels.csv_bytes([header] + changed_rows(LEVEL_GOOD_ROWS, 6, 5, "0A1B2C3D")),
         [], "line 6, column text_hash"),
        ("two model names", levels.csv_bytes([header] + changed_rows(LEVEL_GOOD_ROWS, 7, 3, "llama3:8b")), [],
         "line 7, column model"),
        ("two prompt versions", levels.csv_bytes([header] + changed_rows(LEVEL_GOOD_ROWS, 8, 4, "v2")), [],
         "line 8, column prompt_version"),
        ("no newline at the end", good_bytes[:-1], [], f"line {len(LEVEL_GOOD_ROWS) + 1}"),
        ("a carriage return", good_bytes.replace(b"\n", b"\r\n"), [], "carriage return"),
        ("a BOM at the start", levels.BOM + good_bytes, [], "BOM"),
        ("4 valid votes with --samples 3", levels.csv_bytes([header] + LEVEL_THREE_ROWS), ["--samples", 3],
         "line 6, column votes"),
        ("earnings 2/3 with --min-votes 3", levels.csv_bytes([header] + changed_rows(LEVEL_GOOD_ROWS, 3, 2, "2/3")),
         ["--min-votes", 3], "line 3, column votes"),
        ("disputed 3/5 with --min-votes 3", levels.csv_bytes([header] + changed_rows(LEVEL_GOOD_ROWS, 9, 2, "3/5")),
         ["--min-votes", 3], "line 9, column votes"),
    ]
    review_header = LEVEL_REVIEW_COLUMNS
    broken_reviews = [
        ("prelabel none", levels.csv_bytes([review_header] + changed_rows(LEVEL_GOOD_REVIEW, 3, 1, "none")), [],
         "line 3, column prelabel"),
        ("human_label not one of the seven",
         levels.csv_bytes([review_header] + changed_rows(LEVEL_GOOD_REVIEW, 4, 2, "disputed")), [],
         "line 4, column human_label"),
        ("an id twice", levels.csv_bytes([review_header] + changed_rows(LEVEL_GOOD_REVIEW, 5, 0, "tim:10")), [],
         "line 5, column post_id"),
        ("with --labels, an id not in the label file",
         levels.csv_bytes([review_header] + changed_rows(LEVEL_GOOD_REVIEW, 6, 0, "tim:99")), ["--labels", good_labels],
         "line 6, column post_id"),
        ("with --labels, another prelabel than the label file",
         levels.csv_bytes([review_header] + changed_rows(LEVEL_GOOD_REVIEW, 7, 1, "story_macro")),
         ["--labels", good_labels], "line 7, column prelabel"),
        ("a header that is neither",
         levels.csv_bytes([review_header + ["text"]] + [item + ["x"] for item in LEVEL_GOOD_REVIEW]), [],
         "cannot tell what kind of file"),
    ]
    missed = []
    for what, data, options, expected in broken_labels + broken_reviews:
        path = folder / "broken.csv"
        path.write_bytes(data)
        code, out, err = run_level_check_file(path, *options)
        if code == 0 or expected not in err:
            missed.append(f"{what} (exit {code}, {err.strip()[:200]!r})")
        else:
            report.info(f"check-file, {what}: {[line.strip() for line in err.splitlines() if expected in line][0]}")
        path.unlink()
    report.check(not missed, f"level pre-labels: check-file refuses each of {len(broken_labels)} broken label files "
                 "(header, label, votes 5, 6/5, 0/0 and 2/5 with a label, 1/0, unparsed 1/3, disputed 0/3, id, repeated "
                 "id, fingerprint of 7 digits or in upper case, two models, two prompt versions, no newline at the end, "
                 "carriage returns, a BOM, --samples, --min-votes) and "
                 f"{len(broken_reviews)} broken review tables (prelabel none, human_label, repeated id, --labels twice, "
                 "an unknown header), each with a non-zero exit and the line", "; ".join(missed))

    source = (REPO_DIR / "scripts" / "prelabel_levels.py").read_text(encoding="utf-8")
    saved = os.linesep
    carriage = []
    try:
        os.linesep = "\r\n"  # as on Windows; files written by the script must still have LF only
        output = folder / "linesep.csv"
        posts = [{"post_id": f"tim:{row}", "text": f"line ending probe {row}"} for row in range(6)]
        levels.label_posts(posts, output, LevelFake({}), settings_path=folder / "linesep.settings.json",
                           log=lambda text: None, sleep=lambda seconds: None)
        review = folder / "linesep-review.csv"
        labels_file = folder / "linesep-labels.csv"
        labels_file.write_bytes(good_bytes)
        levels.write_review_template(labels_file, review, size=4)
        carriage = [path.name for path in (output, review, folder / "linesep.settings.json")
                    if b"\r" in path.read_bytes()]
    finally:
        os.linesep = saved
    report.check("to_csv(" not in source and not carriage,
                 "level pre-labels: the script never writes a CSV with pandas (no to_csv( in its source), and with "
                 "os.linesep set to CRLF the label file, the review table and the settings record are written with "
                 "LF only", f"to_csv in the source {'to_csv(' in source}; files with a carriage return {carriage}")
    shutil.rmtree(folder, ignore_errors=True)


def check_levels_sample_file(report):
    """The committed sample label file (static)."""
    path = REPO_DIR / LEVEL_SAMPLE_FILE
    if not report.check(path.is_file(), f"level pre-labels: {LEVEL_SAMPLE_FILE} exists"):
        return
    kind, rows, problems, summary = levels.file_report(path, LEVEL_SAMPLE_SAMPLES, LEVEL_SAMPLE_MIN_VOTES)
    models = sorted({item[3] for item in rows if len(item) == len(LEVEL_COLUMNS)})
    versions = sorted({item[4] for item in rows if len(item) == len(LEVEL_COLUMNS)})
    report.check(kind == "label file" and not problems and len(rows) > 0 and models == [LEVEL_DEFAULT_MODEL]
                 and versions == [levels.PROMPT_VERSION],
                 f"level pre-labels: {LEVEL_SAMPLE_FILE} passes check-file with --samples {LEVEL_SAMPLE_SAMPLES} "
                 f"--min-votes {LEVEL_SAMPLE_MIN_VOTES}, has rows, the model {LEVEL_DEFAULT_MODEL} and prompt "
                 f"{levels.PROMPT_VERSION} only (so no text of any post can be in it)",
                 f"kind {kind}; problems {problems[:3]}; {len(rows)} rows; models {models}; versions {versions}")
    report.info(f"{LEVEL_SAMPLE_FILE}: {summary}")


def check_levels_env(report):
    """scripts/env.sh finds the virtual environment's Python on macOS and Linux, and in Git Bash on Windows."""
    folder = ENV_PROBE_DIR
    shutil.rmtree(folder, ignore_errors=True)
    fake_repo = folder / "repo"
    (fake_repo / "scripts").mkdir(parents=True)
    shutil.copyfile(REPO_DIR / "scripts" / "env.sh", fake_repo / "scripts" / "env.sh")
    environment = {key: value for key, value in os.environ.items() if key != "EYES_PYTHON"}

    def found(preset=None):
        extra = {} if preset is None else {"EYES_PYTHON": preset}
        result = subprocess.run(["bash", "-c", 'source "$1" && printf "%s" "$EYES_PYTHON"', "probe",
                                 str(fake_repo / "scripts" / "env.sh")], capture_output=True, text=True, timeout=60,
                                env={**environment, **extra})
        return result.stdout if result.returncode == 0 else f"exit {result.returncode}: {result.stderr.strip()}"

    def touch(relative):
        path = fake_repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        path.chmod(0o755)

    windows, unix = fake_repo / ".venv" / "Scripts" / "python.exe", fake_repo / ".venv" / "bin" / "python"
    touch(".venv/Scripts/python.exe")
    only_windows = found()
    touch(".venv/bin/python")
    both = found()
    preset = found("/somewhere/else/python")
    report.check(only_windows == str(windows) and both == str(unix) and preset == "/somewhere/else/python",
                 "level pre-labels: scripts/env.sh uses .venv/Scripts/python.exe when only that exists (Git Bash on "
                 "Windows), .venv/bin/python when both exist, and an EYES_PYTHON set beforehand as it is",
                 f"only Scripts {only_windows!r}; both {both!r}; preset {preset!r}")
    shutil.rmtree(folder, ignore_errors=True)


def check_levels_requirements(report):
    """requirements-prelabel.txt pins exactly the five packages with the pins of requirements.txt; .gitattributes keeps
    LF line endings (static)."""
    short = REPO_DIR / "requirements-prelabel.txt"
    full_lines = {line.strip() for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()}
    lines = [line.strip() for line in short.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.strip().startswith("#")] if short.is_file() else []
    names = sorted(normalise(line.split("==")[0]) for line in lines)
    not_in_full = [line for line in lines if line not in full_lines or "==" not in line]
    report.check(short.is_file() and not not_in_full and names == sorted(LEVEL_PRELABEL_PACKAGES),
                 "level pre-labels: requirements-prelabel.txt pins exactly pandas, numpy, pyarrow, huggingface_hub and "
                 "fsspec, each line exactly as in requirements.txt",
                 f"packages {names}; lines not in requirements.txt {not_in_full}")
    attributes = REPO_DIR / ".gitattributes"
    rules = {line.split("#")[0].strip() for line in attributes.read_text(encoding="utf-8").splitlines()} \
        if attributes.is_file() else set()
    missing = [pattern for pattern in LEVEL_LF_PATTERNS if f"{pattern} text eol=lf" not in rules]
    report.check(attributes.is_file() and not missing,
                 "level pre-labels: .gitattributes checks out " + ", ".join(LEVEL_LF_PATTERNS) + " with LF line endings",
                 f"missing {missing}")


def level_review_draw(rows, size, seed):
    """The documented review draw, written here apart from the script: for every level (ladder order) its ids in file
    order shuffled with random.Random(seed); then the levels in turn, one id each per round, a level with none left
    skipped, until size ids are drawn or none are left. none, unparsed and disputed rows never take part."""
    queues = []
    for label in LEVEL_SIX:
        ids = [item[0] for item in rows if item[1] == label]
        random.Random(seed).shuffle(ids)
        queues.append([(post_id, label) for post_id in ids])
    in_turn = [pair for round_ in itertools.zip_longest(*queues) for pair in round_ if pair is not None]
    return in_turn[:size]


def check_levels_full(report):
    """The level pre-label files against the dataset (full check): the sample, and the full file and the review table
    when they are there; prints the numbers."""
    try:
        table, where = levels.load_table()
    except levels.Stop as exc:
        report.check(False, "level pre-labels: the dataset is available to compare the label files with", str(exc))
        return
    posts = levels.posts_from_table(table)
    text_by_id = {post["post_id"]: post["text"] for post in posts}
    selection, summary, steps = levels.select_posts(posts)
    order = [post["post_id"] for post in selection]
    print(f"\nLevel pre-labels (compared with {where})")
    print(f"  dataset rows: {len(table)}   columns {list(table.columns)}   empty texts left out "
          f"{summary['empty_texts_left_out']}   repeated texts left out {summary['repeated_texts_left_out']}")
    print(f"  default draw: {summary['selected']} posts, per step {json.dumps(summary['taken_per_step'])}, "
          f"fingerprint {summary['fingerprint']}")
    found = levels.examples_in_dataset(post["text"] for post in posts)
    report.check(not found, "level pre-labels: none of the prompt's hand-written examples is a post of the dataset "
                 "(stripped, lower case)", f"found {found}")

    def compare(relative, rows):
        missing = [item[0] for item in rows if item[0] not in text_by_id]
        changed = [item[0] for item in rows if item[0] in text_by_id and own_text_hash(text_by_id[item[0]]) != item[5]]
        return missing, changed

    sample_rows = levels.file_report(REPO_DIR / LEVEL_SAMPLE_FILE)[1] if (REPO_DIR / LEVEL_SAMPLE_FILE).is_file() else []
    missing, changed = compare(LEVEL_SAMPLE_FILE, sample_rows)
    ids = [item[0] for item in sample_rows]
    report.check(bool(sample_rows) and not missing and not changed and set(ids) == set(order[:len(ids)]),
                 f"level pre-labels: every id of {LEVEL_SAMPLE_FILE} is a post of the dataset with that post's "
                 "fingerprint, and its ids are exactly the first posts of the default processing order",
                 f"{len(ids)} rows; {len(missing)} ids missing {missing[:3]}; {len(changed)} other texts {changed[:3]}; "
                 f"not among the first {len(ids)}: {sorted(set(ids) - set(order[:len(ids)]))[:3]}")

    full_path = REPO_DIR / LEVEL_FULL_FILE
    if not full_path.is_file():
        print(LEVEL_FULL_SKIPPED)
    else:
        kind, rows, problems, file_summary = levels.file_report(full_path)
        missing, changed = compare(LEVEL_FULL_FILE, rows)
        outside = [item[0] for item in rows if item[0] not in steps]
        report.check(kind == "label file" and not problems and not missing and not changed and not outside,
                     f"level pre-labels: {LEVEL_FULL_FILE} passes check-file, every id is a post of the dataset in the "
                     "default draw and every fingerprint that of its text",
                     f"problems {problems[:3]}; {len(missing)} ids missing {missing[:3]}; {len(changed)} other texts "
                     f"{changed[:3]}; {len(outside)} not in the default draw {outside[:3]}")
        print(f"  {LEVEL_FULL_FILE}: {file_summary}")
        print("  label              rows   share   " + "   ".join(f"{name[:9]:>9s}" for name in LEVEL_SIX[:3] + ["random"]))
        by_step = Counter((item[1], steps.get(item[0])) for item in rows)
        for label in LEVEL_LABELS + LEVEL_FAILURES:
            count = sum(item[1] == label for item in rows)
            print(f"  {label:17s} {count:5d}   {count / len(rows) if rows else 0:5.1%}   "
                  + "   ".join(f"{by_step[(label, step)]:9d}" for step in LEVEL_SIX[:3] + ["random"]))
        per_step = Counter(steps.get(item[0]) for item in rows)
        print("  rows per step of the draw: " + json.dumps({step: per_step.get(step, 0)
                                                            for step in LEVEL_SIX[:3] + ["random"]}))
        natural = Counter(item[1] for item in rows if steps.get(item[0]) == "random")
        print("  random part only (the natural shares): "
              + json.dumps({label: natural.get(label, 0) for label in LEVEL_LABELS + LEVEL_FAILURES}))
        print("  votes: " + json.dumps(dict(sorted(Counter(item[2] for item in rows).items(),
                                                   key=lambda pair: (-pair[1], pair[0])))))
    review_path = REPO_DIR / LEVEL_REVIEW_FILE
    if not review_path.is_file() or not full_path.is_file():
        print(LEVEL_REVIEW_SKIPPED if not review_path.is_file() else
              f"level pre-labels: {LEVEL_REVIEW_FILE} found without {LEVEL_FULL_FILE}")
        if review_path.is_file():
            report.check(False, f"level pre-labels: {LEVEL_REVIEW_FILE} comes with {LEVEL_FULL_FILE}")
        return
    kind, review_rows, problems, review_summary = levels.file_report(review_path, labels_path=full_path)
    expected = level_review_draw(levels.file_report(full_path)[1], LEVEL_DEFAULTS["DEFAULT_REVIEW_SIZE"],
                                 LEVEL_DEFAULTS["DEFAULT_SEED"])
    same = [(item[0], item[1]) for item in review_rows] == expected
    report.check(kind == "review table" and not problems and same,
                 f"level pre-labels: {LEVEL_REVIEW_FILE} passes check-file with --labels {LEVEL_FULL_FILE} and is "
                 "exactly the documented draw (six levels in turn, seed 42), row by row",
                 f"problems {problems[:3]}; same draw {same}")
    print(f"  {LEVEL_REVIEW_FILE}: {review_summary}")


# ---------------------------------------------------------------- the levels chapter and reading the crowd (chapters 9 and 10)

LEVEL_SECOND_REVIEW_FILE = "labels/levels-review-second-model.csv"  # the review by a second, larger language model
LEVEL_SECOND_REVIEW_COLUMNS = ["post_id", "prelabel", "review_label"]  # in order


def check_levels_second_review_file(report):
    """The second review file (static): UTF-8 without BOM, LF only, ending with a newline, exactly three columns, the
    ids and pre-labels of the review table row by row, and review labels among the six levels and none."""
    path, template = REPO_DIR / LEVEL_SECOND_REVIEW_FILE, REPO_DIR / LEVEL_REVIEW_FILE
    if not report.check(path.is_file() and template.is_file(), f"{LEVEL_SECOND_REVIEW_FILE} and {LEVEL_REVIEW_FILE} exist"):
        return
    data = path.read_bytes()
    problems = []
    if data.startswith(b"\xef\xbb\xbf"):
        problems.append("starts with a byte order mark")
    if b"\r" in data:
        problems.append("has carriage returns (CRLF or CR line endings)")
    if not data.endswith(b"\n"):
        problems.append("does not end with a newline")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        text = ""
        problems.append(f"not UTF-8: {exc}")
    rows = list(csv.reader(io.StringIO(text)))
    header, body = (rows[0], rows[1:]) if rows else ([], [])
    if header != LEVEL_SECOND_REVIEW_COLUMNS:
        problems.append(f"header {header}, expected {LEVEL_SECOND_REVIEW_COLUMNS}")
    wrong_width = [number for number, row in enumerate(body, start=2) if len(row) != len(LEVEL_SECOND_REVIEW_COLUMNS)]
    if wrong_width:
        problems.append(f"lines with another number of cells: {wrong_width[:5]}")
    with template.open(encoding="utf-8", newline="") as handle:
        template_rows = [(row["post_id"], row["prelabel"]) for row in csv.DictReader(handle)]
    pairs = [tuple(row[:2]) for row in body]
    if len(body) != LEVEL_DEFAULTS["DEFAULT_REVIEW_SIZE"] or pairs != template_rows:
        problems.append(f"{len(body)} rows; ids and pre-labels equal to {LEVEL_REVIEW_FILE} row by row: "
                        f"{pairs == template_rows}")
    unknown = sorted({row[2] for row in body if len(row) > 2 and row[2] not in LEVEL_LABELS})
    if unknown:
        problems.append(f"review labels outside the six levels and none: {unknown}")
    report.check(not problems,
                 f"levels: {LEVEL_SECOND_REVIEW_FILE} is UTF-8 without BOM with LF line endings and a final newline, has "
                 f"exactly the columns {LEVEL_SECOND_REVIEW_COLUMNS}, {LEVEL_DEFAULTS['DEFAULT_REVIEW_SIZE']} rows with "
                 f"the ids and pre-labels of {LEVEL_REVIEW_FILE} row by row, and review labels among the six levels and "
                 "none", "; ".join(problems))
    counts = Counter(row[2] for row in body if len(row) > 2)
    report.info(f"{LEVEL_SECOND_REVIEW_FILE}: {len(body)} rows; review labels {dict(sorted(counts.items()))}")


LEVEL_RAW_BASE = "https://raw.githubusercontent.com/Zhen131/ZHENNs_Ledger_Eyes/main/"  # this repository on GitHub
LEVEL_SWITCH_DEFAULTS = {  # chapter 9's switches and their defaults
    "LEVEL_LABELS_SOURCE": LEVEL_RAW_BASE + LEVEL_FULL_FILE,
    "LEVEL_REVIEW_SOURCE": LEVEL_RAW_BASE + LEVEL_REVIEW_FILE,
    "LEVEL_SECOND_REVIEW_SOURCE": LEVEL_RAW_BASE + LEVEL_SECOND_REVIEW_FILE,
    "LEVEL_BALANCE": True,
}
START_FORMAT_VERSION = "2.0"  # the format before the levels section; the new version keeps its MAJOR and raises its MINOR
LEVEL_PROBE_READING_DIR = REPO_DIR / "data" / "checks" / "levels-reading-probe"


def check_levels_switches(report, defaults, others):
    """Chapter 9's switches: defaults, LEVEL_BALANCE two-state with False as its other value, LEVEL_EPOCHS a positive
    integer and not two-state, the three sources not two-state."""
    wrong = [f"{name}={defaults.get(name)!r} (expected {value!r})" for name, value in LEVEL_SWITCH_DEFAULTS.items()
             if defaults.get(name) != value or type(defaults.get(name)) is not type(value)]
    epochs = defaults.get("LEVEL_EPOCHS")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        wrong.append(f"LEVEL_EPOCHS={epochs!r} (expected a positive integer)")
    two_state = sorted(name for name in ("LEVEL_LABELS_SOURCE", "LEVEL_REVIEW_SOURCE", "LEVEL_SECOND_REVIEW_SOURCE",
                                         "LEVEL_EPOCHS") if name in others)
    report.check(not wrong and others.get("LEVEL_BALANCE") is False and not two_state,
                 "switch cell: LEVEL_LABELS_SOURCE, LEVEL_REVIEW_SOURCE and LEVEL_SECOND_REVIEW_SOURCE default to the "
                 "repository's files on GitHub, LEVEL_BALANCE defaults to True with False as its other value, and "
                 "LEVEL_EPOCHS is a positive integer; only LEVEL_BALANCE is two-state",
                 f"{'; '.join(wrong)}; LEVEL_BALANCE other {others.get('LEVEL_BALANCE')!r}; two-state {two_state}")


def level_reading_namespace():
    """Chapter 9's class cell and reading cell, executed from the source; returns (namespace, problems)."""
    import pandas as pd
    namespace = {"__name__": "notebook_cells", "pd": pd, "Path": Path}
    return namespace, run_cells(["LEVEL_CLASSES", "match_level_labels"], namespace)


def stops_at_all(function, *args, words=()):
    """The message when function(*args) raises any exception, or None when it does not stop."""
    try:
        function(*args)
    except Exception as exc:  # the reading functions stop with ValueError or FileNotFoundError
        message = str(exc)
        return message if all(word in message for word in words) else f"(without {words}) {message}"
    return None


def check_levels_reading(report):
    """Chapter 9's reading and joining, run from the source on hand-written label tables and posts."""
    import pandas as pd
    shutil.rmtree(LEVEL_PROBE_READING_DIR, ignore_errors=True)
    namespace, problems = level_reading_namespace()
    if not report.check(not problems, "chapter 9: the class cell and the reading cell run from the source",
                        "; ".join(problems)):
        return
    read, check_file, match = (namespace["read_level_table"], namespace["check_level_file"],
                               namespace["match_level_labels"])
    texts = ["dividend yield 6%", "EPS beat", "revenue up", "breakout above 200", "Fed cuts rates", "to the moon",
             "happy friday", "who knows"]
    full = pd.DataFrame({"post_id": [f"tim:{row}" for row in range(len(texts))], "text": texts,
                         "has_ticker": [False] * len(texts)})
    levels_written = ["assets_dividends", "earnings", "revenue_business", "price_chart", "story_macro", "emotion_only",
                      "none", "disputed"]
    good = pd.DataFrame({"post_id": list(full["post_id"]), "level": levels_written, "votes": ["5/5"] * 7 + ["2/5"],
                         "model": ["m"] * 8, "prompt_version": ["v2"] * 8,
                         "text_hash": [own_text_hash(text) for text in texts]})
    LEVEL_PROBE_READING_DIR.mkdir(parents=True, exist_ok=True)
    good_path = LEVEL_PROBE_READING_DIR / "good.csv"
    good.to_csv(good_path, index=False, lineterminator="\n")
    table, digest = read(str(good_path), "LEVEL_LABELS_SOURCE", "labels/levels-prelabel-v2.csv")
    check_file(table)
    matched, counts = match(table, full, False)
    sample = full.iloc[[1, 3, 6]]  # a part of the posts, as the repository's sample is
    sample_matched, sample_counts = match(table, sample, True)
    report.check(digest == sha256_of(good_path) and list(table.columns) == LEVEL_COLUMNS and len(table) == 8
                 and counts == {"n_matched": 8, "n_not_in_data": 0, "n_other_text": 0}
                 and list(matched["post_id"]) == list(full["post_id"]) and list(matched["text"]) == texts
                 and sample_counts == {"n_matched": 3, "n_not_in_data": 5, "n_other_text": 0}
                 and list(sample_matched["post_id"]) == ["tim:1", "tim:3", "tim:6"],
                 "chapter 9 on a good label table: every cell read as text, the SHA-256 of the file's bytes, all 8 rows "
                 "matched to the full posts; on a part of the posts 3 matched and 5 not in the data, without stopping",
                 f"counts {counts}; on the part {sample_counts}; sha256 equal {digest == sha256_of(good_path)}")
    other_text = good.copy()
    other_text.loc[2, "text_hash"] = "00000000"
    missing_id = good.copy()
    missing_id.loc[5, "post_id"] = "tim:999"
    stops = {
        "a fingerprint differs (full)": stops_at_all(match, other_text, full, False, words=("tim:2",)),
        "a fingerprint differs (part)": stops_at_all(match, other_text, full.iloc[[2, 4]], True, words=("tim:2",)),
        "an id not in the full data": stops_at_all(match, missing_id, full, False, words=("tim:999",)),
        "no row matched": stops_at_all(match, good, pd.DataFrame({"post_id": ["tim:50"], "text": ["x"],
                                                                  "has_ticker": [False]}), True, words=("No row",)),
        "other columns": stops_at_all(check_file, good.drop(columns=["votes"]), words=("columns",)),
        "a repeated id": stops_at_all(check_file, pd.concat([good, good.iloc[[0]]]), words=("tim:0",)),
        "an unknown level": stops_at_all(check_file, good.replace({"level": {"earnings": "profits"}}),
                                         words=("profits",)),
        "a source that cannot be read": stops_at_all(read, str(LEVEL_PROBE_READING_DIR / "absent.csv"),
                                                     "LEVEL_LABELS_SOURCE", "labels/levels-prelabel-v2.csv",
                                                     words=("LEVEL_LABELS_SOURCE", "absent.csv")),
    }
    part_with_missing = stops_at_all(match, missing_id, full.iloc[[0, 1]], True)
    report.check(all(message is not None and not message.startswith("(without") for message in stops.values())
                 and part_with_missing is None,
                 "chapter 9 stops on a text fingerprint that differs (full data and part), on an id missing from the "
                 "full data (not from a part), when no row matches, on other columns, a repeated id or an unknown "
                 "level, and on a source it cannot read (naming the switch and the path)",
                 "; ".join(f"{case}: {message}" for case, message in stops.items())
                 + f"; id missing from a part: {part_with_missing or 'did not stop'}")
    shutil.rmtree(LEVEL_PROBE_READING_DIR, ignore_errors=True)


def level_file_expectations(switches):
    """What chapter 9 must report about the label file, computed here from the files the run read."""
    import pandas as pd
    import pyarrow.parquet as pq
    label_path = REPO_DIR / switches["LEVEL_LABELS_SOURCE"]
    with label_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    data_path = REPO_DIR / switches["DATA_SOURCE"]
    columns = pq.ParquetFile(data_path).schema_arrow.names
    from_sample = "row" in columns
    if from_sample:
        ids = {f"tim:{row}" for row in pd.read_parquet(data_path, columns=["row"])["row"]}
    else:
        ids = {f"tim:{row}" for row in range(pq.ParquetFile(data_path).metadata.num_rows)}
    matched = [row for row in rows if row["post_id"] in ids]
    return {
        "sha256": sha256_of(label_path),
        "n_rows": len(rows),
        "label_counts": {value: sum(row["level"] == value for row in rows) for value in LEVEL_LABELS + ["disputed", "unparsed"]},
        "models": sorted({row["model"] for row in rows}),
        "prompt_versions": sorted({row["prompt_version"] for row in rows}),
        "data_is_sample": from_sample,
        "n_matched": len(matched),
        "n_not_in_data": len(rows) - len(matched),
        "dropped": {value: sum(row["level"] == value for row in matched) for value in ("disputed", "unparsed")},
        "counts_before": {value: sum(row["level"] == value for row in matched) for value in LEVEL_LABELS},
    }


def check_levels_brief(report, label, brief, switches, full_run):
    """The levels section of a brief against the files the run read and against itself."""
    levels_section = brief.get("levels")
    if not report.check(isinstance(levels_section, dict) and bool(levels_section), f"{label}: levels section has content"):
        return
    version = str(brief.get("format_version"))
    start_major, start_minor = START_FORMAT_VERSION.split(".")
    major, _, minor = version.partition(".")
    report.check(list(brief) == TOP_LEVEL_FIELDS and major == start_major and minor.isdigit()
                 and int(minor) > int(start_minor),
                 f"{label}: the brief has exactly the top-level fields {TOP_LEVEL_FIELDS}, and its format version keeps "
                 f"the MAJOR of {START_FORMAT_VERSION} with a higher MINOR", f"fields {list(brief)}, version {version}")
    sources = {name: switches.get(name) for name in LOCAL_LEVEL_SOURCES}
    report.check(all(isinstance(value, str) and "://" not in value for value in sources.values())
                 and levels_section.get("labels_source") == switches.get("LEVEL_LABELS_SOURCE"),
                 f"{label}: the run read the three level files from local paths (no web address), and the brief names "
                 "the label source", f"{sources}; brief {levels_section.get('labels_source')!r}")
    expected = level_file_expectations(switches)
    labels_file, matching = as_dict(levels_section.get("labels_file")), as_dict(levels_section.get("matching"))
    found = {"sha256": labels_file.get("sha256"), "n_rows": labels_file.get("n_rows"),
             "label_counts": labels_file.get("label_counts"), "models": labels_file.get("models"),
             "prompt_versions": labels_file.get("prompt_versions"), "data_is_sample": matching.get("data_is_sample"),
             "n_matched": matching.get("n_matched"), "n_not_in_data": matching.get("n_not_in_data"),
             "dropped": levels_section.get("dropped"),
             "counts_before": as_dict(levels_section.get("balance")).get("counts_before")}
    differ = [f"{key}: brief {found[key]!r}, files {expected[key]!r}" for key in expected if found[key] != expected[key]]
    report.check(not differ and matching.get("n_other_text") == 0 and levels_section.get("classes") == LEVEL_LABELS,
                 f"{label}: the label file's fingerprint, rows and counts, the rows matched and not in the data, the "
                 "rows left out (disputed, unparsed) and the posts per class before the draw are those of the files "
                 "the run read; no text fingerprint differs; the classes are the six levels and none",
                 "; ".join(differ) + f"; other text {matching.get('n_other_text')!r}; classes {levels_section.get('classes')}")
    before = as_dict(as_dict(levels_section.get("balance")).get("counts_before"))
    if full_run:
        report.check(matching.get("n_matched") == labels_file.get("n_rows") and matching.get("n_not_in_data") == 0
                     and matching.get("n_other_text") == 0 and matching.get("data_is_sample") is False,
                     f"{label}: every row of the label file is matched to a post of the full data with the same text "
                     "fingerprint", f"{matching}, rows {labels_file.get('n_rows')}")
    else:
        report.check(matching.get("data_is_sample") is True and matching.get("n_other_text") == 0
                     and all(is_count(before.get(value)) and before[value] >= 10 for value in LEVEL_LABELS),
                     f"{label}: on the sample no text fingerprint differs, and each of the seven classes has at least "
                     "MIN_POSTS_PER_CLASS (10) matched posts", f"{matching}; per class {before}")
    balance = as_dict(levels_section.get("balance"))
    after = as_dict(balance.get("counts_after"))
    if switches.get("LEVEL_BALANCE"):
        balance_ok = (sorted(after) == sorted(LEVEL_LABELS) and len(set(after.values())) == 1
                      and after.get("none") == min(before.values() or [-1]))
    else:
        balance_ok = after == before
    report.check(balance.get("enabled") is switches.get("LEVEL_BALANCE") and balance_ok
                 and is_count(after.get("none")) and after["none"] > 0,
                 f"{label}: with LEVEL_BALANCE={switches.get('LEVEL_BALANCE')} the drawn posts are "
                 + ("the same number in every class, as many as the smallest class had" if switches.get("LEVEL_BALANCE")
                    else "every labelled post") + "; none is among them, disputed and unparsed are not",
                 f"before {before}, after {after}")
    split = as_dict(levels_section.get("split"))
    sizes, class_counts = as_dict(split.get("n_posts")), as_dict(split.get("class_counts"))
    split_ok = sorted(sizes) == sorted(SPLIT_PARTS) and all(is_count(sizes[part]) for part in SPLIT_PARTS) \
        and sum(sizes.values()) == sum(after.values())
    for part in SPLIT_PARTS if split_ok else []:
        counts = as_dict(class_counts.get(part))
        split_ok = split_ok and sorted(counts) == sorted(LEVEL_LABELS) and all(
            is_count(count) and count > 0 for count in counts.values()) and sum(counts.values()) == sizes[part]
    if split_ok:
        split_ok = all(sum(class_counts[part][value] for part in SPLIT_PARTS) == after[value] for value in LEVEL_LABELS)
    report.check(split_ok and split.get("stratified_by") == "level",
                 f"{label}: chapter 9's training, validation and test sets add up to the drawn posts, with every one of "
                 "the seven classes in every part", f"{sizes} {class_counts}")
    check_levels_models(report, label, levels_section, switches, sizes, full_run)
    check_levels_review_section(report, label, levels_section, switches)


LEVEL_GROUP_OF = {"assets_dividends": "numbers", "earnings": "numbers", "revenue_business": "numbers",
                  "price_chart": "chart_story", "story_macro": "chart_story", "emotion_only": "emotion"}  # none: no group
LEVEL_GROUP_ORDER = ["numbers", "chart_story", "emotion"]


def six_level_mean(per_class):
    """Mean of the six levels' F1 in a row's per_class (null scores left out); None when none is a number."""
    scores = [as_dict(per_class.get(level)).get("f1") for level in LEVEL_SIX]
    numbers = [score for score in scores if isinstance(score, (int, float)) and not isinstance(score, bool)]
    return sum(numbers) / len(numbers) if numbers else None


def check_levels_models(report, label, levels_section, switches, sizes, full_run):
    """Chapter 9's vectorizers, networks, choice, evaluation table, confusion matrices and groups (from the brief)."""
    n_train, n_test = sizes.get("train"), sizes.get("test")
    vectors = as_dict(levels_section.get("vectorizers"))
    word2vec = as_dict(vectors.get("word2vec"))
    vocab = [as_dict(vectors.get(name)).get("vocab_size") for name in ("count", "tfidf")]
    report.check(vectors.get("fitted_on") == "train" and vectors.get("fitted_on_n_posts") == n_train
                 and word2vec.get("fitted_on_n_posts") == n_train and is_count(word2vec.get("vocab_size"))
                 and all(is_count(size) and 0 < size <= switches["VOCAB_SIZE"] for size in vocab),
                 f"{label}: chapter 9's three vectorizers learned from chapter 9's training set only (at most VOCAB_SIZE "
                 "words for word counts and TF-IDF)",
                 f"fitted on {vectors.get('fitted_on')!r} ({vectors.get('fitted_on_n_posts')} posts, word vectors "
                 f"{word2vec.get('fitted_on_n_posts')}), training posts {n_train}, vocabularies {vocab}")
    models = levels_section.get("models") if isinstance(levels_section.get("models"), list) else []
    training = as_dict(levels_section.get("training"))
    selection = as_dict(levels_section.get("selection"))
    validation = as_dict(selection.get("validation_macro_f1"))
    report.check(models == SENTIMENT_MODELS and training.get("output_layer") == "7 units, softmax"
                 and training.get("loss") == "sparse_categorical_crossentropy"
                 and training.get("target_of_label") == {value: position for position, value in enumerate(LEVEL_LABELS)}
                 and training.get("epochs") == switches.get("LEVEL_EPOCHS")
                 and sorted(as_dict(training.get("history"))) == sorted(models)
                 and sorted(validation) == sorted(models) and all(is_score(score) for score in validation.values())
                 and selection.get("selected") in models,
                 f"{label}: chapter 9 trains three networks with seven softmax outputs for LEVEL_EPOCHS passes and "
                 "chooses one by a validation score that every network has",
                 f"models {models}; output {training.get('output_layer')!r}; epochs {training.get('epochs')}; "
                 f"validation {validation}; selected {selection.get('selected')!r}")

    evaluation = as_dict(levels_section.get("evaluation"))
    rows = [row for row in evaluation.get("rows") or [] if isinstance(row, dict)]
    names = models + SENTIMENT_BASELINES
    found = Counter(row.get("model") for row in rows)
    problems = [f"{name}: {found[name]} rows" for name in names if found[name] != 1]
    if len(rows) != len(names):
        problems.append(f"{len(rows)} rows for {len(names)} models and baselines")
    for row in rows:
        name, per_class, n = row.get("model"), as_dict(row.get("per_class")), row.get("n")
        if row.get("subset") != "all_test" or n != n_test or sorted(per_class) != sorted(LEVEL_LABELS):
            problems.append(f"{name}: subset {row.get('subset')!r}, {n} posts (test {n_test}), classes {sorted(per_class)}")
            continue
        supports = [as_dict(per_class[value]).get("support") for value in LEVEL_LABELS]
        if not all(is_count(support) for support in supports) or sum(supports) != n:
            problems.append(f"{name}: supports {supports} do not add up to {n}")
        scores = [row.get("accuracy"), row.get("macro_f1"), row.get("macro_f1_seven")] + [
            as_dict(per_class[value]).get(measure) for value in LEVEL_LABELS for measure in ("precision", "recall", "f1")]
        if not all(is_score(score) for score in scores):
            problems.append(f"{name}: a score is neither a number in [0, 1] nor null")
        six = six_level_mean(per_class)
        if (six is None) != (row.get("macro_f1") is None) or (six is not None and abs(six - row["macro_f1"]) > 1e-9):
            problems.append(f"{name}: macro_f1 {row.get('macro_f1')} is not the mean F1 of the six levels {six}")
        f1_all = [as_dict(per_class[value]).get("f1") for value in LEVEL_LABELS]
        numbers = [score for score in f1_all if isinstance(score, (int, float))]
        seven = sum(numbers) / len(numbers) if numbers else None
        if (seven is None) != (row.get("macro_f1_seven") is None) or (
                seven is not None and abs(seven - row["macro_f1_seven"]) > 1e-9):
            problems.append(f"{name}: macro_f1_seven {row.get('macro_f1_seven')} is not the mean F1 of the seven classes")
    report.check(bool(rows) and not problems,
                 f"{label}: chapter 9's evaluation table has one row per network and per baseline on all test posts, "
                 "every one of the seven classes scored (none in a column of its own), supports adding up to the test "
                 "posts, macro_f1 the mean F1 of the six levels (recomputed here) and macro_f1_seven that of all seven",
                 "; ".join(problems[:5]))

    accuracy = {row.get("model"): row.get("accuracy") for row in rows}
    matrices = as_dict(levels_section.get("confusion_matrices"))
    groups = as_dict(levels_section.get("groups"))
    group_matrices = as_dict(groups.get("confusion_matrices"))
    group_rows = {row.get("model"): row for row in as_dict(groups.get("evaluation")).get("rows") or []
                  if isinstance(row, dict)}
    called_none = as_dict(groups.get("n_called_none"))
    problems, group_problems = [], []
    none_index = LEVEL_LABELS.index("none")
    for name in models:
        matrix = as_dict(matrices.get(name))
        counts = matrix.get("counts")
        if not (matrix.get("labels") == LEVEL_LABELS and isinstance(counts, list) and len(counts) == 7
                and all(isinstance(line, list) and len(line) == 7 and all(is_count(cell) for cell in line)
                        for line in counts)):
            problems.append(f"{name}: not a 7x7 table of counts with the seven classes")
            continue
        total = sum(map(sum, counts))
        if total != n_test or matrix.get("n") != total:
            problems.append(f"{name}: cells add up to {total}, test posts {n_test}")
        elif n_test and isinstance(accuracy.get(name), float) and \
                abs(sum(counts[i][i] for i in range(7)) / n_test - accuracy[name]) > 1e-9:
            problems.append(f"{name}: diagonal does not match the accuracy")
        # The groups: fold the 7 x 7 matrix into 3 x 3 here and compare; none counted apart.
        folded = [[0] * 3 for _ in range(3)]
        for i, true_value in enumerate(LEVEL_LABELS):
            for j, called in enumerate(LEVEL_LABELS):
                if true_value != "none" and called != "none":
                    folded[LEVEL_GROUP_ORDER.index(LEVEL_GROUP_OF[true_value])][
                        LEVEL_GROUP_ORDER.index(LEVEL_GROUP_OF[called])] += counts[i][j]
        group_matrix = as_dict(group_matrices.get(name))
        both = sum(map(sum, folded))
        if group_matrix.get("labels") != LEVEL_GROUP_ORDER or group_matrix.get("counts") != folded \
                or group_matrix.get("n") != both or as_dict(group_rows.get(name)).get("n") != both:
            group_problems.append(f"{name}: 3x3 {group_matrix.get('counts')} (n {group_matrix.get('n')}), folded from "
                                  f"the 7x7 {folded} (n {both}); row n {as_dict(group_rows.get(name)).get('n')}")
        if groups.get("n_true_none") != sum(counts[none_index]) or \
                called_none.get(name) != sum(line[none_index] for line in counts):
            group_problems.append(f"{name}: none counted apart {groups.get('n_true_none')} true / {called_none.get(name)} "
                                  f"called, the 7x7 has {sum(counts[none_index])} / "
                                  f"{sum(line[none_index] for line in counts)}")
    report.check(bool(models) and not problems,
                 f"{label}: one 7x7 confusion matrix per network, adding up to the test posts and agreeing with the "
                 "accuracy", "; ".join(problems))
    for name, row in group_rows.items():
        per_class = as_dict(row.get("per_class"))
        supports = [as_dict(per_class.get(group)).get("support") for group in LEVEL_GROUP_ORDER]
        if sorted(per_class) != sorted(LEVEL_GROUP_ORDER) or not all(is_count(value) for value in supports) \
                or sum(supports) != row.get("n"):
            group_problems.append(f"{name}: groups {sorted(per_class)}, supports {supports}, n {row.get('n')}")
    report.check(groups.get("mapping") == LEVEL_GROUP_OF and sorted(group_rows) == sorted(names)
                 and sorted(called_none) == sorted(names) and not group_problems,
                 f"{label}: the three groups: every level in its group and none in none; one row per network and per "
                 "baseline; each network's 3x3 matrix is its 7x7 matrix folded (only posts where neither the true "
                 "class nor the call is none), and the none posts counted apart agree with the 7x7 matrix",
                 f"mapping {groups.get('mapping')}; rows {sorted(group_rows)}; " + "; ".join(group_problems[:4]))

    macro = {row.get("model"): row.get("macro_f1") for row in rows}

    def beats_both(name):
        score = macro.get(name)
        return score is not None and all(macro.get(baseline) is not None and score > macro[baseline]
                                         for baseline in SENTIMENT_BASELINES)

    report.check(levels_section.get("beats_baselines") == {name: beats_both(name) for name in models},
                 f"{label}: chapter 9's beats_baselines agrees with its evaluation rows",
                 f"{levels_section.get('beats_baselines')}")
    if full_run:
        for name in models:
            report.check(beats_both(name), f"{label}: chapter 9: {name} macro-F1 over the six levels beats always "
                         "majority and uniform random", f"{macro.get(name)} vs " + ", ".join(
                             f"{baseline} {macro.get(baseline)}" for baseline in SENTIMENT_BASELINES))


def check_levels_groups_cell(report):
    """Chapter 9's LEVEL_GROUPS cell, run from the source: every level folds into its group, none into none."""
    namespace = {"__name__": "notebook_cells"}
    problems = run_cells(["LEVEL_GROUPS"], namespace)
    mapping, order = namespace.get("LEVEL_GROUPS"), namespace.get("LEVEL_GROUP_NAMES")
    example = ["earnings", "none", "story_macro", "assets_dividends", "emotion_only", "price_chart", "revenue_business"]
    folded = [mapping.get(value) for value in example] if isinstance(mapping, dict) else None
    report.check(not problems and mapping == LEVEL_GROUP_OF and order == LEVEL_GROUP_ORDER
                 and folded == ["numbers", None, "chart_story", "numbers", "emotion", "chart_story", "numbers"],
                 "chapter 9: LEVEL_GROUPS (run from the source) folds levels 1 to 3 into numbers, 4 and 5 into "
                 "chart_story and 6 into emotion, and none into no group",
                 f"{'; '.join(problems)} mapping {mapping}; order {order}; {example} -> {folded}")


def own_review_agreement(rows, column, weights):
    """The documented review agreement, written here apart from the notebook. rows: dicts with prelabel and column."""
    filled = [(row["prelabel"].strip(), row[column].strip().lower()) for row in rows if row[column].strip()]
    by_prelabel = {}
    for level in LEVEL_SIX:
        pairs = [pair for pair in filled if pair[0] == level]
        by_prelabel[level] = {"n": len(pairs), "agreement": (sum(a == b for a, b in pairs) / len(pairs)) if pairs else None,
                              "weight": weights.get(level)}
    complete = all(part["agreement"] is not None and part["weight"] is not None for part in by_prelabel.values())
    return {
        "n_reviewed": len(filled),
        "agreement": (sum(a == b for a, b in filled) / len(filled)) if filled else None,
        "weighted_agreement": sum(part["agreement"] * part["weight"] for part in by_prelabel.values()) if complete else None,
        "by_prelabel": by_prelabel,
        "confusion": {"row_labels": LEVEL_LABELS, "column_labels": LEVEL_SIX, "rows": "review label",
                      "columns": "pre-label",
                      "counts": [[sum(pair == (column_label, row_label) for pair in filled) for column_label in LEVEL_SIX]
                                 for row_label in LEVEL_LABELS]},
    }


def close(left, right):
    """Equal, with numbers compared to 1e-9 (inside dicts and lists too)."""
    if isinstance(left, dict) and isinstance(right, dict):
        return sorted(left) == sorted(right) and all(close(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(close(a, b) for a, b in zip(left, right))
    if isinstance(left, float) or isinstance(right, float):
        return isinstance(left, (int, float)) and isinstance(right, (int, float)) and abs(left - right) <= 1e-9
    return left == right


def check_levels_review_section(report, label, levels_section, switches):
    """Both reviews in the brief, recomputed here from the files the run read."""
    review = as_dict(levels_section.get("review"))
    with (REPO_DIR / switches["LEVEL_LABELS_SOURCE"]).open(encoding="utf-8", newline="") as handle:
        label_rows = list(csv.DictReader(handle))
    six = Counter(row["level"] for row in label_rows if row["level"] in LEVEL_SIX)
    weights = {level: six[level] / sum(six.values()) for level in LEVEL_SIX}
    problems = []
    for name, switch, column in (("human", "LEVEL_REVIEW_SOURCE", "human_label"),
                                 ("second_model", "LEVEL_SECOND_REVIEW_SOURCE", "review_label")):
        part = as_dict(review.get(name))
        source = switches.get(switch) or ""
        rows = []
        if source.strip():
            with (REPO_DIR / source).open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        expected = own_review_agreement(rows, column, weights)
        found = {key: part.get(key) for key in expected}
        if not close(found, expected) or part.get("source") != source:
            differing = [key for key in expected if not close(found[key], expected[key])]
            problems.append(f"{name}: differs in {differing} (brief {[found[key] for key in differing][:2]}, "
                            f"files {[expected[key] for key in differing][:2]}); source {part.get('source')!r}")
    human, second = as_dict(review.get("human")), as_dict(review.get("second_model"))
    reviewer = str(second.get("reviewer"))
    names = [meaning for pattern, meaning in FORBIDDEN_PATTERNS if re.search(pattern, reviewer)]
    report.check(not problems and "second" in reviewer and "language model" in reviewer and not names
                 and isinstance(human.get("reviewer"), str),
                 f"{label}: both reviews (counts, strict and weighted agreement, agreement per level, confusion matrix) "
                 "are those recomputed here from the files the run read; the second review is described as by a "
                 "second language model, with no name",
                 "; ".join(problems) + f"; reviewer {reviewer!r} {names}")
    report.info(f"{label}: review human {human.get('n_reviewed')} rows (agreement {human.get('agreement')}), second "
                f"model {second.get('n_reviewed')} rows (agreement {second.get('agreement')}, weighted "
                f"{second.get('weighted_agreement')})")


LEVEL_REVIEW_PROBE = [  # (prelabel, review label as written): only filled labels count, stripped and in lower case
    ("assets_dividends", " Assets_Dividends "), ("assets_dividends", "earnings"), ("earnings", "EARNINGS"),
    ("earnings", ""), ("revenue_business", "none"), ("price_chart", "price_chart"), ("story_macro", "   "),
    ("emotion_only", "emotion_only"),
]
LEVEL_REVIEW_WEIGHTS = {"assets_dividends": 0.1, "earnings": 0.2, "revenue_business": 0.1, "price_chart": 0.2,
                        "story_macro": 0.3, "emotion_only": 0.1}


def check_levels_review_function(report):
    """level_review_agreement (a cell of definitions only), run from the source on hand-written tables whose results
    are worked out by hand here."""
    import numpy as np
    import pandas as pd
    source = cell_defining("level_review_agreement") or ""
    tree = ast.parse(source) if source else None
    only_definitions = bool(tree) and all(isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
                                          for node in tree.body)
    namespace = {"__name__": "notebook_cells", "pd": pd, "np": np}
    problems = run_cells(["LEVEL_CLASSES", "level_review_agreement"], namespace)
    if not report.check(only_definitions and not problems, "chapter 9: level_review_agreement sits in a cell of "
                        "definitions only and runs from the source", "; ".join(problems)):
        return
    function = namespace["level_review_agreement"]

    def table(pairs):
        return pd.DataFrame({"post_id": [f"tim:{index}" for index in range(len(pairs))],
                             "prelabel": [pair[0] for pair in pairs], "x": [pair[1] for pair in pairs]})

    one_level_empty = function(table(LEVEL_REVIEW_PROBE), "x", LEVEL_REVIEW_WEIGHTS)
    complete = function(table(LEVEL_REVIEW_PROBE + [("story_macro", "story_macro")]), "x", LEVEL_REVIEW_WEIGHTS)
    nothing = function(table([]), "x", LEVEL_REVIEW_WEIGHTS)
    # Worked out by hand: 6 filled rows, 4 agree; story_macro has no filled row, so no weighted agreement.
    expected_first = {"n_reviewed": 6, "agreement": 4 / 6, "weighted_agreement": None}
    expected_agreement = {"assets_dividends": 0.5, "earnings": 1.0, "revenue_business": 0.0, "price_chart": 1.0,
                          "story_macro": None, "emotion_only": 1.0}
    # With one story_macro row that agrees: 7 rows, 5 agree; weighted .1*.5 + .2*1 + .1*0 + .2*1 + .3*1 + .1*1 = 0.85.
    expected_counts = [[0] * 6 for _ in range(7)]
    for row_label, column_label in [("assets_dividends", "assets_dividends"), ("earnings", "assets_dividends"),
                                    ("earnings", "earnings"), ("none", "revenue_business"), ("price_chart", "price_chart"),
                                    ("emotion_only", "emotion_only"), ("story_macro", "story_macro")]:
        expected_counts[LEVEL_LABELS.index(row_label)][LEVEL_SIX.index(column_label)] += 1
    found_first = {key: one_level_empty.get(key) for key in expected_first}
    found_agreement = {level: one_level_empty["by_prelabel"][level]["agreement"] for level in LEVEL_SIX}
    ok = (close(found_first, expected_first) and close(found_agreement, expected_agreement)
          and [one_level_empty["by_prelabel"][level]["n"] for level in LEVEL_SIX] == [2, 1, 1, 1, 0, 1]
          and complete["n_reviewed"] == 7 and close(complete["agreement"], 5 / 7)
          and close(complete["weighted_agreement"], 0.85) and complete["confusion"]["counts"] == expected_counts
          and complete["confusion"]["row_labels"] == LEVEL_LABELS and complete["confusion"]["column_labels"] == LEVEL_SIX
          and nothing["n_reviewed"] == 0 and nothing["agreement"] is None and nothing["weighted_agreement"] is None)
    report.check(ok, "chapter 9: level_review_agreement on hand-written tables (spaces, upper case, empty and blank "
                 "labels, a level without reviewed rows, a none review label, no row at all) gives the rows, the strict "
                 "agreement, the agreement per level, the weighted agreement (null when a level has no row) and the "
                 "confusion matrix worked out by hand",
                 f"first {found_first}, per level {found_agreement}; complete {complete['n_reviewed']} rows, "
                 f"{complete['agreement']}, weighted {complete['weighted_agreement']}; none {nothing['n_reviewed']}")


LEVEL_TEXTS_LINE = "level_texts check: "  # the line chapter 9 prints after checking level_texts


def executed_lines(executed):
    """Every printed line of an executed notebook."""
    lines = []
    if executed.exists():
        for cell in json.loads(executed.read_text(encoding="utf-8"))["cells"]:
            for out in cell.get("outputs", []):
                text = out.get("text", "")
                lines += ("".join(text) if isinstance(text, list) else text).splitlines()
    return lines


def check_levels_texts_output(report, label, executed):
    """The executed notebook printed chapter 9's check of level_texts, and every part of it held."""
    lines = [line for line in executed_lines(executed) if line.startswith(LEVEL_TEXTS_LINE)]
    expected = re.compile(r"level_texts check: (\d+) texts, (\d+) rows, columns True, rows sum to 1 True, call is "
                          r"highest True, test calls match the evaluation True")
    match = expected.fullmatch(lines[0]) if len(lines) == 1 else None
    report.check(match is not None and match.group(1) == match.group(2) and int(match.group(1)) > 0,
                 f"{label}: level_texts on the demo sentences gives one row per sentence, the columns p_<class> for the "
                 "seven classes and call, rows adding up to 1 and a call that is the class of the highest probability, "
                 "and level_proba reproduces the evaluated test calls of the chosen network", " | ".join(lines) or "no line")




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


REPEAT_LABEL = "default switches again"  # third quick run: same switches as the first
LOCAL_LEVEL_SOURCES = {"LEVEL_LABELS_SOURCE": LEVEL_FULL_FILE, "LEVEL_REVIEW_SOURCE": LEVEL_REVIEW_FILE,
                       "LEVEL_SECOND_REVIEW_SOURCE": LEVEL_SECOND_REVIEW_FILE}  # every check run reads the local copies


def stable_numbers(brief):
    """The numbers of a brief that do not depend on training: the dataset section, the posts per class in every
    part of the split, and the posts in every subset of the evaluation table."""
    if not isinstance(brief, dict):
        return None
    sentiment = as_dict(brief.get("sentiment"))
    split = as_dict(sentiment.get("split"))
    rows = as_dict(sentiment.get("evaluation")).get("rows") or []
    levels_section = as_dict(brief.get("levels"))
    return {
        "dataset": brief.get("dataset"),
        "split_class_counts": split.get("class_counts"),
        "split_n_posts": split.get("n_posts"),
        "subset_posts": sorted({(row.get("subset"), row.get("n")) for row in rows if isinstance(row, dict)},
                               key=str),
        # The levels section: the label file, the matching, the rows left out, the draw and the split.
        "levels": {key: levels_section.get(key) for key in ("labels_source", "labels_file", "matching", "dropped",
                                                            "classes", "balance")},
        "levels_split": {key: as_dict(levels_section.get("split")).get(key) for key in ("n_posts", "class_counts")},
        "levels_review": levels_section.get("review"),
    }


def compare_stable(report, first, again):
    """Two runs with the same switches must agree on every number that does not depend on training."""
    left, right = stable_numbers(first), stable_numbers(again)
    detail = ""
    if left is None or right is None:
        detail = "a brief is missing"
    elif left != right:
        detail = "differing: " + ", ".join(key for key in left if left[key] != right[key])
    report.check(left is not None and bool(left["dataset"]) and left == right,
                 "default switches run twice: the two briefs agree on the dataset section, the posts per class in "
                 "every part of the split and the posts in every test subset, and on the numbers of the levels section "
                 "that do not depend on training", detail)


def quick(report, defaults, others, overrides, rules=None):
    runs = [("default switches", {}, "quick-default"),
            ("other switches", dict(others), "quick-other"),
            (REPEAT_LABEL, {}, "quick-default-again")]
    briefs = {}
    for label, flipped, folder in runs:
        parameters = {"DATA_SOURCE": SAMPLE_SOURCE, "WORK_DIR": f"data/checks/{folder}", **LOCAL_LEVEL_SOURCES,
                      **flipped, **overrides}
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
            check_levels_brief(report, label, brief, switches, full_run=False)
            check_levels_texts_output(report, label, REPO_DIR / switches["WORK_DIR"] / "executed.ipynb")
        if brief is not None:
            check_format(report, label, brief, rules)
            check_panel_output(report, label, REPO_DIR / switches["WORK_DIR"] / "executed.ipynb")
        if brief is not None and label != REPEAT_LABEL:  # the default run and the run with the other switches
            check_counterexamples(report, label, brief, rules)
        if brief is not None and label == "default switches":
            check_save_refuses(report, label, brief, rules)
    compare_stable(report, briefs.get("default switches"), briefs.get(REPEAT_LABEL))


def full(report, defaults, overrides, rules=None):
    check_levels_full(report)
    parameters = {"DATA_SOURCE": FULL_SOURCE, **LOCAL_LEVEL_SOURCES, **overrides}
    switches = {**defaults, **parameters}
    source = switches["DATA_SOURCE"]
    if "://" not in source:
        path = REPO_DIR / source
        if not report.check(path.is_file(), "full dataset available locally",
                            "" if path.is_file() else
                            f"{source} not found; run scripts/download_data.sh or pass -p DATA_SOURCE <address>"):
            return
        report.info(f"data file {source}: {path.stat().st_size:,} bytes, sha256 {sha256_of(path)}")
    check_fingerprints_full(report, FULL_SOURCE)
    report.info("full run: " + ", ".join(f"{k}={v!r}" for k, v in parameters.items()))
    brief_path = REPO_DIR / switches["WORK_DIR"] / "brief.json"
    brief_path.unlink(missing_ok=True)
    executed = REPO_DIR / switches["WORK_DIR"] / "runs" / "full.ipynb"
    ok, seconds, _ = run_notebook(report, "full run", parameters, executed)
    brief = read_brief(report, "full run", brief_path) if ok else None
    if brief is None:
        return
    check_brief(report, "full run", brief, switches)
    check_sentiment(report, "full run", brief, switches, full_run=True)
    check_levels_brief(report, "full run", brief, switches, full_run=True)
    check_levels_texts_output(report, "full run", executed)
    format_started = time.perf_counter()
    check_format(report, "full run", brief, rules)
    format_seconds = time.perf_counter() - format_started
    check_panel_output(report, "full run", executed)
    dataset = brief.get("dataset") or {}
    sample, cleaning = dataset.get("sample") or {}, dataset.get("cleaning") or {}
    tickers = dataset.get("ticker_mentions") or {}
    print("\nKey numbers (full run)")
    print(f"  posts in total:                {dataset.get('n_posts')}")
    print(f"  dataset's own classes:         {dataset.get('sentiment_counts')}   (0 neutral, 1 bullish, 2 bearish)")
    print(f"  converted:                     {dataset.get('label_from_sentiment')}   (dataset's value -> label)")
    print(f"  posts per class:               {dataset.get('class_counts')}   (0 bearish, 1 neutral, 2 bullish)")
    print(f"  class shares:                  " + json.dumps({k: round(v, 4) for k, v in (dataset.get('class_shares') or {}).items() if v is not None}))
    print(f"  posts with a ticker:           {tickers.get('n_posts')}   per class: {tickers.get('by_class')}")
    print(f"  words per post:                {dataset.get('length_words')}")
    print(f"  repeated texts:                {dataset.get('duplicates')}   posts with links: {dataset.get('n_posts_with_links')}")
    print(f"  empty texts:                   {dataset.get('n_empty_text')}")
    print(f"  working sample per class:      {sample.get('class_counts')}   (requested {sample.get('per_class_requested')})")
    print(f"  average words before / after:  {cleaning.get('avg_words_before')} / {cleaning.get('avg_words_after')}")
    print(f"  empty after cleaning:          {cleaning.get('n_empty_after')}")
    print(f"  run time:                      {seconds:.1f} s")
    print_sentiment(brief)
    print_levels(brief)
    print("\nBrief format (full run)")
    print(f"  format_version:                {brief.get('format_version')}")
    print(f"  format check of the brief:     {format_seconds:.1f} s (the check the notebook runs before saving)")
    print(f"  brief file size:               {brief_path.stat().st_size:,} bytes")
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
    check_numbering_rules(report)
    check_text(report)
    rules = load_format_rules(report)
    check_chapters(report)
    check_switch_cell(report, defaults, others)
    check_reading(report)
    check_ticker_rule(report)
    check_sample_file(report)
    check_split_function(report)
    check_score_functions(report)
    check_interface(report, defaults, others)
    check_levels_interface(report)
    check_levels_prompt(report)
    check_levels_hosts(report)
    check_levels_draw(report)
    check_levels_fake_model(report)
    check_levels_check_file(report)
    check_levels_sample_file(report)
    check_levels_env(report)
    check_levels_requirements(report)
    check_levels_second_review_file(report)
    check_levels_switches(report, defaults, others)
    check_levels_reading(report)
    check_levels_groups_cell(report)
    check_levels_review_function(report)
    unknown = [name for name in overrides if name not in defaults]
    report.check(not unknown, "overrides name existing switches", ", ".join(unknown))
    if defaults and not unknown:
        if args.mode == "quick":
            quick(report, defaults, others, overrides, rules)
        else:
            full(report, defaults, overrides, rules)

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
