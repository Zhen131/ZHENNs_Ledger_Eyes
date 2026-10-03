"""Project checks. Run through scripts/check.sh so the environment from scripts/env.sh is set.

    scripts/check.sh quick [-p NAME VALUE ...]
    scripts/check.sh full  [-p NAME VALUE ...]

quick: static checks, then the notebook runs three times on the bundled
       sample (default switches, every two-state switch flipped, default
       switches again); the first two saved briefs are checked, and the
       topics sections of the two default runs must be identical.
full:  static checks, then one run on the full dataset with the default
       switches; prints the key numbers from the brief.

-p NAME VALUE overrides a switch of the notebook's parameters cell for every
run (VALUE is read as a Python literal when possible, otherwise as text).
Exit code 0 means every check passed.
"""

import argparse
import ast
import hashlib
import json
import re
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
TOP_LEVEL_FIELDS = ["format_version", "generated_at", "scope", "dataset", "sentiment", "topics", "why"]
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


def quick(report, defaults, others, overrides):
    runs = [("default switches", {}, "quick-default"),
            ("other switches", dict(others), "quick-other"),
            (REPEAT_LABEL, {}, "quick-default-again")]
    briefs = {}
    for label, flipped, folder in runs:
        parameters = {"DATA_SOURCE": SAMPLE_SOURCE, "WORK_DIR": f"data/checks/{folder}", **flipped, **overrides}
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
    compare_topics(report, briefs.get("default switches"), briefs.get(REPEAT_LABEL))


def full(report, defaults, overrides):
    parameters = {"DATA_SOURCE": FULL_SOURCE, **overrides}
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
    check_numbering_rules(report)
    check_text(report)
    unknown = [name for name in overrides if name not in defaults]
    report.check(not unknown, "overrides name existing switches", ", ".join(unknown))
    if defaults and not unknown:
        if args.mode == "quick":
            quick(report, defaults, others, overrides)
        else:
            full(report, defaults, overrides)

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
