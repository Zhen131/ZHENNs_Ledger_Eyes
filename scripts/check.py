"""Project checks. Run through scripts/check.sh so the environment from scripts/env.sh is set.

    scripts/check.sh quick [-p NAME VALUE ...]
    scripts/check.sh full  [-p NAME VALUE ...]

quick: static checks, then the notebook runs twice on the bundled sample
       (default switches, then every two-state switch flipped), and both
       saved briefs are checked.
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

# Text scan. Each pattern is written so that it does not match its own source line
# (for example "c[l]aude" matches the word but not this file).
FORBIDDEN_PATTERNS = [
    (r"\b[0-9]{2}[A-D](?:-R)?\b", "private batch code (two digits and a letter A-D)"),
    (r"\bW[0-9]{1,2}\b", "private week code (W and a number)"),
    (r"\bQ[0-9]{2}\b", "private question code (Q and two digits)"),
    ("\u7b2c\\s*[0-9]+\\s*\u5468", "private week number written in Chinese"),
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


def reject_constant(name):
    raise ValueError(f"non-standard JSON constant {name}")


def read_brief(report, label, path):
    if not path.exists():
        report.check(False, f"{label}: brief saved", f"{path.relative_to(REPO_DIR)} not found")
        return None
    try:
        brief = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except ValueError as exc:
        report.check(False, f"{label}: brief is valid JSON without NaN or infinity", str(exc))
        return None
    report.check(True, f"{label}: brief saved as valid JSON without NaN or infinity", str(path.relative_to(REPO_DIR)))
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


def quick(report, defaults, others, overrides):
    runs = [("default switches", {}, "quick-default"),
            ("other switches", dict(others), "quick-other")]
    for label, flipped, folder in runs:
        parameters = {"DATA_SOURCE": SAMPLE_SOURCE, "WORK_DIR": f"data/checks/{folder}", **flipped, **overrides}
        switches = {**defaults, **parameters}
        report.info(f"{label}: " + ", ".join(f"{k}={v!r}" for k, v in parameters.items()))
        brief_path = REPO_DIR / switches["WORK_DIR"] / "brief.json"
        brief_path.unlink(missing_ok=True)  # a stale brief must never pass
        ok, _, _ = run_notebook(report, label, parameters, REPO_DIR / switches["WORK_DIR"] / "executed.ipynb")
        brief = read_brief(report, label, brief_path) if ok else None
        if brief is not None:
            check_brief(report, label, brief, switches)


def full(report, defaults, overrides):
    parameters = {"DATA_SOURCE": FULL_SOURCE, **overrides}
    switches = {**defaults, **parameters}
    source = switches["DATA_SOURCE"]
    if "://" not in source:
        path = REPO_DIR / source
        if not report.check(path.is_file(), "full dataset available locally",
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
