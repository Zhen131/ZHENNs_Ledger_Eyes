# Ledger Eyes

Ledger Eyes reads what the stock crowd writes, learns to tell bullish, neutral and bearish posts apart, and turns what it finds into one small, fixed-format result file, the **brief** (`brief.json`). The brief is the interface: other programs read it, not the notebook.

The work is a single Jupyter notebook that runs from top to bottom on Google Colab. It uses one public dataset from Hugging Face, [TimKoornstra/financial-tweets-sentiment](https://huggingface.co/datasets/TimKoornstra/financial-tweets-sentiment) (38,091 short English posts about stocks and markets, each labelled bullish, neutral or bearish, MIT License), read at a pinned revision; it needs no keys or accounts.

Chapters: 0 The question and the data · 1 Loading · 2 A first look · 3 Cleaning · 4 Tokenization · 5 Training, validation and test sets · 6 Vectorization · 7 Models · 8 Evaluation · 9 The brief: check and save. Chapters 1 and 2 read the posts, convert the dataset's labels to the notebook's order and mark the posts that name a ticker such as `$TSLA`; chapters 3 to 8 clean the posts, cut them into tokens, split them, turn them into three kinds of vectors (word counts, TF-IDF and averaged word vectors trained with Word2Vec), train a one-layer network on each, choose one with the validation set and evaluate all three on the test set against two baselines; chapter 9 checks the brief against its format and saves it.

Besides the notebook, `scripts/prelabel_levels.sh` gives stock tweets one of six levels (what a post pays attention to, from assets and dividends to posts that give no reason at all) through a language model running on the same machine, as training labels for a later chapter (see [Level pre-labels](#level-pre-labels)).

Earlier, longer versions of this notebook read another dataset and had more chapters; they are in the repository's history.

## Repository layout

| Path | What it is |
| --- | --- |
| `notebook/ledger_eyes.py` | The notebook source in jupytext "percent" format (`# %%` marks a cell). **This is the only notebook file to edit.** |
| `notebook/ledger_eyes.ipynb` | The notebook generated from the source; upload this one to Colab. Stored without outputs. |
| `docs/brief-format.md` | The brief's format, field by field: type, whether it is always there, whether it may be `null`, what it means; what each section covers; when the format version changes. |
| `docs/brief.schema.json` | The same rules as a JSON Schema (draft 2020-12). Chapter 9 carries an identical copy and checks every brief against it before saving. |
| `samples/tim-financial-tweets-sample.parquet` | 1,271 posts taken unchanged from the dataset, with their row in the full file; the quick check runs on them (see [The sample](#the-sample)). |
| `samples/levels-prelabel-sample.csv` | Level pre-labels of the first 40 posts of the processing order: ids, levels, votes, model, prompt version and text fingerprints, no text (see [Level pre-labels](#level-pre-labels)). |
| `labels/levels-prelabel-v2.csv`, `labels/levels-review-template.csv` | The level pre-labels of 8,000 stock tweets (`gemma3:12b`, five samples, three votes, labelled on a Windows machine on 2 and 3 November 2026) and their 200-row review table (see [Level pre-labels](#level-pre-labels)). |
| `scripts/check.sh` | The two checks: `quick` and `full`. |
| `scripts/build_notebook.sh` | Regenerates the `.ipynb` from the source. |
| `scripts/download_data.sh`, `scripts/download_data.py` | Downloads the dataset at its pinned revision into `data/raw/tim-financial-tweets-bba60269.parquet` (the same file the level script uses). Never overwrites a file that is already there. |
| `scripts/make_sample.py` | Draws the sample from the downloaded dataset. |
| `scripts/prelabel_levels.sh`, `scripts/prelabel_levels.py` | Level pre-labels of stock tweets through the local Ollama server: the draw, labelling with several samples per post (resumable), the review table, the review rows with their texts (into `data/` only), and a check of label files. |
| `scripts/env.sh` | Shared environment for the scripts (caches under `data/`, no token, no telemetry). |
| `requirements.txt` | Pinned package versions (Python 3.12). |
| `requirements-prelabel.txt` | The five packages `scripts/prelabel_levels.py` needs, with the same pins, for a machine that only pre-labels. |
| `.gitattributes` | Keeps scripts, CSV, JSON, text and Markdown files with LF line endings on every machine (Git on Windows would otherwise check them out with CRLF). |
| `data/` | Downloads, caches, executed notebooks and briefs. Ignored by Git. |

## Run on Google Colab

1. Open <https://colab.research.google.com>, choose *File → Upload notebook* and pick `notebook/ledger_eyes.ipynb`.
2. Optionally change the switches in the first code cell.
3. Choose *Runtime → Run all*.

The first code cell after the switches installs the packages Colab lacks (for example `emoji`, `contractions` and `gensim`) at the pinned versions; everything else, including TensorFlow/Keras and Hugging Face `tokenizers`, is used as Colab ships it, and the next cell prints the versions. The notebook reads the dataset straight from Hugging Face (`DATA_SOURCE`, an `hf://` address at a pinned revision, about 2.6 MB) without a token, and NLTK downloads its stop words and WordNet the first time. Nothing else is downloaded: the word vectors of chapter 6 are trained in the notebook. The last chapter saves the brief to `data/brief.json` in the Colab session.

The notebook has not yet been run on Colab itself; it has been run locally with the same code path, including reading the `hf://` address directly.

## Run locally

Requirements: Python 3.12 and a network connection for the first setup (package index, Hugging Face, NLTK data).

```bash
# 1. Environment
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. The dataset into data/raw/ (only needed for the full check)
scripts/download_data.sh            # TimKoornstra/financial-tweets-sentiment, about 2.6 MB

# 3. Checks
scripts/check.sh quick
scripts/check.sh full
```

Download the dataset first (step 2), then run the full check: it reads the local copy `data/raw/tim-financial-tweets-bba60269.parquet`. The scripts run under `bash` and set up their own environment from `scripts/env.sh`: Hugging Face cache in `data/hf_home`, NLTK data in `data/nltk_data`, pip cache in `data/pip-cache`, Keras settings in `data/keras`, Matplotlib, IPython and Jupyter settings under `data/`; no Hugging Face token is ever sent and telemetry is off.

### The two checks

**`scripts/check.sh quick`** (well under a minute once the environment is ready)

1. `requirements.txt` matches the installed packages, and the notebook's install pins match `requirements.txt`.
2. `notebook/ledger_eyes.ipynb` is exactly what `scripts/build_notebook.sh` produces from the source ("source file and .ipynb are consistent"), and it carries no outputs or execution counts (".ipynb carries no run results").
3. The parameters cell is well formed and holds every switch listed below: `DATA_SOURCE` defaults to the pinned Hugging Face file; `N_CLASSES` defaults to 3 with 2 as its other value; `TICKER_ONLY` defaults to `False` with `True` as its other value; the four word vector switches are positive integers and not two-state; the cell holds exactly the switches of the table below, in order (`ALL_SWITCHES` in `scripts/check.py`), so no switch of a removed chapter is left. The switch check cell, executed from the source, accepts the defaults and the other values and stops with the switch's name on every bad value of `SWITCH_PROBES` (`N_CLASSES` 4, `TICKER_ONLY` `"yes"`, a word vector switch of 0 or below, and others).
4. Text scan (rules in [Text rules](#text-rules)).
5. The brief's format (static, before any run): `docs/brief.schema.json` is a valid JSON Schema; the copy in chapter 9 (`BRIEF_SCHEMA_TEXT`) is identical to it, character for character; `format_version` is the same in the rules, in the brief the notebook writes and in `docs/brief-format.md`; every object of the rules lists its fields (or names its varying keys) and allows no other field; the field tables of `docs/brief-format.md` list exactly the fields of the rules, each with the same type, "required" and "null"; and the notebook saves the brief only through `save_brief`, which runs `validate_brief` before it writes and refuses NaN (`allow_nan=False`).
6. The chapter headings are 0 to 9 in order; chapter 9 is the last one and is the brief; its cells import nothing outside `json`, `math`, `pathlib`, `datetime`, `numpy`, `pandas` and `jsonschema` (so no network package and no web page).
7. Chapter 1's reading cells, executed from the source on small hand-written parquet files (in `data/checks/reading-probe/`, removed afterwards): a file without a `row` column gives the ids `tim:0`, `tim:1`, … and exactly the ids and texts of `posts_from_table` in `scripts/prelabel_levels.py`; a missing text becomes an empty string, spaces stay, the labels are converted (the dataset's 0 → 1, 1 → 2, 2 → 0) and `url` is never read; a file with a `row` column, drawn from the first, gives every post the id, the text and the label it has in the first; an unknown sentiment value (3) and a missing `tweet` or `sentiment` column stop it with that name; loading plus the ticker cell give `df_all` exactly the columns `post_id`, `text`, `label` and `has_ticker`.
8. The ticker rule, executed from the notebook's own cell: the texts of `TICKER_MUST_MATCH` (`$TSLA`, `$tsla`, `$F`, `$BRK.B`, `($AAPL)`, `$NVDA's`, `$SPY.` at the end of a sentence) count and those of `TICKER_MUST_NOT_MATCH` (`$100`, `$5.20`, `$TOOLONG`, `abc$TSLA`, a lone `$`) do not. Both lists may only be added to.
9. The sample (static): exactly the columns `tweet`, `sentiment` and `row`, rows unique, the rows drawn first in order and the rows appended after them in order; no empty text, no account name (`@` followed by a letter, digit or underscore) and no link (`http://`, `https://`, `www.`); at least 200 posts per class after the conversion; at least 80 posts of each of the six levels of `labels/levels-prelabel-v2.csv` and at least 80 labelled `none`, every appended row labelled `none`; at least 100 bearish and 100 bullish posts with a ticker (by the notebook's own rule), so that the run with `TICKER_ONLY = True` and two classes can split; every post that is also in the level file has that file's text fingerprint.
10. Chapter 5's `split_train_val_test`, executed from the source, splits six hand-written classes of 30 posts into 24 / 3 / 3 each and stops on a class in use without posts and on a class of fewer than `MIN_POSTS_PER_CLASS` posts. Chapter 7's `class_scores`, `macro_f1` and `evaluate` sit in a cell of definitions only and, executed from the source on six classes (labels 0 to 5), give scikit-learn's per-class scores and macro-F1.
11. The interface: every name of `INTERFACE_NAMES` is defined at the top level of a code cell, and the functions of `INTERFACE_PARAMETERS` take exactly their locked parameters (the network and score functions take `labels`, by default the classes in use). Chapter 6 trains its word vectors itself and downloads none (no `gensim.downloader`, `api.load` or web address in its cells).
12. The notebook runs from top to bottom on the sample three times: once with the default switches, once with every two-state switch flipped to its other value (`N_CLASSES = 2`, `TICKER_ONLY = True`, `EMOJI_TO_WORDS = False`, `KEEP_NEGATIONS = False`), and once more with the default switches (for item 16). Every run must save a brief, and the briefs of the first two runs are checked: all top-level fields present, `dataset` filled, `scope` matching the switches, `post_id` unique, class counts adding up, the posts with a ticker per class adding up to all posts with a ticker, equal class sizes in the working sample (with `TICKER_ONLY` every sampled post names a ticker), `clean_texts` reproducing the step-by-step cleaning, and the cleaning examples following the switches (negations kept and emoji turned into words with the defaults; with both cleaning switches off, the examples must equal the plain course recipe).
13. The `sentiment` section of both briefs is checked for structure only (the sample is too small for the scores to mean anything): the classes match `N_CLASSES` (three in the first run, two in the second); the tokenization comparison is recorded, and the BPE vocabulary (special tokens + single characters + merged pieces, which must add up) holds at most `BPE_VOCAB_SIZE` pieces, or, when the single characters and special tokens alone are more, exactly those with nothing merged; training, validation and test sets add up to the working sample (about 80/10/10, every class in every part); the test posts with a ticker are counted; the word-count and TF-IDF vectorizers learned at most `VOCAB_SIZE` words from the training set only, and the averaged word vectors learned from the training set only with the four word vector switches, counting the posts left with a vector of zeros; the three networks are listed; the output layer, the loss and the targets follow the classes in use (three softmax units with three classes); the evaluation table has exactly one row per model and per baseline (`always_majority`, `uniform_random`) for all test posts and for the test posts that name a ticker, with the right number of posts, class supports that add up and every score a number in [0, 1] or `null` (`null` for accuracy and macro-F1 when a subset is empty); one confusion matrix per model adding up to the number of test posts and agreeing with the accuracy; one model chosen with a validation score for every model; the scoring function's self-check passed (one column per class, bullish last, rows adding up to 1, the chosen model's evaluated test labels reproduced); `score_texts` gives one row per demo sentence, a probability column per class in use adding up to 1 and a call that is the class of the highest probability; the count of test texts also found in the training set; `beats_baselines` agreeing with the table. The switches `BPE_VOCAB_SIZE`, `VOCAB_SIZE`, `EPOCHS` and `BATCH_SIZE` must exist.
14. Every brief of the three runs passes the format rules (the notebook's own `brief_problems`, executed from the source), and every executed notebook printed that it saved the brief after the format check.
15. Counterexamples (locked; `BROKEN_BRIEFS` and `NULL_ALLOWED` in `scripts/check.py`): the briefs of the first two runs are broken on purpose, one place at a time, and the format check must reject each with a problem at the right field: (1) each top-level field missing; (2) a required field missing inside a section (`scope.granularity`, `dataset.n_posts`, `sentiment.evaluation`, `sentiment.evaluation.rows[0].macro_f1`); (3) a value of the wrong type (a count as text, an object as a list, a boolean as text); (4) a granularity other than `"batch"` (`"weekly"`, `"Batch"`, `""`, `null`, `1`); (5) NaN or infinity, also in fields that may be `null`; (6) `null` in a field that may not be `null`, and a field the format does not know (in `sentiment` and in `scope`); (7) a value the format does not allow (`scope.n_classes` set to 4). Setting a field that may be `null` to `null` (five cases) must still pass. With the default switches, `save_brief` must also write nothing for a brief that breaks the rules, save a NaN in a field that may be `null` as `null`, and write a valid brief unchanged.
16. The two runs with the default switches must agree on every number that does not depend on training: the whole `dataset` section, the posts per class in every part of the split and the posts in every test subset.
17. The level pre-label script (static; Ollama is never contacted and the dataset is not read): its seven labels in order, `unparsed` and `disputed`, the six columns of the label files, the three columns of the review table, the default model, the three default file paths, the review size and seed, the defaults of five samples, temperature 0.7, three votes, 8,000 posts (within 6,000 to 10,000) and a pool cap of 1,500, the dataset and its pinned revision, and the four core keyword groups (dividend; p/b and book value; p/e and pe ratio; revenue) are the interface of [Level pre-labels](#level-pre-labels); prompt v2 has the fingerprint pinned in `LEVEL_PROMPT_FINGERPRINTS` (computed by the check from what the script sends); a request is not streamed and carries the model, the seed and the temperature it is given, room for the longest answer and an answer format that allows only the seven labels, then the system message (the seven labels in ladder order), the hand-written examples (at least one per label) and the post last; a retry differs from the first request only by the reminder at the end of the system message.
18. The level script refuses every address that is not `http://` on a loopback host (the ten cases of `REFUSED_HOSTS` in `scripts/check.py`, also by the label command, which stops before doing anything), accepts `127.0.0.1`, `localhost` and `::1`, defaults to `127.0.0.1`, and uses no proxy even with proxy variables set.
19. The draw, on a hand-written table of 30 rows (no dataset): ids `tim:<row from 0>` with only the id and the text passed on; empty texts and a repeated text (same after stripping, case counts) left out with the smallest row kept, a text that differs in case kept; keywords match whole words in any case (`P/B`, `p/e`, not `dividendo`); a post in several pools belongs to the first; a pool above the cap is drawn down to it; the fill comes from every post not taken; pools holding more than the count stop the draw; the same seed gives the same draw and fingerprint, another seed another; the table's rows shuffled give the same draw and fingerprint; the fingerprint is the SHA-256 of the ids in processing order.
20. The level labelling loop, driven by a fake model on nine hand-written posts (files and settings records in `data/checks/levels-probe/`, removed afterwards): five votes alike give `5/5`, four to one `4/5`, three, one and one `3/5`, two, two and one `disputed,2/5`, a sample that fails three times does not count (`3/4`), every sample failing gives `unparsed,0/0`; labels in another case or with spaces count; two bad attempts and a good one make a vote; three timeouts fail only that sample; sample *i* asks with seed + *i*, its retries keep the seed and carry the reminder, every request has the temperature given; a conspicuous `sentiment` value in the table is in no request; a server that cannot be reached is tried again after the waits and then stops the run without writing that post (although some samples were answered) or asking any later one, while a short outage is ridden out; `--limit 2` and a second run label every post once; a partial last line is dropped and that post labelled again; `--retry-unparsed` asks only the `unparsed` rows, never the `disputed` ones, replaces a row in place when the new result is not `unparsed` (`disputed` too) and leaves it when it is; a file from another model or prompt version stops the run before anything is asked or written; the settings record stops a file with rows when the seed or the number of samples differs and names it, rows with more votes than `--samples` stop the run also without a record, rows without a record give a warning and a new record, an empty file takes new settings, two files with the same name in two folders get two records; `--min-votes` must be more than half of `--samples` and at most `--samples`; `data/prelabel/` is left unchanged.
21. `check-file`: `samples/levels-prelabel-sample.csv --samples 3 --min-votes 2` passes in its own process; a good label file and a good review table (alone and with `--labels`) pass; 22 broken label files (a column missing or out of order, a label that is not one of the nine values, `votes` `5`, `6/5`, `0/0` or `2/5` with a label, `1/0`, `unparsed` with `1/3`, `disputed` with `0/3`, a malformed or repeated id, a fingerprint of 7 digits or in upper case, two models, two prompt versions, no newline at the end, carriage returns, a BOM, too many votes for `--samples 3`, too few for `--min-votes 3`, `disputed` reaching `--min-votes 3`) and 6 broken review tables (`prelabel` `none`, a `human_label` that is not allowed, a repeated id, with `--labels` an id that is not in the label file or another label, an unknown header) each give a non-zero exit and name the line. The script's source has no `to_csv(`, and with `os.linesep` set to CRLF the label file, the review table and the settings record it writes have LF line endings only.
22. `samples/levels-prelabel-sample.csv` (static) passes `check-file --samples 3 --min-votes 2`, has rows, and holds only the default model and `PROMPT_VERSION`.
23. `scripts/env.sh`, copied into a made-up repository in `data/checks/env-probe/` (removed afterwards): with only `.venv/Scripts/python.exe` it uses that, with both it uses `.venv/bin/python`, and an `EYES_PYTHON` set beforehand stays as it is.
24. `requirements-prelabel.txt` pins exactly pandas, numpy, pyarrow, huggingface_hub and fsspec, every line exactly as in `requirements.txt`; `.gitattributes` gives `*.sh`, `*.csv`, `*.py`, `*.json`, `*.txt` and `*.md` LF line endings.

**`scripts/check.sh full`** (under a minute once the environment and the local copy of the dataset are there)

Runs the same static checks, then the notebook once on the local copy `data/raw/tim-financial-tweets-bba60269.parquet` with the default switches, checks the brief as above (including the format rules) and prints the key numbers: the posts, the dataset's own classes and the converted ones, the posts with a ticker per class, the working sample per class, the words per post before and after cleaning, and the whole evaluation table (every model and baseline, for all test posts and for the test posts that name a ticker) with the three confusion matrices. On the full data every model's macro-F1 on all test posts must beat both baselines. Before the run, the notebook's own reading cell (executed from the source) reads the local copy, and every id of `labels/levels-prelabel-v2.csv` must be a post of it with the same text fingerprint, every id of `labels/levels-review-template.csv` a post of it, and every post of the sample the same id, text and label as in it. When the local copy is missing, the check says so and stops; run `scripts/download_data.sh` first.

The full check also compares the level pre-label files with their dataset (read from `data/raw/tim-financial-tweets-bba60269.parquet`, which it downloads from Hugging Face at the pinned revision when it is missing): none of the prompt's hand-written examples may be a post of the dataset; every id of `samples/levels-prelabel-sample.csv` must be a post with that post's fingerprint, and its ids must be exactly the first posts of the default processing order. When `labels/levels-prelabel-v2.csv` is there, every id must be a post of the default draw with that post's fingerprint, and the check prints the rows per label (the seven, `unparsed` and `disputed`) split by the step of the draw, the rows per step, the labels of the random part alone (the natural shares) and the votes; when it is not there (it is labelled on another machine), the check prints `level pre-labels: labels/levels-prelabel-v2.csv not obtained (labelled on another machine); skipped` and does not fail. In the same way, `labels/levels-review-template.csv`, when it is there, must pass `check-file` with `--labels labels/levels-prelabel-v2.csv` and be exactly the documented draw, row by row.

Executed notebooks and briefs land in `data/checks/quick-default/`, `data/checks/quick-other/`, `data/checks/quick-default-again/` and, for the full check, `data/runs/full.ipynb` and `data/brief.json`. The `save_brief` probe uses `data/checks/save-probe/` and removes it, and so do the reading probe with `data/checks/reading-probe/` and the level probes with `data/checks/levels-probe/` and `data/checks/env-probe/`. If you pass `-p WORK_DIR <folder>` to the quick check, all three of its runs use that one folder: each brief is read right after its run, but every run overwrites the previous run's `executed.ipynb` and `brief.json`. `WORK_DIR` may also be a folder outside the repository.

### Changing switches without editing the notebook

Both checks accept `-p NAME VALUE` (repeatable). The value is read as a Python literal when possible, otherwise as text, and applies to every run of that check. A name that is not a switch stops the check.

```bash
scripts/check.sh full -p SAMPLE_PER_CLASS 5000
scripts/check.sh full -p DATA_SOURCE hf://datasets/TimKoornstra/financial-tweets-sentiment@bba602698fd6f18226f6404fd8fb637ef67fa1eb/data/train-00000-of-00001.parquet -p WORK_DIR data/hf-run
scripts/check.sh quick -p RANDOM_SEED 7
```

The checks pass the values to the notebook with [papermill](https://papermill.readthedocs.io), which replaces the cell tagged `parameters`. Without the scripts:

```bash
.venv/bin/papermill notebook/ledger_eyes.ipynb data/out.ipynb -p N_CLASSES 2 -p TICKER_ONLY True
```

## Switches

All switches are in the notebook's first code cell (tag `parameters`), one plain assignment per line, each with a comment of the form `# default: ...; ...`. Results are reported with the defaults. A switch marked "two-state" has one other value, and the quick check's second run flips all of them at once.

| Switch | Default | Other values | Meaning |
| --- | --- | --- | --- |
| `DATA_SOURCE` | `hf://datasets/TimKoornstra/financial-tweets-sentiment@bba602698fd6f18226f6404fd8fb637ef67fa1eb/data/train-00000-of-00001.parquet` | a local `.parquet` path with the columns `tweet` and `sentiment` (and optionally `row`) | Where the posts come from: the dataset's file at a pinned revision. |
| `WORK_DIR` | `data` | any folder | Cache and saved brief. |
| `RANDOM_SEED` | `42` | any integer | Seed for the working sample, the split, the networks and Word2Vec. |
| `SAMPLE_PER_CLASS` | `50000` | any positive integer | Most posts per class in the working sample. |
| `N_CLASSES` | `3` | two-state: `2` | Classes the models use: bearish, neutral and bullish; with 2 the neutral posts are left out. |
| `TICKER_ONLY` | `False` | two-state: `True` | With `True` the working sample keeps only posts that name a ticker. |
| `EMOJI_TO_WORDS` | `True` | two-state: `False` | Emoji become English words before special characters are removed; with `False` they are deleted. |
| `KEEP_NEGATIONS` | `True` | two-state: `False` | Negation words survive the stop word step; with `False` the plain NLTK list is used. |
| `BPE_VOCAB_SIZE` | `5000` | any positive integer | Target vocabulary of the subword tokenizer of chapter 4 (single characters are always kept). |
| `VOCAB_SIZE` | `1000` | any positive integer | Most words the word-count and TF-IDF vectorizers of chapter 6 keep. |
| `EPOCHS` | `5` | any positive integer | Passes over the training set in chapter 7. |
| `BATCH_SIZE` | `50` | any positive integer | Posts per training step in chapter 7. |
| `WORD2VEC_DIM` | `100` | any positive integer | Numbers in every word vector of chapter 6 (gensim's own default). |
| `WORD2VEC_WINDOW` | `5` | any positive integer | Words on each side that count as a word's neighbours in chapter 6 (gensim's own default). |
| `WORD2VEC_MIN_COUNT` | `5` | any positive integer | Uses in the training set a word needs to get a vector (gensim's own default). |
| `WORD2VEC_EPOCHS` | `5` | any positive integer | Passes of Word2Vec over the training set (gensim's own default). |

The cell right after the setup cells checks the switches and stops with the switch's name on a value the notebook cannot work with.

## Interface

These names are fixed; later chapters build on them. Their parameters are locked by the quick check (`INTERFACE_NAMES` and `INTERFACE_PARAMETERS` in `scripts/check.py`).

| Name | Chapter | Meaning |
| --- | --- | --- |
| `df_all` | 1 | Every post of the data source: a pandas table with exactly the columns `post_id`, `text`, `label` and `has_ticker`, in file order. Nothing is removed while reading (empty and repeated posts stay). Created in chapter 1 and kept unchanged until the end of the notebook. |
| `post_id` | 1 | Stable id of a post: `tim:<row>`, the post's row in the full file, from 0. A file with a `row` column (the sample) takes the row from that column, so a post has the same id in the sample and in the full file, and the same id as in the level pre-label files. Derived from the position in the file, never from random numbers. |
| `text` | 1 | The post as written (`tweet` in the file); an empty value becomes an empty string. |
| `label` | 1 | 0 bearish, 1 neutral, 2 bullish (`LABEL_NAMES`), converted from the dataset's `sentiment` (0 neutral, 1 bullish, 2 bearish) by `RAW_TO_LABEL`. |
| `has_ticker` | 1 | True when the text names a ticker: `$`, 1 to 5 letters in any case, optionally a dot and one or two letters, with no letter or digit right before the `$` or right after the code (`TICKER_PATTERN`; the cell of chapter 1 is the only place the rule is written). Also a function, `has_ticker(texts)`, that gives the column for any texts. |
| `df` | 2 | The working sample: the classes in use, the same number of posts in each (at most `SAMPLE_PER_CLASS`), only posts with a ticker when `TICKER_ONLY`; columns `post_id`, `text`, `label`, `has_ticker`, and from chapter 3 `clean_text`. |
| `clean_texts(texts)` | 3 | Cleans raw texts with the current switches, the same steps in the same order as the cells of chapter 3; returns a pandas Series (same index when given a Series). |
| `CLASS_LABELS` | 5 | The labels in use, in column order: `[0, 1, 2]`, or `[0, 2]` with two classes. |
| `split_train_val_test(posts, seed, label_column="label", labels=None)` | 5 | Splits posts into training, validation and test sets, about 80 / 10 / 10, stratified by `label_column`, for any number of classes. Stops when a class of `labels` has no post or a class has fewer than `MIN_POSTS_PER_CLASS` (10) posts. Returns `(train, validation, test)`. |
| `train_df`, `val_df`, `test_df` | 5 | The three parts of the working sample. |
| `count_vectorizer`, `tfidf_vectorizer`, `word2vec_vectorizer` | 6 | The three vectorizers, fitted on the training set only. `word2vec_vectorizer.model` is the trained gensim Word2Vec model; its `transform` returns a dense float32 table, the other two sparse float32 matrices. Each has `vocabulary_` (word → position). |
| `vectorizers` | 6 | `{"count": ..., "tfidf": ..., "word2vec": ...}`. |
| `fit_vectorizers(cleaned_texts)` | 6 | Trains the three vectorizers anew on the given cleaned texts, the way chapter 6 does; returns a dictionary with the same keys. |
| `vectorize_texts(cleaned_texts, fitted=vectorizers)` | 6 | Turns cleaned texts into all three kinds of vectors; returns `{name: matrix}`, one row per text. |
| `to_targets(labels_of_posts, labels=CLASS_LABELS)`, `build_network(n_features, labels=CLASS_LABELS)`, `predict_proba(model, matrix, labels=CLASS_LABELS)`, `predict_labels(model, matrix, labels=CLASS_LABELS)` | 7 | The one-layer network and its helpers, for any number of classes: one sigmoid unit for two classes, one softmax unit per class for more; probabilities have one column per label of `labels`, a tie in `predict_labels` goes to the earlier column. |
| `class_scores(true_labels, predicted_labels, labels=CLASS_LABELS)`, `macro_f1(...)`, `evaluate(...)` | 7 | Per-class precision, recall, F1 and support; their mean F1; one row of the evaluation table. They sit in a cell of definitions only and work with any number of classes (pass the labels, for example `labels=list(range(6))` for six classes). |
| `models` | 7 | `{"count_network", "tfidf_network", "word2vec_network"}`, each with `model`, `vectors` (the key in `vectorizers`) and `history`. |
| `selected_model_name`, `selected_model`, `selected_vectorizer` | 7 | The model chosen by macro-F1 on the validation set (a tie goes to the model listed first), and its vectorizer. |
| `sentiment_proba(cleaned_texts)` | 7 | Class probabilities from the chosen model for any cleaned texts: a NumPy array, one row per text, one column per label of `SENTIMENT_PROBA_LABELS`; every row adds up to 1. |
| `SENTIMENT_PROBA_LABELS`, `BULLISH_COLUMN` | 7 | The label of each column of `sentiment_proba` (bearish first); the probability of bullish is the last column. |
| `score_texts(texts)` | 7 | Raw texts in: cleaned with `clean_texts`, scored with `sentiment_proba`; returns a pandas table, one row per text (same index when given a Series), with one probability column per class in use (`p_bearish`, `p_neutral`, `p_bullish`) and `call`, the class of the highest probability (a tie goes to the earlier column). |
| `train_post_ids` | 7 | The `post_id`s of the training set. |
| `brief` | top | The result dictionary; see below. |
| `BRIEF_SCHEMA_TEXT`, `BRIEF_SCHEMA` | 9 | The format rules, identical to `docs/brief.schema.json`. |
| `brief_problems(document)`, `validate_brief(document)` | 9 | Every way a brief breaks the rules; raise when there is one. |
| `save_brief(brief, path)` | 9 | Converts, checks and writes a brief; writes nothing when the check fails. |
| `brief_path` | 9 | Where the brief was saved: `WORK_DIR/brief.json`. The checks read the brief from there. |

### The brief

The brief has a fixed format, version **2.0**. [docs/brief-format.md](docs/brief-format.md) describes it field by field (type, whether the field is always there, whether it may be `null`, what it means), says which posts each section covers and when the version changes; [docs/brief.schema.json](docs/brief.schema.json) holds the same rules as a JSON Schema. In short:

- Top level, always present: `format_version` (the version above), `generated_at` (UTC, ISO 8601), `scope`, `dataset` (chapters 0 to 3) and `sentiment` (chapters 4 to 8). Every section starts as `null` in the notebook and is filled by its chapter before the brief is saved.
- `scope`: `source` (the `DATA_SOURCE` value), `granularity` (`"batch"`: one result for the whole dataset), `tickers` (`"all"` or `"has_ticker_only"`, describing the working sample only), `n_classes`.
- A value that cannot be computed (for example a share with a zero denominator) is `null`, never 0, NaN or infinity, and only the fields the format document marks "Null: yes" may be `null`. `save_brief` converts numpy and pandas values to plain values (NaN and infinity become `null`), checks the result against the rules, and writes it only when it passes; the JSON writer also refuses NaN.
- The rules allow no other fields. Adding or changing a field means changing the rules, the format document and the format version together (see [Versions](docs/brief-format.md#versions) and [Editing the notebook](#editing-the-notebook)).

## Editing the notebook

1. Edit `notebook/ledger_eyes.py` only.
2. Regenerate the notebook: `scripts/build_notebook.sh` (writes `notebook/ledger_eyes.ipynb`; cell ids come from the cell content, so an unchanged source gives a byte-identical file).
3. Run `scripts/check.sh quick`.

Changing the brief's format (a new field, a changed field): change `docs/brief.schema.json`; copy its text unchanged into `BRIEF_SCHEMA_TEXT` in chapter 9; set the new `format_version` in the rules' `const`, in the cell that creates `brief` and in `docs/brief-format.md`, following [Versions](docs/brief-format.md#versions); update the field tables of `docs/brief-format.md`. The quick check fails until the two copies of the rules are identical, the document lists exactly the fields of the rules and every brief passes them. The locked counterexamples in `scripts/check.py` may only be added to.

Writing rules for the notebook: one cell does one thing; every code cell, or short run of related code cells, is preceded by an explanation cell; explanation cells are English, then a line with `***`, then Chinese; code comments are English only. New switches go into the parameters cell in the same form, all package installs stay in the single install cell right after it, and chapter 9 (check and save the brief) stays the last chapter: a new chapter goes before it.

### Text rules

The quick check scans every text file in the repository outside `data/` and `.venv/` and rejects:

- private numbering, in the forms the private log uses: file-name prefixes (two digits, optionally one of the capital letters A–D or Q, optionally a round suffix `-R` or `-` plus a number, then `_W` and a week number), review codes (two digits, a capital A–D, then `-R`), `W` followed by two digits, `Q` followed by two digits, and week numbers written in Chinese. A bare "two digits and a letter" is deliberately not flagged, because it collides with normal notation such as 50D (50 days) or 10B (10 billion). Before scanning, the check tests these rules on built-in examples (`NUMBERING_MUST_MATCH` and `NUMBERING_MUST_NOT_MATCH` in `scripts/check.py`): every log code form must be caught and no normal notation may be;
- names of chat assistants and their vendors, the two-letter capitalised abbreviation for artificial intelligence, and co-author trailers;
- local home paths and user names.

Chinese text is allowed only in the notebook's explanation cells, after the `***` line. The exact patterns are in `FORBIDDEN_PATTERNS` in `scripts/check.py`.

## The sample

`samples/tim-financial-tweets-sample.parquet` holds 1,271 posts taken unchanged from the dataset, with three columns: `tweet` and `sentiment` exactly as in the dataset (`sentiment` in the dataset's own coding, 0 neutral, 1 bullish, 2 bearish, not converted) and `row`, the post's row in the full file (from 0), so the notebook gives every post the same id, `tim:<row>`, as in the full file. `url` is not copied. The 1,202 rows of the first four steps below are sorted by `row`; the 69 rows that the fifth step appends follow them, sorted by `row` among themselves.

**Source and license.** [TimKoornstra/financial-tweets-sentiment](https://huggingface.co/datasets/TimKoornstra/financial-tweets-sentiment) on Hugging Face, revision `bba602698fd6f18226f6404fd8fb637ef67fa1eb`, released under the MIT License. Its card asks for no particular citation.

**How it was drawn.** `scripts/make_sample.py` read the local copy of the dataset and drew with seed 42, only from posts that are not empty after stripping and contain **no account name (`@` followed by a letter, digit or underscore) and no link (`http://`, `https://`, `www.`)**, so that no one's account name or link is copied into this repository; the posts drawn keep their text exactly. In order, each step drawing at random among the posts not yet taken and only as many as are still missing: 100 posts of each of the six levels of `labels/levels-prelabel-v2.csv`; until every class has 400 posts; until the bearish and the bullish class each have 150 posts with a ticker; until 60 posts contain emoji and 60 a negation; and last, appended after those rows, which stay unchanged and in their order, posts labelled `none` in `labels/levels-prelabel-v2.csv` until the sample holds 80 of them (the first four steps left only 11). The levels chapter learns `none` as a seventh class with the same number of posts in every class, so with too few `none` posts it could not split them in the quick check. Whether a post has a ticker is decided by the notebook's own cell of chapter 1, executed from the source.

**What it must hold** (checked by the quick check, all lower bounds): at least 200 posts per class after the conversion; at least 80 posts of each of the six levels and at least 80 labelled `none` (so the levels chapter can run in the quick check), and every appended row labelled `none`; at least 100 bearish and 100 bullish posts with a ticker (so the run with `TICKER_ONLY = True` and two classes can split); no empty text, no account name, no link.

**Regenerating it.** Download the dataset (`scripts/download_data.sh`), then run `bash -c 'source scripts/env.sh; .venv/bin/python scripts/make_sample.py'`. It prints the counts per class, per level, with a ticker, with emoji and with a negation, and a fingerprint of the list of rows; the same dataset and the same notebook give the same rows.

## Level pre-labels

A second kind of silver label, for stock tweets: **what does a post pay attention to, and how close to cash is it?** `scripts/prelabel_levels.sh` asks a language model (`qwen2.5:7b` by default), through an [Ollama](https://ollama.com) server running on the same machine, to put each post on one rung of a ladder of six levels by what it talks about (a post does not need to give a reason or take a position: a headline or an earnings calendar is labelled by its subject), or to say that it does not talk about stocks or markets at all. The ladder measures the crowd's **appetite for risk, not whether it is bullish or bearish**: the closer to levels 1 and 2, the calmer the attention; the closer to 5 and 6, the more heated. A bullish post and a bearish post can stand on the same level, and the labels are never checked against price moves. These are **silver labels**: a model gave them, not a person; the [review table](#the-level-review-table) lets a person measure how often the model agrees.

The script is written to run the full labelling on another machine (a Windows PC with a graphics card, through Git Bash); this repository holds the script, its checks and a sample of 40 posts labelled on a laptop. The full label file and the review table are added once they have been labelled there.

### The seven labels

In ladder order, closest to cash first (the order of every table and of the review draw):

| # | Label | Meaning (as given to the model) |
| --- | --- | --- |
| 1 | `assets_dividends` | net assets and dividends: price to book (P/B), book value, trading below book value, net assets, the balance sheet, dividends, dividend yield, payouts |
| 2 | `earnings` | profits: price to earnings (P/E), earnings per share (EPS), profit, profit margins, net income, guidance, earnings that beat or miss expectations |
| 3 | `revenue_business` | revenue and the business: price to sales (P/S), revenue, revenue growth, sales, deliveries, units sold, users or subscribers, orders, products, how the business is going |
| 4 | `price_chart` | the chart: the price itself or a price level, its trend, breakouts, support, resistance, moving averages, RSI, volume, turnover, candlestick patterns |
| 5 | `story_macro` | a story: news, analyst ratings and price targets, policy, interest rates, inflation, big industry trends, narratives such as a new technology that will change the world |
| 6 | `emotion_only` | emotion only: the post talks about a stock or the market but only takes a position, shows a feeling or waits, without anything of levels 1 to 5, such as to the moon, it is over, just buy, holding on |
| — | `none` | not about stocks or markets at all: small talk, adverts for airdrops or giveaways, only mentions of people and links, or a post that cannot be read. **Not a level**: it is counted apart, so that such posts do not swell level 6. |

The prompt asks what the post pays attention to, not for the writer's reason: an earnings call or an earnings date is `earnings`, a rating or a price target is `story_macro`, a net asset value announcement is `assets_dividends`, even without a word of opinion. Three rules are part of the prompt:

- **Two or more levels: the one closest to cash.** A post about revenue and about a story is `revenue_business`.
- **About a stock or the market, but only a position, a feeling or waiting, nothing of levels 1 to 5: `emotion_only`.**
- **Not about stocks or markets at all, or unreadable: `none`.**

Two more values are not labels: `unparsed` (no sample gave a usable answer) and `disputed` (the most frequent label did not get enough votes, see below). Neither is used for training; their counts are reported apart.

### The data

[TimKoornstra/financial-tweets-sentiment](https://huggingface.co/datasets/TimKoornstra/financial-tweets-sentiment) on Hugging Face (MIT License), file `data/train-00000-of-00001.parquet` at the pinned revision `bba602698fd6f18226f6404fd8fb637ef67fa1eb` (38,091 rows on 1 November 2026; columns `tweet`, `sentiment` and `url`). The first command that needs it copies the file byte for byte to `data/raw/tim-financial-tweets-bba60269.parquet` (through a temporary file, so an interrupted download leaves nothing half-written) and reads only that copy afterwards; if the copy cannot be read, the script says to delete it and run again. No token is sent. The commands that read the dataset stop when `HF_ENDPOINT` points to anything other than `https://huggingface.co` (no mirror is used silently) and when they are not started through `scripts/prelabel_levels.sh` (which sets `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` and the other settings of `scripts/env.sh`).

- **Post ids**: `tim:<row>`, where `<row>` is the row's position in the parquet file, counted from 0 (the table as read, index reset). An empty `tweet` is taken as an empty text. **Any other code that reads the same file must number its rows the same way**, so that the labels can be joined back to their posts.
- **Text fingerprint** (`text_hash`): the first 8 hexadecimal digits (lower case) of the SHA-256 of the `tweet` text as it is (UTF-8, not stripped, case kept). It cannot give the text back; it shows that an id still points to the same text.
- **Only the text goes to the model.** The dataset's `sentiment` and `url` are never passed to the draw or put into a request.

### Which posts

Three of the six levels are rare in a random draw (a few dividend posts among hundreds), so posts that seem to talk about them are drawn first:

1. Posts whose text is empty after stripping spaces are left out; so are **repeated texts** (the same after stripping spaces, **case counts**), keeping the one with the smallest row.
2. Three keyword pools, in ladder order. A word or phrase matches when no letter or digit touches it on either side, in any case (`p/e` matches `P/E`; `dividend` does not match `dividendo`). A post that matches several pools belongs to the **first** of them (closest to cash, as in the prompt's rule).
   - `assets_dividends`: dividend, dividends, yield, payout, book value, p/b, price to book, nav, balance sheet, net asset, net assets
   - `earnings`: p/e, pe ratio, earnings, eps, profit, profits, margin, margins, guidance, net income
   - `revenue_business`: revenue, revenues, sales, p/s, growth, subscribers, users, deliveries, units
3. Every pool is taken whole, or drawn down to `--pool-cap` (1,500) when it is larger.
4. The remaining places up to `--count` (8,000) are drawn from **every post not taken yet** (pool posts above the cap included). When the pools alone hold more than `--count` posts, the draw stops instead of cutting.
5. All the posts taken are shuffled: that is the processing order. A run stopped half-way has labelled a random part, with about the same shares of the pools.

All randomness comes from one `random.Random(seed)` of Python's standard library (pools above the cap in pool order, then the fill, then the shuffle), on the posts sorted by row, never from NumPy or pandas: two machines with other NumPy or pandas versions draw the same posts in the same order. The **selection fingerprint** is the SHA-256 of the ids in processing order, joined by newlines; `show-selection` prints it, so two machines can compare their draws before a long run. With the defaults, the draw on 1 November 2026 was: 1 empty text and 39 repeated texts left out, 38,051 usable posts; pools of 627, 2,125 and 940 posts, taken 627, 1,500 and 940; 4,933 drawn at random; first ids `tim:26653, tim:2541, tim:33761, tim:26129, tim:6044`; fingerprint `86297c21de2f125962efd1279d6bbca46cab0b107e42ad9f09c4253fe16b0c5c`.

The counts "per pool" and "random" always mean **the step that drew a post**, not which keyword it contains: the random part has posts with keywords too. Because the pools are drawn first, the shares of the levels in the label file are **not** the shares in the crowd; only the random part gives a picture of those (the full check prints it).

**The full run uses the default draw** (`--count`, `--pool-cap` and `--seed` left alone), so that every machine labels the same posts in the same order; only `--samples` and `--min-votes` may be changed, see below.

### Asking the model and counting votes

Every post is asked `--samples` times (5). Sample *i* (counted from 0) uses the seed `--seed` + *i* (42, 43, 44, 45, 46) and the temperature `--temperature` (0.7), so the samples can differ. One request goes to Ollama's `/api/chat`: a system message (what the posts are, the ladder with its meanings, the rules), sixteen hand-written example posts with their answers (at least one per label, one with two levels, two with a position only, an earnings calendar, a price level to wait for and an airdrop advert; none of them is a post of the dataset, which the label command checks before it starts), then the post itself. The request is not streamed, allows at most 64 tokens per answer (the longest answer, `{"level": "revenue_business"}`, needs about 10) and uses Ollama's structured output with a JSON Schema that allows only `{"level": <one of the seven labels>}`. The answer is read as JSON; its `level`, stripped and in lower case, must be one of the seven.

- An answer that cannot be used, or a request that takes longer than `--timeout`: the sample is asked again with a fixed reminder added to the system message and the same seed, up to `--max-attempts` (3) attempts. A sample whose attempts all fail is a **failed vote** and does not count.
- The votes: the most frequent label among the valid votes is recorded when it has at least `--min-votes` (3) votes; otherwise the post is `disputed`; with no valid vote it is `unparsed`.
- The column `votes` holds *votes of the recorded (or most frequent) label* / *valid votes*: five times `earnings` gives `earnings,5/5`; four `earnings` and one `price_chart` give `4/5`; one failed vote and three `earnings` give `3/4`; two, two and one give `disputed,2/5`; no valid vote gives `unparsed,0/0`.
- `--min-votes` must be more than half of `--samples` and at most `--samples` (5 with 3, 3 with 2); the script refuses anything else. A recorded label therefore always has a majority, and there is never a tie for first place.
- The server cannot be reached (connection refused or reset, a server error 5xx): the script waits and tries again; when that fails too, the whole run stops and the current post is not written at all, not even when some of its samples were answered. Start the server and run the same command again.

### Setting up

The script needs Python 3.12 and only five packages, listed in `requirements-prelabel.txt` with the same pins as `requirements.txt` (pandas, NumPy, pyarrow, huggingface_hub, fsspec), and Ollama with the model installed. **Installing Ollama and the model is done by a person** (`ollama pull qwen2.5:7b`, about 4.7 GB); the script never downloads a model and stops before labelling when the model is missing. It talks only to the Ollama server on a loopback address (`127.0.0.1`, `::1` or `localhost` over `http://`), refuses any other address before anything else happens, ignores every proxy setting and needs no key or account. Neither check contacts Ollama.

On macOS or Linux, the environment of [Run locally](#run-locally) has everything; a machine that only pre-labels can use the short list instead:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-prelabel.txt
```

On Windows, use **Git Bash** (it comes with Git for Windows) and the same bash scripts; `scripts/env.sh` finds the Python of a Windows virtual environment (`.venv/Scripts/python.exe`) by itself:

```bash
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-prelabel.txt
```

The repository's `.gitattributes` keeps scripts and CSV files with LF line endings on every machine (Git on Windows would otherwise turn them into CRLF, and bash cannot run a script with carriage returns). The script prints in UTF-8 and line by line, so that tweets with emoji do not stop it in a Windows terminal and its progress shows up in Git Bash.

### Running

```bash
scripts/prelabel_levels.sh show-selection          # the draw, no Ollama: pools, keyword hits, first ids, selection fingerprint
scripts/prelabel_levels.sh show-prompt             # the fixed prompt, the answer format and their fingerprint
scripts/prelabel_levels.sh label --sample --limit 40 --samples 3 --min-votes 2   # the sample of this repository
scripts/prelabel_levels.sh label                   # the full run -> labels/levels-prelabel-v2.csv
scripts/prelabel_levels.sh label --retry-unparsed  # ask again only for the unparsed rows
scripts/prelabel_levels.sh check-file labels/levels-prelabel-v2.csv   # before committing it
scripts/prelabel_levels.sh review-template         # labels/levels-review-template.csv
scripts/prelabel_levels.sh export-review           # the review rows with their texts -> data/review/levels-review-with-text.csv
```

**The order for the full run** (on the machine that labels):

1. `show-selection` and `show-prompt`: the selection fingerprint and the prompt fingerprint must be the ones above and in [Prompt versions](#level-prompt-versions).
2. A short trial into `data/` (ignored by Git), for example `scripts/prelabel_levels.sh label --output data/prelabel/trial.csv --limit 40`, and read its seconds per post. Five samples per post are 40,000 requests for 8,000 posts; at 0.3 to 0.5 seconds per request that is about 3 to 6 hours (on the laptop that labelled the sample, 0.39 seconds per request, 1.17 seconds per post with three samples). The real time is what the trial shows.
3. The full run: `scripts/prelabel_levels.sh label`. Only `--samples` and `--min-votes` may differ from the defaults, and only together: three samples need two votes (`scripts/prelabel_levels.sh label --samples 3 --min-votes 2`). They can only be chosen for an empty (or new) file: a file that already has rows keeps the settings it was started with (see **Resuming**). When the full file is labelled with three samples, write it in the run's notes, because its votes are then out of 3, not 5.
4. `scripts/prelabel_levels.sh check-file labels/levels-prelabel-v2.csv` (with `--samples` and `--min-votes` as used); commit the file only when it passes.
5. `review-template`, then `check-file labels/levels-review-template.csv --labels labels/levels-prelabel-v2.csv`, and commit it.

**Resuming.** Every labelled post is appended as one line and flushed to disk at once. A new run reads the file and skips every id already in it, `unparsed` and `disputed` ones included. A partial last line, left by a run that was stopped while writing, is dropped and that post labelled again. A file whose rows come from another model or prompt version, or whose ids or fingerprints do not fit the data, stops the run instead of being added to. **The settings record**: the first run on a file writes its settings (count, pool cap, seed, samples, temperature, min votes, attempts, model, prompt version, keyword list, data revision and selection fingerprint) to `data/prelabel/<the file's path in the repository, with / as __>.settings.json` (for a file outside the repository, the first 16 hexadecimal digits of the SHA-256 of its full path); every later run compares its own settings with it and stops, naming the setting, when one differs. A file with rows but no record (copied from another machine) gives a warning and takes this run's settings; a row whose valid votes are more than this run's `--samples` stops the run in any case. `--limit N` caps the posts asked in one run. Every run adds one line to `data/prelabel/levels-runs.jsonl` (start and end, settings, posts, requests, seconds, failures, whether and why it stopped; no text). Progress goes to the terminal; redirect it into `data/` when the run is left in the background.

**`--retry-unparsed`** asks again only for the rows written as `unparsed`; `disputed` rows are never asked again, because too few votes is a valid result and asking again would be choosing the answer. A new result that is not `unparsed` (`disputed` too) replaces its row in place (the file is rewritten through a temporary file, so its order and row count stay the same). The same seeds are used, and the same request usually gets the same answer, so it mainly helps after timeouts and passing failures, not after bad answers.

### Options of `label`

| Option | Default | Meaning |
| --- | --- | --- |
| `--sample` | off | Write to `samples/levels-prelabel-sample.csv` instead of `labels/levels-prelabel-v2.csv`. Only the output changes: the data, the draw and the order are the same. |
| `--output` | as above | Label file. |
| `--source` | the pinned dataset | A local parquet file with the dataset's columns (for tests). |
| `--count` | `8000` | Posts to draw. Must stay at the default for the full run. |
| `--pool-cap` | `1500` | Most posts taken from one keyword pool. Must stay at the default for the full run. |
| `--seed` | `42` | Draw, order and review draw; sample *i* asks the model with seed + *i*. Must stay at the default for the full run. |
| `--samples` | `5` | Questions per post. |
| `--min-votes` | `3` | Votes the most frequent label needs; more than half of `--samples` and at most `--samples`. |
| `--temperature` | `0.7` | The model's temperature; at 0 the samples would hardly differ. |
| `--model` | `qwen2.5:7b` | Ollama model; it must already be installed. |
| `--host` | `http://127.0.0.1:11434` | Ollama server; loopback addresses only. |
| `--timeout` | `120` | Seconds one request may take; a timeout is a failed attempt. The first request also loads the model. |
| `--max-attempts` | `3` | Attempts per sample: the first question and two with the reminder. |
| `--connect-retries` | `3` | Extra tries when the server cannot be reached, before the run stops. |
| `--connect-wait` | `5` | Seconds between those tries. |
| `--limit` | none | Label at most this many posts in this run. |
| `--retry-unparsed` | off | Ask again only for the `unparsed` rows. |

`show-selection` takes `--source`, `--count`, `--pool-cap` and `--seed`; it reads the data and draws without contacting Ollama and prints the rows, the posts left out, every pool's posts and how many were taken, the random part, every keyword's hits, the first five ids, the selection fingerprint, the copy's SHA-256 and the keyword list's fingerprint (no text). `review-template` takes `--labels`, `--output`, `--size` (200) and `--seed` (42); `export-review` takes `--template`, `--labels`, `--source` and `--output`, and refuses an output path inside the repository but outside `data/`. Relative paths are relative to the repository.

**`check-file <csv file>`** checks the format of a label file or of a review table, telling them apart by their header, with nothing but the standard library (no dataset, no Ollama; it runs with `requirements-prelabel.txt` alone). Both kinds: UTF-8 without a BOM (a file saved by Excel usually starts with one, and the message says so), LF line endings only, a newline at the end. A label file: the six columns; `level` one of the seven labels, `unparsed` or `disputed`; `votes` as `N/M` with N ≤ M, `unparsed` exactly `0/0`, a label with a majority (N ≥ 1 and N > M / 2), `disputed` with N ≥ 1 and M ≥ 1; ids `tim:<number>`, each once; fingerprints of 8 lower-case hexadecimal digits; one model and one prompt version in the whole file. With `--samples N` no row may count more than N valid votes; with `--min-votes K` every label needs at least K votes and every `disputed` fewer. A review table: the three columns, ids `tim:<number>` each once, `prelabel` one of the six levels, `human_label` empty or one of the six levels or `none`; with `--labels <label file>` every id must be in that label file with that label. It prints one summary line and exits with 0, or lists the first problems with their line and column and exits with 1. `review-template` runs the same checks on the label file first. **Run it on the label file before committing it, and on the review table after filling it in.**

### The level files

| File | Rows |
| --- | --- |
| `labels/levels-prelabel-v2.csv` | the 8,000 posts of the default draw, labelled on a Windows machine (RTX 4060 Ti) with **`gemma3:12b`, five samples and three votes**, 13.5 hours without a break: 7,292 rows with 5/5 votes, 353 with 4/5, 337 with 3/5, 18 `disputed` (2/5), 0 `unparsed`; per label 695 `assets_dividends`, 991 `earnings`, 682 `revenue_business`, 1,186 `price_chart`, 2,506 `story_macro`, 1,157 `emotion_only`, 765 `none`. A run that adds to this file (resuming, `--retry-unparsed`) must pass `--model gemma3:12b --samples 5 --min-votes 3`, or the script stops |
| `samples/levels-prelabel-sample.csv` | the first 40 posts of the same processing order, labelled on a laptop with **`qwen2.5:7b`, three samples and two votes**: not the model of the full file |
| `labels/levels-review-template.csv` | the review table, 200 rows (made from the full file) |

Both label files have the same six columns, in this order, one row per post in processing order:

| Column | Meaning |
| --- | --- |
| `post_id` | The post's id, `tim:<row>` (see [The data](#the-data)). |
| `level` | One of the seven labels, `unparsed` or `disputed`. |
| `votes` | Votes of the recorded (or most frequent) label / valid votes, for example `4/5`. |
| `model` | The Ollama model that gave the label (`gemma3:12b` in the full file, `qwen2.5:7b` in the sample). |
| `prompt_version` | The prompt version (`v2`). |
| `text_hash` | The text fingerprint (see [The data](#the-data)). |

Unlike the emotion files, the sample and the full file number their posts the same way (both come from the same parquet file); they are still two files, because they were labelled with different settings. A label file never mixes models, prompt versions or settings; the script stops instead. **Do not open and save these files with Excel**: it adds a BOM and CRLF line endings, which `check-file` refuses.

Where the files come from: the ids and fingerprints refer to the TimKoornstra dataset (MIT License); the labels were given by a local model through Ollama with prompt v2: silver labels from a model, not human annotation.

### Level prompt versions

The prompt lives only in the script (never in a separate file, whose line endings could change on another machine): `PROMPT_VERSION = "v2"`, the labels and their meanings (`LEVELS_V2`), the hand-written examples (`EXAMPLES_V2`), the system message, the retry reminder and the answer format. Its fingerprint is the SHA-256 of `json.dumps({"first": <messages of a first request>, "retry": <messages of a retry>, "format": <answer format>}, ensure_ascii=True, sort_keys=True, separators=(",", ":"))`, with the placeholder `<post>` in place of the post; for v2 it is `3f8c06a7371553ff57b3894fbfabf2308743542c5089b226d0a40ab8ea0ce7fa`. **Version v1** asked for the reason the writer gives; it labelled only a first trial of the 40 sample posts, was replaced before any file labelled with it stayed in the repository, and is not in the script any more. **The released versions start with v2.** `show-prompt` prints the whole prompt and the fingerprint, and the quick check pins it (`LEVEL_PROMPT_FINGERPRINTS` in `scripts/check.py`). Once a file labelled with a version is committed, the text of that version never changes: a changed character needs a new version in the code, new file names and a new line in `LEVEL_PROMPT_FINGERPRINTS`. The seven labels and the two failure values are part of the files' format and do not change with a new prompt.

### The level review table

`labels/levels-review-template.csv` holds 200 rows for a person to check, with three columns: `post_id`, `prelabel` (the model's level) and `human_label` (empty). The rows are drawn from `labels/levels-prelabel-v2.csv`, leaving out `none`, `unparsed` and `disputed`, **taking the six levels in turn**: the rows of every level (in file order) are shuffled with their own `random.Random(42)`; then, round after round, every level in ladder order gives its next row, a level with no rows left being skipped, until 200 rows are drawn or none are left. The rows of the table are in drawing order. With enough rows in every level this gives 33 or 34 rows per level; a rare level gives all its rows and the other levels fill its places.

To fill it in, run `scripts/prelabel_levels.sh export-review`: it writes the same rows, in the same order, with the original texts to `data/review/levels-review-with-text.csv`, which Git ignores; **this file with the texts must never enter the repository.** Read a text there and write into `human_label` of the template one of the six levels, or `none` when the post does not talk about stocks or markets; then run `check-file labels/levels-review-template.csv --labels labels/levels-prelabel-v2.csv`. A level's agreement is the share of its rows whose `human_label` equals `prelabel`. Because every level has about the same number of rows in the table while the levels are not equally common, **an overall agreement must weight each level's agreement by that level's share of the rows of `labels/levels-prelabel-v2.csv` that have one of the six levels**; the plain share of matching rows in the table is not the overall agreement.

## Known limitations

- The notebook has not been run on Colab itself yet; it has been run locally with the same code path, including reading the `hf://` address directly.
- The dataset is not only the crowd. It merges nine public datasets: part of its posts are financial news headlines posted as tweets, part are retail traders writing in their own words, and one of the nine sources is about cryptocurrencies, so some posts name coins rather than stocks. Its labels come from those sources, with their own rules.
- The posts are in English only.
- There is no independent test set: the test set is cut from the same dataset (10% of the working sample), so the scores describe posts like the training posts, not posts from another source or another year.
- Repeated posts are found only when the text is exactly the same. Near repeats (the same headline with another link, or other capitals) can land in the training set and in the test set at once and make the test scores a little too good; chapter 8 counts the test posts whose original or cleaned text also appears in the training set.
- The models learn from a working sample with the same number of posts in every class, so they never learned the natural shares of the crowd. A share of bullish, neutral and bearish posts read from the models counts the model's call for every post, one by one (`score_texts`, the column `call`); it is not corrected for the models' errors.
- The ticker rule also counts codes with a suffix after the dot that are not stock classes, for example `$DOGE.X`; 40 of the 38,091 posts name a code with a dot and a suffix, most of them share classes such as `$BRK.B`.
- The quick check's sample holds no post with an account name or a link, so it is not a picture of the dataset: in the full dataset many posts carry one.
- The averaged word vectors are trained on the training set only (about 20,000 short posts with the defaults), much less text than ready-made word vectors learn from, so the network on them can score lower than the networks on word counts and TF-IDF.
- Training is seeded, but TensorFlow does not promise bit-identical results across runs and machines, and Word2Vec's starting point also depends on Python's hash of strings, so model scores (and, when the scores are close, the chosen model) may differ slightly. Counts and the two baseline rows do not depend on training.
- The level pre-labels are silver labels from a 7-billion-parameter model, not human annotation; how often they are right is unknown until the level review table is filled in. With a temperature of 0.7 and fixed seeds the same post usually gets the same votes again, but Ollama does not promise it across versions and machines.
- The level pre-labels draw the posts that match the keywords of the first three levels first, so the shares of the levels in `labels/levels-prelabel-v2.csv` lean towards those three levels and are **not** the shares in the crowd; only the randomly drawn part (the full check prints it) gives a picture of those.
- Repeated texts are found case by case: `$TSLA to the moon` and `$tsla to the moon` both stay in the draw. When training and test sets are cut from the level labels, such near repeats can land on both sides and make the test scores a little too good.

## Data policy

The full dataset and everything computed from it (caches, executed notebooks, briefs, tables and quotes produced from the full data) stay in `data/`, which Git ignores. They are never committed, also not in this README or in `docs/`. Only the sample above and examples produced from the sample (or written by hand) may be committed; the examples in `docs/brief-format.md` and the sentences in the notebook are written by hand.

One exception: the three files of [Level pre-labels](#level-pre-labels) (`samples/levels-prelabel-sample.csv`, `labels/levels-prelabel-v2.csv` and `labels/levels-review-template.csv`) are computed from the dataset, but they hold only post ids, levels, vote counts, the model name, the prompt version and a short text fingerprint, not a single character of any post, so they are committed. The copy of the dataset (`data/raw/tim-financial-tweets-bba60269.parquet`), the settings records and run log in `data/prelabel/`, and the level review rows with their texts (`data/review/levels-review-with-text.csv`) stay in `data/`.
