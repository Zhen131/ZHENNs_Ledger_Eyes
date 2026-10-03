# Ledger Eyes

Ledger Eyes reads what the crypto crowd writes and turns it into one small, fixed-format result file, the **brief** (`brief.json`). The brief is the interface: other programs read it, not the notebook.

The work is a single Jupyter notebook that runs from top to bottom on Google Colab. It uses the public StockTwits crypto dataset from Hugging Face and needs no keys or accounts.

Chapters so far: 0 The question and the data · 1 Loading · 2 A first look · 3 Cleaning · 4 Tokenization · 5 Training, validation and test sets · 6 Vectorization · 7 Models · 8 Evaluation · Save the brief. Chapters 4 to 8 tell bullish from bearish posts and fill the brief's `sentiment` section; the `topics` and `why` sections are still `null`.

## Repository layout

| Path | What it is |
| --- | --- |
| `notebook/ledger_eyes.py` | The notebook source in jupytext "percent" format (`# %%` marks a cell). **This is the only notebook file to edit.** |
| `notebook/ledger_eyes.ipynb` | The notebook generated from the source; upload this one to Colab. Stored without outputs. |
| `samples/stocktwits-crypto-sample.xlsx` | 600 posts from the public dataset, used by the quick check (see [The sample](#the-sample)). |
| `scripts/check.sh` | The two checks: `quick` and `full`. |
| `scripts/build_notebook.sh` | Regenerates the `.ipynb` from the source. |
| `scripts/download_data.sh` | Downloads the full dataset into `data/raw/`. |
| `scripts/make_sample.py` | Draws the sample from the downloaded dataset. |
| `scripts/env.sh` | Shared environment for the scripts (caches under `data/`, no token, no telemetry). |
| `requirements.txt` | Pinned package versions (Python 3.12). |
| `data/` | Downloads, caches, executed notebooks and briefs. Ignored by Git. |

## Run on Google Colab

1. Open <https://colab.research.google.com>, choose *File → Upload notebook* and pick `notebook/ledger_eyes.ipynb`.
2. Optionally change the switches in the first code cell.
3. Choose *Runtime → Run all*.

The first code cell after the switches installs the packages Colab lacks (for example `emoji` and `contractions`); everything else, including TensorFlow/Keras and Hugging Face `tokenizers`, is preinstalled on Colab, and a package that is already there is left as it is. The data is read straight from Hugging Face (`hf://datasets/ElKulako/stocktwits-crypto/st-data-full.xlsx`), and the brief is saved to `data/brief.json` inside the Colab session.

The notebook has not yet been run on Colab itself; it has been run locally with the same code path, including reading the `hf://` address directly.

## Run locally

Requirements: Python 3.12 and a network connection for the first setup (package index, Hugging Face, NLTK data).

```bash
# 1. Environment
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Full dataset into data/raw/ (69.6 MB, only needed for the full check)
scripts/download_data.sh

# 3. Checks
scripts/check.sh quick
scripts/check.sh full
```

The scripts run under `bash` and set up their own environment from `scripts/env.sh`: Hugging Face cache in `data/hf_home`, NLTK data in `data/nltk_data`, pip cache in `data/pip-cache`, Keras settings in `data/keras`, Matplotlib, IPython and Jupyter settings under `data/`; no stored Hugging Face token is ever sent (`HF_HUB_DISABLE_IMPLICIT_TOKEN=1`), Hugging Face telemetry is off, and pip does not check for its own updates. The scripts use `.venv/bin/python` unless `EYES_PYTHON` points to another interpreter.

### The two checks

**`scripts/check.sh quick`** (about 25 seconds once the environment is ready)

1. `requirements.txt` matches the installed packages, and the notebook's install pins match `requirements.txt`.
2. `notebook/ledger_eyes.ipynb` is exactly what `scripts/build_notebook.sh` produces from the source ("source file and .ipynb are consistent"), and it carries no outputs or execution counts (".ipynb carries no run results").
3. The parameters cell is well formed and holds every switch listed below.
4. Text scan (rules in [Text rules](#text-rules)).
5. The notebook runs from top to bottom on the sample twice: once with the default switches and once with every two-state switch flipped to its other value. Both runs must save a brief, and each brief is checked: all top-level fields present, `dataset` filled, `scope` matching the switches, `post_id` unique, class counts adding up, equal class sizes in the working sample, `clean_texts` reproducing the step-by-step cleaning, and the cleaning examples following the switches (negations kept and emoji turned into words with the defaults; with both cleaning switches off, the examples must equal the plain course recipe).
6. The `sentiment` section of both briefs is checked for structure only (the sample is too small for the scores to mean anything): the classes match `N_CLASSES` (three classes in the second run); the tokenization comparison is recorded; training, validation and test sets add up to the working sample (about 80/10/10, every class in every part); both vectorizers learned at most `VOCAB_SIZE` words from the training set only; the evaluation table has exactly one row per model (`count_network`, `tfidf_network`) and per baseline (`always_majority`, `uniform_random`) for all test posts and for the test posts that mention Bitcoin, with the right number of posts, class supports that add up and every score a number in [0, 1] or `null` (`null` for accuracy and macro-F1 when a subset is empty); one confusion matrix per model adding up to the number of test posts and agreeing with the accuracy; one model chosen with a validation score for every model; the scoring function's self-check passed (one column per class, bullish last, rows adding up to 1, the chosen model's evaluated test labels reproduced); the count of test texts also found in the training set; `beats_baselines` agreeing with the table. The switches `BPE_VOCAB_SIZE`, `VOCAB_SIZE`, `EPOCHS` and `BATCH_SIZE` must exist.

**`scripts/check.sh full`** (about one minute once the environment is ready)

Runs the same static checks, then the notebook once on the full dataset in `data/raw/st-data-full.xlsx` with the default switches, checks the brief as above and prints the key numbers (posts, posts per class, posts mentioning Bitcoin per class, working sample size, average words before and after cleaning). On top of the structure checks of the quick check, it requires that the rows for the test posts that mention Bitcoin hold posts, and **the bar of chapter 8: every model's macro-F1 on all test posts must be higher than that of both baselines** (with whatever switches the full run uses). It then prints the sentiment numbers: tokenization, split, validation scores and the chosen model, the full evaluation table and the confusion matrices.

Executed notebooks and briefs land in `data/checks/quick-default/`, `data/checks/quick-other/` and, for the full check, `data/runs/full.ipynb` and `data/brief.json`. If you pass `-p WORK_DIR <folder>` to the quick check, both of its runs use that one folder: each brief is checked right after its run, but the second run overwrites the first run's `executed.ipynb` and `brief.json`. `WORK_DIR` may also be a folder outside the repository.

### Changing switches without editing the notebook

Both checks accept `-p NAME VALUE` (repeatable). The value is read as a Python literal when possible, otherwise as text, and applies to every run of that check:

```bash
scripts/check.sh full -p SAMPLE_PER_CLASS 20000
scripts/check.sh full -p DATA_SOURCE hf://datasets/ElKulako/stocktwits-crypto/st-data-full.xlsx -p WORK_DIR data/hf-run
scripts/check.sh quick -p RANDOM_SEED 7
```

The checks pass the values to the notebook with [papermill](https://papermill.readthedocs.io), which replaces the cell tagged `parameters`. Without the scripts:

```bash
.venv/bin/papermill notebook/ledger_eyes.ipynb data/out.ipynb -p N_CLASSES 3 -p BTC_ONLY True
```

## Switches

All switches are in the notebook's first code cell (tag `parameters`), one plain assignment per line, each with a comment of the form `# default: ...; ...`. Results are reported with the defaults. A switch whose comment says `two-state, other: X` has exactly two values; the quick check flips all of them together to `X` for its second run, so a new two-state switch joins that run automatically.

| Switch | Default | Other values | Meaning |
| --- | --- | --- | --- |
| `DATA_SOURCE` | `hf://datasets/ElKulako/stocktwits-crypto/st-data-full.xlsx` | a local `.xlsx` path with the same layout | Where the posts come from. |
| `WORK_DIR` | `data` | any folder | Cache and saved brief. |
| `RANDOM_SEED` | `42` | any integer | Seed for the working sample. |
| `SAMPLE_PER_CLASS` | `50000` | any positive integer | Most posts per class in the working sample. |
| `N_CLASSES` | `2` | `3` (two-state) | 2 = bearish vs bullish; 3 adds neutral. |
| `BTC_ONLY` | `False` | `True` (two-state) | Working sample only from posts that mention Bitcoin. |
| `EMOJI_TO_WORDS` | `True` | `False` (two-state) | Turn emoji into English words instead of deleting them. |
| `KEEP_NEGATIONS` | `True` | `False` (two-state) | Keep not, no, never, nor (and the n't forms) out of the stop word list. |
| `BPE_VOCAB_SIZE` | `5000` | any positive integer | Most pieces the subword (BPE) tokenizer of chapter 4 learns. Display only. |
| `VOCAB_SIZE` | `1000` | any positive integer | Most words each vectorizer of chapter 6 keeps (as in the course). |
| `EPOCHS` | `5` | any positive integer | Passes over the training set in chapter 7 (as in the course). |
| `BATCH_SIZE` | `50` | any positive integer | Posts per training step in chapter 7 (as in the course). |

## Interface

These names are fixed; later chapters build on them.

| Name | Meaning |
| --- | --- |
| `df_all` | Every post of the data source. Created in chapter 1 and kept unchanged until the end of the notebook; no later cell may drop or overwrite it. |
| `df` | The working sample drawn in chapter 2 (see below). |
| `post_id` | Stable id of a post: `<sheet name>:<row number in the sheet>`, for example `stocktwits_1:2` for the first post of the first sheet (row 1 holds the column names). Unique in `df_all`, derived from the position in the file and never from random numbers, so the same data source gives the same ids on every run. `df` and every later table carry it. Ids are only comparable between runs on the same data source. |
| `text` | The original post text. A cell that pandas reads as empty becomes an empty string; in the full file these are 59 formula cells (see [Known limitations](#known-limitations)). |
| `label` | 0 = bearish, 1 = neutral, 2 = bullish. The meaning never changes, also not in the working sample. |
| `mentions_btc` | `True` when the text contains `bitcoin` or `btc` as a whole word, ignoring case (regular expression `\b(?:bitcoin\|btc)\b`). "bitcoin's" counts; "bitcoins", "gbtc" and "btc_usd" do not. |
| `clean_text` | Column of `df` only: the text after chapter 3's cleaning. |
| `clean_texts(texts)` | Cleans a column (or list) of raw texts with the current switches and returns a pandas Series (same index when given a Series). It runs the same step functions as chapter 3's step-by-step cells, in the same order; the notebook checks that both give identical results. |
| `brief` | The result dictionary, saved to `{WORK_DIR}/brief.json` by the last cell. |
| `CLASS_LABELS` | Labels in use, in order: `[0, 2]` with two classes, `[0, 1, 2]` with three. Every class table and every probability array follows this order. |
| `train_df`, `val_df`, `test_df` | Training (80%), validation (10%) and test (10%) parts of `df`, split in chapter 5, stratified by label, with `RANDOM_SEED`. They keep all columns of `df`. |
| `train_post_ids` | `frozenset` of the `post_id`s in `train_df`. A post whose id is not in it was never seen by the models during training. |
| `count_vectorizer`, `tfidf_vectorizer` | Chapter 6's vectorizers, fitted on `train_df["clean_text"]` only (also in the dict `vectorizers`, keys `"count"` and `"tfidf"`). |
| `models` | Chapter 7's networks: `{"count_network": {...}, "tfidf_network": {...}}`, each with `model` (the Keras model), `vectors` (`"count"` or `"tfidf"`) and `history`. |
| `selected_model_name`, `selected_model`, `selected_vectorizer` | The model chosen by macro-F1 on the validation set, and the vectorizer it reads. |
| `sentiment_proba(cleaned_texts)` | Class probabilities from the chosen model for any batch of **cleaned** texts (a list or pandas Series of strings; clean raw posts with `clean_texts` first; a missing value counts as an empty text). Returns a NumPy array of shape `(number of texts, number of classes in use)`; every row adds up to 1; an empty input gives shape `(0, number of classes)`. Column order is fixed and stored in `SENTIMENT_PROBA_LABELS`: two classes `[0, 2]` = [bearish, bullish]; three classes `[0, 1, 2]` = [bearish, neutral, bullish]. |
| `SENTIMENT_PROBA_LABELS` | The label of each column returned by `sentiment_proba` (equal to `CLASS_LABELS`). |
| `BULLISH_COLUMN` | Index of the "probability of bullish" column: always the last one (1 with two classes, 2 with three). |

**Working sample.** Keep the classes in use (`N_CLASSES = 2` drops neutral), keep only Bitcoin posts when `BTC_ONLY`, then draw the same number of posts from every class with `RANDOM_SEED`: the smaller of `SAMPLE_PER_CLASS` and the size of the smallest remaining class. Chapter 2's statistics use `df_all`; chapter 3 cleans `df` only.

**Cleaning.** Lower case → extra whitespace → HTML tags → emoji to words (only with `EMOJI_TO_WORDS`; skin-tone qualifiers dropped, underscores turned into spaces) → special characters (keep letters, digits, whitespace) → contractions (`contractions` package; it also expands apostrophe-free forms such as "dont" and "cant", so negative contractions come back as "not") → stop words (NLTK English list, minus the negation words when `KEEP_NEGATIONS`) → WordNet lemmatisation.

**Sentiment models.** Chapter 4 only shows two tokenizers (NLTK words and a BPE tokenizer trained on the original text of the working sample); nothing from it reaches a model. Chapter 5 splits `df` 80/20 and then the 20% part half and half, both times stratified by label with `RANDOM_SEED`; every class needs at least 10 posts in the working sample, otherwise the notebook stops with a clear message. Chapter 6 fits a `CountVectorizer` and a `TfidfVectorizer` (`max_features=VOCAB_SIZE`, 32-bit floats) on the training set's `clean_text` and only transforms the validation and test sets; the matrices stay sparse instead of `.toarray()`. Chapter 7 trains one Keras network per vector type: a single `Dense` layer, with one sigmoid unit and binary cross-entropy for two classes (inside the network bearish → 0, bullish → 1) or three softmax units and sparse categorical cross-entropy for three classes (labels unchanged), Adam, `EPOCHS` and `BATCH_SIZE`, after `tf.keras.utils.set_random_seed(RANDOM_SEED)`. The validation set is watched during training and then used once to choose the model with the higher macro-F1 (a tie goes to `count_network`). Chapter 8 is the only place that scores the test set.

**Cache.** Chapter 1 stores `df_all`'s three loaded columns as Parquet in `{WORK_DIR}/cache/posts-<fingerprint>.parquet`. The fingerprint of a local file is a SHA-256 hash of its bytes (any edit forces a fresh read); for a web address it is a hash of the address.

### The brief

Top level, always present: `format_version` (a draft value for now), `generated_at` (UTC, ISO 8601), `scope`, `dataset`, `sentiment`, `topics`, `why`. Sections no chapter fills are `null`. Values that cannot be computed (for example a share with a zero denominator) are `null`, never 0 or NaN: None, NaN and infinity are converted to `null` before saving, and the JSON writer also refuses NaN as a safety net.

- `scope`: `source` (the `DATA_SOURCE` value), `granularity` (`"batch"`: one result for the whole dataset; `"daily"` is reserved for dated data), `coins` (`"all"` or `"btc_only"`, describing the working sample), `n_classes`.
- `dataset`: `name`, `file`, `license`, `citation`, `label_meaning`, `sheets` (posts per sheet), `n_posts`, `n_unique_post_ids`, `n_empty_text` (texts that are empty or blank after reading; in the full file, the 59 formula cells), `class_counts`, `class_shares`, `length_words` (`mean`, `median`, `p90`, `p99`, `min`, `max`), `btc_mentions` (`rule`, `n_posts`, `share`, `by_class`, `share_within_class`), `duplicates` (`n_rows_repeating_an_earlier_text`, `n_distinct_texts_repeated`), `n_posts_with_links`, `sample` (`n_classes`, `btc_only`, `seed`, `per_class_requested`, `available_by_class`, `per_class`, `class_counts`, `n_posts`, `n_mentions_btc`), `cleaning` (`emoji_to_words`, `keep_negations`, `steps`, `avg_words_before`, `avg_words_after`, `n_empty_after`, `clean_texts_matches_steps`, `examples`). Class keys are the label values as strings (`"0"`, `"1"`, `"2"`).
- `sentiment` (chapters 4 to 8):
  - `tokenization`: `used_by_models` (`false`), `text`, `n_posts`, `nltk_word_tokenize` and `bpe` (`vocab_size`, `avg_tokens_per_post`; `bpe` also `vocab_size_requested`, `trained_on`), `examples` (`kind`, `post_id` of the posts shown).
  - `classes`: the labels in use as strings, for example `["0", "2"]`.
  - `split`: `seed`, `fractions`, `stratified_by`, and per part (`train`, `validation`, `test`) `n_posts`, `class_counts`, `class_shares`, `n_mentions_btc`.
  - `vectorizers`: `vocab_size_requested`, `fitted_on` (`"train"`), `fitted_on_n_posts`, `text`, `storage`, `count` and `tfidf` (`vocab_size`).
  - `models`: the model names, `["count_network", "tfidf_network"]`.
  - `training`: `epochs`, `batch_size`, `optimizer`, `seed`, `output_layer`, `loss`, `target_of_label` (label → network target), `history` (per model, the loss and accuracy of every pass on the training and validation sets).
  - `selection`: `by`, `validation_macro_f1` (per model), `selected`.
  - `scoring`: `function` (`"sentiment_proba"`), `input`, `output`, `column_labels`, `bullish_column`, `model`, `train_ids` (`"train_post_ids"`), `n_train_ids`, `check` (`n_texts`, `n_columns`, `rows_sum_to_one`, `matches_evaluated_predictions`: the self-check on the test set).
  - `test_texts_also_in_train`: test posts whose original text (`same_text`) or cleaned text (`same_clean_text`) also occurs in the training set. Counted only; nothing is removed.
  - `evaluation`: `subsets` (`all_test`, `test_mentions_btc`), `baselines` (`always_majority`: the most frequent training label, the smallest label on a tie; `uniform_random`: scikit-learn's `DummyClassifier(strategy="uniform")` with `RANDOM_SEED`), `majority_label`, `rows`. One row per model and baseline and per subset: `model`, `kind` (`"model"` or `"baseline"`), `subset`, `n`, `accuracy`, `per_class` (per label: `precision`, `recall`, `f1`, `support`), `macro_f1`. A score that cannot be computed is `null` (scikit-learn's `zero_division=nan`); macro-F1 is the mean of the per-class F1 scores that can be computed, and `null` when none can.
  - `beats_baselines`: per model, whether its macro-F1 on all test posts is higher than that of both baselines.
  - `confusion_matrices`: per model, on all test posts: `labels`, `rows` (true label), `columns` (predicted label), `counts`, `n`.

## Editing the notebook

1. Edit `notebook/ledger_eyes.py` only.
2. Regenerate the notebook: `scripts/build_notebook.sh` (writes `notebook/ledger_eyes.ipynb`; cell ids come from the cell content, so an unchanged source gives a byte-identical file).
3. Run `scripts/check.sh quick`.

Writing rules for the notebook: one cell does one thing; every code cell, or short run of related code cells, is preceded by an explanation cell; explanation cells are English, then a line with `***`, then Chinese; code comments are English only. New switches go into the parameters cell in the same form, all package installs stay in the single install cell right after it, and the "Save the brief" cell stays last.

### Text rules

The quick check scans every text file in the repository outside `data/` and `.venv/` and rejects:

- private numbering, in the forms the private log uses: file-name prefixes (two digits, optionally one of the capital letters A–D or Q, optionally a round suffix `-R` or `-` plus a number, then `_W` and a week number), review codes (two digits, a capital A–D, then `-R`), `W` followed by two digits, `Q` followed by two digits, and week numbers written in Chinese. A bare "two digits and a letter" is deliberately not flagged, because it collides with normal notation such as 50D (50 days) or 10B (10 billion). Before scanning, the check tests these rules on built-in examples (`NUMBERING_MUST_MATCH` and `NUMBERING_MUST_NOT_MATCH` in `scripts/check.py`): every log code form must be caught and notation such as 50D, 30D, 7D, 10B or W2 must not be; if either fails, the check fails;
- names of chat assistants and their vendors, the two-letter capitalised abbreviation for artificial intelligence, and co-author trailers;
- local home paths and user names.

Chinese text is allowed only in the notebook's explanation cells, after the `***` line. The exact patterns are in `FORBIDDEN_PATTERNS` in `scripts/check.py`.

## The sample

`samples/stocktwits-crypto-sample.xlsx` holds 600 posts taken unchanged from the public dataset, in the same layout as the original file (sheets `stocktwits_1` and `stocktwits_2`, columns `text` and `label`), so the notebook reads it with the same code. `scripts/make_sample.py` drew it with seed 42 from `st-data-full.xlsx`: for each class (bearish, neutral, bullish) 80 posts that mention Bitcoin, 20 other posts with emoji, 20 other posts with a negation and 80 other posts, 200 per class. Rows keep their original sheet and order. Posts that read as empty (the formula cells described below), start with `=` or contain characters a spreadsheet cannot store were skipped, so the sample contains no formula cells.

Source: [ElKulako/stocktwits-crypto](https://huggingface.co/datasets/ElKulako/stocktwits-crypto) on Hugging Face, released under the MIT License. For academic use the dataset author asks to cite <https://ieeexplore.ieee.org/document/10223689>.

## Known limitations

- **59 posts lose their text.** In `st-data-full.xlsx`, 59 posts begin with `=`, so the spreadsheet stores them as formulas whose stored value is an error such as `#NAME?` (for example row 365284 of `stocktwits_1`, originally "=the new btc"). pandas, on Colab as well as locally, reads these cells as empty, so they appear in `df_all` as empty strings with their labels intact and are counted in `n_empty_text`. That is 59 of 1,331,697 posts; the notebook keeps them as they are rather than reading the file a second way. Five of them mention Bitcoin, so the Bitcoin count is 5 lower than it would be with the original text.
- The notebook has not been run on Colab itself yet.
- Training is seeded, but TensorFlow does not promise bit-identical results across runs and machines, so model scores (and, when the two are close, the chosen model) may differ slightly. Counts and the two baseline rows do not depend on training.

## Data policy

The full dataset and everything computed from it (caches, executed notebooks, briefs, tables and quotes produced from the full data) stay in `data/`, which Git ignores. They are never committed, also not in this README or in a future `docs/` folder. Only the sample above and examples produced from the sample may be committed.
