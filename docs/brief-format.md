# The brief: format reference

This document describes format **1.0** of the brief, the file `brief.json` that the notebook writes at the end of every run. Other programs read the brief, not the notebook, so this document is written for the authors of those programs.

- **The rules in machine-readable form** are in [`brief.schema.json`](brief.schema.json) (JSON Schema, draft 2020-12). The notebook's last chapter holds an identical copy of that file and checks every brief against it before saving; a brief that breaks a rule is not saved.
- **The checks** (`scripts/check.sh quick` and `full`) make sure that the two copies of the rules are identical, that the tables below list exactly the fields the rules allow (with the same type, the same "required" and the same "null"), and that every brief they produce passes the rules.

## Conventions

- **File.** UTF-8 JSON, indented. Field names are English and never change within a major version (see [Versions](#versions)).
- **null.** A number that cannot be computed is `null`: a share whose denominator is 0, a score on an empty subset, a summary of an empty table. It is never written as 0, and never as NaN or Infinity, which are not valid JSON. Only the fields marked "Null: yes" below may be `null`. Before saving, the notebook turns every NaN or infinite number into `null`; the check that follows then rejects the brief if that `null` sits in a field that may not be `null`. The check also rejects NaN and Infinity themselves, so a brief that contains them never passes.
- **Labels.** `0` = bearish, `1` = neutral, `2` = bullish, the dataset's own coding. As keys of an object they are strings (`"0"`, `"1"`, `"2"`); in `topics.topics[].quotes[].label` the label is an integer.
- **Classes in use.** With two classes (the default) the models know only `"0"` and `"2"`; with three classes also `"1"`. Objects keyed by "classes in use" always have the keys `"0"` and `"2"`, and `"1"` only with three classes.
- **Shares and percentage points.** A share is a fraction from 0 to 1 (0.25, not 25). A difference "in percentage points" (`_pp`) is 100 × (one share − another), from −100 to 100.
- **Counts** are integers, 0 or more.
- **Text fields that describe a method** (for example `why.bullish_share`) are short English sentences for people; programs should not parse them.

How to read the tables:

| Column | Meaning |
| --- | --- |
| Field | The path from the top of the brief. `[]` stands for every item of a list; `<label>`, `<model>`, `<metric>` and `<sheet>` stand for every key of an object whose keys vary (described in the row of that object). |
| Type | The JSON type of the value when it is not `null`: string, integer, number, boolean, object or array. "number" also allows whole numbers. |
| Required | "yes": always present in format 1.0. "no": may be absent. "—": an item of a list or an entry of an object with varying keys; the row above it says which ones are there. |
| Null | "yes": may be `null` (only when the number cannot be computed). "no": never `null`. |

The rules allow no other fields: a writer that adds a field without changing the rules (and this document, and the version) produces a brief that does not pass.

## What each section covers

The top-level `scope` describes the brief as a whole: where the posts come from, the granularity, how many classes the models use, and which coins the **working sample** holds. The sections do not all use the same posts, so **each section's coverage is given by its own fields, not by `scope`**:

| Section | Posts it uses | Fields that say so |
| --- | --- | --- |
| `dataset` | Every post of the data source; then the working sample drawn from them. | `dataset.n_posts`; `dataset.sample` |
| `sentiment` | The working sample (with the default switches: posts about all coins, bearish and bullish only, the same number in each class), split into training, validation and test sets. The evaluation table has rows for all test posts and for the test posts that mention Bitcoin. | `sentiment.classes`; `sentiment.split`; `sentiment.evaluation.subsets` |
| `topics` | The topic post table, taken from every post of the data source, not from the working sample, with all three classes in their natural shares. With the default switches: every post that mentions Bitcoin. | `topics.scope`; `topics.n_posts`; `topics.sampled`; `topics.class_counts` |
| `why` | The same posts as `topics`, scored by the model chosen in `sentiment`. The fair comparison uses only the posts the model never saw. | `why.post_table`; `why.model`; `why.overall.n_posts`; `why.fair_comparison` |

For example, with the default switches `scope.coins` is `"all"`, because the models learn from posts about all coins, while `topics.scope` is `"btc"`, because the topics and the topic × bullish table use only posts that mention Bitcoin.

## Fields

### Top level and scope

| Field | Type | Required | Null | Meaning |
| --- | --- | --- | --- | --- |
| `format_version` | string | yes | no | The format version, `"1.0"` for this document. See [Versions](#versions). |
| `generated_at` | string | yes | no | When the brief was saved: UTC time, ISO 8601, to the second, ending in `Z` (for example `2026-01-31T12:00:00Z`). |
| `scope` | object | yes | no | What the brief as a whole covers. |
| `scope.source` | string | yes | no | The data source as given by the switch `DATA_SOURCE`: a Hugging Face address (`hf://...`) or a path to a local spreadsheet. |
| `scope.granularity` | string | yes | no | `"batch"`: every section describes the whole data source at once. Format 1.0 allows no other value; `"daily"` is reserved (see [Daily results](#daily-results)). |
| `scope.coins` | string | yes | no | Which posts the working sample holds: `"all"` (posts about all coins) or `"btc_only"` (only posts that mention Bitcoin). It describes the working sample only; see [What each section covers](#what-each-section-covers). |
| `scope.n_classes` | integer | yes | no | Classes the models use: 2 (bearish and bullish) or 3 (with neutral). |

### dataset

Chapters 0 to 3: the data source, a first look at every post, the working sample and the cleaning.

| Field | Type | Required | Null | Meaning |
| --- | --- | --- | --- | --- |
| `dataset` | object | yes | no | The data and how it was prepared. |
| `dataset.name` | string | yes | no | Hugging Face dataset id, `"ElKulako/stocktwits-crypto"`. |
| `dataset.file` | string | yes | no | The file inside the dataset, `"st-data-full.xlsx"`. |
| `dataset.license` | string | yes | no | The dataset's license, `"MIT"`. |
| `dataset.citation` | string | yes | no | Address of the paper the dataset author asks to cite. |
| `dataset.label_meaning` | object | yes | no | The meaning of each label. |
| `dataset.label_meaning.0` | string | yes | no | `"bearish"`. |
| `dataset.label_meaning.1` | string | yes | no | `"neutral"`. |
| `dataset.label_meaning.2` | string | yes | no | `"bullish"`. |
| `dataset.sheets` | object | yes | no | Posts per sheet of the spreadsheet, keyed by sheet name, in file order (at least one sheet). |
| `dataset.sheets.<sheet>` | integer | — | no | Number of posts in that sheet. |
| `dataset.n_posts` | integer | yes | no | Every post of the data source. |
| `dataset.n_unique_post_ids` | integer | yes | no | Distinct `post_id` values; equal to `n_posts`. |
| `dataset.n_empty_text` | integer | yes | no | Posts whose text is empty or blank after reading (cells stored as formulas read as empty; see the README's known limitations). |
| `dataset.class_counts` | object | yes | no | Posts per label, over every post. |
| `dataset.class_counts.0` | integer | yes | no | Bearish posts. |
| `dataset.class_counts.1` | integer | yes | no | Neutral posts. |
| `dataset.class_counts.2` | integer | yes | no | Bullish posts. |
| `dataset.class_shares` | object | yes | no | Each label's share of every post. |
| `dataset.class_shares.0` | number | yes | yes | Share of bearish posts; `null` when there are no posts. |
| `dataset.class_shares.1` | number | yes | yes | Share of neutral posts; `null` when there are no posts. |
| `dataset.class_shares.2` | number | yes | yes | Share of bullish posts; `null` when there are no posts. |
| `dataset.length_words` | object | yes | no | Post length in words (runs of characters without spaces), over every post. |
| `dataset.length_words.mean` | number | yes | yes | Mean; `null` when there are no posts. |
| `dataset.length_words.median` | number | yes | yes | Median; `null` when there are no posts. |
| `dataset.length_words.p90` | number | yes | yes | 90th percentile; `null` when there are no posts. |
| `dataset.length_words.p99` | number | yes | yes | 99th percentile; `null` when there are no posts. |
| `dataset.length_words.min` | integer | yes | yes | Shortest post; `null` when there are no posts. |
| `dataset.length_words.max` | integer | yes | yes | Longest post; `null` when there are no posts. |
| `dataset.btc_mentions` | object | yes | no | Posts that mention Bitcoin, over every post. |
| `dataset.btc_mentions.rule` | string | yes | no | The matching rule, written out: `bitcoin` or `btc` as a whole word, ignoring case. |
| `dataset.btc_mentions.n_posts` | integer | yes | no | Posts that mention Bitcoin. |
| `dataset.btc_mentions.share` | number | yes | yes | Their share of every post; `null` when there are no posts. |
| `dataset.btc_mentions.by_class` | object | yes | no | Posts that mention Bitcoin, per label. |
| `dataset.btc_mentions.by_class.0` | integer | yes | no | Bearish posts that mention Bitcoin. |
| `dataset.btc_mentions.by_class.1` | integer | yes | no | Neutral posts that mention Bitcoin. |
| `dataset.btc_mentions.by_class.2` | integer | yes | no | Bullish posts that mention Bitcoin. |
| `dataset.btc_mentions.share_within_class` | object | yes | no | Within each label, the share of posts that mention Bitcoin. |
| `dataset.btc_mentions.share_within_class.0` | number | yes | yes | Among bearish posts; `null` when there is no bearish post. |
| `dataset.btc_mentions.share_within_class.1` | number | yes | yes | Among neutral posts; `null` when there is no neutral post. |
| `dataset.btc_mentions.share_within_class.2` | number | yes | yes | Among bullish posts; `null` when there is no bullish post. |
| `dataset.duplicates` | object | yes | no | Repeated texts (counted only; nothing is removed). |
| `dataset.duplicates.n_rows_repeating_an_earlier_text` | integer | yes | no | Posts whose exact text already appeared earlier in the table (empty texts count as repeats of each other). |
| `dataset.duplicates.n_distinct_texts_repeated` | integer | yes | no | Distinct texts that occur more than once. |
| `dataset.n_posts_with_links` | integer | yes | no | Posts whose text still contains a web link (`http://`, `https://` or `www.`). |
| `dataset.sample` | object | yes | no | The working sample: the posts that chapters 3 to 8 clean, train on and evaluate. |
| `dataset.sample.n_classes` | integer | yes | no | Classes kept (the switch `N_CLASSES`): 2 or 3. |
| `dataset.sample.btc_only` | boolean | yes | no | Whether only posts that mention Bitcoin were kept (the switch `BTC_ONLY`). |
| `dataset.sample.seed` | integer | yes | no | Random seed of the draw (`RANDOM_SEED`). |
| `dataset.sample.per_class_requested` | integer | yes | no | Most posts per class asked for (`SAMPLE_PER_CLASS`). |
| `dataset.sample.available_by_class` | object | yes | no | Posts left in each class in use after the filters, before the draw. Keyed by the classes in use. |
| `dataset.sample.available_by_class.<label>` | integer | — | no | Posts available in that class. |
| `dataset.sample.per_class` | integer | yes | no | Posts drawn from every class: the smaller of `per_class_requested` and the smallest available class. |
| `dataset.sample.class_counts` | object | yes | no | Posts per class in the working sample, keyed by the classes in use. |
| `dataset.sample.class_counts.<label>` | integer | — | no | Posts of that class; the same for every class. |
| `dataset.sample.n_posts` | integer | yes | no | Posts in the working sample. |
| `dataset.sample.n_mentions_btc` | integer | yes | no | Posts in the working sample that mention Bitcoin. |
| `dataset.cleaning` | object | yes | no | Chapter 3's cleaning of the working sample. |
| `dataset.cleaning.emoji_to_words` | boolean | yes | no | Whether emoji became English words (the switch `EMOJI_TO_WORDS`). |
| `dataset.cleaning.keep_negations` | boolean | yes | no | Whether not, no, never, nor and the "n't" forms were kept (the switch `KEEP_NEGATIONS`). |
| `dataset.cleaning.steps` | array | yes | no | The cleaning steps in the order they run (at least one). |
| `dataset.cleaning.steps[]` | string | — | no | Name of a step function, for example `"to_lowercase"`. |
| `dataset.cleaning.avg_words_before` | number | yes | yes | Mean words per post of the working sample before cleaning; `null` when the sample is empty. |
| `dataset.cleaning.avg_words_after` | number | yes | yes | Mean words per post after cleaning; `null` when the sample is empty. |
| `dataset.cleaning.n_empty_after` | integer | yes | no | Posts with no word left after cleaning. |
| `dataset.cleaning.clean_texts_matches_steps` | boolean | yes | no | Whether `clean_texts` gives exactly the step-by-step result. |
| `dataset.cleaning.examples` | array | yes | no | A few hand-written sentences and their cleaned form. |
| `dataset.cleaning.examples[]` | object | — | no | One sentence. |
| `dataset.cleaning.examples[].raw` | string | yes | no | The sentence as written. |
| `dataset.cleaning.examples[].clean` | string | yes | no | The sentence after cleaning. |

### sentiment

Chapters 4 to 8: tokenization (shown only), training, validation and test sets, two one-layer networks, the choice of one model and the evaluation on the test set.

| Field | Type | Required | Null | Meaning |
| --- | --- | --- | --- | --- |
| `sentiment` | object | yes | no | Bullish or bearish. |
| `sentiment.tokenization` | object | yes | no | Chapter 4's comparison of two tokenizers. Display only: no model uses it. |
| `sentiment.tokenization.used_by_models` | boolean | yes | no | Always `false`. |
| `sentiment.tokenization.text` | string | yes | no | Which text was tokenized: the original text of the working sample. |
| `sentiment.tokenization.n_posts` | integer | yes | no | Posts tokenized. |
| `sentiment.tokenization.nltk_word_tokenize` | object | yes | no | NLTK's word tokenizer. |
| `sentiment.tokenization.nltk_word_tokenize.vocab_size` | integer | yes | no | Distinct tokens. |
| `sentiment.tokenization.nltk_word_tokenize.avg_tokens_per_post` | number | yes | yes | Mean tokens per post; `null` when there are no posts. |
| `sentiment.tokenization.bpe` | object | yes | no | A byte-pair encoding (subword) tokenizer trained on the same text. |
| `sentiment.tokenization.bpe.vocab_size` | integer | yes | no | Pieces in its vocabulary = special tokens + single characters + merged pieces. |
| `sentiment.tokenization.bpe.vocab_size_requested` | integer | yes | no | Target size (`BPE_VOCAB_SIZE`). |
| `sentiment.tokenization.bpe.n_special_tokens` | integer | yes | no | Special tokens such as `[UNK]`. |
| `sentiment.tokenization.bpe.n_single_characters` | integer | yes | no | Single characters seen in training (always kept). |
| `sentiment.tokenization.bpe.n_merged_pieces` | integer | yes | no | Pieces learned by merging. |
| `sentiment.tokenization.bpe.trained_on` | string | yes | no | What the tokenizer was trained on. |
| `sentiment.tokenization.bpe.avg_tokens_per_post` | number | yes | yes | Mean pieces per post; `null` when there are no posts. |
| `sentiment.tokenization.examples` | array | yes | no | The posts shown cut both ways (a kind with no short post in the sample is left out). |
| `sentiment.tokenization.examples[]` | object | — | no | One post. |
| `sentiment.tokenization.examples[].kind` | string | yes | no | Why it was picked: `"emoji"`, `"negation"` or `"contraction"`. |
| `sentiment.tokenization.examples[].post_id` | string | yes | no | Its `post_id`. |
| `sentiment.classes` | array | yes | no | The classes in use, in column order: `["0", "2"]` or `["0", "1", "2"]`. |
| `sentiment.classes[]` | string | — | no | A label. |
| `sentiment.split` | object | yes | no | Chapter 5's training, validation and test sets, drawn from the working sample. |
| `sentiment.split.seed` | integer | yes | no | Random seed of both splits. |
| `sentiment.split.fractions` | object | yes | no | Intended size of each part. |
| `sentiment.split.fractions.train` | number | yes | no | 0.8. |
| `sentiment.split.fractions.validation` | number | yes | no | 0.1. |
| `sentiment.split.fractions.test` | number | yes | no | 0.1. |
| `sentiment.split.stratified_by` | string | yes | no | `"label"`: every part keeps the class shares of the working sample. |
| `sentiment.split.n_posts` | object | yes | no | Posts in each part. |
| `sentiment.split.n_posts.train` | integer | yes | no | Training posts. |
| `sentiment.split.n_posts.validation` | integer | yes | no | Validation posts. |
| `sentiment.split.n_posts.test` | integer | yes | no | Test posts. |
| `sentiment.split.class_counts` | object | yes | no | Posts per class in each part. |
| `sentiment.split.class_counts.train` | object | yes | no | In the training set, keyed by the classes in use. |
| `sentiment.split.class_counts.train.<label>` | integer | — | no | Posts of that class. |
| `sentiment.split.class_counts.validation` | object | yes | no | In the validation set, keyed by the classes in use. |
| `sentiment.split.class_counts.validation.<label>` | integer | — | no | Posts of that class. |
| `sentiment.split.class_counts.test` | object | yes | no | In the test set, keyed by the classes in use. |
| `sentiment.split.class_counts.test.<label>` | integer | — | no | Posts of that class. |
| `sentiment.split.class_shares` | object | yes | no | Class shares in each part. |
| `sentiment.split.class_shares.train` | object | yes | no | In the training set, keyed by the classes in use. |
| `sentiment.split.class_shares.train.<label>` | number | — | yes | Share of that class; `null` when the part is empty. |
| `sentiment.split.class_shares.validation` | object | yes | no | In the validation set, keyed by the classes in use. |
| `sentiment.split.class_shares.validation.<label>` | number | — | yes | Share of that class; `null` when the part is empty. |
| `sentiment.split.class_shares.test` | object | yes | no | In the test set, keyed by the classes in use. |
| `sentiment.split.class_shares.test.<label>` | number | — | yes | Share of that class; `null` when the part is empty. |
| `sentiment.split.n_mentions_btc` | object | yes | no | Posts that mention Bitcoin in each part. |
| `sentiment.split.n_mentions_btc.train` | integer | yes | no | In the training set. |
| `sentiment.split.n_mentions_btc.validation` | integer | yes | no | In the validation set. |
| `sentiment.split.n_mentions_btc.test` | integer | yes | no | In the test set. |
| `sentiment.vectorizers` | object | yes | no | Chapter 6's two vectorizers, fitted on the training set only. |
| `sentiment.vectorizers.vocab_size_requested` | integer | yes | no | Most words each keeps (`VOCAB_SIZE`). |
| `sentiment.vectorizers.fitted_on` | string | yes | no | `"train"`. |
| `sentiment.vectorizers.fitted_on_n_posts` | integer | yes | no | Training posts they learned from. |
| `sentiment.vectorizers.text` | string | yes | no | `"clean_text"`: the cleaned text. |
| `sentiment.vectorizers.storage` | string | yes | no | How the vectors are stored, `"sparse float32"`. |
| `sentiment.vectorizers.count` | object | yes | no | The word-count vectorizer. |
| `sentiment.vectorizers.count.vocab_size` | integer | yes | no | Words it kept. |
| `sentiment.vectorizers.tfidf` | object | yes | no | The TF-IDF vectorizer. |
| `sentiment.vectorizers.tfidf.vocab_size` | integer | yes | no | Words it kept. |
| `sentiment.models` | array | yes | no | Model names, `["count_network", "tfidf_network"]`; later versions of the notebook may add more. Every `<model>` key below is one of them. |
| `sentiment.models[]` | string | — | no | A model name. |
| `sentiment.training` | object | yes | no | Chapter 7's training settings and history. |
| `sentiment.training.epochs` | integer | yes | no | Passes over the training set (`EPOCHS`). |
| `sentiment.training.batch_size` | integer | yes | no | Posts per training step (`BATCH_SIZE`). |
| `sentiment.training.optimizer` | string | yes | no | `"adam"`. |
| `sentiment.training.seed` | integer | yes | no | Seed set before building each network. |
| `sentiment.training.output_layer` | string | yes | no | The output layer, for example `"1 unit, sigmoid"`. |
| `sentiment.training.loss` | string | yes | no | The loss function. |
| `sentiment.training.target_of_label` | object | yes | no | What the network learns for each label, keyed by the classes in use. |
| `sentiment.training.target_of_label.<label>` | integer | — | no | The target: the label's position among the classes in use (with two classes: `"0"` → 0, `"2"` → 1). |
| `sentiment.training.history` | object | yes | no | Training history, one entry per model. |
| `sentiment.training.history.<model>` | object | — | no | One entry per Keras metric of that model: `loss`, `accuracy`, `val_loss`, `val_accuracy`. |
| `sentiment.training.history.<model>.<metric>` | array | — | no | One value per pass over the training set. |
| `sentiment.training.history.<model>.<metric>[]` | number | — | yes | The value after that pass; `null` if it is not a finite number. |
| `sentiment.selection` | object | yes | no | How the model for later chapters was chosen. |
| `sentiment.selection.by` | string | yes | no | The rule: macro-F1 on the validation set. |
| `sentiment.selection.validation_macro_f1` | object | yes | no | Every model's macro-F1 on the validation set. |
| `sentiment.selection.validation_macro_f1.<model>` | number | — | yes | The score; `null` when it cannot be computed. |
| `sentiment.selection.selected` | string | yes | no | The chosen model (a tie goes to `count_network`). |
| `sentiment.scoring` | object | yes | no | The function later chapters use to score cleaned texts. |
| `sentiment.scoring.function` | string | yes | no | `"sentiment_proba"`. |
| `sentiment.scoring.input` | string | yes | no | What it takes. |
| `sentiment.scoring.output` | string | yes | no | What it returns. |
| `sentiment.scoring.column_labels` | array | yes | no | The label of each column it returns; equal to `sentiment.classes`. |
| `sentiment.scoring.column_labels[]` | string | — | no | A label. |
| `sentiment.scoring.bullish_column` | integer | yes | no | Index of the "probability of bullish" column: always the last one. |
| `sentiment.scoring.model` | string | yes | no | The model it uses; equal to `selection.selected`. |
| `sentiment.scoring.train_ids` | string | yes | no | Name of the set of training `post_id`s in the notebook, `"train_post_ids"`. |
| `sentiment.scoring.n_train_ids` | integer | yes | no | Training posts. |
| `sentiment.scoring.check` | object | yes | no | Self-check of the function on the test set. |
| `sentiment.scoring.check.n_texts` | integer | yes | no | Texts scored (the test posts). |
| `sentiment.scoring.check.n_columns` | integer | yes | no | Columns returned. |
| `sentiment.scoring.check.rows_sum_to_one` | boolean | yes | no | Whether every row adds up to 1. |
| `sentiment.scoring.check.matches_evaluated_predictions` | boolean | yes | no | Whether it gives exactly the labels evaluated for the chosen model. |
| `sentiment.test_texts_also_in_train` | object | yes | no | Test posts whose text also occurs in the training set (counted only). |
| `sentiment.test_texts_also_in_train.same_text` | integer | yes | no | Same original text. |
| `sentiment.test_texts_also_in_train.same_clean_text` | integer | yes | no | Same cleaned text. |
| `sentiment.evaluation` | object | yes | no | Chapter 8's evaluation on the test set. |
| `sentiment.evaluation.subsets` | object | yes | no | The subsets of the test set, keyed by the names used in `rows[].subset`. |
| `sentiment.evaluation.subsets.all_test` | string | yes | no | Description: all test posts. |
| `sentiment.evaluation.subsets.test_mentions_btc` | string | yes | no | Description: test posts that mention Bitcoin. |
| `sentiment.evaluation.baselines` | array | yes | no | Names of the baselines: `"always_majority"` (always the most frequent training label, the smallest label on a tie) and `"uniform_random"` (a class at random with equal chances, seeded). |
| `sentiment.evaluation.baselines[]` | string | — | no | A baseline name. |
| `sentiment.evaluation.majority_label` | string | yes | no | The most frequent label of the training set. |
| `sentiment.evaluation.rows` | array | yes | no | The evaluation table: one row per model and baseline and per subset. |
| `sentiment.evaluation.rows[]` | object | — | no | One row. |
| `sentiment.evaluation.rows[].model` | string | yes | no | A model name or a baseline name. |
| `sentiment.evaluation.rows[].kind` | string | yes | no | `"model"` or `"baseline"`. |
| `sentiment.evaluation.rows[].subset` | string | yes | no | `"all_test"` or `"test_mentions_btc"`. |
| `sentiment.evaluation.rows[].n` | integer | yes | no | Posts in the subset. |
| `sentiment.evaluation.rows[].accuracy` | number | yes | yes | Share of posts labelled correctly; `null` when the subset is empty. |
| `sentiment.evaluation.rows[].per_class` | object | yes | no | Scores per class, keyed by the classes in use. |
| `sentiment.evaluation.rows[].per_class.<label>` | object | — | no | The scores of that class. |
| `sentiment.evaluation.rows[].per_class.<label>.precision` | number | yes | yes | Of the posts put into this class, the share that belong to it; `null` when none was put into it. |
| `sentiment.evaluation.rows[].per_class.<label>.recall` | number | yes | yes | Of the posts of this class, the share found; `null` when the subset has none. |
| `sentiment.evaluation.rows[].per_class.<label>.f1` | number | yes | yes | Harmonic mean of precision and recall; `null` when it cannot be computed. |
| `sentiment.evaluation.rows[].per_class.<label>.support` | integer | yes | no | Posts of this class in the subset. |
| `sentiment.evaluation.rows[].macro_f1` | number | yes | yes | Mean of the per-class F1 scores that can be computed; `null` when none can. |
| `sentiment.beats_baselines` | object | yes | no | Per model, whether its macro-F1 on all test posts is higher than that of both baselines. |
| `sentiment.beats_baselines.<model>` | boolean | — | no | `false` also when a score is missing. |
| `sentiment.confusion_matrices` | object | yes | no | One confusion matrix per model, on all test posts. |
| `sentiment.confusion_matrices.<model>` | object | — | no | The matrix of that model. |
| `sentiment.confusion_matrices.<model>.labels` | array | yes | no | Labels of the rows and of the columns, in order. |
| `sentiment.confusion_matrices.<model>.labels[]` | string | — | no | A label. |
| `sentiment.confusion_matrices.<model>.rows` | string | yes | no | `"true label"`. |
| `sentiment.confusion_matrices.<model>.columns` | string | yes | no | `"predicted label"`. |
| `sentiment.confusion_matrices.<model>.counts` | array | yes | no | One list per true label. |
| `sentiment.confusion_matrices.<model>.counts[]` | array | — | no | One count per predicted label. |
| `sentiment.confusion_matrices.<model>.counts[][]` | integer | — | no | Posts with this true and this predicted label. |
| `sentiment.confusion_matrices.<model>.n` | integer | yes | no | All cells added up: the test posts. |

### topics

Chapter 9: posts sorted into topics by meaning.

| Field | Type | Required | Null | Meaning |
| --- | --- | --- | --- | --- |
| `topics` | object | yes | no | What the posts are about. |
| `topics.post_table` | string | yes | no | Name of the topic post table in the notebook, `"topic_posts"`. |
| `topics.topic_column` | string | yes | no | Its column with each post's topic, `"topic"`. |
| `topics.scope` | string | yes | no | Which posts the table takes (`TOPIC_SCOPE`): `"btc"` (every post that mentions Bitcoin) or `"all"` (posts about all coins). |
| `topics.max_posts` | integer | yes | no | Most posts in the table (`TOPIC_MAX_POSTS`). |
| `topics.seed` | integer | yes | no | Seed of the draw when more posts are in scope than `max_posts`. |
| `topics.n_available` | integer | yes | no | Posts in scope before any draw. |
| `topics.sampled` | boolean | yes | no | Whether a draw was needed. |
| `topics.n_posts` | integer | yes | no | Posts in the topic post table. |
| `topics.n_unique_post_ids` | integer | yes | no | Distinct `post_id` values; equal to `n_posts`. |
| `topics.class_counts` | object | yes | no | Posts per label in the table (all three labels, natural shares). |
| `topics.class_counts.0` | integer | yes | no | Bearish posts. |
| `topics.class_counts.1` | integer | yes | no | Neutral posts. |
| `topics.class_counts.2` | integer | yes | no | Bullish posts. |
| `topics.class_shares` | object | yes | no | Each label's share of the table. |
| `topics.class_shares.0` | number | yes | yes | Bearish share; `null` when the table is empty. |
| `topics.class_shares.1` | number | yes | yes | Neutral share; `null` when the table is empty. |
| `topics.class_shares.2` | number | yes | yes | Bullish share; `null` when the table is empty. |
| `topics.n_mentions_btc` | integer | yes | no | Posts in the table that mention Bitcoin. |
| `topics.cleaning` | object | yes | no | How the topic posts were cleaned. |
| `topics.cleaning.function` | string | yes | no | `"clean_texts"`, the same cleaning as the working sample. |
| `topics.cleaning.n_empty_after` | integer | yes | no | Posts with no word left after cleaning. |
| `topics.cleaning.n_also_in_working_sample` | integer | yes | no | Posts that are also in the working sample. |
| `topics.cleaning.same_as_working_sample` | boolean | yes | no | Whether those posts got the same cleaned text in both tables. |
| `topics.vectorizer` | object | yes | no | The TF-IDF vectorizer of the topic chapter (separate from chapter 6's). |
| `topics.vectorizer.kind` | string | yes | no | `"tfidf"`. |
| `topics.vectorizer.fitted_on` | string | yes | no | `"topic_posts"`: the whole topic post table. |
| `topics.vectorizer.fitted_on_n_posts` | integer | yes | no | Posts it learned from. |
| `topics.vectorizer.text` | string | yes | no | `"clean_text"`. |
| `topics.vectorizer.vocab_size_requested` | integer | yes | no | Most words it keeps. |
| `topics.vectorizer.vocab_size` | integer | yes | no | Words it kept. |
| `topics.vectorizer.blocked_words` | array | yes | no | Words left out because they would sit in every topic name. |
| `topics.vectorizer.blocked_words[]` | string | — | no | A word. |
| `topics.vectorizer.n_posts_without_words` | integer | yes | no | Posts with no word of the vocabulary (an all-zero vector). |
| `topics.method` | string | yes | no | `"kmeans"`. |
| `topics.method_settings` | object | yes | no | KMeans settings. |
| `topics.method_settings.n_init` | integer | yes | no | Starts; the tightest grouping is kept. |
| `topics.method_settings.random_state` | integer | yes | no | Seed. |
| `topics.method_settings.threads` | integer | yes | no | Threads used (1, so that every run gives the same topics). |
| `topics.n_topics_requested` | integer | yes | no | Topics asked for (`N_TOPICS`). |
| `topics.n_distinct_vectors` | integer | yes | no | Distinct post vectors. |
| `topics.n_topics` | integer | yes | no | Topics made: `n_topics_requested`, or fewer when there are fewer distinct vectors. |
| `topics.name_words` | integer | yes | no | Most words in a topic name. |
| `topics.quotes_per_topic` | integer | yes | no | Most quotes per topic. |
| `topics.n_posts_with_topic` | integer | yes | no | Posts that got a topic; equal to `n_posts`. |
| `topics.topics` | array | yes | no | One entry per topic, in topic-number order. |
| `topics.topics[]` | object | — | no | One topic. |
| `topics.topics[].topic` | integer | yes | no | Topic number, from 0. |
| `topics.topics[].name` | string | yes | no | The top words joined with `" / "`; empty only when none of its posts has a word of the vocabulary. Made by the program, not by a person. |
| `topics.topics[].top_words` | array | yes | no | The words of the name, strongest first. |
| `topics.topics[].top_words[]` | string | — | no | A word. |
| `topics.topics[].n_posts` | integer | yes | no | Posts in the topic (at least 1). |
| `topics.topics[].share` | number | yes | no | Its share of the topic post table. |
| `topics.topics[].n_posts_without_words` | integer | yes | no | Its posts with no word of the vocabulary. |
| `topics.topics[].quotes` | array | yes | no | Original texts of the posts closest to the topic's centre (different posts, different texts). |
| `topics.topics[].quotes[]` | object | — | no | One quote. |
| `topics.topics[].quotes[].post_id` | string | yes | no | The post's `post_id`. |
| `topics.topics[].quotes[].label` | integer | yes | no | The post's label: 0, 1 or 2. |
| `topics.topics[].quotes[].text` | string | yes | no | The post's original text, before cleaning. |

### why

Chapter 10: which topics lean bullish and which lean bearish, by the dataset's labels and by the chosen model.

| Field | Type | Required | Null | Meaning |
| --- | --- | --- | --- | --- |
| `why` | object | yes | no | The topic × bullish table. |
| `why.post_table` | string | yes | no | `"topic_posts"`: the same posts as `topics`. |
| `why.topic_column` | string | yes | no | `"topic"`. |
| `why.model` | string | yes | no | The model used; equal to `sentiment.selection.selected`. |
| `why.scoring` | object | yes | no | How the posts were scored. |
| `why.scoring.function` | string | yes | no | `"sentiment_proba"`. |
| `why.scoring.text` | string | yes | no | `"clean_text"`. |
| `why.scoring.bullish_probability` | string | yes | no | How the probability of bullish is taken from the model's output. |
| `why.scoring.n_scored` | integer | yes | no | Posts scored: every post of the topic post table. |
| `why.scoring.seconds` | number | yes | no | Seconds the scoring took (depends on the machine). |
| `why.thresholds` | object | yes | no | The model's call for a post, from its probability of bullish. |
| `why.thresholds.bullish_at_or_above` | number | yes | no | At or above this: bullish (`BULLISH_THRESHOLD`). |
| `why.thresholds.bearish_at_or_below` | number | yes | no | Otherwise, at or below this: bearish (`BEARISH_THRESHOLD`). In between: unsure. |
| `why.min_topic_posts` | integer | yes | no | Topics with fewer posts are marked `too_few_posts` (`MIN_TOPIC_POSTS`). |
| `why.bullish_share` | string | yes | no | Description of the bullish share: bullish ÷ (bullish + bearish), neutral and unsure posts left out. |
| `why.differences` | string | yes | no | Description of the two differences in percentage points. |
| `why.fair_comparison` | object | yes | no | The posts on which model and labels are compared fairly: not in the training set, labelled bearish or bullish. |
| `why.fair_comparison.posts` | string | yes | no | Description of those posts. |
| `why.fair_comparison.train_ids` | string | yes | no | `"train_post_ids"`. |
| `why.fair_comparison.n_topic_posts_in_training_set` | integer | yes | no | Topic posts the model learned from. |
| `why.fair_comparison.n_unseen_bullish_or_bearish` | integer | yes | no | Posts in the fair comparison, counted from the ids; equal to `overall.fair.n_posts`. |
| `why.fair_comparison.n_same_text_as_a_training_post` | integer | yes | no | Of those, posts whose original text also occurs in the training set (counted only). |
| `why.fair_comparison.n_same_clean_text_as_a_training_post` | integer | yes | no | Of those, posts whose cleaned text also occurs in the training set (counted only). |
| `why.order` | string | yes | no | Description of the order of `topics`. |
| `why.overall` | object | yes | no | The row "all": the whole topic post table, the baseline for every topic. |
| `why.overall.n_posts` | integer | yes | no | Posts. |
| `why.overall.label_counts` | object | yes | no | Posts per label. |
| `why.overall.label_counts.0` | integer | yes | no | Bearish. |
| `why.overall.label_counts.1` | integer | yes | no | Neutral. |
| `why.overall.label_counts.2` | integer | yes | no | Bullish. |
| `why.overall.label_bullish_share` | number | yes | yes | Bullish share by label; `null` when there is no bearish or bullish post. |
| `why.overall.model_counts` | object | yes | no | The model's calls. |
| `why.overall.model_counts.bullish` | integer | yes | no | Called bullish. |
| `why.overall.model_counts.bearish` | integer | yes | no | Called bearish. |
| `why.overall.model_counts.unsure` | integer | yes | no | Unsure (between the thresholds). |
| `why.overall.model_bullish_share` | number | yes | yes | Bullish share by model (unsure left out); `null` when nothing was called bullish or bearish. |
| `why.overall.model_unsure_share` | number | yes | yes | Unsure posts ÷ all posts; `null` when there are no posts. |
| `why.overall.fair` | object | yes | no | The fair comparison over the whole table. |
| `why.overall.fair.n_posts` | integer | yes | no | Posts in it. |
| `why.overall.fair.label_counts` | object | yes | no | Its posts per label (bearish and bullish only). |
| `why.overall.fair.label_counts.0` | integer | yes | no | Bearish. |
| `why.overall.fair.label_counts.2` | integer | yes | no | Bullish. |
| `why.overall.fair.label_bullish_share` | number | yes | yes | Bullish share by label; `null` when the comparison is empty. |
| `why.overall.fair.model_counts` | object | yes | no | The model's calls on these posts. |
| `why.overall.fair.model_counts.bullish` | integer | yes | no | Called bullish. |
| `why.overall.fair.model_counts.bearish` | integer | yes | no | Called bearish. |
| `why.overall.fair.model_counts.unsure` | integer | yes | no | Unsure. |
| `why.overall.fair.model_bullish_share` | number | yes | yes | Bullish share by model; `null` when nothing was called bullish or bearish. |
| `why.overall.fair.model_minus_label_pp` | number | yes | yes | Model minus labels, in percentage points (positive: the model is more bullish); `null` when either share is `null`. |
| `why.topics` | array | yes | no | One row per topic of `topics.topics`, sorted by `label_bullish_share`, highest first, `null` last, ties by topic number. |
| `why.topics[]` | object | — | no | One topic. |
| `why.topics[].topic` | integer | yes | no | Topic number, as in `topics.topics[].topic`. |
| `why.topics[].name` | string | yes | no | Topic name, as in `topics.topics[].name`. |
| `why.topics[].too_few_posts` | boolean | yes | no | Whether the topic has fewer posts than `min_topic_posts`. |
| `why.topics[].n_posts` | integer | yes | no | Posts. |
| `why.topics[].label_counts` | object | yes | no | Posts per label. |
| `why.topics[].label_counts.0` | integer | yes | no | Bearish. |
| `why.topics[].label_counts.1` | integer | yes | no | Neutral. |
| `why.topics[].label_counts.2` | integer | yes | no | Bullish. |
| `why.topics[].label_bullish_share` | number | yes | yes | Bullish share by label; `null` when the topic has no bearish or bullish post. |
| `why.topics[].model_counts` | object | yes | no | The model's calls. |
| `why.topics[].model_counts.bullish` | integer | yes | no | Called bullish. |
| `why.topics[].model_counts.bearish` | integer | yes | no | Called bearish. |
| `why.topics[].model_counts.unsure` | integer | yes | no | Unsure. |
| `why.topics[].model_bullish_share` | number | yes | yes | Bullish share by model; `null` when nothing was called bullish or bearish. |
| `why.topics[].model_unsure_share` | number | yes | yes | Unsure posts ÷ the topic's posts; `null` when the topic has no posts. |
| `why.topics[].fair` | object | yes | no | The fair comparison inside the topic. |
| `why.topics[].fair.n_posts` | integer | yes | no | Posts in it. |
| `why.topics[].fair.label_counts` | object | yes | no | Its posts per label (bearish and bullish only). |
| `why.topics[].fair.label_counts.0` | integer | yes | no | Bearish. |
| `why.topics[].fair.label_counts.2` | integer | yes | no | Bullish. |
| `why.topics[].fair.label_bullish_share` | number | yes | yes | Bullish share by label; `null` when the comparison is empty. |
| `why.topics[].fair.model_counts` | object | yes | no | The model's calls on these posts. |
| `why.topics[].fair.model_counts.bullish` | integer | yes | no | Called bullish. |
| `why.topics[].fair.model_counts.bearish` | integer | yes | no | Called bearish. |
| `why.topics[].fair.model_counts.unsure` | integer | yes | no | Unsure. |
| `why.topics[].fair.model_bullish_share` | number | yes | yes | Bullish share by model; `null` when nothing was called bullish or bearish. |
| `why.topics[].fair.model_minus_label_pp` | number | yes | yes | Model minus labels, in percentage points; `null` when either share is `null`. |
| `why.topics[].label_vs_overall_pp` | number | yes | yes | The topic's bullish share by label minus that of `overall`, in percentage points (positive: more bullish than all topic posts); `null` when either share is `null`. |

A hand-written example of a fair comparison with no posts, as it appears in a small topic:

```json
"fair": {
  "n_posts": 0,
  "label_counts": {"0": 0, "2": 0},
  "label_bullish_share": null,
  "model_counts": {"bullish": 0, "bearish": 0, "unsure": 0},
  "model_bullish_share": null,
  "model_minus_label_pp": null
}
```

## Daily results

Format 1.0 knows one granularity, `"batch"`: every section describes the whole data source at once. The StockTwits data has no dates, so it cannot say "how was today". Results per day need dated data and come with a later version of the format. They will be added in this way, so that no field of format 1.0 changes its name, type or meaning:

1. `scope.granularity` may then also be `"daily"`, and `scope` gets two new fields, `date_from` and `date_to` (UTC dates, `YYYY-MM-DD`), for the period the brief covers. Every existing field keeps describing the whole period.
2. A section with results per day gets one new field, `days`: a list with one entry per day of the period, sorted by date. Each entry has `date` (`YYYY-MM-DD`) and the same fields the section uses for the whole period; in `why`, for example, an entry has `overall` and `topics` with exactly the row fields listed above. A day without posts is still listed, with counts of 0 and `null` shares.
3. Results that do not change from day to day, such as the model's evaluation on the test set, stay where they are and get no `days`.
4. A brief with granularity `"batch"` never has `days`, `date_from` or `date_to`.

These are additions, so the version goes from 1.0 to 1.1 (see below), together with the new rules and new rows in the tables above.

## Versions

`format_version` is `"MAJOR.MINOR"`. It describes the format (which fields exist, their types, units and meanings), not the results: more topics, another model in the evaluation table or different numbers do not change it.

- **MINOR goes up** (1.0 → 1.1) for a change that cannot break a program written for the old version, as long as that program ignores fields it does not know: a new field or a new section; a new list such as `days`; a new allowed value of a field with a fixed list of values (such as `"daily"`); a field that may be `null` becoming one that is never `null`.
- **MAJOR goes up** (1.x → 2.0) for a change that could make such a program read something wrong: a field removed or renamed; a field's type, unit or meaning changed; a field that is never `null` becoming one that may be `null`; a change to the label coding or to the keys of an object whose keys are fixed.
- **Every change to the rules changes the version**, and in the same commit: the `const` of `format_version` in [`brief.schema.json`](brief.schema.json), the identical copy of that file in the notebook's last chapter, the value the notebook writes into `format_version`, the tables of this document and the list below. The checks fail until the two copies of the rules are identical, this document lists exactly the fields they allow, and the notebook's briefs pass them.
- **Readers** should check the MAJOR number, accept any MINOR number of the same MAJOR, and ignore fields they do not know. The rules file describes one exact version; validate a brief with it only when `format_version` matches.

| Version | Change |
| --- | --- |
| 1.0 | First fixed format. |

## Reading a brief

A minimal reader in Python:

```python
import json


def reject(constant):
    raise ValueError(f"{constant} is not allowed in a brief")


with open("data/brief.json", encoding="utf-8") as handle:
    brief = json.load(handle, parse_constant=reject)  # NaN and Infinity are not valid JSON

if brief["format_version"].split(".")[0] != "1":
    raise ValueError(f"cannot read format {brief['format_version']}")
for row in brief["why"]["topics"]:  # most bullish topic first
    share = row["label_bullish_share"]
    print(row["name"], "-" if share is None else f"{share:.1%}")
```

To check a brief against the rules (format 1.0 only), with the `jsonschema` package:

```python
from jsonschema import Draft202012Validator

with open("docs/brief.schema.json", encoding="utf-8") as handle:
    rules = json.load(handle)
for error in Draft202012Validator(rules).iter_errors(brief):
    print(list(error.absolute_path), error.message)
```
