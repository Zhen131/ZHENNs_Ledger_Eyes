# The brief: format reference

This document describes format **1.6** of the brief, the file `brief.json` that the notebook writes at the end of every run. Other programs read the brief, not the notebook, so this document is written for the authors of those programs.

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
| Field | The path from the top of the brief. `[]` stands for every item of a list; `<label>`, `<model>`, `<metric>`, `<sheet>`, `<n_drawers>`, `<emotion>`, `<prelabel>`, `<layer>` and `<drawer>` stand for every key of an object whose keys vary (described in the row of that object). |
| Type | The JSON type of the value when it is not `null`: string, integer, number, boolean, object or array. "number" also allows whole numbers. |
| Required | "yes": always present in this format. "no": may be absent. "—": an item of a list or an entry of an object with varying keys; the row above it says which ones are there. |
| Null | "yes": may be `null` (only when the number cannot be computed). "no": never `null`. |

The rules allow no other fields: a writer that adds a field without changing the rules (and this document, and the version) produces a brief that does not pass.

One rule cannot be written in JSON Schema, because it compares two fields: every day of a layer's `days` lies within that layer's `date_from` and `date_to` (in `dated_data`, `daily` and `new_words`). The notebook's format check (`brief_problems`) checks it besides the rules file, and a brief that breaks it is not saved either.

## What each section covers

The top-level `scope` describes the brief as a whole: where the posts come from, the granularity, how many classes the models use, and which coins the **working sample** holds. The sections do not all use the same posts, so **each section's coverage is given by its own fields, not by `scope`**:

| Section | Posts it uses | Fields that say so |
| --- | --- | --- |
| `dataset` | Every post of the data source; then the working sample drawn from them. | `dataset.n_posts`; `dataset.sample` |
| `sentiment` | The working sample (with the default switches: posts about all coins, bearish and bullish only, the same number in each class), split into training, validation and test sets. The evaluation table has rows for all test posts and for the test posts that mention Bitcoin. | `sentiment.classes`; `sentiment.split`; `sentiment.evaluation.subsets` |
| `topics` | The topic post table, taken from every post of the data source, not from the working sample, with all three classes in their natural shares. With the default switches: every post that mentions Bitcoin. | `topics.scope`; `topics.n_posts`; `topics.sampled`; `topics.class_counts` |
| `drawers` | The same posts as `topics`, put into drawers by a word list (a post can be in several drawers or in none); the posts in no drawer are grouped in `drawers.other`. | `drawers.post_table`; `drawers.n_posts`; `drawers.word_list_version`; `drawers.other.n_posts` |
| `why` | The same posts as `topics`, scored by the model chosen in `sentiment`. The fair comparison uses only the posts the model never saw. The calibration reads the chosen model's recognition rates on the test posts that match the topic posts from `sentiment.evaluation`. The drawer rows use the drawers of `drawers`. | `why.post_table`; `why.model`; `why.overall.n_posts`; `why.fair_comparison`; `why.calibration.rates_from`; `why.any_drawer.n_posts` |
| `emotions` | First the emotion data: the posts of a pre-label file (silver labels from a language model, read from `EMOTION_LABELS_SOURCE`) that fit the data source and are left after four steps, split into training, validation and test parts. Then the same posts as `topics`, each given an emotion by the chosen network; the drawer rows use the drawers of `drawers`. | `emotions.labels_source`; `emotions.labels_file.n_matched`; `emotions.dropped`; `emotions.n_posts`; `emotions.split`; `emotions.scoring.post_table`; `emotions.overall.n_posts` |
| `dated_data` | Not the StockTwits posts: the posts of two dated datasets (tweets as the `retail` layer, news headlines as the `media` layer) that are left after the date, language and Bitcoin steps and the daily cap, counted per layer and UTC day. It covers its own period, given in its own fields. | `dated_data.datasets.tweets.steps`; `dated_data.datasets.news.steps`; `dated_data.date_from`; `dated_data.date_to`; `dated_data.layers.<layer>.date_from`; `dated_data.layers.<layer>.date_to` |
| `daily` | The same posts as `dated_data`, every post cleaned, scored by the sentiment model of `sentiment` (split at 0.5), given an emotion by the network of `emotions` and put into the drawers of `drawers`; counted per layer and UTC day, for the whole period and for every day, with the bullish shares calibrated by recognition rates read from `sentiment.evaluation`. It covers the same period as `dated_data`, given in its own fields. | `daily.post_table`; `daily.scoring.n_scored`; `daily.calibration.rates_from`; `daily.date_from`; `daily.date_to`; `daily.layers.<layer>.date_from`; `daily.layers.<layer>.date_to` |
| `new_words` | The same posts as `dated_data`, by their cleaned texts: the words that suddenly appear in many more posts of a layer than on the days before. It covers the same period as `dated_data`, given in its own fields. | `new_words.post_table`; `new_words.date_from`; `new_words.date_to`; `new_words.layers.<layer>.date_from`; `new_words.layers.<layer>.date_to` |
| `layers` | The days of `daily` that both the media and the retail layer cover (the overlap of their periods), with their calibrated bullish shares side by side; and, over the media layer's whole period, its news headlines with a bullish or bearish reader label, to measure the sentiment model on them. | `layers.compared`; `layers.date_from`; `layers.date_to`; `layers.media_recognition.date_from`; `layers.media_recognition.date_to`; `layers.media_recognition.label_rule` |
| `validation` | The days of `daily` with a calibrated bullish share, every layer over its own period, against the BTC daily closes and the Fear and Greed Index of alternative.me (no daily price or index value is stored). | `validation.date_from`; `validation.date_to`; `validation.layers.<layer>.date_from`; `validation.layers.<layer>.date_to`; `validation.price.source`; `validation.fear_greed.source` |

For example, with the default switches `scope.coins` is `"all"`, because the models learn from posts about all coins, while `topics.scope` is `"btc"`, because the topics and the topic × bullish table use only posts that mention Bitcoin.

## Fields

### Top level and scope

| Field | Type | Required | Null | Meaning |
| --- | --- | --- | --- | --- |
| `format_version` | string | yes | no | The format version, the one named at the top of this document. See [Versions](#versions). |
| `generated_at` | string | yes | no | When the brief was saved: UTC time, ISO 8601, to the second, ending in `Z` (for example `2026-01-31T12:00:00Z`). |
| `scope` | object | yes | no | What the brief as a whole covers. |
| `scope.source` | string | yes | no | The data source as given by the switch `DATA_SOURCE`: a Hugging Face address (`hf://...`) or a path to a local spreadsheet. |
| `scope.granularity` | string | yes | no | `"batch"`: every section computed from the data source (`DATA_SOURCE`) describes it at once; results per day sit in sections of their own that carry their own period (`dated_data`, `daily`, `new_words`, `layers`, `validation`). The format allows no other value; `"daily"` is reserved (see [Daily results](#daily-results)). |
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

### drawers

Chapter 9, after the topics: the same posts put into drawers, broad subjects fixed by a person through a word list; the posts in no drawer are grouped automatically, as a hint for new drawers. With the built-in word list, two runs on the same data give the same section, so results can be compared drawer by drawer.

| Field | Type | Required | Null | Meaning |
| --- | --- | --- | --- | --- |
| `drawers` | object | yes | no | The drawers and the posts in them. |
| `drawers.post_table` | string | yes | no | `"topic_posts"`: the same posts as `topics`. |
| `drawers.text` | string | yes | no | `"text"`: words are matched in the original text, not in the cleaned text. |
| `drawers.match_rule` | string | yes | no | Description of how a post goes into a drawer. |
| `drawers.word_list_version` | string | yes | no | Version of the word list (`"v1"` for the built-in list); it goes up whenever a word changes. |
| `drawers.word_list_source` | string | yes | no | `"built-in"`, or the path given in the switch `DRAWER_WORDS_FILE`. |
| `drawers.word_list_fingerprint` | string | yes | no | First 16 hexadecimal digits of the SHA-256 of the drawers (`[[drawer, name, words, exclude], ...]` as compact ASCII JSON); the same list always gives the same fingerprint. |
| `drawers.n_posts` | integer | yes | no | Posts matched: equal to `topics.n_posts`. |
| `drawers.n_in_any_drawer` | integer | yes | no | Posts in at least one drawer. |
| `drawers.n_in_no_drawer` | integer | yes | no | Posts in no drawer; with `n_in_any_drawer` it adds up to `n_posts`. |
| `drawers.share_in_any_drawer` | number | yes | yes | `n_in_any_drawer` ÷ `n_posts`; `null` when there are no posts. |
| `drawers.share_in_no_drawer` | number | yes | yes | `n_in_no_drawer` ÷ `n_posts`; `null` when there are no posts. |
| `drawers.memberships_per_post` | object | yes | no | Posts by the number of drawers they went into, keyed `"0"`, `"1"`, `"2"`, ... up to the largest number found; the counts add up to `n_posts`. |
| `drawers.memberships_per_post.<n_drawers>` | integer | — | no | Posts in exactly that many drawers; `"0"` is always there and equals `n_in_no_drawer`. |
| `drawers.quotes_per_drawer` | integer | yes | no | Most spot-check quotes per drawer. |
| `drawers.drawers` | array | yes | no | One entry per drawer, in the word list's order (at least one). A post can be in several drawers, so the sizes add up to more than `n_posts`: the sum of `n_posts` over the drawers equals the sum of k × `memberships_per_post.k`. |
| `drawers.drawers[]` | object | — | no | One drawer. |
| `drawers.drawers[].drawer` | string | yes | no | Drawer id: lower-case letters, digits and `_`, starting with a letter (also the column name in `drawer_hits`). |
| `drawers.drawers[].name` | string | yes | no | English name of the drawer. |
| `drawers.drawers[].words` | array | yes | no | The drawer's words and phrases, in the word list's order (at least one). |
| `drawers.drawers[].words[]` | string | — | no | A word or phrase, lower case. |
| `drawers.drawers[].excluded_phrases` | array | yes | no | Phrases blanked out before the drawer's words are looked for (the word list's `exclude`); may be empty. |
| `drawers.drawers[].excluded_phrases[]` | string | — | no | A phrase, lower case. |
| `drawers.drawers[].n_posts` | integer | yes | no | Posts in the drawer. |
| `drawers.drawers[].share` | number | yes | yes | `n_posts` ÷ `drawers.n_posts`; `null` when there are no posts. |
| `drawers.drawers[].quotes` | array | yes | no | Spot checks: up to `quotes_per_drawer` posts of the drawer drawn at random with the seed, different posts with different texts; empty only when the drawer has no posts. |
| `drawers.drawers[].quotes[]` | object | — | no | One quote. |
| `drawers.drawers[].quotes[].post_id` | string | yes | no | The post's `post_id`. |
| `drawers.drawers[].quotes[].label` | integer | yes | no | The post's label: 0, 1 or 2. |
| `drawers.drawers[].quotes[].text` | string | yes | no | The post's original text. |
| `drawers.drawers[].quotes[].matched_words` | array | yes | no | The drawer's words found in the text after its excluded phrases were blanked out, in the word list's order (at least one). |
| `drawers.drawers[].quotes[].matched_words[]` | string | — | no | A word or phrase of the drawer. |
| `drawers.other` | object | yes | no | The posts in no drawer, grouped with the same method as `topics` (a hint for new drawers; never a drawer). |
| `drawers.other.n_posts` | integer | yes | no | Posts in no drawer: equal to `n_in_no_drawer`. |
| `drawers.other.method` | string | yes | no | `"kmeans"`, the same steps and settings as `topics`, learned on these posts only. |
| `drawers.other.n_groups_requested` | integer | yes | no | Groups asked for (`N_OTHER_GROUPS`). |
| `drawers.other.n_distinct_vectors` | integer | yes | no | Distinct post vectors among these posts. |
| `drawers.other.n_groups` | integer | yes | no | Groups made: `n_groups_requested`, or fewer when there are fewer distinct vectors; 0 when every post is in a drawer. |
| `drawers.other.quotes_per_group` | integer | yes | no | Most quotes per group. |
| `drawers.other.groups` | array | yes | no | One entry per group, in group-number order; their sizes add up to `n_posts`. |
| `drawers.other.groups[]` | object | — | no | One group. |
| `drawers.other.groups[].group` | integer | yes | no | Group number, from 0. |
| `drawers.other.groups[].name` | string | yes | no | The top words joined with `" / "`, made by the program like a topic name. |
| `drawers.other.groups[].top_words` | array | yes | no | The words of the name, strongest first. |
| `drawers.other.groups[].top_words[]` | string | — | no | A word. |
| `drawers.other.groups[].n_posts` | integer | yes | no | Posts in the group. |
| `drawers.other.groups[].share` | number | yes | no | Its share of the posts in no drawer. |
| `drawers.other.groups[].quotes` | array | yes | no | Original texts of the posts closest to the group's centre (different posts, different texts). |
| `drawers.other.groups[].quotes[]` | object | — | no | One quote. |
| `drawers.other.groups[].quotes[].post_id` | string | yes | no | The post's `post_id`. |
| `drawers.other.groups[].quotes[].label` | integer | yes | no | The post's label: 0, 1 or 2. |
| `drawers.other.groups[].quotes[].text` | string | yes | no | The post's original text. |

### why

Chapter 10: which topics and which drawers lean bullish and which lean bearish, by the dataset's labels and by the chosen model.

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
| `why.min_bullish_bearish_posts` | integer | yes | no | Rows (every topic and `overall`) with fewer bearish + bullish posts by label are marked `too_few_bullish_bearish` (`MIN_BULLISH_BEARISH_POSTS`). |
| `why.bullish_share` | string | yes | no | Description of the bullish share: bullish ÷ (bullish + bearish), neutral and unsure posts left out. |
| `why.differences` | string | yes | no | Description of the two differences in percentage points. |
| `why.calibration` | object | yes | no | How the model's bullish share is counted at 0.5 and calibrated with the recognition rates measured in `sentiment`. |
| `why.calibration.half_rule` | string | yes | no | Description of the call at 0.5: bullish when the probability of bullish is above 0.5, otherwise bearish (exactly 0.5 is bearish); nothing is unsure. With two classes it gives exactly the labels evaluated in `sentiment`. |
| `why.calibration.formula` | string | yes | no | Description of the calibration: (`half_bullish_share` − `false_bullish_rate`) ÷ (`true_bullish_rate` − `false_bullish_rate`), clipped to [0, 1]. |
| `why.calibration.status` | string | yes | no | `"applied"`: the calibrated shares are computed. Otherwise every calibrated share is `null`, because of `"three_classes"` (with three classes, 1 − the bearish recall also counts bearish posts called neutral, so it is not the false bullish rate), `"rates_missing"` (a rate is `null`) or `"rate_gap_too_small"` (`rate_gap` below `min_rate_gap`, or not above 0); checked in this order. |
| `why.calibration.rates_from` | object | yes | no | The row of `sentiment.evaluation.rows` the rates come from. |
| `why.calibration.rates_from.model` | string | yes | no | The chosen model, `sentiment.selection.selected`. |
| `why.calibration.rates_from.subset` | string | yes | no | The test posts that match the posts being counted: `"test_mentions_btc"` when `topics.scope` is `"btc"`, `"all_test"` when it is `"all"`. |
| `why.calibration.true_bullish_rate` | number | yes | yes | Share of the bullish test posts the model calls bullish: that row's `per_class.2.recall`. `null` with three classes or when the recall is `null`. |
| `why.calibration.false_bullish_rate` | number | yes | yes | Share of the bearish test posts the model calls bullish: 1 − that row's `per_class.0.recall`. `null` with three classes or when the recall is `null`. |
| `why.calibration.rate_gap` | number | yes | yes | `true_bullish_rate` − `false_bullish_rate`, from −1 to 1; `null` when either is `null`. Calibration divides by it. |
| `why.calibration.rate_posts` | object | yes | no | The test posts behind the rates: that row's supports (also with three classes). |
| `why.calibration.rate_posts.0` | integer | yes | no | Bearish test posts (`per_class.0.support`). |
| `why.calibration.rate_posts.2` | integer | yes | no | Bullish test posts (`per_class.2.support`). |
| `why.calibration.min_rate_gap` | number | yes | no | The smallest `rate_gap` that allows a calibration (`MIN_RATE_GAP`). |
| `why.calibration.n_rate_posts_in_fair_comparison` | integer | yes | no | Of those test posts, the ones that are also in `overall.fair` (counted only). The rates and the fair comparison share these posts, so the calibrated share there is a little optimistic. |
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
| `why.overall.too_few_bullish_bearish` | boolean | yes | no | Whether its bearish + bullish posts by label are fewer than `min_bullish_bearish_posts`. The mark changes no number. |
| `why.overall.model_counts` | object | yes | no | The model's calls. |
| `why.overall.model_counts.bullish` | integer | yes | no | Called bullish. |
| `why.overall.model_counts.bearish` | integer | yes | no | Called bearish. |
| `why.overall.model_counts.unsure` | integer | yes | no | Unsure (between the thresholds). |
| `why.overall.model_bullish_share` | number | yes | yes | Bullish share by model (unsure left out); `null` when nothing was called bullish or bearish. |
| `why.overall.model_unsure_share` | number | yes | yes | Unsure posts ÷ all posts; `null` when there are no posts. |
| `why.overall.half_counts` | object | yes | no | The model's calls at 0.5 (see `calibration.half_rule`); they add up to `n_posts`. |
| `why.overall.half_counts.bullish` | integer | yes | no | Called bullish at 0.5. |
| `why.overall.half_counts.bearish` | integer | yes | no | Called bearish at 0.5. |
| `why.overall.half_bullish_share` | number | yes | yes | Bullish share at 0.5; `null` when there are no posts. Neutral posts are split as well, so it is not comparable with `label_bullish_share`. |
| `why.overall.calibrated_bullish_share` | number | yes | yes | `half_bullish_share` calibrated (see `calibration.formula`); `null` when that is `null` or `calibration.status` is not `"applied"`. It covers neutral posts too; it is compared with the labels only in `fair`. |
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
| `why.overall.fair.half_counts` | object | yes | no | The model's calls at 0.5 on these posts. |
| `why.overall.fair.half_counts.bullish` | integer | yes | no | Called bullish at 0.5. |
| `why.overall.fair.half_counts.bearish` | integer | yes | no | Called bearish at 0.5. |
| `why.overall.fair.half_bullish_share` | number | yes | yes | Bullish share at 0.5; `null` when the comparison is empty. |
| `why.overall.fair.calibrated_bullish_share` | number | yes | yes | `half_bullish_share` calibrated; `null` when that is `null` or `calibration.status` is not `"applied"`. The model's bullish share to use; `model_bullish_share` (with the thresholds) stays as a comparison. |
| `why.overall.fair.half_minus_label_pp` | number | yes | yes | At 0.5 minus labels, in percentage points; `null` when either share is `null`. |
| `why.overall.fair.calibrated_minus_label_pp` | number | yes | yes | Calibrated minus labels, in percentage points; `null` when either share is `null`. |
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
| `why.topics[].too_few_bullish_bearish` | boolean | yes | no | Whether the topic's bearish + bullish posts by label are fewer than `min_bullish_bearish_posts`. The mark changes no number. |
| `why.topics[].model_counts` | object | yes | no | The model's calls. |
| `why.topics[].model_counts.bullish` | integer | yes | no | Called bullish. |
| `why.topics[].model_counts.bearish` | integer | yes | no | Called bearish. |
| `why.topics[].model_counts.unsure` | integer | yes | no | Unsure. |
| `why.topics[].model_bullish_share` | number | yes | yes | Bullish share by model; `null` when nothing was called bullish or bearish. |
| `why.topics[].model_unsure_share` | number | yes | yes | Unsure posts ÷ the topic's posts; `null` when the topic has no posts. |
| `why.topics[].half_counts` | object | yes | no | The model's calls at 0.5 (see `calibration.half_rule`); they add up to `n_posts`. |
| `why.topics[].half_counts.bullish` | integer | yes | no | Called bullish at 0.5. |
| `why.topics[].half_counts.bearish` | integer | yes | no | Called bearish at 0.5. |
| `why.topics[].half_bullish_share` | number | yes | yes | Bullish share at 0.5; `null` when there are no posts. Neutral posts are split as well, so it is not comparable with `label_bullish_share`. |
| `why.topics[].calibrated_bullish_share` | number | yes | yes | `half_bullish_share` calibrated (see `calibration.formula`); `null` when that is `null` or `calibration.status` is not `"applied"`. It covers neutral posts too; it is compared with the labels only in `fair`. |
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
| `why.topics[].fair.half_counts` | object | yes | no | The model's calls at 0.5 on these posts. |
| `why.topics[].fair.half_counts.bullish` | integer | yes | no | Called bullish at 0.5. |
| `why.topics[].fair.half_counts.bearish` | integer | yes | no | Called bearish at 0.5. |
| `why.topics[].fair.half_bullish_share` | number | yes | yes | Bullish share at 0.5; `null` when the comparison is empty. |
| `why.topics[].fair.calibrated_bullish_share` | number | yes | yes | `half_bullish_share` calibrated; `null` when that is `null` or `calibration.status` is not `"applied"`. The model's bullish share to use; `model_bullish_share` (with the thresholds) stays as a comparison. |
| `why.topics[].fair.half_minus_label_pp` | number | yes | yes | At 0.5 minus labels, in percentage points; `null` when either share is `null`. |
| `why.topics[].fair.calibrated_minus_label_pp` | number | yes | yes | Calibrated minus labels, in percentage points; `null` when either share is `null`. |
| `why.topics[].label_vs_overall_pp` | number | yes | yes | The topic's bullish share by label minus that of `overall`, in percentage points (positive: more bullish than all topic posts); `null` when either share is `null`. |
| `why.drawers` | array | yes | no | One row per drawer of `drawers.drawers`, in the same order (the word list's order), with the same fields as a row of `topics` and `drawer` in place of `topic`. A post can be in several drawers, so the rows overlap and do not add up to `overall`. |
| `why.drawers[]` | object | — | no | One drawer. |
| `why.drawers[].drawer` | string | yes | no | Drawer id, as in `drawers.drawers[].drawer`. |
| `why.drawers[].name` | string | yes | no | Drawer name, as in `drawers.drawers[].name`. |
| `why.drawers[].too_few_posts` | boolean | yes | no | Whether the drawer has fewer posts than `min_topic_posts`. |
| `why.drawers[].n_posts` | integer | yes | no | Posts. |
| `why.drawers[].label_counts` | object | yes | no | Posts per label. |
| `why.drawers[].label_counts.0` | integer | yes | no | Bearish. |
| `why.drawers[].label_counts.1` | integer | yes | no | Neutral. |
| `why.drawers[].label_counts.2` | integer | yes | no | Bullish. |
| `why.drawers[].label_bullish_share` | number | yes | yes | Bullish share by label; `null` when the drawer has no bearish or bullish post. |
| `why.drawers[].too_few_bullish_bearish` | boolean | yes | no | Whether the drawer's bearish + bullish posts by label are fewer than `min_bullish_bearish_posts`. The mark changes no number. |
| `why.drawers[].model_counts` | object | yes | no | The model's calls. |
| `why.drawers[].model_counts.bullish` | integer | yes | no | Called bullish. |
| `why.drawers[].model_counts.bearish` | integer | yes | no | Called bearish. |
| `why.drawers[].model_counts.unsure` | integer | yes | no | Unsure. |
| `why.drawers[].model_bullish_share` | number | yes | yes | Bullish share by model; `null` when nothing was called bullish or bearish. |
| `why.drawers[].model_unsure_share` | number | yes | yes | Unsure posts ÷ the drawer's posts; `null` when the drawer has no posts. |
| `why.drawers[].half_counts` | object | yes | no | The model's calls at 0.5 (see `calibration.half_rule`); they add up to `n_posts`. |
| `why.drawers[].half_counts.bullish` | integer | yes | no | Called bullish at 0.5. |
| `why.drawers[].half_counts.bearish` | integer | yes | no | Called bearish at 0.5. |
| `why.drawers[].half_bullish_share` | number | yes | yes | Bullish share at 0.5; `null` when there are no posts. Neutral posts are split as well, so it is not comparable with `label_bullish_share`. |
| `why.drawers[].calibrated_bullish_share` | number | yes | yes | `half_bullish_share` calibrated (see `calibration.formula`); `null` when that is `null` or `calibration.status` is not `"applied"`. It covers neutral posts too; it is compared with the labels only in `fair`. |
| `why.drawers[].fair` | object | yes | no | The fair comparison inside the drawer. |
| `why.drawers[].fair.n_posts` | integer | yes | no | Posts in it. |
| `why.drawers[].fair.label_counts` | object | yes | no | Its posts per label (bearish and bullish only). |
| `why.drawers[].fair.label_counts.0` | integer | yes | no | Bearish. |
| `why.drawers[].fair.label_counts.2` | integer | yes | no | Bullish. |
| `why.drawers[].fair.label_bullish_share` | number | yes | yes | Bullish share by label; `null` when the comparison is empty. |
| `why.drawers[].fair.model_counts` | object | yes | no | The model's calls on these posts. |
| `why.drawers[].fair.model_counts.bullish` | integer | yes | no | Called bullish. |
| `why.drawers[].fair.model_counts.bearish` | integer | yes | no | Called bearish. |
| `why.drawers[].fair.model_counts.unsure` | integer | yes | no | Unsure. |
| `why.drawers[].fair.model_bullish_share` | number | yes | yes | Bullish share by model; `null` when nothing was called bullish or bearish. |
| `why.drawers[].fair.model_minus_label_pp` | number | yes | yes | Model minus labels, in percentage points; `null` when either share is `null`. |
| `why.drawers[].fair.half_counts` | object | yes | no | The model's calls at 0.5 on these posts. |
| `why.drawers[].fair.half_counts.bullish` | integer | yes | no | Called bullish at 0.5. |
| `why.drawers[].fair.half_counts.bearish` | integer | yes | no | Called bearish at 0.5. |
| `why.drawers[].fair.half_bullish_share` | number | yes | yes | Bullish share at 0.5; `null` when the comparison is empty. |
| `why.drawers[].fair.calibrated_bullish_share` | number | yes | yes | `half_bullish_share` calibrated; `null` when that is `null` or `calibration.status` is not `"applied"`. The model's bullish share to use; `model_bullish_share` (with the thresholds) stays as a comparison. |
| `why.drawers[].fair.half_minus_label_pp` | number | yes | yes | At 0.5 minus labels, in percentage points; `null` when either share is `null`. |
| `why.drawers[].fair.calibrated_minus_label_pp` | number | yes | yes | Calibrated minus labels, in percentage points; `null` when either share is `null`. |
| `why.drawers[].label_vs_overall_pp` | number | yes | yes | The drawer's bullish share by label minus that of `overall`, in percentage points (positive: more bullish than all topic posts); `null` when either share is `null`. |
| `why.any_drawer` | object | yes | no | The posts in at least one drawer (`drawers.n_in_any_drawer` of them), with the same fields as `overall`. Together with `no_drawer` it adds up to `overall`: every count of the two rows, fair comparison included, adds up to the count of `overall`. |
| `why.any_drawer.n_posts` | integer | yes | no | Posts. |
| `why.any_drawer.label_counts` | object | yes | no | Posts per label. |
| `why.any_drawer.label_counts.0` | integer | yes | no | Bearish. |
| `why.any_drawer.label_counts.1` | integer | yes | no | Neutral. |
| `why.any_drawer.label_counts.2` | integer | yes | no | Bullish. |
| `why.any_drawer.label_bullish_share` | number | yes | yes | Bullish share by label; `null` when there is no bearish or bullish post. |
| `why.any_drawer.too_few_bullish_bearish` | boolean | yes | no | Whether its bearish + bullish posts by label are fewer than `min_bullish_bearish_posts`. The mark changes no number. |
| `why.any_drawer.model_counts` | object | yes | no | The model's calls. |
| `why.any_drawer.model_counts.bullish` | integer | yes | no | Called bullish. |
| `why.any_drawer.model_counts.bearish` | integer | yes | no | Called bearish. |
| `why.any_drawer.model_counts.unsure` | integer | yes | no | Unsure (between the thresholds). |
| `why.any_drawer.model_bullish_share` | number | yes | yes | Bullish share by model (unsure left out); `null` when nothing was called bullish or bearish. |
| `why.any_drawer.model_unsure_share` | number | yes | yes | Unsure posts ÷ these posts; `null` when there are no posts. |
| `why.any_drawer.half_counts` | object | yes | no | The model's calls at 0.5 (see `calibration.half_rule`); they add up to `n_posts`. |
| `why.any_drawer.half_counts.bullish` | integer | yes | no | Called bullish at 0.5. |
| `why.any_drawer.half_counts.bearish` | integer | yes | no | Called bearish at 0.5. |
| `why.any_drawer.half_bullish_share` | number | yes | yes | Bullish share at 0.5; `null` when there are no posts. Neutral posts are split as well, so it is not comparable with `label_bullish_share`. |
| `why.any_drawer.calibrated_bullish_share` | number | yes | yes | `half_bullish_share` calibrated (see `calibration.formula`); `null` when that is `null` or `calibration.status` is not `"applied"`. It covers neutral posts too; it is compared with the labels only in `fair`. |
| `why.any_drawer.fair` | object | yes | no | The fair comparison over these posts. |
| `why.any_drawer.fair.n_posts` | integer | yes | no | Posts in it. |
| `why.any_drawer.fair.label_counts` | object | yes | no | Its posts per label (bearish and bullish only). |
| `why.any_drawer.fair.label_counts.0` | integer | yes | no | Bearish. |
| `why.any_drawer.fair.label_counts.2` | integer | yes | no | Bullish. |
| `why.any_drawer.fair.label_bullish_share` | number | yes | yes | Bullish share by label; `null` when the comparison is empty. |
| `why.any_drawer.fair.model_counts` | object | yes | no | The model's calls on these posts. |
| `why.any_drawer.fair.model_counts.bullish` | integer | yes | no | Called bullish. |
| `why.any_drawer.fair.model_counts.bearish` | integer | yes | no | Called bearish. |
| `why.any_drawer.fair.model_counts.unsure` | integer | yes | no | Unsure. |
| `why.any_drawer.fair.model_bullish_share` | number | yes | yes | Bullish share by model; `null` when nothing was called bullish or bearish. |
| `why.any_drawer.fair.model_minus_label_pp` | number | yes | yes | Model minus labels, in percentage points (positive: the model is more bullish); `null` when either share is `null`. |
| `why.any_drawer.fair.half_counts` | object | yes | no | The model's calls at 0.5 on these posts. |
| `why.any_drawer.fair.half_counts.bullish` | integer | yes | no | Called bullish at 0.5. |
| `why.any_drawer.fair.half_counts.bearish` | integer | yes | no | Called bearish at 0.5. |
| `why.any_drawer.fair.half_bullish_share` | number | yes | yes | Bullish share at 0.5; `null` when the comparison is empty. |
| `why.any_drawer.fair.calibrated_bullish_share` | number | yes | yes | `half_bullish_share` calibrated; `null` when that is `null` or `calibration.status` is not `"applied"`. The model's bullish share to use; `model_bullish_share` (with the thresholds) stays as a comparison. |
| `why.any_drawer.fair.half_minus_label_pp` | number | yes | yes | At 0.5 minus labels, in percentage points; `null` when either share is `null`. |
| `why.any_drawer.fair.calibrated_minus_label_pp` | number | yes | yes | Calibrated minus labels, in percentage points; `null` when either share is `null`. |
| `why.no_drawer` | object | yes | no | The posts in no drawer (`drawers.n_in_no_drawer` of them), with the same fields as `overall`. |
| `why.no_drawer.n_posts` | integer | yes | no | Posts. |
| `why.no_drawer.label_counts` | object | yes | no | Posts per label. |
| `why.no_drawer.label_counts.0` | integer | yes | no | Bearish. |
| `why.no_drawer.label_counts.1` | integer | yes | no | Neutral. |
| `why.no_drawer.label_counts.2` | integer | yes | no | Bullish. |
| `why.no_drawer.label_bullish_share` | number | yes | yes | Bullish share by label; `null` when there is no bearish or bullish post. |
| `why.no_drawer.too_few_bullish_bearish` | boolean | yes | no | Whether its bearish + bullish posts by label are fewer than `min_bullish_bearish_posts`. The mark changes no number. |
| `why.no_drawer.model_counts` | object | yes | no | The model's calls. |
| `why.no_drawer.model_counts.bullish` | integer | yes | no | Called bullish. |
| `why.no_drawer.model_counts.bearish` | integer | yes | no | Called bearish. |
| `why.no_drawer.model_counts.unsure` | integer | yes | no | Unsure (between the thresholds). |
| `why.no_drawer.model_bullish_share` | number | yes | yes | Bullish share by model (unsure left out); `null` when nothing was called bullish or bearish. |
| `why.no_drawer.model_unsure_share` | number | yes | yes | Unsure posts ÷ these posts; `null` when there are no posts. |
| `why.no_drawer.half_counts` | object | yes | no | The model's calls at 0.5 (see `calibration.half_rule`); they add up to `n_posts`. |
| `why.no_drawer.half_counts.bullish` | integer | yes | no | Called bullish at 0.5. |
| `why.no_drawer.half_counts.bearish` | integer | yes | no | Called bearish at 0.5. |
| `why.no_drawer.half_bullish_share` | number | yes | yes | Bullish share at 0.5; `null` when there are no posts. Neutral posts are split as well, so it is not comparable with `label_bullish_share`. |
| `why.no_drawer.calibrated_bullish_share` | number | yes | yes | `half_bullish_share` calibrated (see `calibration.formula`); `null` when that is `null` or `calibration.status` is not `"applied"`. It covers neutral posts too; it is compared with the labels only in `fair`. |
| `why.no_drawer.fair` | object | yes | no | The fair comparison over these posts. |
| `why.no_drawer.fair.n_posts` | integer | yes | no | Posts in it. |
| `why.no_drawer.fair.label_counts` | object | yes | no | Its posts per label (bearish and bullish only). |
| `why.no_drawer.fair.label_counts.0` | integer | yes | no | Bearish. |
| `why.no_drawer.fair.label_counts.2` | integer | yes | no | Bullish. |
| `why.no_drawer.fair.label_bullish_share` | number | yes | yes | Bullish share by label; `null` when the comparison is empty. |
| `why.no_drawer.fair.model_counts` | object | yes | no | The model's calls on these posts. |
| `why.no_drawer.fair.model_counts.bullish` | integer | yes | no | Called bullish. |
| `why.no_drawer.fair.model_counts.bearish` | integer | yes | no | Called bearish. |
| `why.no_drawer.fair.model_counts.unsure` | integer | yes | no | Unsure. |
| `why.no_drawer.fair.model_bullish_share` | number | yes | yes | Bullish share by model; `null` when nothing was called bullish or bearish. |
| `why.no_drawer.fair.model_minus_label_pp` | number | yes | yes | Model minus labels, in percentage points (positive: the model is more bullish); `null` when either share is `null`. |
| `why.no_drawer.fair.half_counts` | object | yes | no | The model's calls at 0.5 on these posts. |
| `why.no_drawer.fair.half_counts.bullish` | integer | yes | no | Called bullish at 0.5. |
| `why.no_drawer.fair.half_counts.bearish` | integer | yes | no | Called bearish at 0.5. |
| `why.no_drawer.fair.half_bullish_share` | number | yes | yes | Bullish share at 0.5; `null` when the comparison is empty. |
| `why.no_drawer.fair.calibrated_bullish_share` | number | yes | yes | `half_bullish_share` calibrated; `null` when that is `null` or `calibration.status` is not `"applied"`. The model's bullish share to use; `model_bullish_share` (with the thresholds) stays as a comparison. |
| `why.no_drawer.fair.half_minus_label_pp` | number | yes | yes | At 0.5 minus labels, in percentage points; `null` when either share is `null`. |
| `why.no_drawer.fair.calibrated_minus_label_pp` | number | yes | yes | Calibrated minus labels, in percentage points; `null` when either share is `null`. |

A hand-written example of a fair comparison with no posts, as it appears in a small topic:

```json
"fair": {
  "n_posts": 0,
  "label_counts": {"0": 0, "2": 0},
  "label_bullish_share": null,
  "model_counts": {"bullish": 0, "bearish": 0, "unsure": 0},
  "model_bullish_share": null,
  "model_minus_label_pp": null,
  "half_counts": {"bullish": 0, "bearish": 0},
  "half_bullish_share": null,
  "calibrated_bullish_share": null,
  "half_minus_label_pp": null,
  "calibrated_minus_label_pp": null
}
```

### emotions

Chapter 11: eight crypto emotions and `none`, learned from **silver labels**: labels that a language model gave to StockTwits posts, not a person. The evaluation shows how well the networks learned those labels, not how well they recognise emotions. The emotion distributions describe what the chosen network says about the topic posts; they include posts the network learned from, so they are not an evaluation.

The emotions are the nine labels of the pre-label files, in this order: `panic`, `greed`, `excitement`, `anger`, `mockery`, `despair`, `wait_and_see`, `confidence`, `none`. An object keyed by `<emotion>` has one key per class in `emotions.classes` (with the default switches all nine; with `EMOTION_KEEP_NONE` off, or with emotions left out, fewer).

| Field | Type | Required | Null | Meaning |
| --- | --- | --- | --- | --- |
| `emotions` | object | yes | no | Chapter 11: eight crypto emotions (and `none`) learned from silver labels, and the emotions of the topic posts. |
| `emotions.label_kind` | string | yes | no | A sentence for people saying what the labels are: silver labels given by a language model, not by a person; the evaluation shows how well the networks learned them, not how well they recognise emotions. |
| `emotions.labels_source` | string | yes | no | The pre-label file as given by the switch `EMOTION_LABELS_SOURCE`: a web address or a local path. |
| `emotions.labels_file` | object | yes | no | The pre-label file as read, after its rows were checked against the data source. |
| `emotions.labels_file.n_rows` | integer | yes | no | Rows of the file. |
| `emotions.labels_file.n_matched` | integer | yes | no | Rows whose `post_id` is a post of the data source and whose text fingerprint is that of the post's text; equal to `n_rows` (otherwise the notebook stops). |
| `emotions.labels_file.label_counts` | object | yes | no | Rows per value of the file's `emotion` column: the nine labels and `unparsed`; they add up to `n_rows`. |
| `emotions.labels_file.label_counts.panic` | integer | yes | no | Rows labelled `panic`. |
| `emotions.labels_file.label_counts.greed` | integer | yes | no | Rows labelled `greed`. |
| `emotions.labels_file.label_counts.excitement` | integer | yes | no | Rows labelled `excitement`. |
| `emotions.labels_file.label_counts.anger` | integer | yes | no | Rows labelled `anger`. |
| `emotions.labels_file.label_counts.mockery` | integer | yes | no | Rows labelled `mockery`. |
| `emotions.labels_file.label_counts.despair` | integer | yes | no | Rows labelled `despair`. |
| `emotions.labels_file.label_counts.wait_and_see` | integer | yes | no | Rows labelled `wait_and_see`. |
| `emotions.labels_file.label_counts.confidence` | integer | yes | no | Rows labelled `confidence`. |
| `emotions.labels_file.label_counts.none` | integer | yes | no | Rows labelled `none`. |
| `emotions.labels_file.label_counts.unparsed` | integer | yes | no | Rows for which the labelling model gave no usable answer. |
| `emotions.labels_file.models` | array | yes | no | The labelling models named in the file, sorted. |
| `emotions.labels_file.models[]` | string | — | no | A model name. |
| `emotions.labels_file.prompt_versions` | array | yes | no | The prompt versions named in the file, sorted. |
| `emotions.labels_file.prompt_versions[]` | string | — | no | A prompt version. |
| `emotions.emotion_names` | array | yes | no | The switch `EMOTION_NAMES`: the emotions the networks may learn, in their order. |
| `emotions.emotion_names[]` | string | — | no | One of the eight emotions (never `none`). |
| `emotions.keep_none` | boolean | yes | no | The switch `EMOTION_KEEP_NONE`: `true` when `none` is learned as a class (nine classes with the defaults), `false` when posts labelled `none` are left out. |
| `emotions.min_class_posts` | integer | yes | no | The switch `EMOTION_MIN_CLASS_POSTS` (at least 10). |
| `emotions.dropped` | array | yes | no | The four steps that leave posts out, in this order: `unparsed`, `none`, `not_in_emotion_names`, `too_few_posts`. The posts they leave out and `n_posts` add up to `labels_file.n_matched`. |
| `emotions.dropped[]` | object | — | no | One step. |
| `emotions.dropped[].step` | string | yes | no | `"unparsed"`, `"none"`, `"not_in_emotion_names"` or `"too_few_posts"`. |
| `emotions.dropped[].rule` | string | yes | no | A sentence for people describing the step. |
| `emotions.dropped[].labels` | array | yes | no | The labels the step leaves out (empty when it leaves out nothing by its rule, for example `none` when `keep_none` is `true`). |
| `emotions.dropped[].labels[]` | string | — | no | A label (one of the nine, or `unparsed`). |
| `emotions.dropped[].n_posts` | integer | yes | no | Posts the step left out. |
| `emotions.classes` | array | yes | no | The classes the networks learned, in the order of `emotion_names` with `none` last (at least two). Every `<emotion>` key below is one of them. |
| `emotions.classes[]` | string | — | no | An emotion. |
| `emotions.n_posts` | integer | yes | no | The emotion data: the posts left after the four steps. |
| `emotions.class_counts` | object | yes | no | Posts of the emotion data per class. |
| `emotions.class_counts.<emotion>` | integer | — | no | Posts labelled with that emotion; they add up to `n_posts`. |
| `emotions.split` | object | yes | no | The emotion data split into a training part, a validation part and a test part. |
| `emotions.split.seed` | integer | yes | no | Random seed of both splits. |
| `emotions.split.fractions` | object | yes | no | Intended size of each part. |
| `emotions.split.fractions.train` | number | yes | no | 0.8. |
| `emotions.split.fractions.validation` | number | yes | no | 0.1. |
| `emotions.split.fractions.test` | number | yes | no | 0.1. |
| `emotions.split.stratified_by` | string | yes | no | `"emotion"`: every part keeps the class shares of the emotion data. |
| `emotions.split.n_posts` | object | yes | no | Posts in each part; they add up to `n_posts`. |
| `emotions.split.n_posts.train` | integer | yes | no | Posts in the emotion training part. |
| `emotions.split.n_posts.validation` | integer | yes | no | Posts in the emotion validation part. |
| `emotions.split.n_posts.test` | integer | yes | no | Posts in the emotion test part. |
| `emotions.split.class_counts` | object | yes | no | Posts per class in each part. |
| `emotions.split.class_counts.train` | object | yes | no | In the training part, keyed by the classes. |
| `emotions.split.class_counts.train.<emotion>` | integer | — | no | Posts of that class (at least one). |
| `emotions.split.class_counts.validation` | object | yes | no | In the validation part, keyed by the classes. |
| `emotions.split.class_counts.validation.<emotion>` | integer | — | no | Posts of that class (at least one). |
| `emotions.split.class_counts.test` | object | yes | no | In the test part, keyed by the classes. |
| `emotions.split.class_counts.test.<emotion>` | integer | — | no | Posts of that class (at least one). |
| `emotions.vectorizers` | object | yes | no | The chapter's own two vectorizers, fitted on the emotion training part only. |
| `emotions.vectorizers.vocab_size_requested` | integer | yes | no | Most words each keeps (`VOCAB_SIZE`). |
| `emotions.vectorizers.fitted_on` | string | yes | no | `"emotion training part"`. |
| `emotions.vectorizers.fitted_on_n_posts` | integer | yes | no | Posts they learned from: `split.n_posts.train`. |
| `emotions.vectorizers.text` | string | yes | no | `"clean_text"`: the cleaned text. |
| `emotions.vectorizers.storage` | string | yes | no | `"sparse float32"`. |
| `emotions.vectorizers.count` | object | yes | no | The word-count vectorizer. |
| `emotions.vectorizers.count.vocab_size` | integer | yes | no | Words it kept. |
| `emotions.vectorizers.tfidf` | object | yes | no | The TF-IDF vectorizer. |
| `emotions.vectorizers.tfidf.vocab_size` | integer | yes | no | Words it kept. |
| `emotions.models` | array | yes | no | Network names, `["count_network", "tfidf_network"]`. Every `<model>` key below is one of them. |
| `emotions.models[]` | string | — | no | A network name. |
| `emotions.training` | object | yes | no | Training settings and history of the networks. |
| `emotions.training.epochs` | integer | yes | no | Passes over the training part (`EPOCHS`). |
| `emotions.training.batch_size` | integer | yes | no | Posts per training step (`BATCH_SIZE`). |
| `emotions.training.optimizer` | string | yes | no | `"adam"`. |
| `emotions.training.seed` | integer | yes | no | Seed set before building each network. |
| `emotions.training.output_layer` | string | yes | no | The output layer: one softmax unit per class, for example `"9 units, softmax"`. |
| `emotions.training.loss` | string | yes | no | `"sparse_categorical_crossentropy"`. |
| `emotions.training.target_of_class` | object | yes | no | What the network learns for each class. |
| `emotions.training.target_of_class.<emotion>` | integer | — | no | The class's position in `classes`, from 0. |
| `emotions.training.history` | object | yes | no | Training history, one entry per network. |
| `emotions.training.history.<model>` | object | — | no | One entry per Keras metric: `loss`, `accuracy`, `val_loss`, `val_accuracy`. |
| `emotions.training.history.<model>.<metric>` | array | — | no | One value per pass over the training part. |
| `emotions.training.history.<model>.<metric>[]` | number | — | yes | The value after that pass; `null` if it is not a finite number. |
| `emotions.selection` | object | yes | no | How the network for the emotion calls was chosen. |
| `emotions.selection.by` | string | yes | no | The rule: macro-F1 on the emotion validation part. |
| `emotions.selection.validation_macro_f1` | object | yes | no | Every network's macro-F1 on the validation part. |
| `emotions.selection.validation_macro_f1.<model>` | number | — | yes | The score; `null` when it cannot be computed. |
| `emotions.selection.selected` | string | yes | no | The chosen network (a tie goes to `count_network`). |
| `emotions.evaluation` | object | yes | no | The evaluation on the emotion test part. |
| `emotions.evaluation.baselines` | array | yes | no | Names of the baselines: `"always_majority"` (always the most frequent class of the training part; on a tie the first in alphabetical order) and `"uniform_random"` (a class at random with equal chances, seeded). |
| `emotions.evaluation.baselines[]` | string | — | no | A baseline name. |
| `emotions.evaluation.majority_label` | string | yes | no | The class `always_majority` guesses. |
| `emotions.evaluation.rows` | array | yes | no | The evaluation table: one row per network and per baseline, all on the emotion test part. |
| `emotions.evaluation.rows[]` | object | — | no | One row. |
| `emotions.evaluation.rows[].model` | string | yes | no | A network name or a baseline name. |
| `emotions.evaluation.rows[].kind` | string | yes | no | `"model"` or `"baseline"`. |
| `emotions.evaluation.rows[].n` | integer | yes | no | Posts of the test part. |
| `emotions.evaluation.rows[].accuracy` | number | yes | yes | Share of posts given their silver label; `null` when there are no posts. |
| `emotions.evaluation.rows[].per_class` | object | yes | no | Scores per class, keyed by the classes. |
| `emotions.evaluation.rows[].per_class.<emotion>` | object | — | no | The scores of that class. |
| `emotions.evaluation.rows[].per_class.<emotion>.precision` | number | yes | yes | Of the posts put into this class, the share labelled with it; `null` when none was put into it. |
| `emotions.evaluation.rows[].per_class.<emotion>.recall` | number | yes | yes | Of the posts labelled with this class, the share found; `null` when there are none. |
| `emotions.evaluation.rows[].per_class.<emotion>.f1` | number | yes | yes | Harmonic mean of precision and recall; `null` when it cannot be computed. |
| `emotions.evaluation.rows[].per_class.<emotion>.support` | integer | yes | no | Test posts labelled with this class; they add up to `n`. |
| `emotions.evaluation.rows[].macro_f1` | number | yes | yes | Mean of the per-class F1 scores that can be computed; `null` when none can. |
| `emotions.beats_baselines` | object | yes | no | Per network, whether its macro-F1 on the test part is higher than that of both baselines. |
| `emotions.beats_baselines.<model>` | boolean | — | no | `false` also when a score is missing. |
| `emotions.confusion_matrices` | object | yes | no | One confusion matrix per network, on the emotion test part. |
| `emotions.confusion_matrices.<model>` | object | — | no | The matrix of that network. |
| `emotions.confusion_matrices.<model>.labels` | array | yes | no | Classes of the rows and of the columns, in order: equal to `classes`. |
| `emotions.confusion_matrices.<model>.labels[]` | string | — | no | An emotion. |
| `emotions.confusion_matrices.<model>.rows` | string | yes | no | `"true emotion"`: the silver label. |
| `emotions.confusion_matrices.<model>.columns` | string | yes | no | `"predicted emotion"`. |
| `emotions.confusion_matrices.<model>.counts` | array | yes | no | One list per true class. |
| `emotions.confusion_matrices.<model>.counts[]` | array | — | no | One count per predicted class. |
| `emotions.confusion_matrices.<model>.counts[][]` | integer | — | no | Posts with this true and this predicted class. |
| `emotions.confusion_matrices.<model>.n` | integer | yes | no | All cells added up: the test posts. |
| `emotions.scoring` | object | yes | no | The emotion call for every topic post. |
| `emotions.scoring.function` | string | yes | no | `"emotion_proba"`: the notebook's function that gives the chosen network's probabilities for cleaned texts, one column per class. |
| `emotions.scoring.model` | string | yes | no | The network used: equal to `selection.selected`. |
| `emotions.scoring.post_table` | string | yes | no | `"topic_posts"`: the same posts as `topics`. |
| `emotions.scoring.table` | string | yes | no | `"emotion_posts"`: the notebook's table with one row per topic post (`post_id`, `emotion`, `emotion_probability`, `in_emotion_data`). |
| `emotions.scoring.n_scored` | integer | yes | no | Topic posts given an emotion: equal to `topics.n_posts`. |
| `emotions.scoring.n_in_emotion_data` | integer | yes | no | Topic posts that are also in the emotion data (the networks learned from them, were chosen with them or were tested on them). |
| `emotions.overall` | object | yes | no | All topic posts. |
| `emotions.overall.n_posts` | integer | yes | no | Topic posts: equal to `scoring.n_scored`. |
| `emotions.overall.counts` | object | yes | no | Posts per class by the chosen network's call, keyed by the classes; they add up to `n_posts`. `any_drawer` + `no_drawer` = `overall`, class by class. |
| `emotions.overall.counts.<emotion>` | integer | — | no | Posts called that emotion. |
| `emotions.overall.shares` | object | yes | no | Share of each class, keyed by the classes. |
| `emotions.overall.shares.<emotion>` | number | — | yes | `counts.<emotion>` ÷ `n_posts`; `null` when there are no posts. |
| `emotions.overall.quotes_per_class` | integer | yes | no | Most example posts per class (3 for `overall`, 1 for every drawer, `any_drawer` and `no_drawer`). |
| `emotions.overall.quotes` | object | yes | no | Example posts, keyed by the classes (every class is there). Only posts not in the emotion data, the highest probability of the class first, each text once in the group; a class has none when no such post was called it. |
| `emotions.overall.quotes.<emotion>` | array | — | no | The examples of that class, at most `quotes_per_class`. |
| `emotions.overall.quotes.<emotion>[]` | object | — | no | One example post, called that emotion by the chosen network. |
| `emotions.overall.quotes.<emotion>[].post_id` | string | yes | no | The post's `post_id`. |
| `emotions.overall.quotes.<emotion>[].text` | string | yes | no | The post's original text. |
| `emotions.overall.quotes.<emotion>[].probability` | number | yes | no | The chosen network's probability of that emotion: the largest of `probabilities`. |
| `emotions.overall.quotes.<emotion>[].probabilities` | object | yes | no | The chosen network's probability of every class, keyed by the classes; they add up to 1. |
| `emotions.overall.quotes.<emotion>[].probabilities.<emotion>` | number | — | no | The probability of that class. |
| `emotions.drawers` | array | yes | no | One entry per drawer of `drawers.drawers`, in the same order. |
| `emotions.drawers[]` | object | — | no | One drawer. |
| `emotions.drawers[].drawer` | string | yes | no | Drawer id, as in `drawers.drawers[].drawer`. |
| `emotions.drawers[].name` | string | yes | no | English name of the drawer. |
| `emotions.drawers[].n_posts` | integer | yes | no | Posts in the drawer: equal to `drawers.drawers[].n_posts`. |
| `emotions.drawers[].counts` | object | yes | no | Posts per class by the chosen network's call, keyed by the classes; they add up to `n_posts`. |
| `emotions.drawers[].counts.<emotion>` | integer | — | no | Posts called that emotion. |
| `emotions.drawers[].shares` | object | yes | no | Share of each class, keyed by the classes. |
| `emotions.drawers[].shares.<emotion>` | number | — | yes | `counts.<emotion>` ÷ `n_posts`; `null` when there are no posts. |
| `emotions.drawers[].quotes_per_class` | integer | yes | no | Most example posts per class (3 for `overall`, 1 for every drawer, `any_drawer` and `no_drawer`). |
| `emotions.drawers[].quotes` | object | yes | no | Example posts, keyed by the classes (every class is there). Only posts not in the emotion data, the highest probability of the class first, each text once in the group; a class has none when no such post was called it. |
| `emotions.drawers[].quotes.<emotion>` | array | — | no | The examples of that class, at most `quotes_per_class`. |
| `emotions.drawers[].quotes.<emotion>[]` | object | — | no | One example post, called that emotion by the chosen network. |
| `emotions.drawers[].quotes.<emotion>[].post_id` | string | yes | no | The post's `post_id`. |
| `emotions.drawers[].quotes.<emotion>[].text` | string | yes | no | The post's original text. |
| `emotions.drawers[].quotes.<emotion>[].probability` | number | yes | no | The chosen network's probability of that emotion: the largest of `probabilities`. |
| `emotions.drawers[].quotes.<emotion>[].probabilities` | object | yes | no | The chosen network's probability of every class, keyed by the classes; they add up to 1. |
| `emotions.drawers[].quotes.<emotion>[].probabilities.<emotion>` | number | — | no | The probability of that class. |
| `emotions.any_drawer` | object | yes | no | The topic posts in at least one drawer. |
| `emotions.any_drawer.n_posts` | integer | yes | no | Equal to `drawers.n_in_any_drawer`. |
| `emotions.any_drawer.counts` | object | yes | no | Posts per class by the chosen network's call, keyed by the classes; they add up to `n_posts`. |
| `emotions.any_drawer.counts.<emotion>` | integer | — | no | Posts called that emotion. |
| `emotions.any_drawer.shares` | object | yes | no | Share of each class, keyed by the classes. |
| `emotions.any_drawer.shares.<emotion>` | number | — | yes | `counts.<emotion>` ÷ `n_posts`; `null` when there are no posts. |
| `emotions.any_drawer.quotes_per_class` | integer | yes | no | Most example posts per class (3 for `overall`, 1 for every drawer, `any_drawer` and `no_drawer`). |
| `emotions.any_drawer.quotes` | object | yes | no | Example posts, keyed by the classes (every class is there). Only posts not in the emotion data, the highest probability of the class first, each text once in the group; a class has none when no such post was called it. |
| `emotions.any_drawer.quotes.<emotion>` | array | — | no | The examples of that class, at most `quotes_per_class`. |
| `emotions.any_drawer.quotes.<emotion>[]` | object | — | no | One example post, called that emotion by the chosen network. |
| `emotions.any_drawer.quotes.<emotion>[].post_id` | string | yes | no | The post's `post_id`. |
| `emotions.any_drawer.quotes.<emotion>[].text` | string | yes | no | The post's original text. |
| `emotions.any_drawer.quotes.<emotion>[].probability` | number | yes | no | The chosen network's probability of that emotion: the largest of `probabilities`. |
| `emotions.any_drawer.quotes.<emotion>[].probabilities` | object | yes | no | The chosen network's probability of every class, keyed by the classes; they add up to 1. |
| `emotions.any_drawer.quotes.<emotion>[].probabilities.<emotion>` | number | — | no | The probability of that class. |
| `emotions.no_drawer` | object | yes | no | The topic posts in no drawer. |
| `emotions.no_drawer.n_posts` | integer | yes | no | Equal to `drawers.n_in_no_drawer`. |
| `emotions.no_drawer.counts` | object | yes | no | Posts per class by the chosen network's call, keyed by the classes; they add up to `n_posts`. |
| `emotions.no_drawer.counts.<emotion>` | integer | — | no | Posts called that emotion. |
| `emotions.no_drawer.shares` | object | yes | no | Share of each class, keyed by the classes. |
| `emotions.no_drawer.shares.<emotion>` | number | — | yes | `counts.<emotion>` ÷ `n_posts`; `null` when there are no posts. |
| `emotions.no_drawer.quotes_per_class` | integer | yes | no | Most example posts per class (3 for `overall`, 1 for every drawer, `any_drawer` and `no_drawer`). |
| `emotions.no_drawer.quotes` | object | yes | no | Example posts, keyed by the classes (every class is there). Only posts not in the emotion data, the highest probability of the class first, each text once in the group; a class has none when no such post was called it. |
| `emotions.no_drawer.quotes.<emotion>` | array | — | no | The examples of that class, at most `quotes_per_class`. |
| `emotions.no_drawer.quotes.<emotion>[]` | object | — | no | One example post, called that emotion by the chosen network. |
| `emotions.no_drawer.quotes.<emotion>[].post_id` | string | yes | no | The post's `post_id`. |
| `emotions.no_drawer.quotes.<emotion>[].text` | string | yes | no | The post's original text. |
| `emotions.no_drawer.quotes.<emotion>[].probability` | number | yes | no | The chosen network's probability of that emotion: the largest of `probabilities`. |
| `emotions.no_drawer.quotes.<emotion>[].probabilities` | object | yes | no | The chosen network's probability of every class, keyed by the classes; they add up to 1. |
| `emotions.no_drawer.quotes.<emotion>[].probabilities.<emotion>` | number | — | no | The probability of that class. |
| `emotions.human_review` | object | yes | no | Agreement of a person with the pre-labels, from a filled-in review table (`labels/emotions-review-template.csv` in the repository). Not weighted: the review table holds about the same number of rows per label, so an overall agreement must weight each label's agreement by the label's share of the labelled rows of the pre-label file. |
| `emotions.human_review.source` | string | yes | no | The switch `EMOTION_REVIEW_SOURCE`; `""` when no table was read. |
| `emotions.human_review.n_reviewed` | integer | yes | no | Rows whose `human_label` is filled in; 0 when no table was read. |
| `emotions.human_review.agreement` | number | yes | yes | Share of those rows whose `human_label` equals `prelabel` (spaces around it and case ignored); `null` when no row is filled in. |
| `emotions.human_review.by_prelabel` | object | yes | no | The same per pre-label, keyed by the nine labels (always all nine). |
| `emotions.human_review.by_prelabel.<prelabel>` | object | — | no | One of the nine labels. |
| `emotions.human_review.by_prelabel.<prelabel>.n_reviewed` | integer | yes | no | Filled-in rows with this pre-label. |
| `emotions.human_review.by_prelabel.<prelabel>.agreement` | number | yes | yes | Share of them on which the person agrees; `null` when there are none. |

### dated_data

Chapter 12: two public datasets **with dates**, read into one dated post table (its shape is in [`dated-posts.md`](dated-posts.md)) and counted per layer and UTC day. Tweets are the `retail` layer, news headlines the `media` layer. This section only counts posts; it holds no score. Unlike the sections above, it describes a period: it carries its own `date_from` and `date_to`, and every layer has its own (see [Daily results](#daily-results)). Both datasets are published under CC BY-NC 4.0.

An object keyed by `<layer>` has one key per layer that has posts in the table: `media`, `kol` or `retail` (with the default switches `media` and `retail`; `kol` has no data yet).

| Field | Type | Required | Null | Meaning |
| --- | --- | --- | --- | --- |
| `dated_data` | object | yes | no | Chapter 12: the dated post table, its posts per layer and UTC day, and where they come from. |
| `dated_data.table_shape` | string | yes | no | The version of the dated post table's shape (`dated-posts-v1`, see `dated-posts.md`). |
| `dated_data.day` | string | yes | no | The unit of every day in this section: `"UTC"`, the date of the post's time in Coordinated Universal Time. |
| `dated_data.switches` | object | yes | no | The switches of chapter 12 that decide which posts are kept. |
| `dated_data.switches.date_start` | string | yes | no | `DATED_START`: the first UTC day kept, `YYYY-MM-DD`. |
| `dated_data.switches.date_end` | string | yes | yes | `DATED_END`: the last UTC day kept, `YYYY-MM-DD`; `null` when there is no last day (to the end of the data). |
| `dated_data.switches.language` | string | yes | yes | `DATED_LANGUAGE`: the language a tweet must be marked with (for example `en`); `null` when the language is not filtered. |
| `dated_data.switches.btc_only` | boolean | yes | no | `DATED_BTC_ONLY`: only posts whose text mentions `bitcoin` or `btc` were kept. |
| `dated_data.switches.daily_cap` | integer | yes | no | `DATED_DAILY_CAP`: the most posts kept per layer and UTC day. |
| `dated_data.switches.seed` | integer | yes | no | `RANDOM_SEED`, which fixes the draw when a day has more posts than the cap. |
| `dated_data.date_from` | string | yes | no | The first UTC day of the whole dated post table (the earliest of the layers' `date_from`), `YYYY-MM-DD`. |
| `dated_data.date_to` | string | yes | no | The last UTC day of the whole dated post table (the latest of the layers' `date_to`), `YYYY-MM-DD`. |
| `dated_data.n_posts` | integer | yes | no | Posts in the dated post table; equal to the sum of the layers' `n_posts`. |
| `dated_data.datasets` | object | yes | no | The two datasets and what every step left of them. |
| `dated_data.datasets.tweets` | object | yes | no | The tweet dataset (`ExponentialScience/DLT-Tweets`), the `retail` layer. |
| `dated_data.datasets.tweets.name` | string | yes | no | The dataset's id on Hugging Face. |
| `dated_data.datasets.tweets.revision` | string | yes | no | The dataset revision that the default address and the download script use. |
| `dated_data.datasets.tweets.license` | string | yes | no | The dataset's license, `CC BY-NC 4.0`. |
| `dated_data.datasets.tweets.citation` | string | yes | no | The paper the dataset's authors ask to cite. |
| `dated_data.datasets.tweets.layer` | string | yes | no | The layer the dataset fills: `retail`. |
| `dated_data.datasets.tweets.source` | string | yes | no | Where the tweets were read, as given by `DATED_TWEETS_SOURCE`: a Hugging Face address or a local `.parquet` file or folder. |
| `dated_data.datasets.tweets.text_column` | string | yes | no | The dataset column used as the text: `tweet`. |
| `dated_data.datasets.tweets.steps` | array | yes | no | The posts left after every step, in this order: `read`, `time_parsed`, `in_dates`, `language`, `mentions_btc`, `unique`, `daily_cap`. Each count is at most the one before it; the last equals the layer's posts in the table. |
| `dated_data.datasets.tweets.steps[]` | object | — | no | One step. |
| `dated_data.datasets.tweets.steps[].step` | string | yes | no | The step: `read` (every row of the files), `time_parsed` (the time could be read), `in_dates` (UTC day within the switches' period), `language` (marked with `DATED_LANGUAGE`, or every row when it is `null`), `mentions_btc` (the text mentions Bitcoin, or every row when `btc_only` is false), `unique` (one row per `post_id`), `daily_cap` (at most `daily_cap` posts per UTC day). |
| `dated_data.datasets.tweets.steps[].n_posts` | integer | yes | no | Posts left after the step. |
| `dated_data.datasets.tweets.time_kinds` | object | yes | no | How the time of every row read is written; the four counts add up to the rows read. |
| `dated_data.datasets.tweets.time_kinds.with_zone` | integer | yes | no | Rows with a time and a time zone (converted to UTC). |
| `dated_data.datasets.tweets.time_kinds.without_zone` | integer | yes | no | Rows with a time but no time zone (taken as UTC). |
| `dated_data.datasets.tweets.time_kinds.date_only` | integer | yes | no | Rows with a date only (`time` stays empty in the table). |
| `dated_data.datasets.tweets.time_kinds.unparseable` | integer | yes | no | Rows whose time cannot be read; they are left out at the step `time_parsed`. |
| `dated_data.datasets.tweets.n_duplicates` | integer | yes | no | Rows dropped at the step `unique` because another row has the same `post_id` (the same original time and the same text after the replacements). |
| `dated_data.datasets.news` | object | yes | no | The news dataset (`ExponentialScience/DLT-Sentiment-News`), the `media` layer. |
| `dated_data.datasets.news.name` | string | yes | no | The dataset's id on Hugging Face. |
| `dated_data.datasets.news.revision` | string | yes | no | The dataset revision that the default address and the download script use. |
| `dated_data.datasets.news.license` | string | yes | no | The dataset's license, `CC BY-NC 4.0`. |
| `dated_data.datasets.news.citation` | string | yes | no | The paper the dataset's authors ask to cite. |
| `dated_data.datasets.news.layer` | string | yes | no | The layer the dataset fills: `media`. |
| `dated_data.datasets.news.source` | string | yes | no | Where the news were read, as given by `DATED_NEWS_SOURCE`. |
| `dated_data.datasets.news.text_column` | string | yes | no | The dataset column used as the text, as given by `NEWS_TEXT`: `title` (the headline) or `text` (headline and summary). |
| `dated_data.datasets.news.steps` | array | yes | no | The posts left after every step, as for the tweets. The dataset has no language column, so the step `language` keeps every row. |
| `dated_data.datasets.news.steps[]` | object | — | no | One step. |
| `dated_data.datasets.news.steps[].step` | string | yes | no | The step, as for the tweets. |
| `dated_data.datasets.news.steps[].n_posts` | integer | yes | no | Posts left after the step. |
| `dated_data.datasets.news.time_kinds` | object | yes | no | How the time of every row read is written, as for the tweets. |
| `dated_data.datasets.news.time_kinds.with_zone` | integer | yes | no | Rows with a time and a time zone. |
| `dated_data.datasets.news.time_kinds.without_zone` | integer | yes | no | Rows with a time but no time zone. |
| `dated_data.datasets.news.time_kinds.date_only` | integer | yes | no | Rows with a date only. |
| `dated_data.datasets.news.time_kinds.unparseable` | integer | yes | no | Rows whose time cannot be read. |
| `dated_data.datasets.news.n_duplicates` | integer | yes | no | Rows dropped at the step `unique` because another row has the same `post_id` (the same original time and the same CryptoPanic address). |
| `dated_data.datasets.news.label_rule` | string | yes | no | How the `label` of a news item was made, as given by `NEWS_LABEL`: `market_direction` (the dataset's class, converted to this project's coding) or `votes` (the side with more reader votes; as many is neutral). |
| `dated_data.datasets.news.class_names_found` | boolean | yes | no | The data file names the classes of `market_direction` (they were checked to be neutral, bearish, bullish in this order); false when it names none and the fixed conversion was used without that check. |
| `dated_data.datasets.news.market_direction_counts` | object | yes | no | The news in the table per value of the dataset's own `market_direction`, **in the dataset's coding**: `"0"` neutral, `"1"` bearish, `"2"` bullish. |
| `dated_data.datasets.news.market_direction_counts.0` | integer | yes | no | News with `market_direction` 0 (neutral). |
| `dated_data.datasets.news.market_direction_counts.1` | integer | yes | no | News with `market_direction` 1 (bearish). |
| `dated_data.datasets.news.market_direction_counts.2` | integer | yes | no | News with `market_direction` 2 (bullish). |
| `dated_data.datasets.news.label_counts` | object | yes | no | The news in the table per `label`, **in this project's coding**: `"0"` bearish, `"1"` neutral, `"2"` bullish. With `label_rule` `market_direction`, `"1"` equals `market_direction_counts."0"`, `"0"` equals `market_direction_counts."1"` and `"2"` equals `market_direction_counts."2"`. |
| `dated_data.datasets.news.label_counts.0` | integer | yes | no | News labelled bearish. |
| `dated_data.datasets.news.label_counts.1` | integer | yes | no | News labelled neutral. |
| `dated_data.datasets.news.label_counts.2` | integer | yes | no | News labelled bullish. |
| `dated_data.layers` | object | yes | no | Every layer that has posts in the table, keyed by `<layer>`. |
| `dated_data.layers.<layer>` | object | — | no | One layer. |
| `dated_data.layers.<layer>.date_from` | string | yes | no | The layer's first UTC day with posts, `YYYY-MM-DD`. |
| `dated_data.layers.<layer>.date_to` | string | yes | no | The layer's last UTC day with posts, `YYYY-MM-DD`. The layers end on different days: the tweets in mid-2023, the news in 2025. |
| `dated_data.layers.<layer>.n_posts` | integer | yes | no | The layer's posts in the table; equal to the sum of its `days`. |
| `dated_data.layers.<layer>.n_mentions_btc` | integer | yes | no | Of these, posts whose text mentions `bitcoin` or `btc` (all of them when `btc_only` is true). |
| `dated_data.layers.<layer>.n_days_with_posts` | integer | yes | no | Days of `days` with at least one post. |
| `dated_data.layers.<layer>.n_days_capped` | integer | yes | no | Days on which the layer had more than `daily_cap` posts after the step `unique`, so that the cap left posts out. |
| `dated_data.layers.<layer>.max_posts_per_day` | integer | yes | no | The most posts of the layer on one day (at most `daily_cap`). |
| `dated_data.layers.<layer>.n_sources` | integer | yes | no | Different values of `source` in the layer (news: domains, `unknown` included; tweets: only `twitter`). |
| `dated_data.layers.<layer>.n_unknown_source` | integer | yes | no | Posts of the layer whose `source` is `unknown` (a news item without a readable address of its original article). |
| `dated_data.layers.<layer>.top_sources` | array | yes | no | The layer's most frequent values of `source`, at most ten, most posts first (ties by name). |
| `dated_data.layers.<layer>.top_sources[]` | object | — | no | One source. |
| `dated_data.layers.<layer>.top_sources[].source` | string | yes | no | The source name (a news site's domain, `unknown` or `twitter`). |
| `dated_data.layers.<layer>.top_sources[].n_posts` | integer | yes | no | The layer's posts from this source. |
| `dated_data.layers.<layer>.days` | array | yes | no | One entry per calendar day from the layer's `date_from` to its `date_to`, sorted by date; a day without posts has 0. |
| `dated_data.layers.<layer>.days[]` | object | — | no | One day. |
| `dated_data.layers.<layer>.days[].date` | string | yes | no | The UTC day, `YYYY-MM-DD`. |
| `dated_data.layers.<layer>.days[].n_posts` | integer | yes | no | The layer's posts on that day (at most `daily_cap`). |

### daily

Chapter 13: every post of the dated post table (see `dated_data`) cleaned with `clean_texts`, scored by the sentiment model chosen in `sentiment` (probability of bullish, split at 0.5), given an emotion by the network chosen in `emotions` and put into the drawers of the word list of `drawers`; then counted per layer and UTC day. Like `dated_data`, it describes a period and carries its own `date_from` and `date_to`, for the section and for every layer; every layer's `days` are exactly the days of `dated_data.layers.<layer>.days` (see [Daily results](#daily-results)). The models learned from StockTwits posts, so the shares and emotions describe what these models make of tweets and news headlines; they are a reference, not a measurement.

A **pile** is either all posts of one layer on one day (or over the layer's whole period), or those of them that sit in one drawer. Every post of a pile is split at 0.5 into bullish or bearish, so `half_counts.bullish` + `half_counts.bearish` = `n_posts`; a drawer's pile records only its bullish posts, `n_half_bullish`. A pile with fewer posts than `daily.min_posts` is marked `too_few_posts`, and both its shares are `null`; a day without posts is a pile of 0 posts and is marked too. An object keyed by `<drawer>` has one key per drawer of `daily.drawer_ids`, in that order; an object keyed by `<emotion>` has one key per class of `daily.emotion_classes`, in that order.

| Field | Type | Required | Null | Meaning |
| --- | --- | --- | --- | --- |
| `daily` | object | yes | no | Chapter 13: the dated posts scored, counted per layer and UTC day, and example posts of chosen days. |
| `daily.post_table` | string | yes | no | The table the posts come from, `dated_posts` (chapter 12). |
| `daily.score_table` | string | yes | no | The notebook's table with every dated post's scores, `dated_scores`. |
| `daily.day` | string | yes | no | The unit of every day in this section: `"UTC"`. |
| `daily.date_from` | string | yes | no | The first UTC day of the section, `YYYY-MM-DD`: the same as `dated_data.date_from`. |
| `daily.date_to` | string | yes | no | The last UTC day of the section, `YYYY-MM-DD`: the same as `dated_data.date_to`. |
| `daily.min_posts` | integer | yes | no | `DAILY_MIN_POSTS`: a pile with fewer posts is marked `too_few_posts` and has no shares. |
| `daily.half_rule` | string | yes | no | How a post is split at 0.5 (bullish above 0.5, otherwise bearish); the same rule as chapter 10's. |
| `daily.scoring` | object | yes | no | How the dated posts were scored. |
| `daily.scoring.text` | string | yes | no | The text the models read: `clean_text`, the post's text after `clean_texts`. |
| `daily.scoring.sentiment_model` | string | yes | no | The sentiment model, the one chosen in `sentiment.selection.selected`. |
| `daily.scoring.emotion_model` | string | yes | no | The emotion network, the one chosen in `emotions.selection.selected`. |
| `daily.scoring.n_scored` | integer | yes | no | Dated posts scored: every post of the dated post table, `dated_data.n_posts`. |
| `daily.scoring.n_distinct_clean_texts` | integer | yes | no | Different cleaned texts among them (each was scored once). |
| `daily.calibration` | object | yes | no | The recognition rates that calibrate every bullish share of this section, and where they come from. |
| `daily.calibration.status` | string | yes | no | As in `why.calibration.status`: `applied`, `rates_missing`, `rate_gap_too_small` or `three_classes`. Unless `applied`, every `calibrated_bullish_share` of this section is `null`. |
| `daily.calibration.rates_from` | object | yes | no | The row of `sentiment.evaluation.rows` the rates come from. |
| `daily.calibration.rates_from.model` | string | yes | no | The chosen sentiment model. |
| `daily.calibration.rates_from.subset` | string | yes | no | `test_mentions_btc` when `dated_data.switches.btc_only` is true, otherwise `all_test` (not `TOPIC_SCOPE`, which decides `why.calibration`). |
| `daily.calibration.subset_rule` | string | yes | no | That rule, as an English sentence. |
| `daily.calibration.true_bullish_rate` | number | yes | yes | That row's bullish recall; `null` with three classes or when it cannot be computed. |
| `daily.calibration.false_bullish_rate` | number | yes | yes | 1 − that row's bearish recall; `null` as above. |
| `daily.calibration.rate_gap` | number | yes | yes | True bullish rate − false bullish rate; `null` when either is `null`. |
| `daily.calibration.rate_posts` | object | yes | no | That row's test posts per class. |
| `daily.calibration.rate_posts.0` | integer | yes | no | Bearish test posts of the row. |
| `daily.calibration.rate_posts.2` | integer | yes | no | Bullish test posts of the row. |
| `daily.calibration.min_rate_gap` | number | yes | no | `MIN_RATE_GAP`, as in `why.calibration.min_rate_gap`. |
| `daily.calibration.formula` | string | yes | no | The calibration, as an English sentence: (counted share − false bullish rate) ÷ (true bullish rate − false bullish rate), clipped to [0, 1]. |
| `daily.calibration.assumption` | string | yes | no | The assumption behind using rates measured on StockTwits test posts for tweets and news, as an English sentence. |
| `daily.emotion_classes` | array | yes | no | The emotions every `emotion_counts` lists, in order: the classes the emotion network actually learned, the same as `emotions.classes`. |
| `daily.emotion_classes[]` | string | — | no | One emotion. |
| `daily.drawer_ids` | array | yes | no | The drawers every `drawers` object lists, in word-list order: the same as the ids of `drawers.drawers`. |
| `daily.drawer_ids[]` | string | — | no | One drawer id. |
| `daily.word_list_version` | string | yes | no | The version of the drawer word list, the same as `drawers.word_list_version`. |
| `daily.layers` | object | yes | no | Every layer of `dated_data.layers`, keyed by `<layer>`. |
| `daily.layers.<layer>` | object | — | no | One layer. |
| `daily.layers.<layer>.date_from` | string | yes | no | The layer's first day, the same as `dated_data.layers.<layer>.date_from`. |
| `daily.layers.<layer>.date_to` | string | yes | no | The layer's last day, the same as `dated_data.layers.<layer>.date_to`. |
| `daily.layers.<layer>.overall` | object | yes | no | The layer's whole period: every count is the sum of the same count over `days`. |
| `daily.layers.<layer>.overall.n_posts` | integer | yes | no | The layer's posts, the same as `dated_data.layers.<layer>.n_posts`. |
| `daily.layers.<layer>.overall.half_counts` | object | yes | no | The posts split at 0.5. |
| `daily.layers.<layer>.overall.half_counts.bullish` | integer | yes | no | Posts with a probability of bullish above 0.5. |
| `daily.layers.<layer>.overall.half_counts.bearish` | integer | yes | no | The other posts; bullish + bearish = `n_posts`. |
| `daily.layers.<layer>.overall.half_bullish_share` | number | yes | yes | The counted bullish share, `half_counts.bullish` ÷ `n_posts`; `null` when `too_few_posts`. |
| `daily.layers.<layer>.overall.calibrated_bullish_share` | number | yes | yes | The counted share calibrated with `daily.calibration`, clipped to [0, 1]; `null` when `too_few_posts` or the status is not `applied`. |
| `daily.layers.<layer>.overall.too_few_posts` | boolean | yes | no | True when `n_posts` is below `daily.min_posts`. |
| `daily.layers.<layer>.overall.emotion_counts` | object | yes | no | Posts per emotion of the network, keyed by `<emotion>`; they add up to `n_posts`. |
| `daily.layers.<layer>.overall.emotion_counts.<emotion>` | integer | — | no | Posts given this emotion. |
| `daily.layers.<layer>.overall.n_in_any_drawer` | integer | yes | no | Posts in at least one drawer. |
| `daily.layers.<layer>.overall.n_in_no_drawer` | integer | yes | no | Posts in no drawer; with `n_in_any_drawer` it adds up to `n_posts`. |
| `daily.layers.<layer>.overall.drawers` | object | yes | no | One pile per drawer, keyed by `<drawer>`. The piles overlap: a post can be in several drawers. |
| `daily.layers.<layer>.overall.drawers.<drawer>` | object | — | no | The posts in this drawer. |
| `daily.layers.<layer>.overall.drawers.<drawer>.n_posts` | integer | yes | no | Posts in the drawer (at most the layer's `n_posts`). |
| `daily.layers.<layer>.overall.drawers.<drawer>.n_half_bullish` | integer | yes | no | Of these, posts with a probability of bullish above 0.5 (bullish at 0.5); the others are bearish at 0.5. |
| `daily.layers.<layer>.overall.drawers.<drawer>.half_bullish_share` | number | yes | yes | The drawer's counted bullish share; `null` when its `too_few_posts`. |
| `daily.layers.<layer>.overall.drawers.<drawer>.calibrated_bullish_share` | number | yes | yes | The drawer's calibrated bullish share; `null` when its `too_few_posts` or the status is not `applied`. |
| `daily.layers.<layer>.overall.drawers.<drawer>.too_few_posts` | boolean | yes | no | True when the drawer has fewer posts than `daily.min_posts`. |
| `daily.layers.<layer>.days` | array | yes | no | One entry per calendar day from the layer's `date_from` to its `date_to`, sorted by date: the same days as `dated_data.layers.<layer>.days`. |
| `daily.layers.<layer>.days[]` | object | — | no | One day: the same fields as `overall`, for the posts of that day, and its date. |
| `daily.layers.<layer>.days[].date` | string | yes | no | The UTC day, `YYYY-MM-DD`. |
| `daily.layers.<layer>.days[].n_posts` | integer | yes | no | The layer's posts on that day, the same as in `dated_data`. |
| `daily.layers.<layer>.days[].half_counts` | object | yes | no | The day's posts split at 0.5. |
| `daily.layers.<layer>.days[].half_counts.bullish` | integer | yes | no | Bullish at 0.5. |
| `daily.layers.<layer>.days[].half_counts.bearish` | integer | yes | no | Bearish at 0.5. |
| `daily.layers.<layer>.days[].half_bullish_share` | number | yes | yes | The day's counted bullish share; `null` when `too_few_posts`. |
| `daily.layers.<layer>.days[].calibrated_bullish_share` | number | yes | yes | The day's calibrated bullish share; `null` when `too_few_posts` or the status is not `applied`. |
| `daily.layers.<layer>.days[].too_few_posts` | boolean | yes | no | True when the day has fewer posts than `daily.min_posts` (also on a day without posts). |
| `daily.layers.<layer>.days[].emotion_counts` | object | yes | no | The day's posts per emotion, keyed by `<emotion>`. |
| `daily.layers.<layer>.days[].emotion_counts.<emotion>` | integer | — | no | Posts given this emotion on that day. |
| `daily.layers.<layer>.days[].n_in_any_drawer` | integer | yes | no | The day's posts in at least one drawer. |
| `daily.layers.<layer>.days[].n_in_no_drawer` | integer | yes | no | The day's posts in no drawer. |
| `daily.layers.<layer>.days[].drawers` | object | yes | no | One pile per drawer for that day, keyed by `<drawer>`. |
| `daily.layers.<layer>.days[].drawers.<drawer>` | object | — | no | The day's posts in this drawer. |
| `daily.layers.<layer>.days[].drawers.<drawer>.n_posts` | integer | yes | no | Posts in the drawer on that day. |
| `daily.layers.<layer>.days[].drawers.<drawer>.n_half_bullish` | integer | yes | no | Of these, posts with a probability of bullish above 0.5 (bullish at 0.5); the others are bearish at 0.5. |
| `daily.layers.<layer>.days[].drawers.<drawer>.half_bullish_share` | number | yes | yes | The counted bullish share; `null` when `too_few_posts`. |
| `daily.layers.<layer>.days[].drawers.<drawer>.calibrated_bullish_share` | number | yes | yes | The calibrated bullish share; `null` when `too_few_posts` or the status is not `applied`. |
| `daily.layers.<layer>.days[].drawers.<drawer>.too_few_posts` | boolean | yes | no | True when the drawer has fewer posts than `daily.min_posts` on that day. |
| `daily.events` | object | yes | no | Example posts of the days in `EVENT_DAYS`. |
| `daily.events.days` | array | yes | no | `EVENT_DAYS`, in the order given. |
| `daily.events.days[]` | string | — | no | One day, `YYYY-MM-DD`. |
| `daily.events.examples_per_side` | integer | yes | no | `EVENT_EXAMPLES`: the most example posts per day, layer and side. |
| `daily.events.bullish_at_or_above` | number | yes | no | `BULLISH_THRESHOLD`: a bullish example has a probability of bullish at or above it. |
| `daily.events.bearish_at_or_below` | number | yes | no | `BEARISH_THRESHOLD`: a bearish example has a probability of bullish at or below it. |
| `daily.events.items` | array | yes | no | One item per day of `days`, in the same order. |
| `daily.events.items[]` | object | — | no | One event day. |
| `daily.events.items[].date` | string | yes | no | The day, `YYYY-MM-DD`. |
| `daily.events.items[].layers` | object | yes | no | Every layer of `daily.layers`, keyed by `<layer>`. |
| `daily.events.items[].layers.<layer>` | object | — | no | The day in one layer. |
| `daily.events.items[].layers.<layer>.in_period` | boolean | yes | no | True when the day lies within the layer's period; when false, the other fields are 0 or empty. |
| `daily.events.items[].layers.<layer>.n_posts` | integer | yes | no | The layer's posts on that day. |
| `daily.events.items[].layers.<layer>.bullish` | array | yes | no | At most `examples_per_side` posts with a probability of bullish at or above `bullish_at_or_above`, highest first, every text once (ties by `post_id`). |
| `daily.events.items[].layers.<layer>.bullish[]` | object | — | no | One example post. |
| `daily.events.items[].layers.<layer>.bullish[].post_id` | string | yes | no | The post's id in the dated post table. |
| `daily.events.items[].layers.<layer>.bullish[].text` | string | yes | no | Its text, with account names and links replaced. |
| `daily.events.items[].layers.<layer>.bullish[].bullish_probability` | number | yes | no | The model's probability of bullish for it. |
| `daily.events.items[].layers.<layer>.bearish` | array | yes | no | At most `examples_per_side` posts with a probability of bullish at or below `bearish_at_or_below`, lowest first, every text once (ties by `post_id`). |
| `daily.events.items[].layers.<layer>.bearish[]` | object | — | no | One example post. |
| `daily.events.items[].layers.<layer>.bearish[].post_id` | string | yes | no | The post's id in the dated post table. |
| `daily.events.items[].layers.<layer>.bearish[].text` | string | yes | no | Its text, with account names and links replaced. |
| `daily.events.items[].layers.<layer>.bearish[].bullish_probability` | number | yes | no | The model's probability of bullish for it. |
| `daily.events.items[].layers.<layer>.new_words` | array | yes | no | The layer's new words of that day, in the order of `new_words` (empty when the day is not judged or has none). |
| `daily.events.items[].layers.<layer>.new_words[]` | string | — | no | One new word. |

### new_words

Chapter 13: per layer and UTC day, the words that suddenly appear in many more posts than on the days before. Words are the pieces of a post's cleaned text (`clean_text`) between spaces; a word counts once per post. A word's count on a day is the number of the layer's posts that day containing it; its past daily mean is its counts on the `window_days` calendar days before, added up and divided by `window_days` (a day without posts counts 0). A word is new when its count is at least `min_count` and at least `ratio` × its past daily mean; with a past daily mean of 0 only the first condition applies. No model is involved, so the same switches always give the same section. Like `dated_data`, it carries its own period, for the section and for every layer, and every layer's `days` are exactly the days of `dated_data.layers.<layer>.days`.

| Field | Type | Required | Null | Meaning |
| --- | --- | --- | --- | --- |
| `new_words` | object | yes | no | Chapter 13: the new words of every layer and day, with example posts. |
| `new_words.post_table` | string | yes | no | The table the posts come from, `dated_posts`. |
| `new_words.text` | string | yes | no | The text the words come from: `clean_text`. |
| `new_words.word_rule` | string | yes | no | What a word and its count are, as an English sentence. |
| `new_words.day` | string | yes | no | The unit of every day in this section: `"UTC"`. |
| `new_words.date_from` | string | yes | no | The first UTC day of the section, the same as `dated_data.date_from`. |
| `new_words.date_to` | string | yes | no | The last UTC day of the section, the same as `dated_data.date_to`. |
| `new_words.window_days` | integer | yes | no | `NEW_WORD_WINDOW`: the calendar days before a day that give the past daily mean. |
| `new_words.min_count` | integer | yes | no | `NEW_WORD_MIN_COUNT`: a new word is in at least this many posts of the day. |
| `new_words.ratio` | number | yes | no | `NEW_WORD_RATIO`: a new word is in at least this many times as many posts as its past daily mean. |
| `new_words.max_per_day` | integer | yes | no | `NEW_WORD_MAX_PER_DAY`: the most new words reported per layer and day. |
| `new_words.quotes_per_word` | integer | yes | no | The most example posts per new word (1 to 3). |
| `new_words.new_word_rule` | string | yes | no | When a word is new, as an English sentence. |
| `new_words.ignored_words` | array | yes | no | Words that are never new: NLTK's English stop words, `never`, `bitcoin`, `btc`, `user` and `http`, sorted. |
| `new_words.ignored_words[]` | string | — | no | One such word. |
| `new_words.ignored_rule` | string | yes | no | The words that never count, also words without a letter and words of a single character, as an English sentence. |
| `new_words.order` | string | yes | no | The order of a day's new words, which decides which ones are kept when there are more than `max_per_day`: most posts first, then the higher ratio (`null` first), then the word. |
| `new_words.quote_rule` | string | yes | no | How the example posts are chosen, as an English sentence. |
| `new_words.layers` | object | yes | no | Every layer of `dated_data.layers`, keyed by `<layer>`. |
| `new_words.layers.<layer>` | object | — | no | One layer. |
| `new_words.layers.<layer>.date_from` | string | yes | no | The layer's first day, the same as in `dated_data`. |
| `new_words.layers.<layer>.date_to` | string | yes | no | The layer's last day, the same as in `dated_data`. |
| `new_words.layers.<layer>.n_days_judged` | integer | yes | no | Days judged: every day but the first `window_days`. |
| `new_words.layers.<layer>.n_days_with_new_words` | integer | yes | no | Days with at least one new word. |
| `new_words.layers.<layer>.n_new_words` | integer | yes | no | New words over all days (a word new on two days counts twice). |
| `new_words.layers.<layer>.days` | array | yes | no | One entry per calendar day from the layer's `date_from` to its `date_to`, sorted by date: the same days as in `dated_data`. |
| `new_words.layers.<layer>.days[]` | object | — | no | One day. |
| `new_words.layers.<layer>.days[].date` | string | yes | no | The UTC day, `YYYY-MM-DD`. |
| `new_words.layers.<layer>.days[].n_posts` | integer | yes | no | The layer's posts on that day, the same as in `dated_data`. |
| `new_words.layers.<layer>.days[].judged` | boolean | yes | no | False for the first `window_days` days of the layer, which have fewer days before them; such a day has no new words. |
| `new_words.layers.<layer>.days[].new_words` | array | yes | no | The day's new words in the order of `new_words.order`, at most `max_per_day`. |
| `new_words.layers.<layer>.days[].new_words[]` | object | — | no | One new word. |
| `new_words.layers.<layer>.days[].new_words[].word` | string | yes | no | The word, as it appears in the cleaned texts. |
| `new_words.layers.<layer>.days[].new_words[].n_posts` | integer | yes | no | Its count: the layer's posts that day whose cleaned text contains it (at least `min_count`). |
| `new_words.layers.<layer>.days[].new_words[].past_daily_mean` | number | yes | no | Its past daily mean (0 when no post of the `window_days` days before contained it). |
| `new_words.layers.<layer>.days[].new_words[].ratio` | number | yes | yes | `n_posts` ÷ `past_daily_mean`, at least `new_words.ratio`; `null` when `past_daily_mean` is 0 (never infinity). |
| `new_words.layers.<layer>.days[].new_words[].quotes` | array | yes | no | 1 to `quotes_per_word` posts of that layer and day whose cleaned text contains the word, in `post_id` order, every text once. |
| `new_words.layers.<layer>.days[].new_words[].quotes[]` | object | — | no | One example post. |
| `new_words.layers.<layer>.days[].new_words[].quotes[].post_id` | string | yes | no | The post's id in the dated post table. |
| `new_words.layers.<layer>.days[].new_words[].quotes[].text` | string | yes | no | Its text, with account names and links replaced. |

### layers

Chapter 14: the two layers with data, the media layer (news headlines) and the retail layer (tweets), side by side, day by day, on the days that both cover; and the sentiment model measured on the news headlines that carry a reader label. Like `dated_data`, this section describes a period and carries its own `date_from` and `date_to`: the overlap of the two layers' periods. The shares are the calibrated bullish shares of `daily`, copied as they are.

A layer **leans bullish** on a day when its calibrated share is at least `bullish_at_or_above` (0.5 + `gap`) and **leans bearish** when it is at most `bearish_at_or_below` (0.5 − `gap`). The two layers **lean opposite ways** on a day when one leans bullish and the other bearish. A day on which a layer's calibrated share is `null` (too few posts, or no calibration) takes no part. An object keyed by `<layer>` has one key per layer of `compared`.

| Field | Type | Required | Null | Meaning |
| --- | --- | --- | --- | --- |
| `layers` | object | yes | no | Chapter 14: the media and the retail layer side by side, and the model on the news headlines. |
| `layers.compared` | array | yes | no | The two layers compared, in this order: `media`, `retail`. |
| `layers.compared[]` | string | — | no | One layer. |
| `layers.share` | string | yes | no | Which share is compared, and which days take no part, as an English sentence. |
| `layers.date_from` | string | yes | yes | The first day both layers cover (the later of the two layers' `date_from` in `daily`), `YYYY-MM-DD`; `null` when a layer has no posts or the two periods do not meet. |
| `layers.date_to` | string | yes | yes | The last day both layers cover (the earlier of the two layers' `date_to`); `null` as above. |
| `layers.gap` | number | yes | no | `LAYER_GAP`: how far from 0.5 a calibrated share must be for a lean, above 0 and at most 0.5. |
| `layers.bullish_at_or_above` | number | yes | no | 0.5 + `gap`: a calibrated share at or above it leans bullish. |
| `layers.bearish_at_or_below` | number | yes | no | 0.5 − `gap`: a calibrated share at or below it leans bearish. |
| `layers.opposite_rule` | string | yes | no | When the two layers lean opposite ways, as an English sentence. |
| `layers.n_days` | integer | yes | no | Calendar days from `date_from` to `date_to` (0 when they are `null`); equal to the length of `days`. |
| `layers.n_days_with_share` | object | yes | no | Per layer of `compared`, keyed by `<layer>`: the days of `days` on which it has a calibrated share. |
| `layers.n_days_with_share.<layer>` | integer | — | no | Days with a calibrated share of this layer. |
| `layers.n_days_both` | integer | yes | no | Days of `days` on which both layers have a calibrated share, so that `opposite` is not `null`. |
| `layers.n_days_opposite` | integer | yes | no | Days on which the two layers lean opposite ways; equal to the length of `opposite_dates`. |
| `layers.opposite_dates` | array | yes | no | The days on which the two layers lean opposite ways, sorted. |
| `layers.opposite_dates[]` | string | — | no | One day, `YYYY-MM-DD`. |
| `layers.days` | array | yes | no | One entry per calendar day from `date_from` to `date_to`, sorted by date; empty when they are `null`. |
| `layers.days[]` | object | — | no | One day. |
| `layers.days[].date` | string | yes | no | The UTC day, `YYYY-MM-DD`. |
| `layers.days[].calibrated_bullish_share` | object | yes | no | Per layer of `compared`, keyed by `<layer>`: that layer's calibrated bullish share of that day in `daily`. |
| `layers.days[].calibrated_bullish_share.<layer>` | number | — | yes | The share; `null` when the layer takes no part that day. |
| `layers.days[].opposite` | boolean | yes | yes | True when one layer leans bullish and the other bearish; `null` when a layer takes no part that day. |
| `layers.media_recognition` | object | yes | no | The sentiment model on the news headlines of the media layer that carry a bullish or bearish reader label, over the media layer's whole period (not only the overlap). It describes news headlines and the readers' votes, not comments. |
| `layers.media_recognition.layer` | string | yes | no | `media`. |
| `layers.media_recognition.date_from` | string | yes | yes | The media layer's first day, as in `daily`; `null` when the layer has no posts. |
| `layers.media_recognition.date_to` | string | yes | yes | The media layer's last day; `null` when the layer has no posts. |
| `layers.media_recognition.label_rule` | string | yes | no | How the news labels were made, the same as `dated_data.datasets.news.label_rule`: `market_direction` or `votes`. |
| `layers.media_recognition.text_column` | string | yes | no | The news text the model read, the same as `dated_data.datasets.news.text_column`: `title` or `text`. |
| `layers.media_recognition.half_rule` | string | yes | no | How a headline is split at 0.5, as in `daily.half_rule`. |
| `layers.media_recognition.n_labelled_bullish` | integer | yes | no | Headlines labelled bullish; equal to `dated_data.datasets.news.label_counts.2`. |
| `layers.media_recognition.n_labelled_bearish` | integer | yes | no | Headlines labelled bearish; equal to `dated_data.datasets.news.label_counts.0`. |
| `layers.media_recognition.n_bullish_called_bullish` | integer | yes | no | Of the headlines labelled bullish, those the model calls bullish at 0.5. |
| `layers.media_recognition.n_bearish_called_bullish` | integer | yes | no | Of the headlines labelled bearish, those the model calls bullish at 0.5. |
| `layers.media_recognition.true_bullish_rate` | number | yes | yes | `n_bullish_called_bullish` ÷ `n_labelled_bullish`; `null` when there is no headline labelled bullish. |
| `layers.media_recognition.false_bullish_rate` | number | yes | yes | `n_bearish_called_bullish` ÷ `n_labelled_bearish`; `null` when there is no headline labelled bearish. |
| `layers.media_recognition.stocktwits_true_bullish_rate` | number | yes | yes | For comparison: the true bullish rate measured on StockTwits test posts, `daily.calibration.true_bullish_rate`. |
| `layers.media_recognition.stocktwits_false_bullish_rate` | number | yes | yes | For comparison: `daily.calibration.false_bullish_rate`. |
| `layers.media_recognition.label_bullish_share` | number | yes | yes | Bullish share of the labelled headlines by the readers' labels: `n_labelled_bullish` ÷ (`n_labelled_bullish` + `n_labelled_bearish`); `null` when there are none. |
| `layers.media_recognition.half_bullish_share` | number | yes | yes | Counted bullish share of the same headlines: (`n_bullish_called_bullish` + `n_bearish_called_bullish`) ÷ (`n_labelled_bullish` + `n_labelled_bearish`); `null` when there are none. |
| `layers.media_recognition.calibrated_bullish_share` | number | yes | yes | `half_bullish_share` calibrated with the StockTwits rates of `daily.calibration`, clipped to [0, 1]; `null` when the status there is not `applied` or the counted share is `null`. The news rates calibrate nothing. |

### validation

Chapter 15: every layer's daily calibrated bullish share (from `daily`) set against two things measured elsewhere, the BTC price and the Crypto Fear and Greed Index of alternative.me, as Spearman rank correlations with the number of days that took part. It describes the past only: no p-value, no test and no forecast. The section carries its own period, every layer's period together, and every layer its own. **No daily price or index value is stored in the brief**: the price sources do not say that their data may be passed on, and the brief is not a price table. An object keyed by `<layer>` has one key per layer of `daily.layers`.

A day d takes part when the layer has a calibrated share that day; it is paired with the index of day d and with the **next day's change**, close of day d + 1 ÷ close of day d − 1 (UTC days). A day missing on either side takes no part.

| Field | Type | Required | Null | Meaning |
| --- | --- | --- | --- | --- |
| `validation` | object | yes | no | Chapter 15: the daily shares against the BTC price and the Fear and Greed Index. |
| `validation.day` | string | yes | no | The unit of every day in this section: `"UTC"`. |
| `validation.date_from` | string | yes | no | The earliest first day of the layers in `daily`, `YYYY-MM-DD`. |
| `validation.date_to` | string | yes | no | The latest last day of the layers in `daily`, `YYYY-MM-DD`. |
| `validation.share` | string | yes | no | Which share is used, and which days take no part, as an English sentence. |
| `validation.method` | string | yes | no | The correlation used, as an English sentence: Spearman's rank correlation, equal values at their average rank. |
| `validation.alignment` | string | yes | no | How the days are paired, as an English sentence. |
| `validation.min_days` | integer | yes | no | `VALIDATION_MIN_DAYS`: with fewer days taking part, a coefficient is `null`. |
| `validation.price` | object | yes | no | The BTC daily closes: where they come from and which days they cover. |
| `validation.price.source` | string | yes | no | The source used: `yfinance` (Yahoo Finance, `BTC-USD`), `coinbase` (Coinbase's public daily candles), `stooq` (Stooq's daily CSV), `local_file` (`PRICE_SOURCE` names a file) or `not_obtained`. |
| `validation.price.address` | string | yes | yes | Where they were read: the source's address, or the local file's path; `null` when not obtained. |
| `validation.price.reason` | string | yes | yes | Why the prices were not obtained (the reason of every source tried, or that `PRICE_SOURCE` is `"none"`); `null` when they were. |
| `validation.price.failed` | array | yes | no | The web sources tried before the one used (or all three when none worked), in the order tried; empty when the first one worked, from a local file or with `"none"`. |
| `validation.price.failed[]` | object | — | no | One source that was tried and gave nothing usable. |
| `validation.price.failed[].source` | string | yes | no | `yfinance`, `coinbase` or `stooq`. |
| `validation.price.failed[].reason` | string | yes | no | Why, as reported (for example an HTTP error, a web page instead of data, a request for a key, or missing days). |
| `validation.price.requested_from` | string | yes | no | The first day needed: `validation.date_from`. |
| `validation.price.requested_to` | string | yes | no | The last day needed: the day after `validation.date_to`, for the last day's change. |
| `validation.price.day_boundary` | string | yes | yes | How the source cuts its days, as an English sentence; `null` when not obtained. |
| `validation.price.fetched_on` | string | yes | yes | The UTC day the closes were fetched from the web, `YYYY-MM-DD` (also when they were read from the cache now); `null` for a local file or when not obtained. |
| `validation.price.from_cache` | boolean | yes | no | True when this run read the closes from its cache in `WORK_DIR/cache/` and connected to nothing. |
| `validation.price.date_from` | string | yes | yes | The first day with a close from `requested_from` to `requested_to`; `null` when there is none. |
| `validation.price.date_to` | string | yes | yes | The last day with a close in that range; `null` when there is none. |
| `validation.price.n_days` | integer | yes | no | Days with a close in that range (0 when not obtained). |
| `validation.fear_greed` | object | yes | no | The Crypto Fear and Greed Index: where it comes from and which days it covers. Data from alternative.me, which asks to be named next to its data. |
| `validation.fear_greed.source` | string | yes | no | `alternative.me` (its public address), `local_file` (`FEAR_GREED_SOURCE` names a file) or `not_obtained`. |
| `validation.fear_greed.address` | string | yes | yes | The address or the local file's path; `null` when not obtained. |
| `validation.fear_greed.reason` | string | yes | yes | Why the index was not obtained (or that `FEAR_GREED_SOURCE` is `"none"`); `null` when it was. |
| `validation.fear_greed.requested_from` | string | yes | no | The first day needed: `validation.date_from`. |
| `validation.fear_greed.requested_to` | string | yes | no | The last day needed: `validation.date_to`. |
| `validation.fear_greed.fetched_on` | string | yes | yes | The UTC day the index was fetched from the web; `null` for a local file or when not obtained. |
| `validation.fear_greed.from_cache` | boolean | yes | no | True when this run read the index from its cache and connected to nothing. |
| `validation.fear_greed.attribution` | string | yes | no | The source to name next to the index: alternative.me, with the address of its page. |
| `validation.fear_greed.date_from` | string | yes | yes | The first day with a value from `requested_from` to `requested_to`; `null` when there is none. |
| `validation.fear_greed.date_to` | string | yes | yes | The last day with a value in that range; `null` when there is none. |
| `validation.fear_greed.n_days` | integer | yes | no | Days with a value in that range (0 when not obtained). |
| `validation.layers` | object | yes | no | Every layer of `daily.layers`, keyed by `<layer>`. |
| `validation.layers.<layer>` | object | — | no | One layer. |
| `validation.layers.<layer>.date_from` | string | yes | no | The layer's first day, as in `daily`. |
| `validation.layers.<layer>.date_to` | string | yes | no | The layer's last day, as in `daily`. |
| `validation.layers.<layer>.n_days_with_share` | integer | yes | no | The layer's days with a calibrated share in `daily`. |
| `validation.layers.<layer>.fear_greed` | object | yes | no | The share against the index of the same day. |
| `validation.layers.<layer>.fear_greed.n_days` | integer | yes | no | Days that took part: a share and an index value on the same day. |
| `validation.layers.<layer>.fear_greed.spearman` | number | yes | yes | Spearman's rank correlation, from −1 to 1; `null` with fewer than `min_days` days or when either side has a single value. |
| `validation.layers.<layer>.next_day_change` | object | yes | no | The share against the next day's change of the BTC close. |
| `validation.layers.<layer>.next_day_change.n_days` | integer | yes | no | Days that took part: a share, a close on that day and a close on the next. |
| `validation.layers.<layer>.next_day_change.spearman` | number | yes | yes | Spearman's rank correlation, from −1 to 1; `null` as above. |

## Daily results

The StockTwits data has no dates, so the sections computed from it (`dataset`, `sentiment`, `topics`, `drawers`, `why` and `emotions`) describe the whole data source at once, and `scope.granularity` is `"batch"`. Results per day come from dated data and are added in this way, so that no field of an earlier format changes its name, type or meaning:

1. Results per day go into **sections of their own** (so far `dated_data`, `daily`, `new_words`, `layers` and `validation`; later versions add more). Each such section carries its own period, `date_from` and `date_to` (UTC days, `YYYY-MM-DD`), for the whole section and, where it has layers, for every layer, because the layers can cover different periods.
2. A list of days has one entry per calendar day from the first to the last day of its period, sorted by date, each with `date` (`YYYY-MM-DD`) and the section's counts for that day; a day without posts is still listed, with counts of 0 (and `null` for a share that cannot be computed).
3. The sections computed from the StockTwits data stay results for the whole data source; they get no `days`.
4. `scope.granularity` stays `"batch"` for such a brief, which is half whole-batch and half per day. `"daily"` is kept for a later brief whose every section is computed per day (for example from comments collected day by day); it is not allowed in the current format.

These are additions, so MINOR goes up by one (see below), together with the new rules and new rows in the tables above.

## Versions

`format_version` is `"MAJOR.MINOR"`. It describes the format (which fields exist, their types, units and meanings), not the results: more topics, another model in the evaluation table or different numbers do not change it.

- **MINOR goes up** (1.0 → 1.1) for a change that cannot break a program written for the old version, as long as that program ignores fields it does not know: a new field or a new section; a new list such as `days`; a new allowed value of a field with a fixed list of values (such as `"daily"`); a field that may be `null` becoming one that is never `null`.
- **MAJOR goes up** (1.x → 2.0) for a change that could make such a program read something wrong: a field removed or renamed; a field's type, unit or meaning changed; a field that is never `null` becoming one that may be `null`; a change to the label coding or to the keys of an object whose keys are fixed.
- **Every change to the rules changes the version**, and in the same commit: the `const` of `format_version` in [`brief.schema.json`](brief.schema.json), the identical copy of that file in the notebook's last chapter, the value the notebook writes into `format_version`, the tables of this document and the list below. The checks fail until the two copies of the rules are identical, this document lists exactly the fields they allow, and the notebook's briefs pass them.
- **Readers** should check the MAJOR number, accept any MINOR number of the same MAJOR, and ignore fields they do not know. The rules file describes one exact version; validate a brief with it only when `format_version` matches.

| Version | Change |
| --- | --- |
| 1.0 | First fixed format. |
| 1.1 | Chapter 10 counts the model's calls at 0.5 and calibrates the bullish share with the recognition rates of chapter 8, and marks rows with too few bullish or bearish posts. New: `why.min_bullish_bearish_posts` and `why.calibration`; in `why.overall` and every `why.topics[]` row `too_few_bullish_bearish`, `half_counts`, `half_bullish_share` and `calibrated_bullish_share`; in every `fair` block `half_counts`, `half_bullish_share`, `calibrated_bullish_share`, `half_minus_label_pp` and `calibrated_minus_label_pp`. Additions only: no field of 1.0 changed. |
| 1.2 | Chapter 9 puts the topic posts into drawers chosen by a person through a versioned word list and groups the posts in no drawer; chapter 10 gives every drawer the same row as a topic. New: the top-level section `drawers`; in `why` the fields `drawers` (one row per drawer), `any_drawer` and `no_drawer`. Additions only: no field of 1.1 changed. |
| 1.3 | Chapter 11 learns eight crypto emotions and `none` from silver labels (a pre-label file), evaluates two networks against two baselines and gives every topic post an emotion, overall and per drawer, with example posts and an interface for a person's review. New: the top-level section `emotions`. Additions only: no field of 1.2 changed. |
| 1.4 | Chapter 12 reads two public datasets with dates, tweets as the `retail` layer and news headlines as the `media` layer, into one dated post table, and counts its posts per layer and UTC day. New: the top-level section `dated_data`, which carries its own period (`date_from`, `date_to`, per layer as well). `scope.granularity` stays `"batch"`; the rules for results per day are rewritten in [Daily results](#daily-results). Additions only: no field of 1.3 changed. |
| 1.5 | Chapter 13 cleans and scores every dated post (bullish share at 0.5 and calibrated, emotion, drawers) and counts it per layer and UTC day, with example posts of chosen days, and finds the new words of every layer and day. New: the top-level sections `daily` and `new_words`, each carrying its own period like `dated_data`. The format check also refuses a day outside its layer's period. Additions only: no field of 1.4 changed. |
| 1.6 | Chapter 14 puts the media and the retail layer side by side on the days both cover and lists the days they lean opposite ways, and measures the sentiment model on the news headlines with a reader label; chapter 15 sets every layer's daily share against the BTC daily closes and the Fear and Greed Index of alternative.me as Spearman rank correlations, with where the data came from. New: the top-level sections `layers` and `validation`, each carrying its own period. Additions only: no field of 1.5 changed. |

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

To check a brief against the rules (only a brief of the version the rules describe), with the `jsonschema` package:

```python
from jsonschema import Draft202012Validator

with open("docs/brief.schema.json", encoding="utf-8") as handle:
    rules = json.load(handle)
for error in Draft202012Validator(rules).iter_errors(brief):
    print(list(error.absolute_path), error.message)
```
