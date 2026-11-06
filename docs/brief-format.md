# The brief: format reference

This document describes format **2.1** of the brief, the file `brief.json` that the notebook writes at the end of every run. Other programs read the brief, not the notebook, so this document is written for the authors of those programs.

- **The rules in machine-readable form** are in [`brief.schema.json`](brief.schema.json) (JSON Schema, draft 2020-12). The notebook's last chapter holds an identical copy of that file and checks every brief against it before saving; a brief that breaks a rule is not saved.
- **The checks** (`scripts/check.sh quick` and `full`) make sure that the two copies of the rules are identical, that the tables below list exactly the fields the rules allow (with the same type, the same "required" and the same "null"), and that every brief they produce passes the rules.

## Conventions

- **File.** UTF-8 JSON, indented. Field names are English and never change within a major version (see [Versions](#versions)).
- **null.** A number that cannot be computed is `null`: a share whose denominator is 0, a score on an empty subset, a summary of an empty table. It is never written as 0, and never as NaN or Infinity, which are not valid JSON. Only the fields marked "Null: yes" below may be `null`. Before saving, the notebook turns every NaN or infinite number into `null`; the check that follows then rejects the brief if that `null` sits in a field that may not be `null`. The check also rejects NaN and Infinity themselves, so a brief that contains them never passes.
- **Labels.** `0` = bearish, `1` = neutral, `2` = bullish. The dataset codes its classes in another order (0 neutral, 1 bullish, 2 bearish); the notebook converts them while reading, and `dataset.label_from_sentiment` records the conversion. As keys of an object they are strings (`"0"`, `"1"`, `"2"`).
- **Classes in use.** With three classes (the default) the models know `"0"`, `"1"` and `"2"`; with two classes only `"0"` and `"2"`. Objects keyed by "classes in use" always have the keys `"0"` and `"2"`, and `"1"` only with three classes.
- **Shares and percentage points.** A share is a fraction from 0 to 1 (0.25, not 25). A difference "in percentage points" (`_pp`) is 100 × (one share − another), from −100 to 100.
- **Counts** are integers, 0 or more.
- **Text fields that describe a method** (for example `sentiment.selection.by`) are short English sentences for people; programs should not parse them.

How to read the tables:

| Column | Meaning |
| --- | --- |
| Field | The path from the top of the brief. `[]` stands for every item of a list; `<label>`, `<model>` and `<metric>` stand for every key of an object whose keys vary (described in the row of that object). |
| Type | The JSON type of the value when it is not `null`: string, integer, number, boolean, object or array. "number" also allows whole numbers. |
| Required | "yes": always present in this format. "no": may be absent. "—": an item of a list or an entry of an object with varying keys; the row above it says which ones are there. |
| Null | "yes": may be `null` (only when the number cannot be computed). "no": never `null`. |

The rules allow no other fields: a writer that adds a field without changing the rules (and this document, and the version) produces a brief that does not pass.

## What each section covers

The top-level `scope` describes the brief as a whole: where the posts come from, the granularity, how many classes the models use, and whether the **working sample** keeps only posts that name a ticker. The two sections do not use the same posts, so **each section's coverage is given by its own fields, not by `scope`**:

| Section | Posts it uses | Fields that say so |
| --- | --- | --- |
| `dataset` | Every post of the data source; then the working sample drawn from them. | `dataset.n_posts`; `dataset.sample` |
| `sentiment` | The working sample (with the default switches: posts with or without a ticker, the classes in use, the same number in each class), split into training, validation and test sets. The evaluation table has rows for all test posts and for the test posts that name a ticker. | `sentiment.classes`; `sentiment.split`; `sentiment.evaluation.subsets` |
| `levels` | The posts of the data source that the level label file (`LEVEL_LABELS_SOURCE`) labels with one of the six levels or `none`, whatever `scope.tickers` and `scope.n_classes` say; with the default switches the same number from each of the seven classes; split into training, validation and test sets. | `levels.matching`; `levels.balance`; `levels.split` |

For example, `dataset.n_posts` counts every post of the data source, while `sentiment.split.n_posts` counts only the posts of the working sample.

## Fields

### Top level and scope

| Field | Type | Required | Null | Meaning |
| --- | --- | --- | --- | --- |
| `format_version` | string | yes | no | The format version, the one named at the top of this document. See [Versions](#versions). |
| `generated_at` | string | yes | no | When the brief was saved: UTC time, ISO 8601, to the second, ending in `Z` (for example `2026-01-31T12:00:00Z`). |
| `scope` | object | yes | no | What the brief as a whole covers. |
| `scope.source` | string | yes | no | The data source as given by the switch `DATA_SOURCE`: a Hugging Face address (`hf://...`) or a path to a local parquet file. |
| `scope.granularity` | string | yes | no | `"batch"`: every section describes the whole data source (`DATA_SOURCE`) at once. The format allows no other value. |
| `scope.tickers` | string | yes | no | Which posts the working sample holds: `"all"` (posts with or without a ticker) or `"has_ticker_only"` (only posts that name a ticker such as `$TSLA`; the switch `TICKER_ONLY`). It describes the working sample only; see [What each section covers](#what-each-section-covers). |
| `scope.n_classes` | integer | yes | no | Classes the models use: 3 (bearish, neutral and bullish; the default) or 2 (bearish and bullish). |

### dataset

Chapters 0 to 3: the data source, a first look at every post, the working sample and the cleaning.

| Field | Type | Required | Null | Meaning |
| --- | --- | --- | --- | --- |
| `dataset` | object | yes | no | The data and how it was prepared. |
| `dataset.name` | string | yes | no | Hugging Face dataset id, `"TimKoornstra/financial-tweets-sentiment"`. |
| `dataset.revision` | string | yes | no | The dataset's pinned revision (a commit id of 40 hexadecimal digits). |
| `dataset.file` | string | yes | no | The file inside the dataset, `"data/train-00000-of-00001.parquet"`. |
| `dataset.license` | string | yes | no | The dataset's license, `"MIT"`. |
| `dataset.citation` | string | yes | no | What to cite: the dataset's page, because its card asks for no particular citation. |
| `dataset.label_meaning` | object | yes | no | The meaning of each label. |
| `dataset.label_meaning.0` | string | yes | no | `"bearish"`. |
| `dataset.label_meaning.1` | string | yes | no | `"neutral"`. |
| `dataset.label_meaning.2` | string | yes | no | `"bullish"`. |
| `dataset.label_from_sentiment` | object | yes | no | How the dataset's own class (`sentiment`) was converted into the label: keyed by the dataset's value. |
| `dataset.label_from_sentiment.0` | integer | yes | no | Label of the dataset's 0 (neutral): `1`. |
| `dataset.label_from_sentiment.1` | integer | yes | no | Label of the dataset's 1 (bullish): `2`. |
| `dataset.label_from_sentiment.2` | integer | yes | no | Label of the dataset's 2 (bearish): `0`. |
| `dataset.sentiment_counts` | object | yes | no | Posts per value of the dataset's own `sentiment` column, over every post (a record; the same posts as `class_counts`, under the dataset's coding). |
| `dataset.sentiment_counts.0` | integer | yes | no | Posts the dataset marks 0 (neutral). |
| `dataset.sentiment_counts.1` | integer | yes | no | Posts the dataset marks 1 (bullish). |
| `dataset.sentiment_counts.2` | integer | yes | no | Posts the dataset marks 2 (bearish). |
| `dataset.n_posts` | integer | yes | no | Every post of the data source. |
| `dataset.n_unique_post_ids` | integer | yes | no | Distinct `post_id` values; equal to `n_posts`. |
| `dataset.n_empty_text` | integer | yes | no | Posts whose text is empty or blank after reading. |
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
| `dataset.ticker_mentions` | object | yes | no | Posts that name a ticker, over every post. |
| `dataset.ticker_mentions.rule` | string | yes | no | The matching rule, written out as a regular expression: `$`, then 1 to 5 letters in any case, optionally a dot and 1 or 2 letters, with no letter or digit right before the `$` or right after the code. |
| `dataset.ticker_mentions.n_posts` | integer | yes | no | Posts that name a ticker. |
| `dataset.ticker_mentions.share` | number | yes | yes | Their share of every post; `null` when there are no posts. |
| `dataset.ticker_mentions.by_class` | object | yes | no | Posts that name a ticker, per label. |
| `dataset.ticker_mentions.by_class.0` | integer | yes | no | Bearish posts that name a ticker. |
| `dataset.ticker_mentions.by_class.1` | integer | yes | no | Neutral posts that name a ticker. |
| `dataset.ticker_mentions.by_class.2` | integer | yes | no | Bullish posts that name a ticker. |
| `dataset.ticker_mentions.share_within_class` | object | yes | no | Within each label, the share of posts that name a ticker. |
| `dataset.ticker_mentions.share_within_class.0` | number | yes | yes | Among bearish posts; `null` when there is no bearish post. |
| `dataset.ticker_mentions.share_within_class.1` | number | yes | yes | Among neutral posts; `null` when there is no neutral post. |
| `dataset.ticker_mentions.share_within_class.2` | number | yes | yes | Among bullish posts; `null` when there is no bullish post. |
| `dataset.duplicates` | object | yes | no | Repeated texts (counted only; nothing is removed). |
| `dataset.duplicates.n_rows_repeating_an_earlier_text` | integer | yes | no | Posts whose exact text already appeared earlier in the table (empty texts count as repeats of each other). |
| `dataset.duplicates.n_distinct_texts_repeated` | integer | yes | no | Distinct texts that occur more than once. |
| `dataset.n_posts_with_links` | integer | yes | no | Posts whose text still contains a web link (`http://`, `https://` or `www.`). |
| `dataset.sample` | object | yes | no | The working sample: the posts that chapters 3 to 8 clean, train on and evaluate. |
| `dataset.sample.n_classes` | integer | yes | no | Classes kept (the switch `N_CLASSES`): 2 or 3. |
| `dataset.sample.ticker_only` | boolean | yes | no | Whether only posts that name a ticker were kept (the switch `TICKER_ONLY`). |
| `dataset.sample.seed` | integer | yes | no | Random seed of the draw (`RANDOM_SEED`). |
| `dataset.sample.per_class_requested` | integer | yes | no | Most posts per class asked for (`SAMPLE_PER_CLASS`). |
| `dataset.sample.available_by_class` | object | yes | no | Posts left in each class in use after the filters, before the draw. Keyed by the classes in use. |
| `dataset.sample.available_by_class.<label>` | integer | — | no | Posts available in that class. |
| `dataset.sample.per_class` | integer | yes | no | Posts drawn from every class: the smaller of `per_class_requested` and the smallest available class. |
| `dataset.sample.class_counts` | object | yes | no | Posts per class in the working sample, keyed by the classes in use. |
| `dataset.sample.class_counts.<label>` | integer | — | no | Posts of that class; the same for every class. |
| `dataset.sample.n_posts` | integer | yes | no | Posts in the working sample. |
| `dataset.sample.n_has_ticker` | integer | yes | no | Posts in the working sample that name a ticker. |
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

Chapters 4 to 8: tokenization (shown only), training, validation and test sets, three kinds of vectors with a one-layer network on each, the choice of one model and the evaluation on the test set.

| Field | Type | Required | Null | Meaning |
| --- | --- | --- | --- | --- |
| `sentiment` | object | yes | no | Bullish, neutral or bearish. |
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
| `sentiment.split.n_has_ticker` | object | yes | no | Posts that name a ticker in each part. |
| `sentiment.split.n_has_ticker.train` | integer | yes | no | In the training set. |
| `sentiment.split.n_has_ticker.validation` | integer | yes | no | In the validation set. |
| `sentiment.split.n_has_ticker.test` | integer | yes | no | In the test set. |
| `sentiment.vectorizers` | object | yes | no | Chapter 6's three vectorizers, fitted on the training set only. |
| `sentiment.vectorizers.vocab_size_requested` | integer | yes | no | Most words the word-count and the TF-IDF vectorizer keep (`VOCAB_SIZE`). |
| `sentiment.vectorizers.fitted_on` | string | yes | no | `"train"`. |
| `sentiment.vectorizers.fitted_on_n_posts` | integer | yes | no | Training posts they learned from. |
| `sentiment.vectorizers.text` | string | yes | no | `"clean_text"`: the cleaned text. |
| `sentiment.vectorizers.storage` | string | yes | no | How the vectors are stored: word counts and TF-IDF as sparse float32 matrices, averaged word vectors as a dense float32 table. |
| `sentiment.vectorizers.count` | object | yes | no | The word-count vectorizer. |
| `sentiment.vectorizers.count.vocab_size` | integer | yes | no | Words it kept. |
| `sentiment.vectorizers.tfidf` | object | yes | no | The TF-IDF vectorizer. |
| `sentiment.vectorizers.tfidf.vocab_size` | integer | yes | no | Words it kept. |
| `sentiment.vectorizers.word2vec` | object | yes | no | The averaged word vectors: a Word2Vec model trained on the cleaned texts of the training set (nothing downloaded), and a post's vector is the mean of its words' vectors. |
| `sentiment.vectorizers.word2vec.algorithm` | string | yes | no | How the word vectors were made, in words. |
| `sentiment.vectorizers.word2vec.vector_size` | integer | yes | no | Numbers in every vector (`WORD2VEC_DIM`). |
| `sentiment.vectorizers.word2vec.window` | integer | yes | no | Words on each side that count as a word's neighbours (`WORD2VEC_WINDOW`). |
| `sentiment.vectorizers.word2vec.min_count` | integer | yes | no | Uses in the training set a word needs to get a vector (`WORD2VEC_MIN_COUNT`). |
| `sentiment.vectorizers.word2vec.epochs` | integer | yes | no | Passes of Word2Vec over the training texts (`WORD2VEC_EPOCHS`). |
| `sentiment.vectorizers.word2vec.vocab_size` | integer | yes | no | Words that got a vector. |
| `sentiment.vectorizers.word2vec.fitted_on_n_posts` | integer | yes | no | Training posts it learned from; equal to the training set. |
| `sentiment.vectorizers.word2vec.unknown_words` | string | yes | no | What happens to a word without a vector: it is skipped when averaging. |
| `sentiment.vectorizers.word2vec.n_posts_all_zero` | object | yes | no | Posts with no word that has a vector (their vector is all zeros), in each part. |
| `sentiment.vectorizers.word2vec.n_posts_all_zero.train` | integer | yes | no | In the training set. |
| `sentiment.vectorizers.word2vec.n_posts_all_zero.validation` | integer | yes | no | In the validation set. |
| `sentiment.vectorizers.word2vec.n_posts_all_zero.test` | integer | yes | no | In the test set. |
| `sentiment.models` | array | yes | no | Model names, `["count_network", "tfidf_network", "word2vec_network"]`; later versions of the notebook may add more. Every `<model>` key below is one of them. |
| `sentiment.models[]` | string | — | no | A model name. |
| `sentiment.training` | object | yes | no | Chapter 7's training settings and history. |
| `sentiment.training.epochs` | integer | yes | no | Passes over the training set (`EPOCHS`). |
| `sentiment.training.batch_size` | integer | yes | no | Posts per training step (`BATCH_SIZE`). |
| `sentiment.training.optimizer` | string | yes | no | `"adam"`. |
| `sentiment.training.seed` | integer | yes | no | Seed set before building each network. |
| `sentiment.training.output_layer` | string | yes | no | The output layer: `"3 units, softmax"` with three classes, `"1 unit, sigmoid"` with two. |
| `sentiment.training.loss` | string | yes | no | The loss function. |
| `sentiment.training.target_of_label` | object | yes | no | What the network learns for each label, keyed by the classes in use. |
| `sentiment.training.target_of_label.<label>` | integer | — | no | The target: the label's position among the classes in use (with three classes `"0"` → 0, `"1"` → 1, `"2"` → 2; with two classes `"0"` → 0, `"2"` → 1). |
| `sentiment.training.history` | object | yes | no | Training history, one entry per model. |
| `sentiment.training.history.<model>` | object | — | no | One entry per Keras metric of that model: `loss`, `accuracy`, `val_loss`, `val_accuracy`. |
| `sentiment.training.history.<model>.<metric>` | array | — | no | One value per pass over the training set. |
| `sentiment.training.history.<model>.<metric>[]` | number | — | yes | The value after that pass; `null` if it is not a finite number. |
| `sentiment.selection` | object | yes | no | How the model for later chapters was chosen. |
| `sentiment.selection.by` | string | yes | no | The rule: macro-F1 on the validation set. |
| `sentiment.selection.validation_macro_f1` | object | yes | no | Every model's macro-F1 on the validation set. |
| `sentiment.selection.validation_macro_f1.<model>` | number | — | yes | The score; `null` when it cannot be computed. |
| `sentiment.selection.selected` | string | yes | no | The chosen model (a tie goes to the model listed first in `sentiment.models`). |
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
| `sentiment.scoring.score_texts` | object | yes | no | `score_texts(texts)`: raw texts in, cleaned and scored; a table with one row per text, a probability column per class in use and the call. |
| `sentiment.scoring.score_texts.function` | string | yes | no | `"score_texts"`. |
| `sentiment.scoring.score_texts.input` | string | yes | no | What it takes. |
| `sentiment.scoring.score_texts.columns` | array | yes | no | The columns of its table, for example `["p_bearish", "p_neutral", "p_bullish", "call"]`. |
| `sentiment.scoring.score_texts.columns[]` | string | — | no | A column name. |
| `sentiment.scoring.score_texts.demo` | object | yes | no | Self-check on the hand-written example sentences of chapter 7. |
| `sentiment.scoring.score_texts.demo.n_texts` | integer | yes | no | Sentences scored. |
| `sentiment.scoring.score_texts.demo.n_rows` | integer | yes | no | Rows returned; equal to `n_texts`. |
| `sentiment.scoring.score_texts.demo.rows_sum_to_one` | boolean | yes | no | Whether the probabilities of every row add up to 1. |
| `sentiment.scoring.score_texts.demo.call_is_highest` | boolean | yes | no | Whether every call is the class of the highest probability. |
| `sentiment.test_texts_also_in_train` | object | yes | no | Test posts whose text also occurs in the training set (counted only). |
| `sentiment.test_texts_also_in_train.same_text` | integer | yes | no | Same original text. |
| `sentiment.test_texts_also_in_train.same_clean_text` | integer | yes | no | Same cleaned text. |
| `sentiment.evaluation` | object | yes | no | Chapter 8's evaluation on the test set. |
| `sentiment.evaluation.subsets` | object | yes | no | The subsets of the test set, keyed by the names used in `rows[].subset`. |
| `sentiment.evaluation.subsets.all_test` | string | yes | no | Description: all test posts. |
| `sentiment.evaluation.subsets.test_has_ticker` | string | yes | no | Description: test posts that name a ticker. |
| `sentiment.evaluation.baselines` | array | yes | no | Names of the baselines: `"always_majority"` (always the most frequent training label, the smallest label on a tie) and `"uniform_random"` (a class at random with equal chances, seeded). |
| `sentiment.evaluation.baselines[]` | string | — | no | A baseline name. |
| `sentiment.evaluation.majority_label` | string | yes | no | The most frequent label of the training set. |
| `sentiment.evaluation.rows` | array | yes | no | The evaluation table: one row per model and baseline and per subset. |
| `sentiment.evaluation.rows[]` | object | — | no | One row. |
| `sentiment.evaluation.rows[].model` | string | yes | no | A model name or a baseline name. |
| `sentiment.evaluation.rows[].kind` | string | yes | no | `"model"` or `"baseline"`. |
| `sentiment.evaluation.rows[].subset` | string | yes | no | `"all_test"` or `"test_has_ticker"`. |
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

### levels

Chapter 9: the level label file, the posts it labels, the same number from every class, training, validation and test sets. The six levels and `none` are written as their English names (`assets_dividends`, `earnings`, `revenue_business`, `price_chart`, `story_macro`, `emotion_only`, `none`); `<class>` stands for each of the seven.

| Field | Type | Required | Null | Meaning |
| --- | --- | --- | --- | --- |
| `levels` | object | yes | no | Chapter 9: what a post pays attention to, on a ladder of six levels from close to cash to far from it, learned from silver labels. |
| `levels.silver_labels` | string | yes | no | One sentence on where the labels come from: a language model gave them, not a person. |
| `levels.labels_source` | string | yes | no | The label file as given by the switch `LEVEL_LABELS_SOURCE`: a web address or a local path. |
| `levels.labels_file` | object | yes | no | The label file as read. |
| `levels.labels_file.sha256` | string | yes | no | SHA-256 of the file's bytes (64 lower-case hexadecimal digits). |
| `levels.labels_file.n_rows` | integer | yes | no | Rows of the file. |
| `levels.labels_file.label_counts` | object | yes | no | Rows per value of the column `level`, over the whole file: the six levels, `none`, `disputed` and `unparsed`, all always present. |
| `levels.labels_file.label_counts.<value>` | integer | — | no | Rows with that value. |
| `levels.labels_file.models` | array | yes | no | The models named in the column `model`, sorted. |
| `levels.labels_file.models[]` | string | — | no | A model name. |
| `levels.labels_file.prompt_versions` | array | yes | no | The prompt versions named in the column `prompt_version`, sorted. |
| `levels.labels_file.prompt_versions[]` | string | — | no | A prompt version. |
| `levels.matching` | object | yes | no | How the rows of the label file were joined to the posts of the data source by `post_id`. |
| `levels.matching.data_is_sample` | boolean | yes | no | `true` when the data source is a part of the full file (its ids come from a column `row`, as in the repository's sample); then ids of the label file that are not in the data are expected and left out. `false` for the full file, where every id must be found. |
| `levels.matching.n_matched` | integer | yes | no | Rows whose id is a post of the data source (with the same text fingerprint). |
| `levels.matching.n_not_in_data` | integer | yes | no | Rows whose id is not a post of the data source; always 0 for the full file. |
| `levels.matching.n_other_text` | integer | yes | no | Rows whose text fingerprint is not that of the post's text; always 0 (the notebook stops otherwise). |
| `levels.dropped` | object | yes | no | Rows among the matched ones that carry no label and are not used. |
| `levels.dropped.disputed` | integer | yes | no | Rows `disputed` (too few votes for the most frequent label). |
| `levels.dropped.unparsed` | integer | yes | no | Rows `unparsed` (no usable answer). |
| `levels.classes` | array | yes | no | The seven classes the networks learn, in column order: the six levels (closest to cash first), then `none`. The format allows only this list. |
| `levels.classes[]` | string | — | no | A class. |
| `levels.balance` | object | yes | no | The posts used for training and evaluation, drawn from the matched posts with a label. |
| `levels.balance.enabled` | boolean | yes | no | The switch `LEVEL_BALANCE`: `true` takes the same number of posts from every class (the size of the smallest class), `false` takes them all. |
| `levels.balance.seed` | integer | yes | no | Random seed of the draw. |
| `levels.balance.counts_before` | object | yes | no | Posts per class before the draw, keyed by the seven classes. |
| `levels.balance.counts_before.<class>` | integer | — | no | Posts of that class. |
| `levels.balance.counts_after` | object | yes | no | Posts per class after the draw, keyed by the seven classes. |
| `levels.balance.counts_after.<class>` | integer | — | no | Posts of that class. |
| `levels.split` | object | yes | no | Training, validation and test sets of the drawn posts, as in `sentiment.split`. |
| `levels.split.seed` | integer | yes | no | Random seed of both splits. |
| `levels.split.fractions` | object | yes | no | Intended size of each part. |
| `levels.split.fractions.train` | number | yes | no | 0.8. |
| `levels.split.fractions.validation` | number | yes | no | 0.1. |
| `levels.split.fractions.test` | number | yes | no | 0.1. |
| `levels.split.stratified_by` | string | yes | no | `"level"`: every part keeps the class shares of the drawn posts. |
| `levels.split.n_posts` | object | yes | no | Posts in each part. |
| `levels.split.n_posts.train` | integer | yes | no | Training posts. |
| `levels.split.n_posts.validation` | integer | yes | no | Validation posts. |
| `levels.split.n_posts.test` | integer | yes | no | Test posts. |
| `levels.split.class_counts` | object | yes | no | Posts per class in each part. |
| `levels.split.class_counts.train` | object | yes | no | In the training set, keyed by the seven classes. |
| `levels.split.class_counts.train.<class>` | integer | — | no | Posts of that class. |
| `levels.split.class_counts.validation` | object | yes | no | In the validation set, keyed by the seven classes. |
| `levels.split.class_counts.validation.<class>` | integer | — | no | Posts of that class. |
| `levels.split.class_counts.test` | object | yes | no | In the test set, keyed by the seven classes. |
| `levels.split.class_counts.test.<class>` | integer | — | no | Posts of that class. |
| `levels.split.class_shares` | object | yes | no | Class shares in each part. |
| `levels.split.class_shares.train` | object | yes | no | In the training set, keyed by the seven classes. |
| `levels.split.class_shares.train.<class>` | number | — | yes | Share of that class; `null` when the part is empty. |
| `levels.split.class_shares.validation` | object | yes | no | In the validation set, keyed by the seven classes. |
| `levels.split.class_shares.validation.<class>` | number | — | yes | Share of that class; `null` when the part is empty. |
| `levels.split.class_shares.test` | object | yes | no | In the test set, keyed by the seven classes. |
| `levels.split.class_shares.test.<class>` | number | — | yes | Share of that class; `null` when the part is empty. |
| `levels.split.n_has_ticker` | object | yes | no | Posts that name a ticker in each part. |
| `levels.split.n_has_ticker.train` | integer | yes | no | In the training set. |
| `levels.split.n_has_ticker.validation` | integer | yes | no | In the validation set. |
| `levels.split.n_has_ticker.test` | integer | yes | no | In the test set. |

## Versions

`format_version` is `"MAJOR.MINOR"`. It describes the format (which fields exist, their types, units and meanings), not the results: more topics, another model in the evaluation table or different numbers do not change it.

- **MINOR goes up** (1.0 → 1.1) for a change that cannot break a program written for the old version, as long as that program ignores fields it does not know: a new field or a new section; a new list; a new allowed value of a field with a fixed list of values; a field that may be `null` becoming one that is never `null`.
- **MAJOR goes up** (1.x → 2.0) for a change that could make such a program read something wrong: a field removed or renamed; a field's type, unit or meaning changed; a field that is never `null` becoming one that may be `null`; a change to the label coding or to the keys of an object whose keys are fixed.
- **Every change to the rules changes the version**, and in the same commit: the `const` of `format_version` in [`brief.schema.json`](brief.schema.json), the identical copy of that file in the notebook's last chapter (chapter 9), the value the notebook writes into `format_version`, the tables of this document and the list below. The checks fail until the two copies of the rules are identical, this document lists exactly the fields they allow, and the notebook's briefs pass them.
- **Readers** should check the MAJOR number, accept any MINOR number of the same MAJOR, and ignore fields they do not know. The rules file describes one exact version; validate a brief with it only when `format_version` matches.

| Version | Change |
| --- | --- |
| 1.0 to 1.7 | The formats of the notebook's earlier, longer version, which had nine more sections. Their rules and tables are in the repository's history. |
| 2.0 | The notebook keeps chapters 0 to 8 and a short last chapter that checks and saves the brief; the later chapters were removed, and their sections with them. Removed: every top-level section except `format_version`, `generated_at`, `scope`, `dataset` and `sentiment` (nine sections), and the format check's rule that compared a list of days with its period. Renamed: `scope.coins` became `scope.tickers` (values `"all"` and `"has_ticker_only"`); the coin-mention fields became the ticker fields: `dataset.ticker_mentions`, `dataset.sample.ticker_only`, `dataset.sample.n_has_ticker` and `sentiment.split.n_has_ticker`; the evaluation subset of those posts became `test_has_ticker` (in `sentiment.evaluation.subsets` and `rows[].subset`). Removed with the old data file: `dataset.sheets`. Added: `dataset.revision`, `dataset.label_from_sentiment`, `dataset.sentiment_counts`, `sentiment.vectorizers.word2vec` and `sentiment.scoring.score_texts`. Removing and renaming fields makes this a MAJOR change. 2.0 covers every change made on the way to this version; the commits in between carry no number of their own. |
| 2.1 | A new top-level section, `levels` (chapter 9: the six levels and `none`, learned from the level label file), and the definitions it uses (`level_class`, `level_class_counts`, `level_class_shares` and `level`). Nothing else changed. Adding a section makes this a MINOR change. 2.1 covers every change made on the way to this version; the commits in between carry no number of their own. |

## Reading a brief

A minimal reader in Python:

```python
import json


def reject(constant):
    raise ValueError(f"{constant} is not allowed in a brief")


with open("data/brief.json", encoding="utf-8") as handle:
    brief = json.load(handle, parse_constant=reject)  # NaN and Infinity are not valid JSON

if brief["format_version"].split(".")[0] != "2":
    raise ValueError(f"cannot read format {brief['format_version']}")
for row in brief["sentiment"]["evaluation"]["rows"]:  # one row per model and subset
    score = row["macro_f1"]
    print(row["model"], row["subset"], "-" if score is None else f"{score:.3f}")
```

To check a brief against the rules (only a brief of the version the rules describe), with the `jsonschema` package:

```python
from jsonschema import Draft202012Validator

with open("docs/brief.schema.json", encoding="utf-8") as handle:
    rules = json.load(handle)
for error in Draft202012Validator(rules).iter_errors(brief):
    print(list(error.absolute_path), error.message)
```
