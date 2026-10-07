# The dated post table

This document describes the **dated post table**, shape version `dated-posts-v1`: the one table that every source of dated posts is turned into before any daily result is computed. Chapter 12 of the notebook builds it from the two dated datasets (tweets and news); a later source, such as video comments, is turned into the same table.

The notebook writes the same shape down as `DATED_POST_SHAPE` and checks every dated post table with `dated_table_problems(table)` before anything else uses it. The quick check makes sure that the table below and `DATED_POST_SHAPE` list exactly the same columns, with the same types and the same "Null".

## Columns

One row per post, the columns in this order:

| Column | Type | Null | Meaning |
| --- | --- | --- | --- |
| `post_id` | string | no | The post's id, unique in the table: `<dataset>:<fingerprint>`, for example `dlt-tweets:3f9a0c2b7d1e4a65`. For the tweets the fingerprint is made from the original time text and the text after the two replacements below; for the news, from the original time text and the CryptoPanic address. The same post gets the same id on every run, whatever order the files are read in. |
| `date` | date | no | The UTC day of the post, as text `YYYY-MM-DD`. Every daily result is counted by this column. When `time` is given, `date` is its UTC day. |
| `time` | utc_time | yes | The post's time in UTC, to the second (pandas `datetime64` with the time zone UTC); empty when the data gives only a date. |
| `text` | string | no | The post's text, with account names and links replaced (see below); it may be an empty string. |
| `layer` | string | no | Who writes: `media` (news), `kol` (accounts with a large audience) or `retail` (ordinary people). No other value is allowed. |
| `source` | string | no | Where the post comes from: for news, the domain of the original article in lower case without a leading `www.` (`unknown` when there is no address or no domain can be read from it); for tweets, `twitter`. |
| `language` | string | yes | The language the data gives for the post (for example `en`); empty when the data gives none per post. |
| `mentions_btc` | boolean | no | The text mentions `bitcoin` or `btc` as a whole word, ignoring case (the rule `\b(?:bitcoin\|btc)\b` of chapter 1), checked after the replacements. |
| `label` | integer | yes | The layer's own label in this project's coding: 0 bearish, 1 neutral, 2 bullish. Empty when the layer has no labels (tweets). |
| `votes_bullish` | integer | yes | Bullish votes of the readers (news); empty when there are none (tweets). |
| `votes_bearish` | integer | yes | Bearish votes of the readers (news); empty when there are none (tweets). |

Types: `string` is text; `date` is text `YYYY-MM-DD` naming a real day; `utc_time` is a time with the UTC time zone; `boolean` is true or false; `integer` is a whole number (a nullable integer type where the column may be empty). "Null: yes" means the column may be empty in a row; "no" means it never is.

## Layers

| Layer | Meaning | Data so far |
| --- | --- | --- |
| `media` | What the news says. | `ExponentialScience/DLT-Sentiment-News` (CryptoPanic headlines, 2021 to 2025) |
| `kol` | What accounts with a large audience say. | none yet (the tweets come without authors) |
| `retail` | What ordinary people say. | `ExponentialScience/DLT-Tweets` (tweets, mainly 2021 to mid-2023) |

## Two replacements in every text

Before an id is made and before `mentions_btc` is decided, every text gets two replacements:

- **Links** become `http`. A link is a piece of text that starts with `http://`, `https://` or `www.` (in any case) and runs up to the next whitespace.
- **Account names** become `@user`. An account name is `@` followed by one or more letters, digits or underscores, where the character right before `@` is not a letter, digit or underscore. An e-mail address such as `name@example.com` is therefore not an account name and stays as it is.

Doing the replacements twice gives the same text as doing them once. They keep other people's account names out of example posts, and they keep account names and links from showing up as odd new words in later counts of words per day.

## Times

The times in the data are written in several ways. They are read as follows: a time with a time zone is converted to UTC; a time without a time zone is taken as UTC; a value with only a date gives `date` and leaves `time` empty; a value that cannot be read as a time puts the row out of the table (the number of such rows is reported in the brief's `dated_data` section).
