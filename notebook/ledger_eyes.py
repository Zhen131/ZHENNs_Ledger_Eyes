# %% [markdown]
# # Ledger Eyes: what is the crypto crowd saying?
#
# This notebook reads a public set of StockTwits crypto posts and turns what it finds into one small, fixed-format result file called the **brief**. Other programs read the brief, not the notebook.
#
# Chapters: 0 The question and the data · 1 Loading · 2 A first look · 3 Cleaning · Save the brief.
#
# Run it from top to bottom. Every setting you may want to change is in the switch cell below.
#
# ***
#
# # 账本的眼：币圈的人在说什么？
#
# 这个笔记本读一份公开的 StockTwits 币圈帖子，把看到的东西整理成一份固定格式的小文件，叫「简报」（brief）。别的程序读的是简报，不是笔记本。
#
# 章节：0 问题与数据 · 1 加载 · 2 看一眼 · 3 清洗 · 保存简报。
#
# 从头到尾运行即可。可能要改的设置都在下面的开关格里。

# %% [markdown]
# ## Switches
#
# All settings live in the next cell, one plain assignment per line. The comment on each line gives the default and what else the value can be. Results are reported with the defaults. Switches marked "two-state" can be flipped to their other value and the notebook still runs from top to bottom. The cell carries the `parameters` tag, so a tool such as papermill can replace the values without editing the notebook.
#
# ***
#
# ## 开关
#
# 所有设置都在下一格，一行一个普通赋值。每行的注释写明默认值和还能改成什么。报告结果以默认值为准。注释里标着 two-state 的开关可以切到另一个值，笔记本照样能从头跑完。这一格带 `parameters` 标签，papermill 之类的工具可以不改笔记本就换掉这些值。

# %% tags=["parameters"]
DATA_SOURCE = "hf://datasets/ElKulako/stocktwits-crypto/st-data-full.xlsx"  # default: the Hugging Face file; or a local .xlsx with the same sheets and columns
WORK_DIR = "data"  # default: "data"; or any folder for the cache and the saved brief
RANDOM_SEED = 42  # default: 42; or any integer
SAMPLE_PER_CLASS = 50000  # default: 50000; or any positive integer (most posts per class in the working sample)
N_CLASSES = 2  # default: 2 (bearish vs bullish); two-state, other: 3 (adds neutral)
BTC_ONLY = False  # default: False (posts about all coins); two-state, other: True (only posts that mention bitcoin or btc)
EMOJI_TO_WORDS = True  # default: True (emoji become English words); two-state, other: False (emoji are deleted with special characters)
KEEP_NEGATIONS = True  # default: True (not, no, never, nor survive cleaning); two-state, other: False (plain NLTK stop word list)

# %% [markdown]
# ## Setup
#
# The first cell below installs the packages that Google Colab does not ship with (for example `emoji` and `contractions`); packages that are already there are left alone. All installs happen in this one cell, before anything is imported, so Colab never asks for a restart halfway through. The cell after it prints the versions of the key packages, which helps when results differ between machines.
#
# ***
#
# ## 准备
#
# 下面第一格安装 Google Colab 没有自带的包（例如 `emoji`、`contractions`），已经有的包不动。所有安装都集中在这一格、排在一切 import 之前，这样 Colab 不会在半路要求重启。再下一格打印关键包的版本，不同机器跑出的结果不一样时方便对照。

# %%
# Install missing packages before anything else is imported.
import importlib.util  # standard library: tells whether a package can be imported
import os  # standard library: environment variables
import subprocess  # standard library: runs pip
import sys  # standard library: the Python that runs this notebook

# Never send a Hugging Face token that may be stored on this machine, and switch off Hugging Face telemetry.
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Import name -> pinned pip requirement. Only the missing ones are installed.
REQUIRED_PACKAGES = {
    "pandas": "pandas==2.2.3",
    "numpy": "numpy==2.2.6",
    "sklearn": "scikit-learn==1.6.1",
    "nltk": "nltk==3.9.4",
    "openpyxl": "openpyxl==3.1.5",
    "pyarrow": "pyarrow==25.0.1",
    "huggingface_hub": "huggingface_hub==0.36.2",
    "fsspec": "fsspec==2026.9.0",
    "matplotlib": "matplotlib==3.10.9",
    "bs4": "beautifulsoup4==4.15.0",
    "tqdm": "tqdm==4.70.1",
    "emoji": "emoji==2.16.0",
    "contractions": "contractions==0.1.73",
}
missing = [requirement for module, requirement in REQUIRED_PACKAGES.items() if importlib.util.find_spec(module) is None]
if missing:
    print("Installing:", ", ".join(missing))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *missing])  # one pip call for all of them
else:
    print("All required packages are already installed.")

# %%
# Print the versions of the key packages.
from importlib.metadata import version  # standard library: version of an installed package

print(f"python {sys.version.split()[0]}")  # Python version
for distribution in ["pandas", "numpy", "scikit-learn", "nltk", "openpyxl", "pyarrow", "huggingface_hub",
                     "fsspec", "matplotlib", "beautifulsoup4", "tqdm", "emoji", "contractions"]:
    print(f"{distribution} {version(distribution)}")  # one line per package

# %% [markdown]
# ## Check the switches
#
# A wrong value should stop the notebook here with a clear message, not deep inside a later chapter.
#
# ***
#
# ## 检查开关
#
# 开关填错了，应该在这里就停下并说清楚原因，而不是跑到后面某一章才莫名出错。

# %%
# Stop early on switch values the notebook cannot work with.
if N_CLASSES not in (2, 3):
    raise ValueError(f"N_CLASSES must be 2 or 3, got {N_CLASSES!r}")
if not isinstance(SAMPLE_PER_CLASS, int) or SAMPLE_PER_CLASS < 1:
    raise ValueError(f"SAMPLE_PER_CLASS must be a positive integer, got {SAMPLE_PER_CLASS!r}")
if not isinstance(RANDOM_SEED, int):
    raise ValueError(f"RANDOM_SEED must be an integer, got {RANDOM_SEED!r}")
for name, value in [("BTC_ONLY", BTC_ONLY), ("EMOJI_TO_WORDS", EMOJI_TO_WORDS), ("KEEP_NEGATIONS", KEEP_NEGATIONS)]:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be True or False, got {value!r}")
print("Switches look fine.")

# %% [markdown]
# ## The brief
#
# The brief is the notebook's real output: one dictionary, `brief`, that the chapters fill in and the last cell saves as `WORK_DIR/brief.json`. Its top level is fixed: `format_version`, `generated_at`, `scope`, `dataset`, `sentiment`, `topics` and `why`. `scope` says what the results cover: the data source, the granularity (`"batch"` means one result for the whole dataset; `"daily"` is kept for dated data), which coins the working sample covers, and how many classes are used. Sections that no chapter fills stay `null`. A number that cannot be computed (for example a share whose denominator is zero) is written as `null`, never as 0 or NaN.
#
# ***
#
# ## 简报
#
# 简报是这个笔记本真正的产出：一个字典 `brief`，各章往里填自己的结果，最后一格把它存成 `WORK_DIR/brief.json`。顶层字段固定为 `format_version`、`generated_at`、`scope`、`dataset`、`sentiment`、`topics`、`why`。`scope` 说明结果覆盖什么：数据来源；粒度（`"batch"` 表示对整份数据给一个结果，`"daily"` 留给带日期的数据）；工作样本包含哪些币；用几分类。没有章节填写的节保持 `null`。算不出来的数（例如分母为 0 的比例）一律写 `null`，不写 0，也不写 NaN。

# %%
# Create the brief. Chapters fill in their own sections; unfilled sections stay None (null in JSON).
brief = {
    "format_version": "0.1-draft",  # draft value until the format is fixed
    "generated_at": None,  # UTC time, set when the brief is saved
    "scope": {
        "source": DATA_SOURCE,  # where the posts come from
        "granularity": "batch",  # one result for the whole dataset
        "coins": "btc_only" if BTC_ONLY else "all",  # which posts the working sample covers
        "n_classes": N_CLASSES,  # 2 = bearish vs bullish, 3 = with neutral
    },
    "dataset": None,  # chapters 0 to 3
    "sentiment": None,
    "topics": None,
    "why": None,
}
list(brief)  # show the top-level fields

# %% [markdown]
# ## 0. The question and the data
#
# **The question.** Crypto prices move with the mood of the crowd. This project reads that mood from what people write: are they bullish or bearish, what are they talking about, and which topics pull the mood up or down.
#
# **The data.** StockTwits crypto posts published on Hugging Face as `ElKulako/stocktwits-crypto` (file `st-data-full.xlsx`, two sheets, roughly 1.3 million posts written between 1 November 2021 and 15 June 2022). Each post has only two columns, `text` and `label`; there is no date, no user and no coin column. The dataset author already removed cashtags such as `$BTC`, hashtags, user names, links, wallet addresses, posts shorter than four words and duplicate posts, and turned all text to lower case.
#
# **The labels.** 0 = bearish, 1 = neutral, 2 = bullish, as described on the model card of the same author's CryptoBERT model. Bullish and bearish are tags that StockTwits users can attach to their own posts; the dataset card does not say how the neutral posts were labelled. The labels are noisy: some posts are hard to call even for a person.
#
# **License and citation.** The dataset is released under the MIT License. For academic use the dataset author asks to cite this paper: <https://ieeexplore.ieee.org/document/10223689>.
#
# ***
#
# ## 0. 问题与数据
#
# **问题。** 币价跟着人群的情绪走。这个项目从人们写的字里读出这种情绪：大家看多还是看空、在聊什么、哪些话题把情绪往上拉或往下拽。
#
# **数据。** StockTwits 的币圈帖子，发布在 Hugging Face 上，名叫 `ElKulako/stocktwits-crypto`（文件 `st-data-full.xlsx`，两个工作表，约 130 万条，写于 2021 年 11 月 1 日至 2022 年 6 月 15 日）。每条帖子只有 `text` 和 `label` 两列，没有日期、用户和币种。数据集作者已经去掉了 `$BTC` 这类币种标签、话题标签、用户名、链接、钱包地址、不足四个词的帖子和重复帖子，并把文字都转成了小写。
#
# **标签。** 0 = 看空，1 = 中性，2 = 看多，出自同一作者 CryptoBERT 模型的说明页。看多、看空是 StockTwits 用户发帖时可以给自己的帖子打的标记；数据集说明没有写中性是怎么来的。标签有噪音：有些帖子连人也很难判断。
#
# **许可证与引用。** 数据集使用 MIT 许可证。学术使用时，数据集作者要求引用这篇论文：<https://ieeexplore.ieee.org/document/10223689>。

# %%
# Record what the data is and what the labels mean.
LABEL_NAMES = {0: "bearish", 1: "neutral", 2: "bullish"}  # label value -> meaning

brief["dataset"] = {
    "name": "ElKulako/stocktwits-crypto",  # Hugging Face dataset id
    "file": "st-data-full.xlsx",  # the file inside the dataset
    "license": "MIT",
    "citation": "https://ieeexplore.ieee.org/document/10223689",  # paper the author asks to cite
    "label_meaning": {str(value): name for value, name in LABEL_NAMES.items()},
}
brief["dataset"]

# %% [markdown]
# ## 1. Loading
#
# The file has two sheets. We read both and stack them into one table, `df_all`, with three columns: `post_id`, `text` and `label`. `df_all` holds every post and stays untouched until the end of the notebook; later chapters take posts from it.
#
# Every post gets a stable id, `post_id`, made from the sheet name and the row number inside the sheet: `stocktwits_1:2` is the first post of the first sheet, because row 1 holds the column names. The id comes from the position in the file, not from random numbers, so with the same data source the same post gets the same id on every run. A text cell that the spreadsheet stores as a number becomes text, and a cell that pandas reads as empty becomes an empty string.
#
# **Known limitation.** In the full file, 59 posts begin with "=", so the spreadsheet stores them as formulas; their stored value is an error such as `#NAME?`, and pandas reads them as empty. Their text is lost and they appear as empty strings (59 of 1,331,697 posts). We keep them as they are, which is also what pandas does on Colab.
#
# Reading a 70 MB spreadsheet takes a while, so the table is saved as a Parquet file (a compact file format that loads quickly) in `WORK_DIR/cache/`, and later runs read that file instead. The cache file name carries a fingerprint of the data source. For a local file the fingerprint is the SHA-256 hash of the file's bytes, so any change to the file leads to a fresh read; for a web address it is a hash of the address. A different data source therefore never picks up another source's cache.
#
# ***
#
# ## 1. 加载
#
# 文件有两个工作表。两个都读进来，上下拼成一张表 `df_all`，三列：`post_id`、`text`、`label`。`df_all` 存着全部帖子，一直原样留到笔记本最后，后面的章节从里面取帖子。
#
# 每条帖子有一个稳定的编号 `post_id`，由工作表名和帖子在表里的行号组成：`stocktwits_1:2` 是第一个工作表的第一条帖子（第 1 行是列名）。编号来自帖子在文件里的位置，不用随机数，所以数据来源不变时，同一条帖子每次运行的编号都一样。被表格存成数字的文字格转成文字；pandas 读成空的格子变成空字符串。
#
# **已知局限。** 完整文件里有 59 条帖子以「=」开头，表格把它们当成了公式，格子里存的是 `#NAME?` 这类错误值，pandas 读出来是空的。这些帖子的原文丢了，表里是空字符串（占 1,331,697 条中的 59 条）。我们原样保留，这和 Colab 上 pandas 的读法一致。
#
# 读一个 70 MB 的表格要花一阵子，所以读完后把表存成 Parquet 文件（一种读得很快的紧凑格式），放在 `WORK_DIR/cache/`，以后运行直接读它。缓存文件名里带着数据来源的「指纹」：本地文件用文件内容的 SHA-256 哈希，文件一改就会重新读；网址用网址本身的哈希。所以换了数据来源，绝不会读到别的来源的缓存。

# %%
# Functions that read the spreadsheet and fingerprint a data source.
import hashlib  # standard library: SHA-256 hashes
import time  # standard library: timing
from pathlib import Path  # standard library: file paths

import pandas as pd  # tables

CACHE_LAYOUT = "posts-v1"  # mixed into every fingerprint, so a change to the cached table's layout never reuses an old cache


def source_fingerprint(source):
    """Return a short fingerprint of a data source: local files by content, web addresses by address."""
    digest = hashlib.sha256(CACHE_LAYOUT.encode("utf-8"))  # start from the layout name
    path = Path(source)
    if "://" not in source and path.is_file():
        with path.open("rb") as handle:  # hash the file's bytes in 1 MiB chunks
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return "file-" + digest.hexdigest()[:16]
    digest.update(source.encode("utf-8"))  # web address: hash the address itself
    return "url-" + digest.hexdigest()[:16]


def read_posts(source):
    """Read every sheet of the spreadsheet and stack them into one table with post_id, text and label."""
    sheets = pd.read_excel(source, sheet_name=None, dtype={"text": str})  # dict: sheet name -> table
    frames = []
    for sheet_name, sheet in sheets.items():
        absent = {"text", "label"} - set(sheet.columns)
        if absent:
            raise ValueError(f"Sheet {sheet_name!r} has no column(s) {sorted(absent)}")
        frames.append(pd.DataFrame({
            # Row 1 holds the column names, so the first post sits in row 2.
            "post_id": [f"{sheet_name}:{row}" for row in range(2, len(sheet) + 2)],
            "text": sheet["text"].fillna("").astype(str),  # cells read as empty (formula cells included) become empty strings
            "label": sheet["label"].astype("int64"),  # fails loudly if a label is missing
        }))
    posts = pd.concat(frames, ignore_index=True)  # sheets in file order, one below the other
    unknown = sorted(set(posts["label"].unique()) - set(LABEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown label values: {unknown}")
    return posts

# %%
# Load all posts: from the Parquet cache when it exists, otherwise from the source (and then write the cache).
def load_posts(source, work_dir):
    """Return the full table of posts, reading the source only when no cache for it exists."""
    cache_file = Path(work_dir) / "cache" / f"posts-{source_fingerprint(source)}.parquet"
    started = time.perf_counter()
    if cache_file.exists():
        posts = pd.read_parquet(cache_file)
        print(f"[load] cache hit: read {cache_file.name} in {time.perf_counter() - started:.1f} s; "
              "the source spreadsheet was not opened")
    else:
        posts = read_posts(source)
        seconds = time.perf_counter() - started
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_file.with_suffix(".tmp")  # write first, rename after, so a crash never leaves half a cache
        posts.to_parquet(temporary, index=False)
        temporary.replace(cache_file)
        print(f"[load] read the source spreadsheet in {seconds:.1f} s; saved cache {cache_file.name}")
    return posts


df_all = load_posts(DATA_SOURCE, WORK_DIR)  # every post; kept unchanged until the end of the notebook
if not df_all["post_id"].is_unique:
    raise ValueError("post_id is not unique")
print(f"{len(df_all):,} posts")
df_all.head()

# %%
# Record the size of each sheet and of the whole table.
sheet_sizes = df_all["post_id"].str.rsplit(":", n=1).str[0].value_counts(sort=False)  # posts per sheet, file order
brief["dataset"]["sheets"] = {name: int(count) for name, count in sheet_sizes.items()}
brief["dataset"]["n_posts"] = int(len(df_all))
brief["dataset"]["n_unique_post_ids"] = int(df_all["post_id"].nunique())
brief["dataset"]["n_empty_text"] = int((df_all["text"].str.strip() == "").sum())  # texts that are empty or blank after reading
{key: brief["dataset"][key] for key in ["sheets", "n_posts", "n_unique_post_ids", "n_empty_text"]}

# %% [markdown]
# ### Which posts mention Bitcoin?
#
# There is no coin column, and the dataset author removed cashtags such as `$BTC`, so we look at the words themselves. A post mentions Bitcoin when its text contains `bitcoin` or `btc` as a whole word, ignoring case. "btc to 50k" and "bitcoin's dominance" count; "bitcoins", "gbtc" and "btc_usd" do not. The answer goes into a True/False column, `mentions_btc`.
#
# ***
#
# ### 哪些帖子提到了比特币？
#
# 数据里没有币种列，数据集作者还去掉了 `$BTC` 这类币种标签，所以只能看字面。帖子正文里出现完整的单词 `bitcoin` 或 `btc`（不分大小写）就算提到。「btc to 50k」「bitcoin's dominance」算；「bitcoins」「gbtc」「btc_usd」不算。结果放在一列真假值 `mentions_btc` 里。

# %%
# Flag posts that mention bitcoin or btc as a whole word.
import re  # standard library: regular expressions

BTC_PATTERN = re.compile(r"\b(?:bitcoin|btc)\b", flags=re.IGNORECASE)  # whole words only, any case
df_all["mentions_btc"] = df_all["text"].str.contains(BTC_PATTERN)
print(f"{int(df_all['mentions_btc'].sum()):,} posts mention bitcoin or btc")
df_all[df_all["mentions_btc"]].head()

# %% [markdown]
# ## 2. A first look
#
# Before cleaning or modelling anything, we count what is there. All counts in this chapter use the full table `df_all`. The last cell then draws the working sample that the rest of the notebook cleans and models.
#
# ***
#
# ## 2. 看一眼
#
# 清洗和建模之前，先数一数手里有什么。本章所有的数都按全量表 `df_all` 算。最后一格再抽出工作样本，本笔记本后面清洗和建模用的都是它。

# %% [markdown]
# ### Posts per class
#
# How many posts carry each label, and what share of all posts that is.
#
# ***
#
# ### 每一类有多少条
#
# 每种标签各有多少条帖子，占全部帖子的比例是多少。

# %%
# Count posts per label and their shares.
def safe_ratio(numerator, denominator):
    """Return numerator / denominator, or None when the denominator is zero (the brief stores it as null)."""
    return None if denominator == 0 else numerator / denominator


class_counts = df_all["label"].value_counts().reindex(list(LABEL_NAMES), fill_value=0)  # labels 0, 1, 2 in order
brief["dataset"]["class_counts"] = {str(label): int(count) for label, count in class_counts.items()}
brief["dataset"]["class_shares"] = {str(label): safe_ratio(int(count), len(df_all)) for label, count in class_counts.items()}
pd.DataFrame({
    "meaning": [LABEL_NAMES[label] for label in class_counts.index],
    "posts": class_counts.values,
    "share": [brief["dataset"]["class_shares"][str(label)] for label in class_counts.index],
}, index=class_counts.index.rename("label"))

# %% [markdown]
# ### How long are the posts?
#
# Length is counted in words, where a word is any run of characters without spaces. The table gives the usual summary numbers and the chart shows the distribution, cut at the 99th percentile so that a few very long posts do not squeeze the picture.
#
# ***
#
# ### 帖子有多长？
#
# 长度按词数算，一个「词」就是一段不含空格的字符。表里是常用的几个汇总数，图是分布；图在第 99 百分位处截断，免得少数特别长的帖子把图压扁。

# %%
# Summarise post length in words and draw a histogram.
import matplotlib.pyplot as plt  # charts


def count_words(texts):
    """Return the number of whitespace-separated words in each text."""
    return texts.str.count(r"\S+")


words_per_post = count_words(df_all["text"])
length_summary = {
    "mean": float(words_per_post.mean()) if len(words_per_post) else None,
    "median": float(words_per_post.median()) if len(words_per_post) else None,
    "p90": float(words_per_post.quantile(0.90)) if len(words_per_post) else None,
    "p99": float(words_per_post.quantile(0.99)) if len(words_per_post) else None,
    "min": int(words_per_post.min()) if len(words_per_post) else None,
    "max": int(words_per_post.max()) if len(words_per_post) else None,
}
brief["dataset"]["length_words"] = length_summary
print(length_summary)

clip_at = int(length_summary["p99"] or 0) + 1  # cut the chart just above the 99th percentile
plt.figure(figsize=(8, 3))
plt.hist(words_per_post.clip(upper=clip_at), bins=range(0, clip_at + 2))
plt.xlabel(f"words per post (last bar: {clip_at} or more)")
plt.ylabel("posts")
plt.title("Post length")
plt.show()

# %% [markdown]
# ### Posts that mention Bitcoin, per class
#
# Later chapters report Bitcoin posts on their own line, so we check how many there are in each class.
#
# ***
#
# ### 每一类里提到比特币的帖子
#
# 后面的章节会把提到比特币的帖子单独列一行，所以先看每一类里各有多少。

# %%
# Count posts that mention Bitcoin, per label.
btc_counts = df_all.loc[df_all["mentions_btc"], "label"].value_counts().reindex(list(LABEL_NAMES), fill_value=0)
brief["dataset"]["btc_mentions"] = {
    "rule": BTC_PATTERN.pattern + " (ignore case)",  # the matching rule, written out
    "n_posts": int(btc_counts.sum()),
    "share": safe_ratio(int(btc_counts.sum()), len(df_all)),
    "by_class": {str(label): int(count) for label, count in btc_counts.items()},
    "share_within_class": {str(label): safe_ratio(int(btc_counts[label]), int(class_counts[label])) for label in LABEL_NAMES},
}
pd.DataFrame({
    "meaning": [LABEL_NAMES[label] for label in btc_counts.index],
    "posts mentioning bitcoin": btc_counts.values,
    "share of the class": [brief["dataset"]["btc_mentions"]["share_within_class"][str(label)] for label in btc_counts.index],
}, index=btc_counts.index.rename("label"))

# %% [markdown]
# ### Repeated texts and links
#
# The dataset card says duplicates and links were removed, so we check. A repeated text is a post whose exact text already appeared earlier in the table (empty texts count as repeats of each other). We only count them here and keep every post as it is.
#
# ***
#
# ### 重复的文本和链接
#
# 数据集说明里说已经去掉了重复帖和链接，这里核对一下。「重复」指一条帖子的文字和表里更早的某条一字不差（空文本之间也算重复）。这里只数条数，帖子一条不删。

# %%
# Count repeated texts and texts that still contain a link.
repeats_earlier = df_all["text"].duplicated(keep="first")  # True when the same text appeared before
repeated_texts = df_all.loc[df_all["text"].duplicated(keep=False), "text"]  # every copy of a repeated text
LINK_PATTERN = re.compile(r"https?://|www\.", flags=re.IGNORECASE)  # a web link
brief["dataset"]["duplicates"] = {
    "n_rows_repeating_an_earlier_text": int(repeats_earlier.sum()),
    "n_distinct_texts_repeated": int(repeated_texts.nunique()),
}
brief["dataset"]["n_posts_with_links"] = int(df_all["text"].str.contains(LINK_PATTERN).sum())
{"duplicates": brief["dataset"]["duplicates"], "n_posts_with_links": brief["dataset"]["n_posts_with_links"]}

# %% [markdown]
# ### The working sample
#
# The full table is too large to clean and model in a classroom session, so we draw a working sample, `df`, following the switches:
#
# 1. Keep the classes in use: with `N_CLASSES = 2` the neutral posts are left out; with 3 all classes stay.
# 2. With `BTC_ONLY = True`, keep only posts that mention Bitcoin.
# 3. Draw the same number of posts from every class: the smaller of `SAMPLE_PER_CLASS` and the size of the smallest remaining class, using `RANDOM_SEED`. Equal class sizes stop the models from simply learning that bullish posts are the most common.
#
# Labels keep their meaning (0 bearish, 1 neutral, 2 bullish), and every post keeps its `post_id`. `df_all` stays untouched.
#
# ***
#
# ### 工作样本
#
# 全量表太大，课堂上的时间不够清洗和建模，所以按开关抽出一份工作样本 `df`：
#
# 1. 只留用到的类别：`N_CLASSES = 2` 时去掉中性；为 3 时三类都留。
# 2. `BTC_ONLY = True` 时，只留提到比特币的帖子。
# 3. 每一类抽一样多：取 `SAMPLE_PER_CLASS` 和剩下最少那一类的条数中较小的那个，用 `RANDOM_SEED` 抽。每类一样多，模型就不会只学会「看多的帖子最多」。
#
# 标签含义不变（0 看空、1 中性、2 看多），每条帖子带着它的 `post_id`。`df_all` 原样不动。

# %%
# Draw the working sample: filter by the switches, then the same number of posts from every class.
def draw_working_sample(posts, n_classes, btc_only, per_class, seed):
    """Return (sample, available): an equal-sized random sample per class, and the class sizes before sampling."""
    labels = [0, 2] if n_classes == 2 else [0, 1, 2]  # bearish and bullish, or all three
    pool = posts[posts["label"].isin(labels)]
    if btc_only:
        pool = pool[pool["mentions_btc"]]
    available = pool["label"].value_counts().reindex(labels, fill_value=0)  # posts left in each class
    n_per_class = int(min(per_class, available.min()))
    if n_per_class == 0:
        raise ValueError(f"No posts left in at least one class after filtering: {available.to_dict()}")
    parts = [pool[pool["label"] == label].sample(n=n_per_class, random_state=seed) for label in labels]
    sample = pd.concat(parts).sample(frac=1, random_state=seed)  # shuffle so the classes are mixed
    return sample.reset_index(drop=True), available


df, available_per_class = draw_working_sample(df_all, N_CLASSES, BTC_ONLY, SAMPLE_PER_CLASS, RANDOM_SEED)
df = df[["post_id", "text", "label", "mentions_btc"]].copy()  # the working sample; df_all is unchanged
sample_counts = df["label"].value_counts().sort_index()
brief["dataset"]["sample"] = {
    "n_classes": N_CLASSES,
    "btc_only": BTC_ONLY,
    "seed": RANDOM_SEED,
    "per_class_requested": SAMPLE_PER_CLASS,
    "available_by_class": {str(label): int(count) for label, count in available_per_class.items()},
    "per_class": int(sample_counts.min()),
    "class_counts": {str(label): int(count) for label, count in sample_counts.items()},
    "n_posts": int(len(df)),
    "n_mentions_btc": int(df["mentions_btc"].sum()),
}
print(brief["dataset"]["sample"])
df.head()

# %% [markdown]
# ## 3. Cleaning
#
# Social media text is noisy. We clean the working sample step by step, one cell per step, following the course recipe: lower case, extra whitespace, HTML tags, special characters, contractions, stop words and lemmatisation. Each step is a small function that takes one text and returns one text, applied to the `clean_text` column; the original `text` column stays as it was.
#
# Two product rules change the recipe, each with its own switch:
#
# - **Emoji carry feeling** (`EMOJI_TO_WORDS`). Instead of deleting emoji with the other special characters, we first turn each one into the English words that describe it, so 🚀 becomes "rocket". With the switch off, emoji are deleted as in the course recipe.
# - **Negations carry meaning** (`KEEP_NEGATIONS`). The NLTK stop word list contains "not", "no" and "nor"; deleting them turns "not bullish" into "bullish". With the switch on, negation words (and the "n't" forms in the list) are taken out of the stop word list; "never" is not in that list, so it always stays. With the switch off, the plain NLTK list is used, as in the course recipe.
#
# A note on order: as in the course recipe, special characters are removed before contractions are expanded, so "don't" first becomes "dont". The `contractions` package also knows these apostrophe-free forms ("dont" → "do not", "cant" → "cannot"), so the negation still comes back as "not". The examples at the end of this chapter show it.
#
# At the end, `clean_texts` chains all the steps in the same order. It is the one function to call when other posts need the same cleaning, and we check that it gives exactly the same result as the step-by-step cells.
#
# ***
#
# ## 3. 清洗
#
# 社交媒体上的文字很乱。我们按课堂上的做法一步一格地清洗工作样本：小写、多余空白、HTML 标签、特殊字符、缩写展开、停用词、词形还原。每一步是一个小函数，进一段文字、出一段文字，作用在 `clean_text` 这一列上；原文 `text` 列保持不动。
#
# 有两条产品要求改动了这套做法，各有一个开关：
#
# - **表情带着情绪**（`EMOJI_TO_WORDS`）。不和其他特殊字符一起删掉，而是先把每个表情换成描述它的英文词，例如 🚀 变成「rocket」。开关关掉时，照课堂做法把表情删掉。
# - **否定词带着意思**（`KEEP_NEGATIONS`）。NLTK 的停用词表里有「not」「no」「nor」，删掉它们，「not bullish」就成了「bullish」。开关打开时，把否定词（以及表里「n't」结尾的那些词）从停用词表里拿出来；「never」本来就不在表里，所以总会留下。开关关掉时，照课堂做法用 NLTK 原样的表。
#
# 关于顺序：和课堂做法一样，先删特殊字符、后展开缩写，所以「don't」会先变成「dont」。`contractions` 包也认得这种没有撇号的写法（「dont」→「do not」，「cant」→「cannot」），否定照样还原成「not」。本章最后的例句会显示这一点。
#
# 最后，`clean_texts` 按同样的顺序把各步串起来。别的帖子需要同样的清洗时就调用它；我们会核对它的结果和逐步做出来的完全一样。

# %%
# Start the cleaned column from the original text, and remember the word counts before cleaning.
from tqdm.auto import tqdm  # progress bars

df["clean_text"] = df["text"]  # every step below rewrites this column
words_before = count_words(df["text"])  # words per post before cleaning
df[["text", "clean_text"]].head(3)

# %% [markdown]
# ### Step 1: lower case
#
# "BTC" and "btc" should count as the same word.
#
# ***
#
# ### 第 1 步：转小写
#
# 「BTC」和「btc」应该算同一个词。

# %%
# Lower case.
def to_lowercase(text):
    """Return the text in lower case."""
    return text.lower()


df["clean_text"] = [to_lowercase(text) for text in tqdm(df["clean_text"], desc="Lower case")]
df[["text", "clean_text"]].head(3)

# %% [markdown]
# ### Step 2: extra whitespace
#
# Remove spaces at both ends and squeeze runs of spaces, tabs and line breaks into one space.
#
# ***
#
# ### 第 2 步：多余空白
#
# 去掉两头的空格，把连续的空格、制表符、换行压成一个空格。

# %%
# Remove extra whitespace.
def remove_extra_whitespace(text):
    """Strip both ends and replace every run of whitespace with a single space."""
    text = text.strip()  # remove leading and trailing whitespace
    return " ".join(text.split())  # split on any whitespace, join with single spaces


df["clean_text"] = [remove_extra_whitespace(text) for text in tqdm(df["clean_text"], desc="Whitespace")]
df[["text", "clean_text"]].head(3)

# %% [markdown]
# ### Step 3: HTML tags
#
# Text copied from web pages can carry HTML markup such as `<br />`. BeautifulSoup parses the text and keeps only the readable part.
#
# ***
#
# ### 第 3 步：HTML 标签
#
# 从网页复制来的文字可能带着 `<br />` 这类 HTML 标记。BeautifulSoup 解析文字，只留下能读的部分。

# %%
# Remove HTML tags.
import warnings  # standard library: silence a harmless warning

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning  # HTML parser

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)  # short texts that look like file names are fine


def remove_html_tags(text):
    """Return the text without HTML tags."""
    return BeautifulSoup(text, "html.parser").get_text()  # parse, then keep the text only


df["clean_text"] = [remove_html_tags(text) for text in tqdm(df["clean_text"], desc="HTML tags")]
df[["text", "clean_text"]].head(3)

# %% [markdown]
# ### Step 4: emoji to words
#
# With `EMOJI_TO_WORDS = True`, each emoji is replaced by its English name from the `emoji` package, with the colons dropped and underscores turned into spaces, so 😂 becomes "face with tears of joy" rather than one glued word. Skin-tone qualifiers are dropped ("raising hands", not "raising hands medium light skin tone"). Text emoticons such as ":)" are not emoji and are not converted. With the switch off, this step leaves the text unchanged and the next step deletes the emoji.
#
# ***
#
# ### 第 4 步：表情换成词
#
# `EMOJI_TO_WORDS = True` 时，每个表情换成 `emoji` 包里它的英文名：去掉两边的冒号，下划线换成空格，所以 😂 变成「face with tears of joy」，而不是粘成一个长词。肤色说明去掉（「raising hands」，而不是「raising hands medium light skin tone」）。「:)」这类用标点拼的表情不是 emoji，不转换。开关关掉时，这一步原样返回，表情由下一步删掉。

# %%
# Turn emoji into English words (only when EMOJI_TO_WORDS is True).
import emoji  # emoji names

SKIN_TONE = re.compile(r"[_,]*(?:medium-light|medium-dark|light|medium|dark)_skin_tone")  # tone qualifiers in emoji names


def emoji_name_as_words(chars, data):
    """Return the English name of one emoji as plain lower-case words with spaces around them."""
    name = SKIN_TONE.sub("", data["en"].strip(":"))  # ":raising_hands_medium-light_skin_tone:" -> "raising_hands"
    return " " + name.replace("_", " ").lower() + " "


def emoji_to_words(text):
    """Replace every emoji with its English name when EMOJI_TO_WORDS is True; otherwise return the text unchanged."""
    if not EMOJI_TO_WORDS:
        return text
    return emoji.replace_emoji(text, replace=emoji_name_as_words)


df["clean_text"] = [emoji_to_words(text) for text in tqdm(df["clean_text"], desc="Emoji to words")]
df.loc[df["text"].map(emoji.emoji_count) > 0, ["text", "clean_text"]].head(3)  # a few posts that had emoji

# %% [markdown]
# ### Step 5: special characters
#
# Keep only letters, digits and whitespace; everything else (punctuation, symbols and any emoji still left) is deleted.
#
# ***
#
# ### 第 5 步：特殊字符
#
# 只留字母、数字和空白，其余的（标点、符号，以及还剩下的表情）都删掉。

# %%
# Remove special characters.
def remove_special_characters(text):
    """Delete every character that is not a letter, a digit or whitespace."""
    return re.sub(r"[^a-zA-Z0-9\s]", "", text)


df["clean_text"] = [remove_special_characters(text) for text in tqdm(df["clean_text"], desc="Special characters")]
df[["text", "clean_text"]].head(3)

# %% [markdown]
# ### Step 6: contractions
#
# Expand short forms into full words, for example "dont" → "do not" and "im" → "i am", with the `contractions` package.
#
# ***
#
# ### 第 6 步：展开缩写
#
# 用 `contractions` 包把缩写展开成完整的词，例如「dont」→「do not」，「im」→「i am」。

# %%
# Expand contractions.
import contractions  # contraction expansion


def expand_contractions(text):
    """Expand contractions such as "dont" into "do not"."""
    return contractions.fix(text)


df["clean_text"] = [expand_contractions(text) for text in tqdm(df["clean_text"], desc="Contractions")]
df[["text", "clean_text"]].head(3)

# %% [markdown]
# ### Step 7: stop words
#
# Remove very common words that carry little meaning ("the", "is", "and"), using the NLTK English stop word list. With `KEEP_NEGATIONS = True`, negation words are taken out of the list first, so they stay in the text. NLTK resources are downloaded only when they are not on the machine yet.
#
# ***
#
# ### 第 7 步：停用词
#
# 用 NLTK 的英文停用词表，删掉「the」「is」「and」这类很常见、意思不大的词。`KEEP_NEGATIONS = True` 时，先把否定词从表里拿出来，让它们留在文字里。NLTK 的资料只在本机还没有时才下载。

# %%
# Remove stop words, keeping negations when KEEP_NEGATIONS is True.
import nltk  # Natural Language Toolkit
from nltk.corpus import stopwords  # stop word lists
from nltk.tokenize import word_tokenize  # splits text into words


def ensure_nltk_resource(resource_path, package):
    """Download an NLTK package only when its resource is not found locally."""
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(package, quiet=True)


ensure_nltk_resource("corpora/stopwords", "stopwords")
ensure_nltk_resource("tokenizers/punkt_tab/english/", "punkt_tab")

NLTK_STOP_WORDS = frozenset(stopwords.words("english"))  # the plain list
NOT_FORMS = {word for word in NLTK_STOP_WORDS if word.endswith("n't")}  # "don't", "isn't", ...
NEGATION_WORDS = frozenset({"no", "not", "nor", "never", "ain"} | NOT_FORMS
                           | {word[:-3] + "n" for word in NOT_FORMS})  # plus their stems: "don", "isn", ...
STOP_WORDS_KEEPING_NEGATIONS = NLTK_STOP_WORDS - NEGATION_WORDS


def remove_stopwords(text):
    """Remove stop words; negation words stay when KEEP_NEGATIONS is True."""
    stop_words = STOP_WORDS_KEEPING_NEGATIONS if KEEP_NEGATIONS else NLTK_STOP_WORDS
    return " ".join(word for word in word_tokenize(text) if word.lower() not in stop_words)


print("Negation words kept when KEEP_NEGATIONS is True:", sorted(NEGATION_WORDS & NLTK_STOP_WORDS) + ["never"])
df["clean_text"] = [remove_stopwords(text) for text in tqdm(df["clean_text"], desc="Stop words")]
df[["text", "clean_text"]].head(3)

# %% [markdown]
# ### Step 8: lemmatisation
#
# Reduce each word to its dictionary form with the WordNet lemmatiser, for example "tears" → "tear". As in the course recipe the lemmatiser treats every word as a noun, so some verb forms stay as they are.
#
# ***
#
# ### 第 8 步：词形还原
#
# 用 WordNet 词形还原器把每个词还原成词典里的形式，例如「tears」→「tear」。和课堂做法一样，它把每个词都当名词处理，所以有些动词形式会原样保留。

# %%
# Lemmatise every word.
from nltk.stem import WordNetLemmatizer  # dictionary-based lemmatiser

ensure_nltk_resource("corpora/wordnet", "wordnet")
LEMMATIZER = WordNetLemmatizer()


def lemmatize_text(text):
    """Replace every word with its WordNet lemma."""
    return " ".join(LEMMATIZER.lemmatize(word) for word in word_tokenize(text))


df["clean_text"] = [lemmatize_text(text) for text in tqdm(df["clean_text"], desc="Lemmatisation")]
df[["text", "clean_text"]].head(3)

# %% [markdown]
# ### All steps in one call: `clean_texts`
#
# `clean_texts` takes a column (or list) of raw texts and returns the cleaned texts, running the same step functions in the same order with the current switches. To make sure it really matches the cells above, we run it again on the working sample's original text and compare with `clean_text`; the two must be identical.
#
# ***
#
# ### 一次做完所有步骤：`clean_texts`
#
# `clean_texts` 接收一列（或一个列表的）原文，返回清洗后的文字；它按同样的顺序、用当前的开关调用上面那些步骤函数。为了确认它和上面逐步做的真的一样，我们用它把工作样本的原文再洗一遍，和 `clean_text` 列比较，两者必须完全相同。

# %%
# Chain the steps into one function and check it against the step-by-step result.
CLEANING_STEPS = [to_lowercase, remove_extra_whitespace, remove_html_tags, emoji_to_words,
                  remove_special_characters, expand_contractions, remove_stopwords, lemmatize_text]


def clean_texts(texts):
    """Clean raw texts with the current switches; returns a pandas Series (same index when given a Series)."""
    texts = texts if isinstance(texts, pd.Series) else pd.Series(list(texts), dtype=object)
    cleaned = []
    for text in tqdm(texts, desc="clean_texts"):
        text = "" if pd.isna(text) else str(text)  # missing values clean to an empty string
        for step in CLEANING_STEPS:
            text = step(text)
        cleaned.append(text)
    return pd.Series(cleaned, index=texts.index, dtype=object)


recleaned = clean_texts(df["text"])
clean_texts_matches_steps = bool(len(recleaned) == len(df) and (recleaned == df["clean_text"]).all())
if not clean_texts_matches_steps:
    raise AssertionError("clean_texts does not reproduce the step-by-step result")
print("clean_texts gives exactly the step-by-step result:", clean_texts_matches_steps)

# %% [markdown]
# ### Before and after
#
# Average number of words per post in the working sample, before and after cleaning, and how many posts end up empty.
#
# ***
#
# ### 清洗前后对比
#
# 工作样本里每条帖子清洗前后的平均词数，以及洗完变成空的帖子有多少。

# %%
# Compare word counts before and after cleaning.
words_after = count_words(df["clean_text"])
brief["dataset"]["cleaning"] = {
    "emoji_to_words": EMOJI_TO_WORDS,
    "keep_negations": KEEP_NEGATIONS,
    "steps": [step.__name__ for step in CLEANING_STEPS],
    "avg_words_before": float(words_before.mean()) if len(df) else None,
    "avg_words_after": float(words_after.mean()) if len(df) else None,
    "n_empty_after": int((words_after == 0).sum()),
    "clean_texts_matches_steps": clean_texts_matches_steps,
}
pd.DataFrame({"average words per post": [brief["dataset"]["cleaning"]["avg_words_before"],
                                         brief["dataset"]["cleaning"]["avg_words_after"]]},
             index=["before cleaning", "after cleaning"])

# %% [markdown]
# ### Examples
#
# A few hand-written sentences with negations and emoji show what the cleaning does to them. The results go into the brief, so automated checks can confirm that, with the default switches, the negations survive and the emoji become words.
#
# ***
#
# ### 例句
#
# 几句手写的、带否定词和表情的句子，看看清洗把它们变成了什么。结果写进简报，自动检查据此确认：默认开关下否定词留下了、表情变成了词。

# %%
# Clean a few example sentences and keep the results in the brief.
CLEANING_EXAMPLES = [
    "I don't think btc will pump, not bullish at all 😭",
    "Never selling my bitcoin 🚀🚀",
    "No way it can't go higher, I won't sell 😂",
    "Nor do I care about this dip 💎🙌",
]
example_results = clean_texts(CLEANING_EXAMPLES)
brief["dataset"]["cleaning"]["examples"] = [
    {"raw": raw, "clean": clean} for raw, clean in zip(CLEANING_EXAMPLES, example_results)
]
pd.DataFrame(brief["dataset"]["cleaning"]["examples"])

# %% [markdown]
# ## Save the brief
#
# The brief is written to `WORK_DIR/brief.json` with the current UTC time. Every value that could not be computed (None, NaN or infinity) is converted to `null` before saving, never to 0 or NaN. As a safety net the JSON writer is also told to refuse NaN, so a NaN that slipped past the conversion would stop the save instead of reaching the file.
#
# ***
#
# ## 保存简报
#
# 简报连同当前的 UTC 时间写到 `WORK_DIR/brief.json`。算不出来的值（None、NaN 或无穷大）在保存前一律转成 `null`，不写 0，也不写 NaN。另设一道保险：写 JSON 时不允许 NaN，万一有 NaN 漏过了转换，保存会报错，而不是写进文件。

# %%
# Save the brief as JSON.
import json  # standard library: JSON files
import math  # standard library: finite-number check
from datetime import datetime, timezone  # standard library: current UTC time

import numpy as np  # numeric types used by pandas


def to_json_safe(value):
    """Convert numpy and pandas values to plain Python; non-finite numbers become None."""
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if value is None or value is pd.NA:
        return None
    return value


brief["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
brief_path = Path(WORK_DIR) / "brief.json"
brief_path.parent.mkdir(parents=True, exist_ok=True)
brief_path.write_text(json.dumps(to_json_safe(brief), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                      encoding="utf-8")
print(f"Saved {brief_path}")
