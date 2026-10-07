# %% [markdown]
# # Ledger Eyes: what is the crypto crowd saying?
#
# This notebook reads a public set of StockTwits crypto posts and turns what it finds into one small, fixed-format result file called the **brief**. Other programs read the brief, not the notebook.
#
# Chapters: 0 The question and the data · 1 Loading · 2 A first look · 3 Cleaning · 4 Tokenization · 5 Training, validation and test sets · 6 Vectorization · 7 Models · 8 Evaluation · 9 Topics · 10 Why · 11 The brief: format, check, save and panel.
#
# Run it from top to bottom. Every setting you may want to change is in the switch cell below.
#
# ***
#
# # 账本的眼：币圈的人在说什么？
#
# 这个笔记本读一份公开的 StockTwits 币圈帖子，把看到的东西整理成一份固定格式的小文件，叫「简报」（brief）。别的程序读的是简报，不是笔记本。
#
# 章节：0 问题与数据 · 1 加载 · 2 看一眼 · 3 清洗 · 4 分词 · 5 训练集、验证集和测试集 · 6 向量化 · 7 模型 · 8 评估 · 9 话题 · 10 为什么 · 11 简报：格式、校验、保存和面板。
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
BPE_VOCAB_SIZE = 5000  # default: 5000; or any positive integer (target vocabulary of the subword tokenizer of chapter 4; single characters are always kept, so a value below their number gives a larger vocabulary; the course example uses 30000)
VOCAB_SIZE = 1000  # default: 1000; or any positive integer (most words each vectorizer of chapter 6 keeps)
EPOCHS = 5  # default: 5; or any positive integer (passes over the training set in chapter 7)
BATCH_SIZE = 50  # default: 50; or any positive integer (posts per training step in chapter 7)
TOPIC_SCOPE = "btc"  # default: "btc" (every post that mentions bitcoin or btc); two-state, other: "all" (posts about all coins)
TOPIC_MAX_POSTS = 100000  # default: 100000; or any positive integer (most posts in the topic table of chapter 9; above it, a random draw with RANDOM_SEED)
N_TOPICS = 8  # default: 8; or any positive integer (topics of chapter 9; fewer only when the table has fewer distinct post vectors)
BULLISH_THRESHOLD = 0.7  # default: 0.7; or any number from 0 to 1, not below BEARISH_THRESHOLD (chapter 10: a probability of bullish at or above it counts as bullish)
BEARISH_THRESHOLD = 0.3  # default: 0.3; or any number from 0 to 1, not above BULLISH_THRESHOLD (chapter 10: at or below it counts as bearish; in between is unsure)
MIN_TOPIC_POSTS = 200  # default: 200; or any positive integer (chapter 10 marks a topic with fewer posts than this as too small)
MIN_RATE_GAP = 0.2  # default: 0.2; or any number from 0 to 1 (chapter 10 calibrates the bullish share only when the bullish recall plus the bearish recall of chapter 8, minus 1, is at least this and above 0)
MIN_BULLISH_BEARISH_POSTS = 100  # default: 100; or any positive integer (chapter 10 marks a row with fewer bearish + bullish posts by label than this; 1 marks none)
LAUNCH_PANEL = True  # default: True (the last cell starts the panel); or False (the panel is only built; the checks always use False)

# %% [markdown]
# ## Setup
#
# The first cell below installs the packages that Google Colab does not ship with (for example `emoji` and `contractions`); packages that are already there are left alone. All installs happen in this one cell, before anything is imported, so Colab never asks for a restart halfway through. The same cell also switches off the usage statistics and version checks of Gradio, the package behind the panel in the last chapter, before anything can import it. The cell after it prints the versions of the key packages, which helps when results differ between machines.
#
# ***
#
# ## 准备
#
# 下面第一格安装 Google Colab 没有自带的包（例如 `emoji`、`contractions`），已经有的包不动。所有安装都集中在这一格、排在一切 import 之前，这样 Colab 不会在半路要求重启。这一格还会在任何代码 import Gradio 之前，关掉它的使用统计和版本查询；Gradio 是最后一章那块面板用的包。再下一格打印关键包的版本，不同机器跑出的结果不一样时方便对照。

# %%
# Install missing packages before anything else is imported.
import importlib.util  # standard library: tells whether a package can be imported
import os  # standard library: environment variables
import subprocess  # standard library: runs pip
import sys  # standard library: the Python that runs this notebook

# Never send a Hugging Face token that may be stored on this machine, and switch off Hugging Face telemetry.
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# Gradio (the panel of the last chapter) sends usage statistics and version checks unless this is "False".
# Set before anything imports gradio, and not left to an existing setting.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

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
    "tokenizers": "tokenizers==0.22.2",
    "tensorflow": "tensorflow==2.20.0",
    "keras": "keras==3.15.1",
    "jsonschema": "jsonschema==4.26.0",
    "gradio": "gradio==6.17.3",
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
                     "fsspec", "matplotlib", "beautifulsoup4", "tqdm", "emoji", "contractions",
                     "tokenizers", "tensorflow", "keras", "jsonschema", "gradio"]:
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
for name, value in [("BTC_ONLY", BTC_ONLY), ("EMOJI_TO_WORDS", EMOJI_TO_WORDS), ("KEEP_NEGATIONS", KEEP_NEGATIONS),
                    ("LAUNCH_PANEL", LAUNCH_PANEL)]:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be True or False, got {value!r}")
for name, value in [("BPE_VOCAB_SIZE", BPE_VOCAB_SIZE), ("VOCAB_SIZE", VOCAB_SIZE), ("EPOCHS", EPOCHS),
                    ("BATCH_SIZE", BATCH_SIZE)]:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
if TOPIC_SCOPE not in ("btc", "all"):
    raise ValueError(f'TOPIC_SCOPE must be "btc" or "all", got {TOPIC_SCOPE!r}')
for name, value in [("TOPIC_MAX_POSTS", TOPIC_MAX_POSTS), ("N_TOPICS", N_TOPICS)]:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
for name, value in [("BULLISH_THRESHOLD", BULLISH_THRESHOLD), ("BEARISH_THRESHOLD", BEARISH_THRESHOLD)]:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a number from 0 to 1, got {value!r}")
if BEARISH_THRESHOLD > BULLISH_THRESHOLD:
    raise ValueError(f"BEARISH_THRESHOLD ({BEARISH_THRESHOLD!r}) must not be above BULLISH_THRESHOLD ({BULLISH_THRESHOLD!r})")
if isinstance(MIN_TOPIC_POSTS, bool) or not isinstance(MIN_TOPIC_POSTS, int) or MIN_TOPIC_POSTS < 1:
    raise ValueError(f"MIN_TOPIC_POSTS must be a positive integer, got {MIN_TOPIC_POSTS!r}")
if isinstance(MIN_RATE_GAP, bool) or not isinstance(MIN_RATE_GAP, (int, float)) or not 0 <= MIN_RATE_GAP <= 1:
    raise ValueError(f"MIN_RATE_GAP must be a number from 0 to 1, got {MIN_RATE_GAP!r}")
if isinstance(MIN_BULLISH_BEARISH_POSTS, bool) or not isinstance(MIN_BULLISH_BEARISH_POSTS, int) or MIN_BULLISH_BEARISH_POSTS < 1:
    raise ValueError(f"MIN_BULLISH_BEARISH_POSTS must be a positive integer, got {MIN_BULLISH_BEARISH_POSTS!r}")
print("Switches look fine.")

# %% [markdown]
# ## The brief
#
# The brief is the notebook's real output: one dictionary, `brief`, that the chapters fill in and chapter 11 saves as `WORK_DIR/brief.json`. Its top level is fixed: `format_version`, `generated_at`, `scope`, `dataset`, `sentiment`, `topics` and `why`. `scope` says what the results cover: the data source, the granularity (`"batch"` means one result for the whole dataset; `"daily"` is kept for dated data), which coins the working sample covers, and how many classes are used. Each section starts as `null` until its chapter fills it. A number that cannot be computed (for example a share whose denominator is zero) is written as `null`, never as 0 or NaN. The format is fixed (version 1.0, described field by field in the repository's `docs/brief-format.md`); chapter 11 checks the finished brief against it before saving.
#
# ***
#
# ## 简报
#
# 简报是这个笔记本真正的产出：一个字典 `brief`，各章往里填自己的结果，第 11 章把它存成 `WORK_DIR/brief.json`。顶层字段固定为 `format_version`、`generated_at`、`scope`、`dataset`、`sentiment`、`topics`、`why`。`scope` 说明结果覆盖什么：数据来源；粒度（`"batch"` 表示对整份数据给一个结果，`"daily"` 留给带日期的数据）；工作样本包含哪些币；用几分类。每一节先是 `null`，等对应的章节来填。算不出来的数（例如分母为 0 的比例）一律写 `null`，不写 0，也不写 NaN。格式是固定的（版本 1.0，仓库里的 `docs/brief-format.md` 逐个字段写了说明）；第 11 章保存之前，会拿它核对写好的简报。

# %%
# Create the brief. Chapters fill in their own sections; unfilled sections stay None (null in JSON).
brief = {
    "format_version": "1.0",  # the format of chapter 11's rules (docs/brief-format.md in the repository)
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
# ## 4. Tokenization: whole words or pieces of words?
#
# A model cannot read text directly; the text is first cut into tokens. This chapter compares two ways of cutting, as in the course:
#
# - **Word level with NLTK** (`word_tokenize`): every word and every punctuation mark becomes a token. The vocabulary is every distinct token ever seen, so it keeps growing with new posts, and a word never seen before is simply unknown.
# - **Subword level with the Hugging Face `tokenizers` package (BPE, byte-pair encoding)**: training starts from single characters and keeps merging the most frequent pair of neighbouring pieces into a new piece, until the vocabulary holds `BPE_VOCAB_SIZE` pieces or no pair is left to merge; the single characters themselves always stay in the vocabulary. Frequent words end up as one piece; rare words are spelled with several pieces, so almost no word is completely unknown. The price is longer token sequences.
#
# We tokenize the original text (`text`), not `clean_text`, because emoji, negations and contractions such as "don't" are still there. **This chapter is for looking only: nothing in it feeds the models.** Chapters 6 and 7 count whole words of the cleaned text instead. Because nothing here reaches a model, the BPE tokenizer is trained on the original text of the whole working sample, test posts included; this leaks nothing into the evaluation.
#
# ***
#
# ## 4. 分词：按整词切，还是按词的片段切？
#
# 模型不能直接读文字，要先把文字切成一个个「词元」（token）。本章照课堂的做法，比较两种切法：
#
# - **用 NLTK 按词切**（`word_tokenize`）：每个词、每个标点都是一个词元。词表就是见过的所有不同词元，所以帖子越多词表越大；没见过的词就是不认识。
# - **用 Hugging Face 的 `tokenizers` 包按子词切（BPE，字节对编码）**：训练从单个字符开始，反复把最常挨在一起的两块合并成一块新的片段，直到词表里的片段数达到 `BPE_VOCAB_SIZE`，或者没有可合并的了；单个字符本身始终留在词表里。常见的词最后是一整块；少见的词由几块拼出来，所以几乎没有完全不认识的词。代价是切出来的序列更长。
#
# 我们切的是原文（`text`），不是 `clean_text`，因为原文里还有表情、否定词和「don't」这类缩写。**本章只用来看，这里的结果不喂给模型。** 第 6、7 章用的是清洗后文字的整词计数。正因为这里的东西不进模型，BPE 分词器是在整个工作样本的原文上学的（测试帖也在内），不会把测试帖泄漏进评估。

# %% [markdown]
# ### Posts to look at
#
# We pick a few short posts from the working sample: some with emoji, some with a negation (not, no, never, nor or an "n't" form) and some with a contraction (an apostrophe between two letters, as in "don't", "i'm" or "it's"). They are taken in the order of the working sample, which is already shuffled with `RANDOM_SEED`. A kind that does not occur in a small sample is skipped with a note.
#
# ***
#
# ### 挑几条帖子来看
#
# 从工作样本里挑几条短帖：有的带表情，有的带否定（not、no、never、nor 或「n't」形式），有的带缩写（两个字母中间夹一个撇号，如「don't」「i'm」「it's」）。按工作样本的顺序取，而工作样本已经用 `RANDOM_SEED` 打乱过。样本小、某一种一条都没有时，跳过并说明。

# %%
# Pick a few short posts with emoji, with a negation and with a contraction.
import numpy as np  # arrays

NEGATION_IN_TEXT = re.compile(r"\b(?:not|no|never|nor)\b|n['’]t\b", flags=re.IGNORECASE)  # also inside "don't"
CONTRACTION_IN_TEXT = re.compile(r"[a-z]['’][a-z]", flags=re.IGNORECASE)  # "don't", "i'm", "it's"
EXAMPLES_PER_KIND = 2  # posts shown for each kind
SHORT_POST_WORDS = 25  # only posts up to this many words, so that the output stays readable

kind_tests = {
    "emoji": lambda text: emoji.emoji_count(text) > 0,
    "negation": lambda text: bool(NEGATION_IN_TEXT.search(text)),
    "contraction": lambda text: bool(CONTRACTION_IN_TEXT.search(text)),
}
short_posts = df[count_words(df["text"]).between(1, SHORT_POST_WORDS)]  # working-sample order is kept
token_examples, picked_ids = [], set()
for kind, has_kind in kind_tests.items():
    found = 0
    for post_id, text in zip(short_posts["post_id"], short_posts["text"]):
        if found == EXAMPLES_PER_KIND:
            break
        if post_id not in picked_ids and has_kind(text):
            token_examples.append({"kind": kind, "post_id": post_id, "text": text})
            picked_ids.add(post_id)
            found += 1
    if found == 0:
        print(f"No short post with {kind} in the working sample; this kind is skipped.")
pd.DataFrame(token_examples, columns=["kind", "post_id", "text"])

# %% [markdown]
# ### Word level with NLTK
#
# `word_tokenize` cuts every post of the working sample into words and punctuation marks. We report the vocabulary size (number of distinct tokens) and the average number of tokens per post.
#
# ***
#
# ### 用 NLTK 按词切
#
# `word_tokenize` 把工作样本里每条帖子切成词和标点。报两个数：词表大小（不同词元的个数）和每帖平均词元数。

# %%
# Cut every post of the working sample into words with NLTK.
nltk_lengths = []  # tokens per post
nltk_vocabulary = set()  # every distinct token
for text in tqdm(df["text"], desc="NLTK word_tokenize"):
    tokens = word_tokenize(text)
    nltk_lengths.append(len(tokens))
    nltk_vocabulary.update(tokens)
nltk_lengths = np.array(nltk_lengths)
print(f"NLTK word level: {len(nltk_vocabulary):,} distinct tokens; "
      f"{nltk_lengths.mean():.2f} tokens per post on average")

# %% [markdown]
# ### Subword level with BPE
#
# As in the course, the BPE tokenizer first splits the text at whitespace and punctuation (`Whitespace` pre-tokenizer), then learns its pieces with a `BpeTrainer`. The special tokens are the course's: `[UNK]` stands for a character never seen in training; `[CLS]`, `[SEP]`, `[PAD]` and `[MASK]` are reserved for later models and are counted in the vocabulary size. The vocabulary always holds every single character seen in training (the alphabet) and the five special tokens; merged pieces are added on top until it reaches `BPE_VOCAB_SIZE`. So the vocabulary can end up **smaller** than `BPE_VOCAB_SIZE` when a small sample has fewer pairs to merge, and **larger** when `BPE_VOCAB_SIZE` is below the number of single characters plus special tokens; then nothing is merged at all. The cell prints how the vocabulary splits into these three parts.
#
# ***
#
# ### 用 BPE 按子词切
#
# 和课堂一样，BPE 分词器先在空白和标点处把文字断开（`Whitespace` 预分词器），再用 `BpeTrainer` 学片段。特殊词元照课堂的：`[UNK]` 代表训练时没见过的字符；`[CLS]`、`[SEP]`、`[PAD]`、`[MASK]` 留给以后的模型用，也算进词表大小。词表里始终有训练时见过的每一个单个字符（字母表）和这五个特殊词元，合并出来的片段加在上面，直到总数达到 `BPE_VOCAB_SIZE`。所以：样本小、可合并的组合不够时，词表会**小于** `BPE_VOCAB_SIZE`；`BPE_VOCAB_SIZE` 比「单个字符数 + 特殊词元数」还小时，词表会**大于**它，而且一个片段也不合并。这一格会打印词表由这三部分各占多少。

# %%
# Train a BPE tokenizer on the working sample's original text and cut every post into pieces.
from tokenizers import Tokenizer  # Hugging Face tokenizers
from tokenizers.models import BPE  # byte-pair encoding model
from tokenizers.pre_tokenizers import Whitespace  # splits at whitespace and punctuation first
from tokenizers.trainers import BpeTrainer  # learns the BPE pieces

BPE_SPECIAL_TOKENS = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]  # the course's special tokens

bpe_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
bpe_tokenizer.pre_tokenizer = Whitespace()
bpe_trainer = BpeTrainer(special_tokens=BPE_SPECIAL_TOKENS, vocab_size=BPE_VOCAB_SIZE, show_progress=False)
bpe_tokenizer.train_from_iterator(df["text"].tolist(), bpe_trainer)  # display only: never used by a model
bpe_lengths = np.array([len(encoding.ids) for encoding in bpe_tokenizer.encode_batch(df["text"].tolist())])
bpe_vocabulary = bpe_tokenizer.get_vocab()  # piece -> id
bpe_single_characters = sum(1 for piece in bpe_vocabulary if len(piece) == 1)  # the alphabet, always kept
bpe_merged_pieces = len(bpe_vocabulary) - bpe_single_characters - len(BPE_SPECIAL_TOKENS)  # learned by merging
print(f"BPE subword level: {len(bpe_vocabulary):,} pieces in the vocabulary ({BPE_VOCAB_SIZE:,} requested) = "
      f"{bpe_single_characters:,} single characters + {len(BPE_SPECIAL_TOKENS)} special tokens + "
      f"{bpe_merged_pieces:,} merged pieces; {bpe_lengths.mean():.2f} tokens per post on average")

# %% [markdown]
# ### Side by side
#
# The chosen posts cut both ways. Tokens are separated by " | ". Look at how each way handles the emoji, "n't" and the apostrophes, and at rare words that BPE spells with several pieces.
#
# ***
#
# ### 并排对比
#
# 把挑出的帖子用两种方法各切一遍，词元之间用「 | 」隔开。看看两种切法各自怎么处理表情、「n't」和撇号，以及 BPE 怎么用几块片段拼出少见的词。

# %%
# Show the chosen posts cut both ways.
for example in token_examples:
    words = word_tokenize(example["text"])
    pieces = bpe_tokenizer.encode(example["text"]).tokens
    print(f"[{example['kind']}] {example['text']}")
    print(f"   NLTK ({len(words)} tokens): {' | '.join(words)}")
    print(f"   BPE  ({len(pieces)} tokens): {' | '.join(pieces)}")
    print()

# %% [markdown]
# ### Into the brief
#
# The `sentiment` section of the brief starts here, with the two vocabulary sizes and average lengths. The numbers describe the tokenizers only; no model uses them.
#
# ***
#
# ### 写进简报
#
# 简报的 `sentiment` 节从这里开始，先写两种切法的词表大小和平均长度。这些数只描述分词器，没有模型用到它们。

# %%
# Start the sentiment section of the brief with the tokenization comparison.
brief["sentiment"] = {
    "tokenization": {
        "used_by_models": False,  # display only
        "text": "original text of the working sample",
        "n_posts": int(len(df)),
        "nltk_word_tokenize": {
            "vocab_size": len(nltk_vocabulary),
            "avg_tokens_per_post": float(nltk_lengths.mean()) if len(nltk_lengths) else None,
        },
        "bpe": {
            "vocab_size": len(bpe_vocabulary),
            "vocab_size_requested": BPE_VOCAB_SIZE,
            "n_special_tokens": len(BPE_SPECIAL_TOKENS),
            "n_single_characters": bpe_single_characters,  # every character seen in training; always kept
            "n_merged_pieces": bpe_merged_pieces,
            "trained_on": "original text of the whole working sample",
            "avg_tokens_per_post": float(bpe_lengths.mean()) if len(bpe_lengths) else None,
        },
        "examples": [{"kind": example["kind"], "post_id": example["post_id"]} for example in token_examples],
    },
}
brief["sentiment"]["tokenization"]

# %% [markdown]
# ## 5. Training, validation and test sets
#
# From here on we build the bullish-or-bearish models. The working sample is split into three parts, as in the course:
#
# - **training set (80%)**: the models learn from it, and the vectorizers of chapter 6 learn their vocabulary from it alone;
# - **validation set (10%)**: watched during training, and used once at the end of chapter 7 to choose the model that later chapters use;
# - **test set (10%)**: kept aside until chapter 8 and used only for the final evaluation, so that the scores describe posts the models have never seen.
#
# As in the course, the sample is first split 80 / 20, and the 20% part is then split in half. Both splits are **stratified** by label, so every part keeps the class shares of the working sample, and both use `RANDOM_SEED`, so the same switches give the same parts on every run. Every class needs at least 10 posts in the working sample for each part to get at least one post of it; with fewer, the cell stops with a clear message.
#
# ***
#
# ## 5. 训练集、验证集和测试集
#
# 从这里开始做「看多还是看空」的模型。照课堂的做法，把工作样本切成三份：
#
# - **训练集（80%）**：模型从它学；第 6 章的向量化器也只在它上面学词表；
# - **验证集（10%）**：训练时用来盯着看，第 7 章最后再用它一次，选出后面章节要用的模型；
# - **测试集（10%）**：一直放到第 8 章，只用来做最后的评估，这样分数说的是模型从没见过的帖子。
#
# 和课堂一样，先按 80 / 20 切，再把 20% 那份对半切。两次切分都按标签**分层**，每一份保持工作样本里的类别占比；两次都用 `RANDOM_SEED`，开关不变时每次切出来都一样。工作样本里每一类至少要有 10 条，每一份才都能分到这一类的帖子；不够时这一格会停下并说明原因。

# %%
# Split the working sample into training, validation and test sets, stratified by label.
from sklearn.model_selection import train_test_split  # random splits

CLASS_LABELS = [0, 2] if N_CLASSES == 2 else [0, 1, 2]  # labels in use; also the column order of every class table below
SPLIT_FRACTIONS = {"train": 0.8, "validation": 0.1, "test": 0.1}
MIN_POSTS_PER_CLASS = 10  # fewest posts per class that still put every class into every part

smallest_class = int(df["label"].value_counts().reindex(CLASS_LABELS, fill_value=0).min())
if smallest_class < MIN_POSTS_PER_CLASS:
    raise ValueError(f"The working sample has only {smallest_class} posts in its smallest class; "
                     f"at least {MIN_POSTS_PER_CLASS} are needed to split it. Raise SAMPLE_PER_CLASS "
                     "or use a larger data source.")
train_df, rest_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED, stratify=df["label"])  # 80 / 20
val_df, test_df = train_test_split(rest_df, test_size=0.5, random_state=RANDOM_SEED,
                                   stratify=rest_df["label"])  # the 20% part, half and half
print(f"Training set: {len(train_df):,} posts; validation set: {len(val_df):,}; test set: {len(test_df):,}")

# %% [markdown]
# ### Class counts and shares in each part
#
# The three parts should show (almost) the same class shares as the working sample. The last column counts the posts that mention Bitcoin, which chapter 8 reports separately.
#
# ***
#
# ### 每一份里各类的条数和占比
#
# 三份的类别占比应当和工作样本（几乎）一样。最后一列是提到比特币的帖子条数，第 8 章会把它们单独报一行。

# %%
# Count posts per class in each part and record the split in the brief.
split_parts = {"train": train_df, "validation": val_df, "test": test_df}
split_counts = pd.DataFrame({name: part["label"].value_counts().reindex(CLASS_LABELS, fill_value=0)
                             for name, part in split_parts.items()}).T  # one row per part, one column per label
brief["sentiment"]["classes"] = [str(label) for label in CLASS_LABELS]
brief["sentiment"]["split"] = {
    "seed": RANDOM_SEED,
    "fractions": SPLIT_FRACTIONS,
    "stratified_by": "label",
    "n_posts": {name: int(len(part)) for name, part in split_parts.items()},
    "class_counts": {name: {str(label): int(split_counts.loc[name, label]) for label in CLASS_LABELS}
                     for name in split_parts},
    "class_shares": {name: {str(label): safe_ratio(int(split_counts.loc[name, label]), len(part))
                            for label in CLASS_LABELS} for name, part in split_parts.items()},
    "n_mentions_btc": {name: int(part["mentions_btc"].sum()) for name, part in split_parts.items()},
}
split_table = split_counts.rename(columns=LABEL_NAMES)
for label in CLASS_LABELS:
    split_table[f"{LABEL_NAMES[label]} share"] = [brief["sentiment"]["split"]["class_shares"][name][str(label)]
                                                  for name in split_parts]
split_table["posts"] = [len(part) for part in split_parts.values()]
split_table["mention bitcoin"] = [brief["sentiment"]["split"]["n_mentions_btc"][name] for name in split_parts]
split_table

# %% [markdown]
# ## 6. Vectorization: from cleaned text to numbers
#
# A model needs numbers, so every cleaned post (`clean_text`) becomes a vector with one position per word of a fixed vocabulary. We build two kinds of vectors, as in the course:
#
# - **Word counts** (`CountVectorizer`): each position holds how often that word occurs in the post.
# - **TF-IDF** (`TfidfVectorizer`): each count is weighted by how rare the word is across posts, so words that appear almost everywhere count less.
#
# Both keep the `VOCAB_SIZE` most frequent words (default 1000, as in the course). **Each vectorizer learns its vocabulary (and, for TF-IDF, its word weights) from the training set only**; the validation and test sets are only transformed with what was learned. Words that occur only outside the training set are therefore ignored, exactly as they would be for a brand-new post.
#
# **Memory.** The course turns the count matrix into an ordinary (dense) table with `.toarray()`. With the full working sample that table would hold 100,000 rows × 1,000 columns, almost all zeros. We keep both kinds of vectors as sparse matrices, which store only the non-zero cells, in 32-bit floats. Keras takes the sparse matrices as they are and turns only one batch at a time (`BATCH_SIZE` posts) into an ordinary table while training. The cell prints how much memory the sparse matrices take next to the size a dense table would have.
#
# ***
#
# ## 6. 向量化：把清洗后的文字变成数
#
# 模型要的是数，所以每条清洗后的帖子（`clean_text`）都变成一个向量，词表里每个词占一个位置。照课堂的做法做两种向量：
#
# - **词数统计**（`CountVectorizer`）：每个位置是这个词在帖子里出现了几次。
# - **TF-IDF**（`TfidfVectorizer`）：在次数上再乘一个权重，一个词在越多帖子里出现，权重越低，几乎处处都有的词就不那么重要。
#
# 两种都只留最常见的 `VOCAB_SIZE` 个词（默认 1000，和课堂一样）。**两个向量化器都只在训练集上学词表（TF-IDF 还要学每个词的权重）**；验证集和测试集只用学好的东西去转换。只在训练集以外出现的词因此被忽略，和对待一条全新帖子完全一样。
#
# **内存。** 课堂上用 `.toarray()` 把词数矩阵变成普通（稠密）的表。用完整的工作样本时，这张表有 100,000 行 × 1,000 列，几乎全是 0。我们把两种向量都存成稀疏矩阵（只记非零的格子），用 32 位浮点数。Keras 直接收下稀疏矩阵，训练时每次只把一批（`BATCH_SIZE` 条）变成普通的表。这一格会打印稀疏矩阵占了多少内存，并和稠密表的大小对照。

# %%
# Word counts: learn the vocabulary on the training set, then transform all three parts.
from sklearn.feature_extraction.text import CountVectorizer  # word counts

count_vectorizer = CountVectorizer(max_features=VOCAB_SIZE, dtype=np.float32)  # the most frequent VOCAB_SIZE words
X_train_count = count_vectorizer.fit_transform(train_df["clean_text"])  # learns the vocabulary: training set only
X_val_count = count_vectorizer.transform(val_df["clean_text"])  # transform only
X_test_count = count_vectorizer.transform(test_df["clean_text"])  # transform only
for matrix in (X_train_count, X_val_count, X_test_count):
    matrix.sort_indices()  # TensorFlow needs the non-zero cells of every row in column order
print(f"Word counts: {len(count_vectorizer.vocabulary_):,} words; training matrix {X_train_count.shape}")

# %%
# TF-IDF: learn the vocabulary and the word weights on the training set, then transform all three parts.
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF weights

tfidf_vectorizer = TfidfVectorizer(max_features=VOCAB_SIZE, dtype=np.float32)  # the most frequent VOCAB_SIZE words
X_train_tfidf = tfidf_vectorizer.fit_transform(train_df["clean_text"])  # learns vocabulary and weights: training set only
X_val_tfidf = tfidf_vectorizer.transform(val_df["clean_text"])  # transform only
X_test_tfidf = tfidf_vectorizer.transform(test_df["clean_text"])  # transform only
for matrix in (X_train_tfidf, X_val_tfidf, X_test_tfidf):
    matrix.sort_indices()  # TensorFlow needs the non-zero cells of every row in column order
print(f"TF-IDF: {len(tfidf_vectorizer.vocabulary_):,} words; training matrix {X_train_tfidf.shape}")

# %% [markdown]
# ### What the vectors look like
#
# For each kind: vocabulary size, matrix shapes, the share of cells that are not zero, and the memory the sparse matrices use compared with dense 32-bit tables. Every count is recorded in the brief, together with the number of training posts the vectorizers learned from.
#
# ***
#
# ### 向量长什么样
#
# 每种向量各报：词表大小、矩阵形状、非零格子的占比，以及稀疏矩阵实际占的内存和 32 位稠密表的对照。这些数都写进简报，连同向量化器是在多少条训练帖上学的。

# %%
# Summarise both kinds of vectors and record them in the brief.
vectorizers = {"count": count_vectorizer, "tfidf": tfidf_vectorizer}  # name -> fitted vectorizer
features = {  # name -> part -> sparse matrix
    "count": {"train": X_train_count, "validation": X_val_count, "test": X_test_count},
    "tfidf": {"train": X_train_tfidf, "validation": X_val_tfidf, "test": X_test_tfidf},
}


def sparse_megabytes(matrix):
    """Memory of a sparse CSR matrix in megabytes (values, column indices and row pointers)."""
    return (matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes) / 1e6


vector_rows = []
for name, parts in features.items():
    cells = sum(matrix.shape[0] * matrix.shape[1] for matrix in parts.values())
    vector_rows.append({
        "vectors": name,
        "vocabulary": len(vectorizers[name].vocabulary_),
        "training shape": parts["train"].shape,
        "non-zero share": safe_ratio(sum(matrix.nnz for matrix in parts.values()), cells),
        "sparse MB": round(sum(sparse_megabytes(matrix) for matrix in parts.values()), 2),
        "dense float32 MB": round(cells * 4 / 1e6, 2),
    })
brief["sentiment"]["vectorizers"] = {
    "vocab_size_requested": VOCAB_SIZE,
    "fitted_on": "train",
    "fitted_on_n_posts": int(X_train_count.shape[0]),
    "text": "clean_text",
    "storage": "sparse float32",
    **{name: {"vocab_size": len(vectorizer.vocabulary_)} for name, vectorizer in vectorizers.items()},
}
pd.DataFrame(vector_rows).set_index("vectors")

# %% [markdown]
# ## 7. Models: a one-layer network
#
# As in the course, the model is a neural network with a single dense layer, built with TensorFlow/Keras: every word of the vocabulary gets one weight per output, and the weighted sum goes through an activation function. We train one network on the word counts and one on the TF-IDF vectors.
#
# **Labels inside the network.** The data keeps its labels (0 bearish, 1 neutral, 2 bullish). The network learns positions instead:
#
# - **Two classes** (`N_CLASSES = 2`): bearish → 0, bullish → 1. The output layer has one unit with a sigmoid activation, which gives the probability of bullish; the loss is binary cross-entropy.
# - **Three classes** (`N_CLASSES = 3`): the labels 0, 1, 2 stay as they are. The output layer has three units with a softmax activation, which gives one probability per class; the loss is sparse categorical cross-entropy.
#
# **Training settings** follow the course: `EPOCHS` = 5 passes over the training set, `BATCH_SIZE` = 50 posts per step, the Adam optimiser. The validation set is only watched during training (its loss and accuracy are printed after every pass); it does not change the weights. Before each network is built, `RANDOM_SEED` fixes the random numbers of Python, NumPy and TensorFlow, so the starting weights and the order of the batches repeat; small differences between runs and machines are still possible, because TensorFlow does not promise bit-identical training.
#
# ***
#
# ## 7. 模型：一层网络
#
# 和课堂一样，模型是只有一个全连接层的神经网络，用 TensorFlow/Keras 搭：词表里每个词对每个输出有一个权重，加权求和后过一个激活函数。词数统计和 TF-IDF 两种向量各训一个网络。
#
# **网络内部的标签。** 数据里的标签含义不变（0 看空、1 中性、2 看多），网络学的是位置号：
#
# - **两类**（`N_CLASSES = 2`）：看空 → 0，看多 → 1。输出层一个单元、sigmoid 激活，输出的是「看多的概率」；损失函数是二元交叉熵。
# - **三类**（`N_CLASSES = 3`）：标签 0、1、2 原样不变。输出层三个单元、softmax 激活，每一类给一个概率；损失函数是稀疏分类交叉熵。
#
# **训练设置**照课堂：`EPOCHS` = 5 轮（把训练集过 5 遍），`BATCH_SIZE` = 每步 50 条，优化器 Adam。验证集在训练时只是盯着看（每过一遍打印它的损失和准确率），不改权重。每搭一个网络之前，先用 `RANDOM_SEED` 固定 Python、NumPy 和 TensorFlow 的随机数，所以初始权重和分批顺序都能重复；但 TensorFlow 不保证训练结果逐位相同，不同次运行、不同机器之间仍可能有细小差别。

# %%
# Helpers: label encoding, the one-layer network and class probabilities.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow's start-up information messages
import tensorflow as tf  # neural networks (Keras)

MODEL_TARGET = {label: position for position, label in enumerate(CLASS_LABELS)}  # two classes: {0: 0, 2: 1}


def to_targets(labels):
    """Turn label values into what the network learns: positions in CLASS_LABELS."""
    targets = np.array([MODEL_TARGET[label] for label in labels])
    return targets.astype("float32") if N_CLASSES == 2 else targets.astype("int64")


def build_network(n_features):
    """One dense layer: 1 sigmoid unit for two classes, one softmax unit per class for three."""
    two_classes = N_CLASSES == 2
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(n_features,)),  # one input per vocabulary word
        tf.keras.layers.Dense(1 if two_classes else N_CLASSES, activation="sigmoid" if two_classes else "softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="binary_crossentropy" if two_classes else "sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def predict_proba(model, matrix):
    """Class probabilities, one column per label in CLASS_LABELS (bearish first, bullish last)."""
    if matrix.shape[0] == 0:
        return np.empty((0, len(CLASS_LABELS)), dtype=np.float32)
    if hasattr(matrix, "has_sorted_indices") and not matrix.has_sorted_indices:
        matrix = matrix.sorted_indices()  # TensorFlow needs the non-zero cells of every row in column order
    output = model.predict(matrix, batch_size=1024, verbose=0)
    if output.shape[1] == 1:  # two classes: the single output is the probability of bullish
        output = np.hstack([1.0 - output, output])
    return output


def predict_labels(model, matrix):
    """Predicted label values: the class with the highest probability (a tie goes to the earlier column)."""
    return np.asarray(CLASS_LABELS)[predict_proba(model, matrix).argmax(axis=1)]

# %% [markdown]
# ### Network on word counts
#
# ***
#
# ### 用词数统计训练的网络

# %%
# Train the network on the word-count vectors.
tf.keras.utils.set_random_seed(RANDOM_SEED)  # fixes Python, NumPy and TensorFlow random numbers
model_count = build_network(X_train_count.shape[1])
history_count = model_count.fit(X_train_count, to_targets(train_df["label"]), epochs=EPOCHS, batch_size=BATCH_SIZE,
                                validation_data=(X_val_count, to_targets(val_df["label"])), verbose=2)

# %% [markdown]
# ### Network on TF-IDF vectors
#
# ***
#
# ### 用 TF-IDF 训练的网络

# %%
# Train the network on the TF-IDF vectors.
tf.keras.utils.set_random_seed(RANDOM_SEED)  # same seed, so both networks start alike
model_tfidf = build_network(X_train_tfidf.shape[1])
history_tfidf = model_tfidf.fit(X_train_tfidf, to_targets(train_df["label"]), epochs=EPOCHS, batch_size=BATCH_SIZE,
                                validation_data=(X_val_tfidf, to_targets(val_df["label"])), verbose=2)

# %% [markdown]
# ### Choosing one model with the validation set
#
# Later chapters need one model to score posts. We choose it by **macro-F1 on the validation set** (chapter 8 explains the measure); the test set plays no part in the choice. If both scores are equal, the word-count network is chosen. Both validation scores and the choice go into the brief.
#
# ***
#
# ### 用验证集选一个模型
#
# 后面的章节要用一个模型给帖子打分。按**验证集上的 macro-F1** 来选（这个指标第 8 章解释）；测试集不参与选择。两个分数一样时选词数统计的网络。两个验证分数和选择结果都写进简报。

# %%
# Score both networks on the validation set and choose the better one.
from sklearn.metrics import precision_recall_fscore_support  # per-class scores


def class_scores(true_labels, predicted_labels):
    """Per-class precision, recall, F1 and support for CLASS_LABELS; NaN where a score cannot be computed."""
    return precision_recall_fscore_support(true_labels, predicted_labels, labels=CLASS_LABELS, zero_division=np.nan)


def macro_f1(true_labels, predicted_labels):
    """Mean of the per-class F1 scores, leaving out classes whose F1 cannot be computed; NaN when none can."""
    f1 = class_scores(true_labels, predicted_labels)[2]
    return float(np.nanmean(f1)) if np.isfinite(f1).any() else float("nan")


models = {  # model name -> fitted network and the vectors it reads
    "count_network": {"model": model_count, "vectors": "count", "history": history_count},
    "tfidf_network": {"model": model_tfidf, "vectors": "tfidf", "history": history_tfidf},
}
validation_macro_f1 = {
    name: macro_f1(val_df["label"].to_numpy(), predict_labels(entry["model"], features[entry["vectors"]]["validation"]))
    for name, entry in models.items()
}
selected_model_name = max(validation_macro_f1, key=lambda name: np.nan_to_num(validation_macro_f1[name], nan=-1.0))
selected_model = models[selected_model_name]["model"]
selected_vectorizer = vectorizers[models[selected_model_name]["vectors"]]

brief["sentiment"]["models"] = list(models)
brief["sentiment"]["training"] = {
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "optimizer": "adam",
    "seed": RANDOM_SEED,
    "output_layer": "1 unit, sigmoid" if N_CLASSES == 2 else f"{N_CLASSES} units, softmax",
    "loss": "binary_crossentropy" if N_CLASSES == 2 else "sparse_categorical_crossentropy",
    "target_of_label": {str(label): position for label, position in MODEL_TARGET.items()},
    "history": {name: {key: [float(value) for value in values] for key, values in entry["history"].history.items()}
                for name, entry in models.items()},
}
brief["sentiment"]["selection"] = {
    "by": "macro-F1 on the validation set",
    "validation_macro_f1": validation_macro_f1,
    "selected": selected_model_name,
}
print("Validation macro-F1:", {name: round(score, 4) for name, score in validation_macro_f1.items()})
print("Chosen for later chapters:", selected_model_name)

# %% [markdown]
# ### The scoring function for later chapters
#
# `sentiment_proba(cleaned_texts)` gives the chosen model's class probabilities for any batch of cleaned texts, not only posts of the working sample. Clean raw posts with `clean_texts` first. It returns a NumPy array with one row per text and one column per class in use; the column order is fixed and is stored in `SENTIMENT_PROBA_LABELS`:
#
# - two classes: columns [bearish, bullish] (labels 0, 2);
# - three classes: columns [bearish, neutral, bullish] (labels 0, 1, 2).
#
# The probability of bullish is always the last column, `BULLISH_COLUMN`. Every row adds up to 1.
#
# `train_post_ids` is the set of `post_id`s of the training set. A post whose id is not in it was never seen by the models during training.
#
# ***
#
# ### 给后面章节用的打分函数
#
# `sentiment_proba(cleaned_texts)` 用选中的模型，给任意一批清洗后的文字算出每一类的概率，不限于工作样本里的帖子。原始帖子先用 `clean_texts` 洗。它返回一个 NumPy 数组，每条文字一行、每个用到的类别一列；列的顺序固定，记在 `SENTIMENT_PROBA_LABELS` 里：
#
# - 两类：[看空、看多]（标签 0、2）；
# - 三类：[看空、中性、看多]（标签 0、1、2）。
#
# 「看多的概率」永远是最后一列，即 `BULLISH_COLUMN`。每一行加起来等于 1。
#
# `train_post_ids` 是训练集帖子的 `post_id` 集合。编号不在里面的帖子，模型训练时从没见过。

# %%
# The scoring function and the ids of the training posts.
SENTIMENT_PROBA_LABELS = list(CLASS_LABELS)  # label of each column returned by sentiment_proba
BULLISH_COLUMN = len(SENTIMENT_PROBA_LABELS) - 1  # the probability of bullish is the last column


def sentiment_proba(cleaned_texts, model=selected_model, vectorizer=selected_vectorizer):
    """Class probabilities from the chosen model; shape (number of texts, number of classes in use).

    Columns follow SENTIMENT_PROBA_LABELS: [bearish, bullish] for two classes, [bearish, neutral, bullish]
    for three. Input: cleaned texts (for example the output of clean_texts); a missing value counts as "".
    """
    texts = ["" if not isinstance(text, str) and pd.isna(text) else str(text) for text in cleaned_texts]
    return predict_proba(model, vectorizer.transform(texts))


train_post_ids = frozenset(train_df["post_id"])  # posts the models learned from
brief["sentiment"]["scoring"] = {
    "function": "sentiment_proba",
    "input": "cleaned texts, for example the output of clean_texts",
    "output": "array of shape (number of texts, number of classes in use); every row adds up to 1",
    "column_labels": [str(label) for label in SENTIMENT_PROBA_LABELS],
    "bullish_column": BULLISH_COLUMN,
    "model": selected_model_name,
    "train_ids": "train_post_ids",
    "n_train_ids": len(train_post_ids),
}
demo_texts = ["btc to the moon 🚀🚀", "not bullish at all, selling everything"]
pd.DataFrame(sentiment_proba(clean_texts(demo_texts)), index=demo_texts,
             columns=[LABEL_NAMES[label] for label in SENTIMENT_PROBA_LABELS])

# %% [markdown]
# ## 8. Evaluation on the test set
#
# Until now the test set was only split off (chapter 5) and turned into vectors with the vocabulary learned on the training set (chapter 6). Only now are its posts labelled and scored. The measures, for each class:
#
# - **precision**: of the posts the model put into this class, the share that really belong to it;
# - **recall**: of the posts that really belong to this class, the share the model found;
# - **F1**: the harmonic mean of precision and recall.
#
# **Macro-F1** is the plain average of the per-class F1 scores, so a large class cannot hide a small one. **Accuracy** is the share of posts the model got right. A **confusion matrix** has one row per true class and one column per predicted class; each cell counts posts.
#
# Two baselines show what "no skill" scores:
#
# - **always majority**: always guess the most frequent class of the training set; if classes are equally frequent, the smallest label (bearish) is taken;
# - **uniform random**: for every post, pick one of the classes at random with equal chances, with `RANDOM_SEED`, so the guesses are the same on every run.
#
# **The bar:** each model's macro-F1 on all test posts must be higher than that of both baselines. Every model and every baseline gets two rows: one on all test posts and one on only the test posts that mention Bitcoin.
#
# A score that cannot be computed (for example the precision of a class the model never predicts, or any score on an empty subset) is left empty: "—" in the tables and `null` in the brief, never 0. Macro-F1 averages only the classes whose F1 can be computed.
#
# ***
#
# ## 8. 在测试集上评估
#
# 在这之前，测试集只是在第 5 章被切出来、在第 6 章用训练集上学好的词表转成了向量。到这里才第一次给测试帖判类别、算分数。每一类的指标：
#
# - **precision（精确率）**：模型判成这一类的帖子里，真是这一类的占多少；
# - **recall（召回率）**：真是这一类的帖子里，被模型找出来的占多少；
# - **F1**：精确率和召回率的调和平均。
#
# **Macro-F1** 是各类 F1 的简单平均，条数多的类盖不住条数少的类。**Accuracy（准确率）**是判对的帖子占多少。**混淆矩阵**每一行是一个真实类别，每一列是一个判出的类别，格子里是帖子条数。
#
# 两条对照线，表示「毫无本事」能拿多少分：
#
# - **永远猜多数**：永远猜训练集里最多的那一类；各类一样多时，取标签最小的那一类（看空）；
# - **随机乱猜**：每条帖子都从各类里等概率随便挑一个，用 `RANDOM_SEED`，所以每次挑得一样。
#
# **过线标准：** 每个模型在全部测试帖上的 macro-F1，必须同时高于这两条对照线。每个模型、每条对照线都有两行：一行是全部测试帖，一行只看提到比特币的测试帖。
#
# 算不出来的分数（比如模型从来没判过的那一类的精确率，或者空子集上的任何分数）一律留空：表里显示「—」，简报里写 `null`，绝不写 0。Macro-F1 只平均那些算得出 F1 的类别。

# %% [markdown]
# ### Test posts whose text also appears in the training set
#
# The dataset author removed duplicate posts, but a few exact repeats remain (chapter 2). A test post with exactly the same text as a training post is easier for the models than a truly new one. We count such test posts, by original text and by cleaned text, and record the numbers in the brief; we keep them in the test set and do not remove anything.
#
# ***
#
# ### 文字和训练集里某条一模一样的测试帖
#
# 数据集作者去过重，但还剩少量一字不差的重复（第 2 章）。和某条训练帖文字完全相同的测试帖，对模型来说比真正的新帖容易。这里按原文和按清洗后的文字各数一遍，写进简报；这些帖子照样留在测试集里，一条不删。

# %%
# Count test posts whose text also appears in the training set (recorded only, nothing is removed).
test_texts_also_in_train = {
    "same_text": int(test_df["text"].isin(set(train_df["text"])).sum()),
    "same_clean_text": int(test_df["clean_text"].isin(set(train_df["clean_text"])).sum()),
}
brief["sentiment"]["test_texts_also_in_train"] = test_texts_also_in_train
print(f"Test posts: {len(test_df):,}; with the same original text as a training post: "
      f"{test_texts_also_in_train['same_text']:,}; with the same cleaned text: {test_texts_also_in_train['same_clean_text']:,}")

# %% [markdown]
# ### Predictions on the test set
#
# Both networks and both baselines label every test post. The baselines ignore the text, so they get a dummy column of zeros as input; scikit-learn's `DummyClassifier` does the guessing, and we check that "always majority" really guesses the most frequent training label.
#
# ***
#
# ### 在测试集上做判断
#
# 两个网络和两条对照线都给每条测试帖判一个类别。对照线不看文字，所以只喂给它一列全是 0 的假输入；猜的活由 scikit-learn 的 `DummyClassifier` 干。我们还核对「永远猜多数」猜的确实是训练集里最多的那一类。

# %%
# Label the test set with both networks and both baselines.
from sklearn.dummy import DummyClassifier  # baselines that ignore the input

BASELINES = ["always_majority", "uniform_random"]
y_test = test_df["label"].to_numpy()
majority_label = int(train_df["label"].value_counts().reindex(CLASS_LABELS, fill_value=0).idxmax())  # tie: smallest label
no_text_train = np.zeros((len(train_df), 1))  # the baselines ignore the text
no_text_test = np.zeros((len(test_df), 1))
always_majority = DummyClassifier(strategy="most_frequent").fit(no_text_train, train_df["label"])
uniform_random = DummyClassifier(strategy="uniform", random_state=RANDOM_SEED).fit(no_text_train, train_df["label"])

test_predictions = {name: predict_labels(entry["model"], features[entry["vectors"]]["test"])
                    for name, entry in models.items()}  # model name -> predicted labels
test_predictions["always_majority"] = always_majority.predict(no_text_test)
test_predictions["uniform_random"] = uniform_random.predict(no_text_test)
if not (test_predictions["always_majority"] == majority_label).all():
    raise AssertionError("the majority baseline does not guess the most frequent training label")
print(f"Most frequent label in the training set: {majority_label} ({LABEL_NAMES[majority_label]})")
pd.DataFrame({name: pd.Series(predicted).value_counts().reindex(CLASS_LABELS, fill_value=0)
              for name, predicted in test_predictions.items()}).rename(index=LABEL_NAMES)  # guesses per class

# %% [markdown]
# ### The evaluation table
#
# One row per model or baseline and per subset: all test posts, and only the test posts that mention Bitcoin. Columns: number of posts, accuracy, precision / recall / F1 for each class, and macro-F1.
#
# ***
#
# ### 评估表
#
# 每个模型或对照线、每个子集一行：全部测试帖；只含提到比特币的测试帖。列：帖子条数、准确率、每一类的精确率／召回率／F1、macro-F1。

# %%
# Score every model and baseline on both subsets of the test set.
SUBSETS = {"all_test": "all test posts", "test_mentions_btc": "test posts that mention bitcoin or btc"}
subset_masks = {"all_test": np.ones(len(test_df), dtype=bool), "test_mentions_btc": test_df["mentions_btc"].to_numpy()}


def evaluate(true_labels, predicted_labels):
    """Accuracy, per-class precision / recall / F1 / support and macro-F1; NaN where a score cannot be computed."""
    precision, recall, f1, support = class_scores(true_labels, predicted_labels)
    return {
        "n": int(len(true_labels)),
        "accuracy": float(np.mean(true_labels == predicted_labels)) if len(true_labels) else float("nan"),
        "per_class": {str(label): {"precision": float(precision[position]), "recall": float(recall[position]),
                                   "f1": float(f1[position]), "support": int(support[position])}
                      for position, label in enumerate(CLASS_LABELS)},
        "macro_f1": macro_f1(true_labels, predicted_labels),
    }


evaluation_rows = [
    {"model": name, "kind": "baseline" if name in BASELINES else "model", "subset": subset,
     **evaluate(y_test[mask], predicted[mask])}
    for name, predicted in test_predictions.items() for subset, mask in subset_masks.items()
]
brief["sentiment"]["evaluation"] = {
    "subsets": SUBSETS,
    "baselines": BASELINES,
    "majority_label": str(majority_label),
    "rows": evaluation_rows,
}


def blank_if_nan(value):
    """Show scores with three decimals and a dash where a score cannot be computed."""
    if isinstance(value, float):
        return "—" if np.isnan(value) else round(value, 3)
    return value


pd.DataFrame([{
    "model": row["model"], "subset": row["subset"], "posts": row["n"], "accuracy": row["accuracy"],
    **{f"{LABEL_NAMES[label]} {measure}": row["per_class"][str(label)][measure]
       for label in CLASS_LABELS for measure in ("precision", "recall", "f1")},
    "macro-F1": row["macro_f1"],
} for row in evaluation_rows]).map(blank_if_nan)

# %% [markdown]
# ### Do the models beat both baselines?
#
# The bar from the start of this chapter, on all test posts.
#
# ***
#
# ### 模型是否同时胜过两条对照线？
#
# 本章开头定的过线标准，在全部测试帖上看。

# %%
# Compare each model's macro-F1 on all test posts with both baselines.
all_test_macro_f1 = {row["model"]: row["macro_f1"] for row in evaluation_rows if row["subset"] == "all_test"}
beats_baselines = {name: bool(all(all_test_macro_f1[name] > all_test_macro_f1[baseline] for baseline in BASELINES))
                   for name in models}  # False when a score is missing
brief["sentiment"]["beats_baselines"] = beats_baselines
pd.DataFrame([{"model": name, "macro-F1": all_test_macro_f1[name],
               **{baseline: all_test_macro_f1[baseline] for baseline in BASELINES},
               "beats both": beats_baselines[name]} for name in models]).set_index("model").map(blank_if_nan)

# %% [markdown]
# ### Confusion matrices
#
# One matrix per model, on all test posts: rows are the true classes, columns the predicted ones. The counts go into the brief; each matrix adds up to the number of test posts.
#
# ***
#
# ### 混淆矩阵
#
# 每个模型一张，用全部测试帖：行是真实类别，列是判出的类别。计数写进简报；每张矩阵的格子加起来等于测试帖条数。

# %%
# Draw a confusion matrix for each model and record the counts.
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix  # confusion matrices

class_names = [LABEL_NAMES[label] for label in CLASS_LABELS]
figure, axes = plt.subplots(1, len(models), figsize=(4.5 * len(models), 4))
confusion_matrices = {}
for axis, name in zip(np.atleast_1d(axes), models):
    counts = confusion_matrix(y_test, test_predictions[name], labels=CLASS_LABELS)  # rows: true, columns: predicted
    confusion_matrices[name] = {"labels": [str(label) for label in CLASS_LABELS], "rows": "true label",
                                "columns": "predicted label", "counts": counts.tolist(), "n": int(counts.sum())}
    ConfusionMatrixDisplay(counts, display_labels=class_names).plot(ax=axis, colorbar=False, values_format="d")
    axis.set_title(name)
    axis.set_xlabel("predicted")
    axis.set_ylabel("true")
plt.tight_layout()
plt.show()
brief["sentiment"]["confusion_matrices"] = confusion_matrices

# %% [markdown]
# ### Checking the scoring function
#
# Finally we make sure that `sentiment_proba`, the function later chapters use, gives exactly the labels that were evaluated above for the chosen model, that it returns one column per class and that every row adds up to 1. The test posts are only read here; nothing is trained or chosen any more.
#
# ***
#
# ### 核对打分函数
#
# 最后核对后面章节要用的 `sentiment_proba`：它给出的类别，和上面评估选中模型时用的完全一样；它每一类返回一列；每一行加起来等于 1。这里只读测试帖，不再训练、也不再做任何选择。

# %%
# Check sentiment_proba against the evaluated predictions of the chosen model.
test_proba = sentiment_proba(test_df["clean_text"])
scoring_check = {
    "n_texts": int(test_proba.shape[0]),
    "n_columns": int(test_proba.shape[1]),
    "rows_sum_to_one": bool(np.allclose(test_proba.sum(axis=1), 1.0, atol=1e-5)),
    "matches_evaluated_predictions": bool(np.array_equal(
        np.asarray(SENTIMENT_PROBA_LABELS)[test_proba.argmax(axis=1)], test_predictions[selected_model_name])),
}
brief["sentiment"]["scoring"]["check"] = scoring_check
scoring_check

# %% [markdown]
# ## 9. Topics: what are the posts about?
#
# Chapters 4 to 8 asked *bullish or bearish?* This chapter asks *about what?* It sorts posts into a few groups by meaning, called **topics**, names every topic after its most typical words, and reports how many posts each topic holds, together with a few of its posts in their original words. The data has no topic labels, so nothing here is checked against labels and nothing gets a score: whether the topics make sense is for a person to judge by reading the table at the end of the chapter.
#
# **Which posts.** Topics use a table of their own, `topic_posts`, taken from the full table `df_all` and not from the working sample. The working sample has the same number of posts in every class, which changes the class shares; topics, and the chapter that will compare bullish and bearish posts per topic, need the natural shares. With the default `TOPIC_SCOPE = "btc"` the table holds every post that mentions Bitcoin, in all three classes, neutral included; with `TOPIC_SCOPE = "all"` it holds posts about all coins. When more posts qualify than `TOPIC_MAX_POSTS`, a plain random draw with `RANDOM_SEED` keeps that many, which leaves the class shares as they are up to chance. The switches `N_CLASSES` and `BTC_ONLY` only describe the working sample and do not touch this table.
#
# **How.** The posts are cleaned with `clean_texts`, the same function as in chapter 3. A TF-IDF vectorizer of their own (not one of chapter 6's) turns them into vectors, and **KMeans** splits the vectors into `N_TOPICS` groups. Same input, same switches: same topics on every run.
#
# ***
#
# ## 9. 话题：帖子都在聊什么？
#
# 第 4～8 章问的是「看多还是看空」，这一章问「在聊什么」：把帖子按意思分成几堆，每一堆叫一个**话题**；每个话题用它最有代表性的几个词起名，报它有多少条帖子，再附几条原话。数据里没有话题标签，所以这里没有拿标签来核对的东西，也不打分；话题分得好不好，要由人读本章最后那张表来判断。
#
# **用哪些帖子。** 话题有自己的一张表 `topic_posts`，从全量表 `df_all` 里取，不从工作样本取。工作样本每一类抽得一样多，类别占比被改过；话题，以及后面要按话题比较看多看空的那一章，要的是自然的占比。默认 `TOPIC_SCOPE = "btc"`：表里是提到比特币的全部帖子，三类都要，含中性；改成 `"all"` 就是全部币种的帖子。符合条件的帖子多于 `TOPIC_MAX_POSTS` 时，用 `RANDOM_SEED` 简单随机抽到这么多条，类别占比除了随机波动之外保持原样。`N_CLASSES` 和 `BTC_ONLY` 两个开关只管工作样本，不影响这张表。
#
# **怎么分。** 帖子用 `clean_texts` 清洗，和第 3 章是同一个函数。再用一个话题专用的 TF-IDF 向量化器（不是第 6 章那两个）把它们变成向量，最后用 **KMeans** 把向量分成 `N_TOPICS` 堆。输入和开关不变，每次分出来的话题都一样。

# %% [markdown]
# ### The posts for topics
#
# Take the posts in scope from `df_all`, keep at most `TOPIC_MAX_POSTS` of them (a random draw with `RANDOM_SEED`, put back into file order), and keep the columns `post_id`, `text`, `label` and `mentions_btc`. With the default switches the table holds exactly the posts that chapter 2 counted as mentioning Bitcoin. The table below gives the posts per class and their shares.
#
# ***
#
# ### 话题用的帖子
#
# 从 `df_all` 里取出范围内的帖子，最多留 `TOPIC_MAX_POSTS` 条（用 `RANDOM_SEED` 随机抽，抽完按文件里的顺序排回去），保留 `post_id`、`text`、`label`、`mentions_btc` 四列。默认开关下，这张表正好是第 2 章数出来的提到比特币的那些帖子。下表是每一类的条数和占比。

# %%
# Build the topic table from the full table: the posts in scope, at most TOPIC_MAX_POSTS of them.
topic_pool = df_all[df_all["mentions_btc"]] if TOPIC_SCOPE == "btc" else df_all  # all three classes, natural shares
n_topic_available = len(topic_pool)
if n_topic_available > TOPIC_MAX_POSTS:
    topic_pool = topic_pool.sample(n=TOPIC_MAX_POSTS, random_state=RANDOM_SEED).sort_index()  # random draw, file order
topic_posts = topic_pool[["post_id", "text", "label", "mentions_btc"]].reset_index(drop=True).copy()  # df_all is unchanged

topic_class_counts = topic_posts["label"].value_counts().reindex(list(LABEL_NAMES), fill_value=0)  # labels 0, 1, 2
brief["topics"] = {
    "post_table": "topic_posts",  # name of the table in the notebook
    "topic_column": "topic",  # its column with the topic of every post
    "scope": TOPIC_SCOPE,
    "max_posts": TOPIC_MAX_POSTS,
    "seed": RANDOM_SEED,
    "n_available": int(n_topic_available),  # posts in scope before any draw
    "sampled": bool(n_topic_available > TOPIC_MAX_POSTS),
    "n_posts": int(len(topic_posts)),
    "n_unique_post_ids": int(topic_posts["post_id"].nunique()),
    "class_counts": {str(label): int(count) for label, count in topic_class_counts.items()},
    "class_shares": {str(label): safe_ratio(int(count), len(topic_posts)) for label, count in topic_class_counts.items()},
    "n_mentions_btc": int(topic_posts["mentions_btc"].sum()),
}
print(f"Topic table: {len(topic_posts):,} of {n_topic_available:,} posts in scope ({TOPIC_SCOPE!r})")
pd.DataFrame({
    "meaning": [LABEL_NAMES[label] for label in topic_class_counts.index],
    "posts": topic_class_counts.values,
    "share": [brief["topics"]["class_shares"][str(label)] for label in topic_class_counts.index],
}, index=topic_class_counts.index.rename("label"))

# %% [markdown]
# ### Cleaning the topic posts
#
# The topic posts go through `clean_texts`, the same function and the same switches as the working sample in chapter 3; the result is the column `clean_text`. As a check, every post that is also in the working sample must come out exactly as its `clean_text` there.
#
# ***
#
# ### 清洗话题用的帖子
#
# 话题用的帖子交给 `clean_texts` 清洗，和第 3 章洗工作样本是同一个函数、同一组开关，结果放在 `clean_text` 列。顺便核对：同时也在工作样本里的帖子，洗出来必须和工作样本里的 `clean_text` 一字不差。

# %%
# Clean the topic posts with clean_texts and compare with the working sample where the posts overlap.
topic_posts["clean_text"] = clean_texts(topic_posts["text"]).to_numpy()
overlap = topic_posts[["post_id", "clean_text"]].merge(df[["post_id", "clean_text"]], on="post_id",
                                                       suffixes=("", "_working_sample"))  # posts in both tables
same_as_working_sample = bool((overlap["clean_text"] == overlap["clean_text_working_sample"]).all())
if not same_as_working_sample:
    raise AssertionError("clean_texts gives the topic posts a different clean_text than the working sample")
brief["topics"]["cleaning"] = {
    "function": "clean_texts",
    "n_empty_after": int((count_words(topic_posts["clean_text"]) == 0).sum()),  # posts with no word left
    "n_also_in_working_sample": int(len(overlap)),
    "same_as_working_sample": same_as_working_sample,
}
print(brief["topics"]["cleaning"])
topic_posts[["text", "clean_text"]].head(3)

# %% [markdown]
# ### TF-IDF vectors for topics
#
# A TF-IDF vectorizer of its own learns its vocabulary and word weights from the whole topic table and keeps the `TOPIC_VOCAB_SIZE` most frequent words (1000, as in the course). It is separate from the two vectorizers of chapter 6, which learned from the training set only. Learning from every topic post is fine here: topics have no labels and no test set, so nothing can leak.
#
# A few words would turn up in the name of every topic without saying anything about it, so this vectorizer leaves them out (the `clean_text` column keeps them):
#
# - with `TOPIC_SCOPE = "btc"`, "bitcoin" and "btc": every post in the table contains one of them;
# - the negation words that chapter 3 keeps (not, no, nor, never and the "n't" forms): they occur in a large share of posts and carry no subject;
# - "face": it only says that an emoji showed a face ("face with tears of joy" is cleaned to "face tear joy").
#
# A post left without any word of the vocabulary gets a vector of zeros; KMeans still puts it into the topic whose centre is closest.
#
# ***
#
# ### 话题用的 TF-IDF 向量
#
# 一个话题专用的 TF-IDF 向量化器，在整张话题表上学词表和每个词的权重，只留最常见的 `TOPIC_VOCAB_SIZE` 个词（1000，和课堂一样）。它和第 6 章那两个只在训练集上学的向量化器是分开的。在全部话题帖子上学没有问题：话题没有标签，也没有测试集，不存在泄漏。
#
# 有几个词会挤进每个话题的名字，却说明不了这个话题在聊什么，所以这个向量化器不收它们（`clean_text` 列里照样留着）：
#
# - `TOPIC_SCOPE = "btc"` 时的「bitcoin」「btc」：表里每条帖子都至少有其中一个；
# - 第 3 章留下来的否定词（not、no、nor、never 以及「n't」那些形式）：出现在很大一部分帖子里，又不指向任何话题；
# - 「face」：它只说明某个表情是一张脸（「face with tears of joy」洗完是「face tear joy」）。
#
# 一个词表里的词都没有的帖子，向量全是 0；KMeans 照样把它归到中心离它最近的那个话题。

# %%
# Learn the topic vectorizer on the whole topic table, leaving out words that would sit in every topic name.
from scipy.sparse import csr_matrix  # sparse matrices
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF weights

TOPIC_VOCAB_SIZE = 1000  # most frequent words the topic vectorizer keeps (as in the course)
TOPIC_BLOCKED_WORDS = sorted(set(NEGATION_WORDS) | {"face"}
                             | ({"bitcoin", "btc"} if TOPIC_SCOPE == "btc" else set()))  # left out of topic vectors

topic_vectorizer = TfidfVectorizer(max_features=TOPIC_VOCAB_SIZE, stop_words=TOPIC_BLOCKED_WORDS)
try:
    X_topics = topic_vectorizer.fit_transform(topic_posts["clean_text"])  # learns on every topic post
    topic_words = topic_vectorizer.get_feature_names_out()  # the word of each column
except ValueError as error:  # a tiny table can be left without a single word; then every vector is empty
    if "empty vocabulary" not in str(error):
        raise
    topic_vectorizer = None
    X_topics = csr_matrix((len(topic_posts), 0))
    topic_words = np.array([], dtype=object)
X_topics.sort_indices()
posts_without_words = X_topics.getnnz(axis=1) == 0  # all-zero vectors
brief["topics"]["vectorizer"] = {
    "kind": "tfidf",
    "fitted_on": "topic_posts",
    "fitted_on_n_posts": int(X_topics.shape[0]),
    "text": "clean_text",
    "vocab_size_requested": TOPIC_VOCAB_SIZE,
    "vocab_size": int(X_topics.shape[1]),
    "blocked_words": TOPIC_BLOCKED_WORDS,
    "n_posts_without_words": int(posts_without_words.sum()),
}
print(f"Topic vectors: {X_topics.shape[0]:,} posts × {X_topics.shape[1]:,} words; "
      f"{int(posts_without_words.sum()):,} posts have no word of the vocabulary")

# %% [markdown]
# ### Grouping the posts with KMeans
#
# **KMeans** starts with `N_TOPICS` centres and repeats two moves until nothing changes any more: put every post into the group of the centre closest to it, then move every centre to the average position of its group. The result depends on where the centres start, so KMeans starts ten times (`n_init=10`) from different places drawn with `RANDOM_SEED` and keeps the tightest grouping. The column `topic` of `topic_posts` then holds the topic number of every post (0 to `N_TOPICS` − 1, KMeans' own numbering).
#
# **Why 8 topics by default.** Few enough that a person can read every topic and its posts in one table, and enough to separate several subjects; with only eight groups in a large table, even a small topic is likely to keep enough posts for the next chapter to compare bullish and bearish posts inside it. Change `N_TOPICS` to try other numbers. A table with fewer distinct post vectors than `N_TOPICS` (only possible with a tiny table) gets as many topics as it has distinct vectors.
#
# **Same result on every run.** KMeans adds up numbers in several threads at once, and the order in which the threads finish can change the last digits of the sums, and in rare cases a topic. The fit therefore runs in a single thread. On some machines the maths library behind numpy reports floating-point warnings inside matrix products even though every number is finite; the cell silences numpy's floating-point warnings while it computes and then checks that every centre is a finite number.
#
# ***
#
# ### 用 KMeans 分堆
#
# **KMeans** 先放 `N_TOPICS` 个中心，然后反复做两件事，直到不再变化：把每条帖子归到离它最近的那个中心，再把每个中心挪到它那一堆帖子的平均位置。结果和中心一开始放在哪有关，所以 KMeans 用 `RANDOM_SEED` 换十个不同的起点各跑一遍（`n_init=10`），留下分得最紧的那一次。之后 `topic_posts` 的 `topic` 列就是每条帖子的话题编号（0 到 `N_TOPICS` − 1，沿用 KMeans 自己的编号）。
#
# **默认为什么是 8 个话题。** 少到一个人能在一张表里把每个话题和它的原话读完，又多到能分开好几件事；一张大表只分八堆，小的话题多半也留得下足够的帖子，下一章还能在话题里面比较看多和看空。想试别的个数就改 `N_TOPICS`。帖子表里不同的向量比 `N_TOPICS` 还少时（只有表很小时才会这样），话题数就等于不同向量的个数。
#
# **每次结果都一样。** KMeans 会用几个线程同时做加法，哪个线程先做完会影响加出来的最后几位小数，极少数情况下还会改变某个话题。所以这一步只用一个线程。有些机器上，numpy 背后的数学库在做矩阵乘法时会报浮点数警告，其实每个数都是有限的；这一格计算时让 numpy 不报这类警告，算完再检查每个中心都是有限的数。

# %%
# Split the topic vectors into N_TOPICS groups with KMeans, in a single thread so that every run gives the same topics.
from sklearn.cluster import KMeans  # k-means clustering
from threadpoolctl import threadpool_limits  # limits the number of threads

TOPIC_N_INIT = 10  # KMeans starts this many times and keeps the tightest grouping
QUIET_FLOATS = {"divide": "ignore", "over": "ignore", "invalid": "ignore"}  # numpy floating-point warnings off


def count_distinct_rows(matrix):
    """Number of different rows in a sparse CSR matrix with sorted indices."""
    return len({(matrix.indices[start:end].tobytes(), matrix.data[start:end].tobytes())
                for start, end in zip(matrix.indptr[:-1], matrix.indptr[1:])})


n_distinct_vectors = count_distinct_rows(X_topics)
n_topics = min(N_TOPICS, n_distinct_vectors)  # fewer topics only when there are fewer distinct vectors
if n_topics and X_topics.shape[1]:
    with threadpool_limits(limits=1), np.errstate(**QUIET_FLOATS):
        topic_model = KMeans(n_clusters=n_topics, n_init=TOPIC_N_INIT, random_state=RANDOM_SEED).fit(X_topics)
    topic_posts["topic"] = topic_model.labels_.astype("int64")
    topic_centres = topic_model.cluster_centers_  # one row per topic, one column per word
else:  # no word at all, or no post: at most one topic, holding every post
    topic_model = None
    topic_posts["topic"] = np.zeros(len(topic_posts), dtype="int64")
    topic_centres = np.zeros((n_topics, X_topics.shape[1]))
if not np.isfinite(topic_centres).all():
    raise ValueError("A topic centre holds a value that is not a finite number")
print(f"{n_topics} topics (asked for {N_TOPICS}; {n_distinct_vectors:,} distinct post vectors)")
topic_posts["topic"].value_counts().sort_index().rename("posts")

# %% [markdown]
# ### The topic table
#
# For every topic:
#
# - **name**: the `TOPIC_NAME_WORDS` words (5) with the largest values in the topic's centre, put together by the program; a person can give the topics better names after reading them;
# - **posts** and **share**: how many posts of the topic table it holds, and what part of the table that is;
# - **quotes**: the original text of the `TOPIC_QUOTES` posts (3) closest to the centre, skipping a text that is already shown, with their `post_id` and label. Only posts with at least one word of the vocabulary are quoted (unless a topic has no other posts): a post without any such word is a vector of zeros, which can sit close to a centre near zero without having anything in common with the topic.
#
# Everything goes into the brief's `topics` section. The table is sorted by size, largest topic first.
#
# ***
#
# ### 话题表
#
# 每个话题给出：
#
# - **名字**：中心里数值最大的 `TOPIC_NAME_WORDS` 个词（5 个），由程序拼出来；人读过之后可以起更好的名字；
# - **帖子数**和**占比**：它有话题表里的多少条帖子，占整张表多少；
# - **原话**：离中心最近的 `TOPIC_QUOTES` 条帖子（3 条）的原文，已经出现过的文字跳过，附上 `post_id` 和标签。只从至少有一个词表里的词的帖子里挑（除非这个话题只有那种帖子）：一个词都没有的帖子向量全是 0，可能离一个靠近 0 的中心很近，却和这个话题没有任何共同点。
#
# 这些都写进简报的 `topics` 节。下表按大小排，最大的话题在最前面。

# %%
# Name every topic, count its posts and pick the posts closest to its centre; record everything in the brief.
TOPIC_NAME_WORDS = 5  # words in a topic name
TOPIC_QUOTES = 3  # original posts shown per topic

topic_of_post = topic_posts["topic"].to_numpy()
with np.errstate(**QUIET_FLOATS):
    squared_lengths = np.asarray(X_topics.multiply(X_topics).sum(axis=1)).ravel()  # |x|^2 of every post vector
topics = []
for topic in range(n_topics):
    members = np.flatnonzero(topic_of_post == topic)  # rows of topic_posts in this topic, file order
    centre = topic_centres[topic]
    strongest = np.argsort(-centre, kind="stable")[:TOPIC_NAME_WORDS]  # ties keep vocabulary order
    top_words = [str(topic_words[column]) for column in strongest if centre[column] > 0]
    with np.errstate(**QUIET_FLOATS):
        distances = squared_lengths[members] - 2 * (X_topics[members] @ centre) + centre @ centre  # |x - c|^2
    if not np.isfinite(distances).all():
        raise ValueError(f"Topic {topic}: a distance to the centre is not a finite number")
    readable = ~posts_without_words[members]  # posts with at least one word of the vocabulary
    if not readable.any():  # quote those only, unless the topic has nothing else
        readable[:] = True
    candidates, candidate_distances = members[readable], distances[readable]
    quotes, seen_texts = [], set()
    for row in candidates[np.lexsort((candidates, candidate_distances))]:  # closest first; ties in file order
        text = topic_posts.at[row, "text"]
        if text in seen_texts:
            continue
        seen_texts.add(text)
        quotes.append({"post_id": topic_posts.at[row, "post_id"], "label": int(topic_posts.at[row, "label"]),
                       "text": text})
        if len(quotes) == TOPIC_QUOTES:
            break
    topics.append({
        "topic": topic,
        "name": " / ".join(top_words),
        "top_words": top_words,
        "n_posts": int(len(members)),
        "share": safe_ratio(int(len(members)), len(topic_posts)),
        "n_posts_without_words": int(posts_without_words[members].sum()),
        "quotes": quotes,
    })

brief["topics"].update({
    "method": "kmeans",
    "method_settings": {"n_init": TOPIC_N_INIT, "random_state": RANDOM_SEED, "threads": 1},
    "n_topics_requested": N_TOPICS,
    "n_distinct_vectors": int(n_distinct_vectors),
    "n_topics": int(n_topics),
    "name_words": TOPIC_NAME_WORDS,
    "quotes_per_topic": TOPIC_QUOTES,
    "n_posts_with_topic": int(topic_posts["topic"].notna().sum()),
    "topics": topics,
})
topic_table = pd.DataFrame([{"topic": item["topic"], "name": item["name"], "posts": item["n_posts"],
                             "share": item["share"]} for item in topics],
                           columns=["topic", "name", "posts", "share"]).set_index("topic")
topic_table.sort_values("posts", ascending=False, kind="stable")

# %% [markdown]
# ### A few posts from every topic
#
# The quotes, topic by topic in the same order as the table above. They are the posts' original words, before cleaning.
#
# ***
#
# ### 每个话题的几条原话
#
# 原话按话题列出，顺序和上表一样。这些是帖子的原文，没有经过清洗。

# %%
# Print the quotes of every topic, largest topic first.
for topic in topic_table.sort_values("posts", ascending=False, kind="stable").index:
    item = topics[topic]
    print(f"Topic {topic}: {item['name']}  ({item['n_posts']:,} posts, {item['share']:.1%})")
    for quote in item["quotes"]:
        print(f"    [{LABEL_NAMES[quote['label']]}, {quote['post_id']}] {quote['text']}")

# %% [markdown]
# ## 10. Why: which topics lean bullish, which lean bearish?
#
# Chapters 4 to 8 tell bullish from bearish posts; chapter 9 sorts posts into topics. This chapter crosses the two: is each topic more bullish or more bearish than the posts as a whole? The answer is one table, from the most bullish topic to the most bearish one.
#
# **Bullish share.** In a group of posts: the bullish posts divided by the bullish plus the bearish posts. Neutral posts are left out of both. Every topic gets two bullish shares:
#
# - **by label**: from the labels that come with the dataset, over every post of the topic in `topic_posts`. The topic post table keeps the natural class shares (unlike the working sample, which has as many posts in every class), so this is the real share. Next to it: the number of bearish, neutral and bullish posts, and the difference from the whole topic post table in **percentage points** (positive = more bullish than all topic posts together, negative = more bearish).
# - **by model**: from the model chosen in chapter 7, through `sentiment_proba`. It scores every post of the topic table, neutral ones included, and gives each a probability of bullish. A probability of at least `BULLISH_THRESHOLD` (0.7) counts as bullish, one of at most `BEARISH_THRESHOLD` (0.3) as bearish, and anything in between as **unsure**: the model is not confident enough, and the post is not forced to either side. Unsure posts are left out of the model's bullish share; their share of all the topic's posts is reported next to it. When the two thresholds are equal, a probability exactly equal to them counts as bullish.
#
# The row **all** does the same for the whole topic post table and is the baseline.
#
# **Too few posts.** A topic with fewer than `MIN_TOPIC_POSTS` posts is marked: its shares rest on so few posts that chance alone can move them a lot. Why 200 by default: if about half of a topic's posts are neutral, 200 posts leave about 100 bullish or bearish posts, and with 100 posts a bullish share near one half can be off by about 10 percentage points by chance (two standard errors: 2 × √(0.5 × 0.5 / 100) = 0.1). A smaller topic's lean says little. This is a rule of thumb for the default, not a statistical test.
#
# A share whose denominator is 0 (for example a topic in which the model calls no post bullish or bearish) cannot be computed: it is "—" in the table and `null` in the brief, and so is every difference computed from it.
#
# ***
#
# ## 10. 为什么：哪些话题偏多，哪些偏空？
#
# 第 4～8 章分辨看多和看空，第 9 章把帖子分成话题。这一章把两者交叉起来：每个话题比全体帖子更偏多还是更偏空？答案是一张表，从最偏多的话题排到最偏空的话题。
#
# **看多比例。** 一堆帖子里，看多的条数除以「看多 + 看空」的条数，中性不算进去。每个话题算两种看多比例：
#
# - **按标签**：用数据集自带的标签，算这个话题在 `topic_posts` 里的全部帖子。话题用的帖子表保持自然的类别占比（不像工作样本那样每类一样多），所以这是真实的比例。旁边列出看空、中性、看多各几条，以及它和整张话题帖子表之差，单位是**百分点**（正数 = 比全体话题帖子更偏多，负数 = 更偏空）。
# - **按模型**：用第 7 章选中的模型，通过 `sentiment_proba` 打分。话题表里每条帖子都打分，中性帖也打，每条得到一个「看多的概率」。概率不低于 `BULLISH_THRESHOLD`（0.7）算看多，不高于 `BEARISH_THRESHOLD`（0.3）算看空，夹在中间的算**说不清**：模型把握不够，不硬判到哪一边。说不清的帖子不进模型的看多比例；它们占这个话题全部帖子的比例另外列出。两个门槛相等时，正好等于门槛的帖子算看多。
#
# **all** 那一行对整张话题帖子表做同样的计算，作为基准。
#
# **帖子太少。** 帖子数少于 `MIN_TOPIC_POSTS` 的话题会被标出来：它的比例建立在很少几条帖子上，光靠偶然就能变很多。默认为什么是 200：如果一个话题大约一半是中性帖，200 条帖子里大约剩 100 条看多或看空；只有 100 条时，一个接近一半的看多比例，光靠偶然就可能差 10 个百分点左右（两个标准误：2 × √(0.5 × 0.5 / 100) = 0.1）。再小的话题，偏多偏空就说明不了什么。这只是给默认值的经验算法，不是统计检验。
#
# 分母为 0 的比例（例如模型在某个话题里一条看多、看空都没判出来）算不出来：表里显示「—」，简报里写 `null`，由它算出的差值也一样。

# %% [markdown]
# ### Three things to keep in mind
#
# **Scoring neutral posts is using the model beyond what it learned.** With the default `N_CLASSES = 2` the models learned only bullish and bearish posts and never saw a neutral one. Scoring neutral posts uses the model outside what it was trained for: many of them land between the two thresholds and become unsure, but some are pushed to one side. The by-label share leaves neutral posts out by definition; the by-model share leaves out only what the model itself is unsure about. With `N_CLASSES = 3` the model has a neutral class; its probability of bullish is then taken as bullish ÷ (bullish + bearish), which asks the same question as with two classes (bullish or bearish?), so that a high neutral probability does not drag a post towards bearish. When both are 0, the value is 0.5.
#
# **Test posts are scored here, and that is fine.** The topic post table contains posts of chapter 5's test set (those that mention Bitcoin). Chapter 8 has finished evaluating; here those posts are only scored and counted. Nothing in this chapter trains, tunes or chooses a model, a threshold or anything else, and nothing flows back into earlier chapters, so the rule "the test set is used only in the evaluation chapter" keeps its purpose: no decision is made after looking at the test results.
#
# **A fair comparison of model and labels.** The topic post table also contains posts of the training set, whose labels the model has learned; on them, agreeing with the labels is partly repeating the answers. So the two shares are compared only on posts the model never saw: posts whose `post_id` is not in `train_post_ids` (validation and test posts, and posts that never entered the working sample) and whose label is bullish or bearish. On these posts both bullish shares are computed as above, and the difference is reported in percentage points as **model minus label** (positive = the model is more bullish than the labels). A post whose text is identical to a training post but whose id is different counts as unseen; how many such posts are in the comparison is recorded in the brief.
#
# ***
#
# ### 三点说明
#
# **给中性帖打分，是把模型拿来用在它没学过的地方。** 默认 `N_CLASSES = 2` 时，模型只学过看多和看空，从没见过中性帖。给中性帖打分超出了它训练时学的范围：很多中性帖会落在两个门槛之间、归为说不清，但也有一些会被推到某一边。按标签的看多比例按定义就不算中性帖；按模型的看多比例只去掉模型自己说不清的那些。`N_CLASSES = 3` 时模型有中性这一类，这时「看多的概率」取 看多 ÷（看多 + 看空），问的和两类时是同一个问题（看多还是看空？），中性概率高的帖子就不会因此被拉向看空。两者都是 0 时，取 0.5。
#
# **这里给测试帖打分，没有问题。** 话题用的帖子表里有第 5 章测试集的帖子（提到比特币的那些）。第 8 章已经评估完；这里只是给它们打分、计数。本章不训练、不调参，也不选任何模型、门槛或别的东西，结果也不回流到前面的章节，所以「测试集只在评估章用」这条规矩的本意没有被破坏：没有任何决定是看了测试结果之后才做的。
#
# **模型和标签的公平比较。** 话题帖子表里也有训练集的帖子，模型学过它们的标签；在这些帖子上和标签一致，一部分是在背答案。所以两种比例只在模型没见过的帖子上比：`post_id` 不在 `train_post_ids` 里（验证集、测试集，以及根本没进工作样本的帖子），而且标签是看多或看空。在这批帖子上照上面的算法分别算两种看多比例，差值用百分点报，方向是**按模型的减去按标签的**（正数 = 模型比标签更偏多）。文字和某条训练帖一模一样、但编号不同的帖子，算没见过；这样的帖子有多少条进了比较，记在简报里。

# %% [markdown]
# ### Scoring the topic posts
#
# `sentiment_proba` scores the `clean_text` of every post in `topic_posts`, and the probability of bullish is turned into a call: bullish, bearish or unsure. The calls go into a table of their own, `why_posts` (`post_id`, `label`, `topic`, `bullish_proba`, `model_call`, `seen_in_training`, `in_fair_comparison`), so that `topic_posts` stays as chapter 9 left it. The cell prints how long the scoring took and how many posts got each call.
#
# ***
#
# ### 给话题帖子打分
#
# `sentiment_proba` 给 `topic_posts` 里每条帖子的 `clean_text` 打分，「看多的概率」再换成一个判断：看多、看空或说不清。判断放进一张单独的表 `why_posts`（`post_id`、`label`、`topic`、`bullish_proba`、`model_call`、`seen_in_training`、`in_fair_comparison`），`topic_posts` 保持第 9 章留下的样子。这一格打印打分花了多久，以及三种判断各几条。

# %%
# Score every topic post with the chosen model and turn its probability of bullish into a call.
MODEL_CALLS = ["bullish", "bearish", "unsure"]  # the model's calls, in this order everywhere


def bullish_probability(proba):
    """Probability of bullish from sentiment_proba's output.

    Two classes: the bullish column. Three classes: bullish / (bullish + bearish), 0.5 when both are 0.
    """
    bullish = proba[:, BULLISH_COLUMN].astype("float64")
    if len(SENTIMENT_PROBA_LABELS) == 2:
        return bullish
    bearish = proba[:, SENTIMENT_PROBA_LABELS.index(0)].astype("float64")
    both = bullish + bearish
    return np.divide(bullish, both, out=np.full(len(both), 0.5), where=both > 0)


def model_call(probability):
    """'bullish' at or above BULLISH_THRESHOLD, otherwise 'bearish' at or below BEARISH_THRESHOLD, otherwise 'unsure'."""
    probability = np.asarray(probability, dtype="float64")
    return np.where(probability >= BULLISH_THRESHOLD, "bullish",
                    np.where(probability <= BEARISH_THRESHOLD, "bearish", "unsure"))


scoring_started = time.perf_counter()
topic_proba = sentiment_proba(topic_posts["clean_text"])  # every topic post, neutral ones included
scoring_seconds = time.perf_counter() - scoring_started

why_posts = topic_posts[["post_id", "label", "topic"]].copy()  # a table of its own; topic_posts stays unchanged
why_posts["bullish_proba"] = bullish_probability(topic_proba)
why_posts["model_call"] = model_call(why_posts["bullish_proba"])
why_posts["seen_in_training"] = why_posts["post_id"].isin(train_post_ids)  # the models learned these labels
why_posts["in_fair_comparison"] = ~why_posts["seen_in_training"] & why_posts["label"].isin([0, 2])
print(f"Scored {len(why_posts):,} topic posts with {selected_model_name} in {scoring_seconds:.1f} s; "
      f"{int(why_posts['seen_in_training'].sum()):,} of them are training posts")
why_posts["model_call"].value_counts().reindex(MODEL_CALLS, fill_value=0).rename("posts").rename_axis("model call")

# %% [markdown]
# ### The topic × bullish table
#
# First the row "all" (every topic post), then one row per topic, from the most bullish to the most bearish by label; topics whose by-label share cannot be computed come last, and equal shares keep the topic order. Columns:
#
# - **posts**, **too few**: the topic's posts, and whether they are fewer than `MIN_TOPIC_POSTS`;
# - **bearish**, **neutral**, **bullish**: posts per label;
# - **label share**, **vs all (pp)**: the bullish share by label, and its difference from the row "all" in percentage points;
# - **model bullish**, **model bearish**, **unsure**, **model share**, **unsure share**: the model's calls, its bullish share, and the unsure posts as a share of all the topic's posts;
# - **fair posts**, **fair label**, **fair model**, **model − label (pp)**: the fair comparison on unseen bullish or bearish posts.
#
# Everything goes into the brief's `why` section, together with the model, the thresholds and the number of posts in the fair comparison whose text equals a training post.
#
# ***
#
# ### 「话题 × 多空」表
#
# 第一行是 all（全部话题帖子），然后每个话题一行，按「按标签」的看多比例从最偏多排到最偏空；按标签的比例算不出来的话题排在最后，比例相同的按话题编号排。各列：
#
# - **posts**、**too few**：这个话题的帖子数，以及是否少于 `MIN_TOPIC_POSTS`；
# - **bearish**、**neutral**、**bullish**：按标签各几条；
# - **label share**、**vs all (pp)**：按标签的看多比例，以及它和 all 那一行之差（百分点）；
# - **model bullish**、**model bearish**、**unsure**、**model share**、**unsure share**：模型判看多、判看空、说不清各几条，按模型的看多比例，以及说不清占这个话题全部帖子的比例；
# - **fair posts**、**fair label**、**fair model**、**model − label (pp)**：在模型没见过、标签为看多或看空的帖子上的公平比较。
#
# 这些都写进简报的 `why` 节，连同用的模型、两个门槛，以及参加公平比较的帖子里文字和某条训练帖相同的有几条。

# %%
# Bullish shares by label and by model for every topic and for the whole topic table; record them in the brief.
def bullish_share(bullish, bearish):
    """Bullish / (bullish + bearish); None when there is neither."""
    return safe_ratio(int(bullish), int(bullish) + int(bearish))


def percentage_points(share, baseline):
    """(share - baseline) in percentage points; None when either share is missing."""
    return None if share is None or baseline is None else 100 * (share - baseline)


def why_row(posts):
    """Label counts and bullish share, model calls and bullish share, and the fair comparison for rows of why_posts."""
    label_counts = posts["label"].value_counts().reindex(list(LABEL_NAMES), fill_value=0)  # labels 0, 1, 2
    calls = posts["model_call"].value_counts().reindex(MODEL_CALLS, fill_value=0)
    fair = posts[posts["in_fair_comparison"]]  # unseen posts labelled bearish or bullish
    fair_labels = fair["label"].value_counts().reindex([0, 2], fill_value=0)
    fair_calls = fair["model_call"].value_counts().reindex(MODEL_CALLS, fill_value=0)
    fair_label_share = bullish_share(fair_labels[2], fair_labels[0])
    fair_model_share = bullish_share(fair_calls["bullish"], fair_calls["bearish"])
    return {
        "n_posts": int(len(posts)),
        "label_counts": {str(label): int(count) for label, count in label_counts.items()},
        "label_bullish_share": bullish_share(label_counts[2], label_counts[0]),
        "model_counts": {call: int(count) for call, count in calls.items()},
        "model_bullish_share": bullish_share(calls["bullish"], calls["bearish"]),
        "model_unsure_share": safe_ratio(int(calls["unsure"]), len(posts)),
        "fair": {
            "n_posts": int(len(fair)),
            "label_counts": {str(label): int(count) for label, count in fair_labels.items()},
            "label_bullish_share": fair_label_share,
            "model_counts": {call: int(count) for call, count in fair_calls.items()},
            "model_bullish_share": fair_model_share,
            "model_minus_label_pp": percentage_points(fair_model_share, fair_label_share),
        },
    }


why_overall = why_row(why_posts)  # the whole topic post table: the baseline
topic_names = {item["topic"]: item["name"] for item in brief["topics"]["topics"]}
why_topics = []
for topic in range(n_topics):
    row = why_row(why_posts[why_posts["topic"] == topic])
    why_topics.append({
        "topic": topic,
        "name": topic_names[topic],
        "too_few_posts": row["n_posts"] < MIN_TOPIC_POSTS,
        **row,
        "label_vs_overall_pp": percentage_points(row["label_bullish_share"], why_overall["label_bullish_share"]),
    })
# Most bullish first by label; a share that cannot be computed goes last; ties keep the topic order.
why_topics.sort(key=lambda row: (row["label_bullish_share"] is None, -(row["label_bullish_share"] or 0.0), row["topic"]))

# The fair comparison, counted a second way from the ids, and its posts whose text equals a training post.
labelled_ids = set(topic_posts.loc[topic_posts["label"].isin([0, 2]), "post_id"])
fair_posts = topic_posts[why_posts["in_fair_comparison"].to_numpy()]
brief["why"] = {
    "post_table": "topic_posts",
    "topic_column": "topic",
    "model": selected_model_name,
    "scoring": {
        "function": "sentiment_proba",
        "text": "clean_text",
        "bullish_probability": ("the bullish column" if len(SENTIMENT_PROBA_LABELS) == 2
                                else "bullish / (bullish + bearish); 0.5 when both are 0"),
        "n_scored": int(topic_proba.shape[0]),
        "seconds": scoring_seconds,
    },
    "thresholds": {"bullish_at_or_above": BULLISH_THRESHOLD, "bearish_at_or_below": BEARISH_THRESHOLD},
    "min_topic_posts": MIN_TOPIC_POSTS,
    "bullish_share": "bullish / (bullish + bearish); neutral posts (by label) and unsure posts (by model) are left out",
    "differences": "percentage points; label_vs_overall_pp is the topic minus all topic posts, "
                   "model_minus_label_pp is the model minus the labels",
    "fair_comparison": {
        "posts": "post_id not in train_post_ids and label bearish or bullish",
        "train_ids": "train_post_ids",
        "n_topic_posts_in_training_set": int(why_posts["seen_in_training"].sum()),
        "n_unseen_bullish_or_bearish": len(labelled_ids - train_post_ids),
        "n_same_text_as_a_training_post": int(fair_posts["text"].isin(set(train_df["text"])).sum()),
        "n_same_clean_text_as_a_training_post": int(fair_posts["clean_text"].isin(set(train_df["clean_text"])).sum()),
    },
    "order": "topics by label_bullish_share from highest to lowest; null last; ties by topic number",
    "overall": why_overall,
    "topics": why_topics,
}


def shown_share(value):
    """A share with three decimals, or a dash when it cannot be computed."""
    return "—" if value is None else round(value, 3)


def shown_points(value):
    """Percentage points with one decimal and a sign, or a dash when they cannot be computed."""
    return "—" if value is None else f"{value:+.1f}"


def table_row(row, topic, name, too_few, versus_all):
    """One line of the topic × bullish table."""
    return {
        "topic": topic, "name": name, "posts": row["n_posts"], "too few": too_few,
        "bearish": row["label_counts"]["0"], "neutral": row["label_counts"]["1"], "bullish": row["label_counts"]["2"],
        "label share": shown_share(row["label_bullish_share"]), "vs all (pp)": versus_all,
        "model bullish": row["model_counts"]["bullish"], "model bearish": row["model_counts"]["bearish"],
        "unsure": row["model_counts"]["unsure"], "model share": shown_share(row["model_bullish_share"]),
        "unsure share": shown_share(row["model_unsure_share"]),
        "fair posts": row["fair"]["n_posts"], "fair label": shown_share(row["fair"]["label_bullish_share"]),
        "fair model": shown_share(row["fair"]["model_bullish_share"]),
        "model − label (pp)": shown_points(row["fair"]["model_minus_label_pp"]),
    }


why_table = pd.DataFrame(
    [table_row(why_overall, "all", "all topic posts", "", "")]
    + [table_row(row, row["topic"], row["name"], "yes" if row["too_few_posts"] else "no",
                 shown_points(row["label_vs_overall_pp"])) for row in why_topics]
).set_index("topic")
print(f"Model: {selected_model_name}; bullish at or above {BULLISH_THRESHOLD}, bearish at or below {BEARISH_THRESHOLD}; "
      f"too few below {MIN_TOPIC_POSTS} posts; fair comparison: {why_overall['fair']['n_posts']:,} posts")
why_table

# %% [markdown]
# ## 11. The brief: format, check, save and panel
#
# Every chapter has written its results into `brief`. This last chapter turns it into the notebook's real output, in four steps:
#
# 1. **The format.** The brief has a fixed format, version 1.0: which fields exist, what type each one has, and which ones may be `null`. The repository describes every field in `docs/brief-format.md` and writes the same rules in a form a program can check, a **JSON Schema** (`docs/brief.schema.json`). JSON Schema is a common way to write such rules as JSON; checkers for it exist in many programming languages. Colab cannot read files from the repository, so the notebook carries an identical copy of the rules, and the repository's checks make sure the two copies never drift apart.
# 2. **The check.** Before saving, the brief is checked against the rules. A missing field, a value of the wrong type, a granularity the format does not know, or a number that is NaN or infinite stops the notebook with a list of the problems, and nothing is saved.
# 3. **Saving.** A brief that passed is written to `WORK_DIR/brief.json`.
# 4. **The panel.** A small web page with the main tables of the saved brief. It reads only the file.
#
# This chapter always stays last: a new chapter goes before it.
#
# ***
#
# ## 11. 简报：格式、校验、保存和面板
#
# 前面每一章都把结果写进了 `brief`。最后这一章把它变成笔记本真正的产出，分四步：
#
# 1. **格式。** 简报有固定的格式，版本 1.0：有哪些字段、每个字段是什么类型、哪些可以是 `null`。仓库里的 `docs/brief-format.md` 逐个字段写了说明，同样的规则还写成了程序能核对的形式，即 **JSON Schema**（`docs/brief.schema.json`）。JSON Schema 是把这类规则本身也写成 JSON 的通行写法，很多编程语言都有现成的校验库。Colab 读不到仓库里的文件，所以笔记本自己带着一份一模一样的规则，仓库的检查命令负责确认两份始终一致。
# 2. **校验。** 保存之前，先拿规则对照简报。缺字段、类型不对、粒度不是格式认识的值、某个数是 NaN 或无穷大，笔记本都会停下并列出问题，什么也不保存。
# 3. **保存。** 通过校验的简报写到 `WORK_DIR/brief.json`。
# 4. **面板。** 一个小网页，显示已保存简报里的几张主要的表。它只读这个文件。
#
# 这一章永远是最后一章：以后新加的章插在它前面。

# %% [markdown]
# ### The rules of the format
#
# The next cell holds the rules as JSON Schema text, exactly as in `docs/brief.schema.json`; it is long because it lists every field, and there is no need to read it line by line. In short: `format_version` must be "1.0" and `scope.granularity` must be "batch"; every object lists its fields, all of them must be there, and no other field is allowed; counts are whole numbers of 0 or more, shares lie between 0 and 1; a field that may be `null` has the type `["number", "null"]` (or `["integer", "null"]`), every other field may not be `null`.
#
# ***
#
# ### 格式的规则
#
# 下一格是写成 JSON Schema 文本的规则，和 `docs/brief.schema.json` 一字不差；它列出了每一个字段，所以很长，不必逐行读。要点：`format_version` 必须是 "1.0"，`scope.granularity` 必须是 "batch"；每个对象都列出了自己的字段，全部必须有，别的字段一概不许出现；条数是 0 或更大的整数，比例在 0 和 1 之间；可以为 `null` 的字段，类型写成 `["number", "null"]`（或 `["integer", "null"]`），其余字段都不许是 `null`。

# %%
# The format rules as JSON Schema text (draft 2020-12), identical to docs/brief.schema.json in the repository.
import json  # standard library: JSON

BRIEF_SCHEMA_TEXT = r"""{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Ledger Eyes brief, format 1.0",
  "description": "The rules for brief.json. The meaning of every field is in docs/brief-format.md. The notebook's last chapter holds an identical copy of this text and checks every brief against it before saving.",
  "type": "object",
  "additionalProperties": false,
  "required": ["format_version", "generated_at", "scope", "dataset", "sentiment", "topics", "why"],
  "properties": {
    "format_version": {"type": "string", "const": "1.0"},
    "generated_at": {"type": "string", "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"},
    "scope": {
      "type": "object",
      "additionalProperties": false,
      "required": ["source", "granularity", "coins", "n_classes"],
      "properties": {
        "source": {"type": "string"},
        "granularity": {"type": "string", "enum": ["batch"]},
        "coins": {"type": "string", "enum": ["all", "btc_only"]},
        "n_classes": {"type": "integer", "enum": [2, 3]}
      }
    },
    "dataset": {
      "type": "object",
      "additionalProperties": false,
      "required": ["name", "file", "license", "citation", "label_meaning", "sheets", "n_posts", "n_unique_post_ids",
                   "n_empty_text", "class_counts", "class_shares", "length_words", "btc_mentions", "duplicates",
                   "n_posts_with_links", "sample", "cleaning"],
      "properties": {
        "name": {"type": "string"},
        "file": {"type": "string"},
        "license": {"type": "string"},
        "citation": {"type": "string"},
        "label_meaning": {
          "type": "object",
          "additionalProperties": false,
          "required": ["0", "1", "2"],
          "properties": {"0": {"type": "string"}, "1": {"type": "string"}, "2": {"type": "string"}}
        },
        "sheets": {"type": "object", "minProperties": 1, "propertyNames": {"title": "sheet"}, "additionalProperties": {"$ref": "#/$defs/count"}},
        "n_posts": {"$ref": "#/$defs/count"},
        "n_unique_post_ids": {"$ref": "#/$defs/count"},
        "n_empty_text": {"$ref": "#/$defs/count"},
        "class_counts": {"$ref": "#/$defs/label_counts"},
        "class_shares": {"$ref": "#/$defs/label_shares"},
        "length_words": {
          "type": "object",
          "additionalProperties": false,
          "required": ["mean", "median", "p90", "p99", "min", "max"],
          "properties": {
            "mean": {"$ref": "#/$defs/amount_or_null"},
            "median": {"$ref": "#/$defs/amount_or_null"},
            "p90": {"$ref": "#/$defs/amount_or_null"},
            "p99": {"$ref": "#/$defs/amount_or_null"},
            "min": {"$ref": "#/$defs/count_or_null"},
            "max": {"$ref": "#/$defs/count_or_null"}
          }
        },
        "btc_mentions": {
          "type": "object",
          "additionalProperties": false,
          "required": ["rule", "n_posts", "share", "by_class", "share_within_class"],
          "properties": {
            "rule": {"type": "string"},
            "n_posts": {"$ref": "#/$defs/count"},
            "share": {"$ref": "#/$defs/share_or_null"},
            "by_class": {"$ref": "#/$defs/label_counts"},
            "share_within_class": {"$ref": "#/$defs/label_shares"}
          }
        },
        "duplicates": {
          "type": "object",
          "additionalProperties": false,
          "required": ["n_rows_repeating_an_earlier_text", "n_distinct_texts_repeated"],
          "properties": {
            "n_rows_repeating_an_earlier_text": {"$ref": "#/$defs/count"},
            "n_distinct_texts_repeated": {"$ref": "#/$defs/count"}
          }
        },
        "n_posts_with_links": {"$ref": "#/$defs/count"},
        "sample": {
          "type": "object",
          "additionalProperties": false,
          "required": ["n_classes", "btc_only", "seed", "per_class_requested", "available_by_class", "per_class",
                       "class_counts", "n_posts", "n_mentions_btc"],
          "properties": {
            "n_classes": {"type": "integer", "enum": [2, 3]},
            "btc_only": {"type": "boolean"},
            "seed": {"type": "integer"},
            "per_class_requested": {"$ref": "#/$defs/positive"},
            "available_by_class": {"$ref": "#/$defs/class_counts"},
            "per_class": {"$ref": "#/$defs/count"},
            "class_counts": {"$ref": "#/$defs/class_counts"},
            "n_posts": {"$ref": "#/$defs/count"},
            "n_mentions_btc": {"$ref": "#/$defs/count"}
          }
        },
        "cleaning": {
          "type": "object",
          "additionalProperties": false,
          "required": ["emoji_to_words", "keep_negations", "steps", "avg_words_before", "avg_words_after", "n_empty_after",
                       "clean_texts_matches_steps", "examples"],
          "properties": {
            "emoji_to_words": {"type": "boolean"},
            "keep_negations": {"type": "boolean"},
            "steps": {"type": "array", "minItems": 1, "items": {"type": "string"}},
            "avg_words_before": {"$ref": "#/$defs/amount_or_null"},
            "avg_words_after": {"$ref": "#/$defs/amount_or_null"},
            "n_empty_after": {"$ref": "#/$defs/count"},
            "clean_texts_matches_steps": {"type": "boolean"},
            "examples": {
              "type": "array",
              "items": {
                "type": "object",
                "additionalProperties": false,
                "required": ["raw", "clean"],
                "properties": {"raw": {"type": "string"}, "clean": {"type": "string"}}
              }
            }
          }
        }
      }
    },
    "sentiment": {
      "type": "object",
      "additionalProperties": false,
      "required": ["tokenization", "classes", "split", "vectorizers", "models", "training", "selection", "scoring",
                   "test_texts_also_in_train", "evaluation", "beats_baselines", "confusion_matrices"],
      "properties": {
        "tokenization": {
          "type": "object",
          "additionalProperties": false,
          "required": ["used_by_models", "text", "n_posts", "nltk_word_tokenize", "bpe", "examples"],
          "properties": {
            "used_by_models": {"type": "boolean"},
            "text": {"type": "string"},
            "n_posts": {"$ref": "#/$defs/count"},
            "nltk_word_tokenize": {
              "type": "object",
              "additionalProperties": false,
              "required": ["vocab_size", "avg_tokens_per_post"],
              "properties": {
                "vocab_size": {"$ref": "#/$defs/count"},
                "avg_tokens_per_post": {"$ref": "#/$defs/amount_or_null"}
              }
            },
            "bpe": {
              "type": "object",
              "additionalProperties": false,
              "required": ["vocab_size", "vocab_size_requested", "n_special_tokens", "n_single_characters", "n_merged_pieces",
                           "trained_on", "avg_tokens_per_post"],
              "properties": {
                "vocab_size": {"$ref": "#/$defs/count"},
                "vocab_size_requested": {"$ref": "#/$defs/positive"},
                "n_special_tokens": {"$ref": "#/$defs/count"},
                "n_single_characters": {"$ref": "#/$defs/count"},
                "n_merged_pieces": {"$ref": "#/$defs/count"},
                "trained_on": {"type": "string"},
                "avg_tokens_per_post": {"$ref": "#/$defs/amount_or_null"}
              }
            },
            "examples": {
              "type": "array",
              "items": {
                "type": "object",
                "additionalProperties": false,
                "required": ["kind", "post_id"],
                "properties": {"kind": {"type": "string"}, "post_id": {"type": "string"}}
              }
            }
          }
        },
        "classes": {"$ref": "#/$defs/labels_in_use"},
        "split": {
          "type": "object",
          "additionalProperties": false,
          "required": ["seed", "fractions", "stratified_by", "n_posts", "class_counts", "class_shares", "n_mentions_btc"],
          "properties": {
            "seed": {"type": "integer"},
            "fractions": {
              "type": "object",
              "additionalProperties": false,
              "required": ["train", "validation", "test"],
              "properties": {"train": {"$ref": "#/$defs/share"}, "validation": {"$ref": "#/$defs/share"}, "test": {"$ref": "#/$defs/share"}}
            },
            "stratified_by": {"type": "string"},
            "n_posts": {"$ref": "#/$defs/part_counts"},
            "class_counts": {
              "type": "object",
              "additionalProperties": false,
              "required": ["train", "validation", "test"],
              "properties": {
                "train": {"$ref": "#/$defs/class_counts"},
                "validation": {"$ref": "#/$defs/class_counts"},
                "test": {"$ref": "#/$defs/class_counts"}
              }
            },
            "class_shares": {
              "type": "object",
              "additionalProperties": false,
              "required": ["train", "validation", "test"],
              "properties": {
                "train": {"$ref": "#/$defs/class_shares"},
                "validation": {"$ref": "#/$defs/class_shares"},
                "test": {"$ref": "#/$defs/class_shares"}
              }
            },
            "n_mentions_btc": {"$ref": "#/$defs/part_counts"}
          }
        },
        "vectorizers": {
          "type": "object",
          "additionalProperties": false,
          "required": ["vocab_size_requested", "fitted_on", "fitted_on_n_posts", "text", "storage", "count", "tfidf"],
          "properties": {
            "vocab_size_requested": {"$ref": "#/$defs/positive"},
            "fitted_on": {"type": "string"},
            "fitted_on_n_posts": {"$ref": "#/$defs/count"},
            "text": {"type": "string"},
            "storage": {"type": "string"},
            "count": {
              "type": "object",
              "additionalProperties": false,
              "required": ["vocab_size"],
              "properties": {"vocab_size": {"$ref": "#/$defs/count"}}
            },
            "tfidf": {
              "type": "object",
              "additionalProperties": false,
              "required": ["vocab_size"],
              "properties": {"vocab_size": {"$ref": "#/$defs/count"}}
            }
          }
        },
        "models": {"type": "array", "minItems": 1, "uniqueItems": true, "items": {"type": "string"}},
        "training": {
          "type": "object",
          "additionalProperties": false,
          "required": ["epochs", "batch_size", "optimizer", "seed", "output_layer", "loss", "target_of_label", "history"],
          "properties": {
            "epochs": {"$ref": "#/$defs/positive"},
            "batch_size": {"$ref": "#/$defs/positive"},
            "optimizer": {"type": "string"},
            "seed": {"type": "integer"},
            "output_layer": {"type": "string"},
            "loss": {"type": "string"},
            "target_of_label": {"$ref": "#/$defs/class_counts"},
            "history": {
              "type": "object",
              "propertyNames": {"title": "model"},
              "additionalProperties": {
                "type": "object",
                "propertyNames": {"title": "metric"},
                "additionalProperties": {"type": "array", "items": {"type": ["number", "null"]}}
              }
            }
          }
        },
        "selection": {
          "type": "object",
          "additionalProperties": false,
          "required": ["by", "validation_macro_f1", "selected"],
          "properties": {
            "by": {"type": "string"},
            "validation_macro_f1": {"type": "object", "propertyNames": {"title": "model"}, "additionalProperties": {"$ref": "#/$defs/share_or_null"}},
            "selected": {"type": "string"}
          }
        },
        "scoring": {
          "type": "object",
          "additionalProperties": false,
          "required": ["function", "input", "output", "column_labels", "bullish_column", "model", "train_ids", "n_train_ids", "check"],
          "properties": {
            "function": {"type": "string"},
            "input": {"type": "string"},
            "output": {"type": "string"},
            "column_labels": {"$ref": "#/$defs/labels_in_use"},
            "bullish_column": {"$ref": "#/$defs/count"},
            "model": {"type": "string"},
            "train_ids": {"type": "string"},
            "n_train_ids": {"$ref": "#/$defs/count"},
            "check": {
              "type": "object",
              "additionalProperties": false,
              "required": ["n_texts", "n_columns", "rows_sum_to_one", "matches_evaluated_predictions"],
              "properties": {
                "n_texts": {"$ref": "#/$defs/count"},
                "n_columns": {"$ref": "#/$defs/count"},
                "rows_sum_to_one": {"type": "boolean"},
                "matches_evaluated_predictions": {"type": "boolean"}
              }
            }
          }
        },
        "test_texts_also_in_train": {
          "type": "object",
          "additionalProperties": false,
          "required": ["same_text", "same_clean_text"],
          "properties": {"same_text": {"$ref": "#/$defs/count"}, "same_clean_text": {"$ref": "#/$defs/count"}}
        },
        "evaluation": {
          "type": "object",
          "additionalProperties": false,
          "required": ["subsets", "baselines", "majority_label", "rows"],
          "properties": {
            "subsets": {
              "type": "object",
              "additionalProperties": false,
              "required": ["all_test", "test_mentions_btc"],
              "properties": {"all_test": {"type": "string"}, "test_mentions_btc": {"type": "string"}}
            },
            "baselines": {"type": "array", "items": {"type": "string"}},
            "majority_label": {"$ref": "#/$defs/label"},
            "rows": {
              "type": "array",
              "items": {
                "type": "object",
                "additionalProperties": false,
                "required": ["model", "kind", "subset", "n", "accuracy", "per_class", "macro_f1"],
                "properties": {
                  "model": {"type": "string"},
                  "kind": {"type": "string", "enum": ["model", "baseline"]},
                  "subset": {"type": "string", "enum": ["all_test", "test_mentions_btc"]},
                  "n": {"$ref": "#/$defs/count"},
                  "accuracy": {"$ref": "#/$defs/share_or_null"},
                  "per_class": {
                    "type": "object",
                    "propertyNames": {"title": "label", "$ref": "#/$defs/label"},
                    "required": ["0", "2"],
                    "additionalProperties": {
                      "type": "object",
                      "additionalProperties": false,
                      "required": ["precision", "recall", "f1", "support"],
                      "properties": {
                        "precision": {"$ref": "#/$defs/share_or_null"},
                        "recall": {"$ref": "#/$defs/share_or_null"},
                        "f1": {"$ref": "#/$defs/share_or_null"},
                        "support": {"$ref": "#/$defs/count"}
                      }
                    }
                  },
                  "macro_f1": {"$ref": "#/$defs/share_or_null"}
                }
              }
            }
          }
        },
        "beats_baselines": {"type": "object", "propertyNames": {"title": "model"}, "additionalProperties": {"type": "boolean"}},
        "confusion_matrices": {
          "type": "object",
          "propertyNames": {"title": "model"},
          "additionalProperties": {
            "type": "object",
            "additionalProperties": false,
            "required": ["labels", "rows", "columns", "counts", "n"],
            "properties": {
              "labels": {"$ref": "#/$defs/labels_in_use"},
              "rows": {"type": "string"},
              "columns": {"type": "string"},
              "counts": {"type": "array", "items": {"type": "array", "items": {"$ref": "#/$defs/count"}}},
              "n": {"$ref": "#/$defs/count"}
            }
          }
        }
      }
    },
    "topics": {
      "type": "object",
      "additionalProperties": false,
      "required": ["post_table", "topic_column", "scope", "max_posts", "seed", "n_available", "sampled", "n_posts",
                   "n_unique_post_ids", "class_counts", "class_shares", "n_mentions_btc", "cleaning", "vectorizer", "method",
                   "method_settings", "n_topics_requested", "n_distinct_vectors", "n_topics", "name_words", "quotes_per_topic",
                   "n_posts_with_topic", "topics"],
      "properties": {
        "post_table": {"type": "string"},
        "topic_column": {"type": "string"},
        "scope": {"type": "string", "enum": ["btc", "all"]},
        "max_posts": {"$ref": "#/$defs/positive"},
        "seed": {"type": "integer"},
        "n_available": {"$ref": "#/$defs/count"},
        "sampled": {"type": "boolean"},
        "n_posts": {"$ref": "#/$defs/count"},
        "n_unique_post_ids": {"$ref": "#/$defs/count"},
        "class_counts": {"$ref": "#/$defs/label_counts"},
        "class_shares": {"$ref": "#/$defs/label_shares"},
        "n_mentions_btc": {"$ref": "#/$defs/count"},
        "cleaning": {
          "type": "object",
          "additionalProperties": false,
          "required": ["function", "n_empty_after", "n_also_in_working_sample", "same_as_working_sample"],
          "properties": {
            "function": {"type": "string"},
            "n_empty_after": {"$ref": "#/$defs/count"},
            "n_also_in_working_sample": {"$ref": "#/$defs/count"},
            "same_as_working_sample": {"type": "boolean"}
          }
        },
        "vectorizer": {
          "type": "object",
          "additionalProperties": false,
          "required": ["kind", "fitted_on", "fitted_on_n_posts", "text", "vocab_size_requested", "vocab_size", "blocked_words",
                       "n_posts_without_words"],
          "properties": {
            "kind": {"type": "string"},
            "fitted_on": {"type": "string"},
            "fitted_on_n_posts": {"$ref": "#/$defs/count"},
            "text": {"type": "string"},
            "vocab_size_requested": {"$ref": "#/$defs/positive"},
            "vocab_size": {"$ref": "#/$defs/count"},
            "blocked_words": {"type": "array", "items": {"type": "string"}},
            "n_posts_without_words": {"$ref": "#/$defs/count"}
          }
        },
        "method": {"type": "string"},
        "method_settings": {
          "type": "object",
          "additionalProperties": false,
          "required": ["n_init", "random_state", "threads"],
          "properties": {
            "n_init": {"$ref": "#/$defs/positive"},
            "random_state": {"type": "integer"},
            "threads": {"$ref": "#/$defs/positive"}
          }
        },
        "n_topics_requested": {"$ref": "#/$defs/positive"},
        "n_distinct_vectors": {"$ref": "#/$defs/count"},
        "n_topics": {"$ref": "#/$defs/count"},
        "name_words": {"$ref": "#/$defs/positive"},
        "quotes_per_topic": {"$ref": "#/$defs/positive"},
        "n_posts_with_topic": {"$ref": "#/$defs/count"},
        "topics": {
          "type": "array",
          "items": {
            "type": "object",
            "additionalProperties": false,
            "required": ["topic", "name", "top_words", "n_posts", "share", "n_posts_without_words", "quotes"],
            "properties": {
              "topic": {"$ref": "#/$defs/count"},
              "name": {"type": "string"},
              "top_words": {"type": "array", "items": {"type": "string"}},
              "n_posts": {"$ref": "#/$defs/count"},
              "share": {"$ref": "#/$defs/share"},
              "n_posts_without_words": {"$ref": "#/$defs/count"},
              "quotes": {
                "type": "array",
                "items": {
                  "type": "object",
                  "additionalProperties": false,
                  "required": ["post_id", "label", "text"],
                  "properties": {
                    "post_id": {"type": "string"},
                    "label": {"type": "integer", "enum": [0, 1, 2]},
                    "text": {"type": "string"}
                  }
                }
              }
            }
          }
        }
      }
    },
    "why": {
      "type": "object",
      "additionalProperties": false,
      "required": ["post_table", "topic_column", "model", "scoring", "thresholds", "min_topic_posts", "bullish_share",
                   "differences", "fair_comparison", "order", "overall", "topics"],
      "properties": {
        "post_table": {"type": "string"},
        "topic_column": {"type": "string"},
        "model": {"type": "string"},
        "scoring": {
          "type": "object",
          "additionalProperties": false,
          "required": ["function", "text", "bullish_probability", "n_scored", "seconds"],
          "properties": {
            "function": {"type": "string"},
            "text": {"type": "string"},
            "bullish_probability": {"type": "string"},
            "n_scored": {"$ref": "#/$defs/count"},
            "seconds": {"type": "number", "minimum": 0}
          }
        },
        "thresholds": {
          "type": "object",
          "additionalProperties": false,
          "required": ["bullish_at_or_above", "bearish_at_or_below"],
          "properties": {"bullish_at_or_above": {"$ref": "#/$defs/share"}, "bearish_at_or_below": {"$ref": "#/$defs/share"}}
        },
        "min_topic_posts": {"$ref": "#/$defs/positive"},
        "bullish_share": {"type": "string"},
        "differences": {"type": "string"},
        "fair_comparison": {
          "type": "object",
          "additionalProperties": false,
          "required": ["posts", "train_ids", "n_topic_posts_in_training_set", "n_unseen_bullish_or_bearish",
                       "n_same_text_as_a_training_post", "n_same_clean_text_as_a_training_post"],
          "properties": {
            "posts": {"type": "string"},
            "train_ids": {"type": "string"},
            "n_topic_posts_in_training_set": {"$ref": "#/$defs/count"},
            "n_unseen_bullish_or_bearish": {"$ref": "#/$defs/count"},
            "n_same_text_as_a_training_post": {"$ref": "#/$defs/count"},
            "n_same_clean_text_as_a_training_post": {"$ref": "#/$defs/count"}
          }
        },
        "order": {"type": "string"},
        "overall": {
          "type": "object",
          "additionalProperties": false,
          "required": ["n_posts", "label_counts", "label_bullish_share", "model_counts", "model_bullish_share",
                       "model_unsure_share", "fair"],
          "properties": {
            "n_posts": {"$ref": "#/$defs/count"},
            "label_counts": {"$ref": "#/$defs/label_counts"},
            "label_bullish_share": {"$ref": "#/$defs/share_or_null"},
            "model_counts": {"$ref": "#/$defs/model_calls"},
            "model_bullish_share": {"$ref": "#/$defs/share_or_null"},
            "model_unsure_share": {"$ref": "#/$defs/share_or_null"},
            "fair": {"$ref": "#/$defs/fair"}
          }
        },
        "topics": {
          "type": "array",
          "items": {
            "type": "object",
            "additionalProperties": false,
            "required": ["topic", "name", "too_few_posts", "n_posts", "label_counts", "label_bullish_share", "model_counts",
                         "model_bullish_share", "model_unsure_share", "fair", "label_vs_overall_pp"],
            "properties": {
              "topic": {"$ref": "#/$defs/count"},
              "name": {"type": "string"},
              "too_few_posts": {"type": "boolean"},
              "n_posts": {"$ref": "#/$defs/count"},
              "label_counts": {"$ref": "#/$defs/label_counts"},
              "label_bullish_share": {"$ref": "#/$defs/share_or_null"},
              "model_counts": {"$ref": "#/$defs/model_calls"},
              "model_bullish_share": {"$ref": "#/$defs/share_or_null"},
              "model_unsure_share": {"$ref": "#/$defs/share_or_null"},
              "fair": {"$ref": "#/$defs/fair"},
              "label_vs_overall_pp": {"$ref": "#/$defs/points_or_null"}
            }
          }
        }
      }
    }
  },
  "$defs": {
    "count": {"type": "integer", "minimum": 0},
    "count_or_null": {"type": ["integer", "null"], "minimum": 0},
    "positive": {"type": "integer", "minimum": 1},
    "amount_or_null": {"type": ["number", "null"], "minimum": 0},
    "share": {"type": "number", "minimum": 0, "maximum": 1},
    "share_or_null": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
    "points_or_null": {"type": ["number", "null"], "minimum": -100, "maximum": 100},
    "label": {"type": "string", "enum": ["0", "1", "2"]},
    "labels_in_use": {"type": "array", "minItems": 2, "maxItems": 3, "uniqueItems": true, "items": {"$ref": "#/$defs/label"}},
    "label_counts": {
      "type": "object",
      "additionalProperties": false,
      "required": ["0", "1", "2"],
      "properties": {"0": {"$ref": "#/$defs/count"}, "1": {"$ref": "#/$defs/count"}, "2": {"$ref": "#/$defs/count"}}
    },
    "label_shares": {
      "type": "object",
      "additionalProperties": false,
      "required": ["0", "1", "2"],
      "properties": {"0": {"$ref": "#/$defs/share_or_null"}, "1": {"$ref": "#/$defs/share_or_null"}, "2": {"$ref": "#/$defs/share_or_null"}}
    },
    "class_counts": {
      "type": "object",
      "propertyNames": {"title": "label", "$ref": "#/$defs/label"},
      "required": ["0", "2"],
      "additionalProperties": {"$ref": "#/$defs/count"}
    },
    "class_shares": {
      "type": "object",
      "propertyNames": {"title": "label", "$ref": "#/$defs/label"},
      "required": ["0", "2"],
      "additionalProperties": {"$ref": "#/$defs/share_or_null"}
    },
    "part_counts": {
      "type": "object",
      "additionalProperties": false,
      "required": ["train", "validation", "test"],
      "properties": {"train": {"$ref": "#/$defs/count"}, "validation": {"$ref": "#/$defs/count"}, "test": {"$ref": "#/$defs/count"}}
    },
    "model_calls": {
      "type": "object",
      "additionalProperties": false,
      "required": ["bullish", "bearish", "unsure"],
      "properties": {"bullish": {"$ref": "#/$defs/count"}, "bearish": {"$ref": "#/$defs/count"}, "unsure": {"$ref": "#/$defs/count"}}
    },
    "fair": {
      "type": "object",
      "additionalProperties": false,
      "required": ["n_posts", "label_counts", "label_bullish_share", "model_counts", "model_bullish_share", "model_minus_label_pp"],
      "properties": {
        "n_posts": {"$ref": "#/$defs/count"},
        "label_counts": {
          "type": "object",
          "additionalProperties": false,
          "required": ["0", "2"],
          "properties": {"0": {"$ref": "#/$defs/count"}, "2": {"$ref": "#/$defs/count"}}
        },
        "label_bullish_share": {"$ref": "#/$defs/share_or_null"},
        "model_counts": {"$ref": "#/$defs/model_calls"},
        "model_bullish_share": {"$ref": "#/$defs/share_or_null"},
        "model_minus_label_pp": {"$ref": "#/$defs/points_or_null"}
      }
    }
  }
}
"""
BRIEF_SCHEMA = json.loads(BRIEF_SCHEMA_TEXT)  # the rules as a dictionary
print(f"{BRIEF_SCHEMA['title']}: rules loaded")

# %% [markdown]
# ### Checking and saving a brief
#
# - `brief_problems(document)` lists every way in which a brief breaks the rules, one readable line each: the path of the field, then what is wrong (for example `scope: 'granularity' is a required property`). An empty list means the brief passes. It uses the `jsonschema` package and adds one rule that JSON Schema cannot express: no number may be NaN or infinite.
# - `validate_brief(document)` stops with an error that lists the problems when there are any.
# - `save_brief(brief, path)` is the only way the brief gets saved. It first turns numpy and pandas values into plain Python values; NaN and infinite numbers become `None`, which is `null` in JSON, because a number that cannot be computed is written as `null`. Then it runs `validate_brief` on exactly what it is about to write, and writes the file only when that passes. A `None` in a field that may not be `null` therefore stops the save. As a last safety net, the JSON writer is told to refuse NaN (`allow_nan=False`).
#
# ***
#
# ### 校验和保存简报
#
# - `brief_problems(document)` 列出简报违反规则的每一处，一行一处：先是字段的路径，再是哪里不对（例如 `scope: 'granularity' is a required property`）。列表为空就是通过。它用 `jsonschema` 包，另外加了一条 JSON Schema 自己写不出来的规则：任何数都不许是 NaN 或无穷大。
# - `validate_brief(document)` 有问题时报错停下，并列出问题。
# - `save_brief(brief, path)` 是保存简报的唯一途径。它先把 numpy、pandas 的值换成普通的 Python 值；NaN 和无穷大变成 `None`，也就是 JSON 里的 `null`，因为算不出的数一律写 `null`。然后拿 `validate_brief` 核对它马上要写出去的内容，通过了才写文件。所以不许为 `null` 的字段一旦是 `None`，保存就会停下。最后再加一道保险：写 JSON 时不允许出现 NaN（`allow_nan=False`）。

# %%
# Check a brief against the rules, and save it only when it passes.
import math  # standard library: finite-number check
from pathlib import Path  # standard library: file paths

import numpy as np  # numeric types used by pandas
import pandas as pd  # pandas' missing value
from jsonschema import Draft202012Validator  # JSON Schema checker

BRIEF_VALIDATOR = Draft202012Validator(BRIEF_SCHEMA)


def field_path(parts):
    """Readable path of a field, for example why.topics[0].fair; "brief" for the top level."""
    text = ""
    for part in parts:
        text += f"[{part}]" if isinstance(part, int) else (f".{part}" if text else str(part))
    return text or "brief"


def non_finite_numbers(value, parts=()):
    """(path parts, number) for every NaN or infinite number inside value."""
    if isinstance(value, (float, np.floating)):
        return [] if math.isfinite(value) else [(parts, value)]
    if isinstance(value, dict):
        return [found for key, item in value.items() for found in non_finite_numbers(item, parts + (key,))]
    if isinstance(value, list):
        return [found for index, item in enumerate(value) for found in non_finite_numbers(item, parts + (index,))]
    return []


def brief_problems(document):
    """Every way in which document breaks the format rules, one line each; an empty list means it passes."""
    problems = [f"{field_path(parts)}: {number!r} is not allowed (a number that cannot be computed is null)"
                for parts, number in non_finite_numbers(document)]
    errors = sorted(BRIEF_VALIDATOR.iter_errors(document), key=lambda error: [str(part) for part in error.absolute_path])
    problems += [f"{field_path(error.absolute_path)}: {error.message}" for error in errors]
    return problems


def validate_brief(document):
    """Raise ValueError, listing the problems, when document breaks the format rules."""
    problems = brief_problems(document)
    if problems:
        listed = "\n".join(f"  - {problem}" for problem in problems[:20])
        more = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
        raise ValueError(f"The brief breaks the format rules in {len(problems)} place(s), so it was not saved:\n"
                         f"{listed}{more}")


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


def save_brief(brief_dict, path):
    """Convert, check and write the brief; nothing is written when the check fails. Returns what was written."""
    document = to_json_safe(brief_dict)
    validate_brief(document)  # stops here, before anything is written
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return document

# %% [markdown]
# ### Save the brief
#
# The brief gets the current UTC time and goes through `save_brief` into `WORK_DIR/brief.json`. If the check fails, this cell stops with the list of problems and no file is written.
#
# ***
#
# ### 保存简报
#
# 简报记上当前的 UTC 时间，经 `save_brief` 写到 `WORK_DIR/brief.json`。校验不通过时，这一格停下并列出问题，不写任何文件。

# %%
# Save the brief as JSON, after checking it against the format rules.
from datetime import datetime, timezone  # standard library: current UTC time

brief["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
brief_path = Path(WORK_DIR) / "brief.json"
save_brief(brief, brief_path)
print(f"[brief] saved {brief_path}: format {brief['format_version']}, passed the format check")

# %% [markdown]
# ### The panel
#
# A small web page, built with Gradio, that shows the saved brief on four tabs: what the brief covers, the evaluation table of chapter 8, the topic table of chapter 9 and the topic × bullish table of chapter 10. A value that cannot be computed is shown as "—".
#
# **It reads only the saved file.** `panel_tables(brief_path)` reads `brief.json` and turns it into the four tables; `build_panel(brief_path)` puts them on the page. Neither function uses any other variable of this notebook, so the panel shows exactly what another program reading the file would see.
#
# **Nothing leaves this machine.** Gradio's usage statistics and version checks were switched off in the setup cell at the top, before Gradio could be imported; the cell below switches them off once more right before its import, and builds the page with `analytics_enabled=False` as well. The panel never asks for a public link (`share=False`) and only listens on this machine (`127.0.0.1`).
#
# ***
#
# ### 面板
#
# 一个用 Gradio 搭的小网页，分四个标签页显示已保存的简报：简报覆盖什么、第 8 章的评估表、第 9 章的话题表、第 10 章的「话题 × 多空」表。算不出的值显示为「—」。
#
# **它只读保存下来的文件。** `panel_tables(brief_path)` 读 `brief.json`，把它变成四张表；`build_panel(brief_path)` 把表放到网页上。两个函数都不用这个笔记本里的任何其他变量，所以面板显示的，正是别的程序读这个文件时看到的内容。
#
# **什么都不往外发。** Gradio 的使用统计和版本查询，在最上面的准备格里就已关掉，那时 Gradio 还不可能被 import；下面这一格在 import 之前再关一次，搭网页时还明确传了 `analytics_enabled=False`。面板从不申请公开链接（`share=False`），只在本机（`127.0.0.1`）上监听。

# %%
# The panel: the saved brief file as four tables on a small Gradio page. Uses nothing else from this notebook.
import json  # standard library: JSON
import os  # standard library: environment variables
from pathlib import Path  # standard library: file paths

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"  # once more, right before the import: no usage statistics or version checks
import gradio as gr  # small web pages
import pandas as pd  # tables

NOT_COMPUTED = "—"  # shown where the brief has null
PANEL_TABS = {  # table name -> (tab title, note above the table)
    "scope": ("What the brief covers", "Every section says for itself which posts it uses."),
    "evaluation": ("Bullish or bearish", "Chapter 8: every model and baseline on all test posts and on the test posts "
                                         "that mention Bitcoin."),
    "topics": ("Topics", "Chapter 9: the topics, largest first, with a few posts in their own words."),
    "why": ("Why", "Chapter 10: the row \"all\", then every topic from the most bullish to the most bearish by label; "
                   "shares are bullish ÷ (bullish + bearish), differences are in percentage points."),
}


def panel_count(value):
    """A count with thousands separators."""
    return f"{value:,}"


def panel_share(value):
    """A share or score with three decimals; null as a dash."""
    return NOT_COMPUTED if value is None else f"{value:.3f}"


def panel_points(value):
    """Percentage points with a sign and one decimal; null as a dash."""
    return NOT_COMPUTED if value is None else f"{value:+.1f}"


def panel_text_table(rows):
    """A pandas DataFrame from a list of dicts, every cell as text."""
    return pd.DataFrame(rows).astype(str)


def panel_why_line(row, topic, name, too_few, versus_all):
    """One line of the topic × bullish table."""
    labels, calls, fair = row["label_counts"], row["model_counts"], row["fair"]
    return {
        "topic": topic, "name": name, "posts": panel_count(row["n_posts"]), "too few": too_few,
        "bearish": panel_count(labels["0"]), "neutral": panel_count(labels["1"]), "bullish": panel_count(labels["2"]),
        "label share": panel_share(row["label_bullish_share"]), "vs all (pp)": versus_all,
        "model bullish": panel_count(calls["bullish"]), "model bearish": panel_count(calls["bearish"]),
        "unsure": panel_count(calls["unsure"]), "model share": panel_share(row["model_bullish_share"]),
        "unsure share": panel_share(row["model_unsure_share"]),
        "fair posts": panel_count(fair["n_posts"]), "fair label": panel_share(fair["label_bullish_share"]),
        "fair model": panel_share(fair["model_bullish_share"]),
        "model − label (pp)": panel_points(fair["model_minus_label_pp"]),
    }


def panel_tables(brief_path):
    """Read the brief file and return the panel's four tables (DataFrames of text), keyed as in PANEL_TABS."""
    content = json.loads(Path(brief_path).read_text(encoding="utf-8"))
    scope, dataset = content["scope"], content["dataset"]
    sentiment, topics, why = content["sentiment"], content["topics"], content["why"]
    names = dataset["label_meaning"]  # "0" -> "bearish", "1" -> "neutral", "2" -> "bullish"
    sample = dataset["sample"]

    scope_rows = [
        ("format version", content["format_version"]),
        ("saved at (UTC)", content["generated_at"]),
        ("data source", scope["source"]),
        ("granularity", scope["granularity"]),
        ("posts in the data source", panel_count(dataset["n_posts"])),
        ("bullish or bearish: working sample",
         f"{panel_count(sample['n_posts'])} posts, coins: {scope['coins']}, "
         f"classes: {', '.join(names[label] for label in sentiment['classes'])}"),
        ("bullish or bearish: test set", f"{panel_count(sentiment['split']['n_posts']['test'])} posts"),
        ("bullish or bearish: model for later chapters", sentiment["selection"]["selected"]),
        ("topics: topic post table",
         f"{panel_count(topics['n_posts'])} posts, scope: {topics['scope']}, all three classes"),
        ("topics: number of topics", panel_count(topics["n_topics"])),
        ("why: posts and model", f"{panel_count(why['overall']['n_posts'])} topic posts scored by {why['model']}"),
        ("why: thresholds", f"bullish at or above {why['thresholds']['bullish_at_or_above']}, "
                            f"bearish at or below {why['thresholds']['bearish_at_or_below']}"),
        ("why: fair comparison", f"{panel_count(why['overall']['fair']['n_posts'])} posts the model never saw"),
    ]
    evaluation_rows = []
    for row in sentiment["evaluation"]["rows"]:
        line = {"model": row["model"], "kind": row["kind"], "subset": row["subset"], "posts": panel_count(row["n"]),
                "accuracy": panel_share(row["accuracy"])}
        for label in sentiment["classes"]:
            scores = row["per_class"][label]
            line.update({f"{names[label]} precision": panel_share(scores["precision"]),
                         f"{names[label]} recall": panel_share(scores["recall"]),
                         f"{names[label]} F1": panel_share(scores["f1"])})
        line["macro-F1"] = panel_share(row["macro_f1"])
        evaluation_rows.append(line)
    topic_rows = [{
        "topic": item["topic"], "name": item["name"], "posts": panel_count(item["n_posts"]),
        "share": panel_share(item["share"]),
        "quotes": " | ".join(f"[{names[str(quote['label'])]}] {quote['text']}" for quote in item["quotes"]),
    } for item in sorted(topics["topics"], key=lambda item: (-item["n_posts"], item["topic"]))]
    why_rows = [panel_why_line(why["overall"], "all", "all topic posts", "", "")] + [
        panel_why_line(row, row["topic"], row["name"], "yes" if row["too_few_posts"] else "no",
                       panel_points(row["label_vs_overall_pp"])) for row in why["topics"]]
    return {
        "scope": panel_text_table([{"item": item, "value": value} for item, value in scope_rows]),
        "evaluation": panel_text_table(evaluation_rows),
        "topics": panel_text_table(topic_rows),
        "why": panel_text_table(why_rows),
    }


def build_panel(brief_path):
    """A Gradio page with the tables of panel_tables(brief_path), one tab each. Built, not started."""
    tables = panel_tables(brief_path)
    with gr.Blocks(title="Ledger Eyes: the brief", analytics_enabled=False) as page:
        gr.Markdown(f"# Ledger Eyes: the brief\nRead from the file `{Path(brief_path).name}`. "
                    f"\"{NOT_COMPUTED}\" marks a value that cannot be computed.")
        for name, (title, note) in PANEL_TABS.items():
            with gr.Tab(title):
                gr.Markdown(note)
                gr.Dataframe(value=tables[name], interactive=False, wrap=True)
    return page

# %% [markdown]
# ### Start the panel
#
# The last cell builds the panel from the saved file and, when the switch `LAUNCH_PANEL` is `True` (the default), starts it: Gradio then shows the page below the cell. With `LAUNCH_PANEL = False` the panel is only built; the repository's checks always run this way.
#
# ***
#
# ### 启动面板
#
# 最后一格用保存下来的文件搭出面板；开关 `LAUNCH_PANEL` 为 `True`（默认）时把它启动，Gradio 会把网页显示在这一格下面。`LAUNCH_PANEL = False` 时只搭不启动；仓库的检查命令总是这样跑。

# %%
# Build the panel from the saved brief file; start it when LAUNCH_PANEL is True (no public link, this machine only).
panel = build_panel(brief_path)  # reads only the saved file
panel_sizes = {name: len(table) for name, table in panel_tables(brief_path).items()}
print(f"[panel] built from {brief_path}: " + ", ".join(f"{name} {rows} rows" for name, rows in panel_sizes.items()))
if LAUNCH_PANEL:
    panel.launch(share=False, server_name="127.0.0.1", inline=True)
else:
    print("[panel] not started (LAUNCH_PANEL = False)")
