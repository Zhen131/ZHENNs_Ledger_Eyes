"""The brief: the result file the Ledger will read once Ledger Eyes is connected to it.

This module does three things: it creates an empty brief, checks a brief against the format rules, and saves it
as JSON only when the check passes. The rules themselves are in docs/brief.schema.json, and every field is
described in docs/brief-format.md. Filling the brief is not done here: that code is spread over the chapters of
notebook/ledger_eyes.ipynb.

Why a module of its own: the short notebook, notebook/ledger_eyes_v2.ipynb, carries no brief (decided on
2026-11-15). The code below comes from the long notebook's cell "The brief" and from its chapter 13; the only
change is that the format rules are read from the file instead of an embedded copy.
"""
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator

FORMAT_VERSION = "2.2"
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "docs" / "brief.schema.json"
BRIEF_SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
BRIEF_VALIDATOR = Draft202012Validator(BRIEF_SCHEMA)


def new_brief(source, ticker_only=False, n_classes=3):
    """Return an empty brief: the five sections are None until a chapter fills them.

    source: the address of the data; ticker_only: True when only posts that name a ticker are read;
    n_classes: the number of sentiment classes.
    """
    return {
        "format_version": FORMAT_VERSION,
        "generated_at": None,
        "scope": {
            "source": source,
            "granularity": "batch",
            "tickers": "has_ticker_only" if ticker_only else "all",
            "n_classes": n_classes,
        },
        "dataset": None,
        "sentiment": None,
        "levels": None,
        "replay": None,
    }


def field_path(parts):
    """Readable path of a field, for example sentiment.evaluation.rows[0]; "brief" for the top level."""
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
    """Raise ValueError, listing the first 20 problems, when document breaks the format rules."""
    problems = brief_problems(document)
    if problems:
        listed = "\n".join(f"  - {problem}" for problem in problems[:20])
        more = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
        raise ValueError(f"The brief breaks the format rules in {len(problems)} place(s), so it was not saved:\n"
                         f"{listed}{more}")


def to_json_safe(value):
    """Convert numpy and pandas values to plain Python; non-finite numbers become None.

    JSON knows neither numpy's number types nor NaN.
    """
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
    """Stamp the time, convert, check and write the brief; returns what was written.

    Nothing is written when the check fails, so the Ledger never reads half a brief or one in a wrong format.
    """
    if brief_dict.get("generated_at") is None:
        brief_dict["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    document = to_json_safe(brief_dict)
    validate_brief(document)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return document
