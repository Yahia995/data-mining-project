"""
src/validation/validate.py
──────────────────────────
Data validation module checking raw dataset against the data contract.

Implements discrete validation check functions specified in DT-28:
  - check_not_empty(df)        -> identifies empty/null text rows
  - check_min_length(df)       -> flags short texts (< min_len)
  - check_valid_emotions(df)   -> validates emotion values against allowed classes
  - check_duplicates(df)       -> finds exact duplicate rows
  - check_encoding(df)         -> validates UTF-8 text encoding
  - check_arabic_content(df)   -> detects records lacking Arabic script
  - validate(df)               -> orchestrates all checks, logs row indices, returns structured report dict

Runnable as a standalone CLI tool:
    python -m src.validation.validate [--csv-path data/raw/dz_emotion_raw.csv]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Sequence
import pandas as pd

from src.validation.contract import ARABIC_UNICODE_RE, VALID_LABELS

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def check_not_empty(df: pd.DataFrame, text_col: str = "Text") -> dict[str, Any]:
    """
    Check for null or empty/whitespace-only text records.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset to inspect.
    text_col : str
        Name of text column (default "Text").

    Returns
    -------
    dict[str, Any]
        Check result dictionary with status, counts, affected row indices, and examples.
    """
    if text_col not in df.columns:
        return {
            "check": "check_not_empty",
            "passed": False,
            "error": f"Column '{text_col}' not found in DataFrame",
            "count": len(df),
            "indices": list(df.index),
            "examples": [],
        }

    null_mask = df[text_col].isnull()
    empty_mask = df[text_col].astype(str).str.strip() == ""
    violating_mask = null_mask | empty_mask

    indices = df[violating_mask].index.tolist()
    count = len(indices)
    examples = df.loc[indices[:5]].to_dict(orient="records") if count > 0 else []

    if count > 0:
        logger.warning(
            "check_not_empty failed: %d empty or null text records found at row indices %s%s",
            count,
            indices[:10],
            "..." if count > 10 else "",
        )
    else:
        logger.info("check_not_empty passed: 0 empty or null text records found.")

    return {
        "check": "check_not_empty",
        "passed": count == 0,
        "count": count,
        "indices": indices,
        "examples": examples,
    }


def check_min_length(df: pd.DataFrame, min_len: int = 3, text_col: str = "Text") -> dict[str, Any]:
    """
    Check for texts shorter than minimum required character length.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset to inspect.
    min_len : int
        Minimum character threshold (default 3).
    text_col : str
        Name of text column (default "Text").

    Returns
    -------
    dict[str, Any]
        Check result dictionary.
    """
    if text_col not in df.columns:
        return {
            "check": "check_min_length",
            "passed": False,
            "error": f"Column '{text_col}' not found in DataFrame",
            "count": len(df),
            "indices": list(df.index),
            "examples": [],
        }

    valid_text = df[text_col].fillna("").astype(str).str.strip()
    short_mask = valid_text.str.len() < min_len

    indices = df[short_mask].index.tolist()
    count = len(indices)
    examples = df.loc[indices[:5], [c for c in ["ID", text_col, "Label"] if c in df.columns]].to_dict(orient="records") if count > 0 else []

    if count > 0:
        logger.warning(
            "check_min_length: %d records shorter than %d characters at indices %s%s",
            count,
            min_len,
            indices[:10],
            "..." if count > 10 else "",
        )
    else:
        logger.info("check_min_length passed: all records >= %d characters.", min_len)

    return {
        "check": "check_min_length",
        "passed": count == 0,
        "min_len": min_len,
        "count": count,
        "indices": indices,
        "examples": examples,
    }


def check_valid_emotions(
    df: pd.DataFrame,
    allowed: Sequence[str] | set[str] = VALID_LABELS,
    label_col: str = "Label",
) -> dict[str, Any]:
    """
    Validate that emotion labels belong to the permitted class set.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset to inspect.
    allowed : Collection[str]
        Set or list of valid canonical emotion labels.
    label_col : str
        Name of the emotion label column (default "Label").

    Returns
    -------
    dict[str, Any]
        Check result dictionary.
    """
    allowed_set = set(allowed)
    if label_col not in df.columns:
        return {
            "check": "check_valid_emotions",
            "passed": False,
            "error": f"Column '{label_col}' not found in DataFrame",
            "count": len(df),
            "indices": list(df.index),
            "invalid_values": [],
            "examples": [],
        }

    invalid_mask = ~df[label_col].isin(allowed_set)
    indices = df[invalid_mask].index.tolist()
    count = len(indices)
    invalid_values = df.loc[invalid_mask, label_col].unique().tolist() if count > 0 else []
    examples = df.loc[indices[:5], [c for c in ["ID", "Text", label_col] if c in df.columns]].to_dict(orient="records") if count > 0 else []

    if count > 0:
        logger.warning(
            "check_valid_emotions failed: %d invalid label records with values %s at indices %s%s",
            count,
            invalid_values,
            indices[:10],
            "..." if count > 10 else "",
        )
    else:
        logger.info("check_valid_emotions passed: all labels conform to allowed set (%s).", allowed_set)

    return {
        "check": "check_valid_emotions",
        "passed": count == 0,
        "allowed": sorted(allowed_set),
        "invalid_values": invalid_values,
        "count": count,
        "indices": indices,
        "examples": examples,
    }


def check_duplicates(
    df: pd.DataFrame,
    subset: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Identify duplicate records across payload columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset to inspect.
    subset : Sequence[str] | None
        Columns to consider for duplicate evaluation. Defaults to ["Text", "Label"] if present,
        or all columns if not.

    Returns
    -------
    dict[str, Any]
        Check result dictionary.
    """
    if subset is None:
        if "Text" in df.columns and "Label" in df.columns:
            subset = ["Text", "Label"]
        else:
            subset = list(df.columns)

    dupe_mask = df.duplicated(subset=subset, keep="first")
    indices = df[dupe_mask].index.tolist()
    count = len(indices)
    examples = df.loc[indices[:5], subset].to_dict(orient="records") if count > 0 else []

    if count > 0:
        logger.warning(
            "check_duplicates: %d duplicate records found (on subset %s) at indices %s%s",
            count,
            subset,
            indices[:10],
            "..." if count > 10 else "",
        )
    else:
        logger.info("check_duplicates passed: 0 duplicate records on subset %s.", subset)

    return {
        "check": "check_duplicates",
        "passed": count == 0,
        "subset": list(subset),
        "count": count,
        "indices": indices,
        "examples": examples,
    }


def check_encoding(df: pd.DataFrame, text_col: str = "Text") -> dict[str, Any]:
    """
    Validate that all text entries can be cleanly encoded as UTF-8 without errors.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset to inspect.
    text_col : str
        Name of text column (default "Text").

    Returns
    -------
    dict[str, Any]
        Check result dictionary.
    """
    if text_col not in df.columns:
        return {
            "check": "check_encoding",
            "passed": False,
            "error": f"Column '{text_col}' not found in DataFrame",
            "count": len(df),
            "indices": list(df.index),
            "examples": [],
        }

    def _is_utf8_valid(val: Any) -> bool:
        if pd.isnull(val):
            return True
        try:
            str(val).encode("utf-8")
            return True
        except (UnicodeEncodeError, UnicodeDecodeError):
            return False

    invalid_mask = ~df[text_col].map(_is_utf8_valid)
    indices = df[invalid_mask].index.tolist()
    count = len(indices)
    examples = df.loc[indices[:5]].to_dict(orient="records") if count > 0 else []

    if count > 0:
        logger.warning(
            "check_encoding failed: %d records with non-UTF-8 encoding at indices %s%s",
            count,
            indices[:10],
            "..." if count > 10 else "",
        )
    else:
        logger.info("check_encoding passed: all text records are valid UTF-8.")

    return {
        "check": "check_encoding",
        "passed": count == 0,
        "count": count,
        "indices": indices,
        "examples": examples,
    }


def check_arabic_content(df: pd.DataFrame, text_col: str = "Text") -> dict[str, Any]:
    """
    Detect text entries that lack any Arabic Unicode characters (U+0600 - U+06FF).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset to inspect.
    text_col : str
        Name of text column (default "Text").

    Returns
    -------
    dict[str, Any]
        Check result dictionary.
    """
    if text_col not in df.columns:
        return {
            "check": "check_arabic_content",
            "passed": False,
            "error": f"Column '{text_col}' not found in DataFrame",
            "count": len(df),
            "indices": list(df.index),
            "examples": [],
        }

    def _has_arabic(val: Any) -> bool:
        if pd.isnull(val):
            return False
        return bool(ARABIC_UNICODE_RE.search(str(val)))

    no_arabic_mask = ~df[text_col].map(_has_arabic)
    indices = df[no_arabic_mask].index.tolist()
    count = len(indices)
    examples = df.loc[indices[:5], [c for c in ["ID", text_col, "Label"] if c in df.columns]].to_dict(orient="records") if count > 0 else []

    if count > 0:
        logger.warning(
            "check_arabic_content: %d records lack Arabic characters at indices %s%s",
            count,
            indices[:10],
            "..." if count > 10 else "",
        )
    else:
        logger.info("check_arabic_content passed: all text records contain Arabic script.")

    return {
        "check": "check_arabic_content",
        "passed": count == 0,
        "count": count,
        "indices": indices,
        "examples": examples,
    }


def validate(
    df: pd.DataFrame,
    allowed_emotions: Sequence[str] | set[str] = VALID_LABELS,
    min_len: int = 3,
    text_col: str = "Text",
    label_col: str = "Label",
) -> dict[str, Any]:
    """
    Orchestrate all 6 contract validation checks on a raw DataFrame.

    Produces a structured summary report dictionary detailing:
      - overall_passed: True if all critical checks passed
      - total_records: count of records examined
      - checks: dictionary mapping each check name to its structured output
      - summary_counts: quick dictionary of violation counts per check

    Parameters
    ----------
    df : pd.DataFrame
        Input raw dataset to validate.
    allowed_emotions : Sequence[str] | set[str]
        Set of valid emotion classes.
    min_len : int
        Minimum character length for text.
    text_col : str
        Column name for text.
    label_col : str
        Column name for label.

    Returns
    -------
    dict[str, Any]
        Comprehensive structured validation report.
    """
    logger.info("Starting data validation checks on %d records...", len(df))

    res_empty = check_not_empty(df, text_col=text_col)
    res_min_len = check_min_length(df, min_len=min_len, text_col=text_col)
    res_emotions = check_valid_emotions(df, allowed=allowed_emotions, label_col=label_col)
    res_duplicates = check_duplicates(df)
    res_encoding = check_encoding(df, text_col=text_col)
    res_arabic = check_arabic_content(df, text_col=text_col)

    checks = {
        "check_not_empty": res_empty,
        "check_min_length": res_min_len,
        "check_valid_emotions": res_emotions,
        "check_duplicates": res_duplicates,
        "check_encoding": res_encoding,
        "check_arabic_content": res_arabic,
    }

    summary_counts = {name: c["count"] for name, c in checks.items()}
    all_passed = all(c["passed"] for c in checks.values())

    report = {
        "overall_passed": all_passed,
        "total_records": len(df),
        "checks": checks,
        "summary_counts": summary_counts,
    }

    logger.info("Validation completed. Summary violation counts: %s", summary_counts)
    return report


def main() -> None:
    """CLI entrypoint for standalone data validation execution."""
    parser = argparse.ArgumentParser(description="Validate raw Arabic emotion dataset against data contract.")
    parser.add_argument(
        "--csv-path",
        default="data/raw/dz_emotion_raw.csv",
        help="Path to raw CSV dataset (default: data/raw/dz_emotion_raw.csv)",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write the structured JSON validation report.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        logger.error("Dataset not found at '%s'. Run ingestion first.", args.csv_path)
        sys.exit(1)

    df = pd.read_csv(args.csv_path)
    report = validate(df)

    print("\n" + "=" * 60)
    print("DATA VALIDATION REPORT (DT-28)")
    print("=" * 60)
    print(f"Total Records Inspected: {report['total_records']:,}")
    print(f"Overall Clean / Passed:  {report['overall_passed']}")
    print("-" * 60)
    for name, res in report["checks"].items():
        status = "PASSED" if res["passed"] else f"VIOLATIONS: {res['count']}"
        print(f"  [{status}] {name}")
    print("=" * 60 + "\n")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("Validation report written to %s", args.output_json)


if __name__ == "__main__":
    main()
