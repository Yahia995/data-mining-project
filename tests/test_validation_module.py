"""
tests/test_validation_module.py
───────────────────────────────
Unit tests for src/validation/validate.py module (DT-28).
"""

import pandas as pd
import pytest
from src.validation.validate import (
    check_not_empty,
    check_min_length,
    check_valid_emotions,
    check_duplicates,
    check_encoding,
    check_arabic_content,
    validate,
)


def _make_df(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def test_check_not_empty():
    df = pd.DataFrame([
        {"ID": 1, "Text": "نص عربي سليم", "Label": "Happy"},
        {"ID": 2, "Text": "   ", "Label": "Anger"},
        {"ID": 3, "Text": None, "Label": "Sadness"},
    ])
    result = check_not_empty(df)
    assert not result["passed"]
    assert result["count"] == 2
    assert result["indices"] == [1, 2]
    assert len(result["examples"]) == 2


def test_check_min_length():
    df = pd.DataFrame([
        {"ID": 1, "Text": "أنا", "Label": "Happy"},  # 3 chars
        {"ID": 2, "Text": "أب", "Label": "Fear"},    # 2 chars
        {"ID": 3, "Text": "", "Label": "Surprise"},   # 0 chars
    ])
    result = check_min_length(df, min_len=3)
    assert not result["passed"]
    assert result["count"] == 2
    assert result["indices"] == [1, 2]


def test_check_valid_emotions():
    df = pd.DataFrame([
        {"ID": 1, "Text": "نص 1", "Label": "Happy"},
        {"ID": 2, "Text": "نص 2", "Label": "Joy"},       # invalid
        {"ID": 3, "Text": "نص 3", "Label": "Excited"},   # invalid
    ])
    result = check_valid_emotions(df)
    assert not result["passed"]
    assert result["count"] == 2
    assert result["indices"] == [1, 2]
    assert set(result["invalid_values"]) == {"Joy", "Excited"}


def test_check_duplicates():
    df = pd.DataFrame([
        {"ID": 1, "Text": "نص مكرر", "Label": "Happy"},
        {"ID": 2, "Text": "نص مكرر", "Label": "Happy"},  # duplicate
        {"ID": 3, "Text": "نص فريد", "Label": "Happy"},
    ])
    result = check_duplicates(df, subset=["Text", "Label"])
    assert not result["passed"]
    assert result["count"] == 1
    assert result["indices"] == [1]


def test_check_encoding():
    df = pd.DataFrame([
        {"ID": 1, "Text": "نص عربي سليم"},
        {"ID": 2, "Text": "Plain english text"},
    ])
    result = check_encoding(df)
    assert result["passed"]
    assert result["count"] == 0


def test_check_arabic_content():
    df = pd.DataFrame([
        {"ID": 1, "Text": "هذا نص عربي صحيح"},
        {"ID": 2, "Text": "just english text here"},
        {"ID": 3, "Text": "123456 !!!???"},
    ])
    result = check_arabic_content(df)
    assert not result["passed"]
    assert result["count"] == 2
    assert result["indices"] == [1, 2]


def test_validate_all_clean_passes():
    df = pd.DataFrame([
        {"ID": 1, "Text": "أنا سعيد جداً اليوم", "Label": "Happy"},
        {"ID": 2, "Text": "هذا أمر محزن للغاية", "Label": "Sadness"},
        {"ID": 3, "Text": "أشعر بالغضب الشديد", "Label": "Anger"},
    ])
    report = validate(df)
    assert report["overall_passed"] is True
    assert report["total_records"] == 3
    assert all(c["passed"] for c in report["checks"].values())
    assert report["summary_counts"]["check_not_empty"] == 0
    assert report["summary_counts"]["check_duplicates"] == 0
