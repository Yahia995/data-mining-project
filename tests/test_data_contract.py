"""
tests/test_data_contract.py
────────────────────────────
Unit tests for the DataContract validation rules (DT-26).
Run with: pytest tests/test_data_contract.py -v
"""

import pandas as pd
import pytest
from src.validation.contract import (
    DataContract,
    VALID_LABELS,
    LABEL_TO_ID,
    RAW_REQUIRED_COLUMNS,
    MIN_TEXT_LENGTH,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_valid_row(
    idx: int = 0,
    text: str = "أنا سعيد جداً",
    label: str = "Happy",
    source: str = "twitter",
) -> dict:
    return {
        "Unnamed: 0": idx,
        "ID": idx + 1,
        "Text": text,
        "Label": label,
        "Source": source,
    }


def _make_df(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def _valid_df(n: int = 10) -> pd.DataFrame:
    labels = list(VALID_LABELS)
    rows = [
        _make_valid_row(i, f"نص عربي صحيح رقم {i}", labels[i % len(labels)])
        for i in range(n)
    ]
    return pd.DataFrame(rows)


def _contract() -> DataContract:
    return DataContract(check_class_balance=False)


# ── VR-001: Required columns ─────────────────────────────────────────────────

class TestVR001SchemaColumns:
    def test_passes_with_all_required_columns(self):
        df = _valid_df()
        report = _contract().validate(df)
        vr = next(r for r in report.results if r.rule_id == "VR-001")
        assert vr.passed

    def test_aborts_when_column_missing(self):
        df = _valid_df().drop(columns=["Label"])
        report = _contract().validate(df)
        assert report.aborted
        vr = next(r for r in report.results if r.rule_id == "VR-001")
        assert not vr.passed
        assert vr.action_taken == "aborted"


# ── VR-002: No extra columns ─────────────────────────────────────────────────

class TestVR002NoExtraColumns:
    def test_passes_with_exact_columns(self):
        df = _valid_df()
        report = _contract().validate(df)
        vr = next(r for r in report.results if r.rule_id == "VR-002")
        assert vr.passed

    def test_warns_with_extra_column(self):
        df = _valid_df()
        df["extra_col"] = "x"
        report = _contract().validate(df)
        vr = next(r for r in report.results if r.rule_id == "VR-002")
        assert not vr.passed
        assert vr.action_taken == "warned"
        assert not report.aborted  # should not abort


# ── VR-003: ID uniqueness ─────────────────────────────────────────────────────

class TestVR003IdUnique:
    def test_passes_with_unique_ids(self):
        df = _valid_df()
        report = _contract().validate(df)
        vr = next(r for r in report.results if r.rule_id == "VR-003")
        assert vr.passed

    def test_warns_and_reindexes_duplicate_ids(self):
        df = _valid_df()
        df.iloc[1, df.columns.get_loc("ID")] = df.iloc[0]["ID"]  # duplicate ID
        report = _contract().validate(df)
        assert not report.aborted
        vr = next(r for r in report.results if r.rule_id == "VR-003")
        assert not vr.passed
        assert vr.action_taken == "warned"
        assert report.cleaned_df["ID"].is_unique


# ── VR-010: Text not null ──────────────────────────────────────────────────────

class TestVR010TextNotNull:
    def test_passes_no_nulls(self):
        df = _valid_df()
        report = _contract().validate(df)
        vr = next(r for r in report.results if r.rule_id == "VR-010")
        assert vr.passed
        assert vr.records_affected == 0

    def test_drops_null_text_rows(self):
        df = _valid_df(n=6)
        df.iloc[2, df.columns.get_loc("Text")] = None
        report = _contract().validate(df)
        vr = next(r for r in report.results if r.rule_id == "VR-010")
        assert not vr.passed
        assert vr.records_affected == 1
        assert vr.action_taken == "dropped"
        assert len(report.cleaned_df) == 5


# ── VR-011: Text not empty ────────────────────────────────────────────────────

class TestVR011TextNotEmpty:
    def test_drops_empty_text(self):
        df = _valid_df(n=5)
        df.iloc[0, df.columns.get_loc("Text")] = "   "
        report = _contract().validate(df)
        vr = next(r for r in report.results if r.rule_id == "VR-011")
        assert not vr.passed
        assert vr.action_taken == "dropped"


# ── VR-012: Text minimum length ───────────────────────────────────────────────

class TestVR012TextMinLength:
    def test_flags_short_text(self):
        df = _valid_df(n=5)
        df.iloc[0, df.columns.get_loc("Text")] = "أب"  # 2 chars < MIN_TEXT_LENGTH
        report = _contract().validate(df)
        vr = next(r for r in report.results if r.rule_id == "VR-012")
        assert not vr.passed
        assert vr.action_taken == "flagged"
        # Record kept, not dropped
        assert len(report.cleaned_df) == 5
        assert "_flag_short_text" in report.cleaned_df.columns


# ── VR-014: Label valid ───────────────────────────────────────────────────────

class TestVR014LabelValid:
    def test_passes_with_valid_labels(self):
        df = _valid_df()
        report = _contract().validate(df)
        vr = next(r for r in report.results if r.rule_id == "VR-014")
        assert vr.passed

    def test_drops_invalid_label(self):
        df = _valid_df(n=5)
        df.iloc[1, df.columns.get_loc("Label")] = "Joy"  # not a valid label
        report = _contract().validate(df)
        vr = next(r for r in report.results if r.rule_id == "VR-014")
        assert not vr.passed
        assert vr.action_taken == "dropped"
        assert "Joy" not in report.cleaned_df["Label"].values

    @pytest.mark.parametrize("label", sorted(VALID_LABELS))
    def test_all_valid_labels_accepted(self, label: str):
        df = _make_df(_make_valid_row(0, "نص عربي صحيح", label))
        report = _contract().validate(df)
        vr = next(r for r in report.results if r.rule_id == "VR-014")
        assert vr.passed


# ── VR-020 & VR-021: Duplicates ──────────────────────────────────────────────

class TestDeduplication:
    def test_vr020_drops_exact_duplicates(self):
        row1 = _make_valid_row(0, "نص مكرر", "Anger")
        row2 = {**_make_valid_row(1, "نص مكرر", "Anger"), "ID": 2}
        df = _make_df(row1, row2)  # duplicate payload, distinct ID
        report = _contract().validate(df)
        vr = next(r for r in report.results if r.rule_id == "VR-020")
        assert not vr.passed
        assert vr.records_affected == 1
        assert len(report.cleaned_df) == 1

    def test_vr021_drops_text_label_duplicates(self):
        row1 = _make_valid_row(0, "نص مكرر", "Anger", source="twitter")
        row2 = {**_make_valid_row(1, "نص مكرر", "Anger", source="facebook"), "ID": 2}
        df = _make_df(row1, row2)
        report = _contract().validate(df)
        vr = next(r for r in report.results if r.rule_id == "VR-021")
        assert not vr.passed
        assert len(report.cleaned_df) == 1


# ── VR-030: Arabic content ───────────────────────────────────────────────────

class TestVR030ArabicContent:
    def test_flags_non_arabic_text(self):
        df = _make_df(_make_valid_row(0, "hello world this is latin text only"))
        report = _contract().validate(df)
        vr = next(r for r in report.results if r.rule_id == "VR-030")
        assert not vr.passed
        assert vr.action_taken == "flagged"
        assert "_flag_no_arabic" in report.cleaned_df.columns

    def test_passes_arabic_text(self):
        df = _make_df(_make_valid_row(0, "هذا نص عربي"))
        report = _contract().validate(df)
        vr = next(r for r in report.results if r.rule_id == "VR-030")
        assert vr.passed


# ── Integration: valid minimal DataFrame ─────────────────────────────────────

class TestIntegration:
    def test_clean_df_passes_all_rules_without_balance_check(self):
        rows = []
        for i, label in enumerate(sorted(VALID_LABELS) * 100):
            rows.append(_make_valid_row(i, f"نص عربي صحيح رقم {i}", label))
        df = pd.DataFrame(rows)
        report = _contract().validate(df)
        assert not report.aborted
        assert len(report.cleaned_df) > 0
        failed_hard = [
            r for r in report.results
            if not r.passed and r.action_taken in ("aborted", "dropped")
        ]
        assert len(failed_hard) == 0, f"Hard failures: {failed_hard}"

    def test_clean_df_passes_full_contract_with_balance_check(self):
        rows = []
        for label in sorted(VALID_LABELS):
            for i in range(1000):
                idx = len(rows)
                rows.append(_make_valid_row(idx, f"نص عربي صحيح فريد رقم {idx} {label}", label))
        df = pd.DataFrame(rows)
        report = DataContract(check_class_balance=True).validate(df)
        assert not report.aborted
        assert len(report.cleaned_df) == 6000

    def test_report_summary_prints(self):
        df = _valid_df(n=12)
        report = _contract().validate(df)
        summary = report.summary()
        assert "VR-001" in summary
        assert "VR-014" in summary
