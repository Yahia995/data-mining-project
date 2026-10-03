"""
src/validation/contract.py
──────────────────────────
Data contract implementation for the Arabic Emotion Detection Pipeline.

Implements the rules defined in docs/methodology/data_contract.md (DT-26).
Grounded in DT-21 inspection findings:
  - Raw columns: Unnamed: 0, ID, Text, Label, Source
  - 6 Ekman emotion classes, 1,000 samples each (perfectly balanced)
  - 1 null in Text column; 3,199 text-level duplicates

Usage
-----
    from src.validation.contract import DataContract, ValidationReport
    report = DataContract().validate(df)
    clean_df = report.cleaned_df
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# Raw input columns (as delivered by HuggingFace / download.py)
RAW_REQUIRED_COLUMNS: list[str] = ["Unnamed: 0", "ID", "Text", "Label", "Source"]

# Canonical emotion classes (case-sensitive, as in raw data)
VALID_LABELS: set[str] = {"Anger", "Disgust", "Fear", "Happy", "Sadness", "Surprise"}

# Label → integer mapping for output schema
LABEL_TO_ID: dict[str, int] = {
    "Anger": 0,
    "Disgust": 1,
    "Fear": 2,
    "Happy": 3,
    "Sadness": 4,
    "Surprise": 5,
}

# Text length constraints
MIN_TEXT_LENGTH: int = 3
MAX_TEXT_LENGTH: int = 2000

# Class balance thresholds (post 3,199 duplicate drop: 2,800 unique samples remain, ~444–481 per class)
MIN_SAMPLES_PER_CLASS: int = 400
MAX_IMBALANCE_RATIO: float = 3.0
MIN_TOTAL_RECORDS: int = 2_500

# Arabic Unicode block (U+0600 – U+06FF)
ARABIC_UNICODE_RE = re.compile(r"[\u0600-\u06FF]")

# Reproducibility seed
RANDOM_SEED: int = 42

# ── Violation Report ─────────────────────────────────────────────────────────


@dataclass
class RuleResult:
    """Outcome of a single validation rule."""

    rule_id: str
    rule_name: str
    passed: bool
    records_affected: int = 0
    message: str = ""
    action_taken: Literal["none", "dropped", "flagged", "aborted", "warned"] = "none"


@dataclass
class ValidationReport:
    """Aggregated result of running the full DataContract."""

    results: list[RuleResult] = field(default_factory=list)
    cleaned_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    aborted: bool = False
    abort_reason: str = ""

    @property
    def passed(self) -> bool:
        return not self.aborted and all(r.passed for r in self.results)

    def summary(self) -> str:
        lines = ["=" * 60, "Data Contract Validation Report", "=" * 60]
        if self.aborted:
            lines.append(f"  ❌  ABORTED: {self.abort_reason}")
        for r in self.results:
            icon = "✅" if r.passed else "⚠️ "
            lines.append(
                f"  {icon}  [{r.rule_id}] {r.rule_name}"
                + (f" — {r.message}" if r.message else "")
            )
        lines += [
            "-" * 60,
            f"  Records in cleaned dataset: {len(self.cleaned_df):,}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ── Data Contract ─────────────────────────────────────────────────────────────


class DataContract:
    """
    Validates a raw Dz-Emotion DataFrame against the contract defined in
    docs/methodology/data_contract.md.

    Parameters
    ----------
    strict : bool
        If True, flag-only violations are treated as errors (default False).
    """

    def __init__(
        self,
        strict: bool = False,
        check_class_balance: bool = True,
        min_total_records: int = MIN_TOTAL_RECORDS,
        min_samples_per_class: int = MIN_SAMPLES_PER_CLASS,
    ) -> None:
        self.strict = strict
        self.check_class_balance = check_class_balance
        self.min_total_records = min_total_records
        self.min_samples_per_class = min_samples_per_class

    # ── Public API ──────────────────────────────────────────────────────────

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """
        Run all validation rules and return a ValidationReport.

        The returned report contains:
          - `results`     : per-rule outcomes
          - `cleaned_df`  : DataFrame after drops/flags applied
          - `aborted`     : True if a hard-abort rule was violated
        """
        report = ValidationReport()
        working = df.copy()

        # ── Column-level rules (VR-001 to VR-003) ──────────────────────────
        abort, working = self._vr001_schema_columns(working, report)
        if abort:
            return report

        self._vr002_no_extra_columns(working, report)

        abort, working = self._vr003_id_unique(working, report)
        if abort:
            return report

        # ── Record-level rules (VR-010 to VR-021) ──────────────────────────
        working = self._vr010_text_not_null(working, report)
        working = self._vr011_text_not_empty(working, report)
        working = self._vr012_text_min_length(working, report)
        working = self._vr013_text_encoding_valid(working, report)
        working = self._vr015_label_not_null(working, report)
        working = self._vr014_label_valid(working, report)
        working = self._vr020_no_exact_duplicates(working, report)
        working = self._vr021_no_text_label_duplicates(working, report)

        # ── Content rules (VR-030 to VR-031) ───────────────────────────────
        working = self._vr030_arabic_content(working, report)
        working = self._vr031_text_max_length(working, report)

        # ── Class-balance check (post-cleaning) ─────────────────────────────
        abort = self._check_class_balance(working, report)
        if abort:
            return report

        report.cleaned_df = working.reset_index(drop=True)
        return report

    # ── VR-001: Required columns present ────────────────────────────────────

    def _vr001_schema_columns(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> tuple[bool, pd.DataFrame]:
        missing = [c for c in RAW_REQUIRED_COLUMNS if c not in df.columns]
        passed = len(missing) == 0
        msg = f"Missing columns: {missing}" if missing else ""
        report.results.append(
            RuleResult(
                rule_id="VR-001",
                rule_name="schema_columns_present",
                passed=passed,
                message=msg,
                action_taken="aborted" if not passed else "none",
            )
        )
        if not passed:
            logger.error("VR-001 ABORT — %s", msg)
            report.aborted = True
            report.abort_reason = f"VR-001: {msg}"
        return not passed, df

    # ── VR-002: No unexpected columns ───────────────────────────────────────

    def _vr002_no_extra_columns(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        extra = [c for c in df.columns if c not in RAW_REQUIRED_COLUMNS]
        passed = len(extra) == 0
        msg = f"Unexpected columns dropped: {extra}" if extra else ""
        report.results.append(
            RuleResult(
                rule_id="VR-002",
                rule_name="no_extra_columns",
                passed=passed,
                records_affected=0,
                message=msg,
                action_taken="warned" if not passed else "none",
            )
        )
        if not passed:
            logger.warning("VR-002 — %s", msg)

    # ── VR-003: ID uniqueness ────────────────────────────────────────────────

    def _vr003_id_unique(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> tuple[bool, pd.DataFrame]:
        n_dupes = df["ID"].duplicated().sum()
        passed = n_dupes == 0
        msg = f"{n_dupes} duplicate raw ID values detected (re-indexed to sequential IDs)" if not passed else ""
        report.results.append(
            RuleResult(
                rule_id="VR-003",
                rule_name="id_unique",
                passed=passed,
                records_affected=int(n_dupes),
                message=msg,
                action_taken="warned" if not passed else "none",
            )
        )
        if not passed:
            logger.warning("VR-003 — %s", msg)
            df = df.copy()
            df["ID"] = range(1, len(df) + 1)
        return False, df

    # ── VR-010: Text not null ────────────────────────────────────────────────

    def _vr010_text_not_null(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> pd.DataFrame:
        mask = df["Text"].isnull()
        n = int(mask.sum())
        passed = n == 0
        report.results.append(
            RuleResult(
                rule_id="VR-010",
                rule_name="text_not_null",
                passed=passed,
                records_affected=n,
                message=f"{n} null Text records dropped" if n else "",
                action_taken="dropped" if n else "none",
            )
        )
        if n:
            logger.info("VR-010 — Dropped %d null Text records", n)
        return df[~mask]

    # ── VR-011: Text not empty ───────────────────────────────────────────────

    def _vr011_text_not_empty(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> pd.DataFrame:
        mask = df["Text"].astype(str).str.strip() == ""
        n = int(mask.sum())
        passed = n == 0
        report.results.append(
            RuleResult(
                rule_id="VR-011",
                rule_name="text_not_empty",
                passed=passed,
                records_affected=n,
                message=f"{n} empty Text records dropped" if n else "",
                action_taken="dropped" if n else "none",
            )
        )
        if n:
            logger.info("VR-011 — Dropped %d empty Text records", n)
        return df[~mask]

    # ── VR-012: Text minimum length ──────────────────────────────────────────

    def _vr012_text_min_length(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> pd.DataFrame:
        lengths = df["Text"].astype(str).str.strip().str.len()
        mask = lengths < MIN_TEXT_LENGTH
        n = int(mask.sum())
        passed = n == 0
        report.results.append(
            RuleResult(
                rule_id="VR-012",
                rule_name="text_min_length",
                passed=passed,
                records_affected=n,
                message=f"{n} short-text records flagged (< {MIN_TEXT_LENGTH} chars)" if n else "",
                action_taken="flagged" if n else "none",
            )
        )
        if n:
            logger.info("VR-012 — Flagged %d short-text records", n)
            df = df.copy()
            df.loc[mask, "_flag_short_text"] = True
        return df

    # ── VR-013: Text UTF-8 encoding valid ────────────────────────────────────

    def _vr013_text_encoding_valid(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> pd.DataFrame:
        def is_valid_utf8(val: object) -> bool:
            try:
                str(val).encode("utf-8")
                return True
            except (UnicodeEncodeError, UnicodeDecodeError):
                return False

        mask = ~df["Text"].map(is_valid_utf8)
        n = int(mask.sum())
        passed = n == 0
        report.results.append(
            RuleResult(
                rule_id="VR-013",
                rule_name="text_encoding_valid",
                passed=passed,
                records_affected=n,
                message=f"{n} invalid-encoding records dropped" if n else "",
                action_taken="dropped" if n else "none",
            )
        )
        if n:
            logger.info("VR-013 — Dropped %d encoding-invalid records", n)
        return df[~mask]

    # ── VR-015: Label not null ───────────────────────────────────────────────

    def _vr015_label_not_null(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> pd.DataFrame:
        mask = df["Label"].isnull()
        n = int(mask.sum())
        passed = n == 0
        report.results.append(
            RuleResult(
                rule_id="VR-015",
                rule_name="label_not_null",
                passed=passed,
                records_affected=n,
                message=f"{n} null Label records dropped" if n else "",
                action_taken="dropped" if n else "none",
            )
        )
        if n:
            logger.info("VR-015 — Dropped %d null Label records", n)
        return df[~mask]

    # ── VR-014: Label value valid ────────────────────────────────────────────

    def _vr014_label_valid(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> pd.DataFrame:
        mask = ~df["Label"].isin(VALID_LABELS)
        n = int(mask.sum())
        passed = n == 0
        invalid_vals = df.loc[mask, "Label"].unique().tolist() if n else []
        report.results.append(
            RuleResult(
                rule_id="VR-014",
                rule_name="label_valid",
                passed=passed,
                records_affected=n,
                message=f"{n} invalid Label records dropped {invalid_vals}" if n else "",
                action_taken="dropped" if n else "none",
            )
        )
        if n:
            logger.info("VR-014 — Dropped %d invalid-label records: %s", n, invalid_vals)
        return df[~mask]

    # ── VR-020: No full-row duplicates ───────────────────────────────────────

    def _vr020_no_exact_duplicates(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> pd.DataFrame:
        mask = df.duplicated(subset=["Text", "Label", "Source"], keep="first")
        n = int(mask.sum())
        passed = n == 0
        report.results.append(
            RuleResult(
                rule_id="VR-020",
                rule_name="no_exact_duplicates",
                passed=passed,
                records_affected=n,
                message=f"{n} exact duplicate records dropped (kept first)" if n else "",
                action_taken="dropped" if n else "none",
            )
        )
        if n:
            logger.info("VR-020 — Dropped %d exact duplicate records", n)
        return df[~mask]

    # ── VR-021: No (Text, Label) duplicates ─────────────────────────────────

    def _vr021_no_text_label_duplicates(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> pd.DataFrame:
        mask = df.duplicated(subset=["Text", "Label"], keep="first")
        n = int(mask.sum())
        passed = n == 0
        report.results.append(
            RuleResult(
                rule_id="VR-021",
                rule_name="no_text_label_duplicates",
                passed=passed,
                records_affected=n,
                message=f"{n} (Text, Label) duplicate records dropped (kept first)" if n else "",
                action_taken="dropped" if n else "none",
            )
        )
        if n:
            logger.info("VR-021 — Dropped %d (Text, Label) duplicate records", n)
        return df[~mask]

    # ── VR-030: Arabic content present ──────────────────────────────────────

    def _vr030_arabic_content(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> pd.DataFrame:
        mask = ~df["Text"].astype(str).str.contains(ARABIC_UNICODE_RE)
        n = int(mask.sum())
        passed = n == 0
        report.results.append(
            RuleResult(
                rule_id="VR-030",
                rule_name="arabic_content_present",
                passed=passed,
                records_affected=n,
                message=f"{n} records with no Arabic characters flagged" if n else "",
                action_taken="flagged" if n else "none",
            )
        )
        if n:
            logger.info("VR-030 — Flagged %d non-Arabic records", n)
            df = df.copy()
            df.loc[mask, "_flag_no_arabic"] = True
        return df

    # ── VR-031: Text maximum length ──────────────────────────────────────────

    def _vr031_text_max_length(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> pd.DataFrame:
        mask = df["Text"].astype(str).str.len() > MAX_TEXT_LENGTH
        n = int(mask.sum())
        passed = n == 0
        report.results.append(
            RuleResult(
                rule_id="VR-031",
                rule_name="text_max_length",
                passed=passed,
                records_affected=n,
                message=f"{n} long-text records flagged (> {MAX_TEXT_LENGTH} chars)" if n else "",
                action_taken="flagged" if n else "none",
            )
        )
        if n:
            logger.info("VR-031 — Flagged %d long-text records", n)
            df = df.copy()
            df.loc[mask, "_flag_long_text"] = True
        return df

    # ── Class balance check ──────────────────────────────────────────────────

    def _check_class_balance(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> bool:
        """Returns True (abort) if balance thresholds are violated."""
        if not self.check_class_balance:
            return False

        if len(df) < self.min_total_records:
            msg = (
                f"Only {len(df):,} records remain after cleaning "
                f"(minimum required: {self.min_total_records:,})"
            )
            report.results.append(
                RuleResult(
                    rule_id="VR-040",
                    rule_name="min_total_records",
                    passed=False,
                    records_affected=self.min_total_records - len(df),
                    message=msg,
                    action_taken="aborted",
                )
            )
            logger.error("VR-040 ABORT — %s", msg)
            report.aborted = True
            report.abort_reason = f"VR-040: {msg}"
            return True

        counts = df["Label"].value_counts()
        min_count = counts.min()
        max_count = counts.max()

        # Per-class minimum
        under = counts[counts < self.min_samples_per_class]
        if not under.empty:
            msg = f"Classes below minimum ({self.min_samples_per_class}): {under.to_dict()}"
            report.results.append(
                RuleResult(
                    rule_id="VR-041",
                    rule_name="min_samples_per_class",
                    passed=False,
                    records_affected=int(under.sum()),
                    message=msg,
                    action_taken="aborted",
                )
            )
            logger.error("VR-041 ABORT — %s", msg)
            report.aborted = True
            report.abort_reason = f"VR-041: {msg}"
            return True

        # Imbalance ratio
        ratio = max_count / max(min_count, 1)
        passed = ratio <= MAX_IMBALANCE_RATIO
        report.results.append(
            RuleResult(
                rule_id="VR-042",
                rule_name="class_imbalance_ratio",
                passed=passed,
                message=f"Max/min class ratio: {ratio:.2f} (limit {MAX_IMBALANCE_RATIO})",
                action_taken="warned" if not passed else "none",
            )
        )
        if not passed:
            logger.warning("VR-042 — Imbalance ratio %.2f exceeds limit %.2f", ratio, MAX_IMBALANCE_RATIO)

        return False
