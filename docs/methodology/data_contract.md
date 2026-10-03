# 📋 Data Contract — Arabic Emotion Detection Pipeline

> **Issue:** [DT-26 — Define data contract](https://linear.app/yahia-workspace/issue/DT-26/define-data-contract)
> **Milestone:** Cycle 1 — Project Validation & Data Foundation (M0 + M1)
> **Dataset:** `Houdna-khilouf/Dz-Emotion` (HuggingFace)
> **Grounded in:** DT-21 initial data inspection findings
> **Status:** ✅ Approved

---

## 1. Purpose & Scope

This document formalizes the **data contract** for the Arabic Emotion Detection Pipeline. It defines:

- What valid raw data looks like (input schema)
- What the cleaned, processed data must contain (output schema)
- Validation rules, violation actions, and the missing-value policy
- Constraints that make the contract directly implementable as validation code

All pipeline components (`src/validation/`, `src/preprocessing/`) must respect this contract. Violations detected at ingestion time must be handled according to the actions specified below before any downstream processing.

---

## 2. Raw Dataset Profile (from DT-21 Inspection)

| Property | Value |
|----------|-------|
| Source | `Houdna-khilouf/Dz-Emotion` (HuggingFace Hub) |
| File | `data/raw/dz_emotion_raw.csv` |
| Shape | 6,000 rows × 5 columns |
| Encoding | UTF-8 |
| Language | Algerian Darija (Arabic script + Arabizi) |
| Class balance | Perfectly balanced — 1,000 samples per class |

---

## 3. Input Schema (Raw CSV)

These are the columns as delivered by the raw source file.

| Column | Type | Required | Constraints | Notes |
|--------|------|----------|-------------|-------|
| `Unnamed: 0` | `int64` | No | ≥ 0, monotonically increasing | Artifact index — **must be dropped** before processing |
| `ID` | `int64` | Yes | > 0, unique | Original record identifier |
| `Text` | `string` | Yes | Not null, not empty, ≥ 3 chars, UTF-8 | Main input feature (Darija text) |
| `Label` | `string` | Yes | One of the 6 canonical emotion classes (see §3.1) | Target classification label |
| `Source` | `string` | Yes | Not null | Provenance metadata; used for stratified splitting |

> [!NOTE]
> Column names in the raw file are **capitalized** (`Text`, `Label`). All pipeline code must reference them by their exact casing. They will be normalised to lowercase in the processed output (§4).

### 3.1 Canonical Emotion Classes

The dataset uses the **Ekman 6-class** emotion taxonomy. Only the following label values are valid:

| Label value | Alias accepted | Class index |
|-------------|----------------|-------------|
| `Anger` | — | 0 |
| `Disgust` | — | 1 |
| `Fear` | — | 2 |
| `Happy` | `Happiness` | 3 |
| `Sadness` | — | 4 |
| `Surprise` | — | 5 |

> [!IMPORTANT]
> Any `Label` value not in the table above is considered **invalid** and the record must be **dropped**.

---

## 4. Output Schema (Processed Dataset)

After ingestion, cleaning, and preprocessing, each record in `data/processed/dz_emotion_processed.csv` must conform to the following schema:

| Column | Type | Nullable | Constraints | Description |
|--------|------|----------|-------------|-------------|
| `id` | `int64` | No | > 0, unique | Sourced from raw `ID` |
| `text_raw` | `string` | No | Not empty | Original verbatim text from `Text` |
| `text_clean` | `string` | No | Not empty, ≥ 3 chars | Normalised text (diacritics stripped, whitespace collapsed) |
| `text_tokens` | `string` | No | Valid JSON array of strings | Space-delimited token list (serialised) |
| `emotion` | `string` | No | One of 6 canonical labels (lowercase) | Normalised label |
| `emotion_id` | `int8` | No | 0 – 5 | Integer encoding of `emotion` |
| `text_length` | `int32` | No | ≥ 3 | Character count of `text_clean` |
| `word_count` | `int16` | No | ≥ 1 | Token count of `text_tokens` |
| `source` | `string` | No | Not null | Provenance from raw `Source` |

---

## 5. Validation Rules

Each rule has a unique identifier (`VR-XXX`), a human-readable description, a measurable condition, and a **violation action**.

### 5.1 Column-Level Rules

| Rule ID | Name | Condition | Violation Action |
|---------|------|-----------|-----------------|
| `VR-001` | `schema_columns_present` | All 5 raw columns (`Unnamed: 0`, `ID`, `Text`, `Label`, `Source`) exist in the file | **Abort pipeline** — schema mismatch is unrecoverable |
| `VR-002` | `no_extra_columns` | No columns beyond the 5 defined exist | **Log warning** — drop unexpected columns and continue |
| `VR-003` | `id_unique` | `ID` values are globally unique | **Abort pipeline** — signals file corruption or merge error |

### 5.2 Record-Level Rules

| Rule ID | Name | Condition | Violation Action |
|---------|------|-----------|-----------------|
| `VR-010` | `text_not_null` | `Text` is not `NaN` / `None` | **Drop record**, log count |
| `VR-011` | `text_not_empty` | `Text.strip()` is not an empty string | **Drop record**, log count |
| `VR-012` | `text_min_length` | `len(Text.strip()) >= 3` | **Flag for review** — add column `_flag_short_text=True`, keep record |
| `VR-013` | `text_encoding_valid` | `Text` can be encoded as UTF-8 without errors | **Drop record**, log count |
| `VR-014` | `label_valid` | `Label` is one of the 6 canonical values (case-sensitive) | **Drop record**, log count |
| `VR-015` | `label_not_null` | `Label` is not `NaN` / `None` | **Drop record**, log count |
| `VR-020` | `no_exact_duplicates` | No full-row duplicate (`ID`, `Text`, `Label`) | **Keep first occurrence**, drop duplicates, log count |
| `VR-021` | `no_text_label_duplicates` | No identical (`Text`, `Label`) pair appears more than once | **Keep first occurrence**, drop duplicates, log count |

> [!WARNING]
> `VR-020` and `VR-021` are separate rules. `VR-021` is the stricter de-duplication step. From DT-21 inspection, **3,199 text-level duplicates** were identified — this rule is expected to have significant impact.

### 5.3 Content Rules (Soft)

| Rule ID | Name | Condition | Violation Action |
|---------|------|-----------|-----------------|
| `VR-030` | `arabic_content_present` | `Text` contains at least one Arabic Unicode character (`\u0600`–`\u06FF`) | **Flag for review** — add column `_flag_no_arabic=True`, keep record |
| `VR-031` | `text_max_length` | `len(Text.strip()) <= 2000` | **Flag for review** — add column `_flag_long_text=True`, keep record |

---

## 6. Missing-Value Policy

| Column | Nulls observed (DT-21) | Policy |
|--------|------------------------|--------|
| `Text` | 1 | Drop the record (`VR-010`) |
| `Label` | 0 | Drop the record if null appears in future data (`VR-015`) |
| `ID` | 0 | Abort pipeline if null appears (`VR-003` extension) |
| `Source` | 0 | Fill with `"unknown"` if null, log warning |
| `Unnamed: 0` | 0 | Drop column unconditionally |

**General policy:** Imputation is not used for core fields (`Text`, `Label`). Records with missing critical fields are dropped and counted in the validation report. Soft fields (`Source`) may be filled with a sentinel value.

---

## 7. Class Distribution Contract

The input dataset must maintain approximate balance. The contract defines acceptable deviation:

| Constraint | Threshold | Violation Action |
|-----------|-----------|-----------------|
| Min samples per class | ≥ 400 after cleaning | **Abort pipeline** with class imbalance error |
| Max class imbalance ratio | ≤ 3:1 (max / min class count) | **Log warning**, continue |
| Expected total records (post-cleaning) | ≥ 2,500 | **Abort pipeline** — too much data lost to cleaning |

---

## 8. Train / Validation / Test Split Contract

| Split | Proportion | Stratified by | Min records |
|-------|-----------|---------------|-------------|
| Train | 70% | `emotion`, `source` | 1,960 |
| Validation | 15% | `emotion`, `source` | 420 |
| Test | 15% | `emotion`, `source` | 420 |

- **Random seed:** `42` (fixed for reproducibility)
- **No data leakage:** identical `Text` values must not appear in more than one split
- Test split is **held out** and must not be used during model selection or hyperparameter tuning

---

## 9. Implementability Notes

This contract is directly implementable via `src/validation/contract.py`. The recommended implementation approach:

```python
# Pseudocode — see src/validation/contract.py for implementation
VALID_LABELS = {"Anger", "Disgust", "Fear", "Happy", "Sadness", "Surprise"}
MIN_TEXT_LENGTH = 3
MAX_TEXT_LENGTH = 2000
MIN_SAMPLES_PER_CLASS = 500
MAX_IMBALANCE_RATIO = 3.0
RANDOM_SEED = 42

RULES = [
    SchemaRule("VR-001", required_cols=[...], action="abort"),
    NullRule("VR-010", column="Text", action="drop"),
    LengthRule("VR-012", column="Text", min_len=3, action="flag"),
    EnumRule("VR-014", column="Label", valid_values=VALID_LABELS, action="drop"),
    DeduplicationRule("VR-021", subset=["Text", "Label"], action="keep_first"),
    ArabicContentRule("VR-030", column="Text", action="flag"),
]
```

---

## 10. Change Log

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-10-03 | Mohamed Bouchoucha | Initial contract — grounded in DT-21 inspection results |
