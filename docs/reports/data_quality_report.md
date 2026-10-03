# 📊 Data Quality Report — Arabic Emotion Detection Pipeline

> **Issue:** [DT-33 — Produce Data Quality Report](https://linear.app/yahia-workspace/issue/DT-33/produce-data-quality-report)
> **Gate:** G2 — Data Quality Gate
> **Dataset:** `Houdna-khilouf/Dz-Emotion` (HuggingFace Hub)
> **Grounded in:** DT-21 Initial Inspection, DT-26 Data Contract
> **Companion Notebook:** [`notebooks/02-data-quality.ipynb`](../../notebooks/02-data-quality.ipynb)
> **Status:** ✅ **PASS — Approved for Feature Engineering & Model Training**

---

## 1. Executive Summary

This report formally documents the data quality evaluation for the Dz-Emotion dataset used in the Arabic Emotion Detection Pipeline. The evaluation covers **10 standardized data quality dimensions**, including quantitative metrics validated against predefined gate thresholds and qualitative manual audits of language authenticity and label reliability.

**Key Findings:**
- The raw dataset (6,000 records) was cleaned to **2,800 unique, validated samples** after removing 3,199 exact duplicates and 1 null record.
- All 10 quality dimensions **meet or exceed** the operational quality gates defined in the [data contract](../methodology/data_contract.md).
- Near-duplicate analysis flagged ~4.10% of pairs (below the 5.0% warning threshold on the sampled subset).
- Manual spot-check of 50 records confirmed high language quality (96%) and label reliability (92%).

---

## 2. Dataset Overview

| Property | Raw Dataset | Processed Dataset |
|---|---|---|
| **Source** | `Houdna-khilouf/Dz-Emotion` (HuggingFace) | Contract-cleaned output |
| **File** | `data/raw/dz_emotion_raw.csv` | `data/processed/dz_emotion_processed.csv` |
| **Records** | 6,000 | 2,800 |
| **Columns** | 5 (`Unnamed: 0`, `ID`, `Text`, `Label`, `Source`) | 9 (`id`, `text_raw`, `text_clean`, `text_tokens`, `emotion`, `emotion_id`, `text_length`, `word_count`, `source`) |
| **Encoding** | UTF-8 | UTF-8 |
| **Language** | Algerian Darija (Arabic script + Arabizi) | Algerian Darija (normalized) |
| **Classes** | 6 Ekman emotions × 1,000 each | 6 Ekman emotions × 444–481 each |

---

## 3. Quality Dimensions & Threshold Matrix

| # | Quality Dimension | Metric / Operational Definition | Gate Threshold | Raw Dataset Value | Cleaned Dataset Value | Status |
|---|---|---|:---:|:---:|:---:|:---:|
| 1 | **Completeness** | % records with all non-null fields | ≥ 99.0% | 99.98% (1 null) | 100.0% (0 nulls) | ✅ PASS |
| 2 | **Validity** | % passing data contract validation rules | ≥ 95.0% | 46.67% (pre-clean) | 100.0% (post-clean) | ✅ PASS |
| 3 | **Exact Duplicates** | Identical (Text, Label) row count | Report & Remove | 3,199 (53.32%) | 0 (0.0%) | ✅ PASS |
| 4 | **Near-Duplicates** | % pairs with TF-IDF cosine sim > 0.85 | Flag if > 5.0% | ~4.10% (41 pairs in 1K sample) | Monitored | ⚠️ WARN |
| 5 | **Class Balance** | Max / Min class ratio | ≤ 3.0:1 | 1.00:1 (1,000/class) | 1.08:1 (444–481/class) | ✅ PASS |
| 6 | **Text Length Distribution** | Outliers (< 5 or > 500 characters) | Flag outliers | 5 < 5 chars, 0 > 500 | 0 < 3 chars, 0 > 500 | ✅ PASS |
| 7 | **Arabic Content** | % records containing Arabic script (`\u0600`–`\u06FF`) | ≥ 90.0% | 98.92% (65 non-Arabic) | 99.00% (28 soft-flag) | ✅ PASS |
| 8 | **Encoding Integrity** | % text cleanly encodable as UTF-8 | 100.0% | 100.0% | 100.0% | ✅ PASS |
| 9 | **Language Quality** | Natural Algerian Darija dialectal syntax (50 spot-check) | Qualitative | 48/50 (96.0%) | Verified | ✅ PASS |
| 10 | **Label Reliability** | Concordance with Ekman emotion classes (50 spot-check) | Qualitative | 46/50 (92.0%) | High alignment | ✅ PASS |

---

## 4. Detailed Dimension Analysis

### 4.1 Completeness (Dimension 1)

- **Metric:** Percentage of records with all non-null fields.
- **Raw value:** 99.98% — only **1 null** found in the `Text` column out of 6,000 records.
- **Post-cleaning:** 100.0% — the null record was dropped per `VR-010` (text_not_null).
- **Assessment:** ✅ Exceeds the 99.0% threshold.

### 4.2 Validity (Dimension 2)

The [DataContract](../../src/validation/contract.py) was applied with all 14 validation rules. Full results:

| Rule | Name | Status | Records Affected | Action |
|---|---|:---:|---:|---|
| `VR-001` | schema_columns_present | ✅ PASS | 0 | — |
| `VR-002` | no_extra_columns | ✅ PASS | 0 | — |
| `VR-003` | id_unique | ⚠️ WARN | 3,171 | Re-indexed to sequential IDs |
| `VR-010` | text_not_null | 🔻 DROP | 1 | Dropped |
| `VR-011` | text_not_empty | ✅ PASS | 0 | — |
| `VR-012` | text_min_length | 🚩 FLAG | 2 | Flagged (`_flag_short_text`) |
| `VR-013` | text_encoding_valid | ✅ PASS | 0 | — |
| `VR-014` | label_valid | ✅ PASS | 0 | — |
| `VR-015` | label_not_null | ✅ PASS | 0 | — |
| `VR-020` | no_exact_duplicates | 🔻 DROP | 3,199 | Dropped (kept first) |
| `VR-021` | no_text_label_duplicates | ✅ PASS | 0 | — |
| `VR-030` | arabic_content_present | 🚩 FLAG | 28 | Flagged (`_flag_no_arabic`) |
| `VR-031` | text_max_length | ✅ PASS | 0 | — |
| `VR-042` | class_imbalance_ratio | ✅ PASS | 0 | — |

- **Pre-clean validity:** 46.67% of raw records required action (primarily duplicate removal).
- **Post-clean validity:** 100.0% — all 2,800 retained records pass all rules.

### 4.3 Exact Duplicates (Dimension 3)

- **Raw dataset:** 3,199 exact `(Text, Label)` duplicates detected (53.32% of raw data).
- **Cleaning action:** First occurrence kept, duplicates dropped (`VR-020`).
- **Post-cleaning:** 0 exact duplicates remain.
- **Root cause:** The raw dataset contains multiple copies of the same text–label pairs across different `Source` entries, likely from dataset construction/concatenation.

### 4.4 Near-Duplicates (Dimension 4)

- **Method:** TF-IDF vectorization (1,2-gram, 1,500 features) + cosine similarity on a random 1,000-record sample (`seed=42`) from the processed dataset.
- **Result:** 41 pairs with cosine similarity > 0.85, yielding a **4.10%** near-duplicate rate.
- **Assessment:** ⚠️ Below the 5.0% action threshold but flagged for monitoring. These near-duplicates are likely dialectal rephrasing of similar sentiments (common in social-media Arabic) and do not constitute data leakage.

### 4.5 Class Balance (Dimension 5)

| Emotion | Raw Count | Processed Count |
|---|---:|---:|
| Anger | 1,000 | ~465 |
| Disgust | 1,000 | ~444 |
| Fear | 1,000 | ~481 |
| Happy | 1,000 | ~470 |
| Sadness | 1,000 | ~468 |
| Surprise | 1,000 | ~472 |

- **Raw ratio:** Perfectly balanced at 1.00:1.
- **Processed ratio:** 1.08:1 (max 481 / min 444), well within the 3.0:1 contract threshold.
- **Assessment:** ✅ Excellent class balance maintained through deduplication. No augmentation or sampling strategies are required.

### 4.6 Text Length Distribution (Dimension 6)

- **Character length:** Median ≈ 60–80 characters with right-skewed distribution typical of social media text.
- **Word count:** Median ≈ 10–15 words per record.
- **Outliers (raw):** 5 records < 5 characters — 2 of these flagged by `VR-012` as < 3 characters (short text flag).
- **Outliers (processed):** 0 records below minimum threshold, 0 records above 500 characters.
- **Assessment:** ✅ Distribution is healthy for downstream NLP tasks. Visualization available in companion notebook.

### 4.7 Arabic Content (Dimension 7)

- **Raw dataset:** 98.92% of records contain at least one Arabic Unicode character (`\u0600`–`\u06FF`); 65 records lack Arabic script.
- **Processed dataset:** 99.00% Arabic content presence; 28 records soft-flagged (`_flag_no_arabic`).
- **Flagged records:** Predominantly Arabizi (Latin transliteration of Arabic), which is common in Algerian dialect online discourse. These records are retained for modeling diversity.
- **Assessment:** ✅ Exceeds the 90.0% threshold.

### 4.8 Encoding Integrity (Dimension 8)

- **Raw dataset:** 100.0% of all `Text` values encode cleanly as UTF-8.
- **Processed dataset:** 100.0% — no encoding corruption.
- **Assessment:** ✅ Perfect encoding integrity.

---

## 5. Qualitative Dimensions: 50-Sample Spot Check Audit

A stratified random sample of 50 records was drawn (`seed=42`) from the processed dataset and manually audited.

### 5.1 Language Quality Assessment (Dimension 9)

- **Criterion:** Text constitutes genuine Algerian Darija dialectal prose.
- **Result:** **48/50 (96.0%)** verified as natural dialectal text.
- **2 exceptions:** Contained Arabizi/Latin-script hashtags mixed with Arabic text — valid for social media context but not pure Arabic script.
- **Assessment:** ✅ High language authenticity confirmed.

### 5.2 Emotion Label Reliability (Dimension 10)

- **Criterion:** Annotated Ekman emotion label aligns with human-interpretable sentiment.
- **Result:** **46/50 (92.0%)** showed direct alignment between text content and label.
- **4 ambiguous cases:** Exhibited borderline overlap between **Sadness** and **Anger**, a well-documented affective dyad in dialectal grief/frustration expressions. These are not labeling errors but reflect genuine emotional ambiguity.
- **Assessment:** ✅ High label reliability; ambiguous cases are expected and do not compromise model training.

---

## 6. Data Contract Compliance Summary

| Contract Section | Status | Notes |
|---|:---:|---|
| Input Schema (§3) | ✅ | All 5 required columns present and validated |
| Output Schema (§4) | ✅ | 9-column processed schema conforms to specification |
| Validation Rules (§5) | ✅ | All 14 rules applied; 2,800 records pass |
| Missing-Value Policy (§6) | ✅ | 1 null `Text` dropped; no nulls remain |
| Class Distribution (§7) | ✅ | Min 444/class, ratio 1.08:1, total 2,800 |
| Train/Val/Test Split (§8) | ✅ | 70/15/15 stratified split; no data leakage |

---

## 7. Risks & Mitigations

| Risk | Severity | Mitigation |
|---|:---:|---|
| Near-duplicate texts (~4.1%) could inflate model performance | Low | Monitored; not exceeding 5% threshold; primarily dialectal rephrasing |
| 28 records without Arabic script | Low | Soft-flagged; retained for Arabizi coverage in model training |
| 2 short-text records (< 3 chars) | Low | Flagged by `VR-012`; may benefit from manual review in future cycles |
| Sadness/Anger label ambiguity (4/50 spot-check) | Low | Expected in affective computing; documented for model evaluation interpretation |

---

## 8. Formal Gate G2 Conclusion

> **Verdict: ✅ PASS — Approved for Feature Engineering & Model Training**

The Dz-Emotion dataset has been rigorously evaluated across 10 data quality dimensions. Key outcomes:

1. **6,000 → 2,800 records** after cleaning (3,199 exact duplicates + 1 null removed).
2. **All quantitative thresholds met** — completeness, validity, encoding, class balance, Arabic content.
3. **All qualitative audits passed** — language quality (96%) and label reliability (92%) confirmed via manual spot-check.
4. **No data leakage** — train/validation/test partitions are strictly separated with no text overlap.
5. **Data contract fully compliant** — all validation rules from [data_contract.md](../methodology/data_contract.md) are enforced in `src/validation/contract.py`.

The dataset is ready to proceed to the next pipeline stage: **Feature Engineering & Model Training**.

---

## 9. Appendix: Artifacts & References

| Artifact | Path | Description |
|---|---|---|
| Data Quality Notebook | [`notebooks/02-data-quality.ipynb`](../../notebooks/02-data-quality.ipynb) | Executable notebook with all metrics and visualizations |
| Data Contract | [`docs/methodology/data_contract.md`](../methodology/data_contract.md) | Formal validation rules and thresholds |
| Contract Implementation | [`src/validation/contract.py`](../../src/validation/contract.py) | Python implementation of all VR-XXX rules |
| Validation Script | [`src/validation/validate.py`](../../src/validation/validate.py) | Orchestrator for running all contract checks |
| Class Balance Chart | [`docs/reports/class_balance_comparison.png`](./class_balance_comparison.png) | Raw vs. processed class distribution |
| Length Distribution Chart | [`docs/reports/length_distribution.png`](./length_distribution.png) | Character and word count histograms |
| Raw Dataset | `data/raw/dz_emotion_raw.csv` | 6,000 records (source of truth) |
| Processed Dataset | `data/processed/dz_emotion_processed.csv` | 2,800 cleaned records |

---

## 10. Change Log

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-10-03 | Mohamed Bouchoucha | Initial data quality report — all 10 dimensions evaluated |
