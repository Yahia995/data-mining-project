"""
src/preprocessing/arabic_cleaner.py
───────────────────────────────────
Arabic and Algerian Darija text normalization and cleaning module.

Implements Arabic-specific orthographic normalization, diacritic removal,
tatweel removal, social media noise removal, and tokenization.
"""

from __future__ import annotations

import json
import re
import logging
from typing import Sequence
import pandas as pd

logger = logging.getLogger(__name__)

# ── Unicode Constants & Regexes ─────────────────────────────────────────────

# Diacritics (Tashkeel) block: U+064B to U+065F, plus U+0670 (Superscript Alef)
DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670]")

# Tatweel / Kashida: U+0640
TATWEEL_RE = re.compile(r"\u0640")

# Alef variants (أ, إ, آ -> ا)
ALEF_VARIANTS_RE = re.compile(r"[أإآ]")

# Yeh variants (ى -> ي)
YEH_VARIANTS_RE = re.compile(r"ى")

# Teh Marbuta (ة -> ه)
TEH_MARBUTA_RE = re.compile(r"ة")

# URLs (http, https, www)
URL_RE = re.compile(r"https?://\S+|www\.\S+")

# User mentions (@username)
MENTION_RE = re.compile(r"@\w+")

# Email addresses
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

# Repeated characters (3 or more consecutive identical characters -> 2)
REPEATED_CHARS_RE = re.compile(r"(.)\1{2,}")

# Whitespace normalization (multiple spaces/tabs/newlines -> single space)
WHITESPACE_RE = re.compile(r"\s+")

# Non-Arabic punctuation / digits / noise cleaning regex (preserves Arabic characters, digits, basic spaces)
NON_ARABIC_REMOVAL_RE = re.compile(r"[^\u0600-\u06FF0-9\s]")


def remove_diacritics(text: str) -> str:
    """Strip Arabic diacritics (harakat / tashkeel)."""
    return DIACRITICS_RE.sub("", text)


def remove_tatweel(text: str) -> str:
    """Remove tatweel / kashida elongations (e.g., مـــــرحبا -> مرحبا)."""
    return TATWEEL_RE.sub("", text)


def normalize_alef(text: str) -> str:
    """Normalize Alef variants (أ, إ, آ) to bare Alef (ا)."""
    return ALEF_VARIANTS_RE.sub("ا", text)


def normalize_yeh(text: str) -> str:
    """Normalize Alef Maksura (ى) to Yeh (ي)."""
    return YEH_VARIANTS_RE.sub("ي", text)


def normalize_teh_marbuta(text: str) -> str:
    """Normalize Teh Marbuta (ة) to Heh (ه)."""
    return TEH_MARBUTA_RE.sub("ه", text)


def normalize_repeated_chars(text: str) -> str:
    """Collapse 3+ repeated characters down to 2 (e.g. ههههههه -> هه)."""
    return REPEATED_CHARS_RE.sub(r"\1\1", text)


def clean_social_noise(text: str) -> str:
    """Remove URLs, user mentions, and email addresses."""
    text = URL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    text = EMAIL_RE.sub("", text)
    return text


def clean_whitespace(text: str) -> str:
    """Collapse consecutive whitespace and trim leading/trailing spaces."""
    return WHITESPACE_RE.sub(" ", text).strip()


def normalize_arabic_text(
    text: str,
    strip_diacritics: bool = True,
    strip_tatweel: bool = True,
    norm_alef: bool = True,
    norm_yeh: bool = True,
    norm_teh_marbuta: bool = True,
    collapse_repeated: bool = True,
    remove_social: bool = True,
) -> str:
    """
    Apply full Arabic text normalization sequence.

    Parameters
    ----------
    text : str
        Input raw text string.
    strip_diacritics : bool
        Remove diacritics / tashkeel (default True).
    strip_tatweel : bool
        Remove tatweel elongation (default True).
    norm_alef : bool
        Normalize Alef variants to bare Alef (default True).
    norm_yeh : bool
        Normalize Alef Maksura to Yeh (default True).
    norm_teh_marbuta : bool
        Normalize Teh Marbuta to Heh (default True).
    collapse_repeated : bool
        Collapse 3+ repeated characters to 2 (default True).
    remove_social : bool
        Remove URLs, mentions, and emails (default True).

    Returns
    -------
    str
        Normalized and cleaned text.
    """
    if not isinstance(text, str):
        return ""

    if remove_social:
        text = clean_social_noise(text)
    if strip_diacritics:
        text = remove_diacritics(text)
    if strip_tatweel:
        text = remove_tatweel(text)
    if norm_alef:
        text = normalize_alef(text)
    if norm_yeh:
        text = normalize_yeh(text)
    if norm_teh_marbuta:
        text = normalize_teh_marbuta(text)
    if collapse_repeated:
        text = normalize_repeated_chars(text)

    return clean_whitespace(text)


def tokenize_arabic_text(text: str) -> list[str]:
    """
    Tokenize normalized text into a list of word tokens.

    Parameters
    ----------
    text : str
        Normalized input text.

    Returns
    -------
    list[str]
        List of whitespace-separated tokens.
    """
    tokens = text.split()
    return tokens


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess a validated raw DataFrame into the processed output schema
    defined in docs/methodology/data_contract.md Section 4.

    Input columns expected: ID, Text, Label, Source
    Output columns created:
      - id : int64
      - text_raw : string
      - text_clean : string
      - text_tokens : JSON string of token list
      - emotion : string (lowercase canonical label)
      - emotion_id : int8 (0-5 integer ID)
      - text_length : int32
      - word_count : int16
      - source : string

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame post-validation.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame conforming to output contract schema.
    """
    label_to_id = {
        "Anger": 0,
        "Disgust": 1,
        "Fear": 2,
        "Happy": 3,
        "Sadness": 4,
        "Surprise": 5,
    }

    records = []
    for idx, row in df.iterrows():
        raw_text = str(row["Text"])
        clean_text = normalize_arabic_text(raw_text)
        tokens = tokenize_arabic_text(clean_text)
        label_raw = str(row["Label"])
        emotion_str = label_raw.lower()
        emotion_id = label_to_id.get(label_raw, -1)
        row_id = int(row["ID"]) if "ID" in row and pd.notnull(row["ID"]) else idx + 1
        source_val = str(row["Source"]) if "Source" in row and pd.notnull(row["Source"]) else "unknown"

        records.append({
            "id": row_id,
            "text_raw": raw_text,
            "text_clean": clean_text,
            "text_tokens": json.dumps(tokens, ensure_ascii=False),
            "emotion": emotion_str,
            "emotion_id": emotion_id,
            "text_length": len(clean_text),
            "word_count": len(tokens),
            "source": source_val,
        })

    out_df = pd.DataFrame(records)
    out_df["id"] = out_df["id"].astype("int64")
    out_df["emotion_id"] = out_df["emotion_id"].astype("int8")
    out_df["text_length"] = out_df["text_length"].astype("int32")
    out_df["word_count"] = out_df["word_count"].astype("int16")
    return out_df
