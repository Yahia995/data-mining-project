"""
tests/test_preprocessing.py
─────────────────────────────
Unit tests for Arabic text normalization, tokenization, and DataFrame preprocessing.
"""

import json
import pandas as pd
import pytest
from src.preprocessing import (
    normalize_arabic_text,
    tokenize_arabic_text,
    remove_diacritics,
    remove_tatweel,
    normalize_alef,
    normalize_yeh,
    normalize_teh_marbuta,
    normalize_repeated_chars,
    clean_social_noise,
    process_dataframe,
)


def test_remove_diacritics():
    text = "أَنَا سَعِيدٌ جِدًّا"
    cleaned = remove_diacritics(text)
    assert "َ" not in cleaned
    assert "ً" not in cleaned
    assert "أنَا" not in cleaned
    assert cleaned == "أنا سعيد جدا"


def test_remove_tatweel():
    text = "جمــــــــيل"
    cleaned = remove_tatweel(text)
    assert cleaned == "جميل"


def test_normalize_alef():
    text = "أحمد إبراهيم آدم"
    cleaned = normalize_alef(text)
    assert cleaned == "احمد ابراهيم ادم"


def test_normalize_yeh():
    text = "على إلى شاطئ"
    cleaned = normalize_yeh(text)
    assert "علي" in cleaned
    assert "إلي" in cleaned


def test_normalize_teh_marbuta():
    text = "مدرسة جميلة"
    cleaned = normalize_teh_marbuta(text)
    assert cleaned == "مدرسه جميله"


def test_normalize_repeated_chars():
    text = "ههههههههه شكرااااا"
    cleaned = normalize_repeated_chars(text)
    assert cleaned == "هه شكراا"


def test_clean_social_noise():
    text = "تابعنا على https://example.com و @user123 contact@test.com"
    cleaned = clean_social_noise(text)
    assert "https://" not in cleaned
    assert "@user123" not in cleaned
    assert "contact@test.com" not in cleaned


def test_tokenize_arabic_text():
    text = "هذا نص عربي بسيط"
    tokens = tokenize_arabic_text(text)
    assert tokens == ["هذا", "نص", "عربي", "بسيط"]


def test_process_dataframe():
    raw_df = pd.DataFrame([
        {
            "Unnamed: 0": 0,
            "ID": 101,
            "Text": "أَنَا سَعِيدٌ جِدًّا هههههه @user https://test.com",
            "Label": "Happy",
            "Source": "twitter",
        },
        {
            "Unnamed: 0": 1,
            "ID": 102,
            "Text": "هَذَا أَمْرٌ مُغْضِبٌ جِدًّا",
            "Label": "Anger",
            "Source": "facebook",
        },
    ])

    out_df = process_dataframe(raw_df)

    assert set(out_df.columns) == {
        "id",
        "text_raw",
        "text_clean",
        "text_tokens",
        "emotion",
        "emotion_id",
        "text_length",
        "word_count",
        "source",
    }
    assert len(out_df) == 2
    assert out_df.iloc[0]["emotion"] == "happy"
    assert out_df.iloc[0]["emotion_id"] == 3
    assert out_df.iloc[1]["emotion"] == "anger"
    assert out_df.iloc[1]["emotion_id"] == 0
    assert "@user" not in out_df.iloc[0]["text_clean"]
    assert "https://" not in out_df.iloc[0]["text_clean"]

    tokens = json.loads(out_df.iloc[0]["text_tokens"])
    assert isinstance(tokens, list)
    assert len(tokens) == out_df.iloc[0]["word_count"]


def test_split_dataset():
    from src.preprocessing import split_dataset
    labels = ["happy", "anger", "disgust", "fear", "sadness", "surprise"]
    rows = []
    for i in range(1000):
        rows.append({
            "id": i + 1,
            "text_raw": f"نص {i}",
            "text_clean": f"نص {i}",
            "text_tokens": json.dumps(["نص", str(i)]),
            "emotion": labels[i % len(labels)],
            "emotion_id": i % len(labels),
            "text_length": 5,
            "word_count": 2,
            "source": "twitter",
        })
    df = pd.DataFrame(rows)
    train_df, val_df, test_df = split_dataset(df, random_state=42)

    assert len(train_df) == 700
    assert len(val_df) == 150
    assert len(test_df) == 150
    assert set(train_df["id"]).isdisjoint(set(val_df["id"]))
    assert set(train_df["id"]).isdisjoint(set(test_df["id"]))
    assert set(val_df["id"]).isdisjoint(set(test_df["id"]))

