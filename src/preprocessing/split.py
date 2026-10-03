"""
src/preprocessing/split.py
───────────────────────────
Stratified Train / Validation / Test splitter for Arabic Emotion Detection Pipeline.

Performs a 70% Train / 15% Validation / 15% Test stratified split by `emotion`
and `source` with fixed random seed (42). Ensures no text leakage across splits.
"""

from __future__ import annotations

import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

RANDOM_SEED: int = 42
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15


def split_dataset(
    df: pd.DataFrame,
    random_state: int = RANDOM_SEED,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split processed DataFrame into Train (70%), Validation (15%), and Test (15%) partitions.

    Stratification is performed on the combined `emotion` and `source` key when possible,
    falling back to `emotion` stratification if individual stratify counts are < 2.

    Parameters
    ----------
    df : pd.DataFrame
        Processed DataFrame conforming to section 4 output schema.
    random_state : int
        Seed for deterministic reproducibility (default 42).
    train_ratio : float
        Proportion of dataset allocated to training (default 0.70).
    val_ratio : float
        Proportion allocated to validation (default 0.15).
    test_ratio : float
        Proportion allocated to test (default 0.15).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-5:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    # Combine emotion and source for stratification key
    df = df.copy()
    strat_key = df["emotion"] + "_" + df["source"].astype(str)
    counts = strat_key.value_counts()

    # Fallback to emotion-only if any group has < 2 samples
    if (counts < 2).any():
        logger.warning("Some (emotion, source) groups have < 2 samples. Falling back to emotion-only stratification.")
        strat_key = df["emotion"]

    temp_ratio = val_ratio + test_ratio  # 0.30
    test_val_relative_test_ratio = test_ratio / temp_ratio  # 0.50

    # First split: 70% Train, 30% Temp (Val + Test)
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=strat_key,
        random_state=random_state,
        shuffle=True,
    )

    # Re-compute stratification key for second split
    temp_strat_key = temp_df["emotion"] + "_" + temp_df["source"].astype(str)
    temp_counts = temp_strat_key.value_counts()
    if (temp_counts < 2).any():
        temp_strat_key = temp_df["emotion"]

    # Second split: 15% Val, 15% Test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_val_relative_test_ratio,
        stratify=temp_strat_key,
        random_state=random_state,
        shuffle=True,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info(
        "Dataset split complete — Train: %d, Val: %d, Test: %d (Total: %d)",
        len(train_df),
        len(val_df),
        len(test_df),
        len(df),
    )

    return train_df, val_df, test_df
