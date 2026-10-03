"""
run_pipeline.py
───────────────
Main pipeline orchestrator for Arabic Emotion Detection Pipeline.

Executes end-to-end data pipeline:
1. Ingest raw dataset (from HuggingFace Hub or local cache)
2. Validate raw data against DataContract (VR-001 to VR-042)
3. Normalize and preprocess Arabic text into canonical schema
4. Perform stratified Train/Val/Test split (70/15/15)
5. Export clean datasets to data/processed/

Usage
-----
    python run_pipeline.py [--force-download]
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
import pandas as pd

from src.ingestion.download import download_dz_emotion
from src.validation.contract import DataContract
from src.preprocessing.arabic_cleaner import process_dataframe
from src.preprocessing.split import split_dataset

# Set up clean logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_pipeline")


def run_pipeline(force_download: bool = False) -> None:
    """Execute the complete data mining pipeline from raw download to processed splits."""
    logger.info("=" * 60)
    logger.info("Starting Arabic Emotion Detection Data Pipeline")
    logger.info("=" * 60)

    raw_csv_path = "data/raw/dz_emotion_raw.csv"

    # Step 1: Data Ingestion
    if force_download or not os.path.exists(raw_csv_path):
        logger.info("Step 1/4: Ingesting dataset from Hugging Face Hub...")
        download_dz_emotion()
    else:
        logger.info("Step 1/4: Raw dataset found at %s (skipping download)", raw_csv_path)

    raw_df = pd.read_csv(raw_csv_path)
    logger.info("Loaded raw dataset: %d records, %d columns", len(raw_df), len(raw_df.columns))

    # Step 2: Data Validation (Data Contract)
    logger.info("Step 2/4: Running Data Contract Validation (DT-26)...")
    contract = DataContract(check_class_balance=True)
    report = contract.validate(raw_df)

    if report.aborted:
        logger.error("Data contract validation ABORTED: %s", report.abort_reason)
        sys.exit(1)

    cleaned_raw_df = report.cleaned_df
    logger.info("Data contract PASSED. Cleaned raw records remaining: %d", len(cleaned_raw_df))

    # Step 3: Text Normalization & Preprocessing
    logger.info("Step 3/4: Normalizing Arabic text & transforming schema...")
    processed_df = process_dataframe(cleaned_raw_df)

    os.makedirs("data/processed", exist_ok=True)
    processed_csv_path = "data/processed/dz_emotion_processed.csv"
    processed_df.to_csv(processed_csv_path, index=False, encoding="utf-8")
    logger.info("Saved processed dataset: %s (%d records)", processed_csv_path, len(processed_df))

    # Step 4: Stratified Train / Val / Test Split
    logger.info("Step 4/4: Performing 70/15/15 Stratified Split...")
    train_df, val_df, test_df = split_dataset(processed_df, random_state=42)

    train_path = "data/processed/train.csv"
    val_path = "data/processed/val.csv"
    test_path = "data/processed/test.csv"

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    logger.info("Exported splits:")
    logger.info("  - Train : %s (%d samples)", train_path, len(train_df))
    logger.info("  - Val   : %s (%d samples)", val_path, len(val_df))
    logger.info("  - Test  : %s (%d samples)", test_path, len(test_df))

    logger.info("=" * 60)
    logger.info("Pipeline Execution Completed Successfully! 🎉")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arabic Emotion Detection Data Pipeline")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-downloading the raw dataset from HuggingFace",
    )
    args = parser.parse_args()
    run_pipeline(force_download=args.force_download)
