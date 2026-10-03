import os
import datetime
from datasets import load_dataset

def download_dz_emotion(output_path: str = "data/raw/dz_emotion_raw.csv") -> str:
    """Download Dz-Emotion dataset from HuggingFace Hub and save to CSV."""
    print("Downloading Dz-Emotion dataset from Hugging Face Hub...")
    dataset = load_dataset("Houdna-khilouf/Dz-Emotion")
    df = dataset["train"].to_pandas()

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_path, index=False)

    timestamp = datetime.datetime.now().isoformat()
    row_count = len(df)
    columns = list(df.columns)

    print("\n--- Download Metadata ---")
    print(f"Timestamp: {timestamp}")
    print(f"Row count: {row_count}")
    print(f"Columns:   {columns}")
    print(f"Saved to:  {output_path}\n")
    return output_path


def main():
    download_dz_emotion()


if __name__ == "__main__":
    main()
