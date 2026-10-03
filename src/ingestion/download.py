import os
import datetime
from datasets import load_dataset

def main():
    print("Downloading Dz-Emotion dataset...")
    
    dataset = load_dataset("Houdna-khilouf/Dz-Emotion")
    
    df = dataset['train'].to_pandas()
    
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "dz_emotion_raw.csv")
    df.to_csv(output_path, index=False)
    
    timestamp = datetime.datetime.now().isoformat()
    row_count = len(df)
    columns = list(df.columns)
    
    print("\n--- Download Metadata ---")
    print(f"Timestamp: {timestamp}")
    print(f"Row count: {row_count}")
    print(f"Columns:   {columns}")
    print(f"Saved to:  {output_path}\n")

if __name__ == "__main__":
    main()
