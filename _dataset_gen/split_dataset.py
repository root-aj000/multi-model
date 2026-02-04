import pandas as pd
from sklearn.model_selection import train_test_split
import os
from pathlib import Path 

# ===================== CONFIG =====================
BASE_DIR = Path("_dataset_gen")
INPUT_CSV = BASE_DIR / "processed" / "Ads_with_images.csv"
OUTPUT_DIR = BASE_DIR / "dataset"

TRAIN_CSV = os.path.join(OUTPUT_DIR, "Train.csv")
VAL_CSV = os.path.join(OUTPUT_DIR, "Val.csv")
TEST_CSV = os.path.join(OUTPUT_DIR, "Test.csv")

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42

# ===================== SPLIT FUNCTION =====================

def split_dataset(input_csv, train_csv, val_csv, test_csv):
    """
    Split dataset into train (70%), validation (15%), and test (15%)
    """
    print("\n" + "="*70)
    print("DATASET SPLITTER")
    print("="*70)
    
    # Load data
    print(f"\n Loading: {input_csv}")
    df = pd.read_csv(input_csv)
    
    total_rows = len(df)
    print(f"   Total samples: {total_rows}")
    
    # First split: separate test set (15%)
    train_val_df, test_df = train_test_split(
        df, 
        test_size=TEST_RATIO, 
        random_state=RANDOM_SEED,
        shuffle=True
    )
    
    # Second split: separate train and validation from remaining 85%
    # val_ratio adjusted: 15% of total = 15/85 ≈ 0.176 of the remaining data
    val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        random_state=RANDOM_SEED,
        shuffle=True
    )
    
    # Save splits
    print(f"\n Saving splits...")
    train_df.to_csv(train_csv, index=False)
    print(f"Train: {len(train_df)} samples ({len(train_df)/total_rows*100:.1f}%) → {train_csv}")
    
    val_df.to_csv(val_csv, index=False)
    print(f"Val:   {len(val_df)} samples ({len(val_df)/total_rows*100:.1f}%) → {val_csv}")
    
    test_df.to_csv(test_csv, index=False)
    print(f"Test:  {len(test_df)} samples ({len(test_df)/total_rows*100:.1f}%) → {test_csv}")
    
    # Summary
    print(f"\n" + "="*70)
    print("SPLIT SUMMARY")
    print("="*70)
    print(f"Total samples:      {total_rows}")
    print(f"Training samples:   {len(train_df)} ({len(train_df)/total_rows*100:.2f}%)")
    print(f"Validation samples: {len(val_df)} ({len(val_df)/total_rows*100:.2f}%)")
    print(f"Test samples:       {len(test_df)} ({len(test_df)/total_rows*100:.2f}%)")
    print("="*70)
    print("Split completed successfully!\n")

if __name__ == "__main__":
    split_dataset(INPUT_CSV, TRAIN_CSV, VAL_CSV, TEST_CSV)