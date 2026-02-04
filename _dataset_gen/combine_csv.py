import pandas as pd
import glob
import os
from pathlib import Path
# 1. Get the absolute path of the folder where this script lives
script_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = Path(script_dir) / "Data_clean"
print(f"Save directory: {save_dir}")
# 2. Construct the full path to your csv_main folder
# Change "csv_main" below if your folder name is different
target_folder = os.path.join(script_dir, "csv")
search_path = os.path.join(target_folder, "*.csv")

print(f"Looking for CSVs in: {search_path}")

files = glob.glob(search_path)

# CHECK: Did we actually find files?
if not files:
    print("ERROR: No CSV files found!")
    print(f"Please make sure the folder '{target_folder}' exists and contains .csv files.")
    exit()

all_data = []

for file in files:
    print(f"Reading {os.path.basename(file)}...")
    try:
        df = pd.read_csv(file)
        # Optional: Skip empty files to prevent errors
        if not df.empty:
            all_data.append(df)
    except Exception as e:
        print(f"Skipping {file} due to error: {e}")

# Double check that we have data before merging
if len(all_data) == 0:
    print("No data loaded. All CSVs might be empty.")
else:
    # 4. Concatenate
    combined_csv = pd.concat(all_data, ignore_index=True)

    # 5. Export
    output_path = os.path.join(save_dir, "To_be_clean.csv")
    combined_csv.to_csv(output_path, index=False)
    print(f"Done! Saved as: {output_path}")