#!/usr/bin/env python3
"""
merge_label_arp_only.py

Merge CICFlowMeter outputs for benign + mitm-arpspoofing pcaps,
assign numeric labels (benign=0, arp=1) and save combined CSV.

Place this script in the parent folder of your `output/` folder,
which should contain the CICFlowMeter CSV/XLSX files.
"""

import pandas as pd
import glob
import os
import sys

# config
INPUT_GLOB_CSV = "output/*_Flow.csv"
INPUT_GLOB_XLSX = "output/*_Flow.xlsx"
OUTPUT_FILE = "combined_arp_dataset.csv"

def read_table(path):
    """Read csv or xlsx into DataFrame."""
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    else:
        # read first sheet of excel
        return pd.read_excel(path, sheet_name=0)

def detect_label_from_filename(fname):
    """Return numeric label: benign -> 0, arp (mitm-arpspoof) -> 1"""
    fn = os.path.basename(fname).lower()
    if "benign" in fn or "normal" in fn:
        return 0
    if "mitm-arpspoof" in fn or "arpspoof" in fn or "arp" in fn:
        return 1
    # fallback: if unsure, raise so you can check files
    raise ValueError(f"Unable to auto-detect label for file: {fname}")

def main():
    files = sorted(glob.glob(INPUT_GLOB_CSV) + glob.glob(INPUT_GLOB_XLSX))
    if not files:
        print("No CICFlowMeter output files found in 'output/'.")
        print("Make sure your CSV/XLSX files match '*_Flow.csv' or '*_Flow.xlsx'.")
        sys.exit(1)

    dfs = []
    print(f"Found {len(files)} files. Listing:")
    for f in files:
        print("  -", f)
    print()

    for f in files:
        try:
            label = detect_label_from_filename(f)
        except ValueError as e:
            print("ERROR:", e)
            print("Skipping file. If this file should be included, rename it to include 'benign' or 'mitm-arpspoof' in the filename.")
            continue

        print(f"Loading {os.path.basename(f)} (label {label}) ...")
        df = read_table(f)

        # optional: add source filename column so you can trace back flows later
        #df["source_file"] = os.path.basename(f)

        # numeric label column (0 or 1)
        df["Label"] = label

        dfs.append(df)

    if not dfs:
        print("No dataframes were loaded. Exiting.")
        sys.exit(1)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined rows: {combined.shape[0]}, columns: {combined.shape[1]}")

    # Save as CSV (no preprocessing)
    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved combined dataset to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
