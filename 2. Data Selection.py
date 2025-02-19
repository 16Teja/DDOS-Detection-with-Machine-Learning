#!/usr/bin/env python
# coding: utf-8

"""
Data Selection:
- Loads AllTraining.parquet and AllTesting.parquet.
- Maps labels: "Benign" → 0, everything else → 1.
- Drops rows with missing numeric feature values.
- Saves CSV files for training and testing that still include src and dst
  for later reporting.
"""

import pandas as pd

def main():
    # Load combined Parquet files
    try:
        all_train_df = pd.read_parquet("AllTraining.parquet")
        print("Loaded AllTraining.parquet with shape:", all_train_df.shape)
    except Exception as e:
        print("Error loading 'AllTraining.parquet':", e)
        return

    try:
        all_test_df = pd.read_parquet("AllTesting.parquet")
        print("Loaded AllTesting.parquet with shape:", all_test_df.shape)
    except Exception as e:
        print("Error loading 'AllTesting.parquet':", e)
        return

    # Map labels (assumes the label column is named "Label")
    print("\n=== Label distribution before mapping (TRAIN) ===")
    print(all_train_df["Label"].value_counts(dropna=False))
    all_train_df["Label"] = all_train_df["Label"].apply(lambda x: 0 if x == "Benign" else 1)
    all_test_df["Label"]  = all_test_df["Label"].apply(lambda x: 0 if x == "Benign" else 1)
    print("\n=== Label distribution after mapping (TRAIN) ===")
    print(all_train_df["Label"].value_counts(dropna=False))

    # Determine feature columns to be used for training.
    # We want to exclude src and dst (non-numeric) from training features,
    # but keep them in the CSV for later reporting.
    all_columns = all_train_df.columns.tolist()
    # Exclude 'Label', 'src', and 'dst' to get numeric features
    numeric_cols = [c for c in all_columns if c not in ["Label", "src", "dst"]]

    # Drop rows with missing numeric feature values
    all_train_df.dropna(subset=numeric_cols, inplace=True)
    all_test_df.dropna(subset=numeric_cols, inplace=True)

    print("\n=== Shapes after dropping missing numeric features ===")
    print("Train shape:", all_train_df.shape)
    print("Test shape:", all_test_df.shape)

    # Save full CSV files (including src and dst)
    all_train_df.to_csv("X_train.csv", index=False)
    all_test_df.to_csv("X_test.csv", index=False)
    # Save labels separately if needed (or can be extracted from the CSV)
    all_train_df[["Label"]].to_csv("y_train.csv", index=False)
    all_test_df[["Label"]].to_csv("y_test.csv", index=False)
    print("\nSaved CSV files: X_train.csv, y_train.csv, X_test.csv, y_test.csv")

if __name__ == "__main__":
    main()
