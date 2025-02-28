#!/usr/bin/env python
# coding: utf-8

"""
Data Selection:
- Loads X_train_res.csv and y_train_res.csv instead of AllTraining.parquet.
- Loads X_test_res.csv and y_test_res.csv instead of AllTesting.parquet.
- Maps labels: "Benign" → 0, everything else → 1.
- Drops rows with missing numeric feature values.
- Saves CSV files (X_train.csv, y_train.csv, X_test.csv, y_test.csv) that include
  src and dst for later reporting. The numeric features are expected to be:
  packets, duration, rate, mean, std, max, min, tcp, udp, dns, icmp, syn, ack,
  psh, fin, urg, rst, sport, dport.
"""

import pandas as pd

# Define the desired numeric feature order:
FEATURE_ORDER = ["packets", "duration", "rate", "mean", "std", "max", "min", 
                 "tcp", "udp", "dns", "icmp", "syn", "ack", "psh", "fin", 
                 "urg", "rst", "sport", "dport"]

def main():
    try:
        all_train_df = pd.read_csv("X_train_res.csv")
        all_train_df["Label"] = pd.read_csv("y_train_res.csv")["output"]
        print("Loaded X_train_res.csv and y_train_res.csv with shape:", all_train_df.shape)
    except Exception as e:
        print("Error loading 'X_train_res.csv' or 'y_train_res.csv':", e)
        return

    try:
        all_test_df = pd.read_csv("X_test_res.csv")
        all_test_df["Label"] = pd.read_csv("y_test_res.csv")["output"]
        print("Loaded X_test_res.csv and y_test_res.csv with shape:", all_test_df.shape)
    except Exception as e:
        print("Error loading 'X_test_res.csv' or 'y_test_res.csv':", e)
        return

    # Map labels (assumes the label column is named "Label")
    print("\n=== Label distribution before mapping (TRAIN) ===")
    print(all_train_df["Label"].value_counts(dropna=False))
    all_train_df["Label"] = all_train_df["Label"].apply(lambda x: 0 if x == "Benign" else 1)
    all_test_df["Label"] = all_test_df["Label"].apply(lambda x: 0 if x == "Benign" else 1)
    print("\n=== Label distribution after mapping (TRAIN) ===")
    print(all_train_df["Label"].value_counts(dropna=False))

    # Ensure that all desired numeric features are present
    missing = [col for col in FEATURE_ORDER if col not in all_train_df.columns]
    if missing:
        print("Warning: The following expected numeric features are missing in training data:", missing)
    
    # Drop rows with missing numeric feature values
    all_train_df.dropna(subset=FEATURE_ORDER, inplace=True)
    all_test_df.dropna(subset=FEATURE_ORDER, inplace=True)

    print("\n=== Shapes after dropping rows with missing numeric features ===")
    print("Train shape:", all_train_df.shape)
    print("Test shape:", all_test_df.shape)

    # Save CSV files including all columns (src, dst, numeric features, Label)
    all_train_df.to_csv("X_train.csv", index=False)
    all_test_df.to_csv("X_test.csv", index=False)
    all_train_df[["Label"]].to_csv("y_train.csv", index=False)
    all_test_df[["Label"]].to_csv("y_test.csv", index=False)
    print("\nSaved CSV files: X_train.csv, y_train.csv, X_test.csv, y_test.csv")

if __name__ == "__main__":
    main()
