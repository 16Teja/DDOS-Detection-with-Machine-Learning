#!/usr/bin/env python
# coding: utf-8

"""
Feature Extraction:
- Combines all *training.parquet files into AllTraining.parquet 
  and all *testing.parquet files into AllTesting.parquet.
- These Parquet files are assumed to contain all fields extracted
  from Wireshark (e.g. src, dst, packets, duration, rate, mean, std, ack, psh, fin, urg, rst, sport, dport, etc.).
"""

import os
import glob
import pandas as pd

def combine_parquet_files(data_folder="."):
    # Find all Parquet files for training and testing
    training_files = glob.glob(os.path.join(data_folder, "*training.parquet"))
    testing_files = glob.glob(os.path.join(data_folder, "*testing.parquet"))

    print("Found training files:", training_files)
    print("Found testing files:", testing_files)

    # Combine training files
    if training_files:
        train_df_list = []
        for fpath in training_files:
            df_temp = pd.read_parquet(fpath)
            train_df_list.append(df_temp)
        all_train_df = pd.concat(train_df_list, ignore_index=True)
        all_train_df.to_parquet("AllTraining.parquet", index=False)
        print("All training data saved to 'AllTraining.parquet'.")
    else:
        print("No training files found!")
        all_train_df = pd.DataFrame()

    # Combine testing files
    if testing_files:
        test_df_list = []
        for fpath in testing_files:
            df_temp = pd.read_parquet(fpath)
            test_df_list.append(df_temp)
        all_test_df = pd.concat(test_df_list, ignore_index=True)
        all_test_df.to_parquet("AllTesting.parquet", index=False)
        print("All testing data saved to 'AllTesting.parquet'.")
    else:
        print("No testing files found!")
        all_test_df = pd.DataFrame()

    return all_train_df, all_test_df

def main():
    data_folder = "."  # Adjust if your Parquet files are in a different folder
    all_train_df, all_test_df = combine_parquet_files(data_folder)

    # Display shapes and a few rows for inspection
    if not all_train_df.empty:
        print("\n=== Training Data ===")
        print("Shape:", all_train_df.shape)
        print(all_train_df.head())
    if not all_test_df.empty:
        print("\n=== Testing Data ===")
        print("Shape:", all_test_df.shape)
        print(all_test_df.head())

if __name__ == "__main__":
    main()
