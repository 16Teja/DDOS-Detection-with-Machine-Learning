#!/usr/bin/env python
# coding: utf-8

"""
ddosdetect.py:
- Loads the saved CNN+Transformer model (cnn_transformer.pt).
- Reads a new CSV (or Parquet) file containing Wireshark-derived features.
- Uses specified src and dst columns (if available) for final reporting.
- Drops src and dst before scaling and inference.
- Writes predictions to prediction.txt with columns: src, dst, prediction.
  
Usage:
  python ddosdetect.py <input_csv_or_parquet> [src_column] [dst_column]
Example:
  python ddosdetect.py new_traffic.csv src dst
"""

import sys
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------
# Model Definition (same as in Train.py)
# ------------------------------------------------
class MultiScaleCNN(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_cnn=0.3):
        super(MultiScaleCNN, self).__init__()
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)
        self.bn3   = nn.BatchNorm1d(out_channels)
        self.bn5   = nn.BatchNorm1d(out_channels)
        self.bn7   = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout_cnn)
    def forward(self, x):
        x3 = F.relu(self.bn3(self.conv3(x)))
        x5 = F.relu(self.bn5(self.conv5(x)))
        x7 = F.relu(self.bn7(self.conv7(x)))
        out = torch.cat([x3, x5, x7], dim=1)
        out = self.dropout(out)
        return out

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, n_heads, num_layers, dim_feedforward=512, dropout=0.3):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    def forward(self, x):
        x = self.embedding(x)
        out = self.transformer_encoder(x)
        return out

class HybridModel(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim, n_heads, num_layers, num_classes=1):
        super(HybridModel, self).__init__()
        self.cnn = MultiScaleCNN(in_channels, out_channels, dropout_cnn=0.3)
        self.transformer = TransformerEncoder(
            input_dim=3*out_channels,
            embed_dim=embed_dim,
            n_heads=n_heads,
            num_layers=num_layers,
            dropout=0.3
        )
        self.fc = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.permute(0, 2, 1)
        transformer_out = self.transformer(cnn_out)
        pooled = torch.mean(transformer_out, dim=1)
        logits = self.fc(pooled)
        if logits.shape[1] == 1:
            return torch.sigmoid(logits)
        else:
            return logits

# ------------------------------------------------
# Dataset & Collate for Inference
# ------------------------------------------------
class DDoSDataset(torch.utils.data.Dataset):
    def __init__(self, features):
        self.features = torch.tensor(features, dtype=torch.float32)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx]

def collate_fn(batch):
    feats = torch.stack(batch, dim=0)  # [batch_size, num_features]
    feats = feats.unsqueeze(1)        # [batch_size, 1, num_features]
    return feats

def main():
    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python ddosdetect.py <input_csv_or_parquet> [src_column] [dst_column]")
        sys.exit(1)

    input_file = sys.argv[1]
    src_col = sys.argv[2] if len(sys.argv) > 2 else None
    dst_col = sys.argv[3] if len(sys.argv) > 3 else None

    # Load the saved model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model on device:", device)
    model = HybridModel(
        in_channels=1,
        out_channels=64,
        embed_dim=128,
        n_heads=4,
        num_layers=3,
        num_classes=1
    ).to(device)
    model.load_state_dict(torch.load("cnn_transformer.pt", map_location=device))
    model.eval()
    print("Loaded 'cnn_transformer.pt' successfully.")

    # Read new data (CSV or Parquet)
    if input_file.endswith(".csv"):
        df = pd.read_csv(input_file)
    elif input_file.endswith(".parquet"):
        df = pd.read_parquet(input_file)
    else:
        print("File format not recognized (must be .csv or .parquet). Exiting.")
        sys.exit(1)

    # Save src and dst for reporting if provided and available
    if src_col and src_col in df.columns:
        src_vals = df[src_col].values
    else:
        src_vals = [None] * len(df)
    if dst_col and dst_col in df.columns:
        dst_vals = df[dst_col].values
    else:
        dst_vals = [None] * len(df)

    # Drop non-numeric columns (Label, src, dst) for inference
    drop_cols = []
    for col in ["Label", src_col, dst_col]:
        if col in df.columns:
            drop_cols.append(col)
    features_df = df.drop(columns=drop_cols, errors="ignore")
    features_df.dropna(inplace=True)

    # Scale features (in practice, use the scaler from training)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df.values)

    # Create DataLoader for inference
    dataset = DDoSDataset(features_scaled)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Run inference
    predictions = []
    with torch.no_grad():
        for batch_features in loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)  # [batch_size, 1]
            preds = (outputs > 0.5).float().squeeze()  # threshold for binary classification
            predictions.extend(preds.cpu().numpy())

    final_labels = ["malicious" if p == 1.0 else "benign" for p in predictions]

    # Write results to prediction.txt
    results_df = pd.DataFrame({
        "src": src_vals[:len(final_labels)],
        "dst": dst_vals[:len(final_labels)],
        "prediction": final_labels
    })
    results_df.to_csv("prediction.txt", index=False)
    print("Predictions saved to 'prediction.txt' with columns [src, dst, prediction].")

if __name__ == "__main__":
    main()
