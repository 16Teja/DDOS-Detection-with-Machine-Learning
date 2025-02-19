#!/usr/bin/env python
# coding: utf-8

"""
Train:
- Reads X_train.csv and y_train.csv (which include src and dst along with numeric features).
- Drops the src and dst columns before scaling and training.
- Creates PyTorch Datasets and DataLoaders.
- Defines and trains the Hybrid CNN+Transformer model.
- Evaluates on X_test.csv/y_test.csv and saves the model state.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# -----------------------
# Dataset & Collate
# -----------------------
class DDoSDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels   = torch.tensor(labels, dtype=torch.float32)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def collate_fn(batch):
    feats, labs = zip(*batch)
    feats = torch.stack(feats, dim=0)  # [batch_size, num_features]
    labs  = torch.stack(labs, dim=0)   # [batch_size]
    # Reshape for 1D CNN: [batch_size, in_channels=1, seq_len=num_features]
    feats = feats.unsqueeze(1)
    return feats, labs

# -----------------------
# CNN & Transformer Model
# -----------------------
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
        out = torch.cat([x3, x5, x7], dim=1)  # [batch_size, 3*out_channels, seq_len]
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
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
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
        # x: [batch_size, in_channels, seq_len]
        cnn_out = self.cnn(x)                   # [batch_size, 3*out_channels, seq_len]
        cnn_out = cnn_out.permute(0, 2, 1)        # [batch_size, seq_len, 3*out_channels]
        transformer_out = self.transformer(cnn_out)  # [batch_size, seq_len, embed_dim]
        pooled = torch.mean(transformer_out, dim=1)    # [batch_size, embed_dim]
        logits = self.fc(pooled)                        # [batch_size, num_classes]
        # For binary classification, apply sigmoid
        if logits.shape[1] == 1:
            return torch.sigmoid(logits)
        else:
            return logits

def main():
    # STEP 1: Load CSV data (which include all fields)
    X_train_df = pd.read_csv("X_train.csv")
    y_train_df = pd.read_csv("y_train.csv")
    X_test_df  = pd.read_csv("X_test.csv")
    y_test_df  = pd.read_csv("y_test.csv")

    # Retain src and dst for potential reporting later, but drop them for training.
    # Assume these columns exist.
    if "src" in X_train_df.columns and "dst" in X_train_df.columns:
        train_ids = X_train_df[["src", "dst"]]
        X_train_features = X_train_df.drop(columns=["src", "dst"])
    else:
        X_train_features = X_train_df.copy()

    if "src" in X_test_df.columns and "dst" in X_test_df.columns:
        test_ids = X_test_df[["src", "dst"]]
        X_test_features = X_test_df.drop(columns=["src", "dst"])
    else:
        X_test_features = X_test_df.copy()

    # Convert to NumPy arrays
    X_train_np = X_train_features.values
    y_train_np = y_train_df.values.squeeze()
    X_test_np  = X_test_features.values
    y_test_np  = y_test_df.values.squeeze()

    # STEP 2: Scale data (using only numeric features)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_np)
    X_test_scaled  = scaler.transform(X_test_np)

    # STEP 3: Create Datasets & DataLoaders
    train_dataset = DDoSDataset(X_train_scaled, y_train_np)
    test_dataset  = DDoSDataset(X_test_scaled, y_test_np)

    BATCH_SIZE = 32
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # STEP 4: Define the Hybrid CNN+Transformer model and training settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    IN_CHANNELS  = 1        # Because data will be reshaped to [batch_size, 1, num_features]
    OUT_CHANNELS = 64
    EMBED_DIM    = 128
    N_HEADS      = 4
    NUM_LAYERS   = 3
    NUM_CLASSES  = 1        # Binary classification
    NUM_EPOCHS   = 10       # Adjust epochs as needed

    model = HybridModel(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        embed_dim=EMBED_DIM,
        n_heads=N_HEADS,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    # STEP 5: Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)         # [batch_size, 1, seq_len]
            batch_labels   = batch_labels.to(device).unsqueeze(1)  # [batch_size, 1]

            optimizer.zero_grad()
            outputs = model(batch_features)  # [batch_size, 1]
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_features.size(0)
        scheduler.step()
        epoch_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}")

    # STEP 6: Evaluate on test set
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            batch_labels   = batch_labels.to(device)
            outputs = model(batch_features)  # [batch_size, 1]
            preds   = (outputs > 0.5).float().squeeze()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {acc:.4f}")

    # STEP 7: Save the trained model and scaler if needed
    torch.save(model.state_dict(), "cnn_transformer.pt")
    print("Model state saved to 'cnn_transformer.pt'")

if __name__ == "__main__":
    main()
