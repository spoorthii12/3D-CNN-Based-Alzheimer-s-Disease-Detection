# test_model.py

import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, random_split
from model import CNN3D  # your CNN3D definition
from train import BrainDataset  # re‑use your Dataset

# — Configuration —
DATA_DIR   = "data/processed"
MODEL_PATH = "best_model.pt"
BATCH_SIZE = 4
TEST_SPLIT = 0.2       # fraction to hold out as test
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # 1) load full dataset
    full_ds = BrainDataset(DATA_DIR)

    # 2) split off a test set
    total = len(full_ds)
    test_size = int(total * TEST_SPLIT)
    train_size = total - test_size
    _, test_ds = random_split(
        full_ds,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 3) load your trained model
    model = CNN3D().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [], []

    # 4) run inference
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE).long()
            logits = model(X).squeeze(1)
            probs  = torch.sigmoid(logits)
            preds  = (probs > 0.5).long()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    # 5) print metrics
    print(f"\nTest set size: {len(all_labels)}")
    print("Accuracy:", accuracy_score(all_labels, all_preds))
    print("\nClassification report:\n", classification_report(all_labels, all_preds, digits=4))

if __name__ == "__main__":
    main()
