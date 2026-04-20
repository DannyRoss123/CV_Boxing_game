"""
Train 3 action-recognition models and evaluate on a held-out test set.

Models:
  1. Naive baseline  - Nearest centroid on flattened sequences
  2. Classical ML     - Random Forest on temporal statistics features
  3. Neural network   - LSTM classifier (PyTorch)

Split: 60% train / 20% val / 20% test (stratified).
LSTM uses val for early stopping. Test set is NEVER seen during training.

Usage:
    python scripts/train_models.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
ACTIONS = ["idle", "jab", "cross", "hook", "uppercut", "block_high"]
SEQUENCE_LENGTH = 60
NUM_LANDMARKS = 33
FEATURES_PER_LM = 4
FEATURE_DIM = NUM_LANDMARKS * FEATURES_PER_LM  # 132
RANDOM_SEED = 42

# Landmark indices for normalization.
L_HIP = 23 * 4
R_HIP = 24 * 4
L_SHOULDER = 11 * 4
R_SHOULDER = 12 * 4


# ── LSTM architecture (shared with live_demo.py) ───────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.dropout(hn[-1])
        return self.fc(out)


# ── 1. Load & Normalize ────────────────────────────────────────────────────
def load_dataset():
    X, y = [], []
    for action in ACTIONS:
        action_dir = os.path.join(DATA_DIR, action)
        if not os.path.isdir(action_dir):
            print(f"  WARNING: no directory for '{action}', skipping.")
            continue
        files = sorted(f for f in os.listdir(action_dir) if f.endswith(".npy"))
        for fname in files:
            sample = np.load(os.path.join(action_dir, fname))
            if sample.shape[0] < SEQUENCE_LENGTH:
                pad = np.zeros((SEQUENCE_LENGTH - sample.shape[0], FEATURE_DIM), dtype=np.float32)
                sample = np.vstack([sample, pad])
            elif sample.shape[0] > SEQUENCE_LENGTH:
                sample = sample[:SEQUENCE_LENGTH]
            X.append(sample)
            y.append(action)
    return np.array(X, dtype=np.float32), np.array(y)


def normalize_poses(X):
    X_norm = X.copy()
    N, T, _ = X_norm.shape
    for i in range(N):
        for t in range(T):
            frame = X_norm[i, t]
            cx = (frame[L_HIP] + frame[R_HIP]) / 2
            cy = (frame[L_HIP + 1] + frame[R_HIP + 1]) / 2
            cz = (frame[L_HIP + 2] + frame[R_HIP + 2]) / 2
            sx = (frame[L_SHOULDER] + frame[R_SHOULDER]) / 2
            sy = (frame[L_SHOULDER + 1] + frame[R_SHOULDER + 1]) / 2
            sz = (frame[L_SHOULDER + 2] + frame[R_SHOULDER + 2]) / 2
            torso_len = np.sqrt((sx - cx)**2 + (sy - cy)**2 + (sz - cz)**2)
            if torso_len < 0.01:
                torso_len = 1.0
            for lm in range(NUM_LANDMARKS):
                idx = lm * FEATURES_PER_LM
                frame[idx]     = (frame[idx]     - cx) / torso_len
                frame[idx + 1] = (frame[idx + 1] - cy) / torso_len
                frame[idx + 2] = (frame[idx + 2] - cz) / torso_len
    return X_norm


# ── 2. Feature Engineering ─────────────────────────────────────────────────
def extract_statistics(X):
    """Motion-only features: std, range, delta. Shape: (N, 396)."""
    feat_std = np.std(X, axis=1)
    feat_range = np.max(X, axis=1) - np.min(X, axis=1)
    feat_delta = X[:, -1, :] - X[:, 0, :]
    return np.concatenate([feat_std, feat_range, feat_delta], axis=1)


# ── 3. Model Training ──────────────────────────────────────────────────────

class NearestCentroidBaseline:
    def __init__(self):
        self.centroids = {}

    def fit(self, X, y):
        X_flat = X.reshape(X.shape[0], -1)
        for label in np.unique(y):
            self.centroids[label] = X_flat[y == label].mean(axis=0)

    def predict(self, X):
        X_flat = X.reshape(X.shape[0], -1)
        labels = list(self.centroids.keys())
        centroid_matrix = np.array([self.centroids[l] for l in labels])
        preds = []
        for x in X_flat:
            dists = np.linalg.norm(centroid_matrix - x, axis=1)
            preds.append(labels[np.argmin(dists)])
        return np.array(preds)


def train_random_forest(X_train_stats, y_train):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train_stats)
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_leaf=2,
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    rf.fit(X_tr, y_train)
    return rf, scaler


def train_lstm(X_train, y_train_enc, X_val, y_val_enc, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  LSTM device: {device}")

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train_enc, dtype=torch.long)
    X_va = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_va_np = y_val_enc

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=16, shuffle=True)

    model = LSTMClassifier(
        input_dim=FEATURE_DIM, hidden_dim=64,
        num_layers=2, num_classes=num_classes, dropout=0.3,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    EPOCHS = 300
    best_val_acc = 0.0
    best_state = None
    patience = 50
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        scheduler.step()

        # Evaluate on val set in eval mode every 10 epochs.
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_va).argmax(dim=1).cpu().numpy()
                val_acc = (val_preds == y_va_np).mean()
                train_logits = model(X_tr.to(device))
                train_acc = (train_logits.argmax(dim=1).cpu().numpy() == y_train_enc).mean()
            avg_loss = total_loss / len(X_tr)
            print(f"    Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.4f}  "
                  f"train={train_acc*100:.1f}%  val={val_acc*100:.1f}%")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 10
            if no_improve >= patience:
                print(f"    Early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
                break

    # Load best checkpoint.
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    print(f"    Best val accuracy: {best_val_acc*100:.1f}%")

    # Save model.
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, "lstm.pt")
    torch.save(model.state_dict(), save_path)

    # Verify save/load roundtrip.
    check = LSTMClassifier(FEATURE_DIM, 64, 2, num_classes, 0.3).to(device)
    check.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    check.eval()
    with torch.no_grad():
        check_preds = check(X_va).argmax(dim=1).cpu().numpy()
        check_acc = (check_preds == y_va_np).mean()
    print(f"    Save/load verify: {check_acc*100:.1f}%")

    return model


# ── 4. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Boxing Action Recognition - Model Training")
    print("=" * 60)

    # Load and normalize.
    print("\n[1] Loading dataset...")
    X, y = load_dataset()
    print(f"  Total samples: {len(X)}")
    for action in ACTIONS:
        print(f"    {action:>12s}: {np.sum(y == action)}")
    print("  Normalizing poses...")
    X = normalize_poses(X)

    # Encode labels.
    le = LabelEncoder()
    le.fit(ACTIONS)
    y_enc = le.transform(y)

    # 60/20/20 split: train / val / test.
    print("\n[2] Splitting data 60/20/20 (train/val/test)...")
    X_trainval, X_test, y_trainval, y_test, y_trainval_enc, y_test_enc = train_test_split(
        X, y, y_enc, test_size=0.2, random_state=RANDOM_SEED, stratify=y,
    )
    X_train, X_val, y_train, y_val, y_train_enc, y_val_enc = train_test_split(
        X_trainval, y_trainval, y_trainval_enc,
        test_size=0.25, random_state=RANDOM_SEED, stratify=y_trainval,
    )
    print(f"  Train: {len(X_train)}  |  Val: {len(X_val)}  |  Test: {len(X_test)}")

    results = {}

    # ── Model 1: Nearest Centroid ────────────────────────────────────────
    print("\n[3] Nearest Centroid Baseline...")
    baseline = NearestCentroidBaseline()
    baseline.fit(X_train, y_train)
    y_pred = baseline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results["Nearest Centroid"] = acc
    print(f"  Test accuracy: {acc*100:.1f}%")
    print(classification_report(y_test, y_pred, zero_division=0))

    # ── Model 2: Random Forest ──────────────────────────────────────────
    print("[4] Random Forest...")
    rf, scaler = train_random_forest(extract_statistics(X_train), y_train)
    y_pred_rf = rf.predict(scaler.transform(extract_statistics(X_test)))
    acc_rf = accuracy_score(y_test, y_pred_rf)
    results["Random Forest"] = acc_rf
    print(f"  Test accuracy: {acc_rf*100:.1f}%")
    print(classification_report(y_test, y_pred_rf, zero_division=0))

    # Save RF.
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "rf_scaler.joblib"))

    # ── Model 3: LSTM ───────────────────────────────────────────────────
    print("[5] LSTM (train on train, early stop on val, report on test)...")
    lstm_model = train_lstm(X_train, y_train_enc, X_val, y_val_enc, num_classes=len(ACTIONS))

    # Final test evaluation — model has NEVER seen test data.
    device = next(lstm_model.parameters()).device
    lstm_model.eval()
    with torch.no_grad():
        X_te_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        test_preds_enc = lstm_model(X_te_t).argmax(dim=1).cpu().numpy()
    y_pred_lstm = le.inverse_transform(test_preds_enc)
    acc_lstm = accuracy_score(y_test, y_pred_lstm)
    results["LSTM"] = acc_lstm
    print(f"  TEST accuracy: {acc_lstm*100:.1f}%")
    print(classification_report(y_test, y_pred_lstm, zero_division=0))

    # ── Summary ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("  RESULTS SUMMARY (all on held-out test set)")
    print("=" * 60)
    for name, acc in results.items():
        bar = "#" * int(acc * 40)
        print(f"  {name:<20s}  {acc*100:5.1f}%  |{bar}")
    print("=" * 60)


if __name__ == "__main__":
    main()
