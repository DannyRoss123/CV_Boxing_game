"""
Verify the saved Random Forest model on a held-out test set.
Also inspect what the training data looks like statistically
vs what the live demo would produce.
"""

import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
RF_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "random_forest.joblib")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "rf_scaler.joblib")

ACTIONS = ["idle", "jab", "cross", "hook", "uppercut", "block_high"]
SEQUENCE_LENGTH = 60
FEATURE_DIM = 33 * 4


def load_dataset():
    X, y = [], []
    for action in ACTIONS:
        action_dir = os.path.join(DATA_DIR, action)
        if not os.path.isdir(action_dir):
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


def extract_statistics(X):
    if X.ndim == 2:
        X = X[np.newaxis, :]
    feat_mean = np.mean(X, axis=1)
    feat_std = np.std(X, axis=1)
    feat_min = np.min(X, axis=1)
    feat_max = np.max(X, axis=1)
    feat_range = feat_max - feat_min
    feat_delta = X[:, -1, :] - X[:, 0, :]
    return np.concatenate([feat_mean, feat_std, feat_min, feat_max, feat_range, feat_delta], axis=1)


def main():
    print("=" * 60)
    print("  Model Verification")
    print("=" * 60)

    # Load saved model.
    rf = joblib.load(RF_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Loaded saved RF model and scaler.\n")

    # Load data.
    X, y = load_dataset()
    print(f"Total samples: {len(X)}")

    # Same split as training (seed=42).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}\n")

    # ── Test 1: Evaluate on the SAME test split ─────────────────────────
    print("--- Test 1: Same split as training (seed=42) ---")
    X_test_stats = extract_statistics(X_test)
    X_test_scaled = scaler.transform(X_test_stats)
    y_pred = rf.predict(X_test_scaled)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.1f}%")
    print(classification_report(y_test, y_pred, zero_division=0))

    # ── Test 2: Completely different random split ────────────────────────
    print("--- Test 2: Different split (seed=99) ---")
    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X, y, test_size=0.2, random_state=99, stratify=y,
    )
    X_test2_stats = extract_statistics(X_test2)
    X_test2_scaled = scaler.transform(X_test2_stats)
    y_pred2 = rf.predict(X_test2_scaled)
    print(f"Accuracy: {accuracy_score(y_test2, y_pred2)*100:.1f}%")
    print(classification_report(y_test2, y_pred2, zero_division=0))

    # ── Test 3: Show individual predictions on test set ──────────────────
    print("--- Individual test predictions (first split) ---")
    wrong = []
    for i in range(len(y_test)):
        proba = rf.predict_proba(X_test_scaled[i:i+1])[0]
        pred = rf.classes_[np.argmax(proba)]
        conf = np.max(proba)
        marker = "OK" if pred == y_test[i] else "WRONG"
        if pred != y_test[i]:
            wrong.append(i)
        print(f"  [{marker:5s}] true={y_test[i]:>12s}  pred={pred:>12s}  conf={conf*100:.0f}%")

    # ── Inspect training data shape ──────────────────────────────────────
    print("\n--- Training data: per-class feature stats ---")
    print("(Avg std across landmarks = how much movement is in each class)\n")
    X_stats = extract_statistics(X)
    for action in ACTIONS:
        mask = y == action
        # The std features are at indices 132:264 in the 792-feature vector.
        avg_std = np.mean(X_stats[mask, 132:264])
        # The delta features are at indices 660:792.
        avg_delta = np.mean(np.abs(X_stats[mask, 660:792]))
        # The range features are at indices 528:660.
        avg_range = np.mean(X_stats[mask, 528:660])
        print(f"  {action:>12s}:  avg_std={avg_std:.4f}  avg_delta={avg_delta:.4f}  avg_range={avg_range:.4f}")

    # ── Inspect a single sample's raw frames ─────────────────────────────
    print("\n--- Sample inspection: first frame vs last frame (wrist y-coord) ---")
    print("(Landmark 16 = right wrist, y-component. Lower y = higher on screen)\n")
    R_WRIST_Y = 16 * 4 + 1  # index 65
    for action in ACTIONS:
        mask = y == action
        samples = X[mask]
        first_y = np.mean(samples[:, 0, R_WRIST_Y])
        last_y = np.mean(samples[:, -1, R_WRIST_Y])
        mid_y = np.mean(samples[:, 30, R_WRIST_Y])
        print(f"  {action:>12s}:  frame0_y={first_y:.3f}  frame30_y={mid_y:.3f}  frame59_y={last_y:.3f}")


if __name__ == "__main__":
    main()
