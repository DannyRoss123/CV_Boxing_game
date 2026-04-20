"""
Experiment: Training Set Size Sensitivity Analysis
===================================================
Motivation: PunchCV was trained on ~699 samples collected by a single user.
We investigate how classification accuracy scales with training data volume
to understand whether more data collection would meaningfully improve the
deployed model, and to identify the point of diminishing returns.

Methodology:
    - Fix the test set (20% stratified split, seed=42) across all runs
    - Train Nearest Centroid and Random Forest at 10 training fractions:
      10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%
    - Repeat each fraction with 5 random seeds for confidence intervals
    - Record per-class F1 alongside overall accuracy to catch class imbalance effects

Outputs (written to data/outputs/):
    - sensitivity_results.csv   — raw results table
    - sensitivity_plot.png      — learning curves with +/-1 std shading
    - sensitivity_summary.txt   — human-readable summary

Usage:
    python scripts/experiment.py
"""

import os
import sys
import json
import math
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT       = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR   = os.path.join(ROOT, "data", "raw")
OUTPUT_DIR = os.path.join(ROOT, "data", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Constants (must match train_models.py) ─────────────────────────────────
ACTIONS         = ["idle", "jab", "cross", "hook", "uppercut", "block_high"]
SEQUENCE_LENGTH = 60
NUM_LANDMARKS   = 33
FEATURE_DIM     = NUM_LANDMARKS * 4   # 132
L_HIP, R_HIP    = 23 * 4, 24 * 4
L_SHO, R_SHO    = 11 * 4, 12 * 4

FRACTIONS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
N_SEEDS   = 5
TEST_FRAC = 0.20
BASE_SEED = 42


# ── Data loading & preprocessing ──────────────────────────────────────────

def load_dataset():
    """Load all .npy samples; return X (N, 60, 132) and y (N,)."""
    X, y = [], []
    for action in ACTIONS:
        d = os.path.join(DATA_DIR, action)
        if not os.path.isdir(d):
            continue
        for fname in sorted(f for f in os.listdir(d) if f.endswith(".npy")):
            sample = np.load(os.path.join(d, fname))
            if sample.shape[0] < SEQUENCE_LENGTH:
                pad = np.zeros((SEQUENCE_LENGTH - sample.shape[0], FEATURE_DIM), np.float32)
                sample = np.vstack([sample, pad])
            else:
                sample = sample[:SEQUENCE_LENGTH]
            X.append(sample)
            y.append(action)
    return np.array(X, np.float32), np.array(y)


def normalize_poses(X):
    """Hip-centered, torso-scaled normalization (matches train_models.py)."""
    X = X.copy()
    for i in range(len(X)):
        for t in range(SEQUENCE_LENGTH):
            f = X[i, t]
            cx = (f[L_HIP] + f[R_HIP]) / 2
            cy = (f[L_HIP+1] + f[R_HIP+1]) / 2
            cz = (f[L_HIP+2] + f[R_HIP+2]) / 2
            sx = (f[L_SHO] + f[R_SHO]) / 2
            sy = (f[L_SHO+1] + f[R_SHO+1]) / 2
            sz = (f[L_SHO+2] + f[R_SHO+2]) / 2
            tl = max(math.sqrt((sx-cx)**2 + (sy-cy)**2 + (sz-cz)**2), 0.01)
            for lm in range(NUM_LANDMARKS):
                idx = lm * 4
                f[idx]   = (f[idx]   - cx) / tl
                f[idx+1] = (f[idx+1] - cy) / tl
                f[idx+2] = (f[idx+2] - cz) / tl
    return X


def extract_statistics(X):
    """Temporal statistics: std + range + delta -> (N, 396)."""
    return np.concatenate([np.std(X, 1), np.max(X, 1)-np.min(X, 1), X[:,-1]-X[:,0]], axis=1)


# ── Baseline model ─────────────────────────────────────────────────────────

class NearestCentroidBaseline:
    def __init__(self):
        self.centroids = {}

    def fit(self, X, y):
        Xf = X.reshape(len(X), -1)
        for label in np.unique(y):
            self.centroids[label] = Xf[y == label].mean(0)

    def predict(self, X):
        Xf = X.reshape(len(X), -1)
        labels = list(self.centroids)
        C = np.array([self.centroids[l] for l in labels])
        return np.array([labels[np.argmin(np.linalg.norm(C - x, axis=1))] for x in Xf])


# ── Experiment logic ───────────────────────────────────────────────────────

def run_fraction(X_trainval, y_trainval, X_test, y_test, fraction, seed):
    """
    Sub-sample fraction of the train+val pool, train both models, return metrics.
    """
    n_total = len(X_trainval)
    n_use   = max(len(ACTIONS), int(round(n_total * fraction)))   # at least 1 per class

    if n_use < n_total:
        _, X_tr, _, y_tr = train_test_split(
            X_trainval, y_trainval,
            test_size=n_use/n_total,
            random_state=seed,
            stratify=y_trainval,
        )
    else:
        X_tr, y_tr = X_trainval.copy(), y_trainval.copy()

    results = {"n_train": len(X_tr)}

    # Nearest Centroid
    nc = NearestCentroidBaseline()
    nc.fit(X_tr, y_tr)
    y_pred_nc = nc.predict(X_test)
    results["nc_acc"]  = float(accuracy_score(y_test, y_pred_nc))
    results["nc_f1"]   = float(f1_score(y_test, y_pred_nc, average="macro", zero_division=0))

    # Random Forest on temporal statistics
    stats_tr   = extract_statistics(X_tr)
    stats_test = extract_statistics(X_test)
    scaler = StandardScaler()
    stats_tr_s   = scaler.fit_transform(stats_tr)
    stats_test_s = scaler.transform(stats_test)
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_leaf=2,
        random_state=seed, n_jobs=-1,
    )
    rf.fit(stats_tr_s, y_tr)
    y_pred_rf = rf.predict(stats_test_s)
    results["rf_acc"]  = float(accuracy_score(y_test, y_pred_rf))
    results["rf_f1"]   = float(f1_score(y_test, y_pred_rf, average="macro", zero_division=0))

    # Per-class F1 for RF (to track which classes benefit most from more data)
    per_class = f1_score(y_test, y_pred_rf, average=None,
                         labels=ACTIONS, zero_division=0)
    for action, f1 in zip(ACTIONS, per_class):
        results[f"rf_f1_{action}"] = float(f1)

    return results


def run_experiment():
    print("=" * 60)
    print("  Training Set Size Sensitivity Analysis")
    print("=" * 60)

    print("\n[1] Loading and normalising dataset...")
    X, y = load_dataset()
    X = normalize_poses(X)
    print(f"    Total samples: {len(X)}")
    for a in ACTIONS:
        print(f"      {a:>12s}: {np.sum(y==a)}")

    # Fixed test set — never changes across fractions or seeds
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_FRAC, random_state=BASE_SEED, stratify=y
    )
    print(f"\n    Train+val pool: {len(X_trainval)}  |  Test (fixed): {len(X_test)}")

    print("\n[2] Running experiment "
          f"({len(FRACTIONS)} fractions × {N_SEEDS} seeds = "
          f"{len(FRACTIONS)*N_SEEDS} fits per model)...")

    all_rows = []
    for frac in FRACTIONS:
        seed_results = []
        for seed in range(N_SEEDS):
            r = run_fraction(X_trainval, y_trainval, X_test, y_test, frac, BASE_SEED + seed)
            r["fraction"] = frac
            r["seed"]     = seed
            seed_results.append(r)
            all_rows.append(r)

        mean_rf  = np.mean([r["rf_acc"]  for r in seed_results])
        std_rf   = np.std( [r["rf_acc"]  for r in seed_results])
        mean_nc  = np.mean([r["nc_acc"]  for r in seed_results])
        n_tr     = seed_results[0]["n_train"]
        print(f"    {int(frac*100):3d}% ({n_tr:4d} samples) -> "
              f"RF: {mean_rf*100:.1f}% +/-{std_rf*100:.1f}%   "
              f"NC: {mean_nc*100:.1f}%")

    # ── Save raw CSV ──────────────────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, "sensitivity_results.csv")
    fieldnames = list(all_rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n[3] Raw results saved -> {csv_path}")

    # ── Aggregate per fraction ────────────────────────────────────────────
    agg = {}
    for frac in FRACTIONS:
        rows = [r for r in all_rows if r["fraction"] == frac]
        agg[frac] = {
            "n_train":  rows[0]["n_train"],
            "rf_acc_mean":  np.mean([r["rf_acc"]  for r in rows]),
            "rf_acc_std":   np.std( [r["rf_acc"]  for r in rows]),
            "rf_f1_mean":   np.mean([r["rf_f1"]   for r in rows]),
            "nc_acc_mean":  np.mean([r["nc_acc"]  for r in rows]),
            "nc_acc_std":   np.std( [r["nc_acc"]  for r in rows]),
        }
        for action in ACTIONS:
            key = f"rf_f1_{action}"
            agg[frac][f"rf_f1_{action}_mean"] = np.mean([r[key] for r in rows])

    # ── Save summary text ─────────────────────────────────────────────────
    summary_path = os.path.join(OUTPUT_DIR, "sensitivity_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Training Set Size Sensitivity Analysis\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total dataset: {len(X)} samples across {len(ACTIONS)} classes\n")
        f.write(f"Fixed test set: {len(X_test)} samples (20% stratified)\n")
        f.write(f"Seeds per fraction: {N_SEEDS}\n\n")
        f.write(f"{'Frac':>5}  {'N':>5}  {'RF Acc':>8}  {'+/-':>6}  {'RF F1':>7}  {'NC Acc':>8}\n")
        f.write("-" * 55 + "\n")
        for frac in FRACTIONS:
            a = agg[frac]
            f.write(f"{int(frac*100):4d}%  {a['n_train']:5d}  "
                    f"{a['rf_acc_mean']*100:7.1f}%  "
                    f"+/-{a['rf_acc_std']*100:4.1f}%  "
                    f"{a['rf_f1_mean']*100:6.1f}%  "
                    f"{a['nc_acc_mean']*100:7.1f}%\n")
        f.write("\nPer-class RF F1 at 100% training data:\n")
        for action in ACTIONS:
            val = agg[1.00][f"rf_f1_{action}_mean"] * 100
            f.write(f"  {action:>12s}: {val:.1f}%\n")
        f.write("\nKey finding: ")
        acc_10  = agg[0.10]["rf_acc_mean"]
        acc_50  = agg[0.50]["rf_acc_mean"]
        acc_100 = agg[1.00]["rf_acc_mean"]
        gain_10_50  = (acc_50  - acc_10)  * 100
        gain_50_100 = (acc_100 - acc_50)  * 100
        f.write(f"RF accuracy increases {gain_10_50:.1f}pp from 10->50% of data, "
                f"then only {gain_50_100:.1f}pp from 50->100%, indicating "
                f"{'strong' if gain_50_100>3 else 'moderate'} data hunger "
                f"and {'room for improvement with more samples' if gain_50_100>3 else 'approaching saturation'}.\n")
    print(f"[4] Summary saved -> {summary_path}")

    # ── Plot (optional — skip gracefully if matplotlib unavailable) ────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fracs_pct = [int(f*100) for f in FRACTIONS]
        rf_means  = [agg[f]["rf_acc_mean"]*100 for f in FRACTIONS]
        rf_stds   = [agg[f]["rf_acc_std"]*100  for f in FRACTIONS]
        nc_means  = [agg[f]["nc_acc_mean"]*100 for f in FRACTIONS]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("PunchCV — Training Set Size Sensitivity", fontsize=14, fontweight="bold")

        # Left: overall accuracy curves
        ax = axes[0]
        ax.fill_between(fracs_pct,
                        [m-s for m,s in zip(rf_means, rf_stds)],
                        [m+s for m,s in zip(rf_means, rf_stds)],
                        alpha=0.2, color="steelblue")
        ax.plot(fracs_pct, rf_means, "o-", color="steelblue",
                linewidth=2, markersize=6, label="Random Forest")
        ax.plot(fracs_pct, nc_means, "s--", color="coral",
                linewidth=2, markersize=6, label="Nearest Centroid (baseline)")
        ax.axhline(y=rf_means[-1], color="steelblue", linestyle=":", alpha=0.5)
        ax.set_xlabel("Training set size (% of available data)", fontsize=11)
        ax.set_ylabel("Test Accuracy (%)", fontsize=11)
        ax.set_title("Overall Accuracy vs. Training Set Size")
        ax.legend(fontsize=10)
        ax.set_xticks(fracs_pct)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)

        # Right: per-class RF F1 at 100% vs 50%
        ax2 = axes[1]
        colors = ["#FF9B50","#FF3737","#FFD732","#32DCFF","#3282FF","#50D750"]
        x = np.arange(len(ACTIONS))
        w = 0.35
        f1_50  = [agg[0.50][f"rf_f1_{a}_mean"]*100 for a in ACTIONS]
        f1_100 = [agg[1.00][f"rf_f1_{a}_mean"]*100 for a in ACTIONS]
        ax2.bar(x - w/2, f1_50,  w, label="50% data",  color=[c+"88" for c in colors])
        ax2.bar(x + w/2, f1_100, w, label="100% data", color=colors)
        ax2.set_xticks(x)
        ax2.set_xticklabels([a.replace("_","\n") for a in ACTIONS], fontsize=9)
        ax2.set_ylabel("F1 Score (%)", fontsize=11)
        ax2.set_title("Per-Class RF F1: 50% vs 100% Training Data")
        ax2.legend(fontsize=10)
        ax2.set_ylim(0, 105)
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, "sensitivity_plot.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[5] Plot saved -> {plot_path}")
    except ImportError:
        print("[5] matplotlib not available — skipping plot (pip install matplotlib)")

    print("\n" + "=" * 60)
    print("  EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"\n  Outputs in: {OUTPUT_DIR}")
    print(f"    sensitivity_results.csv  — raw per-seed results")
    print(f"    sensitivity_summary.txt  — aggregated table + findings")
    print(f"    sensitivity_plot.png     — learning curves figure")


if __name__ == "__main__":
    run_experiment()
