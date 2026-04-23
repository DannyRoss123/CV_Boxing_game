---
title: Boxing Hub
emoji: 🥊
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# Boxing Hub — Real-Time Boxing Training Games

Your body is the controller. PunchCV uses a standard webcam + MediaPipe pose estimation to classify six boxing actions in real time, driving an interactive boxing game.

**Live demo:** [https://huggingface.co/spaces/DanielRoss12345/boxinghub](https://huggingface.co/spaces/DanielRoss12345/boxinghub)

---

## Overview

| Component | Description |
|---|---|
| **Pose pipeline** | MediaPipe PoseLandmarker → 33 landmarks × 4 values per frame |
| **Feature engineering** | 60-frame sliding window → temporal statistics (std, range, delta) → 396 features |
| **Models** | Nearest Centroid (baseline), Random Forest (production, 96.4%), LSTM (92.9%) |
| **Game — Pygame** | `scripts/game.py` — self-contained local boxing game |
| **Game — Godot 3D** | `scripts/cv_server.py` → WebSocket → Godot client |
| **Web app** | `app.py` — Gradio interface, deployed on HuggingFace Spaces |

---

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download pose model + train models (one-time setup)
python setup.py
```

---

## Running the App

### Web interface (Gradio)
```bash
python app.py
# Open http://localhost:7860
```

### Local Pygame boxing game
```bash
python scripts/game.py
```

### Godot 3D game (requires Godot client separately)
```bash
python scripts/cv_server.py
# WebSocket server starts at ws://localhost:9090
```

### Collect your own training data
```bash
python scripts/collect_data.py --action jab
# Records 60 frames per sample. Press SPACE to record.
```

### Train models from scratch
```bash
python scripts/train_models.py
```

---

## Repository Structure

```
├── README.md
├── requirements.txt
├── setup.py                    ← downloads pose model + trains models
├── app.py                      ← Gradio web app (public deployment)
├── scripts/
│   ├── collect_data.py         ← data collection tool
│   ├── train_models.py         ← trains all 3 models
│   ├── verify_model.py         ← evaluates saved RF model
│   ├── game.py                 ← Pygame boxing game
│   ├── cv_server.py            ← WebSocket server for Godot
│   ├── live_demo.py            ← manual-trigger demo
│   └── live_demo_auto.py       ← auto motion-detection demo
├── src/
│   └── basic_cv_boxing.py      ← early proof-of-concept (reference only)
├── models/
│   ├── random_forest.joblib    ← trained RF classifier
│   ├── rf_scaler.joblib        ← StandardScaler for RF features
│   └── lstm.pt                 ← trained LSTM (PyTorch)
├── data/
│   ├── raw/                    ← training data (.npy, one file per sample)
│   │   ├── idle/   (149 samples)
│   │   ├── jab/    (150 samples)
│   │   ├── cross/  (100 samples)
│   │   ├── hook/   (100 samples)
│   │   ├── uppercut/  (100 samples)
│   │   └── block_high/ (100 samples)
│   ├── processed/
│   └── outputs/
├── notebooks/                  ← exploration notebooks
├── report/
│   ├── final_report.md         ← full research paper
│   └── presentation_prompt.md  ← Demo Day pitch prompt
└── pose_landmarker_lite.task   ← MediaPipe model (auto-downloaded by setup.py)
```

---

## Models & Results

| Model | Test Accuracy | Macro F1 | Inference |
|---|---|---|---|
| Nearest Centroid (baseline) | 46.4% | 0.43 | <1 ms |
| **Random Forest (deployed)** | **96.4%** | **0.96** | ~3 ms |
| LSTM (PyTorch) | 92.9% | 0.93 | ~15 ms |

All metrics on 140-sample stratified held-out test set (20% of 699 total samples).

---

## Action Classes

| Class | Description | Damage (game) |
|---|---|---|
| `idle` | Neutral stance | 0 |
| `jab` | Lead-hand straight punch | 8 HP |
| `cross` | Rear-hand straight punch | 12 HP |
| `hook` | Horizontal arc punch | 15 HP |
| `uppercut` | Upward vertical punch | 20 HP |
| `block_high` | Both arms raised | 0 (blocks) |

---

## Tech Stack

- **Pose estimation:** MediaPipe PoseLandmarker (tasks API, v0.10+)
- **Classical ML:** scikit-learn RandomForestClassifier
- **Deep learning:** PyTorch LSTM
- **Game engine:** Pygame (local) / Godot 4 (3D client)
- **Web interface:** Gradio
- **WebSocket:** Python `websockets` library
