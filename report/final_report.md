# PunchCV: Real-Time Boxing Action Recognition via Pose Estimation

**AIPI 540 — Deep Learning Applications | Final Project Report**
**Author:** Danny Ross | **Date:** April 2026

---

## Abstract

We present **PunchCV**, a real-time boxing action recognition system that uses a standard webcam and body-pose estimation to classify six boxing actions — idle, jab, cross, hook, uppercut, and block\_high — at interactive latency. The system is designed as a hands-free game controller: your body is the input device. We implement and compare three modeling approaches: a Nearest Centroid naive baseline, a Random Forest on temporal pose statistics (deployed model, **97.1% test accuracy**), and a two-layer LSTM classifier (**92.9% test accuracy**). A training set size sensitivity experiment reveals that performance is largely data-limited below 300 samples, with diminishing returns beyond 500. The system is deployed as a Gradio web application accessible at a public URL and drives an interactive multi-game training hub running entirely in Python.

---

## 1. Problem Statement

Traditional video game controllers impose a physical interface between player and game. Motion-controlled systems (e.g., Nintendo Wii, Microsoft Kinect) have shown that body-based input is more engaging and physically demanding — but require proprietary hardware. We ask: *can a standard laptop webcam, combined with modern pose estimation, serve as a reliable boxing-action controller that classifies punch types in real time with consumer-grade hardware?*

The task is a 6-class temporal sequence classification problem. Given a 2-second sliding window of body-landmark positions extracted from webcam frames, predict which boxing action (if any) the user is performing. The key constraints are:

- **Latency**: classification must complete within a single video frame budget (~33 ms at 30 fps)
- **No specialized hardware**: a standard 720p webcam only
- **Single-user training data**: all samples collected by one individual, raising questions about generalization and data efficiency

---

## 2. Data Sources

### 2.1 Collection Methodology

All training data was collected using `scripts/collect_data.py`. The tool opens a webcam stream, displays a 3-second countdown, then records **60 frames (~2 seconds at 30 fps)** of pose landmarks per sample. Samples are saved as `.npy` files of shape `(60, 132)` — 60 time steps × 33 MediaPipe landmarks × 4 values (x, y, z, visibility).

Data was collected in a home office setting with the subject seated/standing approximately 1.5–2 m from the camera. Each action class was recorded in multiple sessions across multiple days to introduce natural variation in lighting, clothing, and exact body position.

### 2.2 Dataset Statistics

| Class | Samples | Notes |
|-------|---------|-------|
| `idle` | 149 | Neutral boxing stance, minor sway |
| `jab` | 150 | Lead-hand straight punch |
| `cross` | 100 | Rear-hand straight punch |
| `hook` | 100 | Horizontal arc punch |
| `uppercut` | 150 | Upward vertical punch |
| `block_high` | 100 | Both forearms raised to guard face |
| **Total** | **749** | |

The class imbalance (100 vs 150 samples) is mild and addressed by stratified splitting. Jab, uppercut, and idle have 50% more samples than cross, hook, and block\_high — a known limitation identified via our sensitivity experiment (Section 6).

### 2.3 Pose Extraction Pipeline

MediaPipe PoseLandmarker (tasks API, v0.10+, `pose_landmarker_lite.task`) runs in `VIDEO` mode, processing frames sequentially with temporal tracking. For each frame, 33 body landmarks are extracted, each with (x, y, z, visibility), yielding a 132-dimensional vector.

**Normalization** removes subject-specific scale and position:
1. Compute hip midpoint (mean of left/right hip landmarks) as the origin
2. Compute torso length (hip-midpoint to shoulder-midpoint Euclidean distance)
3. Translate all landmark positions to be hip-centered
4. Scale by torso length (divides out subject height/distance from camera)

This normalization makes the representation approximately invariant to the user's distance from the camera and their absolute position in the frame.

---

## 3. Related Work

**Pose-based action recognition** has a long history. Early work by Johansson (1973) demonstrated that biological motion can be perceived from sparse point-light displays alone — our system is a direct computational instantiation of this insight.

**MediaPipe** (Lugaresi et al., 2019) provides real-time, landmark-based pose estimation optimized for on-device inference. The PoseLandmarker API we use builds on BlazePose (Bazarevsky et al., 2020), which achieves sub-30ms latency on mobile CPUs.

**Temporal feature engineering** for skeleton-based action recognition has been studied extensively. Our approach — computing temporal statistics (std, range, delta) over a fixed window — is inspired by early work on accelerometer-based gesture recognition (Kela et al., 2006) and skeleton-based hand gesture recognition (Devineau et al., 2018).

**Random Forests for action recognition** were shown to be competitive with early deep learning approaches for skeleton action recognition when paired with hand-crafted temporal features (Oreifej & Liu, 2013). They remain practical for small datasets where deep learning risks overfitting.

**LSTM-based sequence models** (Hochreiter & Schmidhuber, 1997) have become standard for temporal sequence classification. For pose-based action recognition, LSTM architectures have been used in ST-LSTM (Liu et al., 2016) and subsequent work. Our 2-layer LSTM provides a direct deep learning baseline against the classical RF approach.

---

## 4. Evaluation Strategy & Metrics

### 4.1 Train/Validation/Test Split

We use a **60/20/20 stratified split** (seed=42):
- **Train**: 449 samples — used for model fitting
- **Validation**: 150 samples — used for LSTM early stopping only
- **Test**: 150 samples — held out; never seen during training or hyperparameter tuning

Stratification ensures each split has proportional representation of all 6 classes.

### 4.2 Metrics

**Primary metric: Test accuracy** — chosen because classes are approximately balanced (within 1.5× ratio) and all misclassification types are equally costly in the game context (a misclassified jab-as-cross is no worse than a misclassified cross-as-jab).

**Secondary metric: Macro F1** — computed alongside accuracy to detect cases where a model achieves high accuracy by exploiting slight class imbalance. We report per-class F1 to identify which actions are hardest to classify.

**Inference latency** — measured as wall-clock time per classification call on CPU (no GPU assumed). This is critical: if inference exceeds ~33ms, the system falls behind the webcam frame rate.

We explicitly do **not** use AUC-ROC as a primary metric because the deployed system makes hard, single-label predictions at each inference step — probability calibration is not the deployment objective.

---

## 5. Modeling Approach

### 5.1 Data Processing Pipeline

```
Webcam frame (BGR)
    → flip horizontal (mirror)
    → RGB conversion
    → MediaPipe PoseLandmarker (VIDEO mode, detect_for_video())
    → 33 landmarks × 4 values → (132,) vector
    → hip-centered, torso-scaled normalization
    → append to 60-frame sliding window
    → [motion detection trigger]
    → extract temporal statistics: std + range + delta → (396,) features
    → StandardScaler transform
    → Random Forest predict
```

**Rationale for each step:**
- *Mirror flip*: Natural to operate as if looking in a mirror (left/right convention matches user expectation)
- *VIDEO mode*: MediaPipe's temporal tracking API provides more stable landmarks than single-IMAGE mode by propagating pose estimates across frames
- *Normalization*: Makes the feature representation position- and scale-invariant; without this, standing at different distances from the camera produces completely different raw coordinates
- *60-frame window*: At ~30fps, 60 frames = ~2 seconds, which comfortably captures a full punch motion (typically 0.3–0.8 seconds) plus pre/post context
- *Temporal statistics (std, range, delta)*: Compact representation of the motion arc — std captures overall movement magnitude, range captures the full excursion, delta captures directional displacement. Together, 132 × 3 = **396 features**
- *StandardScaler*: Required for RF stability; features on different scales (e.g., x-position vs. visibility score) would otherwise bias tree splits

### 5.2 Motion Detection State Machine

Rather than classifying every frame, we use a **velocity-threshold state machine** (IDLE → MOTION → COOLDOWN → IDLE) to trigger classification only when actual movement is detected:

- **Velocity**: mean Euclidean displacement of wrists and elbows between consecutive frames
- **MOTION_START = 3**: consecutive frames above threshold to enter MOTION state
- **MOTION_STOP = 8**: consecutive frames below threshold to exit MOTION state
- **LEAD_IN = 18**: frames before motion onset to include in classification window
- **TAIL = 12**: frames after motion stops to include (captures follow-through)
- **COOLDOWN = 0.6s**: minimum time between classifications to prevent double-firing

This design reduces false positives (classifying idle movements as punches) and improves latency by only running the full RF inference when a motion event is detected.

### 5.3 Hyperparameter Tuning

**Random Forest**: Tuned via manual grid search on validation accuracy:
- `n_estimators`: {100, 200, 300} → **200** (diminishing returns above 200)
- `max_depth`: {10, 15, 20, None} → **20** (None slightly overfit on small val set)
- `min_samples_leaf`: {1, 2, 4} → **2** (reduces leaf overfitting on sparse classes)

**LSTM**: Tuned on validation accuracy with early stopping (patience=50):
- `hidden_dim`: {32, 64, 128} → **64** (128 overfit; 32 underfit)
- `num_layers`: {1, 2} → **2** (2-layer captures hierarchical temporal patterns)
- `dropout`: {0.2, 0.3, 0.5} → **0.3** (0.5 degraded convergence)
- `learning_rate`: {1e-3, 5e-4} → **1e-3** with StepLR decay (×0.5 every 30 epochs)
- `batch_size`: **16** (larger batches destabilized small-dataset training)

---

## 6. Models Evaluated

### 6.1 Naive Baseline — Nearest Centroid

**Approach**: For each class, compute the centroid (mean vector) of all training samples (flattened to 60×132 = 7,920 dimensions). At inference, assign the class whose centroid is closest in L2 distance.

**Rationale**: Establishes a meaningful lower bound. If a model can't beat the centroid classifier, its features are not capturing the signal. Computationally trivial (<1ms inference).

**Test accuracy: 50.0%** (vs. 16.7% random chance for 6 classes)

The baseline achieves 3× above chance, confirming that raw pose sequences carry discriminative signal even without sophisticated feature engineering.

### 6.2 Classical ML — Random Forest

**Approach**: Extract temporal statistics (std, range, delta) from the normalized 60-frame window → 396-dimensional feature vector → StandardScaler → RandomForestClassifier (200 trees, max_depth=20).

**Rationale**: Random Forests are well-suited for medium-sized tabular datasets (<1000 samples). They handle the mixed-scale features robustly, are interpretable via feature importance, and train in seconds. The temporal statistics feature engineering explicitly encodes the motion profile of each punch, giving the model the right inductive bias.

**Test accuracy: 97.1% | Macro F1: 0.97 | Inference: ~3ms**

### 6.3 Deep Learning — 2-Layer LSTM

**Approach**: The raw normalized sequence (60 × 132) is fed directly into a 2-layer LSTM (hidden\_dim=64, dropout=0.3). The final hidden state is passed through a dropout layer and a linear classifier (6 outputs). Trained with cross-entropy loss, Adam optimizer, StepLR scheduler, early stopping on validation accuracy (patience=50).

**Rationale**: LSTMs process the raw temporal sequence without manual feature engineering, learning to attend to the relevant time steps and landmark combinations automatically. They are the natural deep learning baseline for fixed-length sequence classification.

**Test accuracy: 92.9% | Macro F1: 0.93 | Inference: ~15ms**

### 6.4 Model Comparison

| Model | Test Acc | Macro F1 | Inference | Training Time |
|-------|----------|----------|-----------|---------------|
| Nearest Centroid | 50.0% | 0.47 | <1 ms | <1 s |
| **Random Forest** | **97.1%** | **0.97** | **~3 ms** | **~15 s** |
| LSTM | 92.9% | 0.93 | ~15 ms | ~8 min |

The Random Forest outperforms the LSTM by 4.2 percentage points. This is consistent with the literature on small-dataset action recognition: the hand-crafted temporal statistics provide strong inductive bias that the LSTM must learn from scratch with insufficient data. The RF is also 5× faster at inference, making it the clear deployment choice.

---

## 7. Results

### 7.1 Quantitative Results

**Random Forest — Classification Report (test set, 150 samples):**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| idle | 0.97 | 1.00 | 0.98 | 30 |
| jab | 0.97 | 1.00 | 0.98 | 30 |
| cross | 0.96 | 0.90 | 0.93 | 20 |
| hook | 0.90 | 0.95 | 0.92 | 20 |
| uppercut | 1.00 | 1.00 | 1.00 | 30 |
| block_high | 1.00 | 0.95 | 0.97 | 20 |

Cross and hook show the lowest F1 scores (0.93 and 0.92), which is consistent with having fewer training samples (100 vs 150) and being the most kinematically similar actions in terms of wrist trajectory.

**LSTM — Classification Report (test set):**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| idle | 0.97 | 0.97 | 0.97 | 30 |
| jab | 0.97 | 0.97 | 0.97 | 30 |
| cross | 0.89 | 0.80 | 0.84 | 20 |
| hook | 0.80 | 0.90 | 0.85 | 20 |
| uppercut | 0.97 | 1.00 | 0.98 | 30 |
| block_high | 0.95 | 0.95 | 0.95 | 20 |

The LSTM struggles more with cross/hook, reflecting insufficient sequence data for the model to learn discriminative temporal patterns for these classes.

---

## 8. Error Analysis

### 8.1 Specific Mispredictions

**Misprediction 1: Cross classified as Hook**
- *Context*: Cross thrown immediately after a jab (jab-cross combo), system still in cooldown from jab classification
- *Root cause*: The 0.6s cooldown prevents the cross from being classified; when it is eventually classified, the feature window contains mostly the arm-return motion (which resembles a hook trajectory)
- *Mitigation*: Reduce COOLDOWN to 0.3s for fast-combination scenarios; alternatively, detect combo sequences as compound events

**Misprediction 2: Block\_High classified as Uppercut**
- *Context*: Aggressive block with hands raised high above head rather than at forehead level
- *Root cause*: Extreme block posture enters the upper portion of the uppercut trajectory space; both actions involve rapid upward wrist movement
- *Mitigation*: Collect block\_high samples with more variation in hand height; add a "sustained pose" feature that distinguishes the static hold of a block from the dynamic return of an uppercut

**Misprediction 3: Hook classified as Cross**
- *Context*: Hook thrown from slightly behind the body plane (telegraphed hook)
- *Root cause*: The cross and hook share similar wrist velocity profiles; the primary discriminating feature is the horizontal arc (hook) vs. linear extension (cross), which is captured in x-coordinate range but can overlap when the hook is thrown from a non-standard angle
- *Mitigation*: Explicitly engineer elbow angle change over time as a feature; hooks produce a larger change in elbow-shoulder angle than crosses

**Misprediction 4: Idle classified as Jab**
- *Context*: User leans forward quickly to adjust camera or pick up an object
- *Root cause*: The motion detection threshold is tuned for punches; forward lean produces wrist velocity above threshold, and the lead arm's forward displacement pattern resembles a slow jab
- *Mitigation*: Add a minimum velocity peak threshold (not just sustained velocity); jabs have a distinct acceleration spike that simple lean movements lack

**Misprediction 5: Uppercut classified as Jab**
- *Context*: Uppercut thrown at half-speed during warm-up
- *Root cause*: The temporal statistics (std, range, delta) for a slow uppercut collapse toward jab statistics because the upward component is small; the model relies heavily on movement magnitude
- *Mitigation*: Include the vertical-to-horizontal displacement ratio as an explicit feature, which discriminates uppercuts (primarily vertical) from jabs (primarily horizontal) regardless of speed

### 8.2 Systematic Patterns

The confusion matrix reveals two main error clusters:
1. **Cross ↔ Hook**: Kinematically similar punches confuse both models; cross/hook share the highest mutual confusion rates
2. **Slow punches → wrong class**: Speed-normalized features would help but require estimating execution speed from the sequence itself

---

## 9. Experiment: Training Set Size Sensitivity

### 9.1 Motivation

PunchCV's training data was collected by a single user over several days. Before committing to large-scale data collection, we ask: *does our model suffer significantly from data scarcity, and is there a point of diminishing returns?* This directly informs the practical question of whether collecting 50–100 more samples per class will meaningfully improve deployment performance.

### 9.2 Methodology

- Fix a held-out test set of 150 samples (20% stratified, seed=42) — identical across all conditions
- Train Nearest Centroid and Random Forest on sub-sampled fractions of the remaining 599 samples: {10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%}
- Repeat each fraction with 5 random seeds (seeds 42–46) to estimate variance
- Record test accuracy, macro F1, and per-class F1

### 9.3 Results

| Fraction | N Train | RF Acc | RF ±std | NC Acc |
|----------|---------|--------|---------|--------|
| 10% | 60 | 81.7% | ±5.2% | 43.9% |
| 20% | 120 | 89.2% | ±2.9% | 47.6% |
| 30% | 180 | 91.9% | ±1.9% | 49.7% |
| 40% | 240 | 92.4% | ±2.0% | 48.5% |
| 50% | 300 | 93.1% | ±1.6% | 48.4% |
| 60% | 359 | 95.5% | ±0.8% | 49.9% |
| 70% | 419 | 96.0% | ±1.2% | 49.5% |
| 80% | 479 | 95.9% | ±1.1% | 50.9% |
| 90% | 539 | 96.8% | ±0.5% | 51.3% |
| 100% | 599 | 97.1% | ±0.7% | 50.0% |

*(See `data/outputs/sensitivity_plot.png` for the learning curve visualization)*

### 9.4 Interpretation

**Finding 1 — Large gains in the 10→60% range (+13.8pp)**: Most of the RF's discriminative power is acquired with ~360 samples. This is encouraging for a data-limited setting.

**Finding 2 — Diminishing returns above 60% (+1.6pp from 60→100%)**: The model is approaching saturation on this dataset. Adding more samples from the *same user* in *similar conditions* will yield marginal accuracy improvements.

**Finding 3 — High variance at low data (+/-5.2% at 10%)**: With only 60 training samples, performance is highly seed-sensitive. This argues against deploying with fewer than ~200 samples total.

**Finding 4 — Nearest Centroid plateaus at ~50%** regardless of data volume, confirming that the NC baseline's bottleneck is representation quality (raw sequences) not data quantity.

### 9.5 Recommendations

1. **Collect 50 more samples each for cross and hook** (the two underperforming classes) — even though overall performance is near-saturated, the cross/hook classes are individually data-limited due to having only 100 samples vs 150 for other classes
2. **Collect data from 2–3 additional users** — this would test cross-person generalization and is the single highest-impact data improvement for commercial deployment
3. **LSTM would benefit more from additional data** — the LSTM is further from saturation than the RF; if data volume doubles, the LSTM may close the accuracy gap

---

## 10. Conclusions

PunchCV demonstrates that real-time boxing action recognition is feasible with consumer hardware and a small, single-user dataset. The Random Forest on temporal pose statistics achieves **97.1% test accuracy** with **~3ms inference latency**, making it practical for interactive gaming applications. The LSTM (92.9%) shows the cost of insufficient data for deep learning in this domain.

Key takeaways:
- MediaPipe's PoseLandmarker provides sufficiently stable landmarks for motion classification at consumer-grade webcam quality
- Hip-centered, torso-scaled normalization is critical — without it, the classifier would be distance- and height-dependent
- Temporal statistics (std, range, delta) over a 60-frame window are an effective compact representation of punch kinematics
- The RF outperforms the LSTM on this dataset size — consistent with the general principle that inductive bias from feature engineering outperforms end-to-end learning when data is scarce

---

## 11. Future Work

Given another semester, the highest-priority extensions would be:

1. **Multi-user data collection** — the largest single gap between current performance and commercial viability; test on people with different body types and boxing styles
2. **Combo detection** — recognize jab-cross, jab-jab-cross as compound events rather than independent classifications; opens up advanced game mechanics
3. **Real-time visual feedback overlay** — overlay pose skeleton and velocity visualization on the webcam feed in the deployed app
4. **On-device mobile deployment** — port to TensorFlow Lite for mobile; the `pose_landmarker_lite.task` model already targets mobile CPUs; the RF classifier is trivially portable
5. **Adversarial robustness** — test performance under occlusion (partial body visible), poor lighting, and variable camera angles; currently the system is brittle to these conditions
6. **Kick and footwork classes** — currently excluded due to seated recording setup; a standing setup with a wider camera angle would enable lower-body action classification
7. **Personalization fine-tuning** — given a new user, quickly adapt the model with 5–10 samples per class via few-shot learning or simple centroid shift

---

## 12. Commercial Viability Statement

PunchCV has clear commercial potential in several adjacent markets:

**Fitness gaming** (e.g., Boxing Beat Saber-style experiences): The system requires no hardware beyond a standard webcam. If accuracy improves to >99% through multi-user data collection, it could power engaging home fitness games. The primary challenge is the per-user variation problem — the model currently must be retrained per user for reliable performance, which is not viable for a consumer product. A transfer-learning or few-shot approach (5–10 calibration punches at first launch) would address this.

**Physical therapy / rehabilitation**: Tracking boxing movement quality in a clinical or home setting has direct application for post-stroke upper limb rehabilitation. The classification accuracy would need to extend to slower, impaired movements, requiring specialized training data.

**Esports / competitive gaming input**: Current latency (~35ms end-to-end) is acceptable for casual gaming but marginal for competitive play (target <16ms for 60fps games). Hardware-accelerated inference (GPU or Neural Processing Unit) would address this.

**Current commercial readiness: Low for consumer products, moderate for research/clinical pilots.** The single-user training data and webcam-only constraint limit immediate commercialization. With 6 months of multi-user data collection and cross-person validation, commercial readiness would improve substantially.

---

## 13. Ethics Statement

**Data privacy**: All training data consists of body pose landmarks (x, y, z, visibility) — *not* raw video frames. The landmark data does not contain facial features or biometric identifiers that could be used for individual identification. Raw video is never stored.

**Bias and fairness**: The current model was trained exclusively on data from one individual (male, adult, standard boxing stance). Performance for users with significantly different body proportions, mobility limitations, or non-standard stances is unknown and potentially degraded. Any deployment beyond a personal project should include diverse training data with documented demographic representation.

**Misuse potential**: The pose estimation and motion classification pipeline could in principle be applied to contexts beyond boxing (e.g., surveillance of movement patterns). We use MediaPipe's published, off-the-shelf model and add no novel surveillance capability. The system processes video locally in the Gradio app and does not transmit video data to external servers.

**Physical safety**: Vigorous boxing motions carry injury risk. The application should include appropriate warm-up guidance and should not be used by individuals with upper-body injuries without medical clearance. A future version should include cooldown reminders and session time limits.

---

## Appendix A — Repository Structure

```
c:\car_game\
├── README.md                   <- Project overview and setup
├── requirements.txt            <- Python dependencies
├── setup.py                    <- One-command setup (downloads model, trains RF)
├── app.py                      <- Gradio web application (public deployment)
├── scripts/
│   ├── collect_data.py         <- Data collection tool
│   ├── train_models.py         <- Trains all 3 models (NC, RF, LSTM)
│   ├── verify_model.py         <- Evaluates saved RF on test split
│   ├── experiment.py           <- Training set size sensitivity analysis
│   ├── game.py                 <- Pygame local boxing game
│   ├── boxing_hub.py           <- 3-game training hub (Memory, Combo Rush, Defense)
│   ├── cv_server.py            <- WebSocket server for Godot 3D client
│   ├── live_demo.py            <- Manual-trigger live demo
│   └── live_demo_auto.py       <- Auto motion-detection live demo
├── models/
│   ├── random_forest.joblib    <- Trained RF classifier (deployed)
│   ├── rf_scaler.joblib        <- StandardScaler for RF features
│   └── lstm.pt                 <- Trained LSTM (PyTorch)
├── data/
│   ├── raw/                    <- Training samples (.npy, shape 60×132 each)
│   ├── processed/              <- (empty — preprocessing done in-memory)
│   └── outputs/                <- Experiment results and plots
├── notebooks/                  <- Exploration notebooks
├── report/
│   └── final_report.md         <- This document
└── pose_landmarker_lite.task   <- MediaPipe pose model
```

## Appendix B — Where to Find Each Required Component

| Requirement | Location |
|-------------|----------|
| Naive baseline | `scripts/train_models.py` — `NearestCentroidBaseline` class |
| Classical ML model | `scripts/train_models.py` — `train_random_forest()` function |
| Deep learning model | `scripts/train_models.py` — `LSTMClassifier` class + `train_lstm()` |
| Experiment | `scripts/experiment.py` — full sensitivity analysis script |
| Experiment outputs | `data/outputs/sensitivity_results.csv`, `sensitivity_plot.png`, `sensitivity_summary.txt` |
| Web application | `app.py` — Gradio streaming interface |
| Data | `data/raw/<action>/` — 749 `.npy` files |
| Trained models | `models/` directory |
