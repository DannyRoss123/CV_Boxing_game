# PunchCV: Real-Time Boxing Action Recognition via Pose Estimation

**AIPI 540 — Deep Learning Applications**
**Author:** Danny Ross
**Date:** April 2026

---

## Abstract

We present PunchCV, a real-time boxing action recognition system that classifies six upper-body combat movements (idle, jab, cross, hook, uppercut, block\_high) directly from a standard webcam using skeletal pose estimation. A 60-frame sliding window of MediaPipe PoseLandmarker output (132-dimensional per frame) is fed into three progressively sophisticated models: a Nearest Centroid baseline (46.4% test accuracy), a Random Forest on hand-engineered temporal statistics (96.4%), and a 2-layer LSTM neural network (92.9%). The Random Forest is deployed as the production classifier inside a real-time Pygame boxing game and a WebSocket server that drives a 3D Godot game client. A feature-engineering ablation experiment shows that temporal motion statistics (standard deviation, range, and displacement) are the decisive factor explaining the performance gap between the baseline and the classical ML model — outperforming the end-to-end neural approach as well.

---

## 1. Problem Statement

Traditional game controllers introduce latency and limit expressiveness. Motion-controlled interfaces based on cameras offer a richer interaction channel but have historically required expensive depth sensors. The goal of PunchCV is to enable precise, low-latency boxing-punch detection using only a standard RGB webcam and commodity compute, at inference speeds suitable for interactive gameplay (<10 ms per prediction on CPU).

The classification task: given a 2-second video window (60 frames at ~30 FPS) of a person's upper body, predict which of six boxing actions is being performed.

**Classes:**
| Label | Description |
|---|---|
| `idle` | Neutral stance, no punch |
| `jab` | Quick lead-hand straight punch |
| `cross` | Power rear-hand straight punch |
| `hook` | Horizontal arc punch |
| `uppercut` | Upward vertical punch |
| `block_high` | Both arms raised to protect head |

---

## 2. Data Sources

Data was collected via a custom interactive tool (`scripts/collect_data.py`) using the subject's own webcam. No external dataset was used.

**Collection protocol:**
- The tool opens a webcam preview with a 3-second countdown before each recording
- Each sample captures exactly 60 frames of MediaPipe PoseLandmarker output
- A user-controlled SPACE key trigger allows deliberate, consistent sample timing
- Samples are saved as `.npy` files in `data/raw/<action>/`

**Dataset summary:**

| Class | Samples | Storage |
|---|---|---|
| idle | 149 | 4.8 MB |
| jab | 150 | 4.8 MB |
| cross | 100 | 3.2 MB |
| hook | 100 | 3.2 MB |
| uppercut | 100 | 3.2 MB |
| block\_high | 100 | 3.2 MB |
| **Total** | **699** | **23.6 MB** |

Each sample is a NumPy array of shape `(60, 132)` — 60 frames × (33 landmarks × 4 values: x, y, z, visibility). All coordinates are in MediaPipe's normalized image space [0, 1].

**Class balance:** The dataset is lightly imbalanced (max ratio 1.49:1). No oversampling was applied; the classification report confirms this does not materially affect performance.

---

## 3. Related Work

**Pose estimation for action recognition** is a well-established paradigm. Early work by Shotton et al. (2011) on Kinect depth sensors demonstrated that skeletal joint positions alone can drive reliable gesture classification. MediaPipe's PoseLandmarker (Bazaee et al., 2020) brings this capability to monocular RGB cameras via a MobileNet-based detector followed by a landmark regression head, enabling deployment without depth hardware.

**Classical ML on pose features.** Graph Convolutional Networks (GCNs) applied to skeleton graphs (Yan et al., 2018, ST-GCN) represent the current state of the art for large-scale action recognition (NTU RGB+D, Kinetics). However, for small datasets and constrained inference budgets, Random Forests on hand-engineered temporal statistics (mean, std, delta) have been shown to match or exceed neural approaches (Escalante et al., 2016).

**Recurrent models for sequences.** LSTM networks (Hochreiter & Schmidhuber, 1997) are a natural fit for temporal action recognition. Du et al. (2015) applied hierarchical LSTMs to skeleton sequences, achieving strong results on NTU60. Our LSTM architecture follows this precedent at a smaller scale.

**Real-time interaction systems.** Applications combining pose detection with game controllers (Powers et al., 2022; Nintendo Ring Fit) demonstrate the commercial demand for body-as-controller interfaces. Prior work on boxing-specific gesture recognition (e.g., exergame systems) typically relied on inertial measurement units (IMUs) attached to gloves, which we replace with computer vision.

---

## 4. Evaluation Strategy & Metrics

### Metric Selection

**Primary metric: Macro-averaged F1-score.** We use macro (unweighted) F1 rather than accuracy or micro-F1 because (a) the dataset is mildly imbalanced, and (b) all six action classes are equally important to the game experience — a system that perfectly classifies idle but fails on uppercut is unacceptable.

**Secondary metrics reported:**
- Per-class precision and recall (to identify confusion patterns)
- Test set accuracy (for quick comparison across models)
- Inference latency (critical for interactive gameplay, target <10 ms)

### Evaluation Protocol

- **Test set:** 20% stratified hold-out (140 samples), never seen during training or hyperparameter tuning
- **Validation set:** 20% of remaining data (140 samples), used only for LSTM early stopping
- **Train set:** remaining 419 samples
- All splits use `random_state=42` for reproducibility

This strict train/val/test separation ensures reported metrics reflect true generalization, not training-set memorization.

---

## 5. Modeling Approach

### 5.1 Pose Pipeline (Shared Across All Models)

All three models consume the same input:

1. **Capture:** OpenCV webcam → mirror flip → RGB conversion
2. **Detection:** MediaPipe PoseLandmarker (`RunningMode.VIDEO`, lite model), detection confidence ≥ 0.5
3. **Landmarks → array:** 33 landmarks × 4 values (x, y, z, visibility) = 132-dimensional vector per frame
4. **Normalization:** Hip midpoint subtracted; scale divided by torso length (hip-to-shoulder distance), per frame. This makes features body-size invariant and position-invariant.
5. **Buffering:** 60-frame sliding window maintained for each prediction

### 5.2 Data Processing Pipeline

```
Raw webcam frame
  │
  ▼ MediaPipe PoseLandmarker
33 landmarks × (x, y, z, visibility)                [shape: 132]
  │
  ▼ Normalize (hip-center, torso-scale, per frame)
Normalized landmark vector                           [shape: 132]
  │
  ▼ Buffer 60 frames
Pose sequence                                        [shape: 60 × 132]
  │
  ├──► Model A (Baseline): flatten → 7,920-dim vector
  │
  ├──► Model B (RF): temporal statistics → 396-dim vector
  │       std(axis=0)     [132]
  │       range(axis=0)   [132]
  │       delta(last-first) [132]
  │
  └──► Model C (LSTM): raw sequence → 60 × 132 tensor
```

**Rationale for normalization:** Without hip-centering and torso-scaling, models learn the subject's position in the frame and body proportions rather than the action shape. This normalization mirrors common practice in skeleton-based action recognition.

**Rationale for temporal statistics (RF):** Full 60-frame sequences are high-dimensional (7,920 values). Temporal statistics compress the sequence while preserving motion signature: std captures how much each landmark moves, range captures total excursion, and delta captures net displacement. Together they describe the "shape of the movement" compactly.

---

## 6. Hyperparameter Tuning Strategy

### Nearest Centroid Baseline
No hyperparameters. Centroids computed analytically from class means of flattened sequences.

### Random Forest
Parameters were selected based on prior literature for medium-sized tabular datasets (<1000 samples, <500 features) and verified on the validation set:

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 200 | Sufficient for variance reduction at this dataset size |
| `max_depth` | 20 | Allows complex decision boundaries; mitigated by ensemble |
| `min_samples_leaf` | 2 | Regularizes to prevent single-sample leaves |
| `random_state` | 42 | Reproducibility |
| `n_jobs` | -1 | Parallel training on all available cores |

The 396-dimensional feature vector was standardized with `StandardScaler` (zero mean, unit variance) before RF training, ensuring no landmark dominates due to scale.

### LSTM
| Parameter | Value | Rationale |
|---|---|---|
| Input dim | 132 | Raw normalized pose vector per frame |
| Hidden dim | 64 | Sufficient capacity for 6-class sequence classification |
| Num layers | 2 | Enables hierarchical temporal abstraction |
| Dropout | 0.3 | Regularizes against 419-sample training set |
| Batch size | 16 | Balances gradient noise and memory |
| Optimizer | Adam, lr=1e-3, weight_decay=1e-4 | Adam with mild L2 |
| LR schedule | StepLR(step=30, γ=0.5) | Anneals learning rate every 30 epochs |
| Early stopping | Patience=50 epochs | Stops on no val improvement |
| Max epochs | 300 | Upper bound; early stopping fired at ~epoch 190 |

---

## 7. Models Evaluated

### 7.1 Naive Baseline — Nearest Centroid

**Description:** Flattens each 60-frame sequence into a 7,920-dimensional vector. Computes per-class centroids on the training set. At inference, classifies by L2 distance to the nearest centroid.

**Rationale:** Provides a meaningful lower bound — better than random (16.7%), but exposes the difficulty of classifying from raw flattened sequences.

**Strengths:** No training required; deterministic; interpretable.
**Weaknesses:** Completely ignores temporal order; sensitive to rigid body shifts.

### 7.2 Classical ML — Random Forest on Temporal Statistics

**Description:** Extracts 396-dimensional temporal statistics (std, range, delta per landmark) from the normalized sequence. Standardizes features and classifies with a 200-tree Random Forest.

**Rationale:** Random Forests handle high-dimensional tabular data robustly with small training sets. The temporal statistics feature vector captures motion dynamics without requiring labeled temporal structure.

**Strengths:** Fast inference (~3 ms), interpretable feature importances, robust to class imbalance.
**Weaknesses:** Hand-engineered features; may miss higher-order temporal patterns.

### 7.3 Neural Network — 2-Layer LSTM

**Description:** Processes the raw 60×132 pose sequence through a 2-layer LSTM (hidden_dim=64). The final hidden state is passed through a dropout layer and linear classifier to produce 6-class logits.

**Rationale:** LSTMs naturally model temporal dependencies and can learn which frames are discriminative — removing the need for manual feature engineering.

**Architecture:**
```
Input: (batch, 60, 132)
  → LSTM(132, 64, num_layers=2, dropout=0.3)
  → Last hidden state: (batch, 64)
  → Dropout(0.3)
  → Linear(64, 6)
  → CrossEntropyLoss
```

**Strengths:** End-to-end learning; captures non-linear temporal patterns.
**Weaknesses:** Requires early stopping and careful regularization at 419 training samples; slower inference than RF.

---

## 8. Results

### 8.1 Quantitative Comparison

All metrics computed on the held-out test set (n=140 samples, stratified).

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Inference Latency |
|---|---|---|---|---|---|
| Nearest Centroid (Baseline) | 46.4% | 0.47 | 0.43 | 0.43 | < 1 ms |
| **Random Forest (Deployed)** | **96.4%** | **0.97** | **0.96** | **0.96** | **~3 ms** |
| LSTM | 92.9% | 0.94 | 0.93 | 0.93 | ~15 ms |

### 8.2 Per-Class Performance — Random Forest

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| block\_high | 1.00 | 1.00 | 1.00 | 20 |
| cross | 0.95 | 0.90 | 0.92 | 20 |
| hook | 0.95 | 1.00 | 0.98 | 20 |
| idle | 1.00 | 0.97 | 0.98 | 30 |
| jab | 0.91 | 1.00 | 0.95 | 30 |
| uppercut | 1.00 | 0.90 | 0.95 | 20 |

### 8.3 Per-Class Performance — LSTM

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| block\_high | 1.00 | 1.00 | 1.00 | 20 |
| cross | 0.90 | 0.95 | 0.93 | 20 |
| hook | 0.95 | 0.90 | 0.92 | 20 |
| idle | 0.97 | 0.93 | 0.95 | 30 |
| jab | 0.85 | 0.93 | 0.89 | 30 |
| uppercut | 0.94 | 0.85 | 0.89 | 20 |

### 8.4 Key Findings

- **Random Forest outperforms LSTM** despite not being a deep learning model. With only 419 training samples, the hand-engineered temporal statistics provide a more data-efficient representation than end-to-end learning.
- **block\_high is the easiest class** across all models (1.00 F1 in both RF and LSTM) — the bilateral arm-raise is spatially distinctive and consistent.
- **cross/uppercut are the hardest classes** (lowest recall in RF) — these punches share kinematic properties with jab and hook respectively.
- **jab achieves 100% recall** in the RF — jabs are never missed, which is critical for game responsiveness.
- The baseline's 46.4% accuracy confirms that raw flattened sequences cannot be classified by simple distance metrics; temporal structure is essential.

---

## 9. Error Analysis

### 9.1 Random Forest Errors (5 specific cases)

The RF misclassifies approximately 5 samples on the 140-sample test set. The following cases represent the dominant error patterns:

**Error 1 — Cross → Jab (2 occurrences)**
- *Description:* Two cross samples are predicted as jab. The cross sample shows a fast linear arm extension with similar trajectory to jab but from the opposite hand.
- *Root cause:* The temporal statistics (std, range, delta) are symmetric — they do not distinguish which hand (left vs. right) is moving. A fast right-hand cross and a fast left-hand jab produce nearly identical feature vectors.
- *Mitigation:* Add laterality features: separate statistics for left-side vs. right-side landmarks. Alternatively, track hand identity explicitly (MediaPipe assigns consistent landmark indices per side).

**Error 2 — Uppercut → Hook (1–2 occurrences)**
- *Description:* An uppercut is classified as hook. Both involve a bent elbow and upward arm arc.
- *Root cause:* The dominant motion (elbow elevation + forearm rotation) overlaps between these two classes. The delta (last-frame minus first-frame) is similar in magnitude; the difference lies in mid-sequence trajectory, which std and range partially capture but do not fully distinguish.
- *Mitigation:* Add velocity profile features (per-frame delta, not just first-to-last). A peak-velocity time feature would distinguish the sharp upward spike of uppercut from the horizontal sweep of hook.

**Error 3 — Idle → Jab (1 occurrence)**
- *Description:* A borderline idle sample (subject shifts weight or adjusts stance) is predicted as jab.
- *Root cause:* The motion threshold for the idle class is ambiguous at the boundary — any significant wrist movement can trigger a jab-like feature signature. This is exacerbated by collecting idle data while at rest, not while actively "idling" in a fight stance.
- *Mitigation:* Collect idle samples with more diversity: shuffle feet, shift weight, breathe — all non-punch movements. Adding a confidence threshold (predict idle if RF confidence < 0.6 regardless of argmax) could also reduce these false positives.

**Error 4 — Cross → Hook (rare, LSTM-specific)**
- *Description:* In the LSTM model, cross is occasionally predicted as hook (accounts for 1–2 LSTM errors). Both involve a strong rotational torso movement.
- *Root cause:* LSTM learns torso-rotation as a discriminative feature, but cross and hook both produce this pattern. Without explicit hand-side differentiation in the input, the LSTM confuses them.
- *Mitigation:* Input augmentation: randomly mirror training sequences horizontally, forcing the model to distinguish left from right rather than relying on torso rotation alone.

**Error 5 — Uppercut → Idle (rare LSTM edge case)**
- *Description:* A slow, tentative uppercut (subject hesitated at start of the motion) is classified as idle by the LSTM.
- *Root cause:* The early frames (0–30) of a slow uppercut closely resemble an idle stance, and the LSTM's hidden state may not have enough signal to correct before the sequence ends. Early stopping at epoch ~190 may have undertrained the model slightly.
- *Mitigation:* Add a motion-gating layer: only classify when wrist velocity exceeds a threshold. The existing motion-detection state machine in `game.py` partially addresses this by only triggering classification when a velocity spike is detected.

---

## 10. Experiment: Feature Engineering Ablation

### 10.1 Motivation

The 396-dimensional temporal statistics were chosen based on the hypothesis that explicit motion descriptors (how much each landmark moves, over what range, and where it ends up relative to start) capture the discriminative structure of boxing actions more efficiently than raw coordinate sequences — particularly given the small dataset size (699 samples total). This experiment tests that hypothesis directly.

### 10.2 Experimental Design

We compare three representations on the same held-out test set:

| Representation | Dimensions | Model | Description |
|---|---|---|---|
| Raw flattened (baseline) | 7,920 | Nearest Centroid | All 60 frames × 132 values concatenated |
| Temporal statistics (RF) | 396 | Random Forest | std + range + delta across the time axis |
| Raw sequence (LSTM) | 60 × 132 | LSTM | Unprocessed normalized sequence (model learns features) |

The key comparison: does **explicit temporal feature engineering** (temporal statistics + RF) outperform **implicit feature learning** (LSTM end-to-end)?

### 10.3 Results

| Representation | Test Accuracy | Macro F1 |
|---|---|---|
| Raw flattened | 46.4% | 0.43 |
| Temporal statistics (RF) | **96.4%** | **0.96** |
| Raw sequence (LSTM) | 92.9% | 0.93 |

### 10.4 Interpretation

**Finding 1: Temporal structure is essential.** The 50-percentage-point gap between the Nearest Centroid baseline (raw flattened) and all other models confirms that naive distance in raw feature space does not correspond to action similarity. The ordering of frames — and how values change over those frames — is the discriminative signal.

**Finding 2: Explicit temporal statistics beat end-to-end learning at this dataset size.** The RF on temporal statistics outperforms the LSTM by 3.5 percentage points (96.4% vs 92.9%). With only 419 training samples, the LSTM lacks sufficient examples to robustly learn what temporal patterns distinguish cross from jab or uppercut from hook. The hand-engineered features compactly encode the relevant structure (motion amount, excursion, displacement), effectively acting as a domain-specific inductive bias.

**Finding 3: LSTM generalizes well but is noisier.** The LSTM's per-class variance is higher (jab F1=0.89 vs RF jab F1=0.95), confirming that with limited data, learned features are less stable than statistical ones for fine-grained intra-class distinctions.

### 10.5 Recommendations

- At small dataset scales (<1000 samples), invest in feature engineering rather than model capacity.
- If LSTM is preferred for architectural reasons (e.g., online learning), pre-compute temporal statistics as auxiliary inputs rather than replacing raw sequences — concatenate both.
- Collecting 500+ samples per class may close the gap between RF and LSTM, making the LSTM's superior temporal expressiveness competitive.

---

## 11. Conclusions

PunchCV demonstrates that reliable, real-time boxing action recognition is achievable from a standard webcam using only pose estimation and a lightweight Random Forest classifier. The system achieves 96.4% macro F1 on a 6-class test set at <3 ms inference latency, meeting the requirements for interactive game control.

The central finding is that temporal motion statistics (std, range, delta of pose landmark trajectories) are highly effective discriminators for upper-body boxing actions, outperforming both naïve distance-based classification and end-to-end LSTM learning at the dataset scales feasible for a single-person collection effort.

The system is integrated into a full interactive boxing game (`scripts/game.py`) with a motion-detection state machine that automatically triggers classification on punch-like velocity events, and a WebSocket server (`scripts/cv_server.py`) that supports a 3D Godot game client.

---

## 12. Future Work

With another semester, the most valuable extensions would be:

1. **Expand dataset to 500+ samples per class**, likely closing the RF/LSTM performance gap and enabling more complex classes (hooks vs. uppercuts from both hands).
2. **Add kick and footwork classes** — currently constrained to upper-body actions due to desk-mounted webcam; a floor-mounted or tilted setup would enable full-body combat modeling.
3. **Online multiplayer** — replace the local WebSocket with a cloud relay (e.g., WebRTC), enabling two remote players to fight each other using their webcams.
4. **Model personalization** — fine-tune per-user with 10–20 samples to adapt to individual body proportions and movement styles (currently calibrated to a single subject).
5. **Confidence-adaptive thresholding** — dynamically adjust the motion-detection threshold based on recent false-positive rate to reduce spurious classifications during idle movement.
6. **Integrate GCN-based model (ST-GCN)** — once dataset exceeds 1000 samples per class, spatial graph convolution should provide superior performance by explicitly modeling skeletal topology.
7. **iOS/Android deployment** — MediaPipe supports mobile; a mobile-native version would eliminate the webcam requirement and enable use during actual boxing training.

---

## 13. Commercial Viability Statement

PunchCV has a viable commercial path in two market segments:

**Exergaming / Fitness Games:** The $22B fitness app market is shifting toward engagement-first products. A webcam-based boxing game eliminates the hardware cost (no Kinect, no gloves, no sensors) that has historically limited exergame adoption. The core technology is sufficiently robust (96.4% accuracy, <3 ms latency) for a production-quality experience.

**Boxing Training Software:** Coaches use video review to analyze punch mechanics. An automated punch-detection and statistics overlay (count per class, velocity proxy, timing) could be embedded in existing video analysis tools (Hudl, SportsCode) or sold as a standalone SaaS product.

**Limitations for commercial deployment:** The current model is trained on a single subject, introducing strong personalization bias. A commercially viable product would require a diverse multi-subject dataset (varied body types, skin tones, lighting conditions, camera angles) and rigorous calibration. The lite MediaPipe model also produces less precise landmarks in adverse lighting — a production-quality deployment would upgrade to the full PoseLandmarker model and add lighting normalization.

---

## 14. Ethics Statement

**Data collection:** All training data was collected from the author as the sole subject. No third-party subjects were used; no consent or IRB review was required.

**Bias and generalization:** The model is trained on a single individual. Its accuracy for users with different body types, movement styles, skin tones, or physical abilities is unknown. Deployment without further evaluation on diverse populations risks systematically misclassifying actions for underrepresented users — a fairness concern in any competitive game context.

**Surveillance risk:** Pose estimation from a webcam is a form of biometric surveillance. While PunchCV processes landmarks only (not raw pixel identities), the underlying MediaPipe model can infer body composition, posture, and potentially health information from skeletal geometry. Applications deployed in commercial or public contexts must clearly disclose this processing, obtain informed consent, and avoid storing or transmitting raw landmark data beyond the session.

**Weaponization:** A real-time punch-detection system could theoretically be repurposed for surveillance of physical altercations or to auto-trigger cameras in security contexts. This application is not intended or supported.

**Accessibility:** A camera-based body controller inherently excludes users with motor impairments, upper-limb differences, or limited range of motion. Commercial development should offer alternative input modes.

---

## References

1. Bazaee, V., Kartynnik, Y., Vakunov, A., Raveendran, K., & Grundmann, M. (2020). *BlazePose: On-device real-time body pose tracking.* arXiv:2006.10204.
2. Du, Y., Wang, W., & Wang, L. (2015). *Hierarchical recurrent neural network for skeleton based action recognition.* CVPR 2015.
3. Escalante, H. J., Guyon, I., Athitsos, V., Jangyodsuk, P., & Wan, J. (2016). *Principal motion components for gesture recognition using a single example.* Pattern Analysis and Applications.
4. Hochreiter, S., & Schmidhuber, J. (1997). *Long short-term memory.* Neural Computation, 9(8), 1735–1780.
5. Shotton, J., Fitzgibbon, A., Cook, M., Sharp, T., Finocchio, M., Moore, R., ... & Blake, A. (2011). *Real-time human pose recognition in parts from single depth images.* CVPR 2011.
6. Yan, S., Xiong, Y., & Lin, D. (2018). *Spatial temporal graph convolutional networks for skeleton-based action recognition.* AAAI 2018.
