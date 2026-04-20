"""
app.py — PunchCV: Real-Time Boxing Action Recognition
Public web interface deployed on HuggingFace Spaces.

Uses webcam + MediaPipe PoseLandmarker + Random Forest to classify
six boxing actions in real time directly in the browser.

Usage:
    python app.py
    # then open http://localhost:7860
"""

import os
import cv2
import numpy as np
import joblib
import gradio as gr
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Paths ──────────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_PATH = os.path.join(_DIR, "pose_landmarker_lite.task")
RF_PATH = os.path.join(_DIR, "models", "random_forest.joblib")
SCALER_PATH = os.path.join(_DIR, "models", "rf_scaler.joblib")

# ── Constants ──────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 60
NUM_LANDMARKS = 33
FEATURE_DIM = NUM_LANDMARKS * 4  # x, y, z, visibility

# Landmark indices for normalization (matches train_models.py)
L_HIP = 23 * 4
R_HIP = 24 * 4
L_SHOULDER = 11 * 4
R_SHOULDER = 12 * 4

ACTION_LABELS = {
    "idle": "Idle (neutral stance)",
    "jab": "Jab (lead-hand straight)",
    "cross": "Cross (rear-hand straight)",
    "hook": "Hook (horizontal arc)",
    "uppercut": "Uppercut (upward punch)",
    "block_high": "Block High (arms up)",
}

# ── Load models ────────────────────────────────────────────────────────────
rf_model = joblib.load(RF_PATH)
scaler = joblib.load(SCALER_PATH)

# ── MediaPipe setup (IMAGE mode — stateless per-frame) ────────────────────
_base_opts = mp_python.BaseOptions(model_asset_path=TASK_PATH)
_landmarker_opts = mp_vision.PoseLandmarkerOptions(
    base_options=_base_opts,
    running_mode=mp_vision.RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = mp_vision.PoseLandmarker.create_from_options(_landmarker_opts)


# ── Pose processing (matches train_models.py exactly) ─────────────────────

def _landmarks_to_array(pose_landmarks) -> np.ndarray:
    """Convert MediaPipe landmark list to flat float32 array of shape (132,)."""
    return np.array(
        [[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_landmarks],
        dtype=np.float32,
    ).flatten()


def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Hip-center + torso-scale normalization (matches train_models.py)."""
    f = frame.copy()
    cx = (f[L_HIP] + f[R_HIP]) / 2
    cy = (f[L_HIP + 1] + f[R_HIP + 1]) / 2
    cz = (f[L_HIP + 2] + f[R_HIP + 2]) / 2
    sx = (f[L_SHOULDER] + f[R_SHOULDER]) / 2
    sy = (f[L_SHOULDER + 1] + f[R_SHOULDER + 1]) / 2
    sz = (f[L_SHOULDER + 2] + f[R_SHOULDER + 2]) / 2
    torso_len = max(np.sqrt((sx - cx) ** 2 + (sy - cy) ** 2 + (sz - cz) ** 2), 0.01)
    for lm in range(NUM_LANDMARKS):
        idx = lm * 4
        f[idx] = (f[idx] - cx) / torso_len
        f[idx + 1] = (f[idx + 1] - cy) / torso_len
        f[idx + 2] = (f[idx + 2] - cz) / torso_len
    return f


def _extract_statistics(seq: np.ndarray) -> np.ndarray:
    """Temporal statistics: std + range + delta → shape (1, 396)."""
    X = seq[np.newaxis]  # (1, 60, 132)
    std = np.std(X, axis=1)
    rng = np.max(X, axis=1) - np.min(X, axis=1)
    delta = X[:, -1, :] - X[:, 0, :]
    return np.concatenate([std, rng, delta], axis=1)  # (1, 396)


# ── Gradio streaming callback ──────────────────────────────────────────────

def process_frame(image: np.ndarray, frame_buffer: list):
    """
    Called on every webcam frame by Gradio's streaming interface.

    Args:
        image: RGB numpy array from webcam.
        frame_buffer: Stateful list of up to SEQUENCE_LENGTH normalized frames.

    Returns:
        (status_text, label_confidences, updated_frame_buffer)
    """
    if image is None:
        return "Waiting for camera...", {}, frame_buffer

    # Detect pose
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        return (
            "No pose detected — step back so your full upper body is visible.",
            {},
            frame_buffer,
        )

    # Extract + normalize landmark vector
    raw = _landmarks_to_array(result.pose_landmarks[0])
    normed = _normalize_frame(raw)

    # Append to sliding window
    frame_buffer = (frame_buffer + [normed])[-SEQUENCE_LENGTH:]

    n = len(frame_buffer)
    if n < SEQUENCE_LENGTH:
        return f"Warming up… {n}/{SEQUENCE_LENGTH} frames", {}, frame_buffer

    # Build feature vector and classify
    seq = np.array(frame_buffer, dtype=np.float32)  # (60, 132)
    feats = _extract_statistics(seq)                 # (1, 396)
    feats_scaled = scaler.transform(feats)
    proba = rf_model.predict_proba(feats_scaled)[0]

    label_conf = {
        ACTION_LABELS.get(cls, cls): float(p)
        for cls, p in zip(rf_model.classes_, proba)
    }
    pred = rf_model.classes_[int(np.argmax(proba))]
    conf = float(np.max(proba))

    status = f"Detected: {pred.upper().replace('_', ' ')}  ({conf * 100:.0f}% confidence)"
    return status, label_conf, frame_buffer


# ── UI ─────────────────────────────────────────────────────────────────────

_CSS = """
#title { text-align: center; }
#status { font-size: 1.4em; font-weight: bold; text-align: center; padding: 10px; }
"""

with gr.Blocks(title="PunchCV — Boxing Action Recognition", css=_CSS, theme=gr.themes.Monochrome()) as demo:

    gr.Markdown(
        """
# PunchCV — Real-Time Boxing Action Recognition
**Your body is the controller.**
Stand ~2 m from your webcam in a boxing stance. Throw a punch — the model classifies it in real time.

> Best results: good front lighting, full upper body visible, plain background.
        """,
        elem_id="title",
    )

    frame_buffer_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=1):
            webcam_input = gr.Image(
                sources=["webcam"],
                streaming=True,
                mirror_webcam=True,
                label="Your Camera",
                height=400,
            )
        with gr.Column(scale=1):
            status_box = gr.Textbox(
                label="Current Action",
                interactive=False,
                elem_id="status",
            )
            confidence_chart = gr.Label(
                label="Confidence per Class",
                num_top_classes=6,
            )

    gr.Markdown(
        """
---
### How it works
| Step | What happens |
|---|---|
| 1 | Webcam frames stream to the server |
| 2 | [MediaPipe PoseLandmarker](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) extracts 33 body landmarks (x, y, z, visibility) |
| 3 | Per-frame normalization: hip-centered, torso-scaled |
| 4 | 60-frame sliding window → temporal statistics: std + range + Δ = **396 features** |
| 5 | **Random Forest** (200 trees) → action label + confidence |

**96.4% accuracy** on 6-class held-out test set · **~3 ms** inference · Model trained on 699 samples

### Actions recognized
`jab` · `cross` · `hook` · `uppercut` · `block_high` · `idle`
        """
    )

    webcam_input.stream(
        fn=process_frame,
        inputs=[webcam_input, frame_buffer_state],
        outputs=[status_box, confidence_chart, frame_buffer_state],
    )


if __name__ == "__main__":
    demo.launch(share=False)
