"""
app.py — PunchCV: Real-Time Boxing Action Recognition
Public web interface deployed on HuggingFace Spaces.
"""

import os
import time
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
RF_PATH   = os.path.join(_DIR, "models", "random_forest.joblib")
SCALER_PATH = os.path.join(_DIR, "models", "rf_scaler.joblib")

# ── Constants ──────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 60
NUM_LANDMARKS   = 33
FEATURE_DIM     = NUM_LANDMARKS * 4

L_HIP      = 23 * 4
R_HIP      = 24 * 4
L_SHOULDER = 11 * 4
R_SHOULDER = 12 * 4

MIN_CONF = 0.30

PUNCH_LABELS  = ["jab", "cross", "hook", "uppercut", "block_high"]
ALL_LABELS    = PUNCH_LABELS + ["idle"]

LABEL_DISPLAY = {
    "jab":        "JAB",
    "cross":      "CROSS",
    "hook":       "HOOK",
    "uppercut":   "UPPERCUT",
    "block_high": "BLOCK HIGH",
    "idle":       "IDLE",
}

LABEL_COLOR = {
    "jab":        "#F59E0B",
    "cross":      "#EF4444",
    "hook":       "#8B5CF6",
    "uppercut":   "#06B6D4",
    "block_high": "#10B981",
    "idle":       "#6B7280",
}

SKILL_TIPS = {
    "Beginner": {
        "jab":        "Quick lead-hand straight — snap it out and pull back fast.",
        "cross":      "Rear-hand straight with hip rotation for power.",
        "hook":       "Horizontal arc at 90 degrees, elbow parallel to floor.",
        "uppercut":   "Drive upward with your legs — don't just swing your arm.",
        "block_high": "Both arms up, forearms forward — protect that chin.",
    },
    "Intermediate": {
        "jab":        "Use the jab to set up your cross — double-jab variations.",
        "cross":      "Pivot on your rear foot as you throw — full hip drive.",
        "hook":       "Short hook off the jab — same rhythm, tighter arc.",
        "uppercut":   "Slip outside then rip the uppercut — classic counter.",
        "block_high": "High guard then immediately counter-punch — don't just sit there.",
    },
    "Advanced": {
        "jab":        "Paw jab to disrupt rhythm — no tell in the shoulder.",
        "cross":      "Cross-counter off their jab: slip, cross, body-hook.",
        "hook":       "Liver shot hook to the body — bend your knees.",
        "uppercut":   "Uppercut in tight range — great after slipping a cross.",
        "block_high": "Catch and roll — absorb with the glove then turn into it.",
    },
    "Pro": {
        "jab":        "Triple jab: range-finder, set-up, power — vary speed intentionally.",
        "cross":      "Cross to the body then upstairs — work the two levels.",
        "hook":       "Left hook as the money punch after slipping the jab.",
        "uppercut":   "Double uppercut in the pocket — inside fighting game.",
        "block_high": "High guard bait then walk them onto a counter-cross.",
    },
}


# ── Load models ────────────────────────────────────────────────────────────
rf_model = joblib.load(RF_PATH)
scaler   = joblib.load(SCALER_PATH)

# ── MediaPipe setup ────────────────────────────────────────────────────────
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


# ── Pose pipeline ──────────────────────────────────────────────────────────

def _landmarks_to_array(pose_landmarks) -> np.ndarray:
    return np.array(
        [[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_landmarks],
        dtype=np.float32,
    ).flatten()


def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    f = frame.copy()
    cx = (f[L_HIP] + f[R_HIP]) / 2
    cy = (f[L_HIP+1] + f[R_HIP+1]) / 2
    cz = (f[L_HIP+2] + f[R_HIP+2]) / 2
    sx = (f[L_SHOULDER] + f[R_SHOULDER]) / 2
    sy = (f[L_SHOULDER+1] + f[R_SHOULDER+1]) / 2
    sz = (f[L_SHOULDER+2] + f[R_SHOULDER+2]) / 2
    torso = max(np.sqrt((sx-cx)**2 + (sy-cy)**2 + (sz-cz)**2), 0.01)
    for i in range(NUM_LANDMARKS):
        idx = i * 4
        f[idx]   = (f[idx]   - cx) / torso
        f[idx+1] = (f[idx+1] - cy) / torso
        f[idx+2] = (f[idx+2] - cz) / torso
    return f


def _extract_statistics(seq: np.ndarray) -> np.ndarray:
    X   = seq[np.newaxis]
    std = np.std(X, axis=1)
    rng = np.max(X, axis=1) - np.min(X, axis=1)
    dlt = X[:, -1, :] - X[:, 0, :]
    return np.concatenate([std, rng, dlt], axis=1)


# ── HTML builders ──────────────────────────────────────────────────────────

def _punch_counters_html(counts: dict) -> str:
    items = "".join(
        f"""<div class="punch-counter">
              <div class="punch-label" style="color:{LABEL_COLOR[k]}">{LABEL_DISPLAY[k]}</div>
              <div class="punch-count">{counts.get(k, 0)}</div>
            </div>"""
        for k in PUNCH_LABELS
    )
    return f'<div class="punch-counters">{items}</div>'


def _action_display_html(action: str, conf: float, tip: str) -> str:
    color = LABEL_COLOR.get(action, "#6B7280")
    label = LABEL_DISPLAY.get(action, action.upper())
    bar_w = int(conf * 100)
    tip_html = f'<div class="action-tip">{tip}</div>' if tip else ""
    return f"""
<div class="action-display">
  <div class="action-name" style="color:{color}">{label}</div>
  <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{bar_w}%;background:{color}"></div></div>
  <div class="conf-text">{conf*100:.0f}% confidence</div>
  {tip_html}
</div>"""


def _history_html(history: list) -> str:
    if not history:
        return '<div class="history-empty">No punches yet — throw something!</div>'
    rows = "".join(
        f'<div class="hist-row"><span style="color:{LABEL_COLOR.get(a,"#fff")}">{LABEL_DISPLAY.get(a,a)}</span>'
        f'<span class="hist-conf">{c*100:.0f}%</span></div>'
        for a, c in reversed(history[-8:])
    )
    return f'<div class="history-feed">{rows}</div>'


def _session_header_html(name: str, level: str, total: int, elapsed_s: float) -> str:
    mins = int(elapsed_s) // 60
    secs = int(elapsed_s) % 60
    return f"""
<div class="session-header">
  <span class="fighter-name">{name or "Fighter"}</span>
  <span class="skill-badge" style="background:{_level_color(level)}">{level}</span>
  <span class="stat-pill">&#x1F44A; {total} punches</span>
  <span class="stat-pill">&#x23F1; {mins:02d}:{secs:02d}</span>
</div>"""


def _setup_tips_html(level: str) -> str:
    tips = SKILL_TIPS.get(level, {})
    rows = "".join(
        f'<li><strong style="color:{LABEL_COLOR[k]}">{LABEL_DISPLAY[k]}:</strong> {v}</li>'
        for k, v in tips.items()
    )
    return f'<div class="setup-tips"><h3>Tips for {level}</h3><ul>{rows}</ul></div>'


def _summary_html(name: str, level: str, counts: dict, history: list, elapsed_s: float) -> str:
    total = sum(counts.values())
    mins  = int(elapsed_s) // 60
    secs  = int(elapsed_s) % 60
    rate  = total / max(elapsed_s, 1) * 60
    bars  = "".join(
        f"""<div class="sum-bar-row">
              <span class="sum-label" style="color:{LABEL_COLOR[k]}">{LABEL_DISPLAY[k]}</span>
              <div class="sum-bar-bg">
                <div class="sum-bar-fill" style="width:{int(counts.get(k,0)/max(total,1)*100)}%;background:{LABEL_COLOR[k]}"></div>
              </div>
              <span class="sum-count">{counts.get(k,0)}</span>
            </div>"""
        for k in PUNCH_LABELS
    )
    return f"""
<div class="summary-panel">
  <h2 class="sum-title">Session Complete</h2>
  <p class="sum-fighter">{name or "Fighter"} &middot; <span style="color:{_level_color(level)}">{level}</span></p>
  <div class="sum-stats">
    <div class="sum-stat"><div class="sum-num">{total}</div><div class="sum-lbl">Total Punches</div></div>
    <div class="sum-stat"><div class="sum-num">{mins:02d}:{secs:02d}</div><div class="sum-lbl">Duration</div></div>
    <div class="sum-stat"><div class="sum-num">{rate:.1f}</div><div class="sum-lbl">Punches/Min</div></div>
  </div>
  <div class="sum-breakdown">{bars}</div>
</div>"""


def _level_color(level: str) -> str:
    return {"Beginner": "#10B981", "Intermediate": "#F59E0B",
            "Advanced": "#EF4444", "Pro": "#8B5CF6"}.get(level, "#6B7280")


# ── Callbacks ─────────────────────────────────────────────────────────────

def start_session(name: str, level: str):
    state = {
        "name":    name.strip() or "Fighter",
        "level":   level,
        "counts":  {k: 0 for k in PUNCH_LABELS},
        "history": [],
        "start":   time.time(),
        "buffer":  [],
        "last_action": "idle",
        "last_conf":   0.0,
    }
    return (
        state,
        gr.update(visible=False),   # setup panel
        gr.update(visible=True),    # training panel
        gr.update(visible=False),   # summary panel
        _session_header_html(state["name"], level, 0, 0),
        _action_display_html("idle", 0.0, ""),
        _punch_counters_html(state["counts"]),
        _history_html([]),
    )


def end_session(state: dict):
    elapsed = time.time() - state.get("start", time.time())
    return (
        gr.update(visible=False),   # setup
        gr.update(visible=False),   # training
        gr.update(visible=True),    # summary
        _summary_html(state["name"], state["level"], state["counts"], state["history"], elapsed),
    )


def back_to_hub(state: dict):
    return (
        gr.update(visible=True),    # setup
        gr.update(visible=False),   # training
        gr.update(visible=False),   # summary
    )


def update_tips(level: str):
    return _setup_tips_html(level)


def process_frame(image: np.ndarray, state: dict):
    if image is None or not state:
        blank = '<div class="action-display"><div class="action-name">Waiting for camera...</div></div>'
        return state, blank, _punch_counters_html({}), _history_html([]), ""

    # Detect pose
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    result   = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        msg = '<div class="action-display"><div class="action-name" style="color:#6B7280">No pose — step back</div></div>'
        elapsed = time.time() - state.get("start", time.time())
        header  = _session_header_html(state["name"], state["level"],
                                        sum(state["counts"].values()), elapsed)
        return state, msg, _punch_counters_html(state["counts"]), _history_html(state["history"]), header

    raw    = _landmarks_to_array(result.pose_landmarks[0])
    normed = _normalize_frame(raw)

    buf = (state["buffer"] + [normed])[-SEQUENCE_LENGTH:]
    state["buffer"] = buf

    elapsed = time.time() - state.get("start", time.time())
    header  = _session_header_html(state["name"], state["level"],
                                    sum(state["counts"].values()), elapsed)

    if len(buf) < SEQUENCE_LENGTH:
        warming = f'<div class="action-display"><div class="action-name" style="color:#6B7280">Warming up... {len(buf)}/{SEQUENCE_LENGTH}</div></div>'
        return state, warming, _punch_counters_html(state["counts"]), _history_html(state["history"]), header

    seq        = np.array(buf, dtype=np.float32)
    feats      = _extract_statistics(seq)
    feats_s    = scaler.transform(feats)
    proba      = rf_model.predict_proba(feats_s)[0]
    pred       = rf_model.classes_[int(np.argmax(proba))]
    conf       = float(np.max(proba))

    # Count punch if confident and not idle
    if pred in PUNCH_LABELS and conf >= MIN_CONF:
        state["counts"][pred] = state["counts"].get(pred, 0) + 1
        state["history"].append((pred, conf))
        state["last_action"] = pred
        state["last_conf"]   = conf
    else:
        pred  = state.get("last_action", "idle")
        conf  = state.get("last_conf",   0.0)

    tip = SKILL_TIPS.get(state["level"], {}).get(pred, "") if pred != "idle" else ""
    action_html   = _action_display_html(pred, conf, tip)
    counters_html = _punch_counters_html(state["counts"])
    history_html  = _history_html(state["history"])

    return state, action_html, counters_html, history_html, header


# ── CSS ───────────────────────────────────────────────────────────────────

_CSS = """
body, .gradio-container { background:#0f0f0f !important; color:#e5e5e5 !important; }
.gradio-container { max-width:1100px !important; margin:auto; }

/* Hero */
.hero { text-align:center; padding:2rem 1rem 1rem; }
.hero h1 { font-size:2.6rem; font-weight:900; letter-spacing:2px;
           background:linear-gradient(135deg,#EF4444,#F59E0B);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hero p { color:#9ca3af; font-size:1.05rem; }

/* Setup panel */
.setup-panel { background:#1a1a1a; border-radius:16px; padding:2rem; margin-top:1rem; }
.setup-title { font-size:1.5rem; font-weight:700; color:#f3f4f6; margin-bottom:1.5rem; }
.setup-tips { background:#111; border-radius:10px; padding:1rem 1.5rem; margin-top:1.2rem; }
.setup-tips h3 { color:#9ca3af; font-size:0.9rem; text-transform:uppercase; letter-spacing:1px; }
.setup-tips ul { margin:0.5rem 0 0 1rem; padding:0; }
.setup-tips li { color:#d1d5db; font-size:0.92rem; margin-bottom:0.4rem; }

/* Session header */
.session-header { display:flex; align-items:center; gap:1rem; flex-wrap:wrap;
                  background:#1a1a1a; border-radius:12px; padding:0.8rem 1.2rem; margin-bottom:0.8rem; }
.fighter-name { font-size:1.2rem; font-weight:700; color:#f9fafb; }
.skill-badge  { font-size:0.75rem; font-weight:600; padding:2px 10px; border-radius:99px; color:#fff; }
.stat-pill    { font-size:0.9rem; color:#9ca3af; background:#111; border-radius:8px; padding:3px 10px; }

/* Action display */
.action-display { background:#1a1a1a; border-radius:14px; padding:1.5rem; text-align:center; margin-bottom:0.8rem; }
.action-name  { font-size:3rem; font-weight:900; letter-spacing:3px; line-height:1.1; }
.conf-bar-bg  { background:#2d2d2d; border-radius:99px; height:8px; margin:0.7rem 0; overflow:hidden; }
.conf-bar-fill{ height:100%; border-radius:99px; transition:width 0.2s; }
.conf-text    { color:#9ca3af; font-size:0.85rem; }
.action-tip   { margin-top:0.8rem; font-size:0.9rem; color:#d1d5db; font-style:italic;
                border-top:1px solid #2d2d2d; padding-top:0.8rem; }

/* Punch counters */
.punch-counters { display:flex; gap:0.6rem; flex-wrap:wrap; justify-content:center; margin-bottom:0.8rem; }
.punch-counter  { background:#1a1a1a; border-radius:10px; padding:0.7rem 1rem; text-align:center; flex:1; min-width:80px; }
.punch-label    { font-size:0.7rem; font-weight:700; letter-spacing:1px; text-transform:uppercase; }
.punch-count    { font-size:2rem; font-weight:900; color:#f9fafb; }

/* History feed */
.history-feed   { background:#1a1a1a; border-radius:12px; padding:1rem; }
.hist-row       { display:flex; justify-content:space-between; padding:4px 0;
                  border-bottom:1px solid #2d2d2d; font-size:0.9rem; font-weight:600; }
.hist-conf      { color:#6b7280; font-size:0.8rem; }
.history-empty  { color:#4b5563; font-size:0.9rem; text-align:center; padding:1rem; }

/* Summary */
.summary-panel  { background:#1a1a1a; border-radius:16px; padding:2rem; text-align:center; }
.sum-title      { font-size:2rem; font-weight:900; color:#f9fafb; margin-bottom:0.3rem; }
.sum-fighter    { color:#9ca3af; font-size:1.05rem; margin-bottom:1.5rem; }
.sum-stats      { display:flex; justify-content:center; gap:2rem; flex-wrap:wrap; margin-bottom:2rem; }
.sum-stat       { background:#111; border-radius:12px; padding:1rem 1.5rem; }
.sum-num        { font-size:2rem; font-weight:900; color:#f9fafb; }
.sum-lbl        { font-size:0.75rem; color:#6b7280; text-transform:uppercase; letter-spacing:1px; }
.sum-breakdown  { text-align:left; }
.sum-bar-row    { display:flex; align-items:center; gap:0.8rem; margin-bottom:0.6rem; }
.sum-label      { width:90px; font-size:0.8rem; font-weight:700; letter-spacing:1px; }
.sum-bar-bg     { flex:1; background:#2d2d2d; border-radius:99px; height:10px; overflow:hidden; }
.sum-bar-fill   { height:100%; border-radius:99px; }
.sum-count      { width:30px; text-align:right; color:#9ca3af; font-size:0.9rem; }

/* Buttons */
button.enter-btn { background:linear-gradient(135deg,#EF4444,#F59E0B) !important;
                   color:#fff !important; font-size:1.1rem !important; font-weight:700 !important;
                   border-radius:12px !important; padding:0.8rem 2rem !important; width:100%; }
"""

# ── UI Layout ─────────────────────────────────────────────────────────────

with gr.Blocks(title="PunchCV — Boxing Action Recognition", css=_CSS,
               theme=gr.themes.Base()) as demo:

    session_state = gr.State({})

    gr.HTML("""
    <div class="hero">
      <h1>PUNCH CV</h1>
      <p>Real-time boxing action recognition &mdash; your body is the controller</p>
    </div>
    """)

    # ── Setup Panel ───────────────────────────────────────────────────────
    with gr.Column(visible=True, elem_classes=["setup-panel"]) as setup_panel:
        gr.HTML('<div class="setup-title">Enter the Ring</div>')
        with gr.Row():
            name_box  = gr.Textbox(label="Fighter Name", placeholder="e.g. The Hurricane", scale=2)
            level_box = gr.Radio(
                ["Beginner", "Intermediate", "Advanced", "Pro"],
                value="Beginner",
                label="Skill Level",
                scale=3,
            )
        tips_html = gr.HTML(_setup_tips_html("Beginner"))
        enter_btn = gr.Button("ENTER THE RING", elem_classes=["enter-btn"])

    # ── Training Panel ────────────────────────────────────────────────────
    with gr.Column(visible=False) as training_panel:
        header_html   = gr.HTML()
        with gr.Row():
            with gr.Column(scale=3):
                webcam = gr.Image(
                    sources=["webcam"], streaming=True,
                    mirror_webcam=True, label="Camera",
                    height=380,
                )
            with gr.Column(scale=2):
                action_html   = gr.HTML()
                counters_html = gr.HTML()
                history_html  = gr.HTML()
        end_btn = gr.Button("End Session")

    # ── Summary Panel ─────────────────────────────────────────────────────
    with gr.Column(visible=False) as summary_panel:
        summary_html = gr.HTML()
        back_btn = gr.Button("Train Again")

    # ── Wiring ────────────────────────────────────────────────────────────
    level_box.change(fn=update_tips, inputs=[level_box], outputs=[tips_html])

    enter_btn.click(
        fn=start_session,
        inputs=[name_box, level_box],
        outputs=[
            session_state,
            setup_panel, training_panel, summary_panel,
            header_html, action_html, counters_html, history_html,
        ],
    )

    webcam.stream(
        fn=process_frame,
        inputs=[webcam, session_state],
        outputs=[session_state, action_html, counters_html, history_html, header_html],
    )

    end_btn.click(
        fn=end_session,
        inputs=[session_state],
        outputs=[setup_panel, training_panel, summary_panel, summary_html],
    )

    back_btn.click(
        fn=back_to_hub,
        inputs=[session_state],
        outputs=[setup_panel, training_panel, summary_panel],
    )

    gr.HTML("""
    <div style="text-align:center; color:#4b5563; font-size:0.8rem; margin-top:2rem; padding:1rem 0;">
      PunchCV &mdash; AIPI 540 Final Project &middot;
      Random Forest 97.1% accuracy &middot; 6 action classes &middot; MediaPipe PoseLandmarker
    </div>
    """)


if __name__ == "__main__":
    demo.launch(share=False)
