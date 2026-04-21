"""
Classifier test harness.
- Camera fills the screen
- Punch label shown at top
- All results printed to terminal

Usage:
    python scripts/test_classifier.py
"""

import os, time, random, collections, math
import cv2, joblib, numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from mediapipe import Image, ImageFormat

BASE        = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH  = os.path.join(BASE, "pose_landmarker_lite.task")
RF_PATH     = os.path.join(BASE, "models", "random_forest.joblib")
SCALER_PATH = os.path.join(BASE, "models", "rf_scaler.joblib")
DATA_DIR    = os.path.join(BASE, "data", "raw")

# Save live segments as training data (labeled by prompt)
SAVE_LIVE_DATA = True

ACTIONS = ["jab", "cross", "hook", "uppercut", "block_high"]

L_HIP, R_HIP         = 23*4, 24*4
L_SHO, R_SHO         = 11*4, 12*4
_VEL_PAIRS = [(16*4,16*4+1),(15*4,15*4+1),(14*4,14*4+1),(13*4,13*4+1)]

VEL_THRESH    = 0.020   # match cv_pipeline — ignore jitter/breathing
MOTION_START  = 3
MOTION_STOP   = 4
LEAD_IN       = 15
SEQ_LEN       = 60
COOLDOWN_SECS = 0.8
MIN_CONF      = 0.15    # ignore sub-15% noise detections

COLORS_BGR = {
    "jab":        (50, 165, 249),
    "cross":      (50,  50, 239),
    "hook":       (30, 195, 234),
    "uppercut":   (210,182,   6),
    "block_high": (239,130,  40),
}

def lms_to_arr(lms):
    row = []
    for lm in lms: row.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(row, dtype=np.float32)

def normalize(fr):
    fr = fr.copy()
    cx = (fr[L_HIP]+fr[R_HIP])/2;   cy = (fr[L_HIP+1]+fr[R_HIP+1])/2;   cz = (fr[L_HIP+2]+fr[R_HIP+2])/2
    sx = (fr[L_SHO]+fr[R_SHO])/2;   sy = (fr[L_SHO+1]+fr[R_SHO+1])/2;   sz = (fr[L_SHO+2]+fr[R_SHO+2])/2
    t  = max(math.sqrt((sx-cx)**2+(sy-cy)**2+(sz-cz)**2), 0.01)
    for i in range(33):
        idx=i*4; fr[idx]=(fr[idx]-cx)/t; fr[idx+1]=(fr[idx+1]-cy)/t; fr[idx+2]=(fr[idx+2]-cz)/t
    return fr

def velocity(a, b):
    if a is None or b is None: return 0.0
    return sum(math.sqrt((b[ix]-a[ix])**2+(b[iy]-a[iy])**2) for ix,iy in _VEL_PAIRS)/4

def extract_stats(frames):
    X = np.array(frames, dtype=np.float32)
    if X.shape[0] < SEQ_LEN:
        d=SEQ_LEN-X.shape[0]; pb=d//2; pa=d-pb
        parts=[]
        if pb: parts.append(np.tile(X[:1],(pb,1)))
        parts.append(X)
        if pa: parts.append(np.tile(X[-1:],(pa,1)))
        X=np.vstack(parts)
    elif X.shape[0] > SEQ_LEN:
        s=(X.shape[0]-SEQ_LEN)//2; X=X[s:s+SEQ_LEN]
    vel_f=[]
    for ix,iy in _VEL_PAIRS:
        spd=np.sqrt(np.diff(X[:,ix])**2+np.diff(X[:,iy])**2)
        vel_f.extend([spd.mean(),spd.max(),spd.std()])
    disp_f=[]
    for ix,iy in _VEL_PAIRS:
        dx=X[:,ix]-X[0,ix]; dy=X[:,iy]-X[0,iy]
        disp_f.extend([np.abs(dx).max(),np.abs(dy).max(),dx.max(),dx.min()])
    return np.concatenate([np.std(X,0),np.max(X,0)-np.min(X,0),X[-1]-X[0],
                           np.array(vel_f,dtype=np.float32),np.array(disp_f,dtype=np.float32)])

def classify(frames, rf, scaler):
    if len(frames) < 8: return None, None
    fs = scaler.transform(extract_stats(frames).reshape(1,-1))
    proba = rf.predict_proba(fs)[0]
    return list(rf.classes_), proba


def save_segment(frames, label):
    """Pad/trim segment to SEQ_LEN and save as training sample."""
    X = np.array(frames, dtype=np.float32)
    if X.shape[0] < SEQ_LEN:
        d = SEQ_LEN - X.shape[0]; pb = d//2; pa = d-pb
        parts = []
        if pb: parts.append(np.tile(X[:1], (pb, 1)))
        parts.append(X)
        if pa: parts.append(np.tile(X[-1:], (pa, 1)))
        X = np.vstack(parts)
    elif X.shape[0] > SEQ_LEN:
        s = (X.shape[0]-SEQ_LEN)//2; X = X[s:s+SEQ_LEN]
    out_dir = os.path.join(DATA_DIR, label)
    os.makedirs(out_dir, exist_ok=True)
    existing = [f for f in os.listdir(out_dir) if f.endswith('.npy')]
    idx = len(existing)
    path = os.path.join(out_dir, f"{label}_{idx:04d}.npy")
    np.save(path, X)
    return path

def put(img, text, pos, scale, color, thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_DUPLEX, scale, (0,0,0),   thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness,   cv2.LINE_AA)

def main():
    rf     = joblib.load(RF_PATH)
    scaler = joblib.load(SCALER_PATH)

    opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
    )
    lmk = PoseLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cv2.namedWindow("Test", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    buf        = collections.deque(maxlen=120)
    prev       = None
    ts         = 0
    vs         = 0.0
    ms         = "IDLE"
    mc         = sc = mi = fc = 0
    last_t     = 0.0
    votes      = []
    VOTE_EVERY = 5

    prompt         = None
    prompt_hide_at = 0.0
    next_prompt_at = time.time() + random.uniform(2.0, 4.0)

    stats = {a: [0, 0] for a in ACTIONS}

    print("\n── Classifier Test ─────────────────────────────────")
    print("Throw the punch shown on screen. Q to quit.\n")
    print(f"{'ACTION':<14} {'RESULT':<10} {'CONFIDENCE':<12} {'ALL PROBABILITIES'}")
    print("─" * 72)

    while True:
        ok, frame = cap.read()
        if not ok: break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        ts   += 33
        now   = time.time()

        mp_img = Image(image_format=ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res    = lmk.detect_for_video(mp_img, ts)

        result = None
        if res.pose_landmarks:
            arr  = lms_to_arr(res.pose_landmarks[0])
            nrm  = normalize(arr)
            vel  = velocity(prev, nrm)
            vs   = 0.4*vel + 0.6*vs
            prev = nrm
            buf.append(nrm)

            def _vote_seg(seg):
                if len(seg) < 8: return
                fs = scaler.transform(extract_stats(seg).reshape(1,-1))
                votes.append(rf.predict_proba(fs)[0])

            def _resolve():
                if not votes: return None, 0.0
                total = np.sum(votes, axis=0)
                classes = list(rf.classes_)
                if "idle" in classes: total[classes.index("idle")] = 0.0
                best = int(np.argmax(total))
                return classes[best], float(total[best]/len(votes))

            if ms == "IDLE":
                if vs > VEL_THRESH:
                    mc += 1
                    if mc >= MOTION_START:
                        mi = max(0, len(buf)-mc-LEAD_IN)
                        ms = "MOTION"; sc = fc = 0; votes.clear()
                else:
                    mc = 0
            elif ms == "MOTION":
                fc += 1
                if fc % VOTE_EVERY == 0:
                    _vote_seg(list(buf)[mi:])
                if vs < VEL_THRESH:
                    sc += 1
                    if sc >= MOTION_STOP and now-last_t > COOLDOWN_SECS:
                        _vote_seg(list(buf)[mi:])
                        if votes:
                            result = (list(rf.classes_), np.mean(votes, axis=0))
                        last_t = now; ms="IDLE"; mc=sc=fc=0; votes.clear()
                else:
                    sc = 0
                    if len(buf)-mi > 90 and now-last_t > COOLDOWN_SECS:
                        _vote_seg(list(buf)[mi:])
                        if votes:
                            result = (list(rf.classes_), np.mean(votes, axis=0))
                        last_t = now; ms="IDLE"; mc=sc=fc=0; votes.clear()
        else:
            vs = 0.0; prev = None

        # ── Handle result ────────────────────────────────────────────────
        if result and result[0]:
            classes, proba = result
            # zero idle, pick best remaining
            proba_noidl = np.array(proba)
            if "idle" in classes:
                proba_noidl[classes.index("idle")] = 0.0
            best = int(np.argmax(proba_noidl))
            predicted = classes[best]
            conf = float(proba_noidl[best])

            if conf < MIN_CONF:
                continue  # skip noise

            # Save live segment as training data (labeled by prompt)
            if SAVE_LIVE_DATA and prompt and prompt in ACTIONS:
                seg = list(buf)[mi:]
                path = save_segment(seg, prompt)
                print(f"  [saved {prompt} → {os.path.basename(path)}]")

            target  = prompt
            correct = (predicted == target) if target else None

            # print to terminal
            prob_str = "  ".join(f"{c}:{p*100:.0f}%" for c,p in sorted(zip(classes,proba), key=lambda x:-x[1]) if c!="idle")
            if correct is True:
                tag = "✓ CORRECT"
            elif correct is False:
                tag = f"✗ got {predicted}"
            else:
                tag = f"→ {predicted}"

            if target:
                stats[target][1] += 1
                if correct: stats[target][0] += 1
                print(f"{target:<14} {tag:<20} {conf*100:.1f}%      {prob_str}")
            else:
                print(f"{'(no prompt)':<14} {tag:<20} {conf*100:.1f}%      {prob_str}")

            prompt = None
            next_prompt_at = now + random.uniform(2.5, 5.5)

        # ── Prompt scheduling ────────────────────────────────────────────
        if prompt is None and now >= next_prompt_at and ms == "IDLE":
            prompt         = random.choice(ACTIONS)
            prompt_shown_at = now
            prompt_hide_at  = now + random.uniform(1.5, 2.5)
            print(f"\n  >>> THROW: {prompt.upper().replace('_',' ')} <<<")

        # visually hide after hide_at but keep prompt active for matching
        # only clear prompt after 5s total (give time to throw)
        if prompt and now > prompt_shown_at + 5.0:
            print(f"  (missed {prompt} — too slow)")
            prompt = None
            next_prompt_at = now + random.uniform(2.0, 4.0)

        # ── Draw ─────────────────────────────────────────────────────────
        display = frame.copy()

        # Velocity bar along top
        bar_w = int(min(vs/0.05, 1.0) * w)
        bar_color = (60,220,60) if ms=="IDLE" else (50,50,220) if ms=="MOTION" else (60,160,200)
        cv2.rectangle(display, (0,0), (w,8), (40,40,40), -1)
        if bar_w: cv2.rectangle(display, (0,0), (bar_w,8), bar_color, -1)

        # Punch prompt — big text top-center
        if prompt:
            color = COLORS_BGR.get(prompt, (255,255,255))
            label = prompt.replace("_"," ").upper()
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 3.0, 4)
            put(display, label, ((w-tw)//2, 90), 3.0, color, 4)
            put(display, "THROW THIS", ((w-200)//2, 120), 0.7, (200,200,200), 1)

        # Motion state badge top-left
        badge_col = (60,220,60) if ms=="IDLE" else (50,50,220)
        cv2.rectangle(display, (10,14), (170,40), badge_col, -1)
        cv2.putText(display, ms, (18,34), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0,0,0), 2, cv2.LINE_AA)

        cv2.imshow("Test", display)
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    lmk.close()

    print("\n── Final Results ───────────────────────────────────")
    total_c = total_n = 0
    for a in ACTIONS:
        c, n = stats[a]
        total_c += c; total_n += n
        pct = f"{100*c//n}%" if n else "—"
        print(f"  {a:<14}  {c}/{n}  {pct}")
    if total_n:
        print(f"  {'OVERALL':<14}  {total_c}/{total_n}  {100*total_c//total_n}%")

if __name__ == "__main__":
    main()
