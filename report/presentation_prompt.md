# Presentation Prompt for Claude

Use the following prompt to have Claude generate your Demo Day slides or speaker notes:

---

## PROMPT TO PASTE INTO CLAUDE

```
You are helping me prepare a 5-minute Demo Day pitch for my AIPI 540 final project. The audience includes people outside the course — treat it like a startup pitch, not a class presentation. Hard stop at 5 minutes.

PROJECT SUMMARY:
PunchCV is a real-time boxing action recognition system that turns your body into a game controller. It uses a standard webcam + MediaPipe pose estimation to classify 6 boxing actions (idle, jab, cross, hook, uppercut, block_high) and drives a Pygame boxing game in real time.

EXACT NUMBERS (use these, do not make up others):
- Dataset: 699 total samples, 6 classes, ~100-150 samples/class
- Models trained: Nearest Centroid baseline (46.4% accuracy, F1=0.43), Random Forest on temporal statistics (96.4% accuracy, F1=0.96), 2-layer LSTM (92.9% accuracy, F1=0.93)
- Random Forest is the deployed production model
- Input: 60 frames × 132 values (33 pose landmarks × x/y/z/visibility), normalized per-frame to body proportions
- RF feature vector: 396-dimensional (std + range + delta of landmark trajectories over the sequence)
- Inference latency: ~3 ms (RF), ~15 ms (LSTM)
- RF model size: 1.6 MB; LSTM: 333 KB
- Data split: 60/20/20 train/val/test, stratified, seed=42
- Best class: block_high (100% F1 in both RF and LSTM)
- Hardest class: jab/uppercut confusion
- Motion detection state machine: IDLE → MOTION → COOLDOWN, velocity threshold 0.015 on wrist+elbow positions
- Game: Pygame boxer with keyframe animations, damage system (jab=8hp, cross=12hp, hook=15hp, uppercut=20hp), threaded CV + game loop
- WebSocket server: ws://localhost:9090, broadcasts {"type":"action","action":"jab","confidence":0.95} to Godot 3D game client

KEY INSIGHT / EXPERIMENT RESULT:
Temporal statistics (std, range, delta) over the 60-frame pose window outperform both raw flattened sequences (46.4%) AND end-to-end LSTM learning (92.9%). This shows that with ~400 training samples, explicit feature engineering beats learned features — domain knowledge about "what motion looks like" is more data-efficient than gradient descent at this scale.

PITCH STRUCTURE (5 minutes total):
Slide 1 — Problem & Hook (60 seconds)
Slide 2 — Approach (60 seconds)  
[Live demo here — 90 seconds]
Slide 3 — Results & Key Insight (60 seconds)
Slide 4 — Vision & Ask (30 seconds)

TONE: Confident, punchy, startup energy. This is an investor pitch, not a class report. Lead with the problem and the "wow" of it — no controller, just your body. Quantify everything. Avoid jargon.

Please produce:
1. Four slide outlines (title + 3-5 bullet points each, plus any diagrams to describe)
2. Full speaker script for each slide (what I say out loud, timed)
3. Demo Day talking points for the live demo section (what I say while showing the game)
4. One-sentence hook to open the talk
5. One-sentence close / call to action

Remember: the live demo is in the middle. Everything before it builds anticipation; everything after it lands the insight.
```

---

## TIPS FOR YOUR DEMO DAY

**Live demo setup checklist:**
- [ ] Webcam working and positioned at chest-height
- [ ] Good lighting (face lit from front, not backlit)
- [ ] Run `python scripts/game.py` before the presentation starts
- [ ] Have a second window ready if it crashes (live_demo.py as backup)
- [ ] Clear background behind you
- [ ] Test all 6 actions slowly before going live

**If the demo fails mid-presentation:**
- Say: "Let me show you what a successful detection looks like" — switch to a pre-recorded clip
- Have a 30-second screen recording of a working game session as backup

**The key insight to emphasize:**
> "We trained three models. The surprising result: a 100-year-old algorithm — Random Forest — beats our neural network. Why? With 400 training examples, you're better off telling the model what to look for than asking it to figure it out."

**Numbers to memorize:**
- 46 → 96 → 93 (baseline → RF → LSTM accuracy)
- 3 ms inference (RF)
- 699 samples, 6 classes
- 96.4% accuracy on hold-out test set
```
