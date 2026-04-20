from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["training"])

PLANS = {
    "Beginner": {
        "focus": "Learn the 5 basic actions with correct form",
        "sessions_per_week": 3,
        "drills": [
            {"name": "Jab Practice", "sets": 3, "reps": 20,
             "tip": "Quick lead-hand straight — snap it out and pull back fast. Keep your chin down."},
            {"name": "Cross Practice", "sets": 3, "reps": 20,
             "tip": "Rear-hand straight with full hip rotation. Pivot on your rear foot."},
            {"name": "Hook Form", "sets": 3, "reps": 15,
             "tip": "Horizontal arc at 90 degrees, elbow parallel to floor. Short, tight motion."},
            {"name": "Uppercut Drill", "sets": 3, "reps": 15,
             "tip": "Drive upward with your legs — don't just swing your arm. Bend the knees first."},
            {"name": "High Guard", "sets": 3, "reps": 10,
             "tip": "Both arms up, forearms forward — protect that chin. Hold for 2 seconds each rep."},
        ],
        "game_goals": {"memory": 50, "combo_rush": 200, "defense": 150},
        "week_plan": [
            "Mon: Jab x60 + Cross x60 — focus on form over speed",
            "Wed: Hook x50 + Uppercut x50 — slow and deliberate",
            "Fri: Memory game — get to Level 3",
        ],
    },
    "Intermediate": {
        "focus": "Speed and combo chaining",
        "sessions_per_week": 4,
        "drills": [
            {"name": "Jab-Cross Combo", "sets": 4, "reps": 25,
             "tip": "Use the jab to set up your cross — double-jab variations work well."},
            {"name": "Hook-Uppercut Combo", "sets": 4, "reps": 20,
             "tip": "Short hook off the jab — same rhythm, tighter arc. Follow with the uppercut in the pocket."},
            {"name": "Speed Rounds", "sets": 5, "reps": 30,
             "tip": "Max speed for 10 seconds — rest 20s. Focus on staying loose between punches."},
            {"name": "Defense Drill", "sets": 3, "reps": 10,
             "tip": "High guard then immediately counter-punch. Don't just sit in guard."},
            {"name": "Jab-Cross-Hook Combo", "sets": 4, "reps": 15,
             "tip": "Classic 1-2-3. Throw the hook off the cross with no reset — straight into it."},
        ],
        "game_goals": {"memory": 150, "combo_rush": 800, "defense": 400},
        "week_plan": [
            "Mon: Jab-Cross x80 + speed rounds",
            "Tue: Defense Drill game — aim for 0 misses",
            "Thu: Full combo work — all 3-punch combos",
            "Sat: Combo Rush game — beat 800 pts",
        ],
    },
    "Advanced": {
        "focus": "Power, timing, and counter-punching",
        "sessions_per_week": 5,
        "drills": [
            {"name": "Paw Jab Series", "sets": 5, "reps": 30,
             "tip": "Paw jab to disrupt rhythm — no tell in the shoulder. Vary speed intentionally."},
            {"name": "Cross-Counter", "sets": 5, "reps": 20,
             "tip": "Cross-counter off their jab: slip, cross, body-hook. Fast reset after the hook."},
            {"name": "Body-Head Combos", "sets": 4, "reps": 20,
             "tip": "Cross to the body then upstairs — work the two levels to open up the guard."},
            {"name": "Liver Hook", "sets": 4, "reps": 15,
             "tip": "Liver shot hook to the body — bend your knees and drive through on contact."},
            {"name": "Inside Fighting", "sets": 3, "reps": 12,
             "tip": "Double uppercut in the pocket — inside fighting game. Stay tight and turn your hips."},
        ],
        "game_goals": {"memory": 300, "combo_rush": 1600, "defense": 700},
        "week_plan": [
            "Mon: Power combos x100 — maximum hip drive",
            "Tue: Counter-punching drills — react to Defence Drill",
            "Wed: Speed work — 8x 30s max speed rounds",
            "Fri: Memory game — hit Level 7+",
            "Sat: Full session — all three games, beat personal bests",
        ],
    },
    "Pro": {
        "focus": "Competition-level sharpness and ring IQ",
        "sessions_per_week": 6,
        "drills": [
            {"name": "Triple Jab Series", "sets": 6, "reps": 30,
             "tip": "Triple jab: range-finder, set-up, power — vary speed intentionally on each."},
            {"name": "Level-Change Combinations", "sets": 5, "reps": 25,
             "tip": "Cross to the body then upstairs in one fluid motion — no pause between levels."},
            {"name": "Counter Bait Drill", "sets": 4, "reps": 20,
             "tip": "High guard bait then walk them onto a counter-cross. Patience is the weapon."},
            {"name": "Inside Distance Work", "sets": 5, "reps": 20,
             "tip": "Left hook as the money punch after slipping the jab — stay compact, no windup."},
            {"name": "Max Volume Rounds", "sets": 6, "reps": 50,
             "tip": "Volume and variety — no two consecutive punches the same. Think 6+ punch combos."},
        ],
        "game_goals": {"memory": 500, "combo_rush": 2800, "defense": 900},
        "week_plan": [
            "Mon: Triple-jab series + level-change work",
            "Tue: Defense Drill — perfect score, no misses allowed",
            "Wed: Speed rounds 10x 45s max — rest 30s",
            "Thu: Counter-punching focus + Memory game Level 9+",
            "Fri: Long combo session — 8-punch combos",
            "Sat: Full competition sim — all games, max scores",
        ],
    },
}


@router.get("/training/plans")
def get_plans():
    return PLANS


@router.get("/training/plans/{level}")
def get_plan(level: str):
    from fastapi import HTTPException
    if level not in PLANS:
        raise HTTPException(404, f"Level '{level}' not found")
    return PLANS[level]
