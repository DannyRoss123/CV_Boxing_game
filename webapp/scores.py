import json
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from webapp.auth import get_current_user
from webapp import database as db

router = APIRouter(prefix="/api", tags=["scores"])

VALID_GAMES = {"memory", "combo_rush", "defense"}


class SaveScore(BaseModel):
    game: str
    score: int
    metadata: dict = {}


@router.post("/scores")
def save_score(body: SaveScore, user=Depends(get_current_user)):
    if body.game not in VALID_GAMES:
        from fastapi import HTTPException
        raise HTTPException(400, f"Unknown game: {body.game}")
    db.save_score(user["id"], body.game, body.score, json.dumps(body.metadata))
    return {"ok": True}


@router.get("/scores/leaderboard/{game}")
def get_leaderboard(game: str):
    return db.leaderboard(game)


@router.get("/scores/me")
def my_scores(user=Depends(get_current_user)):
    rows = db.user_scores(user["id"])
    grouped: dict = {"memory": [], "combo_rush": [], "defense": []}
    for r in rows:
        g = r["game"]
        if g in grouped:
            grouped[g].append({"score": r["score"], "played_at": r["played_at"]})
    return grouped
