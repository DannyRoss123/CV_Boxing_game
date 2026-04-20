import json
from pathlib import Path
from fastapi import FastAPI, WebSocket, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from jose import JWTError, jwt
from webapp.config import SECRET_KEY, ALGORITHM
from webapp import database as db
from webapp.auth import router as auth_router
from webapp.scores import router as scores_router
from webapp.training import router as training_router
from webapp import ws_handler

_STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Boxing Hub")

app.include_router(auth_router)
app.include_router(scores_router)
app.include_router(training_router)

app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.on_event("startup")
def startup():
    db.init_db()


@app.get("/")
def index():
    return FileResponse(str(_STATIC / "index.html"))


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket, token: str = Query(...)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user    = {"id": payload["uid"], "username": payload["sub"]}
    except JWTError:
        await ws.close(code=4001)
        return

    async def save_score(game: str, score: int):
        db.save_score(user["id"], game, score, json.dumps({}))

    await ws_handler.handle(ws, user, save_score)
