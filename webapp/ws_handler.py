"""
WebSocket handler.
Protocol:
  Browser -> Server:
    {"type":"landmarks", "data":[...132 floats...]}
    {"type":"start_game", "game":"memory"|"combo_rush"|"defense"}
    {"type":"game_action", "action":"advance"}
    {"type":"ping"}

  Server -> Browser:
    {"type":"cv", "action":"jab", "confidence":0.87, "motion_state":"COOLDOWN"}
    {"type":"game_state", "game":"memory", "state":{...}}
    {"type":"game_over", "game":"memory", "score":150}
    {"type":"error", "message":"..."}
"""

import asyncio
import json
import time
from fastapi import WebSocket, WebSocketDisconnect
from webapp.cv_pipeline import MotionPipeline
from webapp.games.memory import MemoryGame
from webapp.games.combo_rush import ComboRushGame
from webapp.games.defense import DefenseDrillGame


def _make_game(name: str):
    if name == "memory":     return MemoryGame()
    if name == "combo_rush": return ComboRushGame()
    if name == "defense":    return DefenseDrillGame()
    return None


async def handle(ws: WebSocket, user: dict, save_score_fn):
    await ws.accept()
    pipeline   = MotionPipeline()
    game       = None
    game_name  = None
    last_tick  = time.time()

    try:
        while True:
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=45.0)
            except asyncio.TimeoutError:
                await ws.send_text(json.dumps({"type": "ping"}))
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            t = msg.get("type")

            if t == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
                continue

            if t == "start_game":
                name = msg.get("game", "")
                g    = _make_game(name)
                if g is None:
                    await ws.send_text(json.dumps({"type": "error", "message": f"Unknown game: {name}"}))
                    continue
                game      = g
                game_name = name
                last_tick = time.time()
                await ws.send_text(json.dumps({
                    "type": "game_state", "game": game_name, "state": game.get_state()
                }))
                continue

            if t == "game_action":
                if game and hasattr(game, "advance"):
                    game.advance()
                    await ws.send_text(json.dumps({
                        "type": "game_state", "game": game_name, "state": game.get_state()
                    }))
                continue

            if t == "landmarks":
                data = msg.get("data")
                if not data or len(data) != 132:
                    continue

                # Tick game clock
                now = time.time()
                dt  = now - last_tick
                last_tick = now

                # Run CV pipeline
                result = await asyncio.to_thread(pipeline.feed, data)

                # Send CV feedback
                cv_msg: dict = {
                    "type":         "cv",
                    "motion_state": pipeline.motion_state,
                    "velocity":     round(pipeline.velocity, 4),
                    "action":       None,
                    "confidence":   0.0,
                }
                if result:
                    action, conf = result
                    cv_msg["action"]     = action
                    cv_msg["confidence"] = round(conf, 3)

                await ws.send_text(json.dumps(cv_msg))

                # Feed punch to game
                if game and result:
                    action, conf = result
                    await asyncio.to_thread(game.on_punch, action, conf)

                # Tick game and send state
                if game:
                    await asyncio.to_thread(game.update, dt)
                    await ws.send_text(json.dumps({
                        "type": "game_state", "game": game_name, "state": game.get_state()
                    }))
                    if game.is_done:
                        final = game.get_state()
                        score = final["score"]
                        await save_score_fn(game_name, score)
                        await ws.send_text(json.dumps({
                            "type":       "game_over",
                            "game":       game_name,
                            "score":      score,
                            "last_state": final,
                        }))
                        game = None; game_name = None

    except WebSocketDisconnect:
        pass
