import sqlite3
from contextlib import contextmanager
from webapp.config import DB_PATH


def init_db():
    with _conn() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at    TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS scores (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id   INTEGER NOT NULL REFERENCES users(id),
                game      TEXT NOT NULL,
                score     INTEGER NOT NULL,
                metadata  TEXT DEFAULT '{}',
                played_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_scores_game ON scores(game, score DESC);
            CREATE INDEX IF NOT EXISTS idx_scores_user ON scores(user_id);
        """)


@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def create_user(username: str, password_hash: str) -> int:
    with _conn() as db:
        cur = db.execute(
            "INSERT INTO users (username, password_hash) VALUES (?,?)",
            (username, password_hash),
        )
        return cur.lastrowid


def get_user_by_name(username: str):
    with _conn() as db:
        row = db.execute(
            "SELECT id, username, password_hash FROM users WHERE username=?",
            (username,),
        ).fetchone()
        return dict(row) if row else None


def save_score(user_id: int, game: str, score: int, metadata: str = "{}"):
    with _conn() as db:
        db.execute(
            "INSERT INTO scores (user_id, game, score, metadata) VALUES (?,?,?,?)",
            (user_id, game, score, metadata),
        )


def leaderboard(game: str, limit: int = 10):
    with _conn() as db:
        rows = db.execute(
            """SELECT u.username, s.score, s.played_at
               FROM scores s JOIN users u ON u.id=s.user_id
               WHERE s.game=?
               ORDER BY s.score DESC LIMIT ?""",
            (game, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def user_scores(user_id: int):
    with _conn() as db:
        rows = db.execute(
            """SELECT game, score, metadata, played_at
               FROM scores WHERE user_id=?
               ORDER BY played_at DESC LIMIT 50""",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]
