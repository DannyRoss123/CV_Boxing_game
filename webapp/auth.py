from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
import bcrypt as _bcrypt
from jose import JWTError, jwt
from webapp.config import SECRET_KEY, ALGORITHM, TOKEN_EXPIRE_DAYS
from webapp import database as db

router = APIRouter(prefix="/api/auth", tags=["auth"])
_oauth2 = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def _hash_pw(pw: str) -> str:
    return _bcrypt.hashpw(pw.encode(), _bcrypt.gensalt()).decode()


def _verify_pw(pw: str, hashed: str) -> bool:
    return _bcrypt.checkpw(pw.encode(), hashed.encode())


class RegisterIn(BaseModel):
    username: str
    password: str


class LoginIn(BaseModel):
    username: str
    password: str


def _make_token(user_id: int, username: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(days=TOKEN_EXPIRE_DAYS)
    return jwt.encode({"sub": username, "uid": user_id, "exp": exp}, SECRET_KEY, ALGORITHM)


def get_current_user(token: str = Depends(_oauth2)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {"id": payload["uid"], "username": payload["sub"]}
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


@router.post("/register")
def register(body: RegisterIn):
    if len(body.username) < 3:
        raise HTTPException(400, "Username must be at least 3 characters")
    if len(body.password) < 4:
        raise HTTPException(400, "Password must be at least 4 characters")
    existing = db.get_user_by_name(body.username)
    if existing:
        raise HTTPException(400, "Username already taken")
    uid = db.create_user(body.username, _hash_pw(body.password))
    return {"token": _make_token(uid, body.username), "username": body.username}


@router.post("/login")
def login(body: LoginIn):
    user = db.get_user_by_name(body.username)
    if not user or not _verify_pw(body.password, user["password_hash"]):
        raise HTTPException(401, "Invalid username or password")
    return {"token": _make_token(user["id"], user["username"]), "username": user["username"]}
