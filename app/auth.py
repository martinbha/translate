"""Auth dependency + brute-force throttling for login."""
import time
from collections import defaultdict

from fastapi import Cookie, Depends, HTTPException, status
from sqlmodel import Session, select

from app.config import settings
from app.db import engine
from app.models import User
from app.security import read_session_token

# --- Simple in-memory login throttle --------------------------------------
# Single-process, single-user app, so an in-memory window is enough. For a
# multi-worker deploy this would move to Redis.
_FAIL_WINDOW = 15 * 60   # 15 minutes
_MAX_FAILS = 5
_failures: dict[str, list[float]] = defaultdict(list)


def _recent_failures(key: str) -> int:
    now = time.time()
    hits = [t for t in _failures[key] if now - t < _FAIL_WINDOW]
    _failures[key] = hits
    return len(hits)


def login_locked(key: str) -> bool:
    return _recent_failures(key) >= _MAX_FAILS


def record_login_failure(key: str) -> None:
    _failures[key].append(time.time())


def reset_login_failures(key: str) -> None:
    _failures.pop(key, None)


# --- Current-user dependency ----------------------------------------------
def get_current_user(
    session_cookie: str | None = Cookie(default=None, alias=settings.cookie_name),
) -> User:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated"
    )
    if not session_cookie:
        raise unauthorized

    user_id = read_session_token(session_cookie)
    if user_id is None:
        raise unauthorized

    with Session(engine) as db:
        user = db.exec(select(User).where(User.id == user_id)).first()
    if user is None:
        raise unauthorized
    return user


CurrentUser = Depends(get_current_user)
