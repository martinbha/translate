"""Password hashing, TOTP, and signed session cookies."""
import time

import pyotp
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from app.config import settings

_ph = PasswordHasher()


# --- Passwords -------------------------------------------------------------
def hash_password(password: str) -> str:
    return _ph.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        _ph.verify(password_hash, password)
        return True
    except VerifyMismatchError:
        return False


# --- TOTP (2FA) ------------------------------------------------------------
def new_totp_secret() -> str:
    return pyotp.random_base32()


def totp_provisioning_uri(secret: str, username: str, issuer: str = "Transcribe") -> str:
    return pyotp.TOTP(secret).provisioning_uri(name=username, issuer_name=issuer)


def verify_totp(secret: str, code: str) -> bool:
    # valid_window=1 tolerates a +/- 30s clock skew.
    return pyotp.TOTP(secret).verify(code.strip(), valid_window=1)


# --- Session cookies -------------------------------------------------------
_serializer = URLSafeTimedSerializer(settings.secret_key, salt="session")


def make_session_token(user_id: int) -> str:
    return _serializer.dumps({"uid": user_id, "iat": int(time.time())})


def read_session_token(token: str) -> int | None:
    try:
        data = _serializer.loads(token, max_age=settings.session_max_age)
        return int(data["uid"])
    except (BadSignature, SignatureExpired, KeyError, ValueError, TypeError):
        return None
