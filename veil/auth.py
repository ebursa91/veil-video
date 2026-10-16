"""Password verification, expiring signed cookies, CSRF, and bounded login throttling."""

import hashlib
import secrets
import threading
import time
from collections import OrderedDict, deque

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from veil.config import Settings

COOKIE = "veil_session"
HASHER = PasswordHasher()


class Auth:
    def __init__(self, settings: Settings):
        self.settings = settings
        fingerprint = hashlib.sha256(settings.password_hash.encode()).hexdigest()[:16]
        self.signer = URLSafeTimedSerializer(
            settings.secret_key, salt=f"veil-session-v1-{fingerprint}"
        )
        self._attempts: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def verify_password(self, password: str) -> bool:
        if not password or len(password) > 1024:
            return False
        try:
            return HASHER.verify(self.settings.password_hash, password)
        except (VerificationError, InvalidHashError):
            return False

    def issue(self) -> tuple[str, dict]:
        session = {"csrf_token": secrets.token_urlsafe(32), "authenticated": True}
        return self.signer.dumps(session), session

    def read(self, cookie: str | None) -> dict | None:
        if not cookie:
            return None
        try:
            session = self.signer.loads(cookie, max_age=self.settings.session_seconds)
            if session.get("authenticated") and isinstance(session.get("csrf_token"), str):
                return session
        except (BadSignature, SignatureExpired, TypeError, AttributeError):
            pass
        return None

    def allow_attempt(self, address: str) -> bool:
        # Also cap all attempts: proxy headers cannot be trusted as client identities.
        now = time.monotonic()
        with self._lock:
            for key, limit in (("__global__", 30), (address, 10)):
                history = self._attempts.setdefault(key, deque())
                while history and now - history[0] > 300:
                    history.popleft()
                if len(history) >= limit:
                    return False
            for key in ("__global__", address):
                self._attempts[key].append(now)
                self._attempts.move_to_end(key)
            while len(self._attempts) > 2048:
                self._attempts.popitem(last=False)
            return True
