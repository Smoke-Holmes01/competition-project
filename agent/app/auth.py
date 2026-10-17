from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import TypedDict

from fastapi import HTTPException, Request, Response, status
from pydantic import BaseModel

from app.config import Settings


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthUser(TypedDict):
    username: str


class AuthResponse(BaseModel):
    authenticated: bool
    username: str


class PasswordHasher:
    iterations = 600_000
    algorithm = "sha256"

    @classmethod
    def hash_password(cls, password: str, salt: str | None = None) -> str:
        salt_bytes = (salt or secrets.token_hex(16)).encode("utf-8")
        digest = hashlib.pbkdf2_hmac(cls.algorithm, password.encode("utf-8"), salt_bytes, cls.iterations)
        return f"pbkdf2_{cls.algorithm}${cls.iterations}${salt_bytes.decode('utf-8')}${base64.b64encode(digest).decode('ascii')}"

    @classmethod
    def verify_password(cls, password: str, stored_hash: str) -> bool:
        try:
            scheme, iterations, salt, encoded = stored_hash.split("$", 3)
        except ValueError as exc:
            raise ValueError("Invalid AUTH_PASSWORD_HASH format") from exc
        if scheme != f"pbkdf2_{cls.algorithm}":
            raise ValueError("Unsupported password hash scheme")
        digest = hashlib.pbkdf2_hmac(cls.algorithm, password.encode("utf-8"), salt.encode("utf-8"), int(iterations))
        return hmac.compare_digest(base64.b64encode(digest).decode("ascii"), encoded)


class SessionManager:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _sign(self, payload: str) -> str:
        return hmac.new(self.settings.auth_secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def create_session_token(self, username: str) -> str:
        issued_at = int(time.time())
        payload = {
            "sub": username,
            "iat": issued_at,
            "exp": issued_at + self.settings.auth_session_ttl_hours * 3600,
        }
        payload_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        payload_b64 = base64.urlsafe_b64encode(payload_json.encode("utf-8")).decode("ascii")
        signature = self._sign(payload_b64)
        return f"{payload_b64}.{signature}"

    def read_session_token(self, token: str | None) -> AuthUser:
        unauthorized = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="未登录或登录已失效")
        if not self.settings.auth_enabled:
            return {"username": self.settings.auth_username}
        if not token:
            raise unauthorized
        try:
            payload_b64, signature = token.split(".", 1)
        except ValueError as exc:
            raise unauthorized from exc
        expected_signature = self._sign(payload_b64)
        if not hmac.compare_digest(signature, expected_signature):
            raise unauthorized
        try:
            payload_raw = base64.urlsafe_b64decode(payload_b64.encode("ascii"))
            payload = json.loads(payload_raw)
        except (ValueError, json.JSONDecodeError) as exc:
            raise unauthorized from exc
        exp = payload.get("exp")
        username = payload.get("sub")
        if not isinstance(exp, int) or exp < int(time.time()) or not isinstance(username, str) or not username:
            raise unauthorized
        return {"username": username}

    def set_session_cookie(self, response: Response, username: str) -> None:
        response.set_cookie(
            key=self.settings.auth_cookie_name,
            value=self.create_session_token(username),
            max_age=self.settings.auth_session_ttl_hours * 3600,
            httponly=True,
            samesite="lax",
            secure=self.settings.auth_cookie_secure,
            path="/",
        )

    def clear_session_cookie(self, response: Response) -> None:
        response.delete_cookie(
            key=self.settings.auth_cookie_name,
            httponly=True,
            samesite="lax",
            secure=self.settings.auth_cookie_secure,
            path="/",
        )

    def authenticate(self, username: str, password: str) -> AuthUser:
        unauthorized = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")
        if not self.settings.auth_enabled:
            return {"username": self.settings.auth_username}
        if username != self.settings.auth_username:
            raise unauthorized
        if not self.settings.auth_password_hash.strip():
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="AUTH_PASSWORD_HASH 未配置")
        try:
            is_valid = PasswordHasher.verify_password(password, self.settings.auth_password_hash)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
        if not is_valid:
            raise unauthorized
        return {"username": username}


def get_session_manager(settings: Settings) -> SessionManager:
    return SessionManager(settings)


def require_auth(settings: Settings, request: Request) -> AuthUser:
    manager = get_session_manager(settings)
    return manager.read_session_token(request.cookies.get(settings.auth_cookie_name))
