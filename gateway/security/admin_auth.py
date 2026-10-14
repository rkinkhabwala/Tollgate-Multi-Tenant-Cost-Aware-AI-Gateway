"""Control-plane authentication (spec §6.3).

`AdminAuthProvider` is the seam where OIDC (Okta/Entra/Google) can replace local
users later: the API layer depends only on `authenticate`, `refresh` and
`verify_access_token`, and on the resulting `AdminPrincipal`.

`LocalAdminAuth` (v1):
* Argon2id password hashes in `admin_users`.
* Short-lived HS256 access JWT (15 min) + opaque refresh token. Refresh tokens are
  stored as SHA-256 and rotated on every use (the old one is revoked).
* Lockout after `login_max_failures` consecutive failures for `login_lockout_s` (§15).
"""

from __future__ import annotations

import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol

import jwt
from sqlalchemy import select, update

from gateway.db.models import AdminRefreshToken, AdminUser
from gateway.db.session import Database
from gateway.security.passwords import verify_password
from gateway.settings import Settings

Role = Literal["platform_admin", "tenant_admin", "tenant_viewer"]


@dataclass(frozen=True)
class AdminPrincipal:
    user_id: uuid.UUID
    email: str
    role: Role
    tenant_id: uuid.UUID | None  # None for platform_admin

    @property
    def actor(self) -> str:
        """Value written to `audit_log.actor`."""
        return f"user:{self.email}"


@dataclass(frozen=True)
class TokenPair:
    access_token: str
    refresh_token: str
    expires_in: int


class AuthError(Exception):
    """Login/refresh/token failure. `reason` is safe to show; never contains secrets."""

    def __init__(self, reason: str, status: int = 401) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status = status


class AdminAuthProvider(Protocol):
    async def authenticate(self, email: str, password: str) -> TokenPair: ...
    async def refresh(self, refresh_token: str) -> TokenPair: ...
    def verify_access_token(self, token: str) -> AdminPrincipal: ...


class LocalAdminAuth:
    """Local users + JWT implementation of `AdminAuthProvider`."""

    ISSUER = "tollgate"

    def __init__(self, db: Database, settings: Settings) -> None:
        self._db = db
        self._s = settings

    # --- tokens ---------------------------------------------------------------
    def _issue_access(self, user: AdminUser) -> str:
        now = datetime.now(UTC)
        claims = {
            "iss": self.ISSUER,
            "sub": str(user.id),
            "email": user.email,
            "role": user.role,
            "tid": str(user.tenant_id) if user.tenant_id else None,
            "typ": "access",
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(seconds=self._s.jwt_access_ttl_s)).timestamp()),
        }
        return jwt.encode(claims, self._s.jwt_secret.get_secret_value(), algorithm="HS256")

    async def _issue_pair(self, session, user: AdminUser) -> TokenPair:  # type: ignore[no-untyped-def]
        refresh = secrets.token_urlsafe(48)
        session.add(
            AdminRefreshToken(
                user_id=user.id,
                tenant_id=user.tenant_id,
                token_hash=hashlib.sha256(refresh.encode()).digest(),
                expires_at=datetime.now(UTC) + timedelta(seconds=self._s.jwt_refresh_ttl_s),
            )
        )
        return TokenPair(self._issue_access(user), refresh, self._s.jwt_access_ttl_s)

    def verify_access_token(self, token: str) -> AdminPrincipal:
        try:
            claims = jwt.decode(
                token,
                self._s.jwt_secret.get_secret_value(),
                algorithms=["HS256"],
                issuer=self.ISSUER,
                options={"require": ["exp", "sub", "role", "iss"]},
            )
        except jwt.ExpiredSignatureError as exc:
            raise AuthError("token expired") from exc
        except jwt.PyJWTError as exc:
            raise AuthError("invalid token") from exc
        if claims.get("typ") != "access":
            raise AuthError("invalid token")
        return AdminPrincipal(
            user_id=uuid.UUID(claims["sub"]),
            email=claims["email"],
            role=claims["role"],
            tenant_id=uuid.UUID(claims["tid"]) if claims.get("tid") else None,
        )

    # --- flows ----------------------------------------------------------------
    async def authenticate(self, email: str, password: str) -> TokenPair:
        # Pre-tenant lookup: the system role is required here (we don't know the tenant yet).
        # Failures are recorded and committed *before* raising, otherwise the
        # transaction rollback would erase the failed-attempt counter.
        failure: AuthError | None = None
        async with self._db.system_session() as s:
            user = (await s.execute(select(AdminUser).where(AdminUser.email == email.lower()))).scalar_one_or_none()
            now = datetime.now(UTC)
            if user is not None and user.locked_until and user.locked_until > now:
                failure = AuthError("account temporarily locked", status=423)
            elif user is None or not verify_password(password, user.password_hash):
                if user is None:
                    verify_password(password, None)  # equalise timing for unknown emails
                else:
                    user.failed_attempts += 1
                    if user.failed_attempts >= self._s.login_max_failures:
                        user.locked_until = now + timedelta(seconds=self._s.login_lockout_s)
                        user.failed_attempts = 0
                failure = AuthError("invalid credentials")
            else:
                user.failed_attempts = 0
                user.locked_until = None
                pair = await self._issue_pair(s, user)
        if failure is not None:
            raise failure
        return pair

    async def refresh(self, refresh_token: str) -> TokenPair:
        token_hash = hashlib.sha256(refresh_token.encode()).digest()
        async with self._db.system_session() as s:
            row = (
                await s.execute(select(AdminRefreshToken).where(AdminRefreshToken.token_hash == token_hash).with_for_update())
            ).scalar_one_or_none()
            if row is None or row.revoked_at is not None or row.expires_at <= datetime.now(UTC):
                raise AuthError("invalid refresh token")
            await s.execute(
                update(AdminRefreshToken).where(AdminRefreshToken.id == row.id).values(revoked_at=datetime.now(UTC))
            )
            user = await s.get(AdminUser, row.user_id)
            if user is None:
                raise AuthError("invalid refresh token")
            return await self._issue_pair(s, user)
