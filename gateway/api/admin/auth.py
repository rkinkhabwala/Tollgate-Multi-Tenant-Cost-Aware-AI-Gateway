"""`POST /admin/auth/login` and `POST /admin/auth/refresh` (spec §6.3, §12).

Login returns a 15-minute access JWT and a rotating refresh token. Repeated failures
lock the account (HTTP 423). Error messages never say whether the email exists.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from gateway.api.deps import get_services
from gateway.errors import GatewayError
from gateway.security.admin_auth import AuthError, TokenPair

router = APIRouter(prefix="/admin/auth", tags=["admin:auth"])


class LoginIn(BaseModel):
    email: str = Field(max_length=320)
    password: str = Field(max_length=1024)


class RefreshIn(BaseModel):
    refresh_token: str = Field(max_length=256)


class TokenOut(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

    @classmethod
    def of(cls, pair: TokenPair) -> TokenOut:
        return cls(access_token=pair.access_token, refresh_token=pair.refresh_token, expires_in=pair.expires_in)


def _auth_error(exc: AuthError) -> GatewayError:
    code = "account_locked" if exc.status == 423 else "invalid_credentials"
    return GatewayError(exc.status, code, exc.reason.capitalize() + ".")


@router.post("/login")
async def login(body: LoginIn, request: Request) -> TokenOut:
    try:
        return TokenOut.of(await get_services(request).admin_auth.authenticate(body.email, body.password))
    except AuthError as exc:
        raise _auth_error(exc) from exc


@router.post("/refresh")
async def refresh(body: RefreshIn, request: Request) -> TokenOut:
    try:
        return TokenOut.of(await get_services(request).admin_auth.refresh(body.refresh_token))
    except AuthError as exc:
        raise _auth_error(exc) from exc
