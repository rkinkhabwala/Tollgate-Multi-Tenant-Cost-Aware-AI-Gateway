"""OpenAI-shaped errors (spec §6.2).

Every error leaves the gateway as:

    {"error": {"message": "...", "type": "...", "param": null, "code": "..."}}

Messages are written by us and never echo request content, prompts or keys.
"""

from __future__ import annotations


class GatewayError(Exception):
    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        *,
        type_: str = "invalid_request_error",
        param: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.type = type_
        self.param = param
        self.headers = headers or {}

    def body(self) -> dict[str, dict[str, str | None]]:
        return {"error": {"message": self.message, "type": self.type, "param": self.param, "code": self.code}}


def invalid_api_key(message: str = "Invalid API key provided.") -> GatewayError:
    return GatewayError(401, "invalid_api_key", message, type_="invalid_request_error")


def permission_denied(message: str) -> GatewayError:
    return GatewayError(403, "permission_denied", message, type_="permission_error")


def budget_exceeded(message: str = "Monthly budget exceeded for this app.") -> GatewayError:
    return GatewayError(402, "budget_exceeded", message, type_="insufficient_quota")


def rate_limited(message: str, headers: dict[str, str]) -> GatewayError:
    return GatewayError(429, "rate_limit_exceeded", message, type_="rate_limit_error", headers=headers)


def bad_request(message: str, code: str = "invalid_request", param: str | None = None) -> GatewayError:
    return GatewayError(400, code, message, param=param)


def not_found(message: str, code: str = "not_found") -> GatewayError:
    return GatewayError(404, code, message)


def upstream_error(status: int, message: str) -> GatewayError:
    code = {503: "model_unavailable", 504: "upstream_timeout"}.get(status, "upstream_error")
    return GatewayError(status, code, message, type_="server_error")
