"""The logging filter must keep API keys, bearer tokens and prompt payloads out of logs."""

import io
import logging

import pytest

from gateway.observability.logging import JsonFormatter, SensitiveDataFilter, scrub

KEY = "tg_live_ab12cd34_" + "A" * 32


@pytest.fixture
def capture() -> tuple[logging.Logger, io.StringIO]:
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(SensitiveDataFilter())
    logger = logging.getLogger("test.sensitive")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    return logger, buf


def test_scrub_api_key_keeps_public_prefix() -> None:
    out = scrub(f"using {KEY}")
    assert "A" * 32 not in out
    assert "tg_live_ab12cd34_[REDACTED]" in out


def test_scrub_bearer_sk_and_jwt() -> None:
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.c2lnbmF0dXJlLXZhbHVl"
    out = scrub(f"Authorization: Bearer abc.def sk-proj-1234567890abc {jwt}")
    assert "abc.def" not in out
    assert "1234567890abc" not in out
    assert jwt not in out


def test_args_are_scrubbed(capture: tuple[logging.Logger, io.StringIO]) -> None:
    logger, buf = capture
    logger.info("key=%s", KEY)
    assert "A" * 32 not in buf.getvalue()


def test_sensitive_extras_dropped(capture: tuple[logging.Logger, io.StringIO]) -> None:
    logger, buf = capture
    logger.info("chat", extra={"messages": [{"content": "my SSN is 123-45-6789"}], "prompt": "secret prompt", "tenant_id": "t1"})
    line = buf.getvalue()
    assert "123-45-6789" not in line
    assert "secret prompt" not in line
    assert '"tenant_id": "t1"' in line


def test_exception_text_scrubbed(capture: tuple[logging.Logger, io.StringIO]) -> None:
    logger, buf = capture
    try:
        raise ValueError(f"bad key {KEY}")
    except ValueError:
        logger.exception("boom")
    assert "A" * 32 not in buf.getvalue()
