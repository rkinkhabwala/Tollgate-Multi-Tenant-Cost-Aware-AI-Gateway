"""Unit-test fixtures. The Presidio engine is loaded once per session (≈ 1–7 s)."""

import pytest

from gateway.redaction.engine import RedactionEngine


@pytest.fixture(scope="session")
def engine() -> RedactionEngine:
    eng = RedactionEngine(workers=2)
    yield eng
    eng.close()
