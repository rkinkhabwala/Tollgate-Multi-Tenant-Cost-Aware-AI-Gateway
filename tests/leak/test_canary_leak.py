"""Leak test: canary PII must never land in logs, Redis or Postgres (spec §4, §7.4, §15).

Runs LAST in the session (see `testpaths` in pyproject.toml), so the scan also
covers everything every earlier test produced.

1. Drive a canary PII set through every path that touches user text:
   - chat JSON and SSE, with `include_usage`;
   - a blocked secret;
   - a validation error that echoes nothing;
   - an over-size body;
   - the `/admin/redaction/test` dry run;
   - a multi-turn history with tool results.
2. Scan these and assert zero canary hits:
   - the session's gateway log file;
   - every key in the test Redis DB (strings, hashes, lists, sets, sorted sets,
     and stream entries);
   - every row of every table and the `usage_daily` view in the test database.

Canaries are checked case-insensitively, and with separators stripped for the
number-like ones.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

import httpx
import pytest
from redis.asyncio import Redis
from sqlalchemy import text

from gateway.db.session import Database
from gateway.settings import Settings
from tests.conftest import Provisioned, TenantFactory

pytestmark = [pytest.mark.stack, pytest.mark.ollama]

CANARIES = {
    "email": "henrietta.leak7731@canary-mail.example",
    "phone": "415-555-0177",
    "ssn": "529-41-7731",
    "card": "4012 8888 8888 1881",
    "name": "Henrietta Castellano",
    "secret": "sk\x2dproj-CANARYleak7731xyzABCDEFGHIJKLmnop",
    "conn": "postgres://canaryuser:Leak7731Pw@db.canary.internal/app",
}


def needles() -> list[str]:
    out = []
    for v in CANARIES.values():
        out.append(v.casefold())
        digits = re.sub(r"\D", "", v)
        if v[0].isdigit() and len(digits) >= 9:
            out.append(digits)  # also catch separator-free forms
    out += ["leak7731"]  # distinctive fragment present in several canaries
    return out


def hits(blob: str) -> list[str]:
    low = blob.casefold()
    squashed = re.sub(r"[\s-]", "", low)
    return [n for n in needles() if n in low or n in squashed]


def test_detector_self_check() -> None:
    """The scanner must flag every canary, including reformatted variants."""
    for value in CANARIES.values():
        assert hits(f"xx {value} yy"), value
    assert hits("card 4012888888881881")  # separators stripped
    assert hits("HENRIETTA.LEAK7731@CANARY-MAIL.EXAMPLE")  # case-folded
    assert not hits("<<EMAIL_1>> <<PHONE_1>> <<SSN>> <<CARD>> <<PERSON_1>>")


@pytest.fixture(scope="module")
async def tenant(tenant_factory: TenantFactory) -> Provisioned:
    return await tenant_factory("leak")


def bearer(key: str) -> dict[str, str]:
    return {"authorization": f"Bearer {key}"}


PII_PROMPT = (
    f"Repeat back exactly: {CANARIES['name']} ({CANARIES['email']}, {CANARIES['phone']}), "
    f"SSN {CANARIES['ssn']}, card {CANARIES['card']}."
)


async def test_drive_canaries_through_every_path(gateway: httpx.AsyncClient, tenant: Provisioned) -> None:
    h = bearer(tenant.key)
    # Chat, non-streaming: the client may get values back (re-hydrated); nothing else may.
    r = await gateway.post(
        "/v1/chat/completions", headers=h, json={"model": "auto", "max_tokens": 60, "temperature": 0, "messages": [{"role": "user", "content": PII_PROMPT}]}
    )
    assert r.status_code == 200 and int(r.headers["x-tollgate-redacted"]) >= 4
    assert "<<EMAIL_1>>" not in r.text  # re-hydrated if echoed

    # Chat, streaming with include_usage + multi-turn history with a tool result.
    r = await gateway.post(
        "/v1/chat/completions",
        headers=h,
        json={
            "model": "auto",
            "stream": True,
            "max_tokens": 40,
            "stream_options": {"include_usage": True},
            "messages": [
                {"role": "system", "content": f"The VIP customer is {CANARIES['name']}."},
                {"role": "user", "content": f"Look up {CANARIES['email']}"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "lookup", "arguments": json.dumps({"email": CANARIES["email"]})}}],
                },
                {"role": "tool", "tool_call_id": "t1", "content": json.dumps({"phone": CANARIES["phone"], "ssn": CANARIES["ssn"]})},
                {"role": "user", "content": "Summarise in one line."},
            ],
        },
    )
    assert r.status_code == 200 and r.text.rstrip().endswith("data: [DONE]")

    # Blocked secrets (never sent upstream, never echoed).
    for secret in (CANARIES["secret"], CANARIES["conn"]):
        r = await gateway.post("/v1/chat/completions", headers=h, json={"model": "auto", "messages": [{"role": "user", "content": f"use {secret}"}]})
        assert r.status_code == 400 and r.json()["error"]["code"] == "sensitive_data_blocked"
        assert not hits(r.text)

    # Validation error carrying canaries.
    r = await gateway.post("/v1/chat/completions", headers=h, json={"model": "auto", "messages": [{"role": "oracle", "content": PII_PROMPT}]})
    assert r.status_code == 400 and not hits(r.text)

    # Over-size body carrying canaries.
    big = PII_PROMPT + " pad" * 600_000
    r = await gateway.post("/v1/chat/completions", headers=h, json={"model": "auto", "messages": [{"role": "user", "content": big}]})
    assert r.status_code == 413 and not hits(r.text)

    # Redaction dry run: response shows placeholders and masks only.
    r = await gateway.post(
        "/admin/redaction/test",
        headers=tenant.admin_headers,
        json={"app_id": tenant.app_id, "text": PII_PROMPT + " " + CANARIES["secret"]},
    )
    assert r.status_code == 200 and r.json()["blocked"] is True
    assert not hits(json.dumps(r.json()["entities"])) and not hits(r.json()["redacted_text"])

    await asyncio.sleep(1.5)  # let the usage writer flush events to Postgres


async def test_logs_contain_no_canaries(test_settings: Settings) -> None:
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert test_settings.log_file is not None
    content = test_settings.log_file.read_text(encoding="utf-8")
    assert content, "expected the gateway to have written logs this session"
    found = hits(content)
    assert not found, f"canary values found in gateway logs: {found}"


async def _dump_redis(redis: Redis) -> str:
    parts: list[str] = []
    async for raw_key in redis.scan_iter(count=1000):
        key = raw_key.decode()
        kind = (await redis.type(raw_key)).decode()
        parts.append(key)
        if kind == "string":
            parts.append(repr(await redis.get(raw_key)))
        elif kind == "hash":
            parts.append(repr(await redis.hgetall(raw_key)))
        elif kind == "list":
            parts.append(repr(await redis.lrange(raw_key, 0, -1)))
        elif kind == "set":
            parts.append(repr(await redis.smembers(raw_key)))
        elif kind == "zset":
            parts.append(repr(await redis.zrange(raw_key, 0, -1)))
        elif kind == "stream":
            parts.append(repr(await redis.xrange(raw_key)))
            for group in await redis.xinfo_groups(raw_key):
                parts.append(repr(group))
    return "\n".join(parts)


async def test_redis_contains_no_canaries(redis: Redis) -> None:
    dump = await _dump_redis(redis)
    assert dump, "expected Redis state from this session"
    found = hits(dump)
    assert not found, f"canary values found in Redis: {found}"


async def test_postgres_contains_no_canaries(db: Database) -> None:
    async with db.system_session() as s:
        await s.execute(text("SELECT refresh_usage_daily()"))
        tables = list(
            (
                await s.execute(
                    text(
                        "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                        "WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p', 'm') AND NOT c.relispartition "
                        "AND c.relname <> 'alembic_version'"
                    )
                )
            ).scalars()
        )
        dumps = {}
        for t in tables:
            rows = (await s.execute(text(f'SELECT row_to_json(x)::text FROM "{t}" x'))).scalars().all()
            dumps[t] = "\n".join(rows)
    assert "usage_events" in dumps and dumps["usage_events"], "expected ledger rows from this session"
    leaks = {t: hits(d) for t, d in dumps.items() if hits(d)}
    assert not leaks, f"canary values found in Postgres: {leaks}"
    # Positive control: the ledger did record PII *counts* for the canary requests.
    assert '"pii_entities":{"' in dumps["usage_events"]
