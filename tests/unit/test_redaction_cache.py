"""Per-message findings cache: identical results, scoped keys, bounded memory, no text stored."""

import json
import uuid

from gateway.pipeline.redact import RedactStage
from gateway.policy.schema import AppPolicy
from gateway.providers.base import ChatMessage, ChatRequest
from gateway.redaction.cache import CacheScope, FindingsCache
from gateway.redaction.engine import Finding, RedactionEngine
from tests.unit.fakes import auth_ctx, make_ctx

CONVERSATION = [
    ChatMessage(role="system", content="Escalations go to Maria Gonzalez (maria.g@northwind.example)."),
    ChatMessage(role="user", content="Hi, I'm Kenji Watanabe, call me at 415-555-0142. Card 4111 1111 1111 1111."),
    ChatMessage(role="assistant", content="Thanks Kenji Watanabe, I've noted 415-555-0142."),
    ChatMessage(role="user", content="Also email kenji.w@contoso.example and cc maria.g@northwind.example."),
]


def snapshot(ctx) -> tuple[str, list[tuple[str, str]], dict]:
    vault = [(ph, ctx.vault.get(ph)) for ph in sorted(ctx.vault._by_placeholder)]
    return ctx.working.model_dump_json(), vault, ctx.pii_counts


async def test_cached_reapplication_is_identical_to_fresh(engine: RedactionEngine) -> None:
    engine.cache.clear()
    stage = RedactStage(engine)
    req = ChatRequest(messages=CONVERSATION)

    fresh = snapshot(await stage.process(make_ctx(req, auth=auth_ctx(), policy=AppPolicy())))
    misses_after_fresh = engine.cache.misses
    cached = snapshot(await stage.process(make_ctx(req, auth=auth_ctx(), policy=AppPolicy())))

    assert engine.cache.misses == misses_after_fresh  # second run analysed nothing
    assert cached == fresh  # same redacted text, same placeholders + numbering, same counts
    assert "<<PERSON_1>>" in fresh[0] and "<<EMAIL_2>>" in fresh[0]


async def test_new_turn_only_analyses_the_new_message(engine: RedactionEngine) -> None:
    engine.cache.clear()
    stage = RedactStage(engine)
    await stage.process(make_ctx(ChatRequest(messages=CONVERSATION), auth=auth_ctx(), policy=AppPolicy()))
    before = engine.cache.misses
    nxt = ChatRequest(messages=[*CONVERSATION, ChatMessage(role="user", content="Any update for Priya Raman?")])
    await stage.process(make_ctx(nxt, auth=auth_ctx(), policy=AppPolicy()))
    assert engine.cache.misses - before == 1


def test_keys_are_scoped_by_tenant_app_and_policy_version() -> None:
    cache = FindingsCache()
    t, a = uuid.uuid4(), uuid.uuid4()
    base = cache.key(CacheScope(t, a, 1), "hello Jane")
    assert base == cache.key(CacheScope(t, a, 1), "hello Jane")
    assert base != cache.key(CacheScope(uuid.uuid4(), a, 1), "hello Jane")
    assert base != cache.key(CacheScope(t, uuid.uuid4(), 1), "hello Jane")
    assert base != cache.key(CacheScope(t, a, 2), "hello Jane")  # policy change invalidates
    assert base != FindingsCache().key(CacheScope(t, a, 1), "hello Jane")  # per-process secret
    assert b"Jane" not in base and len(base) == 32


async def test_policy_version_change_reanalyses(engine: RedactionEngine) -> None:
    engine.cache.clear()
    stage = RedactStage(engine)
    req = ChatRequest(messages=CONVERSATION)
    await stage.process(make_ctx(req, auth=auth_ctx(policy_version=1), policy=AppPolicy()))
    before = engine.cache.misses
    await stage.process(make_ctx(req, auth=auth_ctx(policy_version=2), policy=AppPolicy()))
    assert engine.cache.misses - before == len(CONVERSATION)


def test_bounded_by_entries_and_findings() -> None:
    cache = FindingsCache(max_entries=3, max_findings=5)
    scope = CacheScope(uuid.uuid4(), uuid.uuid4(), 1)
    f = Finding("PERSON", 0, 4, 0.85)
    for i in range(10):
        cache.put(cache.key(scope, f"t{i}"), [f])
    assert len(cache) == 3
    cache.put(cache.key(scope, "big"), [f] * 4)
    assert cache.findings_held <= 5
    assert cache.get(cache.key(scope, "big")) is not None  # newest survives
    assert cache.get(cache.key(scope, "t0")) is None  # oldest evicted


async def test_cache_holds_offsets_and_types_only(engine: RedactionEngine) -> None:
    engine.cache.clear()
    await RedactStage(engine).process(make_ctx(ChatRequest(messages=CONVERSATION), auth=auth_ctx(), policy=AppPolicy()))
    dumped = json.dumps([[(f.entity_type, f.start, f.end, f.score) for f in v] for v in engine.cache._data.values()])
    for value in ("Kenji", "Watanabe", "415-555-0142", "northwind", "4111"):
        assert value not in dumped
    assert all(len(k) == 32 for k in engine.cache._data)  # HMAC digests, not text
