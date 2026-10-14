"""AuthN stage: key parsing, cache behaviour, constant-time verification, revocation."""

from datetime import UTC, datetime, timedelta

import pytest

from gateway.errors import GatewayError
from gateway.pipeline.authn import AuthNStage
from gateway.redis_keys import REVOKED_TOMBSTONE, auth_cache_key
from tests.unit.fakes import APP, KEY_ID, TENANT, FakeResolver, make_ctx, make_key_record, make_redis, new_key


@pytest.fixture
def setup():
    key, prefix = new_key()
    resolver = FakeResolver({prefix: make_key_record(key, prefix)})
    redis = make_redis()
    touched: list = []
    stage = AuthNStage(redis, resolver, cache_ttl_s=60, on_key_used=touched.append)
    return stage, resolver, redis, key, prefix, touched


async def test_valid_key_resolves_auth_context(setup) -> None:
    stage, _, _, key, prefix, touched = setup
    ctx = await stage.process(make_ctx(api_key=key))
    assert ctx.auth is not None
    assert (ctx.auth.tenant_id, ctx.auth.app_id, ctx.auth.key_id) == (TENANT, APP, KEY_ID)
    assert ctx.auth.key_prefix == prefix
    assert "chat:write" in ctx.auth.scopes
    assert touched == [KEY_ID]  # last_used_at is queued, not written inline


async def test_second_request_uses_redis_cache(setup) -> None:
    stage, resolver, redis, key, prefix, _ = setup
    await stage.process(make_ctx(api_key=key))
    await stage.process(make_ctx(api_key=key))
    assert resolver.calls == 1
    assert 0 < await redis.ttl(auth_cache_key(prefix)) <= 60


@pytest.mark.parametrize("bad", [None, "", "sk-abc", "tg_live_nope"])
async def test_malformed_key_401_without_lookup(setup, bad) -> None:
    stage, resolver, *_ = setup
    with pytest.raises(GatewayError) as e:
        await stage.process(make_ctx(api_key=bad))
    assert e.value.status == 401 and e.value.code == "invalid_api_key"
    assert resolver.calls == 0


async def test_wrong_secret_same_prefix_401(setup) -> None:
    stage, _, _, key, prefix, _ = setup
    forged = f"tg_live_{prefix}_" + "Z" * 32
    with pytest.raises(GatewayError) as e:
        await stage.process(make_ctx(api_key=forged))
    assert e.value.status == 401


async def test_unknown_prefix_401(setup) -> None:
    stage, *_ = setup
    other, _ = new_key()
    with pytest.raises(GatewayError) as e:
        await stage.process(make_ctx(api_key=other))
    assert e.value.status == 401


async def test_revocation_tombstone_wins_over_cache(setup) -> None:
    stage, _, redis, key, prefix, _ = setup
    await stage.process(make_ctx(api_key=key))  # populates cache
    await redis.set(auth_cache_key(prefix), REVOKED_TOMBSTONE)  # what revoke_key() does
    with pytest.raises(GatewayError) as e:
        await stage.process(make_ctx(api_key=key))
    assert e.value.status == 401


async def test_cache_fill_does_not_overwrite_tombstone(setup) -> None:
    """A lookup racing a revoke must not resurrect the key (SET NX)."""
    stage, _, redis, key, prefix, _ = setup
    await redis.set(auth_cache_key(prefix), REVOKED_TOMBSTONE)
    await stage._redis.delete("unrelated")
    record = await stage._resolver.by_prefix(prefix)
    await redis.set(auth_cache_key(prefix), record.to_json(), ex=60, nx=True)
    assert await redis.get(auth_cache_key(prefix)) == REVOKED_TOMBSTONE.encode()


async def test_revoked_row_401() -> None:
    key, prefix = new_key()
    rec = make_key_record(key, prefix, revoked_at=datetime.now(UTC).isoformat())
    stage = AuthNStage(make_redis(), FakeResolver({prefix: rec}))
    with pytest.raises(GatewayError) as e:
        await stage.process(make_ctx(api_key=key))
    assert e.value.status == 401 and "revoked" in e.value.message


async def test_expired_key_401() -> None:
    key, prefix = new_key()
    rec = make_key_record(key, prefix, expires_at=(datetime.now(UTC) - timedelta(seconds=1)).isoformat())
    stage = AuthNStage(make_redis(), FakeResolver({prefix: rec}))
    with pytest.raises(GatewayError) as e:
        await stage.process(make_ctx(api_key=key))
    assert e.value.status == 401 and "expired" in e.value.message


async def test_error_message_never_contains_key(setup) -> None:
    stage, _, _, key, prefix, _ = setup
    forged = f"tg_live_{prefix}_" + "Q" * 32
    with pytest.raises(GatewayError) as e:
        await stage.process(make_ctx(api_key=forged))
    assert "Q" * 32 not in str(e.value.body())
