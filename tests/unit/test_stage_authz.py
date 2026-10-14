"""AuthZ stage: spec §6.2 checks in order."""

import pytest

from gateway.errors import GatewayError
from gateway.pipeline.authz import AuthZStage
from gateway.policy.schema import AppPolicy
from tests.unit.fakes import FakePolicyStore, auth_ctx, chat_request, make_catalog, make_ctx


def stage(policy: AppPolicy | None = None) -> AuthZStage:
    return AuthZStage(FakePolicyStore(policy), make_catalog())  # type: ignore[arg-type]


async def test_happy_path_loads_policy_and_estimates_tokens() -> None:
    ctx = await stage().process(make_ctx(chat_request("hello world"), auth=auth_ctx()))
    assert ctx.policy is not None
    assert ctx.estimated_input_tokens > 0


@pytest.mark.parametrize("field", ["tenant_active", "app_active"])
async def test_suspended_tenant_or_app_403(field: str) -> None:
    with pytest.raises(GatewayError) as e:
        await stage().process(make_ctx(chat_request(), auth=auth_ctx(**{field: False})))
    assert e.value.status == 403 and e.value.code == "permission_denied"


async def test_ip_allowlist() -> None:
    ok = make_ctx(chat_request(), auth=auth_ctx(ip_allowlist=("10.0.0.0/24",)))
    await stage().process(ok)
    blocked = make_ctx(chat_request(), auth=auth_ctx(ip_allowlist=("192.168.0.0/16",)))
    with pytest.raises(GatewayError) as e:
        await stage().process(blocked)
    assert e.value.status == 403


async def test_missing_scope_403() -> None:
    ctx = make_ctx(chat_request(), auth=auth_ctx(scopes=frozenset({"usage:read"})))
    with pytest.raises(GatewayError) as e:
        await stage().process(ctx)
    assert e.value.status == 403 and "chat:write" in e.value.message


async def test_usage_scope_checked_for_usage_endpoint() -> None:
    ctx = make_ctx(None, scope="usage:read", auth=auth_ctx(scopes=frozenset({"chat:write"})))
    with pytest.raises(GatewayError) as e:
        await stage().process(ctx)
    assert e.value.status == 403


@pytest.mark.parametrize("model", ["auto", "tollgate/small", "ollama/qwen2.5:7b"])
async def test_auto_alias_and_allowed_models_pass(model: str) -> None:
    await stage().process(make_ctx(chat_request(model=model), auth=auth_ctx()))


async def test_pinned_model_not_in_allowed_models_403() -> None:
    with pytest.raises(GatewayError) as e:
        await stage().process(make_ctx(chat_request(model="anthropic/claude-opus-5-5"), auth=auth_ctx()))
    assert e.value.status == 403


async def test_allowed_but_unknown_model_404() -> None:
    policy = AppPolicy.model_validate({"routing": {"allowed_models": ["made/up-model"]}})
    with pytest.raises(GatewayError) as e:
        await stage(policy).process(make_ctx(chat_request(model="made/up-model"), auth=auth_ctx()))
    assert e.value.status == 404 and e.value.code == "model_not_found"


async def test_max_input_tokens_enforced() -> None:
    policy = AppPolicy.model_validate({"limits": {"max_input_tokens": 50}})
    with pytest.raises(GatewayError) as e:
        await stage(policy).process(make_ctx(chat_request("word " * 400), auth=auth_ctx()))
    assert e.value.status == 400 and e.value.code == "context_length_exceeded"
    assert "word" not in e.value.message  # never echo the prompt
