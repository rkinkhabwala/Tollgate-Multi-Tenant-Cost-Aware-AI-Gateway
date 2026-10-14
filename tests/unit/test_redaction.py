"""Redaction: recognizers, vault, actions, the redact stage, re-hydration, redact_for_local."""

import json

import pytest

from gateway.errors import GatewayError
from gateway.pipeline.context import RouteDecision
from gateway.pipeline.redact import PLACEHOLDER_INSTRUCTION, RedactStage, use_unredacted_for_local
from gateway.pipeline.rehydrate import rehydrate_response
from gateway.policy.schema import AppPolicy, RedactionPolicy
from gateway.providers.base import ChatMessage, ChatRequest, ChatResponse, Choice
from gateway.redaction.engine import Finding, RedactionEngine, apply_actions, effective_actions, resolve_overlaps
from gateway.redaction.vault import Vault
from tests.unit.fakes import auth_ctx, make_ctx


def types(engine: RedactionEngine, text: str, policy: RedactionPolicy | None = None) -> set[str]:
    return {f.entity_type for f in engine.analyze_sync(text, policy or RedactionPolicy())}


# --- recognizers ------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "aws key AK\x49AIOSFODNN7EXAMPLE in config",
        "OPENAI_API_KEY=sk\x2dproj-abcdefghijklmnopqrstuvwx1234",
        "token gh\x70_aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789",
        "slack xo\x78b-123456789012-abcdefghijkl",
        "auth: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.c2lnbmF0dXJlLXZhbHVl",
        "--\x2d--BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA0Z3VS5JJcds3xfn\n-----END RSA PRIVATE KEY-----",
        "DATABASE_URL=postgres://admin:s3cr3tPass@db.internal:5432/app",
        "client_secret: Zq8xK2mP9vL4nR7tW1yB5cF3hJ6dG0sA2eU8iO4pQ",
        'password = "Tr0ub4dor&3"',
        "key tg_live_ab12cd34_" + "A1b2C3d4" * 4,
    ],
)
def test_secret_recognizers(engine: RedactionEngine, text: str) -> None:
    assert "SECRET" in types(engine, text)


def test_low_entropy_long_word_near_keyword_is_not_secret(engine: RedactionEngine) -> None:
    assert "SECRET" not in types(engine, "the token was aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")


@pytest.mark.parametrize(
    ("text", "entity"),
    [
        ("mail jane.roe@corp.internal today", "EMAIL_ADDRESS"),
        ("phone 415-555-0132", "PHONE_NUMBER"),
        ("(212) 555-0187 is the desk", "PHONE_NUMBER"),
        ("+44 20 7946 0958", "PHONE_NUMBER"),
        ("call me at 4155550132 tonight", "PHONE_NUMBER"),
        ("SSN 536-22-8187", "US_SSN"),
        ("card 4111 1111 1111 1111", "CREDIT_CARD"),
        ("Maria Gonzalez approved the refund.", "PERSON"),
    ],
)
def test_pii_recognizers(engine: RedactionEngine, text: str, entity: str) -> None:
    assert entity in types(engine, text)


@pytest.mark.parametrize("text", ["card 4111 1111 1111 1112", "order number 4155550132", "SSN 000-12-3456"])
def test_near_misses_not_flagged(engine: RedactionEngine, text: str) -> None:
    assert not types(engine, text) & {"CREDIT_CARD", "PHONE_NUMBER", "US_SSN"}


def test_custom_patterns_and_deny_terms(engine: RedactionEngine) -> None:
    policy = RedactionPolicy.model_validate(
        {"custom_patterns": [{"name": "ACCOUNT_ID", "regex": r"ACC-\d{8}", "action": "tokenize"}], "deny_terms": ["Project Nightjar"]}
    )
    assert {"ACCOUNT_ID", "DENY_TERM"} <= types(engine, "Move ACC-12345678 under project nightjar.", policy)
    # Another app's policy (no custom patterns) must not see them: recognizers are ad hoc per call.
    assert "ACCOUNT_ID" not in types(engine, "Move ACC-12345678 now.")


def test_allow_action_drops_entity(engine: RedactionEngine) -> None:
    policy = RedactionPolicy.model_validate({"entities": {"EMAIL_ADDRESS": "allow"}})
    assert "EMAIL_ADDRESS" not in types(engine, "write to a@corp.example", policy)


# --- vault + actions ----------------------------------------------------------------
def test_vault_stable_numbering_and_hidden_repr() -> None:
    v = Vault()
    a = v.tokenize("EMAIL_ADDRESS", "Jane@X.com")
    b = v.tokenize("PERSON", "Jane Roe")
    assert a == "<<EMAIL_1>>" and b == "<<PERSON_1>>"
    assert v.tokenize("EMAIL_ADDRESS", "jane@x.com") == a  # same value (case-insensitive email) → same placeholder
    assert v.tokenize("EMAIL_ADDRESS", "bob@x.com") == "<<EMAIL_2>>"
    assert "Jane" not in repr(v) and "Jane" not in str(v)
    assert v.mask("CREDIT_CARD") == "<<CARD>>"


def test_overlap_resolution_prefers_most_protective_action() -> None:
    actions = effective_actions(RedactionPolicy())
    f = [Finding("PERSON", 0, 10, 0.95), Finding("SECRET", 2, 10, 0.6), Finding("EMAIL_ADDRESS", 20, 30, 0.9)]
    assert [x.entity_type for x in resolve_overlaps(f, actions)] == ["SECRET", "EMAIL_ADDRESS"]


def test_apply_actions_tokenize_mask() -> None:
    text = "Jane paid with 4111111111111111"
    vault = Vault()
    out, applied = apply_actions(
        text, [Finding("PERSON", 0, 4, 0.9), Finding("CREDIT_CARD", 15, 31, 1.0)], effective_actions(RedactionPolicy()), vault
    )
    assert out == "<<PERSON_1>> paid with <<CARD>>"
    assert [(a.action, a.placeholder) for a in applied] == [("tokenize", "<<PERSON_1>>"), ("mask", "<<CARD>>")]
    assert vault.get("<<PERSON_1>>") == "Jane"


# --- stage ------------------------------------------------------------------------------
def chat(*messages: ChatMessage, **kw) -> ChatRequest:
    return ChatRequest(messages=list(messages), **kw)


def ctx_for(req: ChatRequest, policy: AppPolicy | None = None):
    return make_ctx(req, auth=auth_ctx(), policy=policy or AppPolicy())


async def test_stage_scans_system_history_tool_results_and_tool_args(engine: RedactionEngine) -> None:
    req = chat(
        ChatMessage(role="system", content="Escalations go to ops-lead@corp.example."),
        ChatMessage(role="user", content="Email jane.roe@corp.example about it."),
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[{"id": "c1", "type": "function", "function": {"name": "send", "arguments": json.dumps({"to": "jane.roe@corp.example", "n": 2})}}],
        ),
        ChatMessage(role="tool", tool_call_id="c1", content='{"delivered_to": "jane.roe@corp.example"}'),
        ChatMessage(role="user", content=[{"type": "text", "text": "Also cc bob@corp.example"}]),
    )
    ctx = await RedactStage(engine).process(ctx_for(req))
    w = ctx.working
    dumped = w.model_dump_json()
    assert "corp.example" not in dumped  # nothing raw goes upstream
    assert w.messages[0].content == PLACEHOLDER_INSTRUCTION  # instruction prepended
    assert w.messages[1].content == "Escalations go to <<EMAIL_1>>."
    assert w.messages[2].content == "Email <<EMAIL_2>> about it."
    args = json.loads(w.messages[3].tool_calls[0]["function"]["arguments"])
    assert args == {"to": "<<EMAIL_2>>", "n": 2}  # same value → same placeholder; JSON still valid
    assert "<<EMAIL_2>>" in w.messages[4].content
    assert w.messages[5].content[0]["text"] == "Also cc <<EMAIL_3>>"
    assert ctx.pii_counts == {"EMAIL_ADDRESS": 5}
    # Original is untouched.
    assert ctx.original.messages[1].content == "Email jane.roe@corp.example about it."


async def test_stage_blocks_secrets_without_echoing(engine: RedactionEngine) -> None:
    secret = "sk\x2dproj-abcdefghijklmnopqrstuvwx1234"
    with pytest.raises(GatewayError) as e:
        await RedactStage(engine).process(ctx_for(chat(ChatMessage(role="user", content=f"use {secret}"))))
    assert e.value.status == 400 and e.value.code == "sensitive_data_blocked"
    assert secret not in e.value.message and "SECRET" in e.value.message


async def test_stage_no_pii_no_instruction(engine: RedactionEngine) -> None:
    ctx = await RedactStage(engine).process(ctx_for(chat(ChatMessage(role="user", content="What is 2 + 2?"))))
    assert not ctx.vault and ctx.pii_counts == {}
    assert ctx.working.messages[0].role == "user"


async def test_disabled_redaction_is_a_noop(engine: RedactionEngine) -> None:
    policy = AppPolicy.model_validate({"redaction": {"enabled": False}})
    ctx = await RedactStage(engine).process(ctx_for(chat(ChatMessage(role="user", content="jane@corp.example")), policy))
    assert ctx.working is None and not ctx.vault


@pytest.mark.parametrize(("tier", "expect_raw"), [("local", True), ("small", False)])
async def test_redact_for_local_false(engine: RedactionEngine, tier: str, expect_raw: bool) -> None:
    policy = AppPolicy.model_validate({"redaction": {"redact_for_local": False}})
    ctx = await RedactStage(engine).process(ctx_for(chat(ChatMessage(role="user", content="Mail jane@corp.example")), policy))
    ctx.route = RouteDecision("x", tier, "test")
    use_unredacted_for_local(ctx)
    sent = ctx.working.model_dump_json()
    assert ("jane@corp.example" in sent) is expect_raw
    assert bool(ctx.vault) is not expect_raw
    assert ctx.unredacted is None


async def test_redact_for_local_false_still_blocks_secrets(engine: RedactionEngine) -> None:
    policy = AppPolicy.model_validate({"redaction": {"redact_for_local": False}})
    with pytest.raises(GatewayError):
        await RedactStage(engine).process(ctx_for(chat(ChatMessage(role="user", content="AK\x49AIOSFODNN7EXAMPLE")), policy))


# --- non-streaming rehydration ----------------------------------------------------------
def test_rehydrate_response_content_and_tool_args() -> None:
    v = Vault()
    e = v.tokenize("EMAIL_ADDRESS", 'jane"quote@corp.example')
    resp = ChatResponse(
        model="m",
        choices=[
            Choice(
                message=ChatMessage(
                    role="assistant",
                    content=f"Sent to {e}; <<EMAIL_9>> unknown; <<CARD>> masked",
                    tool_calls=[{"id": "1", "type": "function", "function": {"name": "f", "arguments": json.dumps({"to": e})}}],
                )
            )
        ],
    )
    out = rehydrate_response(resp, v)
    msg = out.choices[0].message
    assert msg.content == 'Sent to jane"quote@corp.example; <<EMAIL_9>> unknown; <<CARD>> masked'
    assert json.loads(msg.tool_calls[0]["function"]["arguments"]) == {"to": 'jane"quote@corp.example'}
