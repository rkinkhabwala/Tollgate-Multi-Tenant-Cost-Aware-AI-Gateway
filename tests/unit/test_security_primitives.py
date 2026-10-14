"""API key format/hash, Argon2id passwords, admin JWTs."""

import uuid

import pytest

from gateway.security.admin_auth import AuthError, LocalAdminAuth
from gateway.security.api_keys import KEY_RE, env_for_app, generate_api_key, hash_api_key, parse_api_key, verify_api_key
from gateway.security.passwords import hash_password, verify_password
from gateway.settings import Settings


def test_key_format_matches_spec() -> None:
    key, prefix = generate_api_key("live")
    assert KEY_RE.match(key)
    assert key.startswith(f"tg_live_{prefix}_")
    assert len(key.split("_")[3]) == 32


def test_keys_are_unique() -> None:
    assert len({generate_api_key("test")[0] for _ in range(200)}) == 200


@pytest.mark.parametrize("bad", ["", "tg_live_short_x", "sk-123", "tg_prod_abcdefgh_" + "a" * 32, "tg_live_ABCDEFGH_" + "a" * 32])
def test_parse_rejects_malformed(bad: str) -> None:
    assert parse_api_key(bad) is None


def test_hash_and_constant_time_verify() -> None:
    key, _ = generate_api_key("live")
    stored = hash_api_key(key)
    assert len(stored) == 32  # SHA-256
    assert verify_api_key(key, stored)
    assert not verify_api_key(key[:-1] + ("A" if key[-1] != "A" else "B"), stored)


def test_env_mapping() -> None:
    assert env_for_app("prod") == "live"
    assert env_for_app("dev") == env_for_app("staging") == "test"


def test_argon2id_passwords() -> None:
    h = hash_password("correct horse battery")
    assert h.startswith("$argon2id$")
    assert verify_password("correct horse battery", h)
    assert not verify_password("wrong", h)
    assert not verify_password("anything", None)


def test_jwt_roundtrip_and_tamper() -> None:
    from gateway.db.models import AdminUser

    auth = LocalAdminAuth(db=None, settings=Settings())  # type: ignore[arg-type]
    user = AdminUser(id=uuid.uuid4(), email="a@b.c", role="tenant_admin", tenant_id=uuid.uuid4())
    token = auth._issue_access(user)
    p = auth.verify_access_token(token)
    assert (p.user_id, p.role, p.tenant_id) == (user.id, "tenant_admin", user.tenant_id)
    with pytest.raises(AuthError):
        auth.verify_access_token(token[:-2] + "xx")


def test_jwt_wrong_secret_rejected() -> None:
    from gateway.db.models import AdminUser

    a = LocalAdminAuth(db=None, settings=Settings(jwt_secret="one-secret-of-sufficient-length-1234"))  # type: ignore[arg-type]
    b = LocalAdminAuth(db=None, settings=Settings(jwt_secret="another-secret-of-sufficient-len-99"))  # type: ignore[arg-type]
    token = a._issue_access(AdminUser(id=uuid.uuid4(), email="x@y.z", role="platform_admin", tenant_id=None))
    with pytest.raises(AuthError):
        b.verify_access_token(token)
