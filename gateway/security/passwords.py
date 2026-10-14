"""Admin password hashing with Argon2id (spec §6.3)."""

from __future__ import annotations

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

# argon2-cffi defaults to Argon2id with RFC 9106 "low memory" parameters.
_hasher = PasswordHasher()

# Used to equalise timing when the user does not exist.
_DUMMY_HASH = _hasher.hash("tollgate-dummy-password")


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, stored_hash: str | None) -> bool:
    """True if `password` matches. Runs a full hash even for unknown users."""
    try:
        return _hasher.verify(stored_hash or _DUMMY_HASH, password) and stored_hash is not None
    except (VerificationError, InvalidHashError):
        return False
