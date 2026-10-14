"""App API keys for the data plane (spec §6.1).

Format: `tg_{env}_{prefix8}_{secret32}`, e.g. `tg_live_a1b2c3d4_<32 base62 chars>`.

* `env` is `live` for prod apps and `test` for dev/staging apps.
* `prefix8` is public (stored in `api_keys.prefix`, shown in the UI) and is the lookup key.
* Only `SHA-256(full_key)` is stored. Keys carry ~190 bits of entropy, so a fast hash
  is correct here; a slow KDF would only add latency to every request.
* Hashes are compared with `hmac.compare_digest` (constant time).
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import string
from dataclasses import dataclass

BASE62 = string.ascii_letters + string.digits
PREFIX_ALPHABET = string.ascii_lowercase + string.digits
KEY_RE = re.compile(r"^tg_(live|test)_([a-z0-9]{8})_([A-Za-z0-9]{32})$")


@dataclass(frozen=True)
class ParsedKey:
    env: str
    prefix: str
    full_key: str


def env_for_app(environment: str) -> str:
    """Map an app's environment to the key's env segment."""
    return "live" if environment == "prod" else "test"


def generate_api_key(env: str) -> tuple[str, str]:
    """Return `(full_key, prefix8)`. The full key must be shown once and then discarded."""
    if env not in ("live", "test"):
        raise ValueError("env must be 'live' or 'test'")
    prefix = "".join(secrets.choice(PREFIX_ALPHABET) for _ in range(8))
    secret = "".join(secrets.choice(BASE62) for _ in range(32))
    return f"tg_{env}_{prefix}_{secret}", prefix


def parse_api_key(raw: str) -> ParsedKey | None:
    """Validate the format and extract the public prefix. Returns None if malformed."""
    m = KEY_RE.match(raw)
    if not m:
        return None
    return ParsedKey(env=m.group(1), prefix=m.group(2), full_key=raw)


def hash_api_key(full_key: str) -> bytes:
    return hashlib.sha256(full_key.encode()).digest()


def verify_api_key(full_key: str, stored_hash: bytes) -> bool:
    """Constant-time comparison of the presented key's hash with the stored one."""
    return hmac.compare_digest(hash_api_key(full_key), stored_hash)
