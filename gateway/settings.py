"""Process-wide configuration, loaded from environment variables (prefix `TOLLGATE_`).

Everything the gateway needs to start is here. Hosted-provider API keys are optional:
the gateway starts and serves traffic with only Ollama configured (spec §15: provider
keys come from env/secret store, never the DB).
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # No env_file: `.env` holds docker-compose interpolation values (e.g. the Ollama URL
    # *as seen from the container*). The container receives them via compose's
    # `environment:`; host-side tools use these defaults or real env vars.
    model_config = SettingsConfigDict(env_prefix="TOLLGATE_", extra="ignore")

    env: str = "dev"
    log_level: str = "INFO"
    log_file: Path | None = None  # optional extra JSON log sink (tests scan it for leaks)

    # --- PostgreSQL -------------------------------------------------------------
    # Three roles (see docker/postgres/01-roles.sh):
    #  * owner  - runs migrations, owns tables and SECURITY DEFINER helpers
    #  * app    - NOBYPASSRLS; every tenant-scoped query runs as this role
    #  * system - BYPASSRLS; only for pre-tenant lookups (API key by prefix, admin
    #             login) and cross-tenant system work (usage writer, platform admin)
    database_url_owner: str = "postgresql+asyncpg://tollgate:tollgate@localhost:5432/tollgate"
    database_url_app: str = "postgresql+asyncpg://tollgate_app:tollgate_app@localhost:5432/tollgate"
    database_url_system: str = "postgresql+asyncpg://tollgate_system:tollgate_system@localhost:5432/tollgate"

    # --- Redis ------------------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"

    # --- Providers --------------------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout_s: float = 300.0
    # Phase 1 routes every chat request to this single model (spec §16).
    default_model: str = "ollama/qwen2.5:7b"
    openai_api_key: SecretStr | None = None  # optional; adapter arrives in Phase 3
    anthropic_api_key: SecretStr | None = None  # optional; adapter arrives in Phase 3

    # --- Catalog / policy files -------------------------------------------------
    models_yaml: Path = REPO_ROOT / "config" / "models.yaml"
    default_policy_json: Path = REPO_ROOT / "config" / "default_policy.json"

    # --- Data-plane auth --------------------------------------------------------
    auth_cache_ttl_s: int = 60  # spec §6.1
    trust_proxy_headers: bool = False  # honour X-Forwarded-For for IP allowlists

    # --- Admin auth -------------------------------------------------------------
    jwt_secret: SecretStr = SecretStr("dev-only-change-me-dev-only-change-me")
    jwt_access_ttl_s: int = 15 * 60  # spec §6.3: short-lived
    jwt_refresh_ttl_s: int = 7 * 24 * 3600
    login_max_failures: int = 5
    login_lockout_s: int = 15 * 60
    admin_rpm_per_ip: int = 300
    bootstrap_admin_email: str | None = "admin@tollgate.local"
    bootstrap_admin_password: SecretStr | None = SecretStr("admin-dev-password")

    # --- Redaction (spec §7) ----------------------------------------------------
    redaction_workers: int = 4  # thread pool for Presidio/spaCy analysis (see redaction/engine.py)

    # --- Limits -----------------------------------------------------------------
    max_body_bytes: int = 2 * 1024 * 1024  # spec §15
    default_budget_usd: float = 100.0

    # --- Workers ----------------------------------------------------------------
    run_workers: bool = True
    usage_stream: str = "sys:usage_events"
    usage_group: str = "usage-writer"
    usage_flush_interval_s: float = 1.0  # spec §11: batch inserts every second
    usage_batch_size: int = 500
    key_touch_interval_s: float = 5.0
    rollup_interval_s: float = 300.0  # spec §5: refresh usage_daily every 5 minutes

    request_id_header: str = Field(default="x-tollgate-request-id")


@lru_cache
def get_settings() -> Settings:
    return Settings()
