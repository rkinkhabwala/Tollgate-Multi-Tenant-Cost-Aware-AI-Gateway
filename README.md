# Tollgate

A multi-tenant, cost-aware, **OpenAI-compatible AI gateway**. Apps change only `base_url` and `api_key`. Tollgate authenticates the app, enforces its policy and rate limits, calls the model, and records exactly what every tenant spent and saved.

The full design is in [spec.md](spec.md). This repo is at **Phase 2** (spec §16).

| Delivered | Arrives later |
|---|---|
| **Phase 1:** compose stack, Postgres RLS on every tenant table, partitioned usage ledger, admin API (tenants, apps, keys, users) with audit log, OpenAI-compatible `/v1/chat/completions` (JSON + SSE), `/v1/models`, `/v1/usage`, rate limiting, cost and savings metering | Classifier routing, hosted providers, fallback (Phase 3) |
| **Phase 2:** PII and secret redaction (Presidio + spaCy + custom recognizers), tokenize / mask / block / allow, in-memory vault, non-streaming and streaming re-hydration, `GET/PUT` app policy, `/admin/redaction/test`, PII eval gate, canary leak test | Caching, budgets, optimization (Phase 4) |
| | Admin UI, Prometheus, OTel (Phase 5) |
| | Load and chaos testing (Phase 6) |

## Quick start

Prerequisites:
- Docker with Compose v2;
- [uv](https://docs.astral.sh/uv/) for running tests and scripts on the host;
- [Ollama](https://ollama.com) installed natively, plus about 6 GB of disk for the models.

```bash
ollama serve &                            # or: brew services start ollama
ollama pull qwen2.5:7b && ollama pull nomic-embed-text
cp .env.example .env                      # optional; the defaults are baked into docker-compose.yml
docker compose up -d                      # gateway, postgres, redis
curl localhost:8000/readyz                # {"status":"ok","postgres":"ok","redis":"ok"}
```

By default the gateway talks to the **native Ollama on the host** (`TOLLGATE_OLLAMA_BASE_URL=http://host.docker.internal:11434`). That is much faster than a container on macOS, where Docker has no GPU access.

**Fallback: Ollama in Docker.** It runs on CPU on macOS, so it is slow, but it needs nothing installed on the host:

```bash
TOLLGATE_OLLAMA_BASE_URL=http://ollama:11434 docker compose --profile ollama-docker up -d
docker compose logs -f ollama-init        # one-time pull of qwen2.5:7b + nomic-embed-text (~5 GB)
```

Put `TOLLGATE_OLLAMA_BASE_URL=http://ollama:11434` in `.env` to make the fallback stick.

The gateway applies migrations on start and creates a bootstrap **platform admin**: `admin@tollgate.local` / `admin-dev-password`. Override both via `.env` (see [.env.example](.env.example)).

### Seed demo tenants

```bash
docker compose exec gateway python -m scripts.seed_demo_tenants    # or: uv run python -m scripts.seed_demo_tenants
```

This creates 2 tenants (`acme`, `globex`) with 2 apps each. Each app gets the default policy, a monthly budget and one API key. **Keys are printed once.** Re-running reuses the tenants and apps and mints a fresh key per app. Tenant admins are `admin@acme.example` and `admin@globex.example`, password `demo-tenant-password`.

### Call it with curl

```bash
export TG_KEY=tg_live_xxxxxxxx_...   # a key from the seed output

curl -s -D - localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $TG_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"Say hello"}],"max_tokens":20}'

# Streaming (SSE)
curl -N localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $TG_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"auto","stream":true,"messages":[{"role":"user","content":"Count to 5"}]}'

curl -s localhost:8000/v1/usage -H "Authorization: Bearer $TG_KEY"           # this app's usage, last 30 days
curl -s "localhost:8000/v1/usage?from=2026-10-01&to=2026-10-31" -H "Authorization: Bearer $TG_KEY"
```

Every response carries these headers (spec §11):

| Header | Meaning |
|---|---|
| `x-tollgate-request-id` | Request id |
| `x-tollgate-model` | Model the request was routed to |
| `x-tollgate-route-reason` | Why that model was chosen |
| `x-tollgate-cache` | `miss`, `exact` or `semantic` (always `miss` until Phase 4) |
| `x-tollgate-cost-usd` | Actual cost of this call |
| `x-tollgate-saved-usd` | Baseline-model cost minus actual cost |
| `x-tollgate-redacted` | Number of redacted entities (a count; never values) |
| `x-ratelimit-*` | Remaining rate-limit budget |

On streams, cost and savings read `pending` because the headers are sent before the final token count is known. With `stream_options: {"include_usage": true}`, the final usage chunk carries the real numbers in a `tollgate` extension object, which the OpenAI SDK exposes as `chunk.model_extra["tollgate"]`:

```json
{"choices": [], "usage": {...}, "tollgate": {"model": "ollama/qwen2.5:7b", "cache": "miss", "cost_usd": 0.0, "saved_usd": 0.000423, "redacted": 3}}
```

### Call it with the OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="tg_live_...")

resp = client.chat.completions.create(model="auto", messages=[{"role": "user", "content": "Hi!"}])
print(resp.choices[0].message.content, resp.model)          # → ollama/qwen2.5:7b

for chunk in client.chat.completions.create(
    model="auto", messages=[{"role": "user", "content": "Count to 5"}], stream=True
):
    print(chunk.choices[0].delta.content or "", end="") if chunk.choices else None
```

`model` can be `auto`, a tier alias (`tollgate/small`), or a pinned catalog id. A pinned id must be in the app policy's `allowed_models` (otherwise 403). It is honoured only when the policy has `allow_pinning: true` and that provider is configured. In Phase 1 everything else routes to `TOLLGATE_DEFAULT_MODEL` (`ollama/qwen2.5:7b`).

### Admin API (control plane)

```bash
TOKEN=$(curl -s localhost:8000/admin/auth/login -H 'Content-Type: application/json' \
  -d '{"email":"admin@tollgate.local","password":"admin-dev-password"}' | python3 -c 'import json,sys;print(json.load(sys.stdin)["access_token"])')
H="Authorization: Bearer $TOKEN"

curl -s -X POST localhost:8000/admin/tenants -H "$H" -H 'Content-Type: application/json' -d '{"name":"initech"}'
curl -s -X POST localhost:8000/admin/tenants/$TID/users -H "$H" -H 'Content-Type: application/json' \
  -d '{"email":"ops@initech.example","password":"a-long-password","role":"tenant_admin"}'
curl -s -X POST localhost:8000/admin/tenants/$TID/apps -H "$H" -H 'Content-Type: application/json' \
  -d '{"name":"helpdesk","environment":"prod","baseline_model":"anthropic/claude-sonnet-5-5"}'
curl -s -X POST localhost:8000/admin/apps/$AID/keys -H "$H" -H 'Content-Type: application/json' -d '{}'   # key shown once
curl -s localhost:8000/admin/apps/$AID/keys -H "$H"                                                       # list (no secrets)
curl -s -X DELETE localhost:8000/admin/keys/$KID -H "$H"                                                  # revoke (effective immediately)
```

| Endpoint | Roles |
|---|---|
| `POST /admin/auth/login`, `/admin/auth/refresh` | anyone |
| `POST/GET /admin/tenants` | platform_admin |
| `POST /admin/tenants/{tid}/users` | platform_admin, tenant_admin (own tenant) |
| `POST/GET /admin/tenants/{tid}/apps`, `PATCH .../apps/{aid}` | tenant_admin (GET also tenant_viewer) |
| `POST/GET /admin/apps/{aid}/keys`, `DELETE /admin/keys/{kid}` | tenant_admin |

Every mutation writes to `audit_log`. Five failed logins lock an account for 15 minutes. `/admin/*` is rate-limited per IP.

## Sensitive-data redaction (Phase 2)

Every text the model will see is scanned before anything leaves the gateway:
- system and developer prompts, and the *whole* history;
- multi-part content;
- tool results;
- tool-call arguments in history (parsed as JSON, so the result stays valid JSON).

Findings are replaced according to the app policy:

| Action | Effect | Default for |
|---|---|---|
| `tokenize` | `<<EMAIL_1>>`; the original is kept in the per-request in-memory vault and re-hydrated in the response | PERSON, EMAIL_ADDRESS, PHONE_NUMBER, LOCATION, IP_ADDRESS, DENY_TERM |
| `mask` | `<<CARD>>`, never re-hydrated | CREDIT_CARD, US_SSN, US_PASSPORT, US_DRIVER_LICENSE, US_BANK_NUMBER, IBAN_CODE, MEDICAL_LICENSE |
| `block` | `400 sensitive_data_blocked` (the message names entity types only) | SECRET: AWS/OpenAI/Stripe/GitHub/Slack/Google keys, JWTs, PEM private keys, connection strings with passwords, password assignments, high-entropy tokens near key/secret/token/password |
| `allow` | untouched | DATE_TIME and anything not listed |

**How it behaves**
- Numbering is stable within a request: the same value always gets the same placeholder. When anything is tokenized, a system instruction tells the model to keep placeholders verbatim.
- Responses are re-hydrated, including tool-call arguments (JSON-escaped).
- Streams use a 32-char tail buffer, so a placeholder split across chunks (the model really emits `<<`, `EMAIL`, `_`, `1`, `>>`) reaches the client as the real value.
- Only placeholders from *this* request's vault are replaced.
- The vault never touches Redis, Postgres or logs. Usage events record `pii_entities` as counts by type.

**Policy settings** (`GET/PUT /admin/apps/{aid}/policy`, validated against the full spec §9 schema):
- `redaction.entities`: per-type action.
- `redaction.custom_patterns`: `[{name, regex, action}]`, applied per app only.
- `redaction.deny_terms`: literal terms such as codenames.
- `redaction.score_threshold`.
- `redaction.redact_for_local`: when `false`, requests routed to the local tier are sent unredacted. `block` still applies.

Each PUT bumps the policy version, is audited, and applies to the very next request.

```bash
curl -s localhost:8000/admin/redaction/test -H "Authorization: Bearer $ADMIN_TOKEN" -H 'Content-Type: application/json' \
  -d '{"app_id":"…","text":"Email jane@corp.example or call 415-555-0163. Key AK''IAIOSFODNN7EXAMPLE"}'
# → {"blocked": true, "redacted_text": "Email <<EMAIL_1>> or call <<PHONE_1>>. Key <<SECRET>>", "entities": [...], "counts": {...}}
```

**Quality and cost** (`uv run python -m scripts.eval_pii`, `uv run python -m scripts.bench_redaction`; M1 Pro):

| Entity | Recall | Precision | Target |
|---|---|---|---|
| EMAIL_ADDRESS | 100% | 100% | ≥ 95% |
| PHONE_NUMBER | 100% | 100% | ≥ 95% |
| CREDIT_CARD | 100% | 100% | ≥ 95% |
| US_SSN | 100% | 100% | ≥ 95% |
| SECRET | 100% | 100% | ≥ 95% |
| PERSON | 98.2% | 100% | ≥ 85% |

The scores above are on `data/pii_eval.jsonl`: 309 synthetic rows, including near-misses (order numbers, invalid-Luhn cards, invalid SSN ranges), code/log snippets and structured samples (JSON tool results, code comments, YAML).

**Redaction latency** (`scripts/bench_redaction.py`, M1 Pro, 4 worker threads). Phase 2.5 brought it down:

| Scenario | Phase 2 p50 / p95 | Phase 2.5 p50 / p95 |
|---|---|---|
| cold ~1k tokens (6 messages, ~20 entities) | 58 / 65 ms | **28 / 30 ms** |
| cold ~8k tokens (34 messages, ~127 entities) | 310 / 324 ms | **129 / 143 ms** |
| warm ~8k tokens (history seen before + one new ~200-token message) | 309 / 345 ms | **11 / 14 ms** |

What changed:
1. **A per-message findings cache.** The key is an HMAC of tenant, app, policy version and text, and the value holds offsets and types only. It lives in an in-process LRU and is never stored in Redis. Re-sent history is never re-analysed, and the result is byte-identical to a fresh analysis.
2. **Two passes.** Pattern recognizers run on the full text using a tokenizer-only pipeline. spaCy NER runs only on natural-language segments (it skips code fences, JSON, base64 and stack traces, but keeps prose-like JSON values and code comments), using `en_core_web_lg` reduced to its `ner` component: identical entities at ~45% of the cost.
3. **NER spread across the thread pool.** spaCy releases the GIL, so threads scale: 283 → 89 ms with 1 → 4 threads. A process pool gave under 10% more and was not adopted.
4. **Only actionable pattern recognizers run.** This cut the GIL-bound pattern pass from 72 to 40 ms.

`en_core_web_md` was evaluated and rejected: PERSON recall 93.7% vs 97.5%, and it was not faster.

## Demo

```bash
scripts/demo_phase1.sh   # tenancy, auth, SDK usage, isolation, revocation < 1 s
scripts/demo_phase2.sh   # eval gate, dry run, blocking, split-placeholder streaming, live policy change, leak scan
```

The script brings the stack up, seeds it, and then demonstrates the Phase 1 "done when" criteria:
- curl and SDK calls from two tenants, streaming and non-streaming;
- each app's `/v1/usage` shows only its own requests;
- cross-tenant admin probes and forged keys are refused;
- a revoked key returns 401 within 1 s (typically under 10 ms).

## Tests

```bash
uv sync                                   # installs dev deps (pytest, respx, fakeredis, openai)
uv run pytest tests/unit                  # no infrastructure needed
docker compose up -d && uv run pytest     # everything (needs postgres + redis + native Ollama)
uv run pytest -m "not ollama"             # everything except the real-model SDK test
```

**Test runs never touch your dev data.** Each pytest session:
- creates a throwaway database `tollgate_test_<random>` on the compose Postgres and migrates it, then drops it at the end;
- uses Redis DB 15 (dev uses DB 0) and flushes it before and after;
- starts its own gateway: a real uvicorn server, in-process on a random port.

Fixtures purge every tenant they create, and a teardown guard fails the run if row counts in the dev database changed.

| Suite | Needs | What it covers |
|---|---|---|
| `tests/unit` | nothing | One file per stage (authn, authz, ratelimit, route + invoke, meter, runner). Recognizers, vault and the redact/rehydrate stages. Streaming re-hydrator fuzz (every split point, char-by-char, 500 random chunkings). OpenAI SDK against the in-process app, with an echo model proving upstream sees only placeholders. Also the Ollama adapter (respx), security primitives, policy schema, cost math and the logging filter. |
| `tests/eval` | nothing | PII eval gate: recall targets per entity on `data/pii_eval.jsonl` |
| `tests/isolation` | compose stack | RLS bypass attempts as the app DB role. Cross-tenant reads and writes through the data and admin APIs. Forged keys. Redis key prefixing. |
| `tests/integration` | compose stack (+ Ollama) | OpenAI SDK against the real gateway from two tenants. Revocation in under 1 s. Key rotation, scopes, IP allowlist, suspension, audit, lockout, refresh rotation. Policy GET/PUT validation, versioning and immediate effect; redaction dry run; `pii_entities` on the ledger. |
| `tests/leak` | compose stack + Ollama | Runs **last**. Pushes canary PII through every path, then scans the session's gateway log, every Redis key and every Postgres table. Expects zero hits. |

The isolation and integration suites **fail rather than skip** when Postgres or Redis is down (spec §17: the isolation suite must never be skipped). They find the infrastructure through the usual `TOLLGATE_DATABASE_URL_*`, `TOLLGATE_REDIS_URL` and `TOLLGATE_OLLAMA_BASE_URL` variables.

## How it's put together

```
Client ──Bearer tg_live_…──▶ /v1/chat/completions
  authn      sys:auth:{prefix8} (Redis, 60 s) ─miss→ Postgres (system role) · constant-time SHA-256 compare
  authz      tenant/app active · IP allowlist · scope · allowed_models · max_input_tokens
  ratelimit  Lua token buckets t:{tenant}:a:{app}:rl:rpm|tpm (TPM pre-charged, reconciled after)
  budget                                            ← pass-through until Phase 4
  redact     Presidio + spaCy (thread pool) → tokenize/mask/block; vault in memory only
  cache · optimize · classify                       ← pass-through until Phases 3-4
  route      static: TOLLGATE_DEFAULT_MODEL (pinning per policy)
  invoke     Ollama /api/chat (stream pre-fetches first chunk so errors surface before any SSE byte)
  rehydrate  placeholders → values (JSON + SSE with 32-char tail buffer)
  meter      cost + baseline from catalog → Redis counters + XADD sys:usage_events
                                              └─▶ usage writer (1 s batches) ─▶ usage_events (partitioned)
```

**Isolation model**

- Postgres has three roles:
  - `tollgate` owns the schema and runs migrations.
  - `tollgate_app` is NOBYPASSRLS. Every tenant-scoped query runs as this role, inside a transaction that sets `app.tenant_id` via `set_config(..., true)`.
  - `tollgate_system` is BYPASSRLS. It is used only for API-key prefix lookup, admin login, the cross-tenant usage writer, and platform-admin listings.
- Every tenant-owned Redis key is prefixed `t:{tenant}:a:{app}:`. The only exceptions are `sys:` infrastructure keys that are never served to tenants (see [gateway/redis_keys.py](gateway/redis_keys.py)).

**Secrets in logs.** [gateway/observability/logging.py](gateway/observability/logging.py) drops prompt-like fields and scrubs `tg_…` keys, bearer tokens, `sk-…` and JWTs from every log line, including tracebacks. Validation errors never echo request input.

## Layout

The repo follows spec §13. Modules for later phases exist as documented stubs: the pass-through stages in [gateway/pipeline/](gateway/pipeline/), the [routing](gateway/routing/) package, the hosted-provider adapters in [providers/](gateway/providers/), and the admin budget and usage routers.

Model prices for hosted models in [config/models.yaml](config/models.yaml) are **placeholders** (spec §8.1). Fill in current provider pricing before relying on the savings numbers.
