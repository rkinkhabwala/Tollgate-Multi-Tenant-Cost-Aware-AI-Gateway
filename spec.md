# Tollgate — Multi-Tenant, Cost-Aware AI Gateway

> Working name: **Tollgate**. A single OpenAI-compatible endpoint that sits between your apps and every LLM provider. It authenticates the calling app, strips sensitive data, decides which model is *good enough* for the request, caches what it can, enforces per-app budgets, and records exactly what every tenant spent — and saved.

---

## 1. Goals and non-goals

### Goals
1. **Cost savings as the primary product metric.** Every request records `actual_cost` and `baseline_cost` (what it would have cost on the app's default premium model). Savings must be provable per tenant, per app, per day.
2. **Privacy by default.** PII and secrets are removed from prompts *and* context (system prompt, chat history, RAG chunks, tool results) before anything leaves the gateway. Responses are re-hydrated so the calling app sees real values.
3. **Smart routing.** Classify the request (task type + complexity) and send it to the cheapest model tier that meets the app's quality policy, with automatic escalation on failure.
4. **Multi-tenancy.** Tenants own apps; apps own API keys, policies, budgets, and usage. No data, cache entry, or metric crosses a tenant boundary.
5. **Drop-in adoption.** Apps onboard by changing `base_url` and API key in any OpenAI SDK. No code changes otherwise.

### Non-goals (v1)
- Fine-tuning or hosting models (we call Ollama locally and hosted providers remotely).
- Image/audio generation. Text chat + embeddings only.
- A full billing/invoicing system (we produce the usage ledger; invoicing consumes it).

---

## 2. Tech stack

| Concern | Choice | Why |
|---|---|---|
| API server | Python 3.12, **FastAPI**, uvicorn, `httpx` (async) | Async I/O-bound proxy, rich AI ecosystem |
| Config DB | **PostgreSQL 16** with Row-Level Security | Tenants, apps, keys, policies, usage ledger |
| Hot state | **Redis 7** (+ RediSearch / Redis Stack for vectors) | Key cache, rate limits, budget counters, exact + semantic cache |
| PII detection | **Microsoft Presidio** (analyzer + spaCy `en_core_web_lg`) + custom regex recognizers | Mature, pluggable, runs locally |
| Local models | **Ollama** (`qwen2.5:7b` chat, `nomic-embed-text` embeddings, `qwen2.5:1.5b` optional classifier) | Free tier for cheap traffic; local embeddings for routing + semantic cache |
| Hosted models | Provider adapters: OpenAI, Anthropic (extensible) | Premium tiers |
| Migrations | Alembic | |
| Observability | OpenTelemetry traces, Prometheus metrics, Grafana dashboards, structured JSON logs | |
| Admin UI | React + Vite + Recharts (Phase 5) | Tenant/app onboarding and usage dashboards |
| Packaging | Docker Compose (gateway, postgres, redis-stack, ollama, prometheus, grafana, admin-ui) | |
| Testing | pytest, pytest-asyncio, respx (mock providers), k6 or Locust (load) | |

---

## 3. Architecture

### 3.1 Request pipeline

Every `/v1/chat/completions` call goes through these stages in order. Each stage is a class implementing `Stage.process(ctx) -> ctx` so stages can be reordered, toggled per app, and unit-tested in isolation.

```
Client (OpenAI SDK)
   │  Authorization: Bearer tg_live_xxx
   ▼
┌──────────────────────────────────────────────────────────────┐
│ 1. AuthN          resolve API key → tenant, app, key scopes  │
│ 2. AuthZ          scopes, allowed models, IP allowlist       │
│ 3. Rate limit     RPM / TPM token buckets (Redis)            │
│ 4. Budget check   soft limit → force cheaper tier;           │
│                   hard limit → 402                           │
│ 5. Redact         PII + secrets → placeholders; vault kept   │
│                   in request ctx only                        │
│ 6. Cache lookup   exact hash → semantic (tenant-scoped)      │
│        └─ hit ──────────────────────────────┐                │
│ 7. Optimize       trim history, cap max_tokens, dedupe ctx   │
│ 8. Classify       task_type + complexity + confidence        │
│ 9. Route          policy + classification → model tier       │
│10. Invoke         provider adapter, timeout, retry, fallback │
│11. Validate       empty / truncated / JSON-schema → escalate │
│12. Cache write    store redacted prompt + redacted response  │
│13. Rehydrate      placeholders → original values  ◄──────────┘
│14. Meter          usage event: tokens, cost, baseline, saved │
└──────────────────────────────────────────────────────────────┘
   ▼
Client receives a normal OpenAI-format response
(+ x-tollgate-* headers: model used, cache status, cost, saved)
```

### 3.2 Planes
- **Data plane** — `/v1/chat/completions`, `/v1/embeddings`, `/v1/models`. Authenticated by app API keys. Latency-sensitive.
- **Control plane** — `/admin/*`. Authenticated by admin JWTs (local users in v1, OIDC-ready). Onboarding, policies, keys, usage queries.

Both ship in one service in v1 but live in separate routers/modules so they can be split later.

### 3.3 Gateway overhead budget
Target **p50 < 40 ms, p95 < 120 ms** added latency for a non-cached request, excluding model time. Cache hits should return in **< 60 ms p95** end to end. The redaction and classification stages are the main costs; both run locally.

---

## 4. Multi-tenancy model

```
Tenant (an organization / business unit)
 ├── Users (admin console)  role: tenant_admin | tenant_viewer
 └── Apps (one per onboarded application)
      ├── API keys (many, rotatable, scoped)
      ├── Policy (routing, redaction, caching, allowed models)
      ├── Budget (monthly USD, soft %, hard %)
      └── Usage events (ledger)
```

Isolation rules (all must have tests):
- Every table except `providers`/`models` carries `tenant_id`. Postgres **RLS** policies enforce `tenant_id = current_setting('app.tenant_id')`; the gateway sets this per transaction.
- Every Redis key is prefixed `t:{tenant_id}:a:{app_id}:` — including cache entries and vector index tags. **Semantic cache search always filters by tenant_id and app_id** (default) or tenant_id (if the tenant opts in to cross-app sharing).
- Logs never contain raw prompts; only redacted text, and only when the app's `log_prompts` policy is on.
- A platform admin role (`platform_admin`) can see across tenants; nothing else can.

---

## 5. Data model (PostgreSQL)

```sql
CREATE TABLE tenants (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name          TEXT NOT NULL UNIQUE,
  status        TEXT NOT NULL DEFAULT 'active',   -- active | suspended
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE apps (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id     UUID NOT NULL REFERENCES tenants(id),
  name          TEXT NOT NULL,
  environment   TEXT NOT NULL DEFAULT 'prod',     -- dev | staging | prod
  baseline_model TEXT NOT NULL,                   -- used to compute savings
  status        TEXT NOT NULL DEFAULT 'active',
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (tenant_id, name, environment)
);

CREATE TABLE api_keys (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id     UUID NOT NULL REFERENCES tenants(id),
  app_id        UUID NOT NULL REFERENCES apps(id),
  prefix        TEXT NOT NULL UNIQUE,             -- first 8 chars, shown in UI
  secret_hash   BYTEA NOT NULL,                   -- SHA-256 of full key
  scopes        TEXT[] NOT NULL,                  -- chat:write, embeddings:write, usage:read
  ip_allowlist  CIDR[],
  expires_at    TIMESTAMPTZ,
  revoked_at    TIMESTAMPTZ,
  last_used_at  TIMESTAMPTZ,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE app_policies (
  app_id        UUID PRIMARY KEY REFERENCES apps(id),
  tenant_id     UUID NOT NULL REFERENCES tenants(id),
  policy        JSONB NOT NULL,                   -- see §9
  version       INT NOT NULL DEFAULT 1,
  updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE budgets (
  app_id        UUID PRIMARY KEY REFERENCES apps(id),
  tenant_id     UUID NOT NULL REFERENCES tenants(id),
  monthly_usd   NUMERIC(12,4) NOT NULL,
  soft_pct      INT NOT NULL DEFAULT 80,          -- downgrade routing at this %
  hard_pct      INT NOT NULL DEFAULT 100          -- reject at this %
);

CREATE TABLE models (                             -- global catalog, platform-managed
  id                TEXT PRIMARY KEY,             -- e.g. 'ollama/qwen2.5:7b'
  provider          TEXT NOT NULL,                -- ollama | openai | anthropic
  provider_model    TEXT NOT NULL,
  tier              TEXT NOT NULL,                -- local | small | medium | large
  input_usd_per_mtok  NUMERIC(10,4) NOT NULL,
  output_usd_per_mtok NUMERIC(10,4) NOT NULL,
  context_window    INT NOT NULL,
  supports_tools    BOOLEAN NOT NULL DEFAULT false,
  supports_json     BOOLEAN NOT NULL DEFAULT false,
  enabled           BOOLEAN NOT NULL DEFAULT true
);

CREATE TABLE usage_events (                       -- partitioned by month
  id              UUID NOT NULL DEFAULT gen_random_uuid(),
  ts              TIMESTAMPTZ NOT NULL DEFAULT now(),
  tenant_id       UUID NOT NULL,
  app_id          UUID NOT NULL,
  api_key_id      UUID NOT NULL,
  request_id      TEXT NOT NULL,
  requested_model TEXT,                           -- what the client asked for
  routed_model    TEXT,                           -- what we actually used
  route_reason    TEXT,                           -- e.g. 'classifier:extraction/low'
  escalations     INT NOT NULL DEFAULT 0,
  cache_status    TEXT NOT NULL,                  -- miss | exact | semantic
  input_tokens    INT NOT NULL DEFAULT 0,
  output_tokens   INT NOT NULL DEFAULT 0,
  actual_cost_usd   NUMERIC(12,6) NOT NULL,
  baseline_cost_usd NUMERIC(12,6) NOT NULL,
  saved_usd         NUMERIC(12,6) GENERATED ALWAYS AS (baseline_cost_usd - actual_cost_usd) STORED,
  pii_entities    JSONB,                          -- counts by type only, never values
  latency_ms      INT,
  gateway_ms      INT,
  status_code     INT NOT NULL,
  PRIMARY KEY (ts, id)
) PARTITION BY RANGE (ts);

CREATE TABLE audit_log (                          -- control-plane changes
  id          BIGSERIAL PRIMARY KEY,
  ts          TIMESTAMPTZ NOT NULL DEFAULT now(),
  tenant_id   UUID,
  actor       TEXT NOT NULL,
  action      TEXT NOT NULL,                      -- key.create, policy.update, ...
  target      TEXT NOT NULL,
  diff        JSONB
);
```

Add `usage_daily` as a materialized rollup (tenant, app, day, routed_model, cache_status → sums) refreshed every 5 minutes for dashboards.

---

## 6. Authentication and authorization

### 6.1 App API keys (data plane)
- Format: `tg_{env}_{prefix8}_{secret32}` — e.g. `tg_live_a1b2c3d4_<32 random base62 chars>`.
- Shown **once** at creation. Store only `SHA-256(full_key)`. (Keys are high-entropy, so a fast hash is correct here; bcrypt would just add latency.)
- Lookup: parse prefix → fetch row → constant-time compare hash. Cache the resolved `AuthContext{tenant_id, app_id, key_id, scopes, policy_version}` in Redis for 60 s; revocation deletes that entry immediately.
- Rotation: create new key, both valid until old one is revoked or expires.
- Update `last_used_at` asynchronously (batched), never on the hot path.

### 6.2 Authorization checks (in order)
1. Key not revoked/expired; tenant and app `active`.
2. Request IP within `ip_allowlist` if set.
3. Endpoint scope present (`chat:write` for chat, etc.).
4. Requested model (if pinned, not `auto`) is in the app policy's `allowed_models`.
5. Request features allowed by policy (tools, JSON mode, max context).

Failures return OpenAI-shaped errors: `401 invalid_api_key`, `403 permission_denied`, `402 budget_exceeded`, `429 rate_limit_exceeded`.

### 6.3 Admin users (control plane)
- v1: local users with Argon2id password hashes, short-lived JWT (15 min) + refresh token. Design the auth module behind an interface so OIDC (Okta/Entra/Google) can replace it.
- Roles: `platform_admin`, `tenant_admin` (manage apps, keys, policies, budgets in own tenant), `tenant_viewer` (read usage).
- Every control-plane mutation writes to `audit_log`.

---

## 7. Sensitive-data redaction

### 7.1 What gets scanned
Every text-bearing field the provider will see: `system` and all `messages[].content` (including history), tool/function results, and any `context`/RAG chunks. Not just the latest user turn — history and retrieved documents are where most leaks happen.

### 7.2 Detectors
- **Presidio built-ins:** PERSON, EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD (Luhn-validated), US_SSN, IBAN_CODE, IP_ADDRESS, LOCATION, DATE_TIME (configurable), US_BANK_NUMBER, US_DRIVER_LICENSE, US_PASSPORT, MEDICAL_LICENSE.
- **Custom recognizers:** API keys and tokens (AWS `AKIA…`, `sk-…`, GitHub `ghp_…`, Slack `xox…`, JWTs, PEM private keys), connection strings with passwords, high-entropy strings ≥ 32 chars adjacent to words like `key`, `secret`, `token`, `password`.
- **Tenant-defined:** per-app regex list and deny-list terms (e.g. internal project codenames, account-number formats).

### 7.3 Actions (per entity type, set in policy)
| Action | Effect |
|---|---|
| `tokenize` (default) | Replace with stable placeholder `<<EMAIL_1>>`; original kept in request vault; re-hydrated in response |
| `mask` | Replace with `<<EMAIL>>`; not re-hydrated |
| `block` | Reject the request with `400 sensitive_data_blocked` (use for secrets/credentials) |
| `allow` | Leave as is |

Default policy: secrets → `block`; government IDs and card numbers → `mask`; names, emails, phones, addresses → `tokenize`.

### 7.4 Placeholder vault and re-hydration
- Placeholders are numbered per request in order of first appearance, and the same value always maps to the same placeholder within the request (`john@acme.com` → `<<EMAIL_1>>` everywhere).
- The vault (`placeholder → original`) lives **only in the request context in memory**. Never written to Redis, Postgres, or logs.
- Add a system instruction when tokenizing: *"Text like <<TYPE_N>> is a placeholder for redacted data. Keep placeholders exactly as written when you need to refer to them."*
- Re-hydration replaces placeholders in the response text and in tool-call arguments.
- **Streaming:** placeholders can be split across SSE chunks (`<<EMA` … `IL_1>>`). Keep a small tail buffer: hold back any trailing text that could be the start of a placeholder (`<`, `<<`, `<<E…`) until the next chunk resolves it, then flush. Max held: 32 chars.

### 7.5 Why redaction also saves money
Because placeholders are normalized, two users asking "draft a follow-up to <<PERSON_1>> about invoice <<ID_1>>" produce the same redacted prompt. The cache stores the *redacted* response, and re-hydration fills in each caller's own values — so redaction raises cache hit rates instead of lowering them.

### 7.6 Local-tier option
Policy flag `redact_for_local: false` lets an app skip redaction when the request is routed to the local Ollama tier (data never leaves the host). Default is `true`.

---

## 8. Routing (the main cost lever)

### 8.1 Tiers
Defined in `config/models.yaml`, loaded into the `models` table. **Prices below are placeholders — fill in current provider pricing before go-live.**

```yaml
tiers:
  local:   [ollama/qwen2.5:7b]                       # $0, runs on your hardware
  small:   [anthropic/claude-haiku-5-5, openai/<small-model>]
  medium:  [anthropic/claude-sonnet-5-5, openai/<mid-model>]
  large:   [anthropic/claude-opus-5-5]
escalation_order: [local, small, medium, large]
```

Each tier lists models in preference order; the router picks the first healthy one (circuit breaker per model).

### 8.2 Decision flow
1. **Client pinned a model** (`"model": "anthropic/claude-opus-5-5"`) and policy `allow_pinning: true` → honor it (still metered, still redacted). If pinning is disallowed, treat as `auto`.
2. **`"model": "auto"`** (recommended) or a tier alias (`"tollgate/small"`) → continue.
3. **Hard constraints** filter candidate models: needs tools? JSON mode? context length (counted with tokenizer) > window? These eliminate tiers before any classification.
4. **Classifier** produces `{task_type, complexity, confidence}`:
   - `task_type ∈ {chitchat, classification, extraction, summarization, rewrite, qa_with_context, code, reasoning_math, creative_long, agentic_tool_use}`
   - `complexity ∈ {low, medium, high}`
5. **Routing table** (in policy, editable per app) maps `(task_type, complexity)` → minimum tier. Default:

| task_type | low | medium | high |
|---|---|---|---|
| chitchat, classification, extraction | local | local | small |
| summarization, rewrite, qa_with_context | local | small | medium |
| code | small | medium | large |
| reasoning_math, agentic_tool_use | small | medium | large |
| creative_long | small | medium | medium |

6. **Confidence guard:** if classifier confidence < `policy.min_confidence` (default 0.6), bump one tier.
7. **Budget pressure:** past the soft budget limit, cap at `policy.soft_limit_max_tier` (default `small`).
8. **App ceiling/floor:** clamp to `policy.min_tier` / `policy.max_tier`.

### 8.3 Classifier implementation (cheap and local)
Two layers, so the classifier itself never costs more than it saves:
- **Layer A — heuristic features (< 1 ms):** token count, presence of code fences/stack traces, math symbols, number of turns, tools requested, words like "step by step", "prove", "refactor", "summarize", "extract", "translate".
- **Layer B — embedding kNN (~10–20 ms):** embed the last user turn (redacted) with `nomic-embed-text` via Ollama; find k=7 nearest labeled examples in a Redis vector index of ~1,000 seed prompts (`data/router_seed.jsonl`, labeled `task_type` + `complexity`); majority vote → label, vote share → confidence.
- **Optional Layer C:** if A and B disagree and confidence is low, ask `qwen2.5:1.5b` with a strict JSON prompt. Off by default.
- Ship a script to grow the seed set from production: sample redacted prompts, label them (manually or with a large model offline), re-index.

### 8.4 Escalation and fallback
- **Fallback (availability):** provider timeout/5xx/429 → next model in the same tier → next tier up. Circuit breaker opens after 5 failures in 30 s, half-opens after 60 s.
- **Escalation (quality):** after a response, validate:
  - empty or refusal-only text,
  - `finish_reason == "length"` with output under requested max,
  - JSON mode requested and output fails `json.loads` / the provided schema,
  - tool call with invalid arguments.
  On failure, re-run one tier up (max `policy.max_escalations`, default 1). Log both attempts; bill both.
- Streaming requests skip quality escalation (can't take back sent tokens); they only get availability fallback before the first byte.

### 8.5 Learning loop (Phase 6)
Track escalation rate per `(task_type, complexity, tier)`. If a cell escalates > 15% of the time over 7 days, surface a recommendation in the admin UI to raise its minimum tier; if a cell routed to a higher tier could be shadow-tested cheaply, run a sampled shadow evaluation on the lower tier and report agreement.

---

## 9. App policy schema

```json
{
  "routing": {
    "mode": "auto",
    "allow_pinning": false,
    "allowed_models": ["ollama/qwen2.5:7b", "anthropic/claude-haiku-5-5", "anthropic/claude-sonnet-5-5"],
    "min_tier": "local",
    "max_tier": "medium",
    "min_confidence": 0.6,
    "max_escalations": 1,
    "soft_limit_max_tier": "small",
    "table_overrides": { "code": { "low": "small", "medium": "medium", "high": "medium" } }
  },
  "redaction": {
    "enabled": true,
    "redact_for_local": true,
    "entities": {
      "SECRET": "block", "CREDIT_CARD": "mask", "US_SSN": "mask",
      "PERSON": "tokenize", "EMAIL_ADDRESS": "tokenize", "PHONE_NUMBER": "tokenize"
    },
    "custom_patterns": [{ "name": "ACCOUNT_ID", "regex": "ACC-\\d{8}", "action": "tokenize" }],
    "score_threshold": 0.5
  },
  "cache": {
    "exact": true,
    "semantic": true,
    "semantic_threshold": 0.95,
    "ttl_seconds": 86400,
    "scope": "app",
    "skip_if_temperature_above": 0.7
  },
  "optimize": {
    "max_history_turns": 12,
    "summarize_history_over_tokens": 6000,
    "default_max_tokens": 1024,
    "dedupe_context_chunks": true
  },
  "limits": { "rpm": 600, "tpm": 200000, "max_input_tokens": 32000 },
  "logging": { "log_prompts": false, "retention_days": 30 }
}
```

Validate with Pydantic models; reject unknown keys; version every change.

---

## 10. Other cost levers

1. **Exact cache** — key = SHA-256 of normalized `(app_id, redacted messages, tools, response_format, temperature, top_p)`. Only for `temperature ≤ skip_if_temperature_above`.
2. **Semantic cache** — embed the redacted final user turn plus a hash of the system prompt; search Redis vector index filtered by tenant/app and identical system-prompt hash; cosine ≥ `semantic_threshold` → hit. Never semantic-cache requests with tools or multi-turn history longer than 2 turns (too context-dependent). Response header `x-tollgate-cache: semantic` so apps can opt out per request with `x-tollgate-cache-control: no-cache`.
3. **History trimming** — keep system + last N turns; above a token threshold, replace older turns with a summary generated by the local tier (cached per conversation id if the client sends `x-tollgate-conversation-id`).
4. **Context dedupe** — drop duplicate or near-duplicate RAG chunks (same hash or embedding cosine > 0.98).
5. **max_tokens defaulting** — if the client omits it, apply `default_max_tokens` instead of the provider maximum.
6. **Provider prompt caching** — for Anthropic, mark long static system prompts with cache-control breakpoints automatically when the system prompt exceeds the provider minimum; record cached-token discounts in cost.
7. **Budget-aware downgrade** — see §8.2 step 7.

---

## 11. Usage metering and savings

- Cost = `input_tokens × input_price + output_tokens × output_price` using the routed model's catalog row, minus provider cache discounts.
- Baseline = same token counts priced at the app's `baseline_model`. For cache hits, baseline uses the original request's tokens; actual = embedding cost only (≈ 0 on local).
- Write usage events asynchronously: append to a Redis Stream; a background worker batches inserts into Postgres every second. Real-time counters (`spend:month`, `tokens:minute`) update in Redis inline so budgets and rate limits are accurate immediately.
- Response headers on every call:
  - `x-tollgate-request-id`
  - `x-tollgate-model` (routed model)
  - `x-tollgate-route-reason`
  - `x-tollgate-cache` (`miss|exact|semantic`)
  - `x-tollgate-cost-usd`, `x-tollgate-saved-usd`
  - `x-tollgate-redacted` (count of entities, never values)

---

## 12. API surface

### Data plane (OpenAI-compatible)
| Method | Path | Notes |
|---|---|---|
| POST | `/v1/chat/completions` | Streaming + non-streaming; `model: "auto"` or tier alias or pinned |
| POST | `/v1/embeddings` | Routed to local `nomic-embed-text` by default |
| GET | `/v1/models` | Models + aliases this key may use |
| GET | `/v1/usage` | Requires `usage:read`; this app's usage for a date range |

### Control plane
| Method | Path | Role |
|---|---|---|
| POST | `/admin/auth/login`, `/admin/auth/refresh` | any |
| POST/GET | `/admin/tenants` | platform_admin |
| POST/GET/PATCH | `/admin/tenants/{tid}/apps` | tenant_admin |
| POST/GET | `/admin/apps/{aid}/keys`, `DELETE /admin/keys/{kid}` (revoke) | tenant_admin |
| GET/PUT | `/admin/apps/{aid}/policy` | tenant_admin |
| GET/PUT | `/admin/apps/{aid}/budget` | tenant_admin |
| GET | `/admin/usage?tenant_id&app_id&from&to&group_by=day,model,cache_status` | tenant_viewer+ |
| GET | `/admin/savings?tenant_id&app_id&from&to` | tenant_viewer+ |
| POST | `/admin/redaction/test` | tenant_admin — dry-run a text against the app's redaction policy |
| POST | `/admin/routing/test` | tenant_admin — show classification + chosen model for a sample request |
| GET | `/admin/audit` | tenant_admin |

### Onboarding flow (what an app team does)
1. Tenant admin creates app → picks baseline model → gets default policy.
2. Creates API key (shown once).
3. App changes two lines:
   ```python
   client = OpenAI(base_url="https://tollgate.internal/v1", api_key="tg_live_...")
   client.chat.completions.create(model="auto", messages=[...])
   ```
4. Usage and savings appear in the dashboard within 5 minutes.

---

## 13. Project layout

```
tollgate/
├── docker-compose.yml
├── config/
│   ├── models.yaml
│   └── default_policy.json
├── data/
│   ├── router_seed.jsonl          # labeled prompts for kNN classifier
│   └── pii_eval.jsonl             # labeled PII test set
├── gateway/
│   ├── main.py                    # FastAPI app factory
│   ├── settings.py
│   ├── api/
│   │   ├── v1_chat.py  v1_embeddings.py  v1_models.py  v1_usage.py
│   │   └── admin/ (auth, tenants, apps, keys, policies, budgets, usage, tools)
│   ├── pipeline/
│   │   ├── context.py             # RequestContext dataclass
│   │   ├── base.py                # Stage protocol, Pipeline runner
│   │   ├── authn.py  authz.py  ratelimit.py  budget.py
│   │   ├── redact.py  rehydrate.py  stream_rehydrator.py
│   │   ├── cache.py  optimize.py
│   │   ├── classify.py  route.py  invoke.py  validate.py
│   │   └── meter.py
│   ├── providers/
│   │   ├── base.py                # ProviderAdapter protocol (chat, stream, embed, count_tokens)
│   │   ├── ollama.py  openai.py  anthropic.py
│   │   └── circuit_breaker.py
│   ├── redaction/
│   │   ├── engine.py              # Presidio setup
│   │   ├── recognizers/ (secrets.py, custom_regex.py)
│   │   └── vault.py
│   ├── routing/
│   │   ├── features.py  knn.py  table.py
│   ├── policy/ (schema.py, loader.py)
│   ├── db/ (models.py, session.py, rls.py, migrations/)
│   ├── workers/ (usage_writer.py, rollups.py, seed_indexer.py)
│   └── observability/ (otel.py, metrics.py, logging.py)
├── admin-ui/                      # React + Vite (Phase 5)
├── tests/
│   ├── unit/  integration/  isolation/  eval/
│   └── load/ (k6 scripts)
└── scripts/ (seed_demo_tenants.py, label_prompts.py)
```

### Key interfaces

```python
@dataclass
class RequestContext:
    request_id: str
    auth: AuthContext                 # tenant_id, app_id, key_id, scopes
    policy: AppPolicy
    original: ChatRequest             # never logged
    working: ChatRequest              # redacted + optimized copy sent upstream
    vault: dict[str, str]             # placeholder -> original, in-memory only
    pii_counts: dict[str, int]
    classification: Classification | None
    route: RouteDecision | None       # model_id, tier, reason
    attempts: list[Attempt]           # each model call, tokens, cost, latency
    cache_status: Literal["miss", "exact", "semantic"]
    response: ChatResponse | None

class Stage(Protocol):
    name: str
    async def process(self, ctx: RequestContext) -> RequestContext: ...

class ProviderAdapter(Protocol):
    async def chat(self, model: str, req: ChatRequest) -> ChatResponse: ...
    def stream(self, model: str, req: ChatRequest) -> AsyncIterator[ChatChunk]: ...
    async def embed(self, model: str, texts: list[str]) -> list[list[float]]: ...
    def count_tokens(self, model: str, req: ChatRequest) -> int: ...
```

Provider adapters translate between the OpenAI request/response shape and each provider's native API, including tool calls and streaming events.

---

## 14. Observability

Prometheus metrics (all labeled by `tenant_id`, `app_id` where cardinality allows):
- `tollgate_requests_total{status, routed_tier, cache_status}`
- `tollgate_gateway_overhead_ms` histogram, per stage `tollgate_stage_ms{stage}`
- `tollgate_cost_usd_total`, `tollgate_saved_usd_total`
- `tollgate_escalations_total{from_tier, to_tier, reason}`
- `tollgate_pii_entities_total{entity_type, action}`
- `tollgate_provider_errors_total{provider, model, code}`, circuit breaker state gauge

Grafana dashboards: Savings overview, Routing mix, Cache hit rate, Gateway latency by stage, Provider health.

OpenTelemetry span per stage, propagated `traceparent` from clients when present.

---

## 15. Security checklist
- [ ] Raw prompts, vault contents, and API keys never appear in logs, traces, metrics labels, or error messages.
- [ ] RLS enabled and tested on every tenant-scoped table.
- [ ] Redis keys and vector searches always tenant-scoped (isolation test suite).
- [ ] Provider API keys loaded from env/secret store, never from DB in plaintext.
- [ ] Admin endpoints rate-limited; login lockout after repeated failures.
- [ ] Request body size limit (e.g. 2 MB) and `max_input_tokens` enforced before redaction to avoid CPU exhaustion.
- [ ] Prompt-injection note: redaction placeholders are not secrets; the vault is. A model told to "print all placeholders" can only print `<<EMAIL_1>>`. Re-hydration only replaces placeholders that exist in *this request's* vault.
- [ ] TLS terminated at ingress; HSTS on admin UI.

---

## 16. Build phases

Each phase ends with a runnable `docker compose up`, passing tests, and a short demo script.

### Phase 1 — Skeleton, tenancy, auth, passthrough
- Compose stack: gateway, postgres, redis-stack, ollama.
- Tables + Alembic migrations + RLS. Seed script creates 2 tenants × 2 apps.
- API key creation/revocation (admin API, no UI yet), AuthN/AuthZ stages.
- `/v1/chat/completions` non-streaming + streaming, routed to a single configured model (Ollama).
- Rate limiting (token bucket in Redis).
- Usage events via Redis Stream → Postgres worker; cost + baseline computed.
- **Done when:** two apps in different tenants call the gateway with the OpenAI SDK, each sees only its own `/v1/usage`, and revoked keys fail within 1 s.

### Phase 2 — Redaction + re-hydration
- Presidio engine + secret recognizers + custom regex per app.
- Tokenize / mask / block actions; vault; non-streaming and streaming re-hydration with tail buffer.
- `/admin/redaction/test` endpoint.
- **Done when:** PII eval set (`data/pii_eval.jsonl`, ≥ 200 labeled samples) shows ≥ 95% recall on EMAIL/PHONE/CARD/SSN/secrets and ≥ 85% on PERSON; a placeholder split across stream chunks is re-hydrated correctly; grep of all logs during the test run finds zero raw PII.

### Phase 3 — Classification + routing + fallback
- Model catalog from `models.yaml`; OpenAI and Anthropic adapters.
- Heuristic features + kNN classifier on seed set; routing table; policy clamps; pinning rules.
- Circuit breakers, availability fallback, quality validation + escalation.
- `/admin/routing/test` endpoint.
- **Done when:** on a held-out set of 200 labeled prompts, classifier agrees with labels ≥ 80% on task_type; routing a mixed demo workload produces a savings report showing actual vs baseline cost; killing a provider container triggers fallback with no client-visible errors.

### Phase 4 — Caching + budgets + optimization
- Exact and semantic caches (tenant/app-scoped), cache-control headers.
- History trimming/summarization, context dedupe, max_tokens defaulting, Anthropic prompt-cache markers.
- Monthly budgets with soft (downgrade) and hard (reject) limits.
- **Done when:** replaying a demo workload twice shows ≥ 30% cache hits on the second pass with correct per-caller re-hydration; an app crossing its soft limit is visibly downgraded in `x-tollgate-model`; isolation tests prove tenant A can never get a cache hit from tenant B's data.

### Phase 5 — Admin console + observability
- React admin UI: login, tenants, apps, keys (show-once modal), policy editor (JSON with schema validation + form view for common fields), budget settings, usage and savings dashboards (by day, model, cache status), routing and redaction test playgrounds, audit log.
- Prometheus metrics, Grafana dashboards, OTel tracing.
- **Done when:** a new team can be onboarded end to end from the UI in under 5 minutes and see their first request's cost and savings.

### Phase 6 — Hardening + learning loop
- Load test: 500 RPS non-cached mock-provider traffic, gateway overhead within §3.3 targets.
- Escalation-rate analytics and routing recommendations; seed-set growth script.
- Chaos tests (Redis down → fail open for cache, fail closed for auth if key not in local LRU; Postgres down → usage buffered in Redis Stream).
- **Done when:** load and chaos test reports are committed under `tests/load/reports/`.

---

## 17. Testing strategy
- **Unit:** each pipeline stage with fake context; redaction recognizers; stream re-hydrator with adversarial chunk splits; routing table + clamps; cost math.
- **Integration:** full pipeline with `respx`-mocked hosted providers and a real Ollama container.
- **Isolation suite (must never be skipped):** cross-tenant reads via API, RLS bypass attempts, cache and vector index cross-tenant probes, usage endpoint scoping.
- **Eval:** `pii_eval.jsonl` (precision/recall per entity) and `router_eval.jsonl` (classification accuracy + cost of routed vs baseline) run in CI; fail the build on regression > 2 points.
- **Load/chaos:** Phase 6.

---

## 18. Open decisions (defaults chosen; revisit as needed)
| Decision | Default | Alternative |
|---|---|---|
| Language/runtime | Python/FastAPI | Go or Java/Spring Boot for lower overhead at very high RPS |
| Semantic cache scope | per app | per tenant (opt-in) |
| Classifier | heuristics + kNN | small local LLM, or a fine-tuned DistilBERT |
| Usage store | Postgres partitions | ClickHouse once events exceed ~50M/month |
| Admin auth | local users + JWT | OIDC from day one |