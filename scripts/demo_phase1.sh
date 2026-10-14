#!/usr/bin/env bash
# Phase 1 demo — shows the spec §16 "Done when" criteria against the live stack:
#   1. two apps in different tenants call the gateway with the OpenAI SDK (and curl),
#   2. each sees only its own /v1/usage,
#   3. a revoked key fails within 1 s.
#
# Usage:  scripts/demo_phase1.sh        (from the repo root; needs docker + uv)
set -euo pipefail

BASE="${TOLLGATE_BASE_URL:-http://localhost:8000}"
ADMIN_EMAIL="${TOLLGATE_BOOTSTRAP_ADMIN_EMAIL:-admin@tollgate.local}"
ADMIN_PASSWORD="${TOLLGATE_BOOTSTRAP_ADMIN_PASSWORD:-admin-dev-password}"
cd "$(dirname "$0")/.."

say()  { printf '\n\033[1;36m▶ %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m✔ %s\033[0m\n' "$*"; }
fail() { printf '\033[1;31m✘ %s\033[0m\n' "$*"; exit 1; }
json() { python3 -c "import json,sys; d=json.load(sys.stdin); print(eval(sys.argv[1], {'d': d}))" "$1"; }

say "Starting the stack (docker compose up -d)"
docker compose up -d
printf 'waiting for gateway'
for _ in $(seq 1 90); do curl -sf "$BASE/healthz" >/dev/null && break; printf '.'; sleep 2; done; echo
curl -sf "$BASE/readyz" >/dev/null || fail "gateway not ready"
printf 'waiting for model pull (ollama-init)'
until [ "$(docker compose ps -a --format '{{.State}}' ollama-init)" = "exited" ]; do printf '.'; sleep 3; done; echo
[ "$(docker inspect -f '{{.State.ExitCode}}' "$(docker compose ps -aq ollama-init)")" = "0" ] || fail "model pull failed: docker compose logs ollama-init"
ok "stack is up: $(curl -s "$BASE/readyz")"

say "Seeding 2 tenants x 2 apps (keys are printed once)"
SEED=$(docker compose exec -T gateway python -m scripts.seed_demo_tenants --json)
echo "$SEED" | python3 -c "
import json,sys
for r in json.load(sys.stdin): print(f\"  {r['tenant']:<7} {r['app']:<16} {r['api_key'][:17]}…\")"
KEY_A=$(echo "$SEED" | json "[r for r in d if r['tenant']=='acme'][0]['api_key']")
APP_A=$(echo "$SEED" | json "[r for r in d if r['tenant']=='acme'][0]['app_id']")
KEY_B=$(echo "$SEED" | json "[r for r in d if r['tenant']=='globex'][0]['api_key']")
APP_B=$(echo "$SEED" | json "[r for r in d if r['tenant']=='globex'][0]['app_id']")

before_a=$(curl -s "$BASE/v1/usage" -H "Authorization: Bearer $KEY_A" | json "d['totals']['requests']")
before_b=$(curl -s "$BASE/v1/usage" -H "Authorization: Bearer $KEY_B" | json "d['totals']['requests']")

say "Tenant acme / support-bot: curl, non-streaming (first call loads the model; can take ~20 s on CPU)"
curl -s -D - "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $KEY_A" -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":20}' \
  | grep -iE '^x-tollgate|^\{' ; echo

say "Tenant acme / support-bot: curl, streaming (SSE)"
# (subshell + `|| true`: head closes the pipe early, which makes curl exit 23)
(curl -sN "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $KEY_A" -H 'Content-Type: application/json' \
  -d '{"model":"auto","stream":true,"messages":[{"role":"user","content":"Count from 1 to 5."}],"max_tokens":20}' \
  || true) | head -c 600; echo " …"

say "Tenant globex / sales-assistant: official OpenAI Python SDK (streaming + non-streaming)"
KEY_B="$KEY_B" BASE="$BASE" uv run --quiet python - <<'PY'
import os
from openai import OpenAI

client = OpenAI(base_url=os.environ["BASE"] + "/v1", api_key=os.environ["KEY_B"])
raw = client.chat.completions.with_raw_response.create(
    model="auto", messages=[{"role": "user", "content": "Name one planet."}], max_tokens=10
)
c = raw.parse()
print(f"  reply: {c.choices[0].message.content!r}  model={c.model}")
print(f"  x-tollgate-cost-usd={raw.headers['x-tollgate-cost-usd']}  x-tollgate-saved-usd={raw.headers['x-tollgate-saved-usd']}")
stream = client.chat.completions.create(
    model="auto", messages=[{"role": "user", "content": "Name one color."}], max_tokens=10, stream=True,
    stream_options={"include_usage": True},
)
text, usage = "", None
for chunk in stream:
    usage = chunk.usage or usage
    text += "".join(ch.delta.content or "" for ch in chunk.choices)
print(f"  streamed: {text!r}  usage={usage.prompt_tokens}+{usage.completion_tokens} tokens")
PY

say "Each app sees only its own /v1/usage (events land via Redis Stream → Postgres in ~1 s)"
sleep 2
UA=$(curl -s "$BASE/v1/usage" -H "Authorization: Bearer $KEY_A")
UB=$(curl -s "$BASE/v1/usage" -H "Authorization: Bearer $KEY_B")
echo "  acme   → app_id=$(echo "$UA" | json "d['app_id']")  totals=$(echo "$UA" | json "d['totals']")"
echo "  globex → app_id=$(echo "$UB" | json "d['app_id']")  totals=$(echo "$UB" | json "d['totals']")"
[ "$(echo "$UA" | json "d['app_id']")" = "$APP_A" ] || fail "acme usage scoped to wrong app"
[ "$(echo "$UB" | json "d['app_id']")" = "$APP_B" ] || fail "globex usage scoped to wrong app"
[ "$(echo "$UA" | json "d['totals']['requests']")" -eq $((before_a + 2)) ] || fail "acme should see exactly its 2 new requests"
[ "$(echo "$UB" | json "d['totals']['requests']")" -eq $((before_b + 2)) ] || fail "globex should see exactly its 2 new requests"
ok "acme sees +2 requests (its own), globex sees +2 (its own); neither sees the other's"

say "Cross-tenant probes are refused"
ACME_ADMIN=$(curl -s "$BASE/admin/auth/login" -H 'Content-Type: application/json' \
  -d '{"email":"admin@acme.example","password":"demo-tenant-password"}' | json "d['access_token']")
GLOBEX_TID=$(echo "$SEED" | json "[r for r in d if r['tenant']=='globex'][0]['tenant_id']")
code=$(curl -s -o /dev/null -w '%{http_code}' "$BASE/admin/tenants/$GLOBEX_TID/apps" -H "Authorization: Bearer $ACME_ADMIN")
echo "  acme admin lists globex apps      → HTTP $code"; [ "$code" = "403" ] || fail "expected 403"
code=$(curl -s -o /dev/null -w '%{http_code}' "$BASE/admin/apps/$APP_B/keys" -H "Authorization: Bearer $ACME_ADMIN")
echo "  acme admin lists globex app keys  → HTTP $code"; [ "$code" = "404" ] || fail "expected 404"
FORGED="tg_live_$(echo "$KEY_A" | cut -d_ -f3)_$(echo "$KEY_B" | cut -d_ -f4)"
code=$(curl -s -o /dev/null -w '%{http_code}' "$BASE/v1/usage" -H "Authorization: Bearer $FORGED")
echo "  acme prefix + globex secret       → HTTP $code"; [ "$code" = "401" ] || fail "expected 401"
ok "tenant boundaries hold"

say "Revocation: a revoked key fails within 1 s"
NEW=$(curl -s -X POST "$BASE/admin/apps/$APP_A/keys" -H "Authorization: Bearer $ACME_ADMIN" -H 'Content-Type: application/json' -d '{}')
NEW_KEY=$(echo "$NEW" | json "d['key']"); NEW_ID=$(echo "$NEW" | json "d['id']")
BASE="$BASE" NEW_KEY="$NEW_KEY" NEW_ID="$NEW_ID" ADMIN="$ACME_ADMIN" uv run --quiet python - <<'PY'
import os, sys, time, httpx

base, key, kid = os.environ["BASE"], os.environ["NEW_KEY"], os.environ["NEW_ID"]
c = httpx.Client(base_url=base)
auth = {"authorization": f"Bearer {key}"}
print(f"  before revoke: GET /v1/models → {c.get('/v1/models', headers=auth).status_code} (AuthContext now cached for 60 s)")
t0 = time.perf_counter()
c.delete(f"/admin/keys/{kid}", headers={"authorization": f"Bearer {os.environ['ADMIN']}"}).raise_for_status()
while True:
    status = c.get("/v1/models", headers=auth).status_code
    elapsed = time.perf_counter() - t0
    if status == 401 or elapsed > 1:
        break
print(f"  after revoke:  GET /v1/models → {status} after {elapsed * 1000:.0f} ms")
sys.exit(0 if status == 401 and elapsed < 1 else 1)
PY
ok "revoked key rejected in under 1 s"

say "Phase 1 done-when criteria demonstrated ✅"
