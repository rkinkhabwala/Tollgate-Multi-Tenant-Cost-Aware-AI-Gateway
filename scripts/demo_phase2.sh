#!/usr/bin/env bash
# Phase 2 demo — the spec §16 "Done when" criteria, against the live docker stack:
#   1. PII eval: ≥ 95% recall on EMAIL/PHONE/CARD/SSN/secrets and ≥ 85% on PERSON;
#   2. a placeholder split across stream chunks is re-hydrated correctly;
#   3. a scan of gateway logs (plus Redis and Postgres) finds zero raw PII.
# It also shows /admin/redaction/test, blocking, and a policy change applying immediately.
#
# Usage:  scripts/demo_phase2.sh      (repo root; needs docker, uv and a native Ollama)
set -euo pipefail

BASE="${TOLLGATE_BASE_URL:-http://localhost:8000}"
OLLAMA="${OLLAMA_HOST_URL:-http://localhost:11434}"
cd "$(dirname "$0")/.."

say()  { printf '\n\033[1;36m▶ %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m✔ %s\033[0m\n' "$*"; }
fail() { printf '\033[1;31m✘ %s\033[0m\n' "$*"; exit 1; }
json() { python3 -c "import json,sys; d=json.load(sys.stdin); print(eval(sys.argv[1], {'d': d}))" "$1"; }

# Demo PII (synthetic). The leak scan at the end looks for exactly these.
NAME="Rosalind Mercer"
EMAIL="rosalind.mercer@harbor-labs.example"
PHONE="415-555-0163"
SSN="529-38-4417"

say "Checking prerequisites"
curl -sf "$OLLAMA/api/version" >/dev/null || fail "native Ollama not reachable at $OLLAMA (run: ollama serve)"
docker compose up -d >/dev/null 2>&1
for _ in $(seq 1 90); do curl -sf "$BASE/readyz" >/dev/null && break; sleep 2; done
READY=$(curl -s "$BASE/readyz"); echo "  $READY"
[ "$(echo "$READY" | json "d['presidio']")" = "ok" ] || fail "redaction engine not ready"
SINCE=$(date -u +%Y-%m-%dT%H:%M:%SZ)

say "1) PII eval set (data/pii_eval.jsonl) — precision/recall per entity"
uv run --quiet python -m scripts.eval_pii 2>/dev/null || fail "eval below target"
ok "all recall targets met"

say "Seeding demo tenants"
SEED=$(docker compose exec -T gateway python -m scripts.seed_demo_tenants --json)
KEY=$(echo "$SEED" | json "[r for r in d if r['app']=='support-bot'][0]['api_key']")
APP=$(echo "$SEED" | json "[r for r in d if r['app']=='support-bot'][0]['app_id']")
ADMIN=$(curl -s "$BASE/admin/auth/login" -H 'Content-Type: application/json' \
  -d '{"email":"admin@acme.example","password":"demo-tenant-password"}' | json "d['access_token']")
# Start from the default policy so re-runs behave the same.
DEFAULT_POLICY=$(python3 -c "import json;print(json.dumps(json.load(open('config/default_policy.json'))))")
curl -sf -X PUT "$BASE/admin/apps/$APP/policy" -H "Authorization: Bearer $ADMIN" -H 'Content-Type: application/json' -d "$DEFAULT_POLICY" >/dev/null

say "POST /admin/redaction/test — dry run against the app's policy"
SAMPLE="Hi, I'm $NAME. Email $EMAIL or call $PHONE. SSN $SSN. Deploy key AK""IAIOSFODNN7EXAMPLE."
curl -s "$BASE/admin/redaction/test" -H "Authorization: Bearer $ADMIN" -H 'Content-Type: application/json' \
  -d "$(python3 -c 'import json,sys;print(json.dumps({"app_id":sys.argv[1],"text":sys.argv[2]}))' "$APP" "$SAMPLE")" \
  | python3 -c "
import json,sys; d=json.load(sys.stdin)
print('  redacted_text:', d['redacted_text'])
print('  blocked:', d['blocked'], ' counts:', d['counts'])
for e in d['entities']: print(f\"    {e['type']:<14} {e['action']:<9} {e['placeholder']}\")"

say "Secrets are blocked before anything leaves the gateway"
curl -s "$BASE/v1/chat/completions" -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"why does AK''IAIOSFODNN7EXAMPLE fail?"}]}' | sed 's/^/  /'; echo

say "2) Streaming: placeholders split across chunks are re-hydrated"
BASE="$BASE" OLLAMA="$OLLAMA" KEY="$KEY" NAME="$NAME" EMAIL="$EMAIL" PHONE="$PHONE" uv run --quiet python - <<'PY'
import json, os, sys, httpx

base, ollama, key = os.environ["BASE"], os.environ["OLLAMA"], os.environ["KEY"]
name, email, phone = os.environ["NAME"], os.environ["EMAIL"], os.environ["PHONE"]

# What the model actually emits for a placeholder: ask Ollama directly (bypassing the gateway).
pieces = []
with httpx.stream("POST", f"{ollama}/api/chat", timeout=120, json={
        "model": "qwen2.5:7b", "stream": True, "options": {"temperature": 0, "num_predict": 20},
        "messages": [{"role": "user", "content": "Repeat exactly, nothing else: <<EMAIL_1>>"}]}) as r:
    for line in r.iter_lines():
        if line and (c := json.loads(line).get("message", {}).get("content")):
            pieces.append(c)
print("  raw model chunks for a placeholder:", pieces)

# Same idea through the gateway: the client sends real values, upstream sees placeholders,
# and the client receives real values even though the model emits them in pieces.
prompt = f"Repeat exactly, nothing else: {name} <{email}> {phone}"
chunks, text = [], ""
with httpx.stream("POST", f"{base}/v1/chat/completions", timeout=120, headers={"authorization": f"Bearer {key}"},
        json={"model": "auto", "stream": True, "temperature": 0, "max_tokens": 60,
              "stream_options": {"include_usage": True}, "messages": [{"role": "user", "content": prompt}]}) as r:
    redacted_header = r.headers["x-tollgate-redacted"]
    for line in r.iter_lines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[6:])
        for ch in chunk.get("choices", []):
            if piece := ch.get("delta", {}).get("content"):
                chunks.append(piece); text += piece
        if "tollgate" in chunk:
            ext = chunk["tollgate"]
print(f"  gateway stream chunks ({len(chunks)}):", chunks[:12], "…" if len(chunks) > 12 else "")
print(f"  re-hydrated text: {text!r}")
print(f"  x-tollgate-redacted={redacted_header}  final-chunk tollgate={ext}")
leaked_placeholders = [c for c in chunks if "<<" in c or ">>" in c]
ok = email in text and not leaked_placeholders and int(redacted_header) >= 2
sys.exit(0 if ok else 1)
PY
ok "client received real values; no placeholder fragments reached the client"
echo "  (exhaustive check of every split point: uv run pytest tests/unit/test_stream_rehydrator_fuzz.py)"
uv run --quiet pytest -q tests/unit/test_stream_rehydrator_fuzz.py 2>/dev/null | tail -1 | sed 's/^/  /'

say "Policy change applies to the very next request"
red() { curl -s -D - -o /dev/null "$BASE/v1/chat/completions" -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d "{\"model\":\"auto\",\"max_tokens\":3,\"messages\":[{\"role\":\"user\",\"content\":\"Say ok. Contact: $EMAIL\"}]}" \
  | grep -i '^x-tollgate-redacted' | tr -d '\r' | awk '{print $2}'; }
echo "  before: x-tollgate-redacted=$(red)"
NEW=$(echo "$DEFAULT_POLICY" | python3 -c "import json,sys;p=json.load(sys.stdin);p['redaction']['entities']['EMAIL_ADDRESS']='allow';print(json.dumps(p))")
V=$(curl -s -X PUT "$BASE/admin/apps/$APP/policy" -H "Authorization: Bearer $ADMIN" -H 'Content-Type: application/json' -d "$NEW" | json "d['version']")
echo "  PUT policy (EMAIL_ADDRESS → allow) → version $V"
AFTER=$(red); echo "  after:  x-tollgate-redacted=$AFTER"
[ "$AFTER" = "0" ] || fail "policy change not applied immediately"
curl -sf -X PUT "$BASE/admin/apps/$APP/policy" -H "Authorization: Bearer $ADMIN" -H 'Content-Type: application/json' -d "$DEFAULT_POLICY" >/dev/null
ok "applied immediately (policy restored to defaults)"

say "3) Leak scan: gateway logs, Redis and Postgres must not contain the demo PII"
sleep 2
LOGS=$(docker compose logs --since "$SINCE" gateway 2>&1)
REDIS=$(docker compose exec -T redis sh -c 'redis-cli --scan | while read k; do t=$(redis-cli type "$k"); echo "$k"; case $t in string) redis-cli get "$k";; hash) redis-cli hgetall "$k";; stream) redis-cli xrange "$k" - +;; esac; done')
PG=$(docker compose exec -T postgres psql -U tollgate -tAc "SELECT row_to_json(u)::text FROM usage_events u WHERE ts > now() - interval '15 minutes'")
echo "  scanned: $(echo "$LOGS" | wc -l | tr -d ' ') log lines, $(echo "$REDIS" | wc -l | tr -d ' ') redis lines, $(echo "$PG" | wc -l | tr -d ' ') recent usage_events"
LEAKS=0
for v in "$NAME" "$EMAIL" "$PHONE" "$SSN" "AK""IAIOSFODNN7EXAMPLE"; do
  for src in LOGS REDIS PG; do
    if printf '%s' "${!src}" | grep -qiF -- "$v"; then echo "  LEAK: '$v' found in $src"; LEAKS=$((LEAKS + 1)); fi
  done
done
[ "$LEAKS" -eq 0 ] || fail "$LEAKS leak(s) found"
echo "  pii_entities recorded (counts only): $(echo "$PG" | grep -o '"pii_entities":{[^}][^}]*}' | tail -1)"
ok "zero raw PII in logs, Redis or Postgres"

say "Phase 2 done-when criteria demonstrated ✅"
