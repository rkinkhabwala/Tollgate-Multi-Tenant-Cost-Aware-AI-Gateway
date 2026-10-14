"""Generate `data/pii_eval.jsonl`: a synthetic, labeled PII/secrets eval set (spec §16 Phase 2).

    uv run python -m scripts.gen_pii_eval      # rewrites data/pii_eval.jsonl (deterministic)

Everything is synthetic:
* names are invented first/last-name combinations;
* emails use reserved domains (`.example`, `.test`, `.internal`);
* phones are in the fictional 555-01xx range, or Ofcom's drama range for UK numbers;
* cards are the public test PANs, with near-misses failing Luhn;
* SSNs are random but well-formed;
* secrets are random strings in real token formats.

Each row is `{"id", "category", "text", "entities": [{"type", "value"}]}`. Labeled
values occur exactly once in their text, so spans can be recovered with `str.find`.

Categories cover every default entity type, mixed-entity sentences, near-misses
(order/tracking numbers, invalid-Luhn cards, invalid SSN ranges, ISBNs, versions,
hashes), and code/log snippets containing secrets.
"""

from __future__ import annotations

import base64
import json
import random
import string
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "pii_eval.jsonl"
rng = random.Random(2026)

FIRST = ["Maria", "Kenji", "Priya", "Tomasz", "Amara", "Liam", "Sofia", "Mateo", "Hannah", "Arjun", "Elena", "Noah",
         "Fatima", "Lucas", "Ingrid", "Diego", "Chloe", "Rahul", "Yuki", "Grace", "Omar", "Isabel", "Felix", "Nadia"]
LAST = ["Gonzalez", "Watanabe", "Raman", "Kowalczyk", "Okafor", "Brennan", "Lindqvist", "Herrera", "Schultz", "Mehta",
        "Petrova", "Fischer", "Haddad", "Moreau", "Andersen", "Castillo", "Dubois", "Iyer", "Tanaka", "Whitaker",
        "Rahman", "Navarro", "Hoffmann", "Sorensen"]
DOMAINS = ["northwind.example", "contoso.example", "fabrikam.test", "corp.internal", "mail.example.com", "example.org"]
CITIES = ["Seattle", "Lisbon", "Nairobi", "Toronto", "Osaka", "Denver", "Melbourne", "Dublin"]
TEST_CARDS = ["4111111111111111", "5555555555554444", "378282246310005", "6011111111111117", "4012888888881881",
              "5105105105105100", "4000056655665556", "2223003122003222"]

rows: list[dict] = []


def add(category: str, text: str, *entities: tuple[str, str]) -> None:
    for _, value in entities:
        assert text.count(value) == 1, (text, value)
    rows.append({"id": f"pii-{len(rows) + 1:03d}", "category": category, "text": text,
                 "entities": [{"type": t, "value": v} for t, v in entities]})


def name() -> str:
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def email(n: str | None = None) -> str:
    first, last = (n or name()).lower().split(" ")
    style = rng.choice([f"{first}.{last}", f"{first[0]}{last}", f"{first}_{last}{rng.randint(1, 99)}", f"{last}.{first}"])
    return f"{style}@{rng.choice(DOMAINS)}"


def phone() -> str:
    a, s = rng.choice(["415", "212", "312", "646", "503", "617"]), f"01{rng.randint(0, 99):02d}"
    return rng.choice([f"({a}) 555-{s}", f"{a}-555-{s}", f"{a}.555.{s}", f"+1 {a} 555 {s}", f"+1-{a}-555-{s}",
                       f"+44 20 7946 0{rng.randint(100, 999)}"])


def card(fmt: bool = True) -> str:
    pan = rng.choice(TEST_CARDS)
    if not fmt or len(pan) != 16:
        return pan
    sep = rng.choice([" ", "-", ""])
    return sep.join(pan[i : i + 4] for i in range(0, 16, 4))


def bad_luhn() -> str:
    pan = rng.choice([c for c in TEST_CARDS if len(c) == 16])
    last = str((int(pan[-1]) + rng.randint(1, 8)) % 10)
    pan = pan[:-1] + last
    return " ".join(pan[i : i + 4] for i in range(0, 16, 4))


def ssn() -> str:
    area = rng.choice([a for a in range(1, 900) if a != 666])
    return f"{area:03d}-{rng.randint(1, 99):02d}-{rng.randint(1, 9999):04d}"


def rand(chars: str, n: int) -> str:
    return "".join(rng.choice(chars) for _ in range(n))


B62 = string.ascii_letters + string.digits
UPD = string.ascii_uppercase + string.digits


def secret() -> str:
    kind = rng.randrange(9)
    if kind == 0:
        return "AKIA" + rand(UPD, 16)
    if kind == 1:
        return "sk-proj-" + rand(B62 + "_-", 40)
    if kind == 2:
        return "ghp_" + rand(B62, 36)
    if kind == 3:
        return "xoxb-" + "-".join([rand(string.digits, 12), rand(string.digits, 13), rand(B62, 24)])
    if kind == 4:
        def b64(d: dict) -> str:
            return base64.urlsafe_b64encode(json.dumps(d).encode()).decode().rstrip("=")
        return ".".join([b64({"alg": "HS256", "typ": "JWT"}), b64({"sub": rand(string.digits, 8), "exp": 1999999999}), rand(B62 + "_-", 43)])
    if kind == 5:
        return "sk_live_" + rand(B62, 28)
    if kind == 6:
        return "AIza" + rand(B62 + "_-", 35)
    if kind == 7:
        return "github_pat_" + rand(B62 + "_", 70)
    return "tg_live_" + rand(string.ascii_lowercase + string.digits, 8) + "_" + rand(B62, 32)


# --- emails -----------------------------------------------------------------------------
for _ in range(30):
    e = email()
    add("email", rng.choice([
        f"Please send the signed contract to {e} by Friday.",
        f"Reply-To: {e}",
        f"I can't log in with {e}, can you reset it?",
        f"Forwarding this thread to {e} for review.",
        f"Update the billing contact to {e} and confirm.",
    ]), ("EMAIL_ADDRESS", e))

# --- phones -------------------------------------------------------------------------------
for _ in range(26):
    p = phone()
    add("phone", rng.choice([
        f"You can reach the front desk at {p} after 9am.",
        f"Callback number: {p}",
        f"My new mobile is {p}, please update your records.",
        f"For urgent issues dial {p}.",
        f"Fax the form to {p} or upload it to the portal.",
    ]), ("PHONE_NUMBER", p))
for _ in range(6):
    p = f"{rng.choice(['415', '212', '646'])}555{rng.randint(100, 199):04d}"
    add("phone", rng.choice([f"call me at {p} tomorrow", f"Text {p} when you arrive.", f"phone {p} (cell)"]), ("PHONE_NUMBER", p))

# --- cards -------------------------------------------------------------------------------
for _ in range(25):
    c = card()
    add("card", rng.choice([
        f"Charge the order to card {c}, expiry 08/29.",
        f"The customer read out {c} over the phone.",
        f"payment_method: {c}",
        f"Refund the last payment on {c} please.",
    ]), ("CREDIT_CARD", c))

# --- SSNs --------------------------------------------------------------------------------
for _ in range(25):
    s = ssn()
    add("ssn", rng.choice([
        f"Applicant SSN: {s}",
        f"Her social security number is {s} per the W-2.",
        f"Verify {s} against the payroll record.",
        f"SSN on file {s}; do not share.",
    ]), ("US_SSN", s))

# --- persons -----------------------------------------------------------------------------
for _ in range(30):
    n = name()
    add("person", rng.choice([
        f"{n} approved the refund yesterday afternoon.",
        f"Please schedule a call with {n} next week.",
        f"The ticket was escalated by {n} from the finance team.",
        f"Dear {n}, thank you for your patience.",
        f"Our new account manager, {n}, starts Monday.",
        f"Interview notes: {n} has strong Python experience.",
    ]), ("PERSON", n))

# --- secrets in code / logs / configs ------------------------------------------------------
for _ in range(36):
    k = secret()
    add("secret", rng.choice([
        f'client = Client(api_key="{k}")',
        f"export SERVICE_TOKEN={k}",
        f"2026-10-14T10:22:31Z WARN auth failed for token={k} retrying",
        f"Here is my key, why is this failing? {k}",
        f"headers = {{'Authorization': 'Bearer {k}'}}",
        f"# .env\nAPI_KEY={k}\nDEBUG=false",
    ]), ("SECRET", k))
for _ in range(6):
    pw = rand(B62, 6) + rng.choice("!#%&*") + rand(B62, 4)
    add("secret", rng.choice([f'db_password = "{pw}"', f"login with password: {pw}", f"PASSWORD={pw}"]), ("SECRET", pw))
for _ in range(6):
    user, pw = rand(string.ascii_lowercase, 6), rand(B62, 14)
    conn = f"postgres://{user}:{pw}@db-{rng.randint(1, 9)}.corp.internal:5432/app"
    add("secret", f"DATABASE_URL={conn}", ("SECRET", conn))
for _ in range(4):
    tok = rand(B62 + "+/", 40)
    add("secret", f"aws_secret_access_key = {tok}", ("SECRET", tok))
pem_body = "\n".join(rand(B62 + "+/", 64) for _ in range(4))
pem = f"--\x2d--BEGIN RSA PRIVATE KEY-----\n{pem_body}\n-----END RSA PRIVATE KEY-----"
add("secret", f"key file contents:\n{pem}\n", ("SECRET", pem))

# --- mixed -------------------------------------------------------------------------------
for _ in range(30):
    n = name()
    e, p, c, s, city = email(n), phone(), card(), ssn(), rng.choice(CITIES)
    pick = rng.randrange(5)
    if pick == 0:
        add("mixed", f"{n} ({e}, {p}) asked for a callback.", ("PERSON", n), ("EMAIL_ADDRESS", e), ("PHONE_NUMBER", p))
    elif pick == 1:
        add("mixed", f"Customer {n} paid with {c}; receipt sent to {e}.", ("PERSON", n), ("CREDIT_CARD", c), ("EMAIL_ADDRESS", e))
    elif pick == 2:
        add("mixed", f"Employee record: {n}, SSN {s}, phone {p}.", ("PERSON", n), ("US_SSN", s), ("PHONE_NUMBER", p))
    elif pick == 3:
        add("mixed", f"{n} relocated to {city} last spring; new email {e}.", ("PERSON", n), ("LOCATION", city), ("EMAIL_ADDRESS", e))
    else:
        ip = f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
        add("mixed", f"Login for {e} from {ip} was blocked.", ("EMAIL_ADDRESS", e), ("IP_ADDRESS", ip))

# --- near-misses (nothing in the thresholded types should fire) --------------------------------------
near = [
    lambda: f"Order {rng.randint(4000000000, 4999999999)} shipped this morning.",
    lambda: f"Tracking number 1Z{rand(UPD, 6)}{rng.randint(10**9, 10**10 - 1)}",
    lambda: f"The card ending in {bad_luhn()} was declined as invalid.",
    lambda: f"Reference 000-{rng.randint(10, 99)}-{rng.randint(1000, 9999)} is a placeholder id.",
    lambda: f"Ticket 666-{rng.randint(10, 99)}-{rng.randint(1000, 9999)} closed.",
    lambda: f"ISBN 978-0-{rng.randint(100, 999)}-{rng.randint(10000, 99999)}-{rng.randint(0, 9)}",
    lambda: f"Upgraded to v{rng.randint(1, 9)}.{rng.randint(0, 30)}.{rng.randint(0, 20)} last night.",
    lambda: f"commit {rand('0123456789abcdef', 40)} fixed the flaky test",
    lambda: f"Please reset your password from the settings page.",
    lambda: f"The token budget is {rng.randint(1000, 9000)} tokens per request.",
    lambda: f"Part number {rng.randint(100, 999)}-{rng.randint(10, 99)}-{rng.randint(100, 999)} is back in stock.",
    lambda: f"Invoice INV-{rng.randint(2020, 2026)}-{rng.randint(1000, 9999)} is overdue.",
    lambda: f"The meeting is in room {rng.randint(100, 999)} at {rng.randint(1, 12)}pm.",
    lambda: f"request_id={rand('0123456789abcdef', 32)} status=200",
]
for i in range(42):
    add("near_miss", near[i % len(near)]())

# --- other default types (reported, not thresholded) -----------------------------------------------
for _ in range(6):
    ip = f"192.168.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
    add("ip", f"The server at {ip} stopped responding.", ("IP_ADDRESS", ip))
for _ in range(6):
    city = rng.choice(CITIES)
    add("location", f"Our support center in {city} opens at 8am.", ("LOCATION", city))


# --- structured: JSON tool results, code with comments, logs (Phase 2.5 NER segmenting) ----------
for _ in range(10):
    n, e = name(), None
    e = email(n)
    add("structured", json.dumps({"ticket": rng.randint(1000, 9999), "customer": n, "contact": e, "status": "open"}),
        ("PERSON", n), ("EMAIL_ADDRESS", e))
for _ in range(8):
    n = name()
    add("structured", f"```python\n# owner: {n}\ndef refund(order_id: int) -> bool:\n    return True\n```", ("PERSON", n))
for _ in range(6):
    e, k = email(), secret()
    add("structured", f"```yaml\nnotify: {e}\napi_key: {k}\nretries: 3\n```", ("EMAIL_ADDRESS", e), ("SECRET", k))
for _ in range(6):
    n = name()
    add("structured", f'Tool result: {{"results": [{{"name": "{n}", "role": "approver", "score": 0.93}}], "count": 1}}', ("PERSON", n))


def main() -> None:
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
