"""Keep secret-*shaped* test fixtures out of the repository's raw text.

Tollgate's redaction tests and PII eval need strings in real credential formats
(Stripe `sk_live_…`, Slack `xoxb-…`, GitHub `ghp_…`, AWS `AKIA…`, PEM private-key
headers, …) to prove they get blocked. They are fake, but secret scanners such as
GitHub push protection can't tell, and block the push.

This tool rewrites one character of each match into an *escape* that decodes to the
same character. The raw file no longer matches any scanner pattern, but the value the
program sees is unchanged:

| file type | encoding | proof the value is unchanged |
|---|---|---|
| `.py` | `\\xNN` inside the string literal | the module's AST is identical before/after |
| `.json`, `.jsonl` | `\\u00NN` inside the JSON string | every line `json.loads` to the same value |
| `.sh`, `.md` (shell snippets) | an empty quote pair (`AK""IA…` / `AK''IA…`), matching the surrounding quote style | the shell concatenates the pieces back |

    python scripts/secret_scrub.py --check            # list secret-shaped literals in tracked files
    python scripts/secret_scrub.py --fix FILE...      # scrub files in place
    python scripts/secret_scrub.py --fix-tree DIR     # scrub every file under DIR (used for history rewrites)

`tests/unit/test_no_secret_literals.py` runs `--check` logic so new ones can't slip in,
and `scripts/gen_pii_eval.py` scrubs the eval file as it writes it.

The pattern sources below don't match themselves (e.g. `AKIA[0-9A-Z]` is followed by
`[`), so this file is clean too.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

SECRET_SHAPES: list[re.Pattern[str]] = [
    re.compile(r"[sr]k_(?:live|test)_[A-Za-z0-9]{16,}"),  # Stripe
    re.compile(r"xox[abposr]-[A-Za-z0-9-]{10,}"),  # Slack
    re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),  # GitHub
    re.compile(r"github_pat_[A-Za-z0-9_]{50,}"),  # GitHub fine-grained
    re.compile(r"AIza[0-9A-Za-z_-]{35}"),  # Google
    re.compile(r"sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}"),  # OpenAI / Anthropic style
    re.compile(r"(?:AKIA|ASIA)[0-9A-Z]{16}"),  # AWS access key id
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),  # PEM private key
    re.compile(r"tg_(?:live|test)_[a-z0-9]{8}_[A-Za-z0-9]{32}"),  # Tollgate's own key format
]
SPLIT_AT = 2  # which character of a match gets encoded / split


class ScrubError(Exception):
    pass


def matches(text: str) -> list[re.Match[str]]:
    found = [m for p in SECRET_SHAPES for m in p.finditer(text)]
    return sorted(found, key=lambda m: m.start())


def _rewrite(text: str, encode: callable) -> str:  # type: ignore[valid-type]
    # Re-scan after every edit: an encoded character shifts offsets and may split overlaps.
    for _ in range(10_000):
        found = matches(text)
        if not found:
            return text
        m = found[0]
        i = m.start() + SPLIT_AT
        text = text[:i] + encode(text, i) + text[i + 1 :]
    raise ScrubError("scrub did not converge")


def scrub_python(src: str) -> str:
    new = _rewrite(src, lambda t, i: "\\x%02x" % ord(t[i]))
    if new != src and ast.dump(ast.parse(new)) != ast.dump(ast.parse(src)):
        raise ScrubError("python scrub would change program semantics (raw string or non-literal context?)")
    return new


def scrub_json_lines(src: str) -> str:
    new = _rewrite(src, lambda t, i: "\\u%04x" % ord(t[i]))
    old_lines, new_lines = src.splitlines(), new.splitlines()
    if len(old_lines) != len(new_lines):
        raise ScrubError("json scrub changed line count")
    for a, b in zip(old_lines, new_lines, strict=True):
        if a != b and json.loads(a) != json.loads(b):
            raise ScrubError("json scrub would change a value")
    return new


def _quote_state(line: str, upto: int) -> str | None:
    """Shell quote context ("'", '"' or None) at position `upto` of `line`."""
    state: str | None = None
    i = 0
    while i < upto:
        c = line[i]
        if c == "\\" and state != "'":
            i += 2
            continue
        if state is None and c in "'\"":
            state = c
        elif state == c:
            state = None
        i += 1
    return state


def scrub_shell(src: str) -> str:
    def encode(t: str, i: int) -> str:
        line_start = t.rfind("\n", 0, i) + 1
        q = _quote_state(t[line_start:], i - line_start)
        return ("''" if q == "'" else '""') + t[i]

    # Inserting an empty quote pair *before* char i splits the match; re-scan handles the rest.
    return _rewrite(src, encode)


def scrub_text(path: Path, src: str) -> str:
    suffix = path.suffix
    if suffix == ".py":
        return scrub_python(src)
    if suffix in (".json", ".jsonl"):
        return scrub_json_lines(src)
    if suffix in (".sh", ".md"):
        return scrub_shell(src)
    raise ScrubError(f"no scrub strategy for {path} — fix it by hand")


def scrub_file(path: Path) -> bool:
    try:
        src = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, IsADirectoryError):
        return False
    if not matches(src):
        return False
    new = scrub_text(path, src)
    path.write_text(new, encoding="utf-8")
    return True


def tracked_files(root: Path) -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True).stdout.decode()
    return [root / p for p in out.split("\0") if p]


def find_secret_literals(files: list[Path]) -> list[tuple[str, int, str]]:
    """(path, line number, pattern name) for every secret-shaped literal; never the value."""
    hits = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue
        for m in matches(text):
            hits.append((str(f), text.count("\n", 0, m.start()) + 1, m.re.pattern[:24]))
    return hits


def main(argv: list[str]) -> int:
    if not argv or argv[0] == "--check":
        root = Path.cwd()
        hits = find_secret_literals(tracked_files(root))
        for path, line, pattern in hits:
            print(f"{Path(path).relative_to(root)}:{line}: {pattern}")
        return 1 if hits else 0
    if argv[0] == "--fix":
        for p in argv[1:]:
            if scrub_file(Path(p)):
                print(f"scrubbed {p}")
        return 0
    if argv[0] == "--fix-tree":
        root = Path(argv[1])
        for f in sorted(root.rglob("*")):
            if f.is_file() and ".git" not in f.parts:
                try:
                    if scrub_file(f):
                        print(f"scrubbed {f.relative_to(root)}", file=sys.stderr)
                except ScrubError as exc:
                    print(f"ERROR {f.relative_to(root)}: {exc}", file=sys.stderr)
                    return 2
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
