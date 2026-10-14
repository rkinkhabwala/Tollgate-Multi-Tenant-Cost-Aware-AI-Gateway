"""Guard: no tracked file may contain a secret-*shaped* literal.

The redaction tests and the PII eval need strings in real credential formats. They
are fake, but secret scanners (e.g. GitHub push protection) block pushes containing
them. `scripts/secret_scrub.py` stores them with one escaped character, so the raw
text doesn't match while the decoded value is identical. This test keeps it that
way: if it fails, run `python scripts/secret_scrub.py --fix <file>`.
"""

import json

from gateway.settings import REPO_ROOT
from scripts.secret_scrub import find_secret_literals, matches, scrub_json_lines, scrub_python, scrub_shell, tracked_files


def test_no_tracked_file_contains_secret_shaped_literals() -> None:
    hits = find_secret_literals(tracked_files(REPO_ROOT))
    assert not hits, "secret-shaped literals (run scripts/secret_scrub.py --fix):\n" + "\n".join(f"{p}:{n} {pat}" for p, n, pat in hits)


def test_scrub_preserves_values() -> None:
    key = "AK" + "IA" + "IOSFODNN7EXAMPLE"  # built at runtime: the source stays clean
    src = f'x = "deploy {key} now"\n'
    out = scrub_python(src)
    assert not matches(out) and eval(compile(out, "t", "exec"), (ns := {})) is None and ns["x"] == f"deploy {key} now"
    line = json.dumps({"text": f"key {key}"})
    assert not matches(scrub_json_lines(line)) and json.loads(scrub_json_lines(line)) == json.loads(line)
    sh = scrub_shell(f"echo \"{key}\" '{key}'\n")
    assert not matches(sh) and '"AK""IA' in sh and "'AK''IA" in sh


def test_eval_set_still_contains_real_format_secrets_when_loaded() -> None:
    """The eval must still exercise real token formats after decoding."""
    from scripts.eval_pii import load_rows

    values = [e["value"] for r in load_rows() for e in r["entities"] if e["type"] == "SECRET"]
    assert sum(1 for v in values if matches(v)) >= 30
