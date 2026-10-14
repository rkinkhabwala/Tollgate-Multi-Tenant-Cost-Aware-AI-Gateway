"""Fuzz: streaming re-hydration must equal non-streaming re-hydration for ANY chunking.

Texts mix in-vault placeholders, unknown (foreign) placeholders, mask placeholders,
and adversarial `<`/`<<`/`>>` noise. Each text is split
1. at every single boundary (all 2-way splits),
2. one character per chunk,
3. into random chunk sizes over 500 seeds,
and the concatenated stream output is compared with `rehydrate_text(full)`.
"""

import json
import random
import time

import pytest

from gateway.pipeline.rehydrate import rehydrate_stream
from gateway.pipeline.stream_rehydrator import StreamRehydrator, is_placeholder_prefix, rehydrate_text
from gateway.providers.base import ChatChunk, ChunkChoice
from gateway.redaction.vault import MAX_PLACEHOLDER_LEN, Vault

VALUES = ["jane@corp.example", "Jane Roe", "+1 415 555 0132", "<<not a placeholder>>", 'quote"back\\slash', "Zoë Ünïcode"]
NOISE = ["<", "<<", ">>", "<<<", "<<A", "<<EMAIL_", "<<EMAIL_1>", "a<b", "<<x_1>>", "<<CARD>>", "<<EMAIL_99>>", " ", "hi ", "\n"]


def make_vault() -> tuple[Vault, list[str]]:
    v = Vault()
    phs = [v.tokenize("EMAIL_ADDRESS", VALUES[0]), v.tokenize("PERSON", VALUES[1]), v.tokenize("PHONE_NUMBER", VALUES[2])]
    phs += [v.tokenize("PERSON", VALUES[3]), v.tokenize("DENY_TERM", VALUES[4]), v.tokenize("LOCATION", VALUES[5])]
    phs.append(v.tokenize("VERY_LONG_CUSTOM_NAME", "x"))  # 20-char label: longest allowed placeholder
    return v, phs


def random_text(rng: random.Random, phs: list[str]) -> str:
    parts = [rng.choice(phs) if rng.random() < 0.4 else rng.choice(NOISE) for _ in range(rng.randint(1, 25))]
    return "".join(parts)


def stream_out(text: str, cuts: list[int], vault: Vault, json_escape: bool = False) -> str:
    rh = StreamRehydrator(vault, json_escape=json_escape)
    out, prev = [], 0
    for c in cuts + [len(text)]:
        out.append(rh.feed(text[prev:c]))
        assert rh.pending <= MAX_PLACEHOLDER_LEN
        prev = c
    out.append(rh.flush())
    return "".join(out)


def test_longest_placeholder_fits_buffer() -> None:
    _, phs = make_vault()
    assert max(len(p) for p in phs) <= MAX_PLACEHOLDER_LEN


@pytest.mark.parametrize("prefix", ["<", "<<", "<<E", "<<EMAIL", "<<EMAIL_", "<<EMAIL_1", "<<EMAIL_12>"])
def test_prefix_detection(prefix: str) -> None:
    assert is_placeholder_prefix(prefix)


@pytest.mark.parametrize("s", ["<<<", "<<e", "<<EMAIL_1>>", "<a", "<<_1"])
def test_non_prefixes(s: str) -> None:
    assert not is_placeholder_prefix(s)


def test_every_two_way_split_and_char_by_char() -> None:
    vault, phs = make_vault()
    rng = random.Random(7)
    texts = [random_text(rng, phs) for _ in range(150)] + ["".join(phs), "<<" + phs[0], phs[0][:-1] + "x>>"]
    for text in texts:
        expected = rehydrate_text(text, vault)
        for i in range(len(text) + 1):
            assert stream_out(text, [i], vault) == expected, (text, i)
        assert stream_out(text, list(range(1, len(text))), vault) == expected


def test_random_chunkings_500_seeds() -> None:
    vault, phs = make_vault()
    for seed in range(500):
        rng = random.Random(seed)
        text = random_text(rng, phs)
        cuts = sorted(rng.sample(range(1, len(text)), k=min(len(text) - 1, rng.randint(0, 12)))) if len(text) > 1 else []
        assert stream_out(text, cuts, vault) == rehydrate_text(text, vault), (seed, text, cuts)
        assert stream_out(text, cuts, vault, json_escape=True) == rehydrate_text(text, vault, json_escape=True)


def test_split_placeholder_rehydrated() -> None:
    vault, phs = make_vault()
    assert stream_out(f"Hi {phs[1]}!", [5, 9], vault) == "Hi Jane Roe!"  # "<<PE" | "RSON" | "_1>>!"


async def test_chunk_level_wrapper_handles_content_and_tool_args() -> None:
    vault, phs = make_vault()
    content = f"Dear {phs[1]}, mail {phs[0]}."
    args = json.dumps({"to": phs[0], "note": phs[4]})

    async def upstream():
        for i in range(0, len(content), 3):
            yield ChatChunk(id="c", created=0, model="m", choices=[ChunkChoice(delta={"content": content[i : i + 3]})])
        for i in range(0, len(args), 4):
            yield ChatChunk(
                id="c", created=0, model="m", choices=[ChunkChoice(delta={"tool_calls": [{"index": 0, "function": {"arguments": args[i : i + 4]}}]})]
            )
        yield ChatChunk(id="c", created=0, model="m", choices=[ChunkChoice(delta={}, finish_reason="stop")])

    text, arg_text = "", ""
    async for chunk in rehydrate_stream(upstream(), vault):
        for ch in chunk.choices:
            text += ch.delta.get("content", "")
            for tc in ch.delta.get("tool_calls", []):
                arg_text += tc["function"]["arguments"]
    assert text == "Dear Jane Roe, mail jane@corp.example."
    assert json.loads(arg_text) == {"to": "jane@corp.example", "note": 'quote"back\\slash'}


async def test_stream_without_finish_reason_flushes_tail() -> None:
    vault, phs = make_vault()

    async def upstream():
        yield ChatChunk(id="c", created=0, model="m", choices=[ChunkChoice(delta={"content": "bye " + phs[1][:5]})])
        yield ChatChunk(id="c", created=0, model="m", choices=[ChunkChoice(delta={"content": phs[1][5:] + " <"})])

    out = ""
    async for c in rehydrate_stream(upstream(), vault):
        out += "".join(ch.delta.get("content", "") for ch in c.choices)
    assert out == "bye Jane Roe <"


def test_rehydrator_is_fast() -> None:
    vault, phs = make_vault()
    text = " ".join(phs * 200)
    t0 = time.perf_counter()
    stream_out(text, list(range(7, len(text), 7)), vault)
    assert time.perf_counter() - t0 < 1.0
