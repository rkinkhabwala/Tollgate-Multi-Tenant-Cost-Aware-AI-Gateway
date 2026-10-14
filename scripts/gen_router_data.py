"""Generate the router's labeled prompt sets with a local model (spec §8.3, §16 Phase 3).

    uv run python -m scripts.gen_router_data generate   # → data/router_raw.jsonl (resumable)
    uv run python -m scripts.gen_router_data finalize   # → data/router_seed.jsonl + data/router_eval.jsonl

`generate` asks local `qwen2.5:7b` (temperature 0.9) to write realistic user prompts
for each of the 30 (task_type, complexity) cells. To keep the set diverse, it rotates
through 40 business domains and asks for 6 distinct prompts per call. The label is
the cell that was requested (synthetic labeling, spec §8.3: "label them ... with a
large model offline"). Output is appended to `router_raw.jsonl`, so an interrupted
run resumes where it stopped.

`finalize`:
1. embeds every prompt with `nomic-embed-text` (prefix `classification: `);
2. drops near-duplicates globally (cosine > 0.95, keeping the first);
3. per cell, holds out ~7 prompts for `router_eval.jsonl` (200 total), with the
   rest going to `router_seed.jsonl` (target ~1,000);
4. writes the seed embeddings to `router_seed.emb.npz`, keyed by the seed file's
   SHA-256, so the vector index can be rebuilt without Ollama.

Because step 2 is global, no seed/eval pair exceeds cosine 0.95.
`tests/eval/test_router_data.py` verifies that, along with balance and no exact
overlap. Nothing here is real user data.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import httpx
import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
RAW, SEED, EVAL, EMB = DATA / "router_raw.jsonl", DATA / "router_seed.jsonl", DATA / "router_eval.jsonl", DATA / "router_seed.emb.npz"
OLLAMA = "http://localhost:11434"
GEN_MODEL, EMBED_MODEL = "qwen2.5:7b", "nomic-embed-text"
PER_CALL, CALLS_PER_CELL, CONCURRENCY = 6, 8, 3
EVAL_TOTAL, NEAR_DUP = 200, 0.95

TASKS: dict[str, str] = {
    "chitchat": "casual conversation with an assistant: greetings, small talk, opinions, jokes, how-are-you. No real task, no provided document.",
    "classification": "asking the assistant to assign a label, category, sentiment, priority or intent to a piece of text that the prompt INCLUDES verbatim.",
    "extraction": "asking the assistant to pull specific fields, entities, dates, amounts or items out of a text that the prompt INCLUDES verbatim, often into a list/JSON/table.",
    "summarization": "asking the assistant to summarize or condense a text that the prompt INCLUDES verbatim (an email thread, article, meeting notes, report).",
    "rewrite": "asking the assistant to rephrase, translate, simplify, change the tone of, or fix grammar in a text that the prompt INCLUDES verbatim.",
    "qa_with_context": "providing a passage/document/policy excerpt verbatim and asking a question that must be answered from that passage.",
    "code": "asking the assistant to write, debug, refactor, review, or explain source code (may include code snippets or error messages).",
    "reasoning_math": "math word problems, calculations, logic puzzles, probability, proofs, or step-by-step quantitative reasoning.",
    "creative_long": "asking for creative writing: stories, poems, song lyrics, screenplays, speeches, product launch copy, world-building.",
    "agentic_tool_use": "asking an assistant that has tools (web search, calendar, email, databases, APIs, booking systems) to carry out a multi-step task or action on the user's behalf.",
}
COMPLEXITY: dict[str, str] = {
    "low": "short and simple (one sentence to a few lines); a single easy step; any included text is brief.",
    "medium": "moderately detailed; several requirements or a few steps; any included text is a paragraph or two.",
    "high": "long, demanding, expert-level; many constraints or steps; any included text is several paragraphs; requires careful, thorough work.",
}
DOMAINS = [
    "healthcare billing", "e-commerce returns", "human resources", "commercial contracts", "freight logistics",
    "K-12 education", "personal finance", "business travel", "video games", "cybersecurity incident response",
    "residential real estate", "B2B marketing", "factory maintenance", "climate science", "restaurant operations",
    "insurance claims", "software release management", "customer support", "academic research", "public transit",
    "veterinary clinics", "event planning", "agriculture", "banking compliance", "SaaS onboarding",
    "home renovation", "music production", "fitness coaching", "legal intake", "supply chain procurement",
    "nonprofit fundraising", "hotel management", "mobile app reviews", "energy utilities", "biotech lab work",
    "journalism", "automotive repair", "space exploration", "retail inventory", "telecom outages",
]


def cell_prompt(task: str, complexity: str, domain: str, n: int) -> str:
    return (
        f"Write {n} different, realistic messages that a user might send to an AI assistant.\n"
        f"Task type: {task} — {TASKS[task]}\n"
        f"Complexity: {complexity} — {COMPLEXITY[complexity]}\n"
        f"Domain/topic: {domain}.\n"
        "Make them varied in wording, length and persona. Use invented names and data only. "
        "Each message must clearly be this task type and this complexity. "
        'Return JSON: {"prompts": ["...", "..."]}'
    )


async def _gen_call(client: httpx.AsyncClient, task: str, cx: str, domain: str, seed: int) -> list[str]:
    r = await client.post(
        "/api/chat",
        json={
            "model": GEN_MODEL,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.9, "seed": seed, "num_predict": 3000},
            "messages": [{"role": "user", "content": cell_prompt(task, cx, domain, PER_CALL)}],
        },
    )
    r.raise_for_status()
    try:
        prompts = json.loads(r.json()["message"]["content"]).get("prompts", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    return [p.strip() for p in prompts if isinstance(p, str) and len(p.strip()) >= 8]


async def generate() -> None:
    done = Counter()
    if RAW.exists():
        for line in RAW.read_text().splitlines():
            r = json.loads(line)
            done[(r["task_type"], r["complexity"], r["call"])] += 1
    jobs = [
        (t, c, k)
        for t in TASKS
        for c in COMPLEXITY
        for k in range(CALLS_PER_CELL)
        if (t, c, k) not in done
    ]
    print(f"{len(jobs)} generation calls to run ({len(done)} already done)", flush=True)
    sem = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    async with httpx.AsyncClient(base_url=OLLAMA, timeout=600) as client:

        async def run(t: str, c: str, k: int) -> None:
            rng = random.Random(f"{t}/{c}/{k}")
            domain = DOMAINS[(list(TASKS).index(t) * 7 + list(COMPLEXITY).index(c) * 13 + k * 5) % len(DOMAINS)]
            async with sem:
                prompts = await _gen_call(client, t, c, domain, rng.randint(1, 10**6))
            async with lock:
                with RAW.open("a") as f:
                    for p in prompts:
                        f.write(json.dumps({"task_type": t, "complexity": c, "domain": domain, "call": k, "prompt": p}) + "\n")
                print(f"{t}/{c}/{k} {domain}: {len(prompts)}", flush=True)

        await asyncio.gather(*(run(*j) for j in jobs))


async def embed_all(texts: list[str], batch: int = 32) -> np.ndarray:
    out: list[list[float]] = []
    async with httpx.AsyncClient(base_url=OLLAMA, timeout=600) as client:
        for i in range(0, len(texts), batch):
            r = await client.post("/api/embed", json={"model": EMBED_MODEL, "input": [f"classification: {t}" for t in texts[i : i + batch]]})
            r.raise_for_status()
            out.extend(r.json()["embeddings"])
    v = np.asarray(out, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


async def finalize() -> None:
    rows = [json.loads(line) for line in RAW.read_text().splitlines()]
    # exact-duplicate removal first
    seen: set[str] = set()
    rows = [r for r in rows if not (r["prompt"].casefold() in seen or seen.add(r["prompt"].casefold()))]
    vecs = await embed_all([r["prompt"] for r in rows])
    keep: list[int] = []
    kept_vecs = np.zeros((0, vecs.shape[1]), dtype=np.float32)
    for i, v in enumerate(vecs):  # global near-duplicate removal
        if kept_vecs.shape[0] and float((kept_vecs @ v).max()) > NEAR_DUP:
            continue
        keep.append(i)
        kept_vecs = np.vstack([kept_vecs, v])
    by_cell: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i in keep:
        by_cell[(rows[i]["task_type"], rows[i]["complexity"])].append(i)
    rng = random.Random(2026)
    per_cell = EVAL_TOTAL // len(by_cell)
    extra = EVAL_TOTAL - per_cell * len(by_cell)
    eval_idx: list[int] = []
    seed_idx: list[int] = []
    for n, cell in enumerate(sorted(by_cell)):
        idx = by_cell[cell][:]
        rng.shuffle(idx)
        take = per_cell + (1 if n < extra else 0)
        eval_idx += idx[:take]
        seed_idx += idx[take:]

    def dump(path: Path, idx: list[int], prefix: str) -> None:
        with path.open("w") as f:
            for n, i in enumerate(sorted(idx)):
                r = rows[i]
                f.write(json.dumps({"id": f"{prefix}-{n + 1:04d}", "task_type": r["task_type"], "complexity": r["complexity"], "domain": r["domain"], "prompt": r["prompt"]}, ensure_ascii=False) + "\n")

    dump(SEED, seed_idx, "seed")
    dump(EVAL, eval_idx, "eval")
    seed_sorted = sorted(seed_idx)
    np.savez_compressed(EMB, sha256=hashlib.sha256(SEED.read_bytes()).hexdigest(), vectors=vecs[seed_sorted].astype(np.float16))
    print(f"raw {len(rows)} → kept {len(keep)} → seed {len(seed_idx)}, eval {len(eval_idx)}")
    print("seed per task:", dict(Counter(rows[i]["task_type"] for i in seed_idx)))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "generate"
    asyncio.run(generate() if cmd == "generate" else finalize())
