"""Generate the router's labeled prompt sets with a local model (spec §8.3, §16 Phase 3).

    uv run python -m scripts.gen_router_data generate   # → data/router_raw.jsonl (resumable)
    uv run python -m scripts.gen_router_data finalize   # → data/router_seed.jsonl + data/router_eval.jsonl

Generation uses local `qwen2.5:7b` with Ollama structured output (JSON schema), at
temperature 0.9, rotating through 40 business domains. Labels are correct by
construction, because the prompts are generated *for* a given (task_type,
complexity) cell:

* **Direct tasks** (chitchat, code, reasoning_math, creative_long, agentic_tool_use):
  the model writes realistic user messages for the cell, anchored by one example.
* **Text tasks** (classification, extraction, summarization, rewrite,
  qa_with_context): a 7B model is poor at writing "a request *about* an included
  text", so the work is split:
  1. the model writes realistic source texts (reviews, tickets, emails, notes, policy
     excerpts, …) at a length matching the complexity; for QA it also writes
     questions answerable only from the text;
  2. code wraps each text in a task instruction drawn from varied templates, with
     complexity-appropriate requirements (simple label vs multi-criteria scheme,
     one field vs nested JSON, TL;DR vs structured executive summary, …).
  Each source text is used exactly once, so prompts don't collapse into
  near-duplicates.

`finalize`:
1. embeds every prompt with `nomic-embed-text` (prefix `classification: `);
2. drops near-duplicates globally (cosine > 0.95, keeping the first);
3. holds out a balanced 200 for `router_eval.jsonl`, with the rest going to
   `router_seed.jsonl`;
4. writes the seed embeddings to `router_seed.emb.npz`, keyed by the seed file's
   SHA-256.

`tests/eval/test_router_data.py` checks balance, no overlap, and no near-duplicates
across the two sets. Everything is synthetic.
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
PER_CALL, CALLS_PER_CELL, CONCURRENCY = 8, 6, 4
EVAL_TOTAL, NEAR_DUP = 200, 0.95
CX = ("low", "medium", "high")

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

# --- direct tasks --------------------------------------------------------------------------
DIRECT: dict[str, tuple[str, str]] = {
    "chitchat": (
        "casual conversation with an assistant: greetings, small talk, opinions, jokes, feelings. No real task to perform.",
        "Hey! How's your day going? Got any fun plans for the weekend?",
    ),
    "code": (
        "asking the assistant to write, debug, refactor, review, or explain source code (may include code snippets or error messages).",
        "Write a Python function that merges two sorted lists into one sorted list without using sort().",
    ),
    "reasoning_math": (
        "math word problems, calculations, logic puzzles, probability, or step-by-step quantitative reasoning.",
        "A train leaves at 3pm going 60 mph; another leaves the same station at 4pm going 80 mph. When does the second catch up?",
    ),
    "creative_long": (
        "asking for creative writing: stories, poems, song lyrics, screenplays, speeches, launch copy, world-building.",
        "Write a short story about a lighthouse keeper who finds a message in a bottle addressed to her.",
    ),
    "agentic_tool_use": (
        "asking an assistant that has tools (web search, calendar, email, CRM, databases, APIs, booking systems) to carry out actions on the user's behalf.",
        "Search my inbox for last month's invoices from Acme, total them, and add a calendar reminder to pay by Friday.",
    ),
}
DIRECT_CX = {
    "low": "short and simple: one sentence, one easy step.",
    "medium": "a few sentences with several requirements or a couple of steps.",
    "high": "long and demanding (a full paragraph or more): many constraints, multiple steps, expert-level detail.",
}


def direct_prompt(task: str, cx: str, domain: str) -> str:
    desc, example = DIRECT[task]
    return (
        f"Write {PER_CALL} different, realistic messages that users send to an AI assistant.\n"
        f"Kind of message: {desc}\nExample (different domain): {example}\n"
        f"Complexity: {DIRECT_CX[cx]}\nDomain/topic: {domain}.\n"
        "Write each message exactly as the user would type it, first person. Vary wording, persona and length. "
        "Use invented names and data only. Return each message as a plain string."
    )


# --- text tasks ----------------------------------------------------------------------------------
TEXT_TASKS = ("classification", "extraction", "summarization", "rewrite", "qa_with_context")
KINDS = ["customer review", "support ticket", "internal email", "meeting notes", "news paragraph", "policy excerpt",
         "product description", "chat transcript", "incident report", "invoice or order note", "job posting", "forum post"]
TEXT_LEN = {"low": "1-2 sentences", "medium": "one paragraph of 4-6 sentences", "high": "3-4 substantial paragraphs"}


def text_prompt(kind: str, cx: str, domain: str, with_questions: bool) -> str:
    q = " For each text also write 2 questions that can only be answered from that text." if with_questions else ""
    return (
        f"Write {PER_CALL} different realistic {kind} texts from the domain of {domain}. "
        f"Each text is {TEXT_LEN[cx]} long, with concrete invented details (names, dates, amounts, items).{q}"
    )


INSTR: dict[str, dict[str, list[str]]] = {
    "classification": {
        "low": ["Classify the sentiment of this as positive, negative or neutral:", "Is this message a complaint, a question, or praise?",
                "Label this text as urgent or not urgent:", "Which department should handle this: billing, technical, or sales?"],
        "medium": ["Categorize the following into one of: billing, shipping, product quality, account access, other. Give the category and a one-line reason.",
                   "Assign a priority (P1-P4) and a topic label to the text below, and explain the priority in one sentence.",
                   "Classify the text below by intent (request, complaint, inquiry, feedback) and by sentiment.",
                   "Tag this message with all applicable labels from: refund, delay, damaged, wrong item, praise, escalation."],
        "high": ["Apply this taxonomy to the text below: assign a primary category, up to two secondary categories, a sentiment score from -1 to 1, a risk level with justification, and flag any compliance concerns. Return a JSON object.",
                 "Using the following multi-level scheme (domain → issue type → severity), classify each distinct issue mentioned in the text below, cite the sentence that supports each label, and note any ambiguities.",
                 "Perform a detailed classification of the document below: overall intent, per-paragraph topic labels, tone, and whether it requires legal review. Justify each decision briefly."],
    },
    "extraction": {
        "low": ["Pull out the name and the date from this:", "What is the dollar amount mentioned here? Just the number.",
                "Extract the order number from this text:", "List the people mentioned in this:"],
        "medium": ["Extract every date, amount and person from the text below into a JSON list of {type, value}.",
                   "From the text below, extract the customer name, product, issue and requested resolution as key: value lines.",
                   "Pull all action items and their owners from the following text into a table.",
                   "Extract the line items (item, quantity, price) from this text as CSV."],
        "high": ["Extract a complete structured record from the document below as nested JSON: parties, dates (ISO 8601), monetary amounts with currency, obligations with deadlines, and any conditions. Use null for missing fields and do not invent values.",
                 "From the text below, build a normalized dataset: every entity (person, organization, location, product), every event with timestamp, and the relationships between them. Output JSON with separate arrays and IDs linking them.",
                 "Extract all quantitative facts from the following report with their units and context, deduplicate them, flag inconsistencies between paragraphs, and return a JSON array sorted by section."],
    },
    "summarization": {
        "low": ["TL;DR this:", "Summarize in one sentence:", "Give me the gist of this in a few words:", "Short summary please:"],
        "medium": ["Summarize the following in three bullet points:", "Write a two-sentence summary of the text below for a busy manager.",
                   "Summarize this and list the key decisions:", "Condense the following into a short paragraph highlighting what changed."],
        "high": ["Write an executive summary of the document below: a one-paragraph overview, key findings as bullets, risks, open questions, and recommended next steps. Keep it under 250 words and preserve all figures exactly.",
                 "Produce a structured summary of the following for three audiences (executives, engineers, customers), each with its own section and tone, and note anything the original leaves ambiguous.",
                 "Summarize the text below section by section, then give an overall synthesis that reconciles any conflicting statements, and finish with a list of facts that must not be lost in the summary."],
    },
    "rewrite": {
        "low": ["Fix the grammar in this:", "Make this sound more polite:", "Rephrase this more simply:", "Translate this into Spanish:"],
        "medium": ["Rewrite the following in a formal, professional tone suitable for a client email:", "Rewrite this paragraph for a 10-year-old reader:",
                   "Make the text below more concise without losing any facts:", "Rewrite the following as a friendly customer-facing announcement:"],
        "high": ["Rewrite the document below as a polished client-facing letter: formal tone, active voice, under 300 words, keep every date and amount exactly, add a clear call to action, and remove internal jargon.",
                 "Rewrite the following text three ways — formal, conversational, and as a bulleted brief — keeping all facts identical, then note which details you had to rephrase most.",
                 "Translate the text below into French and German, preserving formatting and terminology, and adapt idioms naturally rather than literally."],
    },
    "qa_with_context": {
        "low": ["{q}\n\nText: {t}", "Based on this, {q_lower}\n\n{t}", "{t}\n\nQuestion: {q}"],
        "medium": ["Using only the text below, answer: {q} Quote the sentence that supports your answer.\n\n{t}",
                   "Read the following and answer the question. If the text does not say, reply 'not stated'.\n\nText:\n{t}\n\nQuestion: {q}",
                   "{t}\n\nAccording to the passage above, {q_lower} Explain briefly."],
        "high": ["Answer the questions below strictly from the provided document. For each answer cite the exact sentence, state your confidence, and say 'not in the document' where applicable.\n\nDocument:\n{t}\n\nQuestions:\n1. {q}\n2. {q2}",
                 "You are reviewing the following document. Answer both questions using only its contents, reconcile any contradictions you notice, and list assumptions you had to make.\n\n{t}\n\nQ1: {q}\nQ2: {q2}"],
    },
}

TEXTS_SCHEMA = {
    "type": "object",
    "properties": {"texts": {"type": "array", "items": {"type": "string"}, "minItems": PER_CALL, "maxItems": PER_CALL}},
    "required": ["texts"],
}
QA_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "questions": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
                },
                "required": ["text", "questions"],
            },
            "minItems": PER_CALL,
            "maxItems": PER_CALL,
        }
    },
    "required": ["items"],
}
PROMPTS_SCHEMA = {
    "type": "object",
    "properties": {"prompts": {"type": "array", "items": {"type": "string"}, "minItems": PER_CALL, "maxItems": PER_CALL}},
    "required": ["prompts"],
}


def compose(task: str, cx: str, text: str, questions: list[str], rng: random.Random) -> str:
    tmpl = rng.choice(INSTR[task][cx])
    if task == "qa_with_context":
        q = questions[0].strip()
        q2 = questions[1].strip() if len(questions) > 1 else q
        return tmpl.format(t=text.strip(), q=q, q2=q2, q_lower=q[:1].lower() + q[1:])
    sep = rng.choice(["\n\n", "\n\n---\n", "\n\nText:\n", " "]) if cx == "low" else rng.choice(["\n\n", "\n\n---\n", "\n\nText:\n"])
    return f"{tmpl}{sep}{text.strip()}"


async def _ollama_json(client: httpx.AsyncClient, prompt: str, schema: dict, seed: int, num_predict: int) -> dict:
    r = await client.post(
        "/api/chat",
        json={
            "model": GEN_MODEL,
            "stream": False,
            "format": schema,
            "options": {"temperature": 0.9, "seed": seed, "num_predict": num_predict, "num_ctx": 8192},
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    r.raise_for_status()
    try:
        return json.loads(r.json()["message"]["content"])
    except json.JSONDecodeError:
        return {}


async def run_job(client: httpx.AsyncClient, task: str, cx: str, k: int) -> list[dict]:
    rng = random.Random(f"{task}/{cx}/{k}")
    domain = DOMAINS[(sum(map(ord, task + cx)) + k * 7) % len(DOMAINS)]
    seed = rng.randint(1, 10**6)
    budget = {"low": 1200, "medium": 2500, "high": 5000}[cx]
    rows: list[dict] = []
    if task in DIRECT:
        out = await _ollama_json(client, direct_prompt(task, cx, domain), PROMPTS_SCHEMA, seed, budget)
        for p in out.get("prompts", []):
            if isinstance(p, str) and len(p.strip()) >= 8:
                rows.append({"prompt": p.strip()})
    else:
        kind = KINDS[(k + len(task)) % len(KINDS)]
        if task == "qa_with_context":
            out = await _ollama_json(client, text_prompt(kind, cx, domain, True), QA_SCHEMA, seed, budget + 800)
            items = [(i.get("text", ""), i.get("questions", [])) for i in out.get("items", []) if isinstance(i, dict)]
        else:
            out = await _ollama_json(client, text_prompt(kind, cx, domain, False), TEXTS_SCHEMA, seed, budget)
            items = [(t, []) for t in out.get("texts", []) if isinstance(t, str)]
        for text, qs in items:
            if len(text.strip()) < 20 or (task == "qa_with_context" and not qs):
                continue
            rows.append({"prompt": compose(task, cx, text, [q for q in qs if isinstance(q, str)], rng), "source_kind": kind})
    return [{"task_type": task, "complexity": cx, "domain": domain, "call": k, **r} for r in rows]


async def generate() -> None:
    done = set()
    if RAW.exists():
        for line in RAW.read_text().splitlines():
            r = json.loads(line)
            done.add((r["task_type"], r["complexity"], r["call"]))
    tasks = list(DIRECT) + list(TEXT_TASKS)
    jobs = [(t, c, k) for t in tasks for c in CX for k in range(CALLS_PER_CELL) if (t, c, k) not in done]
    print(f"{len(jobs)} generation calls to run ({len(done)} already done)", flush=True)
    sem, lock = asyncio.Semaphore(CONCURRENCY), asyncio.Lock()
    async with httpx.AsyncClient(base_url=OLLAMA, timeout=900) as client:

        async def run(t: str, c: str, k: int) -> None:
            async with sem:
                try:
                    rows = await run_job(client, t, c, k)
                except httpx.HTTPError as exc:
                    print(f"{t}/{c}/{k}: error {type(exc).__name__}", flush=True)
                    return
            async with lock:
                with RAW.open("a") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                print(f"{t}/{c}/{k}: {len(rows)}", flush=True)

        await asyncio.gather(*(run(*j) for j in jobs))


async def embed_all(texts: list[str], batch: int = 32) -> np.ndarray:
    out: list[list[float]] = []
    async with httpx.AsyncClient(base_url=OLLAMA, timeout=900) as client:
        for i in range(0, len(texts), batch):
            r = await client.post("/api/embed", json={"model": EMBED_MODEL, "input": [f"classification: {t}" for t in texts[i : i + batch]], "truncate": True})
            r.raise_for_status()
            out.extend(r.json()["embeddings"])
    v = np.asarray(out, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


async def finalize() -> None:
    rows = [json.loads(line) for line in RAW.read_text().splitlines()]
    seen: set[str] = set()
    rows = [r for r in rows if not (r["prompt"].casefold() in seen or seen.add(r["prompt"].casefold()))]
    vecs = await embed_all([r["prompt"] for r in rows])
    keep: list[int] = []
    kept = np.zeros((0, vecs.shape[1]), dtype=np.float32)
    for i, v in enumerate(vecs):  # global near-duplicate removal → no cross-set pair > 0.95
        if kept.shape[0] and float((kept @ v).max()) > NEAR_DUP:
            continue
        keep.append(i)
        kept = np.vstack([kept, v])
    by_cell: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i in keep:
        by_cell[(rows[i]["task_type"], rows[i]["complexity"])].append(i)
    rng = random.Random(2026)
    cells = sorted(by_cell)
    per_cell, extra = divmod(EVAL_TOTAL, len(cells))
    eval_idx: list[int] = []
    seed_idx: list[int] = []
    for n, cell in enumerate(cells):
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
    np.savez_compressed(EMB, sha256=hashlib.sha256(SEED.read_bytes()).hexdigest(), vectors=vecs[sorted(seed_idx)].astype(np.float16))
    print(f"raw {len(rows)} → kept {len(keep)} → seed {len(seed_idx)}, eval {len(eval_idx)}")
    print("seed per cell:", dict(sorted(Counter((rows[i]["task_type"], rows[i]["complexity"]) for i in seed_idx).items())))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "generate"
    asyncio.run(generate() if cmd == "generate" else finalize())
