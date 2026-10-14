"""Benchmark the redact stage on typical ~1k- and ~8k-token chat requests (spec §3.3).

    uv run python -m scripts.bench_redaction [--runs 50]

Requests are synthetic multi-turn support conversations: a system prompt plus
alternating user/assistant turns of business prose, with PII sprinkled in at roughly
one entity per ~250 characters (names, emails, phones, cards, SSNs). Each run times
`RedactStage.process()` end to end: the thread-pool hop, Presidio/spaCy analysis of
every message, and action application. Results are p50/p95/max in milliseconds.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import time

from gateway.pipeline.context import RequestContext
from gateway.pipeline.redact import RedactStage
from gateway.policy.schema import AppPolicy
from gateway.providers.base import ChatMessage, ChatRequest, estimate_tokens
from gateway.redaction.engine import RedactionEngine

PROSE = [
    "Thanks for getting back to me so quickly about the renewal.",
    "We reviewed the invoice and the line items for the March billing period look correct.",
    "The deployment to the staging cluster finished without errors, but latency rose slightly.",
    "Could you summarise the open action items from yesterday's planning meeting?",
    "Our team would like to move the quarterly review to the second week of next month.",
    "The customer reported that the export feature times out for reports larger than ten pages.",
    "Please double-check the shipping address before we confirm the replacement order.",
    "I have attached the updated onboarding checklist with the security training steps.",
]
PII = [
    lambda r: f"{r.choice(['Maria Gonzalez', 'Kenji Watanabe', 'Priya Raman', 'Liam Brennan'])} will follow up.",
    lambda r: f"Contact me at {r.choice(['ops', 'billing', 'jane.roe'])}{r.randint(1, 99)}@northwind.example.",
    lambda r: f"My number is 415-555-01{r.randint(10, 99)}.",
    lambda r: "The card on file is 4111 1111 1111 1111.",
    lambda r: f"SSN for the payroll form: {r.randint(100, 599)}-{r.randint(10, 99)}-{r.randint(1000, 9999)}.",
]


def build_request(target_tokens: int, seed: int) -> ChatRequest:
    rng = random.Random(seed)
    messages = [ChatMessage(role="system", content="You are a helpful support assistant for Northwind Traders.")]
    while True:
        turn = []
        while sum(len(s) for s in turn) < 900:
            turn.append(rng.choice(PII)(rng) if rng.random() < 0.28 else rng.choice(PROSE))
        messages.append(ChatMessage(role="user" if len(messages) % 2 else "assistant", content=" ".join(turn)))
        req = ChatRequest(messages=messages)
        if estimate_tokens(req) >= target_tokens:
            return req


def pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(p * (len(ordered) - 1)))]


async def bench(engine: RedactionEngine, target: int, runs: int) -> dict[str, float]:
    stage = RedactStage(engine)
    policy = AppPolicy()
    samples: list[float] = []
    entities = 0
    for i in range(runs + 3):  # 3 warm-up runs
        req = build_request(target, seed=i)
        ctx = RequestContext(request_id="bench", endpoint="chat", required_scope=None, client_ip=None)
        ctx.original, ctx.policy = req, policy
        t0 = time.perf_counter()
        ctx = await stage.process(ctx)
        elapsed = (time.perf_counter() - t0) * 1000
        if i >= 3:
            samples.append(elapsed)
            entities += sum(ctx.pii_counts.values())
    return {
        "tokens": estimate_tokens(build_request(target, 0)),
        "messages": len(build_request(target, 0).messages),
        "entities_avg": entities / runs,
        "p50": pct(samples, 0.50),
        "p95": pct(samples, 0.95),
        "max": max(samples),
        "mean": statistics.fmean(samples),
    }


async def main_async(runs: int) -> None:
    engine = RedactionEngine()
    print(f"redact stage latency, {runs} runs each (after 3 warm-up runs)\n")
    print(f"{'request':<10}{'~tokens':>9}{'msgs':>6}{'PII/req':>9}{'p50 ms':>9}{'p95 ms':>9}{'max ms':>9}")
    for target in (1000, 8000):
        r = await bench(engine, target, runs)
        print(f"{f'{target // 1000}k':<10}{r['tokens']:>9.0f}{r['messages']:>6.0f}{r['entities_avg']:>9.1f}{r['p50']:>9.1f}{r['p95']:>9.1f}{r['max']:>9.1f}")
    engine.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=50)
    asyncio.run(main_async(parser.parse_args().runs))


if __name__ == "__main__":
    main()
