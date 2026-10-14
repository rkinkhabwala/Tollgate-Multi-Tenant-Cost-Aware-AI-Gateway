"""Classifier Layer A: cheap heuristic features (< 1 ms) (spec §8.3).

`extract()` computes structural signals from the request and its (redacted) last
user turn:
* length (Qwen-tokenizer-free approximation);
* turns;
* code fences and stack traces;
* math symbols;
* tools requested;
* JSON mode;
* keyword cues.

`vote()` turns them into an optional task guess with a confidence, plus a
complexity hint.

Hard signals carry most weight:
* the request carries tools → `agentic_tool_use` (0.9);
* a code fence, stack trace or obvious code → `code` (0.8).

Keyword cues are softer (0.55–0.7). They mostly break ties when the kNN vote
(Layer B) is split.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from gateway.providers.base import ChatRequest

_CODE_FENCE = re.compile(r"```|~~~")
_STACK = re.compile(r"Traceback \(most recent call last\)|\bat [\w$.]+\([\w.]+:\d+\)|Exception in thread|Error: .+\n\s+at ")
_CODE_HINT = re.compile(
    r"\b(def |class |function\s*\(|=>|#include|import \w|SELECT .+ FROM|public static|console\.log|npm |pip install|stack trace|regex|SQL query|unit test|refactor|compile|bug in my)",
    re.IGNORECASE,
)
_MATH = re.compile(r"[=+\-*/^√∑∫%]|\b\d+(\.\d+)?\b")
_MATH_WORDS = re.compile(r"\b(solve|calculate|compute|probability|equation|prove|integral|derivative|how many|how much time|percent|average|ratio|mph|interest rate)\b", re.I)

# (task_type, pattern, confidence)
_CUES: list[tuple[str, re.Pattern[str], float]] = [
    ("summarization", re.compile(r"\b(summari[sz]e|summary|tl;?dr|condense|gist|key points|recap)\b", re.I), 0.7),
    ("extraction", re.compile(r"\b(extract|pull (out|all)|list (all|every)|parse .* into|as json|into a (table|list|csv)|line items)\b", re.I), 0.65),
    ("classification", re.compile(r"\b(classify|categori[sz]e|label (this|the|each)|sentiment|which (category|department)|tag (this|each|the)|assign a (priority|category|label))\b", re.I), 0.7),
    ("rewrite", re.compile(r"\b(rewrite|rephrase|paraphrase|proofread|fix (the )?grammar|make (this|it) (sound|more)|translate|simplify this|tone)\b", re.I), 0.65),
    ("qa_with_context", re.compile(r"\b(according to (the|this)|based on (the|this) (text|passage|document|excerpt|policy)|using only|from the (text|passage|document) (above|below))\b", re.I), 0.7),
    ("creative_long", re.compile(r"\b(write (me )?(a|an) (story|poem|song|haiku|limerick|screenplay|script|speech|novel)|lyrics|short story|world-?building|once upon)\b", re.I), 0.7),
    ("agentic_tool_use", re.compile(r"\b(book (a|me)|schedule (a|the)|send (an? )?(email|message|invite)|search (my|the web)|look up|add (a|an|it to) (calendar|reminder)|create a (ticket|calendar)|my inbox|order (me|a))\b", re.I), 0.6),
    ("chitchat", re.compile(r"^\s*(hi|hey|hello|yo|good (morning|evening)|how are you|what'?s up|thanks|thank you|lol|haha)\b", re.I), 0.6),
]
_CONSTRAINTS = re.compile(r"\b(must|at least|no more than|exactly|ensure|include|without|step[- ]by[- ]step|detailed|comprehensive|thorough)\b|^\s*\d+[.)]", re.I | re.M)


@dataclass(frozen=True)
class Features:
    approx_tokens: int  # last user turn, ~4 chars/token
    total_approx_tokens: int
    turns: int
    has_code_fence: bool
    has_stack_trace: bool
    code_hints: int
    math_density: float
    math_words: int
    tools_requested: bool
    json_requested: bool
    constraint_hits: int
    cue_hits: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def last_user_text(req: ChatRequest) -> str:
    for m in reversed(req.messages):
        if m.role == "user":
            return m.text()
    return req.messages[-1].text() if req.messages else ""


def extract(req: ChatRequest, text: str | None = None) -> Features:
    text = last_user_text(req) if text is None else text
    words = max(1, len(text.split()))
    cues: dict[str, float] = {}
    for task, pattern, conf in _CUES:
        if pattern.search(text):
            cues[task] = max(cues.get(task, 0.0), conf)
    tool_choice_none = req.tool_choice == "none"
    rf = (req.response_format or {}).get("type")
    return Features(
        approx_tokens=len(text) // 4 + 1,
        total_approx_tokens=sum(len(m.text()) for m in req.messages) // 4 + 1,
        turns=sum(1 for m in req.messages if m.role in ("user", "assistant")),
        has_code_fence=bool(_CODE_FENCE.search(text)),
        has_stack_trace=bool(_STACK.search(text)),
        code_hints=len(_CODE_HINT.findall(text)),
        math_density=len(_MATH.findall(text)) / words,
        math_words=len(_MATH_WORDS.findall(text)),
        tools_requested=bool(req.tools) and not tool_choice_none,
        json_requested=rf in ("json_object", "json_schema"),
        constraint_hits=len(_CONSTRAINTS.findall(text)),
        cue_hits=cues,
    )


@dataclass(frozen=True)
class HeuristicVote:
    task_type: str | None
    confidence: float
    complexity: str  # hint from length/constraints/turns
    hard: bool  # a structural signal (tools, code), not just a keyword


def complexity_hint(f: Features) -> str:
    score = 0
    score += 1 if f.approx_tokens > 60 else 0
    score += 1 if f.approx_tokens > 250 else 0
    score += 1 if f.approx_tokens > 900 else 0
    score += 1 if f.constraint_hits >= 3 else 0
    score += 1 if f.turns > 8 else 0
    return "low" if score == 0 else "medium" if score <= 2 else "high"


def vote(f: Features) -> HeuristicVote:
    cx = complexity_hint(f)
    if f.tools_requested:
        return HeuristicVote("agentic_tool_use", 0.9, cx, True)
    if f.has_code_fence or f.has_stack_trace or f.code_hints >= 2:
        return HeuristicVote("code", 0.8, cx, True)
    if f.math_words >= 1 and f.math_density > 0.15:
        return HeuristicVote("reasoning_math", 0.65, cx, False)
    if f.cue_hits:
        task, conf = max(f.cue_hits.items(), key=lambda kv: kv[1])
        return HeuristicVote(task, conf, cx, False)
    return HeuristicVote(None, 0.0, cx, False)
