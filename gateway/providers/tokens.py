"""Token counting for limits, context-window checks and TPM reservation (Phase 3).

Two counters, chosen by model:
* **Qwen (Ollama `qwen*` models): exact.** Uses Qwen2.5's own BPE tokenizer
  (`config/tokenizers/qwen2.5.tokenizer.json.gz`, loaded once via HF `tokenizers`),
  plus the chat-template framing (`<|im_start|>role\\n … <|im_end|>\\n`, ~5 tokens per
  message, 3 for the assistant primer) and the tool-schema preamble Qwen's template
  injects when tools are present.
* **Everything else (Anthropic, unknown): conservative.** Claude's tokenizer isn't
  public, and calling `count_tokens` per request would add a network round trip. The
  estimate is `ceil(utf8_bytes / 3 × 1.25)` + 8 per message + tool JSON bytes / 3.
  It over-counts English prose by roughly 50–65%. It stays at or above a real BPE
  count for indentation-heavy code (a 1.10 margin was measured to under-count Python
  vs Qwen's tokenizer) and for CJK (≈ 1 token per 3-byte character). Over-counting only means a context-window or
  `max_input_tokens` check triggers a bit early, never that a request overflows.

Before routing, the model isn't known yet, so authz and the rate limiter use the
conservative count. The router checks context windows per candidate model.
"""

from __future__ import annotations

import gzip
import json
import math
from functools import cached_property
from pathlib import Path

from tokenizers import Tokenizer

from gateway.catalog import CatalogModel
from gateway.providers.base import ChatRequest

QWEN_MSG_OVERHEAD = 5
QWEN_PRIMER = 3
QWEN_TOOLS_PREAMBLE = 120
CONSERVATIVE_BYTES_PER_TOKEN = 3
CONSERVATIVE_MARGIN = 1.25
CONSERVATIVE_MSG_OVERHEAD = 8


def _message_texts(req: ChatRequest) -> list[str]:
    out = []
    for m in req.messages:
        out.append(m.text())
        if m.tool_calls:
            out.append(json.dumps(m.tool_calls))
    return out


def conservative_count(req: ChatRequest) -> int:
    body = sum(len(t.encode("utf-8")) for t in _message_texts(req))
    tokens = math.ceil(body / CONSERVATIVE_BYTES_PER_TOKEN * CONSERVATIVE_MARGIN)
    tokens += CONSERVATIVE_MSG_OVERHEAD * len(req.messages)
    if req.tools:
        tokens += math.ceil(len(json.dumps(req.tools).encode("utf-8")) / CONSERVATIVE_BYTES_PER_TOKEN)
    return tokens + 16


class TokenCounter:
    def __init__(self, tokenizer_dir: Path) -> None:
        self._dir = tokenizer_dir

    @cached_property
    def _qwen(self) -> Tokenizer:
        raw = gzip.decompress((self._dir / "qwen2.5.tokenizer.json.gz").read_bytes()).decode("utf-8")
        return Tokenizer.from_str(raw)

    def qwen_count(self, req: ChatRequest) -> int:
        texts = _message_texts(req)
        n = sum(len(enc.ids) for enc in self._qwen.encode_batch(texts)) if texts else 0
        n += QWEN_MSG_OVERHEAD * len(req.messages) + QWEN_PRIMER
        if req.tools:
            n += len(self._qwen.encode(json.dumps(req.tools)).ids) + QWEN_TOOLS_PREAMBLE
        return n

    def count(self, model: CatalogModel | None, req: ChatRequest) -> int:
        """Input tokens `req` would consume on `model` (conservative if unsure)."""
        if model is not None and model.provider == "ollama" and model.provider_model.startswith("qwen"):
            return self.qwen_count(req)
        return conservative_count(req)
