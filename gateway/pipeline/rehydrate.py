"""Stage 13 — Rehydrate: put the caller's real values back into the response (spec §7.4).

This stage sits directly after `invoke` in the pre-response list. For streams, it
has to wrap `ctx.stream` before the first byte is sent.

* Non-streaming: re-hydrates each choice's `content` and the `arguments` of each
  tool call (JSON-escaped, so the arguments stay valid JSON).
* Streaming: wraps `ctx.stream`. Content deltas go through one `StreamRehydrator`;
  each tool call (by `index`) gets its own JSON-escaping rehydrator. Buffers are
  flushed into the chunk that carries `finish_reason`, or into a trailing chunk if
  the stream ends without one, so no text is ever lost or reordered.

Only placeholders in this request's vault are replaced; nothing else is touched.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from gateway.pipeline.context import RequestContext
from gateway.pipeline.stream_rehydrator import StreamRehydrator, rehydrate_text
from gateway.providers.base import ChatChunk, ChatResponse, ChunkChoice
from gateway.redaction.vault import Vault


def rehydrate_response(resp: ChatResponse, vault: Vault) -> ChatResponse:
    for choice in resp.choices:
        msg = choice.message
        if isinstance(msg.content, str):
            msg.content = rehydrate_text(msg.content, vault)
        for tc in msg.tool_calls or []:
            fn = tc.get("function") or {}
            if isinstance(fn.get("arguments"), str):
                fn["arguments"] = rehydrate_text(fn["arguments"], vault, json_escape=True)
    return resp


async def rehydrate_stream(stream: AsyncIterator[ChatChunk], vault: Vault) -> AsyncIterator[ChatChunk]:
    content: dict[int, StreamRehydrator] = {}  # per choice index
    tools: dict[tuple[int, int], StreamRehydrator] = {}  # per (choice, tool_call index)
    last: ChatChunk | None = None

    def flush_into(choice: ChunkChoice) -> None:
        if (rh := content.pop(choice.index, None)) is not None and (tail := rh.flush()):
            choice.delta["content"] = choice.delta.get("content", "") + tail
        for key in [k for k in tools if k[0] == choice.index]:
            if tail := tools.pop(key).flush():
                choice.delta.setdefault("tool_calls", []).append({"index": key[1], "function": {"arguments": tail}})

    async for chunk in stream:
        last = chunk
        for choice in chunk.choices:
            delta = choice.delta
            if isinstance(delta.get("content"), str):
                rh = content.setdefault(choice.index, StreamRehydrator(vault))
                delta["content"] = rh.feed(delta["content"])
            for tc in delta.get("tool_calls") or []:
                fn = tc.get("function") or {}
                if isinstance(fn.get("arguments"), str):
                    rh = tools.setdefault((choice.index, tc.get("index", 0)), StreamRehydrator(vault, json_escape=True))
                    fn["arguments"] = rh.feed(fn["arguments"])
            if choice.finish_reason is not None:
                flush_into(choice)
        yield chunk

    # Stream ended without a finish_reason for some choice: emit what's still buffered.
    leftovers = sorted({i for i in content} | {k[0] for k in tools})
    if leftovers and last is not None:
        choices = [ChunkChoice(index=i, delta={}) for i in leftovers]
        for c in choices:
            flush_into(c)
        yield ChatChunk(id=last.id, created=last.created, model=last.model, choices=[c for c in choices if c.delta])


class RehydrateStage:
    name = "rehydrate"

    async def process(self, ctx: RequestContext) -> RequestContext:
        if not ctx.vault:
            return ctx
        if ctx.stream is not None:
            ctx.stream = rehydrate_stream(ctx.stream, ctx.vault)
        elif ctx.response is not None:
            ctx.response = rehydrate_response(ctx.response, ctx.vault)
        return ctx
