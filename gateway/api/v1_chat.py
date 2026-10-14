"""`POST /v1/chat/completions` — OpenAI-compatible, streaming and non-streaming (spec §12).

The router only parses and validates the body, builds a `RequestContext`, runs the
chat pipeline, and serialises the result. All policy lives in the stages.

* Non-streaming returns a `chat.completion` JSON body with the `x-tollgate-*` headers.
* Streaming returns `text/event-stream`: `data: {chunk}` lines, then
  `data: [DONE]`. If the client sets `stream_options.include_usage`, a final chunk
  with empty `choices` carries `usage`, as OpenAI does. Post stages (meter) run
  once the stream has drained, or when the client disconnects.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import ValidationError

from gateway.api.deps import Services, get_services, new_context, tollgate_headers
from gateway.errors import GatewayError, bad_request
from gateway.pipeline.context import RequestContext
from gateway.providers.base import ChatChunk, ChatRequest, ProviderError

router = APIRouter()
log = logging.getLogger(__name__)

# Strong references to in-flight post-stream metering tasks (avoid GC mid-flight).
_background: set[asyncio.Task[RequestContext]] = set()


def validation_message(exc: ValidationError) -> str:
    """Describe what is wrong *without* echoing the input (it may contain prompts)."""
    parts = []
    for err in exc.errors(include_input=False, include_url=False)[:5]:
        loc = ".".join(str(p) for p in err["loc"]) or "body"
        parts.append(f"{loc}: {err['msg']}")
    return "Invalid request: " + "; ".join(parts)


async def parse_chat_request(request: Request) -> ChatRequest:
    try:
        raw = json.loads(await request.body())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise bad_request("Request body must be valid JSON.") from exc
    try:
        return ChatRequest.model_validate(raw)
    except ValidationError as exc:
        raise bad_request(validation_message(exc)) from exc


@router.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    svc = get_services(request)
    ctx = new_context(request, svc.settings, "chat", "chat:write")
    ctx.original = await parse_chat_request(request)
    try:
        ctx = await svc.chat_pipeline.run(ctx)
    except GatewayError:
        if ctx.attempts and ctx.auth is not None and ctx.policy is not None:
            await svc.chat_pipeline.finish(ctx)  # bill/record the failed upstream attempt
        raise

    if ctx.stream is None:
        assert ctx.response is not None
        return JSONResponse(ctx.response.model_dump(mode="json", exclude_none=True), headers=tollgate_headers(ctx))
    return StreamingResponse(
        _sse(ctx, svc),
        media_type="text/event-stream",
        headers={**tollgate_headers(ctx, cost_known=False), "cache-control": "no-cache", "x-accel-buffering": "no"},
    )


def _data(payload: str) -> bytes:
    return f"data: {payload}\n\n".encode()


async def _sse(ctx: RequestContext, svc: Services) -> AsyncIterator[bytes]:
    assert ctx.stream is not None and ctx.original is not None
    include_usage = bool(ctx.original.stream_options and ctx.original.stream_options.include_usage)
    last: ChatChunk | None = None
    usage = None
    try:
        async for chunk in ctx.stream:
            last = chunk
            if chunk.usage is not None:
                usage, chunk = chunk.usage, chunk.model_copy(update={"usage": None})
            if chunk.choices:
                yield _data(chunk.model_dump_json(exclude_none=True))
        if include_usage and usage is not None and last is not None:
            final = ChatChunk(id=last.id, created=last.created, model=last.model, choices=[], usage=usage)
            yield _data(final.model_dump_json())
        yield _data("[DONE]")
    except ProviderError as exc:
        err = GatewayError(exc.status, "upstream_error", exc.message, type_="server_error")
        yield _data(json.dumps(err.body()))
    finally:
        # Meter even if the client disconnected mid-stream; shield so cancellation
        # of this generator doesn't abort the write.
        task = asyncio.ensure_future(svc.chat_pipeline.finish(ctx))
        _background.add(task)
        task.add_done_callback(_background.discard)
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            pass
