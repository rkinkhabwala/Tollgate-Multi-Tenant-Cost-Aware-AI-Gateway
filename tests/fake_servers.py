"""Small HTTP servers used by integration tests and the Phase 3 demo.

* `FlakyOllama`: a transparent proxy to a real Ollama that can take individual
  models "down" at runtime. A down model either answers HTTP 503 at once
  (`mode="error"`) or hangs past the client's timeout (`mode="timeout"`), which
  simulates stopping or partitioning a provider mid-run. Control endpoint:
  `POST /_control {"down": {"qwen2.5:1.5b": "error"}}`.
* `FakeAnthropic`: a minimal Messages API. It answers per model from a script
  (`replies: {model: text}`), so escalation can be exercised deterministically with
  a fake key, while usage tokens are reported so billing is non-zero.

Both are Starlette apps run by uvicorn on a free port, in the test's event loop, or
standalone via `python -m tests.fake_servers flaky-ollama --port 11500`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
from typing import Any

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FlakyOllama:
    def __init__(self, upstream: str = "http://localhost:11434") -> None:
        self.upstream = upstream.rstrip("/")
        self.down: dict[str, str] = {}  # model → "error" | "timeout"
        self.hits: dict[str, int] = {}
        self.app = Starlette(routes=[Route("/_control", self.control, methods=["POST", "GET"]), Route("/{path:path}", self.proxy, methods=["GET", "POST"])])

    async def control(self, request: Request) -> Response:
        if request.method == "POST":
            self.down = (await request.json()).get("down", {})
        return JSONResponse({"down": self.down, "hits": self.hits})

    async def proxy(self, request: Request) -> Response:
        body = await request.body()
        model = json.loads(body).get("model") if body else None
        if model:
            self.hits[model] = self.hits.get(model, 0) + 1
        mode = self.down.get(model or "")
        if mode == "timeout":
            await asyncio.sleep(20)  # longer than the 10 s gateway timeout used in tests/demos
        if mode == "error":
            return JSONResponse({"error": "model server unavailable"}, status_code=503)
        client = httpx.AsyncClient(base_url=self.upstream, timeout=600)
        upstream = await client.send(client.build_request(request.method, "/" + request.path_params["path"], content=body, headers={"content-type": "application/json"}), stream=True)

        async def relay():  # type: ignore[no-untyped-def]
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()
                await client.aclose()

        return StreamingResponse(relay(), status_code=upstream.status_code, media_type=upstream.headers.get("content-type"))


class FakeAnthropic:
    def __init__(self, replies: dict[str, str] | None = None) -> None:
        self.replies = replies or {}
        self.requests: list[dict[str, Any]] = []
        self.app = Starlette(routes=[Route("/v1/messages", self.messages, methods=["POST"])])

    async def messages(self, request: Request) -> Response:
        if request.headers.get("x-api-key") != "test-anthropic-key" or request.headers.get("anthropic-version") != "2023-06-01":
            return JSONResponse({"type": "error", "error": {"type": "authentication_error"}}, status_code=401)
        body = await request.json()
        self.requests.append(body)
        text = self.replies.get(body["model"], "ok")
        usage = {"input_tokens": 120, "output_tokens": 30}
        if not body.get("stream"):
            return JSONResponse({"id": "msg_fake", "type": "message", "role": "assistant", "model": body["model"],
                                 "content": [{"type": "text", "text": text}], "stop_reason": "end_turn", "usage": usage})

        def events():  # type: ignore[no-untyped-def]
            seq = [
                {"type": "message_start", "message": {"id": "msg_fake", "usage": {"input_tokens": 120, "output_tokens": 1}}},
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
                {"type": "content_block_stop", "index": 0},
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 30}},
                {"type": "message_stop"},
            ]
            for e in seq:
                yield f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode()

        return StreamingResponse(events(), media_type="text/event-stream")


class ServerHandle:
    def __init__(self, app: Any, port: int | None = None) -> None:
        self.port = port or free_port()
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_config=None, lifespan="off", timeout_graceful_shutdown=1))
        self.server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
        self.task: asyncio.Task[None] | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def start(self) -> ServerHandle:
        self.task = asyncio.create_task(self.server.serve())
        for _ in range(200):
            if self.server.started:
                return self
            await asyncio.sleep(0.02)
        raise RuntimeError("fake server failed to start")

    async def stop(self) -> None:
        self.server.should_exit = True
        if self.task:
            await self.task


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("server", choices=["flaky-ollama"])
    p.add_argument("--port", type=int, default=11500)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--upstream", default="http://localhost:11434")
    a = p.parse_args()
    uvicorn.run(FlakyOllama(a.upstream).app, host=a.host, port=a.port, log_level="warning")
