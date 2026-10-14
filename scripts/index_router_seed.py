"""Build or rebuild the router's kNN vector index (spec §8.3).

    uv run python -m scripts.index_router_seed            # build if missing or stale
    uv run python -m scripts.index_router_seed --force    # drop and rebuild

Reads `data/router_seed.jsonl`, uses the cached embeddings in
`data/router_seed.emb.npz` when they match the seed file's SHA-256 (otherwise it
embeds with Ollama `nomic-embed-text`), and writes the RediSearch index
`sys:router:idx` to `TOLLGATE_ROUTER_REDIS_URL` (DB 0; RediSearch can't index other
DBs). The gateway also runs this at startup if the index is missing, so the script
is mainly for forcing a rebuild after the seed set grows.
"""

from __future__ import annotations

import argparse
import asyncio

from redis.asyncio import Redis

from gateway.providers.ollama import OllamaAdapter
from gateway.routing.knn import VERSION_KEY, KnnClassifier
from gateway.settings import get_settings


async def main(force: bool) -> None:
    s = get_settings()
    redis = Redis.from_url(s.router_redis_url)
    ollama = OllamaAdapter(s.ollama_base_url)
    knn = KnnClassifier(redis, lambda texts: ollama.embed(s.embed_model.split("/", 1)[1], texts))
    if force:
        await redis.delete(VERSION_KEY)
    ok = await knn.ensure_index(s.router_seed_path, s.router_seed_emb_path)
    info = await redis.ft("sys:router:idx").info() if ok else {}
    print(f"index ready={ok} docs={info.get('num_docs')} version={(await redis.get(VERSION_KEY) or b'').decode()[:12]}")
    await ollama.aclose()
    await redis.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true")
    asyncio.run(main(parser.parse_args().force))
