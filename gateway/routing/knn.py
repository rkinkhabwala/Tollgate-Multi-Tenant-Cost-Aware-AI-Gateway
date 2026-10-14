"""Classifier Layer B: embedding kNN over labeled seed prompts (spec §8.3).

* **Seed set:** `data/router_seed.jsonl`, ~1,000 synthetic prompts labeled
  `task_type` and `complexity`. The embeddings are cached in
  `data/router_seed.emb.npz`, keyed by the seed file's SHA-256, so the index can be
  rebuilt without Ollama.
* **Index:** a RediSearch FLAT/COSINE vector index `sys:router:idx` over hashes
  `sys:router:seed:{n}`. It holds global synthetic system data, never tenant data or
  user prompts. RediSearch only indexes Redis DB 0, so this uses
  `settings.router_redis_url`, which defaults to DB 0. `sys:router:version` records
  the seed hash, and `ensure_index()` rebuilds only when that hash changes.
* **Query:** embed the *redacted* last user turn with `nomic-embed-text` (prefix
  `classification: `, head + tail truncated to 2,000 chars), take the k=7 nearest
  seeds, and vote separately on `task_type` and `complexity`. Each vote is weighted
  by similarity, and the winning share is the confidence. The query vector is never
  stored.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from redis.asyncio import Redis
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import ResponseError

log = logging.getLogger(__name__)

INDEX = "sys:router:idx"
PREFIX = "sys:router:seed:"
VERSION_KEY = "sys:router:version"
DIM = 768
EMBED_PREFIX = "classification: "
MAX_CHARS = 2000

Embedder = Callable[[list[str]], Awaitable[list[list[float]]]]


def prepare(text: str) -> str:
    """Prefix for nomic task-type embeddings; keep head + tail of long texts."""
    if len(text) > MAX_CHARS:
        text = text[: MAX_CHARS // 2] + "\n…\n" + text[-MAX_CHARS // 2 :]
    return EMBED_PREFIX + text


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


@dataclass(frozen=True)
class KnnVote:
    task_type: str
    task_confidence: float
    complexity: str
    complexity_confidence: float
    neighbors: tuple[tuple[str, str, float], ...]  # (task, complexity, similarity): labels only


def load_seed(seed_path: Path) -> tuple[list[dict], str]:
    raw = seed_path.read_bytes()
    return [json.loads(line) for line in raw.decode().splitlines() if line.strip()], hashlib.sha256(raw).hexdigest()


async def seed_vectors(seed_path: Path, emb_path: Path, embed: Embedder | None) -> tuple[list[dict], np.ndarray, str]:
    """Seed rows + their normalized vectors, from the cache file if it matches the seed hash."""
    rows, digest = load_seed(seed_path)
    if emb_path.exists():
        cached = np.load(emb_path)
        if str(cached["sha256"]) == digest and len(cached["vectors"]) == len(rows):
            return rows, normalize(cached["vectors"].astype(np.float32)), digest
    if embed is None:
        raise RuntimeError("seed embeddings are stale and no embedder is available")
    vecs: list[list[float]] = []
    for i in range(0, len(rows), 32):
        vecs += await embed([prepare(r["prompt"]) for r in rows[i : i + 32]])
    arr = normalize(np.asarray(vecs, dtype=np.float32))
    np.savez_compressed(emb_path, sha256=digest, vectors=arr.astype(np.float16))
    return rows, arr, digest


async def build_index(redis: Redis, rows: list[dict], vectors: np.ndarray, digest: str) -> None:
    try:
        await redis.ft(INDEX).dropindex(delete_documents=True)
    except ResponseError:
        pass
    await redis.ft(INDEX).create_index(
        [
            TagField("task_type"),
            TagField("complexity"),
            VectorField("vec", "FLAT", {"TYPE": "FLOAT32", "DIM": vectors.shape[1], "DISTANCE_METRIC": "COSINE"}),
        ],
        definition=IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH),
    )
    pipe = redis.pipeline(transaction=False)
    for n, (row, vec) in enumerate(zip(rows, vectors, strict=True)):
        pipe.hset(f"{PREFIX}{n}", mapping={"task_type": row["task_type"], "complexity": row["complexity"], "vec": vec.astype(np.float32).tobytes()})
    pipe.set(VERSION_KEY, digest)
    await pipe.execute()


class KnnClassifier:
    def __init__(self, redis: Redis, embed: Embedder, k: int = 7) -> None:
        self._redis = redis
        self._embed = embed
        self.k = k
        self.ready = False

    async def ensure_index(self, seed_path: Path, emb_path: Path) -> bool:
        """Build the index if missing or built from a different seed file. Returns readiness."""
        if not seed_path.exists() or seed_path.stat().st_size == 0:
            log.warning("router seed file missing; kNN layer disabled")
            return False
        _, digest = load_seed(seed_path)
        current = await self._redis.get(VERSION_KEY)
        try:
            await self._redis.ft(INDEX).info()
            exists = True
        except ResponseError:
            exists = False
        if not exists or (current.decode() if isinstance(current, bytes) else current) != digest:
            rows, vecs, digest = await seed_vectors(seed_path, emb_path, self._embed)
            await build_index(self._redis, rows, vecs, digest)
            log.info("router index built", extra={"seeds": len(rows)})
        self.ready = True
        return True

    async def embed_query(self, text: str) -> np.ndarray:
        (vec,) = await self._embed([prepare(text)])
        return normalize(np.asarray(vec, dtype=np.float32))

    async def vote(self, text: str) -> KnnVote | None:
        if not self.ready or not text.strip():
            return None
        q = await self.embed_query(text)
        query = Query(f"*=>[KNN {self.k} @vec $v AS dist]").sort_by("dist").return_fields("task_type", "complexity", "dist").dialect(2)
        res = await self._redis.ft(INDEX).search(query, query_params={"v": q.astype(np.float32).tobytes()})
        return tally([(d.task_type, d.complexity, 1.0 - float(d.dist)) for d in res.docs])


def tally(neighbors: list[tuple[str, str, float]]) -> KnnVote | None:
    """Similarity-weighted majority vote on task_type and complexity."""
    if not neighbors:
        return None
    tasks: dict[str, float] = defaultdict(float)
    cxs: dict[str, float] = defaultdict(float)
    for task, cx, sim in neighbors:
        w = max(sim, 0.0) + 1e-6
        tasks[task] += w
        cxs[cx] += w
    task, tw = max(tasks.items(), key=lambda kv: kv[1])
    cx, cw = max(cxs.items(), key=lambda kv: kv[1])
    return KnnVote(task, tw / sum(tasks.values()), cx, cw / sum(cxs.values()), tuple(neighbors))
