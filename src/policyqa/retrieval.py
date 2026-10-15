"""Retrievers: BM25, dense (Hugging Face embeddings), hybrid (RRF), and reranking.

All retrievers share one method, ``search(query, k) -> list[Hit]``, so the
evaluation can swap them without touching anything else.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np

from .chunking import Chunk

STOPWORDS = frozenset(
    "a an and are as at be by do does for from how i if in is it me my of on or the to "
    "what when which who will with would can could should that this there their".split()
)

EmbedFn = Callable[[Sequence[str]], np.ndarray]  # texts -> (n, d) array of unit vectors
ScoreFn = Callable[[str, Sequence[str]], Sequence[float]]  # query, passages -> scores


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[Hit]: ...


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())
    out = []
    for tok in tokens:
        if tok in STOPWORDS:
            continue
        # Crude plural stripping so "premiums" matches "premium".
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.append(tok)
    return out


class BM25Retriever:
    """Okapi BM25, written out so every term in the formula can be explained."""

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self._tf = [Counter(tokenize(c.text)) for c in self.chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg_len = sum(self._len) / max(len(self._len), 1)
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(self.chunks)
        self._idf = {t: math.log((n - d + 0.5) / (d + 0.5) + 1.0) for t, d in df.items()}

    def search(self, query: str, k: int = 5) -> list[Hit]:
        terms = tokenize(query)
        scored = []
        for chunk, tf, length in zip(self.chunks, self._tf, self._len):
            norm = self.k1 * (1 - self.b + self.b * length / self._avg_len)
            score = sum(
                self._idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in terms if t in tf
            )
            if score > 0:
                scored.append(Hit(chunk, score))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]


class DenseRetriever:
    """Cosine similarity over embeddings. ``embed_fn`` must return unit vectors."""

    def __init__(self, chunks: Sequence[Chunk], embed_fn: EmbedFn, query_prefix: str = ""):
        self.chunks = list(chunks)
        self._embed = embed_fn
        self._prefix = query_prefix
        self._matrix = np.asarray(embed_fn([c.text for c in self.chunks]), dtype=np.float32)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        q = np.asarray(self._embed([self._prefix + query]), dtype=np.float32)[0]
        sims = self._matrix @ q
        order = np.argsort(-sims)[:k]
        return [Hit(self.chunks[i], float(sims[i])) for i in order]


class HybridRetriever:
    """Reciprocal rank fusion. It combines ranks, so BM25 and cosine scores,
    which live on different scales, never have to be normalised against each other."""

    def __init__(self, retrievers: Sequence[Retriever], rrf_k: int = 60, fetch_k: int = 20):
        self.retrievers = list(retrievers)
        self.rrf_k, self.fetch_k = rrf_k, fetch_k

    def search(self, query: str, k: int = 5) -> list[Hit]:
        fused: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        for retriever in self.retrievers:
            for rank, hit in enumerate(retriever.search(query, self.fetch_k), start=1):
                fused[hit.chunk.id] = fused.get(hit.chunk.id, 0.0) + 1.0 / (self.rrf_k + rank)
                by_id[hit.chunk.id] = hit.chunk
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [Hit(by_id[cid], score) for cid, score in ranked]


class RerankRetriever:
    """Fetch a wide candidate set cheaply, then reorder it with a cross-encoder."""

    def __init__(self, base: Retriever, score_fn: ScoreFn, fetch_k: int = 20):
        self.base, self._score, self.fetch_k = base, score_fn, fetch_k

    def search(self, query: str, k: int = 5) -> list[Hit]:
        candidates = self.base.search(query, self.fetch_k)
        if not candidates:
            return []
        scores = self._score(query, [h.chunk.text for h in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)[:k]
        return [Hit(hit.chunk, float(score)) for hit, score in ranked]


# --- Hugging Face models (imported lazily; install with `pip install -e ".[hf]"`) ---

DEFAULT_EMBEDDER = "BAAI/bge-small-en-v1.5"
# bge v1.5 models expect this instruction in front of short retrieval queries.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def hf_embedder(model_name: str = DEFAULT_EMBEDDER) -> EmbedFn:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return lambda texts: model.encode(list(texts), normalize_embeddings=True)


def hf_reranker(model_name: str = DEFAULT_RERANKER) -> ScoreFn:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name)
    return lambda query, passages: model.predict([(query, p) for p in passages]).tolist()


RETRIEVERS = ("bm25", "dense", "hybrid", "hybrid-rerank")


def build_retriever(
    name: str,
    chunks: Sequence[Chunk],
    embedder: str = DEFAULT_EMBEDDER,
    reranker: str = DEFAULT_RERANKER,
) -> Retriever:
    if name not in RETRIEVERS:
        raise ValueError(f"unknown retriever {name!r}; choose from {RETRIEVERS}")
    bm25 = BM25Retriever(chunks)
    if name == "bm25":
        return bm25
    prefix = BGE_QUERY_PREFIX if "bge" in embedder.lower() else ""
    dense = DenseRetriever(chunks, hf_embedder(embedder), query_prefix=prefix)
    if name == "dense":
        return dense
    hybrid = HybridRetriever([bm25, dense])
    if name == "hybrid":
        return hybrid
    return RerankRetriever(hybrid, hf_reranker(reranker))
