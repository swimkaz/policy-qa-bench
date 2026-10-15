import numpy as np

from policyqa.chunking import Chunk
from policyqa.retrieval import BM25Retriever, DenseRetriever, Hit, HybridRetriever, RerankRetriever, tokenize


def _chunk(cid, body):
    return Chunk(cid, cid, "doc", "Doc", "Heading", body)


CHUNKS = [
    _chunk("a", "The grace period is 31 days after the due date."),
    _chunk("b", "Dividends are not guaranteed."),
    _chunk("c", "Policy loans carry interest."),
]


def test_tokenize_drops_stopwords_and_plurals_keeps_decimals():
    assert tokenize("What are the premiums for 0.0875?") == ["premium", "0.0875"]


def test_bm25_ranks_the_matching_chunk_first():
    hits = BM25Retriever(CHUNKS).search("how long is the grace period", k=2)
    assert hits[0].chunk.id == "a"
    assert all(h.score > 0 for h in hits)


def test_bm25_returns_nothing_when_no_term_matches():
    assert BM25Retriever(CHUNKS).search("zebra", k=3) == []


def _bag_embedder(texts):
    """Toy embedder: one dimension per vocabulary word, unit length."""
    vocab = ["grace", "period", "dividend", "guaranteed", "loan", "interest"]
    rows = []
    for text in texts:
        toks = tokenize(text)
        v = np.array([float(toks.count(w)) for w in vocab]) + 1e-9
        rows.append(v / np.linalg.norm(v))
    return np.vstack(rows)


def test_dense_retriever_uses_cosine_similarity():
    hits = DenseRetriever(CHUNKS, _bag_embedder).search("are dividends guaranteed", k=1)
    assert hits[0].chunk.id == "b"


class _Fixed:
    def __init__(self, order):
        self.order = order

    def search(self, query, k=5):
        return [Hit(c, 1.0) for c in self.order[:k]]


def test_rrf_rewards_agreement_between_retrievers():
    a, b, c = CHUNKS
    fused = HybridRetriever([_Fixed([a, b, c]), _Fixed([b, c, a])]).search("q", k=3)
    assert [h.chunk.id for h in fused] == ["b", "a", "c"]


def test_reranker_reorders_candidates():
    base = _Fixed(CHUNKS)
    rerank = RerankRetriever(base, lambda q, passages: [float("loans" in p) for p in passages])
    assert rerank.search("q", k=1)[0].chunk.id == "c"
