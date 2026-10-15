"""Builds a configured system. Shared by the evaluation, the API and the MCP server."""

from __future__ import annotations

import os
from pathlib import Path

from . import data_dir
from .chunking import load_corpus
from .llm import ChatModel, load_model
from .retrieval import DEFAULT_EMBEDDER, DEFAULT_RERANKER, Retriever, build_retriever
from .systems import AgentSystem, DifySystem, ExtractiveSystem, QASystem, RagSystem
from .tools import RateBook

SYSTEMS = ("extractive", "rag", "agent", "dify")


def models_path() -> Path:
    return Path(os.environ.get("POLICYQA_MODELS", data_dir().parent / "models.toml"))


def load_rates() -> RateBook:
    return RateBook(data_dir() / "rates.csv")


def load_retriever(name: str = "bm25", embedder: str = DEFAULT_EMBEDDER, reranker: str = DEFAULT_RERANKER) -> Retriever:
    return build_retriever(name, load_corpus(data_dir() / "corpus"), embedder, reranker)


def build_system(
    system: str = "extractive",
    retriever: str = "bm25",
    model: str | None = None,
    k: int = 5,
    embedder: str = DEFAULT_EMBEDDER,
    reranker: str = DEFAULT_RERANKER,
) -> tuple[QASystem, ChatModel | None]:
    if system not in SYSTEMS:
        raise ValueError(f"unknown system {system!r}; choose from {SYSTEMS}")
    if system == "dify":
        base, key = os.environ.get("DIFY_BASE_URL"), os.environ.get("DIFY_API_KEY")
        if not base or not key:
            raise RuntimeError("set DIFY_BASE_URL and DIFY_API_KEY to evaluate the Dify app")
        return DifySystem(base, key), None
    retr = load_retriever(retriever, embedder, reranker)
    if system == "extractive":
        return ExtractiveSystem(retr, k), None
    if not model:
        raise ValueError(f"the {system} system needs --model (a name from models.toml)")
    chat = load_model(model, models_path())
    if system == "rag":
        return RagSystem(retr, chat, k), chat
    return AgentSystem(retr, chat, load_rates(), k), chat
