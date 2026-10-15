"""HTTP service.

    uvicorn policyqa.api:create_app --factory --port 8000

Configure with POLICYQA_SYSTEM (extractive | rag | agent), POLICYQA_RETRIEVER and
POLICYQA_MODEL. With nothing set it runs the extractive system, which needs no keys.
"""

from __future__ import annotations

import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .factory import build_system, load_rates
from .systems import QASystem
from .tools import QuoteError, RateBook


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class AskResponse(BaseModel):
    answer: str
    answerable: bool
    citations: list[str]
    retrieved: list[str]
    latency_s: float


class QuoteRequest(BaseModel):
    age: int = Field(ge=0, le=120)
    risk_class: Literal["Preferred Plus", "Preferred", "Standard", "Tobacco"]
    term_years: Literal[10, 20, 30]
    face_amount: int = Field(gt=0)
    mode: Literal["annual", "semiannual", "quarterly", "monthly"] = "annual"


def create_app(system: QASystem | None = None, rates: RateBook | None = None) -> FastAPI:
    if system is None:
        system, _ = build_system(
            os.environ.get("POLICYQA_SYSTEM", "extractive"),
            os.environ.get("POLICYQA_RETRIEVER", "bm25"),
            os.environ.get("POLICYQA_MODEL"),
        )
    rates = rates or load_rates()
    app = FastAPI(title="Larkspur policy QA", version="0.1.0")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    # Plain `def` handlers run in FastAPI's thread pool, so a slow model call
    # does not block other requests.
    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        try:
            result = system.answer(request.question)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"answering failed: {type(exc).__name__}") from exc
        return AskResponse(
            answer=result.text,
            answerable=result.answerable,
            citations=result.citations,
            retrieved=result.retrieved,
            latency_s=round(result.latency_s, 3),
        )

    @app.post("/quote", summary="Price a term life policy", operation_id="quote_term_premium")
    def quote(request: QuoteRequest) -> dict:
        try:
            return rates.quote(**request.model_dump())
        except QuoteError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app
