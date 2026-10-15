import pytest
from fastapi.testclient import TestClient

from policyqa.api import create_app
from policyqa.retrieval import BM25Retriever
from policyqa.systems import ExtractiveSystem


@pytest.fixture()
def client(chunks, rates):
    return TestClient(create_app(ExtractiveSystem(BM25Retriever(chunks)), rates))


def test_health(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_ask(client):
    body = client.post("/ask", json={"question": "How long is the grace period on a whole life policy?"}).json()
    assert "61 days" in body["answer"]
    assert body["citations"] == ["whole_life#grace-period"]


def test_ask_rejects_empty_questions(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_quote(client):
    payload = {"age": 35, "risk_class": "Preferred", "term_years": 20, "face_amount": 500000, "mode": "monthly"}
    assert client.post("/quote", json=payload).json()["payment"] == "49.44"


def test_quote_reports_ineligible_requests(client):
    payload = {"age": 52, "risk_class": "Standard", "term_years": 30, "face_amount": 500000}
    response = client.post("/quote", json=payload)
    assert response.status_code == 422 and "not eligible" in response.json()["detail"]


def test_a_failing_system_returns_502_without_leaking_details(rates):
    class Broken:
        def answer(self, question):
            raise RuntimeError("secret-key-123")

    response = TestClient(create_app(Broken(), rates)).post("/ask", json={"question": "anything"})
    assert response.status_code == 502 and "secret" not in response.text
