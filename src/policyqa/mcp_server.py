"""MCP server exposing the same two tools the agent uses, so any MCP client
(a desktop assistant, another agent framework) can call them.

    python -m policyqa.mcp_server        # stdio transport
"""

from __future__ import annotations

import os
from functools import lru_cache

from mcp.server.mcpserver import MCPServer  # mcp 2.x; in 1.x this was mcp.server.fastmcp.FastMCP

from .factory import load_rates, load_retriever
from .systems import format_hits
from .tools import QuoteError

mcp = MCPServer("larkspur-policy-qa")


@lru_cache(maxsize=1)
def _retriever():
    return load_retriever(os.environ.get("POLICYQA_RETRIEVER", "bm25"))


@lru_cache(maxsize=1)
def _rates():
    return load_rates()


@mcp.tool()
def search_policy_docs(query: str, k: int = 5) -> str:
    """Search Larkspur policy, rider, claims, underwriting and billing documents.
    Returns the most relevant sections, each labelled with a section id to cite."""
    hits = _retriever().search(query, max(1, min(k, 10)))
    return format_hits(hits) or "No matching sections."


@mcp.tool()
def quote_term_premium(age: int, risk_class: str, term_years: int, face_amount: int, mode: str = "annual") -> dict:
    """Price a Larkspur Term Life policy from the official rate table.
    risk_class: Preferred Plus, Preferred, Standard or Tobacco. term_years: 10, 20 or 30.
    mode: annual, semiannual, quarterly or monthly."""
    try:
        return _rates().quote(age, risk_class, term_years, face_amount, mode)
    except QuoteError as exc:
        return {"error": str(exc)}


if __name__ == "__main__":
    mcp.run()
