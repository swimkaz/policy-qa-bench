import asyncio

from policyqa import mcp_server


def test_server_lists_both_tools():
    names = {tool.name for tool in asyncio.run(mcp_server.mcp.list_tools())}
    assert names == {"search_policy_docs", "quote_term_premium"}


def test_tools_return_results_and_readable_errors():
    assert mcp_server.quote_term_premium(47, "Standard", 10, 400_000, "semiannual")["payment"] == "476.32"
    assert "ages 25 through 54" in mcp_server.quote_term_premium(62, "Standard", 10, 200_000)["error"]
    assert mcp_server.search_policy_docs("child rider cost", 1).startswith("[riders#child-term-rider]")
