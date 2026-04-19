from __future__ import annotations

import json
import os
from pathlib import Path

import anyio
import pytest

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXE = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
SERVER_SCRIPT = PROJECT_ROOT / "server.py"


def _server_params() -> StdioServerParameters:
    # Intentionally mirrors the launch config used in MCP clients.
    return StdioServerParameters(
        command=str(PYTHON_EXE),
        args=[str(SERVER_SCRIPT)],
        cwd=str(PROJECT_ROOT),
    )


def _tool_result_payload(result) -> dict:
    if result.structuredContent is not None:
        return result.structuredContent
    if result.content and getattr(result.content[0], "type", None) == "text":
        return json.loads(result.content[0].text)
    return {}


def test_mcp_stdio_launch_and_tools_work() -> None:
    if os.environ.get("RUN_MCP_E2E") != "1":
        pytest.skip("Set RUN_MCP_E2E=1 to run MCP e2e test.")

    async def scenario() -> None:
        async with stdio_client(_server_params()) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                tools_result = await session.list_tools()
                tools = {tool.name: tool for tool in tools_result.tools}
                assert "search_chunks" in tools
                assert "get_chunk_by_id" in tools
                assert tools["search_chunks"].description
                assert tools["get_chunk_by_id"].description

                get_result = await session.call_tool("get_chunk_by_id", {"chunk_id": "missing-id"})
                assert get_result.isError is False
                payload = _tool_result_payload(get_result)
                assert payload.get("found") is False
                assert payload.get("chunk") is None

    anyio.run(scenario)


def test_mcp_stdio_search_chunks_slow_e2e() -> None:
    if os.environ.get("RUN_SLOW_MCP_E2E") != "1":
        pytest.skip("Set RUN_SLOW_MCP_E2E=1 to run slow search_chunks e2e.")

    async def scenario() -> None:
        async with stdio_client(_server_params()) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                with anyio.fail_after(180):
                    result = await session.call_tool(
                        "search_chunks",
                        {"query": "Retriever", "top_k": 10},
                    )
                assert result.isError is False
                payload = _tool_result_payload(result)
                assert payload.get("query") == "Retriever"
                assert payload.get("top_k") == 10
                assert isinstance(payload.get("results"), list)

    anyio.run(scenario)
