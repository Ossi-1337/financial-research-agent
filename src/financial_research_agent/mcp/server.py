from __future__ import annotations

import os
from typing import Any

from .client import DEFAULT_MCP_APP_BASE_URL, McpApplicationClient


class McpDependencyError(RuntimeError):
    pass


def create_mcp_server(
    *,
    client: McpApplicationClient | None = None,
    app_base_url: str | None = None,
) -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise McpDependencyError(
            'MCP support is not installed. Run: pip install -e ".[mcp]"'
        ) from exc

    application = client or McpApplicationClient(
        base_url=app_base_url or os.environ.get("FRA_MCP_APP_BASE_URL", DEFAULT_MCP_APP_BASE_URL)
    )
    server = FastMCP(
        "Financial Research Agent",
        instructions=(
            "Local interface to the Financial Research Agent application. Messages use the same "
            "orchestrator and A2A specialist flow as the Chat UI."
        ),
        log_level="ERROR",
    )

    @server.tool(name="send_message")
    async def send_message(
        content: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Send a message through the canonical application and orchestrator flow."""
        return (
            await application.send_message(
                content=content,
                session_id=session_id,
            )
        ).to_dict()

    @server.tool(name="get_research_status")
    async def get_research_status(job_id: str) -> dict[str, Any]:
        """Get progress for a research job returned by send_message."""
        return (await application.get_research_status(job_id=job_id)).to_dict()

    @server.tool(name="get_research_result")
    async def get_research_result(job_id: str) -> dict[str, Any]:
        """Get a completed source-backed report and evidence for a research job."""
        return (await application.get_research_result(job_id=job_id)).to_dict()

    @server.tool(name="cancel_research")
    async def cancel_research(job_id: str) -> dict[str, Any]:
        """Cancel a queued or running research job."""
        return (await application.cancel_research(job_id=job_id)).to_dict()

    return server


def run_mcp_stdio(*, app_base_url: str | None = None) -> None:
    create_mcp_server(app_base_url=app_base_url).run(transport="stdio")
