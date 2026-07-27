"""Optional A2A 1.0 task server for source-backed company research."""

from financial_research_agent.a2a.runtime import (
    A2AResearchRuntime,
    create_default_a2a_runtime,
)
from financial_research_agent.a2a.server import create_a2a_app, create_agent_card
from financial_research_agent.a2a.store import A2ATaskEventRecord, SQLiteA2ATaskStore

__all__ = [
    "A2AResearchRuntime",
    "A2ATaskEventRecord",
    "SQLiteA2ATaskStore",
    "create_a2a_app",
    "create_agent_card",
    "create_default_a2a_runtime",
]
