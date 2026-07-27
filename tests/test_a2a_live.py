from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from financial_research_agent.a2a import create_a2a_app
from financial_research_agent.settings import Settings


@pytest.mark.skipif(
    os.environ.get("FRA_A2A_LIVE_SMOKE_TEST") != "1",
    reason="set FRA_A2A_LIVE_SMOKE_TEST=1 to run real-data A2A smoke",
)
def test_live_company_research_task_uses_configured_real_providers(tmp_path: Path) -> None:
    settings = Settings.from_env({**os.environ, "FRA_HOME": str(tmp_path)})
    if not settings.data_sources.alpha_vantage_api_key:
        pytest.fail("FRA_ALPHA_VANTAGE_API_KEY is required for live A2A smoke")
    if "contact@financial-research-agent.local" in settings.data_sources.sec_user_agent:
        pytest.fail("a real FRA_SEC_USER_AGENT contact is required for live A2A smoke")

    with TestClient(create_a2a_app(settings=settings)) as client:
        response = client.post(
            "/message:send",
            headers={"A2A-Version": "1.0"},
            json={
                "message": {
                    "messageId": "live-a2a-novo-smoke",
                    "role": "ROLE_USER",
                    "parts": [{"text": "Research Novo Nordisk using available real data."}],
                }
            },
        )

    assert response.status_code == 200
    task = response.json()["task"]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert task["artifacts"][-1]["metadata"]["kind"] == "deterministic_synthesis"
