from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
from a2a.client import ClientConfig, ClientFactory
from a2a.types import AgentCard, Message, Part, Role, SendMessageRequest, TaskState
from a2a.utils.constants import TransportProtocol
from google.protobuf.json_format import ParseDict

ROLES = ("company-research", "financial-report", "stock", "context", "synthesis")


def test_official_sdk_client_completes_five_process_topology_smoke(
    tmp_path: Path,
) -> None:
    ports = {role: _free_port() for role in ROLES}
    processes = [
        _start(role, ports, tmp_path)
        for role in ("financial-report", "stock", "context", "synthesis")
    ]
    processes.append(_start("company-research", ports, tmp_path))
    try:
        for role, process in zip(
            ("financial-report", "stock", "context", "synthesis", "company-research"),
            processes,
            strict=True,
        ):
            _wait_for_server(ports[role], process)
        task = asyncio.run(_send_with_official_client(ports["company-research"]))

        assert task.status.state == TaskState.TASK_STATE_COMPLETED
        assert task.artifacts[-1].name == "source-backed-research-report"
        handoff_artifacts = [
            artifact
            for artifact in task.artifacts
            if artifact.metadata.fields["kind"].string_value == "specialist_handoff"
        ]
        assert len(handoff_artifacts) == 4
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def _start(
    role: str,
    ports: dict[str, int],
    app_home: Path,
) -> subprocess.Popen[str]:
    environment = {
        **os.environ,
        "FRA_HOME": str(app_home),
        "FRA_A2A_ENABLED": "true",
        "FRA_A2A_ROLE": role,
        "FRA_A2A_PUBLIC_BASE_URL": f"http://127.0.0.1:{ports[role]}",
        "FRA_A2A_FINANCIAL_REPORT_URL": f"http://127.0.0.1:{ports['financial-report']}",
        "FRA_A2A_STOCK_URL": f"http://127.0.0.1:{ports['stock']}",
        "FRA_A2A_CONTEXT_URL": f"http://127.0.0.1:{ports['context']}",
        "FRA_A2A_SYNTHESIS_URL": f"http://127.0.0.1:{ports['synthesis']}",
        "FRA_SEC_USER_AGENT": "financial-research-agent tests tests@example.com",
    }
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "tests.a2a_topology_fixture_app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(ports[role]),
            "--log-level",
            "error",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


async def _send_with_official_client(port: int):
    base_url = f"http://127.0.0.1:{port}"
    async with httpx.AsyncClient(timeout=30) as http_client:
        response = await http_client.get(f"{base_url}/.well-known/agent-card.json")
        response.raise_for_status()
        card = ParseDict(response.json(), AgentCard())
        client = ClientFactory(
            ClientConfig(
                streaming=False,
                httpx_client=http_client,
                supported_protocol_bindings=[TransportProtocol.HTTP_JSON.value],
            )
        ).create(card)
        events = [
            event
            async for event in client.send_message(
                SendMessageRequest(
                    message=Message(
                        message_id="five-process-smoke",
                        role=Role.ROLE_USER,
                        parts=[Part(text="Research TEST FIXTURE company")],
                    )
                )
            )
        ]
    return events[-1].task


def _wait_for_server(port: int, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 20
    url = f"http://127.0.0.1:{port}/.well-known/agent-card.json"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            error = process.stderr.read() if process.stderr is not None else ""
            raise AssertionError(f"A2A fixture server exited early: {error}")
        try:
            if httpx.get(url, timeout=0.5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.05)
    raise AssertionError("A2A fixture server did not become ready")


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])
