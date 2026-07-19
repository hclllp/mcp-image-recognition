"""
Tests for the MCP Image Recognition Server.

Spawns the server as a subprocess and communicates via raw JSON-RPC over
stdio ― no dependency on the ``mcp`` package (which pulls in pydantic).

Tests that require real API keys (Anthropic / OpenAI) are skipped when
credentials are not available.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A minimal 1×1 white PNG for testing (valid image, no real content)
TEST_PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080600000001f15c"
    "4a00000009704859730000000ec400000ec401952b0e1b0000001c4944415478"
    "9c636460606062626060606060600000000000ffff030000060001f5f7e3c000"
    "00000049454e44ae426082"
)
TEST_IMAGE_B64 = base64.b64encode(TEST_PNG_BYTES).decode()

SERVER_MODULE = "image_recognition_server.server"


def _has_api_key(provider: str) -> bool:
    """Check whether a real API key is set for the given provider."""
    if provider == "anthropic":
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    elif provider == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    return False


def _send(proc: subprocess.Popen, msg: Dict[str, Any]) -> None:
    """Send a JSON-RPC message to the server's stdin."""
    line = json.dumps(msg, ensure_ascii=False)
    assert proc.stdin is not None
    proc.stdin.write(line + "\n")
    proc.stdin.flush()


def _recv(proc: subprocess.Popen, timeout: float = 10.0) -> Dict[str, Any]:
    """Read one JSON-RPC response line from the server's stdout."""
    assert proc.stdout is not None
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = proc.stdout.readline()
        if line:
            return json.loads(line.strip())
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited with code {proc.returncode}")
        time.sleep(0.05)
    raise TimeoutError(f"No response within {timeout}s")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def server_process() -> subprocess.Popen:
    """Start the MCP server as a subprocess."""
    env = os.environ.copy()
    # Ensure a vision provider is set (fake key — tool calls will fail but
    # the protocol tests don't need real API access).
    env.setdefault("ANTHROPIC_API_KEY", "test_key")
    env.setdefault("VISION_PROVIDER", "anthropic")
    env.setdefault("LOG_LEVEL", "ERROR")

    proc = subprocess.Popen(
        [sys.executable, "-m", SERVER_MODULE],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        cwd=str((Path(__file__).parent.parent / "src").resolve()),
    )
    yield proc
    # Cleanup
    proc.stdin.close()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture
def initialized(server_process: subprocess.Popen) -> subprocess.Popen:
    """Server process that has already completed the initialize handshake."""
    _send(server_process, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                           "params": {"protocolVersion": "2024-11-05",
                                      "capabilities": {},
                                      "clientInfo": {"name": "test", "version": "1.0"}}})
    resp = _recv(server_process)
    assert resp["id"] == 1
    assert "result" in resp
    assert resp["result"]["serverInfo"]["name"] == "mcp-image-recognition"

    # Send initialized notification
    _send(server_process, {"jsonrpc": "2.0", "method": "notifications/initialized"})
    return server_process


# ---------------------------------------------------------------------------
# Protocol tests (no API keys needed)
# ---------------------------------------------------------------------------

class TestProtocol:
    """Tests for JSON-RPC / MCP protocol handling."""

    def test_initialize(self, server_process: subprocess.Popen) -> None:
        """Server responds to initialize with its capabilities."""
        _send(server_process, {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        })
        resp = _recv(server_process)
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 1
        result = resp["result"]
        assert result["protocolVersion"] == "2024-11-05"
        assert "tools" in result["capabilities"]
        assert result["serverInfo"]["name"] == "mcp-image-recognition"

    def test_list_tools(self, initialized: subprocess.Popen) -> None:
        """Server returns the two registered tools."""
        _send(initialized, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        resp = _recv(initialized)
        assert resp["id"] == 2
        tools: List[Dict] = resp["result"]["tools"]
        tool_names = {t["name"] for t in tools}
        assert "describe_image" in tool_names
        assert "describe_image_from_file" in tool_names
        # Each tool must have an inputSchema
        for t in tools:
            assert "inputSchema" in t, f"Missing inputSchema for {t['name']}"

    def test_ping(self, initialized: subprocess.Popen) -> None:
        """Server responds to ping."""
        _send(initialized, {"jsonrpc": "2.0", "id": 3, "method": "ping"})
        resp = _recv(initialized)
        assert resp["id"] == 3
        assert "result" in resp

    def test_unknown_method(self, initialized: subprocess.Popen) -> None:
        """Server returns error for unknown methods."""
        _send(initialized, {"jsonrpc": "2.0", "id": 4, "method": "nonexistent"})
        resp = _recv(initialized)
        assert resp["id"] == 4
        assert "error" in resp
        assert resp["error"]["code"] == -32601  # METHOD_NOT_FOUND

    def test_invalid_json(self, server_process: subprocess.Popen) -> None:
        """Server handles malformed JSON gracefully."""
        assert server_process.stdin is not None
        server_process.stdin.write("this is not json\n")
        server_process.stdin.flush()
        # It should respond with a parse error
        resp = _recv(server_process)
        assert "error" in resp
        assert resp["error"]["code"] == -32700  # PARSE_ERROR


# ---------------------------------------------------------------------------
# Tool tests (require API keys — skipped otherwise)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (_has_api_key("anthropic") or _has_api_key("openai")),
    reason="No ANTHROPIC_API_KEY or OPENAI_API_KEY set",
)
class TestTools:
    """Tests that exercise real tool calls (needs API credentials)."""

    @pytest.fixture
    def vision_provider(self) -> str:
        return "openai" if _has_api_key("openai") else "anthropic"

    @pytest.fixture
    def api_process(self, vision_provider: str) -> subprocess.Popen:
        """Server process with a real API key for the selected provider."""
        env = os.environ.copy()
        env["VISION_PROVIDER"] = vision_provider
        env["LOG_LEVEL"] = "ERROR"

        proc = subprocess.Popen(
            [sys.executable, "-m", SERVER_MODULE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            cwd=str((Path(__file__).parent.parent / "src").resolve()),
        )

        # Initialize
        _send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                     "params": {"protocolVersion": "2024-11-05",
                                "capabilities": {},
                                "clientInfo": {"name": "test", "version": "1.0"}}})
        _recv(proc)
        _send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        return proc

    def test_describe_image(self, api_process: subprocess.Popen) -> None:
        """describe_image returns a non-empty description."""
        _send(api_process, {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {
                "name": "describe_image",
                "arguments": {"image": TEST_IMAGE_B64},
            },
        })
        resp = _recv(api_process, timeout=30.0)
        assert resp["id"] == 10
        assert "result" in resp, f"Error: {resp.get('error')}"
        content = resp["result"]["content"]
        assert len(content) > 0
        assert len(content[0]["text"]) > 0

    def test_describe_image_from_file(
        self, api_process: subprocess.Popen, tmp_path: Path
    ) -> None:
        """describe_image_from_file reads a file and returns a description."""
        image_path = tmp_path / "test.png"
        image_path.write_bytes(TEST_PNG_BYTES)

        _send(api_process, {
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {
                "name": "describe_image_from_file",
                "arguments": {"filepath": str(image_path)},
            },
        })
        resp = _recv(api_process, timeout=30.0)
        assert resp["id"] == 11
        assert "result" in resp, f"Error: {resp.get('error')}"
        content = resp["result"]["content"]
        assert len(content) > 0
        assert len(content[0]["text"]) > 0

    def test_invalid_file_path(self, api_process: subprocess.Popen) -> None:
        """describe_image_from_file returns error for nonexistent files."""
        _send(api_process, {
            "jsonrpc": "2.0",
            "id": 12,
            "method": "tools/call",
            "params": {
                "name": "describe_image_from_file",
                "arguments": {"filepath": "/nonexistent/path.png"},
            },
        })
        resp = _recv(api_process, timeout=10.0)
        assert resp["id"] == 12
        assert "error" in resp
