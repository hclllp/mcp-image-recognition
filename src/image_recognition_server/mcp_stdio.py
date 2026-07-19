"""
Minimal MCP (Model Context Protocol) stdio JSON-RPC server.

Replaces the ``mcp`` / ``FastMCP`` package dependency with a stdlib-only
implementation.  This avoids the heavy pydantic≥2.0 → pydantic-core (Rust)
dependency chain that is problematic on Android/Termux and other ARM64
platforms where compiling native extensions is unreliable.

Implements just enough of the MCP 2024-11-05 spec to register tools and
handle ``tools/list`` + ``tools/call`` over stdin/stdout.

Usage::

    from .mcp_stdio import McpServer, ToolDef

    server = McpServer("my-server")

    @server.tool(ToolDef(name="my_tool", description="...", inputSchema={...}))
    async def my_tool(arg1: str) -> str:
        return f"Result: {arg1}"

    server.run()
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSON-RPC 2.0 constants
# ---------------------------------------------------------------------------
JSONRPC_VERSION = "2.0"
MCP_PROTOCOL_VERSION = "2024-11-05"

# Standard JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


# ---------------------------------------------------------------------------
# Tool definition (plain class – no pydantic)
# ---------------------------------------------------------------------------

class ToolDef:
    """Definition of an MCP tool."""

    def __init__(
        self,
        name: str,
        description: str,
        inputSchema: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.description = description
        self.inputSchema = inputSchema or {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.inputSchema,
        }


# ---------------------------------------------------------------------------
# MCP stdio server
# ---------------------------------------------------------------------------

class McpServer:
    """Minimal MCP JSON-RPC server over stdio.

    Registers tool handlers and runs the read/dispatch/respond loop on
    ``sys.stdin`` / ``sys.stdout``.  All logging goes to *stderr* (or a
    log file) so the stdio channel stays clean for JSON-RPC messages.
    """

    def __init__(self, name: str, version: str = "0.1.0"):
        self.name = name
        self.version = version
        self._tools: Dict[str, ToolDef] = {}
        self._handlers: Dict[str, Callable] = {}

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------

    def tool(self, tool_def: ToolDef):
        """Register a tool with its handler (decorator style).

        Usage::

            @server.tool(ToolDef(name="my_tool", description="..."))
            async def my_tool(image: str) -> str:
                ...
        """

        def decorator(fn: Callable) -> Callable:
            self._tools[tool_def.name] = tool_def
            self._handlers[tool_def.name] = fn
            logger.info(f"Registered tool: {tool_def.name}")
            return fn

        return decorator

    # ------------------------------------------------------------------
    # JSON-RPC helpers
    # ------------------------------------------------------------------

    def _make_response(self, req_id: Any, result: Any) -> str:
        return json.dumps(
            {"jsonrpc": JSONRPC_VERSION, "id": req_id, "result": result},
            ensure_ascii=False,
        )

    def _make_error(self, req_id: Any, code: int, message: str) -> str:
        return json.dumps(
            {
                "jsonrpc": JSONRPC_VERSION,
                "id": req_id,
                "error": {"code": code, "message": message},
            },
            ensure_ascii=False,
        )

    def _write(self, data: str) -> None:
        """Write a JSON-RPC message to stdout (one line per message)."""
        sys.stdout.write(data + "\n")
        sys.stdout.flush()

    # ------------------------------------------------------------------
    # Async request dispatch
    # ------------------------------------------------------------------

    async def _dispatch(self, msg: Dict[str, Any]) -> Optional[str]:
        """Route a single JSON-RPC request/notification to its handler."""
        method = msg.get("method", "")
        req_id = msg.get("id")  # None for notifications
        params = msg.get("params", {})

        try:
            if method == "initialize":
                return self._handle_initialize(req_id, params)
            elif method == "notifications/initialized":
                return None  # Notifications get no response
            elif method == "tools/list":
                return self._handle_list_tools(req_id)
            elif method == "tools/call":
                return await self._handle_call_tool(req_id, params)
            elif method == "ping":
                return self._make_response(req_id, {})
            else:
                logger.warning(f"Unknown method: {method}")
                return self._make_error(
                    req_id, METHOD_NOT_FOUND, f"Unknown method: {method}"
                )
        except Exception as exc:
            logger.error(f"Error handling {method}: {exc}", exc_info=True)
            return self._make_error(req_id or 0, INTERNAL_ERROR, str(exc))

    # ------------------------------------------------------------------
    # Lifecycle & tool handlers
    # ------------------------------------------------------------------

    def _handle_initialize(self, req_id: Any, params: Dict[str, Any]) -> str:
        return self._make_response(
            req_id,
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": self.name,
                    "version": self.version,
                },
            },
        )

    def _handle_list_tools(self, req_id: Any) -> str:
        tools = [t.to_dict() for t in self._tools.values()]
        return self._make_response(req_id, {"tools": tools})

    async def _handle_call_tool(self, req_id: Any, params: Dict[str, Any]) -> str:
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        if tool_name not in self._handlers:
            return self._make_error(
                req_id, METHOD_NOT_FOUND, f"Unknown tool: {tool_name}"
            )

        try:
            result = await self._handlers[tool_name](**arguments)
            return self._make_response(
                req_id,
                {"content": [{"type": "text", "text": str(result)}]},
            )
        except Exception as exc:
            logger.error(f"Tool '{tool_name}' error: {exc}", exc_info=True)
            return self._make_error(req_id, INTERNAL_ERROR, str(exc))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the stdio read/dispatch/respond loop (blocking)."""
        import asyncio

        logger.info(
            f"MCP server '{self.name}' v{self.version} starting on stdio"
        )
        logger.info(f"Registered tools: {list(self._tools.keys())}")

        async def _main_loop() -> None:
            # sys.stdin.readline is blocking — run it in a thread to keep
            # the event loop responsive.
            loop = asyncio.get_event_loop()
            while True:
                try:
                    line = await loop.run_in_executor(None, sys.stdin.readline)
                except (EOFError, KeyboardInterrupt):
                    logger.info("Server shutting down")
                    break

                if not line:
                    # EOF — parent process closed stdin
                    logger.info("stdin closed, exiting")
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    msg = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(f"Invalid JSON received: {exc}")
                    self._write(
                        self._make_error(None, PARSE_ERROR, f"Parse error: {exc}")
                    )
                    continue

                response = await self._dispatch(msg)
                if response is not None:
                    self._write(response)

        try:
            asyncio.run(_main_loop())
        except KeyboardInterrupt:
            pass
        finally:
            logger.info("Server stopped")
