"""
Tool Registry — Agent tool management with dependency injection.

Replaces the global TOOLS list with a registry pattern. Supports:
  - Dynamic tool registration
  - Tool lookup by name (exact match)
  - Safe execution with error handling
  - Tool metadata (description, validation schema placeholder)

Usage:
  >>> from agentic.tools.registry import ToolRegistry
  >>> from agentic.tools.calculator import calculator_tool
  >>> from agentic.tools.weather import weather_tool
  >>> registry = ToolRegistry()
  >>> registry.register(calculator_tool)
  >>> registry.register(weather_tool)
  >>> registry.execute("calculator", "2 + 2")
  "4"
"""

from __future__ import annotations

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field

from agentic.exceptions import ToolNotFoundError, ToolExecutionError
from agentic.observability import get_logger

logger = get_logger(__name__)


# ==========================================================================
# Tool descriptor
# ==========================================================================

@dataclass
class Tool:
    """
    Tool descriptor — the single source of truth for a registered tool.

    Replaces the langchain @tool dependency with a framework-agnostic
    descriptor. LangChain compatibility is preserved via `invoke()`.

    Attributes:
      name:        Unique tool identifier (must match LLM Action value)
      description: Tool description shown in system prompt
      func:        Callable (accepts str, returns str)
      metadata:    Additional metadata (version, timeout_sec, etc.)
    """

    name: str
    description: str
    func: Callable[[str], str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def invoke(self, input_str: str) -> str:
        """Invoke the tool with the given input string."""
        return self.func(input_str)


# ==========================================================================
# Tool Registry
# ==========================================================================

class ToolRegistry:
    """
    Registry for managing tool registration, lookup, and execution.

    Design:
      - Isolated from LLM and Agent — pure tool management
      - Injectable — tests can inject mock tools without global state
      - Safe execution — all tool calls are wrapped in try/except

    Usage:
      >>> registry = ToolRegistry()
      >>> registry.register(Tool("calc", "Calculator tool", lambda x: str(eval(x))))
      >>> registry.list_names()
      ["calc"]
      >>> registry.execute("calc", "2+2")
      "4"
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    # ── Registration ────────────────────────────────────────────────

    def register(self, tool: Tool) -> None:
        """
        Register a tool.

        Args:
          tool: Tool descriptor

        Raises:
          ValueError: Tool with the same name already registered
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        logger.debug(f"Tool registered: {tool.name}")

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry."""
        if name in self._tools:
            del self._tools[name]
            logger.debug(f"Tool unregistered: {name}")

    # ── Lookup ──────────────────────────────────────────────────────

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name. Returns None if not found."""
        return self._tools.get(name)

    def list_tools(self) -> List[Tool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def list_names(self) -> List[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    # ── Execution ───────────────────────────────────────────────────

    def execute(self, name: str, input_str: str) -> str:
        """
        Execute a tool by name with given input.

        Args:
          name:      Tool name (must be registered)
          input_str: Tool input string (from LLM)

        Returns:
          Tool execution result string

        Raises:
          ToolNotFoundError:  Tool not registered
          ToolExecutionError: Tool invocation raised an exception
        """
        tool = self._tools.get(name)
        if tool is None:
            available = self.list_names()
            raise ToolNotFoundError(name, available)

        try:
            result = tool.invoke(input_str)
            return str(result)
        except Exception as e:
            raise ToolExecutionError(name, input_str, e)

    # ── Formatting ─────────────────────────────────────────────────

    def format_for_prompt(self) -> str:
        """
        Format all tools for system prompt display.

        Returns:
          Multi-line string: "- tool_name: description"
        """
        return "\n".join(
            f"- {t.name}: {t.description}"
            for t in self._tools.values()
        )

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
