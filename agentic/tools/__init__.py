"""
Tools package — pluggable tool system for ReAct Agent.

Exports:
  Tool          — Tool descriptor dataclass
  ToolRegistry  — Registry for tool registration, lookup, and execution
  calculator_tool — Math calculator
  weather_tool    — Weather lookup (simulated)

Usage:
  >>> from agentic.tools import ToolRegistry, calculator_tool, weather_tool
  >>> registry = ToolRegistry()
  >>> registry.register(calculator_tool)
  >>> registry.register(weather_tool)
  >>> registry.execute("calculator", "2+2")
  "4"
"""

from agentic.tools.registry import Tool, ToolRegistry
from agentic.tools.calculator import calculator_tool
from agentic.tools.weather import weather_tool

__all__ = ["Tool", "ToolRegistry", "calculator_tool", "weather_tool"]
