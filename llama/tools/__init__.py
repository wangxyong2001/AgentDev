"""
Tools package — pluggable tool system for ReAct Agent.

Exports:
  Tool          — Tool descriptor dataclass
  ToolRegistry  — Registry for tool registration, lookup, and execution
  calculator_tool — Math calculator
  weather_tool    — Weather lookup (simulated)

Usage:
  >>> from llama.tools import ToolRegistry, calculator_tool, weather_tool
  >>> registry = ToolRegistry()
  >>> registry.register(calculator_tool)
  >>> registry.register(weather_tool)
  >>> registry.execute("calculator", "2+2")
  "4"
"""

from llama.tools.registry import Tool, ToolRegistry
from llama.tools.calculator import calculator_tool
from llama.tools.weather import weather_tool

__all__ = ["Tool", "ToolRegistry", "calculator_tool", "weather_tool"]
