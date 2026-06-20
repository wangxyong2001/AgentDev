"""
Protocol loader — YAML-driven prompt template rendering and output formatting.

Splits the monolithic ProtocolLoader into two concerns:
  template.py  — Prompt assembly (system/user/history + full prompt render)
  format.py    — Output formatting (terminal log templates, error messages)

Parsing is delegated to llama.response_parser.ResponseParser (unified).
"""

from agentic.protocol.template import PromptTemplate
from agentic.protocol.format import OutputFormatter

__all__ = ["PromptTemplate", "OutputFormatter"]
