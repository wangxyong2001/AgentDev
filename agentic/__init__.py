"""
ReAct Agent — Enterprise Edition.

Package structure:
  config.py          — 12-Factor app configuration (env vars + dataclass)
  exceptions.py      — Exception hierarchy (RecoverableError vs FatalError)
  logging_config.py  — Structured logging (replaces ts_print)
  response_parser.py — Unified parser (eliminates dual-path rot)
  ReActProtocol.yaml — YAML protocol template
  ReActTraceRenderer.py — HTML trace report renderer
  ReActDemo          — Main program (refactored)
"""

__version__ = "3.0.0"