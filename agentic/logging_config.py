"""
Structured Logging — replaces ts_print() with Python standard logging.

Features:
  - 10 canonical log tags: INIT/AGENT/STEP/LLM/DIFF/TRACE/RESULT/ERROR/WARN/FATAL/HINT
  - Human-readable format (development) + JSON format (production)
  - Millisecond-precision timestamps
  - Log level filtering by environment (REACT_LOG_LEVEL)
  - Tag-aware LoggerAdapter for consistent formatting

Usage:
  >>> from agentic.logging_config import get_logger
  >>> logger = get_logger(__name__)
  >>> logger.agent("ReAct loop start")
  >>> logger.llm(prompt_tokens=188, completion_tokens=34, duration_ms=3914)
  >>> logger.error("Parse failed", extra={"error": str(e)})

Log Format (human):
  [HH:MM:SS.mmm] [TAG] message

Log Format (json):
  {"timestamp": "2026-06-21T02:42:31.152", "level": "INFO", "tag": "AGENT",
   "message": "ReAct loop start", "module": "llama.ReActDemo", ...}
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional


# ==========================================================================
# Custom Formatters
# ==========================================================================

class HumanFormatter(logging.Formatter):
    """
    Human-readable format: [HH:MM:SS.mmm] [TAG] message

    Designed for terminal developers — easy to scan, grep-friendly.
    """

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Millisecond precision
        tag = getattr(record, "tag", record.levelname[:4].upper())
        return f"[{ts}] [{tag}] {record.getMessage()}"


class JSONFormatter(logging.Formatter):
    """
    Machine-parseable JSON format for production log aggregation.

    Compatible with: ELK stack, Datadog, Splunk, Grafana Loki.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "tag": getattr(record, "tag", record.levelname),
            "message": record.getMessage(),
            "module": record.name,
        }
        # Merge any extra fields passed via logger.xxx(msg, extra={...})
        for key in ("error", "prompt_tokens", "completion_tokens", "duration_ms",
                     "step", "thought", "action", "action_input", "observation"):
            val = getattr(record, key, None)
            if val is not None:
                payload[key] = val
        return json.dumps(payload, ensure_ascii=False)


# ==========================================================================
# Tag-Aware Logger Adapter
# ==========================================================================

class TagAdapter(logging.LoggerAdapter):
    """
    LoggerAdapter that injects a [TAG] into every log record.

    Provides convenience methods for each of the 10 canonical tags,
    plus structured kwargs for LLM token/duration data.
    """

    def __init__(self, logger: logging.Logger):
        super().__init__(logger, {"tag": "INFO"})

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        extra = kwargs.pop("extra", {})
        if self.extra:
            for key, value in self.extra.items():
                if key not in extra:
                    extra[key] = value
        kwargs["extra"] = extra
        return msg, kwargs

    def _log_with_tag(self, level: int, tag: str, msg: str, **kwargs):
        extra = kwargs.pop("extra", {})
        extra["tag"] = tag
        # Forward structured fields to the record
        for k, v in kwargs.items():
            extra[k] = v
        self.log(level, msg, extra=extra)

    # ── Canonical tag methods ──────────────────────────────────────

    def init(self, msg: str, **kwargs):
        """[INIT] System initialization — model loading."""
        self._log_with_tag(logging.INFO, "INIT", msg, **kwargs)

    def agent(self, msg: str, **kwargs):
        """[AGENT] Agent session lifecycle — start/question."""
        self._log_with_tag(logging.INFO, "AGENT", msg, **kwargs)

    def step(self, msg: str, **kwargs):
        """[STEP] ReAct loop iteration boundary."""
        self._log_with_tag(logging.INFO, "STEP", msg, **kwargs)

    def llm(self, msg: str = "", *, prompt_tokens: int = 0, completion_tokens: int = 0,
            duration_ms: float = 0, raw_output: str = "", **kwargs):
        """[LLM] LLM inference call — tokens + duration."""
        text = msg or f"prompt={prompt_tokens}  completion={completion_tokens}  duration={duration_ms:.0f}ms"
        if raw_output:
            text = f"Raw: {raw_output[:200]}\n{text}"
        self._log_with_tag(logging.DEBUG, "LLM", text,
                           prompt_tokens=prompt_tokens,
                           completion_tokens=completion_tokens,
                           duration_ms=duration_ms, **kwargs)

    def diff(self, msg: str, **kwargs):
        """[DIFF] Cross-turn data difference summary."""
        self._log_with_tag(logging.DEBUG, "DIFF", msg, **kwargs)

    def trace(self, msg: str, **kwargs):
        """[TRACE] Agent state — Thought/Action/Observation."""
        self._log_with_tag(logging.INFO, "TRACE", msg, **kwargs)

    def result(self, msg: str, **kwargs):
        """[RESULT] Final answer or function return."""
        self._log_with_tag(logging.INFO, "RESULT", msg, **kwargs)

    def error(self, msg: str, **kwargs):
        """[ERROR] Recoverable error (in-loop retry)."""
        self._log_with_tag(logging.WARNING, "ERROR", msg, **kwargs)

    def warn(self, msg: str, **kwargs):
        """[WARN] Warning threshold exceeded."""
        self._log_with_tag(logging.WARNING, "WARN", msg, **kwargs)

    def fatal(self, msg: str, **kwargs):
        """[FATAL] Unrecoverable error — process will exit."""
        self._log_with_tag(logging.CRITICAL, "FATAL", msg, **kwargs)

    def hint(self, msg: str, **kwargs):
        """[HINT] Remediation suggestion — paired with FATAL."""
        self._log_with_tag(logging.INFO, "HINT", msg, **kwargs)


# ==========================================================================
# Logger Factory
# ==========================================================================

def setup_logging(level: str = "INFO", log_format: str = "human") -> None:
    """
    Configure the root logger for the ReAct Agent.

    Args:
      level:   DEBUG | INFO | WARNING | ERROR | CRITICAL
      log_format: "human" (terminal-friendly) or "json" (machine-parseable)

    Called once at process start (main.py). Idempotent.
    """
    root = logging.getLogger("llama")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to ensure idempotency
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)  # Handler passes everything; logger filters

    if log_format == "json":
        formatter = JSONFormatter()
    else:
        formatter = HumanFormatter()

    handler.setFormatter(formatter)
    root.addHandler(handler)


def get_logger(name: str) -> TagAdapter:
    """
    Get a TagAdapter-wrapped logger for the given module name.

    Args:
      name: usually __name__ from the calling module

    Returns:
      TagAdapter with canonical tag methods.

    Usage:
      >>> from agentic.logging_config import get_logger
      >>> logger = get_logger(__name__)
      >>> logger.agent("ReAct loop start")
      >>> logger.llm(prompt_tokens=188, completion_tokens=34, duration_ms=3914.0)
    """
    logger = logging.getLogger(name)
    return TagAdapter(logger)


# ── Auto-initialize from environment (if not already configured) ──

if not logging.getLogger("llama").handlers:
    _level = os.getenv("REACT_LOG_LEVEL", "INFO")
    _format = os.getenv("REACT_LOG_FORMAT", "human")
    setup_logging(level=_level, log_format=_format)