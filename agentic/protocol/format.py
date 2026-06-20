"""
Output Formatter — YAML-driven terminal log and error message formatting.

Handles Section 3 of the ReAct protocol: Agent → User output formatting.

Usage:
  >>> from agentic.protocol.format import OutputFormatter
  >>> fmt = OutputFormatter(yaml_path="agentic/protocol/ReActProtocol.yaml")
  >>> fmt.format("agent_start")         # "[AGENT] === ReAct loop start ==="
  >>> fmt.format("step_header", step=3) # "[STEP 3] ---"
"""

from __future__ import annotations

import os
from typing import Optional, Dict, Any

from agentic.observability import get_logger

logger = get_logger(__name__)


class OutputFormatter:
    """
    Terminal log and output message formatter.

    Reads format templates from YAML protocol file.
    Falls back to built-in defaults when YAML is unavailable.
    """

    def __init__(self, yaml_path: Optional[str] = None):
        self._templates: Dict[str, str] = {}
        self._tool_error_unknown: str = "Unknown action: '{action}'. Available:{tool_list}"
        self._tool_error_exec: str = "Tool error: {error}"

        if yaml_path and os.path.exists(yaml_path):
            self._load_yaml(yaml_path)

    def _load_yaml(self, yaml_path: str) -> None:
        """Load output templates from YAML protocol file."""
        try:
            import yaml
        except ImportError:
            return

        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        output_cfg = cfg.get("output", {})
        self._templates = output_cfg.get("terminal", {})
        self._tool_error_unknown = output_cfg.get(
            "tool_error_unknown", self._tool_error_unknown)
        self._tool_error_exec = output_cfg.get(
            "tool_error_exec", self._tool_error_exec)
        logger.debug("OutputFormatter loaded from YAML")

    def format(self, key: str, **kwargs) -> str:
        """
        Format a log/output message by template key.

        Args:
          key: Template key from YAML output.terminal section
          **kwargs: Placeholder values for the template

        Returns:
          Formatted string, or empty string if key not found
        """
        tpl = self._templates.get(key, "")
        if not tpl:
            return ""
        return tpl.format(**kwargs)

    def tool_error_unknown(self, action: str, tool_list: str) -> str:
        """Format 'unknown tool' error message."""
        return self._tool_error_unknown.format(
            action=action, tool_list=tool_list)

    def tool_error_exec(self, error: str) -> str:
        """Format 'tool execution error' message."""
        return self._tool_error_exec.format(error=error)
