"""
Enterprise Exception Hierarchy for ReAct Agent.

Design principles:
  1. Clear boundary: RecoverableError (retry in-loop) vs FatalError (exit process)
  2. Each exception carries enough context for logging and debugging
  3. Integration with structured logging (exceptions serialize to JSON)

Usage:
  >>> from llama.exceptions import ParseError, ToolNotFoundError, FatalError
  >>> try:
  ...     parsed = parser.parse(output)
  ... except ParseError as e:
  ...     logger.warning("Parse failed, will retry", extra=e.as_dict())

Exception tree:

  ReActError (base)
  ├── RecoverableError        ← catch in-loop, trigger Error Recovery
  │   ├── ParseError           — LLM output doesn't match ReAct format
  │   ├── ToolNotFoundError    — model requested non-existent tool
  │   └── ToolExecutionError   — tool.invoke() raised an exception
  └── FatalError              ← catch in main(), call sys.exit(1)
      ├── ConfigError           — invalid configuration
      ├── ModelLoadError        — LLM initialization failed
      └── DependencyError       — missing required package
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List


# ==========================================================================
# Base Exception
# ==========================================================================

class ReActError(Exception):
    """Base exception for all ReAct Agent errors."""

    def __init__(self, message: str, *, code: str = "REACT_ERR", detail: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.detail = detail or {}

    def as_dict(self) -> Dict[str, Any]:
        """Serialize to structured dict for JSON logging."""
        return {
            "exception_type": self.__class__.__name__,
            "code": self.code,
            "message": str(self),
            "detail": self.detail,
        }


# ==========================================================================
# Recoverable Errors (in-loop retry via Error Recovery)
# ==========================================================================

class RecoverableError(ReActError):
    """
    Base class for errors the ReAct loop can recover from.

    Recovery strategy: inject the error as Observation, let LLM retry.
    After max_steps consecutive failures, the loop terminates safely.
    """
    pass


class ParseError(RecoverableError):
    """
    LLM output cannot be parsed as valid ReAct format.

    Triggered when all four parsing levels (P1~P4) and fallback strategies fail.
    The raw LLM output is injected as Observation for the next retry.
    """

    def __init__(self, raw_output: str, details: str = ""):
        super().__init__(
            message=f"Failed to parse LLM output. Got: {raw_output[:200]}",
            code="REACT_PARSE_ERR",
            detail={"raw_output_snippet": raw_output[:200], "details": details},
        )
        self.raw_output = raw_output


class ToolNotFoundError(RecoverableError):
    """
    LLM requested a tool name that isn't registered.

    Observation includes the list of valid tool names to guide the model.
    """

    def __init__(self, action: str, available_tools: List[str]):
        super().__init__(
            message=f"Unknown action: '{action}'. Available: {available_tools}",
            code="REACT_TOOL_NOT_FOUND",
            detail={"requested": action, "available": available_tools},
        )
        self.action = action
        self.available_tools = available_tools


class ToolExecutionError(RecoverableError):
    """
    Tool invocation raised an unhandled exception.

    The exception message is captured as Observation for the model.
    """

    def __init__(self, tool_name: str, input_str: str, original_error: Exception):
        super().__init__(
            message=f"Tool '{tool_name}' execution error: {original_error}",
            code="REACT_TOOL_EXEC_ERR",
            detail={
                "tool_name": tool_name,
                "input": input_str,
                "original_error": str(original_error),
            },
        )
        self.tool_name = tool_name
        self.original_error = original_error


# ==========================================================================
# Fatal Errors (process exits immediately)
# ==========================================================================

class FatalError(ReActError):
    """
    Base class for unrecoverable errors.

    These terminate the process with sys.exit(1).
    A [HINT] message accompanies each FatalError with remediation steps.
    """
    hint: str = ""

    def __init__(self, message: str, *, code: str = "REACT_FATAL", hint: str = "", detail: Optional[Dict[str, Any]] = None):
        super().__init__(message, code=code, detail=detail)
        self.hint = hint


class ConfigError(FatalError):
    """Configuration validation failed (model_path missing, invalid values)."""

    def __init__(self, message: str):
        super().__init__(
            message=message,
            code="REACT_CONFIG_ERR",
            hint="Check REACT_MODEL_PATH and other REACT_* environment variables.",
        )


class ModelLoadError(FatalError):
    """LLM model failed to load (CUDA OOM, file corrupt, etc.)."""

    def __init__(self, message: str):
        super().__init__(
            message=message,
            code="REACT_MODEL_LOAD_ERR",
            hint="If CUDA OOM, reduce REACT_N_CTX or set REACT_N_GPU_LAYERS=0 (CPU-only).",
        )


class DependencyError(FatalError):
    """Required Python package is not installed."""

    def __init__(self, package: str, install_cmd: str):
        super().__init__(
            message=f"Missing dependency: {package}",
            code="REACT_DEPENDENCY_ERR",
            hint=f"Run: {install_cmd}",
            detail={"package": package, "install_command": install_cmd},
        )