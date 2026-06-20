"""
Tests for llama.exceptions — exception hierarchy and serialization.

Covers:
  - Exception construction with code and detail
  - as_dict() serialization for structured logging
  - ParseError special attributes
  - ToolNotFoundError available_tools list
  - ToolExecutionError original_error capture
  - FatalError hint propagation
  - isinstance relationships (RecoverableError vs FatalError)
"""

import pytest

from llama.exceptions import (
    ReActError,
    RecoverableError,
    ParseError,
    ToolNotFoundError,
    ToolExecutionError,
    FatalError,
    ConfigError,
    ModelLoadError,
    DependencyError,
)


# ==========================================================================
# Base Exception
# ==========================================================================

class TestReActError:
    """Base exception construction and serialization."""

    def test_basic_construction(self):
        e = ReActError("test message")
        assert str(e) == "test message"
        assert e.code == "REACT_ERR"
        assert e.detail == {}

    def test_with_code(self):
        e = ReActError("msg", code="CUSTOM_CODE")
        assert e.code == "CUSTOM_CODE"

    def test_with_detail(self):
        e = ReActError("msg", detail={"key": "value", "count": 42})
        assert e.detail == {"key": "value", "count": 42}

    def test_as_dict(self):
        e = ReActError("fail", code="ERR_001", detail={"line": 10})
        d = e.as_dict()
        assert d["exception_type"] == "ReActError"
        assert d["code"] == "ERR_001"
        assert d["message"] == "fail"
        assert d["detail"]["line"] == 10

    def test_as_dict_without_detail(self):
        e = ReActError("simple")
        d = e.as_dict()
        assert d["detail"] == {}


# ==========================================================================
# ParseError (Recoverable)
# ==========================================================================

class TestParseError:
    """ParseError — recoverable, carries raw LLM output."""

    def test_basic_parse_error(self):
        e = ParseError("garbled output", details="No Action found")
        assert isinstance(e, RecoverableError)
        assert isinstance(e, ReActError)
        assert e.code == "REACT_PARSE_ERR"
        assert e.raw_output == "garbled output"
        assert "garbled output" in str(e)
        assert e.detail["details"] == "No Action found"

    def test_snippet_truncated(self):
        long_output = "x" * 500
        e = ParseError(long_output)
        assert len(e.detail["raw_output_snippet"]) == 200

    def test_as_dict(self):
        e = ParseError("bad output", details="missing action")
        d = e.as_dict()
        assert d["exception_type"] == "ParseError"
        assert d["detail"]["raw_output_snippet"] == "bad output"


# ==========================================================================
# ToolNotFoundError (Recoverable)
# ==========================================================================

class TestToolNotFoundError:
    """ToolNotFoundError — model requested non-existent tool."""

    def test_basic(self):
        e = ToolNotFoundError("fly_to_moon", ["calculator", "get_weather"])
        assert isinstance(e, RecoverableError)
        assert e.action == "fly_to_moon"
        assert e.available_tools == ["calculator", "get_weather"]
        assert "fly_to_moon" in str(e)
        assert "calculator" in str(e)

    def test_as_dict(self):
        e = ToolNotFoundError("xyz", ["a", "b"])
        d = e.as_dict()
        assert d["detail"]["requested"] == "xyz"
        assert d["detail"]["available"] == ["a", "b"]


# ==========================================================================
# ToolExecutionError (Recoverable)
# ==========================================================================

class TestToolExecutionError:
    """ToolExecutionError — tool.invoke() raised."""

    def test_basic(self):
        original = ValueError("division by zero")
        e = ToolExecutionError("calculator", "1/0", original)
        assert isinstance(e, RecoverableError)
        assert e.tool_name == "calculator"
        assert "division by zero" in str(e)
        assert e.detail["original_error"] == "division by zero"

    def test_as_dict(self):
        original = RuntimeError("timeout")
        e = ToolExecutionError("get_weather", "Mars", original)
        d = e.as_dict()
        assert d["detail"]["tool_name"] == "get_weather"
        assert d["detail"]["input"] == "Mars"


# ==========================================================================
# FatalError hierarchy
# ==========================================================================

class TestFatalErrors:
    """Fatal errors — all carry hint messages."""

    def test_config_error(self):
        e = ConfigError("Invalid model path")
        assert isinstance(e, FatalError)
        assert "REACT_MODEL_PATH" in e.hint

    def test_model_load_error(self):
        e = ModelLoadError("CUDA OOM")
        assert isinstance(e, FatalError)
        assert "REACT_N_CTX" in e.hint

    def test_dependency_error(self):
        e = DependencyError("pyyaml", "pip install pyyaml")
        assert isinstance(e, FatalError)
        assert "pip install pyyaml" in e.hint
        assert e.detail["package"] == "pyyaml"

    def test_fatal_is_not_recoverable(self):
        e = ModelLoadError("fail")
        assert not isinstance(e, RecoverableError)
        assert isinstance(e, ReActError)

    def test_recoverable_is_not_fatal(self):
        e = ParseError("bad")
        assert isinstance(e, RecoverableError)
        assert not isinstance(e, FatalError)
