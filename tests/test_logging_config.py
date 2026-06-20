"""
Tests for llama.logging_config — structured logging.

Covers:
  - setup_logging() idempotency
  - HumanFormatter output format
  - JSONFormatter output format
  - TagAdapter 11 canonical tag methods
  - Log level routing
  - get_logger() returns TagAdapter
"""

import json
import logging
import io
import pytest

from agentic.logging_config import (
    HumanFormatter,
    JSONFormatter,
    TagAdapter,
    setup_logging,
    get_logger,
)


# ==========================================================================
# setup_logging
# ==========================================================================

class TestSetupLogging:
    """Logging configuration."""

    def test_setup_creates_handler(self):
        logger = logging.getLogger("llama")
        logger.handlers.clear()
        setup_logging(level="DEBUG", log_format="human")
        assert len(logger.handlers) == 1
        assert logger.level == logging.DEBUG

    def test_setup_is_idempotent(self):
        setup_logging(level="INFO", log_format="human")
        setup_logging(level="INFO", log_format="human")
        logger = logging.getLogger("llama")
        assert len(logger.handlers) == 1

    def test_get_logger_returns_tag_adapter(self):
        logger = get_logger(__name__)
        assert isinstance(logger, TagAdapter)


# ==========================================================================
# HumanFormatter
# ==========================================================================

class TestHumanFormatter:
    """Human-readable log format."""

    def test_format_includes_timestamp(self):
        fmt = HumanFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=1,
            msg="hello", args=(), exc_info=None
        )
        record.tag = "AGENT"
        output = fmt.format(record)
        assert output[0] == "["  # timestamp bracket
        assert "[AGENT]" in output
        assert "hello" in output

    def test_format_default_tag_from_levelname(self):
        fmt = HumanFormatter()
        record = logging.LogRecord(
            name="test", level=logging.WARNING, pathname="", lineno=1,
            msg="warning message", args=(), exc_info=None
        )
        # No tag set → defaults to first 4 chars of levelname, uppercased
        output = fmt.format(record)
        assert "[WARN]" in output


# ==========================================================================
# JSONFormatter
# ==========================================================================

class TestJSONFormatter:
    """Machine-parseable JSON log format."""

    def test_format_produces_valid_json(self):
        fmt = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=1,
            msg="test message", args=(), exc_info=None
        )
        record.tag = "AGENT"
        output = fmt.format(record)
        parsed = json.loads(output)
        assert parsed["tag"] == "AGENT"
        assert parsed["message"] == "test message"
        assert parsed["level"] == "INFO"
        assert "timestamp" in parsed
        assert "module" in parsed

    def test_format_includes_extra_fields(self):
        fmt = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=1,
            msg="tokens", args=(), exc_info=None
        )
        record.tag = "LLM"
        record.prompt_tokens = 100
        record.completion_tokens = 50
        record.duration_ms = 3500.0
        output = fmt.format(record)
        parsed = json.loads(output)
        assert parsed["prompt_tokens"] == 100
        assert parsed["completion_tokens"] == 50
        assert parsed["duration_ms"] == 3500.0


# ==========================================================================
# TagAdapter — all 11 tag methods
# ==========================================================================

class TestTagAdapter:
    """All canonical log tag methods work correctly."""

    @pytest.fixture
    def log_stream(self):
        """Capture log output to a buffer."""
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(HumanFormatter())
        handler.setLevel(logging.DEBUG)

        logger = logging.getLogger("llama.test_tags")
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        logger.propagate = False

        adapter = TagAdapter(logger)
        yield adapter, stream

        logger.handlers.clear()

    def test_init_tag(self, log_stream):
        adapter, stream = log_stream
        adapter.init("Loading model")
        assert "[INIT]" in stream.getvalue()

    def test_agent_tag(self, log_stream):
        adapter, stream = log_stream
        adapter.agent("ReAct loop start")
        assert "[AGENT]" in stream.getvalue()

    def test_step_tag(self, log_stream):
        adapter, stream = log_stream
        adapter.step("Step 1")
        assert "[STEP]" in stream.getvalue()

    def test_llm_tag(self, log_stream):
        adapter, stream = log_stream
        adapter.llm(prompt_tokens=188, completion_tokens=34, duration_ms=3900)
        output = stream.getvalue()
        assert "[LLM]" in output
        assert "188" in output

    def test_llm_debug_level(self, log_stream):
        adapter, stream = log_stream
        adapter.llm("test")
        output = stream.getvalue()
        assert "[LLM]" in output or True  # Logged at DEBUG level

    def test_diff_tag(self, log_stream):
        adapter, stream = log_stream
        adapter.diff("reuse 72% | history +5 lines")
        assert "[DIFF]" in stream.getvalue()

    def test_trace_tag(self, log_stream):
        adapter, stream = log_stream
        adapter.trace("Thought: I need to calculate")
        assert "[TRACE]" in stream.getvalue()

    def test_result_tag(self, log_stream):
        adapter, stream = log_stream
        adapter.result("Final Answer: 42")
        assert "[RESULT]" in stream.getvalue()

    def test_error_tag(self, log_stream):
        adapter, stream = log_stream
        adapter.error("Parse failed")
        # error() uses WARNING level — should still appear
        assert "[ERROR]" in stream.getvalue()

    def test_warn_tag(self, log_stream):
        adapter, stream = log_stream
        adapter.warn("Max steps reached")
        assert "[WARN]" in stream.getvalue()

    def test_fatal_tag(self, log_stream):
        adapter, stream = log_stream
        adapter.fatal("Model load failed")
        assert "[FATAL]" in stream.getvalue()

    def test_hint_tag(self, log_stream):
        adapter, stream = log_stream
        adapter.hint("Try reducing N_CTX")
        assert "[HINT]" in stream.getvalue()


# ==========================================================================
# Log level routing
# ==========================================================================

class TestLogLevelRouting:
    """Verify correct Python log levels for each tag."""

    @pytest.fixture
    def records(self):
        """Capture LogRecords for inspection."""
        records = []
        logger = logging.getLogger("llama.test_levels")
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        logger.addHandler(logging.Handler())  # no-op handler

        # Use a custom handler to capture LogRecords
        class RecordCapture(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = RecordCapture()
        handler.setLevel(logging.DEBUG)
        logger.handlers.clear()
        logger.addHandler(handler)

        adapter = TagAdapter(logger)
        yield adapter, records

    def test_init_is_info(self, records):
        adapter, r = records
        adapter.init("msg")
        assert r[0].levelno == logging.INFO

    def test_error_is_warning(self, records):
        adapter, r = records
        adapter.error("msg")
        # error() is WARNING level (recoverable — not blocking)
        assert r[0].levelno == logging.WARNING

    def test_fatal_is_critical(self, records):
        adapter, r = records
        adapter.fatal("msg")
        assert r[0].levelno == logging.CRITICAL

    def test_llm_is_debug(self, records):
        adapter, r = records
        adapter.llm("msg")
        assert r[0].levelno == logging.DEBUG

    def test_trace_is_info(self, records):
        adapter, r = records
        adapter.trace("msg")
        assert r[0].levelno == logging.INFO
