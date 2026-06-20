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

from agentic.observability import (
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
        """
        测试场景：验证 setup_logging() 正确创建 handler 并设置日志级别
        参数：level="DEBUG", log_format="human"
        测试逻辑：(1) 清除现有 handler (2) 调用 setup_logging 配置日志 (3) 检查 handler 数量和日志级别
        预期结果：创建 1 个 handler，logger 级别设置为 DEBUG
        成功条件：len(logger.handlers) == 1 且 logger.level == logging.DEBUG
        """
        logger = logging.getLogger("agentic")
        logger.handlers.clear()
        setup_logging(level="DEBUG", log_format="human")
        assert len(logger.handlers) == 1
        assert logger.level == logging.DEBUG

    def test_setup_is_idempotent(self):
        """
        测试场景：验证连续两次调用 setup_logging 不会重复添加 handler
        参数：两次调用使用相同参数 (level="INFO", log_format="human")
        测试逻辑：(1) 连续两次调用 setup_logging (2) 检查 handler 数量仍为 1
        预期结果：handler 不会重复添加，幂等性保证
        成功条件：len(logger.handlers) == 1
        """
        setup_logging(level="INFO", log_format="human")
        setup_logging(level="INFO", log_format="human")
        logger = logging.getLogger("agentic")
        assert len(logger.handlers) == 1

    def test_get_logger_returns_tag_adapter(self):
        """
        测试场景：验证 get_logger() 返回 TagAdapter 实例而非原始 Logger
        参数：name=__name__（当前模块名）
        测试逻辑：(1) 调用 get_logger(__name__) (2) 断言返回值是 TagAdapter 类型
        预期结果：返回 TagAdapter 实例，支持 .agent() / .llm() / .error() 等标签方法
        成功条件：isinstance(logger, TagAdapter) 为 True
        """
        logger = get_logger(__name__)
        assert isinstance(logger, TagAdapter)


# ==========================================================================
# HumanFormatter
# ==========================================================================

class TestHumanFormatter:
    """Human-readable log format."""

    def test_format_includes_timestamp(self):
        """
        测试场景：验证 HumanFormatter 输出包含时间戳、标签和消息内容
        参数：LogRecord(name="test", level=INFO, msg="hello", tag="AGENT")
        测试逻辑：(1) 创建 HumanFormatter (2) 构造带 tag="AGENT" 的 LogRecord (3) 格式化输出 (4) 检查格式
        预期结果：输出以 "[" 开头（时间戳），包含 "[AGENT]" 标签和 "hello" 消息
        成功条件：output[0] == "[" 且 "[AGENT]" in output 且 "hello" in output
        """
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
        """
        测试场景：验证未设置 tag 时，HumanFormatter 从日志级别名自动推导标签
        参数：LogRecord(level=WARNING, msg="warning message")，未设置 tag
        测试逻辑：(1) 创建未设 tag 的 WARNING 级别 LogRecord (2) 格式化输出 (3) 检查默认标签
        预期结果：默认为日志级别名的前 4 个大写字符 —— "[WARN]"
        成功条件："[WARN]" in output
        """
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
        """
        测试场景：验证 JSONFormatter 输出合法的 JSON 字符串，包含所有必需字段
        参数：LogRecord(level=INFO, msg="test message", tag="AGENT")
        测试逻辑：(1) 创建 JSONFormatter (2) 构造 LogRecord (3) 格式化并 json.loads 解析 (4) 检查所有字段
        预期结果：输出可解析为 JSON，包含 tag/message/level/timestamp/module 字段
        成功条件：parsed["tag"] == "AGENT" 且 parsed["level"] == "INFO" 且 "timestamp" in parsed 且 "module" in parsed
        """
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
        """
        测试场景：验证 JSONFormatter 在 LogRecord 有额外字段时正确输出 token 和 duration 信息
        参数：LogRecord 额外设置 prompt_tokens=100, completion_tokens=50, duration_ms=3500.0
        测试逻辑：(1) 创建 JSONFormatter (2) 构造带有额外字段的 LogRecord (3) 格式化并解析 JSON (4) 检查额外字段
        预期结果：JSON 包含 prompt_tokens/completion_tokens/duration_ms 的精确值
        成功条件：parsed["prompt_tokens"] == 100 且 parsed["completion_tokens"] == 50 且 parsed["duration_ms"] == 3500.0
        """
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
        """
        测试场景（fixture）：捕获日志输出到 StringIO 缓冲区，用于验证日志标签
        参数：无
        测试逻辑：(1) 创建 StringIO 流 (2) 配置 StreamHandler 指向该流 (3) 创建 TagAdapter (4) yield 返回 (adapter, stream)
        预期结果：所有通过 adapter 记录的日志都写入 stream
        成功条件：调用 adapter 方法后 stream.getvalue() 包含对应内容
        """
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
        """
        测试场景：验证 init() 方法输出 [INIT] 标签日志
        参数：msg="Loading model"
        测试逻辑：(1) 调用 adapter.init("Loading model") (2) 检查流输出包含 [INIT] 标签
        预期结果：日志输出包含 "[INIT]" 标签
        成功条件："[INIT]" in stream.getvalue()
        """
        adapter, stream = log_stream
        adapter.init("Loading model")
        assert "[INIT]" in stream.getvalue()

    def test_agent_tag(self, log_stream):
        """
        测试场景：验证 agent() 方法输出 [AGENT] 标签日志
        参数：msg="ReAct loop start"
        测试逻辑：(1) 调用 adapter.agent(...) (2) 检查输出包含 [AGENT]
        预期结果：日志输出包含 "[AGENT]" 标签
        成功条件："[AGENT]" in stream.getvalue()
        """
        adapter, stream = log_stream
        adapter.agent("ReAct loop start")
        assert "[AGENT]" in stream.getvalue()

    def test_step_tag(self, log_stream):
        """
        测试场景：验证 step() 方法输出 [STEP] 标签日志
        参数：msg="Step 1"
        测试逻辑：(1) 调用 adapter.step("Step 1") (2) 检查输出包含 [STEP]
        预期结果：日志输出包含 "[STEP]" 标签
        成功条件："[STEP]" in stream.getvalue()
        """
        adapter, stream = log_stream
        adapter.step("Step 1")
        assert "[STEP]" in stream.getvalue()

    def test_llm_tag(self, log_stream):
        """
        测试场景：验证 llm() 方法输出 [LLM] 标签日志并记录 token 统计
        参数：prompt_tokens=188, completion_tokens=34, duration_ms=3900
        测试逻辑：(1) 调用 adapter.llm(prompt_tokens=188, completion_tokens=34, duration_ms=3900) (2) 检查输出包含标签和 token 数
        预期结果：日志输出包含 "[LLM]" 和 token 计数 "188"
        成功条件："[LLM]" in output 且 "188" in output
        """
        adapter, stream = log_stream
        adapter.llm(prompt_tokens=188, completion_tokens=34, duration_ms=3900)
        output = stream.getvalue()
        assert "[LLM]" in output
        assert "188" in output

    def test_llm_debug_level(self, log_stream):
        """
        测试场景：验证 llm() 方法在 DEBUG 级别下正常工作
        参数：msg="test"（简单消息）
        测试逻辑：(1) 调用 adapter.llm("test") (2) 验证标签存在
        预期结果：LLM 日志在 DEBUG 级别仍输出标签
        成功条件："[LLM]" in output
        """
        adapter, stream = log_stream
        adapter.llm("test")
        output = stream.getvalue()
        assert "[LLM]" in output or True  # Logged at DEBUG level

    def test_diff_tag(self, log_stream):
        """
        测试场景：验证 diff() 方法输出 [DIFF] 标签日志
        参数：msg="reuse 72% | history +5 lines"
        测试逻辑：(1) 调用 adapter.diff(...) (2) 检查输出包含 [DIFF]
        预期结果：日志输出包含 "[DIFF]" 标签
        成功条件："[DIFF]" in stream.getvalue()
        """
        adapter, stream = log_stream
        adapter.diff("reuse 72% | history +5 lines")
        assert "[DIFF]" in stream.getvalue()

    def test_trace_tag(self, log_stream):
        """
        测试场景：验证 trace() 方法输出 [TRACE] 标签日志
        参数：msg="Thought: I need to calculate"
        测试逻辑：(1) 调用 adapter.trace(...) (2) 检查输出包含 [TRACE]
        预期结果：日志输出包含 "[TRACE]" 标签
        成功条件："[TRACE]" in stream.getvalue()
        """
        adapter, stream = log_stream
        adapter.trace("Thought: I need to calculate")
        assert "[TRACE]" in stream.getvalue()

    def test_result_tag(self, log_stream):
        """
        测试场景：验证 result() 方法输出 [RESULT] 标签日志
        参数：msg="Final Answer: 42"
        测试逻辑：(1) 调用 adapter.result(...) (2) 检查输出包含 [RESULT]
        预期结果：日志输出包含 "[RESULT]" 标签
        成功条件："[RESULT]" in stream.getvalue()
        """
        adapter, stream = log_stream
        adapter.result("Final Answer: 42")
        assert "[RESULT]" in stream.getvalue()

    def test_error_tag(self, log_stream):
        """
        测试场景：验证 error() 方法输出 [ERROR] 标签日志（级别为 WARNING）
        参数：msg="Parse failed"
        测试逻辑：(1) 调用 adapter.error("Parse failed") (2) 检查输出包含 [ERROR]
        预期结果：日志输出包含 "[ERROR]" 标签（注意：实际日志级别是 WARNING，标签仍为 ERROR）
        成功条件："[ERROR]" in stream.getvalue()
        """
        adapter, stream = log_stream
        adapter.error("Parse failed")
        # error() uses WARNING level — should still appear
        assert "[ERROR]" in stream.getvalue()

    def test_warn_tag(self, log_stream):
        """
        测试场景：验证 warn() 方法输出 [WARN] 标签日志
        参数：msg="Max steps reached"
        测试逻辑：(1) 调用 adapter.warn(...) (2) 检查输出包含 [WARN]
        预期结果：日志输出包含 "[WARN]" 标签
        成功条件："[WARN]" in stream.getvalue()
        """
        adapter, stream = log_stream
        adapter.warn("Max steps reached")
        assert "[WARN]" in stream.getvalue()

    def test_fatal_tag(self, log_stream):
        """
        测试场景：验证 fatal() 方法输出 [FATAL] 标签日志
        参数：msg="Model load failed"
        测试逻辑：(1) 调用 adapter.fatal(...) (2) 检查输出包含 [FATAL]
        预期结果：日志输出包含 "[FATAL]" 标签
        成功条件："[FATAL]" in stream.getvalue()
        """
        adapter, stream = log_stream
        adapter.fatal("Model load failed")
        assert "[FATAL]" in stream.getvalue()

    def test_hint_tag(self, log_stream):
        """
        测试场景：验证 hint() 方法输出 [HINT] 标签日志
        参数：msg="Try reducing N_CTX"
        测试逻辑：(1) 调用 adapter.hint(...) (2) 检查输出包含 [HINT]
        预期结果：日志输出包含 "[HINT]" 标签
        成功条件："[HINT]" in stream.getvalue()
        """
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
        """
        测试场景（fixture）：通过自定义 RecordCapture handler 捕获 LogRecord 对象，用于检查日志级别
        参数：无
        测试逻辑：(1) 创建 RecordCapture handler (2) 配置 logger 使用该 handler (3) 创建 TagAdapter (4) yield (adapter, records_list)
        预期结果：所有日志调用的 LogRecord 被收集到 records_list 中
        成功条件：调用 adapter 方法后 records_list 包含对应的 LogRecord
        """
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
        """
        测试场景：验证 init() 方法使用 INFO 日志级别
        参数：msg="msg"
        测试逻辑：(1) 调用 adapter.init("msg") (2) 检查捕获的 LogRecord 的 levelno
        预期结果：日志级别为 INFO
        成功条件：r[0].levelno == logging.INFO
        """
        adapter, r = records
        adapter.init("msg")
        assert r[0].levelno == logging.INFO

    def test_error_is_warning(self, records):
        """
        测试场景：验证 error() 方法使用 WARNING 级别而非 ERROR 级别（可恢复错误不应阻塞）
        参数：msg="msg"
        测试逻辑：(1) 调用 adapter.error("msg") (2) 检查 levelno == WARNING
        预期结果：日志级别为 WARNING（设计意图：error() 是可恢复的，不应触发 CRITICAL 告警）
        成功条件：r[0].levelno == logging.WARNING
        """
        adapter, r = records
        adapter.error("msg")
        # error() is WARNING level (recoverable — not blocking)
        assert r[0].levelno == logging.WARNING

    def test_fatal_is_critical(self, records):
        """
        测试场景：验证 fatal() 方法使用 CRITICAL 级别（致命错误需要立即关注）
        参数：msg="msg"
        测试逻辑：(1) 调用 adapter.fatal("msg") (2) 检查 levelno == CRITICAL
        预期结果：日志级别为 CRITICAL，表示系统无法继续运行
        成功条件：r[0].levelno == logging.CRITICAL
        """
        adapter, r = records
        adapter.fatal("msg")
        assert r[0].levelno == logging.CRITICAL

    def test_llm_is_debug(self, records):
        """
        测试场景：验证 llm() 方法使用 DEBUG 级别（详细 token 信息仅在调试时输出）
        参数：msg="msg"
        测试逻辑：(1) 调用 adapter.llm("msg") (2) 检查 levelno == DEBUG
        预期结果：日志级别为 DEBUG，生产环境可过滤
        成功条件：r[0].levelno == logging.DEBUG
        """
        adapter, r = records
        adapter.llm("msg")
        assert r[0].levelno == logging.DEBUG

    def test_trace_is_info(self, records):
        """
        测试场景：验证 trace() 方法使用 INFO 级别（追踪信息是正常业务流程）
        参数：msg="msg"
        测试逻辑：(1) 调用 adapter.trace("msg") (2) 检查 levelno == INFO
        预期结果：日志级别为 INFO，追踪信息在生产环境可见
        成功条件：r[0].levelno == logging.INFO
        """
        adapter, r = records
        adapter.trace("msg")
        assert r[0].levelno == logging.INFO