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

from agentic.exceptions import (
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
        """
        测试场景：验证 ReActError 最基本的构造行为 —— 传入消息字符串
        参数：message="test message"
        测试逻辑：(1) 构造 ReActError("test message") (2) 检查 str(e)、e.code、e.detail
        预期结果：str() 返回传入的消息，code 默认为 "REACT_ERR"，detail 默认为空字典
        成功条件：str(e) == "test message" 且 e.code == "REACT_ERR" 且 e.detail == {}
        """
        e = ReActError("test message")
        assert str(e) == "test message"
        assert e.code == "REACT_ERR"
        assert e.detail == {}

    def test_with_code(self):
        """
        测试场景：验证可以通过 code 参数自定义错误码
        参数：message="msg", code="CUSTOM_CODE"
        测试逻辑：(1) 构造 ReActError("msg", code="CUSTOM_CODE") (2) 断言 e.code == "CUSTOM_CODE"
        预期结果：错误码被正确设置为自定义值
        成功条件：e.code == "CUSTOM_CODE"
        """
        e = ReActError("msg", code="CUSTOM_CODE")
        assert e.code == "CUSTOM_CODE"

    def test_with_detail(self):
        """
        测试场景：验证可以通过 detail 参数附加结构化上下文信息
        参数：message="msg", detail={"key": "value", "count": 42}
        测试逻辑：(1) 构造 ReActError 并传入 detail 字典 (2) 断言 e.detail 完全匹配
        预期结果：detail 字典原样保留，支持任意可序列化的键值对
        成功条件：e.detail == {"key": "value", "count": 42}
        """
        e = ReActError("msg", detail={"key": "value", "count": 42})
        assert e.detail == {"key": "value", "count": 42}

    def test_as_dict(self):
        """
        测试场景：验证 as_dict() 序列化方法将异常转为结构化字典，用于 JSON 日志输出
        参数：message="fail", code="ERR_001", detail={"line": 10}
        测试逻辑：(1) 构造带有完整信息的 ReActError (2) 调用 e.as_dict() (3) 检查返回字典的各个字段
        预期结果：返回字典包含 exception_type、code、message、detail 四个字段
        成功条件：d["exception_type"] == "ReActError", d["code"] == "ERR_001", d["message"] == "fail", d["detail"]["line"] == 10
        """
        e = ReActError("fail", code="ERR_001", detail={"line": 10})
        d = e.as_dict()
        assert d["exception_type"] == "ReActError"
        assert d["code"] == "ERR_001"
        assert d["message"] == "fail"
        assert d["detail"]["line"] == 10

    def test_as_dict_without_detail(self):
        """
        测试场景：验证不传 detail 参数时 as_dict() 返回空的 detail 字段
        参数：message="simple"（无 detail）
        测试逻辑：(1) 构造不带 detail 的 ReActError (2) 调用 as_dict() (3) 断言 detail 为空字典
        预期结果：detail 字段为空字典而非 None
        成功条件：d["detail"] == {}
        """
        e = ReActError("simple")
        d = e.as_dict()
        assert d["detail"] == {}


# ==========================================================================
# ParseError (Recoverable)
# ==========================================================================

class TestParseError:
    """ParseError — recoverable, carries raw LLM output."""

    def test_basic_parse_error(self):
        """
        测试场景：验证 ParseError 正确携带 LLM 原始输出和解析失败原因
        参数：raw_output="garbled output", details="No Action found"
        测试逻辑：(1) 构造 ParseError (2) 检查类型继承链 (3) 检查 raw_output、code、detail 属性
        预期结果：ParseError 是 RecoverableError 的子类，raw_output 保存原始输出，detail 包含 details 字段
        成功条件：isinstance(e, RecoverableError) 且 e.raw_output == "garbled output" 且 e.detail["details"] == "No Action found"
        """
        e = ParseError("garbled output", details="No Action found")
        assert isinstance(e, RecoverableError)
        assert isinstance(e, ReActError)
        assert e.code == "REACT_PARSE_ERR"
        assert e.raw_output == "garbled output"
        assert "garbled output" in str(e)
        assert e.detail["details"] == "No Action found"

    def test_snippet_truncated(self):
        """
        测试场景：验证超长原始输出会被截断为 200 字符的 snippet
        参数：raw_output="x" * 500（500 个字符）
        测试逻辑：(1) 用 500 个字符的字符串构造 ParseError (2) 检查 detail 中的 raw_output_snippet 长度
        预期结果：raw_output_snippet 被截断为 200 字符，防止日志/追踪中存储超大字符串
        成功条件：len(e.detail["raw_output_snippet"]) == 200
        """
        long_output = "x" * 500
        e = ParseError(long_output)
        assert len(e.detail["raw_output_snippet"]) == 200

    def test_as_dict(self):
        """
        测试场景：验证 ParseError 的 as_dict() 正确序列化包含 raw_output_snippet
        参数：raw_output="bad output", details="missing action"
        测试逻辑：(1) 构造 ParseError (2) 调用 as_dict() (3) 检查 exception_type 和 detail
        预期结果：exception_type 为 "ParseError"，detail 中包含 raw_output_snippet
        成功条件：d["exception_type"] == "ParseError" 且 d["detail"]["raw_output_snippet"] == "bad output"
        """
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
        """
        测试场景：验证当模型请求不存在的工具时，ToolNotFoundError 正确记录请求的工具名和可用工具列表
        参数：action="fly_to_moon", available_tools=["calculator", "get_weather"]
        测试逻辑：(1) 构造 ToolNotFoundError (2) 检查类型继承 (3) 检查 action 和 available_tools 属性 (4) 检查字符串表示
        预期结果：错误消息包含请求的工具名和可用工具列表，类型为 RecoverableError
        成功条件：e.action == "fly_to_moon" 且 e.available_tools == ["calculator", "get_weather"]
        """
        e = ToolNotFoundError("fly_to_moon", ["calculator", "get_weather"])
        assert isinstance(e, RecoverableError)
        assert e.action == "fly_to_moon"
        assert e.available_tools == ["calculator", "get_weather"]
        assert "fly_to_moon" in str(e)
        assert "calculator" in str(e)

    def test_as_dict(self):
        """
        测试场景：验证 ToolNotFoundError 的 as_dict() 正确序列化请求和可用工具列表
        参数：action="xyz", available_tools=["a", "b"]
        测试逻辑：(1) 构造 ToolNotFoundError (2) 调用 as_dict() (3) 检查 detail 中的 requested 和 available 字段
        预期结果：detail 包含 requested="xyz" 和 available=["a", "b"]
        成功条件：d["detail"]["requested"] == "xyz" 且 d["detail"]["available"] == ["a", "b"]
        """
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
        """
        测试场景：验证工具执行时内部抛出异常被正确包装为 ToolExecutionError
        参数：tool_name="calculator", tool_input="1/0", original_error=ValueError("division by zero")
        测试逻辑：(1) 创建一个原始异常 ValueError (2) 构造 ToolExecutionError 包装它 (3) 检查类型、工具名、原始错误信息
        预期结果：ToolExecutionError 是 RecoverableError 的子类，保留原始错误信息
        成功条件：e.tool_name == "calculator" 且 "division by zero" in str(e)
        """
        original = ValueError("division by zero")
        e = ToolExecutionError("calculator", "1/0", original)
        assert isinstance(e, RecoverableError)
        assert e.tool_name == "calculator"
        assert "division by zero" in str(e)
        assert e.detail["original_error"] == "division by zero"

    def test_as_dict(self):
        """
        测试场景：验证 ToolExecutionError 的 as_dict() 正确序列化工具名和输入
        参数：tool_name="get_weather", tool_input="Mars", original_error=RuntimeError("timeout")
        测试逻辑：(1) 构造 ToolExecutionError (2) 调用 as_dict() (3) 检查 detail 中的 tool_name 和 input
        预期结果：detail 包含 tool_name="get_weather" 和 input="Mars"
        成功条件：d["detail"]["tool_name"] == "get_weather" 且 d["detail"]["input"] == "Mars"
        """
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
        """
        测试场景：验证 ConfigError（致命错误）携带 REACT_MODEL_PATH 修复提示
        参数：message="Invalid model path"
        测试逻辑：(1) 构造 ConfigError (2) 检查 isinstance 关系 (3) 检查 hint 中包含 "REACT_MODEL_PATH"
        预期结果：ConfigError 是 FatalError 的子类，hint 指导用户检查环境变量
        成功条件：isinstance(e, FatalError) 且 "REACT_MODEL_PATH" in e.hint
        """
        e = ConfigError("Invalid model path")
        assert isinstance(e, FatalError)
        assert "REACT_MODEL_PATH" in e.hint

    def test_model_load_error(self):
        """
        测试场景：验证 ModelLoadError（致命错误）携带 REACT_N_CTX 修复提示
        参数：message="CUDA OOM"
        测试逻辑：(1) 构造 ModelLoadError (2) 检查 isinstance 关系 (3) 检查 hint 中包含 "REACT_N_CTX"（建议减小上下文窗口）
        预期结果：ModelLoadError 是 FatalError，hint 指导用户减小 N_CTX 以降低显存占用
        成功条件：isinstance(e, FatalError) 且 "REACT_N_CTX" in e.hint
        """
        e = ModelLoadError("CUDA OOM")
        assert isinstance(e, FatalError)
        assert "REACT_N_CTX" in e.hint

    def test_dependency_error(self):
        """
        测试场景：验证 DependencyError（致命错误）正确记录缺失的包名和安装命令
        参数：package="pyyaml", hint="pip install pyyaml"
        测试逻辑：(1) 构造 DependencyError (2) 检查 isinstance 关系 (3) 检查 hint 和 detail["package"]
        预期结果：DependencyError 是 FatalError，hint 是安装命令，detail 包含包名
        成功条件：isinstance(e, FatalError) 且 "pip install pyyaml" in e.hint 且 e.detail["package"] == "pyyaml"
        """
        e = DependencyError("pyyaml", "pip install pyyaml")
        assert isinstance(e, FatalError)
        assert "pip install pyyaml" in e.hint
        assert e.detail["package"] == "pyyaml"

    def test_fatal_is_not_recoverable(self):
        """
        测试场景：验证 FatalError 和 RecoverableError 是互斥的 —— 致命错误不是可恢复错误
        参数：使用 ModelLoadError("fail")
        测试逻辑：(1) 构造 ModelLoadError (2) 断言它不是 RecoverableError 的实例 (3) 断言它仍是 ReActError 的实例
        预期结果：ModelLoadError 是 FatalError/ReActError 但不是 RecoverableError
        成功条件：not isinstance(e, RecoverableError) 且 isinstance(e, ReActError)
        """
        e = ModelLoadError("fail")
        assert not isinstance(e, RecoverableError)
        assert isinstance(e, ReActError)

    def test_recoverable_is_not_fatal(self):
        """
        测试场景：验证反向关系 —— 可恢复错误不是致命错误
        参数：使用 ParseError("bad")
        测试逻辑：(1) 构造 ParseError (2) 断言它是 RecoverableError (3) 断言它不是 FatalError
        预期结果：ParseError 是 RecoverableError 但不是 FatalError
        成功条件：isinstance(e, RecoverableError) 且 not isinstance(e, FatalError)
        """
        e = ParseError("bad")
        assert isinstance(e, RecoverableError)
        assert not isinstance(e, FatalError)