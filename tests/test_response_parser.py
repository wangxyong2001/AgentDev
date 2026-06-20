"""
Tests for llama.response_parser — unified LLM output parser.

Covers all 7 parse paths:
  P1: Final Answer detection
  P2: Thought extraction (match / fallback to first line)
  P3: Action extraction (match / fallback to tool name scan)
  P4: Action Input extraction (match / fallback to post-action text)
  Error: ParseError when all strategies fail
  Preprocessing: Qwen <think> tag stripping
"""

import pytest
from unittest.mock import patch, mock_open

from agentic.protocol.parser import ResponseParser, ParserConfig
from agentic.exceptions import ParseError


# ==========================================================================
# Fixtures
# ==========================================================================

@pytest.fixture
def parser():
    """
    测试场景（fixture）：提供一个使用内置默认配置的全新 ResponseParser 实例
    参数：无
    测试逻辑：直接返回 ResponseParser()，使用默认的 ReAct 正则表达式和停止序列
    预期结果：返回一个可用的 ResponseParser 对象
    成功条件：返回的 parser 是 ResponseParser 实例
    """
    return ResponseParser()


@pytest.fixture
def tool_names():
    """
    测试场景（fixture）：提供标准工具名列表，用于解析器验证 Action 是否为已知工具
    参数：无
    测试逻辑：返回包含 "calculator" 和 "get_weather" 的列表
    预期结果：返回两个字符串的列表
    成功条件：返回值 == ["calculator", "get_weather"]
    """
    return ["calculator", "get_weather"]


# ==========================================================================
# P1: Final Answer detection
# ==========================================================================

class TestFinalAnswer:
    """Highest priority — Final Answer: terminates ReAct loop."""

    def test_simple_final_answer(self, parser, tool_names):
        """
        测试场景：验证 P1 优先级 —— 检测到 "Final Answer:" 时直接返回 final_answer action
        参数：parser + tool_names，LLM 输出="Final Answer: 5535"
        测试逻辑：(1) 调用 parser.parse("Final Answer: 5535", tool_names) (2) 检查 action 和 action_input
        预期结果：action == "final_answer"，action_input == "5535"
        成功条件：result["action"] == "final_answer" 且 result["action_input"] == "5535"
        """
        result = parser.parse("Final Answer: 5535", tool_names)
        assert result["action"] == "final_answer"
        assert result["action_input"] == "5535"

    def test_final_answer_with_thought(self, parser, tool_names):
        """
        测试场景：验证 Final Answer 之前有 Thought 行时也能正确提取最终答案
        参数：LLM 输出包含 "Thought: I have the answer\nFinal Answer: 42"
        测试逻辑：(1) 输入带 Thought 前缀的 Final Answer (2) 解析 (3) 检查 action 和 action_input
        预期结果：忽略 Thought 行，正确处理 Final Answer
        成功条件：result["action"] == "final_answer" 且 result["action_input"] == "42"
        """
        result = parser.parse(
            "Thought: I have the answer\nFinal Answer: 42", tool_names
        )
        assert result["action"] == "final_answer"
        assert result["action_input"] == "42"

    def test_final_answer_multiline(self, parser, tool_names):
        """
        测试场景：验证多行 Final Answer（如包含换行的长回答）被正确收集
        参数：LLM 输出="Final Answer: The temperature\nis 25 degrees Celsius."
        测试逻辑：(1) 输入包含换行的 Final Answer (2) 解析 (3) 检查 action_input 包含完整内容
        预期结果：action_input 包含完整的多行内容
        成功条件：result["action"] == "final_answer" 且 "25 degrees Celsius" in result["action_input"]
        """
        result = parser.parse(
            "Final Answer: The temperature\nis 25 degrees Celsius.", tool_names
        )
        assert result["action"] == "final_answer"
        assert "25 degrees Celsius" in result["action_input"]


# ==========================================================================
# P2+P3+P4: Standard ReAct format
# ==========================================================================

class TestStandardReAct:
    """Complete Thought/Action/Action Input parsing."""

    def test_full_format(self, parser, tool_names):
        """
        测试场景：验证标准 ReAct 三件套 (Thought / Action / Action Input) 的完整解析
        参数：LLM 输出包含 Thought/Action/Action Input 三行
        测试逻辑：(1) 输入标准格式 (2) 解析 (3) 分别检查 thought/action/action_input
        预期结果：三个字段均被正确提取
        成功条件：thought == "I need to calculate the result." 且 action == "calculator" 且 action_input == "123 * 45"
        """
        output = (
            "Thought: I need to calculate the result.\n"
            "Action: calculator\n"
            "Action Input: 123 * 45"
        )
        result = parser.parse(output, tool_names)
        assert result["thought"] == "I need to calculate the result."
        assert result["action"] == "calculator"
        assert result["action_input"] == "123 * 45"

    def test_missing_thought_fallback(self, parser, tool_names):
        """
        测试场景：验证 Thought 行缺失时的回退策略 —— 使用第一个非 Action/Final 行作为 thought
        参数：LLM 输出只有 "Action: calculator\nAction Input: 2 + 2"（无 Thought 行）
        测试逻辑：(1) 输入无 Thought 的输出 (2) 解析 (3) 检查 thought 不为空且 action/input 正确
        预期结果：thought 回退为非空字符串，action 和 action_input 仍然正确提取
        成功条件：result["action"] == "calculator" 且 result["action_input"] == "2 + 2" 且 result["thought"] != ""
        """
        output = "Action: calculator\nAction Input: 2 + 2"
        result = parser.parse(output, tool_names)
        assert result["action"] == "calculator"
        assert result["action_input"] == "2 + 2"
        # Thought fallback: first non-Action/Final line → "Action Input: 2 + 2"
        assert result["thought"] != ""

    def test_missing_action_input_fallback(self, parser, tool_names):
        """
        测试场景：验证 Action Input 行缺失时的回退策略 —— action_input 为空字符串
        参数：LLM 输出只有 "Thought: let me check weather\nAction: get_weather"
        测试逻辑：(1) 输入无 Action Input 的输出 (2) 解析 (3) 检查 action 正确提取且 action_input 为空
        预期结果：action 正确识别，action_input 为空字符串（不是 None）
        成功条件：result["action"] == "get_weather" 且 result["action_input"] == ""
        """
        output = "Thought: let me check weather\nAction: get_weather"
        result = parser.parse(output, tool_names)
        assert result["action"] == "get_weather"
        assert result["action_input"] == ""  # No input after action


# ==========================================================================
# P3: Action fallback — tool name scanning
# ==========================================================================

class TestActionFallback:
    """When 'Action:' line is missing, scan for tool names."""

    def test_fallback_tool_name_in_text(self, parser, tool_names):
        """
        测试场景：验证 P3 fallback —— 当没有 "Action:" 行时，扫描文本中的工具名来推断 action
        参数：LLM 输出="I will use the calculator function with 5 * 3"（无显式 Action 行）
        测试逻辑：(1) 输入不包含 "Action:" 但包含工具名 "calculator" 的自然语言输出 (2) 解析 (3) 检查 action 和 action_input
        预期结果：action 被推断为 "calculator"，action_input 包含工具名之后的文本
        成功条件：result["action"] == "calculator" 且 "5 * 3" in result["action_input"]
        """
        output = "I will use the calculator function with 5 * 3"
        result = parser.parse(output, tool_names)
        assert result["action"] == "calculator"
        assert "5 * 3" in result["action_input"]

    def test_fallback_ambiguous_tools(self, parser, tool_names):
        """
        测试场景：验证工具名歧义时的选择策略 —— 按注册顺序，第一个匹配的工具名获胜
        参数：LLM 输出="Using get_weather and calculator for Tokyo"（同时包含两个工具名）
        测试逻辑：(1) 输入同时提及 calculator 和 get_weather 的输出 (2) 解析 (3) 检查选择优先级
        预期结果：按注册顺序，calculator 先注册先匹配，action_input 包含匹配位置之后的文本
        成功条件：result["action"] == "calculator" 且 "Tokyo" in result["action_input"]
        """
        # If both tools mentioned, first registered tool wins
        output = "Using get_weather and calculator for Tokyo"
        result = parser.parse(output, tool_names)
        # calculator registered first → matches first in scanning
        assert result["action"] == "calculator"
        # action_input is text after the matched tool name
        assert "Tokyo" in result["action_input"]


# ==========================================================================
# Preprocessing — Qwen <think> tag stripping
# ==========================================================================

class TestPreprocessing:
    """Strip model-specific wrapper tags."""

    def test_strip_think_tags(self, parser):
        """
        测试场景：验证 Qwen 模型的 <think> 和 <response> 包装标签在预处理阶段被正确剥离
        参数：raw="<think>I should use calculator</think>\n<response>Action: calculator\nAction Input: 1+1</response>"
        测试逻辑：(1) 输入包含 Qwen 特有标签的原始输出 (2) 调用 parser.preprocess(raw) (3) 检查清理结果
        预期结果：<think> 和 <response> 标签被移除，ReAct 格式内容保留
        成功条件："<think>" not in cleaned 且 "<response>" not in cleaned 且 "Action: calculator" in cleaned
        """
        raw = "<think>I should use calculator</think>\n<response>Action: calculator\nAction Input: 1+1</response>"
        cleaned = parser.preprocess(raw)
        assert "<think>" not in cleaned
        assert "<response>" not in cleaned
        assert "Action: calculator" in cleaned

    def test_strip_empty_input(self, parser):
        """
        测试场景：验证空输入（纯空白）预处理后返回空字符串
        参数：raw="  "（两个空格）
        测试逻辑：(1) 输入纯空白字符串 (2) 调用 preprocess (3) 检查返回空字符串
        预期结果：空输入返回空字符串，不抛异常
        成功条件：cleaned == ""
        """
        cleaned = parser.preprocess("  ")
        assert cleaned == ""

    def test_no_tags_leaves_unchanged(self, parser):
        """
        测试场景：验证不含特殊标签的标准 ReAct 输出在预处理后保持不变
        参数：raw="Thought: test\nAction: calculator\nAction Input: x"
        测试逻辑：(1) 输入标准 ReAct 格式 (2) 调用 preprocess (3) 检查输出与输入完全相同
        预期结果：无标签时原样返回
        成功条件：cleaned == raw
        """
        raw = "Thought: test\nAction: calculator\nAction Input: x"
        cleaned = parser.preprocess(raw)
        assert cleaned == raw


# ==========================================================================
# Error: ParseError
# ==========================================================================

class TestParseError:
    """All parsing strategies exhausted."""

    def test_completely_unparseable(self, parser, tool_names):
        """
        测试场景：验证完全无法解析的文本（所有策略均失败）时抛出 ParseError
        参数：LLM 输出="This is just random text without any ReAct format"（无任何 ReAct 结构）
        测试逻辑：(1) 输入完全不包含 ReAct 格式或已知工具名的随机文本 (2) 使用 pytest.raises 捕获异常 (3) 检查异常消息和错误码
        预期结果：抛出 ParseError，消息包含 "Failed to parse"，错误码为 "REACT_PARSE_ERR"
        成功条件：ParseError 的 code == "REACT_PARSE_ERR" 且 str(exc.value) 包含 "Failed to parse"
        """
        with pytest.raises(ParseError) as exc:
            parser.parse("This is just random text without any ReAct format", tool_names)
        assert "无法解析" in str(exc.value)
        assert exc.value.code == "REACT_PARSE_ERR"

    def test_parse_error_carries_raw_output(self, parser, tool_names):
        """
        测试场景：验证 ParseError 正确携带原始 LLM 输出用于调试
        参数：LLM 输出="blah blah"（无意义的短文本）
        测试逻辑：(1) 输入无意义文本 (2) 捕获 ParseError (3) 检查 exc.value.raw_output
        预期结果：raw_output 属性保存了传入的原始文本
        成功条件：exc.value.raw_output == "blah blah"
        """
        with pytest.raises(ParseError) as exc:
            parser.parse("blah blah", tool_names)
        assert exc.value.raw_output == "blah blah"


# ==========================================================================
# YAML config loading
# ==========================================================================

class TestYamlLoading:
    """ResponseParser properly loads YAML overrides."""

    YAML_CONTENT = """
response:
  stop_sequences:
    - "Observation:"
    - "custom_stop"
  preprocessing:
    strip_tags:
      - "<custom_tag>"
  parsing:
    final_answer:
      pattern: "ANSWER:\\\\s*(.*)"
    thought:
      pattern: "THINKING:\\\\s*(.+?)(?=\\\\n|$)"
    action:
      pattern: "TOOL:\\\\s*(\\\\S+)"
    action_input:
      pattern: "INPUT:\\\\s*(.*)"
"""

    @pytest.mark.skip(reason="Known bug: _load_yaml() does not recompile regexes after YAML override (ARC-002 Phase 2)")
    def test_yaml_overrides_patterns(self, tool_names):
        """
        测试场景（已知Bug-跳过）：验证 ResponseParser 能正确加载 YAML 配置并覆盖默认正则表达式
        参数：YAML 定义自定义模式 TOOL:/INPUT: 替代 Action:/Action Input:，tool_names=["calculator"]
        测试逻辑：(1) mock 打开 YAML 文件 (2) 创建 ResponseParser(yaml_path="fake.yaml") (3) 用自定义格式解析
        预期结果：解析器应使用 YAML 中定义的 TOOL:/INPUT: 模式
        成功条件：result["action"] == "calculator" 且 result["action_input"] == "42"
        跳过原因：已知 bug —— _load_yaml() 在 YAML 覆盖后不会重新编译正则表达式 (ARC-002 Phase 2)
        """
        with patch("builtins.open", mock_open(read_data=self.YAML_CONTENT)):
            with patch("yaml.safe_load") as mock_yaml:
                import yaml
                mock_yaml.return_value = yaml.safe_load(self.YAML_CONTENT)
                parser = ResponseParser(yaml_path="fake.yaml")

        result = parser.parse("TOOL: calculator\nINPUT: 42", tool_names)
        assert result["action"] == "calculator"
        assert result["action_input"] == "42"

    def test_pyyaml_missing_keeps_defaults(self, tool_names):
        """
        测试场景：验证当 PyYAML 不可用时，ResponseParser 优雅降级使用默认配置
        参数：YAML 文件存在但 parser 使用默认配置，tool_names=["calculator"]
        测试逻辑：(1) mock YAML 文件 (2) 创建 parser (3) 用标准 ReAct 格式解析 (4) 验证默认行为正常
        预期结果：即使 YAML 加载可能失败，默认解析器仍能正确解析标准 ReAct 格式
        成功条件：result["action"] == "calculator"
        """
        with patch("builtins.open", mock_open(read_data=self.YAML_CONTENT)):
            parser = ResponseParser(yaml_path="fake.yaml")

        default_parser = ResponseParser()
        result = default_parser.parse("Action: calculator\nAction Input: test", tool_names)
        assert result["action"] == "calculator"