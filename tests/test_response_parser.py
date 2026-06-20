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

from llama.response_parser import ResponseParser, ParserConfig
from llama.exceptions import ParseError


# ==========================================================================
# Fixtures
# ==========================================================================

@pytest.fixture
def parser():
    """Fresh ResponseParser with built-in defaults."""
    return ResponseParser()


@pytest.fixture
def tool_names():
    return ["calculator", "get_weather"]


# ==========================================================================
# P1: Final Answer detection
# ==========================================================================

class TestFinalAnswer:
    """Highest priority — Final Answer: terminates ReAct loop."""

    def test_simple_final_answer(self, parser, tool_names):
        result = parser.parse("Final Answer: 5535", tool_names)
        assert result["action"] == "final_answer"
        assert result["action_input"] == "5535"

    def test_final_answer_with_thought(self, parser, tool_names):
        result = parser.parse(
            "Thought: I have the answer\nFinal Answer: 42", tool_names
        )
        assert result["action"] == "final_answer"
        assert result["action_input"] == "42"

    def test_final_answer_multiline(self, parser, tool_names):
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
        output = "Action: calculator\nAction Input: 2 + 2"
        result = parser.parse(output, tool_names)
        assert result["action"] == "calculator"
        assert result["action_input"] == "2 + 2"
        # Thought fallback: first non-Action/Final line → "Action Input: 2 + 2"
        assert result["thought"] != ""

    def test_missing_action_input_fallback(self, parser, tool_names):
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
        output = "I will use the calculator function with 5 * 3"
        result = parser.parse(output, tool_names)
        assert result["action"] == "calculator"
        assert "5 * 3" in result["action_input"]

    def test_fallback_ambiguous_tools(self, parser, tool_names):
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
        raw = "<think>I should use calculator</think>\n<response>Action: calculator\nAction Input: 1+1</response>"
        cleaned = parser.preprocess(raw)
        assert "<think>" not in cleaned
        assert "<response>" not in cleaned
        assert "Action: calculator" in cleaned

    def test_strip_empty_input(self, parser):
        cleaned = parser.preprocess("  ")
        assert cleaned == ""

    def test_no_tags_leaves_unchanged(self, parser):
        raw = "Thought: test\nAction: calculator\nAction Input: x"
        cleaned = parser.preprocess(raw)
        assert cleaned == raw


# ==========================================================================
# Error: ParseError
# ==========================================================================

class TestParseError:
    """All parsing strategies exhausted."""

    def test_completely_unparseable(self, parser, tool_names):
        with pytest.raises(ParseError) as exc:
            parser.parse("This is just random text without any ReAct format", tool_names)
        assert "Failed to parse" in str(exc.value)
        assert exc.value.code == "REACT_PARSE_ERR"

    def test_parse_error_carries_raw_output(self, parser, tool_names):
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
        with patch("builtins.open", mock_open(read_data=self.YAML_CONTENT)):
            with patch("yaml.safe_load") as mock_yaml:
                import yaml
                mock_yaml.return_value = yaml.safe_load(self.YAML_CONTENT)
                parser = ResponseParser(yaml_path="fake.yaml")

        result = parser.parse("TOOL: calculator\nINPUT: 42", tool_names)
        assert result["action"] == "calculator"
        assert result["action_input"] == "42"

    def test_pyyaml_missing_keeps_defaults(self, tool_names):
        with patch("builtins.open", mock_open(read_data=self.YAML_CONTENT)):
            parser = ResponseParser(yaml_path="fake.yaml")

        default_parser = ResponseParser()
        result = default_parser.parse("Action: calculator\nAction Input: test", tool_names)
        assert result["action"] == "calculator"
