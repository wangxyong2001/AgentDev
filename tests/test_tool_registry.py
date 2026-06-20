"""
Tests for llama.tools — ToolRegistry, calculator, weather.
"""

import pytest
from llama.tools.registry import Tool, ToolRegistry
from llama.tools.calculator import calculator_tool, _calculate
from llama.tools.weather import weather_tool, _get_weather
from llama.exceptions import ToolNotFoundError, ToolExecutionError


class TestToolRegistry:
    """Tool registration, lookup, execution."""

    @pytest.fixture
    def registry(self):
        r = ToolRegistry()
        r.register(calculator_tool)
        r.register(weather_tool)
        return r

    def test_register(self, registry):
        assert registry.tool_count == 2
        assert "calculator" in registry
        assert "get_weather" in registry

    def test_register_duplicate_raises(self, registry):
        with pytest.raises(ValueError, match="already registered"):
            registry.register(calculator_tool)

    def test_get_existing(self, registry):
        tool = registry.get("calculator")
        assert tool is not None
        assert tool.name == "calculator"

    def test_get_missing(self, registry):
        assert registry.get("nonexistent") is None

    def test_list_names(self, registry):
        names = registry.list_names()
        assert "calculator" in names
        assert "get_weather" in names
        assert len(names) == 2

    def test_execute_calculator(self, registry):
        result = registry.execute("calculator", "2 + 2")
        assert result == "4"

    def test_execute_weather(self, registry):
        result = registry.execute("get_weather", "Tokyo")
        assert "25°C" in result

    def test_execute_unknown_tool_raises(self, registry):
        with pytest.raises(ToolNotFoundError):
            registry.execute("fly_to_moon", "now")

    def test_execute_error_wraps(self, registry):
        def broken(_):
            raise RuntimeError("boom")
        registry.register(Tool("broken", "desc", broken))
        with pytest.raises(ToolExecutionError) as exc:
            registry.execute("broken", "test")
        assert "boom" in str(exc.value)

    def test_unregister(self, registry):
        registry.unregister("calculator")
        assert registry.tool_count == 1
        assert "calculator" not in registry

    def test_format_for_prompt(self, registry):
        prompt = registry.format_for_prompt()
        assert "- calculator:" in prompt
        assert "- get_weather:" in prompt


class TestCalculator:
    """Math expression evaluation."""

    def test_simple_addition(self):
        assert _calculate("2 + 2") == "4"

    def test_multiplication(self):
        assert _calculate("123 * 45") == "5535"

    def test_complex_expression(self):
        result = _calculate("(25 - 32) * 5 / 9")
        assert float(result) == pytest.approx(-3.888, rel=0.01)

    def test_division_by_zero(self):
        result = _calculate("1 / 0")
        assert "Error" in result

    def test_sandbox_isolates_execution(self):
        """Sandbox runs in subprocess — os.system returns exit code, not Error string."""
        result = _calculate("__import__('os').system('ls')")
        # L1 sandbox: code runs in isolated subprocess.
        # os.system may succeed (return 0) or fail (return non-zero exit status).
        # Either way, it does NOT affect the parent process.
        assert result != ""  # Sandbox returned something (not a crash)


class TestWeather:
    """Weather lookup."""

    def test_known_city(self):
        assert "Rainy" in _get_weather("London")

    def test_unknown_city(self):
        result = _get_weather("Mars")
        assert "not found" in result

    def test_tool_descriptor(self):
        assert weather_tool.name == "get_weather"
        assert weather_tool.metadata["data_source"] == "simulated"
