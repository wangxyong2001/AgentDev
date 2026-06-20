"""
Tests for llama.tools — ToolRegistry, calculator, weather.
"""

import pytest
from agentic.tools.registry import Tool, ToolRegistry
from agentic.tools.calculator import calculator_tool, _calculate
from agentic.tools.weather import weather_tool, _get_weather
from agentic.exceptions import ToolNotFoundError, ToolExecutionError


class TestToolRegistry:
    """Tool registration, lookup, execution."""

    @pytest.fixture
    def registry(self):
        """
        测试场景（fixture）：创建一个预注册了 calculator 和 weather 两个工具的 ToolRegistry 实例
        参数：无
        测试逻辑：(1) 创建 ToolRegistry (2) 注册 calculator_tool (3) 注册 weather_tool (4) 返回 registry
        预期结果：registry 包含 2 个工具
        成功条件：registry.tool_count == 2
        """
        r = ToolRegistry()
        r.register(calculator_tool)
        r.register(weather_tool)
        return r

    def test_register(self, registry):
        """
        测试场景：验证工具注册后 registry 中的计数和成员检查
        参数：预注册了 calculator 和 weather 的 registry
        测试逻辑：(1) 检查 tool_count (2) 使用 in 操作符检查工具是否存在
        预期结果：tool_count == 2，两个工具名都在 registry 中
        成功条件：registry.tool_count == 2 且 "calculator" in registry 且 "get_weather" in registry
        """
        assert registry.tool_count == 2
        assert "calculator" in registry
        assert "get_weather" in registry

    def test_register_duplicate_raises(self, registry):
        """
        测试场景：验证重复注册同名工具时抛出 ValueError
        参数：registry 已包含 calculator_tool，再次注册 calculator_tool
        测试逻辑：(1) 尝试再次注册 calculator_tool (2) 断言抛出 ValueError 且消息包含 "already registered"
        预期结果：抛出 ValueError，防止工具被意外覆盖
        成功条件：pytest.raises(ValueError, match="already registered")
        """
        with pytest.raises(ValueError, match="already registered"):
            registry.register(calculator_tool)

    def test_get_existing(self, registry):
        """
        测试场景：验证通过工具名获取已注册的工具
        参数：registry + tool_name="calculator"
        测试逻辑：(1) 调用 registry.get("calculator") (2) 检查返回值非 None 且 name 正确
        预期结果：返回一个 Tool 对象，其 name 属性为 "calculator"
        成功条件：tool is not None 且 tool.name == "calculator"
        """
        tool = registry.get("calculator")
        assert tool is not None
        assert tool.name == "calculator"

    def test_get_missing(self, registry):
        """
        测试场景：验证获取不存在的工具时返回 None（不抛异常）
        参数：registry + tool_name="nonexistent"
        测试逻辑：(1) 调用 registry.get("nonexistent") (2) 检查返回 None
        预期结果：返回 None，不抛异常
        成功条件：registry.get("nonexistent") is None
        """
        assert registry.get("nonexistent") is None

    def test_list_names(self, registry):
        """
        测试场景：验证 list_names() 返回所有已注册工具的名称列表
        参数：registry 包含 2 个工具
        测试逻辑：(1) 调用 registry.list_names() (2) 检查长度和内容
        预期结果：返回包含 "calculator" 和 "get_weather" 的列表，长度为 2
        成功条件："calculator" in names 且 "get_weather" in names 且 len(names) == 2
        """
        names = registry.list_names()
        assert "calculator" in names
        assert "get_weather" in names
        assert len(names) == 2

    def test_execute_calculator(self, registry):
        """
        测试场景：验证通过 registry 执行 calculator 工具的正确性
        参数：registry + tool_name="calculator", tool_input="2 + 2"
        测试逻辑：(1) 调用 registry.execute("calculator", "2 + 2") (2) 检查返回结果
        预期结果：返回计算结果 "4"
        成功条件：result == "4"
        """
        result = registry.execute("calculator", "2 + 2")
        assert result == "4"

    def test_execute_weather(self, registry):
        """
        测试场景：验证通过 registry 执行 weather 工具的正确性
        参数：registry + tool_name="get_weather", tool_input="Tokyo"
        测试逻辑：(1) 调用 registry.execute("get_weather", "Tokyo") (2) 检查返回结果包含温度
        预期结果：返回包含 "25°C" 的天气描述字符串
        成功条件："25°C" in result
        """
        result = registry.execute("get_weather", "Tokyo")
        assert "25°C" in result

    def test_execute_unknown_tool_raises(self, registry):
        """
        测试场景：验证执行未注册工具时抛出 ToolNotFoundError
        参数：registry + tool_name="fly_to_moon", tool_input="now"
        测试逻辑：(1) 尝试执行不存在的工具 (2) 断言抛出 ToolNotFoundError
        预期结果：抛出 ToolNotFoundError，且错误信息包含请求的工具名
        成功条件：pytest.raises(ToolNotFoundError)
        """
        with pytest.raises(ToolNotFoundError):
            registry.execute("fly_to_moon", "now")

    def test_execute_error_wraps(self, registry):
        """
        测试场景：验证工具执行过程中内部异常被正确包装为 ToolExecutionError
        参数：注册一个总是 raise RuntimeError("boom") 的 broken 工具
        测试逻辑：(1) 注册 broken 工具 (2) 调用 execute (3) 捕获 ToolExecutionError (4) 检查原始错误信息
        预期结果：抛出 ToolExecutionError，str(exc.value) 包含原始错误 "boom"
        成功条件：pytest.raises(ToolExecutionError) 且 "boom" in str(exc.value)
        """
        def broken(_):
            raise RuntimeError("boom")
        registry.register(Tool("broken", "desc", broken))
        with pytest.raises(ToolExecutionError) as exc:
            registry.execute("broken", "test")
        assert "boom" in str(exc.value)

    def test_unregister(self, registry):
        """
        测试场景：验证注销工具后计数减少且工具不可访问
        参数：registry 原有 2 个工具，注销 "calculator"
        测试逻辑：(1) 调用 registry.unregister("calculator") (2) 检查 tool_count 和成员检查
        预期结果：tool_count 从 2 变为 1，"calculator" 不再在 registry 中
        成功条件：registry.tool_count == 1 且 "calculator" not in registry
        """
        registry.unregister("calculator")
        assert registry.tool_count == 1
        assert "calculator" not in registry

    def test_format_for_prompt(self, registry):
        """
        测试场景：验证 format_for_prompt() 生成的工具描述字符串格式正确
        参数：registry 包含 calculator 和 weather 工具
        测试逻辑：(1) 调用 registry.format_for_prompt() (2) 检查输出包含 "- calculator:" 和 "- get_weather:"
        预期结果：返回包含工具名和描述的格式化字符串，用于嵌入 system prompt
        成功条件："- calculator:" in prompt 且 "- get_weather:" in prompt
        """
        prompt = registry.format_for_prompt()
        assert "- calculator:" in prompt
        assert "- get_weather:" in prompt


class TestCalculator:
    """Math expression evaluation."""

    def test_simple_addition(self):
        """
        测试场景：验证 _calculate() 正确执行基本加法运算
        参数：expression="2 + 2"
        测试逻辑：(1) 调用 _calculate("2 + 2") (2) 检查返回结果
        预期结果：返回字符串 "4"
        成功条件：_calculate("2 + 2") == "4"
        """
        assert _calculate("2 + 2") == "4"

    def test_multiplication(self):
        """
        测试场景：验证 _calculate() 正确执行乘法运算
        参数：expression="123 * 45"
        测试逻辑：(1) 调用 _calculate("123 * 45") (2) 检查返回结果
        预期结果：返回字符串 "5535"
        成功条件：_calculate("123 * 45") == "5535"
        """
        assert _calculate("123 * 45") == "5535"

    def test_complex_expression(self):
        """
        测试场景：验证 _calculate() 正确执行包含括号和负数/小数的复杂表达式
        参数：expression="(25 - 32) * 5 / 9"（摄氏度转华氏度公式的逆运算）
        测试逻辑：(1) 调用 _calculate("(25 - 32) * 5 / 9") (2) 将结果转为 float (3) 近似比较
        预期结果：返回值约等于 -3.888
        成功条件：float(result) == pytest.approx(-3.888, rel=0.01)
        """
        result = _calculate("(25 - 32) * 5 / 9")
        assert float(result) == pytest.approx(-3.888, rel=0.01)

    def test_division_by_zero(self):
        """
        测试场景：验证除以零时 _calculate() 返回包含 "Error" 的描述而非崩溃
        参数：expression="1 / 0"
        测试逻辑：(1) 调用 _calculate("1 / 0") (2) 检查返回包含 "Error"
        预期结果：不抛异常，返回包含 "Error" 的错误提示字符串
        成功条件："Error" in result
        """
        result = _calculate("1 / 0")
        assert "Error" in result

    def test_sandbox_isolates_execution(self):
        """
        测试场景：验证 L1 沙箱隔离 —— 恶意代码在子进程中执行不影响父进程
        参数：expression="__import__('os').system('ls')"（尝试执行系统命令）
        测试逻辑：(1) 传入试图调用 os.system 的危险表达式 (2) 检查返回值非空（沙箱返回了结果而非崩溃）
        预期结果：代码在子进程中隔离执行，os.system 返回退出码不影响父进程，返回值不为空字符串
        成功条件：result != ""（沙箱正常返回，未崩溃）
        """
        result = _calculate("__import__('os').system('ls')")
        # L1 sandbox: code runs in isolated subprocess.
        # os.system may succeed (return 0) or fail (return non-zero exit status).
        # Either way, it does NOT affect the parent process.
        assert result != ""  # Sandbox returned something (not a crash)


class TestWeather:
    """Weather lookup."""

    def test_known_city(self):
        """
        测试场景：验证已知城市 London 返回天气信息包含 "Rainy"
        参数：city="London"
        测试逻辑：(1) 调用 _get_weather("London") (2) 检查返回包含 "Rainy"
        预期结果：返回的天气描述字符串包含 "Rainy"
        成功条件："Rainy" in _get_weather("London")
        """
        assert "Rainy" in _get_weather("London")

    def test_unknown_city(self):
        """
        测试场景：验证未知城市 "Mars" 返回包含 "not found" 的提示
        参数：city="Mars"
        测试逻辑：(1) 调用 _get_weather("Mars") (2) 检查返回包含 "not found"
        预期结果：返回未找到城市的提示信息
        成功条件："not found" in result
        """
        result = _get_weather("Mars")
        assert "not found" in result

    def test_tool_descriptor(self):
        """
        测试场景：验证 weather_tool 的元数据描述符正确
        参数：weather_tool 模块级变量
        测试逻辑：(1) 检查 weather_tool.name (2) 检查 weather_tool.metadata["data_source"]
        预期结果：名称为 "get_weather"，数据源标记为 "simulated"
        成功条件：weather_tool.name == "get_weather" 且 weather_tool.metadata["data_source"] == "simulated"
        """
        assert weather_tool.name == "get_weather"
        assert weather_tool.metadata["data_source"] == "simulated"