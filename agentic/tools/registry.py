"""
ToolRegistry 工具注册中心 — 基于依赖注入的 Agent 工具管理。

取代全局 TOOLS 列表，采用注册中心模式。支持：
  - 动态工具注册
  - 按名称查询工具（精确匹配）
  - 安全执行与错误处理
  - 工具元数据（描述、验证模式占位符）

用法:
  >>> from agentic.tools.registry import ToolRegistry
  >>> from agentic.tools.calculator import calculator_tool
  >>> from agentic.tools.weather import weather_tool
  >>> registry = ToolRegistry()
  >>> registry.register(calculator_tool)
  >>> registry.register(weather_tool)
  >>> registry.execute("calculator", "2 + 2")
  "4"
"""

from __future__ import annotations

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field

from agentic.exceptions import ToolNotFoundError, ToolExecutionError
from agentic.observability import get_logger

logger = get_logger(__name__)


# ==========================================================================
# 工具描述符
# ==========================================================================

@dataclass
class Tool:
    """
    工具描述符 — 已注册工具的单一信源。

    取代 langchain 的 @tool 依赖，采用框架无关的描述符。
    通过 `invoke()` 保持与 LangChain 的兼容性。

    属性:
      name:        工具唯一标识（必须匹配 LLM Action 中的值）
      description: 工具描述，显示在系统提示词中
      func:        可调用对象（接收 str，返回 str）
      metadata:    附加元数据（版本、超时秒数等）
    """

    name: str
    description: str
    func: Callable[[str], str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def invoke(self, input_str: str) -> str:
        """使用给定的输入字符串调用工具。"""
        return self.func(input_str)


# ==========================================================================
# 工具注册中心
# ==========================================================================

class ToolRegistry:
    """
    注册中心 — 管理工具的注册、查找和执行。

    设计原则:
      - 与 LLM 和 Agent 解耦 — 纯粹的工具管理
      - 可注入 — 测试时可注入 Mock 工具，无需全局状态
      - 安全执行 — 所有工具调用均包裹在 try/except 中

    用法:
      >>> registry = ToolRegistry()
      >>> registry.register(Tool("calc", "计算器工具", lambda x: str(eval(x))))
      >>> registry.list_names()
      ["calc"]
      >>> registry.execute("calc", "2+2")
      "4"
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    # ── 注册 ────────────────────────────────────────────────────────

    def register(self, tool: Tool) -> None:
        """
        注册一个工具。

        参数:
          tool: 工具描述符

        处理逻辑:
          1. 检查工具名称是否已存在
          2. 若已存在，抛出 ValueError
          3. 将工具存入内部字典
          4. 记录调试日志

        异常:
          ValueError: 同名工具已注册
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        logger.debug(f"Tool registered: {tool.name}")

    def unregister(self, name: str) -> None:
        """从注册中心移除一个工具。"""
        if name in self._tools:
            del self._tools[name]
            logger.debug(f"Tool unregistered: {name}")

    # ── 查询 ────────────────────────────────────────────────────────

    def get(self, name: str) -> Optional[Tool]:
        """按名称获取工具。未找到时返回 None。"""
        return self._tools.get(name)

    def list_tools(self) -> List[Tool]:
        """返回所有已注册的工具列表。"""
        return list(self._tools.values())

    def list_names(self) -> List[str]:
        """返回所有已注册工具的名称列表。"""
        return list(self._tools.keys())

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    # ── 执行 ────────────────────────────────────────────────────────

    def execute(self, name: str, input_str: str) -> str:
        """
        按名称执行已注册的工具。

        参数:
          name:      工具名称，必须在注册表中存在
          input_str: 传给工具的输入字符串（来自 LLM 的 Action Input）

        处理逻辑:
          1. 在注册表中按名称查找工具
          2. 若工具不存在，抛出 ToolNotFoundError 并列出可用工具
          3. 调用 tool.invoke(input_str) 执行工具
          4. 捕获执行异常，包装为 ToolExecutionError

        返回值:
          工具执行结果的字符串形式

        异常:
          ToolNotFoundError:  工具未注册
          ToolExecutionError: 工具执行时抛出异常
        """
        tool = self._tools.get(name)
        if tool is None:
            available = self.list_names()
            raise ToolNotFoundError(name, available)

        try:
            result = tool.invoke(input_str)
            return str(result)
        except Exception as e:
            raise ToolExecutionError(name, input_str, e)

    # ── 格式化 ─────────────────────────────────────────────────────

    def format_for_prompt(self) -> str:
        """
        将所有工具格式化为系统提示词显示形式。

        处理逻辑:
          1. 遍历所有已注册工具
          2. 每行生成 "- 工具名称: 描述" 格式
          3. 用换行符连接所有行

        返回值:
          多行字符串，格式为："- tool_name: description"
        """
        return "\n".join(
            f"- {t.name}: {t.description}"
            for t in self._tools.values()
        )

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
