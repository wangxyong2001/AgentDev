"""
ReAct Agent 企业级异常层次结构。

设计原则:
  1. 明确边界: RecoverableError（循环内重试）vs FatalError（退出进程）
  2. 每个异常携带足够的上下文用于日志记录和调试
  3. 与结构化日志集成（异常可序列化为 JSON）

用法:
  >>> from agentic.exceptions import ParseError, ToolNotFoundError, FatalError
  >>> try:
  ...     parsed = parser.parse(output)
  ... except ParseError as e:
  ...     logger.warning("解析失败，将重试", extra=e.as_dict())

异常树:

  ReActError（基类）
  ├── RecoverableError        ← 循环内捕获，触发错误恢复
  │   ├── ParseError           — LLM 输出不符合 ReAct 格式
  │   ├── ToolNotFoundError    — 模型请求了不存在的工具
  │   └── ToolExecutionError   — tool.invoke() 抛出异常
  └── FatalError              ← 在 main() 中捕获，调用 sys.exit(1)
      ├── ConfigError           — 配置无效
      ├── ModelLoadError        — LLM 初始化失败
      └── DependencyError       — 缺少必需的 Python 包
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List


# ==========================================================================
# 基类异常
# ==========================================================================

class ReActError(Exception):
    """所有 ReAct Agent 错误的基类。"""

    def __init__(self, message: str, *, code: str = "REACT_ERR", detail: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.detail = detail or {}

    def as_dict(self) -> Dict[str, Any]:
        """
        序列化为结构化的字典，用于 JSON 日志记录。

        功能描述:
          将异常信息转换为字典格式，包含异常类型、错误码、
          错误消息和详细信息，便于结构化日志输出。

        返回:
          包含 exception_type、code、message、detail 的字典
        """
        return {
            "exception_type": self.__class__.__name__,
            "code": self.code,
            "message": str(self),
            "detail": self.detail,
        }


# ==========================================================================
# 可恢复错误（循环内通过错误恢复机制重试）
# ==========================================================================

class RecoverableError(ReActError):
    """
    可通过 ReAct 循环恢复的错误基类。

    功能描述:
      这些错误不会终止 Agent 运行，而是将错误信息作为 Observation 注入，
      让 LLM 在下一轮重试。

    处理逻辑:
      1. 在 ReAct 循环中捕获
      2. 将错误消息作为 Observation 注入
      3. LLM 在下一轮根据错误信息修正行为
      4. 连续超过 max_steps 次失败后，循环安全终止
    """
    pass


class ParseError(RecoverableError):
    """
    LLM 输出无法解析为有效的 ReAct 格式。

    功能描述:
      当所有四个解析层级（P1~P4）和后备策略都失败时触发。
      原始 LLM 输出作为 Observation 注入以进行下一轮重试。

    参数:
      raw_output: LLM 原始输出文本
      details: 解析失败的详细原因
    """
    def __init__(self, raw_output: str, details: str = ""):
        super().__init__(
            message=f"无法解析 LLM 输出。内容: {raw_output[:200]}",
            code="REACT_PARSE_ERR",
            detail={"raw_output_snippet": raw_output[:200], "details": details},
        )
        self.raw_output = raw_output


class ToolNotFoundError(RecoverableError):
    """
    LLM 请求了一个未注册的工具名。

    功能描述:
      Observation 中包含可用工具列表，用于引导模型使用正确的工具名。

    参数:
      action: LLM 请求的动作名
      available_tools: 所有已注册的工具名称列表
    """
    def __init__(self, action: str, available_tools: List[str]):
        super().__init__(
            message=f"未知动作: '{action}'。可用工具: {available_tools}",
            code="REACT_TOOL_NOT_FOUND",
            detail={"requested": action, "available": available_tools},
        )
        self.action = action
        self.available_tools = available_tools


class ToolExecutionError(RecoverableError):
    """
    工具调用时抛出未处理的异常。

    功能描述:
      异常消息被捕获并作为 Observation 提供给模型。

    参数:
      tool_name: 工具名
      input_str: 工具的输入参数
      original_error: 原始异常对象
    """
    def __init__(self, tool_name: str, input_str: str, original_error: Exception):
        super().__init__(
            message=f"工具 '{tool_name}' 执行错误: {original_error}",
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
# 致命错误（进程立即退出）
# ==========================================================================

class FatalError(ReActError):
    """
    不可恢复错误的基类。

    功能描述:
      这些错误会终止进程并调用 sys.exit(1)。
      每个 FatalError 附带 [HINT] 提示信息，提供修复指导。

    参数:
      message: 错误描述
      code: 错误码，默认 "REACT_FATAL"
      hint: 修复建议
      detail: 可选的详细上下文
    """
    hint: str = ""

    def __init__(self, message: str, *, code: str = "REACT_FATAL", hint: str = "", detail: Optional[Dict[str, Any]] = None):
        super().__init__(message, code=code, detail=detail)
        self.hint = hint


class ConfigError(FatalError):
    """
    配置验证失败（模型路径不存在、无效值等）。

    功能描述:
      当配置加载或验证阶段发现错误时触发。

    参数:
      message: 配置错误描述
    """
    def __init__(self, message: str):
        super().__init__(
            message=message,
            code="REACT_CONFIG_ERR",
            hint="请检查 REACT_MODEL_PATH 和其他 REACT_* 环境变量。",
        )


class ModelLoadError(FatalError):
    """
    LLM 模型加载失败（CUDA OOM、文件损坏等）。

    功能描述:
      当 LLM 模型初始化过程中发生不可恢复错误时触发。

    参数:
      message: 加载错误的描述
    """
    def __init__(self, message: str):
        super().__init__(
            message=message,
            code="REACT_MODEL_LOAD_ERR",
            hint="如果遇到 CUDA OOM，请减小 REACT_N_CTX 或设置 REACT_N_GPU_LAYERS=0（仅使用 CPU）。",
        )


class DependencyError(FatalError):
    """
    缺少必需的 Python 包。

    功能描述:
      当所需的第三方 Python 包未安装时触发。

    参数:
      package: 缺失的包名
      install_cmd: 安装该包的命令
    """
    def __init__(self, package: str, install_cmd: str):
        super().__init__(
            message=f"缺少依赖: {package}",
            code="REACT_DEPENDENCY_ERR",
            hint=f"请运行: {install_cmd}",
            detail={"package": package, "install_command": install_cmd},
        )
