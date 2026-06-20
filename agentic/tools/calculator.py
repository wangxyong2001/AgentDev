"""
计算器工具 — 通过 OS 级进程沙箱进行数学表达式求值。

安全: 使用 execute_in_sandbox()，以 python3 -I -E 运行，
配合 RLIMIT_AS、RLIMIT_CPU、RLIMIT_NPROC 和每次调用的临时文件系统。
"""

from __future__ import annotations

from agentic.tools.registry import Tool
from agentic.tools.sandbox import execute_in_sandbox


def _calculate(expression: str) -> str:
    """
    在 OS 级沙箱内计算数学表达式。

    处理逻辑:
      1. 将表达式包裹在 ``print(...)`` 中
      2. 作为独立 Python 脚本在子进程中执行
      3. 子进程应用资源限制、最小环境和可选的网络隔离
      4. 返回计算结果或沙箱错误

    参数:
      expression: 数学表达式字符串，例如 '2 + 2'、'10 * 5'、'(25-32)*5/9'

    返回值:
      计算结果字符串，失败时返回沙箱错误消息。
    """
    sandbox_code = f"print({expression})"
    result = execute_in_sandbox(sandbox_code, timeout_sec=5, memory_mb=64)
    return result.strip()


# ── 工具描述符 ─────────────────────────────────────────────────────

calculator_tool = Tool(
    name="calculator",
    description="Math calculator — evaluates expressions like '2+2' or '10*5' or '(25-32)*5/9'",
    func=_calculate,
    metadata={"version": "2.0.0", "sandbox": "subprocess_rlimit"},
)
