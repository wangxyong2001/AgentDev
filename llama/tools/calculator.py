"""
Calculator tool — math expression evaluation via OS-level process sandbox.

Security: Uses execute_in_sandbox() which runs python3 -I -E with
RLIMIT_AS, RLIMIT_CPU, RLIMIT_NPROC and per-invocation temp filesystem.
"""

from __future__ import annotations

from llama.tools.registry import Tool
from llama.tools.sandbox import execute_in_sandbox


def _calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression inside an OS-level sandbox.

    The expression is wrapped in ``print(...)`` and executed as a
    standalone Python script in a subprocess with resource limits,
    minimal environment, and optional network isolation.

    Args:
      expression: Math expression string, e.g. '2 + 2', '10 * 5', '(25-32)*5/9'

    Returns:
      Calculation result string, or sandbox error message on failure.
    """
    sandbox_code = f"print({expression})"
    result = execute_in_sandbox(sandbox_code, timeout_sec=5, memory_mb=64)
    return result.strip()


# ── Tool descriptor ─────────────────────────────────────────────────

calculator_tool = Tool(
    name="calculator",
    description="Math calculator — evaluates expressions like '2+2' or '10*5' or '(25-32)*5/9'",
    func=_calculate,
    metadata={"version": "2.0.0", "sandbox": "subprocess_rlimit"},
)
