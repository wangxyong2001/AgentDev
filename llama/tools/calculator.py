"""
Calculator tool — safe math expression evaluation.

Security: Uses eval() with empty __builtins__ for sandboxing.
Production upgrade path: replace with ast.literal_eval() or numexpr.
"""

from __future__ import annotations

from llama.tools.registry import Tool


def _calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression.

    Args:
      expression: Math expression string, e.g. '2 + 2', '10 * 5', '(25-32)*5/9'

    Returns:
      Calculation result string, or "Error: {message}" on failure

    Security:
      eval(expr, {"__builtins__": {}}, {}) — disables all builtins
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


# ── Tool descriptor ─────────────────────────────────────────────────

calculator_tool = Tool(
    name="calculator",
    description="Math calculator — evaluates expressions like '2+2' or '10*5' or '(25-32)*5/9'",
    func=_calculate,
    metadata={"version": "1.0.0", "sandbox": "eval_empty_builtins"},
)
