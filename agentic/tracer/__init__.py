"""
Tracer — full-chain ReAct call observability.

Extracted from LlamaTracer monolith into:
  collector.py  — TraceRecord storage + record_turn() + session summary
  diff.py       — PromptDecomposition + TurnDiff algorithm (pure functions)

The diff algorithm is stateless and independently testable.
The collector manages session state and exports.
"""

from agentic.tracer.diff import PromptDecomposition, TurnDiff
from agentic.tracer.collector import TraceCollector, TraceRecord

__all__ = [
    "PromptDecomposition",
    "TurnDiff",
    "TraceRecord",
    "TraceCollector",
]
