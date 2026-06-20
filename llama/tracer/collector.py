"""
Trace Collector — per-turn recording and session-level aggregation.

Manages:
  - TraceRecord storage (append-only list)
  - Cross-turn state tracking (_last_* variables)
  - Session summary statistics (tokens, cache hit rate, cost)
  - HTML report export (delegates to ReActTraceRenderer)

Usage:
  >>> from llama.tracer.collector import TraceCollector
  >>> collector = TraceCollector(model_name="qwen3.6-35b", pricing=(0.004, 0.012))
  >>> collector.record_turn(turn=1, question="2+2?", prompt=..., ...)
  >>> collector.summary()  # {"total_turns": 1, "total_tokens": 222, ...}
"""

from __future__ import annotations

import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from llama.tracer.diff import (
    PromptDecomposition,
    TurnDiff,
    decompose_prompt,
    compute_diff,
)
from llama.logging_config import get_logger

logger = get_logger(__name__)


# ==========================================================================
# TraceRecord
# ==========================================================================

@dataclass
class TraceRecord:
    """Complete per-turn ReAct call record."""

    turn: int
    question: str
    timestamp: str
    prompt_full: str
    prompt_snapshot: str
    raw_output: str
    cleaned_output: str
    thought: str
    action: str
    action_input: str
    observation: str
    prompt_tokens: int
    completion_tokens: int
    duration_ms: float
    decomp: Optional[PromptDecomposition] = None
    diff: Optional[TurnDiff] = None
    parse_error: Optional[str] = None
    status: str = "success"


# ==========================================================================
# TraceCollector
# ==========================================================================

class TraceCollector:
    """
    Full-chain ReAct call trace collector.

    Records every turn, computes cross-turn diffs, aggregates session stats,
    and exports HTML visualizations.
    """

    def __init__(self, model_name: str = "",
                 pricing: Tuple[float, float] = (0.004, 0.012)):
        self.records: List[TraceRecord] = []
        self.session_start = datetime.now()
        self.model_name = model_name
        self.pricing = pricing  # (input_price_per_1k, output_price_per_1k)

        # Cross-turn state
        self._last_prompt: str = ""
        self._last_decomp: Optional[PromptDecomposition] = None
        self._last_response: str = ""
        self._last_thought: str = ""
        self._last_action: str = ""

    # ── Recording ────────────────────────────────────────────────────

    def record_turn(
        self,
        turn: int,
        question: str,
        prompt: str,
        raw_output: str,
        cleaned_output: str,
        thought: str,
        action: str,
        action_input: str,
        observation: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration_ms: float,
        parse_error: Optional[str] = None,
        status: str = "success",
    ):
        """
        Record one complete ReAct turn.

        Automatically:
          1. Decomposes the prompt into system/user/history
          2. Computes cross-turn diff (skip for turn 1)
          3. Stores TraceRecord
          4. Updates cross-turn state
          5. Logs diff summary
        """
        # Decompose + diff
        decomp = decompose_prompt(prompt)
        diff = compute_diff(
            turn=turn,
            current_prompt=prompt,
            prev_prompt=self._last_prompt,
            decomp=decomp,
            prev_decomp=self._last_decomp,
            thought=thought,
            prev_thought=self._last_thought,
            action=action,
            prev_action=self._last_action,
            response_len=len(cleaned_output),
            prev_response_len=len(self._last_response),
        )

        # Create record
        record = TraceRecord(
            turn=turn,
            question=question,
            timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3],
            prompt_full=prompt,
            prompt_snapshot=prompt[-500:],
            raw_output=raw_output,
            cleaned_output=cleaned_output,
            thought=thought,
            action=action,
            action_input=action_input,
            observation=observation,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_ms=duration_ms,
            decomp=decomp,
            diff=diff,
            parse_error=parse_error,
            status=status,
        )
        self.records.append(record)

        # Update cross-turn state
        self._last_prompt = prompt
        self._last_decomp = decomp
        self._last_response = cleaned_output
        self._last_thought = thought
        self._last_action = action

        # Log diff
        if diff and turn > 1:
            logger.diff(diff.summary_line)
        elif diff and turn == 1:
            logger.diff(
                f"initial prompt | system={decomp.system_len}ch "
                f"user={decomp.user_len}ch history={decomp.history_len}ch"
            )

    # ── Aggregation ──────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Compute session-level aggregate statistics."""
        total_turns = len(self.records)
        total_prompt = sum(r.prompt_tokens for r in self.records)
        total_completion = sum(r.completion_tokens for r in self.records)
        total_duration = sum(r.duration_ms for r in self.records)
        error_turns = [r for r in self.records if r.status != "success"]
        tool_calls = [r for r in self.records
                      if r.action not in ("final_answer", "none", "")]

        # Cache statistics
        total_cached = sum(r.diff.cached_tokens for r in self.records if r.diff)
        total_new = sum(r.diff.new_tokens for r in self.records if r.diff)
        total_est = total_cached + total_new
        cache_hit_rate = (total_cached / total_est * 100) if total_est > 0 else 0.0

        # System change canary — should always be 0
        system_change_count = sum(
            1 for r in self.records
            if r.diff and not r.diff.system_unchanged
        )

        # Average prompt reuse (exclude turn 1)
        reuse_rates = [r.diff.prompt_reuse_pct
                       for r in self.records[1:] if r.diff]
        avg_reuse = sum(reuse_rates) / len(reuse_rates) if reuse_rates else 0.0

        # Cost
        cost_rmb = (
            total_prompt * self.pricing[0] +
            total_completion * self.pricing[1]
        ) / 1000

        return {
            "model": self.model_name,
            "session_duration_s": (datetime.now() - self.session_start).total_seconds(),
            "total_turns": total_turns,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "total_duration_ms": total_duration,
            "error_turns": len(error_turns),
            "tool_calls": len(tool_calls),
            "cost_rmb": cost_rmb,
            "estimated_cached_tokens": total_cached,
            "estimated_new_tokens": total_new,
            "cache_hit_rate": cache_hit_rate,
            "avg_prompt_reuse_pct": avg_reuse,
            "system_change_count": system_change_count,
        }

    # ── Reporting ────────────────────────────────────────────────────

    def print_summary(self):
        """Print ASCII art summary to terminal."""
        s = self.summary()
        cache_bar = _bar(s['cache_hit_rate'], 30)
        reuse_bar = _bar(s['avg_prompt_reuse_pct'], 30)
        logger.info(f"""
╔══════════════════════════════════════════════════════════╗
║              ReAct Trace Report                          ║
╠══════════════════════════════════════════════════════════╣
║  Model: {s['model'][:46]:<46s} ║
║  Turns: {s['total_turns']:<5}  Duration: {s['session_duration_s']:.1f}s{'':>30s} ║
╠══════════════════════════════════════════════════════════╣
║  Token Usage                                            ║
║    Prompt tokens:      {s['total_prompt_tokens']:>8,d}                          ║
║    Completion tokens:  {s['total_completion_tokens']:>8,d}                          ║
║    Total:              {s['total_tokens']:>8,d}                          ║
╠══════════════════════════════════════════════════════════╣
║  Cache Analysis (estimated)                             ║
║    Cached tokens:      {s['estimated_cached_tokens']:>8,d}  [{cache_bar}] ║
║    New tokens:         {s['estimated_new_tokens']:>8,d}                          ║
║    Cache hit rate:     {s['cache_hit_rate']:>6.1f}%                             ║
║    Avg prompt reuse:   {s['avg_prompt_reuse_pct']:>6.1f}%  [{reuse_bar}] ║
║    System changes:     {s['system_change_count']} (should be 0)                          ║
╠══════════════════════════════════════════════════════════╣
║  Cost: ¥{s['cost_rmb']:.6f}  |  Tools: {s['tool_calls']}  |  Errors: {s['error_turns']}  |  LLM: {s['total_duration_ms']:.0f}ms ║
╚══════════════════════════════════════════════════════════╝
""")

    def export_html(self, filepath: str):
        """Export self-contained HTML trace report."""
        from ReActTraceRenderer import render_html
        render_html(self.records, self.summary(), self.session_start, filepath)


# ==========================================================================
# Helpers
# ==========================================================================

def _bar(pct: float, width: int) -> str:
    """Draw a terminal progress bar."""
    filled = int(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)

