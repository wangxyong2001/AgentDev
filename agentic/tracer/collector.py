"""
Trace Collector — per-turn recording and session-level aggregation.

Manages:
  - TraceRecord storage (append-only list)
  - Cross-turn state tracking (_last_* variables)
  - Session summary statistics (tokens, cache hit rate, cost)
  - HTML report export (delegates to ReActTraceRenderer)

Data Flow (record_turn):
  1. Caller provides raw LLM I/O data (prompt, output, tokens, timing)
  2. Prompt is decomposed into system/user/history segments
  3. Cross-turn diff is computed against the previous turn
  4. TraceRecord is created and appended to the records list
  5. Cross-turn state is updated for the next turn's diff
  6. Diff summary is logged (per-turn visibility)

Thread Safety:
  TraceCollector is NOT thread-safe. The record_turn method updates
  multiple instance variables (_last_prompt, _last_decomp, etc.) in
  sequence, and concurrent calls from multiple threads would interleave
  these updates, producing incorrect diffs and corrupted state.
  If thread-safe operation is needed, wrap calls with an external lock.

Usage:
  >>> from agentic.tracer.collector import TraceCollector
  >>> collector = TraceCollector(model_name="qwen3.6-35b", pricing=(0.004, 0.012))
  >>> collector.record_turn(turn=1, question="2+2?", prompt=..., ...)
  >>> collector.summary()  # {"total_turns": 1, "total_tokens": 222, ...}
"""

from __future__ import annotations

import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from agentic.tracer.diff import (
    PromptDecomposition,
    TurnDiff,
    decompose_prompt,
    compute_diff,
)
from agentic.observability import get_logger

logger = get_logger(__name__)


# ==========================================================================
# TraceRecord
# ==========================================================================

@dataclass
class TraceRecord:
    """Complete per-turn ReAct call record.

    Captures everything needed for:
      - Per-turn analysis (what did the model do this turn?)
      - Cross-turn analysis (how did the prompt change?)
      - Session aggregates (total tokens, cost, error count)
      - HTML report rendering (visual trace output)
    """

    turn: int
    question: str
    timestamp: str
    prompt_full: str
    # prompt_snapshot stores the LAST 500 chars of the prompt (the history
    # tail + trigger). This is sufficient for most debugging without
    # duplicating the full prompt (which is already in prompt_full).
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

    State machine:
      Initial state: _last_prompt = "", no records
      Each record_turn() call:
        - Computes diff against _last_* state
        - Appends TraceRecord
        - Updates _last_* state to current turn's values
      State is reset only by creating a new TraceCollector instance.
    """

    def __init__(self, model_name: str = "",
                 pricing: Tuple[float, float] = (0.004, 0.012)):
        """
        Args:
            model_name: Human-readable model identifier for reports
                (e.g. "qwen3.6-35b", "llama3-8b").
            pricing: (input_price_per_1k_tokens, output_price_per_1k_tokens)
                in RMB. Used for cost estimation in summary().
                Defaults to Qwen2-72B approximate edge pricing.
        """
        self.records: List[TraceRecord] = []
        self.session_start = datetime.now()
        self.model_name = model_name
        self.pricing = pricing  # (input_price_per_1k, output_price_per_1k)

        # Cross-turn state: tracks the previous turn's data for diff computation.
        #
        # These variables are the "memory" of the collector. They are updated
        # at the end of every record_turn() call so the next call can diff
        # against them.
        #
        # NOTE: These are NOT thread-safe — see module-level docstring.
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

        Steps performed (in order):
          1. Decompose: split the full prompt into system/user/history
             segments using regex patterns from _SEPARATORS in diff.py.
          2. Diff: compute cross-turn differences against the previous
             turn's data (_last_prompt, _last_decomp, etc.). For turn 1,
             prev_prompt is an empty string and the diff shows 0% reuse.
          3. Store: create a TraceRecord with all parsed and computed
             data, append to records list.
          4. Update: set _last_* variables to current turn's values for
             the next call's diff computation.
          5. Log: emit per-turn diff summary via logger.diff().

        Cross-turn state flow:
          record_turn(turn=1) -> _last_prompt = prompt1, _last_decomp = decomp1
          record_turn(turn=2) -> diff(turn2, prompt2, _last_prompt=prompt1, ...)
                                 _last_prompt = prompt2, _last_decomp = decomp2
          record_turn(turn=3) -> diff(turn3, prompt3, _last_prompt=prompt2, ...)
          ...etc.

        Args:
            turn:             Turn number (1-indexed)
            question:         The user's question for this turn
            prompt:           Full prompt sent to the LLM
            raw_output:       Raw (unprocessed) LLM output text
            cleaned_output:   Preprocessed/parsed LLM output
            thought:          Extracted Thought text
            action:           Extracted Action text
            action_input:     Extracted Action Input text
            observation:      Tool output returned to the agent
            prompt_tokens:    Token count of the input prompt
            completion_tokens: Token count of the LLM output
            duration_ms:      Wall-clock duration of the LLM call
            parse_error:      Error string if parsing failed (None if success)
            status:           "success" or "error"
        """
        # Step 1: Decompose the prompt into segments.
        # This is always done fresh because the prompt changes each turn
        # (history grows, question may change).
        decomp = decompose_prompt(prompt)

        # Step 2: Compute cross-turn diff against the previous turn.
        # For turn 1, _last_prompt is "" and _last_decomp is None,
        # so compute_diff returns a baseline diff (0% reuse, etc.).
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

        # Step 3: Create and store the TraceRecord.
        # The timestamp uses millisecond precision for accurate
        # turn-by-turn timing in the HTML report.
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

        # Step 4: Update cross-turn state for the next diff.
        # ORDER MATTERS: these assignments must happen AFTER the
        # diff computation above, otherwise the diff would be
        # comparing current_turn with current_turn (100% reuse).
        self._last_prompt = prompt
        self._last_decomp = decomp
        self._last_response = cleaned_output
        self._last_thought = thought
        self._last_action = action

        # Step 5: Log the diff summary for real-time visibility.
        # Turn 1 gets a baseline log showing the initial prompt structure.
        # Turns 2+ get the reuse% and change summary.
        if diff and turn > 1:
            logger.diff(diff.summary_line)
        elif diff and turn == 1:
            logger.diff(
                f"initial prompt | system={decomp.system_len}ch "
                f"user={decomp.user_len}ch history={decomp.history_len}ch"
            )

    # ── Aggregation ──────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Compute session-level aggregate statistics.

        Statistics derived:

          Token counts:
            - total_prompt_tokens: sum of all input tokens across turns
            - total_completion_tokens: sum of all output tokens across turns
            - total_tokens: total LLM consumption (billing basis)

          Cache hit rate:
            - estimated_cached_tokens: sum of cached_tokens from each
              turn's diff (char-level common prefix / 3.5 chars/token)
            - estimated_new_tokens: sum of new_tokens from each turn's diff
            - cache_hit_rate: cached_tokens / (cached_tokens + new_tokens) * 100
            - avg_prompt_reuse_pct: average of per-turn prompt_reuse_pct
              (excludes turn 1 which is always 0%)

          Derivations:
            - cache_hit_rate measures how much of the prompt content was
              reusable across turns from the LLM's KV-cache. Higher is better
              (lower latency, lower cost).
            - avg_prompt_reuse_pct measures the average overlap between
              consecutive prompts at the character level. This is a proxy
              for cache_hit_rate but measured differently.
            - The two measures (cache_hit_rate vs avg_prompt_reuse_pct) may
              diverge because cache_hit_rate includes turn 1 in the
              denominator (treating it as all "new" tokens) while
              avg_prompt_reuse_pct excludes turn 1.

          System change canary:
            - Count of turns where system_unchanged == False.
            - Should ALWAYS be 0 in normal operation.
            - Any non-zero value indicates a system prompt drift anomaly
              that invalidates the KV-cache and requires investigation.

          Cost:
            - cost_rmb = (total_prompt_tokens * input_price + total_completion_tokens * output_price) / 1000
            - Pricing is provided at init via the pricing tuple.

        Returns:
            Dict with full session statistics (see implementation for keys).
        """
        total_turns = len(self.records)
        total_prompt = sum(r.prompt_tokens for r in self.records)
        total_completion = sum(r.completion_tokens for r in self.records)
        total_duration = sum(r.duration_ms for r in self.records)
        error_turns = [r for r in self.records if r.status != "success"]
        tool_calls = [r for r in self.records
                      if r.action not in ("final_answer", "none", "")]

        # Cache statistics:
        # cached_tokens = common_prefix_len / 3.5 per turn
        # new_tokens = new_prompt_len / 3.5 per turn
        # cache_hit_rate = cached / (cached + new) as percentage
        #
        # Note: total_est = total_cached + total_new represents the
        # estimated total prompt tokens from the diff perspective.
        # This may differ slightly from total_prompt (actual tokeniser
        # count) due to the 3.5 chars/token approximation.
        total_cached = sum(r.diff.cached_tokens for r in self.records if r.diff)
        total_new = sum(r.diff.new_tokens for r in self.records if r.diff)
        total_est = total_cached + total_new
        cache_hit_rate = (total_cached / total_est * 100) if total_est > 0 else 0.0

        # System change canary — should always be 0.
        # If non-zero, the system prompt changed mid-session which
        # invalidates the entire KV-cache and is a serious anomaly.
        system_change_count = sum(
            1 for r in self.records
            if r.diff and not r.diff.system_unchanged
        )

        # Average prompt reuse (exclude turn 1).
        # Turn 1's reuse_pct is always 0% (no previous prompt to
        # compare against), so including it would unfairly depress
        # the average.
        reuse_rates = [r.diff.prompt_reuse_pct
                       for r in self.records[1:] if r.diff]
        avg_reuse = sum(reuse_rates) / len(reuse_rates) if reuse_rates else 0.0

        # Cost calculation in RMB:
        #   Cost = (prompt_tokens * input_token_price + completion_tokens * output_token_price) / 1000
        # The division by 1000 converts from per-1K-tokens to per-token.
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
        """Print ASCII art summary to terminal.

        Format is a box-drawn table with sections for model info,
        token usage, cache analysis, and cost. Designed for quick
        visual scan during development and debugging.
        """
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
        """Export self-contained HTML trace report.

        Delegates rendering to ReActTraceRenderer. The HTML file is
        self-contained (no external dependencies) — it embeds all CSS
        and JS inline for easy sharing and archival.

        Args:
            filepath: Output path for the HTML file.
        """
        from agentic.tracer.renderer import render_html
        render_html(self.records, self.summary(), self.session_start, filepath)


# ==========================================================================
# Helpers
# ==========================================================================

def _bar(pct: float, width: int) -> str:
    """Draw a terminal progress bar.

    Args:
        pct: Percentage value (0-100) to visualise.
        width: Character width of the bar.

    Returns:
        String of filled (█) and empty (░) blocks representing the
        percentage visually.

    Example:
        >>> _bar(73.5, 10)
        '███████░░░'
    """
    filled = int(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)
