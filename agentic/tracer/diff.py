"""
Cross-turn Diff Algorithm — pure functions for prompt comparison and cache estimation.

Key algorithms:
  - decompose_prompt: split ReAct prompt into system/user/history segments
  - compute_diff:      character-level common-prefix comparison between turns
  - estimate_cache:    token-level KV-cache hit rate estimation (3.5 char/token)

Stateless — no side effects, no global state. Independently testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# ==========================================================================
# Data types
# ==========================================================================

@dataclass
class PromptDecomposition:
    """
    ReAct prompt decomposed into three segments.

    Segments:
      system_prompt: Fixed ReAct rules + tool definitions (KV-cache best friend)
      user_message:  User question or Observation injection (varies per turn)
      history_text:  Accumulated Thought/Action/Observation records (grows each turn)
      full_prompt:    Original complete prompt (raw input)
    """

    system_prompt: str
    user_message: str
    history_text: str
    full_prompt: str

    @property
    def system_len(self) -> int:
        """Character length of system prompt (should remain constant)."""
        return len(self.system_prompt)

    @property
    def user_len(self) -> int:
        """Character length of user message."""
        return len(self.user_message)

    @property
    def history_len(self) -> int:
        """Character length of history (grows each turn)."""
        return len(self.history_text)


@dataclass
class TurnDiff:
    """
    Full cross-turn difference analysis.

    Dimensions:
      Prompt:   common_prefix_len, reuse_pct, system_unchanged, history_lines_added
      Response: thought_vs_prev, action_vs_prev, response_len_delta
      Cache:    estimated cached_tokens, new_tokens
    """

    turn: int

    # Prompt dimensions
    common_prefix_len: int
    new_prompt_len: int
    prompt_reuse_pct: float
    system_unchanged: bool
    history_lines_added: int
    user_changed: bool

    # Response dimensions
    thought_vs_prev: str  # "new" | "same" | "different"
    action_vs_prev: str   # "new" | "same" | "different"
    response_len_delta: int

    # Cache estimation
    cached_tokens: int
    new_tokens: int

    # One-line summary
    summary_line: str


# ==========================================================================
# Prompt decomposition
# ==========================================================================

# Separator patterns for chat templates
_SEPARATORS = {
    "qwen2": {
        "system": r'<\|im_start\|>system\n(.*?)<\|im_end\|>',
        "user": r'<\|im_start\|>user\n(.*?)<\|im_end\|>',
        "assistant": r'<\|im_start\|>assistant\n(.*)',
    },
    "llama3": {
        "system": r'<\|begin_of_text\|><\|start_header_id\|>system<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>',
        "user": r'<\|start_header_id\|>user<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>',
        "assistant": r'<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*)',
    },
}


def decompose_prompt(prompt: str, chat_format: str = "qwen2") -> PromptDecomposition:
    """
    Decompose a ReAct prompt into system/user/history segments.

    Args:
      prompt:       Full ReAct prompt string
      chat_format:  Chat template format ("qwen2" | "llama3")

    Returns:
      PromptDecomposition with three segments

    Note:
      The assistant segment intentionally lacks a closing delimiter —
      this is by design to let the model continue generating.
    """
    seps = _SEPARATORS.get(chat_format, _SEPARATORS["qwen2"])

    system_prompt = ""
    user_message = ""
    history_text = ""

    sys_match = re.search(seps["system"], prompt, re.DOTALL)
    if sys_match:
        system_prompt = sys_match.group(1).strip()

    user_match = re.search(seps["user"], prompt, re.DOTALL)
    if user_match:
        user_message = user_match.group(1).strip()

    asst_match = re.search(seps["assistant"], prompt, re.DOTALL)
    if asst_match:
        history_text = asst_match.group(1).strip()

    return PromptDecomposition(
        system_prompt=system_prompt,
        user_message=user_message,
        history_text=history_text,
        full_prompt=prompt,
    )


# ==========================================================================
# Cross-turn diff
# ==========================================================================

def compute_diff(
    turn: int,
    current_prompt: str,
    prev_prompt: str,
    decomp: PromptDecomposition,
    prev_decomp: Optional[PromptDecomposition],
    thought: str,
    prev_thought: str,
    action: str,
    prev_action: str,
    response_len: int,
    prev_response_len: int,
) -> TurnDiff:
    """
    Compute full cross-turn difference between two consecutive ReAct calls.

    Algorithm:
      1. Character-by-character common prefix scan: O(min(len(a), len(b)))
      2. Cache token estimation: common_len / 3.5 (empirical CN/EN average)
      3. System prompt anomaly detection: direct string comparison
      4. Response semantic comparison: thought/action equality check

    Args:
      turn:              Current turn number
      current_prompt:    This turn's full prompt
      prev_prompt:       Last turn's full prompt (empty for first turn)
      decomp:            This turn's prompt decomposition
      prev_decomp:       Last turn's decomposition (None for first turn)
      thought:           This turn's parsed Thought
      prev_thought:      Last turn's Thought
      action:            This turn's parsed Action
      prev_action:       Last turn's Action
      response_len:      This turn's cleaned output char length
      prev_response_len: Last turn's cleaned output char length

    Returns:
      TurnDiff with all 13 dimensions populated
    """

    # ═══ Prompt dimension: character-level common prefix ═══
    common_len = 0
    min_len = min(len(current_prompt), len(prev_prompt)) if prev_prompt else 0
    for i in range(min_len):
        if current_prompt[i] == prev_prompt[i]:
            common_len = i + 1
        else:
            break

    new_len = len(current_prompt) - common_len
    reuse_pct = (common_len / len(current_prompt) * 100) if len(current_prompt) > 0 else 0.0

    # System prompt anomaly detection
    system_unchanged = True
    if prev_decomp and decomp:
        system_unchanged = (prev_decomp.system_prompt == decomp.system_prompt)

    # History growth
    history_added = 0
    if prev_decomp and decomp:
        prev_lines = prev_decomp.history_text.count('\n')
        curr_lines = decomp.history_text.count('\n')
        history_added = max(0, curr_lines - prev_lines)

    # User message change detection
    user_changed = True
    if prev_decomp and decomp:
        user_changed = (prev_decomp.user_message != decomp.user_message)

    # ═══ Response dimension: semantic comparison ═══
    thought_vs = "new" if not prev_thought else (
        "same" if thought == prev_thought else "different")
    action_vs = "new" if not prev_action else (
        "same" if action == prev_action else "different")
    resp_delta = response_len - prev_response_len if prev_response_len else response_len

    # ═══ Cache dimension: token estimation ═══
    # 3.5 chars/token is empirical for CN/EN mixed text.
    # ±5% accuracy for cost estimation purposes.
    cached_tokens = int(common_len / 3.5)
    new_token_est = int(new_len / 3.5)

    # ═══ One-line summary ═══
    parts = [f"reuse {reuse_pct:.0f}%"]
    if history_added > 0:
        parts.append(f"history +{history_added} lines")
    parts.append(f"user {'changed' if user_changed else 'same'}")
    parts.append(f"Thought:{thought_vs}")
    parts.append(f"Action:{action_vs}")
    if resp_delta != 0:
        parts.append(f"resp {resp_delta:+d} chars")

    return TurnDiff(
        turn=turn,
        common_prefix_len=common_len,
        new_prompt_len=new_len,
        prompt_reuse_pct=reuse_pct,
        system_unchanged=system_unchanged,
        history_lines_added=history_added,
        user_changed=user_changed,
        thought_vs_prev=thought_vs,
        action_vs_prev=action_vs,
        response_len_delta=resp_delta,
        cached_tokens=cached_tokens,
        new_tokens=new_token_est,
        summary_line=" | ".join(parts),
    )
