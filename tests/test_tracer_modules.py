"""
Tests for llama.tracer — diff algorithm and collector.
"""

import pytest
from agentic.tracer.diff import (
    PromptDecomposition,
    TurnDiff,
    decompose_prompt,
    compute_diff,
)
from agentic.tracer.collector import TraceCollector, TraceRecord


# ==========================================================================
# decompose_prompt
# ==========================================================================

class TestDecomposePrompt:
    """Prompt decomposition into system/user/history."""

    PROMPT = (
        "<|im_start|>system\n"
        "You are a ReAct agent. Tools: calculator.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "What is 2+2?\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "Thought: need to calculate\n"
        "Action: calculator\n"
        "Action Input: 2+2\n"
        "Observation: 4\n"
        "Thought:"
    )

    def test_system_segment(self):
        d = decompose_prompt(self.PROMPT)
        assert "ReAct agent" in d.system_prompt
        assert "calculator" in d.system_prompt

    def test_user_segment(self):
        d = decompose_prompt(self.PROMPT)
        assert "What is 2+2?" in d.user_message

    def test_history_segment(self):
        d = decompose_prompt(self.PROMPT)
        assert "Observation: 4" in d.history_text

    def test_length_properties(self):
        d = decompose_prompt(self.PROMPT)
        assert d.system_len > 0
        assert d.user_len > 0
        assert d.history_len > 0


# ==========================================================================
# compute_diff
# ==========================================================================

class TestComputeDiff:
    """Cross-turn diff computation."""

    PROMPT1 = (
        "<|im_start|>system\nSYS\n<|im_end|>\n"
        "<|im_start|>user\nQ1\n<|im_end|>\n"
        "<|im_start|>assistant\nHIST1\nThought:"
    )
    PROMPT2 = (
        "<|im_start|>system\nSYS\n<|im_end|>\n"
        "<|im_start|>user\nQ2\n<|im_end|>\n"
        "<|im_start|>assistant\nHIST1\nHIST2\nThought:"
    )

    def test_common_prefix(self):
        decomp1 = decompose_prompt(self.PROMPT1)
        decomp2 = decompose_prompt(self.PROMPT2)
        diff = compute_diff(
            turn=2,
            current_prompt=self.PROMPT2,
            prev_prompt=self.PROMPT1,
            decomp=decomp2,
            prev_decomp=decomp1,
            thought="t2", prev_thought="t1",
            action="a2", prev_action="a1",
            response_len=50, prev_response_len=40,
        )
        assert diff.common_prefix_len > 0
        assert diff.system_unchanged  # SYS is same in both

    def test_first_turn(self):
        decomp = decompose_prompt(self.PROMPT1)
        diff = compute_diff(
            turn=1,
            current_prompt=self.PROMPT1,
            prev_prompt="",
            decomp=decomp,
            prev_decomp=None,
            thought="t1", prev_thought="",
            action="a1", prev_action="",
            response_len=40, prev_response_len=0,
        )
        assert diff.thought_vs_prev == "new"
        assert diff.action_vs_prev == "new"
        assert diff.common_prefix_len == 0


# ==========================================================================
# TraceCollector
# ==========================================================================

class TestTraceCollector:
    """Trace recording and aggregation."""

    @pytest.fixture
    def collector(self):
        return TraceCollector(model_name="test-model", pricing=(0.004, 0.012))

    def test_record_turn(self, collector):
        collector.record_turn(
            turn=1, question="Q?", prompt="<|im_start|>system\nSYS\n<|im_end|>\n<|im_start|>user\nQ?\n<|im_end|>\n<|im_start|>assistant\nThought:",
            raw_output="Thought: t\nAction: calc\nAction Input: 1+1",
            cleaned_output="Thought: t\nAction: calc\nAction Input: 1+1",
            thought="t", action="calc", action_input="1+1",
            observation="2",
            prompt_tokens=50, completion_tokens=10, duration_ms=100,
        )
        assert len(collector.records) == 1
        assert collector.records[0].turn == 1
        assert collector.records[0].status == "success"

    def test_record_parse_error(self, collector):
        collector.record_turn(
            turn=1, question="Q?", prompt="prompt",
            raw_output="garbled", cleaned_output="garbled",
            thought="(Parse Error)", action="none", action_input="none",
            observation="garbled",
            prompt_tokens=30, completion_tokens=5, duration_ms=50,
            parse_error="bad format", status="parse_error",
        )
        assert collector.records[0].parse_error == "bad format"

    def test_summary(self, collector):
        collector.record_turn(
            turn=1, question="Q?", prompt="prompt",
            raw_output="Thought: t\nAction: final_answer\nAction Input: 42",
            cleaned_output="Thought: t\nAction: final_answer\nAction Input: 42",
            thought="t", action="final_answer", action_input="42",
            observation="(final answer)",
            prompt_tokens=100, completion_tokens=20, duration_ms=200,
        )
        s = collector.summary()
        assert s["total_turns"] == 1
        assert s["total_tokens"] == 120
        assert s["cost_rmb"] == pytest.approx(0.00064, rel=0.01)
        assert s["system_change_count"] == 0
