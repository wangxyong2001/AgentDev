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
        """
        测试场景：验证 decompose_prompt() 正确提取 system prompt 片段
        参数：PROMPT — 包含 <|im_start|>system ... <|im_end|> 的 Qwen2 格式 prompt
        测试逻辑：(1) 调用 decompose_prompt(self.PROMPT) (2) 检查返回对象的 system_prompt 属性
        预期结果：system_prompt 包含 "ReAct agent" 和 "calculator"
        成功条件："ReAct agent" in d.system_prompt 且 "calculator" in d.system_prompt
        """
        d = decompose_prompt(self.PROMPT)
        assert "ReAct agent" in d.system_prompt
        assert "calculator" in d.system_prompt

    def test_user_segment(self):
        """
        测试场景：验证 decompose_prompt() 正确提取 user message 片段
        参数：PROMPT — user message 为 "What is 2+2?"
        测试逻辑：(1) 调用 decompose_prompt (2) 检查 user_message 属性
        预期结果：user_message == "What is 2+2?"
        成功条件："What is 2+2?" in d.user_message
        """
        d = decompose_prompt(self.PROMPT)
        assert "What is 2+2?" in d.user_message

    def test_history_segment(self):
        """
        测试场景：验证 decompose_prompt() 正确提取 assistant history 片段
        参数：PROMPT — history 包含 Observation: 4
        测试逻辑：(1) 调用 decompose_prompt (2) 检查 history_text 包含 Observation
        预期结果：history_text 包含 "Observation: 4"
        成功条件："Observation: 4" in d.history_text
        """
        d = decompose_prompt(self.PROMPT)
        assert "Observation: 4" in d.history_text

    def test_length_properties(self):
        """
        测试场景：验证 decompose_prompt() 的 system_len/user_len/history_len 均为正数
        参数：PROMPT
        测试逻辑：(1) 调用 decompose_prompt (2) 检查三个长度属性均 > 0
        预期结果：三个段的长度均为正整数
        成功条件：d.system_len > 0 且 d.user_len > 0 且 d.history_len > 0
        """
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
        """
        测试场景：验证第二轮对话的 compute_diff() 正确计算 common_prefix_len 和 system_unchanged
        参数：PROMPT1（第一轮）和 PROMPT2（第二轮），system 相同、user 不同、history 递增
        测试逻辑：(1) 分别分解两个 prompt (2) 调用 compute_diff (3) 检查 common_prefix_len > 0 和 system_unchanged == True
        预期结果：common_prefix_len 为正（system + 公共部分被复用），system_unchanged 为 True
        成功条件：diff.common_prefix_len > 0 且 diff.system_unchanged == True
        """
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
        """
        测试场景：验证首轮对话（无历史）时 compute_diff() 的边界行为
        参数：turn=1, prev_prompt=""（空）, prev_decomp=None, prev_thought="", prev_action=""
        测试逻辑：(1) 仅分解当前 prompt (2) 调用 compute_diff 传入空历史 (3) 检查 thought/action 标记为 "new"
        预期结果：common_prefix_len == 0，thought_vs_prev == "new"，action_vs_prev == "new"
        成功条件：diff.thought_vs_prev == "new" 且 diff.action_vs_prev == "new" 且 diff.common_prefix_len == 0
        """
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
        """
        测试场景（fixture）：创建一个使用测试模型和模拟定价的 TraceCollector 实例
        参数：model_name="test-model", pricing=(0.004, 0.012) — 模拟 qwen3.6-35b 定价
        测试逻辑：直接返回 TraceCollector 实例
        预期结果：返回一个可用的 TraceCollector
        成功条件：isinstance(collector, TraceCollector)
        """
        return TraceCollector(model_name="test-model", pricing=(0.004, 0.012))

    def test_record_turn(self, collector):
        """
        测试场景：验证 record_turn() 正确记录一轮完整的 ReAct 循环
        参数：turn=1, 包含 question/prompt/raw_output/thought/action/observation/token stats
        测试逻辑：(1) 调用 collector.record_turn() 传入完整参数 (2) 检查 records 长度 (3) 检查记录的 turn 和 status
        预期结果：records 列表包含 1 条记录，turn==1，status=="success"
        成功条件：len(collector.records) == 1 且 records[0].turn == 1 且 records[0].status == "success"
        """
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
        """
        测试场景：验证 record_turn() 正确记录解析失败的轮次
        参数：turn=1，status="parse_error"，parse_error="bad format"
        测试逻辑：(1) 调用 record_turn 传入 parse_error 和 status (2) 检查记录的 parse_error 字段
        预期结果：记录的 parse_error 字段正确保存，status 为 "parse_error"
        成功条件：collector.records[0].parse_error == "bad format"
        """
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
        """
        测试场景：验证 summary() 方法正确聚合会话级统计数据
        参数：最终答案为 "42" 的单轮 final_answer 记录
        测试逻辑：(1) 记录一轮 final_answer (2) 调用 collector.summary() (3) 检查 turns/tokens/cost/system_changes
        预期结果：total_turns=1, total_tokens=120(100+20), cost_rmb≈0.00064, system_change_count=0
        成功条件：s["total_turns"]==1 且 s["total_tokens"]==120 且 s["cost_rmb"]==pytest.approx(0.00064, rel=0.01) 且 s["system_change_count"]==0
        """
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