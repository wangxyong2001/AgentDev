"""
Trace Collector — 每轮记录与会话级聚合。

功能描述:
  管理:
    - TraceRecord 存储（追加列表）
    - 跨轮状态追踪（_last_* 变量）
    - 会话摘要统计（token 数、缓存命中率、成本）
    - HTML 报告导出（委托给 ReActTraceRenderer）

数据流（record_turn）:
  1. 调用方提供原始 LLM I/O 数据（prompt、output、tokens、timing）
  2. 提示词被分解为 system/user/history 片段
  3. 计算与上一轮的跨轮差异
  4. 创建 TraceRecord 并追加到记录列表
  5. 更新跨轮状态，用于下一个轮次的差异计算
  6. 记录差异摘要（每轮可见性）

线程安全:
  TraceCollector 不是线程安全的。record_turn 方法按顺序更新
  多个实例变量（_last_prompt、_last_decomp 等），
  多线程并发调用会交错这些更新，产生错误的差异和损坏的状态。
  如果需要线程安全的操作，请使用外部锁包装调用。

用法:
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
    """
    完整的每轮 ReAct 调用记录。

    功能描述:
      捕获单轮 ReAct 调用的所有数据，用于:
        - 单轮分析（模型本轮做了什么？）
        - 跨轮分析（提示词如何变化？）
        - 会话聚合（总 token 数、成本、错误计数）
        - HTML 报告渲染（可视化 trace 输出）

    参数:
      turn: 轮次号（从 1 开始）
      question: 本轮的用户问题
      timestamp: 毫秒精度的时间戳
      prompt_full: 发送给 LLM 的完整提示词
      prompt_snapshot: 提示词的最后 500 个字符（历史尾部 + 触发词）
      raw_output: 原始（未处理）LLM 输出文本
      cleaned_output: 预处理/解析后的 LLM 输出
      thought: 提取的 Thought 文本
      action: 提取的 Action 文本
      action_input: 提取的 Action Input 文本
      observation: 返回给 Agent 的工具输出
      prompt_tokens: 输入提示词的 token 数
      completion_tokens: LLM 输出的 token 数
      duration_ms: LLM 调用的实际耗时
      decomp: 可选的提示词分解
      diff: 可选的跨轮差异
      parse_error: 解析失败时的错误字符串（成功时为 None）
      status: "success" 或 "error"
    """
    turn: int
    question: str
    timestamp: str
    prompt_full: str
    # prompt_snapshot 存储提示词的最后 500 个字符（历史尾部 + 触发词）。
    # 对于大多数调试已足够，无需复制完整提示词（已存在于 prompt_full 中）。
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
    全链路 ReAct 调用 Trace 收集器。

    功能描述:
      记录每一轮，计算跨轮差异，聚合会话统计，导出 HTML 可视化。

    状态机:
      初始状态: _last_prompt = ""，无记录
      每次 record_turn() 调用:
        - 计算与 _last_* 状态的差异
        - 追加 TraceRecord
        - 将 _last_* 状态更新为当前轮的值
      状态仅通过创建新的 TraceCollector 实例来重置。

    参数:
      model_name: 用于报告的人类可读模型标识符
      pricing: (input_price_per_1k_tokens, output_price_per_1k_tokens)
               单位为人民币。用于 summary() 中的成本估算。
               默认近似为 Qwen2-72B 边缘定价。
    """
    def __init__(self, model_name: str = "",
                 pricing: Tuple[float, float] = (0.004, 0.012)):
        self.records: List[TraceRecord] = []
        self.session_start = datetime.now()
        self.model_name = model_name
        self.pricing = pricing  # (输入价格每千 token, 输出价格每千 token)

        # 跨轮状态: 跟踪上一轮的数据用于差异计算。
        #
        # 这些变量是收集器的"记忆"。它们在每次 record_turn() 调用
        # 结束时更新，以便下一次调用可以与之做差异比较。
        #
        # 注意: 这些不是线程安全的 — 见模块级文档字符串。
        self._last_prompt: str = ""
        self._last_decomp: Optional[PromptDecomposition] = None
        self._last_response: str = ""
        self._last_thought: str = ""
        self._last_action: str = ""

    # ── 记录 ────────────────────────────────────────────────────

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
        记录一个完整的 ReAct 轮次。

        执行步骤（按顺序）：

        **第 1 步: 分解**
          使用 diff.py 中 _SEPARATORS 的正则模式将完整提示词拆分为
          system/user/history 片段。

        **第 2 步: 差异计算**
          计算与上一轮数据的跨轮差异（_last_prompt、_last_decomp 等）。
          对于第 1 轮，prev_prompt 为空字符串，差异显示 0% 复用率。

        **第 3 步: 存储**
          创建包含所有解析和计算数据的 TraceRecord，追加到记录列表。

        **第 4 步: 更新**
          将 _last_* 变量设置为当前轮的值，用于下一轮差异计算。

        **第 5 步: 记录**
          通过 logger.diff() 输出每轮差异摘要。

        跨轮状态流:
          record_turn(turn=1) -> _last_prompt = prompt1, _last_decomp = decomp1
          record_turn(turn=2) -> diff(turn2, prompt2, _last_prompt=prompt1, ...)
                                _last_prompt = prompt2, _last_decomp = decomp2
          record_turn(turn=3) -> diff(turn3, prompt3, _last_prompt=prompt2, ...)
          ...以此类推

        参数:
            turn:             轮次号（从 1 开始）
            question:         本轮的用户问题
            prompt:           发送给 LLM 的完整提示词
            raw_output:       原始（未处理）LLM 输出文本
            cleaned_output:   预处理/解析后的 LLM 输出
            thought:          提取的 Thought 文本
            action:           提取的 Action 文本
            action_input:     提取的 Action Input 文本
            observation:      返回给 Agent 的工具输出
            prompt_tokens:    输入提示词的 token 数
            completion_tokens: LLM 输出的 token 数
            duration_ms:      LLM 调用的实际耗时（毫秒）
            parse_error:      解析失败时的错误字符串（成功时为 None）
            status:           "success" 或 "error"
        """
        # 第 1 步: 将提示词分解为片段。
        # 每次都重新分解，因为每轮提示词都会变化（历史记录增长，问题可能变化）。
        decomp = decompose_prompt(prompt)

        # 第 2 步: 计算与上一轮的跨轮差异。
        # 对于第 1 轮，_last_prompt 为 "" 且 _last_decomp 为 None，
        # 因此 compute_diff 返回基准差异（0% 复用率等）。
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

        # 第 3 步: 创建并存储 TraceRecord。
        # 时间戳使用毫秒精度，以确保 HTML 报告中每轮时序准确。
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

        # 第 4 步: 更新跨轮状态，用于下一个差异计算。
        # 顺序很关键: 这些赋值必须在上述差异计算之后执行，
        # 否则差异会比较当前轮与当前轮（100% 复用率）。
        self._last_prompt = prompt
        self._last_decomp = decomp
        self._last_response = cleaned_output
        self._last_thought = thought
        self._last_action = action

        # 第 5 步: 记录差异摘要，用于实时可见性。
        # 第 1 轮记录基准日志，显示初始提示词结构。
        # 第 2+ 轮记录复用率和变化摘要。
        if diff and turn > 1:
            logger.diff(diff.summary_line)
        elif diff and turn == 1:
            logger.diff(
                f"初始提示词 | system={decomp.system_len}字符 "
                f"user={decomp.user_len}字符 history={decomp.history_len}字符"
            )

    # ── 聚合 ──────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """
        计算会话级聚合统计。

        功能描述:
          汇总所有记录的轮次数据，生成完整的会话报告。

        统计指标:

          Token 计数:
            - total_prompt_tokens: 所有轮次的输入 token 总和
            - total_completion_tokens: 所有轮次的输出 token 总和
            - total_tokens: LLM 总消耗（计费基础）

          缓存命中率:
            - estimated_cached_tokens: 每轮缓存 token 的总和
              （字符级公共前缀 / 3.5 字符/token）
            - estimated_new_tokens: 每轮新 token 的总和
            - cache_hit_rate: cached_tokens / (cached_tokens + new_tokens) * 100
            - avg_prompt_reuse_pct: 每轮 prompt_reuse_pct 的平均值
              （排除始终为 0% 的第 1 轮）

          衍生指标:
            - cache_hit_rate 衡量提示词内容中可从 LLM 的 KV-cache 中
              跨轮复用的比例。越高越好（更低延迟，更低成本）。
            - avg_prompt_reuse_pct 衡量相邻提示词在字符级别的平均重叠率。
              这是 cache_hit_rate 的代理指标，但计算方式不同。
            - 两个指标（cache_hit_rate vs avg_prompt_reuse_pct）可能不同，
              因为 cache_hit_rate 在分母中包含第 1 轮（视为所有 token
              "新"），而 avg_prompt_reuse_pct 排除第 1 轮。

          系统变化金丝雀:
            - system_unchanged == False 的轮次计数。
            - 正常操作下应始终为 0。
            - 任何非零值表示系统提示词漂移异常，
              使 KV-cache 失效并需要调查。

          成本:
            - cost_rmb = (total_prompt_tokens * input_price +
                          total_completion_tokens * output_price) / 1000
            - 定价在 init 时通过 pricing 元组提供。

        返回:
            包含完整会话统计信息的字典（参见实现了解键名）。
        """
        total_turns = len(self.records)
        total_prompt = sum(r.prompt_tokens for r in self.records)
        total_completion = sum(r.completion_tokens for r in self.records)
        total_duration = sum(r.duration_ms for r in self.records)
        error_turns = [r for r in self.records if r.status != "success"]
        tool_calls = [r for r in self.records
                      if r.action not in ("final_answer", "none", "")]

        # 缓存统计:
        # cached_tokens = common_prefix_len / 3.5 每轮
        # new_tokens = new_prompt_len / 3.5 每轮
        # cache_hit_rate = cached / (cached + new) 百分比
        #
        # 注意: total_est = total_cached + total_new 表示从差异角度
        # 估计的提示词 token 总数。这可能与实际 token 数略有差异
        # （来自实际 tokeniser 计数），因 3.5 字符/token 的近似。
        total_cached = sum(r.diff.cached_tokens for r in self.records if r.diff)
        total_new = sum(r.diff.new_tokens for r in self.records if r.diff)
        total_est = total_cached + total_new
        cache_hit_rate = (total_cached / total_est * 100) if total_est > 0 else 0.0

        # 系统变化金丝雀 — 应始终为 0。
        # 如果非零，系统提示词在会话中途发生了变化，
        # 使整个 KV-cache 失效，这是一个严重异常。
        system_change_count = sum(
            1 for r in self.records
            if r.diff and not r.diff.system_unchanged
        )

        # 平均提示词复用率（排除第 1 轮）。
        # 第 1 轮的 reuse_pct 始终为 0%（没有之前的提示词可比对），
        # 因此包含它会不公平地拉低平均值。
        reuse_rates = [r.diff.prompt_reuse_pct
                       for r in self.records[1:] if r.diff]
        avg_reuse = sum(reuse_rates) / len(reuse_rates) if reuse_rates else 0.0

        # 成本计算（人民币）:
        #   成本 = (prompt_tokens * 输入 token 价格 + completion_tokens * 输出 token 价格) / 1000
        # 除以 1000 将每千 token 价格转换为每 token 价格。
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

    # ── 报告 ────────────────────────────────────────────────────

    def print_summary(self):
        """
        在终端中打印 ASCII 艺术风格摘要。

        功能描述:
          格式为带框表格，分为模型信息、token 使用、缓存分析和成本等部分。
          专为开发和调试期间的快速视觉扫描设计。
        """
        s = self.summary()
        cache_bar = _bar(s['cache_hit_rate'], 30)
        reuse_bar = _bar(s['avg_prompt_reuse_pct'], 30)
        logger.info(f"""
╔══════════════════════════════════════════════════════════╗
║              ReAct Trace 报告                            ║
╠══════════════════════════════════════════════════════════╣
║  模型: {s['model'][:46]:<46s} ║
║  轮次: {s['total_turns']:<5}  耗时: {s['session_duration_s']:.1f}s{'':>30s} ║
╠══════════════════════════════════════════════════════════╣
║  Token 使用                                             ║
║    Prompt token:      {s['total_prompt_tokens']:>8,d}                          ║
║    Completion token:  {s['total_completion_tokens']:>8,d}                          ║
║    总计:              {s['total_tokens']:>8,d}                          ║
╠══════════════════════════════════════════════════════════╣
║  缓存分析（估计）                                       ║
║    缓存 token:        {s['estimated_cached_tokens']:>8,d}  [{cache_bar}] ║
║    新 token:          {s['estimated_new_tokens']:>8,d}                          ║
║    缓存命中率:        {s['cache_hit_rate']:>6.1f}%                             ║
║    平均提示词复用:    {s['avg_prompt_reuse_pct']:>6.1f}%  [{reuse_bar}] ║
║    系统变化:          {s['system_change_count']}（应为 0）                         ║
╠══════════════════════════════════════════════════════════╣
║  成本: ¥{s['cost_rmb']:.6f}  |  工具: {s['tool_calls']}  |  错误: {s['error_turns']}  |  LLM: {s['total_duration_ms']:.0f}ms ║
╚══════════════════════════════════════════════════════════╝
""")

    def export_html(self, filepath: str):
        """
        导出自包含的 HTML Trace 报告。

        功能描述:
          委托给 ReActTraceRenderer 进行渲染。
          HTML 文件是自包含的（无外部依赖）— 嵌入所有 CSS 和 JS，
          便于共享和归档。

        参数:
            filepath: HTML 文件的输出路径。
        """
        from agentic.tracer.renderer import render_html
        render_html(self.records, self.summary(), self.session_start, filepath)


# ==========================================================================
# 辅助函数
# ==========================================================================

def _bar(pct: float, width: int) -> str:
    """
    绘制终端进度条。

    功能描述:
      根据百分比值生成视觉化的填充条。

    参数:
        pct: 百分比值（0-100），用于可视化。
        width: 进度条的字符宽度。

    返回:
        由填充块（█）和空块（░）组成的字符串，表示百分比。

    示例:
        >>> _bar(73.5, 10)
        '███████░░░'
    """
    filled = int(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)
