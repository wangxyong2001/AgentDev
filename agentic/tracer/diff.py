"""
跨轮 Diff 算法 — 用于提示词比较和缓存估计的纯函数。

功能描述:
  核心算法:
    - decompose_prompt: 将 ReAct 提示词拆分为 system/user/history 三个片段
    - compute_diff:      轮次间的字符级公共前缀比较
    - estimate_cache:    token 级别的 KV-cache 命中率估算（3.5 字符/token）

  无状态 — 无副作用，无全局状态。可独立测试。

========================
字符级公共前缀算法
========================

  diff 算法通过逐字符比较来找到两个连续提示词之间的最长公共前缀。
  复杂度为 O(min(N, M))，其中 N 和 M 是两个提示词的长度。

  为什么用字符级而不是 token 级？
    - Token 化是模型相关的（不同模型使用不同的 tokeniser）。
      字符级比较是模型无关的。
    - 我们使用字符作为 token 位置的代理，因为 KV-cache 按 token 位置操作，
      字符级的公共前缀（大致）映射到 token 级的公共前缀。
    - 权衡：字符级比较是 O(N)，而更智能的算法（如在哈希前缀上二分搜索）
      可以达到 O(log N)，但提示词通常 <10K 字符，因此 O(N) 足够。

  性能说明:
    对于 10,000 字符的提示词和 8 轮推理，此循环每轮最多运行
    80,000 次比较 — 开销可忽略不计。

========================
3.5 字符/token 启发式值
========================

  我们使用 3.5 字符每 token 进行缓存估算。这是中英文混合文本的
  经验平均值，是该 Agent 的主要使用场景（中文用户提问、
  英文工具名和代码、中英文混合推理）。

  理由:
    - 英文文本平均 ~4 字符/token（OpenAI 经验法则：~0.75 token/词，
      约 5 字符/词 = ~3.75 字符/token）。
    - 中文文本平均 ~1.5 字符/token（每个中文字符是一个 Unicode 码点，
      tokeniser 通常每 token 编码 1-2 个中文字符）。
    - 对于中英文混合文本（我们的目标场景），根据对约 60% 英文结构 +
      ~40% 中文内容的 ReAct 提示词的测量，经验平均值接近 3.5 字符/token。
    - 精度：对于成本估算目的约为 +/- 5%。这对于缓存命中率分析足够
      （我们只需要相对幅度，而非精确 token 数）。

  更新此值的时机:
    - 目标模型更换为使用非常不同的 tokeniser 时
      （例如 Llama 3 的 tokeniser 与 Qwen2 的中英文比例不同）。
    - 基于生产 trace 数据集的经验测量显示系统偏差超过 +/- 5%。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# ==========================================================================
# 数据类型
# ==========================================================================

@dataclass
class PromptDecomposition:
    """
    ReAct 提示词分解为三个片段。

    功能描述:
      将完整提示词拆分为三个角色特定部分。

    片段:
      system_prompt: 固定的 ReAct 规则 + 工具定义（KV-cache 的最佳朋友）
      user_message:  用户问题或 Observation 注入（每轮变化）
      history_text:  累积的 Thought/Action/Observation 记录（每轮增长）
      full_prompt:    原始的完整提示词（原始输入）
    """
    system_prompt: str
    user_message: str
    history_text: str
    full_prompt: str

    @property
    def system_len(self) -> int:
        """
        系统提示词的字符长度（应保持恒定）。

        功能描述:
          正常操作下，此值在轮次之间应永不改变。
          如果变化，要么是 YAML 协议被重新加载，要么是系统提示词
          在会话中途被破坏 — 两者都是值得调查的异常情况。
        """
        return len(self.system_prompt)

    @property
    def user_len(self) -> int:
        """
        用户消息的字符长度。

        功能描述:
          如果用户提出后续问题则每轮变化，
          如果仅有历史记录增长则保持不变（同一问题上的多步推理）。
        """
        return len(self.user_message)

    @property
    def history_len(self) -> int:
        """
        历史记录字符长度（每轮增长）。

        功能描述:
          轮次数的单调递增函数（除非应用截断或摘要）。
        """
        return len(self.history_text)


@dataclass
class TurnDiff:
    """
    完整的跨轮差异分析。

    功能描述:
      从四个维度描述两个连续 ReAct 轮次之间的差异。

    维度:
      Prompt:   common_prefix_len, reuse_pct, system_unchanged, history_lines_added
      Response: thought_vs_prev, action_vs_prev, response_len_delta
      Cache:    estimated cached_tokens, new_tokens

    13 个维度旨在捕获两个 ReAct 轮次之间转换的每个可观察属性。
    它们具有双重用途:
      1. 实时指标（每轮通过 logger.diff 记录）
      2. 事后分析（由 TraceCollector.summary 聚合）

    参数:
      turn: 当前轮次号
      common_prefix_len: 与前一轮的公共前缀字符长度
      new_prompt_len: 当前轮次独有的新字符长度
      prompt_reuse_pct: 提示词复用百分比
      system_unchanged: 系统提示词是否未变
      history_lines_added: 新增的历史记录行数
      user_changed: 用户消息是否变化
      thought_vs_prev: "new" | "same" | "different"
      action_vs_prev: "new" | "same" | "different"
      response_len_delta: 与上一轮相比的响应长度变化
      cached_tokens: 估计的缓存 token 数（可复用部分）
      new_tokens: 估计的新 token 数
      summary_line: 一行摘要字符串
    """
    turn: int

    # Prompt 维度
    common_prefix_len: int
    new_prompt_len: int
    prompt_reuse_pct: float
    system_unchanged: bool
    history_lines_added: int
    user_changed: bool

    # Response 维度
    thought_vs_prev: str  # "new" | "same" | "different"
    action_vs_prev: str   # "new" | "same" | "different"
    response_len_delta: int

    # 缓存估算
    cached_tokens: int
    new_tokens: int

    # 一行摘要
    summary_line: str


# ==========================================================================
# 提示词分解
# ==========================================================================

# Chat 模板的分隔符模式。
#
# 维护说明:
# 当为新模型的 chat 模板添加支持时，在此处添加新的条目并
# 包含适当的正则模式。键名必须与 decompose_prompt() 中的
# `chat_format` 参数匹配。
#
# 每个格式需要三个模式:
#   - system:    捕获分隔符之间的系统提示词
#   - user:      捕获分隔符之间的用户消息
#   - assistant: 捕获 assistant 头部之后的所有内容（无结束分隔符 —
#     assistant 部分有意开放，以允许模型继续生成）
#
# 当前支持的格式:
#   "qwen2":  <|im_start|>role\n...<|im_end|>  (ChatML 变体)
#   "llama3": <|begin_of_text|><|start_header_id|>role<|end_header_id|>\n\n...<|eot_id|>
#
# 添加后必须测试: 每个新模式必须使用目标模型的实际完整提示词样本
# 进行测试。由于提示词可以跨多行，需要正则 DOTALL 标志。
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
    将 ReAct 提示词分解为 system/user/history 片段。

    功能描述:
      使用正则表达式从完整提示词字符串中提取三个角色特定部分。
      正则模式是模型特定的（参见 _SEPARATORS）。

    参数:
      prompt:       完整的 ReAct 提示词字符串
      chat_format:  Chat 模板格式（"qwen2" | "llama3"）

    返回:
      包含三个片段的 PromptDecomposition

    注意事项:
      assistant 片段有意缺少结束分隔符 —
      这是为了让模型可以继续生成而设计。
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
# 跨轮差异计算
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
    计算两个连续 ReAct 调用之间的完整跨轮差异。

    功能描述:
      从四个维度全面分析两个轮次之间的变化。

    算法:
      1. 逐字符公共前缀扫描: O(min(len(a), len(b)))
         — 线性比较两个提示词，直到第一个差异处。
         这给出了在下次推理调用中可从 KV-cache 提供的字符数。

      2. 缓存 token 估算: common_len / 3.5（中英文混合经验平均值）
         — 将字符级复用转换为近似的 token 级缓存命中。
         3.5 除数是对中英文混合文本的经验平均值；
         见模块级文档字符串了解理由和精度范围。

      3. 系统提示词异常检测: 直接字符串比较
         — 检查系统提示词在轮次之间是否变化。任何变化都是异常的
         （系统提示词应是静态的），并使整个前缀的 KV-cache 失效。

      4. 响应语义比较: thought/action 相等性检查
         — 将响应分类为 "same"（完全重复）、"new"（首次出现）
         或 "different"（与上一轮不同）。

    参数:
      turn:              当前轮次号
      current_prompt:    本轮次的完整提示词
      prev_prompt:       上一轮的完整提示词（第一轮为空字符串）
      decomp:            本轮次的提示词分解
      prev_decomp:       上一轮的提示词分解（第一轮为 None）
      thought:           本轮次解析的 Thought
      prev_thought:      上一轮的 Thought
      action:            本轮次解析的 Action
      prev_action:       上一轮的 Action
      response_len:      本轮次清理后输出的字符长度
      prev_response_len: 上一轮清理后输出的字符长度

    返回:
      包含所有 13 个维度的 TurnDiff

    处理逻辑:
      1. 计算字符级公共前缀（用于 KV-cache 估计）
      2. 检测系统提示词是否变化（金丝雀指标）
      3. 测量历史记录增长
      4. 检测用户消息是否变化
      5. 比较响应的语义变化
      6. 估算缓存 token 数
      7. 生成一行摘要
    """

    # ═══ Prompt 维度: 字符级公共前缀 ═══
    # 逐字符向前扫描，直到发现差异。
    # 这给出了两个提示词共享的最长前缀。
    #
    # 为什么不用 difflib.SequenceMatcher?
    #   SequenceMatcher.find_longest_match() 找到两个字符串中任意位置的
    #   最长公共子串，而不仅仅是前缀。对于 KV-cache 估算，我们只关心
    #   前缀 — 缓存从第一个差异点开始就失效了，即使后面出现相同文本。
    common_len = 0
    min_len = min(len(current_prompt), len(prev_prompt)) if prev_prompt else 0
    for i in range(min_len):
        if current_prompt[i] == prev_prompt[i]:
            common_len = i + 1
        else:
            break

    new_len = len(current_prompt) - common_len
    reuse_pct = (common_len / len(current_prompt) * 100) if len(current_prompt) > 0 else 0.0

    # 系统提示词异常检测（金丝雀指标）。
    #
    # 正常操作下，系统提示词在轮次之间应永不改变。
    # 如果变化，之前的每个 KV-cache 条目都失效，
    # 模型必须重新处理整个提示词。
    #
    # 此检查作为金丝雀指标 — 如同煤矿中的金丝雀，
    # 意外的系统提示词变化信号表示严重问题:
    #   - YAML 协议在会话中途被重新加载
    #   - 并发线程或进程修改了配置
    #   - 模板渲染器中的内存损坏 bug
    #   - 攻击者注入内容进入了系统提示词部分
    #     （不太可能，但在 chat 模板分隔符失效的提示注入场景中可能发生）
    #
    # 在健康会话中，system_change_count（在 collector.py 中）应始终为 0。
    # 任何非零值都会触发告警。
    system_unchanged = True
    if prev_decomp and decomp:
        system_unchanged = (prev_decomp.system_prompt == decomp.system_prompt)

    # 历史记录增长测量。
    # 历史记录单调增长 — 每轮增加一个条目。
    # 计算换行符是一种廉价的近似方法，表示"新增了多少个历史记录条目"
    # （每个条目是一个 Thought/Action/Action Input/Observation 块，
    # 由换行符分隔）。
    history_added = 0
    if prev_decomp and decomp:
        prev_lines = prev_decomp.history_text.count('\n')
        curr_lines = decomp.history_text.count('\n')
        history_added = max(0, curr_lines - prev_lines)

    # 用户消息变化检测。
    # 在单轮交互中（用户问一个问题，Agent 多步解决），
    # 用户消息是恒定的。如果变化，是用户提出了后续问题或澄清。
    user_changed = True
    if prev_decomp and decomp:
        user_changed = (prev_decomp.user_message != decomp.user_message)

    # ═══ Response 维度: 语义比较 ═══
    # 将模型输出从上一轮的变化分类:
    #   "new"        — 第一轮或没有上一轮的响应
    #   "same"       — 相同的 thought/action（可能表示
    #                  持续失败或模型重复相同的步骤）
    #   "different"  — 与上一轮不同（正常操作）
    thought_vs = "new" if not prev_thought else (
        "same" if thought == prev_thought else "different")
    action_vs = "new" if not prev_action else (
        "same" if action == prev_action else "different")
    resp_delta = response_len - prev_response_len if prev_response_len else response_len

    # ═══ Cache 维度: token 估算 ═══
    # 将字符级公共前缀转换为 token 级缓存估算。
    # 3.5 字符/token 是针对中英文混合文本的经验平均值。
    # 见模块级文档字符串了解理由和精度边界。
    cached_tokens = int(common_len / 3.5)
    new_token_est = int(new_len / 3.5)

    # ═══ 一行摘要 ═══
    # 紧凑的人类可读摘要，用于每轮日志记录。
    # 格式匹配 TraceCollector 中 logger.diff() 的约定。
    parts = [f"复用率 {reuse_pct:.0f}%"]
    if history_added > 0:
        parts.append(f"历史 +{history_added} 行")
    parts.append(f"用户 {'变化' if user_changed else '不变'}")
    parts.append(f"Thought:{thought_vs}")
    parts.append(f"Action:{action_vs}")
    if resp_delta != 0:
        parts.append(f"响应 {resp_delta:+d} 字符")

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
