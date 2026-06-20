"""
AgentCore — 状态机构建的 ReAct 推理循环。

取代 170 行单体函数 run_react_agent()，采用基于类的状态机实现，便于独立测试。
每次迭代遵循严格的状态流转，对解析失败和工具未找到错误提供显式恢复机制。

状态（9 个，含恢复态）：
  IDLE → PROMPT_BUILD → LLM_CALL → PARSE → DISPATCH →
    FINAL_ANSWER | TOOL_EXEC → HISTORY_APPEND → NEXT_ITERATION
  PARSE 失败 → ERROR_RECOVERY → PROMPT_BUILD（重试）
  终止状态：    FINAL_ANSWER | MAX_STEPS

安全边界：
  Agent 将 LLM 输出视为不可信数据。工具执行的观测结果以文本字符串形式
  追加到提示历史中——不会反序列化、评估或作为代码执行。这可以防止通过
  精心构造的工具输出逃离 ReAct 循环的提示注入攻击。

用法：
  >>> from agentic.agent import AgentCore
  >>> agent = AgentCore(llm=backend, registry=tools, template=prompt_tpl, parser=parser)
  >>> result = agent.run("What is 2+2?")
  "4"
"""

from __future__ import annotations

import time
from typing import List, Dict, Optional, Protocol

from agentic.exceptions import ParseError, ToolNotFoundError, ToolExecutionError
from agentic.observability import get_logger

logger = get_logger(__name__)


# ==========================================================================
# AgentCore
# ==========================================================================

class AgentCore:
    """
    ReAct Agent 状态机。

    编排流程：prompt -> LLM -> 解析 -> 分发 -> 历史记录
    具备解析失败和工具缺失时的错误恢复能力。

    依赖通过注入方式提供——不依赖全局状态。这使得组件可以通过将真实 LLM
    后端替换为实现了 ``LLMBackend`` 协议的测试替身进行独立测试。

    安全性：所有 LLM 生成的文本均被视为不可信数据。解析器从原始文本中
    提取结构化字段，但不执行 ``eval()``、``exec()`` 或任何超出字符串匹配
    范围的输出解释操作。工具输出以纯文本字符串的形式追加到提示上下文
    中——永不反序列化或执行。

    用法：
        agent = AgentCore(llm=backend, registry=tools,
                          template=prompt_tpl, parser=parser)
        result = agent.run("What is 2+2?")
    """

    def __init__(
        self,
        llm,                    # LLMBackend（协议接口）
        registry,               # ToolRegistry
        template,               # PromptTemplate
        parser,                 # ResponseParser
        collector=None,         # TraceCollector（可选）
        formatter=None,         # OutputFormatter（可选）
        max_steps: int = 8,
    ):
        """
        初始化 AgentCore 实例，注入所有依赖组件。

        参数:
          llm: 实现了 ``LLMBackend`` 协议的对象（包含 ``generate()`` 方法和 ``model_name`` 属性）。
          registry: ``ToolRegistry`` 实例，通过 ``execute()`` 提供工具名称查找和执行功能。
          template: ``PromptTemplate`` 实例，用于渲染提示词和历史条目。定义 LLM 调用的停止序列和 token 限制。
          parser: ``ResponseParser`` 实例，从 LLM 原始输出中提取 ``thought``、``action`` 和 ``action_input``。
          collector: 可选 ``TraceCollector`` 实例，用于可观测性。设置后记录每一步的完整跟踪信息（token 数、耗时、错误）。
          formatter: 可选 ``OutputFormatter`` 实例，用于返回前对最终答案进行格式化。
          max_steps: 强制终止前的最大 ReAct 迭代次数。默认 8。防止因 LLM 不配合输出导致无限循环。
        """
        self._llm = llm
        self._registry = registry
        self._template = template
        self._parser = parser
        self._collector = collector
        self._formatter = formatter
        self.max_steps = max_steps

    # ── 公共 API ───────────────────────────────────────────────────

    def run(self, question: str) -> str:
        """
        执行完整的 ReAct 推理循环。

        该方法通过状态机迭代：构建提示词、LLM 生成、响应解析、动作分发
        （工具调用或最终答案）、历史追加和循环继续。解析失败时将错误记录
        到历史中并重试，而不是崩溃退出。

        参数:
          question: 用户自然语言问题字符串。

        处理逻辑:
          1. 组装工具描述、用户问题、历史记录构建提示词
          2. 调用 LLM 生成 Thought/Action/Action Input
          3. 正则解析 LLM 输出，提取结构化字段
          4. 根据 Action 类型分发：final_answer 终止 / 工具名调用执行
          5. 工具执行结果作为 Observation 追加到历史
          6. 循环直到 Final Answer 或达到 max_steps 上限

        错误恢复:
          - 解析失败 → 将原始输出作为 Observation 注入，提示模型重试
          - 工具不存在 → 返回可用工具列表作为 Observation
          - 工具执行异常 → 返回错误信息作为 Observation

        返回值:
          最终答案字符串，或 "Agent stopped due to max steps."

        异常:
          仅致命错误会向外抛出；可恢复错误在循环内处理，不会传播异常。
        """
        logger.info(f"{'═'*50}\n  Question: {question}")

        history: List[str] = []
        current_input = question
        tool_names = self._registry.list_names()
        tool_descs = {t.name: t.description for t in self._registry.list_tools()}

        for step in range(1, self.max_steps + 1):
            logger.info(f"{'─'*40}\n  Step {step}/{self.max_steps}")
            t_start = time.time()

            # ── 状态：PROMPT_BUILD ──────────────────────────────────
            # 渲染完整的 ReAct 提示词：工具描述、用户问题和累积的
            # 思考/动作/观测历史。模板控制停止序列和 token 限制，
            # 为 LLM 提供输出格式的显式指导。
            history_str = "\n".join(history) if history else ""
            prompt = self._template.render_full_prompt(
                tool_names, tool_descs, current_input, history_str,
            )

            # ── 状态：LLM_CALL ──────────────────────────────────────
            # 将提示词发送给 LLM 后端。后端负责分词、推理和停止序列检测。
            # response 包含 token 计数和实时时间统计。
            response = self._llm.generate(
                prompt=prompt,
                stop=self._template.stop_sequences,
                max_tokens=self._template.max_tokens,
                temperature=self._template.temperature,
            )

            logger.info(f"LLM: {response.prompt_tokens}p+{response.completion_tokens}c tokens, {response.duration_ms:.0f}ms")
            logger.debug(f"Raw: {response.text[:200]}")

            # ── 状态：PARSE ─────────────────────────────────────────
            # 从非结构化的 LLM 输出中提取结构化字段
            # （thought、action、action_input）。解析器使用正则表达式
            # 匹配 ReAct 格式（例如 "Thought: ...\nAction: ..."）。
            # 安全性：LLM 输出不可信。解析器仅进行字符串/正则匹配——
            # 不执行 eval() 或 exec()。提取的字段作为字符串传递给
            # 工具执行；各工具负责自身的输入清理。
            parse_error = None
            try:
                parsed = self._parser.parse(response.text, tool_names)
            except ParseError as e:
                logger.error(f"Parse failed: {e}")
                parse_error = str(e)

                # ── 错误恢复路径 ──────────────────────────────────
                # 当 LLM 偏离预期的 ReAct 输出格式时发生解析失败
                # （例如生成了自由形式的散文而非
                # "Thought: ...\nAction: ..."）。
                #
                # 恢复策略：
                #   1. 将原始输出记录为 observation，使 LLM 在下次
                #      迭代中看到自己的输出。
                #   2. 向历史追加格式化的 "skip" 条目，保持提示结构
                #      完整。
                #   3. 提供元指令（"Based on the observation, try again."）
                #      引导 LLM 回到 ReAct 格式。
                #   4. ``continue`` 到下一次迭代而非崩溃——使 agent
                #      对偶尔的格式错误输出具韧性。
                #
                # 安全说明：原始 LLM 输出原样追加到历史中。它在下次
                # 提示词中作为纯文本渲染——不反序列化，不评估。
                # 这防止了在 LLM 输出中嵌入控制序列的提示注入逃逸尝试。
                if self._collector:
                    self._collector.record_turn(
                        turn=step, question=question, prompt=prompt,
                        raw_output=response.text,
                        cleaned_output=self._parser.preprocess(response.text),
                        thought="(Parse Error)", action="none", action_input="none",
                        observation=response.text,
                        prompt_tokens=response.prompt_tokens,
                        completion_tokens=response.completion_tokens,
                        duration_ms=response.duration_ms,
                        parse_error=parse_error, status="parse_error",
                    )
                history.append(
                    f"Thought: (Parse Error)\n"
                    f"Action: none\n"
                    f"Action Input: none\n"
                    f"Observation: {response.text}"
                )
                current_input = (
                    f"Based on the observation, try again.\n"
                    f"Observation: {response.text}"
                )
                continue

            thought = parsed["thought"]
            action = parsed["action"]
            action_input = parsed["action_input"]

            logger.info(f"  Thought: {thought}")
            logger.info(f"  Action: {action}({action_input})")
            logger.trace(f"Input: {action_input}")

            # ── 状态：DISPATCH -> FINAL_ANSWER ─────────────────────
            # LLM 通过发出 "final_answer" 作为 action 来信号完成。
            # action_input 成为最终输出。这是除 MAX_STEPS 之外的
            # 唯一终止状态。
            if action == "final_answer":
                logger.info(f"  Final Answer: {action_input}")
                if self._collector:
                    self._collector.record_turn(
                        turn=step, question=question, prompt=prompt,
                        raw_output=response.text,
                        cleaned_output=self._parser.preprocess(response.text),
                        thought=thought, action=action, action_input=action_input,
                        observation="(final answer)",
                        prompt_tokens=response.prompt_tokens,
                        completion_tokens=response.completion_tokens,
                        duration_ms=response.duration_ms, status="success",
                    )
                return action_input

            # ── 状态：TOOL_EXEC ─────────────────────────────────────
            # 将动作分发到工具注册表。注册表按名称查找工具并调用
            # 其 execute() 方法。
            #
            # 两种可恢复的故障模式：
            #   1. ToolNotFoundError——LLM 幻觉出一个不存在的工具名。
            #      Observation 列出可用工具，以便 LLM 自我修正。
            #   2. ToolExecutionError——工具本身执行失败
            #      （例如网络超时、无效输入）。透出原始错误消息
            #      （非异常堆栈），避免内部信息泄露到提示上下文中。
            #
            # 安全性：action_input 字符串由用户提供（经 LLM 输出）
            # 并原样传递给工具。每个工具负责自身的输入验证和清理。
            # 沙箱层（agentic.tools.sandbox）为代码执行工具提供
            # 操作系统级隔离。
            tool_error = False
            try:
                observation = self._registry.execute(action, action_input)
            except ToolNotFoundError:
                observation = (
                    f"Unknown action: '{action}'. "
                    f"Available: {tool_names}"
                )
                tool_error = True
            except ToolExecutionError as e:
                observation = f"Tool execution error: {e.original_error}"
                tool_error = True

            logger.info(f"  Observation: {observation}")

            # ── 记录跟踪（如果配置了 collector）───────────────────
            if self._collector:
                self._collector.record_turn(
                    turn=step, question=question, prompt=prompt,
                    raw_output=response.text,
                    cleaned_output=self._parser.preprocess(response.text),
                    thought=thought, action=action, action_input=action_input,
                    observation=observation,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    duration_ms=response.duration_ms,
                    status="tool_error" if tool_error else "success",
                )

            # ── 状态：HISTORY_APPEND ───────────────────────────────
            # 将完成的轮次（思考/动作/结果）添加到运行历史中。
            # 模板控制每个条目的格式化方式，影响 LLM 在下一次迭代
            # 中看到的内容。
            #
            # 注入防御：Observation 作为字符串值追加到结构化模板中。
            # 模板引擎不会将其解释为模板指令——仅进行纯文本替换。
            # 这防止包含 "Thought: ..." 的观测值突破 ReAct 格式
            # 并注入伪造上下文。
            history.append(self._template.render_history_entry(
                thought=thought, action=action,
                action_input=action_input, observation=observation,
            ))

            # ── 状态：NEXT_ITERATION ────────────────────────────────
            # 为下一次循环迭代准备用户消息。本次迭代的观察结果
            # 成为主要输入，引导 LLM 推理下一步操作。
            current_input = (
                f"Based on the observation, what should I do next?\n"
                f"Observation: {observation}"
            )

        # ── 终止状态：MAX_STEPS ───────────────────────────────────
        # Agent 耗尽了步骤预算而未达到 final_answer。
        # 这防止 LLM 无限循环地持续调用工具。
        # 调用方可以通过哨兵字符串区分真实答案与超时终止。
        logger.warn(f"Max steps ({self.max_steps}) reached. Loop terminated.")
        return "Agent stopped due to max steps."
