"""
Prompt 模板渲染器 — 从 YAML 协议组装 ReAct 提示词。

功能描述:
  处理 ReAct 协议的第 1 节：Agent → LLM 提示词构建。
  将系统提示词、用户消息、历史记录和触发词渲染为完整的提示词字符串，
  供 LLM 消费。可通过 YAML 协议文件或内置 Qwen2 默认值配置。

用法:
  >>> from agentic.protocol.template import PromptTemplate
  >>> tpl = PromptTemplate(yaml_path="agentic/protocol/ReActProtocol.yaml")  # YAML 驱动
  >>> tpl = PromptTemplate()                                 # 内置默认值
  >>> prompt = tpl.render_full_prompt(tool_names, question, history)
"""

from __future__ import annotations

import os
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, fields, MISSING

from agentic.observability import get_logger

logger = get_logger(__name__)


@dataclass
class PromptTemplate:
    """
    ReAct 提示词模板渲染器。

    功能描述:
      渲染系统提示词 + 用户消息 + 历史记录 + 触发词
      为完整的提示词字符串供 LLM 使用。
      可通过 YAML 协议文件或内置 Qwen2 默认值配置。

    =====================
    缓存性策略
    =====================

    提示词组装顺序旨在最大化跨轮次的 KV-cache 复用。
    因为系统提示词 XML 部分（role_text）每轮都相同，
    缓存友好的推理引擎可以复用上一轮前缀的已计算 KV 条目，
    仅需为用户消息和历史记录计算新 token。

    可缓存（跨轮次静态）:
      - role_text（所有 XML 标签部分）
      - format_rules（工具名变化，但规则字符串固定）
      - tools_header + tool_item_format（工具*描述*变化，
        但结构前缀可复用）

    每轮变化（不可缓存）:
      - user_template（问题变化）
      - history_entry_format（历史记录增长）
      - trigger（始终为 "Thought:" — 这是模型开始生成的提示，
        标记输入和输出的边界）

    参数:
      所有字段通过 dataclass 声明，可通过 YAML 文件覆盖。
      参见各字段文档获取详细信息。
    """

    # ── 分隔符 ───────────────────────────────────────────────────
    # Chat 模板标记，包裹每个角色的内容。这些是模型特定的：
    # 下面的默认值针对 Qwen2（ChatML 格式），但可以通过 YAML
    # 为其他模型覆盖（例如 Llama 3 使用
    # <|start_header_id|>...<|end_header_id|>）。
    system_start: str = "<|im_start|>system\n"
    system_end: str = "<|im_end|>"
    user_start: str = "<|im_start|>user\n"
    user_end: str = "<|im_end|>"
    assistant_start: str = "<|im_start|>assistant\n"
    # 注意: assistant_end 有意为空字符串 ("")。
    # 我们希望模型继续生成而不带结束分隔符。
    # 如果在此处放置 <|im_end|>，模型将在产生输出之前停止。
    assistant_end: str = ""

    # ── 系统提示词模板 ───────────────────────────────────────
    #
    # 重要说明 — 缓存性:
    # 下面的 role_text 是一个静态/可缓存的前缀 — 它永不随轮次变化。
    # XML 标签化的部分锚定模型注意力，实现跨对话的 KV-cache 复用。
    # 只有 format_rules（行内提醒）和 tool_items 每轮变化。
    #
    # 每个 XML 部分都有特定的设计理由:
    #
    #   <identity>
    #     预先确立 Agent 的角色身份。这锚定了模型的自我认知，
    #     早于任何指令，减少了后续注入内容试图重新定义 Agent 角色的
    #     角色劫持攻击。
    #
    #   <objectives>
    #     高层任务分解指南。在工具细节之前设定推理策略，
    #     使模型先考虑步骤，其次才是工具选择。
    #
    #   <first_turn_behavior>
    #     显式处理模糊问题的情况。没有这个，模型可能猜测
    #     而不是请求澄清，导致浪费轮次。
    #
    #   <tone_and_style>
    #     防止"社交填充"（如 "好问题！"），浪费 token 并增加延迟。
    #     "每步一行"约束保持输出可被正则解析器解析。
    #
    #   <tools>
    #     行内工具定义。这些是静态示例 — 实际的动态工具列表
    #     稍后通过 tools_header + tool_items 追加。
    #     静态示例作为后备，以防动态列表被截断或排序错误。
    #
    #   <workflow>
    #     核心 ReAct 循环规范。{tool_names} 占位符在渲染时替换。
    #     "每个 Thought 一行"约束对可解析性至关重要 — 多行 Thought
    #     会混淆基于正则的解析器。
    #
    #   <guardrails>
    #     显式安全约束。与指令分离，使其作为不可变规则突出显示，
    #     而非程序性建议。"CONFIRMATION REQUIRED" 头部模式
    #     被下游安全过滤器使用。
    #
    #   <output_format>
    #     解析器期望的精确格式。解析器的正则模式（P1-P4）
    #     针对此格式设计 — 修改此部分而不更新 parser.py
    #     将导致解析失败。
    #
    #   <error_handling>
    #     常见失败模式的恢复策略。没有显式的错误处理指导，
    #     模型倾向于重复重试相同的失败动作（持续性失败）。
    #
    #   <internal_logic>
    #     工具选择的决策优先级规则。"Cache-aware" 行提醒模型
    #     系统提示词始终可用（通过 KV-cache），因此无需重新读取。
    #
    role_text: str = (
        "请严格遵守以下格式。\n\n"
        "<identity>\n"
        "你是一个运行在本地边缘硬件上的 ReAct Agent。\n"
        "你的目标是通过逐步推理和调用工具来回答问题。\n"
        "你不得: 执行任意代码、访问网络或修改系统文件。\n"
        "</identity>\n\n"
        "<objectives>\n"
        "1. 理解用户的提问\n"
        "2. 将其分解为可解决的步骤\n"
        "3. 需要计算或外部数据时使用工具\n"
        "4. 提供清晰的最终答案，附带推理过程\n"
        "</objectives>\n\n"
        "<first_turn_behavior>\n"
        "如果问题明确: 立即以 \"Thought:\" 开始推理\n"
        "如果问题模糊: 在继续之前提出一个澄清性问题\n"
        "</first_turn_behavior>\n\n"
        "<tone_and_style>\n"
        "- 简洁 — 每步一个 Thought，每个 Action 一个步骤\n"
        "- 直接开始回答 — 不需要 \"好问题！\" 或奉承\n"
        "- 使用简洁的语言进行推理\n"
        "</tone_and_style>\n\n"
        "<tools>\n"
        "可用工具如下。根据任务选择正确的工具:\n"
        '- calculator: 数学表达式计算（如 "2+2", "10*5"）\n'
        '- get_weather: 当前天气查询（如 "London", "Tokyo"）\n'
        "仅在必要时使用工具。如果你已经知道答案，直接使用 Final Answer。\n"
        "</tools>\n\n"
        "<workflow>\n"
        "遵循以下序列:\n"
        "1. Thought: 在一行内写出你的推理\n"
        "2. Action: 仅写出工具名（{tool_names}）\n"
        "3. Action Input: 写出工具参数\n"
        "4. 等待 Observation 后再进行下一步\n"
        '5. 当答案准备好时: 输出 "Final Answer: 你的答案"\n'
        "</workflow>\n\n"
        "<guardrails>\n"
        "需要确认: 无（只读工具，安全的执行环境）\n"
        "禁止: 生成除数学表达式外的代码、访问文件、发出网络请求\n"
        "操作限制: 每个问题最多 8 步推理\n"
        "</guardrails>\n\n"
        "<output_format>\n"
        "你必须严格遵守以下格式。不要添加额外解释，不要使用 think 标签:\n\n"
        "Thought: [一行内的推理]\n"
        'Action: [工具名或 "final_answer"]\n'
        "Action Input: [工具输入，或最终答案文本]\n"
        "</output_format>\n\n"
        "<error_handling>\n"
        "- 如果工具返回错误: 阅读错误信息，调整方法，使用修正后的输入重试或使用不同的工具\n"
        "- 如果同一步骤尝试 3 次后仍无法解决: 说明原因并提供最佳的部分答案\n"
        "- 如果工具未知: 检查可用工具列表并使用有效的工具名重试\n"
        "</error_handling>\n\n"
        "<internal_logic>\n"
        "多个工具都可能工作时，优先级顺序:\n"
        "1. 如果需要数学计算 → calculator\n"
        "2. 如果需要天气/位置数据 → get_weather\n"
        "3. 如果无需工具即可得出答案 → 直接 Final Answer\n"
        "缓存感知: 系统提示词永不改变，因此格式规则始终可用。\n"
        "</internal_logic>"
    )
    format_rules: List[str] = field(default_factory=lambda: [
        '在 "Thought:" 之后写你一行内的推理',
        '在 "Action:" 之后仅写出工具名: {tool_names}',
        '在 "Action Input:" 之后写出工具输入',
        '得到答案后使用 "Final Answer:"',
    ])
    tools_header: str = "可用工具:"
    tool_item_format: str = "- {name}: {description}"

    # ── 用户消息模板 ────────────────────────────────────────
    # 简单的替换模板。问题是唯一可变的部分 —
    # 用 <user> 分隔符包裹由 render_full_prompt 方法的组装顺序处理。
    user_template: str = "{question}"

    # ── 历史记录条目格式 ─────────────────────────────────────────
    # 每个 ReAct 轮次产生一个历史记录条目。
    # 格式匹配 LLM 输出的内容（Thought/Action/Action Input/Observation），
    # 使得组装后的历史记录看起来像是对话的自然延续。
    history_entry_format: str = (
        "Thought: {thought}\n"
        "Action: {action}\n"
        "Action Input: {action_input}\n"
        "Observation: {observation}"
    )

    # ── 触发词 ──────────────────────────────────────────────────────
    # 触发词文本在历史记录之后追加。它提示模型开始生成下一个 Thought。
    # "Thought:" 是标准的 ReAct 触发词，因为它是每个 Agent 输出的第一个 token。
    trigger: str = "Thought:"

    # ── 停止序列 ───────────────────────────────────────────────
    # LLM 在生成这些 token 时应停止。
    # "Observation:" 标记工具输出开始的位置（我们外部生成 Observation，
    # 而非通过 LLM）。
    # <|im_end|> 是 Qwen2 的轮次结束标记。
    stop_sequences: List[str] = field(default_factory=lambda: [
        "Observation:", "<|im_end|>",
    ])

    # ── 生成参数 ────────────────────────────────────────────
    max_tokens: int = 512
    temperature: float = 0.2
    max_steps: int = 8

    # ═════════════════════════════════════════════════════════════════
    # XML 标签（用于文档/内省）
    # ═════════════════════════════════════════════════════════════════

    @property
    def xml_tags(self) -> List[str]:
        """
        返回 role_text 中使用的 XML 标签名列表。

        功能描述:
          这些标签作为提示词部分的规范清单。
          每个标签锚定一个特定的行为指令，标签名被下游指标代码
          用于测量每个部分的提示词复用率。

        返回:
            XML 标签名列表，按显示顺序排列（与 role_text 中的顺序相同）。
        """
        return [
            "identity",
            "objectives",
            "first_turn_behavior",
            "tone_and_style",
            "tools",
            "workflow",
            "guardrails",
            "output_format",
            "error_handling",
            "internal_logic",
        ]

    # ═════════════════════════════════════════════════════════════════
    # YAML 加载
    # ═════════════════════════════════════════════════════════════════

    def __init__(self, yaml_path: Optional[str] = None, **kwargs):
        """
        自定义 __init__ 方法 — 因为 @dataclass 会自动生成一个，
        但我们需要在字段初始化和 YAML 加载之间进行协调。

        为什么需要手动字段循环:
          dataclass 会自动生成带有每个字段位置参数的 __init__。
          当我们用自定义签名覆盖 __init__（yaml_path, **kwargs）时，
          dataclass 生成的 __init__ 被替换 — 因此我们必须从默认值/
          default_factory 手动初始化每个字段，然后应用 kwargs 覆盖，
          最后可选地加载 YAML。

        三层覆盖链:
          1. Dataclass 默认值（上面硬编码的 Qwen2 值）
          2. **kwargs（调用方的程序化覆盖）
          3. YAML 协议文件（磁盘上的配置，最高优先级）

        参数:
            yaml_path: agentic/protocol/ReActProtocol.yaml 的路径。
                如果为 None，仅使用默认值和 kwargs。
            **kwargs: 字段级别的覆盖，在 YAML 之前应用，
                因此 YAML 仍然可以覆盖它们。这使得调用方可以设置
                一个 YAML 可以替换的后备值。

        处理逻辑:
            1. 将所有 dataclass 字段初始化为其默认值
            2. 应用 kwargs 覆盖到 dataclass 字段
            3. 如果提供 yaml_path，加载 YAML 覆盖（最高优先级）
        """
        # 步骤 1: 将所有 dataclass 字段初始化为其默认值。
        # 这是必要的，因为自定义 __init__ 替换了自动生成的 __init__，
        # 否则字段将保持未设置状态。
        for f in fields(self.__class__):
            if f.default is not MISSING:
                # 简单默认值（字符串、整数等）
                setattr(self, f.name, f.default)
            elif f.default_factory is not MISSING:
                # 带有 default_factory 的字段（列表、字典等）需要
                # 调用工厂方法创建新实例 — 共享的可变默认值是
                # 经典的 Python 陷阱。
                setattr(self, f.name, f.default_factory())

        # 步骤 2: 将 kwargs 应用到 dataclass 字段。
        # 允许调用方如 PromptTemplate(temperature=0.1) 一样
        # 覆盖单个字段而不触碰 YAML。
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        # 步骤 3: 加载 YAML 覆盖（最高优先级）。
        if yaml_path:
            self._load_yaml(yaml_path)

    def _load_yaml(self, yaml_path: str) -> None:
        """
        从 YAML 协议文件加载提示词配置。

        功能描述:
          从 YAML 字典选择性地覆盖实例属性。
          仅修改 YAML 配置中存在的键；YAML 中不存在的字段保持当前值
          （无论是来自默认值还是 kwargs）。

        参数:
          yaml_path: YAML 协议文件的路径。

        预期的 YAML 结构（来自 ReActProtocol.yaml）:
          prompt:
            delimiters: {system_start, system_end, ...}
            sections:
              system: {role, format_rules, tools_header, tool_item_format}
              user: {template}
              history: {entry_format}
              trigger: "..."
          response:
            generation: {max_tokens, temperature, max_steps}
            stop_sequences: [...]

        优雅降级:
          - 如果 pyyaml 未安装: 记录警告并保持默认值。
          - 如果文件不存在: 记录警告并保持默认值。
          - 如果 YAML 中缺少某个键: 保持当前值。
        """
        try:
            import yaml
        except ImportError:
            logger.warning("pyyaml 未安装 — 使用内置默认值")
            return

        if not os.path.exists(yaml_path):
            logger.warning(f"YAML 协议文件不存在: {yaml_path}")
            return

        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        prompt_cfg = cfg.get("prompt", {})

        # 分隔符 — 模型特定的 chat 模板标记。
        # 这些与 sections 分开，因为它们是模型属性，而非内容属性。
        d = prompt_cfg.get("delimiters", {})
        if d:
            self.system_start = d.get("system_start", self.system_start)
            self.system_end = d.get("system_end", self.system_end)
            self.user_start = d.get("user_start", self.user_start)
            self.user_end = d.get("user_end", self.user_end)
            self.assistant_start = d.get("assistant_start", self.assistant_start)
            self.assistant_end = d.get("assistant_end", self.assistant_end)

        # Sections — 实际的提示词内容，按角色块划分。
        sections = prompt_cfg.get("sections", {})
        sys_sec = sections.get("system", {})
        if sys_sec:
            self.role_text = sys_sec.get("role", self.role_text)
            self.format_rules = sys_sec.get("format_rules", self.format_rules)
            self.tools_header = sys_sec.get("tools_header", self.tools_header)
            self.tool_item_format = sys_sec.get("tool_item_format", self.tool_item_format)

        user_sec = sections.get("user", {})
        if user_sec:
            self.user_template = user_sec.get("template", self.user_template)

        history_sec = sections.get("history", {})
        if history_sec:
            self.history_entry_format = history_sec.get("entry_format", self.history_entry_format)

        self.trigger = sections.get("trigger", self.trigger)

        # 响应配置 — 生成参数和停止序列。
        resp_cfg = cfg.get("response", {})
        gen_cfg = resp_cfg.get("generation", {})
        self.max_tokens = gen_cfg.get("max_tokens", self.max_tokens)
        self.temperature = gen_cfg.get("temperature", self.temperature)
        self.max_steps = gen_cfg.get("max_steps", self.max_steps)
        self.stop_sequences = resp_cfg.get("stop_sequences", self.stop_sequences)

        logger.debug("PromptTemplate 已从 YAML 加载")

    # ═════════════════════════════════════════════════════════════════
    # 渲染
    # ═════════════════════════════════════════════════════════════════

    def render_system_prompt(self, tool_names: List[str],
                             tool_descriptions: Dict[str, str]) -> str:
        """
        渲染系统提示词部分（固定前缀）。

        功能描述:
          XML 标签部分（role_text）跨轮次静态/可缓存 —
          只有 format_rules 和 tool_items 每次请求变化。

        组装顺序:
          1. 替换了 {tool_names} 的 role_text（可缓存的基座）
          2. 替换了 {tool_names} 的 format_rules（小幅变化）
          3. tools_header + 动态 tool_items（工具集变化时变化）

        参数:
            tool_names: 用于 {tool_names} 占位符的工具名列表。
            tool_descriptions: 将名称映射到描述的字典，用于渲染
                动态工具列表。

        返回:
            完整渲染的系统提示词字符串（不含分隔符）。
        """
        names_str = ', '.join(tool_names)
        tool_items = "\n".join(
            self.tool_item_format.format(name=name, description=desc)
            for name, desc in tool_descriptions.items()
        )
        rules = "\n".join(
            r.format(tool_names=names_str) for r in self.format_rules
        )
        # role_text 可能包含 {tool_names} 占位符（XML <workflow> 部分）
        formatted_role = self.role_text.format(tool_names=names_str)
        return (
            f"{formatted_role}\n"
            f"{rules}\n\n"
            f"{self.tools_header}\n"
            f"{tool_items}"
        )

    def render_user_message(self, question: str) -> str:
        """
        渲染用户消息部分。

        功能描述:
          使用 user_template 格式替换 {question} 占位符。

        参数:
            question: 用户的输入字符串。

        返回:
            用户消息内容（不含分隔符 — 分隔符由 render_full_prompt 添加）。
        """
        return self.user_template.format(question=question)

    def render_history_entry(self, thought: str, action: str,
                             action_input: str, observation: str) -> str:
        """
        渲染单个 ReAct 历史记录条目。

        功能描述:
          将 Thought、Action、Action Input 和 Observation 格式化为
          一个历史记录条目字符串。

        参数:
            thought: 模型的推理行。
            action: 调用的工具名（或 "final_answer"）。
            action_input: 传递给工具的参数。
            observation: 工具输出。

        返回:
            格式化后的历史记录条目字符串，准备追加到 assistant 部分。
        """
        return self.history_entry_format.format(
            thought=thought, action=action,
            action_input=action_input, observation=observation,
        )

    def render_full_prompt(self, tool_names: List[str],
                           tool_descriptions: Dict[str, str],
                           question: str, history: str) -> str:
        """
        渲染完整的 ReAct 提示词供 LLM 使用。

        功能描述:
          按照缓存友好的标准顺序组装完整提示词:
          <system> → <user> → <assistant + history> → Thought:

        组装顺序（规范的缓存友好顺序）:
          <system> → <user> → <assistant + history> → Thought:

        此顺序专为 KV-cache 复用设计:
          - <system> 块跨轮次相同（可缓存）。
          - <user> 块通常每轮变化（不可缓存）。
          - <assistant> + 历史记录单调增长（部分可缓存 —
            之前轮次的历史记录前缀可以复用）。
          - "Thought:" 是生成触发词（始终相同）。

        参数:
            tool_names: 用于提示词渲染的工具名列表。
            tool_descriptions: 工具名到描述的字典。
            question: 当前的用户问题。
            history: 累积的 ReAct 历史记录字符串（预先格式化）。

        返回:
            完整的提示词字符串，准备用于 LLM 推理。
        """
        system = self.render_system_prompt(tool_names, tool_descriptions)
        user_msg = self.render_user_message(question)

        return (
            f"{self.system_start}{system}{self.system_end}\n"
            f"{self.user_start}{user_msg}{self.user_end}\n"
            f"{self.assistant_start}{history}\n"
            f"{self.trigger}"
        )
