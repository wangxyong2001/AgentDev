"""ReAct Agent 循环的 Observation 消毒器。

在工具输出进入 ReAct 历史 / LLM 上下文之前对其进行消毒，
阻断嵌入在工具/API 响应中的提示注入攻击。

纵深防御原则:
  1. 先检测可疑内容（日志记录 + 严格模式哨兵）
  2. 剥离聊天模板令牌（角色边界注入）
  3. 剥离 ANSI/控制序列（隐藏载荷）
  4. 阻断 ReAct 格式令牌（Agent 步骤欺骗）
  5. 阻断命令执行模式（Shell 注入）
  6. 截断至长度上限（令牌预算保护）
  7. 在输出前添加 [sanitized] 和审计标志（透明性）

本模块是工具输出进入 LLM 上下文窗口之前的**最后一道防线**。
它不能替代工具层本身的输入验证。
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 聊天模板 / 特殊令牌
# ---------------------------------------------------------------------------
# 这些令牌被聊天格式模型（Qwen、Llama 等）用于分隔对话角色。
# 如果它们出现在工具输出中，攻击者可能试图注入虚假的系统提示或
# 用户/助手边界以覆盖 Agent 的指令。剥离它们可防止角色边界注入。
_CHAT_TEMPLATE_TOKENS = {
    "<|im_start|>", "<|im_end|>",        # Qwen / ChatML format
    "<|begin_of_text|>", "<|eot_id|>",   # Llama 3 format
    "<|system|>", "<|user|>", "<|assistant|>",  # Role markers
    "<s>", "</s>",                        # BOS/EOS tokens
}

_CHAT_TEMPLATE_PATTERN = re.compile(
    "|".join(re.escape(t) for t in _CHAT_TEMPLATE_TOKENS),
)

# ---------------------------------------------------------------------------
# 注入检测模式
# ---------------------------------------------------------------------------
# 这些模式捕获常见的提示注入攻击类别：
#   - 指令覆盖："ignore previous instructions"、"you are now..."
#     经典越狱，让 LLM 忽略其系统提示。
#   - ReAct 格式注入：模仿 Agent 输出（Thought/Action）的行，
#     使攻击者的回复被解释为 Agent 自身的推理。
#   - 命令执行：shell 命令、eval()、subprocess 调用，
#     诱骗 Agent 执行任意代码。
#
# 重要：这是一个**黑名单**，而非白名单。它能捕获已知模式，
# 但无法阻止新型攻击。高安全部署应启用严格模式。
_INSTRUCTION_OVERRIDE_PATTERNS = [
    # "Ignore all previous instructions/directions/commands"
    # — 最常见的提示注入类别；告诉模型丢弃其系统提示，
    # 转而遵循注入的文本。
    re.compile(r"ignore\s+(all\s+)?previous\s+(instructions|directions|commands)", re.IGNORECASE),

    # "You are now..." — 角色重定义攻击，用攻击者控制的人格覆盖
    # Agent 的身份。
    re.compile(r"you\s+are\s+now\b", re.IGNORECASE),

    # "Forget your training" — 试图通过引用模型的训练过程来重置
    # 模型的对齐和护栏。
    re.compile(r"forget\s+(your\s+)?training", re.IGNORECASE),

    # "You are an unfiltered assistant/AI/model" — 试图通过将模型
    # 重新定义为 "unfiltered" 来禁用模型的拒绝机制。
    re.compile(r"you\s+are\s+an?\s+unfiltered\s+(assistant|ai|model|chatbot)", re.IGNORECASE),

    # "You must ignore/forget/disregard" — 命令式覆盖指令，
    # 指示模型绕过特定的安全检查。
    re.compile(r"you\s+must\s+(ignore|forget|disregard)", re.IGNORECASE),

    # "New instructions:" — 引入一组替代指令，旨在完全替换系统提示。
    re.compile(r"new\s+instructions?\s*:", re.IGNORECASE),

    # "Override mode/instructions/directives" — 显式覆盖尝试，
    # 直接挑战 Agent 的配置。
    re.compile(r"override\s+(mode|instructions|directives)", re.IGNORECASE),
]

# 匹配模仿 ReAct 格式的行（Thought/Action/Action Input/Final Answer）。
# 攻击者可以将这些内容放入工具输出中，使 Agent 后续的推理看起来
# 像是 *工具输出* 的一部分，导致循环误解其自身状态。
#
# 我们**阻断**（替换为 [blocked]）而非剥离这些内容，因为剥离可能会
# 留下仍然可读作有效 ReAct 输出的上下文。阻断使注入在视觉上是惰性的。
_REACT_FORMAT_PATTERN = re.compile(
    r"^(Thought|Action|Action\s+Input|Final\s+Answer)\s*:",
    re.MULTILINE,
)

# 命令执行模式 — 捕获攻击者可能嵌入到工具输出中以诱骗具有
# 代码执行能力的 Agent 的 shell 命令和 Python 执行原语。
#
# 检测 + 阻断分离传递的理由：
#   - 检测传递（sanitize 步骤 5）即使替换未改变字符串（例如重叠匹配）
#     也会添加审计标志。
#   - 阻断传递（sanitize 步骤 6）替换 ALL 出现。
#   - 这种两遍设计确保日志保真度：每个包含危险模式的 observation
#     只记录一次，而不是每个匹配记录一次。
_COMMAND_EXECUTION_PATTERNS = [
    # "Execute" — 通用执行命令，常用于触发工具调用或 shell 命令。
    re.compile(r"\bExecute\b", re.IGNORECASE),

    # "run command" — 显式命令执行尝试。
    re.compile(r"\brun\s+command\b", re.IGNORECASE),

    # "rm -rf" — 破坏性文件系统操作（Unix）。
    re.compile(r"\brm\s+-[rf]\b"),

    # curl / wget — 通过 HTTP 进行网络数据外泄。
    re.compile(r"\bcurl\b"),
    re.compile(r"\bwget\b"),

    # sudo — 权限提升。
    re.compile(r"\bsudo\b"),

    # exec / subprocess / os.system — Python 进程执行。
    re.compile(r"\bexec\b"),
    re.compile(r"\bsubprocess\b"),
    re.compile(r"\bos\.system\b"),

    # __import__ / eval( — Python 动态代码执行。
    re.compile(r"\b__import__\b"),
    re.compile(r"\beval\s*\("),
]

# ---------------------------------------------------------------------------
# ANSI 转义码 / 控制字符
# ---------------------------------------------------------------------------
# ANSI 转义序列可用于在终端输出中隐藏文本
# （例如将前景色设为背景色）。控制字符
# (\x00-\x1f, \x7f) 包括响铃、退格和其他终端控制码，
# 可能被用于混淆解析器或在日志文件中隐藏内容。
#
# 这些被**剥离**（完全移除），因为它们在 Agent 的推理上下文中
# 没有合法的语义价值——它们是纯粹的渲染产物，可能携带隐藏载荷。
_ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class ObservationSanitizer:
    """在工具输出进入 ReAct 历史 / LLM 上下文之前对其进行消毒。

    应用纵深防御流水线：
        1. 检测并记录注入尝试
        2. 剥离聊天模板令牌
        3. 剥离 ANSI 转义序列和控制字符
        4. 截断至 *max_chars*
        5. 若有任何修改，添加 ``[sanitized]`` 前缀

    设计原则 — 剥离 vs. 阻断：
      - 剥离（完全移除）：聊天令牌、ANSI 码、控制字符。
        这些在推理上下文中没有语义价值，移除不会改变 observation 的含义。
      - 阻断（替换为 [blocked]）：ReAct 格式令牌、命令模式。
        这些带有语义含义——将其替换为可见哨兵可保留"有内容被移除"的事实，
        同时使其丧失活性。
    """

    def __init__(
        self,
        max_chars: int = 2000,
        log_violations: bool = True,
        strict_mode: bool = False,
    ) -> None:
        """
        初始化 Observation 消毒器。

        参数:
            max_chars: 返回的 observation 最大长度。防止令牌预算耗尽攻击，
                即攻击者发送超长载荷以消耗模型的上下文窗口。
            log_violations: 检测到注入时发出 ``logging.warning`` 日志。
                日志记录包含原始文本的 500 字符预览，用于取证审计追踪。
                仅在高吞吐量环境中禁用（此时日志量可能成为问题）。
            strict_mode: 当为 *True* 时，检测到注入时返回哨兵字符串
                而非消毒后的 observation。这是最安全的选择：LLM 永远不会
                看到潜在的恶意内容。代价是包含误报匹配的合法工具输出也会被丢弃。
        """
        self.max_chars = max_chars
        self.log_violations = log_violations
        self.strict_mode = strict_mode

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def sanitize(self, observation: str) -> str:
        """清理工具输出以防止提示注入。

        返回值可安全地包含在 ReAct 历史中。

        处理逻辑:
          1. 对原始文本进行注入检测
          2. 剥离聊天模板令牌
          3. 剥离 ANSI 转义序列和控制字符
          4. 阻断 ReAct 格式注入令牌
          5. 检测并阻断危险命令模式
          6. 截断至长度上限
          7. 若有修改，添加 [sanitized] 前缀和审计标志

        返回值:
            安全字符串，可安全用于 ReAct 历史。
        """
        if not observation:
            return observation

        original = observation
        flags: list[str] = []

        # 1. 判断 observation 是否*看起来*有恶意
        #    此检查在原始文本上运行，在剥离之前进行，
        #    因此攻击者无法通过混合单独看起来良性但组合后恶意的
        #    载荷来逃避检测。
        is_suspicious = self._detect_injection(observation)
        if is_suspicious:
            self._log_violation(observation, "injection pattern detected")
            if self.strict_mode:
                return "[sanitized] Observation blocked by strict mode"

        # 2. 剥离聊天模板令牌
        #    诸如 <|im_start|>system<|im_end|> 的特殊令牌用于分隔
        #    对话角色。如果攻击者将这些嵌入工具输出，LLM 可能将它们
        #    解释为实际的角色边界，从而注入虚假的系统提示或用户消息。
        #    剥离它们是安全的，因为在 observation 上下文中它们没有语义含义。
        cleaned, count = _CHAT_TEMPLATE_PATTERN.subn("", observation)
        if count:
            flags.append(f"stripped {count} special token(s)")

        # 3. 剥离 ANSI 转义序列和控制字符
        #    ANSI 码可以隐藏文本（例如零宽序列）或引发基于终端的渲染攻击。
        #    控制字符可能混淆文本解析器或在下游触发意外行为。
        #    两者都被完全移除。
        cleaned = _ANSI_ESCAPE_PATTERN.sub("", cleaned)
        cleaned = _CONTROL_CHAR_PATTERN.sub("", cleaned)

        # 4. 剥离 ReAct 格式注入（看起来像 Agent 步骤的行）
        #    如果工具输出包含 "Thought:" 或 "Action:" 行，模型在组装历史时
        #    可能会将它们解释为自身的输出。我们**阻断**这些内容（替换为
        #    [blocked]）而非剥离，因为可见的哨兵清楚地告诉模型那些令牌已被移除。
        if _REACT_FORMAT_PATTERN.search(cleaned):
            cleaned = _REACT_FORMAT_PATTERN.sub("[blocked]", cleaned)
            flags.append("blocked ReAct format tokens")

        # 5. 检测危险的命令模式
        #    单独的检测传递确保即使多个命令模式匹配，我们也只记录一次。
        #    这保持了审计日志的可读性。
        for pattern in _COMMAND_EXECUTION_PATTERNS:
            if pattern.search(cleaned):
                flags.append("blocked command pattern(s)")
                break

        # 6. 剥离危险的命令模式（将匹配项替换为 [blocked]）
        #    阻断（而非剥离）保留了文本的结构完整性，同时使危险项丧失活性。
        #    [blocked] 哨兵也作为提示中可见的审计标记。
        for pattern in _COMMAND_EXECUTION_PATTERNS:
            cleaned = pattern.sub("[blocked]", cleaned)

        # 7. 截断
        #    强制执行 max_chars 预算。这防止令牌桶耗尽攻击，
        #    即攻击者发送超长载荷以填充模型的上下文窗口，
        #    从而挤出合法内容（系统提示、历史记录等）。
        if len(cleaned) > self.max_chars:
            cleaned = cleaned[: self.max_chars]
            flags.append(f"truncated to {self.max_chars} chars")

        # 8. 如果有任何修改，添加前缀
        #    前缀使消毒完全透明。模型可以清楚地看到什么被移除/阻断
        #    以及原因，这防止了 observation 部分缺失时的混淆。
        if flags:
            prefix = "[sanitized] " + "; ".join(flags) + ": "
            cleaned = prefix + cleaned

        return cleaned

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _strip_control_sequences(self, text: str) -> str:
        """移除指令覆盖、角色标记和格式指令。

        处理逻辑:
          将三重剥离传递（聊天令牌、ANSI、控制字符）合并为一次调用。
          供需要"快速清理"而无需完整注入检测流水线的调用方使用。

        参数:
          text: 要清理的输入文本

        返回值:
          清理后的文本字符串
        """
        result = _CHAT_TEMPLATE_PATTERN.sub("", text)
        result = _ANSI_ESCAPE_PATTERN.sub("", result)
        result = _CONTROL_CHAR_PATTERN.sub("", result)
        return result

    def _truncate(self, text: str, max_chars: Optional[int] = None) -> str:
        """截断过长的工具输出。

        参数:
            text: 要截断的输入文本。
            max_chars: 最大字符长度。未指定时使用实例默认值。

        返回值:
            截断后的字符串（此方法不添加前缀或审计标志；
            由调用方处理）。
        """
        limit = max_chars if max_chars is not None else self.max_chars
        return text[:limit]

    def _detect_injection(self, text: str) -> bool:
        """检测常见的注入模式。如果可疑则返回 True。

        检查三个类别:
          1. 指令覆盖模式（忽略之前的指令、角色重定义等）
          2. 聊天模板令牌（角色边界注入）
          3. ReAct 格式令牌（Agent 输出欺骗）

        注意：此处**不检查**命令执行模式——它们由阻断传递处理
        （sanitize() 中的步骤 5-6）。这是有意为之：命令模式可能出现在
        合法的工具输出中（例如返回 curl 使用文档的工具），
        我们希望阻断这些术语而非拒绝整个 observation。
        """
        for pattern in _INSTRUCTION_OVERRIDE_PATTERNS:
            if pattern.search(text):
                return True
        # Check for chat-template tokens
        if _CHAT_TEMPLATE_PATTERN.search(text):
            return True
        # Check for ReAct format injection
        if _REACT_FORMAT_PATTERN.search(text):
            return True
        return False

    def _log_violation(self, original: str, reason: str) -> None:
        """记录注入尝试以供审计。

        格式: ``ObservationSanitizer violation | reason=... | preview=...``

        500 字符预览的理由:
          - 足够短，避免在高流量部署中日志泛滥
          - 足够长，可捕获用于取证分析的注入载荷
          - ``%.500r`` 格式确保预览被安全地 repr 转义，
            因此原始数据中的二进制数据或控制字符不会破坏日志输出

        审计轨迹说明:
          此行日志是原始（消毒前）observation 的唯一记录。
          如果启用了 strict_mode，LLM 永远不会看到该内容——
          日志是唯一的痕迹。请确保根据您的安全事件保留策略保留日志。

        参数:
            original: 原始的（消毒前）observation 文本
            reason: 注入检测的原因描述
        """
        if not self.log_violations:
            return
        # Only log the first 500 chars of the original to avoid log flooding
        preview = original[:500]
        logger.warning(
            "ObservationSanitizer violation | reason=%s | preview=%.500r",
            reason,
            preview,
        )
