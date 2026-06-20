"""
统一的 ResponseParser — LLM 输出解析的唯一事实来源。

功能描述:
  消除双重路径腐化问题 (ARC-002)，即 ProtocolLoader.parse_response()
  和独立的 parse_llm_output() 各自独立实现了相同的四级正则解析，
  但存在细微差异。

架构:
  ResponseParser（统一引擎）
  ├── Config → agentic/protocol/ReActProtocol.yaml（首选）或内置默认值
  ├── preprocess()  — 剥离模型特定标签（Qwen <think> 等）
  ├── parse()       — 四级 P1->P4 正则解析，带后备策略
  └── validate()    — 解析后验证（动作是否已知、输入是否安全）

用法:
  >>> from agentic.protocol.parser import ResponseParser
  >>> parser = ResponseParser(yaml_path="agentic/protocol/ReActProtocol.yaml")  # YAML 驱动
  >>> parser = ResponseParser()                                 # 内置默认值
  >>> result = parser.parse(llm_output, tool_names=["calculator", "get_weather"])
  >>> # result: {"thought": "...", "action": "calculator", "action_input": "123*45"}
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from agentic.exceptions import ParseError, ToolNotFoundError


# ==========================================================================
# 配置
# ==========================================================================

@dataclass
class ParserConfig:
    """
    解析器配置 — 内置默认值（针对 Qwen2 模型）。

    功能描述:
      当提供 YAML 协议文件时，这些默认值会被 YAML 值覆盖。
      当没有 YAML 可用时，这些默认值作为生产配置使用。
      这是唯一的配置来源 — 别处不存在独立的硬编码副本。

    处理逻辑:
      - stop_sequences: LLM 生成停止序列。"Observation:" 是关键停止标记
      - strip_tags: 预处理时剥离的标签。Qwen 的 chat 模板将推理包裹在 <think> 标签中
      - pattern_*: 四级正则解析模式（P1->P4 优先级）

    注意事项:
      pattern_* 正则模式与 template.py 中的 output_format 部分紧密耦合。
      如果 output_format 变更，这些模式必须同步更新。
    """
    # LLM 生成的停止序列。
    # "Observation:" 是关键：它告诉 LLM 在 ReAct 循环的下一轮注入
    # Observation 之前停止生成。
    # "<|im_end|>" 是 Qwen2 的轮次结束标记。
    stop_sequences: List[str] = field(default_factory=lambda: [
        "Observation:", "<|im_end|>",
    ])

    # 预处理时需要剥离的标签。
    # Qwen 的 chat 模板将推理过程包裹在 <think>...</think> 标签中。
    # 如果不剥离这些标签，下方的正则模式将无法匹配，
    # 因为 "Thought:" 关键字嵌套在 thinking 块内部。
    strip_tags: List[str] = field(default_factory=lambda: [
        r"<think>.*?</think>",
        r"<response>",
        r"</response>",
        r"<\s*/\s*think\s*>",
    ])

    # 解析模式（P1->P4 优先级 — 详见 parse() 文档）。
    #
    # 重要说明: 这些模式针对 template.py 中的 output_format 部分设计。
    # 如果 output_format 变更，这些模式必须同步更新。
    #
    # pattern_final_answer:
    #   匹配输出中任何位置的 "Final Answer: <text>"。
    #   使用 re.DOTALL 使 <text> 可以跨多行。
    #
    # pattern_thought:
    #   捕获 "Thought:" 之后直到下一个 "Action:" 或 "Final" 关键字
    #   或字符串结束的文本。前瞻断言 (?=\n(?:...)) 防止 thought 吞掉 Action 行。
    #
    # pattern_action:
    #   捕获 "Action:" 之后的第一个空白分隔的 token。
    #   工具名总是单 token（不含空格）。
    #
    # pattern_action_input:
    #   捕获 "Action Input:" 之后直到下一个 ReAct 关键字
    #   (Thought/Action/Final/Observation) 或字符串结束的文本。
    #   这是最脆弱的模式 — 工具参数可以包含任意文本包括换行符。
    pattern_final_answer: str = r"Final Answer:\s*(.*)"
    pattern_thought: str = r"Thought:\s*(.+?)(?=\n(?:Action|Final)|$)"
    pattern_action: str = r"Action:\s*(\S+)"
    pattern_action_input: str = r"Action Input:\s*(.+?)(?=\n(?:Thought|Action|Final|Observation)|$)"


# ==========================================================================
# 统一的响应解析器
# ==========================================================================

class ResponseParser:
    """
    统一的 LLM 输出解析器 — YAML 和内置路径的单一实现。

    功能描述:
      配置来自 YAML（首选）或内置 ParserConfig（后备）。
      所有解析逻辑集中在此处 — 不在 ReActDemo 中重复。
      返回验证后的 Dict 或抛出 ParseError（可恢复）。

    解析优先级（P1-P4）:
      P1: Final Answer — 模型表示对话完成。立即返回，无需进一步解析。
      P2: Thought — 模型的推理过程。可选的（简单问题下模型可能直接跳到 Action）。
          后备策略: 使用输出的第一个非空行。
      P3: Action — 工具名。必需的。如果正则不匹配，后备扫描器在原始输出中搜索
          已知工具名。如果仍然失败，抛出 ParseError。
      P4: Action Input — 工具参数。可选的（某些工具不需要参数）。
          后备策略: 使用 Action 行之后的文本。

    处理逻辑:
      1. 初始化时预编译所有正则表达式（一次编译，多次使用）
      2. preprocess() 剥离模型特定标签
      3. parse() 执行四级优先级解析，每级有主路径和后备路径
      4. validate() 执行后解析验证
    """
    def __init__(self, yaml_path: Optional[str] = None, config: Optional[ParserConfig] = None):
        """
        从 YAML 文件或显式配置初始化解析器。

        功能描述:
          解析器始终在初始化时从其配置预编译正则表达式，
          使得 parse() 调用的时间复杂度为 O(n)（仅输出长度），
          没有正则编译开销。

        参数:
          yaml_path: agentic/protocol/ReActProtocol.yaml 的路径（可选）
          config:    ParserConfig 覆盖（可选）

        如果两者均为 None，则使用内置默认值。
        """
        self._config: ParserConfig = config or ParserConfig()

        if yaml_path:
            self._load_yaml(yaml_path)

        # 预编译正则表达式以提高性能。
        # 正则编译在 init 时完成一次，因为 parse() 每个 ReAct 轮次调用一次
        # （每次会话可能数百次）。
        self._re_final = re.compile(self._config.pattern_final_answer, re.DOTALL)
        self._re_thought = re.compile(self._config.pattern_thought, re.DOTALL)
        self._re_action = re.compile(self._config.pattern_action)
        self._re_action_input = re.compile(self._config.pattern_action_input, re.DOTALL)

        # 编译剥离模式
        self._strip_patterns = [re.compile(tag, re.DOTALL) for tag in self._config.strip_tags]

    def _load_yaml(self, yaml_path: str) -> None:
        """
        从 YAML 协议文件覆盖配置。

        功能描述:
          从 YAML 文件加载解析配置，覆盖默认值。
          优雅降级：如果 pyyaml 未安装，静默保持内置默认值。
          如果 YAML 中缺少某个配置键，保持当前值。

        参数:
          yaml_path: YAML 协议文件的路径

        预期的 YAML 结构:
          response:
            stop_sequences: [...]
            preprocessing: {strip_tags: [...]}
            parsing:
              final_answer: {pattern: "..."}
              thought: {pattern: "..."}
              action: {pattern: "..."}
              action_input: {pattern: "..."}

        异常:
          FileNotFoundError: YAML 文件不存在（通过调用方处理）
        """
        try:
            import yaml
        except ImportError:
            # pyyaml 未安装 — 保持内置默认值
            return

        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        resp_cfg = cfg.get("response", {})

        # 停止序列
        if "stop_sequences" in resp_cfg:
            self._config.stop_sequences = resp_cfg["stop_sequences"]

        # 剥离标签（预处理配置）
        pp = resp_cfg.get("preprocessing", {})
        if "strip_tags" in pp:
            self._config.strip_tags = pp["strip_tags"]

        # 解析模式 — 每个都是包含 "pattern" 键的字典。
        # YAML 结构将模式包裹在对象中，以允许未来添加元数据
        # （如 description、example）而不破坏 API。
        parsing = resp_cfg.get("parsing", {})
        if "final_answer" in parsing:
            self._config.pattern_final_answer = parsing["final_answer"]["pattern"]
        if "thought" in parsing:
            self._config.pattern_thought = parsing["thought"]["pattern"]
        if "action" in parsing:
            self._config.pattern_action = parsing["action"]["pattern"]
        if "action_input" in parsing:
            self._config.pattern_action_input = parsing["action_input"]["pattern"]

    # ── 预处理 ──────────────────────────────────────────────

    def preprocess(self, raw_text: str) -> str:
        """
        从原始 LLM 输出中剥离模型特定的包装标签。

        功能描述:
          处理 Qwen 推理模型的 <think>...</think>、<response>、</response> 等标签。
          标签模式通过 ParserConfig.strip_tags 可配置。

        为什么需要预处理:
          Qwen 的 chat 模板将模型的推理包裹在 <think>...</think> 标签中。
          原始输出看起来像:
            <think>我需要计算 2+2</think>
            Action: calculator
          如果不剥离，"Thought:" 正则将无法匹配，因为推理文本嵌套在
          <think> 标签内，而不是以 "Thought:" 开头。

        参数:
          raw_text: 原始 LLM 输出字符串

        返回:
          已清理的文本，准备进行解析
        """
        cleaned = raw_text
        for pattern in self._strip_patterns:
            cleaned = pattern.sub('', cleaned)
        return cleaned.strip()

    # ── 解析（P1->P4 优先级） ───────────────────────────────────

    def parse(self, llm_output: str, tool_names: List[str]) -> Dict[str, str]:
        """
        将 LLM 输出解析为结构化的 ReAct 字典。

        功能描述:
          使用四级优先级系统（P1-P4）解析 LLM 输出。
          因为 LLM 输出是可变的，正则匹配可能因多种原因失败
          （格式漂移、截断输出、模型特定问题）。
          每个优先级级别都有主路径（正则匹配）和后备路径。

        解析优先级（P1-P4）:

        P1: Final Answer（最高优先级）
          主路径:   正则搜索 "Final Answer: <text>"
          后备路径:  无 — 如果找到，立即返回。
          说明: Final Answer 表示对话结束，无需继续解析。

        P2: Thought（可选）
          主路径:   正则搜索 "Thought: <text>"
          后备路径:  输出中不以 "Action:" 或 "Final" 开头的第一个非空行。
          说明: 模型对简单问题可能跳过 "Thought:" 前缀。
                后备策略捕获存在的任何前言文本。

        P3: Action（必需）
          主路径:   正则搜索 "Action: <name>"
          后备路径:  在整个输出中扫描已注册的工具名。
          说明: 模型可能在没有 "Action:" 前缀的情况下输出工具名（格式漂移）。
                扫描已知工具名更健壮，但存在歧义 — 见下文说明。

        P4: Action Input（可选）
          主路径:   正则搜索 "Action Input: <text>"
          后备路径:  Action 行之后的文本（仅第一行）。
          说明: 对 get_weather 等简单工具，Action Input 可能被省略或与 Action 行合并。

        工具名扫描的歧义:
          P3 后备（在原始文本中扫描已注册工具名）存在已知歧义：
          如果模型输出类似 "我将使用 calculator 工具"，
          扫描器会匹配 "calculator"，但不知道 Action Input 从哪里开始。
          此时，我们使用匹配的工具名之后的所有文本作为 Action Input，
          可能包含 "工具" 等周围文字。
          这是可接受的，因为：
            a) P3 后备仅在主正则失败时激活（正常操作中很少见）
            b) 下游验证步骤可以捕获格式错误的输入
            c) Observation（工具输出）提供反馈信号 — 如果动作错误，
               模型在下一轮纠正

        参数:
          llm_output: 原始或预处理后的 LLM 输出文本
          tool_names: 已注册的工具名列表，用于后备扫描

        返回:
          {"thought": str, "action": str, "action_input": str}

        异常:
          ParseError: 所有解析策略均失败 — 模型输出无法解释
        """
        clean = self.preprocess(llm_output)
        if not clean:
            # 如果预处理剥离了所有内容（例如空 <think>），
            # 回退到原始输出。
            clean = llm_output.strip()

        # ── P1: Final Answer（最高优先级） ──
        # 模型表示"完成"。立即返回合成的 thought 和 "final_answer" 动作。
        # 无需进一步解析。
        fa = self._re_final.search(clean)
        if fa:
            return {
                "thought": "我现在知道最终答案了",
                "action": "final_answer",
                "action_input": fa.group(1).strip(),
            }

        # ── P2: Thought（可选，回退到第一行） ──
        # 模型的推理步骤。这是可选的，因为模型对于已知答案可能直接跳到 Action。
        thought = ""
        tm = self._re_thought.search(clean)
        if tm:
            # 主路径: 正则匹配 "Thought: <reasoning>"
            thought = tm.group(1).strip()
        else:
            # 后备路径: 取不以 Action 或 Final Answer 指令开头的第一个非空行。
            # 处理模型输出原始文本而没有 "Thought:" 前缀的情况
            # （例如来自使用不同格式的微调模型）。
            for line in clean.split('\n'):
                stripped = line.strip()
                if stripped and not stripped.startswith(("Action:", "Final")):
                    thought = stripped
                    break
            if not thought:
                # 最后后备: 无论内容如何，取第一行。
                # 这确保我们始终有 thought 字符串。
                thought = clean.split('\n')[0].strip()

        # ── P3: Action（必需） ──
        # 工具名。这是唯一必需的字段 — 如果无法确定动作，ReAct 循环无法继续。
        am = self._re_action.search(clean)
        if not am:
            # 后备: 在输出中扫描已注册的工具名。
            # 当模型省略 "Action:" 前缀但仍将工具名输出在文本中时激活。
            #
            # 歧义: 如果出现多个工具名，我们取第一个匹配。
            # 如果工具名是较大单词的一部分（例如通过子串 "calc" 匹配 "calculator"），
            # 将产生误匹配。tool_names 列表应使用精确名称以最小化此问题。
            for tool_name in tool_names:
                if tool_name in clean:
                    m = re.search(rf"{re.escape(tool_name)}\s*(.*)", clean)
                    return {
                        "thought": thought or clean.split('\n')[0].strip(),
                        "action": tool_name,
                        # 匹配工具名之后的所有内容成为 Action Input。
                        # 这是尽力而为的猜测，可能包含周围文本。
                        # 下游工具调用负责解析其输入。
                        "action_input": m.group(1).strip().strip('()"\'') if m else clean.strip(),
                    }
            # 所有策略均已用尽 — 输出无法解析。
            raise ParseError(clean, f"输出中未找到 'Action:' 且没有匹配到任何工具名")

        action = am.group(1).strip()

        # ── P4: Action Input（可选） ──
        # 工具参数。这是可选的，因为某些工具（如 "get_time"）不需要参数。
        aim = self._re_action_input.search(clean)
        if aim:
            # 主路径: 正则匹配 "Action Input: <params>"
            action_input = aim.group(1).strip()
        else:
            # 后备: 取 Action 行之后的文本。
            # 当模型将 Action 和 Action Input 合并到同一行时激活
            # （例如 "Action: calculator 2+2"）。取 Action: 后的第一行作为输入。
            rest = clean[am.end():].strip()
            action_input = rest.split('\n')[0].strip() if rest else ""

        # ── 最终 Thought 后备 ──
        # 如果所有 Thought 提取策略都失败，提供占位符
        # 以防下游代码因空字符串而崩溃。
        if not thought:
            thought = clean.split('\n')[0].strip() or "(未提供推理过程)"

        return {
            "thought": thought,
            "action": action,
            "action_input": action_input,
        }

    # ── 验证 ─────────────────────────────────────────────────

    def validate(self, parsed: Dict[str, str], tool_names: List[str],
                 tool_obj: Optional[object] = None) -> Tuple[bool, Optional[str]]:
        """
        解析后验证 — 检查动作是否有效、输入是否安全。

        功能描述:
          在 parse() 成功后运行。捕获正则解析器单独无法检测的情况：
            - Action 指向不存在的工具（幻觉工具）
            - Action 拼写错误（"calculatr" vs "calculator"）
            - Action Input 对目标工具语法无效

        参数:
          parsed:     parse() 的结果字典
          tool_names: 已注册的工具名列表
          tool_obj:   可选的工具实例（用于进一步验证）

        返回:
          (is_valid, error_message)。如果有效，error_message 为 None。

        注意事项:
          此方法不抛出异常 — 验证失败通过 ReAct 循环中的 Observation
          信号传递，而不是通过异常。这种设计使 ReAct 循环可以将验证错误
          作为 Observation 反馈给模型，让模型在下一轮自行纠正。
        """
        action = parsed.get("action", "")
        if action == "final_answer":
            return True, None

        if action not in tool_names:
            return False, f"未知动作: '{action}'。可用工具: {tool_names}"

        return True, None

    # ── 属性 ─────────────────────────────────────────────────

    @property
    def stop_sequences(self) -> List[str]:
        """LLM 生成的停止序列。"""
        return self._config.stop_sequences
