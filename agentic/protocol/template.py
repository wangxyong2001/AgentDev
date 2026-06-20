"""
Prompt Template Renderer — assembles ReAct prompts from YAML protocol.

Handles Section 1 of the ReAct protocol: Agent → LLM prompt construction.

Usage:
  >>> from agentic.protocol.template import PromptTemplate
  >>> tpl = PromptTemplate(yaml_path="agentic/protocol/ReActProtocol.yaml")  # YAML-driven
  >>> tpl = PromptTemplate()                                 # Built-in defaults
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
    ReAct prompt template renderer.

    Renders the system prompt + user message + history + trigger
    into a complete prompt string for LLM consumption.

    Configurable via YAML protocol file or built-in Qwen2 defaults.

    =====================
    Cacheability Strategy
    =====================

    The prompt assembly order is designed to maximise KV-cache reuse
    across turns. Because the system prompt XML sections (role_text)
    are IDENTICAL across every turn, a cache-friendly inference engine
    can reuse the computed KV entries for the prefix from the previous
    turn, only computing new tokens for the user message and history.

    Cacheable (static across turns):
      - role_text (all XML-tagged sections)
      - format_rules (tool names vary, but the rule strings are fixed)
      - tools_header + tool_item_format (tool *descriptions* vary,
        but the structural prefix is reused)

    Varies per turn (not cacheable):
      - user_template (question changes)
      - history_entry_format (history grows)
      - trigger (always "Thought:" — this is the model's cue to start
        generating, and marks the boundary between input and output)
    """

    # ── Delimiters ───────────────────────────────────────────────────
    # Chat-template markers that wrap each role's content. These are
    # model-specific: the defaults below target Qwen2 (ChatML format),
    # but they can be overridden via YAML for other models (e.g. Llama 3
    # uses <|start_header_id|>...<|end_header_id|>).
    system_start: str = "<|im_start|>system\n"
    system_end: str = "<|im_end|>"
    user_start: str = "<|im_start|>user\n"
    user_end: str = "<|im_end|>"
    assistant_start: str = "<|im_start|>assistant\n"
    # NOTE: assistant_end is intentionally empty (""). We want the model
    # to continue generating without a closing delimiter. If we placed
    # <|im_end|> here, the model would stop before producing its output.
    assistant_end: str = ""

    # ── System prompt template ───────────────────────────────────────
    #
    # IMPORTANT — Cacheability note:
    # The role_text below is a STATIC/CACHEABLE prefix — it never
    # changes between turns. The XML-tagged sections anchor model
    # attention and enable KV-cache reuse across the conversation.
    # Only format_rules (inline reminder) and tool_items vary per turn.
    #
    # Each XML section has a specific design rationale:
    #
    #   <identity>
    #     Establishes the agent's persona upfront. This anchors the
    #     model's self-perception before any instructions, reducing
    #     persona-jacking attacks where injected content later in
    #     the prompt tries to redefine the agent's role.
    #
    #   <objectives>
    #     High-level task decomposition guide. This sets the reasoning
    #     strategy before tool details, so the model thinks in terms
    #     of steps first, tool selection second.
    #
    #   <first_turn_behavior>
    #     Handles the ambiguous-question case explicitly. Without this,
    #     the model might guess rather than ask for clarification,
    #     leading to wasted turns.
    #
    #   <tone_and_style>
    #     Prevents "social padding" (phrases like "Great question!")
    #     that wastes tokens and adds latency. The "one line per step"
    #     constraint keeps the output parseable by the regex parser.
    #
    #   <tools>
    #     Inline tool definitions. These are static examples — the
    #     actual dynamic tool list is appended via tools_header +
    #     tool_items later. The static examples serve as a fallback
    #     in case the dynamic list is truncated or mis-ordered.
    #
    #   <workflow>
    #     The core ReAct loop specification. The {tool_names} placeholder
    #     is substituted at render time. The "one line per Thought"
    #     constraint is critical for parseability — multi-line thoughts
    #     confuse the regex-based parser.
    #
    #   <guardrails>
    #     Explicit safety constraints. These are separated from the
    #     instructions so they stand out as immutable rules, not
    #     procedural suggestions. The "CONFIRMATION REQUIRED" header
    #     pattern is used by downstream safety filters.
    #
    #   <output_format>
    #     The exact format the parser expects. The parser's regex patterns
    #     (P1-P4) are designed against this format — changing this section
    #     without updating parser.py will break parsing.
    #
    #   <error_handling>
    #     Recovery strategies for common failure modes. Without explicit
    #     error-handling guidance, the model tends to retry the same
    #     failing action repeatedly (perseveration failure).
    #
    #   <internal_logic>
    #     Decision-priority rules for tool selection. The "Cache-aware"
    #     line reminds the model that the system prompt is always available
    #     (via KV-cache), so it doesn't need to re-read it.
    #
    role_text: str = (
        "Follow this EXACT format.\n\n"
        "<identity>\n"
        "You are a ReAct Agent running on local edge hardware.\n"
        "Your purpose is to answer questions by reasoning step-by-step and calling tools.\n"
        "You do NOT: execute arbitrary code, access the network, or modify system files.\n"
        "</identity>\n\n"
        "<objectives>\n"
        "1. Understand the user's question\n"
        "2. Break it into solvable steps\n"
        "3. Use tools when calculation or external data is needed\n"
        "4. Provide a clear, final answer with reasoning shown\n"
        "</objectives>\n\n"
        "<first_turn_behavior>\n"
        'If the question is clear: start reasoning immediately with "Thought:"\n'
        'If the question is ambiguous: ask ONE clarifying question before proceeding\n'
        "</first_turn_behavior>\n\n"
        "<tone_and_style>\n"
        "- Be concise — one Thought per step, one Action per step\n"
        '- Start responses directly — no "Great question!" or flattery\n'
        "- Use plain English for reasoning\n"
        "</tone_and_style>\n\n"
        "<tools>\n"
        "Available tools listed below. Choose the correct tool based on the task:\n"
        '- calculator: Math expression evaluation (e.g. "2+2", "10*5")\n'
        '- get_weather: Current weather lookup (e.g. "London", "Tokyo")\n'
        "Only use a tool when necessary. If you already know the answer, use Final Answer directly.\n"
        "</tools>\n\n"
        "<workflow>\n"
        "Follow this exact sequence:\n"
        "1. Thought: Write your reasoning in ONE line\n"
        "2. Action: Write ONLY the tool name ({tool_names})\n"
        "3. Action Input: Write the tool parameters\n"
        "4. Wait for Observation before next step\n"
        '5. When answer is ready: write "Final Answer: your answer"\n'
        "</workflow>\n\n"
        "<guardrails>\n"
        "CONFIRMATION REQUIRED: None (read-only tools, safe execution environment)\n"
        "PROHIBITED: Generating code other than math expressions, accessing files, making network requests\n"
        "OPERATIONAL LIMITS: Maximum 8 reasoning steps per question\n"
        "</guardrails>\n\n"
        "<output_format>\n"
        "You MUST follow this EXACT format. Do NOT add extra explanations, do NOT use think tags:\n\n"
        "Thought: [your reasoning in one line]\n"
        'Action: [tool name or "final_answer"]\n'
        "Action Input: [tool input, or final answer text]\n"
        "</output_format>\n\n"
        "<error_handling>\n"
        "- If tool returns an error: read the error, adjust your approach, try the tool again with corrected input or use a different tool\n"
        "- If you cannot solve after 3 attempts on the same step: explain why and provide best partial answer\n"
        "- If tool is unknown: check available tools list and retry with a valid tool name\n"
        "</error_handling>\n\n"
        "<internal_logic>\n"
        "Priority order when multiple tools could work:\n"
        "1. If math calculation needed → calculator\n"
        "2. If weather/location data needed → get_weather\n"
        "3. If answer can be derived without tools → Final Answer directly\n"
        "Cache-aware: the system prompt never changes, so your format rules are always available.\n"
        "</internal_logic>"
    )
    format_rules: List[str] = field(default_factory=lambda: [
        'After "Thought:" write your reasoning in ONE line',
        'After "Action:" write ONLY the tool name: {tool_names}',
        'After "Action Input:" write the tool input',
        'After you have the answer, use "Final Answer:"',
    ])
    tools_header: str = "Available tools:"
    tool_item_format: str = "- {name}: {description}"

    # ── User message template ────────────────────────────────────────
    # Simple substitution template. The question is the only variable
    # part — wrapping it in <user> delimiters is handled by the
    # render_full_prompt method's assembly order.
    user_template: str = "{question}"

    # ── History entry format ─────────────────────────────────────────
    # Each ReAct turn produces one history entry. The format matches
    # what the LLM outputs (Thought/Action/Action Input/Observation)
    # so the assembled history looks like a natural continuation of
    # the conversation.
    history_entry_format: str = (
        "Thought: {thought}\n"
        "Action: {action}\n"
        "Action Input: {action_input}\n"
        "Observation: {observation}"
    )

    # ── Trigger ──────────────────────────────────────────────────────
    # The trigger text is appended after the history. It cues the model
    # to start generating its next Thought. "Thought:" is the standard
    # ReAct trigger because it's the first token of every agent output.
    trigger: str = "Thought:"

    # ── Stop sequences ───────────────────────────────────────────────
    # The LLM should stop generating when it produces these tokens.
    # "Observation:" marks the boundary where tool output begins (we
    # generate the Observation externally, not via the LLM).
    # <|im_end|> is the Qwen2 end-of-turn token.
    stop_sequences: List[str] = field(default_factory=lambda: [
        "Observation:", "<|im_end|>",
    ])

    # ── Generation params ────────────────────────────────────────────
    max_tokens: int = 512
    temperature: float = 0.2
    max_steps: int = 8

    # ═════════════════════════════════════════════════════════════════
    # XML tags (for documentation / introspection)
    # ═════════════════════════════════════════════════════════════════

    @property
    def xml_tags(self) -> List[str]:
        """Return the list of XML tag names used in role_text.

        These tags serve as the canonical inventory of prompt sections.
        Each tag anchors a specific behavioural directive, and the tag
        names are used by downstream metrics code to measure per-section
        prompt reuse.

        Returns:
            List of XML tag names in display order (same order as role_text).
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
    # YAML loading
    # ═════════════════════════════════════════════════════════════════

    def __init__(self, yaml_path: Optional[str] = None, **kwargs):
        """
        Custom __init__ because @dataclass generates one, but we need
        to interleave YAML loading with field initialisation.

        Why the manual field loop:
          A dataclass auto-generates __init__ with positional args for
          every field. When we override __init__ with a custom signature
          (yaml_path, **kwargs), the dataclass-generated __init__ is
          replaced — so we must manually initialise each field from its
          default/default_factory, then apply kwargs overrides, then
          optionally load YAML on top.

        This three-layer override chain is:
          1. Dataclass defaults (hardcoded Qwen2 values above)
          2. **kwargs (programmatic overrides from callers)
          3. YAML protocol file (on-disk configuration, highest priority)

        Args:
            yaml_path: Path to agentic/protocol/ReActProtocol.yaml.
                If None, only defaults + kwargs are used.
            **kwargs: Field-by-field overrides applied BEFORE YAML,
                so YAML can still override them. This enables callers
                to set a fallback value that YAML can replace.
        """
        # Step 1: Initialise all dataclass fields with their defaults.
        # This is necessary because the custom __init__ replaces the
        # auto-generated one, so fields would remain unset otherwise.
        for f in fields(self.__class__):
            if f.default is not MISSING:
                # Simple default value (string, int, etc.)
                setattr(self, f.name, f.default)
            elif f.default_factory is not MISSING:
                # Fields with default_factory (list, dict, etc.) need
                # the factory called to create a fresh instance — shared
                # mutable defaults are a classic Python gotcha.
                setattr(self, f.name, f.default_factory())

        # Step 2: Apply kwargs to dataclass fields.
        # This allows callers like PromptTemplate(temperature=0.1) to
        # override individual fields without touching YAML.
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        # Step 3: Load YAML overrides (highest priority).
        if yaml_path:
            self._load_yaml(yaml_path)

    def _load_yaml(self, yaml_path: str) -> None:
        """Load prompt configuration from YAML protocol file.

        This method selectively overrides instance attributes from the
        YAML dict. Only keys that exist in the YAML config are changed;
        any field not present in YAML keeps its current value (whether
        from defaults or kwargs).

        YAML structure expected (from ReActProtocol.yaml):
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

        Graceful degradation:
          - If pyyaml is not installed: log a warning and keep defaults.
          - If the file does not exist: log a warning and keep defaults.
          - If a key is missing from the YAML: keep the current value.
        """
        try:
            import yaml
        except ImportError:
            logger.warning("pyyaml not installed — using built-in defaults")
            return

        if not os.path.exists(yaml_path):
            logger.warning(f"YAML protocol file not found: {yaml_path}")
            return

        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        prompt_cfg = cfg.get("prompt", {})

        # Delimiters — model-specific chat template tokens.
        # These are kept separate from sections because they're a
        # model property, not a content property.
        d = prompt_cfg.get("delimiters", {})
        if d:
            self.system_start = d.get("system_start", self.system_start)
            self.system_end = d.get("system_end", self.system_end)
            self.user_start = d.get("user_start", self.user_start)
            self.user_end = d.get("user_end", self.user_end)
            self.assistant_start = d.get("assistant_start", self.assistant_start)
            self.assistant_end = d.get("assistant_end", self.assistant_end)

        # Sections — the actual prompt content broken into role blocks.
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

        # Response config — generation parameters and stop sequences.
        resp_cfg = cfg.get("response", {})
        gen_cfg = resp_cfg.get("generation", {})
        self.max_tokens = gen_cfg.get("max_tokens", self.max_tokens)
        self.temperature = gen_cfg.get("temperature", self.temperature)
        self.max_steps = gen_cfg.get("max_steps", self.max_steps)
        self.stop_sequences = resp_cfg.get("stop_sequences", self.stop_sequences)

        logger.debug("PromptTemplate loaded from YAML")

    # ═════════════════════════════════════════════════════════════════
    # Rendering
    # ═════════════════════════════════════════════════════════════════

    def render_system_prompt(self, tool_names: List[str],
                             tool_descriptions: Dict[str, str]) -> str:
        """Render the system prompt section (fixed prefix).

        The XML-tagged sections (role_text) are static/cacheable across
        turns — only format_rules and tool_items vary per request.

        Assembly order:
          1. role_text with {tool_names} substituted (cacheable base)
          2. format_rules with {tool_names} substituted (small variation)
          3. tools_header + dynamic tool_items (varies if tool set changes)

        Args:
            tool_names: List of tool names for {tool_names} placeholder.
            tool_descriptions: Dict mapping name -> description for
                dynamic tool list rendering.

        Returns:
            Fully rendered system prompt string (without delimiters).
        """
        names_str = ', '.join(tool_names)
        tool_items = "\n".join(
            self.tool_item_format.format(name=name, description=desc)
            for name, desc in tool_descriptions.items()
        )
        rules = "\n".join(
            r.format(tool_names=names_str) for r in self.format_rules
        )
        # role_text may contain {tool_names} placeholder (XML <workflow> section)
        formatted_role = self.role_text.format(tool_names=names_str)
        return (
            f"{formatted_role}\n"
            f"{rules}\n\n"
            f"{self.tools_header}\n"
            f"{tool_items}"
        )

    def render_user_message(self, question: str) -> str:
        """Render the user message section.

        Args:
            question: The user's input string.

        Returns:
            The user message content (without delimiters — delimiters
            are added by render_full_prompt).
        """
        return self.user_template.format(question=question)

    def render_history_entry(self, thought: str, action: str,
                             action_input: str, observation: str) -> str:
        """Render a single ReAct history entry.

        Args:
            thought: Model's reasoning line.
            action: Tool name called (or "final_answer").
            action_input: Parameters passed to the tool.
            observation: Tool output.

        Returns:
            Formatted history entry string ready to be appended to
            the assistant section.
        """
        return self.history_entry_format.format(
            thought=thought, action=action,
            action_input=action_input, observation=observation,
        )

    def render_full_prompt(self, tool_names: List[str],
                           tool_descriptions: Dict[str, str],
                           question: str, history: str) -> str:
        """
        Render a complete ReAct prompt for LLM.

        Assembly order (Canonical cache-friendly order):
          <system> → <user> → <assistant + history> → Thought:

        This order is designed for KV-cache reuse:
          - The <system> block is identical across turns (cacheable).
          - The <user> block typically changes per turn (not cacheable).
          - The <assistant> + history grows monotonically (partially
            cacheable — the prefix of history from previous turns can
            be reused).
          - "Thought:" is the generation trigger (always the same).

        Args:
            tool_names: List of tool names for prompt rendering.
            tool_descriptions: Dict of tool name -> description.
            question: The current user question.
            history: Accumulated ReAct history string (pre-formatted).

        Returns:
            Complete prompt string ready for LLM inference.
        """
        system = self.render_system_prompt(tool_names, tool_descriptions)
        user_msg = self.render_user_message(question)

        return (
            f"{self.system_start}{system}{self.system_end}\n"
            f"{self.user_start}{user_msg}{self.user_end}\n"
            f"{self.assistant_start}{history}\n"
            f"{self.trigger}"
        )
