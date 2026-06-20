"""
Prompt Template Renderer — assembles ReAct prompts from YAML protocol.

Handles Section 1 of the ReAct protocol: Agent → LLM prompt construction.

Usage:
  >>> from agentic.protocol.template import PromptTemplate
  >>> tpl = PromptTemplate(yaml_path="ReActProtocol.yaml")  # YAML-driven
  >>> tpl = PromptTemplate()                                 # Built-in defaults
  >>> prompt = tpl.render_full_prompt(tool_names, question, history)
"""

from __future__ import annotations

import os
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, fields, MISSING

from agentic.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PromptTemplate:
    """
    ReAct prompt template renderer.

    Renders the system prompt + user message + history + trigger
    into a complete prompt string for LLM consumption.

    Configurable via YAML protocol file or built-in Qwen2 defaults.
    """

    # ── Delimiters ───────────────────────────────────────────────────
    system_start: str = "<|im_start|>system\n"
    system_end: str = "<|im_end|>"
    user_start: str = "<|im_start|>user\n"
    user_end: str = "<|im_end|>"
    assistant_start: str = "<|im_start|>assistant\n"
    assistant_end: str = ""  # Intentional: let model continue

    # ── System prompt template ───────────────────────────────────────
    # NOTE: The role_text below is a STATIC/CACHEABLE prefix — it never
    # changes between turns. The XML-tagged sections anchor model
    # attention and enable KV-cache reuse across the conversation.
    # Only format_rules (inline reminder) and tool_items vary per turn.
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
    user_template: str = "{question}"

    # ── History entry format ─────────────────────────────────────────
    history_entry_format: str = (
        "Thought: {thought}\n"
        "Action: {action}\n"
        "Action Input: {action_input}\n"
        "Observation: {observation}"
    )

    # ── Trigger ──────────────────────────────────────────────────────
    trigger: str = "Thought:"

    # ── Stop sequences ───────────────────────────────────────────────
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
        """Return the list of XML tag names used in role_text."""
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
        # Initialize all dataclass fields with their defaults
        # (needed because custom __init__ overrides dataclass-generated one)
        for f in fields(self.__class__):
            if f.default is not MISSING:
                setattr(self, f.name, f.default)
            elif f.default_factory is not MISSING:
                setattr(self, f.name, f.default_factory())

        # Apply kwargs to dataclass fields
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        if yaml_path:
            self._load_yaml(yaml_path)

    def _load_yaml(self, yaml_path: str) -> None:
        """Load prompt configuration from YAML protocol file."""
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

        # Delimiters
        d = prompt_cfg.get("delimiters", {})
        if d:
            self.system_start = d.get("system_start", self.system_start)
            self.system_end = d.get("system_end", self.system_end)
            self.user_start = d.get("user_start", self.user_start)
            self.user_end = d.get("user_end", self.user_end)
            self.assistant_start = d.get("assistant_start", self.assistant_start)
            self.assistant_end = d.get("assistant_end", self.assistant_end)

        # Sections
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

        # Response config
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
        """Render the user message section."""
        return self.user_template.format(question=question)

    def render_history_entry(self, thought: str, action: str,
                             action_input: str, observation: str) -> str:
        """Render a single ReAct history entry."""
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
        """
        system = self.render_system_prompt(tool_names, tool_descriptions)
        user_msg = self.render_user_message(question)

        return (
            f"{self.system_start}{system}{self.system_end}\n"
            f"{self.user_start}{user_msg}{self.user_end}\n"
            f"{self.assistant_start}{history}\n"
            f"{self.trigger}"
        )
