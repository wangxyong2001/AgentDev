"""
Prompt Template Renderer — assembles ReAct prompts from YAML protocol.

Handles Section 1 of the ReAct protocol: Agent → LLM prompt construction.

Usage:
  >>> from llama.protocol.template import PromptTemplate
  >>> tpl = PromptTemplate(yaml_path="ReActProtocol.yaml")  # YAML-driven
  >>> tpl = PromptTemplate()                                 # Built-in defaults
  >>> prompt = tpl.render_full_prompt(tool_names, question, history)
"""

from __future__ import annotations

import os
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

from llama.logging_config import get_logger

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
    role_text: str = (
        "You are a ReAct agent. You MUST follow the EXACT format below.\n"
        "Do NOT add extra explanations, do NOT use <think> tags, just follow the format:"
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
    # YAML loading
    # ═════════════════════════════════════════════════════════════════

    def __init__(self, yaml_path: Optional[str] = None, **kwargs):
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
        """Render the system prompt section (fixed prefix)."""
        names_str = ', '.join(tool_names)
        tool_items = "\n".join(
            self.tool_item_format.format(name=name, description=desc)
            for name, desc in tool_descriptions.items()
        )
        rules = "\n".join(
            r.format(tool_names=names_str) for r in self.format_rules
        )
        return (
            f"{self.role_text}\n"
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
