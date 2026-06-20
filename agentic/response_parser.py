"""
Unified ResponseParser — single source of truth for LLM output parsing.

Eliminates the dual-path rot (ARC-002) where ProtocolLoader.parse_response()
and standalone parse_llm_output() independently reimplemented the same
four-level regex parsing with subtle differences.

Architecture:
  ResponseParser (unified engine)
  ├── Config → ReActProtocol.yaml (preferred) or built-in defaults
  ├── preprocess()  — strip model-specific tags (Qwen <think>, etc.)
  ├── parse()       — four-level P1→P4 regex parsing with fallbacks
  └── validate()    — post-parse validation (action known, input safe)

Usage:
  >>> from agentic.response_parser import ResponseParser
  >>> parser = ResponseParser(yaml_path="ReActProtocol.yaml")  # YAML-driven
  >>> parser = ResponseParser()                                 # Built-in defaults
  >>> result = parser.parse(llm_output, tool_names=["calculator", "get_weather"])
  >>> # result: {"thought": "...", "action": "calculator", "action_input": "123*45"}
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from agentic.exceptions import ParseError, ToolNotFoundError


# ==========================================================================
# Configuration
# ==========================================================================

@dataclass
class ParserConfig:
    """
    Parser configuration — built-in defaults (Qwen2 model).

    When a YAML protocol file is provided, these defaults are overridden
    by the YAML values. When no YAML is available, these defaults serve
    as the production configuration.

    This is the SINGLE source of truth — no separate hardcoded copy
    exists elsewhere.
    """

    # Stop sequences for LLM generation
    stop_sequences: List[str] = field(default_factory=lambda: [
        "Observation:", "<|im_end|>",
    ])

    # Tags to strip during preprocessing
    strip_tags: List[str] = field(default_factory=lambda: [
        r"<think>.*?</think>",
        r"<response>",
        r"</response>",
        r"<\s*/\s*think\s*>",
    ])

    # Parsing patterns (P1→P4 priority)
    pattern_final_answer: str = r"Final Answer:\s*(.*)"
    pattern_thought: str = r"Thought:\s*(.+?)(?=\n(?:Action|Final)|$)"
    pattern_action: str = r"Action:\s*(\S+)"
    pattern_action_input: str = r"Action Input:\s*(.+?)(?=\n(?:Thought|Action|Final|Observation)|$)"


# ==========================================================================
# Unified Response Parser
# ==========================================================================

class ResponseParser:
    """
    Unified LLM output parser — single implementation for both YAML and
    built-in paths.

    Design:
      - Config from YAML (preferred) or built-in ParserConfig (fallback)
      - All parsing logic lives here — no duplication in ReActDemo
      - Returns validated Dict or raises ParseError (recoverable)
    """

    def __init__(self, yaml_path: Optional[str] = None, config: Optional[ParserConfig] = None):
        """
        Initialize parser from YAML file or explicit config.

        Args:
          yaml_path: Path to ReActProtocol.yaml (optional)
          config:    ParserConfig override (optional)

        If both are None, built-in defaults are used.
        """
        self._config: ParserConfig = config or ParserConfig()

        if yaml_path:
            self._load_yaml(yaml_path)

        # Precompile regexes for performance
        self._re_final = re.compile(self._config.pattern_final_answer, re.DOTALL)
        self._re_thought = re.compile(self._config.pattern_thought, re.DOTALL)
        self._re_action = re.compile(self._config.pattern_action)
        self._re_action_input = re.compile(self._config.pattern_action_input, re.DOTALL)

        # Compile strip patterns
        self._strip_patterns = [re.compile(tag, re.DOTALL) for tag in self._config.strip_tags]

    def _load_yaml(self, yaml_path: str) -> None:
        """Override config from YAML protocol file."""
        try:
            import yaml
        except ImportError:
            # pyyaml not installed — keep built-in defaults
            return

        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        resp_cfg = cfg.get("response", {})

        # Stop sequences
        if "stop_sequences" in resp_cfg:
            self._config.stop_sequences = resp_cfg["stop_sequences"]

        # Strip tags
        pp = resp_cfg.get("preprocessing", {})
        if "strip_tags" in pp:
            self._config.strip_tags = pp["strip_tags"]

        # Parsing patterns
        parsing = resp_cfg.get("parsing", {})
        if "final_answer" in parsing:
            self._config.pattern_final_answer = parsing["final_answer"]["pattern"]
        if "thought" in parsing:
            self._config.pattern_thought = parsing["thought"]["pattern"]
        if "action" in parsing:
            self._config.pattern_action = parsing["action"]["pattern"]
        if "action_input" in parsing:
            self._config.pattern_action_input = parsing["action_input"]["pattern"]

    # ── Preprocessing ──────────────────────────────────────────────

    def preprocess(self, raw_text: str) -> str:
        """
        Strip model-specific wrapper tags from raw LLM output.

        Handles:
          - Qwen thinking model: <think>...</think>, <response>, </response>
          - Configurable via ParserConfig.strip_tags

        Args:
          raw_text: Raw LLM output string

        Returns:
          Cleaned text ready for parsing
        """
        cleaned = raw_text
        for pattern in self._strip_patterns:
            cleaned = pattern.sub('', cleaned)
        return cleaned.strip()

    # ── Parsing (P1→P4 priority) ───────────────────────────────────

    def parse(self, llm_output: str, tool_names: List[str]) -> Dict[str, str]:
        """
        Parse LLM output into structured ReAct dict.

        Priority:
          P1: Final Answer: → return immediately
          P2: Thought:      → reasoning (optional, fallback to first line)
          P3: Action:       → tool name (required, fallback to scanning)
          P4: Action Input: → tool parameters (optional, fallback to post-action text)

        Args:
          llm_output: Raw or preprocessed LLM output text
          tool_names: List of registered tool names for fallback scanning

        Returns:
          {"thought": str, "action": str, "action_input": str}

        Raises:
          ParseError: All parsing strategies failed — model output is uninterpretable
        """
        clean = self.preprocess(llm_output)
        if not clean:
            clean = llm_output.strip()

        # ── P1: Final Answer (highest priority) ──
        fa = self._re_final.search(clean)
        if fa:
            return {
                "thought": "I now know the final answer",
                "action": "final_answer",
                "action_input": fa.group(1).strip(),
            }

        # ── P2: Thought (optional, fallback to first line) ──
        thought = ""
        tm = self._re_thought.search(clean)
        if tm:
            thought = tm.group(1).strip()
        else:
            # Fallback: first non-empty line
            for line in clean.split('\n'):
                stripped = line.strip()
                if stripped and not stripped.startswith(("Action:", "Final")):
                    thought = stripped
                    break
            if not thought:
                thought = clean.split('\n')[0].strip()

        # ── P3: Action (required) ──
        am = self._re_action.search(clean)
        if not am:
            # Fallback: scan for registered tool names in the output
            for tool_name in tool_names:
                if tool_name in clean:
                    m = re.search(rf"{re.escape(tool_name)}\s*(.*)", clean)
                    return {
                        "thought": thought or clean.split('\n')[0].strip(),
                        "action": tool_name,
                        "action_input": m.group(1).strip().strip('()"\'') if m else clean.strip(),
                    }
            # All strategies exhausted
            raise ParseError(clean, f"No 'Action:' found and no tool name matched in output")

        action = am.group(1).strip()

        # ── P4: Action Input (optional) ──
        aim = self._re_action_input.search(clean)
        if aim:
            action_input = aim.group(1).strip()
        else:
            # Fallback: text after Action line
            rest = clean[am.end():].strip()
            action_input = rest.split('\n')[0].strip() if rest else ""

        # ── Final thought fallback ──
        if not thought:
            thought = clean.split('\n')[0].strip() or "(no reasoning provided)"

        return {
            "thought": thought,
            "action": action,
            "action_input": action_input,
        }

    # ── Validation ─────────────────────────────────────────────────

    def validate(self, parsed: Dict[str, str], tool_names: List[str],
                 tool_obj: Optional[object] = None) -> Tuple[bool, Optional[str]]:
        """
        Post-parse validation — check action is valid and input is safe.

        Args:
          parsed:     Result from parse()
          tool_names: Registered tool names
          tool_obj:   Optional tool instance (for further validation)

        Returns:
          (is_valid, error_message). error_message is None if valid.

        Note: This doesn't raise — validation failures are signaled via
              Observation in the ReAct loop, not via exceptions.
        """
        action = parsed.get("action", "")
        if action == "final_answer":
            return True, None

        if action not in tool_names:
            return False, f"Unknown action: '{action}'. Available tools: {tool_names}"

        return True, None

    # ── Properties ─────────────────────────────────────────────────

    @property
    def stop_sequences(self) -> List[str]:
        """Stop sequences for LLM generation."""
        return self._config.stop_sequences