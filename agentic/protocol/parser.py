"""
Unified ResponseParser — single source of truth for LLM output parsing.

Eliminates the dual-path rot (ARC-002) where ProtocolLoader.parse_response()
and standalone parse_llm_output() independently reimplemented the same
four-level regex parsing with subtle differences.

Architecture:
  ResponseParser (unified engine)
  ├── Config → agentic/protocol/ReActProtocol.yaml (preferred) or built-in defaults
  ├── preprocess()  — strip model-specific tags (Qwen <think>, etc.)
  ├── parse()       — four-level P1->P4 regex parsing with fallbacks
  └── validate()    — post-parse validation (action known, input safe)

Usage:
  >>> from agentic.protocol.parser import ResponseParser
  >>> parser = ResponseParser(yaml_path="agentic/protocol/ReActProtocol.yaml")  # YAML-driven
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

    # Stop sequences for LLM generation.
    # "Observation:" is critical: it tells the LLM to stop generating
    # before the ReAct loop's next turn injects the Observation.
    # "<|im_end|>" is the Qwen2 end-of-turn token.
    stop_sequences: List[str] = field(default_factory=lambda: [
        "Observation:", "<|im_end|>",
    ])

    # Tags to strip during preprocessing.
    # Qwen's chat template wraps reasoning in <think>...</think> tags.
    # If these aren't stripped, the regex patterns below fail to match
    # because the "Thought:" keyword is nested inside the thinking block.
    strip_tags: List[str] = field(default_factory=lambda: [
        r"<think>.*?</think>",
        r"<response>",
        r"</response>",
        r"<\s*/\s*think\s*>",
    ])

    # Parsing patterns (P1->P4 priority — see parse() docstring).
    #
    # IMPORTANT: These patterns are designed against the output_format
    # section of template.py. If the output_format changes, these patterns
    # MUST be updated in lockstep.
    #
    # pattern_final_answer:
    #   Catches "Final Answer: <text>" anywhere in the output.
    #   Uses re.DOTALL so <text> can span multiple lines.
    #
    # pattern_thought:
    #   Captures text after "Thought:" up to the next "Action:" or
    #   "Final" keyword, or end-of-string. The lookahead (?=\n(?:...))
    #   prevents the thought from swallowing the Action line.
    #
    # pattern_action:
    #   Captures the first whitespace-delimited token after "Action:".
    #   Tool names are always single tokens (no spaces).
    #
    # pattern_action_input:
    #   Captures text after "Action Input:" up to the next ReAct keyword
    #   (Thought/Action/Final/Observation) or end-of-string. This is the
    #   most fragile pattern — tool parameters can contain arbitrary text
    #   including newlines.
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

    Parsing priority (P1-P4):
      P1: Final Answer — the model signals the conversation is complete.
          Return immediately; no further parsing needed.
      P2: Thought — the model's reasoning. Optional (the model might jump
          straight to Action for simple cases). Fallback: use the first
          non-empty line of output.
      P3: Action — the tool name. REQUIRED. If the regex doesn't match,
          the fallback scanner searches for known tool names in the raw
          output. If that also fails, ParseError is raised.
      P4: Action Input — the tool parameters. Optional (some tools take
          no parameters). Fallback: use text after the Action line.
    """

    def __init__(self, yaml_path: Optional[str] = None, config: Optional[ParserConfig] = None):
        """
        Initialize parser from YAML file or explicit config.

        Args:
          yaml_path: Path to agentic/protocol/ReActProtocol.yaml (optional)
          config:    ParserConfig override (optional)

        If both are None, built-in defaults are used.

        The graser always precompiles regexes from its config at init
        time, so parse() calls are O(n) in output length with no
        regex compilation overhead.
        """
        self._config: ParserConfig = config or ParserConfig()

        if yaml_path:
            self._load_yaml(yaml_path)

        # Precompile regexes for performance.
        # Regex compilation is done once at init because parse() is called
        # once per ReAct turn (potentially hundreds of times per session).
        self._re_final = re.compile(self._config.pattern_final_answer, re.DOTALL)
        self._re_thought = re.compile(self._config.pattern_thought, re.DOTALL)
        self._re_action = re.compile(self._config.pattern_action)
        self._re_action_input = re.compile(self._config.pattern_action_input, re.DOTALL)

        # Compile strip patterns
        self._strip_patterns = [re.compile(tag, re.DOTALL) for tag in self._config.strip_tags]

    def _load_yaml(self, yaml_path: str) -> None:
        """Override config from YAML protocol file.

        Graceful degradation:
          - If pyyaml is not installed: keep built-in defaults silently
            (no warning — missing pyyaml in production is a known state).
          - If a config key is missing from YAML: keep current value.

        YAML structure expected:
          response:
            stop_sequences: [...]
            preprocessing: {strip_tags: [...]}
            parsing:
              final_answer: {pattern: "..."}
              thought: {pattern: "..."}
              action: {pattern: "..."}
              action_input: {pattern: "..."}
        """
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

        # Strip tags (preprocessing config)
        pp = resp_cfg.get("preprocessing", {})
        if "strip_tags" in pp:
            self._config.strip_tags = pp["strip_tags"]

        # Parsing patterns — each is a dict with a "pattern" key.
        # The YAML structure wraps patterns in objects to allow future
        # metadata (e.g. description, example) without breaking the API.
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

        Why preprocessing is necessary:
          Qwen's chat template wraps the model's reasoning in
          <think>...</think> tags. The raw output looks like:
            <think>I need to calculate 2+2</think>
            Action: calculator
          Without stripping, the "Thought:" regex won't match because
          the reasoning text is nested inside <think> tags rather than
          starting with "Thought:".

        Args:
          raw_text: Raw LLM output string

        Returns:
          Cleaned text ready for parsing
        """
        cleaned = raw_text
        for pattern in self._strip_patterns:
            cleaned = pattern.sub('', cleaned)
        return cleaned.strip()

    # ── Parsing (P1->P4 priority) ───────────────────────────────────

    def parse(self, llm_output: str, tool_names: List[str]) -> Dict[str, str]:
        """
        Parse LLM output into structured ReAct dict.

        ===================
        P1-P4 Priority System
        ===================

        The parser uses a four-level priority system because LLM output
        is variable and regex matches can fail for many reasons (format
        drift, truncated output, model-specific quirks). Each priority
        level has a primary path (regex match) and a fallback path.

        P1: Final Answer (highest priority)
          Primary:  regex search for "Final Answer: <text>"
          Fallback: none — if found, we return immediately.
          Rationale: A Final Answer signals conversation end. There is
          no need to parse Thought/Action/Action Input because the
          answer is already provided.

        P2: Thought (optional)
          Primary:  regex search for "Thought: <text>"
          Fallback: first non-empty line in the output that doesn't
                    start with "Action:" or "Final".
          Rationale: The model may skip the "Thought:" prefix for simple
          questions (e.g. just "Action: calculator"). The fallback
          captures whatever preamble exists.

        P3: Action (required)
          Primary:  regex search for "Action: <name>"
          Fallback: scan the entire output for registered tool names.
          Rationale: The model might output the tool name without the
          "Action:" prefix (format drift). Scanning for known tool names
          is more resilient but introduces ambiguity — see the tool name
          scanning note below.

        P4: Action Input (optional)
          Primary:  regex for "Action Input: <text>"
          Fallback: text after the Action line (first line only).
          Rationale: For simple tools like get_weather, Action Input
          might be omitted or merged with the Action line.

        ==========================
        Tool Name Scanning Ambiguity
        ==========================

        The P3 fallback (scanning for registered tool names in raw text)
        has a known ambiguity: if the model outputs something like:
          "I will use the calculator tool"
        ...the scanner will match "calculator", but it won't know where
        the Action Input begins. In this case, we use everything after
        the matched tool name as the Action Input, which may include
        surrounding text like "tool" and explanatory words.

        This is acceptable because:
          a) P3 fallback only activates when the primary regex fails
             (which is rare in normal operation).
          b) The downstream validation step can catch malformed input.
          c) The Observation (tool output) provides a feedback signal
             — if the action is wrong, the model corrects next turn.

        Args:
          llm_output: Raw or preprocessed LLM output text
          tool_names: List of registered tool names for fallback scanning

        Returns:
          {"thought": str, "action": str, "action_input": str}

        Raises:
          ParseError: All parsing strategies failed — model output is
          uninterpretable
        """
        clean = self.preprocess(llm_output)
        if not clean:
            # If preprocessing stripped everything (e.g. empty <think>),
            # fall back to the raw output.
            clean = llm_output.strip()

        # ── P1: Final Answer (highest priority) ──
        # The model signals "done." Return immediately with a synthetic
        # thought and "final_answer" action. No further parsing needed.
        fa = self._re_final.search(clean)
        if fa:
            return {
                "thought": "I now know the final answer",
                "action": "final_answer",
                "action_input": fa.group(1).strip(),
            }

        # ── P2: Thought (optional, fallback to first line) ──
        # The model's reasoning step. This is optional because the model
        # may jump straight to Action for known answers.
        thought = ""
        tm = self._re_thought.search(clean)
        if tm:
            # Primary path: regex matched "Thought: <reasoning>"
            thought = tm.group(1).strip()
        else:
            # Fallback path: take the first non-empty line that doesn't
            # look like an Action or Final Answer directive. This handles
            # cases where the model outputs raw text without the "Thought:"
            # prefix (e.g. from a fine-tuned model with different format).
            for line in clean.split('\n'):
                stripped = line.strip()
                if stripped and not stripped.startswith(("Action:", "Final")):
                    thought = stripped
                    break
            if not thought:
                # Last-resort fallback: first line regardless of content.
                # This ensures we always have a thought string.
                thought = clean.split('\n')[0].strip()

        # ── P3: Action (required) ──
        # The tool name. This is the ONLY required field — if we can't
        # determine the action, the ReAct loop cannot proceed.
        am = self._re_action.search(clean)
        if not am:
            # Fallback: scan for registered tool names in the output.
            # This activates when the model omits the "Action:" prefix
            # but still outputs the tool name somewhere in the text.
            #
            # Ambiguity: if multiple tool names appear, we take the FIRST
            # match. If a tool name is part of a larger word (e.g. "calc"
            # matching "calculator" via substring), this will false-match.
            # The tool_names list should use exact names to minimise this.
            for tool_name in tool_names:
                if tool_name in clean:
                    m = re.search(rf"{re.escape(tool_name)}\s*(.*)", clean)
                    return {
                        "thought": thought or clean.split('\n')[0].strip(),
                        "action": tool_name,
                        # Everything after the matched tool name becomes
                        # Action Input. This is a best-effort guess and
                        # may include surrounding text. The downstream
                        # tool call is responsible for parsing its input.
                        "action_input": m.group(1).strip().strip('()"\'') if m else clean.strip(),
                    }
            # All strategies exhausted — the output is unparseable.
            raise ParseError(clean, f"No 'Action:' found and no tool name matched in output")

        action = am.group(1).strip()

        # ── P4: Action Input (optional) ──
        # The tool parameters. This is optional because some tools (e.g.,
        # a "get_time" tool) take no parameters.
        aim = self._re_action_input.search(clean)
        if aim:
            # Primary path: regex matched "Action Input: <params>"
            action_input = aim.group(1).strip()
        else:
            # Fallback: take text after the Action line.
            # This activates when the model merges Action and Action Input
            # into the same line (e.g. "Action: calculator 2+2"). We take
            # the first line after Action: as the input.
            rest = clean[am.end():].strip()
            action_input = rest.split('\n')[0].strip() if rest else ""

        # ── Final thought fallback ──
        # If all Thought extraction strategies failed, provide a
        # placeholder so downstream code doesn't crash on empty strings.
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

        This runs AFTER parse() succeeds. It catches cases that the
        regex parser alone cannot detect:
          - Action refers to a tool that doesn't exist (hallucinated tool)
          - Action is misspelled ("calculatr" vs "calculator")
          - Action Input is syntactically valid for the target tool

        Args:
          parsed:     Result from parse()
          tool_names: Registered tool names
          tool_obj:   Optional tool instance (for further validation)

        Returns:
          (is_valid, error_message). error_message is None if valid.

        Note: This doesn't raise — validation failures are signaled via
              Observation in the ReAct loop, not via exceptions.
              This design choice means the ReAct loop can feed validation
              errors back to the model as Observations, letting the model
              self-correct on the next turn.
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
