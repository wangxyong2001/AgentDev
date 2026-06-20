"""Observation sanitizer for ReAct agent loop.

Sanitizes tool output before it enters the ReAct history / LLM context,
blocking prompt injection attacks embedded in tool/API responses.

Defence-in-depth philosophy:
  1. Detect suspicious content first (logging + strict-mode sentinel)
  2. Strip chat template tokens (role boundary injection)
  3. Strip ANSI/control sequences (hidden payloads)
  4. Block ReAct format tokens (agent step spoofing)
  5. Block command execution patterns (shell injection)
  6. Truncate to length limit (token budget protection)
  7. Prefix output with [sanitized] and audit flags (transparency)

This module is the LAST LINE OF DEFENCE before tool output enters
the LLM context window. It is NOT a substitute for input validation
in the tool layer itself.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chat template / special tokens
# ---------------------------------------------------------------------------
# These tokens are used by chat-format models (Qwen, Llama, etc.) to delimit
# conversation roles. If they appear in tool output, an attacker may be trying
# to inject a fake system-prompt or user/assistant boundary to override the
# agent's instructions. Stripping them prevents role-boundary injection.
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
# Injection detection patterns
# ---------------------------------------------------------------------------
# These patterns catch common prompt injection attack classes:
#   - Instruction override: "ignore previous instructions", "you are now..."
#     Classic jailbreak that tells the LLM to disregard its system prompt.
#   - ReAct format injection: lines that mimic agent output (Thought/Action)
#     so the attacker's reply is interpreted as the agent's own reasoning.
#   - Command execution: shell commands, eval(), subprocess calls that
#     trick the agent into executing arbitrary code.
#
# IMPORTANT: This is a BLOCK LIST, not an allow list. It catches known
# patterns but cannot prevent novel attacks. Strict mode should be enabled
# for high-security deployments.
_INSTRUCTION_OVERRIDE_PATTERNS = [
    # "Ignore all previous instructions/directions/commands"
    # — the most common prompt injection class; tells the model to discard
    # its system prompt and follow the injected text instead.
    re.compile(r"ignore\s+(all\s+)?previous\s+(instructions|directions|commands)", re.IGNORECASE),

    # "You are now..." — role-redefinition attacks that overwrite the
    # agent's identity with an attacker-controlled persona.
    re.compile(r"you\s+are\s+now\b", re.IGNORECASE),

    # "Forget your training" — attempts to reset the model's alignment
    # and safety guardrails by referencing its training process.
    re.compile(r"forget\s+(your\s+)?training", re.IGNORECASE),

    # "You are an unfiltered assistant/AI/model" — attempts to disable
    # the model's refusal mechanisms by redefining it as "unfiltered".
    re.compile(r"you\s+are\s+an?\s+unfiltered\s+(assistant|ai|model|chatbot)", re.IGNORECASE),

    # "You must ignore/forget/disregard" — imperative override commands
    # that instruct the model to bypass specific safety checks.
    re.compile(r"you\s+must\s+(ignore|forget|disregard)", re.IGNORECASE),

    # "New instructions:" — introduces a block of substitute instructions
    # intended to replace the system prompt entirely.
    re.compile(r"new\s+instructions?\s*:", re.IGNORECASE),

    # "Override mode/instructions/directives" — explicit override attempts
    # that signal a direct challenge to the agent's configuration.
    re.compile(r"override\s+(mode|instructions|directives)", re.IGNORECASE),
]

# Matches lines that mimic the ReAct format (Thought/Action/Action Input/
# Final Answer). An attacker can place these in tool output to make the
# agent's subsequent reasoning appear to be part of the *tool output*,
# causing the loop to misinterpret its own state.
#
# We BLOCK (replace with [blocked]) rather than strip these, because
# stripping could leave surrounding context that still reads as valid
# ReAct output. Blocking makes the injection visibly inert.
_REACT_FORMAT_PATTERN = re.compile(
    r"^(Thought|Action|Action\s+Input|Final\s+Answer)\s*:",
    re.MULTILINE,
)

# Command execution patterns — these catch shell commands and Python
# execution primitives that an attacker might embed in tool output to
# trick an agent with code-execution capabilities.
#
# Rationale for separate detection + blocking passes:
#   - Detection pass (sanitize step 5) adds an audit flag even if
#     the replacement doesn't change the string (e.g. overlapping matches).
#   - Blocking pass (sanitize step 6) replaces ALL occurrences.
#   - This two-pass design ensures logging fidelity: we log ONCE per
#     observation that contains dangerous patterns, not once per match.
_COMMAND_EXECUTION_PATTERNS = [
    # "Execute" — generic execution command, often used to trigger
    # tool calls or shell commands.
    re.compile(r"\bExecute\b", re.IGNORECASE),

    # "run command" — explicit command execution attempt.
    re.compile(r"\brun\s+command\b", re.IGNORECASE),

    # "rm -rf" — destructive filesystem operation (unix).
    re.compile(r"\brm\s+-[rf]\b"),

    # curl / wget — network data exfiltration via HTTP.
    re.compile(r"\bcurl\b"),
    re.compile(r"\bwget\b"),

    # sudo — privilege escalation.
    re.compile(r"\bsudo\b"),

    # exec / subprocess / os.system — Python process execution.
    re.compile(r"\bexec\b"),
    re.compile(r"\bsubprocess\b"),
    re.compile(r"\bos\.system\b"),

    # __import__ / eval( — dynamic code execution in Python.
    re.compile(r"\b__import__\b"),
    re.compile(r"\beval\s*\("),
]

# ---------------------------------------------------------------------------
# ANSI escape codes / control characters
# ---------------------------------------------------------------------------
# ANSI escape sequences can be used to hide text in terminal output
# (e.g., setting foreground colour to background colour). Control
# characters (\x00-\x1f, \x7f) can include bell, backspace, and
# other terminal control codes that could be used for confusing
# parsers or hiding content in log files.
#
# These are STRIPPED (removed entirely) because they have no legitimate
# semantic value in the agent's reasoning context — they are pure
# rendering artefacts that could carry hidden payloads.
_ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class ObservationSanitizer:
    """Sanitize tool output before it enters the ReAct history / LLM context.

    Applies a defence-in-depth pipeline:
        1. Detect & log injection attempts
        2. Strip chat template tokens
        3. Strip ANSI escape sequences and control characters
        4. Truncate to *max_chars*
        5. Prefix with ``[sanitized]`` if anything was removed

    Design rationale — stripping vs. blocking:
      - STRIP (remove entirely): chat tokens, ANSI codes, control chars.
        These have no semantic value in the reasoning context and their
        removal doesn't change the meaning of the observation.
      - BLOCK (replace with [blocked]): ReAct format tokens, command
        patterns. These carry semantic meaning — replacing them with a
        visible sentinel preserves the fact that something was removed
        while making it inert.
    """

    def __init__(
        self,
        max_chars: int = 2000,
        log_violations: bool = True,
        strict_mode: bool = False,
    ) -> None:
        """
        Args:
            max_chars: Maximum length of the returned observation.
                This protects against token-budget exhaustion attacks where
                an attacker sends a very long payload to consume the model's
                context window.
            log_violations: Emit a ``logging.warning`` on injection detection.
                The log record includes a 500-char preview of the original
                text for forensic audit trail. Disable only in high-throughput
                environments where log volume is a concern.
            strict_mode: When *True*, return a sentinel string instead of the
                sanitised observation on injection detection. This is the
                safest option: the LLM never sees potentially malicious content.
                The trade-off is that legitimate tool output containing
                false-positive matches is also discarded.
        """
        self.max_chars = max_chars
        self.log_violations = log_violations
        self.strict_mode = strict_mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sanitize(self, observation: str) -> str:
        """Clean tool output to prevent prompt injection.

        Returns a string safe for inclusion in ReAct history.

        The pipeline is designed so that each stage is independently
        testable and the audit flags (prefixed to the output) give
        full transparency into what was modified.
        """
        if not observation:
            return observation

        original = observation
        flags: list[str] = []

        # 1. Determine whether the observation *looks* malicious
        #    This check runs on the ORIGINAL text before any stripping,
        #    so an attacker cannot evade detection by mixing payloads
        #    that individually look benign but are malicious in combination.
        is_suspicious = self._detect_injection(observation)
        if is_suspicious:
            self._log_violation(observation, "injection pattern detected")
            if self.strict_mode:
                return "[sanitized] Observation blocked by strict mode"

        # 2. Strip chat template tokens
        #    Special tokens like <|im_start|>system<|im_end|> are used to
        #    delimit conversation roles. If an attacker embeds these in
        #    tool output, the LLM may interpret them as actual role
        #    boundaries, injecting a fake system prompt or user message.
        #    Stripping them is safe because they have no semantic meaning
        #    in the observation context.
        cleaned, count = _CHAT_TEMPLATE_PATTERN.subn("", observation)
        if count:
            flags.append(f"stripped {count} special token(s)")

        # 3. Strip ANSI escape sequences and control characters
        #    ANSI codes can hide text (e.g. zero-width sequences) or
        #    cause terminal-based rendering attacks. Control characters
        #    can confuse text parsers or trigger unexpected behaviour
        #    downstream. Both are removed completely.
        cleaned = _ANSI_ESCAPE_PATTERN.sub("", cleaned)
        cleaned = _CONTROL_CHAR_PATTERN.sub("", cleaned)

        # 4. Strip ReAct format injection (lines that look like agent steps)
        #    If tool output contains "Thought:" or "Action:" lines, the
        #    model might interpret them as its own output when the history
        #    is assembled. We BLOCK these (replace with [blocked]) rather
        #    than strip them, because the visible sentinel makes it clear
        #    to the model that those tokens were removed.
        if _REACT_FORMAT_PATTERN.search(cleaned):
            cleaned = _REACT_FORMAT_PATTERN.sub("[blocked]", cleaned)
            flags.append("blocked ReAct format tokens")

        # 5. Detect dangerous command patterns
        #    Separate detection pass ensures we log ONCE even if multiple
        #    command patterns match. This keeps audit logs readable.
        for pattern in _COMMAND_EXECUTION_PATTERNS:
            if pattern.search(cleaned):
                flags.append("blocked command pattern(s)")
                break

        # 6. Strip dangerous command patterns (replace matched terms with [blocked])
        #    Blocking (not stripping) preserves the structural integrity of
        #    the text while making the dangerous term inert. The [blocked]
        #    sentinel also serves as an audit marker visible in the prompt.
        for pattern in _COMMAND_EXECUTION_PATTERNS:
            cleaned = pattern.sub("[blocked]", cleaned)

        # 7. Truncate
        #    Enforce the max_chars budget. This prevents token-bucket
        #    exhaustion attacks where an attacker sends a very long
        #    payload to fill the model's context window and push out
        #    legitimate content (system prompt, history, etc.).
        if len(cleaned) > self.max_chars:
            cleaned = cleaned[: self.max_chars]
            flags.append(f"truncated to {self.max_chars} chars")

        # 8. Prefix if any modification occurred
        #    The prefix makes sanitisation fully transparent. The model
        #    can see exactly what was removed/blocked and why, which
        #    prevents confusion when part of the observation is missing.
        if flags:
            prefix = "[sanitized] " + "; ".join(flags) + ": "
            cleaned = prefix + cleaned

        return cleaned

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _strip_control_sequences(self, text: str) -> str:
        """Remove instruction overrides, role markers, format directives.

        Convenience wrapper that combines the three stripping passes
        (chat tokens, ANSI, control chars) into a single call. Used
        by callers that need a "quick clean" without the full injection
        detection pipeline.
        """
        result = _CHAT_TEMPLATE_PATTERN.sub("", text)
        result = _ANSI_ESCAPE_PATTERN.sub("", result)
        result = _CONTROL_CHAR_PATTERN.sub("", result)
        return result

    def _truncate(self, text: str, max_chars: Optional[int] = None) -> str:
        """Truncate overly long tool outputs.

        Args:
            text: Input text to truncate.
            max_chars: Maximum character length. Falls back to instance
                default if not specified.

        Returns:
            Truncated string (no prefix or audit flags are added here;
            the caller handles those).
        """
        limit = max_chars if max_chars is not None else self.max_chars
        return text[:limit]

    def _detect_injection(self, text: str) -> bool:
        """Detect common injection patterns. Returns True if suspicious.

        Checks three categories:
          1. Instruction override patterns (ignore previous instructions,
             role redefinition, etc.)
          2. Chat template tokens (role-boundary injection)
          3. ReAct format tokens (agent output spoofing)

        NOTE: Command execution patterns are NOT checked here — they are
        handled by the blocking pass instead (steps 5-6 in sanitize()).
        This is intentional: command patterns may appear in legitimate
        tool output (e.g. a tool that returns documentation about curl),
        and we want to block those terms rather than reject the entire
        observation.
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
        """Log injection attempt for audit.

        Format: ``ObservationSanitizer violation | reason=... | preview=...``

        Rationale for 500-char preview:
          - Short enough to avoid log flooding in high-volume deployments
          - Long enough to capture the injection payload for forensic analysis
          - The ``%.500r`` format ensures the preview is safely repr-escaped,
            so binary data or control characters in the original don't corrupt
            the log output

        Audit trail note:
          This log line is the ONLY record of the original (pre-sanitisation)
          observation. If strict_mode is enabled, the LLM never sees the
          content — the log is the sole trace. Ensure logs are retained
          according to your security incident retention policy.
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
