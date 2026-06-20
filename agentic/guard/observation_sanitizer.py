"""Observation sanitizer for ReAct agent loop.

Sanitizes tool output before it enters the ReAct history / LLM context,
blocking prompt injection attacks embedded in tool/API responses.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chat template / special tokens
# ---------------------------------------------------------------------------
_CHAT_TEMPLATE_TOKENS = {
    "<|im_start|>", "<|im_end|>",
    "<|begin_of_text|>", "<|eot_id|>",
    "<|system|>", "<|user|>", "<|assistant|>",
    "<s>", "</s>",
}

_CHAT_TEMPLATE_PATTERN = re.compile(
    "|".join(re.escape(t) for t in _CHAT_TEMPLATE_TOKENS),
)

# ---------------------------------------------------------------------------
# Injection detection patterns
# ---------------------------------------------------------------------------
_INSTRUCTION_OVERRIDE_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+(instructions|directions|commands)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\b", re.IGNORECASE),
    re.compile(r"forget\s+(your\s+)?training", re.IGNORECASE),
    re.compile(r"you\s+are\s+an?\s+unfiltered\s+(assistant|ai|model|chatbot)", re.IGNORECASE),
    re.compile(r"you\s+must\s+(ignore|forget|disregard)", re.IGNORECASE),
    re.compile(r"new\s+instructions?\s*:", re.IGNORECASE),
    re.compile(r"override\s+(mode|instructions|directives)", re.IGNORECASE),
]

_REACT_FORMAT_PATTERN = re.compile(
    r"^(Thought|Action|Action\s+Input|Final\s+Answer)\s*:",
    re.MULTILINE,
)

_COMMAND_EXECUTION_PATTERNS = [
    re.compile(r"\bExecute\b", re.IGNORECASE),
    re.compile(r"\brun\s+command\b", re.IGNORECASE),
    re.compile(r"\brm\s+-[rf]\b"),
    re.compile(r"\bcurl\b"),
    re.compile(r"\bwget\b"),
    re.compile(r"\bsudo\b"),
    re.compile(r"\bexec\b"),
    re.compile(r"\bsubprocess\b"),
    re.compile(r"\bos\.system\b"),
    re.compile(r"\b__import__\b"),
    re.compile(r"\beval\s*\("),
]

# ---------------------------------------------------------------------------
# ANSI escape codes / control characters
# ---------------------------------------------------------------------------
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
            log_violations: Emit a ``logging.warning`` on injection detection.
            strict_mode: When *True*, return a sentinel string instead of the
                sanitised observation on injection detection.
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
        """
        if not observation:
            return observation

        original = observation
        flags: list[str] = []

        # 1. Determine whether the observation *looks* malicious
        is_suspicious = self._detect_injection(observation)
        if is_suspicious:
            self._log_violation(observation, "injection pattern detected")
            if self.strict_mode:
                return "[sanitized] Observation blocked by strict mode"

        # 2. Strip chat template tokens
        cleaned, count = _CHAT_TEMPLATE_PATTERN.subn("", observation)
        if count:
            flags.append(f"stripped {count} special token(s)")

        # 3. Strip ANSI escape sequences and control characters
        cleaned = _ANSI_ESCAPE_PATTERN.sub("", cleaned)
        cleaned = _CONTROL_CHAR_PATTERN.sub("", cleaned)

        # 4. Strip ReAct format injection (lines that look like agent steps)
        if _REACT_FORMAT_PATTERN.search(cleaned):
            cleaned = _REACT_FORMAT_PATTERN.sub("[blocked]", cleaned)
            flags.append("blocked ReAct format tokens")

        # 5. Detect dangerous command patterns
        for pattern in _COMMAND_EXECUTION_PATTERNS:
            if pattern.search(cleaned):
                flags.append("blocked command pattern(s)")
                break

        # 6. Strip dangerous command patterns (replace matched terms with [blocked])
        for pattern in _COMMAND_EXECUTION_PATTERNS:
            cleaned = pattern.sub("[blocked]", cleaned)

        # 7. Truncate
        if len(cleaned) > self.max_chars:
            cleaned = cleaned[: self.max_chars]
            flags.append(f"truncated to {self.max_chars} chars")

        # 8. Prefix if any modification occurred
        if flags:
            prefix = "[sanitized] " + "; ".join(flags) + ": "
            cleaned = prefix + cleaned

        return cleaned

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _strip_control_sequences(self, text: str) -> str:
        """Remove instruction overrides, role markers, format directives."""
        result = _CHAT_TEMPLATE_PATTERN.sub("", text)
        result = _ANSI_ESCAPE_PATTERN.sub("", result)
        result = _CONTROL_CHAR_PATTERN.sub("", result)
        return result

    def _truncate(self, text: str, max_chars: Optional[int] = None) -> str:
        """Truncate overly long tool outputs."""
        limit = max_chars if max_chars is not None else self.max_chars
        return text[:limit]

    def _detect_injection(self, text: str) -> bool:
        """Detect common injection patterns. Returns True if suspicious."""
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
        """Log injection attempt for audit."""
        if not self.log_violations:
            return
        # Only log the first 500 chars of the original to avoid log flooding
        preview = original[:500]
        logger.warning(
            "ObservationSanitizer violation | reason=%s | preview=%.500r",
            reason,
            preview,
        )
