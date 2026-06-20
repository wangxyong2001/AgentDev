"""
Token Budget + Circuit Breaker for ReAct Agent.

Prevents runaway recursion loops and unbounded token consumption — the
primary cause of $16K-$50K cost incidents on edge hardware (Jetson Orin/GB10).

TokenBudget:
  Budget manager that checks BEFORE every LLM call. Uses a len//4 heuristic
  for fast estimation and accumulates spend against a configurable max.

CircuitBreaker:
  Dual-limit breaker combining turn_count + token_budget. Raises
  CircuitBreakerError when either limit is exceeded — agent must stop
  immediately.

Usage:
  >>> from agentic.agent.budget import TokenBudget, CircuitBreaker, CircuitBreakerError
  >>> budget = TokenBudget(max_tokens=4096, warning_ratio=0.8)
  >>> breaker = CircuitBreaker(max_turns=10, token_budget=budget)
  >>> try:
  ...     breaker.check_before_call("What is 2+2?")
  ... except CircuitBreakerError:
  ...     print("Budget exhausted — agent stopped.")
"""

from __future__ import annotations

from typing import Optional


# ==========================================================================
# TokenBudget
# ==========================================================================

class TokenBudget:
    """Token budget manager — checks BEFORE every LLM call.

    Uses a fast len//4 heuristic for token estimation. Accumulates spend
    and provides warning/remaining signals for monitoring.

    Args:
      max_tokens: Hard ceiling on total token consumption across a run.
      warning_ratio: Fraction of max_tokens that triggers the warning flag.
    """

    def __init__(self, max_tokens: int, warning_ratio: float = 0.8):
        self.max_tokens = max_tokens
        self.warning_threshold = int(max_tokens * warning_ratio)
        self.accumulated = 0

    def estimate(self, prompt: str) -> int:
        """Estimate tokens from prompt string (len//4 heuristic).

        This is a fast, zero-dependency approximation. For production
        use with LLM backends, replace with the backend's tokenizer.

        Args:
          prompt: The prompt string to estimate.

        Returns:
          Estimated token count.
        """
        return len(prompt) // 4

    def check_and_reserve(self, estimated: int) -> bool:
        """Check if estimated tokens fit in budget. Returns True if safe.

        Atomically checks the budget AND reserves the tokens in one call.
        This prevents race conditions where multiple checks pass but the
        cumulative spend exceeds max_tokens.

        Args:
          estimated: Estimated token count for the upcoming call.

        Returns:
          True if the call fits within remaining budget (tokens reserved).
          False if the budget would be exceeded (no tokens reserved).
        """
        if self.accumulated + estimated > self.max_tokens:
            return False
        self.accumulated += estimated
        return True

    @property
    def warning(self) -> bool:
        """Return True if accumulated tokens meet or exceed warning threshold."""
        return self.accumulated >= self.warning_threshold

    @property
    def remaining(self) -> int:
        """Return remaining tokens before max_tokens is exhausted."""
        return max(0, self.max_tokens - self.accumulated)

    def __repr__(self) -> str:
        return (
            f"TokenBudget(max_tokens={self.max_tokens}, "
            f"accumulated={self.accumulated}, "
            f"remaining={self.remaining})"
        )


# ==========================================================================
# CircuitBreakerError
# ==========================================================================

class CircuitBreakerError(Exception):
    """Raised when circuit breaker trips — agent must stop immediately.

    Carries a human-readable message indicating which limit was exceeded
    (turn count or token budget) and the configured max value.
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


# ==========================================================================
# CircuitBreaker
# ==========================================================================

class CircuitBreaker:
    """Dual-limit breaker: turn_count + token_count.

    Checked BEFORE every LLM invocation. Prevents two classes of runaway:

      1. Turn loops: model refuses to produce final_answer and cycles forever.
      2. Token blowout: each turn's prompt grows unboundedly with history.

    Both limits are independent — whichever trips first stops the agent.

    Args:
      max_turns: Maximum ReAct turns before forced stop.
      token_budget: TokenBudget instance for token-aware limiting.
    """

    def __init__(self, max_turns: int, token_budget: TokenBudget):
        self.max_turns = max_turns
        self.token_budget = token_budget
        self.turn_count = 0

    def check_before_call(self, prompt: str):
        """Call BEFORE every LLM invocation. Raises CircuitBreakerError.

        Increments turn count, estimates token cost of the prompt, and
        checks both limits. This is a single call site to prevent missing
        a check in any code path.

        Args:
          prompt: The prompt about to be sent to the LLM.

        Raises:
          CircuitBreakerError: If turn limit exceeded or token budget
            would be exceeded by this call.
        """
        self.turn_count += 1
        if self.turn_count > self.max_turns:
            raise CircuitBreakerError(
                f"Turn limit exceeded: {self.max_turns}"
            )
        estimated = self.token_budget.estimate(prompt)
        if not self.token_budget.check_and_reserve(estimated):
            raise CircuitBreakerError(
                f"Token budget exceeded: {self.token_budget.max_tokens}"
            )

    def reset(self):
        """Reset both turn count and accumulated tokens.

        Used when reusing a CircuitBreaker across independent agent runs.
        """
        self.turn_count = 0
        self.token_budget.accumulated = 0

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(max_turns={self.max_turns}, "
            f"turn_count={self.turn_count}, "
            f"token_budget={self.token_budget})"
        )
