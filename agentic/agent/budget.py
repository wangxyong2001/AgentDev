"""
ReAct Agent 的 Token 预算与断路器。

防止失控的递归循环和无限制的 token 消耗——这是在边缘硬件
（Jetson Orin/GB10）上产生 $16K-$50K 成本事故的主要原因。

TokenBudget:
  预算管理器，在每次 LLM 调用前进行检查。使用 len//4 启发式方法
  进行快速估算，并针对可配置的上限累积消耗。

CircuitBreaker:
  双重限制断路器，结合 turn_count（轮次数）+ token_budget（token 预算）。
  任一种限制超限时抛出 CircuitBreakerError——Agent 必须立即停止。

用法：
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
    """Token 预算管理器——在每次 LLM 调用前进行检查。

    使用快速 len//4 启发式方法估算 token 数量。累积消耗并提供
    警告和剩余信号用于监控。

    参数:
      max_tokens: 单次运行中总 token 消耗的硬上限。
      warning_ratio: 触发警告标志的 max_tokens 比例阈值。
    """

    def __init__(self, max_tokens: int, warning_ratio: float = 0.8):
        self.max_tokens = max_tokens
        self.warning_threshold = int(max_tokens * warning_ratio)
        self.accumulated = 0

    def estimate(self, prompt: str) -> int:
        """根据提示字符串估算 token 数量（len//4 启发式方法）。

        这是一个快速、零依赖的近似估算。生产环境使用 LLM 后端时，
        应替换为后端的 tokenizer。

        参数:
          prompt: 待估算的提示字符串。

        返回值:
          估算的 token 数量。
        """
        return len(prompt) // 4

    def check_and_reserve(self, estimated: int) -> bool:
        """检查估算的 token 是否在预算内。安全时返回 True。

        在一个调用中原子性地检查预算并预留 token。
        这防止了多次检查通过但累积消耗超出 max_tokens 的竞态条件。

        参数:
          estimated: 即将进行的调用所需的估算 token 数。

        返回值:
          True 表示调用在剩余预算内（token 已预留）。
          False 表示会超出预算（未预留任何 token）。
        """
        if self.accumulated + estimated > self.max_tokens:
            return False
        self.accumulated += estimated
        return True

    @property
    def warning(self) -> bool:
        """如果累积 token 达到或超过警告阈值则返回 True。"""
        return self.accumulated >= self.warning_threshold

    @property
    def remaining(self) -> int:
        """返回耗尽 max_tokens 前的剩余 token 数。"""
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
    """断路器跳闸时抛出——Agent 必须立即停止。

    携带人类可读的消息，指示哪个限制被超出
    （轮次数或 token 预算）以及配置的最大值。
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


# ==========================================================================
# CircuitBreaker
# ==========================================================================

class CircuitBreaker:
    """双重限制断路器：turn_count（轮次数）+ token_count（token 数）。

    在每次 LLM 调用前进行检查。防止两类失控情况：

      1. 轮次循环：模型拒绝产生 final_answer 并无限循环。
      2. Token 暴增：每轮的提示词因历史累积而无限制增长。

    两个限制相互独立——任一个先触及时即停止 Agent。

    参数:
      max_turns: 强制停止前的最大 ReAct 轮次数。
      token_budget: TokenBudget 实例，用于 token 感知的限制。
    """

    def __init__(self, max_turns: int, token_budget: TokenBudget):
        self.max_turns = max_turns
        self.token_budget = token_budget
        self.turn_count = 0

    def check_before_call(self, prompt: str):
        """在每次 LLM 调用前调用。超出限制时抛出 CircuitBreakerError。

        增加轮次计数，估算提示词的 token 成本，并检查两项限制。
        这是单个调用点，防止在任何代码路径中遗漏检查。

        参数:
          prompt: 即将发送给 LLM 的提示词。

        异常:
          CircuitBreakerError: 如果超出轮次限制，或本次调用
            将超出 token 预算。
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
        """重置轮次计数和累积 token 数。

        用于在跨独立 Agent 运行间复用 CircuitBreaker 时。
        """
        self.turn_count = 0
        self.token_budget.accumulated = 0

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(max_turns={self.max_turns}, "
            f"turn_count={self.turn_count}, "
            f"token_budget={self.token_budget})"
        )
