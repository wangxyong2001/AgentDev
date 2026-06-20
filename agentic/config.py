"""
Enterprise Configuration — 12-Factor App compliant.

Design principles:
  1. All config sourced from environment variables (os.environ)
  2. Sensible defaults for Jetson Orin / DGX Spark edge devices
  3. Single dataclass instance (no global mutable state)
  4. Validation on load — fail fast on misconfiguration

Usage:
  >>> from agentic.config import ReActConfig
  >>> cfg = ReActConfig.from_env()
  >>> llm = Llama(model_path=cfg.model_path, n_ctx=cfg.n_ctx, ...)

Adding a new model:
  >>> ReActConfig.MODEL_PRICES["deepseek-v3"] = {"input": 0.001, "output": 0.002}
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class ReActConfig:
    """
    Immutable configuration for ReAct Agent.

    All fields have defaults tuned for NVIDIA GB10 / Jetson Orin edge devices.
    Override via environment variables (prefixed with REACT_).

    Frozen dataclass — once created, config cannot be mutated. This prevents
    the "config changed mid-session" bug that BRD §3.2 tracks via
    system_change_count.
    """

    # ── LLM Model ──────────────────────────────────────────────────

    model_path: str = field(default_factory=lambda: os.getenv(
        "REACT_MODEL_PATH",
        "/home/nvidia/llama/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf",
    ))

    chat_format: str = field(default_factory=lambda: os.getenv(
        "REACT_CHAT_FORMAT", "qwen2",
    ))

    # ── Inference Parameters ───────────────────────────────────────

    n_ctx: int = field(default_factory=lambda: int(os.getenv(
        "REACT_N_CTX", "32768",
    )))

    n_gpu_layers: int = field(default_factory=lambda: int(os.getenv(
        "REACT_N_GPU_LAYERS", "-1",
    )))

    n_threads: int = field(default_factory=lambda: int(os.getenv(
        "REACT_N_THREADS", "10",
    )))

    n_batch: int = field(default_factory=lambda: int(os.getenv(
        "REACT_N_BATCH", "1024",
    )))

    flash_attn: bool = field(default_factory=lambda: os.getenv(
        "REACT_FLASH_ATTN", "true",
    ).lower() == "true")

    rope_scaling_type: int = field(default_factory=lambda: int(os.getenv(
        "REACT_ROPE_SCALING", "1",
    )))  # 1=LINEAR, 2=YARN

    # ── Sampling Parameters ────────────────────────────────────────

    temperature: float = field(default_factory=lambda: float(os.getenv(
        "REACT_TEMPERATURE", "0.2",
    )))

    top_p: float = field(default_factory=lambda: float(os.getenv(
        "REACT_TOP_P", "0.95",
    )))

    repeat_penalty: float = field(default_factory=lambda: float(os.getenv(
        "REACT_REPEAT_PENALTY", "1.15",
    )))

    # ── Generation Parameters ──────────────────────────────────────

    max_tokens: int = field(default_factory=lambda: int(os.getenv(
        "REACT_MAX_TOKENS", "512",
    )))

    # ── Agent Parameters ───────────────────────────────────────────

    max_steps: int = field(default_factory=lambda: int(os.getenv(
        "REACT_MAX_STEPS", "8",
    )))

    # ── Logging ────────────────────────────────────────────────────

    log_level: str = field(default_factory=lambda: os.getenv(
        "REACT_LOG_LEVEL", "INFO",
    ))

    log_format: str = field(default_factory=lambda: os.getenv(
        "REACT_LOG_FORMAT", "human",  # "human" | "json"
    ))

    # ── Trace Output ───────────────────────────────────────────────

    trace_output_dir: str = field(default_factory=lambda: os.getenv(
        "REACT_TRACE_OUTPUT_DIR",
        os.path.dirname(os.path.abspath(__file__)),
    ))

    # ── Pricing ────────────────────────────────────────────────────

    MODEL_PRICES: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "qwen3.6-35b":     {"input": 0.004, "output": 0.012},
        "deepseek-v3":     {"input": 0.001, "output": 0.002},
        "glm-4-plus":      {"input": 0.050, "output": 0.100},
    })

    # ── Validation ─────────────────────────────────────────────────

    def __post_init__(self):
        """Validate configuration on load — fail fast principle."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                f"Set REACT_MODEL_PATH to the correct GGUF file path."
            )
        if self.n_ctx < 256:
            raise ValueError(f"n_ctx must be >= 256, got {self.n_ctx}")
        if self.n_ctx > 131072:
            raise ValueError(f"n_ctx must be <= 131072, got {self.n_ctx}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be 0.0-2.0, got {self.temperature}")
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError(f"Invalid log_level: {self.log_level}")
        if self.log_format not in ("human", "json"):
            raise ValueError(f"Invalid log_format: {self.log_format}")

    def price_for(self, model_name: str) -> Tuple[float, float]:
        """
        Get (input_price, output_price) in ¥/thousand tokens for a model.

        Args:
          model_name: model key in MODEL_PRICES dict

        Returns:
          (input_price, output_price) — defaults to Qwen pricing if unknown
        """
        default = self.MODEL_PRICES.get("qwen3.6-35b", {"input": 0.004, "output": 0.012})
        prices = self.MODEL_PRICES.get(model_name, default)
        return prices["input"], prices["output"]

    @classmethod
    def from_env(cls) -> "ReActConfig":
        """
        Factory method — create config from current environment.

        This is the canonical entry point. All defaults are resolved
        from os.environ at construction time.

        Returns:
          Validated ReActConfig instance.

        Raises:
          FileNotFoundError: model_path doesn't exist
          ValueError: any numeric/range constraint violated
        """
        return cls()


# ── Singleton — initialized once at module import ──────────────────

try:
    _config = ReActConfig.from_env()
except (FileNotFoundError, ValueError) as _e:
    import sys
    print(f"[FATAL] Configuration error: {_e}", file=sys.stderr)
    print("[HINT] Set REACT_MODEL_PATH to a valid GGUF file, or check other REACT_* env vars.", file=sys.stderr)
    _config = None


def get_config() -> ReActConfig:
    """
    Get the global configuration singleton.

    Returns:
      ReActConfig instance, or None if initialization failed.

    Usage:
      >>> from agentic.config import get_config
      >>> cfg = get_config()
      >>> llm = Llama(model_path=cfg.model_path, n_ctx=cfg.n_ctx)
    """
    return _config