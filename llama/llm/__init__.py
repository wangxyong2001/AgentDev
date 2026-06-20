"""
LLM Backend — abstract interface + local GGUF factory.

Decouples Agent core loop from the LLM implementation. Supports:
  - Local GGUF (llama-cpp-python) — current
  - OpenAI-compatible API (future)
  - Ollama local server (future)

Usage:
  >>> from llama.llm import create_llm, LLMBackend, LLMResponse
  >>> from llama.config import get_config
  >>> llm = create_llm(get_config())
  >>> response = llm.generate("What is 2+2?", stop=["Observation:"], max_tokens=512, temperature=0.2)
"""

from __future__ import annotations

import os
import sys
import time
from typing import List, Protocol, Dict, Any, Optional
from dataclasses import dataclass

from llama.exceptions import ModelLoadError
from llama.logging_config import get_logger

logger = get_logger(__name__)


# ==========================================================================
# Data types
# ==========================================================================

@dataclass
class LLMResponse:
    """Standardized LLM response across all backends."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    duration_ms: float
    raw_response: Optional[Dict[str, Any]] = None


# ==========================================================================
# Backend interface (Protocol — structural subtyping)
# ==========================================================================

class LLMBackend(Protocol):
    """
    LLM backend interface.

    Any backend implementing this protocol can be used with AgentCore.
    No inheritance required — structural subtyping via Protocol.
    """

    def generate(
        self,
        prompt: str,
        stop: List[str],
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """
        Generate text from a prompt.

        Args:
          prompt:      Full ReAct prompt string
          stop:        Stop sequences (e.g. ["Observation:", "<|im_end|>"])
          max_tokens:  Maximum tokens to generate
          temperature: Sampling temperature (0.0 = greedy)

        Returns:
          LLMResponse with text, token counts, and duration
        """
        ...

    @property
    def model_name(self) -> str:
        """Human-readable model identifier."""
        ...


# ==========================================================================
# Local GGUF Backend
# ==========================================================================

class LocalGGUFBackend:
    """
    llama-cpp-python backend for local GGUF model inference.

    Wraps llama_cpp.Llama with the LLMBackend protocol.
    """

    def __init__(self, model_path: str, n_ctx: int, n_gpu_layers: int,
                 n_threads: int, n_batch: int, chat_format: str,
                 temperature: float, top_p: float, repeat_penalty: float,
                 rope_scaling_type: int, flash_attn: bool):
        from llama_cpp import Llama

        self._model_path = model_path
        logger.init(f"Loading model: {model_path}")
        logger.init(f"context={n_ctx}  gpu_layers={n_gpu_layers}  threads={n_threads}")

        try:
            self._llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                n_batch=n_batch,
                verbose=False,
                chat_format=chat_format,
                temp=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                rope_scaling_type=rope_scaling_type,
                flash_attn=flash_attn,
            )
            logger.init("Model loaded successfully.")
        except Exception as e:
            raise ModelLoadError(str(e)) from e

    @property
    def model_name(self) -> str:
        return os.path.basename(self._model_path)

    def generate(self, prompt: str, stop: List[str],
                 max_tokens: int, temperature: float) -> LLMResponse:
        t_start = time.time()

        response = self._llm(
            prompt,
            max_tokens=max_tokens,
            stop=stop,
            temperature=temperature,
        )

        duration_ms = (time.time() - t_start) * 1000
        raw_output = response["choices"][0]["text"]
        usage = response.get("usage", {})

        return LLMResponse(
            text=raw_output,
            prompt_tokens=usage.get("prompt_tokens", len(prompt) // 4),
            completion_tokens=usage.get("completion_tokens", len(raw_output) // 4),
            duration_ms=duration_ms,
            raw_response=response,
        )


# ==========================================================================
# Factory
# ==========================================================================

def create_llm(config) -> LocalGGUFBackend:
    """
    Create an LLM backend from configuration.

    Args:
      config: ReActConfig instance

    Returns:
      LocalGGUFBackend instance

    Raises:
      ModelLoadError: On model file missing or load failure
      FileNotFoundError: If model file doesn't exist
    """
    if not os.path.exists(config.model_path):
        raise ModelLoadError(f"Model file not found: {config.model_path}")

    return LocalGGUFBackend(
        model_path=config.model_path,
        n_ctx=config.n_ctx,
        n_gpu_layers=config.n_gpu_layers,
        n_threads=config.n_threads,
        n_batch=config.n_batch,
        chat_format=config.chat_format,
        temperature=config.temperature,
        top_p=config.top_p,
        repeat_penalty=config.repeat_penalty,
        rope_scaling_type=config.rope_scaling_type,
        flash_attn=config.flash_attn,
    )
