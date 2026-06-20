"""
LLM Backend — abstract interface + local GGUF factory.

Decouples Agent core loop from the LLM implementation. Supports:
  - Local GGUF (llama-cpp-python) — current
  - OpenAI-compatible API (future)
  - Ollama local server (future)

Architecture:
  ``LLMBackend`` is a ``Protocol`` class — the consumer (AgentCore)
  depends on the interface, not on a concrete class.  New backends
  only need to implement ``generate()`` and the ``model_name``
  property; no inheritance is required (structural subtyping).

  The ``create_llm()`` factory reads a configuration object and
  instantiates the appropriate backend.  Currently only
  ``LocalGGUFBackend`` is implemented; future backends would be
  selected via a ``backend_type`` config field.

Edge cases handled:
  - Missing model file -> ModelLoadError (before Llama() init).
  - Model load failure (corrupted file, OOM, wrong arch) ->
    ModelLoadError wrapping the underlying exception.
  - Token count fallback: if llama-cpp-python does not return
    usage stats, token counts are estimated as len(text) // 4
    (a rough approximation: ~4 chars per token for English text).

Usage:
  >>> from agentic.llm import create_llm, LLMBackend, LLMResponse
  >>> from agentic.config import get_config
  >>> llm = create_llm(get_config())
  >>> response = llm.generate("What is 2+2?", stop=["Observation:"], max_tokens=512, temperature=0.2)
"""

from __future__ import annotations

import os
import sys
import time
from typing import List, Protocol, Dict, Any, Optional
from dataclasses import dataclass

from agentic.exceptions import ModelLoadError
from agentic.observability import get_logger

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
    LLM backend interface (structural subtyping via Protocol).

    Any object with a ``generate()`` method and a ``model_name``
    property matching these signatures is a valid ``LLMBackend`` —
    no ``@abstractmethod`` or inheritance required.  This lets us
    use test doubles (MagicMock, custom stubs) without a shared
    base class.

    Interface contract — implementers must guarantee:
      1. ``generate()`` always returns an ``LLMResponse``, never
         raises (errors are wrapped in the response or logged).
      2. ``generate()`` respects ``stop`` sequences — it terminates
         generation when any stop string is encountered.
      3. ``generate()`` caps output at ``max_tokens`` tokens.
      4. ``model_name`` returns a stable, human-readable identifier
         suitable for logging and trace attribution.

    Consumers (AgentCore) MUST NOT assume:
      - The backend is stateful or stateless (may cache tokens).
      - The backend is thread-safe (serialize calls externally).
      - Token counts are exact (some backends estimate).

    Usage:
        class MyBackend:
            @property
            def model_name(self) -> str:
                return "my-model"

            def generate(self, prompt, stop, max_tokens, temperature):
                return LLMResponse(text="...", prompt_tokens=0,
                                   completion_tokens=0, duration_ms=0)
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
          prompt:      Full ReAct prompt string, including tool
                       descriptions, history, and the user's question.
          stop:        Stop sequences (e.g. ``["Observation:", "<|im_end|>"]``).
                       The backend MUST halt generation when any of these
                       strings is produced (they are NOT included in the
                       output text).
          max_tokens:  Maximum tokens to generate.  The backend MUST
                       respect this bound even if the model could produce
                       more tokens.
          temperature: Sampling temperature (0.0 = greedy / deterministic
                       argmax).  Values > 0 introduce randomness.

        Returns:
          LLMResponse with generated text, token counts, and
          wall-clock duration in milliseconds.
        """
        ...

    @property
    def model_name(self) -> str:
        """Human-readable model identifier (e.g. "qwen2.5-7b-instruct")."""
        ...


# ==========================================================================
# Local GGUF Backend
# ==========================================================================

class LocalGGUFBackend:
    """
    llama-cpp-python backend for local GGUF model inference.

    Wraps ``llama_cpp.Llama`` with the ``LLMBackend`` protocol.  Runs
    inference on the local machine (CPU or GPU via cuBLAS/Metal).

    Model loading edge cases:
      - **Missing file**: The factory (``create_llm``) checks file
        existence before calling this constructor, raising
        ``ModelLoadError`` with a clear message.
      - **Corrupted model**: ``Llama.__init__()`` will raise.  We
        catch and re-raise as ``ModelLoadError``, preserving the
        original traceback via ``raise ... from e``.
      - **GPU OOM**: ``n_gpu_layers`` > 0 on a GPU with insufficient
        VRAM causes an ``Exception`` from llama.cpp during mmap or
        tensor allocation.  Caught and wrapped.
      - **Incompatible chat_format**: Some GGUF models require a
        specific chat format string (e.g. ``"chatml"``, ``"qwen"``).
        Mismatch may produce garbled output or a load error.
      - **flash_attn**: Requires a build of llama-cpp-python with
        flash attention support.  Silently ignored if unsupported
        at the C++ level (``Llama`` constructor swallows it).

    Thread safety: llama-cpp-python's Llama object is NOT thread-safe
    for concurrent generation.  Serialize ``generate()`` calls or use
    a lock in multi-threaded contexts.
    """

    def __init__(self, model_path: str, n_ctx: int, n_gpu_layers: int,
                 n_threads: int, n_batch: int, chat_format: str,
                 temperature: float, top_p: float, repeat_penalty: float,
                 rope_scaling_type: int, flash_attn: bool):
        # Lazy import: llama_cpp is a heavy native dependency.  Importing
        # it here rather than at the top of the module means the rest of
        # the agent can be imported without it (useful for testing, docs,
        # and environments where the model is not available).
        from llama_cpp import Llama

        self._model_path = model_path
        logger.init(f"Loading model: {model_path}")
        logger.init(f"context={n_ctx}  gpu_layers={n_gpu_layers}  threads={n_threads}")

        try:
            # ── Llama constructor ───────────────────────────────────
            # This call:
            #   - Memory-maps the GGUF file (mmap).
            #   - Allocates the KV cache (n_ctx * n_layer * kv_size).
            #   - Moves n_gpu_layers to GPU if applicable.
            #   - Loads the tokenizer and config from the GGUF metadata.
            # The constructor is the most likely point of failure
            # (corrupted file, OOM, unsupported architecture).
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
            # Wrap ANY exception from llama-cpp-python into our
            # domain-specific ModelLoadError.  This prevents leaky
            # abstraction (callers don't need to know about Llama's
            # exception hierarchy).
            raise ModelLoadError(str(e)) from e

    @property
    def model_name(self) -> str:
        """Return just the filename of the GGUF model (e.g. qwen2.5-7b-instruct.Q4_K_M.gguf).

        This is a concise, human-readable identifier used in logging
        and trace attribution.  The full path is available via
        ``self._model_path`` if needed.
        """
        return os.path.basename(self._model_path)

    def generate(self, prompt: str, stop: List[str],
                 max_tokens: int, temperature: float) -> LLMResponse:
        """Generate text using the local GGUF model.

        Delegates to ``llama_cpp.Llama.__call__()``, which handles
        tokenization, inference, and stop-sequence detection in C++
        via the llama.cpp library.

        Args:
            prompt: Full prompt string (already rendered by AgentCore's
                PromptTemplate).  Must be a raw string, not tokenized.
            stop: Stop sequences.  Llama.cpp handles these natively.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.  0.0 = greedy.

        Returns:
            LLMResponse with the generated text, token counts (or
            estimated counts if llama-cpp-python doesn't provide them),
            and wall-clock duration.

        Notes on token count fallback:
            Some builds of llama-cpp-python do not return ``usage``
            in the response dict.  In that case we estimate:
              prompt_tokens = len(prompt) // 4
              completion_tokens = len(raw_output) // 4
            This approximation (~4 chars/token) is reasonable for
            English text but may be inaccurate for code or non-English
            languages.  It is a fallback — exact counts are preferred.
        """
        t_start = time.time()

        # ── Inference call ──────────────────────────────────────────
        # The Llama object's __call__ method handles:
        #   - Tokenization (string -> token IDs)
        #   - KV cache management (reuse if same prompt prefix)
        #   - Parallel decoding (batch_size)
        #   - Stop sequence detection (truncates output at match)
        #   - Temperature-based sampling (top-k, top-p, repeat penalty)
        response = self._llm(
            prompt,
            max_tokens=max_tokens,
            stop=stop,
            temperature=temperature,
        )

        duration_ms = (time.time() - t_start) * 1000
        raw_output = response["choices"][0]["text"]
        usage = response.get("usage", {})

        # ── Token count fallback ────────────────────────────────────
        # llama-cpp-python's usage dict may be missing in some builds
        # (e.g. without server bindings).  Fall back to crude
        # character-based estimation.
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

    Factory function that reads a configuration object and maps its
    fields to ``LocalGGUFBackend`` constructor parameters.  This is
    the only public entry point for backend instantiation — consumers
    should not call ``LocalGGUFBackend.__init__()`` directly.

    Configuration parameter mapping (``config`` -> ``LocalGGUFBackend``):

        config.model_path       model_path   Path to the .gguf file on disk
        config.n_ctx            n_ctx        Context window size (tokens)
        config.n_gpu_layers     n_gpu_layers GPU layers to offload (0 = CPU)
        config.n_threads        n_threads    CPU threads for evaluation
        config.n_batch          n_batch      Tokens per evaluation batch
        config.chat_format      chat_format  Chat template (e.g. "chatml")
        config.temperature      temperature  Sampling temperature
        config.top_p            top_p        Nucleus sampling threshold
        config.repeat_penalty   repeat_penalty Repetition penalty
        config.rope_scaling_type rope_scaling_type RoPE scaling for long ctx
        config.flash_attn       flash_attn   Use flash attention (bool)

    Edge cases:
        1. **Missing model file**: Checked explicitly BEFORE calling
           the constructor, producing a clear ``ModelLoadError``
           rather than a cryptic ``FileNotFoundError`` from
           llama-cpp-python.
        2. **Network filesystem**: ``os.path.exists()`` may succeed
           on a stale NFS mount but the file is inaccessible at load
           time.  The constructor will raise ``ModelLoadError``.
        3. **Symbolic links**: ``os.path.exists()`` follows symlinks.
           A broken symlink will be caught as a missing file.
        4. **Config with missing fields**: If ``config`` is a
           ``ReActConfig`` (dataclass), all fields are guaranteed
           present.  If a plain dict is accidentally passed, the
           attribute accesses will raise ``AttributeError`` — this
           is intended as a fail-fast design.

    Args:
      config: A configuration object (typically ``ReActConfig``
          dataclass) with the attributes listed above.

    Returns:
      LocalGGUFBackend instance ready for ``generate()`` calls.

    Raises:
      ModelLoadError: If ``config.model_path`` does not exist, or if
          the model file cannot be loaded (corrupted, OOM, incompatible
          architecture).
    """
    # ── Pre-flight check ────────────────────────────────────────────
    # Check file existence before delegating to the Llama constructor.
    # This gives a clean error for the most common deployment mistake
    # (wrong path) without waiting for the native library to initialize.
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
