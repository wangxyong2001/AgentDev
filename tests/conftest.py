"""
Shared pytest fixtures for ReAct Agent tests.

Provides:
  - Environment variable isolation
  - Temporary file system
  - LLM mock response factory
  - Logging capture and reset
"""

import os
import sys
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import MagicMock, patch

import pytest

# Ensure llama package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ==========================================================================
# Environment isolation
# ==========================================================================

@pytest.fixture(autouse=True)
def clean_env():
    """Isolate environment variables — prevent test contamination."""
    saved = os.environ.copy()
    # Wipe REACT_ vars to force defaults
    for key in list(os.environ.keys()):
        if key.startswith("REACT_"):
            del os.environ[key]
    yield
    # Restore
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging state between tests."""
    logger = logging.getLogger("llama")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    yield
    logger.handlers.clear()
    logger.setLevel(logging.INFO)


# ==========================================================================
# Temporary files
# ==========================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory that self-cleans."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def temp_model_file(temp_dir):
    """Create a minimal placeholder GGUF file for config validation."""
    model = temp_dir / "test-model.gguf"
    model.write_text("fake-gguf-content")
    return model


# ==========================================================================
# LLM Mock
# ==========================================================================

@pytest.fixture
def mock_llm_response() -> Dict[str, Any]:
    """Standard mock LLM response compatible with llama-cpp-python."""
    return {
        "choices": [{"text": "I need to use calculator.\nAction: calculator\nAction Input: 123 * 45"}],
        "usage": {"prompt_tokens": 188, "completion_tokens": 34},
    }


@pytest.fixture
def mock_llm():
    """Create a MagicMock that quacks like llama_cpp.Llama."""
    llm = MagicMock()
    llm.return_value = {
        "choices": [{"text": "Thought: test\nAction: calculator\nAction Input: 2+2"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }
    return llm


# ==========================================================================
# Sample data
# ==========================================================================

@pytest.fixture
def sample_tool_names():
    """Standard tool name list used in tests."""
    return ["calculator", "get_weather"]


@pytest.fixture
def sample_prompt():
    """A complete Qwen2-format ReAct prompt for testing decompose."""
    return (
        "<|im_start|>system\n"
        "You are a ReAct agent. Use tools: calculator, get_weather.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "What is 2+2?\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "Thought: I need to calculate 2+2.\n"
        "Action: calculator\n"
        "Action Input: 2+2\n"
        "Observation: 4\n"
        "Thought:"
    )


@pytest.fixture
def sample_prompt_next():
    """A follow-up prompt (next turn) for testing diff."""
    return (
        "<|im_start|>system\n"
        "You are a ReAct agent. Use tools: calculator, get_weather.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Based on observation, what next?\nObservation: 4\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "Thought: I need to calculate 2+2.\n"
        "Action: calculator\n"
        "Action Input: 2+2\n"
        "Observation: 4\n"
        "Thought: I now have the answer.\n"
        "Action: final_answer\n"
        "Action Input: 4\n"
        "Observation: (final answer)\n"
        "Thought:"
    )
