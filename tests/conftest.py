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
    """
    测试场景：确保每个测试在干净的环境变量中运行，避免 REACT_* 环境变量跨测试污染
    参数：无（autouse=True 自动应用于所有测试）
    测试逻辑：(1) 保存当前环境变量快照 (2) 删除所有 REACT_ 前缀的环境变量 (3) yield 执行测试 (4) 恢复原始环境变量
    预期结果：测试执行期间 REACT_* 环境变量不存在，测试完成后环境恢复原状
    成功条件：测试结束后 os.environ 与测试前完全一致
    """
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
    """
    测试场景：确保每个测试的日志状态独立，防止 handler 累积导致重复输出
    参数：无（autouse=True 自动应用于所有测试）
    测试逻辑：(1) 获取 "llama" logger (2) 清除所有 handler (3) 重置日志级别为 INFO (4) yield 执行测试 (5) 再次清除 handler
    预期结果：每个测试开始时 logger 处于干净状态，无残留 handler
    成功条件：测试后 logger.handlers 为空列表，logger.level == logging.INFO
    """
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
    """
    测试场景：需要一个临时目录来创建测试文件，测试结束后自动清理
    参数：无
    测试逻辑：使用 tempfile.TemporaryDirectory 创建临时目录，yield 返回 Path 对象，测试结束后自动删除
    预期结果：返回一个可用的临时目录路径，测试结束后目录被删除
    成功条件：目录存在且可写入，with 块退出后目录不存在
    """
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def temp_model_file(temp_dir):
    """
    测试场景：需要一个占位 GGUF 模型文件来通过配置验证（ReActConfig 要求 model_path 存在）
    参数：依赖 temp_dir fixture 提供临时目录
    测试逻辑：(1) 在临时目录中创建 test-model.gguf (2) 写入伪内容 (3) 返回文件路径
    预期结果：返回一个存在的 .gguf 文件路径，可用于 ReActConfig 验证
    成功条件：返回的 Path 对象指向一个存在的文件，内容为 "fake-gguf-content"
    """
    model = temp_dir / "test-model.gguf"
    model.write_text("fake-gguf-content")
    return model


# ==========================================================================
# LLM Mock
# ==========================================================================

@pytest.fixture
def mock_llm_response() -> Dict[str, Any]:
    """
    测试场景：模拟 llama-cpp-python 的 LLM 调用返回值，用于测试响应解析
    参数：无
    测试逻辑：返回一个标准字典，结构与 llama_cpp.Llama.create_completion() 返回值一致
    预期结果：返回包含 choices[0].text 和 usage 的字典
    成功条件：字典包含 "choices" 和 "usage" 键，choices[0]["text"] 包含完整的 ReAct 格式输出
    """
    return {
        "choices": [{"text": "I need to use calculator.\nAction: calculator\nAction Input: 123 * 45"}],
        "usage": {"prompt_tokens": 188, "completion_tokens": 34},
    }


@pytest.fixture
def mock_llm():
    """
    测试场景：需要一个 MagicMock 对象来替代真实的 llama_cpp.Llama 实例
    参数：无
    测试逻辑：创建 MagicMock，设置 return_value 为包含 Thought/Action/Action Input 的标准 ReAct 输出
    预期结果：调用 mock_llm() 返回模拟的 LLM 响应
    成功条件：mock_llm.return_value["choices"][0]["text"] 包含 "Thought: test"，mock_llm() 被调用后可验证 call_count
    """
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
    """
    测试场景：提供标准工具名列表，用于测试工具注册、解析和 prompt 构建
    参数：无
    测试逻辑：直接返回包含 "calculator" 和 "get_weather" 的列表
    预期结果：返回包含两个字符串的列表
    成功条件：返回值 == ["calculator", "get_weather"]，len == 2
    """
    return ["calculator", "get_weather"]


@pytest.fixture
def sample_prompt():
    """
    测试场景：提供一个完整的 Qwen2 格式 ReAct prompt，用于测试 prompt 分解 (decompose) 算法
    参数：无
    测试逻辑：拼接 system / user / assistant 三个 Qwen2 chat template 段落，包含一轮完整的 Thought→Action→Observation
    预期结果：返回包含 <|im_start|>system/user/assistant 标记的完整 prompt 字符串
    成功条件：返回值包含 "<|im_start|>system"、"<|im_start|>user"、"<|im_start|>assistant" 三个标记
    """
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
    """
    测试场景：提供第二轮对话的 Qwen2 格式 prompt，用于测试跨轮 diff 算法
    参数：无
    测试逻辑：在第一轮基础上追加新的 assistant 回复（包含 final_answer），history 比 sample_prompt 更长
    预期结果：返回一个比 sample_prompt 内容更多的 prompt，可用于验证 common_prefix 计算
    成功条件：返回值包含 "final_answer" 关键字，且长度大于 sample_prompt
    """
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