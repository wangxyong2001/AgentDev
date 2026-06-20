"""
LLM 后端——抽象接口 + 本地 GGUF 工厂。

将 Agent 核心循环与 LLM 实现解耦。支持：
  - 本地 GGUF（llama-cpp-python）——当前
  - OpenAI 兼容 API（未来）
  - Ollama 本地服务器（未来）

架构：
  ``LLMBackend`` 是一个 ``Protocol`` 类——消费者（AgentCore）
  依赖接口而非具体类。新后端只需实现 ``generate()`` 方法和
  ``model_name`` 属性；无需继承（结构子类型化）。

  ``create_llm()`` 工厂读取配置对象并实例化相应的后端。
  当前仅实现了 ``LocalGGUFBackend``；未来后端将通过
  ``backend_type`` 配置字段选择。

已处理的边缘情况：
  - 模型文件缺失 -> ModelLoadError（在 Llama() 初始化之前）。
  - 模型加载失败（文件损坏、OOM、架构不兼容）-> ModelLoadError
    包装底层异常。
  - Token 计数回退：如果 llama-cpp-python 未返回用量统计，
    token 数估算为 len(text) // 4（粗略近似：英文约 4 字符/token）。

用法：
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
# 数据类型
# ==========================================================================

@dataclass
class LLMResponse:
    """所有后端共用的标准化 LLM 响应。"""

    text: str
    prompt_tokens: int
    completion_tokens: int
    duration_ms: float
    raw_response: Optional[Dict[str, Any]] = None


# ==========================================================================
# 后端接口（Protocol——结构子类型化）
# ==========================================================================

class LLMBackend(Protocol):
    """
    LLM 后端接口（通过 Protocol 实现结构子类型化）。

    任何具有与以下签名匹配的 ``generate()`` 方法和 ``model_name``
    属性的对象都是有效的 ``LLMBackend``——不需要 ``@abstractmethod``
    或继承。这让我们可以使用测试替身（MagicMock、自定义桩）
    而无需共享基类。

    接口契约——实现者必须保证：
      1. ``generate()`` 始终返回 ``LLMResponse``，永不抛出异常
         （错误包装在响应中或记录日志）。
      2. ``generate()`` 尊重 ``stop`` 序列——遇到任何停止字符串
         即终止生成。
      3. ``generate()`` 将输出上限限制在 ``max_tokens``。
      4. ``model_name`` 返回适合日志记录和追踪归属的稳定、
         人类可读的标识符。

    消费者（AgentCore）不得假设：
      - 后端是有状态或无状态的（可能缓存 token）。
      - 后端是线程安全的（需在外部序列化调用）。
      - Token 计数是精确的（某些后端为估算值）。

    用法：
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
        根据提示词生成文本。

        参数:
          prompt:      完整的 ReAct 提示词字符串，包含工具描述、
                       历史记录和用户问题。
          stop:        停止序列（例如 ``["Observation:", "<|im_end|>"]``）。
                       后端必须在生成任何这些字符串时停止生成
                       （它们不包含在输出文本中）。
          max_tokens:  最大生成 token 数。即使模型可以生成更多，
                       后端也必须遵守此上限。
          temperature: 采样温度（0.0 = 贪婪/确定性 argmax）。
                       大于 0 的值引入随机性。

        返回值:
          LLMResponse，包含生成的文本、token 计数和挂钟耗时（毫秒）。
        """
        ...

    @property
    def model_name(self) -> str:
        """人类可读的模型标识符（例如 "qwen2.5-7b-instruct"）。"""
        ...


# ==========================================================================
# 本地 GGUF 后端
# ==========================================================================

class LocalGGUFBackend:
    """
    llama-cpp-python 的本地 GGUF 模型推理后端。

    将 ``llama_cpp.Llama`` 包装为 ``LLMBackend`` 协议。在本地
    机器上运行推理（CPU 或通过 cuBLAS/Metal 使用 GPU）。

    模型加载边缘情况：
      - **文件缺失**：工厂函数（``create_llm``）在调用此构造函数
        前检查文件是否存在，如缺失则抛出 ``ModelLoadError`` 并附清晰消息。
      - **模型损坏**：``Llama.__init__()`` 会抛出异常。我们捕获并
        重新抛出为 ``ModelLoadError``，通过 ``raise ... from e``
        保留原始堆栈跟踪。
      - **GPU OOM**：在显存不足的 GPU 上设置 ``n_gpu_layers`` > 0
        会导致 llama.cpp 在 mmap 或张量分配时抛出 ``Exception``。
        已捕获并包装。
      - **不兼容的 chat_format**：某些 GGUF 模型需要特定的聊天格式
        字符串（例如 ``"chatml"``、``"qwen"``）。不匹配可能产生乱码
        输出或加载错误。
      - **flash_attn**：需要构建支持 flash attention 的
        llama-cpp-python。如果 C++ 级别不支持，则静默忽略
        （``Llama`` 构造函数会吞掉该参数）。

    线程安全性：llama-cpp-python 的 Llama 对象不支持并发生成。
    请序列化 ``generate()`` 调用或在多线程环境中使用锁。
    """

    def __init__(self, model_path: str, n_ctx: int, n_gpu_layers: int,
                 n_threads: int, n_batch: int, chat_format: str,
                 temperature: float, top_p: float, repeat_penalty: float,
                 rope_scaling_type: int, flash_attn: bool):
        # 延迟导入：llama_cpp 是重型原生依赖。在这里而非模块顶部
        # 导入意味着无需它即可导入 agent 的其他部分（对测试、文档
        # 和模型不可用的环境很有用）。
        from llama_cpp import Llama

        self._model_path = model_path
        logger.init(f"Loading model: {model_path}")
        logger.init(f"context={n_ctx}  gpu_layers={n_gpu_layers}  threads={n_threads}")

        try:
            # ── Llama 构造函数 ───────────────────────────────────
            # 此调用：
            #   - 内存映射 GGUF 文件（mmap）。
            #   - 分配 KV 缓存（n_ctx * n_layer * kv_size）。
            #   - 如果适用，将 n_gpu_layers 移至 GPU。
            #   - 从 GGUF 元数据加载 tokenizer 和配置。
            # 构造函数是最可能的失败点
            # （文件损坏、OOM、不支持的架构）。
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
            # 将 llama-cpp-python 的任何异常包装为我们领域特定的
            # ModelLoadError。这防止了抽象泄漏（调用者不需要了解
            # Llama 的异常层次结构）。
            raise ModelLoadError(str(e)) from e

    @property
    def model_name(self) -> str:
        """返回 GGUF 模型的文件名（例如 qwen2.5-7b-instruct.Q4_K_M.gguf）。

        这是一个简洁、人类可读的标识符，用于日志记录和追踪归属。
        完整路径可通过 ``self._model_path`` 获取。
        """
        return os.path.basename(self._model_path)

    def generate(self, prompt: str, stop: List[str],
                 max_tokens: int, temperature: float) -> LLMResponse:
        """使用本地 GGUF 模型生成文本。

        委托给 ``llama_cpp.Llama.__call__()``，该函数通过 llama.cpp
        库在 C++ 中处理分词、推理和停止序列检测。

        参数:
            prompt: 完整的提示词字符串（已由 AgentCore 的 PromptTemplate
                渲染）。必须是原始字符串，而非已分词的。
            stop: 停止序列。Llama.cpp 原生处理这些。
            max_tokens: 最大生成 token 数。
            temperature: 采样温度。0.0 = 贪婪解码。

        返回值:
            LLMResponse，包含生成的文本、token 计数（如果
            llama-cpp-python 未提供则为估算值）和挂钟耗时。

        关于 token 计数回退的说明：
            某些 llama-cpp-python 构建不在响应字典中返回 ``usage``。
            在这种情况下，我们估算：
              prompt_tokens = len(prompt) // 4
              completion_tokens = len(raw_output) // 4
            此近似值（约 4 字符/token）对英文文本合理，但对代码或
            非英文语言可能不准确。这是一个回退方案——精确计数更优。
        """
        t_start = time.time()

        # ── 推理调用 ──────────────────────────────────────────────
        # Llama 对象的 __call__ 方法处理：
        #   - 分词（字符串 -> token ID）
        #   - KV 缓存管理（相同提示前缀时重用）
        #   - 并行解码（batch_size）
        #   - 停止序列检测（在匹配处截断输出）
        #   - 基于温度的采样（top-k、top-p、重复惩罚）
        response = self._llm(
            prompt,
            max_tokens=max_tokens,
            stop=stop,
            temperature=temperature,
        )

        duration_ms = (time.time() - t_start) * 1000
        raw_output = response["choices"][0]["text"]
        usage = response.get("usage", {})

        # ── Token 计数回退 ────────────────────────────────────────
        # llama-cpp-python 的 usage 字典在某些构建中可能缺失
        # （例如没有服务器绑定的情况）。回退到粗略的基于字符的估算。
        return LLMResponse(
            text=raw_output,
            prompt_tokens=usage.get("prompt_tokens", len(prompt) // 4),
            completion_tokens=usage.get("completion_tokens", len(raw_output) // 4),
            duration_ms=duration_ms,
            raw_response=response,
        )


# ==========================================================================
# 工厂函数
# ==========================================================================

def create_llm(config) -> LocalGGUFBackend:
    """
    根据配置创建 LLM 后端实例。

    工厂函数，读取配置对象并将其字段映射到 ``LocalGGUFBackend``
    构造函数参数。这是后端实例化的唯一公共入口——消费者不应直接
    调用 ``LocalGGUFBackend.__init__()``。

    配置参数映射（``config`` -> ``LocalGGUFBackend``）：

        config.model_path       model_path   磁盘上 .gguf 文件路径
        config.n_ctx            n_ctx        上下文窗口大小（token 数）
        config.n_gpu_layers     n_gpu_layers 卸载到 GPU 的层数（0 = CPU）
        config.n_threads        n_threads    评估用的 CPU 线程数
        config.n_batch          n_batch      每批评估的 token 数
        config.chat_format      chat_format  聊天模板（例如 "chatml"）
        config.temperature      temperature  采样温度
        config.top_p            top_p        核采样阈值
        config.repeat_penalty   repeat_penalty 重复惩罚系数
        config.rope_scaling_type rope_scaling_type 长上下文的 RoPE 缩放
        config.flash_attn       flash_attn   是否使用 flash attention

    边缘情况：
        1. **模型文件缺失**：在调用构造函数前显式检查，产生清晰的
           ``ModelLoadError`` 而非 llama-cpp-python 难以理解的
           ``FileNotFoundError``。
        2. **网络文件系统**：``os.path.exists()`` 可能在失效的 NFS
           挂载点上返回成功，但加载时文件不可访问。构造函数将抛出
           ``ModelLoadError``。
        3. **符号链接**：``os.path.exists()`` 跟随符号链接。
           断链符号链接会被检测为文件缺失。
        4. **缺少字段的配置**：如果 ``config`` 是 ``ReActConfig``
           （dataclass），所有字段保证存在。如果意外传入了普通字典，
           属性访问将抛出 ``AttributeError``——这是有意为之的快速失败设计。

    参数:
      config: 配置对象（通常为 ``ReActConfig`` dataclass），
          包含上述所有属性。

    返回值:
      可用于 ``generate()`` 调用的 LocalGGUFBackend 实例。

    异常:
      ModelLoadError: 如果 ``config.model_path`` 不存在，或模型文件
          无法加载（文件损坏、OOM、不兼容的架构）。
    """
    # ── 前置检查 ──────────────────────────────────────────────────
    # 在委托给 Llama 构造函数前检查文件是否存在。
    # 这能在不让原生库初始化的前提下，针对最常见的部署错误
    # （路径错误）给出清晰的错误信息。
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
