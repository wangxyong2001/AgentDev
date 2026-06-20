"""
企业级配置模块 — 符合 12-Factor App 规范。

设计原则:
  1. 所有配置从环境变量 (os.environ) 获取
  2. 为 Jetson Orin / DGX Spark 边缘设备提供合理默认值
  3. 单一 dataclass 实例（无全局可变状态）
  4. 加载时验证 — 配置错误及早失败

用法:
  >>> from agentic.config import ReActConfig
  >>> cfg = ReActConfig.from_env()
  >>> llm = Llama(model_path=cfg.model_path, n_ctx=cfg.n_ctx, ...)

添加新模型:
  >>> ReActConfig.MODEL_PRICES["deepseek-v3"] = {"input": 0.001, "output": 0.002}
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class ReActConfig:
    """
    不可变的 ReAct Agent 配置数据类。

    功能描述:
      从环境变量 REACT_* 加载所有配置项，提供合理默认值。
      Frozen dataclass — 初始化后不可修改，防止"会话中途配置变更"类 bug。
      BRD §3.2 通过 system_change_count 追踪此类变更。

    参数:
      所有字段通过 factory 从 os.environ 获取，前缀为 REACT_。
      默认值针对 NVIDIA GB10 / Jetson Orin 边缘设备调优。

    处理逻辑:
      1. 每个字段通过 field(default_factory=lambda: os.getenv(...)) 延迟加载
      2. __post_init__ 执行完整验证（范围检查、文件存在性）
      3. from_env() 工厂方法创建实例

    验证项:
      - model_path 必须存在
      - n_ctx 范围 [256, 131072]
      - temperature 范围 [0.0, 2.0]
      - max_steps >= 1
      - log_level 必须是有效级别
      - log_format 必须是 "human" 或 "json"

    异常:
      FileNotFoundError: 模型文件路径不存在
      ValueError: 数值参数超出合法范围
    """

    # ── LLM 模型 ──────────────────────────────────────────────────

    model_path: str = field(default_factory=lambda: os.getenv(
        "REACT_MODEL_PATH",
        "/home/nvidia/llama/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf",
    ))

    chat_format: str = field(default_factory=lambda: os.getenv(
        "REACT_CHAT_FORMAT", "qwen2",
    ))

    # ── 推理参数 ───────────────────────────────────────────────────────

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

    # ── 采样参数 ────────────────────────────────────────────────────────

    temperature: float = field(default_factory=lambda: float(os.getenv(
        "REACT_TEMPERATURE", "0.2",
    )))

    top_p: float = field(default_factory=lambda: float(os.getenv(
        "REACT_TOP_P", "0.95",
    )))

    repeat_penalty: float = field(default_factory=lambda: float(os.getenv(
        "REACT_REPEAT_PENALTY", "1.15",
    )))

    # ── 生成参数 ──────────────────────────────────────────────────────

    max_tokens: int = field(default_factory=lambda: int(os.getenv(
        "REACT_MAX_TOKENS", "512",
    )))

    # ── Agent 参数 ───────────────────────────────────────────────────────

    max_steps: int = field(default_factory=lambda: int(os.getenv(
        "REACT_MAX_STEPS", "8",
    )))

    # ── 日志 ────────────────────────────────────────────────────

    log_level: str = field(default_factory=lambda: os.getenv(
        "REACT_LOG_LEVEL", "INFO",
    ))

    log_format: str = field(default_factory=lambda: os.getenv(
        "REACT_LOG_FORMAT", "human",  # "human" | "json"
    ))

    # ── Trace 输出 ───────────────────────────────────────────────

    trace_output_dir: str = field(default_factory=lambda: os.getenv(
        "REACT_TRACE_OUTPUT_DIR",
        os.path.dirname(os.path.abspath(__file__)),
    ))

    # ── 定价 ────────────────────────────────────────────────────

    MODEL_PRICES: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "qwen3.6-35b":     {"input": 0.004, "output": 0.012},
        "deepseek-v3":     {"input": 0.001, "output": 0.002},
        "glm-4-plus":      {"input": 0.050, "output": 0.100},
    })

    # ── 验证 ─────────────────────────────────────────────────

    def __post_init__(self):
        """
        加载时执行配置验证 — 及早失败原则。

        功能描述:
          在对象初始化完成后立即验证所有配置项。

        验证项:
          - model_path 指向的文件必须存在
          - n_ctx 必须在 [256, 131072] 范围内
          - temperature 必须在 [0.0, 2.0] 范围内
          - max_steps 必须 >= 1
          - log_level 必须是 DEBUG/INFO/WARNING/ERROR/CRITICAL 之一
          - log_format 必须是 "human" 或 "json"

        异常:
          FileNotFoundError: 模型文件路径不存在
          ValueError: 数值参数超出合法范围或日志配置无效
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"模型文件不存在: {self.model_path}\n"
                f"请设置 REACT_MODEL_PATH 为正确的 GGUF 文件路径。"
            )
        if self.n_ctx < 256:
            raise ValueError(f"n_ctx 必须 >= 256，当前值为 {self.n_ctx}")
        if self.n_ctx > 131072:
            raise ValueError(f"n_ctx 必须 <= 131072，当前值为 {self.n_ctx}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature 必须在 0.0-2.0 范围内，当前值为 {self.temperature}")
        if self.max_steps < 1:
            raise ValueError(f"max_steps 必须 >= 1，当前值为 {self.max_steps}")
        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError(f"无效的 log_level: {self.log_level}")
        if self.log_format not in ("human", "json"):
            raise ValueError(f"无效的 log_format: {self.log_format}")

    def price_for(self, model_name: str) -> Tuple[float, float]:
        """
        获取指定模型的（输入价格，输出价格），单位为 元/千 token。

        功能描述:
          从 MODEL_PRICES 字典中查找模型定价。

        参数:
          model_name: MODEL_PRICES 字典中的模型键名

        返回:
          (input_price, output_price) — 如果模型未知，默认返回 Qwen 定价
        """
        default = self.MODEL_PRICES.get("qwen3.6-35b", {"input": 0.004, "output": 0.012})
        prices = self.MODEL_PRICES.get(model_name, default)
        return prices["input"], prices["output"]

    @classmethod
    def from_env(cls) -> "ReActConfig":
        """
        工厂方法 — 从当前环境变量创建配置实例。

        功能描述:
          这是规范的入口点。所有默认值在构造时从 os.environ 解析。
          内部调用 __init__，__post_init__ 自动执行验证。

        返回:
          已验证的 ReActConfig 实例。

        异常:
          FileNotFoundError: model_path 路径不存在
          ValueError: 数值/范围约束被违反
        """
        return cls()


# ── 单例 — 在模块导入时初始化一次 ──────────────────

try:
    _config = ReActConfig.from_env()
except (FileNotFoundError, ValueError) as _e:
    import sys
    print(f"[致命错误] 配置错误: {_e}", file=sys.stderr)
    print("[提示] 请设置 REACT_MODEL_PATH 为有效的 GGUF 文件路径，或检查其他 REACT_* 环境变量。", file=sys.stderr)
    _config = None


def get_config() -> ReActConfig:
    """
    获取全局配置单例。

    功能描述:
      返回模块级单例 _config。该实例在模块导入时从环境变量初始化。
      如果初始化失败，返回 None。

    返回:
      ReActConfig 实例，如果初始化失败则返回 None。

    用法:
      >>> from agentic.config import get_config
      >>> cfg = get_config()
      >>> llm = Llama(model_path=cfg.model_path, n_ctx=cfg.n_ctx)
    """
    return _config
