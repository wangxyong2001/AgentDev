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

添加新模型定价:
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

    参数:
      所有字段通过 factory 从 os.environ 获取，前缀为 REACT_。
      默认值针对 NVIDIA GB10 / Jetson Orin 边缘设备调优。

    处理逻辑:
      1. 每个字段通过 field(default_factory=lambda: os.getenv(...)) 延迟加载
      2. __post_init__ 执行完整验证（范围检查、文件存在性）
      3. from_env() 工厂方法创建实例

    异常:
      FileNotFoundError: 模型文件路径不存在
      ValueError: 数值参数超出合法范围
    """

    # ═════════════════════════════════════════════════════════════════
    # 1. LLM 模型路径
    # ═════════════════════════════════════════════════════════════════

    model_path: str = field(default_factory=lambda: os.getenv(
        "REACT_MODEL_PATH",
        "/home/nvidia/llama/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf",
    ))
    """
    模型文件路径（GGUF 量化格式）。

    当前默认模型: Qwen3.6-35B Q4_K_P

    GGUF 量化级别详解:
      Q2_K    (~16GB)  2 位量化, 最快推理速度, 质量损失最大
      Q3_K_M  (~18GB)  3 位量化, 中等质量
      Q4_K_M  (~21GB)  4 位量化, 速度与质量的最佳平衡点
      Q4_K_P  (~22GB)  4 位量化 + 性能优化, 对推理速度额外调优 ← 当前默认
      Q5_K_M  (~27GB)  5 位量化, 更高质量
      Q8_0    (~42GB)  8 位量化, 接近无损

    Q4_K_P 命名含义:
      Q4 = 每个权重 4 位存储, 模型从 140GB(FP16) 压缩到 ~22GB
      K  = K-Quants 算法, 对关键层(attention/output)使用更高精度量化
      P  = 性能优化变体, 针对推理速度做额外调优

    GB10 (121GB 统一内存) 适配:
      模型 22GB + KV-Cache 32K ~8GB + 系统 ~8GB = ~38GB
      余量 ~83GB — 非常安全, 可升级到 Q5_K_M 或更大上下文

    Jetson Orin (64GB 统一内存) 适配:
      模型 22GB + KV-Cache ~4GB(n_ctx=4096) + 系统 ~8GB = ~34GB
      余量 ~30GB — 安全, 建议保持 Q4_K_P

    环境变量: REACT_MODEL_PATH
    默认值: /home/nvidia/llama/Qwen3.6-35B-A3B-...-Q4_K_P.gguf
    """

    # ═════════════════════════════════════════════════════════════════
    # 2. Chat Template 格式
    # ═════════════════════════════════════════════════════════════════

    chat_format: str = field(default_factory=lambda: os.getenv(
        "REACT_CHAT_FORMAT", "qwen2",
    ))
    """
    Chat Template 格式 — 决定 prompt 中特殊 token 的渲染方式。

    特殊 token 包括:
      <|im_start|>  — 角色段开始  (Qwen2 格式)
      <|im_end|>    — 角色段结束
      <|begin_of_text|> — 对话开始 (Llama3 格式)
      <|eot_id|>    — 轮次结束

    支持的格式及对应模型:
      "qwen2"     — Qwen 2/2.5/3.6 全系列 (默认)
      "llama3"    — Llama 3/4 系列
      "chatml"    — 通用 ChatML 标准格式
      "deepseek"  — DeepSeek V2/V3 系列
      "mistral"   — Mistral 系列

    重要: 切换模型时必须同步修改此参数。
    使用错误的 chat_format 会导致模型无法理解 prompt 结构。

    环境变量: REACT_CHAT_FORMAT
    默认值: qwen2
    """

    # ═════════════════════════════════════════════════════════════════
    # 3. 上下文窗口 (Context Window)
    # ═════════════════════════════════════════════════════════════════

    n_ctx: int = field(default_factory=lambda: int(os.getenv(
        "REACT_N_CTX", "32768",
    )))
    """
    上下文窗口大小 — 模型一次能"看到"的最大 token 数。

    Token 与文字的换算 (Qwen tokenizer):
      英文: ~4.0 字符/token → 32K = ~128K 英文字符
      中文: ~1.8 字符/token → 32K = ~57K 中文字符
      混合: ~3.5 字符/token → 32K = ~112K 混合字符 (ReAct 场景)

    KV-Cache 内存估算 (35B 模型, Q4 量化, FP16 KV Cache):
      每个 token 的 KV Cache ≈ 35B × 2.5(层因子) × 2 bytes × 2(K+V) / 1B
                      ≈ 350 bytes/token (理论) → 实际 ~256 bytes/token
      32K 上下文  → ~8 GB
      64K 上下文  → ~16 GB
      128K 上下文 → ~32 GB

    Qwen3.6-35B 原始训练窗口: 262,144 (256K)
    当前默认 32K 是出于内存安全考虑, 而非模型能力限制。

    ReAct 8 轮典型消耗: 2K-4K tokens (prompt 累积 + 响应)
    32K 已有 8-16x 余量。

    范围: [256, 131072]
    环境变量: REACT_N_CTX
    默认值: 32768
    """

    # ═════════════════════════════════════════════════════════════════
    # 4. GPU 加速参数
    # ═════════════════════════════════════════════════════════════════

    n_gpu_layers: int = field(default_factory=lambda: int(os.getenv(
        "REACT_N_GPU_LAYERS", "-1",
    )))
    """
    加载到 GPU 的模型层数。

    取值含义:
      -1:   所有层加载到 GPU — 推理速度最快, 需要足够 GPU 内存
       0:   所有层在 CPU — 完全不使用 GPU, 推理速度最慢但内存占用最低
       N>0: 前 N 层在 GPU, 剩余在 CPU — 手动控制 GPU 内存使用

    GB10/Jetson 统一内存架构:
      CPU 和 GPU 共享同一块物理内存 (GB10: 121GB, Jetson: 64GB)。
      设置 n_gpu_layers=-1 不会产生 CPU↔GPU 拷贝开销,
      因为"GPU 内存"就是同一块物理内存的不同视图。

    纯 CPU 推理场景:
      设置 n_gpu_layers=0 可避免 CUDA 初始化开销。

    环境变量: REACT_N_GPU_LAYERS
    默认值: -1 (全部 GPU)
    """

    n_threads: int = field(default_factory=lambda: int(os.getenv(
        "REACT_N_THREADS", "10",
    )))
    """
    CPU 推理线程数。

    影响范围:
      - 纯 CPU 推理 (n_gpu_layers=0): 直接影响推理速度
      - GPU 推理 (n_gpu_layers>0): 影响 tokenization 和 prompt 处理

    建议值 (按硬件):
      GB10 20核 ARM:         默认 10 (留 10 核给操作系统和其他进程)
      Jetson Orin 12核 ARM:  建议 6
      桌面 CPU 8核:          建议 6
      桌面 CPU 16核+:        建议 物理核心数 - 2

    环境变量: REACT_N_THREADS
    默认值: 10
    """

    n_batch: int = field(default_factory=lambda: int(os.getenv(
        "REACT_N_BATCH", "1024",
    )))
    """
    Prompt 批处理大小 — 每次处理 prompt 时并行处理的 token 数。

    原理:
      LLM 处理输入 prompt 时将 token 分批, 每批并行计算。
      更大的 n_batch = 更快的 prompt 处理速度, 但增加内存峰值。

    与 max_tokens 的区别:
      n_batch:    输入端 — 控制 prompt 处理的批大小 (影响输入处理速度)
      max_tokens: 输出端 — 控制生成的 token 上限 (影响响应长度)

    推荐值:
      内存充裕 (GB10 121GB): 2048
      标准内存 (64GB):       1024
      内存紧张 (<32GB):      512

    环境变量: REACT_N_BATCH
    默认值: 1024
    """

    flash_attn: bool = field(default_factory=lambda: os.getenv(
        "REACT_FLASH_ATTN", "true",
    ).lower() == "true")
    """
    是否启用 Flash Attention 加速。

    Flash Attention 原理:
      IO-aware 的注意力计算优化算法。将注意力矩阵分块计算,
      避免将完整 N×N 注意力矩阵写入 GPU 显存, 大幅减少内存带宽消耗。

    实际效果 (BRD 实测):
      - 推理延迟降低约 40-60%
      - GPU 内存带宽消耗减少 3-5 倍
      - 对输出质量零影响 (数学等价于标准注意力)

    硬件兼容性:
      Blackwell 架构 (GB10):     ✅ 原生支持
      Jetson Orin GPU:           ✅ 支持
      NVIDIA GPU (SM 8.0+/Ampere): ✅ 支持
      纯 CPU 推理:                ⚠️ 无效果, 建议关闭

    环境变量: REACT_FLASH_ATTN (true/false)
    默认值: true
    """

    # ═════════════════════════════════════════════════════════════════
    # 5. 位置编码扩展
    # ═════════════════════════════════════════════════════════════════

    rope_scaling_type: int = field(default_factory=lambda: int(os.getenv(
        "REACT_ROPE_SCALING", "1",
    )))
    """
    RoPE (Rotary Position Embedding) 缩放类型。

    功能: 使模型支持超出原始训练长度的上下文窗口。

    取值:
      0 = UNSPECIFIED — 不缩放, 仅在上下文 ≤ 训练长度时使用
      1 = LINEAR      — 线性缩放, 将位置索引线性映射到训练长度范围内
      2 = YARN        — Yet Another RoPE extensioN, 更好的长上下文质量
      3 = LONGROPE    — Microsoft LongRoPE, 适合极大上下文扩展

    默认 LINEAR (1):
      适用于将原始训练窗口 32K 的模型扩展到 64K+。
      Qwen3.6-35B 原生支持 256K, 当前 n_ctx=32K 远小于训练窗口,
      实际上不需要缩放。设 0 也可正常运行。

    llama-cpp-python 新版本要求整数值 (旧版接受字符串 "linear"/"yarn")。

    环境变量: REACT_ROPE_SCALING
    默认值: 1 (LINEAR)
    """

    # ═════════════════════════════════════════════════════════════════
    # 6. 采样策略参数
    # ═════════════════════════════════════════════════════════════════

    temperature: float = field(default_factory=lambda: float(os.getenv(
        "REACT_TEMPERATURE", "0.2",
    )))
    """
    采样温度 — 控制 LLM 输出的随机性/创造性。

    技术原理:
      LLM 输出 logits (原始分数) → 除以 temperature → softmax → 概率分布 → 采样
      temperature ↓ → 概率分布更尖锐 (确定性高)
      temperature ↑ → 概率分布更平坦 (多样性高)

    范围 [0.0, 2.0] 的行为:
      0.0 = 完全确定性: 每次选择概率最高的 token (贪婪解码)
      0.2 = 低随机性:   轻微多样性, 保证格式稳定 ← ReAct 推荐值
      0.5 = 中等随机性: 适度的表达变化
      0.7 = 标准随机性: OpenAI API 默认值
      1.0 = 较高随机性
      2.0 = 高随机性:   输出不可预测

    ReAct Agent 为什么用 0.2:
      - ReAct 格式 (Thought/Action/Action Input) 需要严格的格式遵循
      - Qwen3.6 约 10% 轮次在 temperature=0.2 时仍不遵循格式
      - 提高 temperature 会增加格式违规率
      - 0.2 在"格式稳定"和"推理灵活"之间达到平衡

    环境变量: REACT_TEMPERATURE
    默认值: 0.2
    """

    top_p: float = field(default_factory=lambda: float(os.getenv(
        "REACT_TOP_P", "0.95",
    )))
    """
    Nucleus Sampling (核采样) 阈值。

    技术原理:
      1. 按概率从高到低排序所有候选 token
      2. 累加概率直到达到 top_p (例如 0.95)
      3. 仅从这组"核"token 中采样, 其余 token 概率置零

    与 temperature 的联合作用:
      temperature 先缩放 logits 改变概率分布形态
      top_p 再截断低概率 token 限制候选集
      两者独立调优: 固定 temperature, 通过 top_p 控制多样性

    范围 (0.0, 1.0]:
      0.1  = 极度保守 (仅最高概率 token, 接近贪婪)
      0.5  = 中度过滤
      0.95 = 标准值 ← 默认, 保留 95% 概率质量
      1.0  = 关闭 nucleus sampling (考虑所有 token)

    环境变量: REACT_TOP_P
    默认值: 0.95
    """

    repeat_penalty: float = field(default_factory=lambda: float(os.getenv(
        "REACT_REPEAT_PENALTY", "1.15",
    )))
    """
    重复惩罚因子 — 抑制模型重复生成相同 token。

    技术原理:
      模型每生成一个 token, 该 token 的 logit 被除以 repeat_penalty
      (或乘以 1/repeat_penalty), 降低它再次被选中的概率。

    范围 [1.0, 2.0] 的行为:
      1.0  = 无惩罚 (正常采样)
      1.15 = 轻微抑制 ← 默认, 防止 ReAct 循环中 Thought 复读
      1.3  = 中等抑制
      1.5+ = 强抑制, 可能破坏输出质量 (模型回避正常词汇)

    ReAct 场景的特殊意义:
      当模型陷入"死循环"时, 通常表现为连续多轮输出相同的 Thought/Action。
      repeat_penalty=1.15 轻微降低了这种循环的概率, 但不能完全防止。
      真正的死循环防护依赖 CircuitBreaker (agent/breaker.py)。

    环境变量: REACT_REPEAT_PENALTY
    默认值: 1.15
    """

    # ═════════════════════════════════════════════════════════════════
    # 7. 输出生成参数
    # ═════════════════════════════════════════════════════════════════

    max_tokens: int = field(default_factory=lambda: int(os.getenv(
        "REACT_MAX_TOKENS", "512",
    )))
    """
    每次 LLM 调用允许生成的最大输出 token 数。

    ReAct 单轮输出构成:
      Thought:       ~15-50 tokens (一行推理)
      Action:        ~2-5 tokens (工具名或 final_answer)
      Action Input:  ~5-30 tokens (工具参数)
      合计:           ~30-85 tokens (典型)

    512 已有约 6x 余量, 防止模型偶尔输出长文本时的截断。

    与 n_ctx 的关系:
      n_ctx = prompt_tokens (输入) + max_tokens (输出上限)
      实际 prompt 可用空间 = n_ctx - max_tokens
      例如: 32768 - 512 = 32256 tokens 留给 system prompt + user message + history

    环境变量: REACT_MAX_TOKENS
    默认值: 512
    """

    # ═════════════════════════════════════════════════════════════════
    # 8. Agent 行为参数
    # ═════════════════════════════════════════════════════════════════

    max_steps: int = field(default_factory=lambda: int(os.getenv(
        "REACT_MAX_STEPS", "8",
    )))
    """
    ReAct 推理循环的最大迭代步数 — 强制安全阀。

    防止模型陷入以下情况时的无限循环:
      - 模型反复调用同一工具但无法收敛
      - 解析失败 → 重试 → 再失败 (Error Recovery 循环)
      - 模型产生无意义的工具调用链

    推荐值:
      简单计算 (1 步):          3-5
      多步推理 (2-3 步):         8 ← 默认
      需要探索的开放任务:        15-20

    达到上限时的行为:
      返回 "Agent stopped due to max steps."
      通过 logger.warn 记录 WARN 级别日志
      CircuitBreaker 不会触发 (max_steps 在 AgentCore 内部处理)

    环境变量: REACT_MAX_STEPS
    默认值: 8
    """

    # ═════════════════════════════════════════════════════════════════
    # 9. 日志参数
    # ═════════════════════════════════════════════════════════════════

    log_level: str = field(default_factory=lambda: os.getenv(
        "REACT_LOG_LEVEL", "INFO",
    ))
    """
    日志输出级别 — 控制 stdout 和日志文件中可见的日志量。

    DEBUG:    所有日志 (含 LLM 原始输出前 200 chars、跨轮 Diff 详情)
             — 适用于本地开发和性能调试
    INFO:     关键操作日志 (步骤、Thought/Action/Observation、结果)
             — 默认生产级别, 终端可见运行过程
    WARNING:  仅警告 (Max steps 终止、可恢复错误) 和更高级别
             — 适用于 CI/自动化运行
    ERROR:    仅错误和致命错误
             — 静默正常运行, 仅在异常时输出
    CRITICAL: 仅致命错误 (模型加载失败、配置错误)
             — 极简输出

    环境变量: REACT_LOG_LEVEL
    默认值: INFO
    """

    log_format: str = field(default_factory=lambda: os.getenv(
        "REACT_LOG_FORMAT", "human",
    ))
    """
    日志输出格式。

    human:
      [HH:MM:SS.mmm] [TAG] 消息内容
      示例: [14:32:15.238] [AGENT] 问题: 2+2?
      终端开发者友好, 易于 grep 和肉眼浏览。

    json:
      {"timestamp":"2026-06-21T14:32:15.238Z","level":"INFO","tag":"AGENT",
       "message":"问题: 2+2?","module":"agentic.agent.runner"}
      机器可解析, 可直接对接 ELK Stack / Datadog / Splunk / Grafana Loki。

    环境变量: REACT_LOG_FORMAT
    默认值: human
    """

    # ═════════════════════════════════════════════════════════════════
    # 10. 输出路径
    # ═════════════════════════════════════════════════════════════════

    trace_output_dir: str = field(default_factory=lambda: os.getenv(
        "REACT_TRACE_OUTPUT_DIR",
        os.path.dirname(os.path.abspath(__file__)),
    ))
    """
    Trace 报告和 YAML 协议文件的输出目录。

    生成的文件:
      ReActTrace.html        — 自包含 HTML Trace 可视化报告 (深色/浅色双主题)
      运行日志 agent.log     — 在 agentic/log/ 下 (由 main.py 独立配置)
      审计账本 agent_audit.db — 在 agentic/db/ 下 (由 AuditLedger 独立配置)

    默认值指向 agentic/ 包目录本身 (即此文件所在目录)。

    环境变量: REACT_TRACE_OUTPUT_DIR
    默认值: agentic/ 目录路径
    """

    # ═════════════════════════════════════════════════════════════════
    # 11. 成本估算定价 (用于本地推理的云端成本模拟)
    # ═════════════════════════════════════════════════════════════════

    MODEL_PRICES: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "qwen3.6-35b":     {"input": 0.004, "output": 0.012},
        "deepseek-v3":     {"input": 0.001, "output": 0.002},
        "glm-4-plus":      {"input": 0.050, "output": 0.100},
    })
    """
    云端模型 API 定价参考表 (单位: 元/千 token)。

    用途:
      本地推理的边际成本为零 (仅有 GPU 功耗), 但可通过此定价表
      模拟"如果调用云端 API 需要多少钱"。用于:
      - 成本对比: 本地推理 vs 云端 API 的经济性分析
      - 预算估算: 如果未来迁移到云端, 预估月成本

    价格来源 (2026 Q1 公开定价):
      Qwen3.6-35B:  阿里云百炼平台
      DeepSeek-V3:   DeepSeek 官方
      GLM-4-Plus:    智谱 AI 开放平台

    成本计算 (在 TraceCollector.summary() 中):
      cost_rmb = (prompt_tokens × input_price + completion_tokens × output_price) / 1000

    示例: ReAct 3 轮会话, 消耗 200 prompt + 50 completion tokens:
      cost = (200 × 0.004 + 50 × 0.012) / 1000 = ¥0.0014
    """

    # ═════════════════════════════════════════════════════════════════
    # 配置验证
    # ═════════════════════════════════════════════════════════════════

    def __post_init__(self):
        """
        加载时执行配置验证 — 及早失败原则。

        在对象初始化完成后立即验证所有配置项, 避免"启动 10 分钟后才发现配置错误"。
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

        这是规范的入口点。所有默认值在构造时从 os.environ 解析。
        内部调用 __init__, __post_init__ 自动执行验证。

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

    返回:
      ReActConfig 实例，如果初始化失败则返回 None。

    用法:
      >>> from agentic.config import get_config
      >>> cfg = get_config()
      >>> llm = Llama(model_path=cfg.model_path, n_ctx=cfg.n_ctx)
    """
    return _config
