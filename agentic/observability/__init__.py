"""
结构化日志模块 — 用 Python 标准 logging 替代 ts_print()。

功能:
  - 10 个规范日志标签: INIT/AGENT/STEP/LLM/DIFF/TRACE/RESULT/ERROR/WARN/FATAL/HINT
  - 人类可读格式（开发环境）+ JSON 格式（生产环境）
  - 毫秒精度时间戳
  - 按环境变量（REACT_LOG_LEVEL）进行日志级别过滤
  - 标签感知的 LoggerAdapter，确保一致格式化

用法:
  >>> from agentic.observability import get_logger
  >>> logger = get_logger(__name__)
  >>> logger.agent("ReAct 循环开始")
  >>> logger.llm(prompt_tokens=188, completion_tokens=34, duration_ms=3914)
  >>> logger.error("解析失败", extra={"error": str(e)})

日志格式（human）:
  [HH:MM:SS.mmm] [TAG] message

日志格式（json）:
  {"timestamp": "2026-06-21T02:42:31.152", "level": "INFO", "tag": "AGENT",
   "message": "ReAct 循环开始", "module": "llama.ReActDemo", ...}
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List


# ==========================================================================
# 自定义格式化器
# ==========================================================================

class HumanFormatter(logging.Formatter):
    """
    人类可读的日志格式: [HH:MM:SS.mmm] [TAG] message

    功能描述:
      专为终端开发者设计 — 易于浏览，grep 友好。
      输出格式简单明了，每行一条日志，包含时间戳、标签和消息。

    处理逻辑:
      从 LogRecord 中提取或推断标签，使用毫秒精度的时间戳。
    """
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # 毫秒精度
        tag = getattr(record, "tag", record.levelname[:4].upper())
        return f"[{ts}] [{tag}] {record.getMessage()}"


class JSONFormatter(logging.Formatter):
    """
    机器可解析的 JSON 日志格式，适用于生产环境日志聚合。

    功能描述:
      输出结构化的 JSON 格式日志，兼容 ELK Stack、Datadog、Splunk、Grafana Loki。

    处理逻辑:
      1. 构建包含 timestamp、level、tag、message、module 的 JSON 对象
      2. 合并通过 extra 参数传递的额外字段（如 tokens、duration）
      3. 使用 json.dumps 序列化，确保非 ASCII 字符正常显示

    输出说明:
      JSON 格式包含：
        - timestamp: ISO 8601 格式时间戳（UTC）
        - level: 日志级别
        - tag: 日志标签
        - message: 日志消息
        - module: 模块名
        - 可选的额外字段（error, prompt_tokens, completion_tokens 等）
    """
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "tag": getattr(record, "tag", record.levelname),
            "message": record.getMessage(),
            "module": record.name,
        }
        # 合并通过 logger.xxx(msg, extra={...}) 传递的额外字段
        for key in ("error", "prompt_tokens", "completion_tokens", "duration_ms",
                     "step", "thought", "action", "action_input", "observation"):
            val = getattr(record, key, None)
            if val is not None:
                payload[key] = val
        return json.dumps(payload, ensure_ascii=False)


# ==========================================================================
# 标签感知的日志适配器
# ==========================================================================

class TagAdapter(logging.LoggerAdapter):
    """
    向每一条日志记录注入 [TAG] 标签的 LoggerAdapter。

    功能描述:
      提供 10 个规范标签的便捷方法，以及用于 LLM token/时长数据的结构化 kwargs。
      简化日志调用，避免手动设置标签。

    使用方法:
      logger.agent("消息") → [HH:MM:SS.mmm] [AGENT] 消息
      logger.llm(prompt_tokens=100, completion_tokens=50) → [HH:MM:SS.mmm] [LLM] ...
    """
    def __init__(self, logger: logging.Logger):
        super().__init__(logger, {"tag": "INFO"})

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        extra = kwargs.pop("extra", {})
        if self.extra:
            for key, value in self.extra.items():
                if key not in extra:
                    extra[key] = value
        kwargs["extra"] = extra
        return msg, kwargs

    def _log_with_tag(self, level: int, tag: str, msg: str, **kwargs):
        extra = kwargs.pop("extra", {})
        extra["tag"] = tag
        # 将结构化字段转发到日志记录
        for k, v in kwargs.items():
            extra[k] = v
        self.log(level, msg, extra=extra)

    # ── 规范标签方法 ──────────────────────────────────────

    def init(self, msg: str, **kwargs):
        """[INIT] 系统初始化 — 模型加载等。"""
        self._log_with_tag(logging.INFO, "INIT", msg, **kwargs)

    def agent(self, msg: str, **kwargs):
        """[AGENT] Agent 会话生命周期 — 开始/问题。"""
        self._log_with_tag(logging.INFO, "AGENT", msg, **kwargs)

    def step(self, msg: str, **kwargs):
        """[STEP] ReAct 循环迭代边界。"""
        self._log_with_tag(logging.INFO, "STEP", msg, **kwargs)

    def llm(self, msg: str = "", *, prompt_tokens: int = 0, completion_tokens: int = 0,
            duration_ms: float = 0, raw_output: str = "", **kwargs):
        """[LLM] LLM 推理调用 — token 数量 + 耗时。"""
        text = msg or f"prompt={prompt_tokens}  completion={completion_tokens}  duration={duration_ms:.0f}ms"
        if raw_output:
            text = f"原始输出: {raw_output[:200]}\n{text}"
        self._log_with_tag(logging.DEBUG, "LLM", text,
                           prompt_tokens=prompt_tokens,
                           completion_tokens=completion_tokens,
                           duration_ms=duration_ms, **kwargs)

    def diff(self, msg: str, **kwargs):
        """[DIFF] 跨轮数据差异摘要。"""
        self._log_with_tag(logging.DEBUG, "DIFF", msg, **kwargs)

    def trace(self, msg: str, **kwargs):
        """[TRACE] Agent 状态 — Thought/Action/Observation。"""
        self._log_with_tag(logging.INFO, "TRACE", msg, **kwargs)

    def result(self, msg: str, **kwargs):
        """[RESULT] 最终答案或函数返回。"""
        self._log_with_tag(logging.INFO, "RESULT", msg, **kwargs)

    def error(self, msg: str, **kwargs):
        """[ERROR] 可恢复错误（循环内重试）。"""
        self._log_with_tag(logging.WARNING, "ERROR", msg, **kwargs)

    def warn(self, msg: str, **kwargs):
        """[WARN] 超过警告阈值。"""
        self._log_with_tag(logging.WARNING, "WARN", msg, **kwargs)

    def fatal(self, msg: str, **kwargs):
        """[FATAL] 不可恢复错误 — 进程将退出。"""
        self._log_with_tag(logging.CRITICAL, "FATAL", msg, **kwargs)

    def hint(self, msg: str, **kwargs):
        """[HINT] 修复建议 — 与 FATAL 配对使用。"""
        self._log_with_tag(logging.INFO, "HINT", msg, **kwargs)


# ==========================================================================
# 日志工厂函数
# ==========================================================================

def setup_logging(level: str = "INFO", log_format: str = "human",
                  log_file: Optional[str] = None) -> None:
    """
    配置 ReAct Agent 的根日志记录器。

    功能描述:
      初始化日志系统，配置输出目标和格式。
      可同时输出到标准输出和日志文件。

    参数:
      level:     日志级别，可选 DEBUG | INFO | WARNING | ERROR | CRITICAL
      log_format: "human"（终端友好）或 "json"（机器可解析）
      log_file:  可选的日志文件路径。自动创建父目录。
                 如果为 None，仅输出到 stdout。

    处理逻辑:
      1. 获取名为 "agentic" 的根日志记录器
      2. 设置日志级别
      3. 清除已有 handler 确保幂等性
      4. 添加 stdout handler
      5. 如果指定了 log_file，添加文件 handler（自动创建目录）

    注意事项:
      在进程启动时调用一次（main.py 中）。幂等操作，可重复调用。
    """
    root = logging.getLogger("agentic")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 清除已有 handler 以确保幂等性
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)  # Handler 不过滤；由 logger 级别控制

    if log_format == "json":
        formatter = JSONFormatter()
    else:
        formatter = HumanFormatter()

    handler.setFormatter(formatter)
    root.addHandler(handler)

    # ── 可选的文件 handler，用于持久化日志 ─────────────────
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        root.addHandler(fh)


def get_logger(name: str) -> TagAdapter:
    """
    获取为指定模块名包装的 TagAdapter 日志器。

    功能描述:
      创建或获取一个标准 logging.Logger，并包装为 TagAdapter，
      提供便捷的标签化日志方法。

    参数:
      name: 通常是调用模块的 __name__

    返回:
      带有规范标签方法的 TagAdapter 实例。

    用法:
      >>> from agentic.observability import get_logger
      >>> logger = get_logger(__name__)
      >>> logger.agent("ReAct 循环开始")
      >>> logger.llm(prompt_tokens=188, completion_tokens=34, duration_ms=3914.0)
    """
    logger = logging.getLogger(name)
    return TagAdapter(logger)


# ── 从环境变量自动初始化（如果尚未配置） ──

if not logging.getLogger("agentic").handlers:
    _level = os.getenv("REACT_LOG_LEVEL", "INFO")
    _format = os.getenv("REACT_LOG_FORMAT", "human")
    setup_logging(level=_level, log_format=_format)
