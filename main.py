#!/usr/bin/env python3
"""
ReAct Agent v3.0 — 企业版入口点。

功能描述:
  组装所有企业级模块:
  config → logging → sandbox → parser → template → tools → LLM → agent → tracer → ledger

用法:
  cd /path/to/AgentDev && python main.py
  REACT_MODEL_PATH=/path/to/model.gguf python main.py
"""

import os
import sys
from datetime import datetime

# 确保 AgentDev/ 在 sys.path 中
_PARENT = os.path.dirname(os.path.abspath(__file__))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from agentic.config import get_config
from agentic.observability import setup_logging, get_logger
from agentic.exceptions import FatalError, ModelLoadError
from agentic.protocol.parser import ResponseParser
from agentic.protocol.template import PromptTemplate
from agentic.protocol.format import OutputFormatter
from agentic.tools import ToolRegistry, calculator_tool, weather_tool
from agentic.llm import create_llm
from agentic.agent import AgentCore, TokenBudget, CircuitBreaker, AuditLedger
from agentic.tracer.collector import TraceCollector
from agentic.guard.observation_sanitizer import ObservationSanitizer

# ── 引导 ──────────────────────────────────────────────────────
config = get_config()
if config is None:
    print("[致命错误] 配置失败。请检查 REACT_MODEL_PATH。", file=sys.stderr)
    sys.exit(1)

setup_logging(level=config.log_level, log_format=config.log_format,
              log_file=os.path.join(os.path.dirname(__file__), "agentic", "log", "agent.log"))
logger = get_logger(__name__)

logger.init(f"ReAct Agent v3.0 — 企业版")
logger.init(f"模型={os.path.basename(config.model_path)}  context大小={config.n_ctx}  "
            f"格式={config.chat_format}  日志={config.log_level}/{config.log_format}")


def main():
    # ── 第 1 步: LLM 后端 ────────────────────────────────────────
    try:
        llm = create_llm(config)
    except (FatalError, ModelLoadError) as e:
        logger.fatal(f"LLM 初始化失败: {e}")
        sys.exit(1)

    # ── 第 2 步: 工具注册 ──────────────────────────────────────
    registry = ToolRegistry()
    registry.register(calculator_tool)
    registry.register(weather_tool)
    logger.init(f"已注册工具: {registry.list_names()}")

    # ── 第 3 步: 提示词模板 + 解析器 + 清理器 ────────────────
    yaml_path = os.path.join(os.path.dirname(__file__), "agentic", "protocol", "ReActProtocol.yaml")
    template = PromptTemplate(yaml_path if os.path.exists(yaml_path) else None)
    parser = ResponseParser(yaml_path if os.path.exists(yaml_path) else None)
    sanitizer = ObservationSanitizer()

    # ── 第 4 步: Token 预算 + 熔断器 ──────────────────────
    budget = TokenBudget(max_tokens=config.n_ctx)
    breaker = CircuitBreaker(max_turns=config.max_steps, token_budget=budget)

    # ── 第 5 步: Trace 收集器 + 审计账本 ──────────────────────
    tracer = TraceCollector(
        model_name=llm.model_name,
        pricing=config.price_for("qwen3.6-35b"),
    )
    ledger = AuditLedger()  # 自动: agentic/db/agent_audit.db
    logger.init(f"审计账本: {ledger._db_path}")

    # ── 第 6 步: AgentCore ──────────────────────────────────────────
    agent = AgentCore(
        llm=llm,
        registry=registry,
        template=template,
        parser=parser,
        collector=tracer,
        formatter=OutputFormatter(yaml_path if os.path.exists(yaml_path) else None),
        ledger=ledger,
        max_steps=config.max_steps,
        session_id=f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )

    # ── 第 7 步: 运行 ────────────────────────────────────────
    test_questions = [
        "123 乘以 45 等于多少？",
        "告诉我上海的天气，然后加上 10。",
    ]

    with ledger:
        for q in test_questions:
            logger.agent(f"问题: {q}")
            try:
                # ── 执行前的熔断检查 ──
                breaker.check_before_call(q)
            except Exception as e:
                logger.warn(f"熔断器: {e}")
                break

            result = agent.run(q)

            # ── 在记录/存储之前清理结果 ──
            safe_result = sanitizer.sanitize(result)
            logger.result(safe_result)

    # ── 第 8 步: 报告 ─────────────────────────────────────────────
    tracer.print_summary()

    # 验证审计链完整性
    chain_ok = ledger.verify_chain()
    logger.info(f"审计链完整性: {'OK' if chain_ok else '已损坏'}")

    html_path = os.path.join(config.trace_output_dir, "ReActTrace.html")
    tracer.export_html(html_path)
    logger.info(f"Trace 报告: {html_path}")


if __name__ == "__main__":
    main()
