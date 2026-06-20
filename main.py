#!/usr/bin/env python3
"""
ReAct Agent v3.0 — Enterprise Edition Entry Point.

Wires together all enterprise modules:
  config → logging → sandbox → parser → template → tools → LLM → agent → tracer → ledger

Usage:
  cd /path/to/AgentDev && python main.py
  REACT_MODEL_PATH=/path/to/model.gguf python main.py
"""

import os
import sys

# Ensure AgentDev/ is on sys.path
_PARENT = os.path.dirname(os.path.abspath(__file__))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from agentic.config import get_config
from agentic.logging_config import setup_logging, get_logger
from agentic.exceptions import FatalError, ModelLoadError
from agentic.response_parser import ResponseParser
from agentic.protocol.template import PromptTemplate
from agentic.protocol.format import OutputFormatter
from agentic.tools import ToolRegistry, calculator_tool, weather_tool
from agentic.llm import create_llm
from agentic.agent import AgentCore, TokenBudget, CircuitBreaker, AuditLedger
from agentic.tracer.collector import TraceCollector
from agentic.guard.observation_sanitizer import ObservationSanitizer

# ── Bootstrap ──────────────────────────────────────────────────────
config = get_config()
if config is None:
    print("[FATAL] Configuration failed. Check REACT_MODEL_PATH.", file=sys.stderr)
    sys.exit(1)

setup_logging(level=config.log_level, log_format=config.log_format)
logger = get_logger(__name__)

logger.init(f"ReAct Agent v3.0 — Enterprise Edition")
logger.init(f"Model={os.path.basename(config.model_path)}  ctx={config.n_ctx}  "
            f"format={config.chat_format}  log={config.log_level}/{config.log_format}")


def main():
    # ── Step 1: LLM Backend ────────────────────────────────────────
    try:
        llm = create_llm(config)
    except (FatalError, ModelLoadError) as e:
        logger.fatal(f"LLM init failed: {e}")
        sys.exit(1)

    # ── Step 2: Tool Registry ──────────────────────────────────────
    registry = ToolRegistry()
    registry.register(calculator_tool)
    registry.register(weather_tool)
    logger.init(f"Tools registered: {registry.list_names()}")

    # ── Step 3: Prompt Template + Parser + Sanitizer ────────────────
    yaml_path = os.path.join(os.path.dirname(__file__), "llama", "ReActProtocol.yaml")
    template = PromptTemplate(yaml_path if os.path.exists(yaml_path) else None)
    parser = ResponseParser(yaml_path if os.path.exists(yaml_path) else None)
    sanitizer = ObservationSanitizer()

    # ── Step 4: Token Budget + Circuit Breaker ──────────────────────
    budget = TokenBudget(max_tokens=config.n_ctx)
    breaker = CircuitBreaker(max_turns=config.max_steps, token_budget=budget)

    # ── Step 5: Trace Collector + Audit Ledger ──────────────────────
    tracer = TraceCollector(
        model_name=llm.model_name,
        pricing=config.price_for("qwen3.6-35b"),
    )
    ledger = AuditLedger(db_path=os.path.join(config.trace_output_dir, "agent_audit.db"))
    logger.init(f"Audit ledger: {ledger._db_path}")

    # ── Step 6: AgentCore ──────────────────────────────────────────
    agent = AgentCore(
        llm=llm,
        registry=registry,
        template=template,
        parser=parser,
        collector=tracer,
        formatter=OutputFormatter(yaml_path if os.path.exists(yaml_path) else None),
        max_steps=config.max_steps,
    )

    # ── Step 7: Run ────────────────────────────────────────────────
    test_questions = [
        "What is 123 multiplied by 45?",
        "What is the weather in London multiplied by 2?",
        "Tell me the weather in Tokyo and then add 10 to it.",
    ]

    with ledger:
        for q in test_questions:
            logger.agent(f"Question: {q}")
            try:
                # ── Pre-flight breaker check ──
                breaker.check_before_call(q)
            except Exception as e:
                logger.warn(f"Circuit breaker: {e}")
                break

            result = agent.run(q)

            # ── Sanitize result before logging/storing ──
            safe_result = sanitizer.sanitize(result)
            logger.result(safe_result)

    # ── Step 8: Report ─────────────────────────────────────────────
    tracer.print_summary()

    # Verify audit chain integrity
    chain_ok = ledger.verify_chain()
    logger.info(f"Audit chain integrity: {'OK' if chain_ok else 'CORRUPTED'}")

    html_path = os.path.join(config.trace_output_dir, "ReActTrace.html")
    tracer.export_html(html_path)
    logger.info(f"Trace report: {html_path}")


if __name__ == "__main__":
    main()
