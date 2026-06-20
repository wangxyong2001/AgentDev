"""
ReAct Agent — Enterprise Edition (v3.0).

Package structure (25 modules, 0 loose .py files):
  config/             — 12-Factor app configuration (env vars + frozen dataclass)
  exceptions/         — Exception hierarchy (RecoverableError vs FatalError)
  observability/      — Structured logging (TagAdapter + JSON/Human formatter)
  protocol/
    template.py       — PromptTemplate (XML 10-section)
    format.py         — OutputFormatter
    parser.py         — Unified ResponseParser (eliminates dual-path rot)
    ReActProtocol.yaml — YAML protocol template
  tools/
    registry.py       — ToolRegistry (dependency injection)
    calculator.py     — Math tool (L1 subprocess sandbox)
    weather.py        — Weather tool
    sandbox.py        — L1 process isolation (RLIMIT_AS/CPU/NPROC)
  llm/                — LLMBackend Protocol + LocalGGUFBackend + factory
  tracer/
    diff.py           — Pure-function cross-turn diff algorithm
    collector.py      — TraceCollector + session summary
    renderer.py       — HTML trace report (self-contained, dark/light theme)
  agent/
    runner.py         — AgentCore (8-state ReAct state machine)
    budget.py         — TokenBudget + CircuitBreaker
    ledger.py         — AuditLedger (SHA-256 hash chain, SQLite)
  guard/
    observation_sanitizer.py — Prompt injection defense (5 attack classes)
  ReActDemo           — Legacy entry point (standalone, backward compatible)

Usage:
  python main.py                        # Enterprise entry point
  cd agentic && python ReActDemo        # Legacy entry point
"""

__version__ = "3.0.0"
