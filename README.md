# Enterprise-Grade Agent Practice

> **Agent = Model + Harness** — 基于行业调研的工程化实践

[![Tests](https://img.shields.io/badge/tests-116%20passed-brightgreen)](tests/)
[![Version](https://img.shields.io/badge/version-3.0.0-blue)](pyproject.toml)
[![Python](https://img.shields.io/badge/python-3.12+-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green)](pyproject.toml)

## What is this?

A **fully offline ReAct (Reasoning + Acting) Agent** running on NVIDIA Jetson Orin / DGX Spark edge devices. The agent loads a 22GB quantized Qwen3.6-35B GGUF model locally, calls tools (calculator, weather), and reasons through multi-step problems — all without network connectivity.

```
User: "What's the weather in Shanghai multiplied by 2?"
  │
  ├─ Thought: Need Shanghai weather first
  │  Action: get_weather("Shanghai")  →  Observation: "Sunny, 25°C"
  │
  ├─ Thought: Now multiply by 2
  │  Action: calculator("25 * 2")  →  Observation: 50
  │
  └─ Final Answer: 50
```

## Why this exists

This project is the engineering output of a systematic industry research effort covering **LangChain 1.0, OpenAI Agents SDK, and Anthropic Claude Agent** ecosystems. Every architecture decision is backed by evidence — production incidents, academic papers, CVE databases, and source code analysis.

📚 **[Full Industry Research Report](doc/INDUSTRY_RESEARCH.md)** — 14 chapters, 1658 lines, 40+ cited sources

📋 **[Architecture Review Meeting Minutes](doc/MEETING_MINUTES_2026-06-21.md)** — 9 consensus items from 5 roles

📎 **[Consensus Evidence Traceability](doc/CONSENSUS_EVIDENCE.md)** — Every consensus mapped to verifiable evidence

## Architecture

```
agentic/
├── config.py              # 12-Factor config (frozen dataclass + env vars)
├── exceptions.py          # Exception hierarchy (Recoverable vs Fatal)
├── logging_config.py      # Structured logging (10 tags + JSON/human dual format)
├── response_parser.py     # Unified parser (eliminates 3-path regex rot)
├── tools/
│   ├── registry.py        # ToolRegistry with dependency injection
│   ├── calculator.py      # Math tool — L1 subprocess sandbox
│   ├── weather.py         # Weather tool — simulated data
│   └── sandbox.py         # L1 process isolation (RLIMIT_AS/CPU/NPROC)
├── llm/
│   └── __init__.py        # LLMBackend Protocol + LocalGGUFBackend
├── protocol/
│   ├── template.py        # XML 10-section system prompt
│   └── format.py          # Output formatter
├── tracer/
│   ├── diff.py            # Pure-function cross-turn diff algorithm
│   └── collector.py       # TraceCollector + session summary
├── agent/
│   ├── runner.py          # AgentCore — 8-state ReAct state machine
│   ├── budget.py          # TokenBudget + CircuitBreaker (pre-flight checks)
│   └── ledger.py          # SQLite AuditLedger (SHA-256 hash chain)
├── guard/
│   └── observation_sanitizer.py  # Prompt injection defense (5 attack classes)
└── eval/                  # Evaluation framework (Phase 2)
```

## Security Levels

| Component | Level | Mechanism |
|-----------|-------|-----------|
| `calculator` | **L1** | Subprocess + RLIMIT_AS/CPU/NPROC + tmpfs |
| `get_weather` | N/A | Pure data lookup, no code execution |
| Observation sanitizer | — | 5-class injection detection + audit logging |
| Audit ledger | SHA-256 chain | Append-only, tamper-evident, daily Merkle root |

```
L0 eval("__builtins__":{})  →  L1 subprocess sandbox
  Known escape: ().__class__.__bases__  →  Blast radius: 128MB/30s/no-fork
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure (or use defaults)
export REACT_MODEL_PATH=/path/to/your/model.gguf
export REACT_N_CTX=32768
export REACT_LOG_FORMAT=json   # "human" for development

# Run
cd llama && python3 ReActDemo
```

## Test Status

```bash
pytest tests/ -q
# 116 passed, 1 skipped in 0.16s
```

| Module | Cases | Status |
|--------|-------|--------|
| config | 19 | ✅ |
| exceptions | 15 | ✅ |
| logging | 20 | ✅ |
| response_parser | 14 | ✅ |
| tracer | 22 | ✅ |
| tools/registry | 17 | ✅ |

## Maturity Roadmap

| Phase | Target Score | Key Deliverables |
|-------|-------------|-----------------|
| **Current** | 2.7/10 | Modular foundation (16 modules) |
| **Phase 2** (in progress) | 4.9/10 | L1 sandbox ✅ · TokenBudget ✅ · XML prompt ✅ · Observation sanitizer ✅ · SQLite Ledger ✅ · EvalScorer |
| **Phase 3** (planned) | 7.0/10 | Reflection · Grounding · Guardrails · Hooks · OTel-lite · Docker |

**H0→H2 Maturity**: Script → Framework → Platform (targeting H2)

## Key Design Decisions

| Decision | Evidence |
|----------|----------|
| L1 sandbox over eval() | PEP 551 rejected, `().__class__.__bases__` escape vector |
| Token Budget + Circuit Breaker | $16K-$50K recursion incidents (2025) |
| Observation sanitizer | 36% public Skills contain prompt injection (Snyk 2026) |
| XML 10-section system prompt | Anthropic guidance, Blackmon Lab attention anchor experiments |
| SQLite hash-chain ledger | SOC2 CC7.2, GDPR Art.32, HIPAA compliance baseline |
| Six-dimension trajectory eval | Compound error math: 0.95⁸ ≈ 66% end-to-end |

## Industry Alignment

| Paradigm | Status |
|----------|--------|
| OpenAI Runner pattern | ✅ AgentCore (injectable LLM + tools + parser) |
| Anthropic Explore/Plan/Code modes | ⏳ Phase 3 |
| LangGraph Token Budget + Reflection | ⏳ Phase 3 |
| Harness-Sandbox separation | ✅ L1 subprocess isolation |
| Agent Eval (6-dim trajectory scoring) | ⏳ Phase 2 (EvalScorer planned) |
| Production Circuit Breaker + Ledger | ✅ `agent/budget.py` + `agent/ledger.py` |
| SOC2/ISO27001 audit readiness | ✅ Hash-chain ledger + compliance_tags |

## Contributing

This project follows the architecture decisions documented in:
- [Industry Research Report](doc/INDUSTRY_RESEARCH.md)
- [Meeting Minutes](doc/MEETING_MINUTES_2026-06-21.md)
- [Consensus Evidence](doc/CONSENSUS_EVIDENCE.md)

Before contributing, review the 9 architecture consensus items. PRs should include tests (target: maintain >100 passing).

---

> **"An agent is not a model. Evaluating one as if it were is the most common reason production agents fail."** — FutureAGI (2026)
