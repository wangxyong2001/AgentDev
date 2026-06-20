"""
Agent Core — ReAct reasoning loop as a state machine.

Extracted from the monolithic run_react_agent():

  runner.py  — AgentCore: ReAct state machine with 8 states
  hooks.py   — AgentHooks: Pre/PostToolUse interception points

Phase 3 (future):
  budget.py      — Token Budget management
  reflection.py  — Dead-loop detection via TurnDiff data
  grounding.py   — Conclusion verification against tool outputs
"""

from llama.agent.runner import AgentCore

__all__ = ["AgentCore"]
