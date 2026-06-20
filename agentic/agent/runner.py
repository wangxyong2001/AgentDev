"""
AgentCore — ReAct reasoning loop as a state machine.

Replaces the 170-line monolithic run_react_agent() with a class-based
state machine that's independently testable.

States (8):
  IDLE → PROMPT_BUILD → LLM_CALL → PARSE → DISPATCH →
    FINAL_ANSWER | TOOL_EXEC → HISTORY_APPEND → NEXT_ITERATION

Error recovery: PARSE failure → ERROR_RECOVERY → PROMPT_BUILD
Termination:    FINAL_ANSWER | MAX_STEPS

Usage:
  >>> from agentic.agent import AgentCore
  >>> agent = AgentCore(llm=backend, registry=tools, template=prompt_tpl, parser=parser)
  >>> result = agent.run("What is 2+2?")
  "4"
"""

from __future__ import annotations

import time
from typing import List, Dict, Optional, Protocol

from agentic.exceptions import ParseError, ToolNotFoundError, ToolExecutionError
from agentic.observability import get_logger

logger = get_logger(__name__)


# ==========================================================================
# AgentCore
# ==========================================================================

class AgentCore:
    """
    ReAct Agent state machine.

    Orchestrates: prompt → LLM → parse → dispatch → history
    With error recovery for parse failures and missing tools.

    Dependencies are injected — no global state.
    """

    def __init__(
        self,
        llm,                    # LLMBackend (Protocol)
        registry,               # ToolRegistry
        template,               # PromptTemplate
        parser,                 # ResponseParser
        collector=None,         # TraceCollector (optional)
        formatter=None,         # OutputFormatter (optional)
        max_steps: int = 8,
    ):
        self._llm = llm
        self._registry = registry
        self._template = template
        self._parser = parser
        self._collector = collector
        self._formatter = formatter
        self.max_steps = max_steps

    # ── Public API ───────────────────────────────────────────────────

    def run(self, question: str) -> str:
        """
        Execute a complete ReAct reasoning loop.

        Args:
          question: Natural language question

        Returns:
          Final answer string, or "Agent stopped due to max steps."

        Raises:
          Only fatal errors — recoverable errors are handled in-loop.
        """
        logger.agent("=== ReAct loop start ===")
        logger.agent(f"Question: {question}")

        history: List[str] = []
        current_input = question
        tool_names = self._registry.list_names()
        tool_descs = {t.name: t.description for t in self._registry.list_tools()}

        for step in range(1, self.max_steps + 1):
            logger.step(f"--- Step {step} ---")
            t_start = time.time()

            # ── State: PROMPT_BUILD ──
            history_str = "\n".join(history) if history else ""
            prompt = self._template.render_full_prompt(
                tool_names, tool_descs, current_input, history_str,
            )

            # ── State: LLM_CALL ──
            response = self._llm.generate(
                prompt=prompt,
                stop=self._template.stop_sequences,
                max_tokens=self._template.max_tokens,
                temperature=self._template.temperature,
            )

            logger.debug(f"Raw: {response.text[:200]}")
            logger.llm(
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                duration_ms=response.duration_ms,
            )

            # ── State: PARSE ──
            parse_error = None
            try:
                parsed = self._parser.parse(response.text, tool_names)
            except ParseError as e:
                logger.error(f"Parse failed: {e}")
                parse_error = str(e)
                # ── ERROR RECOVERY ──
                if self._collector:
                    self._collector.record_turn(
                        turn=step, question=question, prompt=prompt,
                        raw_output=response.text,
                        cleaned_output=self._parser.preprocess(response.text),
                        thought="(Parse Error)", action="none", action_input="none",
                        observation=response.text,
                        prompt_tokens=response.prompt_tokens,
                        completion_tokens=response.completion_tokens,
                        duration_ms=response.duration_ms,
                        parse_error=parse_error, status="parse_error",
                    )
                history.append(
                    f"Thought: (Parse Error)\n"
                    f"Action: none\n"
                    f"Action Input: none\n"
                    f"Observation: {response.text}"
                )
                current_input = (
                    f"Based on the observation, try again.\n"
                    f"Observation: {response.text}"
                )
                continue

            thought = parsed["thought"]
            action = parsed["action"]
            action_input = parsed["action_input"]

            logger.trace(f"Thought: {thought}")
            logger.trace(f"Action: {action}")
            logger.trace(f"Input: {action_input}")

            # ── State: DISPATCH → FINAL_ANSWER ──
            if action == "final_answer":
                logger.result(f"Final Answer: {action_input}")
                if self._collector:
                    self._collector.record_turn(
                        turn=step, question=question, prompt=prompt,
                        raw_output=response.text,
                        cleaned_output=self._parser.preprocess(response.text),
                        thought=thought, action=action, action_input=action_input,
                        observation="(final answer)",
                        prompt_tokens=response.prompt_tokens,
                        completion_tokens=response.completion_tokens,
                        duration_ms=response.duration_ms, status="success",
                    )
                return action_input

            # ── State: TOOL_EXEC ──
            tool_error = False
            try:
                observation = self._registry.execute(action, action_input)
            except ToolNotFoundError:
                observation = (
                    f"Unknown action: '{action}'. "
                    f"Available: {tool_names}"
                )
                tool_error = True
            except ToolExecutionError as e:
                observation = f"Tool execution error: {e.original_error}"
                tool_error = True

            logger.trace(f"Observation: {observation}")

            # ── Trace ──
            if self._collector:
                self._collector.record_turn(
                    turn=step, question=question, prompt=prompt,
                    raw_output=response.text,
                    cleaned_output=self._parser.preprocess(response.text),
                    thought=thought, action=action, action_input=action_input,
                    observation=observation,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    duration_ms=response.duration_ms,
                    status="tool_error" if tool_error else "success",
                )

            # ── State: HISTORY_APPEND ──
            history.append(self._template.render_history_entry(
                thought=thought, action=action,
                action_input=action_input, observation=observation,
            ))

            # ── State: NEXT_ITERATION ──
            current_input = (
                f"Based on the observation, what should I do next?\n"
                f"Observation: {observation}"
            )

        # ── Terminal state: MAX_STEPS ──
        logger.warn(f"Max steps ({self.max_steps}) reached. Loop terminated.")
        return "Agent stopped due to max steps."
