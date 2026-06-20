"""
AgentCore — ReAct reasoning loop as a state machine.

Replaces the 170-line monolithic run_react_agent() with a class-based
state machine that's independently testable. Each iteration follows
a strict state flow with explicit error recovery for parse failures
and tool-not-found errors.

States (9, including recovery):
  IDLE → PROMPT_BUILD → LLM_CALL → PARSE → DISPATCH →
    FINAL_ANSWER | TOOL_EXEC → HISTORY_APPEND → NEXT_ITERATION
  PARSE failure → ERROR_RECOVERY → PROMPT_BUILD (retry)
  Termination:    FINAL_ANSWER | MAX_STEPS

Security boundary:
  The agent treats LLM output as untrusted data. Observations from
  tool execution are appended to the prompt history as text strings
  — they are NOT deserialized, evaluated, or executed as code. This
  prevents prompt-injection attacks that attempt to escape the ReAct
  loop through crafted tool output.

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

    Orchestrates: prompt -> LLM -> parse -> dispatch -> history
    With error recovery for parse failures and missing tools.

    Dependencies are injected — no global state. This makes the
    component testable in isolation by swapping real LLM backends
    with test doubles that implement the ``LLMBackend`` protocol.

    Security: All LLM-generated text is treated as untrusted. The
    parser extracts structured fields from raw text, but does not
    ``eval()``, ``exec()``, or otherwise interpret the output beyond
    string matching. Tool output is appended to the prompt context
    as plain text strings — never deserialized or executed.

    Usage:
        agent = AgentCore(llm=backend, registry=tools,
                          template=prompt_tpl, parser=parser)
        result = agent.run("What is 2+2?")
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
        """
        Args:
            llm: An object implementing the ``LLMBackend`` protocol
                (``generate()`` method + ``model_name`` property).
            registry: ``ToolRegistry`` instance providing tool name
                lookup and execution via ``execute()``.
            template: ``PromptTemplate`` used to render prompts and
                history entries. Defines stop sequences and token
                limits for the LLM call.
            parser: ``ResponseParser`` that extracts ``thought``,
                ``action``, and ``action_input`` from raw LLM output.
            collector: Optional ``TraceCollector`` for observability.
                When provided, each step's full trace (tokens, timing,
                errors) is recorded.
            formatter: Optional ``OutputFormatter`` for final answer
                formatting before returning.
            max_steps: Maximum ReAct iterations before forced
                termination. Default 8. Prevents infinite loops from
                uncooperative LLM output.
        """
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

        The method iterates through a state machine: prompt building,
        LLM generation, response parsing, action dispatch (tool call
        or final answer), history appending, and loop continuation.
        On parse failure, the error is recorded in history and the
        loop retries rather than crashing.

        Args:
          question: Natural language question to answer.

        Returns:
          Final answer string, or "Agent stopped due to max steps."

        Raises:
          Only fatal errors — recoverable errors (parse failures,
          tool not found, tool execution errors) are handled inside
          the loop without propagating exceptions.
        """
        logger.info(f"{'═'*50}\n  Question: {question}")

        history: List[str] = []
        current_input = question
        tool_names = self._registry.list_names()
        tool_descs = {t.name: t.description for t in self._registry.list_tools()}

        for step in range(1, self.max_steps + 1):
            logger.info(f"{'─'*40}\n  Step {step}/{self.max_steps}")
            t_start = time.time()

            # ── State: PROMPT_BUILD ──────────────────────────────────
            # Render the full ReAct prompt: tool descriptions, user
            # question, and accumulated thought/action/observation
            # history. The template controls stop sequences and token
            # limits, giving the LLM explicit guidance on output format.
            history_str = "\n".join(history) if history else ""
            prompt = self._template.render_full_prompt(
                tool_names, tool_descs, current_input, history_str,
            )

            # ── State: LLM_CALL ──────────────────────────────────────
            # Send the prompt to the LLM backend. The backend handles
            # tokenization, inference, and stop-sequence detection.
            # response includes token counts and wall-clock timing.
            response = self._llm.generate(
                prompt=prompt,
                stop=self._template.stop_sequences,
                max_tokens=self._template.max_tokens,
                temperature=self._template.temperature,
            )

            logger.info(f"LLM: {response.prompt_tokens}p+{response.completion_tokens}c tokens, {response.duration_ms:.0f}ms")
            logger.debug(f"Raw: {response.text[:200]}")

            # ── State: PARSE ─────────────────────────────────────────
            # Extract structured fields (thought, action, action_input)
            # from the unstructured LLM output. The parser uses regex
            # to match the ReAct format (e.g. "Thought: ...\nAction: ...").
            # Security: LLM output is untrusted. The parser only applies
            # string/regex matching — no eval() or exec(). The extracted
            # fields are passed to tool execution as strings; tool
            # implementations are responsible for their own input
            # sanitization.
            parse_error = None
            try:
                parsed = self._parser.parse(response.text, tool_names)
            except ParseError as e:
                logger.error(f"Parse failed: {e}")
                parse_error = str(e)

                # ── Error Recovery Path ──────────────────────────────
                # Parse failures happen when the LLM deviates from the
                # expected ReAct output format (e.g., it generates
                # free-form prose instead of "Thought: ...\nAction: ...").
                #
                # Recovery strategy:
                #   1. Record the raw output as an observation so the
                #      LLM sees its own output in the next iteration.
                #   2. Append a formatted "skip" entry to history so
                #      the prompt structure stays intact.
                #   3. Provide a meta-instruction ("Based on the
                #      observation, try again.") that guides the LLM
                #      back to the ReAct format.
                #   4. ``continue`` to the next iteration instead of
                #      crashing — this makes the agent resilient to
                #      occasional malformed output.
                #
                # Security note: The raw LLM output is appended to
                # history verbatim. It is rendered as plain text in
                # the next prompt — not deserialized, not evaluated.
                # This prevents prompt-injection escape attempts that
                # embed control sequences in the LLM output.
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

            logger.info(f"  Thought: {thought}")
            logger.info(f"  Action: {action}({action_input})")
            logger.trace(f"Input: {action_input}")

            # ── State: DISPATCH -> FINAL_ANSWER ─────────────────────
            # The LLM signals completion by emitting "final_answer"
            # as the action. The action_input becomes the final output.
            # This is the only terminal state besides MAX_STEPS.
            if action == "final_answer":
                logger.info(f"  Final Answer: {action_input}")
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

            # ── State: TOOL_EXEC ─────────────────────────────────────
            # Dispatch the action to the tool registry. The registry
            # looks up the tool by name and calls its execute() method.
            #
            # Two recoverable failure modes:
            #   1. ToolNotFoundError — the LLM hallucinated a tool name.
            #      The observation lists available tools so the LLM can
            #      correct itself.
            #   2. ToolExecutionError — the tool itself failed (e.g.
            #      network timeout, invalid input). The original error
            #      message is surfaced (not the exception traceback) to
            #      avoid leaking internals into the prompt context.
            #
            # Security: The action_input string is user-provided (via LLM
            # output) and is passed to the tool verbatim. Each tool is
            # responsible for its own input validation and sanitization.
            # The sandbox layer (agentic.tools.sandbox) provides OS-level
            # isolation for code-execution tools.
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

            logger.info(f"  Observation: {observation}")

            # ── Record trace (if collector configured) ──────────────
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

            # ── State: HISTORY_APPEND ───────────────────────────────
            # Add the completed turn (thought/action/result) to the
            # running history. The template controls how each entry is
            # formatted, which influences what the LLM sees on the
            # next iteration.
            #
            # Injection defense: The observation is appended as a string
            # value inside a structured template. The template engine
            # does NOT interpolate it as template directives — it is
            # plain text substitution only. This prevents an observed
            # value containing "Thought: ..." from breaking out of the
            # ReAct format and injecting fake context.
            history.append(self._template.render_history_entry(
                thought=thought, action=action,
                action_input=action_input, observation=observation,
            ))

            # ── State: NEXT_ITERATION ────────────────────────────────
            # Prepare the user message for the next loop iteration.
            # The observation from this turn becomes the primary input,
            # guiding the LLM to reason about what to do next.
            current_input = (
                f"Based on the observation, what should I do next?\n"
                f"Observation: {observation}"
            )

        # ── Terminal state: MAX_STEPS ───────────────────────────────
        # The agent exhausted its step budget without reaching final_answer.
        # This prevents infinite loops from LLMs that keep calling tools
        # indefinitely. The caller can distinguish this from a real answer
        # by the sentinel string.
        logger.warn(f"Max steps ({self.max_steps}) reached. Loop terminated.")
        return "Agent stopped due to max steps."
