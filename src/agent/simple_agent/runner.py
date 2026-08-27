"""In-process ReAct-style agent runner.

Built to sidestep two problems observed with the built-in runners:

* ``plan-execute`` resolves each step's tool arguments with a single,
  isolated LLM call and never sees the tool's actual error — a bad call
  just fails silently into the final summary.
* ``openai-agent`` registers all ~86 tools across every MCP server for
  every run; with a large open-weight model via OpenRouter this was
  observed to break tool-calling entirely (empty completions, zero tool
  calls, on every question that needed one).

This runner takes an explicit, small ``servers`` list per scenario, executes
tools through :class:`mcphub.ToolUniverse` (one persistent MCP session per
server, not one per call), and runs a plain multi-turn OpenAI-compatible
tool-calling loop: every tool result — including errors, MCP-protocol or
JSON-error-body alike — is fed back to the model as an observation, so it
gets a real chance to correct its next call instead of a single unrecoverable
attempt.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import time

from mcphub import ToolUniverse
from observability import agent_run_span, persist_trajectory

from llm.routers import resolve_model, resolve_router_creds
from ._prompts import SIMPLE_AGENT_SYSTEM_PROMPT
from .schema_patches import apply_schema_patches
from .verifiers import verify_before_finalizing, verify_tool_call
from ..models import AgentResult, ToolCall, Trajectory, TurnRecord
from ..runner import AgentRunner

_log = logging.getLogger(__name__)

_DEFAULT_MODEL = "litellm_proxy/openai/gpt-4o-mini"


def _build_openai_tools(
    tool_specs: dict[str, dict],
) -> tuple[list[dict], dict[str, str]]:
    """Convert mcphub tool specs into OpenAI tool defs + a name map.

    Tool names come back from mcphub as ``<server>.<tool>`` (e.g.
    ``iot.sites``); dots aren't safe in every OpenAI-compatible function-name
    validator, so each is sanitized to ``<server>__<tool>`` and the map is
    used to recover the qualified name when the model calls it.
    """
    openai_tools: list[dict] = []
    name_map: dict[str, str] = {}
    for qualified, spec in tool_specs.items():
        sanitized = qualified.replace(".", "__")
        name_map[sanitized] = qualified
        parameters = spec.get("parameters") or {"type": "object", "properties": {}}
        parameters = apply_schema_patches(qualified, parameters)
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": sanitized,
                    "description": spec.get("description", ""),
                    "parameters": parameters,
                },
            }
        )
    return openai_tools, name_map


class SimpleAgentRunner(AgentRunner):
    """Minimal in-process tool-calling loop over a curated MCP server subset.

    Args:
        llm: Unused — accepted for :class:`AgentRunner` interface
             compatibility; this runner talks to the model directly via an
             OpenAI-compatible client resolved from ``model``.
        server_paths: Unused by this runner (mcphub owns server launch);
             accepted for interface compatibility.
        model: Provider-prefixed model id, e.g.
               ``litellm_proxy/meta-llama/llama-4-maverick``.
        servers: MCP servers to register for this run, e.g. ``["iot", "fmsr"]``.
               Keeping this small and scenario-specific is the fix for the
               large-tool-catalog failure seen with ``openai-agent``.
               ``None`` registers every server mcphub knows about.
        max_turns: Maximum model turns before giving up (default 8).
        enable_verifiers: Run in-loop grounding checks (see
               :mod:`.verifiers`) after each tool call and feed a corrective
               observation back to the model when one fires. Default True;
               set False to isolate how much a run's outcome depends on
               this mechanism vs. the model's own behavior.
    """

    def __init__(
        self,
        llm=None,
        server_paths=None,
        model: str = _DEFAULT_MODEL,
        servers: list[str] | None = None,
        max_turns: int = 8,
        enable_verifiers: bool = True,
    ) -> None:
        super().__init__(llm, server_paths)
        self._model_id = model
        self._model = resolve_model(model)
        self._servers = servers
        self._max_turns = max_turns
        self._enable_verifiers = enable_verifiers

    async def run(self, question: str) -> AgentResult:
        with agent_run_span(
            "simple-agent", model=self._model_id, question=question
        ) as span:
            run_started = time.perf_counter()
            started_at = _dt.datetime.now(_dt.timezone.utc).isoformat()

            from openai import AsyncOpenAI

            creds = resolve_router_creds(self._model_id)
            client = AsyncOpenAI(base_url=creds.base_url, api_key=creds.api_key)

            tu = ToolUniverse()
            turns: list[TurnRecord] = []
            answer = ""
            try:
                tu.load_tools(servers=self._servers)
                openai_tools, name_map = _build_openai_tools(tu.all_tools)
                _log.info(
                    "SimpleAgentRunner: %d tool(s) registered from servers=%s",
                    len(openai_tools),
                    self._servers or "all",
                )

                messages: list[dict] = [
                    {"role": "system", "content": SIMPLE_AGENT_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ]
                all_tool_calls: list[ToolCall] = []
                nudged_finalize_categories: set[str] = set()

                for turn_index in range(self._max_turns):
                    turn_started = time.perf_counter()
                    response = await client.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        tools=openai_tools,
                        tool_choice="auto",
                        temperature=0.0,
                    )
                    choice = response.choices[0].message
                    usage = response.usage
                    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

                    assistant_msg: dict = {
                        "role": "assistant",
                        "content": choice.content or "",
                    }
                    if choice.tool_calls:
                        assistant_msg["tool_calls"] = [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in choice.tool_calls
                        ]
                    messages.append(assistant_msg)

                    if not choice.tool_calls:
                        pending_nudges: list[tuple[str, str]] = []
                        if self._enable_verifiers:
                            pending_nudges = [
                                (cat, msg)
                                for cat, msg in verify_before_finalizing(
                                    question, all_tool_calls
                                )
                                if cat not in nudged_finalize_categories
                            ]

                        turns.append(
                            TurnRecord(
                                index=turn_index,
                                text=choice.content or "",
                                tool_calls=[],
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                                duration_ms=(time.perf_counter() - turn_started)
                                * 1000,
                            )
                        )

                        if pending_nudges:
                            nudged_finalize_categories.update(
                                cat for cat, _ in pending_nudges
                            )
                            messages.append(
                                {
                                    "role": "user",
                                    "content": "[Verification] "
                                    + " ".join(msg for _, msg in pending_nudges),
                                }
                            )
                            continue

                        answer = choice.content or ""
                        break

                    tool_calls: list[ToolCall] = []
                    for tc in choice.tool_calls:
                        qualified = name_map.get(tc.function.name, tc.function.name)
                        try:
                            args = json.loads(tc.function.arguments or "{}")
                        except json.JSONDecodeError as exc:
                            args = {}
                            output = {"error": f"invalid JSON arguments: {exc}"}
                        else:
                            try:
                                output = tu.run(
                                    {"name": qualified, "arguments": args}
                                )
                            except Exception as exc:  # noqa: BLE001
                                output = {"error": str(exc)}
                        tool_calls.append(
                            ToolCall(name=qualified, input=args, id=tc.id, output=output)
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": json.dumps(output, default=str),
                            }
                        )

                    all_tool_calls.extend(tool_calls)
                    if self._enable_verifiers:
                        nudges = [
                            nudge
                            for tc_obj in tool_calls
                            if (
                                nudge := verify_tool_call(
                                    tc_obj,
                                    question=question,
                                    prior_tool_calls=all_tool_calls,
                                )
                            )
                        ]
                        if nudges:
                            messages.append(
                                {
                                    "role": "user",
                                    "content": "[Verification] "
                                    + " ".join(nudges),
                                }
                            )

                    turns.append(
                        TurnRecord(
                            index=turn_index,
                            text=choice.content or "",
                            tool_calls=tool_calls,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            duration_ms=(time.perf_counter() - turn_started) * 1000,
                        )
                    )
                else:
                    answer = (
                        f"Max turns ({self._max_turns}) reached without a "
                        "final answer."
                    )
            finally:
                tu.close()

            trajectory = Trajectory(turns=turns, started_at=started_at)
            duration_ms = (time.perf_counter() - run_started) * 1000
            span.set_attribute("agent.answer.length", len(answer))
            span.set_attribute("agent.turns", len(trajectory.turns))
            span.set_attribute("agent.tool_calls", len(trajectory.all_tool_calls))
            span.set_attribute("agent.duration_ms", duration_ms)
            span.set_attribute(
                "gen_ai.usage.input_tokens", trajectory.total_input_tokens
            )
            span.set_attribute(
                "gen_ai.usage.output_tokens", trajectory.total_output_tokens
            )

            persist_trajectory(
                runner_name="simple-agent",
                model=self._model_id,
                question=question,
                answer=answer,
                trajectory=trajectory,
            )
            return AgentResult(question=question, answer=answer, trajectory=trajectory)
