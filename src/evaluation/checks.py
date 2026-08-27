"""Deterministic, no-LLM trust checks over agent trajectories.

Two families, chosen for signal that's cheap and unambiguous to compute
without a model in the loop:

* **Trajectory integrity** — did every tool call actually succeed (no
  silently-swallowed error hiding behind a plausible-sounding final
  answer), and for scenarios that declare which servers they expect —
  a structured ``servers`` list where one exists, else keyword-matched
  out of AssetOpsBench's free-text ``hint`` field (e.g. "IoT Agent
  handles ingestion; TSFM Agent detects anomalies...") — were those
  servers actually invoked at all.
* **Schema/format validation** — do the arguments a runner actually
  sent a tool match that tool's declared MCP JSON schema, and does the
  final answer's parsed shape match what ``characteristic_form`` says
  to expect (e.g. "a JSON object" vs "a list").

Every check returns a :class:`CheckResult` with ``passed`` in
``{True, False, None}`` — ``None`` means *not applicable* to this
scenario/trajectory (e.g. no ``hint`` field, no live schemas supplied)
and is excluded from pass/fail aggregation rather than counted as a
failure.

Callable from two places:

* :mod:`evaluation.scorers.deterministic` wraps :func:`run_checks` as a
  ``deterministic_checks`` scorer for the standard ``uv run evaluate``
  pipeline.
* :mod:`benchmark.custom_harness` calls :func:`run_checks` directly
  right after each scenario run, using live schemas from
  :func:`fetch_tool_schemas`.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

_log = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """Outcome of one deterministic check.

    ``passed=None`` means the check did not apply (skipped), and is
    excluded from pass-rate aggregation.
    """

    name: str
    passed: bool | None
    detail: str = ""


@dataclass
class ToolInvocation:
    """One tool call, normalized across the plan-execute and SDK trajectory shapes."""

    server: str | None
    tool: str | None
    args: dict
    output: Any
    error: str | None


KNOWN_SERVERS: tuple[str, ...] = ("iot", "utilities", "fmsr", "tsfm", "wo", "vibration")

# Free-text keywords that identify a server in a scenario's ``hint`` field,
# e.g. "IoT Agent handles ingestion; TSFM Agent detects anomalies; FMSR
# Agent interprets failure modes; WO Agent plans maintenance actions."
_SERVER_KEYWORDS: dict[str, tuple[str, ...]] = {
    "iot": ("iot agent", "iot"),
    "tsfm": (
        "tsfm agent",
        "tsfm",
        "time series",
        "time-series",
        "forecast",
        "anomaly detection",
    ),
    "fmsr": ("fmsr agent", "fmsr", "failure mode"),
    "wo": ("wo agent", "work order", "work-order"),
    "vibration": ("vibration agent", "vibration"),
    "utilities": ("utilities agent", "utility"),
}

_ERROR_OUTPUT_RE = re.compile(r"^\s*(error|exception|traceback)\b", re.IGNORECASE)


def _get_field(scenario: Any, name: str) -> Any:
    """Read a field from either a dict or a pydantic Scenario (extra='allow')."""
    if isinstance(scenario, dict):
        return scenario.get(name)
    return getattr(scenario, name, None)


def _normalize_trajectory(trajectory: Any) -> list[ToolInvocation]:
    """Reduce a persisted trajectory to a flat list of tool invocations.

    Handles the two shapes this repo persists: plan-execute's
    ``list[StepResult]`` (server/tool are already bare slugs) and the SDK
    runners' ``Trajectory`` dict with ``turns[].tool_calls[]`` (server is
    guessed from the MCP-qualified tool name).
    """
    if trajectory is None:
        return []

    if isinstance(trajectory, list):
        out = []
        for step in trajectory:
            if not isinstance(step, dict):
                continue
            tool = step.get("tool")
            error = step.get("error")
            no_tool = not tool or str(tool).lower() in ("none", "null", "")
            # A step with no tool AND no error is a legitimate reasoning-only
            # step the planner marked tool="none" on purpose — skip it. But a
            # step that failed *before* resolving to a tool (e.g. the executor
            # rejecting an invalid/hallucinated server name) also ends up with
            # tool="" — that must still surface as a failure, not disappear.
            if no_tool and not error:
                continue
            out.append(
                ToolInvocation(
                    server=step.get("server"),
                    tool=tool or None,
                    args=step.get("tool_args") or {},
                    output=step.get("response"),
                    error=error,
                )
            )
        return out

    if isinstance(trajectory, dict) and "turns" in trajectory:
        out = []
        for turn in trajectory.get("turns") or []:
            for tc in turn.get("tool_calls") or []:
                name = tc.get("name") or ""
                server, tool = _split_server_and_tool(name)
                out.append(
                    ToolInvocation(
                        server=server,
                        tool=tool,
                        args=tc.get("input") or {},
                        output=tc.get("output"),
                        error=None,
                    )
                )
        return out

    return []


def _split_server_and_tool(name: str) -> tuple[str | None, str]:
    """Split an SDK tool-call name into ``(server, bare_tool_name)``.

    SDK runners qualify tool names differently — claude-agent's
    ``mcp__iot__asset_ids``, this repo's own ``iot.installed_sensors`` — but
    all of them wrap a snake_case tool name around a recognizable server
    token. Splitting on every non-alphanumeric run (which also breaks the
    tool name's own internal underscores) and rejoining the remainder with
    ``_`` recovers the original bare name, which is what schema lookups key
    on. Falls back to the untouched name when no known server token is found.
    """
    tokens = [t for t in re.split(r"[^a-zA-Z0-9]+", name) if t]
    for i, token in enumerate(tokens):
        if token.lower() in KNOWN_SERVERS:
            remainder = tokens[i + 1 :]
            return token.lower(), ("_".join(remainder) if remainder else name)
    return None, name


def _looks_like_error_output(output: Any) -> bool:
    """Heuristic error detection for tool outputs with no structured error
    flag — a softer signal than plan-execute's explicit ``error`` field.

    Covers two shapes seen in practice: an MCP-protocol ``isError``/prefix
    signal (mainly SDK trajectories), and an MCP tool that returns a normal,
    non-erroring response whose *body* is ``{"error": "..."}`` — several of
    this repo's MCP servers use that convention, and plan-execute's
    ``StepResult.response`` is always the flattened text of that body, so it
    has to be parsed back to catch it.
    """
    if isinstance(output, dict):
        if output.get("isError") is True:
            return True
        if "content" in output:
            return _looks_like_error_output(output["content"])
        return bool(output.get("error"))
    if isinstance(output, list):
        return any(_looks_like_error_output(item) for item in output)
    if isinstance(output, str):
        text = output.strip()
        if _ERROR_OUTPUT_RE.match(text):
            return True
        if text.startswith("{"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return False
            return isinstance(parsed, dict) and bool(parsed.get("error"))
        return False
    return False


# ── checks ──────────────────────────────────────────────────────────────────


def check_no_silent_failures(trajectory: Any) -> CheckResult:
    """No tool call in the trajectory failed without the answer saying so."""
    invocations = _normalize_trajectory(trajectory)
    if not invocations:
        return CheckResult(
            "no_silent_failures", None, "no tool invocations in trajectory"
        )

    failures = []
    for inv in invocations:
        if inv.error:
            failures.append(f"{inv.server or '?'}.{inv.tool}: {inv.error}")
        elif _looks_like_error_output(inv.output):
            failures.append(
                f"{inv.server or '?'}.{inv.tool}: output looks like an error (heuristic)"
            )

    passed = not failures
    detail = (
        f"all {len(invocations)} tool call(s) succeeded"
        if passed
        else "; ".join(failures)
    )
    return CheckResult("no_silent_failures", passed, detail)


def check_expected_servers_invoked(scenario: Any, trajectory: Any) -> CheckResult:
    """Verify every server the scenario declares as expected was invoked.

    Prefers a scenario's own structured ``servers`` list — an exact,
    unambiguous declaration — when present. Falls back to keyword-matching
    a free-text ``hint`` field (as AssetOpsBench's original multiagent
    utterances carry, e.g. "IoT Agent handles ingestion; TSFM Agent detects
    anomalies...") only for scenarios that carry a hint but no structured
    list. Checked empirically against every scenario carrying both fields:
    ``servers`` never disagrees with ``hint`` except by being a superset —
    every disagreement found was ``servers`` correctly including ``iot`` for
    an implicit sensor-ingestion prerequisite the hint's free-text summary
    never named explicitly. Prefer the exact declaration.
    """
    declared = _get_field(scenario, "servers")
    if isinstance(declared, list) and declared:
        expected = {str(s).strip().lower() for s in declared if str(s).strip()}
        source = "servers"
    else:
        hint = _get_field(scenario, "hint")
        if not hint:
            return CheckResult(
                "expected_servers_invoked",
                None,
                "scenario declares no 'servers' list and no 'hint' field",
            )
        hint_lower = str(hint).lower()
        expected = {
            server
            for server, keywords in _SERVER_KEYWORDS.items()
            if any(kw in hint_lower for kw in keywords)
        }
        if not expected:
            return CheckResult(
                "expected_servers_invoked",
                None,
                "hint present but no known server keywords matched",
            )
        source = "hint"

    actual = {inv.server for inv in _normalize_trajectory(trajectory) if inv.server}
    missing = expected - actual
    passed = not missing
    detail = (
        f"all expected servers invoked ({source}): {', '.join(sorted(expected))}"
        if passed
        else (
            f"missing expected server(s) ({source}): {', '.join(sorted(missing))} "
            f"(expected {sorted(expected)}, saw {sorted(actual)})"
        )
    )
    return CheckResult("expected_servers_invoked", passed, detail)


def check_tool_args_match_schema(
    trajectory: Any,
    tool_schemas: dict[str, dict[str, dict]] | None,
) -> CheckResult:
    """Every resolved tool-call arg dict includes that tool's required params."""
    if not tool_schemas:
        return CheckResult(
            "tool_args_match_schema", None, "no live tool schemas supplied"
        )

    invocations = _normalize_trajectory(trajectory)
    checked = 0
    failures = []
    for inv in invocations:
        if not inv.server or not inv.tool:
            continue
        schema = tool_schemas.get(inv.server, {}).get(inv.tool)
        if schema is None:
            continue
        checked += 1
        required = schema.get("required", [])
        missing = [p for p in required if p not in (inv.args or {})]
        if missing:
            failures.append(f"{inv.server}.{inv.tool}: missing required arg(s) {missing}")

    if checked == 0:
        return CheckResult(
            "tool_args_match_schema", None, "no tool call matched a known schema"
        )

    passed = not failures
    detail = (
        f"{checked}/{checked} tool call(s) match their schema"
        if passed
        else "; ".join(failures)
    )
    return CheckResult("tool_args_match_schema", passed, detail)


def check_answer_shape_matches_characteristic_form(
    scenario: Any, answer: str
) -> CheckResult:
    """If characteristic_form specifies a JSON object/list shape, verify the answer parses as it."""
    characteristic = _get_field(scenario, "characteristic_form")
    if not characteristic:
        return CheckResult(
            "answer_shape_matches_characteristic_form",
            None,
            "scenario has no characteristic_form",
        )

    text = str(characteristic).lower()
    if "json object" in text or re.search(r"\bdict(ionary)?\b", text):
        expected_type: type = dict
    # "a list of X" / "as a list" is a shape instruction; bare "list" as a
    # verb ("should list the sensors") or in a noun compound ("sensor list")
    # is prose asking for an enumeration, not a literal JSON array — matching
    # on it produced false positives against real answers phrased as prose.
    elif re.search(r"\blist of\b|\bas a list\b|\ba list\b|\barray\b", text):
        expected_type = list
    else:
        return CheckResult(
            "answer_shape_matches_characteristic_form",
            None,
            "characteristic_form does not specify a JSON object/list shape",
        )

    from .scorers.static_json import parse_structured_answer

    parsed = parse_structured_answer(answer)
    passed = isinstance(parsed, expected_type)
    detail = (
        f"answer parses as {type(parsed).__name__}, matches expected {expected_type.__name__}"
        if passed
        else f"answer parses as {type(parsed).__name__}, expected {expected_type.__name__}"
    )
    return CheckResult("answer_shape_matches_characteristic_form", passed, detail)


# Domain-general English patterns for "this question is asking for a
# time-series prediction/anomaly judgment" and "this question is asking
# whether a maintenance action should be taken" — independent of any
# benchmark-specific metadata (hint, characteristic_form). Deliberately not
# derived from the wording of any specific scenario: these are the generic
# phrasings any predictive-maintenance question would use, not a fingerprint
# of the runs that motivated adding this check.
PREDICTIVE_INTENT_RE = re.compile(
    r"\bpredict|\bforecast|\banomal|\btrend(ing)?\b|"
    r"\brisk of\b|\bapproaching (total )?failure|\bnearing failure|"
    r"\bwithin the (next|following)\b|"
    r"\blikely to fail|\bfail(ure)?\s*(soon|imminent)|"
    r"\b(could|might|will)\b[^.?!]{0,40}\bfail",
    re.IGNORECASE,
)
MAINTENANCE_RECOMMENDATION_RE = re.compile(
    r"schedule(d)? maintenance|maintenance (be )?scheduled|"
    r"recommend(ed)? (a )?(maintenance )?action|plan (a )?(repair|maintenance)|"
    r"\bwork order\b",
    re.IGNORECASE,
)
# Deliberately narrow: only explicit "enumerate/diagnose the failure modes"
# language, not any mention of a named failure type ("air leak failure").
# The broader phrasing would also catch pure-forecasting questions that have
# nothing to do with failure-mode diagnosis (e.g. "forecast energy
# consumption"), which is a worse trade than leaving some real fmsr-relevant
# questions unflagged. Validated empirically: fires on the benchmark's own
# "List all failure modes..." scenarios (9/46 in the diverse set) and does
# not fire on any of the 16 predictive-maintenance scenarios.
FMSR_INTENT_RE = re.compile(
    r"\bfailure mode|\broot cause|\bwhat (?:could|can|might) cause|"
    r"\bwhich fault|\btype(?:s)? of failure|\bknown failure|\bcause of\b",
    re.IGNORECASE,
)


def check_predictive_task_uses_expected_tools(
    scenario: Any, trajectory: Any
) -> CheckResult:
    """If the question's own wording implies prediction/planning/diagnosis,
    verify the matching tool category was actually used — independent of any
    hint or characteristic_form field, so it applies even to scenarios with
    no benchmark-authored metadata at all.

    Three independent signals, each optional:
    - predictive/anomaly language (e.g. "predict", "risk of ... failure",
      "within the next N days") implies a ``tsfm`` call should exist.
    - maintenance-action language (e.g. "should maintenance be scheduled",
      "recommend an action") implies a ``wo`` call should exist.
    - failure-mode/diagnosis language (e.g. "list all failure modes",
      "what could cause") implies an ``fmsr`` call should exist.

    Skips when none of the three signals are present in the question text.
    """
    text = _get_field(scenario, "text") or ""
    invoked = {inv.server for inv in _normalize_trajectory(trajectory) if inv.server}

    expected: set[str] = set()
    if PREDICTIVE_INTENT_RE.search(text):
        expected.add("tsfm")
    if MAINTENANCE_RECOMMENDATION_RE.search(text):
        expected.add("wo")
    if FMSR_INTENT_RE.search(text):
        expected.add("fmsr")

    if not expected:
        return CheckResult(
            "predictive_task_uses_expected_tools",
            None,
            "question text has no predictive/maintenance/diagnostic language",
        )

    missing = expected - invoked
    passed = not missing
    detail = (
        f"question implies {sorted(expected)}, all invoked"
        if passed
        else f"question implies {sorted(expected)} but only saw {sorted(invoked)}"
    )
    return CheckResult("predictive_task_uses_expected_tools", passed, detail)


def run_checks(
    scenario: Any,
    answer: str,
    trajectory: Any,
    tool_schemas: dict[str, dict[str, dict]] | None = None,
) -> list[CheckResult]:
    """Run every registered deterministic check for one scenario run."""
    return [
        check_no_silent_failures(trajectory),
        check_expected_servers_invoked(scenario, trajectory),
        check_tool_args_match_schema(trajectory, tool_schemas),
        check_answer_shape_matches_characteristic_form(scenario, answer),
        check_predictive_task_uses_expected_tools(scenario, trajectory),
    ]


# ── live schema fetch (optional, used by the standalone harness) ────────────


async def fetch_tool_schemas(
    server_paths: dict[str, str] | None = None,
) -> dict[str, dict[str, dict]]:
    """Connect to each MCP server and build ``{server: {tool: {"required": [...]}}}``.

    Best-effort per server: a server that fails to start (e.g. missing
    credentials for its LLM-use tools) is logged and skipped rather than
    failing the whole fetch, so :func:`check_tool_args_match_schema` still
    runs against whatever schemas were successfully collected.
    """
    from agent.runner import DEFAULT_SERVER_PATHS
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    paths = server_paths or DEFAULT_SERVER_PATHS
    schemas: dict[str, dict[str, dict]] = {}
    for name, entry_point in paths.items():
        try:
            params = StdioServerParameters(command="uv", args=["run", str(entry_point)])
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    server_schema: dict[str, dict] = {}
                    for tool in result.tools:
                        input_schema = tool.inputSchema or {}
                        server_schema[tool.name] = {
                            "required": list(input_schema.get("required", [])),
                            "properties": list(input_schema.get("properties", {})),
                        }
                    schemas[name] = server_schema
        except Exception as exc:  # noqa: BLE001
            _log.warning("fetch_tool_schemas: %s unavailable: %s", name, exc)
    return schemas
