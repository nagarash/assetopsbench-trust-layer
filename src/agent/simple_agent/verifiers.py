"""In-loop grounding verifiers — the trust layer wired into the tool loop.

Deterministic checks (``evaluation.checks``) score a trajectory only after
the run is over: useful for reporting, useless for the run itself. Two real
answer-content failures turned up in real runs that every structural check
passed cleanly (schema-valid arguments, no tool error, right servers
touched):

* ``fmsr.get_failure_modes`` called with the wrong ``asset_class`` for the
  asset actually in question (schema-valid string, wrong value).
* ``wo.list_workorders`` called with an unrequested ``status`` filter, then
  the narrowed result presented as an exhaustive answer.

Both are invisible to a post-hoc structural check, but each has direct
evidence sitting in the trajectory: an ``iot.asset_detail`` call already
returned the asset's real ``assettype``, or the question text itself never
mentioned the filter value used. These verifiers look for that evidence
right after the tool call that used it and, if it contradicts the call,
inject a corrective observation into the conversation — the same mechanism
the loop already uses for tool errors — so the model gets a chance to
re-query before finalizing, instead of a mismatch only surfacing in a report
after the run is done.
"""

from __future__ import annotations

import re
from typing import Any

from ..models import ToolCall


def _asset_class_key(value: str) -> str:
    """Normalize an asset class/type string for loose comparison.

    Mirrors ``servers.fmsr.main._asset_class_key`` (strip digits, collapse
    separators, lowercase) so the comparison uses the same normalization the
    failure-mode database itself keys on.
    """
    key = re.sub(r"\d+", "", value or "")
    key = re.sub(r"[_\-]+", " ", key)
    return re.sub(r"\s+", " ", key).strip().lower()


def _unwrap_result(output: Any) -> dict | None:
    if isinstance(output, dict):
        result = output.get("result", output)
        return result if isinstance(result, dict) else None
    return None


def _class_mismatch_message(asset_class: str, real_type: str) -> str:
    return (
        f"Warning: fmsr.get_failure_modes was called with "
        f"asset_class={asset_class!r}, but iot.asset_detail reported "
        f"assettype={real_type!r} for this asset — these are inconsistent. "
        "Re-call fmsr.get_failure_modes now with the corrected asset_class "
        "before answering; do not finalize your answer without the failure "
        "modes for the asset's actual type, and do not simply note the "
        "discrepancy without resolving it."
    )


def _verify_failure_mode_asset_class(
    args: dict, prior_tool_calls: list[ToolCall]
) -> str | None:
    """Cross-check ``fmsr.get_failure_modes``'s asset_class against any
    ``iot.asset_detail`` result already seen in this trajectory.

    Only fires against evidence that already exists — no ``asset_detail``
    call yet means "go confirm," not "you're wrong," since we have no
    grounds for the latter.
    """
    asset_class = args.get("asset_class")
    if not asset_class:
        return None

    asset_detail_calls = [tc for tc in prior_tool_calls if tc.name == "iot.asset_detail"]
    if not asset_detail_calls:
        return (
            f"Before treating the failure modes for asset_class={asset_class!r} as "
            "final: this trajectory hasn't confirmed that class against the asset's "
            "actual registered type yet. If you haven't already, call "
            "iot.asset_detail for the asset in question and re-check — its "
            "'assettype' field is the ground truth for asset_class."
        )

    given = _asset_class_key(asset_class)
    for tc in asset_detail_calls:
        result = _unwrap_result(tc.output)
        real_type = result.get("assettype") if result else None
        if not real_type:
            continue
        real = _asset_class_key(str(real_type))
        if real and given and real not in given and given not in real:
            return _class_mismatch_message(asset_class, str(real_type))
    return None


def _verify_asset_detail_against_prior_failure_modes(
    output: Any, prior_tool_calls: list[ToolCall]
) -> str | None:
    """The mirror-image check: ``asset_detail`` often arrives *after* the
    model already called ``get_failure_modes`` with a guessed class (it's
    reacting to the nudge above) — that earlier call is already in the past
    by the time the confirming evidence shows up, so it never gets
    re-checked unless something re-checks it here, on arrival of the
    evidence itself."""
    result = _unwrap_result(output)
    real_type = result.get("assettype") if result else None
    if not real_type:
        return None
    real = _asset_class_key(str(real_type))

    for tc in prior_tool_calls:
        if tc.name != "fmsr.get_failure_modes":
            continue
        asset_class = tc.input.get("asset_class") if tc.input else None
        if not asset_class:
            continue
        given = _asset_class_key(str(asset_class))
        if given and real and real not in given and given not in real:
            return _class_mismatch_message(str(asset_class), str(real_type))
    return None


_WO_NARROWING_PARAMS = ("status", "worktype", "wopriority")


def _verify_workorder_filter_requested(
    tool: str, args: dict, question: str
) -> str | None:
    """Flag a work-order query filter that wasn't present in the question.

    A model finalizing "there is no X" off a result it silently narrowed is
    exactly the overstatement pattern this repo's own failure analysis (and
    this project's own test runs) called out — not a tool error, so no
    structural check catches it.
    """
    if not tool.startswith("wo."):
        return None

    question_lower = question.lower()
    unrequested = [
        f"{param}={args[param]!r}"
        for param in _WO_NARROWING_PARAMS
        if args.get(param) and str(args[param]).lower() not in question_lower
    ]
    if not unrequested:
        return None

    return (
        f"Note: this work-order query used {', '.join(unrequested)}, which "
        "wasn't requested in the question. If you're about to state an "
        "absolute conclusion (e.g. \"there is no such work order\"), that "
        "conclusion only holds for this filtered subset — either re-query "
        "without the filter to confirm, or state the limitation explicitly "
        "in your answer."
    )


_TSFM_CATALOG_TOOLS = frozenset(
    {"find_models", "search_models", "list_models", "describe_candidates"}
)


def _verify_tsfm_empty_catalog(tc: ToolCall) -> str | None:
    """If a tsfm catalog-lookup tool comes back empty, say so explicitly
    instead of leaving it ambiguous.

    Observed in production: a model facing an empty ``models: []`` result
    reads it as "maybe I searched wrong" and retries with a different
    ``task_id``/``domain``/search text — repeatedly, for many turns, because
    nothing tells it the search was correct and the catalog is just empty.
    This fires on the *first* empty result, not after a retry pattern
    emerges, so the model never has to guess.
    """
    tool_name = tc.name.rsplit(".", 1)[-1]
    if tool_name not in _TSFM_CATALOG_TOOLS:
        return None
    result = _unwrap_result(tc.output)
    if not isinstance(result, dict):
        return None
    items = result.get("models") if "models" in result else result.get("candidates")
    if items is None or len(items) != 0:
        return None
    return (
        f"tsfm.{tool_name} returned no results — no model is registered for "
        "this task/filter combination in the catalog. This is a definitive "
        "answer, not an ambiguous one: do not retry with a different "
        "task_id, domain, or search text. Move on to whatever this question "
        "still needs (fmsr, wo), or state plainly that no model is "
        "available for this task."
    )


def verify_tool_call(
    tc: ToolCall,
    *,
    question: str,
    prior_tool_calls: list[ToolCall],
) -> str | None:
    """Return a corrective observation for *tc* if it looks ungrounded, else None.

    ``prior_tool_calls`` is expected to already include ``tc`` itself (the
    caller extends its running list before verifying) so the asset_class
    checks can see *tc*'s own output when relevant.
    """
    if tc.name == "fmsr.get_failure_modes":
        return _verify_failure_mode_asset_class(tc.input, prior_tool_calls)
    if tc.name == "iot.asset_detail":
        return _verify_asset_detail_against_prior_failure_modes(
            tc.output, prior_tool_calls
        )
    if tc.name.startswith("wo."):
        return _verify_workorder_filter_requested(tc.name, tc.input, question)
    if tc.name.startswith("tsfm."):
        return _verify_tsfm_empty_catalog(tc)
    return None


# ── pre-finalization gate ─────────────────────────────────────────────────
#
# The three checks above fire right after a specific tool call, using
# evidence that call produced. This one fires at the opposite moment: right
# before the model's turn with no tool calls is accepted as the final
# answer — using the same question-text patterns that
# evaluation.checks.check_predictive_task_uses_expected_tools scores after
# the fact, but here as a chance to actually fix it instead of just
# reporting it. Deliberately advisory, not a hard block: a genuinely empty
# sensor stream is a legitimate reason to skip tsfm (see the false-positive
# noted against scenario 509 in evaluation.checks' docstring), so the nudge
# asks the model to weigh whether it applies rather than asserting it must.

from evaluation.checks import (  # noqa: E402
    FMSR_INTENT_RE,
    MAINTENANCE_RECOMMENDATION_RE,
    PREDICTIVE_INTENT_RE,
)

_TSFM_NUDGE = (
    "this question asks about predicting or detecting an anomaly, but no "
    "time-series (tsfm) tool has been used yet — if you found relevant "
    "sensor data earlier, consider running anomaly detection or a forecast "
    "on it before concluding; if no data was available for the requested "
    "window, say so explicitly rather than reasoning from raw numbers alone"
)
_WO_NUDGE = (
    "this question asks whether maintenance or an action should be taken, "
    "but no work-order (wo) tool has been used yet — if a concrete "
    "recommendation is warranted, consider checking or generating a work "
    "order before concluding"
)
_FMSR_NUDGE = (
    "this question asks about failure modes or their causes, but no "
    "failure-mode (fmsr) tool has been used yet — look up the known failure "
    "modes for the relevant asset class before concluding, rather than "
    "answering from general knowledge alone"
)


def verify_before_finalizing(
    question: str, prior_tool_calls: list[ToolCall]
) -> list[tuple[str, str]]:
    """Return ``[(category_key, nudge_message), ...]`` for implied-but-unused
    tool categories, checked against the whole trajectory so far.

    Category keys let the caller deduplicate — nudge once per category, not
    every remaining turn, so a model that has a legitimate reason to skip a
    tool (e.g. no data exists) isn't stuck being re-nudged until max_turns.
    """
    invoked = {tc.name.split(".")[0] for tc in prior_tool_calls if tc.name}
    pending: list[tuple[str, str]] = []
    if PREDICTIVE_INTENT_RE.search(question) and "tsfm" not in invoked:
        pending.append(("tsfm_predictive", _TSFM_NUDGE))
    if MAINTENANCE_RECOMMENDATION_RE.search(question) and "wo" not in invoked:
        pending.append(("wo_maintenance", _WO_NUDGE))
    if FMSR_INTENT_RE.search(question) and "fmsr" not in invoked:
        pending.append(("fmsr_diagnostic", _FMSR_NUDGE))
    return pending
