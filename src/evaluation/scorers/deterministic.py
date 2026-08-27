"""Deterministic, no-LLM trust checks — trajectory integrity + schema/format validation.

Plugs into the evaluation pipeline as ``deterministic_checks``:

    uv run evaluate --scorer-default deterministic_checks --trajectories ... --scenarios ...

Unlike ``llm_judge``, every check here is a pure function over the
persisted trajectory: no model call, fully reproducible. Checks that
don't apply to a given scenario (e.g. no ``hint`` field, no live tool
schemas supplied) are skipped rather than failed, and don't count
against the pass rate. See :mod:`evaluation.checks` for the check
implementations.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from ..checks import run_checks
from ..models import Scenario, ScorerResult
from . import register


class DeterministicChecksScorer:
    """Evaluation scorer wrapper around :func:`evaluation.checks.run_checks`."""

    def __init__(
        self,
        tool_schemas: dict[str, dict[str, dict]] | None = None,
        name: str = "deterministic_checks",
    ) -> None:
        self._tool_schemas = tool_schemas
        self.name = name

    def __call__(
        self, scenario: Scenario, answer: str, trajectory_text: str
    ) -> ScorerResult:
        trajectory = None
        if trajectory_text:
            try:
                trajectory = json.loads(trajectory_text)
            except json.JSONDecodeError:
                trajectory = None

        checks = run_checks(
            scenario, answer, trajectory, tool_schemas=self._tool_schemas
        )
        applicable = [c for c in checks if c.passed is not None]
        passed = all(c.passed for c in applicable) if applicable else True
        score = (
            sum(1 for c in applicable if c.passed) / len(applicable)
            if applicable
            else 1.0
        )
        failed = [c for c in applicable if not c.passed]
        rationale = (
            "; ".join(f"{c.name}: {c.detail}" for c in failed)
            if failed
            else (
                "all applicable checks passed"
                if applicable
                else "no checks were applicable to this scenario"
            )
        )
        return ScorerResult(
            scorer=self.name,
            passed=passed,
            score=round(score, 3),
            rationale=rationale,
            details={"checks": [asdict(c) for c in checks]},
        )


def install(
    tool_schemas: dict[str, dict[str, dict]] | None = None,
    name: str = "deterministic_checks",
) -> None:
    """Register a deterministic-checks scorer bound to optional ``tool_schemas``."""
    register(name, DeterministicChecksScorer(tool_schemas=tool_schemas, name=name))
