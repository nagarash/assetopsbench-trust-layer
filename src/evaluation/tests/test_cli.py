"""Tests for the evaluation CLI argument surface."""

from __future__ import annotations

from evaluation.cli import _build_parser, _resolve_scenario_ids


def test_cli_accepts_optional_scenario_selector() -> None:
    args = _build_parser().parse_args(
        [
            "--trajectories",
            "trajectories",
            "--scenarios",
            "scenarios",
            "--scenario-ids",
            "fcc+fmsr_all",
        ]
    )

    assert args.scenario_ids == "fcc+fmsr_all"


def test_resolve_scenario_ids_is_optional() -> None:
    assert _resolve_scenario_ids(None) is None
