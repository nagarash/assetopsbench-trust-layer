"""Custom scenario harness with deterministic trust checks.

Loops scenarios from a JSONL file (the AssetOpsBench HF dataset shape —
``id``/``text``/optional ``hint``/``characteristic_form``/...) through one
of the installed agent-runner CLIs, persists each trajectory the same way
the built-in runners do, then layers deterministic (no-LLM) trust checks
from :mod:`evaluation.checks` on top of every run: did every tool call
actually succeed, did the run touch the agents/servers the scenario names
as expected, do the resolved tool-call arguments match the tool's live MCP
schema, and does the final answer's shape match what the scenario's
``characteristic_form`` says to expect.

This is separate from ``benchmark.scenario_suite_runner``, which expects
scenarios laid out as ``scenario_<id>/{question.txt,groundtruth.txt}``
folders. This harness reads the HF dataset's JSONL scenario files directly.

Usage:

    uv run run-scenario-checks \\
        --scenarios assetopsbench-data/data/asset/compressor_utterance.jsonl \\
        --runner plan-execute \\
        --model-id litellm_proxy/openai/gpt-4o-mini \\
        --limit 5

Trajectories persist to ``--trajectory-dir`` in the same schema the built-in
runners use, so they also compose with ``uv run evaluate``
(``--scorer-default deterministic_checks`` or ``llm_judge``) after the fact.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

from evaluation.checks import fetch_tool_schemas, run_checks
from evaluation.loader import load_scenarios
from evaluation.models import Scenario

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_agent(
    *,
    runner: str,
    model_id: str,
    scenario_id: str,
    question: str,
    trajectory_dir: Path,
    run_id: str,
) -> None:
    env = os.environ.copy()
    env["AGENT_TRAJECTORY_DIR"] = str(trajectory_dir)
    cmd = [
        "uv",
        "run",
        runner,
        "--model-id",
        model_id,
        "--scenario-id",
        str(scenario_id),
        "--run-id",
        run_id,
        question,
    ]
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)


def _load_trajectory_record(trajectory_dir: Path, run_id: str) -> dict | None:
    path = trajectory_dir / f"{run_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _run_one_scenario(
    scenario: Scenario,
    *,
    runner: str,
    model_id: str,
    trajectory_dir: Path,
    tool_schemas: dict[str, dict[str, dict]],
) -> dict:
    run_id = f"{runner}_{scenario.id}"
    print(f"\n[{scenario.id}] {scenario.text[:80]}")

    _run_agent(
        runner=runner,
        model_id=model_id,
        scenario_id=scenario.id,
        question=scenario.text,
        trajectory_dir=trajectory_dir,
        run_id=run_id,
    )

    record = _load_trajectory_record(trajectory_dir, run_id)
    if record is None:
        raise RuntimeError(f"no trajectory persisted for run_id={run_id!r}")

    checks = run_checks(
        scenario,
        record.get("answer", ""),
        record.get("trajectory"),
        tool_schemas=tool_schemas,
    )
    for c in checks:
        status = "SKIP" if c.passed is None else ("PASS" if c.passed else "FAIL")
        print(f"  [{status}] {c.name}: {c.detail}")

    return {
        "scenario_id": scenario.id,
        "run_id": run_id,
        "question": scenario.text,
        "answer": record.get("answer", ""),
        "checks": [asdict(c) for c in checks],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run-scenario-checks",
        description=(
            "Run scenarios through an agent runner and score them with "
            "deterministic (no-LLM) trust checks."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--scenarios", type=Path, required=True, help="JSONL scenario file."
    )
    parser.add_argument(
        "--runner",
        default="plan-execute",
        help="Agent runner CLI to invoke (default: plan-execute).",
    )
    parser.add_argument(
        "--model-id",
        default="litellm_proxy/openai/gpt-4o-mini",
        help="Model id passed to the runner.",
    )
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=REPO_ROOT / "traces/trajectories/custom_harness",
        help="Directory to persist trajectory JSON into.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to write a JSON report of all scenario results.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only run the first N scenarios."
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running later scenarios if one scenario's agent run fails.",
    )
    parser.add_argument(
        "--skip-schema-fetch",
        action="store_true",
        help=(
            "Skip the live MCP schema fetch; tool_args_match_schema will be "
            "reported as skipped for every scenario."
        ),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    scenarios = load_scenarios(args.scenarios)
    if args.limit:
        scenarios = scenarios[: args.limit]
    if not scenarios:
        raise SystemExit(f"no scenarios found in {args.scenarios}")

    args.trajectory_dir.mkdir(parents=True, exist_ok=True)

    tool_schemas: dict[str, dict[str, dict]] = {}
    if not args.skip_schema_fetch:
        print("Fetching live tool schemas from MCP servers...")
        tool_schemas = asyncio.run(fetch_tool_schemas())

    report_rows: list[dict] = []
    check_totals: dict[str, dict[str, int]] = {}

    for scenario in scenarios:
        try:
            row = _run_one_scenario(
                scenario,
                runner=args.runner,
                model_id=args.model_id,
                trajectory_dir=args.trajectory_dir,
                tool_schemas=tool_schemas,
            )
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            print(f"  ERROR: {exc}")
            if not args.continue_on_error:
                raise
            continue

        report_rows.append(row)
        for c in row["checks"]:
            bucket = check_totals.setdefault(c["name"], {"pass": 0, "fail": 0, "skip": 0})
            key = "skip" if c["passed"] is None else ("pass" if c["passed"] else "fail")
            bucket[key] += 1

    print("\n" + "=" * 60)
    print("Deterministic check summary")
    print("=" * 60)
    for name, counts in sorted(check_totals.items()):
        total = sum(counts.values())
        print(
            f"  {name}: {counts['pass']} pass / {counts['fail']} fail / "
            f"{counts['skip']} skip (of {total})"
        )

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {"scenarios": report_rows, "summary": check_totals}, indent=2
            ),
            encoding="utf-8",
        )
        print(f"\nReport written: {args.report}")


if __name__ == "__main__":
    main()
