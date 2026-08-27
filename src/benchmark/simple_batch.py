"""In-process batch runner for our own scenario set + SimpleAgentRunner.

Companion to :mod:`benchmark.custom_harness`, but everything runs in one
asyncio process — no ``subprocess.run(["uv", "run", ...])`` per scenario.
That subprocess-per-scenario pattern is what corrupted this project's
editable install twice in one session (repeated rapid ``uv run`` invocations
racing the venv's ``.pth``/metadata files); this harness calls
:class:`agent.simple_agent.runner.SimpleAgentRunner` directly instead.

Each scenario declares its own ``servers`` list (see
``src/couchdb/scenarios_data/custom/scenarios.jsonl``), so the tool catalog
handed to the model stays small and scenario-specific — the fix for the
large-tool-catalog failure observed with ``openai-agent``.

Usage:

    uv run python -m benchmark.simple_batch \\
        --scenarios src/couchdb/scenarios_data/custom/scenarios.jsonl \\
        --model-id litellm_proxy/meta-llama/llama-4-maverick \\
        --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path

from agent.simple_agent.runner import SimpleAgentRunner
from evaluation.checks import fetch_tool_schemas, run_checks
from evaluation.loader import load_scenarios
from evaluation.metrics import metrics_from_trajectory
from evaluation.models import PersistedTrajectory
from observability import set_run_context

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_trajectory_record(trajectory_dir: Path, run_id: str) -> dict | None:
    path = trajectory_dir / f"{run_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


async def _run_one(
    scenario,
    *,
    model_id: str,
    max_turns: int,
    trajectory_dir: Path,
    tool_schemas: dict[str, dict[str, dict]],
    enable_verifiers: bool,
) -> dict:
    servers = getattr(scenario, "servers", None)
    run_id = f"simple-agent_{scenario.id}"
    print(f"\n[{scenario.id}] {scenario.text[:80]} (servers={servers or 'all'})")

    set_run_context(run_id=run_id, scenario_id=str(scenario.id))
    runner = SimpleAgentRunner(
        model=model_id,
        servers=servers,
        max_turns=max_turns,
        enable_verifiers=enable_verifiers,
    )
    result = await runner.run(scenario.text)

    record = _load_trajectory_record(trajectory_dir, run_id)
    trajectory = record.get("trajectory") if record else None

    checks = run_checks(scenario, result.answer, trajectory, tool_schemas=tool_schemas)
    for c in checks:
        status = "SKIP" if c.passed is None else ("PASS" if c.passed else "FAIL")
        print(f"  [{status}] {c.name}: {c.detail}")

    ops = metrics_from_trajectory(PersistedTrajectory.from_raw(record)) if record else None
    if ops is not None:
        cost = f"${ops.est_cost_usd:.4f}" if ops.est_cost_usd is not None else "unknown"
        print(
            f"  tokens: {ops.tokens_in} in / {ops.tokens_out} out"
            f"  turns: {ops.turn_count}  est. cost: {cost}"
        )

    return {
        "scenario_id": scenario.id,
        "run_id": run_id,
        "question": scenario.text,
        "answer": result.answer,
        "checks": [asdict(c) for c in checks],
        "ops": ops.model_dump() if ops is not None else None,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simple-batch",
        description=(
            "Run our own scenario set through simple-agent, in-process, "
            "with deterministic trust checks applied live."
        ),
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=REPO_ROOT / "src/couchdb/scenarios_data/custom/scenarios.jsonl",
        help="JSONL scenario file (default: our own custom scenario set).",
    )
    parser.add_argument(
        "--model-id",
        default="litellm_proxy/openai/gpt-4o-mini",
        help="Model id passed to simple-agent.",
    )
    parser.add_argument(
        "--max-turns", type=int, default=8, help="Max turns per scenario (default: 8)."
    )
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=REPO_ROOT / "traces/trajectories/simple_batch",
        help="Directory to persist trajectory JSON into.",
    )
    parser.add_argument(
        "--report", type=Path, default=None, help="Optional path for a JSON report."
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--skip-schema-fetch",
        action="store_true",
        help="Skip the live MCP schema fetch (tool_args_match_schema will be skipped).",
    )
    parser.add_argument(
        "--no-verifiers",
        action="store_true",
        help=(
            "Disable the in-loop verifiers (verify_tool_call / "
            "verify_before_finalizing) — isolates the system prompt's own "
            "effect from the enforced finalization gate, for comparison."
        ),
    )
    return parser


async def _main_async(args: argparse.Namespace) -> None:
    scenarios = load_scenarios(args.scenarios)
    if args.limit:
        scenarios = scenarios[: args.limit]
    if not scenarios:
        raise SystemExit(f"no scenarios found in {args.scenarios}")

    args.trajectory_dir.mkdir(parents=True, exist_ok=True)
    os.environ["AGENT_TRAJECTORY_DIR"] = str(args.trajectory_dir)

    tool_schemas: dict[str, dict[str, dict]] = {}
    if not args.skip_schema_fetch:
        print("Fetching live tool schemas from MCP servers...")
        tool_schemas = await fetch_tool_schemas()

    report_rows: list[dict] = []
    check_totals: dict[str, dict[str, int]] = {}

    for scenario in scenarios:
        try:
            row = await _run_one(
                scenario,
                model_id=args.model_id,
                max_turns=args.max_turns,
                trajectory_dir=args.trajectory_dir,
                tool_schemas=tool_schemas,
                enable_verifiers=not args.no_verifiers,
            )
        except Exception as exc:  # noqa: BLE001
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

    ops_rows = [row["ops"] for row in report_rows if row.get("ops")]
    tokens_in_total = sum(o["tokens_in"] for o in ops_rows)
    tokens_out_total = sum(o["tokens_out"] for o in ops_rows)
    costs = [o["est_cost_usd"] for o in ops_rows if o.get("est_cost_usd") is not None]
    cost_total = round(sum(costs), 6) if costs else None
    ops_summary = {
        "scenarios_with_ops": len(ops_rows),
        "tokens_in_total": tokens_in_total,
        "tokens_out_total": tokens_out_total,
        "tool_calls_total": sum(o["tool_call_count"] for o in ops_rows),
        "est_cost_usd_total": cost_total,
        "est_cost_usd_missing_for": len(ops_rows) - len(costs),
    }

    print("\n" + "=" * 60)
    print("Ops / cost summary")
    print("=" * 60)
    print(f"  model: {args.model_id}")
    print(f"  tokens: {tokens_in_total} in / {tokens_out_total} out")
    print(f"  tool calls: {ops_summary['tool_calls_total']}")
    if cost_total is not None:
        note = (
            ""
            if ops_summary["est_cost_usd_missing_for"] == 0
            else f"  ({ops_summary['est_cost_usd_missing_for']} scenario(s) missing a price entry)"
        )
        print(f"  est. total cost: ${cost_total:.4f}{note}")
    else:
        print(
            "  est. total cost: unknown (no price entry for this model in "
            "evaluation.metrics._PRICE_PER_1M)"
        )

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "scenarios": report_rows,
                    "summary": check_totals,
                    "ops_summary": ops_summary,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nReport written: {args.report}")


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    args = _build_parser().parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
