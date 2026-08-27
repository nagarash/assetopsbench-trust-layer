"""CLI entry point for the simple-agent runner.

Usage:
    simple-agent --servers iot,fmsr "List all failure modes of asset Chiller 6."
    simple-agent --model-id litellm_proxy/meta-llama/llama-4-maverick \\
        --servers iot,tsfm "Forecast Chiller 6's Tonnage for next week."
"""

from __future__ import annotations

import argparse

from agent._cli_common import add_common_args, print_result, run_sdk_cli

_DEFAULT_MODEL = "litellm_proxy/openai/gpt-4o-mini"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simple-agent",
        description=(
            "Run a question through a minimal in-process tool-calling loop "
            "over a curated MCP server subset."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
model-id format:
  litellm_proxy/<model>    LiteLLM-proxy-shaped OpenAI-compatible endpoint
                            (e.g. OpenRouter): litellm_proxy/openai/gpt-4o-mini
  tokenrouter/<model>       TokenRouter (OpenAI-compatible) model

environment variables:
  LITELLM_API_KEY / LITELLM_BASE_URL         for litellm_proxy/* models
  TOKENROUTER_API_KEY / TOKENROUTER_BASE_URL for tokenrouter/* models

examples:
  simple-agent --servers iot "What sensors are on Chiller 6?"
  simple-agent --servers iot,fmsr --model-id litellm_proxy/meta-llama/llama-4-maverick \\
      "List all failure modes of asset Chiller 6."
""",
    )
    add_common_args(parser, default_model=_DEFAULT_MODEL)
    parser.add_argument(
        "--servers",
        default=None,
        metavar="SERVER,SERVER,...",
        help=(
            "Comma-separated MCP servers to register for this run "
            "(e.g. iot,fmsr,wo). Keep this small — a large tool catalog is "
            "the failure mode this runner exists to avoid. Default: all servers."
        ),
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=8,
        metavar="N",
        help="Maximum model turns before giving up (default: 8).",
    )
    return parser


async def _run(args: argparse.Namespace) -> None:
    from agent.simple_agent.runner import SimpleAgentRunner

    servers = args.servers.split(",") if args.servers else None
    runner = SimpleAgentRunner(
        model=args.model_id, servers=servers, max_turns=args.max_turns
    )
    result = await runner.run(args.question)

    print_result(
        result,
        show_trajectory=args.show_trajectory,
        output_json=args.output_json,
    )


def main() -> None:
    run_sdk_cli("simple-agent", _build_parser, _run)


if __name__ == "__main__":
    main()
