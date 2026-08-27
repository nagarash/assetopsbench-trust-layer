# AssetOpsBench — Trust Layer

A private research fork of [IBM/AssetOpsBench](https://github.com/IBM/AssetOpsBench)
(the original upstream README is preserved at [`docs/UPSTREAM_README.md`](docs/UPSTREAM_README.md)).
This repo keeps the parts of the framework this project actually uses — the domain MCP
servers, the MCP client, and the evaluation subsystem — and adds a deterministic **trust
layer**: post-hoc checks that score a finished agent trajectory without an LLM judge,
and in-loop verifiers that inspect tool calls as they happen and correct the agent
mid-run.

**Full write-up:** [`reports/trust-vs-outcome.pdf`](reports/trust-vs-outcome.pdf)

## Results at a glance

Same 12-scenario sample, same trust layer, three models:

| Metric | gpt-4o-mini | GLM-4.6 | Claude Sonnet 4.5 |
|---|---|---|---|
| Success rate (0 failed checks) | 58% (7/12) | 58% (7/12) | 67% (8/12) |
| Cost per scenario | $0.0085 | $0.0513 | $0.263 |
| Total cost, 12 scenarios | $0.10 | $0.62 | $3.15 |

The small model plus trust layer lands within 9 points of a frontier model's success
rate at roughly 1/31st the cost. The trust layer itself moves the hardest tool-invocation
category (work orders) from 6% to 31% on a separate 16-scenario study. See the report for
the full methodology, the diagnostic trace behind each check, and the caveats.

## Setup

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/), Docker (or
[Colima](https://github.com/abiosoft/colima) on macOS without Docker Desktop), and an
[OpenRouter](https://openrouter.ai/) API key.

```bash
# 1. Install dependencies
uv sync

# 2. Configure environment
cp .env.public .env
# Edit .env and set:
#   LITELLM_API_KEY=<your OpenRouter key>
#   LITELLM_BASE_URL=https://openrouter.ai/api/v1
# Model ids then look like litellm_proxy/openai/gpt-4o-mini,
# litellm_proxy/anthropic/claude-sonnet-4.5, litellm_proxy/z-ai/glm-4.6, etc.

# 3. Start CouchDB (backs the iot/fmsr/wo servers)
colima start                                        # skip if using Docker Desktop
docker compose -f src/couchdb/docker-compose.yaml up -d

# 4. Seed the custom scenario data (assets, sensors, failure modes, work orders)
uv run python -m couchdb.init_data custom --reset

# 5. Materialize the TSFM CSVs the tsfm-heavy scenarios read from
uv run python -m couchdb.materialize_tsfm
```

Verify the stack is healthy:

```bash
curl -s http://localhost:5984/ | head -c 200          # CouchDB
uv run pytest src/ -k "not integration" -q            # unit tests
```

## Reproducing the results

All three studies run through `benchmark.simple_batch`, an in-process batch runner over
`SimpleAgentRunner` (see [How the harness works](#how-the-harness-works) below). Each run
prints a per-check pass/fail breakdown and a cost summary; pass `--report <path>` to also
write a JSON report.

### 1. Structural checks vs. correctness (10 hand-graded scenarios)

```bash
uv run python -m benchmark.simple_batch \
  --scenarios src/couchdb/scenarios_data/custom/scenarios.jsonl \
  --model-id litellm_proxy/openai/gpt-4o-mini \
  --max-turns 8
```

Correctness against the hand-graded ground truth (not itself a deterministic check) is
where the reported 80%→100% figure comes from — see the report for how that was scored.

### 2. Baseline vs. trust layer (16 predictive-maintenance scenarios)

```bash
# Baseline: verifiers and the pre-finalization gate disabled
uv run python -m benchmark.simple_batch \
  --scenarios src/couchdb/scenarios_data/scenario_custom/predictive_maintenance_16.jsonl \
  --model-id litellm_proxy/openai/gpt-4o-mini \
  --max-turns 12 --no-verifiers

# Trust layer: default (verifiers + gate enabled)
uv run python -m benchmark.simple_batch \
  --scenarios src/couchdb/scenarios_data/scenario_custom/predictive_maintenance_16.jsonl \
  --model-id litellm_proxy/openai/gpt-4o-mini \
  --max-turns 12
```

### 3. Cost vs. accuracy across models (12-scenario sample)

Run once per model — trust layer is on by default in all three:

```bash
uv run python -m benchmark.simple_batch \
  --scenarios src/couchdb/scenarios_data/scenario_custom/frontier_sample_12.jsonl \
  --model-id litellm_proxy/openai/gpt-4o-mini \
  --max-turns 12 --report traces/reports/gpt4o_mini.json

uv run python -m benchmark.simple_batch \
  --scenarios src/couchdb/scenarios_data/scenario_custom/frontier_sample_12.jsonl \
  --model-id litellm_proxy/z-ai/glm-4.6 \
  --max-turns 12 --report traces/reports/glm46.json

uv run python -m benchmark.simple_batch \
  --scenarios src/couchdb/scenarios_data/scenario_custom/frontier_sample_12.jsonl \
  --model-id litellm_proxy/anthropic/claude-sonnet-4.5 \
  --max-turns 12 --report traces/reports/sonnet45.json
```

> **Note on reproducibility:** these are real model calls, not replays. Expect run-to-run
> variance from the models themselves — the report documents one instance of this
> (Claude Sonnet 4.5's clean-run count moved between two otherwise-identical runs). Read
> the percentages as directional, not exact reruns.

> **If `uv run` starts failing with `ModuleNotFoundError: No module named 'benchmark'`**
> after a `uv sync`, the editable install's metadata has raced itself — this is a known
> `uv` quirk with rapid successive invocations, not a code bug. Fix: `rm -rf .venv && uv sync`.

## How the harness works

`SimpleAgentRunner` (`src/agent/simple_agent/`) is a minimal in-process ReAct-style tool-
calling loop: one model, one shared conversation thread, connected to a scenario-scoped
subset of MCP servers via `mcphub.ToolUniverse`. It was built to sidestep two failure
modes observed in the upstream runners — an isolated per-step tool-argument resolution
that never sees the tool's actual error, and a full ~86-tool catalog that broke tool-
calling entirely with a smaller model.

The trust layer wraps that loop in two places:

- **Post-hoc checks** (`src/evaluation/checks.py`) score a completed trajectory after
  the fact — no LLM judge, pure functions over the tool-call log.
- **In-loop verifiers** (`src/agent/simple_agent/verifiers.py`) run during the loop and
  inject a corrective observation back into the conversation when a tool call looks
  ungrounded, using the same mechanism the loop already uses to surface tool errors.

The full mechanism-by-mechanism breakdown — what each check verifies, why it exists, and
how it was validated against a broader scenario set to avoid overfitting to the specific
failure that motivated it — is in the report.

## Repo layout

```
src/agent/simple_agent/   SimpleAgentRunner, in-loop verifiers, schema patches
src/evaluation/checks.py  post-hoc deterministic checks
src/benchmark/            in-process batch runner (simple_batch.py) + subprocess harness
src/couchdb/              seed data loader, TSFM materialization, custom scenario sets
src/servers/               MCP servers: iot, fmsr, tsfm, wo, vibration
src/mcphub/                ToolUniverse — the MCP client used directly by the harness
src/observability/         trajectory persistence used by both harnesses
reports/                   the written report (PDF, artifact HTML, Substack export)
```

## Tests

```bash
uv run pytest src/ -k "not integration" -q
```

A handful of pre-existing failures are unrelated to this project's changes: six
`car_score` tests in `evaluation/tests/test_static_json_scorer.py` (a pre-existing bug
in an upstream scorer path this project doesn't use) and two `observability` tests that
need `google-protobuf` installed. Everything else should pass.

## What was trimmed from upstream

This fork removes the parts of the original framework this project never exercised, to
keep the repo focused on the trust-layer work above:

- The `claude_agent`, `deep_agent`, `direct_llm_agent`, `openai_agent`, `opencode_agent`,
  `plan_execute`, and `stirrup_agent` runners, and their heavy dependencies (LangChain,
  DeepAgents, the OpenAI Agents SDK, the Claude Agent SDK, Stirrup).
- The `utilities` MCP server (unused by every scenario evaluated here).
- IBM's own stirrup-specific benchmark-suite runner and its `benchmarks/` scenario
  profiles, and the TSFM showcase notebooks under `notebook/`.

`upstream` still points at `github.com/IBM/AssetOpsBench` if you want to diff against or
pull from the original.

## License

Apache 2.0, inherited from the upstream project — see [`LICENSE`](LICENSE).
