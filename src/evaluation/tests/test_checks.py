"""Tests for the deterministic (no-LLM) trust checks in evaluation.checks."""

from __future__ import annotations

from evaluation.checks import (
    check_answer_shape_matches_characteristic_form,
    check_expected_servers_invoked,
    check_no_silent_failures,
    check_predictive_task_uses_expected_tools,
    check_tool_args_match_schema,
    run_checks,
)

_PLAN_EXECUTE_TRAJ_OK = [
    {
        "step_number": 1,
        "task": "list sites",
        "server": "iot",
        "tool": "sites",
        "tool_args": {},
        "response": '["MAIN"]',
        "error": None,
    },
]

_PLAN_EXECUTE_TRAJ_ERROR = [
    {
        "step_number": 1,
        "task": "list sites",
        "server": "iot",
        "tool": "sites",
        "tool_args": {},
        "response": "",
        "error": "connection refused",
    },
]

# Real failure mode seen in production: the plan-execute planner hallucinates
# an invalid server name ("none"); the executor rejects it before ever
# resolving a tool, so the failed step's `tool` field is empty. This must
# still be caught, not silently dropped for lacking a `tool` value.
_PLAN_EXECUTE_TRAJ_UNRESOLVED_SERVER = [
    {
        "step_number": 1,
        "task": "list sites",
        "server": "iot",
        "tool": "sites",
        "tool_args": {},
        "response": '["MAIN"]',
        "error": None,
    },
    {
        "step_number": 2,
        "task": "do something the planner invented",
        "server": "none",
        "tool": "",
        "tool_args": {},
        "response": "",
        "error": "Unknown server 'none'. Registered servers: ['iot', 'wo']",
    },
]

# A step the planner legitimately marks as reasoning-only (no tool needed,
# no error) must still be skipped, not treated as a failure.
_PLAN_EXECUTE_TRAJ_NO_TOOL_NEEDED = [
    {
        "step_number": 1,
        "task": "summarize prior results",
        "server": "",
        "tool": "none",
        "tool_args": {},
        "response": "Based on the above, the answer is X.",
        "error": None,
    },
]

_SDK_TRAJ_OK = {
    "turns": [
        {
            "index": 0,
            "text": "",
            "tool_calls": [
                {"name": "mcp__iot__sites", "input": {}, "output": '["MAIN"]'}
            ],
        }
    ]
}

_SDK_TRAJ_ERROR_OUTPUT = {
    "turns": [
        {
            "index": 0,
            "text": "",
            "tool_calls": [
                {
                    "name": "mcp__iot__sites",
                    "input": {},
                    "output": "Error: asset not found",
                }
            ],
        }
    ]
}


class TestNoSilentFailures:
    def test_skips_when_no_tool_calls(self):
        r = check_no_silent_failures([])
        assert r.passed is None

    def test_passes_when_plan_execute_step_has_no_error(self):
        r = check_no_silent_failures(_PLAN_EXECUTE_TRAJ_OK)
        assert r.passed is True

    def test_fails_when_plan_execute_step_has_error(self):
        r = check_no_silent_failures(_PLAN_EXECUTE_TRAJ_ERROR)
        assert r.passed is False
        assert "connection refused" in r.detail

    def test_passes_on_clean_sdk_output(self):
        r = check_no_silent_failures(_SDK_TRAJ_OK)
        assert r.passed is True

    def test_flags_error_like_sdk_output_heuristically(self):
        r = check_no_silent_failures(_SDK_TRAJ_ERROR_OUTPUT)
        assert r.passed is False
        assert "heuristic" in r.detail

    def test_catches_step_that_failed_before_resolving_a_tool(self):
        r = check_no_silent_failures(_PLAN_EXECUTE_TRAJ_UNRESOLVED_SERVER)
        assert r.passed is False
        assert "Unknown server" in r.detail

    def test_skips_legitimate_no_tool_needed_step(self):
        r = check_no_silent_failures(_PLAN_EXECUTE_TRAJ_NO_TOOL_NEEDED)
        assert r.passed is None

    def test_catches_json_error_body_in_a_nominally_successful_step(self):
        # Real shape seen in production: the MCP tool call itself succeeds
        # (StepResult.error is None) but the response body it returns is
        # {"error": "..."} — several servers use this convention.
        traj = [
            {
                "step_number": 1,
                "task": "run a forecast recipe",
                "server": "tsfm",
                "tool": "run_recipe",
                "tool_args": {},
                "response": '{\n  "error": "recipe must include an \'estimator\' or an \'ensemble\'"\n}',
                "error": None,
            }
        ]
        r = check_no_silent_failures(traj)
        assert r.passed is False
        assert "heuristic" in r.detail

    def test_does_not_flag_a_response_that_merely_mentions_the_word_error(self):
        # A JSON body without a truthy top-level "error" key should not
        # false-positive just because "{" starts the string.
        traj = [
            {
                "step_number": 1,
                "task": "list sensors",
                "server": "iot",
                "tool": "measured_sensors",
                "tool_args": {},
                "response": '{"sensors": ["Error Rate Sensor"], "count": 1}',
                "error": None,
            }
        ]
        r = check_no_silent_failures(traj)
        assert r.passed is True


class TestExpectedServersInvoked:
    def test_skips_without_hint_field(self, make_scenario):
        r = check_expected_servers_invoked(make_scenario(), _PLAN_EXECUTE_TRAJ_OK)
        assert r.passed is None

    def test_passes_when_hinted_server_was_invoked(self, make_scenario):
        s = make_scenario(hint="IoT Agent handles raw sensor data ingestion.")
        r = check_expected_servers_invoked(s, _PLAN_EXECUTE_TRAJ_OK)
        assert r.passed is True

    def test_fails_when_hinted_server_was_not_invoked(self, make_scenario):
        s = make_scenario(
            hint=(
                "IoT Agent handles ingestion; TSFM Agent detects anomalies; "
                "FMSR Agent interprets failure modes; WO Agent plans maintenance."
            )
        )
        r = check_expected_servers_invoked(s, _PLAN_EXECUTE_TRAJ_OK)
        assert r.passed is False
        assert "tsfm" in r.detail
        assert "fmsr" in r.detail
        assert "wo" in r.detail

    def test_works_against_sdk_trajectory_shape(self, make_scenario):
        s = make_scenario(hint="IoT Agent handles raw sensor data ingestion.")
        r = check_expected_servers_invoked(s, _SDK_TRAJ_OK)
        assert r.passed is True

    def test_uses_structured_servers_list_when_present_even_without_hint(
        self, make_scenario
    ):
        s = make_scenario(servers=["iot"])
        r = check_expected_servers_invoked(s, _PLAN_EXECUTE_TRAJ_OK)
        assert r.passed is True
        assert "servers" in r.detail

    def test_fails_against_structured_servers_list_when_one_is_missing(
        self, make_scenario
    ):
        s = make_scenario(servers=["iot", "fmsr"])
        r = check_expected_servers_invoked(s, _PLAN_EXECUTE_TRAJ_OK)
        assert r.passed is False
        assert "fmsr" in r.detail

    def test_prefers_structured_servers_list_over_hint_when_both_present(
        self, make_scenario
    ):
        # servers includes 'iot' as an implicit ingestion prerequisite that
        # the free-text hint never names — the real disagreement pattern
        # found across the scenario corpus (servers is always a superset).
        s = make_scenario(
            servers=["iot", "fmsr"],
            hint="FMSR Agent interprets failure modes.",
        )
        traj = [
            {
                "step_number": 1,
                "task": "get failure modes",
                "server": "fmsr",
                "tool": "get_failure_modes",
                "tool_args": {},
                "response": "{}",
                "error": None,
            }
        ]
        r = check_expected_servers_invoked(s, traj)
        assert r.passed is False
        assert "iot" in r.detail

    def test_falls_back_to_hint_when_servers_field_is_absent(self, make_scenario):
        s = make_scenario(hint="IoT Agent handles raw sensor data ingestion.")
        r = check_expected_servers_invoked(s, _PLAN_EXECUTE_TRAJ_OK)
        assert r.passed is True
        assert "hint" in r.detail

    def test_falls_back_to_hint_when_servers_field_is_an_empty_list(
        self, make_scenario
    ):
        s = make_scenario(servers=[], hint="IoT Agent handles raw sensor data ingestion.")
        r = check_expected_servers_invoked(s, _PLAN_EXECUTE_TRAJ_OK)
        assert r.passed is True
        assert "hint" in r.detail

class TestToolArgsMatchSchema:
    def test_skips_without_schemas(self):
        r = check_tool_args_match_schema(_PLAN_EXECUTE_TRAJ_OK, None)
        assert r.passed is None

    def test_skips_when_no_call_matches_a_known_schema(self):
        schemas = {"wo": {"generate_work_order": {"required": ["asset_id"]}}}
        r = check_tool_args_match_schema(_PLAN_EXECUTE_TRAJ_OK, schemas)
        assert r.passed is None

    def test_passes_when_required_args_present(self):
        schemas = {"iot": {"sites": {"required": []}}}
        r = check_tool_args_match_schema(_PLAN_EXECUTE_TRAJ_OK, schemas)
        assert r.passed is True

    def test_fails_when_required_arg_missing(self):
        traj = [
            {
                "step_number": 1,
                "task": "get sensors",
                "server": "iot",
                "tool": "installed_sensors",
                "tool_args": {},
                "response": "",
                "error": None,
            }
        ]
        schemas = {"iot": {"installed_sensors": {"required": ["asset_id"]}}}
        r = check_tool_args_match_schema(traj, schemas)
        assert r.passed is False
        assert "asset_id" in r.detail


class TestAnswerShapeMatchesCharacteristicForm:
    def test_skips_without_characteristic_form(self, make_scenario):
        s = make_scenario(characteristic_form=None)
        r = check_answer_shape_matches_characteristic_form(s, "42")
        assert r.passed is None

    def test_skips_when_no_shape_keyword(self, make_scenario):
        s = make_scenario(characteristic_form="Should be a friendly summary.")
        r = check_answer_shape_matches_characteristic_form(s, "Sure thing.")
        assert r.passed is None

    def test_passes_when_answer_is_a_json_object(self, make_scenario):
        s = make_scenario(
            characteristic_form="The response should be a JSON object with site names."
        )
        r = check_answer_shape_matches_characteristic_form(s, '{"site": "MAIN"}')
        assert r.passed is True

    def test_fails_when_answer_is_not_a_list(self, make_scenario):
        s = make_scenario(characteristic_form="Return a list of all failure modes.")
        r = check_answer_shape_matches_characteristic_form(s, "just some prose")
        assert r.passed is False

    def test_skips_bare_verb_list_as_prose_not_shape(self, make_scenario):
        # "should list X" asks for an enumeration in prose, not a literal
        # JSON array — must not trigger the list-shape check.
        s = make_scenario(
            characteristic_form="The response should list the installed sensors."
        )
        r = check_answer_shape_matches_characteristic_form(
            s, "Chiller 6 has these sensors: Tonnage, Power Input, Efficiency."
        )
        assert r.passed is None

    def test_skips_noun_compound_list_as_prose_not_shape(self, make_scenario):
        s = make_scenario(
            characteristic_form="Retrieve the sensor list and summarize it."
        )
        r = check_answer_shape_matches_characteristic_form(s, "Some prose summary.")
        assert r.passed is None


class TestToolArgsMatchSchemaSdkTrajectory:
    def test_matches_claude_agent_style_qualified_name(self):
        # mcp__<server>__<tool> — the tool name itself has an internal
        # underscore, which must survive the split/rejoin to match the
        # schema's bare key.
        traj = {
            "turns": [
                {
                    "index": 0,
                    "tool_calls": [
                        {"name": "mcp__iot__asset_ids", "input": {"site_name": "MAIN"}}
                    ],
                }
            ]
        }
        schemas = {"iot": {"asset_ids": {"required": ["site_name"]}}}
        r = check_tool_args_match_schema(traj, schemas)
        assert r.passed is True

    def test_matches_dot_qualified_name(self):
        traj = {
            "turns": [
                {
                    "index": 0,
                    "tool_calls": [
                        {"name": "iot.installed_sensors", "input": {}}
                    ],
                }
            ]
        }
        schemas = {"iot": {"installed_sensors": {"required": ["asset_id"]}}}
        r = check_tool_args_match_schema(traj, schemas)
        assert r.passed is False
        assert "asset_id" in r.detail


# Deliberately a different asset and phrasing style than any scenario used to
# motivate this check — this is a generalization test, not a regression test
# for the specific runs that surfaced the failure pattern.
_MOTOR_TRAJ_IOT_ONLY = [
    {
        "step_number": 1,
        "task": "pull vibration history",
        "server": "iot",
        "tool": "history",
        "tool_args": {},
        "response": "[]",
        "error": None,
    }
]

_MOTOR_TRAJ_WITH_TSFM_AND_WO = [
    {
        "step_number": 1,
        "task": "pull vibration history",
        "server": "iot",
        "tool": "history",
        "tool_args": {},
        "response": "[]",
        "error": None,
    },
    {
        "step_number": 2,
        "task": "run anomaly detection",
        "server": "tsfm",
        "tool": "run_recipe",
        "tool_args": {},
        "response": "{}",
        "error": None,
    },
    {
        "step_number": 3,
        "task": "log a repair job",
        "server": "wo",
        "tool": "generate_work_order",
        "tool_args": {},
        "response": "{}",
        "error": None,
    },
]


class TestPredictiveTaskUsesExpectedTools:
    def test_skips_when_question_has_no_predictive_or_maintenance_language(
        self, make_scenario
    ):
        s = make_scenario(text="What sensors are installed on Motor 12?")
        r = check_predictive_task_uses_expected_tools(s, _MOTOR_TRAJ_IOT_ONLY)
        assert r.passed is None

    def test_fails_when_predictive_language_present_but_tsfm_never_called(
        self, make_scenario
    ):
        s = make_scenario(
            text=(
                "Given the last week of vibration readings, could this motor "
                "experience a bearing failure soon?"
            )
        )
        r = check_predictive_task_uses_expected_tools(s, _MOTOR_TRAJ_IOT_ONLY)
        assert r.passed is False
        assert "tsfm" in r.detail

    def test_fails_when_maintenance_language_present_but_wo_never_called(
        self, make_scenario
    ):
        s = make_scenario(
            text="Should maintenance be scheduled for this motor this week?"
        )
        r = check_predictive_task_uses_expected_tools(s, _MOTOR_TRAJ_IOT_ONLY)
        assert r.passed is False
        assert "wo" in r.detail

    def test_passes_when_both_signals_present_and_both_tools_called(
        self, make_scenario
    ):
        s = make_scenario(
            text=(
                "Given the last week of vibration readings, could this motor "
                "experience a bearing failure soon, and do we need to log a "
                "repair job?"
            )
        )
        r = check_predictive_task_uses_expected_tools(
            s, _MOTOR_TRAJ_WITH_TSFM_AND_WO
        )
        assert r.passed is True

    def test_within_the_next_phrasing_implies_tsfm(self, make_scenario):
        s = make_scenario(
            text="Will this pump fail within the next 48 hours?"
        )
        r = check_predictive_task_uses_expected_tools(s, _MOTOR_TRAJ_IOT_ONLY)
        assert r.passed is False
        assert "tsfm" in r.detail

    def test_failure_mode_language_implies_fmsr(self, make_scenario):
        s = make_scenario(text="List all failure modes of this motor.")
        r = check_predictive_task_uses_expected_tools(s, _MOTOR_TRAJ_IOT_ONLY)
        assert r.passed is False
        assert "fmsr" in r.detail

    def test_root_cause_language_implies_fmsr(self, make_scenario):
        s = make_scenario(text="What is the root cause of this vibration pattern?")
        r = check_predictive_task_uses_expected_tools(s, _MOTOR_TRAJ_IOT_ONLY)
        assert r.passed is False
        assert "fmsr" in r.detail

    def test_no_nudge_when_fmsr_already_called(self, make_scenario):
        s = make_scenario(text="List all failure modes of this motor.")
        r = check_predictive_task_uses_expected_tools(s, _MOTOR_TRAJ_WITH_TSFM_AND_WO)
        # fmsr wasn't called in this fixture trajectory either, so this
        # should still fail — asserting the negative would be a tautology.
        assert r.passed is False
        assert "fmsr" in r.detail

    def test_named_failure_type_does_not_trigger_fmsr(self, make_scenario):
        # Deliberately narrow: a specific named failure type ("bearing
        # failure") embedded in a risk question is not the same signal as
        # explicit "failure mode" language — see the check's own docstring
        # for why broadening this trades away more than it buys.
        s = make_scenario(
            text="Is this asset at risk of a bearing failure within the next week?"
        )
        r = check_predictive_task_uses_expected_tools(s, _MOTOR_TRAJ_IOT_ONLY)
        assert "fmsr" not in (r.detail or "")


class TestRunChecks:
    def test_returns_all_five_checks(self, make_scenario):
        checks = run_checks(make_scenario(), "answer", _PLAN_EXECUTE_TRAJ_OK)
        assert {c.name for c in checks} == {
            "no_silent_failures",
            "expected_servers_invoked",
            "tool_args_match_schema",
            "answer_shape_matches_characteristic_form",
            "predictive_task_uses_expected_tools",
        }
