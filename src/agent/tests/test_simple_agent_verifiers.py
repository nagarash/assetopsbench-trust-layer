"""Tests for the in-loop grounding verifiers."""

from __future__ import annotations

from agent.models import ToolCall
from agent.simple_agent.verifiers import verify_before_finalizing, verify_tool_call


class TestFailureModeAssetClassGrounding:
    def test_no_nudge_when_no_asset_detail_call_yet(self):
        tc = ToolCall(name="fmsr.get_failure_modes", input={"asset_class": "compressor"})
        nudge = verify_tool_call(
            tc,
            question="List all failure modes of asset Chiller 6.",
            prior_tool_calls=[tc],
        )
        assert nudge is not None
        assert "iot.asset_detail" in nudge

    def test_flags_mismatch_against_a_confirmed_assettype(self):
        detail = ToolCall(
            name="iot.asset_detail",
            input={"asset_id": "Chiller 6"},
            output={"result": {"assettype": "CHILLER"}},
        )
        fmsr = ToolCall(name="fmsr.get_failure_modes", input={"asset_class": "compressor"})
        nudge = verify_tool_call(
            fmsr,
            question="List all failure modes of asset Chiller 6.",
            prior_tool_calls=[detail, fmsr],
        )
        assert nudge is not None
        assert "CHILLER" in nudge
        assert "compressor" in nudge

    def test_no_nudge_when_asset_class_matches_confirmed_type(self):
        detail = ToolCall(
            name="iot.asset_detail",
            input={"asset_id": "Chiller 6"},
            output={"result": {"assettype": "CHILLER"}},
        )
        fmsr = ToolCall(name="fmsr.get_failure_modes", input={"asset_class": "chiller"})
        nudge = verify_tool_call(
            fmsr,
            question="List all failure modes of asset Chiller 6.",
            prior_tool_calls=[detail, fmsr],
        )
        assert nudge is None

    def test_no_nudge_without_an_asset_class_arg(self):
        tc = ToolCall(name="fmsr.get_failure_modes", input={})
        nudge = verify_tool_call(
            tc,
            question="List all failure modes of asset Chiller 6.",
            prior_tool_calls=[tc],
        )
        assert nudge is None

    def test_flags_mismatch_when_asset_detail_arrives_after_the_bad_fmsr_call(self):
        # The real production sequence: the model guesses asset_class wrong,
        # gets nudged to confirm, THEN calls asset_detail — the earlier fmsr
        # call is already in the past and must be re-checked against this
        # newly-arriving evidence, not just evidence that predates it.
        fmsr = ToolCall(name="fmsr.get_failure_modes", input={"asset_class": "compressor"})
        detail = ToolCall(
            name="iot.asset_detail",
            input={"asset_id": "Chiller 6"},
            output={"result": {"assettype": "CHILLER"}},
        )
        nudge = verify_tool_call(
            detail,
            question="List all failure modes of asset Chiller 6.",
            prior_tool_calls=[fmsr, detail],
        )
        assert nudge is not None
        assert "CHILLER" in nudge
        assert "compressor" in nudge

    def test_no_nudge_from_asset_detail_when_no_prior_fmsr_call(self):
        detail = ToolCall(
            name="iot.asset_detail",
            input={"asset_id": "Chiller 6"},
            output={"result": {"assettype": "CHILLER"}},
        )
        nudge = verify_tool_call(
            detail,
            question="List all failure modes of asset Chiller 6.",
            prior_tool_calls=[detail],
        )
        assert nudge is None

    def test_no_nudge_from_asset_detail_when_prior_fmsr_call_already_matched(self):
        fmsr = ToolCall(name="fmsr.get_failure_modes", input={"asset_class": "chiller"})
        detail = ToolCall(
            name="iot.asset_detail",
            input={"asset_id": "Chiller 6"},
            output={"result": {"assettype": "CHILLER"}},
        )
        nudge = verify_tool_call(
            detail,
            question="List all failure modes of asset Chiller 6.",
            prior_tool_calls=[fmsr, detail],
        )
        assert nudge is None


class TestWorkOrderFilterGrounding:
    def test_flags_unrequested_status_filter(self):
        tc = ToolCall(
            name="wo.list_workorders", input={"asset_num": "MP1", "status": "OPEN"}
        )
        nudge = verify_tool_call(
            tc,
            question="Is there a preventive maintenance work order recorded for equipment MP1?",
            prior_tool_calls=[tc],
        )
        assert nudge is not None
        assert "status='OPEN'" in nudge

    def test_no_nudge_when_status_was_requested(self):
        tc = ToolCall(
            name="wo.list_workorders", input={"asset_num": "MP1", "status": "OPEN"}
        )
        nudge = verify_tool_call(
            tc,
            question="Show me the OPEN work orders for equipment MP1.",
            prior_tool_calls=[tc],
        )
        assert nudge is None

    def test_no_nudge_without_a_narrowing_param(self):
        tc = ToolCall(name="wo.list_workorders", input={"asset_num": "MP1"})
        nudge = verify_tool_call(
            tc,
            question="Get the work order for equipment MP1.",
            prior_tool_calls=[tc],
        )
        assert nudge is None

    def test_ignores_non_wo_tools(self):
        tc = ToolCall(name="iot.history", input={"status": "OPEN"})
        nudge = verify_tool_call(
            tc,
            question="What is the status?",
            prior_tool_calls=[tc],
        )
        assert nudge is None


class TestVerifyBeforeFinalizing:
    def test_no_pending_when_no_predictive_or_maintenance_language(self):
        pending = verify_before_finalizing(
            "What sensors are installed on Motor 12?", []
        )
        assert pending == []

    def test_flags_missing_tsfm_for_predictive_language(self):
        calls = [ToolCall(name="iot.history", input={})]
        pending = verify_before_finalizing(
            "Could this motor experience a bearing failure soon?", calls
        )
        assert [cat for cat, _ in pending] == ["tsfm_predictive"]

    def test_flags_missing_wo_for_maintenance_language(self):
        calls = [ToolCall(name="iot.history", input={})]
        pending = verify_before_finalizing(
            "Should maintenance be scheduled for this motor?", calls
        )
        assert [cat for cat, _ in pending] == ["wo_maintenance"]

    def test_flags_both_when_both_implied_and_both_missing(self):
        calls = [ToolCall(name="iot.history", input={})]
        pending = verify_before_finalizing(
            "Could this motor fail soon, and should maintenance be scheduled?",
            calls,
        )
        assert {cat for cat, _ in pending} == {"tsfm_predictive", "wo_maintenance"}

    def test_no_nudge_when_tsfm_already_called(self):
        calls = [
            ToolCall(name="iot.history", input={}),
            ToolCall(name="tsfm.run_recipe", input={}),
        ]
        pending = verify_before_finalizing(
            "Could this motor experience a bearing failure soon?", calls
        )
        assert pending == []

    def test_no_nudge_when_wo_already_called(self):
        calls = [
            ToolCall(name="iot.history", input={}),
            ToolCall(name="wo.generate_work_order", input={}),
        ]
        pending = verify_before_finalizing(
            "Should maintenance be scheduled for this motor?", calls
        )
        assert pending == []

    def test_flags_missing_fmsr_for_failure_mode_language(self):
        calls = [ToolCall(name="iot.history", input={})]
        pending = verify_before_finalizing(
            "List all failure modes of this motor.", calls
        )
        assert [cat for cat, _ in pending] == ["fmsr_diagnostic"]

    def test_no_nudge_when_fmsr_already_called(self):
        calls = [
            ToolCall(name="iot.history", input={}),
            ToolCall(name="fmsr.get_failure_modes", input={}),
        ]
        pending = verify_before_finalizing(
            "List all failure modes of this motor.", calls
        )
        assert pending == []


class TestTsfmEmptyCatalogGrounding:
    def test_flags_empty_find_models_result_on_first_try(self):
        tc = ToolCall(
            name="tsfm.find_models",
            input={"task_id": "tsfm_anomaly_detection"},
            output={"result": {"models": []}},
        )
        nudge = verify_tool_call(tc, question="q", prior_tool_calls=[tc])
        assert nudge is not None
        assert "no results" in nudge
        assert "do not retry" in nudge

    def test_flags_empty_search_models_result(self):
        tc = ToolCall(
            name="tsfm.search_models",
            input={"text": "air leak"},
            output={"result": {"models": []}},
        )
        nudge = verify_tool_call(tc, question="q", prior_tool_calls=[tc])
        assert nudge is not None

    def test_flags_empty_describe_candidates_result(self):
        tc = ToolCall(
            name="tsfm.describe_candidates",
            input={"task_id": "tsfm_anomaly_detection"},
            output={"result": {"candidates": []}},
        )
        nudge = verify_tool_call(tc, question="q", prior_tool_calls=[tc])
        assert nudge is not None

    def test_no_nudge_when_catalog_has_results(self):
        tc = ToolCall(
            name="tsfm.find_models",
            input={"task_id": "tsfm_forecasting"},
            output={"result": {"models": [{"model_id": "ttm_96_28"}]}},
        )
        nudge = verify_tool_call(tc, question="q", prior_tool_calls=[tc])
        assert nudge is None

    def test_no_nudge_for_non_catalog_tsfm_tools(self):
        tc = ToolCall(
            name="tsfm.run_recipe",
            input={},
            output={"result": {"status": "success"}},
        )
        nudge = verify_tool_call(tc, question="q", prior_tool_calls=[tc])
        assert nudge is None
