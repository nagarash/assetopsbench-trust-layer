"""Tests for client-side tool-parameter schema enrichment."""

from __future__ import annotations

from agent.simple_agent.schema_patches import apply_schema_patches


class TestApplySchemaPatches:
    def test_unpatched_tool_passes_through_unchanged(self):
        params = {"type": "object", "properties": {"asset_class": {"type": "string"}}}
        result = apply_schema_patches("fmsr.get_failure_modes", params)
        assert result is params

    def test_run_recipe_recipe_property_gets_real_structure(self):
        params = {
            "type": "object",
            "properties": {
                "dataset_path": {"type": "string"},
                "recipe": {"type": "object", "additionalProperties": True},
            },
            "required": ["dataset_path", "recipe"],
        }
        result = apply_schema_patches("tsfm.run_recipe", params)
        recipe_schema = result["properties"]["recipe"]
        assert "estimator" in recipe_schema["properties"]
        assert "model_id" in recipe_schema["properties"]["estimator"]["properties"]
        assert "sktime_class" in recipe_schema["properties"]["estimator"]["properties"]
        assert "ensemble" in recipe_schema["properties"]

    def test_other_properties_on_a_patched_tool_are_untouched(self):
        params = {
            "type": "object",
            "properties": {
                "dataset_path": {"type": "string", "title": "Dataset Path"},
                "recipe": {"type": "object", "additionalProperties": True},
            },
        }
        result = apply_schema_patches("tsfm.run_recipe", params)
        assert result["properties"]["dataset_path"] == {
            "type": "string",
            "title": "Dataset Path",
        }

    def test_does_not_mutate_the_input_dict(self):
        params = {
            "type": "object",
            "properties": {"recipe": {"type": "object", "additionalProperties": True}},
        }
        apply_schema_patches("tsfm.run_recipe", params)
        assert params["properties"]["recipe"] == {
            "type": "object",
            "additionalProperties": True,
        }

    def test_missing_property_is_not_injected(self):
        # A tool spec that doesn't even have a `recipe` property shouldn't
        # gain one — only enrich what the server actually declared.
        params = {"type": "object", "properties": {"dataset_path": {"type": "string"}}}
        result = apply_schema_patches("tsfm.run_recipe", params)
        assert "recipe" not in result["properties"]

    def test_impute_enum_matches_the_tools_valid_values(self):
        params = {
            "type": "object",
            "properties": {"recipe": {"type": "object", "additionalProperties": True}},
        }
        result = apply_schema_patches("tsfm.run_recipe", params)
        impute_schema = result["properties"]["recipe"]["properties"]["impute"]
        assert set(impute_schema["enum"]) == {"interpolate", "drop", "zero"}
