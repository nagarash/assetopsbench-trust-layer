"""Client-side JSON-schema enrichment for tool parameters the upstream MCP
servers advertise with no real structure.

``tsfm.run_recipe``'s ``recipe`` parameter is typed
``{"type": "object", "additionalProperties": true}`` by the server — no
nested schema at all for its most complex, most failure-prone argument.
Everything about the correct shape lives in prose (the tool's docstring,
and a separate ``recipe_template()`` tool the model has to think to call)
rather than in anything the function-calling API can validate or hint from.
In production this produced a real, repeatable failure: the model invented
a literal placeholder value (``{"model_id": "<model_id>"}``) instead of
either calling ``recipe_template()`` or using an inline ``sktime_class``.

This module patches the schema client-side, mirroring exactly what
``recipe_template()`` already documents, so the model sees real structure
at the moment it's building the call instead of needing a separate,
easy-to-skip lookup first. It changes only what the model is shown —
``mcphub.ToolUniverse.run`` still passes the resolved argument dict through
to the real ``run_recipe`` tool unchanged, which still accepts the same
loosely-typed ``recipe: dict`` it always has, so nothing server-side (or
for any other runner using these servers) is affected.

The right long-term fix is upstreaming this into the server's own Pydantic
schema; this is a stopgap for this runner in the meantime.
"""

from __future__ import annotations

from typing import Any

_RECIPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "Forecasting/anomaly-detection recipe. Must include exactly one of "
        "'estimator' or 'ensemble'."
    ),
    "properties": {
        "task": {
            "type": "string",
            "description": "Omit for forecasting. Set for other tasks.",
            "enum": [
                "tsfm_anomaly_detection",
                "tsfm_classification",
                "tsfm_regression",
                "tsfm_clustering",
            ],
        },
        "estimator": {
            "type": "object",
            "description": (
                "A single model. Provide exactly one of model_id or "
                "sktime_class+params — never both, and never a placeholder "
                "string. Omit this if using 'ensemble' instead."
            ),
            "properties": {
                "model_id": {
                    "type": "string",
                    "description": (
                        "An id that already exists in the catalog (confirm "
                        "via find_models/search_models first — do not guess "
                        "an id). If no catalog model exists for this task, "
                        "use sktime_class instead of guessing here."
                    ),
                },
                "sktime_class": {
                    "type": "string",
                    "description": (
                        "Fully-qualified class path to use when no catalog "
                        "model applies, e.g. 'sktime.detection.lof.SubLOF' "
                        "for anomaly detection or "
                        "'sktime.forecasting.naive.NaiveForecaster' for "
                        "forecasting."
                    ),
                },
                "params": {
                    "type": "object",
                    "description": "Constructor kwargs for sktime_class.",
                },
            },
        },
        "ensemble": {
            "type": "object",
            "description": (
                "Use INSTEAD of estimator: {members: [<estimator spec>, "
                "...], combine: 'mean'|'median'|'min'|'max'|'weighted'|"
                "'stack', weights: [...]}."
            ),
        },
        "fh": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Forecast horizon steps, e.g. [1,2,3]. Default [1,2,3,4,5].",
        },
        "transforms": {
            "type": "array",
            "description": "Transform specs applied to the target before fitting.",
        },
        "conformal": {
            "type": "object",
            "description": "{coverage: 0.9} for calibrated prediction intervals.",
        },
        "finetune": {
            "type": "object",
            "description": "Training block (lr, epochs, batch_size, ...).",
        },
        "anomaly": {
            "type": "object",
            "description": "Detector block (false_alarm, ad_model_type, window_size, ...).",
        },
        "impute": {
            "type": "string",
            "enum": ["interpolate", "drop", "zero"],
            "description": "Fill gaps before fitting classical forecasters.",
        },
        "eval": {
            "type": "object",
            "description": "{metrics: ['smape', ...]} — the first metric scores the backtest.",
        },
        "save_to": {
            "type": "string",
            "description": "Directory path to persist fitted weights after fine-tuning.",
        },
    },
    "additionalProperties": True,
}

# qualified tool name -> {parameter name -> enriched schema}
TOOL_PARAMETER_SCHEMA_PATCHES: dict[str, dict[str, dict]] = {
    "tsfm.run_recipe": {"recipe": _RECIPE_SCHEMA},
}


def apply_schema_patches(qualified_name: str, parameters: dict) -> dict:
    """Return *parameters* with any registered client-side enrichments applied.

    Only replaces the specific properties this module patches; every other
    property and every other tool passes through completely unchanged, so
    this is safe to call unconditionally for every tool.
    """
    patch = TOOL_PARAMETER_SCHEMA_PATCHES.get(qualified_name)
    if not patch:
        return parameters

    properties = dict(parameters.get("properties") or {})
    for prop_name, schema in patch.items():
        if prop_name in properties:
            properties[prop_name] = schema
    return {**parameters, "properties": properties}
