"""Pre-materialize seeded telemetry into TSFM-readable CSV files.

There is no MCP tool anywhere in this framework that bridges live IoT
telemetry into TSFM's file-based tools (``profile_series``, ``run_recipe``,
etc. all require a ``dataset_path`` as *input*; nothing produces one from a
live query). ``servers.tsfm.io.refs.materialize_iot`` looks like it should
be that bridge, but it carries no ``@mcp.tool`` decorator — it's a Python
test helper, not something any agent can call.

Rather than build and maintain a new server-side tool for this, this script
writes the same telemetry already seeded into CouchDB (see
``scenario_custom/manifest.json``) to fixed, predictable ``file://`` paths
under ``TSFM_WORKDIR`` — so the tsfm tool chain has something real to read,
consistent with whatever ``iot.history``/``iot.measured_sensors`` report for
the same asset. Run it once after (re)loading the ``custom`` CouchDB
scenario; the paths are stable across runs as long as the source JSON files
don't change.

Usage:
    uv run python -m couchdb.materialize_tsfm
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]

WORKDIR = Path(os.environ.get("TSFM_WORKDIR", "/tmp/tsfm_work"))

# asset_id -> source JSON file (same files loaded into the `iot` CouchDB
# collection for the `custom` scenario manifest).
ASSET_SOURCES: dict[str, Path] = {
    "Chiller 6": _HERE / "scenarios_data/shared/iot/chiller_6.json",
    "mp_1": _HERE / "scenarios_data/shared/iot/metro_pump_1.json",
    "hp_1": _HERE / "scenarios_data/scenario_custom/hydraulic_pump_hp1.json",
}


def _safe_filename(asset_id: str) -> str:
    return asset_id.replace(" ", "_").replace("/", "_") + ".csv"


def dataset_path_for(asset_id: str) -> str | None:
    """Return the file:// pointer for asset_id if it's a known, materializable asset."""
    if asset_id not in ASSET_SOURCES:
        return None
    return f"file://{WORKDIR / _safe_filename(asset_id)}"


def materialize() -> dict[str, str]:
    """Write each known asset's telemetry to a CSV and return {asset_id: file://path}."""
    WORKDIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for asset_id, src in ASSET_SOURCES.items():
        records = json.loads(src.read_text(encoding="utf-8"))
        df = pd.DataFrame(records)
        df = df.drop(columns=["asset_id"], errors="ignore")
        df = df.sort_values("timestamp")
        ordered_cols = ["timestamp"] + [c for c in df.columns if c != "timestamp"]
        df = df[ordered_cols]

        out_path = WORKDIR / _safe_filename(asset_id)
        df.to_csv(out_path, index=False)
        paths[asset_id] = f"file://{out_path}"
        print(f"{asset_id}: {len(df)} rows, {len(df.columns) - 1} sensor(s) -> {out_path}")
    return paths


def main() -> None:
    materialize()


if __name__ == "__main__":
    main()
