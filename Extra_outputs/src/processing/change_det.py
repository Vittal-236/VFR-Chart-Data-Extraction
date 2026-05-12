import json
from pathlib import Path
from deepdiff import DeepDiff
from src.utils.logger import get_logger

log = get_logger(__name__)

COORD_DRIFT_THRESHOLD_DEG = 0.01


class ChangeDetector:

    def compare(self, current_path, previous_path):
        current = self._load(current_path)
        previous = self._load(previous_path)

        if not current or not previous:
            return {"error": "Could not load one or both JSON files."}

        diff = DeepDiff(previous, current, ignore_order=True, report_repetition=True)

        report = {
            "added": [],
            "removed": [],
            "modified": [],
            "coordinate_drifts": [],
            "summary": "",
        }

        for path in diff.get("iterable_item_added", {}).values():
            if isinstance(path, dict):
                report["added"].append(str(path))

        for path in diff.get("iterable_item_removed", {}).values():
            if isinstance(path, dict):
                report["removed"].append(str(path))

        for key, change in diff.get("values_changed", {}).items():
            report["modified"].append({
                "field": key,
                "old": change["old_value"],
                "new": change["new_value"],
            })

        report["coordinate_drifts"] = self._detect_drifts(
            previous.get("waypoints", []),
            current.get("waypoints", []),
        )

        n_add = len(report["added"])
        n_rem = len(report["removed"])
        n_mod = len(report["modified"])
        n_drift = len(report["coordinate_drifts"])
        report["summary"] = (
            f"{n_add} added, {n_rem} removed, {n_mod} modified, "
            f"{n_drift} coordinate drift(s) detected."
        )

        log.info(f"Change detection: {report['summary']}")
        return report

    def _load(self, path):
        path = Path(path)
        if not path.exists():
            log.error(f"File not found: {path}")
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _detect_drifts(self, prev_wps, curr_wps):
        drifts = []
        prev_map = {wp.get("name"): wp for wp in prev_wps if wp.get("name")}
        curr_map = {wp.get("name"): wp for wp in curr_wps if wp.get("name")}

        for name, curr in curr_map.items():
            prev = prev_map.get(name)
            if not prev:
                continue
            dlat = abs((curr.get("lat") or 0) - (prev.get("lat") or 0))
            dlon = abs((curr.get("lon") or 0) - (prev.get("lon") or 0))
            if dlat > COORD_DRIFT_THRESHOLD_DEG or dlon > COORD_DRIFT_THRESHOLD_DEG:
                drifts.append({
                    "waypoint": name,
                    "prev_lat": prev.get("lat"),
                    "prev_lon": prev.get("lon"),
                    "curr_lat": curr.get("lat"),
                    "curr_lon": curr.get("lon"),
                    "delta_lat_deg": round(dlat, 5),
                    "delta_lon_deg": round(dlon, 5),
                })
        return drifts
