"""
validator.py  — ENHANCED

Added full schema validation for:
  - holding_patterns (inbound_track_deg range, turn enum)
  - routes (track_deg range)
  - nhp (coordinate check)
  - heli_routes (altitude sanity)
"""

import jsonschema
from dataclasses import dataclass, field
from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)

AERO_SCHEMA = {
    "type": "object",
    "properties": {
        "waypoints": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string"},
                    "lat":  {"type": ["number", "null"]},
                    "lon":  {"type": ["number", "null"]},
                    "type": {"type": ["string", "null"]},
                },
            },
        },
        "frequencies": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["value_mhz"],
                "properties": {
                    "value_mhz": {"type": "number"},
                    "callsign":  {"type": ["string", "null"]},
                    "type":      {"type": ["string", "null"]},
                },
            },
        },
        "holding_patterns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name":              {"type": ["string", "null"]},
                    "fix":               {"type": ["string", "null"]},
                    "inbound_track_deg": {"type": ["number", "null"]},
                    "turn":              {"type": ["string", "null"], "enum": ["left", "right", None]},
                    "lat":               {"type": ["number", "null"]},
                    "lon":               {"type": ["number", "null"]},
                },
            },
        },
        "routes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name":        {"type": ["string", "null"]},
                    "from_fix":    {"type": ["string", "null"]},
                    "to_fix":      {"type": ["string", "null"]},
                    "track_deg":   {"type": ["number", "null"]},
                    "distance_nm": {"type": ["number", "null"]},
                },
            },
        },
        "nhp": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string"},
                    "lat":  {"type": ["number", "null"]},
                    "lon":  {"type": ["number", "null"]},
                },
            },
        },
        "obstacles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "height_ft_amsl": {"type": ["number", "null"]},
                    "height_ft_agl":  {"type": ["number", "null"]},
                    "lat":            {"type": ["number", "null"]},
                    "lon":            {"type": ["number", "null"]},
                    "lighted":        {"type": ["boolean", "null"]},
                },
            },
        },
        "heli_routes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name":        {"type": ["string", "null"]},
                    "from":        {"type": ["string", "null"]},
                    "to":          {"type": ["string", "null"]},
                    "altitude_ft": {"type": ["number", "null"]},
                },
            },
        },
    },
}


@dataclass
class ValidationResult:
    is_valid: bool = True
    errors:   list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def add_error(self, msg):
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg):
        self.warnings.append(msg)

    def summary(self):
        return (
            f"Valid={self.is_valid} | "
            f"{len(self.errors)} error(s) | "
            f"{len(self.warnings)} warning(s)"
        )


class Validator:

    def validate(self, data):
        result = ValidationResult()
        self._validate_schema(data, result)
        self._validate_frequencies(data.get("frequencies", []), result)
        self._validate_coordinates(data.get("waypoints", []),        "waypoints", result)
        self._validate_coordinates(data.get("nhp", []),              "nhp",       result)
        self._validate_coordinates(data.get("holding_patterns", []), "holding_patterns", result)
        self._validate_obstacles(data.get("obstacles", []),   result)
        self._validate_holdings(data.get("holding_patterns", []), result)
        self._validate_routes(data.get("routes", []),          result)
        self._validate_heli_routes(data.get("heli_routes", []), result)
        log.info(f"  Validation: {result.summary()}")
        return result

    def _validate_schema(self, data, result):
        try:
            jsonschema.validate(instance=data, schema=AERO_SCHEMA)
        except jsonschema.ValidationError as e:
            result.add_error(f"Schema error: {e.message} at {list(e.absolute_path)}")

    def _validate_frequencies(self, freqs, result):
        lo_vhf, hi_vhf = cfg.VALID_VHF_RANGE
        lo_uhf, hi_uhf = cfg.VALID_UHF_RANGE
        for freq in freqs:
            mhz = freq.get("value_mhz")
            if mhz is None:
                result.add_warning("Frequency missing value_mhz.")
                continue
            if not ((lo_vhf <= mhz <= hi_vhf) or (lo_uhf <= mhz <= hi_uhf)):
                result.add_error(
                    f"Frequency {mhz} MHz outside valid VHF ({lo_vhf}–{hi_vhf}) "
                    f"and UHF ({lo_uhf}–{hi_uhf}) ranges."
                )

    def _validate_coordinates(self, items, label, result):
        for item in items:
            lat  = item.get("lat")
            lon  = item.get("lon")
            name = item.get("name") or item.get("fix") or "unnamed"
            if lat is None or lon is None:
                result.add_warning(f"{label} '{name}' missing coordinates.")
                continue
            if not (-90 <= lat <= 90):
                result.add_error(f"{label} '{name}' has invalid lat={lat}.")
            if not (-180 <= lon <= 180):
                result.add_error(f"{label} '{name}' has invalid lon={lon}.")

    def _validate_obstacles(self, obstacles, result):
        for obs in obstacles:
            height = obs.get("height_ft_amsl")
            if height is not None and height > cfg.MAX_OBSTACLE_HEIGHT_FT:
                result.add_error(
                    f"Obstacle height {height} ft AMSL exceeds maximum "
                    f"({cfg.MAX_OBSTACLE_HEIGHT_FT} ft)."
                )
            if obs.get("lat") is None or obs.get("lon") is None:
                result.add_warning("Obstacle missing coordinates.")

    def _validate_holdings(self, holdings, result):
        for h in holdings:
            track = h.get("inbound_track_deg")
            if track is not None and not (0 <= track <= 360):
                result.add_error(
                    f"Holding '{h.get('name')}': inbound_track_deg {track} "
                    "is out of range [0, 360]."
                )
            turn = h.get("turn")
            if turn is not None and turn not in ("left", "right"):
                result.add_error(
                    f"Holding '{h.get('name')}': turn must be 'left', 'right', or null."
                )

    def _validate_routes(self, routes, result):
        for r in routes:
            track = r.get("track_deg")
            if track is not None and not (0 <= track <= 360):
                result.add_error(
                    f"Route '{r.get('name')}': track_deg {track} out of range [0, 360]."
                )
            dist = r.get("distance_nm")
            if dist is not None and dist < 0:
                result.add_error(f"Route '{r.get('name')}': negative distance_nm {dist}.")

    def _validate_heli_routes(self, heli_routes, result):
        for hr in heli_routes:
            alt = hr.get("altitude_ft")
            if alt is not None and (alt < 0 or alt > 50000):
                result.add_warning(
                    f"Heli-route '{hr.get('name')}': altitude_ft {alt} seems implausible."
                )
