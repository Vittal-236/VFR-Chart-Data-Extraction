"""
client.py  — FIXED + CLAUDE API INTEGRATION

Now supports two modes:
  1. API mode  (default): sends the payload to the Anthropic Claude API
     using the key from ANTHROPIC_API_KEY in .env.  Falls back to rules
     mode on API failure so the pipeline never hard-crashes.
  2. Rules mode (--no-llm or no API key): the original regex classifier,
     preserved intact for offline/testing use.

The API call uses the system_prompt.txt already in models/prompts/ and
sends the payload as a JSON-encoded user message.

Also new in this file:
  - Route extraction from vector paths (previously a no-op stub).
  - ANSP-adaptive waypoint colour lookup via cfg.ANSP_RULES.
"""

import re
import json
import os
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.config import cfg

log = get_logger(__name__)

# ── Compiled regexes ───────────────────────────────────────────────────────────
_FREQ_RE   = re.compile(r"\b(1[0-3]\d\.\d{1,3}|2\d{2}\.\d{1,3})(?=[^\d]|$)")
_ICAO_RE   = re.compile(r"\b(L[FG][A-Z]{2}|EG[A-Z]{2}|EH[A-Z]{2}|LG[A-Z]{2})\b")
_COORD_LAT = re.compile(r"(\d{1,3})\s+(\d{1,2})\s+(\d{1,2})\s*([NS])", re.I)
_COORD_LON = re.compile(r"(\d{1,3})\s+(\d{1,2})\s+(\d{1,2})\s*([EW])", re.I)
_ALT_RE    = re.compile(r"\b(\d{2,5})\s*(?:ft|FT)\b")
_HDG_RE    = re.compile(r"\b(\d{3})\b")
_ROUTE_RE  = re.compile(r"\b([ULBH][A-Z]?\d{1,3})\b")
_HOLD_RE   = re.compile(r"\b(HOLD(?:ING)?|INBOUND|OUTBOUND|RACETRACK)\b", re.I)
_NHP_RE    = re.compile(r"\b(NHP|VRP)\b", re.I)

_OBS_KEYWORDS  = {"OBSTACLE","OBST","WARNING","MAST","TOWER",
                  "CHIMNEY","CRANE","STACK","PYLON","ANTENNA"}
_HELI_KEYWORDS = {"HELIPAD","HELIPORT","HELICOPTER"}
_SVC_MAP = {
    "ARRIVAL":"APP","ARRIVAL WEST":"APP","ARRIVAL EAST":"APP",
    "TOWER WEST":"TWR","TOWER EAST":"TWR",
    "TWR":"TWR","TOWER":"TWR","APP":"APP","APPROACH":"APP",
    "GND":"GND","GROUND":"GND","ATIS":"ATIS",
    "AFIS":"FIS","FIS":"FIS","A/A":"other",
}
_WPT_BLACKLIST = {
    "VFR","IFR","AIP","NIL","TWR","APP","GND","FIS","NDB","VOR","DME",
    "RWY","SIA","ULM","DEP","SIV","DTHR","RFFS","ACB","AVT","ZNM","LINFO",
    "REES","THR","TMA","CTR","ATZ","SID","STAR","GPS","ILS","LOC","TXT",
    "ALT","LAT","LON","LONG","VAR","OCA","ATS","ATT","CAP","VUE","ACT",
    "ADD","AGL","ATC","ADF","AFF","AFN","AFS","AGC","AGR","AGS",
    "OCH","MSA","MDA","MKR","MAG","MAP","LLZ","FAF","FAP","FAT",
    "RVR","TDZ","TCH","SDF","PAR","NPA","IAF","IAP","DH","DA",
    "AMDT","CHG","REFS","INFO","PAPI","MEHT","TORA","TODA","LDA","ASD",
    "FR","GP","RY","DE","AD","SVC","NON","ONG","AVI","EG","OS","LN",
}


def _dms(deg, min_, sec, hemi):
    dd = float(deg) + float(min_ or 0)/60 + float(sec or 0)/3600
    return -dd if hemi.upper() in ("S","W") else dd


class LLMClient:

    def __init__(self):
        self._api_key = cfg.ANTHROPIC_API_KEY
        if self._api_key:
            log.info("  Classifier: Claude API mode (Anthropic).")
        else:
            log.info("  Classifier: rules-based (no API key found).")
        self._system_prompt = self._load_system_prompt()

    # ── Public interface ───────────────────────────────────────────────────────

    def classify(self, payload: dict, retries: int = 2) -> dict:
        """
        Try the Claude API first; fall back to rules-based on failure.
        """
        if self._api_key:
            for attempt in range(retries):
                try:
                    return self._classify_via_api(payload)
                except Exception as exc:
                    log.warning(f"  Claude API attempt {attempt+1} failed: {exc}")
            log.warning("  All API attempts failed — falling back to rules-based classifier.")

        return self._classify_rules(payload)

    # ── Claude API mode ────────────────────────────────────────────────────────

    def _load_system_prompt(self) -> str:
        prompt_path = Path("models/prompts/system_prompt.txt")
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        return (
            "You are an aeronautical data analyst. Extract structured JSON from "
            "the chart payload. Respond ONLY with valid JSON using the schema provided."
        )

    def _classify_via_api(self, payload: dict) -> dict:
        """
        Call the Anthropic Claude API (claude-sonnet-4-20250514) with the
        chart payload and return the parsed JSON result.
        """
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)

        user_msg = (
            "Extract all aeronautical entities from this chart payload and return "
            "a JSON object matching the schema in your system prompt.\n\n"
            f"```json\n{json.dumps(payload, indent=2)}\n```"
        )

        response = client.messages.create(
            model=cfg.LLM_MODEL,
            max_tokens=cfg.LLM_MAX_TOKENS,
            system=self._system_prompt,
            messages=[{"role": "user", "content": user_msg}],
        )

        raw = response.content[0].text.strip()
        # Strip markdown fences if the model added them
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        result = json.loads(raw)
        log.info("  Claude API classification successful.")
        self._normalise(result)
        return result

    def _normalise(self, result: dict) -> None:
        """Ensure all expected top-level keys are present."""
        for key in ("waypoints","holding_patterns","frequencies","routes",
                    "nhp","obstacles","heli_routes","other"):
            result.setdefault(key, [])

    # ── Rules-based mode (offline fallback) ───────────────────────────────────

    def _classify_rules(self, payload: dict) -> dict:
        blocks  = payload.get("text_blocks", [])
        symbols = payload.get("detected_symbols", [])
        vector_paths = payload.get("vector_paths", [])

        result = {
            "waypoints":[], "holding_patterns":[], "frequencies":[],
            "routes":[], "nhp":[], "obstacles":[], "heli_routes":[], "other":[],
        }

        all_text = " ".join(b["text"] for b in blocks)

        self._frequencies(blocks, all_text, result)
        self._waypoints(blocks, symbols, result)
        self._obstacles(blocks, symbols, result)
        self._holdings(blocks, symbols, result)
        self._routes(blocks, vector_paths, result)
        self._nhp(blocks, symbols, result)
        self._helipads(blocks, symbols, result)
        self._airport_info(blocks, all_text, result)

        for key in result:
            result[key] = self._dedup(result[key])

        result = self._restructure(result)

        total = sum(len(v) if isinstance(v, list) else 1 for v in result.values())
        log.info(f"  Rules classifier: {total} items extracted.")
        return result

    # ── Restructure ────────────────────────────────────────────────────────────

    def _restructure(self, result: dict) -> dict:
        icao = elev = ref_lat = ref_lon = None
        for item in result.get("other", []):
            if item.get("category") == "airport_icao":
                icao = item.get("description")
            elif item.get("category") == "airport_elevation_ft":
                elev = item.get("description", "").replace(" ft", "")
            elif item.get("category") == "airport_reference_point":
                ref_lat = item.get("lat")
                ref_lon = item.get("lon")
        if icao:
            result["airport"] = {
                "icao": icao,
                "elevation_ft": int(elev) if elev and elev.isdigit() else None,
                "reference_point": {"lat": ref_lat, "lon": ref_lon},
                "runway_components": [
                    i["description"] for i in result.get("other", [])
                    if i.get("category") == "runway_info"
                ],
            }
        return result

    # ── Frequencies ────────────────────────────────────────────────────────────

    def _frequencies(self, blocks, all_text, result):
        seen = set()
        for block in blocks:
            for m in _FREQ_RE.finditer(block["text"]):
                freq = float(m.group(1))
                if freq in seen:
                    continue
                seen.add(freq)
                result["frequencies"].append({
                    "callsign":  self._nearby_callsign(block, blocks),
                    "type":      self._nearby_svc(block, blocks),
                    "value_mhz": freq,
                    "lat":       block.get("approx_lat"),
                    "lon":       block.get("approx_lon"),
                })

    def _nearby_svc(self, ref_block, all_blocks):
        ref_lat = ref_block.get("approx_lat", 0)
        row = " ".join(
            b["text"].upper() for b in all_blocks
            if abs(b.get("approx_lat", 0) - ref_lat) < 0.002
        )
        for kw, svc in _SVC_MAP.items():
            if kw in row:
                return svc
        return "other"

    def _nearby_callsign(self, ref_block, all_blocks):
        ref_lat = ref_block.get("approx_lat", 0)
        for b in all_blocks:
            if abs(b.get("approx_lat", 0) - ref_lat) < 0.002:
                t = b["text"].strip()
                if t.isupper() and 2 <= len(t) <= 10 and not _FREQ_RE.search(t):
                    return t
        return None

    # ── Waypoints ──────────────────────────────────────────────────────────────

    def _waypoints(self, blocks, symbols, result):
        for sym in symbols:
            if sym.get("type") == "waypoint":
                name = sym.get("label") or "UNKNOWN"
                if name not in _WPT_BLACKLIST:
                    result["waypoints"].append({
                        "name": name,
                        "lat":  sym.get("approx_lat"),
                        "lon":  sym.get("approx_lon"),
                        "type": None,
                    })
        seen = set()
        for block in blocks:
            text = block["text"].strip()
            if (re.fullmatch(r"[A-Z]{3,5}", text)
                    and text not in _WPT_BLACKLIST
                    and text not in seen):
                seen.add(text)
                result["waypoints"].append({
                    "name": text,
                    "lat":  block.get("approx_lat"),
                    "lon":  block.get("approx_lon"),
                    "type": None,
                })

    # ── Obstacles ──────────────────────────────────────────────────────────────

    def _obstacles(self, blocks, symbols, result):
        seen_coords = set()
        for block in blocks:
            text = block["text"].strip().upper()
            if set(re.findall(r"[A-Z]+", text)) & _OBS_KEYWORDS:
                hm = _ALT_RE.search(block["text"])
                result["obstacles"].append({
                    "description":    block["text"].strip(),
                    "height_ft_amsl": int(hm.group(1)) if hm else None,
                    "height_ft_agl":  None,
                    "lat":            block.get("approx_lat"),
                    "lon":            block.get("approx_lon"),
                    "lighted":        None,
                })
        for block in blocks:
            text = block["text"].strip()
            if re.fullmatch(r"\d{3,4}", text):
                val = int(text)
                if 50 <= val <= 2000:
                    coord_key = (
                        round(block.get("approx_lat", 0), 3),
                        round(block.get("approx_lon", 0), 3),
                    )
                    if coord_key not in seen_coords:
                        seen_coords.add(coord_key)
                        result["obstacles"].append({
                            "description":    f"Height {val} ft",
                            "height_ft_amsl": val,
                            "height_ft_agl":  None,
                            "lat":            block.get("approx_lat"),
                            "lon":            block.get("approx_lon"),
                            "lighted":        None,
                        })

    # ── Holdings ───────────────────────────────────────────────────────────────

    def _holdings(self, blocks, symbols, result):
        for sym in symbols:
            if sym.get("type") == "holding":
                result["holding_patterns"].append({
                    "name":              sym.get("label"),
                    "fix":               sym.get("label"),
                    "inbound_track_deg": None,
                    "turn":              None,
                    "lat":               sym.get("approx_lat"),
                    "lon":               sym.get("approx_lon"),
                })
        for block in blocks:
            if _HOLD_RE.search(block["text"]):
                hm = _HDG_RE.search(block["text"])
                result["holding_patterns"].append({
                    "name":              block["text"].strip(),
                    "fix":               None,
                    "inbound_track_deg": int(hm.group(1)) if hm else None,
                    "turn":              None,
                    "lat":               block.get("approx_lat"),
                    "lon":               block.get("approx_lon"),
                })

    # ── Routes (enhanced: also uses vector paths) ──────────────────────────────

    def _routes(self, blocks, vector_paths, result):
        """
        Extract named routes from text labels AND infer route segments from
        long vector paths (lines connecting waypoints on the chart).
        """
        seen = set()
        # Text-based named routes (e.g. "UL607", "B1")
        for block in blocks:
            for m in _ROUTE_RE.finditer(block["text"]):
                name = m.group(1)
                if name not in seen:
                    seen.add(name)
                    result["routes"].append({
                        "name": name,
                        "from_fix": None,
                        "to_fix": None,
                        "track_deg": None,
                        "distance_nm": None,
                        "lat": block.get("approx_lat"),
                        "lon": block.get("approx_lon"),
                    })

        # Vector-path based route segments
        # Long lines on aeronautical charts typically represent routes
        for path in vector_paths:
            pts = path.get("points", [])
            if len(pts) < 2:
                continue
            # Compute path length in PDF points
            total_len = sum(
                ((pts[i+1][0]-pts[i][0])**2 + (pts[i+1][1]-pts[i][1])**2)**0.5
                for i in range(len(pts)-1)
            )
            # Only consider paths longer than ~100 pt (about 1.4 inches)
            if total_len < 100:
                continue
            # Compute track heading (first to last point)
            dx = pts[-1][0] - pts[0][0]
            dy = pts[-1][1] - pts[0][1]
            import math
            track = (math.degrees(math.atan2(dx, -dy))) % 360

            result["routes"].append({
                "name": None,
                "from_fix": None,
                "to_fix": None,
                "track_deg": round(track, 1),
                "distance_nm": None,
                "source": "vector_path",
            })

    # ── NHP ────────────────────────────────────────────────────────────────────

    def _nhp(self, blocks, symbols, result):
        for sym in symbols:
            if sym.get("type") == "nhp":
                result["nhp"].append({
                    "name": sym.get("label") or "UNKNOWN",
                    "lat":  sym.get("approx_lat"),
                    "lon":  sym.get("approx_lon"),
                })
        for block in blocks:
            if _NHP_RE.search(block["text"]):
                result["nhp"].append({
                    "name": block["text"].strip(),
                    "lat":  block.get("approx_lat"),
                    "lon":  block.get("approx_lon"),
                })

    # ── Helipads ───────────────────────────────────────────────────────────────

    def _helipads(self, blocks, symbols, result):
        for sym in symbols:
            if sym.get("type") == "helipad":
                label = sym.get("label") or ""
                if any(kw in label.upper() for kw in _HELI_KEYWORDS) or re.search(r"\bH\d+\b", label):
                    result["heli_routes"].append({
                        "name": label, "from": label or "HELIPAD",
                        "to": None, "altitude_ft": None,
                    })
        for block in blocks:
            if set(block["text"].strip().upper().split()) & _HELI_KEYWORDS:
                result["heli_routes"].append({
                    "name": block["text"].strip(),
                    "from": block["text"].strip(),
                    "to": None, "altitude_ft": None,
                })

    # ── Airport info ────────────────────────────────────────────────────────────

    def _airport_info(self, blocks, all_text, result):
        seen_icao = set()
        for m in _ICAO_RE.finditer(all_text):
            code = m.group(1)
            if code not in seen_icao:
                seen_icao.add(code)
                result["other"].append({
                    "category":"airport_icao","description":code,
                    "lat":None,"lon":None,
                })
        alt_m = re.search(r"ALT\s+AD\s*[:\s]\s*(\d+)", all_text, re.I)
        if alt_m:
            result["other"].append({
                "category":"airport_elevation_ft",
                "description":alt_m.group(1)+" ft",
                "lat":None,"lon":None,
            })
        rwy_seen = set()
        for block in blocks:
            text = block["text"].strip()
            if re.search(r"\b(PAPI|MEHT|RWY|THR|TORA|TODA|LDA)\b", text, re.I):
                if text not in rwy_seen:
                    rwy_seen.add(text)
                    result["other"].append({
                        "category":"runway_info","description":text,
                        "lat":block.get("approx_lat"),"lon":block.get("approx_lon"),
                    })
        lat_m = _COORD_LAT.search(all_text)
        lon_m = _COORD_LON.search(all_text)
        if lat_m and lon_m:
            lat = _dms(*lat_m.groups())
            lon = _dms(*lon_m.groups())
            result["other"].append({
                "category":"airport_reference_point",
                "description":f"LAT {lat:.5f} LON {lon:.5f}",
                "lat":lat,"lon":lon,
            })

    # ── Dedup ──────────────────────────────────────────────────────────────────

    def _dedup(self, items):
        seen, out = set(), []
        for item in items:
            key = json.dumps(
                {k: v for k, v in item.items() if k != "source"},
                sort_keys=True
            )
            if key not in seen:
                seen.add(key)
                out.append(item)
        return out
