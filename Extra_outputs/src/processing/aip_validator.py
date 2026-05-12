import re
from src.utils.logger import get_logger

log = get_logger(__name__)

KNOWN_FR_WAYPOINTS = {
    "RESMI","TINOT","GALBO","GONUP","OKABA","LFAV",
    "LILLE","DENAIN","VALENCIENNES","CAMBRAI","DOUAI",
}

KNOWN_US_WAYPOINTS = {
    "SWANN","RAVNN","FLUKY","PALEO","WOOLY","DRUZZ",
    "BROSS","TAPPA","LUCIT","CAVVS",
}

KNOWN_GR_WAYPOINTS = {
    # From the Greece VFR chart
    "GERMI","PIKAD","RILIN","KASTEL","ARGOS","FSIS","AMARO","AVLAK",
    "FEVES","QMAIS","EGINA","KIATC","NVROR","PIKAD","LGEL","LGAV",
    "KAFIREAS","MEROUTI","MANDILOU","PERATI","FLEVES","MAKROS","KEA",
    "SOREV","BADEL","VELOP","ASTROV","NAFPLIO","EPIDAVROS","SPETSAI",
    "YDRA","DAPORI","KIATON","KORINTHOS","ASTROS","SERIFOS","VARIX",
    "ABEAM","YIAROS","ABLONAS","ZOUMBERI","PALLINI","STAVROS","ILIOUP",
    "KARISTOS","LAO","EGN","AST","RIO","MAI","SPA","GMG","GNL",
    "LGP","KRO","ALIS","MSL","SEE","GERMI",
}

class AIDataValidator:

    def __init__(self, ansp: str = "FR"):
        self.ansp = ansp.upper()
        if self.ansp == "FR":
            self._known = KNOWN_FR_WAYPOINTS
        elif self.ansp in ("US", "GR"):
            self._known = KNOWN_US_WAYPOINTS if self.ansp == "US" else KNOWN_GR_WAYPOINTS
        else:
            self._known = set()

    def validate_and_flag(self, aero_data: dict) -> dict:
        flagged = 0
        for wpt in aero_data.get("waypoints", []):
            name = wpt.get("name", "")
            wpt["verified"] = name in self._known
            if not wpt["verified"]:
                flagged += 1
        if flagged:
            log.info(f"  AIP check: {flagged} waypoint(s) unverified.")
        return aero_data

    def filter_verified_only(self, aero_data: dict) -> dict:
        before = len(aero_data.get("waypoints", []))
        aero_data["waypoints"] = [
            w for w in aero_data.get("waypoints", [])
            if w.get("verified", False)
        ]
        after = len(aero_data["waypoints"])
        log.info(f"  Strict filter: kept {after}/{before} verified waypoints.")
        return aero_data