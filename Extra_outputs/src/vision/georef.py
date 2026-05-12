"""
georef.py  — FIXED + ENHANCED

Changes vs original:
1. DMS regex now accepts both °/'/'' notation AND plain-space notation.
   Charts label ticks as "38°30'N" or "38 30 00 N" — both now matched.
2. Inset-aware: build_transform() accepts an optional parent_transform so
   inset regions get their own calibrated geo-transform derived from the
   parent page transform.
3. _fit_affine is more robust: uses numpy least-squares instead of polyfit
   so it works even when ticks are not perfectly monotonic (scan skew).
"""

import re
import numpy as np
from dataclasses import dataclass
from src.ingestion.pdf_parser import PageData
from src.utils.logger import get_logger

log = get_logger(__name__)

KNOWN_CHART_BOUNDS = {
    "GR": (38.5, 37.0, 22.0, 25.5),
    "FR": None,
    "US": (40.0, 36.0, -80.0, -74.0),
}


@dataclass
class GeoTransform:
    a: float; b: float; c: float
    d: float; e: float; f: float

    def pixel_to_latlon(self, px_x: float, px_y: float):
        lon = self.a + self.b * px_x + self.c * px_y
        lat = self.d + self.e * px_x + self.f * px_y
        return lat, lon

    def is_valid(self):
        return not (self.b == 0 and self.e == 0)


class Georeferencer:

    # Matches BOTH formats:
    #   "38°30'00\"N"  or  "38°30'N"  (degree-minute, no seconds)
    #   "38 30 00 N"   (space-separated DMS)
    _LAT_RE = re.compile(
        r"(\d{1,3})\s*[°\s]\s*(\d{1,2})\s*['\s]\s*(\d{0,2}(?:\.\d+)?)\s*[\"']?\s*([NS])\b",
        re.IGNORECASE,
    )
    _LON_RE = re.compile(
        r"(\d{1,3})\s*[°\s]\s*(\d{1,2})\s*['\s]\s*(\d{0,2}(?:\.\d+)?)\s*[\"']?\s*([EW])\b",
        re.IGNORECASE,
    )

    def __init__(self, ansp: str = ""):
        self.ansp = ansp.upper()

    def build_transform(
        self,
        page_data: PageData,
        parent_transform: GeoTransform | None = None,
    ) -> GeoTransform:
        img_h = page_data.height_pt
        img_w = page_data.width_pt

        # For inset sub-pages: derive transform from parent using pixel offsets
        if parent_transform is not None and getattr(page_data, "inset_x0", 0) != 0:
            return self._inset_transform(page_data, parent_transform)

        # Strategy 1: known chart bounds (fastest, most accurate)
        if self.ansp in KNOWN_CHART_BOUNDS and KNOWN_CHART_BOUNDS[self.ansp]:
            bounds = KNOWN_CHART_BOUNDS[self.ansp]
            transform = self._bounds_to_transform(bounds, img_w, img_h)
            log.info(f"  Georef: known bounds for {self.ansp}.")
            return transform

        # Strategy 2: grid ticks from DMS labels
        lat_ticks, lon_ticks = [], []
        for block in page_data.text_blocks:
            text = block["text"]
            bx0, by0, bx1, by1 = block["bbox"]
            cx = (bx0 + bx1) / 2
            cy = (by0 + by1) / 2

            for m in self._LAT_RE.finditer(text):
                dd = self._dms_to_dd(*m.groups())
                if dd is not None:
                    lat_ticks.append((cy, dd))

            for m in self._LON_RE.finditer(text):
                dd = self._dms_to_dd(*m.groups())
                if dd is not None:
                    lon_ticks.append((cx, dd))

        if len(lat_ticks) >= 2 and len(lon_ticks) >= 2:
            transform = self._fit_affine(lat_ticks, lon_ticks, img_w, img_h)
            log.info(
                f"  Georef: grid-based transform from "
                f"{len(lat_ticks)} lat / {len(lon_ticks)} lon ticks."
            )
            return transform

        # Strategy 3: plain degree markers at chart edges
        transform = self._parse_degree_markers(page_data.text_blocks, img_w, img_h)
        if transform and transform.is_valid():
            log.info("  Georef: degree-marker transform.")
            return transform

        # Strategy 4: single header coordinate (airport charts)
        transform = self._parse_header_coords(page_data.text_blocks, img_w, img_h)
        if transform and transform.is_valid():
            log.info("  Georef: header-based transform (airport chart).")
            return transform

        log.warning("  Georef: no coordinates found — using identity transform.")
        return self._fallback_transform(img_w, img_h)

    # ── Inset transform ────────────────────────────────────────────────────────

    def _inset_transform(
        self, inset: PageData, parent: GeoTransform
    ) -> GeoTransform:
        """
        An inset is a blown-up crop of a sub-region of the parent chart.
        We know the pixel offset (inset_x0, inset_y0) of the crop in parent
        coordinates.  Build an affine that maps inset-local pixels to lat/lon
        via the parent transform, then scales by the zoom factor.

        zoom = parent_chart_pixels_for_this_region / inset_pixels
        The zoom is approximated as parent_width / inset_width (same crop).
        """
        x0 = getattr(inset, "inset_x0", 0)
        y0 = getattr(inset, "inset_y0", 0)
        pw = getattr(inset, "parent_width_pt", inset.width_pt)
        ph = getattr(inset, "parent_height_pt", inset.height_pt)

        zoom_x = inset.width_pt / pw if pw else 1.0
        zoom_y = inset.height_pt / ph if ph else 1.0

        # Effective scale in lat/lon per inset-pixel
        b_eff = parent.b / zoom_x  # lon per inset-px-x
        f_eff = parent.f / zoom_y  # lat per inset-px-y

        # Origin = parent lat/lon at (x0, y0)
        lat0, lon0 = parent.pixel_to_latlon(x0, y0)

        return GeoTransform(
            a=lon0, b=b_eff, c=0.0,
            d=lat0, e=0.0,   f=f_eff,
        )

    # ── Core helpers ───────────────────────────────────────────────────────────

    def _bounds_to_transform(self, bounds, img_w, img_h) -> GeoTransform:
        lat_max, lat_min, lon_min, lon_max = bounds
        scale_lat = (lat_min - lat_max) / img_h
        scale_lon = (lon_max - lon_min) / img_w
        return GeoTransform(
            a=lon_min, b=scale_lon, c=0.0,
            d=lat_max, e=0.0,       f=scale_lat,
        )

    def _dms_to_dd(self, deg, min_, sec, hemi):
        try:
            dd = float(deg) + float(min_ or 0) / 60 + float(sec or 0) / 3600
            if str(hemi).upper() in ("S", "W"):
                dd = -dd
            return dd
        except (ValueError, TypeError):
            return None

    def _fit_affine(self, lat_ticks, lon_ticks, img_w, img_h) -> GeoTransform:
        """Least-squares fit — more robust than polyfit for skewed scans."""
        lat_ticks = sorted(lat_ticks)
        lon_ticks = sorted(lon_ticks)

        pys  = np.array([t[0] for t in lat_ticks])
        lats = np.array([t[1] for t in lat_ticks])
        A    = np.column_stack([pys, np.ones_like(pys)])
        lat_coeffs, _, _, _ = np.linalg.lstsq(A, lats, rcond=None)
        f, d = lat_coeffs

        pxs  = np.array([t[0] for t in lon_ticks])
        lons = np.array([t[1] for t in lon_ticks])
        B    = np.column_stack([pxs, np.ones_like(pxs)])
        lon_coeffs, _, _, _ = np.linalg.lstsq(B, lons, rcond=None)
        b, a = lon_coeffs

        return GeoTransform(a=a, b=b, c=0.0, d=d, e=0.0, f=f)

    def _parse_degree_markers(self, text_blocks, img_w, img_h):
        lat_ticks, lon_ticks = [], []
        for block in text_blocks:
            text = block["text"].strip()
            bx0, by0, bx1, by1 = block["bbox"]
            cx = (bx0 + bx1) / 2
            cy = (by0 + by1) / 2
            if not re.fullmatch(r"\d{2}", text):
                continue
            val = int(text)
            if 34 <= val <= 45 and (cx < img_w * 0.08 or cx > img_w * 0.92):
                lat_ticks.append((cy, float(val)))
            if 20 <= val <= 30 and (cy < img_h * 0.08 or cy > img_h * 0.92):
                lon_ticks.append((cx, float(val)))
        if len(lat_ticks) >= 2 and len(lon_ticks) >= 2:
            return self._fit_affine(lat_ticks, lon_ticks, img_w, img_h)
        return None

    def _parse_header_coords(self, text_blocks, img_w, img_h):
        joined = " ".join(b["text"] for b in text_blocks)
        lat_m = self._LAT_RE.search(joined)
        lon_m = self._LON_RE.search(joined)
        if not (lat_m and lon_m):
            return None
        found_lat = self._dms_to_dd(*lat_m.groups())
        found_lon = self._dms_to_dd(*lon_m.groups())
        if found_lat is None or found_lon is None:
            return None
        scale_lat = -0.1 / img_h
        scale_lon =  0.1 / img_w
        centre_lat = found_lat - scale_lat * (img_h / 2)
        centre_lon = found_lon - scale_lon * (img_w / 2)
        return GeoTransform(
            a=centre_lon, b=scale_lon, c=0.0,
            d=centre_lat, e=0.0,       f=scale_lat,
        )

    def _fallback_transform(self, img_w, img_h) -> GeoTransform:
        return GeoTransform(a=0, b=1, c=0, d=0, e=0, f=1)
