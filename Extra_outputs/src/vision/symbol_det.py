"""
symbol_det.py  — ENHANCED

Changes vs original:
1. Waypoint colour is now read from cfg.ANSP_RULES[ansp] so Greece (red
   triangles) and France (blue triangles) use the correct colour masks.
2. SymbolDetector accepts an optional ansp= parameter (defaults to empty
   string → falls back to magenta/red).
3. Minor: obstacle blob params relaxed to reduce false-negative rate.
4. Route-line detection added: long coloured paths → "route" symbols.
"""

from pathlib import Path
import cv2
import numpy as np
import math
from dataclasses import dataclass
from src.ingestion.pdf_parser import PageData
from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class DetectedSymbol:
    type: str
    bbox_px: tuple[int, int, int, int]
    confidence: float
    label: str | None = None


class SymbolDetector:

    def __init__(self, ansp: str = ""):
        self.ansp = ansp.upper()
        # Read waypoint colour from ANSP config; default to red/magenta
        ansp_rules = cfg.ANSP_RULES.get(self.ansp, {})
        bgr = ansp_rules.get("waypoint_color_bgr", (0, 0, 255))  # default red
        self._waypoint_bgr = bgr

    def detect(self, page_data: PageData) -> list[DetectedSymbol]:
        img = page_data.image_cv2
        if img is None:
            return []

        symbols: list[DetectedSymbol] = []
        symbols += self._detect_waypoints(img)
        symbols += self._detect_holding_patterns(img)
        symbols += self._detect_obstacles(img)
        symbols += self._detect_helipads(img)
        symbols += self._detect_route_lines(img)
        symbols += self._run_yolo(img)

        self._attach_labels(symbols, page_data.text_blocks)

        log.info(f"  Detected {len(symbols)} symbols on page {page_data.page_number}.")
        return symbols

    # ── Waypoints (ANSP-colour-aware) ──────────────────────────────────────────

    def _detect_waypoints(self, img: np.ndarray) -> list[DetectedSymbol]:
        """
        Detect filled triangles in the ANSP-specific waypoint colour.
        Greece = red, France = blue, default = magenta/red.
        """
        symbols = []
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        b, g, r = self._waypoint_bgr

        # Build HSV mask tuned to the waypoint colour
        if r > 200 and g < 60 and b < 60:
            # Red: two ranges in HSV
            mask1 = cv2.inRange(hsv, (0, 60, 60), (10, 255, 255))
            mask2 = cv2.inRange(hsv, (170, 60, 60), (180, 255, 255))
            mask  = cv2.bitwise_or(mask1, mask2)
        elif b > 200 and r < 60:
            # Blue
            mask = cv2.inRange(hsv, (100, 80, 80), (130, 255, 255))
        else:
            # Magenta / other — combine red and magenta ranges
            mask1 = cv2.inRange(hsv, (140, 60, 60), (170, 255, 255))
            mask2 = cv2.inRange(hsv, (0, 60, 60), (10, 255, 255))
            mask  = cv2.bitwise_or(mask1, mask2)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (30 < area < 2000):
                continue
            peri  = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) == 3:
                x, y, w, h = cv2.boundingRect(cnt)
                symbols.append(DetectedSymbol(
                    type="waypoint",
                    bbox_px=(x, y, w, h),
                    confidence=0.75,
                ))
        return symbols

    # ── Holding patterns ───────────────────────────────────────────────────────

    def _detect_holding_patterns(self, img: np.ndarray) -> list[DetectedSymbol]:
        symbols = []
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if len(cnt) < 5:
                continue
            area = cv2.contourArea(cnt)
            if not (500 < area < 50000):
                continue
            ellipse = cv2.fitEllipse(cnt)
            (_, _), (ma, mi), _ = ellipse
            if mi == 0:
                continue
            aspect = ma / mi
            if 1.8 < aspect < 6.0:
                x, y, w, h = cv2.boundingRect(cnt)
                symbols.append(DetectedSymbol(
                    type="holding",
                    bbox_px=(x, y, w, h),
                    confidence=0.65,
                ))
        return symbols

    # ── Obstacles ──────────────────────────────────────────────────────────────

    def _detect_obstacles(self, img: np.ndarray) -> list[DetectedSymbol]:
        symbols = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 15          # slightly relaxed
        params.maxArea = 400
        params.filterByCircularity = False
        detector = cv2.SimpleBlobDetector_create(params)

        keypoints = detector.detect(255 - binary)
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            r = int(kp.size / 2) + 5
            symbols.append(DetectedSymbol(
                type="obstacle",
                bbox_px=(x - r, y - r, 2 * r, 2 * r),
                confidence=0.55,
            ))
        return symbols

    # ── Helipads ───────────────────────────────────────────────────────────────

    def _detect_helipads(self, img: np.ndarray) -> list[DetectedSymbol]:
        symbols = []
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=50, param2=30, minRadius=8, maxRadius=40,
        )
        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            for x, y, r in circles:
                symbols.append(DetectedSymbol(
                    type="helipad",
                    bbox_px=(x - r, y - r, 2 * r, 2 * r),
                    confidence=0.60,
                ))
        return symbols

    # ── Route lines ────────────────────────────────────────────────────────────

    def _detect_route_lines(self, img: np.ndarray) -> list[DetectedSymbol]:
        """
        Detect thick coloured lines that represent VFR routes on the chart.
        Uses probabilistic Hough line detection after isolating coloured pixels.
        """
        symbols = []
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Target route colours: blue, cyan, magenta on most European charts
        masks = [
            cv2.inRange(hsv, (100, 80, 80), (130, 255, 255)),  # blue
            cv2.inRange(hsv, (80,  80, 80), (100, 255, 255)),  # cyan
            cv2.inRange(hsv, (140, 80, 80), (170, 255, 255)),  # magenta
        ]
        combined = masks[0]
        for m in masks[1:]:
            combined = cv2.bitwise_or(combined, m)

        combined = cv2.dilate(combined, np.ones((3, 3), np.uint8), iterations=1)
        lines = cv2.HoughLinesP(
            combined, 1, np.pi / 180, threshold=80,
            minLineLength=60, maxLineGap=20
        )
        if lines is None:
            return symbols

        for line in lines:
            x1, y1, x2, y2 = line[0]
            cx = min(x1, x2)
            cy = min(y1, y2)
            w  = abs(x2 - x1) + 1
            h  = abs(y2 - y1) + 1
            symbols.append(DetectedSymbol(
                type="route",
                bbox_px=(cx, cy, w, h),
                confidence=0.60,
            ))
        return symbols

    # ── YOLO stub ──────────────────────────────────────────────────────────────

    def _run_yolo(self, img: np.ndarray) -> list[DetectedSymbol]:
        weights_path = Path(cfg.YOLO_WEIGHTS)
        if not weights_path.exists():
            return []
        try:
            from ultralytics import YOLO
            model   = YOLO(str(weights_path))
            results = model(img, conf=cfg.SYMBOL_CONF_THRESHOLD, verbose=False)
            symbols = []
            for box in results[0].boxes:
                cls_id   = int(box.cls[0])
                cls_name = results[0].names[cls_id]
                conf     = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                symbols.append(DetectedSymbol(
                    type=cls_name,
                    bbox_px=(x1, y1, x2 - x1, y2 - y1),
                    confidence=conf,
                ))
            log.info(f"  YOLO detected {len(symbols)} additional symbols.")
            return symbols
        except ImportError:
            log.warning("  ultralytics not installed — YOLO detection skipped.")
            return []
        except Exception as e:
            log.warning(f"  YOLO failed: {e}")
            return []

    # ── Label association ───────────────────────────────────────────────────────

    def _attach_labels(
        self,
        symbols: list[DetectedSymbol],
        text_blocks: list[dict],
        search_radius_px: int = 60,
    ) -> None:
        for sym in symbols:
            sx, sy, sw, sh = sym.bbox_px
            cx, cy = sx + sw / 2, sy + sh / 2
            best_dist = float("inf")
            best_text = None
            for block in text_blocks:
                bx0, by0, bx1, by1 = block["bbox"]
                bcx = (bx0 + bx1) / 2
                bcy = (by0 + by1) / 2
                dist = math.hypot(cx - bcx, cy - bcy)
                if dist < search_radius_px and dist < best_dist:
                    best_dist = dist
                    best_text = block["text"]
            sym.label = best_text
