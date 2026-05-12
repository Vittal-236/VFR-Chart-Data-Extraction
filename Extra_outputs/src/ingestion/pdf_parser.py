"""
pdf_parser.py  — FIXED
Extracts text blocks, raster images, and vector drawing instructions from PDFs.

BUG FIX: _extract_text_blocks had the ratio<0.5 guard *outside* the inner span
loop.  That meant the ratio from the very last span decided whether to discard
ALL blocks for the whole page.  Now every span is checked individually.
"""

import fitz  # PyMuPDF
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import cv2

from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class PageData:
    page_number: int
    width_pt: float
    height_pt: float
    text_blocks: list[dict]
    image_cv2: Optional[np.ndarray] = None
    vector_paths: list[dict] = field(default_factory=list)
    insets: list["PageData"] = field(default_factory=list)
    # Set by _detect_insets so georef can build a separate transform
    inset_x0: int = 0
    inset_y0: int = 0
    parent_width_pt: float = 0.0
    parent_height_pt: float = 0.0


class PDFParser:
    DPI = 200

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"PDF not found: {self.path}")

    def parse(self) -> list[PageData]:
        doc = fitz.open(str(self.path))
        results = []
        for i, page in enumerate(doc):
            log.info(f"Parsing page {i + 1}/{len(doc)} of '{self.path.name}'")
            pd = self._parse_page(page, i)
            results.append(pd)
        doc.close()
        return results

    def _parse_page(self, page: fitz.Page, page_idx: int) -> PageData:
        rect = page.rect
        text_blocks = self._extract_text_blocks(page)
        image_cv2   = self._render_page(page)
        vector_paths = self._extract_vector_paths(page)

        pd = PageData(
            page_number=page_idx,
            width_pt=rect.width,
            height_pt=rect.height,
            text_blocks=text_blocks,
            image_cv2=image_cv2,
            vector_paths=vector_paths,
        )
        pd.insets = self._detect_insets(pd)
        return pd

    def _extract_text_blocks(self, page: fitz.Page) -> list[dict]:
        """
        FIXED: ratio check is now INSIDE the span loop.
        Each garbage span is skipped individually; readable spans are kept.
        If fewer than 5 readable spans survive, return [] so OCR takes over.
        """
        blocks = []
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if not text:
                        continue
                    readable = sum(1 for c in text if 32 <= ord(c) < 127)
                    if readable / len(text) < 0.5:
                        continue          # encoded glyph — skip this span
                    blocks.append({
                        "text": text,
                        "bbox": span["bbox"],
                        "size": span["size"],
                        "font": span["font"],
                        "color": span.get("color", 0),
                    })

        if len(blocks) < 5:
            log.info("  Too few readable spans — OCR will handle this page.")
            return []
        return blocks

    def _render_page(self, page: fitz.Page) -> np.ndarray:
        mat = fitz.Matrix(self.DPI / 72, self.DPI / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3
        )
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def _extract_vector_paths(self, page: fitz.Page) -> list[dict]:
        paths = []
        for path in page.get_drawings():
            paths.append({
                "points": [
                    (item[1].x, item[1].y)
                    for item in path.get("items", [])
                    if item[0] in ("l", "m")
                ],
                "stroke_color": path.get("color"),
                "fill_color":   path.get("fill"),
                "line_width":   path.get("width", 1.0),
                "rect":         path.get("rect"),
            })
        return paths

    def _detect_insets(self, page_data: "PageData") -> list["PageData"]:
        img = page_data.image_cv2
        if img is None:
            return []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        insets = []
        img_area = img.shape[0] * img.shape[1]

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            aspect = w / h if h > 0 else 0
            area_ratio = area / img_area
            if not (0.05 < area_ratio < 0.40):
                continue
            if not (0.5 < aspect < 3.0):
                continue

            cropped = img[y: y + h, x: x + w]
            inset_pd = PageData(
                page_number=page_data.page_number,
                width_pt=float(w),
                height_pt=float(h),
                text_blocks=[],
                image_cv2=cropped,
                inset_x0=x,
                inset_y0=y,
                parent_width_pt=page_data.width_pt,
                parent_height_pt=page_data.height_pt,
            )
            insets.append(inset_pd)
            log.info(f"  Detected inset region at ({x},{y}) size {w}x{h}px")

        return insets
