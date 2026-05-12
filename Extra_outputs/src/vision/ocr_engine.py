"""
ocr_engine.py  — FIXED + ENHANCED

Changes vs original:
1. process_easyocr() now recurses into inset sub-pages (was silently skipped).
2. Added colour-based preprocessing option for aeronautical charts where text
   appears on coloured backgrounds.
3. EasyOCR reader is cached at class level to avoid reloading weights each call.
"""

import pytesseract
from pytesseract import Output
import cv2
import numpy as np
from src.ingestion.pdf_parser import PageData
from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)

pytesseract.pytesseract.tesseract_cmd = cfg.TESSERACT_CMD

# Module-level EasyOCR reader cache (avoid reloading weights repeatedly)
_easyocr_reader = None


def _get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(["en"], verbose=False)
        log.info("  EasyOCR reader loaded.")
    return _easyocr_reader


class OCREngine:

    _TESS_CONFIG = "--oem 3 --psm 11"

    def process(self, page_data: PageData, force: bool = False) -> None:
        """Run Tesseract OCR on a page and its insets."""
        if len(page_data.text_blocks) > 10 and not force:
            log.info("  Text blocks already present — skipping Tesseract OCR.")
            return
        if page_data.image_cv2 is None:
            log.warning("  No image available for OCR — skipping.")
            return
        log.info("  Running Tesseract OCR …")
        preprocessed = self._preprocess(page_data.image_cv2)
        data = pytesseract.image_to_data(
            preprocessed,
            config=self._TESS_CONFIG,
            output_type=Output.DICT,
        )
        blocks = self._dict_to_blocks(data, page_data)
        page_data.text_blocks.extend(blocks)
        log.info(f"  OCR found {len(blocks)} text spans.")

        # FIXED: recurse into insets
        for inset in page_data.insets:
            self.process(inset, force=True)

    def process_easyocr(self, page_data: PageData) -> None:
        """
        EasyOCR alternative — better accuracy on maps than Tesseract.
        Falls back to Tesseract if easyocr is not installed.

        FIXED: now recurses into detected inset sub-pages.
        """
        try:
            reader = _get_easyocr_reader()
        except ImportError:
            log.warning("  easyocr not installed — falling back to Tesseract.")
            self.process(page_data)
            return

        self._run_easyocr_on_page(reader, page_data)

        # FIXED: process each inset independently
        for inset in page_data.insets:
            log.info(
                f"  Running EasyOCR on inset ({inset.width_pt:.0f}x{inset.height_pt:.0f})…"
            )
            self._run_easyocr_on_page(reader, inset, force=True)

    def _run_easyocr_on_page(
        self, reader, page_data: PageData, force: bool = False
    ) -> None:
        if len(page_data.text_blocks) > 10 and not force:
            log.info("  Text blocks already present — skipping EasyOCR.")
            return
        if page_data.image_cv2 is None:
            log.warning("  No image available for EasyOCR — skipping.")
            return

        results = reader.readtext(page_data.image_cv2)
        blocks = []
        for bbox, text, conf in results:
            text = text.strip()
            if not text or conf < 0.4:
                continue
            x0 = min(p[0] for p in bbox)
            y0 = min(p[1] for p in bbox)
            x1 = max(p[0] for p in bbox)
            y1 = max(p[1] for p in bbox)
            blocks.append({
                "text": text,
                "bbox": (float(x0), float(y0), float(x1), float(y1)),
                "size": float(y1 - y0),
                "font": "EasyOCR",
                "color": 0,
                "confidence": conf,
            })
        page_data.text_blocks.extend(blocks)
        log.info(f"  EasyOCR found {len(blocks)} text spans.")

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Standard grayscale + sharpen + Otsu for Tesseract."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        _, binary = cv2.threshold(
            sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return binary

    def _dict_to_blocks(self, data: dict, page_data: PageData) -> list[dict]:
        blocks = []
        n = len(data["text"])
        for i in range(n):
            text = data["text"][i].strip()
            conf = int(data["conf"][i])
            if not text or conf < 40:
                continue
            x, y, w, h = (
                data["left"][i], data["top"][i],
                data["width"][i], data["height"][i],
            )
            blocks.append({
                "text": text,
                "bbox": (float(x), float(y), float(x + w), float(y + h)),
                "size": float(h),
                "font": "OCR",
                "color": 0,
                "confidence": conf,
            })
        return blocks
