"""
mapper.py  — ENHANCED

Changes vs original:
1. vector_paths are now included in the payload so the LLM / rules
   classifier can use them for route extraction.
2. Inset text blocks are tagged with inset=True so the classifier
   knows they came from a zoomed-in region.
3. MAX_TEXT_BLOCKS increased to 400 (Claude's context is large enough).
"""

from src.ingestion.pdf_parser import PageData
from src.vision.symbol_det import DetectedSymbol
from src.vision.georef import GeoTransform
from src.utils.logger import get_logger

log = get_logger(__name__)


class PayloadMapper:

    MAX_TEXT_BLOCKS = 400

    def build(
        self,
        page_data: PageData,
        symbols: list[DetectedSymbol],
        transform: GeoTransform,
        chart_metadata: dict,
    ) -> dict:
        text_payload   = self._format_text_blocks(page_data.text_blocks, transform)
        symbol_payload = self._format_symbols(symbols, transform)

        payload = {
            "chart_metadata":    chart_metadata,
            "text_blocks":       text_payload,
            "detected_symbols":  symbol_payload,
            "vector_paths":      [],   # populated by main.py after build()
        }

        log.info(
            f"  Payload built: {len(text_payload)} text blocks, "
            f"{len(symbol_payload)} symbols."
        )
        return payload

    def _format_text_blocks(
        self, blocks: list[dict], transform: GeoTransform
    ) -> list[dict]:
        result = []
        for block in blocks[: self.MAX_TEXT_BLOCKS]:
            bx0, by0, bx1, by1 = block["bbox"]
            cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
            lat, lon = transform.pixel_to_latlon(cx, cy)
            result.append({
                "text":       block["text"],
                "approx_lat": round(lat, 5),
                "approx_lon": round(lon, 5),
                "font_size":  round(block.get("size", 0), 1),
                "inset":      block.get("inset", False),
            })
        return result

    def _format_symbols(
        self, symbols: list[DetectedSymbol], transform: GeoTransform
    ) -> list[dict]:
        result = []
        for sym in symbols:
            x, y, w, h = sym.bbox_px
            cx, cy = x + w / 2, y + h / 2
            lat, lon = transform.pixel_to_latlon(cx, cy)
            result.append({
                "type":       sym.type,
                "label":      sym.label,
                "confidence": round(sym.confidence, 2),
                "approx_lat": round(lat, 5),
                "approx_lon": round(lon, 5),
            })
        return result
