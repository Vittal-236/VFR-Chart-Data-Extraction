"""
image_loader.py
Loads PNG / JPEG chart files and wraps them in the same PageData structure
used by the PDF parser, so the rest of the pipeline is format-agnostic.
"""

from pathlib import Path
import cv2
import numpy as np
from src.ingestion.pdf_parser import PageData
from src.utils.logger import get_logger

log = get_logger(__name__)

# Target long-edge resolution.  Charts under this are upscaled for better OCR.
TARGET_LONG_EDGE_PX = 3000


class ImageLoader:
    """
    Usage:
        loader = ImageLoader("data/raw/france_vfr.png")
        pages = loader.load()   # returns list with a single PageData
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Image not found: {self.path}")
        if self.path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            raise ValueError(f"Unsupported image format: {self.path.suffix}")

    def load(self) -> list[PageData]:
        log.info(f"Loading image '{self.path.name}'")
        img = cv2.imread(str(self.path))
        if img is None:
            raise IOError(f"OpenCV could not read image: {self.path}")

        img = self._normalise_resolution(img)
        h, w = img.shape[:2]

        pd = PageData(
            page_number=0,
            width_pt=float(w),    # for images, 1 pt = 1 px
            height_pt=float(h),
            text_blocks=[],       # populated later by OCR engine
            image_cv2=img,
        )
        return [pd]

    # â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _normalise_resolution(self, img: np.ndarray) -> np.ndarray:
        """Upscale small images so OCR has enough pixels to work with."""
        h, w = img.shape[:2]
        long_edge = max(h, w)
        if long_edge >= TARGET_LONG_EDGE_PX:
            return img
        scale = TARGET_LONG_EDGE_PX / long_edge
        new_w, new_h = int(w * scale), int(h * scale)
        log.info(f"  Upscaling image from {w}Ã—{h} â†’ {new_w}Ã—{new_h}")
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
