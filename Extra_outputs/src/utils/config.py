"""
config.py
Central configuration â€” loads API keys, thresholds, ANSP-specific rules.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env file at project root


class Config:
    # â”€â”€ LLM â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    LLM_MODEL: str = "claude-sonnet-4-20250514"
    LLM_MAX_TOKENS: int = 2048

    # â”€â”€ OCR â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    TESSERACT_CMD: str = os.getenv("TESSERACT_CMD", "tesseract")
    # If using Azure Read API instead:
    AZURE_VISION_ENDPOINT: str = os.getenv("AZURE_VISION_ENDPOINT", "")
    AZURE_VISION_KEY: str = os.getenv("AZURE_VISION_KEY", "")

    # â”€â”€ Vision / Detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    SYMBOL_CONF_THRESHOLD: float = 0.5   # min confidence for detected symbols
    YOLO_WEIGHTS: str = "models/vision/aero_symbols.pt"  # custom YOLO weights path

    # â”€â”€ Georeferencing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Pixel tolerance when snapping detected text to a coordinate grid line
    GRID_SNAP_TOLERANCE_PX: int = 20

    # â”€â”€ Output â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    OUTPUT_DIR: str = "data/output"
    PROCESSED_DIR: str = "data/processed"

    # â”€â”€ ANSP-specific rules â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Each ANSP can override symbol colours, coordinate formats, etc.
    ANSP_RULES: dict = {
        "GR": {   # Greece
            "waypoint_color_bgr": (0, 0, 255),    # red triangles
            "coordinate_format": "DD MM SS",
        },
        "FR": {   # France
            "waypoint_color_bgr": (255, 0, 0),    # blue triangles
            "coordinate_format": "DD MM.mmm",
        },
        "US": {
            "waypoint_color_bgr": (0, 0, 200),
            "coordinate_format": "DD MM SS",
        }
    }

    # â”€â”€ Aeronautical validation rules â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    VALID_VHF_RANGE = (108.0, 137.975)   # MHz
    VALID_UHF_RANGE = (225.0, 400.0)     # MHz
    MAX_OBSTACLE_HEIGHT_FT = 99999


cfg = Config()
