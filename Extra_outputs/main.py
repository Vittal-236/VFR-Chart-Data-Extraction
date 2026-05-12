"""
main.py  — FIXED + ENHANCED
AeroExtract-AI — Master Orchestrator

Changes vs original:
1. SymbolDetector now receives ansp= so ANSP colour rules are applied.
2. Inset sub-pages are processed through OCR + symbol detection + their
   own georef transform derived from the parent page transform.
3. vector_paths are passed through to the mapper for route extraction.
4. LLMClient.classify() now falls through to the real Claude API first.

Usage:
    python main.py --input data/raw/Greece_vfr.pdf --ansp GR
    python main.py --input data/raw/France_vfr.pdf --ansp FR --prev data/output/france_prev.json
    python main.py --input data/raw/chart.pdf --ansp GR --no-llm
"""

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.ingestion.pdf_parser import PDFParser, PageData
from src.ingestion.image_loader import ImageLoader
from src.vision.ocr_engine import OCREngine
from src.vision.symbol_det import SymbolDetector
from src.vision.georef import Georeferencer
from src.llm.client import LLMClient
from src.llm.mapper import PayloadMapper
from src.processing.validator import Validator
from src.processing.change_det import ChangeDetector
from src.processing.aip_validator import AIDataValidator
from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger("main")
console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AeroExtract-AI: Extract structured aeronautical data from charts."
    )
    parser.add_argument("--input",      required=True, help="Path to chart (PDF/PNG/JPG).")
    parser.add_argument("--ansp",       default="GR",  help="ANSP country code (e.g. GR, FR).")
    parser.add_argument("--prev",       default=None,  help="Previous-cycle JSON for change detection.")
    parser.add_argument("--no-llm",     action="store_true", help="Skip LLM — use rules-based only.")
    parser.add_argument("--output-dir", default=cfg.OUTPUT_DIR)
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(Panel(
        f"[bold cyan]AeroExtract-AI[/bold cyan]\n"
        f"Input : {input_path.name}\n"
        f"ANSP  : {args.ansp}\n"
        f"LLM   : {'disabled' if args.no_llm else 'enabled'}",
        title="Pipeline Start",
    ))

    # ── Step 1: Ingest ─────────────────────────────────────────────────────────
    log.info("[1/7] Ingesting chart …")
    suffix = input_path.suffix.lower()
    if suffix == ".pdf":
        pages = PDFParser(input_path).parse()
    elif suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        pages = ImageLoader(input_path).load()
    else:
        log.error(f"Unsupported file format: {suffix}")
        sys.exit(1)
    log.info(f"  Loaded {len(pages)} page(s).")

    # ── Step 2: OCR ────────────────────────────────────────────────────────────
    log.info("[2/7] Running OCR …")
    ocr = OCREngine()
    for page in pages:
        ocr.process_easyocr(page)   # now also recurses into insets

    # ── Step 3: Symbol detection ───────────────────────────────────────────────
    log.info("[3/7] Detecting symbols …")
    detector   = SymbolDetector(ansp=args.ansp)   # FIXED: pass ANSP
    all_symbols = []
    for page in pages:
        syms = detector.detect(page)
        all_symbols.extend(syms)
        # Also detect symbols in insets
        for inset in page.insets:
            inset_syms = detector.detect(inset)
            all_symbols.extend(inset_syms)
    log.info(f"  Total symbols detected: {len(all_symbols)}")

    # ── Step 4: Georeferencing ─────────────────────────────────────────────────
    log.info("[4/7] Georeferencing …")
    georef    = Georeferencer(ansp=args.ansp)
    transform = georef.build_transform(pages[0])

    # Build separate transforms for each inset (FIXED: was missing)
    for page in pages:
        for inset in page.insets:
            inset.transform = georef.build_transform(inset, parent_transform=transform)

    # ── Step 5: LLM / rules classification ────────────────────────────────────
    aero_data: dict = {}
    if not args.no_llm:
        log.info("[5/7] Running LLM classification …")
        mapper = PayloadMapper()
        llm    = LLMClient()

        all_text_blocks = [tb for page in pages for tb in page.text_blocks]
        # Also include inset text blocks
        for page in pages:
            for inset in page.insets:
                all_text_blocks.extend(inset.text_blocks)

        # Collect all vector paths across pages
        all_vector_paths = [vp for page in pages for vp in page.vector_paths]

        combined = PageData(
            page_number=0,
            width_pt=pages[0].width_pt,
            height_pt=pages[0].height_pt,
            text_blocks=all_text_blocks,
        )

        chart_metadata = {
            "ansp":        args.ansp,
            "source_file": input_path.name,
            "total_pages": len(pages),
        }

        payload = mapper.build(combined, all_symbols, transform, chart_metadata)
        payload["vector_paths"] = all_vector_paths   # pass paths for route extraction
        aero_data = llm.classify(payload)
    else:
        log.info("[5/7] LLM step skipped (--no-llm flag).")
        aero_data = _build_offline_data(all_symbols, transform)

    # ── Step 6: Validation ─────────────────────────────────────────────────────
    log.info("[6/7] Validating …")
    validator  = Validator()
    val_result = validator.validate(aero_data)
    for err in val_result.errors:
        log.error(f"  VALIDATION ERROR: {err}")
    for warn in val_result.warnings:
        log.warning(f"  VALIDATION WARNING: {warn}")

    aip_val   = AIDataValidator(ansp=args.ansp)
    aero_data = aip_val.validate_and_flag(aero_data)

    # ── Step 7: Save output ────────────────────────────────────────────────────
    log.info("[7/7] Saving output …")
    output_path = output_dir / (input_path.stem + "_extracted.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(aero_data, f, indent=2, ensure_ascii=False)
    log.info(f"  Output saved to: {output_path}")

    # ── Optional: Change detection ─────────────────────────────────────────────
    if args.prev:
        log.info("[+] Running change detection vs previous cycle …")
        change_det  = ChangeDetector()
        report      = change_det.compare(output_path, args.prev)
        report_path = output_dir / (input_path.stem + "_changes.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        console.print(f"\n[bold yellow]Change Report:[/bold yellow] {report['summary']}")
        log.info(f"  Change report saved to: {report_path}")

    _print_summary(aero_data, val_result, output_path)


def _build_offline_data(symbols, transform) -> dict:
    data = {
        "waypoints":[], "holding_patterns":[], "frequencies":[],
        "routes":[], "nhp":[], "obstacles":[], "heli_routes":[], "other":[],
    }
    for sym in symbols:
        x, y, w, h = sym.bbox_px
        lat, lon = transform.pixel_to_latlon(x + w/2, y + h/2)
        entry = {"name": sym.label or "UNKNOWN", "lat": round(lat, 5), "lon": round(lon, 5)}
        if sym.type == "waypoint":
            data["waypoints"].append(entry)
        elif sym.type == "holding":
            data["holding_patterns"].append({
                "name": sym.label, "lat": round(lat, 5), "lon": round(lon, 5),
                "fix": sym.label, "inbound_track_deg": None, "turn": None,
            })
        elif sym.type == "obstacle":
            data["obstacles"].append({
                "description": sym.label, "lat": round(lat, 5), "lon": round(lon, 5),
                "height_ft_amsl": None, "height_ft_agl": None, "lighted": None,
            })
        else:
            data["other"].append({
                "category": sym.type,
                "description": sym.label or "",
                "lat": round(lat, 5), "lon": round(lon, 5),
            })
    return data


def _print_summary(aero_data: dict, val_result, output_path: Path) -> None:
    table = Table(title="Extraction Summary", show_header=True, header_style="bold magenta")
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right")
    categories = [
        ("Waypoints",        "waypoints"),
        ("Holding Patterns", "holding_patterns"),
        ("Frequencies",      "frequencies"),
        ("Routes",           "routes"),
        ("NHP",              "nhp"),
        ("Obstacles",        "obstacles"),
        ("Heli-Routes",      "heli_routes"),
        ("Other",            "other"),
    ]
    for label, key in categories:
        count = len(aero_data.get(key, []))
        table.add_row(label, str(count))

    console.print(table)
    status = (
        "[green]✓ Valid[/green]" if val_result.is_valid
        else f"[red]✗ {len(val_result.errors)} error(s)[/red]"
    )
    console.print(f"\nValidation: {status}")
    console.print(f"Output: [bold]{output_path}[/bold]\n")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
