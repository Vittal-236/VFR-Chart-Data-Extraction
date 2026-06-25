import fitz
from pathlib import Path
from PIL import Image

# =============================================================================
# CONFIG
# =============================================================================

PDF_DIR = Path("charts_pdf")
OUTPUT_DIR = Path("rendered_png")

DPI = 300

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# RENDER
# =============================================================================

zoom = DPI / 72
matrix = fitz.Matrix(zoom, zoom)

pdf_files = list(PDF_DIR.glob("*.pdf"))

print(f"\nFound {len(pdf_files)} PDFs")

for pdf_path in pdf_files:

    print(f"\nRendering: {pdf_path.name}")

    doc = fitz.open(pdf_path)

    pdf_out_dir = OUTPUT_DIR / pdf_path.stem
    pdf_out_dir.mkdir(exist_ok=True)

    metadata = {
        "pdf_name": pdf_path.name,
        "dpi": DPI,
        "pages": len(doc),
    }

    for page_num in range(len(doc)):

        page = doc.load_page(page_num)

        pix = page.get_pixmap(
            matrix=matrix,
            alpha=False
        )

        out_path = pdf_out_dir / f"page_{page_num+1}.png"

        img = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples
        )

        img.save(out_path)

        print(f"  Saved: {out_path.name}")

    doc.close()

print("\nDONE")