# VFR Chart Extraction Pipeline — FAA Base Model

Deterministic aeronautical data extraction from FAA VFR Sectional Charts.
No deep learning. Every step is mathematically explainable and reproducible.

---

## Project Structure

```
vfr_pipeline/
├── phase1_preprocessing.py      ← INPUT: raw PDF/PNG  →  OUTPUT: clean RGB + binary arrays
├── phase2_layer_separation.py   ←
├── phase3_georeferencing.py     ←
├── phase3_5_inset_detection.py  ←
├── phase4_symbol_detection.py   ← (supervisor-assigned algorithm)
├── phase5_ocr.py                ←
├── phase6_metadata_mapping.py   ←
├── phase7_vectorisation.py      ←
├── phase8_validation.py         ←
├── phase9_storage.py            ←
├── requirements.txt
└── outputs/
    ├── phase1/                  ← <stem>_rgb.png, <stem>_binary.png, <stem>_log.json
    └── ...
```

---

## Setup

```bash
conda create -n vfr_pipeline python=3.10
conda activate vfr_pipeline
pip install -r requirements.txt
```

---

## Phase 1 — Input Preprocessing

### What it does

| Step | Method | Why |
|------|--------|-----|
| Rasterise PDF | PyMuPDF at 300 DPI | FAA standard resolution for all symbology |
| Denoise | Non-local means (NLM) | Preserves thin text strokes better than Gaussian/median |
| Binarise | Sauvola adaptive threshold | Handles uneven topo shading on sectionals |
| Deskew | Probabilistic Hough → median angle → rotate | Corrects scan/print skew before georef |

### CLI

```bash
# Full run
python phase1_preprocessing.py --input Washington.pdf --output-dir outputs/phase1

# Skip denoising for a fast debug pass
python phase1_preprocessing.py --input Washington.pdf --output-dir outputs/phase1 --skip-denoise

# Custom DPI (150 for quick preview)
python phase1_preprocessing.py --input Washington.pdf --output-dir outputs/phase1 --dpi 150
```

### Module usage (from later phases)

```python
from phase1_preprocessing import preprocess_chart

result = preprocess_chart(
    input_path="Washington.pdf",
    output_dir="outputs/phase1",
    dpi=300,
)

# result.rgb_array    → (H, W, 3) uint8 — denoised colour image
# result.binary_array → (H, W) bool   — Sauvola binarised
# result.skew_angle_deg               — angle that was corrected
# result.rgb_path                     — path to saved PNG
```

### Outputs

| File | Description |
|------|-------------|
| `<stem>_rgb.png` | Denoised, deskewed colour image |
| `<stem>_binary.png` | Sauvola binary mask (white = foreground) |
| `<stem>_log.json` | All parameters, timings, and notes |

### Key parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--dpi` | 300 | 300 is FAA standard. Use 150 for fast debug previews. |
| `--sauvola-window` | 25 | Increase to 51 if text is very large; decrease to 15 for dense small text |
| `--sauvola-k` | 0.2 | Lower = more aggressive binarisation. Range: 0.1–0.5 |
| `--skew-threshold` | 0.5 | Only correct if skew > this many degrees |
| `--skip-denoise` | False | Flag for fast debug runs; disables NLM |

---

## Memory note

A 300-DPI render of the Washington sectional (~56" × 41") produces an array
roughly **16 800 × 12 300 px → ~620 MB** for the RGB array alone. Ensure at
least 4–6 GB of free RAM before running at full DPI. Use `--dpi 150` during
development.
