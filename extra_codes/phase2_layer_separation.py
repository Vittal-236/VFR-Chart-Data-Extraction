"""
Phase 2 — Layer Separation
VFR Chart Extraction Pipeline (FAA Base Model)

Splits the Phase 1 binary mask into six semantically distinct ink layers:

    SYMBOL   — aeronautical symbols (obstacle triangles, airport glyphs,
                VOR rosettes, NDB circles, heliport H, seaplane anchors)
    TEXT     — alphanumeric annotation labels and legend text
    AIRSPACE — airspace boundary arcs and rings (Class B/C/D/E circles,
                ADIZ boundary, warning-area polygons)
    AIRWAY   — victor airway centre-lines and MEA/MOCA altitude boxes
    TOPO     — topographic contour lines and elevation tinting edges
    GRID     — lat/lon tick marks and neat-line border

Each CC in the binary is assigned to exactly one layer based on five
geometric discriminators computed from skimage.regionprops:

    chord      = area / perimeter          (stroke half-width proxy)
    aspect     = max_dim / min_dim         (elongation)
    solidity   = area / convex_hull_area   (fill density)
    euler      = euler_number              (< 0 ↔ character holes)
    local_density = Gaussian blur response at centroid (neighbourhood ink density)

Decision tree (ordered, first match wins):
    1. area ≤ 3                          → discard (noise)
    2. aspect > 50 AND max_dim > 50      → GRID (tick lines)
    3. min_dim == 1 AND area < 15        → GRID (thin tick fragments)
    4. euler < −3 AND area < 400         → TEXT (multi-hole glyphs)
    5. local_density > 0.13 AND area<100 → AIRSPACE (arc cluster)
    6. max_dim > 25 AND chord < 1.25
         AND solidity < 0.45             → branch:
           aspect > 6                   → AIRWAY (elongated stroke)
           else                         → AIRSPACE (arc/ring)
    7. max_dim > 60 AND solidity > 0.05
         AND chord < 1.5                → TOPO (long contour stroke)
    8. area > 300 AND max_dim < 80
         AND solidity > 0.3             → SYMBOL (large compound glyph)
    9. area < 200 AND max_dim < 40
         AND aspect < 6                 → SYMBOL (compact isolated glyph)
   10. remainder                        → TEXT

Outputs (per layer, in output_dir):
    <stem>_layer_symbol.png
    <stem>_layer_text.png
    <stem>_layer_airspace.png
    <stem>_layer_airway.png
    <stem>_layer_topo.png
    <stem>_layer_grid.png
    <stem>_layer_separation_log.json

PowerShell CLI:
    python phase2_layer_separation.py `
        --input outputs/phase1/Washington_binary.png `
        --output-dir outputs/phase2

Module:
    from phase2_layer_separation import separate_layers
    result = separate_layers(
        binary_path="outputs/phase1/Washington_binary.png",
        output_dir="outputs/phase2",
    )
    # result.layers["SYMBOL"]  → (H,W) bool array
    # result.layer_paths       → dict  name → PNG path
    # result.cc_table          → list of dicts (one per CC, with all metrics)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from skimage import filters, measure

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  [%(levelname)s]  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("phase2")

LAYERS = ["SYMBOL", "TEXT", "AIRSPACE", "AIRWAY", "TOPO", "GRID", "NOISE"]
LAYER_COLOURS = {          # for composite debug image (RGB)
    "SYMBOL":   (255, 200,   0),   # yellow
    "TEXT":     (255, 255, 255),   # white
    "AIRSPACE": ( 80, 160, 255),   # blue
    "AIRWAY":   (255, 100, 100),   # red
    "TOPO":     (100, 220, 100),   # green
    "GRID":     (160, 160, 160),   # grey
    "NOISE":    ( 40,  40,  40),   # dark (invisible on black bg)
}


@dataclass
class LayerResult:
    layers:      Dict[str, np.ndarray]   # name → (H,W) bool
    cc_table:    List[dict]              # per-CC metrics + assignment
    layer_paths: Dict[str, str]          # name → PNG path (if saved)
    composite_path: Optional[str]
    input_path:  str
    stats:       dict = field(default_factory=dict)
    elapsed_sec: float = 0.0

    def summary(self) -> str:
        lines = ["=== Phase 2 Layer Separation ===",
                 f"  Input   : {self.input_path}"]
        for name in LAYERS:
            n = self.stats.get(name, 0)
            lines.append(f"  {name:<10}: {n:>5} CCs")
        lines.append(f"  Elapsed : {self.elapsed_sec:.1f} s")
        return "\n".join(lines)


# ── helpers ──────────────────────────────────────────────────────────────────

def _classify(area, max_d, min_d, aspect, chord, solidity,
              euler, local_density) -> str:
    if area <= 3:
        return "NOISE"
    if aspect > 50 and max_d > 50:
        return "GRID"
    if min_d == 1 and area < 15:
        return "GRID"
    if euler < -3 and area < 400:
        return "TEXT"
    if local_density > 0.13 and area < 100:
        return "AIRSPACE"
    if max_d > 25 and chord < 1.25 and solidity < 0.45:
        return "AIRWAY" if aspect > 6 else "AIRSPACE"
    if max_d > 60 and 0.02 < solidity and chord < 1.5:
        return "TOPO"
    if area > 300 and max_d < 80 and solidity > 0.3:
        return "SYMBOL"
    if area < 200 and max_d < 40 and aspect < 6:
        return "SYMBOL"
    return "TEXT"


# ── public API ────────────────────────────────────────────────────────────────

def separate_layers(
    binary_path: str,
    output_dir: Optional[str] = None,
    density_sigma: float = 8.0,
    save_composite: bool = True,
) -> LayerResult:
    t0  = time.time()
    src = Path(binary_path)
    arr = np.array(Image.open(str(src)).convert("L"))
    fg  = arr > 128
    H, W = fg.shape
    log.info(f"Loaded {src.name}  {W}×{H}  fg={fg.mean()*100:.2f}%")

    log.info("Computing density map …")
    density = filters.gaussian(fg.astype(np.float32), sigma=density_sigma)

    log.info("Labelling connected components …")
    labeled = measure.label(fg, connectivity=2)
    props   = measure.regionprops(labeled)
    log.info(f"  {len(props)} CCs found")

    r_dens = int(density_sigma * 2)
    layer_masks: Dict[str, np.ndarray] = {
        name: np.zeros((H, W), dtype=bool) for name in LAYERS
    }
    cc_table: List[dict] = []
    stats: Dict[str, int] = {name: 0 for name in LAYERS}

    for p in props:
        bb   = p.bbox
        h, w = bb[2]-bb[0], bb[3]-bb[1]
        max_d = max(h, w);  min_d = max(min(h, w), 1)
        aspect    = max_d / min_d
        area      = p.area
        chord     = area / max(p.perimeter, 1.0)
        solidity  = p.solidity
        euler     = p.euler_number
        cy = int(np.clip(p.centroid[0], 0, H-1))
        cx = int(np.clip(p.centroid[1], 0, W-1))
        y0,y1 = max(0,cy-r_dens), min(H,cy+r_dens)
        x0,x1 = max(0,cx-r_dens), min(W,cx+r_dens)
        ld = float(density[y0:y1, x0:x1].mean())

        layer = _classify(area, max_d, min_d, aspect, chord, solidity, euler, ld)
        layer_masks[layer][labeled == p.label] = True
        stats[layer] += 1

        cc_table.append({
            "label": int(p.label), "layer": layer,
            "area": int(area), "max_dim": int(max_d),
            "aspect": round(aspect, 2), "chord": round(chord, 3),
            "solidity": round(solidity, 3), "euler": int(euler),
            "local_density": round(ld, 4),
            "centroid_y": round(p.centroid[0], 1),
            "centroid_x": round(p.centroid[1], 1),
        })

    log.info("Layer stats: " + " | ".join(
        f"{k}={v}" for k,v in stats.items() if k != "NOISE"))

    # ── save ─────────────────────────────────────────────────────────────
    layer_paths: Dict[str, str] = {}
    composite_path = None

    if output_dir:
        out = Path(output_dir);  out.mkdir(parents=True, exist_ok=True)
        stem = src.stem
        for name, mask in layer_masks.items():
            if name == "NOISE":
                continue
            p_path = out / f"{stem}_layer_{name.lower()}.png"
            Image.fromarray((mask * 255).astype(np.uint8)).save(str(p_path))
            layer_paths[name] = str(p_path)
            log.info(f"  Saved {name} layer → {p_path.name}")

        if save_composite:
            comp = np.zeros((H, W, 3), dtype=np.uint8)
            for name, mask in layer_masks.items():
                r, g, b = LAYER_COLOURS[name]
                comp[mask, 0] = r;  comp[mask, 1] = g;  comp[mask, 2] = b
            cp = out / f"{stem}_layer_composite.png"
            Image.fromarray(comp).save(str(cp))
            composite_path = str(cp)
            log.info(f"  Saved composite → {cp.name}")

        log_path = out / f"{stem}_layer_separation_log.json"
        with open(log_path, "w") as f:
            json.dump({"input": str(src), "stats": stats,
                       "density_sigma": density_sigma,
                       "elapsed_sec": round(time.time()-t0, 2),
                       "cc_count": len(cc_table)}, f, indent=2)

    result = LayerResult(
        layers=layer_masks, cc_table=cc_table,
        layer_paths=layer_paths, composite_path=composite_path,
        input_path=str(src), stats=stats,
        elapsed_sec=time.time()-t0,
    )
    log.info(result.summary())
    return result


def _cli():
    p = argparse.ArgumentParser(
        description="Phase 2 — VFR chart layer separation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input",      required=True)
    p.add_argument("--output-dir", default="outputs/phase2")
    p.add_argument("--density-sigma", type=float, default=8.0)
    p.add_argument("--no-composite", action="store_true")
    a = p.parse_args()
    separate_layers(a.input, a.output_dir, a.density_sigma,
                    save_composite=not a.no_composite)

if __name__ == "__main__":
    _cli()