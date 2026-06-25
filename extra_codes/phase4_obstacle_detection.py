"""
Phase 4 — Obstacle Detection (Blob-based Multi-Scale Template Matching)
VFR Chart Extraction Pipeline (FAA Base Model)

Changes from original:
    B1 — build_template() fixed:
           - Base edge REMOVED (FAA symbol is open ∧, not closed △)
           - Dot anchor moved to base-centre (bottom-centre of bbox),
             not 55% from apex
    B5 — Overlap scoring now uses two separate templates:
           - Triangle template (left + right edges only)
           - Dot template (filled circle at base-centre)
         Both are scored independently. A blob must pass the triangle
         threshold to be accepted. The dot score is logged separately
         for future use but does not gate acceptance yet, because the
         phase 2a mask may contain triangle-only fragments (dot filtered
         out by the area minimum in phase 2a).

Input:  Phase 2a obstacle mask PNG
Output: Annotated RGB PNG + detection JSON

Usage (PowerShell):
    python phase4_obstacle_detection.py `
        --mask   outputs/phase2a/Washington_obstacle_mask.png `
        --rgb    outputs/phase1_150/Washington_rgb.png `
        --output-dir outputs/phase4
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw
from skimage.measure import label, regionprops
from skimage.transform import resize as sk_resize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase4")


def build_triangle_template(size: int = 64) -> np.ndarray:
    """
    Build a canonical open-V triangle template at internal resolution.

    FAA obstacle symbol — triangle component:
      - TWO edges only: left edge (apex → base-left) and
        right edge (apex → base-right)
      - NO base edge — the real symbol is an open ∧ shape
      - All symbols on the map are at the same fixed angle (no rotation)

    Built at size x size pixels. Gets resized to each blob's bounding
    box during detection so it adapts to any DPI automatically.
    """
    img = np.zeros((size, size), dtype=np.float64)

    apex_r  = 2
    apex_c  = size // 2
    base_r  = size - 3
    base_lc = 2
    base_rc = size - 3

    def draw_edge(r0, c0, r1, c1, thickness=3):
        steps = max(abs(r1 - r0), abs(c1 - c0), 1) * 3
        for i in range(steps + 1):
            t  = i / steps
            r  = int(round(r0 + t * (r1 - r0)))
            c  = int(round(c0 + t * (c1 - c0)))
            for dr in range(-thickness // 2, thickness // 2 + 1):
                for dc in range(-thickness // 2, thickness // 2 + 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < size and 0 <= nc < size:
                        img[nr, nc] = 1.0

    draw_edge(apex_r, apex_c, base_r, base_lc)  # left edge  ✓
    draw_edge(apex_r, apex_c, base_r, base_rc)  # right edge ✓
    # NO base edge — this was the B1 bug

    return img


def build_dot_template(size: int = 64) -> np.ndarray:
    """
    Build a canonical dot template at internal resolution.

    FAA obstacle symbol — dot component:
      - Filled circle sitting at the midpoint of where the base would be
        (bottom-centre of the bounding box)
      - Radius approximately 8% of symbol width
    """
    img = np.zeros((size, size), dtype=np.float64)

    # Dot is at base-centre — bottom of the bounding box, horizontally centred
    dot_r      = size - 3   # same row as where base would be
    dot_c      = size // 2
    dot_radius = max(2, round(size * 0.08))

    for dr in range(-dot_radius, dot_radius + 1):
        for dc in range(-dot_radius, dot_radius + 1):
            if dr * dr + dc * dc <= dot_radius * dot_radius:
                rr, cc = dot_r + dr, dot_c + dc
                if 0 <= rr < size and 0 <= cc < size:
                    img[rr, cc] = 1.0

    return img


def score_blob(
    blob_patch:   np.ndarray,
    tmpl_bin:     np.ndarray,
    bh:           int,
    bw:           int,
) -> float:
    """
    Resize tmpl_bin to (bh, bw) and compute pixel overlap ratio
    against blob_patch.

    Returns overlap ratio in [0, 1]:
        overlap_pixels / template_pixels
    """
    tmpl_resized = sk_resize(
        tmpl_bin.astype(np.float64),
        (bh, bw),
        order=1,
        anti_aliasing=True,
    ) > 0.4

    n_tmpl = tmpl_resized.sum()
    if n_tmpl == 0:
        return 0.0

    overlap = np.logical_and(tmpl_resized, blob_patch > 0.5).sum()
    return float(overlap) / float(n_tmpl)


def detect_obstacles(
    obstacle_mask:     np.ndarray,
    rgb:               np.ndarray,
    output_dir:        Optional[str] = None,
    stem:              str           = "Washington",
    overlap_threshold: float         = 0.45,
) -> dict:
    """
    Detect obstacles from the Phase 2a mask.

    For each blob:
        - Resize triangle template to blob bounding box, compute overlap
        - Resize dot template to blob bounding box, compute overlap (logged)
        - Reject legend strip (left 18%)
        - Accept if triangle overlap >= overlap_threshold
    """
    t0 = time.time()
    log.info("=" * 55)
    log.info("Phase 4 — Obstacle Detection  (v2)")
    log.info("=" * 55)
    log.info(f"  Overlap threshold: {overlap_threshold}")

    H, W = obstacle_mask.shape

    # Build both templates at internal resolution
    tri_tmpl = build_triangle_template(size=64) > 0.5
    dot_tmpl = build_dot_template(size=64)      > 0.5

    legend = int(W * 0.18)

    labelled = label(obstacle_mask, connectivity=2)
    props    = regionprops(labelled)
    log.info(f"  Blobs to test: {len(props)}")

    detections       = []
    n_legend         = 0
    n_overlap_fail   = 0
    n_accepted       = 0

    for p in props:
        r0, c0, r1, c1 = p.bbox
        bh = r1 - r0
        bw = c1 - c0
        cx = (c0 + c1) // 2
        cy = (r0 + r1) // 2

        # Reject legend strip
        if cx < legend:
            n_legend += 1
            continue

        if bh < 2 or bw < 2:
            continue

        # Extract blob pixels normalised to [0, 1]
        blob_patch = obstacle_mask[r0:r1, c0:c1].astype(np.float64) / 255.0

        # Score against triangle template (primary gate)
        tri_ratio = score_blob(blob_patch, tri_tmpl, bh, bw)

        # Score against dot template (logged, not gating yet)
        dot_ratio = score_blob(blob_patch, dot_tmpl, bh, bw)

        if tri_ratio >= overlap_threshold:
            detections.append({
                "pixel_x":         int(cx),
                "pixel_y":         int(cy),
                "tri_overlap":     round(tri_ratio, 4),
                "dot_overlap":     round(dot_ratio, 4),
                "blob_area_px2":   int(p.area),
                "blob_w":          int(bw),
                "blob_h":          int(bh),
                "solidity":        round(float(p.solidity), 4),
                "bbox":            [int(r0), int(c0), int(r1), int(c1)],
            })
            n_accepted += 1
            log.info(
                f"  ✓ ({cx:5d},{cy:5d})  tri={tri_ratio:.3f}  "
                f"dot={dot_ratio:.3f}  {bw}x{bh}px"
            )
        else:
            n_overlap_fail += 1
            log.info(
                f"  ✗ ({cx:5d},{cy:5d})  tri={tri_ratio:.3f}  "
                f"dot={dot_ratio:.3f}  {bw}x{bh}px  REJECTED"
            )

    # Sort top-to-bottom, left-to-right
    detections.sort(key=lambda d: (d["pixel_y"], d["pixel_x"]))

    log.info(f"  Legend rejected:  {n_legend}")
    log.info(f"  Overlap rejected: {n_overlap_fail}")
    log.info(f"  Accepted:         {n_accepted}")

    # Draw on RGB
    log.info("Drawing detections ...")
    vis  = Image.fromarray(rgb)
    draw = ImageDraw.Draw(vis)
    for d in detections:
        r0, c0, r1, c1 = d["bbox"]
        m = 5
        draw.ellipse([c0 - m, r0 - m, c1 + m, r1 + m], outline=(255, 0, 0), width=2)
        draw.text(
            (c1 + m + 2, r0),
            f"t={d['tri_overlap']:.2f}",
            fill=(255, 0, 0),
        )

    elapsed = round(time.time() - t0, 2)
    result  = {
        "stem":              stem,
        "overlap_threshold": overlap_threshold,
        "n_blobs_tested":    len(props),
        "n_legend_rejected": n_legend,
        "n_detections":      len(detections),
        "detections":        detections,
        "elapsed_sec":       elapsed,
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        img_path = out / f"{stem}_phase4_obstacles.png"
        vis.save(str(img_path))
        log.info(f"  Saved → {img_path}")
        log_path = out / f"{stem}_phase4_log.json"
        with open(log_path, "w") as f:
            json.dump(result, f, indent=2)
        log.info(f"  Saved → {log_path}")

    log.info(f"Done in {elapsed}s  |  {len(detections)} detections")
    return result


if __name__ == "__main__":
    Image.MAX_IMAGE_PIXELS = None

    p = argparse.ArgumentParser(
        description="Phase 4 — Obstacle detection from Phase 2a mask (v2)"
    )
    p.add_argument("--mask",       required=True,
                   help="Phase 2a obstacle mask PNG.")
    p.add_argument("--rgb",        required=True,
                   help="Phase 1 RGB image PNG.")
    p.add_argument("--output-dir", default="outputs/phase4")
    p.add_argument("--stem",       default=None)
    p.add_argument("--overlap",    type=float, default=0.45,
                   help="Triangle overlap threshold [0-1]. Default 0.45.")
    args = p.parse_args()

    log.info(f"Loading mask: {args.mask}")
    mask = np.array(
        Image.open(args.mask).convert("L"), dtype=np.uint8
    )
    log.info(f"  {mask.shape[1]}x{mask.shape[0]}px  coverage={mask.mean()*100:.3f}%")

    log.info(f"Loading RGB: {args.rgb}")
    rgb  = np.array(Image.open(args.rgb).convert("RGB"), dtype=np.uint8)
    log.info(f"  {rgb.shape[1]}x{rgb.shape[0]}px")

    stem = args.stem or Path(args.mask).stem.replace("_obstacle_mask", "")

    detect_obstacles(
        obstacle_mask=mask,
        rgb=rgb,
        output_dir=args.output_dir,
        stem=stem,
        overlap_threshold=args.overlap,
    )