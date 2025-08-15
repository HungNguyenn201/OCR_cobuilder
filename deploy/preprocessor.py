from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np
from header import (
    PREFER_TILE, 
    MAX_TILE_LIMIT, 
    OVERLAP_RATIO, 
    MIN_PAD, 
    MAX_PAD
)

# ---------------------- Small utils ---------------------- #
def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

# ---------------------- Fast font height estimator ---------------------- #
def estimate_font_height(tile_bgr: np.ndarray) -> float:
    """
    Very fast proxy for average glyph height in a tile.
    Uses Canny->contours and returns robust median of bounding-rect heights.
    """
    if tile_bgr is None or tile_bgr.size == 0:
        return 12.0
    g = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    # light denoise to stabilize edges
    g = cv2.fastNlMeansDenoising(g, None, 7, 7, 21)
    edges = cv2.Canny(g, 60, 180)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = g.shape[:2]
    hs: List[int] = []
    # collect "text-like" components (not full-frame lines/rects)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if h < 6:               # too tiny
            continue
        if h > 0.60 * H:        # huge object -> not a glyph
            continue
        if w > 0.90 * W:        # long line
            continue
        hs.append(h)

    if not hs:
        return 12.0
    # robust: median of middle 60% to avoid outliers
    hs_sorted = np.sort(np.array(hs))
    lo = int(0.20 * len(hs_sorted))
    hi = int(0.80 * len(hs_sorted))
    return float(np.median(hs_sorted[lo: max(lo+1, hi)]))

# ---------------------- Textual tile quick check ---------------------- #
def is_textual_tile(tile_bgr: np.ndarray) -> bool:
    """
    Heuristic rất nhẹ (không dùng MSER mặc định):
      - Otsu foreground density
      - Edge density (Canny) với ngưỡng cố định
    """
    g = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - bw
    density = inv.mean() / 255.0

    edges = cv2.Canny(g, 60, 180)
    edge_density = edges.mean() / 255.0

    # Ngưỡng rất lỏng để không bỏ sót tile chữ mảnh
    return (density > 0.010) or (edge_density > 0.010)

# ---------------------- Optional enhancer (off by default) ---------------------- #

def _green_ratio(tile_bgr: np.ndarray) -> float:
    """Heuristic green-ink ratio commonly used in drawings labels."""
    try:
        hsv = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # green-ish band; tune-safe for drawings
        mask = ((h >= 35) & (h <= 95) & (s > 40) & (v > 40)).astype(np.uint8)
        return float(mask.sum()) / mask.size
    except Exception:
        return 0.0
    
def is_textual_tile(
    tile_bgr: np.ndarray,
    *,
    white_threshold: int = 240,
    white_ratio_thresh: float = 0.998,   # allow lots of white but not "all white"
    edge_density_thresh: float = 0.0015, # very loose to keep thin text
    green_ratio_thresh: float = 5e-4,    # if some green text -> keep
) -> bool:
    """
    Decide whether a tile is worth OCR:
      1) If green ratio > threshold -> keep (common in CAD labels)
      2) Else: keep if (white_ratio < thr) AND (edge_density > thr)
    """
    if tile_bgr is None or tile_bgr.size == 0:
        return False

    # 1) color hint
    if _green_ratio(tile_bgr) > green_ratio_thresh:
        return True

    # 2) structure/contrast hints
    g = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    white_ratio = float(np.mean(g > white_threshold))
    edges = cv2.Canny(g, 50, 150)
    edge_density = float(np.mean(edges > 0))

    return (white_ratio < white_ratio_thresh) and (edge_density > edge_density_thresh)

def enhance_tile(tile_bgr: np.ndarray) -> np.ndarray:
    """
    Preserve colored text, gently improve contrast:
      - Boost S/V for green mask
      - CLAHE on L (LAB), then light unsharp mask in gray
    """
    if tile_bgr is None or tile_bgr.size == 0:
        return tile_bgr

    out = tile_bgr.copy()

    # Boost green text a bit
    try:
        hsv16 = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.int16)
        h, s, v = cv2.split(hsv16)
        mask = ((h >= 35) & (h <= 95) & (s > 30) & (v > 30))
        if mask.any():
            s[mask] = np.clip(s[mask] * 1.30, 0, 255)
            v[mask] = np.clip(v[mask] * 1.12, 0, 255)
            out = cv2.cvtColor(
                cv2.merge((h.astype(np.uint8), s.astype(np.uint8), v.astype(np.uint8))),
                cv2.COLOR_HSV2BGR
            )
    except Exception:
        pass

    # CLAHE on L channel
    try:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        L2 = clahe.apply(L)
        out = cv2.cvtColor(cv2.merge((L2, A, B)), cv2.COLOR_LAB2BGR)
    except Exception:
        pass

    # light unsharp mask (in gray) to enhance thin strokes
    try:
        g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(g, (0, 0), 1.0)
        sharp = cv2.addWeighted(g, 1.5, blur, -0.5, 0)
        out = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)
    except Exception:
        pass

    return out

# ---------------------- Adaptive tile size ---------------------- #
def get_tile_size_adaptive(
            image_shape: Tuple[int, int, int],
            max_size: Tuple[int, int] = MAX_TILE_LIMIT,
            prefer_size: Tuple[int, int] = PREFER_TILE
        ) -> Tuple[int, int]:
    """
    - Tiny images: return full image (no tiling)
    - Medium: use prefer_size
    - Large: split ~3 parts per axis, clamp [256 .. max_size]
    """
    H, W = image_shape[:2]
    pw, ph = prefer_size
    mw, mh = max_size

    if W <= pw and H <= ph:
        return (int(W), int(H))

    if W <= int(mw * 1.5) and H <= int(mh * 1.5):
        return (int(pw), int(ph))

    tile_w = min(mw, max(256, W // 3))
    tile_h = min(mh, max(256, H // 3))
    return (int(tile_w), int(tile_h))

def split_image_to_tiles(
    image_bgr: np.ndarray,
    tile_size: Optional[Tuple[int, int]] = None,
    *,
    overlap_ratio: float = OVERLAP_RATIO,
    min_pad: int = MIN_PAD,
    max_pad: int = MAX_PAD,
    keep_border_tiles: bool = True,
    apply_enhance: bool = True,
    detect_textual: bool = True,
    debug_print: bool = False,
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """
    Split image to padded tiles ready for OCR.
    - Filters non-textual tiles (but keeps border tiles to avoid losing titles/frames)
    - Pads each tile with dynamic replicate-border pad derived from local font height
    - (Optionally) enhances tiles for thin strokes and green labels

    Returns:
      tiles: list of padded BGR tiles
      metas: list of dicts with mapping info:
        {
          'tile_id': int,
          'orig_x': int, 'orig_y': int,        # top-left in full image
          'tile_w': int, 'tile_h': int,        # crop size before pad
          'pad': int,                          # symmetric pad used (approx.)
          'pad_l': int, 'pad_t': int, 'pad_r': int, 'pad_b': int,  # exact pads
          'step_x': int, 'step_y': int,        # stride (for seam-aware post)
          'full_w': int, 'full_h': int,        # image size
          'grid_x': int, 'grid_y': int,        # grid indices
          'overlap_ratio': float
        }
    """
    assert image_bgr is not None and image_bgr.size > 0, "Empty image passed to splitter"

    H, W = image_bgr.shape[:2]
    if tile_size is None:
        tile_size = get_tile_size_adaptive(image_bgr.shape)

    tw, th = map(int, tile_size)
    overlap_ratio = _clamp(float(overlap_ratio), 0.0, 0.5)
    step_x = max(1, int(round(tw * (1.0 - overlap_ratio))))
    step_y = max(1, int(round(th * (1.0 - overlap_ratio))))

    # Enumerate grid starts (ensure last tile covers the end)
    xs = list(range(0, max(1, W), step_x))
    ys = list(range(0, max(1, H), step_y))
    if xs and xs[-1] + tw < W:
        xs.append(max(0, W - tw))
    if ys and ys[-1] + th < H:
        ys.append(max(0, H - th))

    tiles: List[np.ndarray] = []
    metas: List[Dict[str, Any]] = []
    tid = 0

    for gi, y0 in enumerate(ys):
        for gj, x0 in enumerate(xs):
            x1 = min(x0 + tw, W)
            y1 = min(y0 + th, H)
            crop = image_bgr[y0:y1, x0:x1]
            if crop is None or crop.size == 0:
                continue

            at_border = keep_border_tiles and (x0 == 0 or y0 == 0 or x1 == W or y1 == H)
            if detect_textual:
                if not (is_textual_tile(crop) or at_border):
                    # skip non-textual inner tiles
                    continue

            # dynamic pad from local font height
            fH = estimate_font_height(crop)
            dyn_pad = int(_clamp(round(0.5 * fH), min_pad, max_pad))

            # symmetric replicate pad (mapping subtracts 'pad' back)
            pad_l = pad_t = pad_r = pad_b = dyn_pad
            tile_padded = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)

            if apply_enhance:
                tile_padded = enhance_tile(tile_padded)

            tiles.append(tile_padded)
            metas.append({
                'tile_id': tid,
                'orig_x': int(x0),
                'orig_y': int(y0),
                'tile_w': int(x1 - x0),
                'tile_h': int(y1 - y0),
                'pad': int(dyn_pad),                     # backward-compat field
                'pad_l': int(pad_l), 'pad_t': int(pad_t),
                'pad_r': int(pad_r), 'pad_b': int(pad_b),
                'step_x': int(step_x), 'step_y': int(step_y),
                'full_w': int(W), 'full_h': int(H),
                'grid_x': int(gj), 'grid_y': int(gi),
                'overlap_ratio': float(overlap_ratio),
            })
            tid += 1

    if debug_print:
        print(f"[PRE] img={W}x{H} tile={tw}x{th} overlap={overlap_ratio:.2f} "
              f"stride=({step_x},{step_y}) grid=({len(xs)}x{len(ys)}) tiles={len(tiles)}")

    return tiles, metas
