from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional, Generator
import cv2
import numpy as np

# ---- Project config ----
from header import (
    PREFER_TILE,
    MAX_TILE_LIMIT,
    OVERLAP_RATIO,
    MIN_PAD,
    MAX_PAD,
    FONT_HEIGHT_SMALL_PX,            # ngưỡng “chữ nhỏ” để quyết định scale 1.25
    APPLY_ENHANCE_CONDITIONAL,       # bật/tắt enhance có điều kiện
)

# ---------------------- Small utils ---------------------- #
def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _pad_rect(x0: int, y0: int, x1: int, y1: int, pad_l: int, pad_t: int, pad_r: int, pad_b: int,
              w: int, h: int) -> Tuple[int, int, int, int, int, int, int, int]:
    """
    Pad bbox trong biên ảnh, trả về bbox đã pad + pad thực tế (có thể bị clamp).
    """
    nx0 = _clamp(x0 - pad_l, 0, w)
    ny0 = _clamp(y0 - pad_t, 0, h)
    nx1 = _clamp(x1 + pad_r, 0, w)
    ny1 = _clamp(y1 + pad_b, 0, h)
    # pad thực tế (sau clamp)
    pl = int(x0 - nx0)
    pt = int(y0 - ny0)
    pr = int(nx1 - x1)
    pb = int(ny1 - y1)
    return int(nx0), int(ny0), int(nx1), int(ny1), pl, pt, pr, pb

# ---------------------- Tile size policy ---------------------- #
def get_tile_size_adaptive(image_shape: Tuple[int, int, int],
                           max_size: Tuple[int, int] = MAX_TILE_LIMIT,
                           prefer_size: Tuple[int, int] = PREFER_TILE) -> Tuple[int, int]:
    """
    Adaptive tile size:
      - Ảnh nhỏ hơn prefer -> return (w,h) (không tách tile)
      - Ảnh tầm trung -> return prefer_size
      - Ảnh lớn -> chia ~1/3 mỗi chiều, clamp [256 .. max_size]
    """
    h, w = image_shape[:2]
    if w <= prefer_size[0] and h <= prefer_size[1]:
        return (int(w), int(h))
    if w <= int(max_size[0] * 1.5) and h <= int(max_size[1] * 1.5):
        return (int(prefer_size[0]), int(prefer_size[1]))
    tile_w = min(max_size[0], max(256, int(w // 3)))
    tile_h = min(max_size[1], max(256, int(h // 3)))
    return (int(tile_w), int(tile_h))

# ---------------------- Tile textual quick check ---------------------- #
def is_textual_tile(tile_bgr: np.ndarray,
                    white_threshold: int = 235,
                    white_ratio_thresh: float = 0.985,
                    edge_density_thresh: float = 0.010) -> bool:
    """
    Heuristic rất nhẹ:
      - White ratio cao -> tile có thể trống
      - Edge density (Canny) -> gần đúng mật độ nét chữ/đường
    Trả True nếu tile có khả năng chứa chữ/đồ hoạ hữu ích.
    """
    if tile_bgr is None or tile_bgr.size == 0:
        return False
    g = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    white_ratio = float(np.mean(g > white_threshold))
    edges = cv2.Canny(g, 50, 150)
    edge_density = float(np.mean(edges > 0))
    return (white_ratio < white_ratio_thresh) and (edge_density > edge_density_thresh)

# ---------------------- Enhance tile ---------------------- #
def _boost_green_hsv(tile_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # Vùng xanh/lam thường thấy trong CAD
    green_mask = ((h >= 60) & (h <= 110) & (s >= 40) & (v >= 50))
    s = np.where(green_mask, np.clip(s.astype(np.int32) + 20, 0, 255).astype(np.uint8), s)
    v = np.where(green_mask, np.clip(v.astype(np.int32) + 10, 0, 255).astype(np.uint8), v)
    hsv2 = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)

def enhance_tile(tile_bgr: np.ndarray) -> np.ndarray:
    """
    Tăng nhẹ độ tương phản & sắc nét nhưng giữ màu chữ:
      - Boost S/V cho mực xanh CAD
      - CLAHE kênh L (LAB)
      - Unsharp mask nhẹ ở grayscale
    """
    if tile_bgr is None or tile_bgr.size == 0:
        return tile_bgr
    bgr = _boost_green_hsv(tile_bgr)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, A, B])
    bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    g = cv2.cvtColor(bgr2, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(g, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(g, 1.25, blur, -0.25, 0)
    bgr2[:, :, 0] = cv2.min(bgr2[:, :, 0], sharp)
    bgr2[:, :, 1] = cv2.min(bgr2[:, :, 1], sharp)
    bgr2[:, :, 2] = cv2.min(bgr2[:, :, 2], sharp)
    return bgr2

# ---------------------- Font height estimation ---------------------- #
def estimate_font_height(image_bgr: np.ndarray) -> float:
    """
    Ước lượng chiều cao chữ xấp xỉ (median contour height),
    dùng cho quyết định scale/overlap toàn ảnh.
    """
    if image_bgr is None or image_bgr.size == 0:
        return 0.0
    g = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(g, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    hs: List[int] = []
    H = g.shape[0]
    for c in cnts[:4000]:  # giới hạn để nhanh
        x, y, w, h = cv2.boundingRect(c)
        if 4 <= h <= int(0.2 * H):
            hs.append(h)
    if not hs:
        return 0.0
    hs_sorted = np.sort(np.array(hs))
    lo = int(0.20 * len(hs_sorted))
    hi = int(0.80 * len(hs_sorted))
    return float(np.median(hs_sorted[lo:max(lo + 1, hi)]))

# ===== quyết định có enhance tile không  =====
def should_enhance_tile(tile_bgr: np.ndarray,
                        green_ratio_thresh: float = 0.010,
                        edge_density_thresh: float = 0.020,
                        white_threshold: int = 235) -> bool:
    """
    Trả True nếu tile có khả năng chữ mảnh/nhạt (nên enhance).
    Heuristic: có “mực xanh” CAD hoặc edge density thấp (thiếu tương phản) nhưng không gần trắng hoàn toàn.
    """
    if tile_bgr is None or tile_bgr.size == 0:
        return False
    hsv = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    green_mask = (((h >= 60) & (h <= 110)) & (s >= 40) & (v >= 50)).astype(np.uint8)
    green_ratio = float(np.mean(green_mask > 0))

    g = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(g, 50, 150)
    edge_density = float(np.mean(edges > 0))
    white_ratio = float(np.mean(g > white_threshold))

    return (green_ratio >= green_ratio_thresh) or ((edge_density < edge_density_thresh) and (white_ratio < 0.98))

# =====: ước lượng font theo tile  =====
def estimate_font_height_tile(tile_bgr: np.ndarray) -> float:
    if tile_bgr is None or tile_bgr.size == 0:
        return 0.0
    g = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(g, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    hs: List[int] = []
    H = g.shape[0]
    for c in cnts[:2000]:
        x, y, w, h = cv2.boundingRect(c)
        if 4 <= h <= int(0.2 * H):
            hs.append(h)
    if not hs:
        return 0.0
    hs.sort()
    return float(hs[len(hs) // 2])

# ---------------------- Split image to tiles (compat API) ---------------------- #
def split_image_to_tiles(image_bgr: np.ndarray,
                         tile_size: Optional[Tuple[int, int]] = None,
                         overlap_ratio: float = OVERLAP_RATIO,
                         add_padding: bool = True,
                         keep_empty_border: bool = True,
                         detect_textual: bool = True,
                         apply_enhance: bool = True,
                         debug_print: bool = False) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """
    Tách ảnh thành list tiles + metas (không streaming).
    - detect_textual: True -> lọc bỏ tile gần trắng/ít nét.
    - apply_enhance: True -> enhance có điều kiện (chỉ khi tile 'khó').
    - Giữ đủ meta để map bbox & vẽ debug (step_x/step_y, grid, ...).
    """
    assert image_bgr is not None and image_bgr.size > 0
    H, W = image_bgr.shape[:2]
    if tile_size is None:
        tile_size = get_tile_size_adaptive(image_bgr.shape)
    tw, th = int(tile_size[0]), int(tile_size[1])

    step_x = max(1, int(tw * (1.0 - float(overlap_ratio))))
    step_y = max(1, int(th * (1.0 - float(overlap_ratio))))

    # lưới đảm bảo "chạm mép"
    xs = list(range(0, max(1, W - tw + 1), step_x))
    ys = list(range(0, max(1, H - th + 1), step_y))
    if not xs or xs[-1] != W - tw:
        xs.append(max(0, W - tw))
    if not ys or ys[-1] != H - th:
        ys.append(max(0, H - th))

    # pad động theo font height toàn ảnh (nhẹ)
    fh_global = estimate_font_height(image_bgr)
    pad_auto = int(_clamp(fh_global * 0.30, MIN_PAD, MAX_PAD)) if add_padding else 0

    tiles: List[np.ndarray] = []
    metas: List[Dict[str, Any]] = []

    tid = 0
    for gi, y0 in enumerate(ys):
        for gj, x0 in enumerate(xs):
            x1 = min(W, x0 + tw)
            y1 = min(H, y0 + th)

            # crop + optional pad (clamp biên)
            nx0, ny0, nx1, ny1, pl, pt, pr, pb = _pad_rect(
                x0, y0, x1, y1,
                pad_auto, pad_auto, pad_auto, pad_auto,
                W, H
            ) if add_padding else (x0, y0, x1, y1, 0, 0, 0, 0)

            crop = image_bgr[ny0:ny1, nx0:nx1]
            if crop is None or crop.size == 0:
                continue

            # lọc tile "trống" nếu bật detect_textual
            if detect_textual and not is_textual_tile(crop):
                if keep_empty_border:
                    # vẫn thêm meta (để vẽ lưới) nhưng bỏ ảnh tile
                    pass
                else:
                    continue

            # enhance có điều kiện nếu bật
            if apply_enhance and should_enhance_tile(crop):
                crop = enhance_tile(crop)

            tiles.append(crop)
            metas.append({
                'id': int(tid),
                'orig_x': int(x0), 'orig_y': int(y0),
                'tile_w': int(x1 - x0), 'tile_h': int(y1 - y0),
                'pad_l': int(pl), 'pad_t': int(pt), 'pad_r': int(pr), 'pad_b': int(pb),
                'step_x': int(step_x), 'step_y': int(step_y),
                'full_w': int(W), 'full_h': int(H),
                'grid_x': int(gj), 'grid_y': int(gi),
                'overlap_ratio': float(overlap_ratio),
            })
            tid += 1

    if debug_print:
        print(f"[PRE] img={W}x{H} tile={tw}x{th} overlap={overlap_ratio:.2f} "
              f"stride=({step_x},{step_y}) grid=({len(xs)}x{len(ys)}) tiles={len(tiles)})")

    return tiles, metas

# ===== NEW: stream tiles để giảm RAM (khuyến nghị dùng trong pipeline) =====
def iter_tiles(image_bgr: np.ndarray,
               tile_size: Tuple[int, int],
               overlap_ratio: float,
               apply_enhance_conditional: bool = APPLY_ENHANCE_CONDITIONAL
               ) -> Generator[Tuple[np.ndarray, Dict[str, Any], bool], None, None]:
    """
    Generator trả từng (tile_bgr, meta, small_font:bool).
    - Không giữ list tiles trong RAM.
    - Mỗi tile có meta đủ để map bbox về toạ độ ảnh gốc.
    - small_font: True nếu font ước lượng theo tile < FONT_HEIGHT_SMALL_PX.
    """
    assert image_bgr is not None and image_bgr.size > 0
    H, W = image_bgr.shape[:2]
    tw, th = tile_size
    step_x = max(1, int(tw * (1.0 - overlap_ratio)))
    step_y = max(1, int(th * (1.0 - overlap_ratio)))

    xs = list(range(0, max(1, W - tw + 1), step_x))
    ys = list(range(0, max(1, H - th + 1), step_y))
    if xs[-1] != W - tw:
        xs.append(max(0, W - tw))
    if ys[-1] != H - th:
        ys.append(max(0, H - th))

    tid = 0
    for gi, y0 in enumerate(ys):
        for gj, x0 in enumerate(xs):
            x1 = min(W, x0 + tw)
            y1 = min(H, y0 + th)
            crop = image_bgr[y0:y1, x0:x1]
            if crop is None or crop.size == 0:
                continue

            # Ước lượng font theo tile
            fh_tile = estimate_font_height_tile(crop)
            small_font = (fh_tile > 0) and (fh_tile < float(FONT_HEIGHT_SMALL_PX))

            # Enhance có điều kiện (P2)
            if apply_enhance_conditional and should_enhance_tile(crop):
                crop = enhance_tile(crop)

            meta = {
                'id': int(tid),
                'orig_x': int(x0), 'orig_y': int(y0),
                'tile_w': int(x1 - x0), 'tile_h': int(y1 - y0),
                'pad_l': 0, 'pad_t': 0, 'pad_r': 0, 'pad_b': 0, 
                'step_x': int(step_x), 'step_y': int(step_y),
                'full_w': int(W), 'full_h': int(H),
                'grid_x': int(gj), 'grid_y': int(gi),
                'overlap_ratio': float(overlap_ratio),
            }
            yield crop, meta, small_font
            tid += 1
