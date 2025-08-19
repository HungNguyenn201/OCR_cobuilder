from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import os, time
import cv2, numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from postprocessor import (
    _canonicalize_device_token,
    _normalize_sel_series,
    _sanitize_device_near_number,
    advanced_nms_text,
    merge_split_texts,
    remove_duplicate_substrings,
    is_invalid_fixed_pattern,
    is_cut_by_tile,
    detect_voltage_levels_from_texts,
)

# ================== header-driven config ==================
from header import (
    PIPELINE_DEBUG,
    # OCR engine
    OCR_USE_GPU, LANG_CODE, REC_BATCH_NUM, OCR_USE_ANGLE_CLS,
    # tiling / worker (cũ)
    OVERLAP_RATIO, TILE_DOWNSCALE, MAX_WORKERS,
    # multi-scale policy
    OCR_MULTI_SCALE_SMALL, OCR_MULTI_SCALE_LARGE, FONT_HEIGHT_SMALL_PX,
    # NMS thresholds
    IOU_NMS, CENTER_THRESH_PX,
    OUTPUT_INCLUDE_TAGS,
    # ===== NEW toggles / params =====
    USE_SHARED_ENGINE_POOL, HARD_MAX_WORKERS,
    DYNAMIC_OVERLAP, OVERLAP_MIN, OVERLAP_MAX,
    PER_TILE_MULTISCALE, ROTATION_PASS_90,
    HIGH_RECALL_TILES,              
    DENSE_PAGE_MODE, IOU_NMS_DENSE,
    TILE_QUEUE_MAX,
)
from preprocessor import (
    get_tile_size_adaptive,
    iter_tiles,               # streaming tiles 
    estimate_font_height,     # ước lượng font toàn ảnh
)

# ================== small utils ==================
def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

# ================== OCR Engine (PP-OCRv4) ==================
class _PaddleEngine:
    def __init__(self,
                 use_gpu: bool = OCR_USE_GPU,
                 lang: str = None,
                 rec_batch_num: int = REC_BATCH_NUM,
                 use_angle_cls: bool = OCR_USE_ANGLE_CLS):
        from paddleocr import PaddleOCR
        lang = (lang if lang else LANG_CODE)
        self.use_angle_cls = bool(use_angle_cls)
        self.ocr = PaddleOCR(
            use_angle_cls=self.use_angle_cls,
            lang=lang,
            det=True, rec=True,
            ocr_version="PP-OCRv4",
            show_log=False,
            use_gpu=use_gpu,
            rec_batch_num=rec_batch_num,
        )

    @staticmethod
    def _poly_to_xyxy(poly: List[List[float]]) -> Tuple[int, int, int, int]:
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

    def predict(self, bgr: np.ndarray) -> List[Dict[str, Any]]:
        if bgr is None or bgr.size == 0:
            return []
        res = self.ocr.ocr(bgr, cls=self.use_angle_cls) or []
        items: List[Dict[str, Any]] = []
        for line in res or []:
            if not line:
                continue
            for det in line:
                poly, (txt, conf) = det
                x0, y0, x1, y1 = self._poly_to_xyxy(poly)
                items.append({
                    "text": txt or "",
                    "bbox": [x0, y0, x1, y1],
                    "center": [int((x0+x1)/2), int((y0+y1)/2)],
                    "confidence": float(conf or 0.0),
                })
        return items

# ================== EnginePool (thread-safe, hard-capped) ==================
class EnginePool:
    def __init__(self, size: int):
        if size <= 0:
            raise ValueError("EnginePool size must be > 0")
        from queue import Queue
        size = max(1, min(size, HARD_MAX_WORKERS))  # hard cap
        self.num_workers = size
        self._q = Queue(maxsize=size)
        for _ in range(size):
            self._q.put(_PaddleEngine(lang= LANG_CODE,
                                      use_gpu=OCR_USE_GPU,
                                      rec_batch_num=REC_BATCH_NUM,
                                      use_angle_cls=OCR_USE_ANGLE_CLS))

    def acquire(self) -> _PaddleEngine:
        return self._q.get(block=True)

    def release(self, eng: _PaddleEngine) -> None:
        try:
            self._q.put(eng, block=False)
        except Exception:
            pass

# ------------------ Shared engine pool (singleton) ------------------
_SHARED_POOL: Optional[EnginePool] = None

def get_shared_engine_pool() -> EnginePool:
    global _SHARED_POOL
    if _SHARED_POOL is None:
        # default sử dụng HARD_MAX_WORKERS
        _SHARED_POOL = EnginePool(max(1, HARD_MAX_WORKERS))
    return _SHARED_POOL

# ================== Map tile → full + seam tag ==================
def _map_tile_preds_to_full(items: List[Dict[str, Any]], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Map bbox/center từ tile về toạ độ ảnh gốc và gắn 'cut_boundary'
    dựa trên stride thực (step_x/step_y) và tile_w/tile_h.
    """
    orig_x = int(meta["orig_x"]); orig_y = int(meta["orig_y"])
    pad_l  = int(meta.get("pad_l", meta.get("pad", 0)))
    pad_t  = int(meta.get("pad_t", meta.get("pad", 0)))

    tile_w = int(meta["tile_w"]); tile_h = int(meta["tile_h"])
    step_x = int(meta["step_x"]); step_y = int(meta["step_y"])
    full_w = int(meta["full_w"]); full_h = int(meta["full_h"])

    mapped: List[Dict[str, Any]] = []
    for it in items:
        x0, y0, x1, y1 = it["bbox"]
        # bỏ padding, cộng origin
        bx0 = int(x0 - pad_l + orig_x)
        by0 = int(y0 - pad_t + orig_y)
        bx1 = int(x1 - pad_l + orig_x)
        by1 = int(y1 - pad_t + orig_y)

        # clamp
        bx0 = int(_clamp(bx0, 0, full_w))
        by0 = int(_clamp(by0, 0, full_h))
        bx1 = int(_clamp(bx1, 0, full_w))
        by1 = int(_clamp(by1, 0, full_h))
        cx = int((bx0 + bx1) / 2); cy = int((by0 + by1) / 2)

        tags = set(it.get("tags", []))
        # seam check dùng stride thực & kích thước crop
        if is_cut_by_tile((bx0,by0,bx1,by1),
                          tile_size=(tile_w, tile_h),
                          img_size=(full_w, full_h),
                          stride=(step_x, step_y)):
            tags.add("cut_boundary")

        mapped.append({
            "text": str(it.get("text","")),
            "bbox": [bx0,by0,bx1,by1],
            "center": [cx,cy],
            "confidence": float(it.get("confidence",0.0)),
            **({"tags": list(tags)} if tags else {}),
        })
    return mapped

# ================== Main Pipeline ==================
class OCRPipeline:
    def __init__(self,
                 image_path: str,
                 tile_size: Optional[Tuple[int, int]] = None,
                 overlap: Optional[float] = None,
                 multi_scale: Tuple[float, ...] | None = None,
                 tile_downscale: float = TILE_DOWNSCALE,
                 num_workers: Optional[int] = MAX_WORKERS,
                 engine_pool: Optional[EnginePool] = None):
        self.image_path = image_path
        self.overlap = float(OVERLAP_RATIO if overlap is None else overlap)
        self.tile_size = tile_size
        self.tile_downscale = float(tile_downscale)
        self._user_workers = int(num_workers) if (num_workers and num_workers > 0) else None
        self._scales_arg = tuple(s for s in (multi_scale or ()) if s > 0)
        # chọn engine pool
        if engine_pool is not None:
            self.pool = engine_pool
        elif USE_SHARED_ENGINE_POOL:
            self.pool = get_shared_engine_pool()
        else:
            planned = self._user_workers or (os.cpu_count() or 2)
            self.pool = EnginePool(max(1, min(planned, HARD_MAX_WORKERS)))

    @staticmethod
    def _filter_trivial(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for it in items:
            t = (it.get("text") or "").strip()
            if not t:
                continue
            # ký tự đơn lẻ với độ tin quá thấp -> Loại
            if len(t) == 1 and float(it.get("confidence", 0.0)) < 0.10:
                continue
            out.append(it)
        return out

    def _run_on_image(self, bgr: np.ndarray) -> List[Dict[str, Any]]:
        H, W = bgr.shape[:2]
        fH_global = max(12, int(estimate_font_height(bgr)))

        # 1) Tile size & overlap
        if self.tile_size is None:
            self.tile_size = get_tile_size_adaptive(bgr.shape)

        if self.tile_downscale < 1.0 and self.tile_size is not None:
            tw, th = self.tile_size
            self.tile_size = (max(512, int(tw*self.tile_downscale)),
                              max(512, int(th*self.tile_downscale)))

        overlap = self.overlap
        if DYNAMIC_OVERLAP:
            # chữ to/đậm → giảm overlap để tăng tốc
            if fH_global >= int(FONT_HEIGHT_SMALL_PX * 1.2):
                overlap = OVERLAP_MIN
            else:
                overlap = OVERLAP_MAX

        # 2) Streaming tiles (bộ nhớ thấp)
        tile_gen = iter_tiles(
            bgr,
            tile_size=self.tile_size,
            overlap_ratio=overlap,
            apply_enhance_conditional=True,   # điều kiện trong preprocessor
        )

        # 3) per-tile OCR (multi-scale theo tile) + local NMS + map
        def run_one(tile_img: np.ndarray, meta: Dict[str, Any], small_font: bool) -> List[Dict[str, Any]]:
            eng = self.pool.acquire()
            try:
                # chọn scales: per-tile
                if self._scales_arg:
                    scales = self._scales_arg
                else:
                    scales = (OCR_MULTI_SCALE_SMALL if small_font else OCR_MULTI_SCALE_LARGE) if PER_TILE_MULTISCALE else OCR_MULTI_SCALE_LARGE
                # high-recall: thêm 1.5 nếu chưa có và tile nhỏ/khó
                if HIGH_RECALL_TILES and small_font and (1.5 not in scales):
                    scales = tuple(sorted(set(list(scales) + [1.5])))

                local: List[Dict[str, Any]] = []
                for s in scales:
                    if s == 1.0:
                        img_s = tile_img
                    else:
                        img_s = cv2.resize(tile_img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
                    preds = eng.predict(img_s)
                    if preds and s != 1.0:
                        inv = 1.0 / s
                        for it in preds:
                            it["bbox"] = [
                                int(it["bbox"][0] * inv), int(it["bbox"][1] * inv),
                                int(it["bbox"][2] * inv), int(it["bbox"][3] * inv)
                            ]
                            if "center" in it and it["center"] is not None:
                                it["center"] = [int(it["center"][0] * inv), int(it["center"][1] * inv)]
                    local.extend(preds or [])

                local = self._filter_trivial(local)

                # per-tile NMS nhẹ (không seam), hint theo fH_global
                nms_local = advanced_nms_text(
                    local,
                    iou_thresh=IOU_NMS,
                    center_thresh=CENTER_THRESH_PX,
                    tile_size=None, img_size=None,
                    avg_h_hint=fH_global
                )
                # Map về ảnh gốc + gắn seam đúng stride
                return _map_tile_preds_to_full(nms_local, meta)
            finally:
                self.pool.release(eng)

        all_items: List[Dict[str, Any]] = []
        # xử lý theo lô nhỏ để hạn chế RAM
        with ThreadPoolExecutor(max_workers=self.pool.num_workers) as ex:
            batch_tasks = []
            for tile, meta, small_font in tile_gen:
                batch_tasks.append(ex.submit(run_one, tile, meta, small_font))
                # Khi đủ batch → đợi và gom kết quả
                if len(batch_tasks) >= max(1, TILE_QUEUE_MAX):
                    for fu in as_completed(batch_tasks):
                        all_items.extend(fu.result() or [])
                    batch_tasks.clear()
            # flush các task còn lại
            for fu in as_completed(batch_tasks):
                all_items.extend(fu.result() or [])

        return all_items

    def run(self) -> Dict[str, Any]:
        t0 = time.perf_counter()
        bgr = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(self.image_path)

        # ---- Pass 0° ----
        base_items = self._run_on_image(bgr)

        # ---- Pass 90° (nếu bật) ----
        if ROTATION_PASS_90:
            rot = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
            rot_items = self._run_on_image(rot)
            # map bbox từ ảnh xoay về hệ ảnh gốc
            H, W = bgr.shape[:2]
            mapped_rot: List[Dict[str, Any]] = []
            for it in rot_items:
                x0, y0, x1, y1 = it["bbox"]
                # Inverse mapping for 90° CW:
                # original x0 = W - y1, x1 = W - y0; y0 = x0, y1 = x1
                nx0 = W - y1
                nx1 = W - y0
                ny0 = x0
                ny1 = x1
                nx0, ny0 = int(_clamp(nx0, 0, W)), int(_clamp(ny0, 0, H))
                nx1, ny1 = int(_clamp(nx1, 0, W)), int(_clamp(ny1, 0, H))
                b = [min(nx0, nx1), min(ny0, ny1), max(nx0, nx1), max(ny0, ny1)]
                mapped_rot.append({
                    **it,
                    "bbox": b,
                    "center": [int((b[0]+b[2])//2), int((b[1]+b[3])//2)],
                })
            base_items.extend(mapped_rot)

        t1 = time.perf_counter()

        # ---- Global post: NMS (seam-aware), merge, dedupe ----
        iou_glob = IOU_NMS_DENSE if DENSE_PAGE_MODE else IOU_NMS
        glob = advanced_nms_text(
            base_items,
            iou_thresh=iou_glob,
            center_thresh=CENTER_THRESH_PX,
            tile_size=None, img_size=None,
        )
        merged = merge_split_texts(glob)
        deduped = remove_duplicate_substrings(merged)

        # ---- Filter fixed-pattern & format output ----
        final_items: List[Dict[str, Any]] = []
        for it in deduped:
            raw_text = it.get("text", "")
            txt = _sanitize_device_near_number(
                    _canonicalize_device_token(_normalize_sel_series(raw_text))
                )
            if is_invalid_fixed_pattern(txt):
                continue
            b = it["bbox"]; c = it["center"]
            out_obj = {
                "text": str(txt),
                "bbox": [int(b[0]), int(b[1]), int(b[2]), int(b[3])],
                "center": [int(c[0]), int(c[1])],
                "confidence": float(it.get("confidence", 0.0)),
            }
            if OUTPUT_INCLUDE_TAGS and it.get("tags"):
                out_obj["tags"] = it.get("tags")
            final_items.append(out_obj)

        # ---- Voltage detection ----
        voltage = detect_voltage_levels_from_texts([it["text"] for it in final_items])

        t2 = time.perf_counter()
        if PIPELINE_DEBUG:
            print(f"[PIPE] workers={self.pool.num_workers} "
                  f"time: ocr={(t1-t0):.2f}s post={(t2-t1):.2f}s total={(t2-t0):.2f}s "
                  f"items={len(final_items)}")

        return {"ocr": final_items, "voltage_level": voltage}
