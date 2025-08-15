from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import os, time
import cv2, numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from postprocessor import (_canonicalize_device_token, 
                           _normalize_sel_series, 
                           _sanitize_device_near_number)
# ================== header-driven config ==================
from header import (
    PIPELINE_DEBUG,
    # OCR engine
    OCR_USE_GPU, LANG_CODE, REC_BATCH_NUM, OCR_USE_ANGLE_CLS,
    # (optional) angle classifier toggle; default True nếu không có
    # tiling / worker
    OVERLAP_RATIO, TILE_DOWNSCALE, MAX_WORKERS,
    # multi-scale policy
    OCR_MULTI_SCALE_SMALL, OCR_MULTI_SCALE_LARGE, FONT_HEIGHT_SMALL_PX,
    # NMS thresholds
    IOU_NMS, CENTER_THRESH_PX,
    OUTPUT_INCLUDE_TAGS
)

from preprocessor import (
    get_tile_size_adaptive,
    split_image_to_tiles,
    estimate_font_height,
)

from postprocessor import (
    advanced_nms_text,
    merge_split_texts,
    remove_duplicate_substrings,
    is_invalid_fixed_pattern,
    is_cut_by_tile,
    detect_voltage_levels_from_texts,  
)

# ================== small utils ==================

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

# ================== OCR Engine (PPOCRv4) ==================
class _PaddleEngine:
    def __init__(self,
                 use_gpu: bool = OCR_USE_GPU,
                 lang: str = LANG_CODE,
                 rec_batch_num: int = REC_BATCH_NUM,
                 use_angle_cls: bool = OCR_USE_ANGLE_CLS):
        from paddleocr import PaddleOCR
        
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

# ================== EnginePool (thread-safe) ==================
class EnginePool:
    def __init__(self, size: int):
        if size <= 0:
            raise ValueError("EnginePool size must be > 0")
        from queue import Queue
        self._q = Queue(maxsize=size)
        for _ in range(size):
            self._q.put(_PaddleEngine())

    def acquire(self) -> _PaddleEngine:
        return self._q.get(block=True)

    def release(self, eng: _PaddleEngine) -> None:
        try:
            self._q.put(eng, block=False)
        except Exception:
            pass

# ================== Map tile → full + seam tag ==================
def _map_tile_preds_to_full(items: List[Dict[str, Any]], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Map bbox/center từ PADDING tile về toạ độ ảnh gốc và gắn 'cut_boundary'
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

# ================== Pipeline ==================
class OCRPipeline:
    def __init__(self,
                 image_path: str,
                 tile_size: Optional[Tuple[int, int]] = None,
                 overlap: Optional[float] = None,
                 multi_scale: Tuple[float, ...] | None = None,
                 tile_downscale: float = TILE_DOWNSCALE,
                 num_workers: Optional[int] = MAX_WORKERS):
        self.image_path = image_path
        self.overlap = float(OVERLAP_RATIO if overlap is None else overlap)
        self.tile_size = tile_size
        self.tile_downscale = float(tile_downscale)
        self._user_workers = int(num_workers) if (num_workers and num_workers > 0) else None
        self._scales_arg = tuple(s for s in (multi_scale or ()) if s > 0)

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

    def run(self) -> Dict[str, Any]:
        t0 = time.perf_counter()
        bgr = cv2.imread(self.image_path)
        if bgr is None:
            raise FileNotFoundError(self.image_path)
        H, W = bgr.shape[:2]
        fH_global = max(12, int(estimate_font_height(bgr)))

        # 1) Tile & chọn scale
        if self.tile_size is None:
            self.tile_size = get_tile_size_adaptive(bgr.shape)

        if self.tile_downscale < 1.0:
            tw, th = self.tile_size
            self.tile_size = (max(512, int(tw*self.tile_downscale)),
                              max(512, int(th*self.tile_downscale)))

        if self._scales_arg:
            scales = self._scales_arg
        else:
            # font nhỏ → dùng cấu hình multi-scale dày hơn
            scales = OCR_MULTI_SCALE_SMALL if fH_global < FONT_HEIGHT_SMALL_PX else OCR_MULTI_SCALE_LARGE

        tiles, metas = split_image_to_tiles(
            bgr,
            tile_size=self.tile_size,
            overlap_ratio=self.overlap,
            apply_enhance=True,       
            detect_textual=True,        # lọc tile rỗng
            debug_print=PIPELINE_DEBUG  
        )
        if not tiles:
            return {"ocr": [], "voltage_level": []}

        # 2) Worker pool
        cpu = os.cpu_count() or 4
        planned = self._user_workers or cpu
        workers = min(planned, len(tiles))
        pool = EnginePool(max(1, workers))

        # 3) OCR từng tile (multi-scale) + local NMS + map + seam-tag
        def process_one(tile_img: np.ndarray, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
            eng = pool.acquire()
            try:
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

                # Per-tile NMS (không seam)
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
                pool.release(eng)

        t1 = time.perf_counter()
        mapped_chunks: List[List[Dict[str, Any]]] = [None] * len(tiles)  # type: ignore

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(process_one, tile, meta): i for i, (tile, meta) in enumerate(zip(tiles, metas))}
                for fu in as_completed(futs):
                    mapped_chunks[futs[fu]] = fu.result()
        else:
            for i, (tile, meta) in enumerate(zip(tiles, metas)):
                mapped_chunks[i] = process_one(tile, meta)

        all_items: List[Dict[str, Any]] = [it for chunk in mapped_chunks for it in chunk]
        t2 = time.perf_counter()
        if PIPELINE_DEBUG:
            print(f"[PIPE] workers={workers} tiles={len(tiles)} scales={tuple(scales)} "
                  f"time: tiling={(t1-t0):.2f}s ocr={(t2-t1):.2f}s")

        # 4) Global NMS (seam-aware nhờ tag) + merge + dedupe
        glob = advanced_nms_text(
            all_items,
            iou_thresh=IOU_NMS,
            center_thresh=CENTER_THRESH_PX,
            tile_size=None, img_size=None,  
            avg_h_hint=fH_global
        )
        merged = merge_split_texts(glob)
        deduped = remove_duplicate_substrings(merged)

        # 5) Filter fixed-pattern & format output
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

        t3 = time.perf_counter()
        if PIPELINE_DEBUG:
            print(f"[PIPE] after_filter={len(final_items)} | post={(t3-t2):.2f}s total={(t3-t0):.2f}s")

        # 6) Voltage detection (dùng detector chung từ postprocessor)
        voltage = detect_voltage_levels_from_texts([it["text"] for it in final_items])
        return {"ocr": final_items, 
                "voltage_level": voltage}
