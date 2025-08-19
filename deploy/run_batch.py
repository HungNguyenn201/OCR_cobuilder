from __future__ import annotations
from typing import List, Dict, Any, Tuple
import json
import cv2
from pipeline import OCRPipeline, get_shared_engine_pool
from text_extract_pdf import add_missing_from_pdf, try_load_pdf_text_for_image
from header import (
    PIPELINE_DEBUG,
    PDF_VALIDATE_DEFAULT,
    PDF_IOU_THRESH,
)

def _read_image_size(path: str) -> Tuple[int, int]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return (img.shape[1], img.shape[0])  # (w, h)

def run_batch_ocr_from_cache(
    cache_json_path: str,
    use_pdf_validate: bool | None = None,
    iou_thresh: float | None = None
) -> List[Dict[str, Any]]:
    """
    Chạy OCR theo danh sách ảnh trong cache.json:
      1) OCRPipeline.run()
      2) Nếu có JSON words từ PDF -> append-only vào kết quả (không re-postprocess)
    Giữ nguyên cấu trúc output: [{"id","path","voltage_level","ocr"}, ...]
    """

    if use_pdf_validate is None:
        use_pdf_validate = bool(PDF_VALIDATE_DEFAULT)
    if iou_thresh is None:
        iou_thresh = float(PDF_IOU_THRESH)

    with open(cache_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_entries = data.get("images", []) or []
    all_results: List[Dict[str, Any]] = []

    # Khởi tạo shared engine pool MỘT LẦN cho toàn bộ batch (giảm RAM & thời gian warmup)
    pool = get_shared_engine_pool()

    for entry in img_entries:
        img_id = entry["id"]
        img_path = entry["path"]
        if PIPELINE_DEBUG:
            print(f"[INFO] Processing img id={img_id}, path={img_path}")

        # 1) Run OCR pipeline (dùng pool chia sẻ)
        pipeline = OCRPipeline(image_path=img_path, engine_pool=pool)
        result = pipeline.run()

        # 2) (Tuỳ chọn) Append-only với PDF words
        if use_pdf_validate:
            pdf_text_data = try_load_pdf_text_for_image(img_path, img_id)
            if pdf_text_data and pdf_text_data.get("pdf_texts"):
                image_size = _read_image_size(img_path)
                result["ocr"] = add_missing_from_pdf(
                    result["ocr"],
                    pdf_text_data["pdf_texts"],
                    pdf_text_data["pdf_size"],
                    image_size,
                    iou_thresh=iou_thresh
                )
            elif PIPELINE_DEBUG:
                print(f"[INFO] Skipping validation: no PDF text found for id={img_id}")

        all_results.append({
            "id": img_id,
            "path": img_path,
            "voltage_level": result.get("voltage_level", []),
            "ocr": result.get("ocr", []),
        })

    return all_results

def run_full_ocr_pipeline_from_cache(
    cache_path: str,
    use_pdf_validate: bool | None = None,
    iou_thresh: float | None = None
) -> List[Dict[str, Any]]:
    return run_batch_ocr_from_cache(
        cache_path,
        use_pdf_validate=use_pdf_validate,
        iou_thresh=iou_thresh
    )
