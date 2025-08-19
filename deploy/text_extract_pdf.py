from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import os
import re, unicodedata, math
from difflib import SequenceMatcher
import json
from PyPDF2 import PdfReader, PdfWriter
import pdfplumber

from postprocessor import (_iou as iou, is_invalid_fixed_pattern, 
                           _canonicalize_device_token, 
                            _normalize_sel_series,
                            _sanitize_device_near_number)
# ---- header-driven cache dir (optional) ----
from header import (PDF_CACHE_DIR,  PDF_CENTER_MERGE_PX, 
                    PDF_PREFER_SOURCE, PDF_IOU_THRESH)



# ============== Helpers ==============
def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    # Chuẩn hoá Đ/đ -> D/d để nhất quán với postprocessor
    s = s.replace("Đ", "D").replace("đ", "d")
    return s

def _canon(s: str) -> str:
    s = _strip_accents(s).upper().strip()
    return re.sub(r"[^A-Z0-9]+", "", s)

def _sim(a: str, b: str) -> float:
    return SequenceMatcher(a=_canon(a), b=_canon(b)).ratio()

def _center(b):
    x0,y0,x1,y1=b; return ((x0+x1)/2.0, (y0+y1)/2.0)

def _area(b):
    x0,y0,x1,y1=b; return max(0,x1-x0)*max(0,y1-y0)

def _quality(text: str, conf: float, bbox) -> float:
    alnum = sum(c.isalnum() for c in (text or ""))
    return alnum + 0.35*math.sqrt(max(0.0, conf)) + 0.15*math.log(_area(bbox)+1.0)

def _normalize_final(s: str) -> str:
    return _sanitize_device_near_number(_canonicalize_device_token(_normalize_sel_series(s or ""))).strip()


def _doc_name_from_path(img_path: str) -> Optional[str]:
    """
    Lấy tên tài liệu (folder chứa ảnh trang):
    .../<doc_name>/<image_file>
    """
    parts = Path(img_path).parts
    if len(parts) >= 2:
        return parts[-2]
    return None


def _page_json_path(doc_name: str, page_id: int) -> str:
    """
    Build path: cache/pdf_cache/<doc_name>/<doc_name>_page{page_id}.pdf.json
    (Giữ nguyên quy ước cũ)
    """
    out_dir = os.path.join(PDF_CACHE_DIR, doc_name)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{doc_name}_page{page_id}.pdf.json")


# ============== Public API ==============
def try_load_pdf_text_for_image(img_path: str, img_id: str | int) -> Optional[Dict[str, Any]]:
    """
    Đọc JSON text đã tách sẵn theo quy ước:
      cache/pdf_cache/<doc>/<doc>_page<img_id>.pdf.json
    """
    doc_name = _doc_name_from_path(img_path)
    if not doc_name:
        return None
    json_path = _page_json_path(doc_name, int(img_id))
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def add_missing_from_pdf(ocr_items, pdf_texts, pdf_size, image_size, iou_thresh=None):
    """
    Hòa giải OCR↔PDF, chống trùng “lồng nhau”:
      - Nếu trùng vị trí (IoU cao, hoặc IoF cao, hoặc tâm rất gần) và cùng nội dung → giữ đúng 1 mục theo PDF_PREFER_SOURCE.
      - Nếu khác nội dung → chỉ thêm PDF khi thật sự xa.
      - Không thêm trường 'tags'.
    """
    if iou_thresh is None:
        iou_thresh = float(PDF_IOU_THRESH)

    # --- helpers ---
    def _area(b): x0,y0,x1,y1=b; return max(0,x1-x0)*max(0,y1-y0)
    def _inter(a,b):
        ax0,ay0,ax1,ay1=a; bx0,by0,bx1,by1=b
        ix0=max(ax0,bx0); iy0=max(ay0,by0); ix1=min(ax1,bx1); iy1=min(ay1,by1)
        return max(0,ix1-ix0)*max(0,iy1-iy0)
    def _iof(a,b):
        inter=_inter(a,b)
        return inter / max(1.0, min(_area(a), _area(b)))

    def _center(b): x0,y0,x1,y1=b; return (0.5*(x0+x1), 0.5*(y0+y1))
    def _h(b): return max(1.0, b[3]-b[1])
    def _canon_noaccent(s: str) -> str:
        return re.sub(r"[^A-Z0-9]+", "", _strip_accents((s or "").upper()))

    def _same_text(a: str, b: str) -> bool:
        ca = _canon_noaccent(_normalize_final(a))
        cb = _canon_noaccent(_normalize_final(b))
        return (ca == cb) or (ca in cb) or (cb in ca)

    def _quality(text: str, conf: float, bbox) -> float:
        alnum = sum(c.isalnum() for c in (text or ""))
        return alnum + 0.35*math.sqrt(max(0.0, conf)) + 0.15*math.log(_area(bbox)+1.0)

    merged = list(ocr_items)

    for pdf_obj in (pdf_texts or []):
        mapped_bbox = convert_pdf_bbox_to_image_bbox(pdf_obj["bbox"], pdf_size, image_size)
        raw = (pdf_obj.get("text") or "").strip()
        if not raw or is_invalid_fixed_pattern(raw):
            continue
        pdf_text = _normalize_final(raw)

        # --- tìm ứng viên OCR “trùng chỗ” ---
        best_j, best_score = -1, -1.0
        for j, o in enumerate(merged):
            b0, b1 = o["bbox"], mapped_bbox
            iou_ = iou(b0, b1)
            iof_ = _iof(b0, b1)
            cx0, cy0 = o.get("center", _center(b0))
            cx1, cy1 = _center(b1)
            # ngưỡng tâm linh hoạt theo chữ cao hơn trong 2 box
            center_tol = max(PDF_CENTER_MERGE_PX, 0.35*min(_h(b0), _h(b1)))
            center_hit = (abs(cx0-cx1) <= center_tol and abs(cy0-cy1) <= center_tol)

            # “điểm gần” – ưu tiên IoU, sau đó IoF, sau đó tâm
            score = max(iou_/max(1e-6,iou_thresh), iof_/0.5, 1.0 if center_hit else 0.0)
            if score >= 1.0 and score > best_score:
                best_score, best_j = score, j

        if best_j < 0:
            # hoàn toàn chưa có gì ở gần → thêm mới
            cx, cy = _center(mapped_bbox)
            merged.append({
                "text": pdf_text,
                "bbox": mapped_bbox,
                "confidence": 1.0,
                "center": [cx, cy],
                # "tags": ["from_pdf_only"],
            })
            continue

        # --- có OCR gần đó ---
        o = merged[best_j]
        if _same_text(o.get("text",""), pdf_text):
            # giữ 1 bản theo prefer
            prefer_pdf = (str(PDF_PREFER_SOURCE).lower() == "pdf")
            keep_pdf = prefer_pdf or (_quality(pdf_text, 1.0, mapped_bbox) >
                                      _quality(o.get("text",""), float(o.get("confidence",0.0)), o["bbox"]))
            if keep_pdf:
                cx, cy = _center(mapped_bbox)
                merged[best_j] = {
                    "text": pdf_text,
                    "bbox": mapped_bbox,
                    "confidence": max(1.0, float(o.get("confidence",0.0))),
                    "center": [cx, cy],
                }
        else:
            # khác nội dung → chỉ thêm nếu thật sự xa
            b0, b1 = o["bbox"], mapped_bbox
            iou_ = iou(b0, b1); iof_ = _iof(b0, b1)
            cx0, cy0 = o.get("center", _center(b0)); cx1, cy1 = _center(b1)
            center_tol = max(PDF_CENTER_MERGE_PX, 0.35*min(_h(b0), _h(b1)))
            center_far = (abs(cx0-cx1) > 1.2*center_tol or abs(cy0-cy1) > 1.2*center_tol)
            if (iou_ < 0.05) and (iof_ < 0.25) and center_far:
                merged.append({
                    "text": pdf_text,
                    "bbox": mapped_bbox,
                    "confidence": 1.0,
                    "center": [cx1, cy1],
                })

    # --- sweep dedupe nhẹ (sau merge) ---
    out = []
    for it in merged:
        keep = True
        for k, sel in enumerate(out):
            same = _same_text(it.get("text",""), sel.get("text",""))
            if not same:
                continue
            b0, b1 = it["bbox"], sel["bbox"]
            iou_ = iou(b0, b1); inter = _inter(b0, b1); mn = min(_area(b0), _area(b1))
            cx0, cy0 = it.get("center", _center(b0)); cx1, cy1 = sel.get("center", _center(b1))
            center_tol = max(PDF_CENTER_MERGE_PX, 0.35*min(_h(b0), _h(b1)))
            if (iou_ >= 0.05) or (mn > 0 and inter/mn >= 0.5) or (abs(cx0-cx1) <= center_tol and abs(cy0-cy1) <= center_tol):
                # giữ bản theo prefer
                prefer_pdf = (str(PDF_PREFER_SOURCE).lower() == "pdf")
                q_it = _quality(it.get("text",""), float(it.get("confidence",0.0)), b0)
                q_sel= _quality(sel.get("text",""), float(sel.get("confidence",0.0)), b1)
                keep_it = prefer_pdf or (q_it > q_sel)
                if keep_it:
                    out[k] = it
                keep = False
                break
        if keep:
            out.append(it)

    return out


def split_pdf_to_pages_and_extract_texts(pdf_path: str, pdf_output_dir: str = PDF_CACHE_DIR) -> None:
    """
    Cắt PDF thành từng trang (file .pdf riêng) và trích xuất words + bbox bằng pdfplumber.
    Lưu JSON cạnh file trang, tên: <doc>_page{i}.pdf.json
    Giữ nguyên cấu trúc JSON: {"pdf_texts": [...], "pdf_size": [w, h]}
    """
    pdf_name = Path(pdf_path).stem
    output_dir = os.path.join(pdf_output_dir, pdf_name)
    os.makedirs(output_dir, exist_ok=True)

    if is_pdf_scanned(pdf_path):
        print(f"[INFO] PDF '{pdf_path}' is scanned — skipping text extraction.")
        return

    reader = PdfReader(pdf_path)
    for i, page in enumerate(reader.pages, start=1):
        single_page_pdf_path = os.path.join(output_dir, f"{pdf_name}_page{i}.pdf")
        # write one-page pdf
        writer = PdfWriter()
        writer.add_page(page)
        with open(single_page_pdf_path, "wb") as f:
            writer.write(f)

        # extract words
        with pdfplumber.open(single_page_pdf_path) as pdf:
            pdf_page = pdf.pages[0]
            pdf_w, pdf_h = pdf_page.width, pdf_page.height
            words = pdf_page.extract_words() or []
            text_objs = [{
                "text": w.get("text", ""),
                "bbox": [w.get("x0", 0.0), w.get("top", 0.0), w.get("x1", 0.0), w.get("bottom", 0.0)]
            } for w in words]

        # save json beside the one-page pdf
        with open(single_page_pdf_path + ".json", "w", encoding="utf-8") as f:
            json.dump({
                "pdf_texts": text_objs,
                "pdf_size": [pdf_w, pdf_h]
            }, f, ensure_ascii=False, indent=2)


def convert_pdf_bbox_to_image_bbox(
    pdf_bbox: List[float] | Tuple[float, float, float, float],
    pdf_size: Tuple[float, float] | List[float],
    image_size: Tuple[int, int] | List[int]
) -> List[float]:
    """
    Chuyển bbox từ toạ độ PDF (theo pdfplumber: x0, top, x1, bottom; gốc trên-trái)
    sang toạ độ ảnh (gốc trên-trái). Chỉ scale theo tỉ lệ w,h (không flip trục).
    """
    x0, top, x1, bottom = pdf_bbox  # pdfplumber: top < bottom
    pdf_w, pdf_h = float(pdf_size[0]), float(pdf_size[1])
    img_w, img_h = float(image_size[0]), float(image_size[1])

    scale_x = img_w / pdf_w if pdf_w else 1.0
    scale_y = img_h / pdf_h if pdf_h else 1.0

    new_x0 = x0 * scale_x
    new_x1 = x1 * scale_x
    new_y0 = top * scale_y
    new_y1 = bottom * scale_y

    return [new_x0, new_y0, new_x1, new_y1]


def is_pdf_scanned(pdf_path: str) -> bool:
    """
    True nếu PDF hầu như không có text vector.
    Ưu tiên dùng PyMuPDF (fitz); nếu không có, fallback bằng pdfplumber.
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        for page in doc:
            if len(page.get_text("text").strip()) > 10:
                return False
        return True
    except Exception:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for p in pdf.pages:
                    txt = p.extract_text() or ""
                    if len((txt or "").strip()) > 10:
                        return False
                return True
        except Exception:
            return True
