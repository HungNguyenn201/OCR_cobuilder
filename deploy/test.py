from __future__ import annotations
import os
import json
from pathlib import Path
import cv2
from typing import Tuple, List, Dict, Any

from header import OVERLAP_RATIO, PIPELINE_DEBUG, PDF_VALIDATE_DEFAULT
from pdftoimage import convert_pdf_to_images
from text_extract_pdf import split_pdf_to_pages_and_extract_texts
from run_batch import run_full_ocr_pipeline_from_cache

from preprocessor import split_image_to_tiles, get_tile_size_adaptive


# ---------- small drawing utils ----------
def _draw_rect(img, x0, y0, x1, y1, color, thickness=2):
    cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), color, thickness, lineType=cv2.LINE_AA)

def _draw_label(img, x, y, text, color=(0, 0, 0), bg=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thick = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    pad = 2
    cv2.rectangle(img, (int(x), int(y) - th - 2 * pad), (int(x) + tw + 2 * pad, int(y)), bg, -1)
    cv2.putText(img, text, (int(x) + pad, int(y) - pad), font, scale, color, thick, cv2.LINE_AA)


def visualize_pipeline_debug(
    img_path: str,
    ocr_items: List[Dict[str, Any]],
    save_path: str,
    tile_size: Tuple[int, int] | None = None,
    overlap_ratio: float = OVERLAP_RATIO,
    draw_tiles: bool = True,
    draw_ocr: bool = True,
    show_text: bool = True
) -> Dict[str, Any]:
    """
    Vẽ:
      - Lưới tile (cam) + "core" không overlap (xanh lá)
      - Bbox OCR: xanh lá = OCR engine; xanh dương = from_pdf_only; đỏ = cut_boundary
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    overlay = img.copy()
    H, W = img.shape[:2]

    # ---- Tiles ----
    tiles_info: List[Dict[str, Any]] = []
    if draw_tiles:
        if tile_size is None:
            tile_size = get_tile_size_adaptive(img.shape)

        # API mới: trả về (tiles, metas)
        tiles, metas = split_image_to_tiles(
            img,
            tile_size=tile_size,
            overlap_ratio=overlap_ratio,
            apply_enhance=False,         # chỉ vẽ grid, không cần enhance
            detect_textual=True,
            debug_print=PIPELINE_DEBUG
        )

        color_tile = (0, 165, 255)   # cam
        color_core = (0, 255, 0)     # xanh lá

        # stride fallback nếu rỗng
        ref_sw = int(tile_size[0] * (1.0 - overlap_ratio))
        ref_sh = int(tile_size[1] * (1.0 - overlap_ratio))

        for idx, m in enumerate(metas):
            x0 = int(m["orig_x"]); y0 = int(m["orig_y"])
            tw = int(m["tile_w"]);  th = int(m["tile_h"])
            x1 = x0 + tw;           y1 = y0 + th
            sw = int(m.get("step_x", ref_sw))
            sh = int(m.get("step_y", ref_sh))

            # viền tile
            _draw_rect(overlay, x0, y0, x1, y1, color_tile, 2)

            # core = vùng không overlap (tile - margins dựa trên stride)
            mw = max(2, int(round((tw - sw) / 2.0)))
            mh = max(2, int(round((th - sh) / 2.0)))
            cx0, cy0 = x0 + mw, y0 + mh
            cx1, cy1 = x1 - mw, y1 - mh
            cx0 = max(cx0, x0); cy0 = max(cy0, y0)
            cx1 = min(cx1, x1); cy1 = min(cy1, y1)
            _draw_rect(overlay, cx0, cy0, cx1, cy1, color_core, 2)

            _draw_label(overlay, x0 + 3, y0 + 14, f"id={idx} {x0},{y0}-{x1},{y1}")
            tiles_info.append({
                "id": idx,
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "tile_w": tw, "tile_h": th,
                "stride_w": sw, "stride_h": sh
            })

        # legend
        _draw_rect(overlay, 10, 10, 250, 90, (255, 255, 255), -1)
        _draw_rect(overlay, 10, 10, 250, 90, (0, 0, 0), 1)
        _draw_rect(overlay, 20, 25, 60, 45, (0, 165, 255), 2); cv2.putText(overlay, "Tile border", (70, 43), 0, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        _draw_rect(overlay, 20, 55, 60, 75, (0, 255, 0), 2);   cv2.putText(overlay, "Core (non-overlap)", (70, 73), 0, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # ---- OCR boxes ----
    if draw_ocr and ocr_items:
        for it in ocr_items:
            x0, y0, x1, y1 = map(int, it["bbox"])
            tags = set(it.get("tags", []))
            cutb = ("cut_boundary" in tags) or bool(it.get("cut_boundary", False))

            # màu theo nguồn (nếu có tag PDF)
            if "from_pdf_only" in tags or "from_pdf_line_only" in tags:
                color = (255, 0, 0)     # BLUE (BGR)
            else:
                color = (0, 200, 0)     # GREEN (BGR)
            if cutb:
                color = (0, 0, 255)     # RED (BGR)

            _draw_rect(overlay, x0, y0, x1, y1, color, 2)
            if show_text:
                txt = it.get("text", "")
                if "from_pdf_only" in tags:      txt = "[PDF] " + txt
                if "from_pdf_line_only" in tags: txt = "[PDF-L] " + txt
                if cutb:                          txt = "[CUT] " + txt
                _draw_label(overlay, x0 + 1, max(y0 - 2, 12), txt, color=(0, 0, 0), bg=(255, 255, 255))

        # legend OCR
        basey = 100
        _draw_rect(overlay, 10, basey, 300, basey + 90, (255, 255, 255), -1)
        _draw_rect(overlay, 10, basey, 300, basey + 90, (0, 0, 0), 1)
        _draw_rect(overlay, 20, basey + 15, 60, basey + 35, (0, 200, 0), 2);  cv2.putText(overlay, "OCR (engine)", (70, basey + 33), 0, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        _draw_rect(overlay, 20, basey + 45, 60, basey + 65, (255, 0, 0), 2);  cv2.putText(overlay, "PDF merge", (70, basey + 63), 0, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        _draw_rect(overlay, 20, basey + 75, 60, basey + 95, (0, 0, 255), 2);  cv2.putText(overlay, "Cut boundary", (70, basey + 93), 0, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # ---- save ----
    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(save_path, overlay)

    return {
        "image_size": (W, H),
        "tile_size": tile_size,
        "overlap_ratio": overlap_ratio,
        "num_tiles": len(tiles_info),
        "tiles": tiles_info,
        "out": save_path
    }


def visualize_final_ocr(img_path: str, ocr_data, save_path: str):
    """Vẽ kết quả OCR cuối cùng lên ảnh."""
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    for item in ocr_data:
        x1, y1, x2, y2 = map(int, item["bbox"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            item["text"],
            (x1, max(y1 - 3, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 200, 0),
            1
        )

    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(save_path, image)


def test_full_pipeline(pdf_path: str):
    # Step 1: Convert PDF -> images + cache.json
    convert_result = convert_pdf_to_images(pdf_path)
    cache_path = convert_result["cache_path"]

    # Step 2: Extract PDF texts (cache). Hàm tự bỏ qua PDF scan.
    split_pdf_to_pages_and_extract_texts(pdf_path)

    # Step 3: Run OCR pipeline (batch). Dùng header để bật/tắt PDF merge.
    results = run_full_ocr_pipeline_from_cache(
        cache_path,
        use_pdf_validate=PDF_VALIDATE_DEFAULT
    )

    # Step 4: Vẽ debug (grid + OCR) cho từng trang
    for item in results:
        img_path = item["path"]
        img_id = int(item["id"])
        doc_name = os.path.basename(os.path.dirname(img_path))

        vis_dbg = f"outputs/vis_overlay/{doc_name}_page{img_id}_DEBUG.png"
        _ = visualize_pipeline_debug(
            img_path=img_path,
            ocr_items=item["ocr"],      # list sau khi merge/dedupe + PDF (nếu bật)
            save_path=vis_dbg,
            overlap_ratio=OVERLAP_RATIO,
            tile_size=None,             # None -> get_tile_size_adaptive
            draw_tiles=True,
            draw_ocr=True,
            show_text=True
        )

        # (optional) Vẽ lớp cuối chỉ OCR
        vis_final = f"outputs/vis_overlay/{doc_name}_page{img_id}_FINAL.png"
        visualize_final_ocr(img_path, item["ocr"], vis_final)

    # Step 5: Save JSON kết quả
    out_path = os.path.join("outputs", f"{Path(pdf_path).stem}_ocr_final.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[✓] Final result saved to {out_path}")


if __name__ == "__main__":
    pdf_path = "data/pdf_file/pdf/1.CSE.pdf"
    test_full_pipeline(pdf_path)
