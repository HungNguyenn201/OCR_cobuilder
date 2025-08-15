import os
import json
from pdf2image import convert_from_path
from pathlib import Path


POPPLER_PATH = r"C:\Program Files\poppler-24.08.0\Library\bin"

def convert_multiple_pdfs_to_images(pdf_paths: list[str], base_output_dir: str = "cache/images") -> list[str]:
    """
    Convert multiple PDFs to image folders.
    Returns a list of cache.json paths for all converted PDFs.
    """
    cache_paths = []
    for pdf_path in pdf_paths:
        result = convert_pdf_to_images(pdf_path, base_output_dir=base_output_dir)
        cache_paths.append(result["cache_path"])
    return cache_paths


def convert_pdf_to_images(pdf_path: str, base_output_dir: str = "cache/images") -> dict:
    """
    Convert a single PDF to images, saved under a subfolder matching the PDF name.
    Return a JSON dict of image ids and paths, and also save it to cache.json.
    """
    # Get PDF name without extension
    pdf_name = Path(pdf_path).stem
    output_folder = os.path.join(base_output_dir, pdf_name)
    os.makedirs(output_folder, exist_ok=True)

    images = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)
    result = {"images": []}

    for i, img in enumerate(images):
        img_id = i + 1
        filename = f"page{img_id}.png"
        output_path = os.path.join(output_folder, filename)
        img.save(output_path, "PNG")
        result["images"].append({
            "id": img_id,
            "path": output_path.replace("\\", "/")  # For cross-platform path consistency
        })

    # Save cache.json
    cache_path = os.path.join(output_folder, "cache.json")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    result["cache_path"] = cache_path
    return result


