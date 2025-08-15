from __future__ import annotations
from typing import Tuple
import re

# ---------- Preprocessor defaults ----------
PREFER_TILE: Tuple[int, int] = (1024, 1024)
MAX_TILE_LIMIT: Tuple[int, int] = (1536, 1536)
OVERLAP_RATIO: float = 0.25
MIN_PAD: int = 24
MAX_PAD: int = 192

# ---------- Postprocessor thresholds ----------
# NMS
IOU_NMS: float = 0.30
CENTER_THRESH_PX: int = 24       # giới hạn trên cho center-thresh
GRID_MULTIPLIER: float = 3.5         # grid_px ≈ GRID_MULTIPLIER * avg_h

# merge lines
LINE_VOV_THRESH: float = 0.55
SMALL_GAP_FACTOR: float = 0.35
CHAR_GAP_FACTOR: float = 0.50

# dedupe cuối
IOU_DEDUP: float = 0.40
SIM_DEDUP: float = 0.88
VOV_DUP_THRESH: float = 0.60
SIM_SHORT: float = 0.96               # text ngắn => cần giống hơn
SIM_LONG: float  = 0.85

# validate text
CB_LIKE_RE = re.compile(r"^(?:C|G)(?:B|8|E)$", re.I)
NUM_SHORT_RE = re.compile(r"^\d{1,2}$")
DEV_HEAD_RE = re.compile(r"^(BB|C|TC)\d{1,2}$", re.I)  # thiết bị đầu dòng
ALNUM = re.compile(r"[A-Za-z0-9]")
DEVICE_VOLTAGE_PAT  = re.compile(r"^(BB|C|TC)([156234897])\d{1,2}$", re.I)
HEADER_VOLTAGE_PAT  = re.compile(r"(phia|so\s*do).*?(\d{2,3})kV", re.IGNORECASE)
DS_ES_FULL_RE = re.compile(r"^(?:DS|ES)(?:[1-9]{1,2})?$", re.I)  # fullmatch, không lệ thuộc header

# 2 mẫu: [DS9] [4]  hoặc  [4] [DS9]
DS_ES_WITH_NEAR_NUM_R = re.compile(r"^((?:DS|ES)[1-9]{1,2}|(?:DS|ES))\s+\d{1,2}$", re.I)
DS_ES_WITH_NEAR_NUM_L = re.compile(r"^\d{1,2}\s+((?:DS|ES)[1-9]{1,2}|(?:DS|ES))$", re.I)

# tile boundary heuristic
TILE_MARGIN_PX: int = 6
OUTER_BORDER_PX: int = 12
CUT_JOIN_BONUS_FACTOR = 0.5
OUTPUT_INCLUDE_TAGS = False
# ---------- Domain regex ----------
DEVICE_PAT = re.compile(r"^(CB|DS|ES|VT|CT|SS|TC|MC)\d+[A-Z]*$", re.I)
DEVICE_HEAD_PAT = re.compile(r"^(CB|DS|ES|VT|CT|SS|TC|MC)$", re.I)
SEL_SERIES_RE = re.compile(r"\bSEL[-\s]?([0-9OIl]{3,5}[A-Z]?)\b", re.I)
DS_ES_ALLOWED = re.compile(r"^(?:DS|ES)(?:[1-9]{1,2})?$", re.I)
DS_ES_LIKE = re.compile(r"^(?:DS|ES)[A-Z0-9]+$", re.I)
VOLTAGE_DIGIT_MAP  = {"1":110,
                      "2":220,
                      "3":35,
                      "4":22,
                      "5":500,
                      "6":6,
                      "7":66,
                      "8":15,
                      "9":10}
REVERSE_VOLTAGE_MAP = {v: k for k, v in VOLTAGE_DIGIT_MAP.items()}
# === Pipeline / OCR defaults (tối giản, nhanh) ===
PIPELINE_DEBUG: bool = True
OCR_USE_ANGLE_CLS = False         # True để in log thời gian/NMS
OCR_USE_GPU: bool = False            # để True nếu máy có GPU
OCR_MULTI_SCALE_SMALL: tuple = (1.0, 1.25)   # cho chữ nhỏ
OCR_MULTI_SCALE_LARGE: tuple = (1.0,)        # cho chữ to/thuận lợi
FONT_HEIGHT_SMALL_PX: int = 18       # ngưỡng quyết định small vs large
TILE_DOWNSCALE: float = 1.0          # luôn feed full tile vào OCR
MAX_WORKERS: int | None = None       # None => auto theo CPU và số tile
REC_BATCH_NUM: int = 16              # PaddleOCR rec batch
LANG_CODE: str = "en"                # ngôn ngữ PaddleOCR

# === PDF merge config (tinh chỉnh ở đây) ===
PDF_CACHE_DIR = "cache/pdf_cache"
PDF_VALIDATE_DEFAULT = True
PDF_IOU_THRESH = 0.2
PDF_CENTER_MERGE_PX    = 14       # fallback: khoảng cách tâm tối đa khi IoU nhỏ
PDF_SIM_THRESH         = 0.84     # coi là “cùng nội dung” (0..1)
PDF_PREFER_SOURCE      = "ocr"    # "pdf" hoặc "ocr" khi nội dung trùng