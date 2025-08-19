from __future__ import annotations
from typing import Tuple
import re

# ---------- Preprocessor defaults ----------
PREFER_TILE: Tuple[int, int] = (1408, 1408)
MAX_TILE_LIMIT: Tuple[int, int] = (1536, 1536)
OVERLAP_RATIO: float = 0.2
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
DEV_HEAD_RE = re.compile(r"^(BB|C|TC|T)\d{1,2}$", re.I)  
ALNUM = re.compile(r"[A-Za-z0-9]")
DEVICE_VOLTAGE_PAT  = re.compile(r"^(BB|C|TC|T)([156234897])\d{1,2}$", re.I)
HEADER_VOLTAGE_PAT = re.compile(
    r"\bPHIA(?:\s+SO\s*DO)?\b.*?(\d{2,3})\s*K\s*V"
    r"|"
    r"\bSO\s*DO\b.*?\bPHIA\b.*?(\d{2,3})\s*K\s*V",
    re.IGNORECASE
)

HEADER_VOLTAGE_PAT_EN = re.compile(
    r"\b(\d{2,3})\s*K\s*V\b\s*(?:SIDE|BAY)\b",
    re.IGNORECASE
)
DS_ES_FULL_RE = re.compile(r'^(?:DS|ES)[1-9]{1,3}[A-Z]?$', re.I)  # fullmatch, không lệ thuộc header

# 2 config DE/ES
DS_ES_WITH_NEAR_NUM_R = re.compile(r'^\s*((?:DS|ES)[1-9]{1,2})\s+([1-9])\s*$', re.I)
DS_ES_WITH_NEAR_NUM_L = re.compile(r'^\s*([1-9])\s+((?:DS|ES)[1-9]{1,2})\s*$', re.I)
DS_ES_ALLOWED = re.compile(r'^(?:DS|ES)(?:[1-9]{1,3})?$', re.I)
DS_ES_LIKE = re.compile(r"^(?:DS|ES)[A-Z0-9]+$", re.I)
# tile boundary heuristic
TILE_MARGIN_PX: int = 6
OUTER_BORDER_PX: int = 12
CUT_JOIN_BONUS_FACTOR = 0.5
OUTPUT_INCLUDE_TAGS = False
# ---------- Domain regex ----------
DEVICE_PAT = re.compile(r"^(CB|DS|ES|VT|CT|SS|TC|MC)\d+[A-Z]*$", re.I)
DEVICE_HEAD_PAT = re.compile(r"^(CB|DS|ES|VT|CT|SS|TC|MC)$", re.I)
SEL_SERIES_RE = re.compile(r"\bSEL[-\s]?([0-9OIl]{3,5}[A-Z]?)\b", re.I)
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
PDF_VALIDATE_DEFAULT = True        # True để bật validate PDF
PDF_IOU_THRESH = 0.2
PDF_CENTER_MERGE_PX    = 14       # fallback: khoảng cách tâm tối đa khi IoU nhỏ
PDF_SIM_THRESH         = 0.84     # coi là “cùng nội dung” (0..1)
PDF_PREFER_SOURCE      = "ocr"    # "pdf" hoặc "ocr" khi nội dung trùng

# ---------- Performance & Memory control ----------
# ---------- Pipeline toggles (NEW) ----------
USE_SHARED_ENGINE_POOL: bool = True          # P1: pool OCR dùng chung toàn batch
HARD_MAX_WORKERS: int = 2                    # Yêu cầu: tối đa 2 worker
APPLY_ENHANCE_CONDITIONAL: bool = True       # P2: chỉ enhance tiles “khó”
DYNAMIC_OVERLAP: bool = True                 # P3: overlap động theo avg font
OVERLAP_MIN: float = 0.15                    # khi font to/ảnh thuận lợi
OVERLAP_MAX: float = OVERLAP_RATIO           # giữ mặc định 0.25

PER_TILE_MULTISCALE: bool = True             # P3/B: scale 1.25 chỉ cho tiles small-font
ROTATION_PASS_90: bool = False                # A: chạy pass xoay 90° để bắt chữ dọc
HIGH_RECALL_TILES: bool = True               # B: tiles “khó” chạy det high-recall



# Dày đặc (E): hạ ngưỡng NMS khi trang dày đặc khung
DENSE_PAGE_MODE: bool = True
IOU_NMS_DENSE: float = 0.25

# Memory guard (stream tiles)
TILE_QUEUE_MAX: int = 4                      # chỉ giữ tối đa 4 tile trong pipeline
