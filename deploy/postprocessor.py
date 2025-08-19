# postprocessor.py (updated)
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional, Union
import math, re, unicodedata
from difflib import SequenceMatcher
import numpy as np

from header import (
    # NMS / merge / dedupe
    IOU_NMS, CENTER_THRESH_PX, GRID_MULTIPLIER,
    LINE_VOV_THRESH, SMALL_GAP_FACTOR, CHAR_GAP_FACTOR,
    IOU_DEDUP, SIM_DEDUP, VOV_DUP_THRESH, SIM_SHORT, SIM_LONG,
    TILE_MARGIN_PX, OUTER_BORDER_PX,

    # domain regex & rules
    DEVICE_PAT, DEVICE_HEAD_PAT, SEL_SERIES_RE, DS_ES_ALLOWED,
    VOLTAGE_DIGIT_MAP, CUT_JOIN_BONUS_FACTOR, DS_ES_LIKE,
    CB_LIKE_RE, NUM_SHORT_RE, DEV_HEAD_RE, ALNUM, DEVICE_VOLTAGE_PAT,
    HEADER_VOLTAGE_PAT, DS_ES_FULL_RE, DS_ES_WITH_NEAR_NUM_L, DS_ES_WITH_NEAR_NUM_R,
    REVERSE_VOLTAGE_MAP, HEADER_VOLTAGE_PAT_EN,

    # density mode (E)
    DENSE_PAGE_MODE, IOU_NMS_DENSE,
)

# =============================== geometry ===============================
def _to_int_box(b): x0,y0,x1,y1=b; return (int(round(x0)),int(round(y0)),int(round(x1)),int(round(y1)))
def _area(b): return max(0,b[2]-b[0]) * max(0,b[3]-b[1])
def _iou(a,b):
    a=_to_int_box(a); b=_to_int_box(b)
    ix0=max(a[0],b[0]); iy0=max(a[1],b[1]); ix1=min(a[2],b[2]); iy1=min(a[3],b[3])
    iw=max(0,ix1-ix0); ih=max(0,iy1-iy0); inter=iw*ih
    den=_area(a)+_area(b)-inter
    return float(inter)/float(den) if den>0 else 0.0
def _union(a,b): return (min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3]))
def _center(b): return (int((b[0]+b[2])//2), int((b[1]+b[3])//2))
def _h(b): return max(0,b[3]-b[1])
def _w(b): return max(0,b[2]-b[0])

def _vertical_overlap_ratio(a,b):
    top,bot=max(a[1],b[1]),min(a[3],b[3]); inter=max(0,bot-top)
    return inter / max(1, min(_h(a), _h(b)))
def _hgap(a,b): return (b[0]-a[2])
def _avg_h(a,b): return max(1.0, 0.5*((a[3]-a[1]) + (b[3]-b[1])))
def _min_nonneg_gap(a,b)->float:
    gr = _hgap(a,b); gl = _hgap(b,a); cand=[]
    if gr>=0: cand.append(gr)
    if gl>=0: cand.append(gl)
    return min(cand) if cand else -1.0

# =============================== text utils ===============================
def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def _canon(s: str) -> str:
    """Accent-insensitive + remove non-alnum."""
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.replace("Đ","D").replace("đ","d")
    return re.sub(r"[^A-Za-z0-9]+", "", _norm_space(s)).upper()

def _sim(a: str, b: str) -> float:
    return SequenceMatcher(a=_norm_space(a).lower(), b=_norm_space(b).lower()).ratio()

def _near_duplicate(a: str, b: str, th: float) -> bool:
    ca, cb = _canon(a), _canon(b)
    if not ca or not cb: return False
    if ca == cb or ca in cb or cb in ca: return True
    return SequenceMatcher(a=ca, b=cb).ratio() >= th

def _alnum_len(s:str)->int: return sum(1 for ch in s if ch.isalnum())
def _punct_frac(s:str)->float:
    if not s: return 0.0
    p=sum(1 for ch in s if not ch.isalnum() and not ch.isspace())
    return p/max(1,len(s))

def _text_quality_score(item: Dict[str,Any]) -> float:
    """Ưu tiên text dài, ít ký tự phụ, box hợp lý, confidence cao."""
    txt=_norm_space(str(item.get("text",""))); conf=float(item.get("confidence",0.0))
    box=_to_int_box(item.get("bbox",[0,0,0,0])); alnum=_alnum_len(txt)
    area=max(1,_area(box)); pfrac=_punct_frac(txt)
    return (alnum + 0.15*math.log(area+1.0) + 0.35*math.sqrt(max(0.0,conf)) - 0.5*pfrac)

def _almost_same(a: str, b: str, th: float = 0.97) -> bool:
    if not a and not b: return True
    if not a or not b:  return False
    ca, cb = _canon(a), _canon(b)
    if not ca or not cb: return False
    if ca == cb or ca in cb or cb in ca:
        return True
    return SequenceMatcher(a=ca, b=cb).ratio() >= th

def _looks_like_repeat_concat(a: str, b: str) -> bool:
    """Phát hiện 'AB' + 'AB' hoặc nối tạo lặp rõ ràng."""
    ca, cb = _canon(a), _canon(b)
    if not ca or not cb: return False
    if ca == cb: return True
    s = ca + cb
    # tìm chu kỳ ngắn
    for k in range(1, len(s)//2 + 1):
        if len(s) % k == 0 and s == s[:k] * (len(s)//k):
            return True
    # hoặc dạng 'AB' + 'B'/'A' (đuôi/trước trùng dài)
    return ca.endswith(cb) or cb.startswith(ca)

def _is_pure_short_num(s: str) -> bool:
    cs = _canon(s)
    return bool(NUM_SHORT_RE.fullmatch(cs))

def _is_device_like(s: str) -> bool:
    """Thiết bị/đầu mục: DS/ES (đúng), DS/ES dạng đầy đủ, hoặc header token."""
    up = _norm_space(s).upper()
    comp = re.sub(r"[^A-Z0-9]", "", up)
    return bool(DS_ES_ALLOWED.fullmatch(comp) or DS_ES_FULL_RE.fullmatch(comp) or DEV_HEAD_RE.fullmatch(up))

def _sanitize_device_near_number(text: str) -> str:
    """
    Nếu chuỗi là dạng 'DS9 4' hoặc '4 DS9' (DS/ES cạnh số ngắn),
    trả về 'DS9'/'ES9'… để loại số lẻ dính cạnh. Ngược lại trả nguyên văn.
    Dùng chung với _canonicalize_device_token để làm sạch token thiết bị.
    """
    s = _norm_space(text).upper()

    # Case đặc thù: 'ES1 0', 'DS2 0' (số 0 nhiễu trong CÙNG item) -> cắt bỏ '0'
    m = re.fullmatch(r'\s*((?:DS|ES)[1-9]{1,2})\s+0+\s*', s, re.I)
    if m:
        return m.group(1)

    # Giữ các rule cũ
    m = DS_ES_WITH_NEAR_NUM_R.fullmatch(s)
    if m:  # 'DS9 4' (không ghép), trả phần cơ sở
        return m.group(1)
    m = DS_ES_WITH_NEAR_NUM_L.fullmatch(s)
    if m:  # '4 DS9'
        return m.group(2)
    return text

# =========================== smart join guard ===========================
def _should_join_strict(cur: Dict[str,Any], nxt: Dict[str,Any], avg_h: float) -> bool:
    """Chỉ cho phép nối khi có bằng chứng mạnh; chặn 'DS/ES' dính số lẻ."""
    a, b = (cur["text"] or "").strip(), (nxt["text"] or "").strip()
    if not a or not b:
        return False
    # 0) không nối nếu gần như trùng hoặc nối tạo chuỗi lặp
    if _almost_same(a, b) or _looks_like_repeat_concat(a, b):
        return False

    vov = _vertical_overlap_ratio(cur["bbox"], nxt["bbox"])
    gap = _hgap(cur["bbox"], nxt["bbox"])
    overlap_x = (gap < 0)
    base_diff = abs(cur["center"][1] - nxt["center"][1])
    cut_bonus = CUT_JOIN_BONUS_FACTOR * avg_h if (
        "cut_boundary" in set(cur.get("tags", [])) or "cut_boundary" in set(nxt.get("tags", []))
    ) else 0.0

    # 1) CHẶN: “số ngắn” cạnh token thiết bị → không bao giờ nối
    if (_is_pure_short_num(a) and _is_device_like(b)) or (_is_pure_short_num(b) and _is_device_like(a)):
        return False

    # 2) cầu nối '-' hoặc '.'
    if (a.endswith('-') and gap < 0.5*avg_h + cut_bonus) or (b.startswith('-') and gap < 0.5*avg_h + cut_bonus):
        return True
    if ((a.endswith('.') and b[:1].isdigit()) or (a[-1:].isalnum() and b.startswith('.'))) and gap < 0.35*avg_h + cut_bonus:
        return True

    # 3) overlap-x: yêu cầu căn baseline + vov cao + mức chồng vừa
    if overlap_x:
        if (vov >= 0.60 and base_diff <= 0.30*avg_h and abs(gap) <= 0.25*_avg_h(cur["bbox"], nxt["bbox"])):
            return True
        return False

    # 4) substring thật + khoảng cách rất nhỏ
    if vov >= 0.60 and gap <= (0.20*avg_h + cut_bonus):
        if a in b or b in a:
            return True

    # 5) NỚI RIÊNG cho seam: cả hai cùng bị cắt ở rãnh tile → nới điều kiện
    cur_tags = set(cur.get("tags", [])); nxt_tags = set(nxt.get("tags", []))
    if ("cut_boundary" in cur_tags) and ("cut_boundary" in nxt_tags):
        if base_diff <= 0.35*avg_h and abs(gap) <= 0.30*avg_h:
            if not ((_is_pure_short_num(a) and _is_device_like(b)) or (_is_pure_short_num(b) and _is_device_like(a))):
                return True

    # 6) mặc định: không nối (loại hẳn DS9 + 4)
    return False

# =========================== tile-edge heuristic ===========================
def is_cut_by_tile(
    bbox,
    tile_size: Optional[Union[int, Tuple[int,int]]] = None,
    img_size: Optional[Tuple[int,int]] = None,
    stride: Optional[Tuple[int,int]] = None,
    margin: int = TILE_MARGIN_PX,
    origin: Tuple[int,int] = (0,0),
) -> bool:
    """Phát hiện box nằm sát “đường rãnh” tile để ưu tiên giữ/ghép ở hậu kỳ."""
    if tile_size is None or img_size is None:
        return False
    if isinstance(tile_size, int):
        tw, th = int(tile_size), int(tile_size)
    else:
        tw, th = int(tile_size[0]), int(tile_size[1])
    sx, sy = stride if stride else (tw, th)
    b = _to_int_box(bbox); W,H = img_size; ox,oy = origin

    def _near_to_grid(a0, a1, step, o):
        d0 = (a0 - o) % step
        d1 = (a1 - o) % step
        return min(d0, step - d1)

    near_x = _near_to_grid(b[0], b[2], sx, ox)
    near_y = _near_to_grid(b[1], b[3], sy, oy)
    on_outer = (min(b[0], b[1], W - b[2], H - b[3]) <= OUTER_BORDER_PX)
    return (near_x <= margin) or (near_y <= margin) or on_outer

# ========================= normalize special patterns =========================
def _normalize_sel_series(text: str) -> str:
    def _fix(m):
        s = m.group(1)
        s = s.replace('O','0').replace('o','0').replace('I','1').replace('l','1')
        return "SEL-" + s
    return SEL_SERIES_RE.sub(_fix, text)

def _clean_digits_no_zero(num: str) -> Optional[str]:
    """Bỏ tất cả '0'. Hợp lệ khi còn lại 1–3 chữ số (1–9)."""
    if not num:
        return None
    nz = ''.join(ch for ch in num if ch != '0')
    if re.fullmatch(r'[1-9]{1,3}', nz or ''):
        return nz
    return None

def _canonicalize_device_token(text: str) -> str:
    """
    Chuẩn hoá 1 token ngắn theo các luật:
      - DS/ES đúng chuẩn: DS, ES, DS[1-9], DS[1-9]{2} (không chữ số 0 đầu).
      - Kéo DS/ES xuất hiện 'bên trong' chuỗi (có rác hai đầu) về token sạch, ví dụ: 'r +eS11' -> 'ES11'.
      - Mẫu ghép cạnh số ngắn: 'DS9 4' hoặc '4 DS9' -> 'DS9'.
      - CB đứng một mình: sửa các biến thể OCR phổ biến (CE, C8, GB) -> 'CB'.
      - DS/ES 'na ná' nhưng sai (dính rác mà không trích được token hợp lệ) -> trả "" để bị filter.
      - Không rơi vào các trường hợp trên -> trả nguyên văn.
    """
    if not text:
        return text

    s = _norm_space(text)
    su = s.upper()
    s_comp = re.sub(r"[^A-Z0-9]", "", su)

    # 0) 'DS9 4' hoặc '4 DS9' -> lấy phần DS/ES<n>
    m = DS_ES_WITH_NEAR_NUM_R.fullmatch(su)  # ví dụ 'DS9 4'
    if m:
        return m.group(1)
    m = DS_ES_WITH_NEAR_NUM_L.fullmatch(su)  # ví dụ '4 DS9'
    if m:
        return m.group(1)

    # 1) DS/ES hợp lệ tuyệt đối (không số 0 đứng đầu)
    m_ok = DS_ES_ALLOWED.fullmatch(s_comp)   # ^(?:DS|ES)(?:[1-9]{1,2})?$
    if m_ok:
        return m_ok.group(0)

    # 1b) TÌM DS/ES NẰM TRONG CHUỖI (loại rác hai đầu). Ví dụ: 'r +eS11' -> 'ES11'
    m = re.search(r'(?:^|[^A-Z0-9])(DS|ES)\s*([0-9]{1,3})?(?:[^A-Z0-9]|$)', su)
    m = re.search(r'(?:^|[^A-Z0-9])(DS|ES)\s*([0-9]{1,3})?(?:[^A-Z0-9]|$)', su)
    if m:
        head, num = m.group(1), m.group(2)
        if not num:
            return head  # 'DS' / 'ES' đơn
        num_clean = _clean_digits_no_zero(num)   # <-- bỏ '0' nhiễu
        if num_clean:
            return f"{head}{num_clean}"
        return "" 

    # 2) DS/ES giống mà sai -> loại bỏ (để không lẫn với token sạch)
    if ("DS" in s_comp or "ES" in s_comp):
        return ""

    # 3) CB 2 ký tự đứng 1 mình: sửa các biến thể OCR hay gặp
    if len(s_comp) == 2:
        if s_comp == "CB":
            return "CB"
        # Biến thể OCR thường gặp: CE / C8 / GB
        if CB_LIKE_RE.fullmatch(s_comp):    # ^(?:C|G)(?:B|8|E)$
            return "CB"

    # 4) Không matches gì: giữ nguyên văn bản gốc
    return text


# =========================== smart join for drawings ===========================
def _join_text(a_txt: str, b_txt: str, gap: int, avg_h: float) -> str:
    a = (a_txt or "").strip(); b = (b_txt or "").strip()
    if gap <= 0: return (a + b).strip()
    # gắn theo ngữ pháp đơn giản:
    if (a.endswith('-') and gap < 0.5 * avg_h) or (b.startswith('-') and gap < 0.5 * avg_h):
        return (a + b).replace(' -','-').replace('- ','-').strip()
    if (a.endswith('.') and b[:1].isdigit() and gap < 0.35 * avg_h) \
       or (a[-1:].isalnum() and b.startswith('.') and gap < 0.35 * avg_h):
        return (a + b).replace('. ','.').replace(' .','.').strip()
    # Alnum-alnum rất gần → dính liền
    a_is = bool(ALNUM.search(a[-1:] or "")); b_is = bool(ALNUM.search(b[:1] or ""))
    if a_is and b_is and gap < 0.35 * avg_h:
        return (a + b).strip()
    return (a + " " + b).strip()

# ============================ grid NMS (seam-aware) ============================
def advanced_nms_text(
    items: List[Dict[str, Any]],
    iou_thresh: float = IOU_NMS,
    center_thresh: int = CENTER_THRESH_PX,
    tile_size: Optional[Union[int, Tuple[int, int]]] = None,
    img_size: Optional[Tuple[int, int]] = None,
    stride: Optional[Tuple[int,int]] = None,
    stage: Optional[str] = None,
    grid_px: Optional[int] = None,
    avg_h_hint: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    NMS lân cận theo grid (O(N)), seam-aware:
      - Ưu tiên text chất lượng (dài/hợp lý/độ tin cao)
      - Không drop thiết bị/header pattern khi trùng vị trí
      - Nếu dính seam, làm nhẹ hơn (để ghép ở merge)
    """
    if not items: return []

    # relax IoU khi trang dày đặc
    if DENSE_PAGE_MODE:
        iou_thresh = min(iou_thresh, IOU_NMS_DENSE)

    # chuẩn hoá & auto-tag seam
    std: List[Dict[str, Any]] = []
    for it in items:
        b = _to_int_box(it["bbox"])
        c = it.get("center") or _center(b)
        txt = _norm_space(str(it.get("text","")))
        conf = float(it.get("confidence", 0.0))
        tags = set(it.get("tags", []))
        if tile_size is not None and img_size is not None:
            if is_cut_by_tile(b, tile_size=tile_size, img_size=img_size, stride=stride):
                tags.add("cut_boundary")
        std.append({
            "text": txt, "bbox": b, "center": c, "confidence": conf,
            "tags": list(tags),
            "_canon": _canon(txt),
            "_area": _area(b),
        })

    std.sort(key=_text_quality_score, reverse=True)

    def _median_h(arr: List[Dict[str, Any]]) -> float:
        hs = [max(1, _to_int_box(it["bbox"])[3] - _to_int_box(it["bbox"])[1]) for it in arr]
        return float(np.median(hs)) if hs else 24.0

    base_h = float(avg_h_hint) if (avg_h_hint and avg_h_hint > 0) else _median_h(std)
    if grid_px is None or grid_px <= 0:
        grid_px = int(max(48, min(256, round(GRID_MULTIPLIER * base_h))))
    # Nới center_thresh theo cỡ chữ (không “cap” xuống nhỏ hơn)
    center_thresh = int(round(max(center_thresh, 0.75 * base_h)))

    picked: List[Dict[str, Any]] = []
    grid: Dict[Tuple[int,int], List[int]] = {}

    def _cells_for(b):
        x0,y0,x1,y1 = _to_int_box(b)
        xs = range(x0 // grid_px, max(0,(x1-1)) // grid_px + 1)
        ys = range(y0 // grid_px, max(0,(y1-1)) // grid_px + 1)
        return xs, ys

    def _neighbors(b):
        xs, ys = _cells_for(b)
        for gx in range(min(xs)-1, max(xs)+2):
            for gy in range(min(ys)-1, max(ys)+2):
                for idx in grid.get((gx,gy), []):
                    yield idx

    for it in std:
        keep = True
        if picked:
            for j in _neighbors(it["bbox"]):
                sel = picked[j]
                it_tags  = set(it.get("tags", []))
                sel_tags = set(sel.get("tags", []))
                seam_involved = ("cut_boundary" in it_tags) or ("cut_boundary" in sel_tags)

                # 1) IoU lớn
                if _iou(it["bbox"], sel["bbox"]) >= iou_thresh:
                    if DEVICE_PAT.match(it["text"]) or DEVICE_PAT.match(sel["text"]) \
                       or DEVICE_HEAD_PAT.match(it["text"]) or DEVICE_HEAD_PAT.match(sel["text"]):
                        continue
                    # near-dup → drop cái mới
                    if _near_duplicate(it["text"], sel["text"], SIM_LONG):
                        keep = False; break
                    # text khác nhau: giữ cả 2 (để merge), nhất là nếu có seam
                    continue

                # 2) Tâm gần
                cx0, cy0 = it["center"]; cx1, cy1 = sel["center"]
                if abs(cx0-cx1) <= center_thresh and abs(cy0-cy1) <= center_thresh:
                    if DEVICE_PAT.match(it["text"]) or DEVICE_PAT.match(sel["text"]) \
                       or DEVICE_HEAD_PAT.match(it["text"]) or DEVICE_HEAD_PAT.match(sel["text"]):
                        continue
                    # Không seam: nếu khá giống -> drop cái mới
                    if not seam_involved and _near_duplicate(it["text"], sel["text"], SIM_LONG):
                        keep = False; break
                    continue

                # 3) cùng dòng & gap nhỏ (không seam) → coi là dup
                if not seam_involved:
                    vov = _vertical_overlap_ratio(it["bbox"], sel["bbox"])
                    if vov >= VOV_DUP_THRESH:
                        gap = _min_nonneg_gap(it["bbox"], sel["bbox"])
                        if 0 <= gap <= SMALL_GAP_FACTOR * _avg_h(it["bbox"], sel["bbox"]) \
                           and _near_duplicate(it["text"], sel["text"], SIM_LONG):
                            keep = False; break

        if keep:
            idx = len(picked); picked.append(it)
            xs, ys = _cells_for(it["bbox"])
            for gx in xs:
                for gy in ys:
                    grid.setdefault((gx,gy), []).append(idx)

    return picked

# ================================ line merge ================================
def _assign_to_line(lines: List[Dict[str,Any]], it: Dict[str,Any], vov_thresh: float)->None:
    b=it["bbox"]; best=(-1,0.0)
    for i,L in enumerate(lines):
        vov=_vertical_overlap_ratio(L["bbox"], b)
        if vov>=vov_thresh and vov>best[1]: best=(i,vov)
    if best[0]>=0:
        idx=best[0]; lines[idx]["bbox"]=_union(lines[idx]["bbox"], b)
        lines[idx]["items"].append(it); lines[idx]["heights"].append(_h(b))
    else:
        lines.append({"bbox":b,"items":[it],"heights":[_h(b)]})

def merge_split_texts(items: List[Dict[str,Any]],
                      line_vov_thresh:float=LINE_VOV_THRESH,
                      gap_factor:float=SMALL_GAP_FACTOR,
                      char_gap_factor:float=CHAR_GAP_FACTOR,
                      log: bool=False) -> List[Dict[str,Any]]:
    if not items: return []

    # sort & gán vào dòng
    items_sorted = sorted(items, key=lambda it: (it["center"][1], it["bbox"][1], it["bbox"][0]))
    lines: List[Dict[str,Any]] = []
    for it in items_sorted:
        _assign_to_line(lines, it, vov_thresh=line_vov_thresh)

    merged_all: List[Dict[str,Any]] = []

    for L in lines:
        L_items = sorted(L["items"], key=lambda it: it["bbox"][0])
        avg_h = float(np.median(L["heights"])) if L["heights"] else 24.0
        if avg_h <= 0: avg_h = 24.0

        # (1) TRIỆT DUP TRONG DÒNG trước khi nối
        kept_line: List[Dict[str,Any]] = []
        for it in L_items:
            drop = False
            for j, sel in enumerate(kept_line):
                vov = _vertical_overlap_ratio(it["bbox"], sel["bbox"])
                gap = _min_nonneg_gap(it["bbox"], sel["bbox"])
                center_close = abs(it["center"][1] - sel["center"][1]) <= 0.45*avg_h
                if (vov >= 0.55 and (gap < 0 or gap <= 0.30*avg_h) and _almost_same(it["text"], sel["text"]) and center_close):
                    # chọn bản tốt hơn
                    better = it if _text_quality_score(it) >= _text_quality_score(sel) else sel
                    kept_line[j] = {**better, "tags": sorted(set(better.get("tags",[])) | set(sel.get("tags",[])) | set(it.get("tags",[])))}
                    drop = True; break
            if not drop:
                kept_line.append(it)

        # (2) NỐI CHẶT (seam-aware bằng _should_join_strict)
        out_line: List[Dict[str,Any]] = []
        cur = None
        for it in kept_line:
            if cur is None:
                cur = dict(it); continue

            if not _should_join_strict(cur, it, avg_h):
                # không nối → push cur
                cur["text"] = _sanitize_device_near_number(
                 _canonicalize_device_token(_normalize_sel_series(cur["text"]))
              )
                out_line.append(cur)
                cur = dict(it)
                continue

            # nối an toàn
            gap = _hgap(cur["bbox"], it["bbox"])
            u   = _union(cur["bbox"], it["bbox"])
            jt  = _join_text(cur["text"], it["text"], gap, avg_h)

            # nếu kết quả nối tạo lặp → HUỶ nối, chọn bản tốt hơn
            if _looks_like_repeat_concat(cur["text"], it["text"]) or _almost_same(jt, cur["text"]) or _almost_same(jt, it["text"]):
                better = cur if _text_quality_score(cur) >= _text_quality_score(it) else it
                better = dict(better); better["tags"] = sorted(set(cur.get("tags",[])) | set(it.get("tags",[])))
                cur = better
                continue

            txt = _canonicalize_device_token(_normalize_sel_series(jt))
            cur = {**cur, "text": txt, "bbox": u, "center": _center(u),
                   "confidence": max(float(cur.get("confidence",0)), float(it.get("confidence",0))),
                   "tags": sorted(set(cur.get("tags", [])) | set(it.get("tags", [])))}

        if cur is not None:
            cur["text"] = _sanitize_device_near_number(
                 _canonicalize_device_token(_normalize_sel_series(cur["text"]))
              )
            out_line.append(cur)

        merged_all.extend(out_line)

    if log:
        print(f"[POST] merge(strict) lines={len(lines)} in={len(items)} out={len(merged_all)}")
    return merged_all

# ================================ dedupe cuối ================================
def remove_duplicate_substrings(items: List[Dict[str,Any]],
                                iou_thresh:float=IOU_DEDUP,
                                sim_thresh:float=SIM_DEDUP,
                                same_line_gap_factor: float = SMALL_GAP_FACTOR) -> List[Dict[str,Any]]:
    """
    Gỡ các bản sao/substring gần nhau (ưu tiên giữ item dài/chất lượng hơn).
    Seam-aware: nếu có 'cut_boundary', chỉ xoá khi chắc chắn là dup.
    + Bảo vệ trường hợp KHÁC DÒNG (baseline lệch lớn) không bị xoá nhầm.
    """
    if not items: return []
    items_sorted = sorted(items, key=_text_quality_score, reverse=True)
    kept: List[Dict[str,Any]] = []

    for it in items_sorted:
        t_it  = _norm_space(it["text"]); c_it = _canon(t_it)
        if not t_it: continue
        it_tags = set(it.get("tags", []))
        keep=True

        for sel in kept:
            # NEW: nếu lệch baseline lớn → bỏ qua so sánh dup
            baseline_diff = abs(it["center"][1] - sel["center"][1])
            if baseline_diff > 0.5 * _avg_h(it["bbox"], sel["bbox"]):
                continue

            t_sel = _norm_space(sel["text"]); c_sel = _canon(t_sel)
            sel_tags = set(sel.get("tags", []))
            seam_involved = ("cut_boundary" in it_tags) or ("cut_boundary" in sel_tags)

            # 1) IoU lớn
            if _iou(it["bbox"], sel["bbox"]) >= iou_thresh:
                if DEVICE_PAT.match(t_it) or DEVICE_PAT.match(t_sel) or DEVICE_HEAD_PAT.match(t_it) or DEVICE_HEAD_PAT.match(t_sel):
                    continue
                # với seam: chỉ xoá nếu canonical y hệt/substring rõ
                if seam_involved:
                    if (t_it in t_sel or t_sel in t_it) or (c_it and c_sel and (c_it==c_sel or c_it in c_sel or c_sel in c_it)):
                        keep=False; break
                else:
                    if (t_it in t_sel or t_sel in t_it) or (c_it and c_sel and (c_it==c_sel or c_it in c_sel or c_sel in c_it)):
                        keep=False; break
                    short=(len(t_it)<=3 or len(t_sel)<=3)
                    if _sim(t_it, t_sel) >= (SIM_SHORT if short else sim_thresh):
                        keep=False; break

            # 2) near-dup cùng dòng (không seam)
            if not seam_involved:
                vov = _vertical_overlap_ratio(it["bbox"], sel["bbox"])
                if vov >= VOV_DUP_THRESH:
                    gap = _min_nonneg_gap(it["bbox"], sel["bbox"])
                    if 0 <= gap <= same_line_gap_factor * _avg_h(it["bbox"], sel["bbox"]) and (c_it and c_sel and (c_it==c_sel or c_it in c_sel or c_sel in c_it)):
                        keep=False; break

        if keep:
            kept.append(it)

    return kept

# ============================ fixed-pattern filter ============================
def is_invalid_fixed_pattern(text: str) -> bool:
    s = (text or "").strip()
    if not s: return True
    if len(s)==1 and not s.isalnum(): return True
    if re.fullmatch(r"[^\w\s]+", s): return True
    if re.fullmatch(r"[(){}\[\]<>]+", s): return True
    if re.fullmatch(r"[-_/\\=.,:;|`~^'\"–—•·…]{2,}", s): return True

    comp = re.sub(r"[^A-Z0-9]", "", s.upper())
    # Hợp lệ -> OK; Giống DS/ES nhưng KHÔNG hợp lệ -> loại
    if ("DS" in comp or "ES" in comp) and not DS_ES_ALLOWED.fullmatch(comp):
        return True
    if DS_ES_ALLOWED.fullmatch(comp):
        return False
    if DS_ES_LIKE.fullmatch(comp):
        return True
    return False

# ===================== voltage level detector (optional) =====================

def _noacc_upper(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.replace("đ","d").replace("Đ","D")
    return _norm_space(s).upper()

def detect_voltage_levels_from_texts(text_list: List[str]) -> List[int]:
    from header import PIPELINE_DEBUG
    levels_found = set()
    DBG = bool(PIPELINE_DEBUG)
    # ---------- priority 1 (UNCHANGED) ----------
    for text in text_list:
        m = DEVICE_VOLTAGE_PAT.search((text or "").strip())
        if m:
            digit = m.group(2)
            voltage = VOLTAGE_DIGIT_MAP.get(digit)
            if DBG:
                print(f"[VOLT:P1] hit='{text}'  match='{m.group(0)}'  digit='{digit}'  -> {voltage}")
            if voltage:
                levels_found.add(voltage)
    if levels_found:
        if DBG:
            print(f"[VOLT:FINAL] {sorted(levels_found)}  (source=P1/device_token)")
        return sorted(levels_found)

    # ---------- priority 2: header (VN + EN) ----------
    def _pick_digit(m: re.Match) -> Optional[int]:
        if not m:
            return None
        for g in reversed(m.groups() or []):
            if g and g.isdigit():
                try:
                    return int(g)
                except Exception:
                    pass
        return None

    for raw in text_list:
        t = _noacc_upper(raw)  # normalize: strip accents, uppercase, collapse spaces
        m = HEADER_VOLTAGE_PAT.search(t) or HEADER_VOLTAGE_PAT_EN.search(t)
        v = _pick_digit(m)
        if DBG and m:
            src = "VN_HEADER" if m else ("EN_HEADER" if m else "SIDE")
            print(f"[VOLT:P2:{src}] raw='{raw}'  norm='{t}'  match='{m.group(0)}'  -> {v}")
        if v is not None and v in REVERSE_VOLTAGE_MAP:  # REVERSE_VOLTAGE_MAP must use int keys
            levels_found.add(v)
    if DBG:
        print(f"[VOLT:FINAL] {sorted(levels_found)}  (source=P2/header)")
    return sorted(levels_found)



# ================================= exports =================================
__all__ = [
    "advanced_nms_text",
    "merge_split_texts",
    "remove_duplicate_substrings",
    "is_cut_by_tile",
    "is_invalid_fixed_pattern",
    "detect_voltage_levels_from_texts",
    "_iou",
    "_sanitize_device_near_number"
]
