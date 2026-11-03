"""
Chuẩn hoá văn bản tiếng Việt cho bộ phân loại cảm xúc (v5.0).

- NFC + chữ thường + bỏ ký tự lặp kéo dài ("vuiiii" -> "vui")
- Thống nhất vị trí dấu thanh kiểu cũ/mới ("hoà" == "hòa", "thuý" == "thúy")
- Teencode / viết tắt phổ biến ("ko" -> "không", "dc" -> "được")
- strip_accents(): bản bỏ dấu, để model nhận ra cả câu gõ không dấu
  ("hom nay buon qua") dù dữ liệu train toàn câu có dấu.
"""
import re
import unicodedata

from emotion_mlp import _expand_emojis

# Dấu thanh đặt ở nguyên âm đầu (kiểu cũ) -> nguyên âm sau (kiểu mới, chuẩn
# của bộ gõ Unikey mặc định). Chỉ các cặp oa/oe/uy hay gặp khác nhau giữa 2 kiểu.
_TONE_FIX = {
    "òa": "oà", "óa": "oá", "ỏa": "oả", "õa": "oã", "ọa": "oạ",
    "òe": "oè", "óe": "oé", "ỏe": "oẻ", "õe": "oẽ", "ọe": "oẹ",
    "ùy": "uỳ", "úy": "uý", "ủy": "uỷ", "ũy": "uỹ", "ụy": "uỵ",
}
_TONE_RE = re.compile("|".join(_TONE_FIX))

# Teencode chỉ thay khi đứng thành 1 từ riêng (so khớp nguyên token).
TEENCODE = {
    "ko": "không", "k": "không", "kh": "không", "hok": "không", "hông": "không",
    "khum": "không", "hem": "không", "kg": "không", "khg": "không",
    "dc": "được", "đc": "được", "dk": "được", "đk": "được",
    "j": "gì", "gi": "gì", "cx": "cũng", "cg": "cũng", "r": "rồi", "rùi": "rồi",
    "vs": "với", "bh": "bây giờ", "bjo": "bây giờ", "h": "giờ",
    "mk": "mình", "mik": "mình", "mn": "mọi người", "ng": "người",
    "ny": "người yêu", "iu": "yêu", "thik": "thích",
    "ns": "nói", "trc": "trước", "ntn": "như thế nào", "lm": "làm",
    "nma": "nhưng mà", "nhma": "nhưng mà", "tks": "cảm ơn", "thanks": "cảm ơn",
    "oke": "ổn", "ok": "ổn", "okela": "ổn", "z": "vậy", "v": "vậy", "zậy": "vậy",
    "wá": "quá", "qá": "quá", "wa": "quá", "bùn": "buồn",
    "vl": "vãi", "vcl": "vãi", "vkl": "vãi",
}

_REPEAT_RE = re.compile(r"(\w)\1{2,}")
_NON_WORD_RE = re.compile(r"[^\w\s]")
_SPACE_RE = re.compile(r"\s+")


def normalize(text):
    t = _expand_emojis(text or "")
    t = unicodedata.normalize("NFC", t).lower()
    t = _TONE_RE.sub(lambda m: _TONE_FIX[m.group(0)], t)
    t = _REPEAT_RE.sub(r"\1", t)
    t = _NON_WORD_RE.sub(" ", t)
    words = []
    for w in t.split():
        words.append(TEENCODE.get(w, w))
    return _SPACE_RE.sub(" ", " ".join(words)).strip()


def strip_accents(text):
    t = unicodedata.normalize("NFD", text)
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return t.replace("đ", "d").replace("Đ", "D")
