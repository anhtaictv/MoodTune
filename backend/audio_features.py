"""
MoodTune Audio Feature Engine (Ý tưởng 2 - bản lite)
=====================================================
Phân tích đặc trưng âm thanh (BPM, Spectral Centroid, MFCC) của track Jamendo
chạy NGẦM (background thread) + cache vào audio_cache.json, KHÔNG chặn
/api/music/search.

Nếu librosa không cài được (ví dụ Python quá mới) → AUDIO_ENABLED=False, mọi
hàm trở thành no-op, hệ thống vẫn chạy bình thường (graceful fallback).
"""
import os
import json
import threading
import urllib.request
import io

try:
    import numpy as np
    import librosa
    AUDIO_ENABLED = True
except Exception as e:
    AUDIO_ENABLED = False
    print(f"[AudioFeatures] Tắt tính năng phân tích âm thanh (thiếu thư viện: {e})")

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_cache.json")
ANALYZE_DURATION = 30   # giây đầu của track
MAX_BG_PER_CALL = 5     # số track tối đa phân tích nền mỗi lần gọi search

_lock = threading.Lock()
_in_progress = set()
_cache = {}

if os.path.exists(CACHE_PATH):
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            _cache = json.load(f)
    except Exception:
        _cache = {}


def _save_cache():
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(_cache, f, ensure_ascii=False)


def _audio_to_emotion(bpm, centroid):
    """Heuristic map BPM (nhịp/phút) + Spectral Centroid (độ sáng âm thanh)
    -> 1 trong 10 nhãn cảm xúc."""
    if bpm >= 120 and centroid >= 2500:
        return "energetic"
    if bpm >= 120:
        return "happy"
    if bpm < 70 and centroid < 1500:
        return "sad"
    if bpm < 70:
        return "relaxed"
    if 90 <= bpm < 120 and centroid >= 2200:
        return "happy"
    if centroid >= 3000:
        return "stressed"
    if 70 <= bpm < 100 and centroid < 1800:
        return "lonely"
    if 70 <= bpm < 100:
        return "focused"
    return "romantic"


def analyze_track(track_id, audio_url):
    """Download + phân tích 1 track. Trả về dict đặc trưng hoặc None nếu lỗi."""
    if not AUDIO_ENABLED or not audio_url:
        return None
    try:
        req = urllib.request.Request(audio_url, headers={"User-Agent": "MoodTune/2.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read()
        y, sr = librosa.load(io.BytesIO(raw), sr=22050, duration=ANALYZE_DURATION, mono=True)
        if y.size == 0:
            return None
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo)
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)

        result = {
            "bpm": round(bpm, 1),
            "centroid": round(centroid, 1),
            "mfcc": [round(float(x), 3) for x in mfcc],
            "audio_emotion": _audio_to_emotion(bpm, centroid),
        }
        with _lock:
            _cache[track_id] = result
            _save_cache()
        return result
    except Exception as e:
        print(f"[AudioFeatures] Lỗi phân tích track {track_id}: {e}")
        return None


def get_cached(track_id):
    return _cache.get(track_id)


def analyze_async(tracks):
    """Phân tích nền tối đa MAX_BG_PER_CALL track Jamendo chưa có cache,
    không chặn request hiện tại."""
    if not AUDIO_ENABLED:
        return

    todo = []
    for t in tracks:
        tid = t.get("id")
        if not tid or tid in _cache or tid in _in_progress or not t.get("preview"):
            continue
        todo.append(t)
        if len(todo) >= MAX_BG_PER_CALL:
            break
    if not todo:
        return

    def worker():
        for t in todo:
            tid = t["id"]
            _in_progress.add(tid)
            try:
                analyze_track(tid, t["preview"])
            finally:
                _in_progress.discard(tid)

    threading.Thread(target=worker, daemon=True).start()
