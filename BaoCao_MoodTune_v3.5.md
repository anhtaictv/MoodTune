# MoodTune v3.5 — Báo cáo thay đổi so với v3.1

**Phiên bản:** `v3.5` (so với `v3.1` trong `BaoCao_MoodTune_v3.1.md`)
**Tên đầy đủ:** MoodTune — AI Cảm Xúc Tự Xây (RLUF Bandit · Knowledge Graph · Self-Attention) + Gợi ý nhạc Jamendo Hybrid
**Chủ đề nâng cấp:** Chuẩn hoá hệ thống cảm xúc theo mô hình khoa học Valence-Arousal (GEMS/Circumplex)

v3.5 gồm **5 thay đổi/cải tiến chính**, mỗi mục trình bày theo cấu trúc:
**Mô tả → Cài đặt kỹ thuật → Kết quả kiểm thử**.

---

## Thay đổi 1: Rút gọn hệ thống cảm xúc (15 → 10 class)

### Mô tả
v3.1 có 15 class cảm xúc nhưng phân tích lại cho thấy một số class không đạt tiêu chí "cảm xúc đơn giản" (simple emotion) phù hợp với trải nghiệm nghe nhạc:

| Class bị loại | Lý do |
|---|---|
| `vui_nhon` 🤪 | Trùng lặp với `vui_ve` (happy) — cùng nhóm High Arousal + Positive Valence, không cần class riêng |
| `phieu_luu` ⚔️ | Là trạng thái tâm trí (adventurous/curious), không phải cảm xúc thuần — khó ánh xạ nhạc chính xác |
| `bi_an` 🔮 | Không phải cảm xúc thuần, là đặc tính âm nhạc (mysterious/dark) — dễ nhầm với căng thẳng |
| `tu_tin` 💪 | Không phải cảm xúc thuần theo mô hình Valence-Arousal, gần với `nang_dong` (energetic) |
| `biet_on` 🙏 | Trạng thái nhận thức (gratefulness), không được biểu diễn rõ ràng trong không gian cảm xúc âm nhạc |

Kết quả: **10 "cảm xúc đơn giản"** được giữ lại, phủ đều 4 vùng Valence-Arousal.

### Cài đặt kỹ thuật
- **`lexicon.py` → `EMOTIONS`**: Rút từ 15 → 10 class, đổi tên nội bộ sang tiếng Anh cho nhất quán:

| Tên cũ (v3.1) | Tên mới (v3.5) | Nhãn hiển thị |
|---|---|---|
| `vui_ve` | `happy` | Vui vẻ 😄 |
| `buon_ba` | `sad` | Buồn bã 😢 |
| `lang_man` | `romantic` | Lãng mạn 🥰 |
| `nang_dong` | `energetic` | Năng động ⚡ |
| `thu_gian` | `relaxed` | Thư giãn 🌿 |
| `co_don` | `lonely` | Cô đơn 🌙 |
| `cang_thang` | `stressed` | Căng thẳng 🔥 |
| `tap_trung` | `focused` | Tập trung 🎯 |
| `hoai_niem` | `nostalgic` | Hoài niệm 🍂 |
| `tuc_gian` | `angry` | Tức giận 😡 |

- Vocab của `vui_nhon` (hài hước, buồn cười, lầy lội, lol, lmao, comedy, funny, v.v.) được **gộp vào `happy`** — không mất dữ liệu từ điển.
- Các class bị loại (`phieu_luu`, `bi_an`, `tu_tin`, `biet_on`) và toàn bộ vocab riêng của chúng bị xoá khỏi `LEXICON` — `cang_thang`/`stressed` giữ nguyên phần vocab "stress/lo âu" (không có từ giận dữ, đã được hotfix ở v3.1 Chức năng 7).
- `VOCAB_SIZE` thực tế thay đổi do loại bỏ các entry không còn xuất hiện ở class nào khác.

### Kết quả kiểm thử
- `python -m py_compile lexicon.py` → OK.
- Log khởi động backend: `[Init] Vocab=... | Classes=10` — đúng 10 class.
- `POST /api/predict` trả đúng **10 entries** trong `scores`/`graph`.

---

## Thay đổi 2: Áp dụng mô hình Valence-Arousal (GEMS/Circumplex)

### Mô tả
Thêm toạ độ khoa học **Valence** (Sắc thái: âm/dương) và **Arousal** (Cường độ năng lượng: thấp/cao) vào metadata của mỗi cảm xúc, dựa trên mô hình **Geneva Emotion Music Scale (GEMS)** kết hợp **Circumplex Model of Affect (Russell)**. Đây là chuẩn nghiên cứu phổ biến trong ngành âm nhạc học tính toán (Music Information Retrieval).

### Cài đặt kỹ thuật
- **`lexicon.py` → `EMOTION_META`**: Thêm 2 trường `valence` và `arousal` (giá trị trong `[-1, 1]`) cho mỗi class:

| Class | Valence | Arousal | Vùng Circumplex |
|---|---|---|---|
| `happy` | +0.85 | +0.50 | High Arousal, Positive (gộp vui vẻ + vui nhộn → trung bình) |
| `sad` | −0.80 | −0.60 | Low Arousal, Negative |
| `romantic` | +0.70 | +0.20 | Low-Mid Arousal, Positive |
| `energetic` | +0.60 | +0.90 | High Arousal, Positive |
| `relaxed` | +0.80 | −0.40 | Low Arousal, Positive |
| `lonely` | −0.70 | −0.50 | Low Arousal, Negative |
| `stressed` | −0.60 | +0.80 | High Arousal, Negative |
| `focused` | +0.40 | −0.50 | Low-Mid Arousal, Slightly Positive |
| `nostalgic` | −0.20 | −0.30 | Low Arousal, Slightly Negative |
| `angry` | −0.75 | +0.85 | High Arousal, Negative (cực đoan hơn stressed) |

- **`app.py` → `/api/predict`**: Response bổ sung 2 field `valence` và `arousal` của emotion chính, và mỗi entry trong `scores` list cũng kèm `valence`/`arousal` — frontend/đồ án có thể dùng để vẽ biểu đồ 2D.
- **`audio_features.py` → `_audio_to_emotion()`**: Hàm heuristic ánh xạ BPM/Spectral Centroid sang nhãn cảm xúc được cập nhật dùng 10 tên class tiếng Anh mới (thay vì các key tiếng Việt cũ).

### Kết quả kiểm thử
- `GET /api/predict` → response bao gồm `"valence": 0.85, "arousal": 0.5` (ví dụ với emotion=happy) — đúng theo bảng.
- `GET /api/health` → `output_size: 10` — xác nhận kiến trúc đầu ra đúng.

---

## Thay đổi 3: Cập nhật từ điển LEXICON & tag Jamendo

### Mô tả
Hai tác vụ song song: (1) dọn dẹp từ điển sau khi bỏ 5 class, gộp vocab `vui_nhon` vào `happy`; (2) cập nhật pool tag Jamendo trong `app.py` để khớp với 10 key tiếng Anh mới.

### Cài đặt kỹ thuật
**Từ điển (lexicon.py):**
- `happy` được mở rộng đáng kể bằng cách hấp thụ toàn bộ vocab `vui_nhon` (hài hước, buồn cười, tếu, nhí nhố, lầy lội, troll, comedy, funny, lol, lmao, rofl, joke, humor, cười bể bụng, cười vỡ bụng, stand up, roast, gag, prank, silly, goofy, playful, witty, bông đùa, v.v.).
- Các từ riêng của `phieu_luu`, `bi_an`, `tu_tin`, `biet_on` bị xoá.
- `angry` giữ nguyên từ điển của `tuc_gian` từ v3.1 (đã được hotfix — loại bỏ overlap với `cang_thang`).

**EMOTION_TAGS (app.py):**
```
Đổi key:  "nang_dong" → "energetic", "thu_gian" → "relaxed", v.v.
Bổ sung:  "angry" pool (thay thế "stressed" pool tag metal/punk, stressed giữ riêng)
Bỏ đi:   "tu_tin", "biet_on", "vui_nhon", "phieu_luu", "bi_an" pool
```
- `stressed` giữ pool tag `["stress+heavy", "intense+industrial", "rock+intense", "metal+power", "punk+fast"]` (năng lượng cao, tiêu cực — đúng arousal +0.8).
- `angry` có pool riêng `["anger+aggressive", "metal+hardcore", "metal+aggressive", "punk+rock", "rock+heavy"]` để phân biệt rõ với stressed.

### Kết quả kiểm thử
- `GET /api/music/search?emotion=happy` → tags lấy từ pool `happy` mới (đã gộp vui nhộn) — đa dạng hơn.
- `GET /api/music/search?emotion=angry` vs `?emotion=stressed` → trả về pool tag khác nhau.
- `GET /api/mix-ratio?emotion=happy` → 200 OK, bandit init đúng 10 key.

---

## Thay đổi 4: Retrain Self-Attention MLP cho 10 class

### Mô tả
Kiến trúc `AttentionMLP` co giãn theo `len(EMOTIONS)` — khi số class thay đổi (15→10), **bắt buộc phải train lại từ đầu** vì ma trận đầu ra `W2`/`b2` có shape phụ thuộc trực tiếp vào số class.

### Cài đặt kỹ thuật
- `W2`: `(64, 15)` → `(64, 10)`; `b2`: `(15,)` → `(10,)`.
- **`SEED_DATA`**: 144 câu → **92 câu** (bỏ seed của 5 class đã loại; angry/tuc_gian giữ nguyên 20 câu để bù vị trí khó về overlap với stressed):
  - 8 câu × 9 class (happy → nostalgic) = 72 câu
  - 20 câu cho `angry` (index 9) = 20 câu
  - **Tổng: 92 câu**
- Quy trình retrain: Xoá `weights.npz` → `pm2 restart moodtune-backend` → `pretrain(SEED_DATA + replay, epochs=400)`.
- **Replay buffer cũ**: bị xoá (label 0-14 không còn khớp schema 10 class mới) → `weights_replay.json` được reset.

| Tham số | v3.1 | v3.5 |
|---|---|---|
| Số class đầu ra | 15 | **10** |
| `W2` shape | (64, 15) | **(64, 10)** |
| `SEED_DATA` | 144 câu | **92 câu** |
| Replay buffer khi deploy | 500 mẫu (label 0-14) | **Reset** (label 0-9) |

### Kết quả kiểm thử
- Pretrain 400 epoch hoàn tất, `weights.npz` shape mới — không lỗi shape-mismatch khi load.
- `POST /api/predict` ("hôm nay tôi rất vui") → `emotion=happy`, confidence cao — ~290ms/request.
- Các câu kiểm thử cũ (angry/stressed) vẫn phân biệt đúng sau khi retrain.

---

## Thay đổi 5: Cập nhật giao diện (Frontend)

### Mô tả
Đồng bộ UI với 10 class mới, loại bỏ 5 nút cảm xúc dư, cập nhật version và changelog.

### Cài đặt kỹ thuật
- **Card "Nhập cảm xúc"** (`.emo-grid`): Bỏ 5 `.emo-btn` của class đã loại (`vui_nhon`, `phieu_luu`, `bi_an`, `tu_tin`, `biet_on`) → còn **10 nút** (grid 4 cột: 4+4+2).
- **"Dạy AI"** (`.correct-row`): Bỏ 5 `.correct-btn` tương ứng → còn **10 nút**.
- **Stat pill "Classes"**: Hard-code hiển thị `10` cảm xúc.
- **Version tag**: `v3.1` → **`v3.5`**.
- **Modal "About"**: `"Phiên bản v3.1 — 15 cảm xúc..."` → **`"Phiên bản v3.5 — 10 cảm xúc · RLUF Bandit · Knowledge Graph · Self-Attention · Jamendo"`**.
- **Modal "Changelog"**: Thêm entry v3.5 lên đầu (đánh dấu "hiện tại") — *"Rút gọn từ 15 → 10 cảm xúc đơn giản... Áp dụng mô hình Valence-Arousal (GEMS/Circumplex)..."*; bỏ badge "hiện tại" khỏi v3.1.

### Kết quả kiểm thử
- Frontend (`anhtaictv.me`) → 200 OK.
- Đếm `.emo-btn` / `.correct-btn` → đúng **10/10**.
- Version tag hiển thị đúng **"v3.5"**.
- Knowledge Graph Canvas vẫn render đúng — code lặp động theo `scores`, tự scale xuống 10 bong bóng.

---

## Bảng so sánh tổng quan v3.1 vs v3.5

| Khía cạnh | v3.1 | v3.5 |
|---|---|---|
| Số class cảm xúc | 15 | **10** (rút gọn theo Valence-Arousal) |
| Tên class nội bộ | Tiếng Việt (`vui_ve`, `buon_ba`...) | **Tiếng Anh** (`happy`, `sad`...) |
| Mô hình cảm xúc | Tự thiết kế | **GEMS + Circumplex of Affect (Russell)** |
| Valence/Arousal trong API | Không có | **Có** (mỗi emotion kèm toạ độ 2D) |
| `EMOTION_META` | `vi`, `emoji` | **`vi`, `emoji`, `valence`, `arousal`** |
| `VOCAB_SIZE` | 805 | **Thay đổi** (bỏ vocab 5 class, gộp vui_nhon vào happy) |
| `SEED_DATA` | 144 câu | **92 câu** (-52, bỏ seed 5 class đã loại) |
| Kiến trúc MLP | `W2 (64,15)`, `b2 (15,)` | **`W2 (64,10)`, `b2 (10,)`** (retrain từ đầu) |
| `EMOTION_TAGS` (Jamendo) | 15 pool tag (key tiếng Việt) | **10 pool tag (key tiếng Anh)** |
| Frontend `.emo-btn` / `.correct-btn` | 15 / 15 | **10 / 10** |
| Replay buffer khi deploy | 500 mẫu (label 0-14) | **Reset** (label 0-9) |
| Version trên UI | "v3.1" | **"v3.5"** + entry Changelog mới |
| Thư viện ngoài thêm vào | — | **Không** (100% NumPy thuần) |

---

## Lý do nhảy thẳng từ v3.1 lên v3.5

Phiên bản được đánh số **v3.5** (thay vì v3.2) để phản ánh mức độ thay đổi cấu trúc đáng kể:
- Thay đổi số class đầu ra (`output_size`) của mô hình → phá vỡ tính tương thích với `weights.npz` cũ.
- Thay đổi tên key trong toàn bộ API, `LEXICON`, `EMOTION_TAGS`, `EMOTION_META` — không backward-compatible.
- Áp dụng chuẩn khoa học mới (GEMS/Circumplex) → thay đổi quan điểm thiết kế hệ thống, không chỉ là bổ sung tính năng nhỏ.

> **Định hướng tiếp theo:** Với 10 class chuẩn hoá theo Valence-Arousal, có thể xây dựng bộ lọc nhạc theo toạ độ 2D (ví dụ: thanh trượt Valence-Arousal trên UI) và cải thiện `_audio_to_emotion()` dựa trên BPM/centroid ánh xạ chính xác hơn theo mô hình Circumplex.
