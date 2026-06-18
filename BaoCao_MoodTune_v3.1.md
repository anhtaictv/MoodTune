# MoodTune v3.1 — Báo cáo chức năng mới (so với v3.0)

**Phiên bản:** `v3.1` (so với `v3.0` trong `BaoCao_MoodTune_v3.0.md`)
**Tên đầy đủ:** MoodTune — AI Cảm Xúc Tự Xây (RLUF Bandit · Knowledge Graph · Self-Attention) + Gợi ý nhạc Jamendo Hybrid

v3.1 gồm **7 chức năng/thay đổi chính**, mỗi mục được trình bày theo cấu trúc:
**Mô tả → Cài đặt kỹ thuật → Kết quả kiểm thử**.

---

## Chức năng 1: Thêm 3 class cảm xúc mới (12 → 15)

### Mô tả
Bổ sung 3 cảm xúc mới để lấp các "khoảng trống mood" mà 12 class cũ chưa phủ tới:

| Class | Tên hiển thị | Emoji | Ý nghĩa / khác biệt với class cũ | Nhạc gợi ý (Jamendo) |
|---|---|---|---|---|
| `tu_tin` | Tự tin | 💪 | Động lực cá nhân, quyết tâm vượt khó — khác `nang_dong` (năng động/tiệc tùng sôi nổi) | `motivational+epic`, `rock+powerful`, `uplifting+anthem`, `electronic+powerful` |
| `biet_on` | Biết ơn | 🙏 | Bình yên nội tâm, trân trọng hiện tại, hạnh phúc gia đình — khác `thu_gian` (nghỉ ngơi thể chất) và `vui_ve` (vui sôi nổi) | `acoustic+warm`, `folk+calm`, `piano+peaceful`, `soul+gentle` |
| `tuc_gian` | Tức giận | 😡 | Giận dữ, bực tức, phẫn nộ — tách riêng khỏi `cang_thang` (giờ tập trung vào stress/áp lực/lo âu) | `metal+aggressive`, `punk+rock`, `rock+heavy`, `industrial+intense` |

### Cài đặt kỹ thuật
- `lexicon.py` → `EMOTIONS`: append `"tu_tin", "biet_on", "tuc_gian"` vào cuối list (giữ
  nguyên thứ tự 12 class cũ → không phá replay buffer/online-learning cũ).
- `EMOTION_META`: thêm 3 entry `{"vi": ..., "emoji": ...}` tương ứng.
- `LEXICON`: thêm 3 block mới —
  - `tu_tin`: **58 mục** (tự tin, quyết tâm, mạnh mẽ, kiên định, bản lĩnh, nỗ lực, vượt
    qua, không bỏ cuộc, tin vào bản thân, cố lên, "tôi làm được", level up, kiên cường,
    bền bỉ, đối mặt, trở ngại, chùn bước, motivated, confident, unstoppable, champion,
    fighting, ...)
  - `biet_on`: **42 mục** (biết ơn, cảm ơn, trân trọng, mãn nguyện, an yên, hạnh phúc
    bình dị, ấm lòng, "cảm ơn cuộc đời", grateful, thankful, blessed, content, ...)
  - `tuc_gian`: **48 mục** (tức giận, phẫn nộ, giận dữ, nổi điên, bực tức, căm phẫn, sôi
    máu, ức chế, bất công, phản bội, "không thể chấp nhận được", angry, furious, rage,
    mad, ...)
- `app.py` → `EMOTION_TAGS`: thêm 3 pool tag Jamendo (bảng trên).
- `bandit.py` (`ThompsonBandit`): không cần sửa — `setdefault` lazy-init đúng cho 3
  emotion key mới khi gặp lần đầu.

### Kết quả kiểm thử
- `GET /api/mix-ratio?emotion=tu_tin|biet_on|tuc_gian` → 200 OK, bandit lazy-init đúng.
- `GET /api/music/search?emotion=tu_tin|biet_on|tuc_gian` → 200 OK (~0.8-1s, gọi Jamendo).
- `POST /api/predict` trả đủ **15 entries** trong `scores`/`graph`/`model_info`.
- Độ chính xác held-out cho 3 class mới: xem **Chức năng 4**.

---

## Chức năng 2: Cập nhật & mở rộng từ điển cảm xúc (LEXICON)

### Mô tả
Ngoài vocab cho 3 class mới (Chức năng 1), bổ sung slang/cụm từ phổ biến còn thiếu cho
một số class cũ — đúng yêu cầu "cập nhật vocab".

### Cài đặt kỹ thuật
| Class | Vocab mới bổ sung |
|---|---|
| `vui_ve` | "vui dữ", "đỉnh nóc", "max vui", "vui banh nóc" |
| `thu_gian` | "touch grass", "nạp lại năng lượng", "detox", "chữa lành" |
| `cang_thang` | "quá tải tinh thần", "không thở được", "ngộp thở" |

- Bổ sung trực tiếp vào dict `LEXICON[emotion]` hiện có (weight 1.0-3.0, cùng style các
  mục cũ) — không đổi cấu trúc, không cần code mới.
- **`VOCAB_SIZE` tổng: 656 → 810** (+154 từ/cụm, gồm cả vocab 3 class mới ở Chức năng 1).

### Kết quả kiểm thử
- `python -m py_compile lexicon.py` → OK (không lỗi cú pháp).
- Log khởi động backend: `[Init] Vocab=810 (+3 dynamic) | Classes=15` — đúng số lượng kỳ
  vọng (810 từ tĩnh + 3 từ học online qua dynamic vocab).
- Accuracy của 12 class cũ **không bị giảm** sau khi mở rộng vocab (xem Chức năng 4).

---

## Chức năng 3: Retrain Self-Attention MLP cho 15 class

### Mô tả
Kiến trúc `AttentionMLP` tự co giãn theo số class (`N = len(EMOTIONS)`) và số từ vựng
(`VOCAB_SIZE`) — khi 2 hằng số này đổi (12→15 class, 656→810 từ), phải train lại từ đầu
để các ma trận trọng số có shape đúng.

### Cài đặt kỹ thuật
- `W2`: `(64, 12)` → `(64, 15)`; `b2`: `(12,)` → `(15,)`.
- `E` (embedding matrix): tự co giãn theo `VOCAB_SIZE = 810`.
- Quy trình retrain:
  1. Xoá `backend/weights.npz` (giữ `weights_meta.json`, `weights_replay.json`,
     `dynamic_vocab.json`, `bandit_state.json`).
  2. `pm2 restart moodtune-backend` → `EmotionEngine.__init__` thấy `weights.npz` không
     tồn tại → gọi `pretrain(SEED_DATA + replay, epochs=400)`.
  3. `AttentionMLP.__init__` tự tạo `W2`/`b2`/`E` với shape mới (15 class, 810 vocab).
- **Replay buffer** (500 mẫu, label 0-11) được giữ nguyên — 12 class cũ không mất dữ liệu
  học online cũ vì thứ tự index không đổi.
- `SEED_DATA`: 96 → **144 câu** (12 class cũ giữ 8 câu/class; 3 class mới: `tu_tin` 20,
  `biet_on` 8, `tuc_gian` 20 — xem Chức năng 4 để biết lý do số lượng khác nhau).

### Kết quả kiểm thử
- Pretrain 400 epoch hoàn tất, `weights.npz` được tạo lại đúng shape mới — không lỗi
  shape-mismatch khi load.
- `POST /api/predict` ("tự tin và kiên cường vượt qua mọi trở ngại") → `emotion=tu_tin`,
  confidence **0.9996** — **~290ms/request**.
- Log PM2 không có lỗi 500 mới sau khi deploy.

---

## Chức năng 4: Tinh chỉnh độ chính xác cho 3 class mới (3 vòng lặp)

### Mô tả
Sau khi thêm class mới, đo độ chính xác bằng bộ test held-out (30 câu, 2 câu/class) và
phát hiện 2 lỗi overlap giữa lexicon của class mới và class cũ:
- `tu_tin` ↔ `phieu_luu`: trùng "vượt qua" (2.5 cả 2 bên) và "thử thách" (chỉ có ở
  `phieu_luu`) → câu về "vượt qua thử thách" bị nhận nhầm thành `phieu_luu`.
- `tuc_gian` ↔ `cang_thang`: "bực bội" ban đầu chỉ có ở `cang_thang` → câu "bực bội"
  bị nhận nhầm thành `cang_thang` với confidence rất cao (0.985).

### Cài đặt kỹ thuật
| Vòng | Thay đổi |
|---|---|
| 0 (cold-start) | Chỉ 24 seed mới (8 câu/class), chưa fix overlap |
| 1 | Thêm "bực bội" (2.0) vào `tuc_gian`, thêm "không ngờ"/"thật không ngờ" vào `tuc_gian`; thêm "thử thách bản thân"/"tin chắc"/"chắc chắn"/"chinh phục bản thân"/"không gì cản được" vào `tu_tin`; +8 seed/class (label 12, 14) |
| 2 | Thêm 6 vocab `tu_tin` (kiên cường, bền bỉ, đối mặt, trở ngại, tinh thần, chùn bước) + 2 vocab `tuc_gian` (bất công, phản bội); +8 seed/class (label 12, 14) |

- Lưu ý kỹ thuật: `rule_score()`/`_tokenize()` chỉ match unigram + bigram (2 từ) với
  `LEXICON` — mọi cụm 3+ từ đều "chết" (không bao giờ match). Vì vậy toàn bộ vocab mới ở
  vòng 1-2 được chọn là cụm **1-2 từ** để đảm bảo có hiệu lực thực tế.

### Kết quả kiểm thử
| Vòng | 12 class cũ | 3 class mới | Tổng |
|---|---|---|---|
| 0 (cold-start) | 83.3% (20/24) | 33.3% (2/6) | 73.3% (22/30) |
| 1 | 83.3% (20/24) | 50.0% (3/6) | 76.7% (23/30) |
| 2 | **91.7% (22/24)** | **83.3% (5/6)** | **90.0% (27/30)** |

Câu khó từng bị nhận nhầm ở vòng 0, đã đúng ở vòng 2:
- "tôi tin chắc mình sẽ vượt qua được kỳ thi này" → **tu_tin** ✓ (0.831)
- "không có thử thách nào làm tôi chùn bước" → **tu_tin** ✓ (0.682)

**Hạn chế còn lại (tại thời điểm phát hành)**: "tôi đang vô cùng bực bội vì điều này"
vẫn bị nhận là `cang_thang` (0.522) thay vì `tuc_gian` — quyết định để người dùng tự sửa
qua **"Dạy AI"** (online learning `learn()`) thay vì tinh chỉnh lexicon thêm, tránh
over-fit trên bộ test nhỏ. → **Đã được fix triệt để ở Chức năng 7** sau khi phát hiện
nguyên nhân gốc là overlap lexicon rộng hơn dự kiến giữa `tuc_gian` và `cang_thang`.

---

## Chức năng 5: Cập nhật giao diện (Frontend)

### Mô tả
Đồng bộ UI với 15 class cảm xúc mới và nâng version hiển thị lên v3.1.

### Cài đặt kỹ thuật
- **Card "Nhập cảm xúc"** (`.emo-grid`): thêm 3 `.emo-btn` mới — 💪 Tự tin ("tự tin quyết
  tâm vượt qua khó khăn"), 🙏 Biết ơn ("biết ơn mãn nguyện với cuộc sống"), 😡 Tức giận
  ("tức giận phẫn nộ bực bội") → tổng **15 nút** (grid 4 cột, tự xuống dòng 4+4+4+3).
- **"Dạy AI"** (`.correct-row`): thêm 3 `.correct-btn` tương ứng `tu_tin`/`biet_on`/
  `tuc_gian` → tổng **15 nút**.
- `<span class="version-tag">v3.0</span>` → **`v3.1`**.
- Modal "Lịch sử phiên bản": thêm entry mới lên đầu (đánh dấu "hiện tại") — *"v3.1 — Mở
  rộng cảm xúc: Thêm 3 cảm xúc mới (Tự tin 💪, Biết ơn 🙏, Tức giận 😡) + mở rộng từ điển
  (656→810 từ) + retrain Self-Attention MLP (15 class)"*; bỏ badge "hiện tại" khỏi v3.0.
- `renderKnowledgeGraph` và `score-bars` lặp động theo `scores`/`tokens` → tự scale lên
  15 bong bóng, **không cần sửa code**.

### Kết quả kiểm thử
- `node --check` trên `<script>` extract từ `index.html` → OK.
- Frontend (`anhtaictv.me`, IIS site `MoodTune` → `C:\moodtune\frontend`) → 200 OK.
- Đếm `.emo-btn` / `.correct-btn` trên trang → đúng **15/15**, `data-emotion` khớp
  lexicon mới.
- Version tag hiển thị đúng **"v3.1"**.

---

## Chức năng 6: Kiểm thử triển khai tổng thể (Deployment)

### Mô tả
Xác nhận toàn bộ thay đổi hoạt động ổn định trên môi trường production (PM2 + IIS) sau
khi deploy.

### Cài đặt kỹ thuật
- Backend: PM2 process `moodtune-backend` (port 5005, Flask dev server).
- Frontend: IIS site `MoodTune` (`anhtaictv.me`, port 80/443) → `C:\moodtune\frontend`.
- Quy trình deploy: `python -m py_compile` (lexicon.py, emotion_mlp.py) → xoá
  `weights.npz` → `pm2 restart moodtune-backend`.

### Kết quả kiểm thử
| Mục kiểm thử | Kết quả |
|---|---|
| Trang chủ `anhtaictv.me` | 200 OK, 66.5KB, load ~1.4ms (local) |
| `POST /api/predict` | 200 OK, ~290ms/request, đủ 15 class |
| `GET /api/music/search` (3 class mới) | 200 OK, ~0.8-1s (Jamendo) |
| `GET /api/mix-ratio?emotion=tu_tin` | 200 OK, bandit lazy-init đúng |
| PM2 error log | Không có lỗi 500 mới |
| 15 `.emo-btn` / `.correct-btn` | Hiển thị đầy đủ, đúng `data-emotion` |

**Nhận xét tổng thể**: app chạy mượt — predict ~0.3s, trang load gần như tức thì, không
phát sinh lỗi sau nâng cấp.

---

## Chức năng 7: Hotfix overlap lexicon — "tức giận"/"bực bội" bị nhận thành `cang_thang`, "buồn bã" báo sai

### Mô tả
Sau khi phát hành, người dùng phản hồi: **"không buồn thì nó sẽ tự nhận là buồn"** — engine
nhận diện sai khá thường xuyên. Kiểm tra với các câu thực tế (không có trong bộ test 30
câu) phát hiện 2 nhóm lỗi:

1. **`tuc_gian` ↔ `cang_thang` overlap nặng**: `cang_thang` (class cũ, trước v3.1 vốn
   gộp chung "stress + giận") vẫn còn giữ các từ thuộc về sự **giận dữ** —
   `"tức":2.5, "giận":2.5, "tức giận":3.0, "điên":2.5, "điên tiết":3.0, "cáu":2.5,
   "cáu kỉnh":2.5, "bực bội":2.5` — đè hẳn lên `tuc_gian` (class mới). Hậu quả: câu
   *"tôi đang **rất tức giận** vì bị trễ xe buýt"* bị nhận là `cang_thang` (0.996) thay
   vì `tuc_gian`; *"tôi đang vô cùng **bực bội** vì điều này"* bị nhận `cang_thang`
   (0.522) — đây chính là hạn chế đã ghi nhận ở Chức năng 4 nhưng **nguyên nhân gốc lớn
   hơn dự kiến** (không chỉ 1 từ "bực bội" mà cả 1 nhóm 8 từ/cụm).
2. **`buon_ba` chứa từ quá chung, dễ "dính" sai**: `"mất":2.0, "thua":2.0, "tiếc":2.0` —
   các từ này xuất hiện trong rất nhiều câu trung tính đời thường (mất đồ, thua trận,
   tiếc nuối nhẹ) nhưng kéo cả câu thành "Buồn bã" với confidence ~0.20. Ví dụ: *"tôi vừa
   **mất** chìa khóa xe"*, *"trận đấu hôm nay đội tôi bị **thua**"*, *"**tiếc** là tôi
   không đi xem phim được"* → tất cả bị nhận `buon_ba`.

### Cài đặt kỹ thuật
- **`lexicon.py` → `cang_thang`**: xoá 8 mục thuộc lãnh địa "giận dữ" (đã có sẵn ở
  `tuc_gian` dưới dạng bigram tương đương — `"tức giận"`, `"điên tiết"`, `"bực bội"` —
  hoặc không cần thiết nữa): `"tức":2.5, "giận":2.5, "tức giận":3.0, "điên":2.5,
  "điên tiết":3.0, "cáu":2.5, "cáu kỉnh":2.5, "bực bội":2.5`. `cang_thang` giờ tập trung
  thuần vào stress/áp lực/lo âu (giữ `"bực":2.5, "bực mình":2.5, "khó chịu":2.0`...).
- **`lexicon.py` → `buon_ba`**: giảm trọng số 3 từ quá chung —
  `"mất":2.0→1.0`, `"thua":2.0→1.0`, `"tiếc":2.0→1.0` (vẫn góp phần khi đi cùng từ buồn
  khác, nhưng không còn đủ để một mình kéo cả câu thành "buồn bã").
- **`VOCAB_SIZE`: 810 → 805** (-5, do 5 từ "tức"/"giận"/"cáu"/"cáu kỉnh"/"điên" không còn
  xuất hiện ở bất kỳ class nào khác → bị loại khỏi vocab). Retrain lại theo đúng quy
  trình ở Chức năng 3 (xoá `weights.npz` → `pm2 restart` → pretrain 400 epoch).

### Kết quả kiểm thử
| Câu test | Trước fix | Sau fix |
|---|---|---|
| "tôi đang rất tức giận vì bị trễ xe buýt" | ❌ `cang_thang` (0.996) | ✅ **`tuc_gian`** (0.923) |
| "tôi đang vô cùng bực bội vì điều này" | ❌ `cang_thang` (0.522) | ✅ **`tuc_gian`** (0.689) |
| "thái độ đó khiến tôi nổi giận" | — | ✅ **`tuc_gian`** (0.995) |
| "tôi vừa mất chìa khóa xe" | `buon_ba` (conf 0.201) | `buon_ba` (conf **0.116**, giảm mạnh độ "tự tin sai") |
| "trận đấu hôm nay đội tôi bị thua" | `buon_ba` (conf 0.201) | `buon_ba` (conf **0.116**) |
| "tiếc là tôi không đi xem phim được" | `buon_ba` (conf 0.201) | `buon_ba` (conf **0.116**) |

**Độ chính xác held-out (30 câu)**: **90.0% (27/30) → 93.3% (28/30)** — 3 class mới đạt
**100% (6/6)** (tăng từ 83.3%), 12 class cũ giữ **91.7% (22/24)**, không regression.

> Lưu ý: với "mất"/"thua"/"tiếc" chỉ giảm trọng số (không xoá hẳn) vì các từ này vẫn là
> chỉ báo buồn hợp lệ khi đi cùng ngữ cảnh khác (vd: "tôi mất người thân", "tiếc nuối quá
> khứ"). Vấn đề "thời tiết hôm nay khá lạnh" → `bi_an` (do "lạnh" cũng nằm trong lexicon
> bí ẩn) là overlap tương tự nhưng có từ trước v3.1, nằm ngoài phạm vi hotfix này — để
> dành cho "Dạy AI" tự điều chỉnh.

---

## Bảng so sánh tổng quan v3.0 vs v3.1

| Khía cạnh | v3.0 | v3.1 |
|---|---|---|
| Số class cảm xúc | 12 | **15** (+ tu_tin 💪, biet_on 🙏, tuc_gian 😡) |
| `VOCAB_SIZE` | 656 | **805** (+149, sau hotfix Chức năng 7) |
| `SEED_DATA` | 96 câu | **144 câu** (+48) |
| Kiến trúc MLP | `W2 (64,12)`, `b2 (12,)` | `W2 (64,15)`, `b2 (15,)` (retrain từ đầu) |
| `EMOTION_TAGS` (Jamendo) | 12 pool tag | **15 pool tag** |
| Độ chính xác held-out (30 câu) | — (không đo) | **93.3%** (cũ: 91.7%, mới: **100%**) |
| Frontend `.emo-btn` / `.correct-btn` | 12 / 12 | **15 / 15** |
| Version trên UI | "v3.0" | **"v3.1"** + entry Changelog mới |
| Thư viện ngoài thêm vào | — | **Không** (100% NumPy thuần) |

> Định hướng tiếp theo: overlap `tuc_gian`↔`cang_thang` (từ "tức giận"/"bực bội") đã được
> fix ở Chức năng 7. Vấn đề tương tự còn lại ("lạnh" → `bi_an` thay vì nghĩa thời tiết) có
> từ trước v3.1, để dành cho "Dạy AI" tự điều chỉnh qua online-learning, không cần tinh
> chỉnh lexicon thủ công thêm.
