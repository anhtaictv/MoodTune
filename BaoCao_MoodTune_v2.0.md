# MoodTune v2.0 — Báo cáo nâng cấp & So sánh với v1.1

**Phiên bản:** `v2.0` (so với `v1.1` trong `BaoCao_MoodTune_v1.1.md`)
**Tên đầy đủ:** MoodTune — AI Cảm Xúc Tự Xây (Self-Attention) + Gợi ý nhạc Jamendo (Hybrid Online/Offline)

---

## 1. Tóm tắt thay đổi chính

So với v1.1 (Hybrid Bag-of-Words + MLP, chỉ nghe nhạc online qua Jamendo), v2.0 nâng cấp
**3/5 ý tưởng** trong đề xuất nâng cấp (`nangcap.txt` / `capnhat_utf8.txt`) đã được code hóa thực tế:

| # | Ý tưởng đề xuất | Trạng thái trong code hiện tại |
|---|---|---|
| 1 | Dynamic Weight Expansion (mở rộng từ điển/ma trận runtime) | ✅ Đã làm — `dynamic_vocab.json`, `expand_weights()`, `expand_vocab()` |
| 2 | Audio Feature Multimodal Engine (BPM/Spectral/MFCC) | ✅ Đã làm (bản lite) — `audio_features.py`, soft re-rank kết quả search |
| 3 | Self-Attention Layer thuần NumPy + Embedding | ✅ Đã làm — `AttentionMLP` trong `emotion_mlp.py` |
| 4 | Hybrid Online/Offline (Kho nhạc Local) | ✅ Đã làm — tab "Kho nhạc Local" + `/api/predict/batch` |
| 4b | Leaky ReLU + L2 động theo feedback | ❌ Chưa làm — vẫn ReLU thường + L2 cố định `1e-4` (giữ như v1.1) |
| 5 | Đóng gói Docker / PyInstaller / Electron | ❌ Chưa làm — chưa thấy `Dockerfile`/`.spec` trong project |

---

## 2. Thay đổi trong AI Engine (`emotion_mlp.py`)

### 2.1. Kiến trúc mạng — thay đổi LỚN NHẤT

| | v1.1 | v2.0 |
|---|---|---|
| Input | Bag-of-Words 653 chiều (nhị phân/tần suất, **mất thứ tự từ**) | Sequence token-id (`to_token_ids`, giữ **đúng thứ tự từ trong câu**, max 24 token) |
| Lớp nhúng | Không có | **Embedding Matrix `E` (VOCAB_SIZE × d=32)**, tra cứu vector từng từ |
| Lớp giữa | Dense(653→64, ReLU) | **Self-Attention(Q,K,V)** → mean-pool → Dense(32→64, ReLU) |
| Công thức attention | — | `Attention(Q,K,V) = Softmax(QKᵀ/√d) · V` — forward + backward thủ công bằng NumPy |
| Output | Dense(64→12, Softmax) | Dense(64→12, Softmax) *(giữ nguyên)* |
| Tối ưu | SGD + Momentum 0.9 + L2 1e-4 + He init | **Giữ nguyên** SGD + Momentum 0.9 + L2 1e-4 + He init, **thêm Gradient Clipping (norm ≤ 5.0)** vì kiến trúc attention dễ nổ gradient hơn |
| Lưu trọng số | `weights.npz` chứa `W1,b1,W2,b2` | `weights.npz` chứa `E, Wq, Wk, Wv, W1, b1, W2, b2` — **format mới, không tương thích ngược** với weights v1.1 (có check `"E" not in d.files` → tự pretrain lại) |

**Ý nghĩa:** v1.1 không phân biệt được câu đảo ngữ/phủ định phức tạp do BoW làm mất trật tự từ
(ví dụ "Không thể nào vui nổi" dễ bị hiểu lẫn với "vui"). v2.0 giữ trật tự từ qua sequence
token-id + Self-Attention nên mô hình tự học được mối quan hệ giữa từ phủ định và từ cảm xúc
đứng gần nó, đồng thời vẫn giữ triết lý "100% NumPy thuần, không framework ML".

### 2.2. Dynamic Vocab / Weight Expansion (Ý tưởng 1) — MỚI hoàn toàn

- File mới `dynamic_vocab.json` lưu các từ mới phát hiện khi người dùng "Dạy AI" (hiện có
  3 từ: `ko`, `zdui`, `mỏi`).
- Hàm mới `find_oov_words()` quét câu, tìm từ ≥2 ký tự chưa có trong `VOCAB_IDX`.
- Hàm mới `add_vocab_words()` thêm từ vào `VOCAB`/`VOCAB_IDX` + ghi lại `dynamic_vocab.json`.
- Hàm mới `expand_weights(E, k, d)` dùng `np.vstack()` chèn thêm `k` dòng vào ma trận
  Embedding `E`, khởi tạo **He Init × 0.01** để không phá vỡ đặc trưng đã học.
- `AttentionMLP.expand_vocab(k)` gọi hàm trên + reset momentum của `E` (vì shape đổi).
- Toàn bộ xảy ra **ngay trong `engine.learn()`**, không cần restart server.
- Kết quả thực tế: vocab đã tăng từ **653 → 656** (theo `weights_meta.json`), feedback đã
  học **58 lần**, alpha hiện tại ≈ **0.554** (so với 0.85 ban đầu — hệ thống đang tin MLP
  nhiều hơn rõ rệt so với lúc mới khởi động).

### 2.3. Những phần KHÔNG đổi (giữ từ v1.1)

- Rule Scorer (lexicon 653 từ/cụm, 12 nhóm cảm xúc, bigram ×1.5, phủ định ×-0.6, softmax).
- Hybrid blend: `final = alpha*rule + (1-alpha)*mlp`, công thức alpha `max(0.35, 0.85 - 0.5*fc/(fc+40))`.
- Online learning 30 bước + Experience Replay (≤500 mẫu) chống catastrophic forgetting.
- Lưu trạng thái vào `weights.npz` / `weights_meta.json` / `weights_replay.json`.
- **Leaky ReLU và L2 động theo feedback** (đề xuất trong `nangcap.txt`) **chưa được áp dụng** —
  `backward()` vẫn dùng ReLU thường (`z1 > 0`) và `l2 = 1e-4` cố định.

---

## 3. Audio Feature Engine — `audio_features.py` (Ý tưởng 2, bản lite) — MỚI

File hoàn toàn mới, chạy **nền (background thread)**, không chặn `/api/music/search`:

- Dùng `librosa` để phân tích 30 giây đầu của track Jamendo: **BPM** (tempo), **Spectral
  Centroid** (độ sáng âm thanh), **MFCC** (13 hệ số).
- Heuristic `_audio_to_emotion(bpm, centroid)` map đặc trưng âm thanh → 1 trong 12 nhãn cảm
  xúc (ví dụ BPM≥120 & centroid≥2500 → "năng động"; BPM<70 & centroid<1500 → "buồn bã"...).
- Cache kết quả vào `audio_cache.json`, tối đa 5 track/lần gọi (`MAX_BG_PER_CALL=5`).
- **Khác với đề xuất gốc**: đề xuất ban đầu muốn ghép vector audio vào thẳng input của MLP
  (concatenate với vector text). Bản thực tế làm theo hướng nhẹ hơn — dùng kết quả phân tích
  audio để **soft re-rank** danh sách bài hát trả về từ Jamendo (ưu tiên bài có
  `audio_emotion` khớp với cảm xúc người dùng đang tìm), không sửa input/shape của mạng MLP.
- **Graceful fallback**: nếu `librosa` không cài được (`AUDIO_ENABLED=False`), mọi hàm
  thành no-op, hệ thống vẫn chạy bình thường như v1.1.

---

## 4. Hybrid Online/Offline — "Kho nhạc Local" (Ý tưởng 4) — MỚI

### 4.1. Frontend (`index.html`)
- Tab mới **"📁 Kho nhạc Local (Hybrid Online/Offline)"**.
- Nút "Chọn thư mục nhạc" gọi `window.showDirectoryPicker()` (File System Access API) —
  có kiểm tra `'showDirectoryPicker' in window`, fallback nếu trình duyệt không hỗ trợ.
- `indexLocalFiles()`: gửi danh sách tên file tới `/api/predict/batch` để AI gắn nhãn cảm xúc
  cho từng file dựa trên **tên bài hát**.
- Bản đồ `{filename: {emotion, label, emoji}}` lưu vào `localStorage` (`mt_local_library`),
  không upload nội dung file lên server (privacy-first).
- `getLocalTracksForEmotion(emotion, limit=6)`: lấy tối đa 6 bài local khớp cảm xúc.
- `playLocalTrack()`: phát file cục bộ qua `URL.createObjectURL(file)` (Blob URL), không cần
  server.

### 4.2. Backend — Endpoint mới `/api/predict/batch`
- Nhận `{"items": ["ten_bai_1.mp3", ...]}` (tối đa 200 item/lần).
- Chạy `engine.predict()` cho từng tên file → trả về `emotion/label/emoji` cho mỗi item.
- Log lại số lượng qua `log_event("predict_batch", ...)`.

### 4.3. Hybrid Playlist Mixer
- Khi người dùng phân tích cảm xúc xong, hệ thống lấy đồng thời:
  - 6 bài **Online** từ `/api/music/search` (Jamendo).
  - 6 bài **Offline** từ Kho nhạc Local khớp cảm xúc (qua `getLocalTracksForEmotion`).
- Cả hai trộn vào cùng một danh sách phát trên player.

---

## 5. So sánh tổng quan v1.1 vs v2.0

| Khía cạnh | v1.1 | v2.0 |
|---|---|---|
| Hiển thị version trên UI | "v1.1" | **"v2.0"** (`index.html`, logo + modal) |
| Input representation | Bag-of-Words 653-d | Embedding sequence (token-id, giữ thứ tự) |
| Lớp học sâu | MLP đơn giản (Dense→ReLU→Dense→Softmax) | + **Self-Attention(Q,K,V)** trước MLP |
| Mở rộng từ điển runtime | Không có (phải retrain/restart khi thêm từ) | **Có** — `dynamic_vocab.json` + `expand_weights()` |
| Phân tích nhạc | Chỉ dựa vào tag Jamendo (text-only) | + Phân tích **audio thực** (BPM/Centroid/MFCC) để re-rank |
| Nguồn nhạc | Chỉ Online (Jamendo) | **Hybrid**: Online (Jamendo) + Offline (thư mục nhạc local qua File System Access API) |
| Endpoint API | 8 endpoint (health, predict, learn, stats, music/search, recommend, track/event, time-suggestion) | **+1 endpoint mới**: `/api/predict/batch` (9 endpoint) |
| Health response | Không có field `architecture` | Thêm `architecture: {vocab_size, embed_dim, hidden_size, output_size, attention:true}` |
| Format file weights | `W1,b1,W2,b2` | `E,Wq,Wk,Wv,W1,b1,W2,b2` (không tương thích ngược, tự pretrain lại nếu thiếu `E`) |
| Gradient clipping | Không có | Có (norm ≤ 5.0) — cần do attention dễ nổ gradient |
| Leaky ReLU / L2 động | — (đề xuất, chưa áp dụng ở cả 2 bản) | — (vẫn chưa áp dụng) |
| Đóng gói Docker/Desktop | Chưa | Chưa (đề xuất ý tưởng 5, chưa code) |

---

## 6. Lịch sử phiên bản (cập nhật)

- **v2.0** *(hiện tại)* — Self-Attention + Embedding Layer thay Bag-of-Words; Dynamic Vocab/Weight
  Expansion runtime; Audio Feature Engine (librosa, BPM/Centroid/MFCC) soft re-rank kết quả
  Jamendo; Hybrid Online/Offline với tab "Kho nhạc Local" (File System Access API,
  `/api/predict/batch`, Blob URL playback, Hybrid Playlist Mixer).
- **v1.1** — Hiển thị số phiên bản trên giao diện; hoàn thiện đầy đủ các chức năng: phân tích
  cảm xúc Hybrid AI (Bag-of-Words + MLP), gợi ý/nghe nhạc Jamendo, online learning, gợi ý cá
  nhân hoá, gợi ý theo giờ, lịch sử phân tích.
- **v1.0** — Phiên bản nền tảng: Rule + MLP Hybrid Engine, tích hợp Jamendo, feedback & online
  learning cơ bản.

> Định hướng tiếp theo (chưa làm trong v2.0): Leaky ReLU + L2 regularization động theo mức độ
> feedback; đóng gói Docker (Web) và PyInstaller/Electron (Desktop) cho bản chạy biên (Edge AI).
