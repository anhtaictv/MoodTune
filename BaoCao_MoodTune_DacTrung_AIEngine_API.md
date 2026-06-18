# MoodTune — Điểm đặc trưng, AI Engine chi tiết & API Endpoints

**Phiên bản:** v3.1 · **Domain:** https://anhtaictv.me

---

## 1. Điểm đặc trưng của phần mềm

### 1.1. 100% NumPy thuần — không dùng framework Machine Learning
Toàn bộ AI Engine (`emotion_mlp.py`) — bao gồm **Embedding**, **Self-Attention (Q,K,V)**,
**forward** và **backward (backpropagation)**, **optimizer (SGD+Momentum)**, **Leaky
ReLU**, **Adaptive L2 Regularization**, **Gradient Clipping**, **Softmax** — đều được tự
viết bằng `numpy`, **không** dùng TensorFlow/PyTorch/Keras/scikit-learn. Đây là triết lý
xuyên suốt dự án, dùng để loại bỏ các đề xuất nâng cấp vi phạm (ví dụ: face-api.js cho
nhận diện cảm xúc qua khuôn mặt ở v3.0).

### 1.2. Hybrid Model — Rule-based + Self-Attention MLP
Kết quả cuối cùng là **trung bình có trọng số (alpha)** giữa:
- Một bộ **từ điển trọng số tay** (`lexicon.py`, 805 từ/cụm, 15 nhóm cảm xúc) — đảm bảo độ
  chính xác ổn định ngay cả khi MLP chưa học gì.
- Một **mạng nơ-ron tự học** (`AttentionMLP`) — ngày càng chính xác hơn theo thời gian nhờ
  online learning.

### 1.3. Online Learning — AI học ngay từ người dùng
Tính năng **"Dạy AI"**: khi AI đoán sai, người dùng chọn nhãn đúng → model train ngay
**30 bước gradient descent** trên câu đó, không cần restart hay retrain toàn bộ. Có
**Experience Replay** (đệm 500 mẫu) để chống "quên" kiến thức cũ (catastrophic
forgetting).

### 1.4. Dynamic Vocab Expansion — tự mở rộng từ điển khi học
Khi gặp từ mới (OOV) trong câu dạy, hệ thống tự thêm từ vào vocab và **mở rộng ma trận
Embedding** (`np.vstack`) ngay trong runtime — không cần dừng server.

### 1.5. RLUF — Reinforcement Learning từ phản hồi người dùng (Thompson Sampling Bandit)
Mỗi cảm xúc có 2 "bandit" Beta-Bernoulli (100% NumPy, `bandit.py`):
- Học **tỉ lệ trộn nhạc Online (Jamendo) : Local (máy người dùng)** theo Like/Dislike/Bỏ qua.
- Học **bộ tag Jamendo** nào hợp gu người dùng nhất cho từng cảm xúc.

### 1.6. Hybrid Online/Offline — Kho nhạc Local
Người dùng có thể quét thư mục nhạc trên máy (File System Access API), AI tự gắn nhãn
cảm xúc theo **tên file** (`/api/predict/batch`), rồi trộn vào playlist cùng nhạc online —
**không upload nội dung file lên server** (privacy-first).

### 1.7. Audio Feature Engine (bản lite)
Phân tích BPM / Spectral Centroid / MFCC (qua `librosa`, chạy nền) của track Jamendo để
**soft re-rank** kết quả tìm kiếm theo cảm xúc — có **graceful fallback** (tự tắt nếu
thiếu `librosa`, không crash hệ thống).

### 1.8. Trực quan hoá AI — Sơ đồ tri thức cảm xúc (Knowledge Graph)
Vẽ bằng **Canvas 2D thuần** (không thêm D3.js): hiển thị 15 "bong bóng" cảm xúc, các từ
khoá có **trọng số Attention** cao nhất trong câu, và "tia năng lượng" nối từ khoá tới
cảm xúc tương ứng — giúp người dùng thấy AI "đang nghĩ gì".

### 1.9. 15 nhóm cảm xúc, mở rộng linh hoạt
Kiến trúc `W2`, `b2`, `E` tự co giãn theo `N = len(EMOTIONS)` và `VOCAB_SIZE` — thêm class
cảm xúc mới chỉ cần sửa `lexicon.py` + xoá `weights.npz` để pretrain lại, không cần đổi
code kiến trúc.

### 1.10. Single-Page App, Vanilla JS, Responsive/Auto-scale
Toàn bộ frontend là **1 file `index.html`** (HTML/CSS/JS thuần, không React/Vue), dark
theme, CSS dùng `clamp()` để font/kích thước tự co giãn theo màn hình, grid 15 nút cảm
xúc tự chuyển 4→3 cột trên điện thoại.

### 1.11. Triển khai production thật, không chỉ demo local
Frontend host qua **IIS** tại domain riêng (`anhtaictv.me`, HTTP+HTTPS), backend chạy bền
bằng **PM2**, IIS URL Rewrite proxy `/api/*` → Flask port 5005. Toàn bộ trạng thái AI
(weights, vocab, bandit, replay buffer) **persist ra file**, không mất khi restart.

---

## 2. AI Engine chi tiết (`emotion_mlp.py` + `lexicon.py`)

### 2.1. Tổng quan pipeline dự đoán (`EmotionEngine.predict(text)`)

```
text
 ├─► Tầng 1: rule_score(text)   ──┐
 │      (lexicon, bigram, phủ định) ├─► combined = alpha·rule + (1-alpha)·mlp
 └─► Tầng 2: AttentionMLP.predict(text) ──┘
                                          │
                                          ▼
                         emotion, label, emoji, confidence,
                         all_scores, rule_scores, mlp_scores,
                         alpha, graph_tokens (Knowledge Graph)
```

### 2.2. Tầng 1 — Rule-based Scorer (`rule_score`)

- Nguồn dữ liệu: `LEXICON` (dict 15 nhóm × danh sách `{từ/cụm: trọng số 1.0-3.0}`),
  tổng **805 từ/cụm** (vocab tĩnh).
- **Tokenize** câu, quét theo cặp **bigram trước** (cụm 2 từ, ví dụ `"tức giận"`,
  `"ngày xưa"`) — nếu khớp, cộng điểm × **1.5** (boost); sau đó quét **unigram** (từ đơn)
  còn lại.
- **Xử lý phủ định**: tập `NEGATIONS = {"không","chẳng","chả","đâu","chưa","khỏi","ko","k"}`
  — nếu 1 từ trong tập này đứng ngay trước từ cảm xúc, điểm của từ đó bị **đảo dấu/giảm**
  (tránh "không vui" bị tính như "vui").
- Điểm âm sau xử lý phủ định bị **clamp về 0**.
- Tổng điểm theo từng nhóm cảm xúc được đưa qua **Softmax** → `rule_scores` (tổng = 1).
- Nếu câu không khớp bất kỳ từ nào trong lexicon → trả về phân phối **đều** `1/15` cho mỗi
  nhóm (tránh thiên vị ngẫu nhiên).

### 2.3. Tầng 2 — Self-Attention MLP (`AttentionMLP`)

**Kiến trúc:**

| Lớp | Shape / Công thức |
|---|---|
| Tokenize → token-id | `_tokenize(text, max_len=24)` — bigram-trước-unigram, giữ **thứ tự từ trong câu**, OOV → `id=None` |
| Embedding `E` | `(VOCAB_SIZE, 32)` — tra cứu vector 32 chiều cho mỗi token-id |
| Self-Attention | `Q = X·Wq`, `K = X·Wk`, `V = X·Wv` (mỗi ma trận `32×32`)<br>`Attention(Q,K,V) = Softmax(QKᵀ/√32) · V` |
| Mean-pool | Trung bình theo chiều token → vector `32` chiều / câu |
| Dense 1 | `W1 (32×64)`, `b1 (64,)` + **Leaky ReLU**: `a1 = where(z1>0, z1, 0.01·z1)` |
| Dense 2 (Output) | `W2 (64×15)`, `b2 (15,)` + **Softmax** → `mlp_scores` |

**Forward** trả `1/15` đều nếu câu rỗng / toàn OOV (không có token nào hợp lệ).

**Tối ưu (training, `backward` + `EmotionEngine.learn`):**
- **SGD + Momentum (0.9)** cho tất cả tham số (`E, Wq, Wk, Wv, W1, b1, W2, b2`).
- **He Initialization** khi khởi tạo trọng số mới (kể cả khi mở rộng vocab).
- **Gradient Clipping** theo norm ≤ **5.0** — cần thiết vì Self-Attention dễ gây "nổ"
  gradient khi train liên tục với batch nhỏ (online learning).
- **Adaptive L2 Regularization**:
  ```python
  l2 = min(l2_base * 5, l2_base * (1 + feedback_count / 100))   # l2_base = 1e-4
  ```
  → model mới (ít feedback) regularize nhẹ để học nhanh; model học nhiều → regularize
  mạnh hơn (tối đa `5e-4`) để chống overfit vào các mẫu feedback gần nhất.

### 2.4. Hybrid Blend (kết hợp Rule + MLP)

```python
combined = alpha * rule_scores + (1 - alpha) * mlp_scores
alpha = max(0.35, 0.85 - 0.5 * feedback_count / (feedback_count + 40))
```

- `feedback_count = 0` → `alpha = 0.85` (tin Rule gần như tuyệt đối — MLP chưa học gì).
- `feedback_count` tăng dần qua mỗi lần "Dạy AI" → `alpha` giảm dần, **chặn dưới 0.35**
  (Rule luôn còn đóng góp tối thiểu 35% để giữ ổn định, tránh MLP "lệch" hoàn toàn theo
  vài mẫu gần nhất).
- **Trạng thái thực tế hiện tại**: `feedback_count = 59` → `alpha ≈ 0.552`,
  `l2 ≈ 0.000159`, `vocab_size = 805` (+3 từ học online = 808).

### 2.5. Online Learning chi tiết (`EmotionEngine.learn(text, correct_emotion, steps=30)`)

1. **Phát hiện từ mới (OOV)**: `find_oov_words(text)` quét các từ ≥2 ký tự chưa có trong
   `VOCAB_IDX`.
2. **Mở rộng vocab động**: `add_vocab_words()` thêm từ mới vào `VOCAB`/`VOCAB_IDX`, ghi
   lại `dynamic_vocab.json`; `expand_weights(E, k, 32)` dùng `np.vstack()` chèn `k` dòng
   mới vào `E` (He Init × 0.01) + reset momentum của `E` (vì shape đổi).
3. **Training loop (30 bước)**:
   - Mỗi bước: forward + backward trên câu mới với nhãn đúng (`correct_emotion`).
   - **Mỗi 2 bước**, train xen kẽ thêm 1 mẫu **ngẫu nhiên từ Experience Replay buffer**
     (chống catastrophic forgetting).
4. **Cập nhật replay buffer**: thêm `(text, label)` vào `weights_replay.json`, giữ tối đa
   **500 mẫu** gần nhất (bao gồm cả `SEED_DATA` ban đầu, 144 câu).
5. **Cập nhật trạng thái**: `feedback_count += 1` → tính lại `alpha` và `l2` (Adaptive L2)
   → lưu `weights.npz`, `weights_meta.json`.

### 2.6. Knowledge Graph — `graph_tokens` (cho Canvas trực quan hoá)

Với mỗi token có `tid is not None` trong câu:
- `attention = attn_received[i] / max(attn_received)` — `attn_received` = tổng cột ma
  trận attention `self.mlp._attn` (token nào được các token khác "chú ý" đến nhiều nhất),
  chuẩn hoá về `[0, 1]`.
- `emotions = {emo: LEXICON[emo][token] for emo in EMOTIONS if token trong LEXICON[emo]}`
  — các nhóm cảm xúc mà từ này có mặt trong lexicon, kèm trọng số.

Kết quả mỗi token: `{"text": "...", "attention": 0.0-1.0, "emotions": {"cang_thang": 3.0, ...}}`
→ frontend dùng để vẽ 8 từ khoá attention cao nhất + tia năng lượng nối tới các "bong
bóng" cảm xúc tương ứng.

### 2.7. 15 nhóm cảm xúc (`EMOTIONS` trong `lexicon.py`)

`vui_ve, buon_ba, lang_man, nang_dong, thu_gian, co_don, cang_thang, tap_trung,
hoai_niem, phieu_luu, bi_an, vui_nhon` (12 nhóm gốc) + **`tu_tin, biet_on, tuc_gian`**
(3 nhóm mới ở v3.1, label index 12-14).

### 2.8. Persistence của AI Engine

| File | Nội dung |
|---|---|
| `weights.npz` | `E, Wq, Wk, Wv, W1, b1, W2, b2` |
| `weights_meta.json` | `feedback_count, alpha, l2, arch, embed_dim, vocab_size` |
| `weights_replay.json` | Experience Replay buffer (≤500 mẫu) |
| `dynamic_vocab.json` | Từ mới học online (mở rộng `E` runtime) |

---

## 3. 10 API Endpoints (Flask, `app.py`)

| # | Method | Endpoint | Mục đích / Tham số chính | Trả về (response) |
|---|---|---|---|---|
| 1 | GET | `/api/health` | Kiểm tra trạng thái server + thông tin kiến trúc model | `status, model, feedback_count, alpha, emotions, architecture{vocab_size, embed_dim, hidden_size, output_size, attention, activation, l2}` |
| 2 | POST | `/api/predict` | **Phân tích cảm xúc** từ văn bản. Body: `{"text": "..."}` | `emotion, label, emoji, confidence, scores[] (15 nhóm), graph[] (Knowledge Graph), model_info{alpha_rule, alpha_mlp, feedback_count}` |
| 3 | POST | `/api/learn` | **"Dạy AI"** — online learning. Body: `{"text": "...", "correct_emotion": "..."}` | `status, feedback_count, new_alpha, message` |
| 4 | POST | `/api/predict/batch` | Gắn nhãn cảm xúc cho **danh sách tên file nhạc local** (Kho nhạc Local). Body: `{"items": [...]}` (≤200) | `results[] = [{name, emotion, label, emoji}, ...]` |
| 5 | GET | `/api/stats` | Thống kê tổng lượt predict/feedback + phân bố 15 cảm xúc (đọc `feedback_log.jsonl`) | `total_predicts, total_feedback, emotion_counts{}, model_alpha, feedback_count` |
| 6 | GET | `/api/music/search` | **Tìm nhạc Jamendo** theo `emotion`/`q` (từ khoá)/`genre`/`intensity`/`offset`/`limit`. Nếu tìm theo cảm xúc → chọn tag bằng Thompson Sampling (RLUF) + soft re-rank theo Audio Feature | `tracks[], query, tags_used, source, offset, limit, order, genre, has_more, tag_arm` |
| 7 | GET | `/api/mix-ratio` | `?emotion=...` — **RLUF**: tỉ lệ trộn nhạc Online:Local (12 bài tiếp theo) theo Thompson Sampling | `emotion, online (số bài), local (số bài), ratio{online, local}` |
| 8 | POST | `/api/track/event` | Ghi log hành vi nghe nhạc (`play/like/dislike/next`) + **cập nhật RLUF Bandit** theo reward (`like`=+1, `dislike/next`=-1). Body: `{type, track_id, name, artist, tags, emotion, source, tag_arm}` | `{"status": "logged"}` |
| 9 | GET | `/api/recommend` | **Gợi ý cá nhân hoá**: theo `liked_artists`/`liked_tags` (query, phân tách `,`) → nhạc của nghệ sĩ/tag yêu thích, fallback theo giờ trong ngày nếu chưa đủ dữ liệu | `tracks[] (≤20), source, based_on{artists, tags}, personalized` |
| 10 | GET | `/api/time-suggestion` | Gợi ý cảm xúc theo **giờ hiện tại trên máy server** (sáng→năng động, trưa→tập trung, tối→thư giãn...) | cảm xúc gợi ý (emotion/label/emoji/...) |

> Lưu ý: endpoint #6, #7, #8 cùng phối hợp với `bandit.py` (`ThompsonBandit`,
> `bandit_state.json`) để hiện thực RLUF — Like/Dislike/Bỏ qua ở #8 là "reward" cập nhật
> trực tiếp tham số Beta(a,b) dùng cho #6 và #7 ở lượt tìm kiếm tiếp theo.
