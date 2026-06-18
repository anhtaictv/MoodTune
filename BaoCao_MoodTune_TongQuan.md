# MoodTune — Báo cáo tổng hợp hệ thống (v3.1)

**Tên đầy đủ:** MoodTune — AI Cảm Xúc Tự Xây (Self-Attention · RLUF Bandit · Knowledge Graph) + Gợi ý nhạc Jamendo Hybrid
**Domain production:** https://anhtaictv.me
**Phiên bản hiện tại:** v3.1

---

## 1. Giới thiệu chung

MoodTune là web app phân tích **cảm xúc từ văn bản tiếng Việt** bằng một mô hình AI
**tự xây dựng từ đầu bằng NumPy thuần (không dùng TensorFlow/PyTorch/sklearn)**, kết hợp
với hệ thống **gợi ý & nghe nhạc miễn phí qua Jamendo API** (full track, không cần đăng nhập).

### Điểm đặc trưng của dự án

- **100% NumPy thuần**: toàn bộ AI Engine (forward, backward, attention, optimizer) tự
  viết — không dùng framework Machine Learning có sẵn.
- **Hybrid Model**: kết hợp Rule-based Scorer (từ điển trọng số) + Self-Attention MLP
  (mạng nơ-ron tự học).
- **Online Learning**: AI học ngay từ phản hồi người dùng ("Dạy AI"), không cần restart
  hay retrain lại toàn bộ.
- **Reinforcement Learning từ phản hồi người dùng (RLUF)**: Multi-Armed Bandit (Thompson
  Sampling) tối ưu tỉ lệ nhạc Online/Local và lựa chọn tag nhạc theo gu từng người.
- **Hybrid Online/Offline**: vừa nghe nhạc trực tuyến (Jamendo) vừa quét thư viện nhạc
  local trên máy người dùng (File System Access API), AI tự gắn nhãn cảm xúc cho file local.
- **Trực quan hoá AI**: Sơ đồ tri thức cảm xúc (Knowledge Graph) vẽ bằng Canvas 2D, thể
  hiện cơ chế Self-Attention đang "nhìn" vào từ nào trong câu.
- Toàn bộ trải nghiệm (nhập cảm xúc → AI phân tích → gợi ý nhạc → nghe → phản hồi → AI tự
  học) diễn ra liên tục trong **một trang duy nhất (SPA, vanilla JS, không framework)**.

---

## 2. Kiến trúc tổng quan hệ thống

```
┌──────────────────────────────┐        ┌───────────────────────────────┐
│        FRONTEND (SPA)         │  REST  │           BACKEND               │
│  index.html (HTML/CSS/JS)     │◄──────►│  Flask (Python) — app.py         │
│  - Vanilla JS, không framework│  API   │  - EmotionEngine (emotion_mlp.py)│
│  - Responsive / auto-scale    │        │  - Lexicon 15 class (lexicon.py) │
│  - LocalStorage (lịch sử,     │        │  - ThompsonBandit (bandit.py)    │
│    feedback, sở thích, kho    │        │  - Audio Feature Engine          │
│    nhạc local)                │        │  - Feedback log (jsonl)          │
└──────────────────────────────┘        └──────────────┬────────────────┘
        ▲                                               │
        │ IIS (anhtaictv.me)                            │ urllib (HTTP)
        │ - Serve static files                          ▼
        │ - URL Rewrite /api/* → 127.0.0.1:5005 ┌─────────────────────────┐
        └───────────────────────────────────────│  Jamendo Music API       │
                                                  │  (free, full-track MP3)  │
                                                  └─────────────────────────┘
```

### Thành phần triển khai

| Thành phần | Công nghệ | Vị trí |
|---|---|---|
| Frontend | HTML/CSS/JS tĩnh (1 file `index.html`, responsive) | `C:\moodtune\frontend`, host bởi IIS site **MoodTune** (`anhtaictv.me`) |
| Backend API | Python Flask (port `5005`) | `C:\moodtune\backend\app.py`, chạy nền bằng **PM2** (`moodtune-backend`) |
| Reverse proxy / CORS | IIS URL Rewrite + Flask-CORS | `web.config` proxy `/api/*` → `http://127.0.0.1:5005/api/*` |
| Nguồn nhạc | Jamendo API (free, full track, không login) | gọi trực tiếp từ backend |
| Trạng thái AI | File `.npz` / `.json` trên ổ đĩa | `weights.npz`, `weights_meta.json`, `weights_replay.json`, `dynamic_vocab.json`, `bandit_state.json` |
| Audio cache | File JSON | `audio_cache.json` (BPM/Spectral/MFCC từng track) |
| Log hành vi / feedback | File JSON Lines | `feedback_log.jsonl` |

---

## 3. AI Engine — Hybrid Emotion Model (NumPy thuần)

### 3.1. Tầng 1 — Rule-based Scorer (`rule_score`)

- Dựa trên `lexicon.py`: từ điển **805 từ/cụm từ** (+3 từ học online → 808) cho **15 nhóm
  cảm xúc**.
- Mỗi từ có trọng số 1.0 – 3.0 theo mức độ liên quan tới cảm xúc.
- **Bigram boost ×1.5**: cụm 2 từ (ví dụ "ngày xưa", "tức giận", "bực bội") được tính điểm
  cao hơn từ đơn — match bigram trước, unigram sau.
- **Xử lý phủ định**: nếu từ phủ định (`không, chẳng, chả, đâu, chưa, khỏi, ko, k`) đứng
  ngay trước từ cảm xúc → đảo dấu điểm, tránh hiểu sai ("không vui" ≠ "vui").
- Điểm âm bị clamp về 0, sau đó chuẩn hoá qua **Softmax**. Câu không khớp từ nào → phân
  phối đều `1/15`.

### 3.2. Tầng 2 — Self-Attention MLP (`AttentionMLP`, tự viết forward + backward)

```
Input: sequence token-id (giữ thứ tự từ, tối đa 24 token)
   → Embedding Matrix E (VOCAB_SIZE × 32)
   → Self-Attention(Q, K, V) = Softmax(QKᵀ/√d) · V   (tự cài forward + backward NumPy)
   → Mean-pool theo chiều token
   → Dense(32 → 64) + Leaky ReLU (slope 0.01, chống "Dying ReLU")
   → Dense(64 → 15) + Softmax
```

- **Tối ưu**: SGD + Momentum (0.9) + **Adaptive L2 Regularization** + He Initialization +
  **Gradient Clipping** (norm ≤ 5.0, cần thiết vì attention dễ nổ gradient).
- **Adaptive L2**: `l2 = min(5e-4, 1e-4 × (1 + feedback_count/100))` — model mới học ít
  thì regularize nhẹ, học nhiều thì regularize mạnh hơn để chống overfit vào các mẫu
  feedback gần nhất.
- Giữ trật tự từ trong câu (khác Bag-of-Words ở các bản đầu) → mô hình tự học được mối
  quan hệ giữa từ phủ định và từ cảm xúc đứng gần nó (ví dụ "không vui nổi").

### 3.3. Hybrid Blend — Kết hợp 2 tầng

```
final_score = alpha * rule_score + (1 - alpha) * mlp_score
alpha = max(0.35, 0.85 - 0.5 * feedback_count / (feedback_count + 40))
```

- Lúc mới khởi động: `alpha = 0.85` → tin **Rule** nhiều hơn (MLP chưa học gì).
- Mỗi lần "Dạy AI" thành công, `feedback_count` tăng → `alpha` giảm dần về tối thiểu
  **0.35** → hệ thống ngày càng tin **MLP** hơn.
- **Trạng thái hiện tại** (production): `feedback_count = 59`, `alpha ≈ 0.552`,
  `l2 ≈ 0.000159`, `vocab_size = 805` (+3 dynamic).

### 3.4. Online Learning + Experience Replay + Dynamic Vocab

- Khi người dùng bấm "Dạy AI" với nhãn đúng, model train **30 bước** trên câu mới.
- Chống **catastrophic forgetting**: mỗi 2 bước xen kẽ train lại 1 mẫu ngẫu nhiên từ
  **replay buffer** (tối đa 500 mẫu gần nhất, bao gồm cả seed data ban đầu).
- **Dynamic Vocab Expansion**: phát hiện từ mới ngoài từ điển (OOV) trong câu dạy → tự
  thêm vào `VOCAB`/`VOCAB_IDX`, mở rộng ma trận Embedding `E` bằng `np.vstack()` (khởi tạo
  He Init × 0.01, không phá đặc trưng đã học) — xảy ra **ngay trong runtime**, không cần
  restart server.
- Toàn bộ trọng số (`weights.npz`), trạng thái (`weights_meta.json`), replay buffer
  (`weights_replay.json`) và vocab động (`dynamic_vocab.json`) được lưu lại ngay sau mỗi
  lần học → không mất dữ liệu khi restart server.

### 3.5. 15 nhóm cảm xúc (classes)

| # | Mã | Tên hiển thị | Emoji | Tag nhạc Jamendo gợi ý |
|---|---|---|---|---|
| 1 | `vui_ve` | Vui vẻ | 😄 | pop+happy, dance+upbeat... |
| 2 | `buon_ba` | Buồn bã | 😢 | acoustic+sad, piano+melancholy... |
| 3 | `lang_man` | Lãng mạn | 🥰 | romantic, love+ballad... |
| 4 | `nang_dong` | Năng động | ⚡ | dance+energetic, edm+party... |
| 5 | `thu_gian` | Thư giãn | 🌿 | chill+lofi, ambient+calm... |
| 6 | `co_don` | Cô đơn | 🌙 | sad+piano, lonely+night... |
| 7 | `cang_thang` | Căng thẳng | 🔥 | intense+focus, dramatic... |
| 8 | `tap_trung` | Tập trung | 🎯 | study+lofi, instrumental+focus... |
| 9 | `hoai_niem` | Hoài niệm | 🍂 | nostalgic, retro+oldies... |
| 10 | `phieu_luu` | Phiêu lưu | ⚔️ | epic+adventure, cinematic... |
| 11 | `bi_an` | Bí ẩn | 🔮 | dark+mysterious, ambient+dark... |
| 12 | `vui_nhon` | Vui nhộn | 🤪 | fun+quirky, comedy+upbeat... |
| 13 | `tu_tin` | Tự tin | 💪 | motivational+epic, rock+powerful... |
| 14 | `biet_on` | Biết ơn | 🙏 | acoustic+warm, folk+calm... |
| 15 | `tuc_gian` | Tức giận | 😡 | metal+aggressive, punk+rock... |

> 3 nhóm cuối (`tu_tin`, `biet_on`, `tuc_gian`) là điểm mới của v3.1, lấp các "khoảng
> trống mood" mà 12 nhóm ban đầu chưa phủ tới.

---

## 4. Hệ thống gợi ý & phát nhạc

### 4.1. Jamendo Integration (nền tảng, từ v1.0)

- Gọi trực tiếp **Jamendo API** (miễn phí, full track MP3, không cần đăng nhập/API key
  trả phí).
- Sau khi phân tích cảm xúc, tự động tìm nhạc theo **tag tương ứng cảm xúc** (mỗi cảm xúc
  có nhiều bộ tag → kết quả đa dạng).
- Tìm kiếm thủ công theo tên bài hát/nghệ sĩ, phân trang "⬇ Xem thêm bài".
- Slider **"Cường độ" (1–10)** điều chỉnh tiêu chí xếp hạng kết quả: 1–3 nhạc mới nổi
  trong tháng, 4–7 nhạc hot mọi thời đại, 8–10 nhạc trending trong tuần.

### 4.2. Audio Feature Engine (`audio_features.py`, từ v2.0)

- Chạy nền (background thread), không chặn `/api/music/search`.
- Dùng `librosa` phân tích 30 giây đầu track: **BPM** (tempo), **Spectral Centroid**, **MFCC**.
- Heuristic map đặc trưng âm thanh → 1 trong 15 nhãn cảm xúc, dùng để **soft re-rank**
  danh sách bài hát (ưu tiên bài có `audio_emotion` khớp cảm xúc người dùng).
- Cache vào `audio_cache.json`. **Graceful fallback**: nếu thiếu `librosa`, mọi hàm
  thành no-op, hệ thống vẫn chạy bình thường.

### 4.3. Hybrid Online/Offline — "Kho nhạc Local" (từ v2.0)

- Tab **"📁 Kho nhạc Local"**: chọn thư mục nhạc qua `window.showDirectoryPicker()` (File
  System Access API).
- `/api/predict/batch`: AI gắn nhãn cảm xúc cho từng file dựa trên **tên bài hát** (tối đa
  200 file/lần).
- Bản đồ `{filename: {emotion, label, emoji}}` lưu trong `localStorage`
  (`mt_local_library`) — **không upload nội dung file** lên server (privacy-first).
- Phát file local qua `URL.createObjectURL(file)` (Blob URL), không cần server.
- **Hybrid Playlist Mixer**: trộn nhạc Online (Jamendo) + Local theo tỉ lệ động (xem 4.4).

### 4.4. RLUF — Thompson Sampling Multi-Armed Bandit (`bandit.py`, từ v3.0)

- Mỗi cảm xúc có 2 "bandit" độc lập (Beta-Bernoulli, `np.random.beta`, 100% NumPy):
  - **Source bandit**: học tỉ lệ trộn **Online : Local** (thay tỉ lệ cố định 6:6).
  - **Tag bandit**: học cặp tag Jamendo nào hay được "Like" nhất cho từng cảm xúc.
- `GET /api/mix-ratio?emotion=...` trả về tỉ lệ kỳ vọng hiện tại (Online X% / Local Y%),
  hiển thị trên UI dưới dạng **"🎯 Gu của bạn: Online X% · Local Y%"**.
- Phản hồi người dùng (**Like / Dislike / Bỏ qua**) → cập nhật tham số Beta(a,b) →
  `bandit_state.json`, theo từng cảm xúc, theo thời gian thực.

---

## 5. Danh sách chức năng giao diện (Frontend, `index.html` — vanilla JS, SPA)

| # | Chức năng | Mô tả ngắn |
|---|---|---|
| 1 | **Phân tích cảm xúc** | Nhập văn bản tự do hoặc chọn 1 trong 15 icon cảm xúc → `POST /api/predict` → hiển thị emoji, tên cảm xúc, % tin cậy, biểu đồ điểm 15 nhóm cảm xúc, tỉ lệ Rule vs MLP |
| 2 | **Sơ đồ tri thức AI (Knowledge Graph)** | Canvas 2D vẽ 15 "bong bóng" cảm xúc + từ khoá có attention cao nhất + "tia năng lượng" nối từ khoá → cảm xúc tương ứng, có animation |
| 3 | **Slider cường độ (1–10)** | Điều chỉnh tiêu chí tìm nhạc (mới nổi / hot mọi thời / trending tuần) |
| 4 | **Gợi ý & tìm nhạc Jamendo** | Tự tìm theo cảm xúc (tag bandit chọn) + tìm thủ công + load more |
| 5 | **Trình phát nhạc (Now Playing)** | Phát/dừng qua `<audio>`, hiển thị bìa/tên/nghệ sĩ/album, tự chuyển bài kế |
| 6 | **Feedback nhạc (Like/Dislike/Bỏ qua)** | Cập nhật `localStorage` + gửi `/api/track/event` → cập nhật Bandit + gợi ý cá nhân hoá |
| 7 | **Gợi ý cá nhân hoá ("✨ Gợi ý cho bạn")** | `GET /api/recommend` theo nghệ sĩ/tag yêu thích, fallback theo giờ trong ngày |
| 8 | **Gợi ý theo thời gian trong ngày** | Tự chọn sẵn cảm xúc theo giờ máy (sáng→năng động, trưa→tập trung, tối→thư giãn...) |
| 9 | **Kho nhạc Local (Hybrid Online/Offline)** | Quét thư mục nhạc máy người dùng, AI gắn nhãn cảm xúc theo tên file, trộn vào playlist |
| 10 | **"Dạy AI" (Online Learning UI)** | 15 nút sửa nhãn cảm xúc đúng → `POST /api/learn` → model học ngay (30 bước + replay) |
| 11 | **Lịch sử phân tích** | Lưu tối đa 20 lượt phân tích gần nhất (text, emoji, % tin cậy, thời gian) trong `localStorage` |
| 12 | **Thanh trạng thái hệ thống** | Vocab size, số Classes, kiến trúc Layers, Activation, L2, tổng Feedback, Alpha — từ `GET /api/health`, đèn online/offline |
| 13 | **Modal "About" + "Lịch sử phiên bản"** | Giới thiệu nhóm/đề tài + changelog đầy đủ v1.0 → v3.1, click vào tag version để xem |
| 14 | **Responsive / Auto-scale (mobile)** | CSS dùng `clamp()` cho font/kích thước ảnh, grid 15 nút cảm xúc tự co 4→3 cột trên điện thoại, sửa lỗi tràn ngang ở 2 card chính (grid item `min-width:0`) |

---

## 6. API Endpoints (Backend Flask)

| Method | Endpoint | Chức năng |
|---|---|---|
| GET | `/api/health` | Trạng thái server, kiến trúc model (`vocab_size, embed_dim, hidden_size, output_size, attention, activation, l2`), feedback count, alpha |
| POST | `/api/predict` | Phân tích cảm xúc → `emotion, label, emoji, confidence, all_scores, rule_scores, mlp_scores, alpha, graph` (Knowledge Graph) |
| POST | `/api/learn` | Online learning — dạy AI nhãn đúng cho 1 câu (30 bước + replay + dynamic vocab) |
| GET | `/api/stats` | Thống kê tổng lượt predict/feedback, phân bố cảm xúc |
| POST | `/api/predict/batch` | Gắn nhãn cảm xúc cho danh sách tên file nhạc local (tối đa 200) |
| GET | `/api/music/search` | Tìm nhạc Jamendo theo emotion/tag/genre/cường độ/từ khoá, trả `tag_arm` (Bandit) |
| GET | `/api/mix-ratio` | `?emotion=...` → tỉ lệ trộn Online/Local hiện tại (Thompson Sampling) |
| GET | `/api/recommend` | Gợi ý nhạc cá nhân hoá (nghệ sĩ/tag yêu thích + giờ trong ngày) |
| POST | `/api/track/event` | Ghi log hành vi `play/like/dislike/next`, cập nhật Bandit theo reward |
| GET | `/api/time-suggestion` | Trả cảm xúc gợi ý theo giờ trong ngày |

---

## 7. Lưu trữ dữ liệu (Persistence)

| File / Storage | Nội dung |
|---|---|
| `weights.npz` | Trọng số mạng (`E, Wq, Wk, Wv, W1, b1, W2, b2`) |
| `weights_meta.json` | `feedback_count`, `alpha`, `l2`, `arch`, `embed_dim`, `vocab_size` |
| `weights_replay.json` | Experience Replay buffer (≤500 mẫu) chống quên kiến thức |
| `dynamic_vocab.json` | Từ mới học online qua "Dạy AI", mở rộng vocab runtime |
| `bandit_state.json` | Tham số Beta(a,b) cho Source/Tag bandit theo từng cảm xúc |
| `audio_cache.json` | Kết quả phân tích audio (BPM/Centroid/MFCC) từng track Jamendo |
| `feedback_log.jsonl` | Log JSONL: mọi lượt predict, dạy AI, play/like/dislike/next |
| `localStorage` (browser) | Lịch sử phân tích, feedback bài hát, nghệ sĩ/tag yêu thích, kho nhạc local |

---

## 8. Triển khai Production

- **Frontend**: serve tĩnh qua IIS, site `MoodTune` → `C:\moodtune\frontend`, domain
  **anhtaictv.me** (HTTP + HTTPS).
- **Backend**: chạy bằng **PM2** (`ecosystem.config.js`), process `moodtune-backend`,
  `app.py`, lắng nghe `0.0.0.0:5005`.
- **Kết nối Frontend ↔ Backend**: IIS URL Rewrite chuyển tiếp `/api/*` →
  `http://127.0.0.1:5005/api/*`. Flask-CORS giới hạn origin (production domain + localhost
  dev).
- **Nguồn nhạc**: gọi trực tiếp Jamendo API, không cần login, trả MP3 full track.

---

## 9. Kết quả kiểm thử & đánh giá tổng thể

| Mục kiểm thử | Kết quả |
|---|---|
| Độ chính xác held-out (30 câu, 2 câu/class) | **93.3% (28/30)** — 15 class mới đạt 100% (6/6), 12 class cũ 91.7% (22/24) |
| Thời gian phản hồi `/api/predict` | ~290ms/request |
| Trang chủ `anhtaictv.me` | 200 OK, ~66.5KB |
| `/api/music/search`, `/api/mix-ratio`, `/api/recommend`, `/api/predict/batch` | 200 OK, hoạt động đúng với 15 class |
| Log PM2 | Không có lỗi 500 |
| Giao diện | 15/15 nút cảm xúc + 15/15 nút "Dạy AI" hiển thị đúng, responsive trên di động |

---

## 10. Lịch sử phát triển (Changelog v1.0 → v3.1)

| Phiên bản | Thay đổi chính |
|---|---|
| **v1.0** | Nền tảng: Rule + MLP Hybrid Engine (Bag-of-Words 653-d), tích hợp Jamendo, online learning + replay buffer cơ bản |
| **v1.1** | Hiển thị version trên UI, hoàn thiện toàn bộ chức năng cơ bản (12 class, gợi ý cá nhân hoá, gợi ý theo giờ, lịch sử phân tích) |
| **v2.0** | **Self-Attention + Embedding** thay Bag-of-Words; Dynamic Vocab/Weight Expansion runtime; Audio Feature Engine (librosa); Hybrid Online/Offline "Kho nhạc Local" |
| **v2.5** | **Leaky ReLU** thay ReLU (chống Dying ReLU); **Adaptive L2 Regularization** theo feedback; modal "Lịch sử phiên bản" (Changelog UI) |
| **v3.0** | **RLUF Thompson Sampling Bandit** (tối ưu tỉ lệ Online:Local + chọn tag nhạc); **Sơ đồ tri thức cảm xúc** (Knowledge Graph Canvas, trực quan hoá Self-Attention) |
| **v3.1** | **Mở rộng 12 → 15 class cảm xúc** (Tự tin 💪, Biết ơn 🙏, Tức giận 😡); mở rộng từ điển (656 → 805 từ); retrain Self-Attention MLP cho 15 class; hotfix overlap lexicon `tuc_gian`↔`cang_thang` và `buon_ba`; cải thiện responsive UI cho di động (auto-scale) |

---

## 11. Định hướng phát triển tiếp theo

- **Đóng gói phân phối**: Docker (Web) / PyInstaller hoặc Electron (Desktop) cho bản chạy
  biên (Edge AI) — đề xuất từ v2.0, chưa triển khai.
- **Đa phương thức Face/Voice**: bị loại có chủ đích ở v3.0 vì vi phạm triết lý "100%
  NumPy thuần" (yêu cầu thư viện AI ngoài như face-api.js).
- **Tinh chỉnh lexicon còn lại**: một số overlap nhỏ còn tồn đọng (ví dụ "lạnh" → `bi_an`
  thay vì nghĩa thời tiết) — để dành cho người dùng tự sửa qua **"Dạy AI"**
  (online learning) thay vì can thiệp lexicon thủ công, tránh overfit trên bộ test nhỏ.
- **Mở rộng vocab/seed data liên tục**: thông qua phản hồi thực tế từ người dùng và "Dạy AI".
