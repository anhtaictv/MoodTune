# MoodTune — Tài liệu tổng hợp chức năng & sơ lược hệ thống

**Phiên bản hiện tại:** `v1.1`
**Tên đầy đủ:** MoodTune — AI Cảm Xúc Tự Xây + Gợi ý nhạc Jamendo
**Domain production:** https://anhtaictv.me

---

## 1. Giới thiệu chung

MoodTune là web app phân tích **cảm xúc từ văn bản tiếng Việt** bằng một mô hình AI
**tự xây dựng từ đầu (numpy thuần, không dùng framework ML)**, kết hợp với hệ thống
**gợi ý/nghe nhạc miễn phí qua Jamendo API** (full track, không cần đăng nhập).

Điểm đặc trưng của dự án:
- AI Engine là **Hybrid Model** (Rule-based + MLP tự viết bằng NumPy) — không dùng
  TensorFlow/PyTorch/sklearn.
- Có khả năng **học online (online learning)** từ phản hồi người dùng, không cần
  retrain lại từ đầu.
- Toàn bộ trải nghiệm (phân tích → gợi ý nhạc → nghe → phản hồi → AI tự học) diễn
  ra liên tục trong một giao diện duy nhất (single-page app).

---

## 2. Sơ đồ kiến trúc tổng quan

```
┌──────────────────────────────┐        ┌───────────────────────────────┐
│        FRONTEND (SPA)         │  REST  │           BACKEND               │
│  index.html (HTML/CSS/JS)     │◄──────►│  Flask (Python) — app.py         │
│  - Vanilla JS, không framework│  API   │  - EmotionEngine (emotion_mlp.py)│
│  - LocalStorage (lịch sử,     │        │  - Lexicon (lexicon.py)          │
│    feedback, sở thích)        │        │  - Feedback log (jsonl)          │
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
| Frontend | HTML/CSS/JS tĩnh (1 file `index.html`) | `C:\moodtune\frontend`, host bởi IIS site **MoodTune** (`anhtaictv.me`) |
| Backend API | Python Flask (port `5005`) | `C:\moodtune\backend\app.py`, chạy nền bằng **PM2** (`moodtune-backend`) |
| Reverse proxy / CORS | IIS URL Rewrite + Flask-CORS | `web.config` proxy `/api/*` → `http://127.0.0.1:5005/api/*` |
| Nguồn nhạc | Jamendo API (free, full track, không login) | gọi trực tiếp từ backend |
| Lưu trữ trạng thái AI | File `.npz` / `.json` trên ổ đĩa | `weights.npz`, `weights_meta.json`, `weights_replay.json` |
| Log hành vi / feedback | File JSON Lines | `feedback_log.jsonl` |

---

## 3. AI Engine — Hybrid Emotion Model (numpy thuần)

### 3.1. Kiến trúc 2 tầng

**Tầng 1 — Rule Scorer (Từ điển trọng số)**
- Dựa trên `lexicon.py`: từ điển **653 từ/cụm từ** cho **12 nhóm cảm xúc**.
- Mỗi từ có trọng số 1.0 – 3.0 theo mức độ liên quan.
- **Bigram boost ×1.5**: cụm 2 từ (ví dụ "ngày xưa", "tập trung") được tính điểm
  cao hơn từ đơn.
- **Xử lý phủ định**: nếu từ phủ định ("không", "chẳng"...) đứng ngay trước từ
  cảm xúc → đảo dấu điểm (×-0.6) để tránh hiểu sai ("không vui" ≠ "vui").
- Kết quả được chuẩn hóa qua Softmax.

**Tầng 2 — MLP Learner (tự viết forward + backprop bằng NumPy)**
- Kiến trúc: `Input(653) → Hidden(64, ReLU) → Output(12, Softmax)`
- Tối ưu: **SGD + Momentum (0.9) + L2 regularization (1e-4)**
- Khởi tạo trọng số: **He Initialization**
- Input là vector **Bag-of-Words có trọng số bigram**, chuẩn hóa theo norm.

### 3.2. Hybrid Blend — Kết hợp 2 tầng
```
final_score = alpha * rule_score + (1 - alpha) * mlp_score
alpha = max(0.35, 0.85 - 0.5 * feedback_count / (feedback_count + 40))
```
- Lúc mới khởi động: `alpha = 0.85` → tin **Rule** nhiều hơn (vì MLP chưa học gì).
- Mỗi lần người dùng "dạy" AI (feedback đúng), `feedback_count` tăng → `alpha` giảm
  dần về tối thiểu **0.35** → hệ thống ngày càng tin **MLP** hơn.

### 3.3. Online Learning + Experience Replay
- Khi người dùng bấm "dạy AI" với nhãn đúng, model train **30 bước** trên câu mới.
- Để tránh **catastrophic forgetting** (quên kiến thức cũ), mỗi 2 bước xen kẽ
  train lại 1 mẫu ngẫu nhiên từ **replay buffer** (tối đa 500 mẫu gần nhất, gồm
  cả seed data ban đầu).
- Trọng số (`weights.npz`), trạng thái (`weights_meta.json`) và replay buffer
  (`weights_replay.json`) được lưu lại ngay sau mỗi lần học → **không mất dữ liệu
  khi restart server**.

### 3.4. 12 nhóm cảm xúc (classes)
| # | Mã | Tên hiển thị | Emoji | Số từ/cụm trong lexicon |
|---|---|---|---|---|
| 1 | `vui_ve` | Vui vẻ | 😄 | 55 |
| 2 | `buon_ba` | Buồn bã | 😢 | 50 |
| 3 | `lang_man` | Lãng mạn | 🥰 | 50 |
| 4 | `nang_dong` | Năng động | ⚡ | 53 |
| 5 | `thu_gian` | Thư giãn | 🌿 | 55 |
| 6 | `co_don` | Cô đơn | 🌙 | 49 |
| 7 | `cang_thang` | Căng thẳng | 🔥 | 61 |
| 8 | `tap_trung` | Tập trung | 🎯 | 52 |
| 9 | `hoai_niem` | Hoài niệm | 🍂 | 55 |
| 10 | `phieu_luu` | Phiêu lưu | ⚔️ | 58 |
| 11 | `bi_an` | Bí ẩn | 🔮 | 59 |
| 12 | `vui_nhon` | Vui nhộn | 🤪 | 62 |

---

## 4. Danh sách chức năng (Frontend)

### 4.1. Phân tích cảm xúc
- Người dùng nhập văn bản tự do **hoặc** chọn 1 trong 12 icon cảm xúc có sẵn.
- Bấm "✦ Phân tích & tìm nhạc" → gọi `POST /api/predict`.
- Hiển thị: emoji + tên cảm xúc, % độ tin cậy, **biểu đồ điểm số của cả 12 nhóm
  cảm xúc**, tỉ lệ đóng góp Rule vs MLP, tổng số feedback đã học.

### 4.2. Slider "Cường độ" (1–10)
- Điều chỉnh tiêu chí xếp hạng kết quả nhạc trả về từ Jamendo:
  - 1–3: nhạc mới nổi trong tháng (mellow)
  - 4–7: nhạc hot mọi thời đại (all-time hits)
  - 8–10: nhạc trending trong tuần (hot)

### 4.3. Gợi ý & tìm nhạc (Jamendo — miễn phí, full track)
- Sau khi phân tích cảm xúc, hệ thống tự động gọi `/api/music/search` với **tag
  nhạc tương ứng cảm xúc** (mỗi cảm xúc có nhiều bộ tag, chọn ngẫu nhiên → kết quả
  luôn đa dạng).
- Tìm kiếm thủ công theo tên bài hát / nghệ sĩ.
- Phân trang "⬇ Xem thêm bài" (load more).
- Mỗi bài hiển thị ảnh bìa, tên, nghệ sĩ, album, badge "FULL" nếu có file nghe đầy đủ.

### 4.4. Trình phát nhạc (Now Playing)
- Phát/tạm dừng trực tiếp ngay trong trang qua `<audio>`.
- Hiển thị ảnh bìa, tên bài, nghệ sĩ, album đang phát.
- Tự động chuyển bài kế tiếp khi bài hiện tại kết thúc.

### 4.5. Feedback nhạc (Like / Dislike)
- Người dùng Like/Dislike bài đang nghe → lưu vào `localStorage` và gửi lên
  `POST /api/track/event` để server thống kê.
- Dữ liệu like/dislike (nghệ sĩ, tag) được dùng làm input cho gợi ý cá nhân hoá.

### 4.6. Gợi ý cá nhân hoá ("✨ Gợi ý cho bạn")
- Gọi `GET /api/recommend` kèm danh sách nghệ sĩ/tag yêu thích (từ `localStorage`).
- Backend kết hợp dữ liệu client + thống kê log server để trả về:
  1. Nhạc của các nghệ sĩ user hay thích
  2. Nhạc theo tag ưa thích
  3. Fallback: gợi ý theo cảm xúc tương ứng giờ trong ngày (nếu chưa đủ dữ liệu)

### 4.7. Gợi ý theo thời gian trong ngày
- Khi load trang, tự động chọn sẵn 1 cảm xúc gợi ý dựa theo giờ máy người dùng
  (sáng → năng động, trưa → tập trung, chiều → vui vẻ, tối → thư giãn, khuya →
  thư giãn nhẹ) và tự tìm nhạc luôn.
- Đồng bộ với endpoint `GET /api/time-suggestion` ở backend.

### 4.8. "Dạy AI" — sửa kết quả sai (Online Learning UI)
- Nếu AI đoán sai, người dùng bấm vào 1 trong 12 nút cảm xúc đúng.
- Gọi `POST /api/learn` → model học ngay (30 bước + replay), trả về
  `feedback_count` và `alpha` mới, cập nhật trực tiếp trên UI.

### 4.9. Lịch sử phân tích
- Lưu tối đa 20 lượt phân tích gần nhất (text, emoji, label, % tin cậy, thời gian)
  vào `localStorage`, hiển thị ở cuối trang.

### 4.10. Thanh trạng thái hệ thống (Stat row)
- Hiển thị real-time: Vocab size, số Classes, kiến trúc Layers, tổng Feedback đã
  học, giá trị Alpha hiện tại — lấy từ `GET /api/health`.
- Đèn trạng thái (xanh = online, đỏ = offline) cho biết kết nối backend.

---

## 5. API Endpoints (Backend Flask)

| Method | Endpoint | Chức năng |
|---|---|---|
| GET | `/api/health` | Trạng thái server, thông tin kiến trúc model, feedback count, alpha |
| POST | `/api/predict` | Phân tích cảm xúc từ văn bản → trả về emotion, confidence, điểm 12 nhóm |
| POST | `/api/learn` | Online learning — dạy AI nhãn đúng cho 1 câu |
| GET | `/api/stats` | Thống kê tổng số lượt predict/feedback, phân bố cảm xúc |
| GET | `/api/music/search` | Tìm nhạc trên Jamendo theo emotion/tag/genre/cường độ/từ khoá |
| GET | `/api/recommend` | Gợi ý nhạc cá nhân hoá (theo nghệ sĩ/tag yêu thích + giờ trong ngày) |
| POST | `/api/track/event` | Ghi log hành vi nghe/like/dislike để phục vụ gợi ý |
| GET | `/api/time-suggestion` | Trả về cảm xúc gợi ý theo giờ trong ngày |

---

## 6. Lưu trữ dữ liệu (Persistence)

| File | Nội dung |
|---|---|
| `weights.npz` | Trọng số MLP (`W1, b1, W2, b2`) |
| `weights_meta.json` | `feedback_count`, `alpha` hiện tại |
| `weights_replay.json` | Bộ nhớ replay (tối đa 500 mẫu gần nhất) chống quên kiến thức |
| `feedback_log.jsonl` | Log JSONL: mọi lượt predict, feedback dạy AI, sự kiện play/like/dislike |
| `localStorage` (browser) | Lịch sử phân tích, feedback bài hát, nghệ sĩ/tag yêu thích |

---

## 7. Triển khai Production

- **Frontend**: serve tĩnh qua IIS, site `MoodTune` → `C:\moodtune\frontend`,
  domain **anhtaictv.me** (HTTP + HTTPS).
- **Backend**: chạy bằng **PM2** (`ecosystem.config.js`), process
  `moodtune-backend`, Python script `app.py`, lắng nghe `0.0.0.0:5005`.
- **Kết nối Frontend ↔ Backend**: IIS URL Rewrite chuyển tiếp mọi request
  `/api/*` → `http://127.0.0.1:5005/api/*`. Flask-CORS giới hạn origin được phép
  (production domain + localhost dev).
- **Nguồn nhạc**: gọi trực tiếp Jamendo API (`api.jamendo.com`), không cần API key
  trả phí, không cần user login, trả về MP3 full track.

---

## 8. Lịch sử phiên bản

- **v1.1** *(hiện tại)* — Hiển thị số phiên bản trên giao diện; hoàn thiện đầy đủ
  các chức năng: phân tích cảm xúc Hybrid AI, gợi ý/nghe nhạc Jamendo, online
  learning, gợi ý cá nhân hoá, gợi ý theo giờ, lịch sử phân tích.
- **v1.0** — Phiên bản nền tảng: Rule + MLP Hybrid Engine, tích hợp Jamendo,
  feedback & online learning cơ bản.

> Các định hướng phát triển tiếp theo (v2.0): mở rộng động từ điển/ma trận trọng
> số, trích xuất đặc trưng âm thanh đa phương thức (audio features), Self-Attention
> viết bằng NumPy thuần, kiến trúc Hybrid Online/Offline (kho nhạc local), và đóng
> gói phân phối độc lập (Docker/PyInstaller).
