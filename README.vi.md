<div align="center">
  <img src="MoodTune.png" alt="MoodTune" width="120" />

  # MoodTune

  **Ứng dụng gợi ý nhạc theo cảm xúc, phân tích văn bản tiếng Việt bằng AI tự xây dựng**

  [![Version](https://img.shields.io/badge/phiên_bản-v4.0-blue?style=flat-square)](https://github.com/anhtaictv/MoodTunes/releases)
  [![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
  [![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=flat-square&logo=flask)](https://flask.palletsprojects.com)
  [![NumPy](https://img.shields.io/badge/NumPy-only-013243?style=flat-square&logo=numpy)](https://numpy.org)

  [🌐 Xem Demo](https://anhtaictv.me) · [🇬🇧 English version](README.md)
</div>

---

## Tổng quan

MoodTune phân tích một đoạn văn bản tiếng Việt, đoán cảm xúc của người dùng bằng AI engine tự xây dựng, rồi trả về các bài nhạc phù hợp từ Jamendo API — miễn phí, không cần đăng nhập.

AI engine được viết **hoàn toàn bằng NumPy thuần** (không PyTorch, không TensorFlow): kết hợp rule scorer trên lexicon cảm xúc tiếng Việt và một MLP có self-attention, học online từ feedback người dùng.

## Tính năng

- **Nhận diện cảm xúc** — 10 class theo mô hình Valence-Arousal (GEMS/Circumplex)
- **MLP Self-Attention** — Embedding + Q/K/V attention tự viết bằng NumPy; hiểu thứ tự từ và phủ định
- **Online learning** — model cập nhật ngay từng feedback (thích / không thích / bỏ qua)
- **Thompson Sampling Bandit** — học gu nhạc cá nhân qua nhiều phiên
- **Audio re-ranking** — BPM, spectral centroid, MFCC qua librosa để tinh chỉnh kết quả
- **Giao diện đa theme** — Tối (Midnight), Sáng (Aurora), Rực rỡ (Sunset); nhớ qua localStorage
- **PWA** — cài được như app, hỗ trợ offline với nội dung đã cache
- **Không phụ thuộc ML framework** — toàn bộ AI chạy trên NumPy + Python stdlib

## Công nghệ sử dụng

| Tầng | Công nghệ |
|---|---|
| Backend | Python 3, Flask, Waitress |
| AI Engine | NumPy (tự viết MLP, attention, bandit) |
| Music API | Jamendo |
| Frontend | HTML / CSS / JS thuần (không build step) |
| Process manager | PM2 |
| Reverse proxy | IIS / Nginx |

## Cấu trúc dự án

```
moodtune/
├── backend/
│   ├── app.py                  # Entrypoint Flask
│   ├── emotion_mlp.py          # MLP Self-Attention + online learning
│   ├── lexicon.py              # Lexicon cảm xúc & rule scorer
│   ├── bandit.py               # Thompson Sampling bandit
│   ├── audio_features.py       # Re-ranking BPM / spectral / MFCC
│   ├── weights.npz             # Trọng số model đã huấn luyện
│   ├── dynamic_vocab.json      # Vocab mở rộng runtime
│   ├── requirements.txt
│   └── ecosystem.config.js     # Config PM2
└── frontend/
    └── index.html              # Single-page app
```

## Chạy nhanh

**Backend**

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Chạy tại `http://localhost:5005`. Kiểm tra `/api/health` để xác nhận.

**Frontend**

Mở `frontend/index.html` bằng trình duyệt (hoặc qua static server trên port 5500). Khi chạy ở `localhost` / `file://`, frontend tự gọi thẳng `http://localhost:5005/api`.

**Production (PM2)**

```bash
pm2 start backend/ecosystem.config.js
```

## Biến môi trường

Tất cả đều có giá trị mặc định — không bắt buộc khi chạy local.

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `MOODTUNE_SECRET_KEY` | `moodtune_secret_2024` | Flask secret key. Nên đổi khi deploy thật. |
| `MOODTUNE_FRONTEND` | `https://anhtaictv.me` | Origin frontend để cấu hình CORS. |
| `JAMENDO_CLIENT_ID` | `cf31dbfd` | Client ID gọi Jamendo API. |

## Deploy Production

Frontend gọi `/api` (relative path). Cần cấu hình reverse proxy (IIS hoặc Nginx) trỏ `/api` → backend Flask port 5005. Xem `frontend/web.config` để biết cấu hình IIS mẫu.

## Dữ liệu model

| File | Nội dung |
|---|---|
| `weights.npz` | Trọng số MLP + embedding matrix |
| `weights_meta.json` | Metadata vocabulary |
| `weights_replay.json` | Replay buffer cho online learning |
| `dynamic_vocab.json` | Vocab mở rộng trong runtime |

## Lịch sử phiên bản

Bấm vào tag phiên bản trên UI để xem changelog trong app. Báo cáo đầy đủ từng phiên bản (sơ đồ kiến trúc, so sánh trước/sau, kết quả kiểm thử) được link bên dưới.

| Phiên bản | Tên | Nội dung chính |
|---|---|---|
| `v4.0` | Word Segmentation | Tổng quát hoá tokenizer sang **longest-match N-gram** (tối đa 5 từ); sửa lỗi "vocab chết" với 54 cụm từ không được nhận diện. ([báo cáo](BaoCao_MoodTune_v4.0.md)) |
| `v3.8` | Giao diện đa theme | 3 theme: Midnight / Aurora / Sunset; đạt WCAG AA ở cả 3 theme; sơ đồ tri thức theo theme. ([báo cáo](BaoCao_MoodTune_v3.8.md)) |
| `v3.7` | Production ổn định | PWA manifest, presence widget, Waitress server, rate limiting, tự vá lệch vocab. ([báo cáo](BaoCao_MoodTune_v3.7.md)) |
| `v3.6` | Sửa phủ định | Sửa rule scorer nhận diện phủ định nhiều từ (`"không hề"`, `"chẳng bao giờ"`, ...). ([báo cáo](BaoCao_MoodTune_v3.6.md)) |
| `v3.5` | Valence-Arousal | Rút về 10 class cảm xúc theo mô hình GEMS/Circumplex. ([báo cáo](BaoCao_MoodTune_v3.5.md)) |
| `v3.1` | Thêm cảm xúc | Tự tin 💪, Biết ơn 🙏, Tức giận 😡 — từ 12 lên 15 class. ([báo cáo](BaoCao_MoodTune_v3.1.md)) |
| `v3.0` | Bandit + Sơ đồ tri thức | Thompson Sampling bandit học gu nhạc; sơ đồ tri thức Canvas 2D tương tác. ([báo cáo](BaoCao_MoodTune_v3.0.md)) |
| `v2.5` | Adaptive Learning | ReLU → Leaky ReLU; L2 adaptive; modal changelog trong app. ([báo cáo](BaoCao_MoodTune_v2.5.md)) |
| `v2.0` | Self-Attention | Embedding + self-attention (Q,K,V) NumPy; audio feature engine; dynamic vocab. ([báo cáo](BaoCao_MoodTune_v2.0.md)) |
| `v1.1` | Hoàn thiện UI | Badge phiên bản; gợi ý cá nhân hoá & theo giờ; lịch sử phân tích. ([báo cáo](BaoCao_MoodTune_v1.1.md)) |
| `v1.0` | Nền tảng | Rule scorer + MLP hybrid, Jamendo API, online learning cơ bản. |

Tài liệu bổ sung: [`BaoCao_MoodTune_TongQuan.md`](BaoCao_MoodTune_TongQuan.md) (kiến trúc tổng quan) và [`BaoCao_MoodTune_DacTrung_AIEngine_API.md`](BaoCao_MoodTune_DacTrung_AIEngine_API.md) (AI Engine & API chi tiết).
