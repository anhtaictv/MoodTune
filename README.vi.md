<div align="center">
  <img src="frontend/logo.png" alt="MoodTune" width="110" />

  <h1>MoodTune</h1>

  <p><strong>Gõ cảm xúc — nhận ngay nhạc phù hợp</strong><br/>
  AI nhận diện cảm xúc từ văn bản tiếng Việt, gợi ý nhạc miễn phí, không cần đăng nhập.</p>

  [![Version](https://img.shields.io/badge/phiên_bản-v4.0-brightgreen?style=flat-square)](https://github.com/anhtaictv/MoodTunes/releases)
  [![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
  [![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=flat-square&logo=flask)](https://flask.palletsprojects.com)
  [![NumPy](https://img.shields.io/badge/NumPy_thuần-không_framework-013243?style=flat-square&logo=numpy)](https://numpy.org)
  [![Live](https://img.shields.io/badge/live-anhtaictv.me-1db954?style=flat-square&logo=googlechrome&logoColor=white)](https://anhtaictv.me)

  [🌐 Xem Demo](https://anhtaictv.me) · [🇬🇧 English](README.en.md)
</div>

---

## Giới thiệu

MoodTune là ứng dụng gợi ý nhạc theo cảm xúc: bạn gõ một câu mô tả tâm trạng (hoặc chọn emoji), AI tự phân tích cảm xúc và trả về danh sách nhạc phù hợp từ Jamendo — miễn phí, full track, không cần tài khoản.

**Điểm đặc biệt:** toàn bộ AI engine được viết từ đầu bằng **NumPy thuần** — không PyTorch, không TensorFlow. Kiến trúc gồm rule scorer trên lexicon tiếng Việt kết hợp với Self-Attention MLP học online từ phản hồi người dùng.

```
"hôm nay mệt mỏi muốn thư giãn" → 😌 Thư giãn → lofi + ambient
"năng lượng tràn trề muốn workout" → ⚡ Năng động → electronic + dance
"nhớ về kỷ niệm ngày xưa"        → 🍂 Hoài niệm → retro + acoustic
```

## Tính năng

| Tính năng | Mô tả |
|---|---|
| 🧠 **Nhận diện cảm xúc** | 10 class theo mô hình Valence-Arousal (GEMS/Circumplex) |
| 🔤 **Word Segmentation** | Tokenizer longest-match N-gram (v4.0), nhận diện cụm tới 5 từ |
| 📚 **Online learning** | Model cập nhật ngay từ mỗi feedback like / dislike / skip |
| 🎰 **Thompson Sampling** | Bandit học gu nhạc cá nhân qua từng phiên sử dụng |
| 🎵 **Audio re-ranking** | BPM, spectral centroid, MFCC để tinh chỉnh danh sách nhạc |
| 🌓 **Đa theme** | Midnight · Aurora · Sunset — nhớ qua localStorage |
| 📱 **PWA** | Cài như app, hỗ trợ offline với nội dung đã cache |
| ⚡ **Zero dependency** | Toàn bộ AI chỉ cần NumPy + Python stdlib |

## Công nghệ

| Tầng | Công nghệ |
|---|---|
| Backend | Python 3, Flask, Waitress |
| AI Engine | NumPy — MLP, Self-Attention, Thompson Bandit tự viết |
| Music API | Jamendo (miễn phí, full track) |
| Frontend | HTML / CSS / JS thuần (không build step) |
| Process manager | PM2 |
| Reverse proxy | IIS / Nginx |

## Cấu trúc dự án

```
MoodTunes/
├── backend/
│   ├── app.py                  # Flask entrypoint + REST API
│   ├── emotion_mlp.py          # Self-Attention MLP + online learning
│   ├── lexicon.py              # Lexicon cảm xúc tiếng Việt & rule scorer
│   ├── bandit.py               # Thompson Sampling bandit
│   ├── audio_features.py       # Re-ranking BPM / spectral / MFCC
│   ├── weights.npz             # Trọng số model đã huấn luyện
│   ├── dynamic_vocab.json      # Vocab mở rộng runtime
│   ├── requirements.txt
│   └── ecosystem.config.js     # PM2 config
└── frontend/
    └── index.html              # Single-page app (vanilla JS)
```

## Chạy nhanh

**Backend**

```bash
cd backend
pip install -r requirements.txt
python app.py
# → http://localhost:5005  |  kiểm tra: /api/health
```

**Frontend**

Mở `frontend/index.html` trong trình duyệt (hoặc static server port 5500). Khi chạy ở `localhost` / `file://`, frontend tự gọi `http://localhost:5005/api`.

**Production (PM2)**

```bash
pm2 start backend/ecosystem.config.js
```

## Biến môi trường

Tất cả đều có giá trị mặc định — không bắt buộc khi chạy local.

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `MOODTUNE_SECRET_KEY` | *(không an toàn)* | Flask secret key — **bắt buộc đổi khi deploy thật** |
| `MOODTUNE_FRONTEND` | `https://anhtaictv.me` | Origin frontend cho CORS |
| `JAMENDO_CLIENT_ID` | `cf31dbfd` | Client ID Jamendo API |
| `MOODTUNE_PYTHON` | *(path Python có waitress)* | Override interpreter cho PM2 |

## Lịch sử phiên bản

| Phiên bản | Nội dung |
|---|---|
| `v4.0` | Word Segmentation — longest-match N-gram tokenizer, sửa "vocab chết" cho 54 cụm từ |
| `v3.8` | Giao diện đa theme: Midnight / Aurora / Sunset |
| `v3.7` | Production: PWA, presence widget, Waitress, rate limiting |
| `v3.6` | Sửa rule scorer phủ định nhiều từ (`"không hề"`, `"chẳng bao giờ"`) |
| `v3.5` | Rút về 10 class Valence-Arousal (GEMS/Circumplex), chuẩn hoá emotion system |
| `v3.1` | Mở rộng 12 → 15 class: thêm tự_tin 💪, biết_ơn 🙏, tức_giận 😡 |
| `v3.0` | Thompson Sampling bandit (RLUF) + sơ đồ tri thức Canvas 2D tương tác |
| `v2.5` | Leaky ReLU + Adaptive L2 Regularization theo feedback_count |
| `v2.0` | Self-Attention (Q/K/V NumPy), audio features (BPM/MFCC), dynamic vocab |
| `v1.1` | Hybrid Rule+MLP hoàn chỉnh, online learning, Jamendo API, secret_key env |
| `v1.0` | Phiên bản đầu tiên — rule scorer + MLP cơ bản |

Báo cáo chi tiết từng phiên bản: xem các file `BaoCao_MoodTune_v*.md` trong repo.
