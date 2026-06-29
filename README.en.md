<div align="center">
  <img src="frontend/logo.png" alt="MoodTune" width="110" />

  <h1>MoodTune</h1>

  <p><strong>Type how you feel — get music that matches</strong><br/>
  AI detects emotion from Vietnamese text and recommends free, full-length tracks. No login required.</p>

  [![Version](https://img.shields.io/badge/version-v4.0-brightgreen?style=flat-square)](https://github.com/anhtaictv/MoodTunes/releases)
  [![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
  [![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=flat-square&logo=flask)](https://flask.palletsprojects.com)
  [![NumPy](https://img.shields.io/badge/NumPy_only-no_framework-013243?style=flat-square&logo=numpy)](https://numpy.org)
  [![Live](https://img.shields.io/badge/live-anhtaictv.me-1db954?style=flat-square&logo=googlechrome&logoColor=white)](https://anhtaictv.me)

  [🌐 Live Demo](https://anhtaictv.me) · [🇻🇳 Bản tiếng Việt](README.vi.md)
</div>

---

## Overview

MoodTune analyzes a Vietnamese text description of how you're feeling (or an emoji pick), infers your emotion using a self-built AI engine, and returns a curated playlist from the Jamendo API — completely free, full tracks, no account needed.

**What makes it different:** the entire AI stack is written **from scratch in pure NumPy** — no PyTorch, no TensorFlow. It combines a rule-based scorer over a hand-crafted Vietnamese emotion lexicon with a Self-Attention MLP that updates online from every user feedback signal.

```
"hôm nay mệt mỏi muốn thư giãn"  → 😌 Relaxed   → lofi + ambient
"năng lượng tràn trề muốn workout" → ⚡ Energetic → electronic + dance
"nhớ về kỷ niệm ngày xưa"         → 🍂 Nostalgic → retro + acoustic
```

## Features

| Feature | Description |
|---|---|
| 🧠 **Emotion detection** | 10-class Valence-Arousal model (GEMS/Circumplex framework) |
| 🔤 **Word segmentation** | Longest-match N-gram tokenizer (v4.0), phrases up to 5 tokens |
| 📚 **Online learning** | Model updates instantly from every like / dislike / skip |
| 🎰 **Thompson Sampling** | Bandit learns individual music taste across sessions |
| 🎵 **Audio re-ranking** | BPM, spectral centroid, MFCC via librosa to refine results |
| 🌓 **Multi-theme UI** | Midnight · Aurora · Sunset — persisted via localStorage |
| 📱 **PWA** | Installable, works offline for cached content |
| ⚡ **Zero ML dependency** | Entire AI stack on NumPy + Python stdlib only |

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3, Flask, Waitress |
| AI Engine | NumPy — custom MLP, Self-Attention, Thompson Bandit |
| Music API | Jamendo (free, full tracks) |
| Frontend | Vanilla HTML / CSS / JS (no build step) |
| Process manager | PM2 |
| Reverse proxy | IIS / Nginx |

## Project Structure

```
MoodTunes/
├── backend/
│   ├── app.py                  # Flask entrypoint + REST API
│   ├── emotion_mlp.py          # Self-Attention MLP + online learning
│   ├── lexicon.py              # Vietnamese emotion lexicon & rule scorer
│   ├── bandit.py               # Thompson Sampling bandit
│   ├── audio_features.py       # BPM / spectral / MFCC re-ranking
│   ├── weights.npz             # Trained model weights
│   ├── dynamic_vocab.json      # Runtime-expanded vocabulary
│   ├── requirements.txt
│   └── ecosystem.config.js     # PM2 config
└── frontend/
    └── index.html              # Single-page app (vanilla JS)
```

## Quick Start

**Backend**

```bash
cd backend
pip install -r requirements.txt
python app.py
# → http://localhost:5005  |  verify: /api/health
```

**Frontend**

Open `frontend/index.html` in a browser (or serve via any static server on port 5500). When running on `localhost` / `file://`, the frontend automatically targets `http://localhost:5005/api`.

**Production (PM2)**

```bash
pm2 start backend/ecosystem.config.js
```

## Environment Variables

All have safe defaults — none are required for local development.

| Variable | Default | Description |
|---|---|---|
| `MOODTUNE_SECRET_KEY` | *(insecure default)* | Flask secret key — **override in production** |
| `MOODTUNE_FRONTEND` | `https://anhtaictv.me` | Frontend origin for CORS |
| `JAMENDO_CLIENT_ID` | `cf31dbfd` | Jamendo API client ID |
| `MOODTUNE_PYTHON` | *(path to Python with waitress)* | Override PM2 interpreter path |

## Version History

| Version | Highlights |
|---|---|
| `v4.0` | Word Segmentation — longest-match N-gram tokenizer, fixed "dead vocab" for 54 multi-word phrases |
| `v3.8` | Multi-theme UI: Midnight / Aurora / Sunset |
| `v3.7` | Production: PWA, presence widget, Waitress server, rate limiting |
| `v3.6` | Rule scorer fix for multi-word negations (`"không hề"`, `"chẳng bao giờ"`) |
| `v3.5` | Reduced to 10-class Valence-Arousal (GEMS/Circumplex), standardized emotion system |
| `v3.1` | Expanded 12 → 15 classes: added confident 💪, grateful 🙏, angry 😡 |
| `v3.0` | Thompson Sampling bandit (RLUF) + interactive Canvas 2D knowledge graph |
| `v2.5` | Leaky ReLU + Adaptive L2 regularization (scaled by feedback_count) |
| `v2.0` | Self-Attention (Q/K/V in NumPy), audio features (BPM/MFCC), dynamic vocab |
| `v1.1` | Full Hybrid Rule+MLP, online learning, Jamendo API, secret_key env var |
| `v1.0` | Initial release — rule scorer + basic MLP |

Full per-version writeups (architecture diagrams, before/after comparisons, test results) are in the `BaoCao_MoodTune_v*.md` files in this repo.
