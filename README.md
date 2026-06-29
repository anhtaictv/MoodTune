<div align="center">
  <img src="MoodTune.png" alt="MoodTune" width="120" />

  # MoodTune

  **AI-powered music recommendation based on emotion detected from Vietnamese text**

  [![Version](https://img.shields.io/badge/version-v4.0-blue?style=flat-square)](https://github.com/anhtaictv/MoodTunes/releases)
  [![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
  [![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=flat-square&logo=flask)](https://flask.palletsprojects.com)
  [![NumPy](https://img.shields.io/badge/NumPy-only-013243?style=flat-square&logo=numpy)](https://numpy.org)

  [🌐 Live Demo](https://anhtaictv.me) · [🇻🇳 Bản tiếng Việt](README.vi.md)
</div>

---

## Overview

MoodTune analyzes a piece of Vietnamese text, infers the user's emotion using a custom AI engine, and returns matching music tracks from the Jamendo API — free, no login required.

The AI engine is built **from scratch in pure NumPy** (no PyTorch, no TensorFlow): a hybrid of a hand-crafted rule scorer over a Vietnamese emotion lexicon and a self-attention MLP that learns online from user feedback.

## Features

- **Emotion detection** — 10-class Valence-Arousal model aligned with the GEMS/Circumplex framework
- **Self-attention MLP** — Embedding + Q/K/V attention written in NumPy; understands word order and negation
- **Online learning** — model updates in real time from every feedback signal (like / dislike / skip)
- **Thompson Sampling bandit** — learns individual music taste across sessions
- **Audio feature re-ranking** — BPM, spectral centroid, MFCC via librosa to refine track selection
- **Multi-theme UI** — Dark (Midnight), Light (Aurora), Vivid (Sunset); remembered via localStorage
- **PWA** — installable, works offline for cached content
- **Zero ML framework dependency** — entire AI stack runs on NumPy + Python stdlib

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3, Flask, Waitress |
| AI Engine | NumPy (custom MLP, attention, bandit) |
| Music API | Jamendo |
| Frontend | Vanilla HTML / CSS / JS (no build step) |
| Process manager | PM2 |
| Reverse proxy | IIS / Nginx |

## Project Structure

```
moodtune/
├── backend/
│   ├── app.py                  # Flask entrypoint
│   ├── emotion_mlp.py          # Self-attention MLP + online learning
│   ├── lexicon.py              # Vietnamese emotion lexicon & rule scorer
│   ├── bandit.py               # Thompson Sampling bandit
│   ├── audio_features.py       # BPM / spectral / MFCC re-ranking
│   ├── weights.npz             # Trained model weights
│   ├── dynamic_vocab.json      # Runtime vocabulary
│   ├── requirements.txt
│   └── ecosystem.config.js     # PM2 config
└── frontend/
    └── index.html              # Single-page app
```

## Quick Start

**Backend**

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Runs at `http://localhost:5005`. Check `/api/health` to verify.

**Frontend**

Open `frontend/index.html` in a browser (or serve via any static server on port 5500). When running on `localhost` / `file://`, the frontend automatically targets `http://localhost:5005/api`.

**Production (PM2)**

```bash
pm2 start backend/ecosystem.config.js
```

## Environment Variables

All variables have defaults — none are required for local development.

| Variable | Default | Description |
|---|---|---|
| `MOODTUNE_SECRET_KEY` | `moodtune_secret_2024` | Flask secret key. Override in production. |
| `MOODTUNE_FRONTEND` | `https://anhtaictv.me` | Frontend origin for CORS. |
| `JAMENDO_CLIENT_ID` | `cf31dbfd` | Jamendo API client ID. |

## Production Deployment

The frontend calls `/api` (relative path). Set up a reverse proxy (IIS or Nginx) to route `/api` → Flask backend on port 5005. See `frontend/web.config` for a sample IIS configuration.

## Model Data

| File | Contents |
|---|---|
| `weights.npz` | MLP weights + embedding matrix |
| `weights_meta.json` | Vocabulary metadata |
| `weights_replay.json` | Replay buffer for online learning |
| `dynamic_vocab.json` | Runtime-expanded vocabulary |

## Version History

Click the version tag in the app UI to view the changelog. Full writeups (architecture diagrams, before/after comparisons, test results) are linked below.

| Version | Name | Highlights |
|---|---|---|
| `v4.0` | Word Segmentation | Generalized tokenizer to **longest-match N-gram** (up to 5 words); fixed "dead vocab" bug for 54 unreachable multi-word phrases. ([report](BaoCao_MoodTune_v4.0.md)) |
| `v3.8` | Multi-theme UI | 3 themes: Midnight / Aurora / Sunset; WCAG AA contrast on all themes; theme-aware knowledge graph. ([report](BaoCao_MoodTune_v3.8.md)) |
| `v3.7` | Production reliability | PWA manifest, presence widget, Waitress server, rate limiting, auto-heal on vocab mismatch. ([report](BaoCao_MoodTune_v3.7.md)) |
| `v3.6` | Negation fix | Fixed rule scorer to handle multi-word negations (`"không hề"`, `"chẳng bao giờ"`, etc.). ([report](BaoCao_MoodTune_v3.6.md)) |
| `v3.5` | Valence-Arousal | Reduced to 10 emotion classes aligned with the GEMS/Circumplex model. ([report](BaoCao_MoodTune_v3.5.md)) |
| `v3.1` | New emotions | Added confident 💪, grateful 🙏, angry 😡 — 12 → 15 classes. ([report](BaoCao_MoodTune_v3.1.md)) |
| `v3.0` | Bandit + Knowledge Graph | Thompson Sampling bandit for music taste; interactive Canvas 2D emotion knowledge graph. ([report](BaoCao_MoodTune_v3.0.md)) |
| `v2.5` | Adaptive Learning | ReLU → Leaky ReLU; adaptive L2 regularization; in-app changelog modal. ([report](BaoCao_MoodTune_v2.5.md)) |
| `v2.0` | Self-Attention | Embedding + self-attention (Q,K,V) in NumPy; audio feature engine; dynamic vocab expansion. ([report](BaoCao_MoodTune_v2.0.md)) |
| `v1.1` | UI polish | Version badge; personalized & time-of-day recommendations; analysis history. ([report](BaoCao_MoodTune_v1.1.md)) |
| `v1.0` | Foundation | Rule scorer + MLP hybrid, Jamendo API, basic online learning. |

Additional documents: [`BaoCao_MoodTune_TongQuan.md`](BaoCao_MoodTune_TongQuan.md) (overall architecture) and [`BaoCao_MoodTune_DacTrung_AIEngine_API.md`](BaoCao_MoodTune_DacTrung_AIEngine_API.md) (AI engine & API deep dive).
