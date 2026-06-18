# MoodTune v3.0 — Báo cáo nâng cấp & So sánh với v2.5

**Phiên bản:** `v3.0` (so với `v2.5` trong `BaoCao_MoodTune_v2.5.md`)
**Tên đầy đủ:** MoodTune — AI Cảm Xúc Tự Xây (RLUF Bandit · Knowledge Graph · Self-Attention) + Gợi ý nhạc Jamendo Hybrid

---

## 1. Tóm tắt thay đổi chính

v3.0 triển khai **2 trong 3 ý tưởng** đề xuất ở `nangcap2.txt`:

| # | Ý tưởng (nangcap2.txt) | Trạng thái |
|---|---|---|
| 1 | **RLUF — Multi-Armed Bandit** (Thompson Sampling) tối ưu Hybrid Playlist Mixer | ✅ Đã làm |
| 2 | Phân tích cảm xúc đa phương thức (Face + Voice qua face-api.js/Web Audio API) | ❌ **Bỏ qua có chủ đích** |
| 3 | **Đồ thị tri thức cảm xúc tương tác** (Interactive Knowledge Graph, Canvas) | ✅ Đã làm |

Ý tưởng 2 bị loại vì **vi phạm triết lý "100% NumPy thuần"** của dự án — nó yêu cầu nhúng thư
viện AI ngoài (face-api.js chạy WebGL/WASM) ở phía frontend, khác hẳn hướng "tự viết AI" mà
toàn bộ engine (`emotion_mlp.py`, và giờ thêm `bandit.py`) đang theo.

Cả 2 ý tưởng được chọn đều mở rộng tự nhiên trên kiến trúc có sẵn:
- Bandit dùng **Thompson Sampling (Beta-Bernoulli, `np.random.beta`)** — code mới 100% NumPy
  thuần, state lưu JSON theo đúng pattern `weights_meta.json` / `dynamic_vocab.json`.
- Knowledge Graph tận dụng **ma trận attention `self.mlp._attn`** đã có từ v2.0 + dữ liệu
  `LEXICON` có sẵn — chỉ expose thêm qua API, vẽ bằng **Canvas 2D thuần** (không thêm D3.js,
  giữ đúng "Vanilla JS không framework").

---

## 2. RLUF: Thompson Sampling Multi-Armed Bandit (file mới `bandit.py`)

### Vấn đề ở v2.5
Hybrid Playlist Mixer (tab "Kho nhạc Local") luôn chia cứng **6 bài Online : 6 bài Local**
cho mọi cảm xúc/người dùng. Việc chọn cặp tag Jamendo (`EMOTION_TAGS[emotion]`, 4-5 cặp/cảm
xúc) cũng **random thuần** (`random.choice`) — không học được gu nghe của từng người.

### Thiết kế `ThompsonBandit` (class mới trong `bandit.py`)
Mỗi cảm xúc có 2 "bandit" độc lập, mỗi arm là 1 phân phối **Beta(a, b)**, khởi tạo
`[a,b]=[1,1]` (uniform prior), lazy-init khi gặp emotion/arm mới:

```json
{
  "source": { "<emotion>": { "online": [a,b], "local": [a,b] } },
  "tags":   { "<emotion>": { "0": [a,b], "1": [a,b], ... } }
}
```

- **`sample_mix(emotion, total=12)`**: lấy mẫu `θ_online ~ Beta(a,b)`, `θ_local ~ Beta(a,b)`,
  chia 12 bài theo tỉ lệ `θ_online / (θ_online + θ_local)` → `(n_online, n_local)`.
- **`sample_tag_index(emotion, n_options)`**: lấy mẫu Beta cho từng cặp tag trong
  `EMOTION_TAGS[emotion]`, trả `argmax` → cặp tag "đáng thử nhất" lúc này.
- **`update_source(emotion, source, reward)` / `update_tag(emotion, tag_idx, reward)`**:
  `reward=+1` (Like) → `a += 1`; `reward=-1` (Dislike/Bỏ qua) → `b += 1`. Tự `save()` ngay.
- **`get_summary(emotion)`**: tỉ lệ kỳ vọng `a/(a+b)` (mean Beta), dùng hiển thị UI ổn định
  (không sample ngẫu nhiên mỗi lần load).
- State lưu `bandit_state.json` (cạnh `weights.npz`, `weights_meta.json`).

### Tích hợp `app.py`
- Khởi tạo `mab = ThompsonBandit("bandit_state.json")` cạnh `engine`.
- `pick_emotion_tags_bandit(emotion)`: thay `random.choice` bằng `mab.sample_tag_index(...)`
  trong `/api/music/search` (chỉ khi tìm theo cảm xúc, không áp dụng cho tìm thủ công `q=`).
  Response trả thêm `"tag_arm": <int|null>`.
- **Endpoint mới `GET /api/mix-ratio?emotion=<emo>`**:
  ```json
  { "emotion": "vui_ve", "online": 8, "local": 4,
    "ratio": {"online": 0.67, "local": 0.33} }
  ```
- **`/api/track/event` mở rộng**:
  - Thêm `type="next"` (bấm "Bỏ qua") bên cạnh `play/like/dislike`.
  - Nhận thêm `source` (`"online"`/`"local"`), `tag_arm` (int).
  - Với `like/dislike/next` (có `emotion`): `reward = +1` nếu `like`, ngược lại `-1` →
    `mab.update_source(emotion, source, reward)`; nếu `source=="online"` và có `tag_arm` →
    `mab.update_tag(emotion, tag_arm, reward)`.

### Thay đổi Frontend (`index.html`)
- `loadMusicSongs(emotion)`: gọi `GET /api/mix-ratio` (best-effort, lỗi → fallback 6/4 → 6/6
  như cũ), dùng `mix.local`/`mix.online` thay cho hằng số `6` khi chia Online/Local.
- Card "Gợi ý nhạc" hiện thêm nhãn **"🎯 Gu của bạn: Online X% · Local Y%"** (`#mix-hint`,
  lấy từ `ratio` — tỉ lệ kỳ vọng, ổn định, không nhảy số mỗi lần load).
- Global `currentTagArm` lưu `tag_arm` trả về từ `/api/music/search`, gửi kèm trong mọi
  `recordBehavior()`.
- **Nút mới "⏭ Bỏ qua"** trong `fb-row` (luôn hiện) → `skipSong()` → ghi `type="next"` +
  chuyển bài kế tiếp. Like/Dislike chỉ hiện với nhạc Online (nhạc Local ẩn 2 nút này, vẫn có
  nút Bỏ qua).
- `recordBehavior()` gửi thêm `source` (`'local'`/`'online'`) và `tag_arm`.

### Kết quả thực tế
- Người dùng càng Like nhạc Online → `sample_mix` dần nghiêng tỉ lệ Online cao hơn cho lần
  tìm tiếp theo (và ngược lại với Local/Dislike/Bỏ qua).
- Với từng cảm xúc, cặp tag Jamendo nào được Like nhiều → `sample_tag_index` ưu tiên chọn lại
  cặp đó nhiều hơn (nhưng vẫn có xác suất "khám phá" cặp khác nhờ bản chất Thompson Sampling).
- Đã test round-trip: `sample_mix` trả tổng đúng `total`, `update_source`/`update_tag` cập
  nhật đúng `(a,b)`, `save()`/load lại từ `bandit_state.json` khớp dữ liệu.

---

## 3. Interactive Emotion Knowledge Graph (`emotion_mlp.py` + Canvas)

### Vấn đề ở v2.5
12 thanh progress bar (`score-bars`) cho thấy AI "nghĩ gì" nhưng khá khô khan — không thể
hiện được **vì sao** một từ trong câu lại đẩy điểm về một cảm xúc cụ thể, cũng không tận
dụng trực quan ma trận Self-Attention đã tính sẵn.

### Backend — refactor tokenization + `graph_tokens`
- `to_token_ids(text)` (v2.0) chỉ trả về danh sách ID, bỏ qua từ OOV → không đủ thông tin để
  vẽ graph. Refactor thành 2 hàm:
  - **`_tokenize(text, max_len=24)`** (mới): trả `[(token_str, token_id_or_None), ...]`, giữ
    cả token OOV (id=`None`) — cùng logic match bigram-trước-unigram như cũ.
  - **`to_token_ids(text)`**: viết lại = `[tid for _, tid in _tokenize(text) if tid is not None]`
    — **chữ ký không đổi**, không ảnh hưởng `AttentionMLP`/`learn`/`pretrain`.
- `EmotionEngine.predict(text)` thêm field **`graph_tokens`**:
  - Với mỗi token có `tid is not None`, tính `attention = attn_received[i] / max(attn_received)`
    trong đó `attn_received = self.mlp._attn.sum(axis=0)` (tổng attention các token khác "đổ
    vào" token này, chuẩn hoá [0,1]).
  - `emotions`: quét `LEXICON[emo].get(token_str, 0)` cho 12 cảm xúc, giữ lại các giá trị > 0.
  - Kết quả mỗi token: `{"text": "...", "attention": 0.0-1.0, "emotions": {"cang_thang": 3.0, ...}}`.

Test thực tế với câu *"hôm nay mệt quá, stress deadline áp lực"* (kết quả `cang_thang`,
86.1%):
```
mệt       attention=0.992  emotions={cang_thang: 1.5}
stress    attention=0.977  emotions={cang_thang: 3.0}
deadline  attention=1.000  emotions={cang_thang: 2.5, tap_trung: 1.5}
áp lực    attention=0.991  emotions={cang_thang: 3.0}
```

### API
- `POST /api/predict` trả thêm field **`"graph": result["graph_tokens"]`**.

### Frontend — Canvas 2D thuần (`renderKnowledgeGraph`)
- Card mới **"🕸️ Sơ đồ tri thức AI"** (`#kg-canvas`) nằm ngay trong khung kết quả phân tích,
  dưới `score-bars`.
- Vẽ bằng Canvas 2D (scale theo `devicePixelRatio`, không thêm thư viện):
  - **12 bong bóng cảm xúc** xếp vòng ngoài (từ `data.scores`), kích thước & độ sáng theo
    `score`.
  - **Tối đa 8 từ khoá** có `attention` cao nhất xếp vòng trong, kích thước theo `attention`.
  - **Tia năng lượng**: với mỗi từ khoá có `emotions` khớp LEXICON, vẽ đường nối tới bong
    bóng cảm xúc tương ứng — độ rộng/độ sáng tỉ lệ `emotions[emo] × attention`.
  - **Animation ~2.5s** (`requestAnimationFrame`): các tia "nhấp nháy" theo `sin(t)`, sau đó
    dừng ở trạng thái mờ tĩnh.
- Gọi `renderKnowledgeGraph(data.graph, data.scores)` ngay sau khi build `score-bars` trong
  `analyze()` — mỗi lần phân tích câu mới, đồ thị tự vẽ lại theo `alpha`/attention hiện tại.

---

## 4. API thay đổi

| Endpoint | Thay đổi |
|---|---|
| `POST /api/predict` | Trả thêm `"graph": [{"text","attention","emotions"}, ...]` |
| `GET /api/music/search` | Trả thêm `"tag_arm": <int|null>`; chọn tag bằng Thompson Sampling thay vì random |
| `GET /api/mix-ratio` *(mới)* | `?emotion=<emo>` → `{"online", "local", "ratio":{"online","local"}}` |
| `POST /api/track/event` | `type` nhận thêm `"next"`; nhận thêm `source`, `tag_arm`; cập nhật `bandit_state.json` theo reward |
| (file mới) `bandit_state.json` | State Beta(a,b) cho `source`/`tags` theo từng cảm xúc |

---

## 5. Thay đổi Frontend (`index.html`)

### 5.1. Version bump
- Logo: `<span class="version-tag">v2.5</span>` → **`v3.0`**.
- Modal "About": `"Phiên bản v2.5 — Self-Attention · Leaky ReLU · Adaptive L2 · Jamendo"` →
  **`"Phiên bản v3.0 — RLUF Bandit · Knowledge Graph · Self-Attention · Jamendo"`**.

### 5.2. Modal "Lịch sử phiên bản" (Changelog)
- Thêm entry mới lên đầu, đánh dấu **"hiện tại"**:
  - **v3.0 — RLUF & Knowledge Graph**: Thompson Sampling Multi-Armed Bandit học gu nhạc từ
    Like/Dislike/Bỏ qua (tối ưu tỉ lệ Online:Local + chọn tag Jamendo); Sơ đồ tri thức cảm
    xúc tương tác (Canvas, tia năng lượng theo Attention).
  - Badge "hiện tại" bỏ khỏi entry v2.5.

### 5.3. Card "Nhập cảm xúc"
- Thêm khối **"🕸️ Sơ đồ tri thức AI"** (`#kg-canvas`) dưới `score-bars`.

### 5.4. Card "Gợi ý nhạc"
- Thêm nhãn **"🎯 Gu của bạn: Online X% · Local Y%"** (`#mix-hint`) ở góc tiêu đề.
- `fb-row` (Now Playing) có thêm nút **"⏭ Bỏ qua"**, luôn hiển thị; Like/Dislike chỉ hiện
  với nhạc Online.

---

## 6. So sánh tổng quan v2.5 vs v3.0

| Khía cạnh | v2.5 | v3.0 |
|---|---|---|
| Hiển thị version trên UI | "v2.5" | **"v3.0"** + entry mới trong Changelog |
| Tỉ lệ trộn nhạc Online:Local | Cố định 6:6 | **Thompson Sampling**, học theo Like/Dislike/Bỏ qua từng cảm xúc |
| Chọn cặp tag Jamendo | `random.choice` trong `EMOTION_TAGS` | **Thompson Sampling** (`tag_arm`), ưu tiên cặp được Like nhiều |
| Phản hồi người dùng | `play/like/dislike` | + **`next`** (nút "⏭ Bỏ qua") — dùng làm reward cho Bandit |
| `/api/predict` | `emotion, label, scores, model_info` | + **`graph`** (token + attention + emotions) |
| Trực quan hoá AI | 12 thanh progress bar | + **Sơ đồ tri thức Canvas** (bong bóng cảm xúc + tia năng lượng theo Attention) |
| File state mới | — | **`bandit_state.json`** (Beta params source/tags theo cảm xúc) |
| Thư viện ngoài thêm vào | — | **Không** (100% NumPy + Canvas 2D thuần, đúng triết lý dự án) |

---

## 7. Lịch sử phiên bản (cập nhật)

- **v3.0** *(hiện tại)* — Tự cài đặt **Thompson Sampling Multi-Armed Bandit** (Beta-Bernoulli,
  NumPy thuần, `bandit.py`) học gu nhạc từng người dùng theo từng cảm xúc từ phản hồi
  Like/Dislike/Bỏ qua — tối ưu tỉ lệ trộn nhạc Online:Local (`/api/mix-ratio`) và chọn bộ tag
  Jamendo (`tag_arm`). Thêm **Sơ đồ tri thức cảm xúc tương tác** (`/api/predict.graph` +
  Canvas 2D `renderKnowledgeGraph`): từ khoá trong câu phát "tia năng lượng" theo trọng số
  Attention bắn vào 12 bong bóng cảm xúc, có animation.
- **v2.5** — Thay ReLU bằng Leaky ReLU ở lớp ẩn Attention MLP (chống Dying ReLU); thêm
  Adaptive L2 Regularization tăng dần theo số lần feedback (chặn ở 5× giá trị gốc); expose
  `activation`/`l2` qua `/api/health`; thêm modal "Lịch sử phiên bản" (changelog v1.0→v2.5).
- **v2.0** — Self-Attention + Embedding Layer thay Bag-of-Words; Dynamic Vocab/Weight
  Expansion runtime; Audio Feature Engine (librosa, BPM/Centroid/MFCC) soft re-rank kết quả
  Jamendo; Hybrid Online/Offline với tab "Kho nhạc Local" (File System Access API,
  `/api/predict/batch`, Blob URL playback, Hybrid Playlist Mixer).
- **v1.1** — Hiển thị số phiên bản trên giao diện; hoàn thiện đầy đủ các chức năng: phân tích
  cảm xúc Hybrid AI (Bag-of-Words + MLP), gợi ý/nghe nhạc Jamendo, online learning, gợi ý cá
  nhân hoá, gợi ý theo giờ, lịch sử phân tích.
- **v1.0** — Phiên bản nền tảng: Rule + MLP Hybrid Engine, tích hợp Jamendo, feedback & online
  learning cơ bản.

> Định hướng còn lại: ý tưởng 2 trong `nangcap2.txt` (đầu vào đa phương thức Face/Voice) bị
> loại có chủ đích vì xung đột triết lý "100% NumPy thuần". Đóng gói Docker/PyInstaller
> (`capnhat.txt` — "Local Edge Deployment Edition") vẫn chưa triển khai.
