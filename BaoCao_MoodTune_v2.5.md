# MoodTune v2.5 — Báo cáo nâng cấp & So sánh với v2.0

**Phiên bản:** `v2.5` (so với `v2.0` trong `BaoCao_MoodTune_v2.0.md`)
**Tên đầy đủ:** MoodTune — AI Cảm Xúc Tự Xây (Self-Attention · Leaky ReLU · Adaptive L2) + Gợi ý nhạc Jamendo Hybrid

---

## 1. Tóm tắt thay đổi chính

v2.5 hoàn tất phần **còn thiếu cuối cùng** trong đề xuất nâng cấp `nangcap.txt` — mục
"Tối ưu hóa MLP" (Leaky ReLU + L2 Regularization động) — vốn chưa được làm ở v2.0. Đây là
bản nâng cấp **gọn, tập trung vào AI Engine**, không thay đổi kiến trúc tổng thể hay thêm
nguồn dữ liệu mới.

| # | Nội dung | Trạng thái |
|---|---|---|
| 1 | Thay **ReLU → Leaky ReLU** (`max(0.01x, x)`) ở lớp ẩn của Attention MLP | ✅ Mới |
| 2 | **Adaptive L2 Regularization**: hệ số weight decay tăng dần theo `feedback_count` | ✅ Mới |
| 3 | Expose `activation` & `l2` hiện tại qua `/api/health.architecture` | ✅ Mới |
| 4 | Cập nhật version UI: tag `v2.0 → v2.5` + modal "About" | ✅ Mới |
| 5 | **Modal Lịch sử phiên bản (Changelog)**: bấm vào tag version ở logo → hiện danh sách tên + thay đổi của v1.0 → v2.5 | ✅ Mới |

Với v2.5, **toàn bộ 4 mục yêu cầu trong `nangcap.txt`** (Embedding, Self-Attention, Leaky
ReLU + Adaptive L2, Hybrid Web Online/Offline) đã được code hóa đầy đủ.

---

## 2. Leaky ReLU thay cho ReLU (`emotion_mlp.py`)

### Vấn đề ở v2.0
Lớp ẩn của `AttentionMLP` dùng ReLU thường (`np.maximum(0, z1)`). Với online learning chạy
liên tục trên các mẫu nhỏ (mỗi lần feedback train 30 bước), một số neuron có thể rơi vào
vùng âm vĩnh viễn (`z1 ≤ 0`) → gradient luôn = 0 → neuron "chết" (Dying ReLU), không học
được nữa.

### Thay đổi
- **Forward** (`AttentionMLP.forward`):
  ```python
  a1 = np.where(z1 > 0, z1, 0.01 * z1)   # Leaky ReLU (slope 0.01)
  ```
  (trước đó: `a1 = np.maximum(0, z1)`)

- **Backward** (`AttentionMLP.backward`):
  ```python
  dz1 = da1 * np.where(self._z1 > 0, 1.0, 0.01)   # Leaky ReLU grad
  ```
  (trước đó: `dz1 = da1 * (self._z1 > 0)`)

Với `z1 ≤ 0`, neuron vẫn truyền một gradient nhỏ (×0.01) thay vì 0 hoàn toàn, giúp neuron có
thể "hồi sinh" trong các vòng học online tiếp theo.

---

## 3. Adaptive L2 Regularization (`emotion_mlp.py`)

### Vấn đề ở v2.0
`self.l2 = 1e-4` cố định trong suốt vòng đời model — không phân biệt giữa model mới (ít
feedback, cần học nhanh, ít cần regularize) và model đã học nhiều (nhiều feedback, dễ
overfit vào các mẫu gần nhất do train lại 30 bước/lần).

### Thay đổi
- `AttentionMLP.__init__`: thêm `self.l2_base = 1e-4`, `self.l2 = self.l2_base`.
- Hàm mới `AttentionMLP.update_l2(feedback_count)`:
  ```python
  self.l2 = min(self.l2_base * 5, self.l2_base * (1 + feedback_count / 100))
  ```
  → `l2` tăng tỷ lệ thuận với số lần feedback, **chặn ở 5× giá trị gốc** (`5e-4`) để không
  làm "đông cứng" trọng số khi feedback quá nhiều.
- `EmotionEngine.__init__`: gọi `self.mlp.update_l2(self.feedback_count)` ngay sau khi
  khôi phục `feedback_count` từ `weights_meta.json` → l2 đúng giá trị kể cả sau restart.
- `EmotionEngine.learn()`: gọi lại `update_l2()` sau mỗi lần `feedback_count += 1`.
- `EmotionEngine.save()`: lưu thêm field `"l2"` vào `weights_meta.json` để minh bạch trạng
  thái.

### Kết quả thực tế (sau restart PM2)
Với `feedback_count = 58` hiện tại:
```
l2 = 1e-4 * (1 + 58/100) = 1.58e-4
```
Xác nhận qua `/api/health`:
```json
"architecture": {
  "activation": "leaky_relu",
  "l2": 0.000158,
  "attention": true,
  "vocab_size": 656, "embed_dim": 32, "hidden_size": 64, "output_size": 12
}
```

---

## 4. API thay đổi

| Endpoint | Thay đổi |
|---|---|
| `GET /api/health` | `architecture` thêm 2 field mới: `"activation": "leaky_relu"`, `"l2": <float>` |
| (file) `weights_meta.json` | thêm field `"l2"` bên cạnh `feedback_count`, `alpha`, `arch`, `embed_dim`, `vocab_size` |

Không có endpoint mới, không đổi format `weights.npz` (vẫn `E, Wq, Wk, Wv, W1, b1, W2, b2`)
→ **weights v2.0 load trực tiếp được ở v2.5**, không cần pretrain lại.

---

## 5. Thay đổi Frontend (`index.html`)

### 5.1. Version bump
- Logo: `<span class="version-tag">v2.0</span>` → **`v2.5`**.
- Modal "About": `"Phiên bản v2.0 — AI Cảm Xúc Tự Xây (Self-Attention) · Jamendo"` →
  **`"Phiên bản v2.5 — Self-Attention · Leaky ReLU · Adaptive L2 · Jamendo"`**.

### 5.2. Modal "Lịch sử phiên bản" (Changelog) — MỚI hoàn toàn
- Tag version ở logo giờ **clickable** (`onclick="showVersions()"`, `cursor:pointer`,
  hover highlight).
- Modal mới `#version-overlay` hiển thị danh sách 4 phiên bản (mới nhất lên đầu, có badge
  "hiện tại" cho v2.5):
  - **v2.5 — Adaptive Learning**: Leaky ReLU + Adaptive L2.
  - **v2.0 — Self-Attention**: Embedding + Self-Attention, Dynamic Vocab/Weight Expansion,
    Audio Feature Engine, Kho nhạc Local (Hybrid Online/Offline).
  - **v1.1 — Hoàn thiện giao diện**: hiển thị version, hoàn thiện toàn bộ chức năng cơ bản.
  - **v1.0 — Phiên bản nền tảng**: Rule + MLP Hybrid Engine, Jamendo, feedback cơ bản.
- CSS mới: `.version-list`, `.version-item` (`.current` để highlight bản hiện tại),
  `.version-name`, `.version-tag-sm`, `.version-desc`.
- JS mới: `showVersions()` / `hideVersions()`, theo cùng pattern với `showAbout()`/`hideAbout()`.

---

## 6. So sánh tổng quan v2.0 vs v2.5

| Khía cạnh | v2.0 | v2.5 |
|---|---|---|
| Hiển thị version trên UI | "v2.0" | **"v2.5"** + clickable → modal Changelog |
| Activation lớp ẩn | ReLU (`max(0, x)`) | **Leaky ReLU** (`max(0.01x, x)`) |
| L2 regularization | Cố định `1e-4` | **Adaptive**, `1e-4 → 5e-4` theo `feedback_count` |
| `/api/health.architecture` | `vocab_size, embed_dim, hidden_size, output_size, attention` | + **`activation`, `l2`** |
| `weights_meta.json` | `feedback_count, alpha, arch, embed_dim, vocab_size` | + **`l2`** |
| Tương thích weights cũ | — | ✅ Tương thích trực tiếp với weights v2.0 (cùng shape) |
| Modal version history | Không có | **Có** — xem toàn bộ changelog v1.0→v2.5 trong UI |

---

## 7. Lịch sử phiên bản (cập nhật)

- **v2.5** *(hiện tại)* — Thay ReLU bằng Leaky ReLU ở lớp ẩn Attention MLP (chống Dying ReLU);
  thêm Adaptive L2 Regularization tăng dần theo số lần feedback (chặn ở 5× giá trị gốc);
  expose `activation`/`l2` qua `/api/health`; thêm modal "Lịch sử phiên bản" (changelog
  v1.0→v2.5) khi bấm vào tag version trên UI.
- **v2.0** — Self-Attention + Embedding Layer thay Bag-of-Words; Dynamic Vocab/Weight
  Expansion runtime; Audio Feature Engine (librosa, BPM/Centroid/MFCC) soft re-rank kết quả
  Jamendo; Hybrid Online/Offline với tab "Kho nhạc Local" (File System Access API,
  `/api/predict/batch`, Blob URL playback, Hybrid Playlist Mixer).
- **v1.1** — Hiển thị số phiên bản trên giao diện; hoàn thiện đầy đủ các chức năng: phân tích
  cảm xúc Hybrid AI (Bag-of-Words + MLP), gợi ý/nghe nhạc Jamendo, online learning, gợi ý cá
  nhân hoá, gợi ý theo giờ, lịch sử phân tích.
- **v1.0** — Phiên bản nền tảng: Rule + MLP Hybrid Engine, tích hợp Jamendo, feedback & online
  learning cơ bản.

> Với v2.5, toàn bộ 4 yêu cầu trong `nangcap.txt`/`capnhat_utf8.txt` đã hoàn tất. Định hướng
> còn lại trong `capnhat.txt` (Đóng gói Docker/PyInstaller/Electron - "Local Edge Deployment
> Edition") vẫn chưa triển khai.
