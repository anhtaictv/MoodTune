# So sánh 3 mô hình AI Cảm Xúc MoodTune — Tài liệu chuẩn bị thuyết trình

> **Quy ước đặt tên cho buổi bảo vệ** (theo yêu cầu): gọi 3 giai đoạn kiến trúc của
> `AttentionMLP` (`backend/emotion_mlp.py`) là **Model 1 / 2 / 3**.
>
> | Tên gọi | Tương ứng | Trạng thái |
> |---|---|---|
> | **Model 1** | Kiến trúc Self-Attention dùng ReLU thường (giai đoạn `v2.0`–`v2.4`) | Đã thay thế, không còn chạy |
> | **Model 2** | Kiến trúc hiện tại — Leaky ReLU + Adaptive L2 (`v2.5` → `v3.8`, **đang chạy production**) | Đang dùng thật |
> | **Model 3** | Đề xuất nâng cấp Word Segmentation tổng quát | **CHƯA code, mới ở mức phân tích/đề xuất** |

⚠️ **Lưu ý quan trọng khi thuyết trình:** Model 3 trong tài liệu này là **đề xuất**, dựa
trên việc đọc trực tiếp code đang chạy thật (`backend/emotion_mlp.py`,
`backend/lexicon.py`) để tìm điểm nghẽn kỹ thuật cụ thể — **không phải đã code và đo kết
quả**. Nếu giảng viên hỏi "chạy thử chưa, số liệu đâu" thì câu trả lời thành thật là
"chưa, đây là phần phân tích/đề xuất nâng cấp tiếp theo", xem mục 6.

---

## 1. Pipeline chung của cả 3 model (không đổi)

```
Câu nhập vào
   │
   ├─► RULE SCORER (lexicon.py + rule_score())
   │     từ điển trọng số + xử lý phủ định (1-3 từ) → vector điểm 10 lớp cảm xúc
   │
   └─► ATTENTION MLP LEARNER (AttentionMLP, numpy thuần)
         Embedding(VOCAB_SIZE, 32) → Self-Attention(Q,K,V) → mean-pool
           → Dense(32→64, activation) → Dense(64→10, Softmax)

   final = alpha * rule + (1 - alpha) * mlp
   alpha = max(0.35, 0.85 - 0.5 * feedback_count / (feedback_count + 40))
```

`alpha` giảm dần khi có nhiều feedback (tin MLP hơn) nhưng **không bao giờ xuống dưới
0.35** — rule-based luôn giữ ít nhất 35% trọng số quyết định cuối cùng. Điều này không
đổi qua cả 3 model.

---

## 2. Model 1 — ReLU (giai đoạn v2.0–v2.4)

- Activation lớp ẩn: **ReLU thường** — `a1 = max(0, z1)`.
- L2 regularization: cố định `1e-4` suốt vòng đời model.
- **Vấn đề:** online learning chạy liên tục trên mẫu nhỏ (mỗi lần feedback train 30 bước
  lặp, `learn(steps=30)`) → một số neuron rơi vào vùng `z1 ≤ 0` **vĩnh viễn**, gradient
  luôn bằng 0 ở neuron đó → neuron "chết" (Dying ReLU), không học được nữa cho cả các câu
  sau này, kể cả khi câu đó liên quan tới neuron đã chết.
- Đã được thay thế hoàn toàn từ `v2.5`, không còn dấu vết trong code hiện tại (chỉ còn
  trong báo cáo lịch sử `BaoCao_MoodTune_v2.5.md`).

## 3. Model 2 — Leaky ReLU + Adaptive L2 (hiện tại, v2.5 → v3.8)

- Activation lớp ẩn: **Leaky ReLU** — `a1 = max(0.01·z1, z1)` (`emotion_mlp.py:351`,
  gradient tương ứng ở `emotion_mlp.py:371`). Neuron ở vùng âm vẫn nhận gradient nhỏ
  (×0.01) → có thể "hồi sinh" ở các vòng học online sau, không chết vĩnh viễn như Model 1.
- L2 **động**: `l2 = min(5e-4, 1e-4 * (1 + feedback_count/100))` (`emotion_mlp.py:433`) —
  tăng dần theo số lần feedback, chặn ở 5× giá trị gốc để chống overfit khi học online
  nhiều lần trên mẫu nhỏ, nhưng không "đông cứng" trọng số khi model còn mới/ít feedback.
- **Tokenization** (`_tokenize()`, `emotion_mlp.py:176-196`): quét câu, **ưu tiên thử ghép
  bigram trước unigram** — nhưng chỉ ghép được nếu cụm 2 từ đó **đã có sẵn** trong
  `VOCAB_IDX` (từ `LEXICON` tĩnh trong `lexicon.py`, hoặc cụm được tự học qua
  `dynamic_lexicon.json` sau khi có feedback chứa cụm đó). Giới hạn: tối đa **đúng 2 từ**,
  và chỉ hoạt động với cụm **đã từng được đăng ký** — không tổng quát cho từ ghép bất kỳ
  trong tiếng Việt.
- **Điểm quan trọng dễ bị hỏi vặn:** `to_token_ids()` (`emotion_mlp.py:199-201`) loại bỏ
  hoàn toàn token có `tid is None` (từ chưa từng nằm trong `VOCAB_IDX`). Vì `VOCAB` chỉ
  được dựng từ các khoá trong `LEXICON` (`emotion_mlp.py:57-64`), **mọi từ không phải từ
  cảm xúc — bao gồm cả các từ phủ định `"không", "chẳng", "chưa"`... (`lexicon.py:65`) —
  hoàn toàn không có embedding và bị loại khỏi input của Attention MLP.** Nói cách khác:
  **nhánh học sâu (Embedding/Attention) không tự "nhìn thấy" và không tự học được phủ
  định** — toàn bộ việc nhận diện phủ định hiện tại do Rule Scorer xử lý cứng theo danh
  sách `NEGATIONS`, và `alpha ≥ 0.35` là cơ chế đảm bảo tín hiệu đó luôn có trọng số trong
  quyết định cuối.

## 4. Model 3 — Word Segmentation tổng quát (ĐỀ XUẤT, chưa code)

Giữ nguyên toàn bộ phần đã tốt của Model 2 (Leaky ReLU, Adaptive L2) — chỉ thay đổi tầng
**tiền xử lý/tokenization**:

1. **Word segmentation tổng quát bằng longest-match**: biên soạn một từ điển từ ghép
   tiếng Việt riêng (pure Python list/dict, không gọi thư viện NLP ngoài như
   underthesea/pyvi/VnCoreNLP — đúng tinh thần "tự code, chỉ NumPy"), rồi quét câu theo
   thuật toán so khớp từ dài nhất trước (longest-match, độ phức tạp gần tuyến tính nhờ
   tra dictionary O(1) mỗi vị trí). Khác với Model 2, cách này áp dụng cho **bất kỳ từ
   ghép nào có trong từ điển**, không bị giới hạn ở các cụm đã từng xuất hiện trong
   feedback, và không bị chặn cứng ở 2 từ.
2. **(Mở rộng tuỳ chọn, nên làm cùng lúc)** Đăng ký các từ trong `NEGATIONS` (và có thể cả
   từ nhấn mạnh như "rất", "hơi", "cực") thành các **token chức năng** riêng trong `VOCAB`
   — để nhánh Attention có cơ hội tự học pattern phủ định/nhấn mạnh qua dữ liệu thật, thay
   vì phụ thuộc 100% vào rule cứng như Model 2.

### Hệ quả kỳ vọng
- Chuỗi token đưa vào Attention **ngắn hơn** và **đúng đơn vị nghĩa hơn** (ví dụ "làm
  việc" thành 1 token thay vì 2 token rời rạc bị bag-of-context hoá qua attention).
- Học **hiệu quả hơn với cùng lượng dữ liệu hạn chế** (~3500 mẫu feedback hiện tại) — vì
  mô hình không phải tự "đoán" mối liên hệ giữa các âm tiết của cùng một từ ghép từ vài
  nghìn mẫu, mà mối liên hệ đó đã được mã hoá sẵn ở tầng tokenizer.
- Nếu làm luôn bước 2: hệ thống có thêm một con đường thứ hai (học được, không chỉ rule
  cứng) để nhận diện phủ định — hữu ích cho các kiểu phủ định/nhấn mạnh chưa được liệt kê
  thủ công trong `NEGATIONS`.

### Đánh đổi (trade-off) — nên nói thẳng khi thuyết trình
- Từ điển từ ghép biên soạn tay sẽ không bao giờ đầy đủ như một bộ tokenizer tiếng Việt
  được nghiên cứu/huấn luyện chuyên sâu (underthesea...) — đổi lại là giữ đúng yêu cầu
  "tự code 100% bằng NumPy", và phạm vi từ ghép cần cho bài toán cảm xúc nhỏ hơn nhiều so
  với một tokenizer tổng quát cho mọi domain.
- Đổi tokenizer làm thay đổi `VOCAB_SIZE` và ý nghĩa từng `token_id` → **không tương thích
  trực tiếp với `weights.npz` hiện có**, cần pretrain lại từ đầu (xem mục 6).

---

## 5. Bảng so sánh tổng hợp

| Khía cạnh | Model 1 (ReLU) | Model 2 (hiện tại) | Model 3 (đề xuất) |
|---|---|---|---|
| Activation lớp ẩn | ReLU | **Leaky ReLU** | Leaky ReLU (kế thừa) |
| Dying ReLU | Có rủi ro | Đã vá | Đã vá (kế thừa) |
| L2 regularization | Cố định `1e-4` | **Adaptive** `1e-4→5e-4` | Adaptive (kế thừa) |
| Tokenization cho MLP | Unigram thuần (tách theo âm tiết) | Bigram-aware, **chỉ với cụm đã đăng ký**, tối đa 2 từ | **Word segmentation tổng quát**, không giới hạn 2 từ, dựa từ điển riêng |
| Token phủ định (`không`, `chẳng`...) có vào `VOCAB`/Attention không | Không | **Không** — bị `to_token_ids()` loại bỏ, chỉ Rule Scorer xử lý | Có thể có, nếu làm thêm bước đăng ký function-token |
| Tương thích `weights.npz` khi chuyển sang | — | Cần pretrain lại (đổi activation) | Cần pretrain lại (đổi `VOCAB_SIZE`/ý nghĩa token_id) |
| Trạng thái | Đã thay thế từ `v2.5` | **Đang chạy production** | **Đề xuất, chưa code** |

---

## 6. Ví dụ minh hoạ cụ thể — "không hề tập trung làm việc"

Hữu ích để trả lời câu "cho ví dụ cụ thể" của giảng viên.

**Rule Scorer** (giống nhau ở cả 3 model, không đổi từ v3.6): bắt đúng `"không hề"` trong
`NEGATIONS` đứng ngay trước bigram `"tập trung"` (có trong `LEXICON["focused"]`) →
`_is_negated()` trả `True` → điểm "focused" bị nhân `-0.6` (đảo dấu) thay vì cộng dương.
Phần này đã đúng, không liên quan đến đề xuất Model 3.

**Nhánh Attention MLP (Model 2 — hiện tại):**
- `"không"`, `"hề"` → không có trong `VOCAB_IDX` → bị `to_token_ids()` loại bỏ hoàn toàn.
- `"tập trung"` → có trong `LEXICON` → ghép đúng thành 1 token.
- `"làm việc"` → nếu chưa từng xuất hiện trong feedback gắn với cảm xúc nào, tách thành 2
  token rời `"làm"`, `"việc"` — và nếu cả hai cũng chưa từng học, **cả hai cũng bị loại
  bỏ luôn**.
- → Chuỗi vào Attention thực tế chỉ còn **đúng 1 token**: `"tập trung"`. Toàn bộ phần phủ
  định và phần còn lại của câu hoàn toàn vô hình với nhánh học sâu — quyết định "có phủ
  định hay không" của hệ thống ở câu này gần như 100% do Rule Scorer gánh.

**Nhánh Attention MLP (Model 3 — đề xuất):**
- `"làm việc"` được nhận diện là 1 từ ghép qua từ điển segmentation tổng quát (dù chưa
  từng xuất hiện trong feedback/lexicon cảm xúc) → giữ đúng 1 đơn vị nghĩa.
- Nếu làm thêm bước đăng ký function-token: `"không hề"` cũng trở thành 1 token riêng mà
  Attention nhìn thấy được — cho mô hình thêm một nguồn tín hiệu để tự học phủ định qua dữ
  liệu, không chỉ phụ thuộc rule cứng.

---

## 7. Trả lời: dùng Model 3 + chạy thêm một đợt huấn luyện thì có thông minh hơn nữa không?

**Có — và đây là hai cơ chế bổ trợ nhau, không thay thế nhau:**

- **Model 3 (đổi tokenizer)** nâng "trần năng lực" (representation ceiling) — tức loại
  thông tin mà mô hình *có khả năng* học được, với cùng lượng dữ liệu.
- **Chạy thêm huấn luyện** (`pretrain()` với `SEED_DATA` mở rộng, hoặc tích luỹ thêm
  feedback qua `learn()`) quyết định mô hình *tiến gần* tới trần năng lực đó tới đâu.

| Tình huống | Kỳ vọng |
|---|---|
| Chỉ đổi Model 3, không train thêm | Có lợi ngay phần tokenization (embedding gán đúng đơn vị nghĩa từ đầu) nhưng các trọng số `Wq/Wk/Wv/W1/W2` vẫn là *trọng số mới khởi tạo lại* (vì đổi `VOCAB_SIZE` không tương thích `weights.npz` cũ) — cần ít nhất chạy lại `pretrain()` trên `SEED_DATA` để có một baseline dùng được. |
| Chỉ train thêm, giữ Model 2 | Bị chặn ở giới hạn cấu trúc hiện tại (token phủ định/OOV vẫn bị `to_token_ids()` lọc mất) dù feed bao nhiêu mẫu cũng không vượt qua được giới hạn đó. |
| **Model 3 + train thêm (pretrain lại + tích luỹ feedback)** | Cải thiện rõ rệt nhất — vì lượng dữ liệu hiện có (~3500 feedback + `SEED_DATA`) được tận dụng hiệu quả hơn nhờ tokenization đúng đơn vị nghĩa, đúng tinh thần câu hỏi gốc "học hiệu quả hơn với ít dữ liệu" đã trao đổi trước đó. |

**Ràng buộc thực tế cần nói rõ:** đổi tokenizer làm thay đổi ý nghĩa `token_id` →
**không** thể nạp trực tiếp `weights.npz` hiện có, phải pretrain lại từ đầu — giống hệt
mỗi lần đổi kiến trúc output trước đây (ví dụ `v3.1` thêm class, `v3.5` đổi từ 15→10
class). Đây không phải rủi ro mất dữ liệu: toàn bộ feedback cũ vẫn còn trong
`weights_replay.json` (tối đa 500 mẫu gần nhất, dùng để train lại ngay) và
`feedback_log.jsonl` (toàn bộ lịch sử) — `pretrain()` (`emotion_mlp.py:601-627`) đã có sẵn
cơ chế nạp `SEED_DATA + replay` để huấn luyện lại từ đầu mà không cần thu thập dữ liệu mới.

---

## 8. Câu hỏi giảng viên có thể hỏi + gợi ý trả lời

**Q: Vì sao không dùng thư viện tách từ tiếng Việt có sẵn (underthesea, pyvi,
VnCoreNLP)?**
A: Yêu cầu của đồ án là tự xây engine bằng NumPy thuần, không phụ thuộc framework/thư viện
NLP ngoài, để thể hiện việc tự hiểu và tự cài thuật toán. Word segmentation bằng
longest-match trên từ điển tự biên soạn vẫn nằm trong tinh thần "tự code" — chỉ là một
thuật toán tra-từ-điển, không gọi API/model ngoài nào.

**Q: Model 3 đã chạy thử chưa, số liệu accuracy trước/sau thế nào?**
A: Chưa — đây là phần phân tích/đề xuất nâng cấp, xác định được điểm nghẽn kỹ thuật cụ thể
bằng cách đọc trực tiếp code đang chạy thật (`to_token_ids()` lọc mất token phủ định/OOV,
bigram giới hạn trong phạm vi từ điển cảm xúc). Bước tiếp theo nếu triển khai là dùng tập
feedback hiện có (~3500 mẫu) làm validation set, đo accuracy trước/sau khi đổi tokenizer.

**Q: Sao biết Model 3 sẽ tốt hơn nếu chưa đo thực nghiệm?**
A: Dự đoán có cơ sở kỹ thuật rõ ràng (từ quan sát trực tiếp trong code, không phải đoán
mò): tiếng Việt mang nghĩa theo từ chứ không theo âm tiết, và nhánh học sâu hiện tại đang
*mất hoàn toàn* tín hiệu phủ định/từ ngoài-từ-điển do bị lọc ở tầng tokenize — nhưng kết
luận cuối cùng vẫn cần đo thực nghiệm trước khi khẳng định mức cải thiện cụ thể.

**Q: Đổi tokenizer có mất kiến thức đã học (~3500 feedback) không?**
A: Không — `VOCAB_SIZE` đổi nên phải pretrain lại, nhưng dữ liệu gốc (câu + nhãn) vẫn còn
nguyên trong `weights_replay.json`/`feedback_log.jsonl`, dùng lại ngay để train kiến trúc
mới, không phải thu thập lại từ đầu.

**Q: Word segmentation tổng quát có làm chậm hệ thống không?**
A: Không đáng kể — longest-match tra dictionary là O(1) mỗi vị trí, gần tuyến tính theo độ
dài câu, trong khi Self-Attention đã có sẵn chi phí O(T²) (T = số token) lớn hơn nhiều;
câu cảm xúc đầu vào thường rất ngắn (dưới 24 token, theo `max_len=24` đang dùng).

---

## 9. Trạng thái triển khai

- **Model 1, Model 2**: đã triển khai thật trong code (`backend/emotion_mlp.py`,
  `backend/lexicon.py`), mốc chuyển đổi ghi trong `BaoCao_MoodTune_v2.5.md`.
- **Model 3**: đề xuất, **chưa code**. Toàn bộ phân tích trong tài liệu này dựa trên việc
  đọc trực tiếp file đang chạy production (`backend/emotion_mlp.py`), không phải giả định
  lý thuyết suông.
