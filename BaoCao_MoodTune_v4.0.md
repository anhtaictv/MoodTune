# MoodTune v4.0 — Báo cáo Model 3: Word Segmentation tổng quát

**Phiên bản:** `v4.0` (so với `v3.8` trong `BaoCao_MoodTune_v3.8.md`)
**Tên đầy đủ:** MoodTune — AI Cảm Xúc Tự Xây (RLUF Bandit · Knowledge Graph · Self-Attention
· Word Segmentation) + Gợi ý nhạc Jamendo Hybrid
**Chủ đề nâng cấp:** Generalize tokenizer của AI Engine từ "bigram-only" sang
"longest-match N-gram" (= **Model 3** trong `BaoCao_MoodTune_SoSanhModel.md`) — sửa lỗi
**"vocab chết"** phát hiện qua đo thực nghiệm trực tiếp trên code production, đăng ký
`NEGATIONS` như function-token, kèm script dạy lại model (`teach_model3.py`).

> **Đối chiếu với tài liệu chuẩn bị thuyết trình** (`BaoCao_MoodTune_SoSanhModel.md`,
> viết trước khi code Model 3): khi triển khai thực tế và đo thử trên code đang chạy
> production, phát hiện ra vấn đề **cụ thể hơn và nghiêm trọng hơn** giả thuyết ban đầu.
> Giả thuyết ban đầu là "MLP không tự nhìn thấy từ phủ định". Thực tế đo được:
> 1. Từ phủ định (`không`, `chẳng`...) **đã có embedding** trong VOCAB của bản production
>    hiện tại — không qua đăng ký chủ động, mà "tình cờ" lọt vào qua `add_vocab_words()`
>    mỗi khi chúng xuất hiện trong một câu feedback nào đó (xác nhận bằng
>    `dynamic_vocab.json` thực tế, xem mục 4).
> 2. Vấn đề thật sự lớn hơn: **cả `rule_score()` lẫn tokenizer của MLP đều có 54/591
>    (~9.1%) cụm cảm xúc dài 3-5 từ trong `LEXICON` hoàn toàn không bao giờ được nhận
>    diện** — dù tác giả lexicon đã cố tình viết ra (ví dụ `"không thể chấp nhận được"`
>    dưới `angry`) — vì cả hai chỗ đó trước v4.0 chỉ thử ghép đúng 2 từ liền kề.
>
> Báo cáo này phản ánh đúng phát hiện thực nghiệm, không lặp lại nguyên văn giả thuyết
> ban đầu trong tài liệu chuẩn bị thuyết trình.

---

## 1. Phát hiện: "vocab chết" — đo được bằng số liệu cụ thể

`VOCAB` được dựng từ các khoá (key) trong `LEXICON` (`emotion_mlp.py:59-63`), bất kể
khoá đó là 1 từ hay nhiều từ — nên một cụm 5 từ như `"không thể chấp nhận được"` vẫn có
một dòng embedding riêng trong ma trận `E` ngay từ lúc khởi tạo. Vấn đề là **tokenizer
cũ không có cách nào tạo ra token đó**: `_tokenize()` (dùng cho nhánh MLP) chỉ thử ghép
đúng 2 từ liền kề, và `rule_score()` cũng chỉ có một loop bigram riêng — không có chỗ
nào thử ghép 3, 4, hay 5 từ.

Đo trực tiếp trên `lexicon.py` (production):

```
Tổng số entry trong LEXICON: 591
Entry nhiều hơn 1 từ: 367 (62%)
  - 2 từ:  313  (đã reachable qua bigram-only — không vấn đề)
  - 3 từ:   42  ← KHÔNG BAO GIỜ reachable trước v4.0
  - 4 từ:   11  ← KHÔNG BAO GIỜ reachable trước v4.0
  - 5 từ:    1  ← KHÔNG BAO GIỜ reachable trước v4.0
```

→ **54 entry (~9.1% toàn bộ lexicon) là "vocab chết"**: có embedding, có trong `VOCAB`,
nhưng `to_token_ids()` không bao giờ trả ra `id` của chúng, nên `backward()` không bao
giờ cộng gradient cho dòng đó (`emotion_mlp.py:396-398`, loop scatter theo `self._ids`,
chỉ chứa các id thực sự được `to_token_ids()` trả ra) — *trừ* phần suy giảm rất nhỏ do
L2 weight decay áp dụng đều lên toàn ma trận `E` mỗi lần `backward()`
(`dE += self.l2 * self.E`, `emotion_mlp.py:399`), khiến các dòng "chết" này còn bị co dần
về 0 qua hàng nghìn lần học online — không chỉ vô dụng mà còn suy giảm theo thời gian.

Tái hiện trực tiếp trên engine production (trước khi sửa):

```python
>>> _tokenize("không thể chấp nhận được")
[('không', 619), ('thể', 967), ('chấp', 968), ('nhận', 969), ('được', 648)]
# 5 token rời rạc — KHÔNG có token nào là cụm "không thể chấp nhận được"
# dù cụm đó có sẵn id riêng trong VOCAB_IDX từ lúc khởi tạo module.
```

---

## 2. Cài đặt 1: Word Segmentation longest-match cho nhánh MLP

### Mô tả
Generalize `_tokenize()` (`emotion_mlp.py:215-243`): tại mỗi vị trí, thử ghép cụm **dài
nhất** có sẵn trong `VOCAB_IDX` trước (từ `MAX_PHRASE_LEN` từ giảm dần xuống 2), rồi mới
rơi về từ đơn — thay cho bản cũ chỉ thử đúng 2 từ.

### Cài đặt kỹ thuật
- `MAX_PHRASE_LEN` (`emotion_mlp.py:137`): tính một lần lúc module load —
  `max(len(w.split()) for w in VOCAB if " " in w)` — tự động bằng 5 với lexicon hiện tại,
  tự cập nhật nếu sau này lexicon có cụm dài hơn, không cần sửa code.
- `_tokenize()` mới: vòng `for length in range(max_try, 1, -1)` thử từ dài xuống ngắn,
  khớp ngay khi `" ".join(words[i:i+length])` có trong `VOCAB_IDX`, advance con trỏ theo
  đúng độ dài vừa khớp. Token OOV vẫn giữ `id=None` như cũ (phục vụ Knowledge Graph).

### Kết quả kiểm thử
```python
>>> _tokenize("không thể chấp nhận được")
[('không thể chấp nhận được', 556)]   # ĐÚNG 1 token, reachable
```

---

## 3. Cài đặt 2: Generalize `rule_score()` sang N-gram

### Mô tả
`rule_score()` (`emotion_mlp.py:180-213`) có loop bigram **riêng**, độc lập với
`_tokenize()` — cũng bị giới hạn đúng 2 từ, nên 54 cụm dài cũng chưa từng được rule-based
scorer (vốn không phụ thuộc trạng thái học, luôn đúng ngay khi sửa code) chấm điểm.

### Cài đặt kỹ thuật
- `MAX_LEXICON_PHRASE_LEN` (`emotion_mlp.py:163-166`): độ dài cụm dài nhất tính trực
  tiếp từ `LEXICON` (bằng 5 hiện tại).
- Loop bigram cũ → loop N-gram: `for length in range(2, MAX_LEXICON_PHRASE_LEN + 1)`,
  giữ nguyên hệ số boost `×1.5` và cách gọi `_is_negated(words, i)` — **không đổi**
  `_is_negated()` (giữ đúng signature/hành vi đã được 14 test case cũ pin chặt).

### Kết quả kiểm thử
```python
>>> rule_score("không thể chấp nhận được")   # trước: uniform (không match gì)
→ argmax = angry, không còn uniform

>>> rule_score("tôi không hề tràn đầy năng lượng")   # phủ định cụm 4 từ
→ uniform (đúng — phủ định triệt tiêu điểm)
```

---

## 4. Cài đặt 3: Đăng ký `NEGATIONS` như function-token

### Mô tả
Đo trực tiếp `dynamic_vocab.json` (production) phát hiện `"không"`, `"chẳng"`, `"chưa"`
**đã** có trong `VOCAB_IDX` — nhưng KHÔNG phải vì được thiết kế vào từ đầu: `learn()` gọi
`find_oov_words()` (`emotion_mlp.py` — không lọc theo từ cảm xúc, nhận MỌI từ ≥2 ký tự
chưa có trong VOCAB) mỗi lần có feedback, nên các từ phủ định chỉ tình cờ lọt vào nếu đã
từng xuất hiện trong một câu feedback nào đó. Không đảm bảo *toàn bộ* `NEGATIONS` đều có
mặt, và embedding của chúng học được "tình cờ" qua bất kỳ câu nào chúng từng xuất hiện
— không có nguồn tín hiệu nào nói rõ "đây là token phủ định".

### Cài đặt kỹ thuật
- `emotion_mlp.py:65-79`: ngay sau loop dựng `VOCAB` từ `LEXICON`, thêm loop đăng ký toàn
  bộ `NEGATIONS` vào `VOCAB_IDX` — đảm bảo **chắc chắn 100%**, không phụ thuộc việc từ đó
  có từng xuất hiện gần một từ cảm xúc trong feedback hay chưa.

### Kết quả kiểm thử
```python
>>> all(neg in VOCAB_IDX for neg in NEGATIONS)
True   # đúng cho TOÀN BỘ NEGATIONS, không chỉ vài từ tình cờ học được trước đó
```

---

## 5. Test — không có regression

Chạy lại toàn bộ `backend/tests/test_rule_score.py` sau khi sửa:

```
37 test cũ (TestIsNegated, TestRuleScoreBasics, TestRuleScoreNegation) — PASSED, không đổi
 8 test mới (TestWordSegmentationModel3):
   - 5 test parametrize: cụm 3/4/5 từ trước đây "chết" giờ match đúng emotion
   - test_long_phrase_tokenizes_as_single_token: nhánh MLP nhận đúng 1 token
   - test_negation_cancels_long_phrase: phủ định vẫn triệt tiêu đúng cụm dài
   - test_all_negations_are_registered_in_vocab: toàn bộ NEGATIONS có trong VOCAB_IDX
======================== 45 passed in 3.84s ========================
```

`_is_negated()` không bị sửa — toàn bộ 14 test cũ pin hành vi đó tiếp tục pass nguyên
vẹn, không có rủi ro regression ở phần phủ định.

---

## 6. Script dạy lại model: `teach_model3.py`

### Vì sao cần
Generalize tokenizer làm 54 cụm "chết" lần đầu tiên *reachable* — nhưng giá trị embedding
hiện tại của chúng vẫn là rác cũ (random init, bị L2 decay suốt thời gian "chết"), cần
chạy lại một đợt huấn luyện để thực sự học được ý nghĩa.

### Cách hoạt động
- `collect_dead_phrase_examples()`: tự động sinh 1 mẫu huấn luyện (chính cụm từ đó) cho
  **toàn bộ 54** cụm ≥3 từ trong `LEXICON` — đảm bảo coverage đầy đủ, không phải tập con
  chọn tay.
- `HAND_WRITTEN_EXAMPLES`: 10 câu tự nhiên đặt các cụm đó trong ngữ cảnh câu đầy đủ.
- Pretrain tiếp tục **từ trọng số hiện có** (warm-start, không reset về random) trên
  `SEED_DATA + 54 cụm chết + 10 câu viết tay + replay buffer hiện tại` — giữ nguyên toàn
  bộ kiến thức đã học từ 3551 feedback thật, chỉ bổ sung tín hiệu cho phần đang thiếu.
- An toàn theo thiết kế: mặc định lưu vào `--weights weights_model3_test` (không phải
  `weights` production) — phải truyền `--weights weights` rõ ràng mới đụng tới model đang
  chạy thật.

### Kết quả dry-run (copy của weights production, KHÔNG đụng bản thật)

```
[Engine] Loaded | feedback=3551 | alpha=0.356 | replay=500
Pretrain trên 656 mẫu, 400 epochs → loss 0.014 → 0.008 (hội tụ, không nổ gradient)

Câu                              MLP trước   MLP sau     Hybrid trước  Hybrid sau
---------------------------------------------------------------------------------
không thể chấp nhận được         focused     angry       angry         angry
ghét cay ghét đắng                focused     angry       focused       angry      <-- đổi
tràn đầy năng lượng                focused     energetic   energetic     energetic
áp lực nặng                       focused     stressed    stressed      stressed
quá tải tinh thần                  focused     stressed    stressed      stressed
nhớ ngày xưa                      focused     nostalgic   nostalgic     nostalgic
hôm nay tôi tràn đầy năng lượng    happy       energetic   happy         energetic  <-- đổi
dạo này quá tải tinh thần quá      focused     stressed    focused       stressed   <-- đổi
```

**Quan sát quan trọng:** trước khi dạy lại, nhánh MLP-thuần (`MLP trước`) đoán **"focused"
cho cả 6/8 câu** bất kể nội dung thật — đúng như dự đoán từ phân tích "vocab chết": với
embedding gần như nhiễu/zero, đầu ra của mạng bị chi phối bởi bias đã học từ trước, cho
ra một đáp án "mặc định" giống nhau bất kể input. Sau khi dạy lại, MLP-thuần đoán đúng cả
8/8. Vì `alpha` hiện tại ở production chỉ 0.356 (MLP chiếm 64.4% trọng số quyết định cuối
— xem `BaoCao_MoodTune_SoSanhModel.md` mục 1), tín hiệu MLP sai trước đây **đủ mạnh để
kéo cả kết quả hybrid cuối cùng sang sai** ở 3/8 câu (`ghét cay ghét đắng`,
`hôm nay tôi tràn đầy năng lượng`, `dạo này quá tải tinh thần quá`) — sau khi dạy lại,
MLP và rule đồng thuận, hybrid đúng cả 8/8.

---

## 7. Bảng so sánh tổng quan v3.8 vs v4.0

| Khía cạnh | v3.8 | v4.0 |
|---|---|---|
| Tokenizer nhánh MLP | Bigram-only (tối đa 2 từ) | **Longest-match N-gram** (tối đa `MAX_PHRASE_LEN`, hiện =5) |
| `rule_score()` phrase matching | Bigram-only (loop riêng) | **N-gram** (2..`MAX_LEXICON_PHRASE_LEN`) |
| Cụm cảm xúc 3-5 từ trong LEXICON | 54/591 "chết" — không bao giờ reachable | **Reachable**, xác nhận bằng test + dry-run |
| `NEGATIONS` trong VOCAB | Tình cờ, qua `add_vocab_words()` khi gặp trong feedback | **Đăng ký chắc chắn 100%** lúc khởi động |
| Test suite | 37 test | **45 test** (8 mới, 0 regression) |
| Script dạy lại model | Không có (đề cập `gemini_teacher.py` ở v3.7 nhưng không có trong repo) | **`teach_model3.py`** — auto-collect 54 cụm chết + replay, an toàn theo mặc định |
| Tương thích `weights.npz` cũ | — | Tự self-heal qua `expand_vocab()` (đã có từ v3.7) khi `VOCAB_SIZE` tăng do thêm `NEGATIONS`; cần chạy `teach_model3.py` để 54 cụm chết thực sự học được điều gì đó hữu ích |
| Trạng thái | Đã triển khai | **Code đã xong, đã test, đã dry-run an toàn** — **weights production CHƯA được dạy lại thật** (xem mục 8) |

---

## 8. Triển khai

- **Code đã thay đổi:** `backend/emotion_mlp.py` (+104/-30 dòng), thêm
  `backend/tests/test_rule_score.py` (+52 dòng, 8 test mới), thêm mới
  `backend/teach_model3.py`. Chưa commit.
- **Đã verify an toàn:** dry-run `teach_model3.py` chạy trên **bản copy**
  (`weights_model3_test.*`), không đụng tới `weights.npz`/`weights_meta.json`/
  `weights_replay.json` production đang backing service thật (`feedback_count=3551`) —
  xác nhận bằng `git status` (3 file production không xuất hiện là modified).
- **CHƯA làm (cần quyết định riêng, không tự động thực hiện):**
  1. Chạy `teach_model3.py --weights weights` để dạy lại weights **production thật**.
  2. Restart service (`pm2`/`waitress`) để nạp lại weights vừa dạy — `AttentionMLP` chỉ
     đọc `weights.npz` lúc khởi động.
  3. Commit code + cập nhật version tag trên UI/README/changelog (hiện vẫn ghi `v3.8`)
     nếu quyết định "lên" v4.0 chính thức.

> **Định hướng tiếp theo:** sau khi production học thêm dữ liệu thật (không chỉ
> `SEED_DATA`/cụm tự sinh), nên đo lại tỉ lệ các cụm 3-5 từ này thực sự xuất hiện trong
> `feedback_log.jsonl` để biết mức ưu tiên thực tế — nếu hiếm gặp trong câu người dùng
> thật, lợi ích chính của v4.0 sẽ nằm ở việc dọn "nợ kỹ thuật" (vocab chết, dễ gây nhiễu
> khi lexicon mở rộng thêm) hơn là cải thiện accuracy đo được ngay lập tức.
