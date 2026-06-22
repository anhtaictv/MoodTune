"""
Dạy lại Attention MLP sau khi nâng cấp Word Segmentation (v4.0 - Model 3).

Vì sao cần script này: trước v4.0, _tokenize() chỉ ghép được đúng 2 từ liền
kề (bigram-only) - các cụm cảm xúc 3-5 từ trong LEXICON (54/591 entry, xem
BaoCao_MoodTune_v4.0.md) có embedding sẵn trong VOCAB nhưng KHÔNG BAO
GIỜ được tokenizer cũ tạo ra token đó -> embedding "chết", chỉ bị L2 decay
dần về 0 suốt thời gian chạy, chưa từng học được gì có ích. Generalize
tokenizer sang longest-match (emotion_mlp.py) làm các cụm này LẦN ĐẦU TIÊN
reachable - nhưng giá trị embedding hiện tại của chúng vẫn là rác cũ, cần
chạy lại một đợt huấn luyện (pretrain) để thực sự học được ý nghĩa của
chúng. Đây chính là "đợt dạy" cho Model 3.

Cách dùng AN TOÀN (khuyến nghị) - chạy thử trên 1 bản COPY của weights
production, không đụng tới model đang chạy thật:

    cd backend
    cp weights.npz weights_model3_test.npz
    cp weights_meta.json weights_model3_test_meta.json
    cp weights_replay.json weights_model3_test_replay.json
    python teach_model3.py --weights weights_model3_test

Chỉ sau khi xem kết quả before/after và thấy ổn, mới áp dụng vào production
thật (cần dừng/restart service vì AttentionMLP nạp weights.npz lúc khởi
động, xem README.md "Running the backend"):

    python teach_model3.py --weights weights

Tham số:
    --weights PATH   prefix của bộ weights cần dạy (PATH.npz, PATH_meta.json,
                      PATH_replay.json). Mặc định "weights_model3_test" để
                      KHÔNG vô tình đụng vào production nếu quên truyền cờ.
    --epochs N        số epoch pretrain (mặc định 400, giống cold-start
                      trong EmotionEngine.__init__).
"""
import argparse
import sys

import numpy as np

from emotion_mlp import EmotionEngine, SEED_DATA, EMOTIONS, LEXICON, to_token_ids


def _label(name):
    return EMOTIONS.index(name)


def collect_dead_phrase_examples():
    """Tự động sinh 1 câu ví dụ (chính cụm từ đó) cho MỌI cụm cảm xúc >=3 từ
    trong LEXICON - đảm bảo toàn bộ 54 cụm từng "chết" trước v4.0 đều có ít
    nhất 1 mẫu huấn luyện trực tiếp, không chỉ một tập con chọn tay."""
    examples = []
    for emo, words in LEXICON.items():
        label = _label(emo)
        for phrase in words:
            if len(phrase.split()) >= 3:
                examples.append((phrase, label))
    return examples


# Vài câu tự nhiên hơn (cụm dài nằm trong câu đầy đủ, có ngữ cảnh xung
# quanh) - bổ sung cho DEAD_PHRASE_EXAMPLES (chỉ là cụm trần trụi) để model
# không chỉ thấy cụm đó một mình, cũng học được cách nó xuất hiện giữa câu.
HAND_WRITTEN_EXAMPLES = [
    ("thật sự không thể chấp nhận được chuyện này", _label("angry")),
    ("tôi ghét cay ghét đắng cái kiểu đó", _label("angry")),
    ("hôm nay tôi tràn đầy năng lượng", _label("energetic")),
    ("dạo này quá tải tinh thần quá", _label("stressed")),
    ("áp lực nặng từ công việc khiến tôi mệt", _label("stressed")),
    ("cứ nhớ ngày xưa lúc còn là tuổi học trò", _label("nostalgic")),
    ("nghe bài hát cũ lại nhớ ký ức tuổi thơ", _label("nostalgic")),
    ("muốn tìm một buổi sáng yên tĩnh để nạp lại năng lượng", _label("relaxed")),
    ("không ai tâm sự nên thấy cô đơn lắm", _label("lonely")),
    ("cần im lặng để làm bài cho xong", _label("focused")),
]

DEAD_PHRASE_EXAMPLES = collect_dead_phrase_examples()


DEMO_SENTENCES = [
    "không thể chấp nhận được",
    "ghét cay ghét đắng",
    "tràn đầy năng lượng",
    "áp lực nặng",
    "quá tải tinh thần",
    "nhớ ngày xưa",
    "hôm nay tôi tràn đầy năng lượng",
    "dạo này quá tải tinh thần quá",
]


def predict_mlp_only(engine, text):
    ids = to_token_ids(text)
    p = engine.mlp.predict(ids)
    return EMOTIONS[int(np.argmax(p))], float(p.max())


def predict_final(engine, text):
    out = engine.predict(text)
    return out["emotion"]


def snapshot(engine, sentences):
    return {
        s: {
            "mlp": predict_mlp_only(engine, s),
            "final": predict_final(engine, s),
        }
        for s in sentences
    }


def print_comparison(before, after, expected_by_sentence):
    print()
    print(f"{'Câu':45} {'MLP trước':12} {'MLP sau':12} {'Hybrid trước':14} {'Hybrid sau':12}")
    print("-" * 100)
    for s in before:
        b_mlp, b_conf = before[s]["mlp"]
        a_mlp, a_conf = after[s]["mlp"]
        mark = " <-- đổi" if b_mlp != a_mlp else ""
        print(f"{s:45} {b_mlp:12} {a_mlp:12} {before[s]['final']:14} {after[s]['final']:12}{mark}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="weights_model3_test",
                         help="prefix bộ weights cần dạy (mặc định: weights_model3_test, KHÔNG phải production)")
    parser.add_argument("--epochs", type=int, default=400)
    args = parser.parse_args()

    print(f"[teach_model3] Nạp engine từ '{args.weights}'...")
    engine = EmotionEngine(args.weights)
    print(f"[teach_model3] feedback_count={engine.feedback_count} | alpha={engine.alpha:.3f} "
          f"| vocab_size={engine.vocab_size} | replay={len(engine.replay)} mẫu")

    print(f"[teach_model3] Thu thập {len(DEAD_PHRASE_EXAMPLES)} cụm cảm xúc >=3 từ từng 'chết' trước v4.0 "
          f"+ {len(HAND_WRITTEN_EXAMPLES)} câu tự nhiên viết tay.")

    before = snapshot(engine, DEMO_SENTENCES)

    data = SEED_DATA + DEAD_PHRASE_EXAMPLES + HAND_WRITTEN_EXAMPLES + [(t, l) for t, l in engine.replay]
    print(f"[teach_model3] Pretrain trên {len(data)} mẫu (SEED_DATA + cụm chết + viết tay + replay), "
          f"{args.epochs} epochs...")
    engine.pretrain(data, epochs=args.epochs)

    after = snapshot(engine, DEMO_SENTENCES)

    print_comparison(before, after, None)
    print()
    print(f"[teach_model3] Đã lưu lại weights tại '{args.weights}.npz' / '{args.weights}_meta.json' "
          f"/ '{args.weights}_replay.json'.")


if __name__ == "__main__":
    sys.exit(main())
