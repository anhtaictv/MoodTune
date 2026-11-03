"""
Train lại TfidfEmotionModel từ seed data + feedback_log.jsonl rồi ghi tfidf_model.npz.

Chạy trong tiến trình RIÊNG (python train_tfidf.py) - app.py gọi qua subprocess
sau khi có feedback mới, để sklearn/scipy không bao giờ nằm trong RAM backend.
"""
import json
import os
import sys
import time

import numpy as np

import emotion_mlp as em
from emotion_mlp import EMOTIONS
from text_classifier import MODEL_FILE, PSEUDO_WEIGHT, train

HERE = os.path.dirname(os.path.abspath(__file__))
FEEDBACK_LOG = os.path.join(HERE, "feedback_log.jsonl")
# Chỉ lấy nhãn giả từ predict của engine CŨ (trước v5.0). Predict sau mốc này là
# output của chính model TF-IDF -> học lại sẽ tự củng cố lỗi, và ai gọi
# /api/predict cũng lái được model mà không qua rate limit của /api/learn.
PSEUDO_CUTOFF = "2026-11-03"


def seed_pairs():
    return (em.SEED_DATA + em.NEGATION_AUGMENT_DATA + em.ANGER_AUGMENT_DATA
            + em.SAD_AUGMENT_DATA + em.ROMANTIC_AUGMENT_DATA)


def load_feedback(path=FEEDBACK_LOG):
    """-> (labeled, pseudo): dict text->label. Feedback (người dùng/teacher sửa
    nhãn) là nhãn thật; predict chỉ là nhãn model cũ tự đoán -> trọng số thấp."""
    labeled, pseudo = {}, {}
    if not os.path.exists(path):
        return labeled, pseudo
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            text = (d.get("text") or "").strip()
            if not text:
                continue
            if d.get("type") == "feedback" and d.get("correct") in EMOTIONS:
                labeled[text] = EMOTIONS.index(d["correct"])
            elif (d.get("type") == "predict" and d.get("result") in EMOTIONS
                  and (d.get("time") or "") < PSEUDO_CUTOFF):
                pseudo[text] = EMOTIONS.index(d["result"])
    return labeled, {t: l for t, l in pseudo.items() if t not in labeled}


def build_dataset():
    labeled, pseudo = load_feedback()
    pairs = seed_pairs() + list(labeled.items())
    texts = [t for t, _ in pairs] + list(pseudo)
    labels = [l for _, l in pairs] + list(pseudo.values())
    weights = np.array([1.0] * len(pairs) + [PSEUDO_WEIGHT] * len(pseudo))
    return texts, labels, weights, len(labeled), len(pseudo)


def main():
    t0 = time.time()
    texts, labels, weights, n_labeled, n_pseudo = build_dataset()
    train(texts, labels, weights, len(EMOTIONS), os.path.join(HERE, MODEL_FILE))
    print(f"[TF-IDF] trained on {len(texts)} samples (feedback={n_labeled}, pseudo={n_pseudo}) "
          f"in {time.time() - t0:.1f}s -> {MODEL_FILE}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
